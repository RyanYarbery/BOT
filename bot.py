# https://trade.stage.dydx.exchange/trade/ETH-USD
import asyncio
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
# from DYDXInterface.dydx_interface import DydxInterface
from DYDXInterface.dydx_v4_interface import DydxInterface
import os
import torch
import logging
import time
from Model.model import ModelType, TradingModel
from Model.model_data import Data, TimeSeriesDataset
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from config import load_config
# Was using bot_env 3.10.14
config = load_config('config.yaml')
print("ALL THESE FRIGGEN' VALUES SHOULD BE AQUIRED FROM THE LOADED MODEL OBJECT SO WE DON'T HAVE TO UPDATE THIS ISH ALL THE TIME.")
SEQUENCE_LENGTH = config['model']['sequence_length']
NUM_OUTPUTS = config['model']['num_outputs']
MAX_HISTORY_LEN = config['model']['action_sequence_length']

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers= [
                        logging.FileHandler('trading.log'),
                        logging.StreamHandler()
                    ])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# dydx = DydxInterface() # Testnet
dydx = None

async def initialize_dydx():
    global dydx
    dydx = DydxInterface()

asyncio.run(initialize_dydx())
# dydx = DydxInterface(environment = 'main') # Mainnet



def load_model():
    model = TradingModel(
        num_types=19,
        timeframes=config['data']['resample_timeframes'],
        num_heads=config['model']['nhead'],
        d_models_pre=config['model']['preprocess_layers'],
        d_models_joint=config['model']['process_layers'],
        model_type=ModelType(2),
        dropout=config['model']['dropout'],
        target_tf=config['data']['target_tf'],
        pre_process=False,
        max_seq_length=config['model']['sequence_length'],
        action_sequence_length=config['model']['action_sequence_length'],
        dim_feedforward=config['model']['dim_feedforward'],
    )
    
    
    model_info = torch.load('Models/Model.pth')
    # model = model_info['network_class']() # Create instance of model class
    model.load_state_dict(model_info['model_state_dict'])
    
    model.eval()
    return model, model_info

# Use integer constants for positions
SHORT = np.int8(-1)
NONE = np.int8(0)
LONG = np.int8(1)

class Portfolio:
    """In the future, take in (and make) a client class for this portfolio."""
    def __init__(self, dydx_balance, tsl_pct, withdrawal_pct, leverage):
        self.tsl_pct_ = tsl_pct
        # self.withdrawal_pct_ = withdrawal_pct
        self.trade_balance = dydx_balance #* (1 - withdrawal_pct)
        # self.reserve_balance = dydx_balance * withdrawal_pct
        self.position = NONE
        self.entry_price = 0
        self.leverage = leverage
        self.effective_leverage = 0
        self.open_position_value = 0
        self.tsl_price = 0
        self.tsl_hit = 0
                    
    def short(self, current_price, open_price=None):
        new_position_value = self.leverage * dydx.fetch_equity()
        new_position_size = new_position_value / current_price
        side_input = 'sell'
        entry_price = dydx.calculate_new_price(current_price, operation='subtract', buffer_value=2)
        
        if self.position == LONG:
            self.close_position(current_price)
        elif self.position == SHORT:
            self.trade_balance += self.open_position_value / self.leverage - self.trade_balance
            
            # Current position size will be negative for short.
            current_position_size = dydx.fetch_position_size()            
            # if current position size abs is greater than new_position_size, buy since size is too big.
            if abs(current_position_size) > abs(new_position_size):
                new_position_size = float(abs(current_position_size)) - float(abs(new_position_size))
                side_input = 'buy'
                entry_price = dydx.calculate_new_price(current_price, operation='add', buffer_value=2)
            else:
                # Positive val (new_position_size) + negative curr size (current_position_size) = delta size needed to buy
                new_position_size = float(abs(new_position_size)) - float(abs(current_position_size))
            logging.info(f'Short changing. Current size: {current_position_size}, Chanting to this size: {new_position_size} (additive sizes may be negative)')
        
        response = dydx.place_limit_order(side_input=side_input, size=new_position_size, price=entry_price)
                
        logging.info(f'Short order placed. Response: {response}')
        
        self.tsl_price = current_price * (1 + self.tsl_pct_)
        self.position = SHORT
        self.entry_price = current_price

    def long(self, current_price, open_price=None):
        new_position_value = self.leverage * dydx.fetch_equity()
        new_position_size = new_position_value / current_price
        side_input = 'buy'
        
        entry_price = dydx.calculate_new_price(current_price, operation='add', buffer_value=2)
        
        if self.position == SHORT:
            self.close_position(current_price)
        elif self.position == LONG:
            self.trade_balance += self.open_position_value / self.leverage - self.trade_balance

            current_position_size = dydx.fetch_position_size()
            # if current position size abs is greater than new_position_size, sell since size is too big.
            if abs(current_position_size) > abs(new_position_size):
                new_position_size = float(abs(current_position_size)) - float(abs(new_position_size))
                side_input = 'sell'
                entry_price = dydx.calculate_new_price(current_price, operation='subtract', buffer_value=2)
            else:
                # Positive val (new_position_size) + negative curr size (current_position_size) = delta size needed to buy
                new_position_size = float(abs(new_position_size)) - float(abs(current_position_size))
            logging.info(f'Long changing. Current size: {current_position_size}, Adding this size: {new_position_size} (additive sizes may be negative)')
        
        response = dydx.place_limit_order(side_input=side_input, size=new_position_size, price=entry_price)
        
        logging.info(f'Short order placed. Response: {response}')
        
        self.tsl_price = current_price * (1 - self.tsl_pct_) #(exit_price - self.entry_price[self.idx_]) * trade_size / self.entry_price[self.idx_] < Why closing provides a better end profit.
        self.position = LONG
        self.entry_price = current_price
        
    def calculate_tsl_price(self, entry_price):
        pass # This method needs to be implemented to calculate the TSL price based on the entry price and the TSL percentage.
        
        
    def update_tsl(self, current_price):
        if self.position == SHORT:
            new_tsl_price = current_price * (1 + self.tsl_pct_)
            if new_tsl_price < self.tsl_price:
                self.tsl_price = new_tsl_price
        elif self.position == LONG:
            new_tsl_price = current_price * (1 - self.tsl_pct_)
            if new_tsl_price > self.tsl_price:
                self.tsl_price = new_tsl_price

    def check_tsl(self, current_price):
        if self.position == SHORT and current_price >= self.tsl_price:
            logging.info(f"SHORT TSL hit. Closing position at price: {current_price}")
            self.close_position(current_price)
            self.tsl_hit += 1
            return True
        elif self.position == LONG and current_price <= self.tsl_price:
            logging.info(f"LONG TSL hit. Closing position at price: {current_price}")
            self.close_position(current_price)
            self.tsl_hit += 1
            return True
        return False

    def close_position(self, exit_price):
        open_positions = dydx.fetch_open_positions()
        for position in open_positions:
            if position['status'] == 'OPEN':
                size = position['size']
                break  # Exit the loop once the first open position is found
        trade_size = self.trade_balance * self.leverage
        if self.position == SHORT:
            price = dydx.calculate_new_price(exit_price, operation='add', buffer_value=2)
            order = dydx.place_limit_order('buy', size, price)
            logging.info(f"Closing order with order: {order}, size: {size}, price: {price}")
            
            self.trade_balance += (self.entry_price - exit_price) * trade_size / self.entry_price
        elif self.position == LONG:
            price = dydx.calculate_new_price(exit_price, operation='subtract', buffer_value=2)
            order = dydx.place_limit_order('sell', size, price)
            logging.info(f"Closing order with order: {order}, size: {size}, price: {price}")
            
            self.trade_balance += (exit_price - self.entry_price) * trade_size / self.entry_price
        
        self.position = NONE
        self.entry_price = 0
        self.tsl_price = 0
        self.open_position_value = 0
        self.effective_leverage = 0
        
    def calculate_effective_leverage(self, current_price):
        if self.position == NONE:
            return 0
        if self.position == SHORT:
            new_equity = self.trade_balance + (self.entry_price / current_price - 1) * (self.trade_balance * self.leverage)
            effective_leverage = self.open_position_value / new_equity
        elif self.position == LONG:
            new_equity = self.trade_balance + (current_price / self.entry_price - 1) * (self.trade_balance * self.leverage)
            effective_leverage = self.open_position_value / new_equity
        else:
            print("Invalid position")
        self.effective_leverage = effective_leverage
        
    def update_open_position_value(self, current_price):
        if self.position == NONE:
            self.open_position_value = 0
        else:
            if self.position == LONG:
                current_position_value = (current_price / self.entry_price) * (self.trade_balance * abs(self.leverage))
            elif self.position[self.idx_] == SHORT:
                current_position_value = (self.entry_price / current_price) * (self.trade_balance * abs(self.leverage))
                
            self.open_position_value = current_position_value

class FixedSizeQueue:
    def __init__(self, batch_size, max_size, device=None, default_value=(2, 0, 2, 0)): # current_position (0: long, 1: short, 2: none), end_tick_effective_leverage, action (0: long, 1: short, 2: none, 3:close), hold_time
        """
        Initializes a fixed-size queue for multiple sequences (batches).

        Parameters:
        - batch_size (int): Number of sequences in the batch.
        - max_size (int): Maximum size of the queue.
        - device (torch.device, optional): The device on which tensors are allocated.
        - default_value (tuple): The default value to initialize the queue with.
          Format: (current_position, end_tick_effective_leverage, action, hold_time)
        """
        self.batch_size = batch_size
        self.max_size = max_size
        self.device = device if device else torch.device('cpu')
        
        # Create a tensor from default_value and repeat it across batch_size and max_size
        default_tensor = torch.tensor(default_value, dtype=torch.float32, device=self.device)
        self.queue = default_tensor.unsqueeze(0).unsqueeze(0).repeat(batch_size, max_size, 1)
        
        # Initialize pointers for each sequence in the batch
        self.pointers = torch.zeros(batch_size, dtype=torch.long, device=self.device)
    
    def add_item(self, items):
        """
        Adds a batch of items to the queue.

        Parameters:
        - items (torch.Tensor or list of tuples): Batch of items to add.
          Each item should be a tuple/list of (current_position, end_tick_effective_leverage, action, hold_time).
          Shape: (batch_size, 4)
        """
        if isinstance(items, list):
            items = torch.tensor(items, dtype=torch.float32, device=self.device)
        elif isinstance(items, torch.Tensor):
            items = items.to(self.device, dtype=torch.float32)
        else:
            raise TypeError("Items must be a list of tuples or a torch.Tensor")
        
        if items.shape != (self.batch_size, 4):
            raise ValueError(f"Expected items of shape ({self.batch_size}, 4), got {items.shape}")
        
        # Update the queue at the current pointers
        batch_indices = torch.arange(self.batch_size, device=self.device)
        self.queue[batch_indices, self.pointers] = items
        
        # Increment pointers and wrap around using modulo for circular buffer
        self.pointers = (self.pointers + 1) % self.max_size # WTF is this doing?https://chatgpt.com/c/674cfab6-63fc-8006-9f43-f5e1c013e295
    
    def get_items(self):
        """
        Retrieves the queue items in the correct order for each sequence in the batch.

        Returns:
        - torch.Tensor: A tensor of shape (batch_size, max_size, 4) containing the queue items.
        """
        # Create an index tensor for each position in the queue
        indices = (self.pointers.unsqueeze(1) + torch.arange(self.max_size, device=self.device)) % self.max_size
        # Gather the items in the correct order
        ordered_queue = torch.gather(self.queue, 1, indices.unsqueeze(-1).expand(-1, -1, 4))
        return ordered_queue
    
    def reset(self):
        """
        Resets the queue to the default values.
        """
        default_tensor = torch.tensor((2, 0, 2, 0), dtype=torch.float32, device=self.device)
        self.queue = default_tensor.unsqueeze(0).unsqueeze(0).repeat(self.batch_size, self.max_size, 1)
        self.pointers.zero_()

def collect_csv_filenames():
    base_date = datetime.now()
    filenames = []
    filenames.append('CSVs/' + base_date.strftime('%Y-%m') + '_ETHUSDTKlines.csv')
    prev_month = base_date - relativedelta(months=1)
    filenames.append('CSVs/' + prev_month.strftime('%Y-%m') + '_ETHUSDTKlines.csv')

    if base_date.day < 2:
        two_months_ago = base_date - relativedelta(months=2)
        filenames.append('CSVs/' + two_months_ago.strftime('%Y-%m') + '_ETHUSDTKlines.csv')

    return filenames

def merge_csv_files(filenames):
    data_frames = []

    for filename in filenames:

        if os.path.exists(filename):

            try:
                print(f"Attempting to read file: {filename}")
                df = pd.read_csv(filename, header=0, parse_dates=[0], 
                                 names=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'],
                                 usecols=[0, 1, 2, 3, 4, 5],
                                 skiprows=1)
                data_frames.append(df)
                
            except FileNotFoundError:
                logging.debug(f"File {filename} not found. Skipping.")

        else:
            print(f"File {filename} does not exist.")  # Additional check

    if not data_frames:
        raise ValueError("No data frames were created. Ensure the CSV files exist and are accessible.")
    
    data = pd.concat(data_frames).drop_duplicates(subset='DateTime').sort_values('DateTime')#.reset_index(drop=True)
    
    data.dropna(inplace=True)
    
    data['DateTime'] = pd.to_datetime(data['DateTime'], format='%Y-%m-%d %H:%M:%S')
    data = data.set_index('DateTime')
    
    cutoff_date = data.index.max() - pd.Timedelta(days=30)
    data = data[data.index > cutoff_date]

    return data

def prepare_input(model_data, sequence_length, num_outputs):
#     xception has occurred: TypeError
# TimeSeriesDataset.__init__() missing 3 required positional arguments: 'epoch', 'valid_indices', and 'data_type'
#   File "/home/andrew/Programming/BOT-main/BOT/bot.py", line 349, in prepare_input
#     return TimeSeriesDataset(model_data, sequence_length, num_outputs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/andrew/Programming/BOT-main/BOT/bot.py", line 369, in process_data
#     prepped_model_data = prepare_input(model_data, SEQUENCE_LENGTH, NUM_OUTPUTS)
#                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/andrew/Programming/BOT-main/BOT/bot.py", line 423, in main
#     predicted_target = process_data(model, history_queue)
#                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/andrew/Programming/BOT-main/BOT/bot.py", line 447, in <module>
#     asyncio.run(main())
# TypeError: TimeSeriesDataset.__init__() missing 3 required positional arguments: 'epoch', 'valid_indices', and 'data_type'
    return TimeSeriesDataset(model_data, sequence_length, num_outputs)

def process_data(model, history_queue): # Should probably improve these fn names here:
    file_names = collect_csv_filenames()
    
    min_df = merge_csv_files(file_names)
    min_df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }, inplace=True)
    model_data = Data(
        config,
        interpolate_to_target=False,
        scaler=StandardScaler(),
        live_trading=True,
        dataframe=min_df,
        )
    prepped_model_data = prepare_input(model_data, SEQUENCE_LENGTH, NUM_OUTPUTS)
    
    last_item = prepped_model_data[-1]
    
    # Convert action_leverage_history into tensors
    history_items = history_queue.get_items()
    # Extract positions, leverages, and actions using tensor slicing
    positions_tensor = history_items[:, :, 0].long()       # Shape: [batch_size, max_size]
    leverages_tensor = history_items[:, :, 1].float()      # Shape: [batch_size, max_size]
    actions_tensor = history_items[:, :, 2].long()         # Shape: [batch_size, max_size]
    hold_time_tensor = history_items[:, :, 3].float()      # Shape: [batch_size, max_size]
    
    with torch.no_grad():        
        input_datums = last_item['input'].to(device)
        
        if input_datums.dim() == 3: # IS THIS STILL VALID? - 12/12/24
            input_datums = input_datums.unsqueeze(0)
        
        q_values = model(
            tensors = input_datums,
            positions = positions_tensor,
            effective_leverages = leverages_tensor,
            actions = actions_tensor,
            hold_time = hold_time_tensor,
        )
        action = q_values.max(1)[1]
        
    
async def main():
    print("Is data retreival running first?")
    dydx.clear_existing_orders_and_positions()
    dydx_balance = dydx.fetch_equity()
    
    portfolio = Portfolio(dydx_balance, tsl_pct=0.01, withdrawal_pct=0.1, leverage=10) # withdrawal_pct not implemented yet.
    
    model, model_info = load_model()
    model = model.to(device)
    
    # SEQUENCE_LENGTH = model_info['sequence_length']
    # NUM_OUTPUTS = model_info['num_outputs']
    # predicted_target = process_data(model)
    history_queue = FixedSizeQueue(batch_size=1, max_size=MAX_HISTORY_LEN, device=device)
    
    while True:
        current_time = datetime.now()
        dydx_trading_price = await dydx.fetch_eth_price()
        open_position = await dydx.fetch_open_positions()
        if open_position:
            open_position = SHORT if open_position[0]['side'] == 'SHORT' else LONG if open_position[0]['side'] == 'LONG' else NONE
        else:
            open_position = NONE
        portfolio.position = open_position
        
        if (current_time.minute % 5 == 2): # 2 because of min delay in fetching data. and another min to ensure csv saved
            predicted_target = process_data(model, history_queue)
            
            logging.info(f"Predicted target: {predicted_target}")
            
            if predicted_target == 0:
                portfolio.long(dydx_trading_price)
            elif predicted_target == 1:
                portfolio.short(dydx_trading_price)
            else:
                portfolio.close_position(dydx_trading_price)
        elif portfolio.position != NONE:
            # IMPLEMENT THIS: Ensure that the TSL price is calculated correctly.
            # portfolio.tsl_price = portfolio.calculate_tsl_price(dydx_trading_price)
            stop_loss_hit = portfolio.check_tsl(dydx_trading_price)
            if not stop_loss_hit:
                portfolio.update_tsl(dydx_trading_price)
        
        current_time = datetime.now()
        next_minute = current_time.replace(second=0, microsecond=0) + timedelta(minutes=1)
        delta_time = (next_minute - current_time).total_seconds()        
        time.sleep(delta_time) #This should make sure we start at the top of the minute, every minute.
    

if __name__ == "__main__":
    asyncio.run(main())
    
    
# Potential improvements, have reserve balance fluctuate depending on volitility.

# need to ensure that positions ARE closed, especially when changing positional direction, pull size on the position! also, ensure that the position fills/doesnt expire. 
