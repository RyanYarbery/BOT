import logging
import numpy as np
from torch.utils.data import RandomSampler
import torch
# from config import load_config
from model import ModelType
from torch.utils.data import DataLoader, random_split
import pandas_ta as ta
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from torch.utils.data import Subset
from math import ceil

"""
Remember that when you're using this data for prediction later, you should use the same scaler to transform new data. You can save the scaler object for later use:
import joblib

# Save the scaler
joblib.dump(scaler, 'standard_scaler.pkl')

# Later, to load:
# scaler = joblib.load('standard_scaler.pkl')

This way, you can ensure that your future data is scaled in the same way as your training data."""

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers= [
                        logging.FileHandler('train.log'),
                        logging.StreamHandler()
                    ])

class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, data, sequence_length, num_outputs):       
        self.data = data
        self.num_outputs = num_outputs
        self.sequence_length = sequence_length
                
        # self.target = data.df_dict[data.target_tf]['position_class'] Delete this line when proven working!
        
        # Separate MACD and RSI columns
        self.indicator_cols = [col for col in data.df_dict[data.target_tf].columns if col not in ['leverage_target', 'position_class', 'open', 'high', 'low', 'close', 'volume', 'target_leverage', 'open_pct', 'high_pct', 'low_pct', 'close_pct']]
        self.value_cols = ['open_pct', 'high_pct', 'low_pct', 'close_pct']
        print(f"Indicator columns: {self.indicator_cols}")
        self.valid_indices = self.get_valid_indices()

    def encode_time_cyclical(self, value, max_value):
        """
        Encode a time feature as sine and cosine components.
        
        :param value: The time value (e.g., hour of the day)
        :param max_value: The maximum value for this time feature (e.g., 24 for hours)
        :return: A tuple of (sin, cos) encoded values
        """
        # Convert the value to a proportion of the cycle
        angle = 2 * np.pi * value / max_value
        
        # Compute the sine and cosine components
        sine_component = np.sin(angle)
        cosine_component = np.cos(angle)
        
        return sine_component, cosine_component

    def encode_datetime(self, datetime_index):
        # ensure you are getting an array of size seq len!
        minute_sine, minute_cosine = self.encode_time_cyclical(datetime_index.minute.values, 59.0)  # Normalize to [0, 1]
        hour_sine, hour_cosine = self.encode_time_cyclical(datetime_index.hour.values, 23.0)  # Normalize to [0, 1]
        day_sine, day_cosine = self.encode_time_cyclical(datetime_index.dayofweek.values, 6.0)  # Normalize to [0, 1]
        month_sine, month_cosine = self.encode_time_cyclical(datetime_index.month.values, 12.0)  # Normalize to [0, 1]
        
        return np.column_stack((minute_sine, minute_cosine, hour_sine, hour_cosine, day_sine, day_cosine, month_sine, month_cosine))
            
    def preprocess_data(self, data_list): #data, datetime_index):
        third_dimension = 1 + 8 + 1 + 1 # 1 for base value, 8 for sin/cos, 1 for timeframe (ie 30min, 4hr...), and 1 for type (rsi, macd...)
        shape = (len(data_list), self.sequence_length * (len(self.indicator_cols) + len(self.value_cols)), third_dimension)
        tensor_of_tensors = torch.zeros(shape, dtype=torch.float32)
        for timeframe in range(len(data_list)):
            data = data_list[timeframe]
        
            open_pct = data['open_pct'].values
            high_pct  = data['high_pct'].values
            low_pct   = data['low_pct'].values
            close_pct = data['close_pct'].values
        
            # values aren't minmaxed because max value is ~+/-.33, could minmax in Data class to see results tho...
            pct_returns = np.column_stack((open_pct, high_pct, low_pct, close_pct))
            
            # ensure scaling is occuring in the correct dimension!
            # # Process indicators
            indicators = data[self.indicator_cols].values # REMEMBER, NO MIN MAX SCALING HERE!!!  ADD ME BACK??!?!?!?!?!?!?!?
            # scaler = MinMaxScaler()
            # scaled_data = scaler.fit_transform(indicators) We're passing on scaling for now...
            
            # Encode datetime
            datetime_features = torch.tensor(self.encode_datetime(data.index))
            dtf_3d = datetime_features.unsqueeze(1)
            joined = torch.cat((torch.tensor(pct_returns), torch.tensor(indicators)), dim=1)
            joined_3d = joined.unsqueeze(-1)
            
            # Below we add the time back to the sequence of values to their respective indicator for their respective timeframe
            combined = torch.cat((joined_3d, dtf_3d.expand(-1, joined_3d.shape[1], -1)), dim=-1)
            timeframe_tensor = torch.full((combined.shape[0], combined.shape[1], 1), fill_value=timeframe) #, dtype=torch.long)
            combined = torch.cat([combined, timeframe_tensor], dim=-1)
            
            index_tensor = torch.arange(combined.shape[1]).unsqueeze(0).unsqueeze(-1)
            
            index_tensor_expanded = index_tensor.expand(combined.shape[0], -1, -1)
            
            # DO NOT DELETE THIS COMMENT:
            # Result is of shape (sequence_length, input values, vectors) in this case, (32, 484, 6) the six are at place 0, the value, 1-4, time embeddings, and 5th/lastly, the index tensor, which acts as the type embedding as we treat all 484 items as unique types.
            result = torch.cat([combined, index_tensor_expanded], dim=-1)
            
            final_tensor = result.reshape(-1, result.shape[-1]).to(dtype=torch.float32)
            
            tensor_of_tensors[timeframe] = final_tensor
        
        return tensor_of_tensors #you got the right binary embeddings now on dimension 2. Now you gotta pass it to the model and see how you do that. continue here: https://chatgpt.com/c/819e12b2-0b6b-46ad-b5ba-562f793ddef5 Don't worry about embedding macdlen and rsi len. Though, you may consider normalizing the values of RSI and MACD at the moment of creation so that they are scaled appropriately between 0 and 1.
    
    def get_valid_indices(self):
        # Calculate timedelta_high and round it up to the nearest integer
        timedelta_high = ceil(pd.Timedelta(self.data.highest_timeframe) / pd.Timedelta(self.data.target_tf))
        
        first_index = timedelta_high + self.sequence_length
        last_index = len(self.data.df_dict[self.data.highest_timeframe])# - first_index
        
        return range(first_index, last_index)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        try:
            real_idx = self.valid_indices[idx]
            datetime_index_at_idx = self.data.df_dict[self.data.highest_timeframe].index[real_idx]
            
            df_list = []
            for tf_key, delta in self.data.tf_tdelta_tuple: # Remember, we need delta to prevent look ahead for next tf data.
                start_position = self.data.df_dict[tf_key].index.get_loc(datetime_index_at_idx) + 1 # +1 or else time is off by one, idk why.
                data = self.data.df_dict[tf_key].iloc[start_position-self.sequence_length-delta:start_position-delta]
                df_list.append(data)
                if tf_key == self.data.target_tf:
                    # Target tf and lower tfs won't have delta btw.
                    target = self.data.df_dict[tf_key].iloc[start_position:start_position+self.num_outputs]['position_class']
                
            processed_data = self.preprocess_data(df_list)
            
            return {
                'input': torch.FloatTensor(processed_data),
                'target': torch.LongTensor(target.values),
                'timeframe': target.index,
            }
        except Exception as e:
            print(f"Error in __getitem__ at idx {idx}: {str(e)}")
            print(f"valid_indices[{idx}] = {self.valid_indices[idx]}")
            # print(f"data_30min shape: {self.data_30min.shape}")
            # print(f"data_4hr shape: {self.data_4hr.shape}")
            # print(f"real_idx_30min: {real_idx_30min}, real_idx_4hr: {real_idx_4hr}")
            raise

    def set_model_type(self, model_type):
        if isinstance(model_type, ModelType):
            self.model_type = model_type
        else:
            raise ValueError("model_type must be an instance of ModelType")

def custom_collate(batch):
    elem = batch[0]
    batch = {key: [d[key] for d in batch] for key in elem}
    
    for key in batch:
        if isinstance(batch[key][0], torch.Tensor):
            batch[key] = torch.stack(batch[key])
        elif isinstance(batch[key][0], (int, float)):
            batch[key] = torch.tensor(batch[key])
    
    return batch

def get_data_loaders(data, config):
    dataset = TimeSeriesDataset(data, config['model']['sequence_length'], config['model']['num_outputs'])

    total_samples = len(dataset)
    train_size = int(0.7 * total_samples)
    
    train_dataset = Subset(dataset, range(train_size))
    val_dataset = Subset(dataset, range(train_size, total_samples))

    train_sampler = RandomSampler(train_dataset, replacement=False, num_samples=len(train_dataset))
    val_sampler = RandomSampler(val_dataset, replacement=False, num_samples=len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        sampler=train_sampler,
        pin_memory=True,
        num_workers=config['training']['num_workers'],
        persistent_workers=True,
        collate_fn=custom_collate
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        sampler=val_sampler,
        pin_memory=True,
        num_workers=config['training']['num_workers'],
        persistent_workers=True,
        collate_fn=custom_collate
    )
    
    return train_loader, val_loader

def calculate_rsi(df, periods=[14]):
    for period in periods:
        df[f"rsi_high_{period}"] = ta.rsi(df["high"], length=period)
        df[f"rsi_low_{period}"] = ta.rsi(df["low"], length=period)
        df[f"rsi_close_{period}"] = ta.rsi(df["close"], length=period)
        
        # # Apply min-max scaling to each RSI column
        # for col in [f"rsi_high_{period}", f"rsi_low_{period}", f"rsi_close_{period}"]:
        #     min_val = df[col].min()
        #     max_val = df[col].max()
        #     df[col] = (df[col] - min_val) / (max_val - min_val)
        
    return df

def calculate_macd(df, fast_periods=[12], slow_periods=[26], signal_periods=[9]):
    for fast in fast_periods:
        for slow in slow_periods:
            if fast < slow:  # Ensure fast period is smaller than slow period
                for signal in signal_periods:
                    # Categories, MACD, MACDh, MACDs
                    macd = ta.macd(close=df["close"], fast=fast, slow=slow, signal=signal)
                    df = df.join(macd)
                    
                    # # Apply min-max scaling to each MACD column
                    # for col in macd.columns:
                    #     min_val = df[col].min()
                    #     max_val = df[col].max()
                    #     df[col] = (df[col] - min_val) / (max_val - min_val)
    return df

def value_to_class(value):
    if -10 <= value <= 10:
        return 2  # Neutral position
    elif -100 <= value < -10:
        return 1  # Short position
    elif 10 < value <= 100:
        return 0  # Long position
    else:
        raise ValueError(f"Value {value} is out of the expected range [-100, 100]")

def series_to_classes(series):
    return series.apply(value_to_class)
# add back volume!
class Data:
    def __init__(self, timeframes, target_tf, dataframe=None):
        self.timeframes = timeframes
        self.target_tf = target_tf
        self.highest_timeframe = self.get_highest_timeframe()        
        self.tf_tdelta_tuple = self.get_timedeltas()
        
        self.rsi_period = range(7, 15, 7)
        self.macd_period = range(9, 10, 3)
        self.macd_fast_periods = range(12, 13, 4)
        self.macd_slow_periods = range(26, 27, 12)
        
        self.df_dict = self.get_data(dataframe=dataframe)#start_date='2020-07-01', end_date='2021-04-01')
        
        self.data_types = len([col for col in self.df_dict[self.target_tf].columns if col not in ['leverage_target', 'position_class']])
    
    def get_highest_timeframe(self):
        # Convert the timeframes to Timedelta objects
        timedeltas = [pd.Timedelta(t) for t in self.timeframes]
        # Find the maximum Timedelta and its corresponding original timeframe
        max_timedelta = max(timedeltas)
        return self.timeframes[timedeltas.index(max_timedelta)]
    
    def get_timedeltas(self):
        "Justification: We need to do this crap because when there are timeframes greater than the target, the close time will be leaking information!"
        " For example, if a timeframe is 4 hours and we have a 30-minute target tf, and the latest token is midnight, "
        " The time is 00:00:00, and the close time will be 03:59:59... but the target of 30-minutes will be looking to predict 00:30:00!"
        " To solve this we will use the delta between the target tf and each timeframe, and subtract that delta for the given timeframe indice(s)."
        tf_tdelta_tuple = []
        for timeframe in self.timeframes:
            tf_td = pd.Timedelta(timeframe)
            target_tf_td = pd.Timedelta(self.target_tf)
            delta = 0 if tf_td <= target_tf_td else int(tf_td // target_tf_td)
            tf_tdelta_tuple.append((timeframe, delta))
        return tf_tdelta_tuple
    
    def get_data(self, filepath='data/sorted_data_1min3.parquet', dataframe=None, start_date=None, end_date=None):
        print("Caution: Will need to preserve scale of data when running on realtime data if using global min-max scaling.")
        print("Remember: on a 4hr scale with idex of 2021-01-01 00:00:00', the close time will always be 03:59:59. Ensure that timeframes larger than the target are not leaking information!")
        if dataframe is not None and not dataframe.empty:
            df = dataframe.copy()
            df.index = pd.to_datetime(df.index)
        else:
            df = pd.read_parquet(filepath)
            df.index = pd.to_datetime(df.index)
        
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        newest_old_timestamp = None
        resampled_df_tfs_dict = {}
        target_td = pd.Timedelta(self.target_tf)
        for timeframe in self.timeframes:
            resampled_df_list = []
            interaval = int(pd.Timedelta(timeframe) // target_td)
            interaval = 1 if interaval < 1 else interaval
            
            for i in range(interaval):
                offset = i * target_td                
                data = df.resample(timeframe, offset=offset).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                resampled_df_list.append(data)
                
            # Concatenate all DataFrames for different offsets
            resampled_df = pd.concat(resampled_df_list).sort_index()
            resampled_df = calculate_rsi(resampled_df, periods=self.rsi_period)
            resampled_df = calculate_macd(resampled_df, fast_periods=self.macd_fast_periods, slow_periods=self.macd_slow_periods, signal_periods=self.macd_period)
            resampled_df.dropna(inplace=True)
            
            long_leverage  = (resampled_df["low"]  - (resampled_df["close"]- resampled_df["open"]))  / (resampled_df["close"]- resampled_df["open"])
            short_leverage = (resampled_df["high"] - (resampled_df["open"] - resampled_df["close"])) / (resampled_df["open"] - resampled_df["close"])
            long_leverage = long_leverage.replace([np.inf, -np.inf], 0)
            short_leverage = short_leverage.replace([np.inf, -np.inf], 0)
            long_leverage[long_leverage < 0] = 0
            short_leverage[short_leverage < 0] = 0
            
            resampled_df["leverage_target"] = np.where(
                resampled_df["close"] > resampled_df["open"],
                long_leverage,
                np.where(
                    resampled_df["close"] < resampled_df["open"],
                    -short_leverage, # ENSURE SHORT IS NEGATIVE
                    0
                )
            )
            # INSTEAD OF CLIPPING consider, SCALE to 100 rather than CLIP!
            resampled_df["leverage_target"] = np.clip(resampled_df["leverage_target"], -100, 100)
            resampled_df["position_class"] = series_to_classes(resampled_df["leverage_target"])
            
            long = (resampled_df["position_class"] == 0).mean() * 100
            print(f"{timeframe} Percentage of position_class that is Long: {long:.2f}%")
            short = (resampled_df["position_class"] == 1).mean() * 100
            print(f"{timeframe} Percentage of position_class that is Short: {short:.2f}%")
            neutral = (resampled_df["position_class"] == 2).mean() * 100
            print(f"{timeframe} Percentage of position_class that is Neutral: {neutral:.2f}%")
            
            # Consider getting the normalized value, simliar to this, but with pct: df['Open_norm'] = (df['Open'] - df['Open'].min()) / (df['Open'].max() - df['Open'].min())
            # Assuming df is your DataFrame with columns: 'open', 'high', 'low', 'close'
            resampled_df['open_pct'] = (resampled_df['open'] - resampled_df['open'].shift(1)) / resampled_df['open'].shift(1)
            resampled_df['high_pct'] = (resampled_df['high'] - resampled_df['open']) / resampled_df['open']
            resampled_df['low_pct']  = (resampled_df['low'] - resampled_df['open']) / resampled_df['open']
            resampled_df['close_pct'] = (resampled_df['close'] - resampled_df['open']) / resampled_df['open']
            
            print(f"{timeframe} open_pct min, max: {resampled_df['open_pct'].min()}, {resampled_df['open_pct'].max()}")
            print(f"{timeframe} high_pct min, max: {resampled_df['high_pct'].min()}, {resampled_df['high_pct'].max()}")
            print(f"{timeframe} low_pct min, max: {resampled_df['low_pct'].min()}, {resampled_df['low_pct'].max()}")
            print(f"{timeframe} close_pct min, max: {resampled_df['close_pct'].min()}, {resampled_df['close_pct'].max()}\n")
            
            # List of columns to scale
            columns_to_scale = ['open_pct', 'high_pct', 'low_pct', 'close_pct']

            # Create a StandardScaler object
            scaler = StandardScaler()
            
            for column in columns_to_scale: # 4 loop is dumb, but oh well.
                resampled_df[column] = scaler.fit_transform(resampled_df[column].values.reshape(-1, 1))
                
            for col in resampled_df.columns:
                if col not in ['leverage_target', 'position_class'] and col not in columns_to_scale and col not in ['open', 'high', 'low', 'close', 'volume']:
                    resampled_df[col]
            
            resampled_df.dropna(inplace=True) # drop nan again
            
            old_timestamp = resampled_df.index.min()
            if newest_old_timestamp is None or old_timestamp > newest_old_timestamp:
                newest_old_timestamp = old_timestamp
            resampled_df_tfs_dict[f"{timeframe}"] = resampled_df.astype(np.float32)
            
            # TRY IMPLEMENTING STANDARD DEVIATION SCALING HERE (StandardScaler from sklearn.preprocessing)
            
        for key in resampled_df_tfs_dict.keys():
            tfdf = resampled_df_tfs_dict[key]
            
            oldest_timestamp = newest_old_timestamp + pd.Timedelta(self.timeframes[-1])
            
            # Filter the data to exclude data points before the oldest timestamp
            tfdf = tfdf[tfdf.index >= oldest_timestamp]
            resampled_df_tfs_dict[key] = tfdf
        
        # save_path = "data/RSI-n-MACD_30min_4hr_combined.parquet" IMPLEMENT ME AGAIN FAM!
        # if save_path:
        #     # Save both dataframes to a single parquet file
        #     combined_df = pd.concat([data_30min_filtered, data_4hr_shifted], keys=['30min', '4hr'])
        #     combined_df.to_parquet(save_path)
        #     print(f"Data saved to {save_path}")
        
        return resampled_df_tfs_dict

# def load_processed_data(filepath, start_date=None, end_date=None):
#     # Load the combined dataframe
#     combined_df = pd.read_parquet(filepath)
    
#     # Check if the DataFrame has a multi-index
#     if not isinstance(combined_df.index, pd.MultiIndex):
#         raise ValueError("The loaded data does not have a multi-index as expected.")

#     # Convert the second level of the index to datetime if not already
#     combined_df.index = combined_df.index.set_levels(pd.to_datetime(combined_df.index.levels[1]), level=1)
    
#     # Filter by date range if specified
#     if start_date:
#         combined_df = combined_df[combined_df.index.get_level_values(1) >= pd.to_datetime(start_date)]
#     if end_date:
#         combined_df = combined_df[combined_df.index.get_level_values(1) <= pd.to_datetime(end_date)]
    
#     # Split the combined dataframe back into 30min and 4hr dataframes
#     data_30min = combined_df.loc['30min']
#     data_4hr = combined_df.loc['4hr']
    
#     return data_30min, data_4hr

def denormalize_output(predicted_returns, last_price):
    # Implement scaler to denormalize the predicted returns???
    predicted_prices = last_price * torch.exp(predicted_returns)
    return predicted_prices

# if __name__ == '__main__':
    # config = load_config('config.yaml')
    # logging.info(f"Starting training with configuration: {config}")
    
    # # Load and preprocess data
    # data_30min, data_4hr = Data() # get_data(config['data']['filepath']) #load_processed_data('data/RSI-n-MACD_30min_4hr_combined.parquet')   #get_data(config['data']['filepath']) #, start_date='2020-07-01', end_date='2021-07-01')
    # train_data_loader, val_data_loader = get_data_loaders(data_30min, data_4hr, config)
    
implment try catch for this error:
    Traceback (most recent call last):
  File "/opt/anaconda3/envs/bot_env/lib/python3.10/site-packages/urllib3/connectionpool.py", line 789, in urlopen
    response = self._make_request(
  File "/opt/anaconda3/envs/bot_env/lib/python3.10/site-packages/urllib3/connectionpool.py", line 536, in _make_request
    response = conn.getresponse()
  File "/opt/anaconda3/envs/bot_env/lib/python3.10/site-packages/urllib3/connection.py", line 464, in getresponse
    httplib_response = super().getresponse()
  File "/opt/anaconda3/envs/bot_env/lib/python3.10/http/client.py", line 1375, in getresponse
    response.begin()
  File "/opt/anaconda3/envs/bot_env/lib/python3.10/http/client.py", line 318, in begin
    version, status, reason = self._read_status()
  File "/opt/anaconda3/envs/bot_env/lib/python3.10/http/client.py", line 287, in _read_status
    raise RemoteDisconnected("Remote end closed connection without"
http.client.RemoteDisconnected: Remote end closed connection without response

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/andrew/.local/lib/python3.10/site-packages/requests/adapters.py", line 486, in send
    resp = conn.urlopen(
  File "/opt/anaconda3/envs/bot_env/lib/python3.10/site-packages/urllib3/connectionpool.py", line 843, in urlopen
    retries = retries.increment(
  File "/opt/anaconda3/envs/bot_env/lib/python3.10/site-packages/urllib3/util/retry.py", line 474, in increment
    raise reraise(type(error), error, _stacktrace)
  File "/opt/anaconda3/envs/bot_env/lib/python3.10/site-packages/urllib3/util/util.py", line 38, in reraise
    raise value.with_traceback(tb)
  File "/opt/anaconda3/envs/bot_env/lib/python3.10/site-packages/urllib3/connectionpool.py", line 789, in urlopen
    response = self._make_request(
  File "/opt/anaconda3/envs/bot_env/lib/python3.10/site-packages/urllib3/connectionpool.py", line 536, in _make_request
    response = conn.getresponse()
  File "/opt/anaconda3/envs/bot_env/lib/python3.10/site-packages/urllib3/connection.py", line 464, in getresponse
    httplib_response = super().getresponse()
  File "/opt/anaconda3/envs/bot_env/lib/python3.10/http/client.py", line 1375, in getresponse
    response.begin()
  File "/opt/anaconda3/envs/bot_env/lib/python3.10/http/client.py", line 318, in begin
    version, status, reason = self._read_status()
  File "/opt/anaconda3/envs/bot_env/lib/python3.10/http/client.py", line 287, in _read_status
    raise RemoteDisconnected("Remote end closed connection without"
urllib3.exceptions.ProtocolError: ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/andrew/Programming/BOT-main/BOT/bot.py", line 324, in <module>
    main()
  File "/home/andrew/Programming/BOT-main/BOT/bot.py", line 291, in main
    dydx_trading_price = dydx.fetch_eth_price()
  File "/home/andrew/Programming/BOT-main/BOT/DYDXInterface/dydx_interface.py", line 216, in fetch_eth_price
    eth_data = self.fetch_eth_market_data()
  File "/home/andrew/Programming/BOT-main/BOT/DYDXInterface/dydx_interface.py", line 206, in fetch_eth_market_data
    market_data_response = self.client.public.get_markets()
  File "/opt/anaconda3/envs/bot_env/lib/python3.10/site-packages/dydx3/modules/public.py", line 84, in get_markets
    return self._get(uri, {'market': market})
  File "/opt/anaconda3/envs/bot_env/lib/python3.10/site-packages/dydx3/modules/public.py", line 19, in _get
    return request(
  File "/opt/anaconda3/envs/bot_env/lib/python3.10/site-packages/dydx3/helpers/requests.py", line 24, in request
    response = send_request(
  File "/opt/anaconda3/envs/bot_env/lib/python3.10/site-packages/dydx3/helpers/requests.py", line 43, in send_request
    return getattr(session, method)(uri, headers=headers, **kwargs)
  File "/home/andrew/.local/lib/python3.10/site-packages/requests/sessions.py", line 602, in get
    return self.request("GET", url, **kwargs)
  File "/home/andrew/.local/lib/python3.10/site-packages/requests/sessions.py", line 589, in request
    resp = self.send(prep, **send_kwargs)
  File "/home/andrew/.local/lib/python3.10/site-packages/requests/sessions.py", line 703, in send
    r = adapter.send(request, **kwargs)
  File "/home/andrew/.local/lib/python3.10/site-packages/requests/adapters.py", line 501, in send
    raise ConnectionError(err, request=request)
requests.exceptions.ConnectionError: ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))