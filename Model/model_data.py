import logging
import numpy as np
from torch.utils.data import RandomSampler
import torch
# from ..config import load_config
# from model import ModelType
from torch.utils.data import DataLoader, random_split
import pandas_ta as ta
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from torch.utils.data import Subset, Sampler, SequentialSampler
from math import ceil
import sys
import os
# Add the parent directory to sys.path#+
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))#+
sys.path.append(parent_dir)#+
from config import load_config
config = load_config('config.yaml')
from Model.reward import Reward_Simple
# import warnings
import copy

# warnings.filterwarnings("ignore", message="No module named 'imp'", category=ImportWarning)




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
                        logging.FileHandler('model_data.log'),
                        logging.StreamHandler()
                    ])

class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, data, sequence_length, num_outputs):#, epoch, valid_indices, data_type):       
        # self.live_trading = live_trading
        self.data = data
        self.num_outputs = num_outputs
        self.sequence_length = sequence_length
        # self.type = data_type
                
        # self.target = data.df_dict[data.target_tf]['position_class'] Delete this line when proven working!
        
        # Separate MACD and RSI columns
        self.indicator_cols = [col for col in data.df_dict[data.target_tf].columns if col not in ['volatility','leverage_target', 'position_class', 'open', 'high', 'low', 'close', 'volume', 'target_leverage', 'open_pct', 'high_pct', 'low_pct', 'close_pct', 'weights']]
        self.value_cols = ['open_pct', 'high_pct', 'low_pct', 'close_pct']
        print(f"Indicator columns: {self.indicator_cols}")
        self.valid_indices = self.get_valid_indices(len(data))
        
        # Suppose you have num_sequences sequences
        self.batch_size = config['training']['batch_size']
        self.reward_objects = [Reward_Simple() for _ in range(self.batch_size)]
        self.tf_overlap_price = [0 for _ in range(self.batch_size)]
        # self.epoch = epoch

    def decode_time_cyclical(sine, cosine, max_value): #https://chatgpt.com/c/672f3420-9874-8006-9513-26f9cf65b0ff
        # Calculate the angle
        angle = np.arctan2(sine, cosine)
        
        # Convert the angle back to the value in the original range
        value = angle * max_value / (2 * np.pi)
        
        # Adjust if the value is negative to ensure it's within [0, max_value]
        if value < 0:
            value += max_value
        
        return round(value)  # Round to get the closest integer

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
        month_sine, month_cosine = self.encode_time_cyclical(datetime_index.month.values, 12.0)  # Normalize to [0, 1] 11.0 instead of 12?
        
        return np.column_stack((minute_sine, minute_cosine, hour_sine, hour_cosine, day_sine, day_cosine, month_sine, month_cosine))
        
    def preprocess_data(self, data_list): #data, datetime_index):
        third_dimension = 1 + 8 + 1 + 1 # 1 for base value, 8 for sin/cos, 1 for timeframe (ie 30min, 4hr...), and 1 for type (rsi, macd...)
        shape = (len(data_list), self.sequence_length * (len(self.indicator_cols) + len(self.value_cols)), third_dimension)
        tensor_of_tensors = torch.zeros(shape, dtype=torch.float32)
        for timeframe in range(len(data_list)):
            data = data_list[timeframe]
        
            open_pct  = data['open_pct'].values
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
    
    def get_valid_indices(self, valid_indices):
        # if not self.data.live_trading:
        #     # Calculate timedelta_high and round it up to the nearest integer
        #     timedelta_high = ceil(pd.Timedelta(self.data.highest_timeframe) / pd.Timedelta(self.data.shortest_timeframe))
            
        #     first_index = timedelta_high * self.sequence_length
        #     last_index = (len(valid_indices) - self.data.future_ticks ) - ((len(valid_indices) - self.data.future_ticks) % config['training']['batch_size'])
            
        #     return range(first_index, last_index)
        # else:
        timedelta_high = ceil(pd.Timedelta(self.data.highest_timeframe) / pd.Timedelta(self.data.shortest_timeframe))
        
        first_index = len(valid_indices) - (timedelta_high * self.sequence_length)
        last_index = len(valid_indices)
        
        return range(first_index, last_index)

    def __len__(self):
        return len(self.valid_indices)#- self.data.future_ticks #config['data']['future_sample_ticks'] # - 1 removed - 1 as _i_think_ config['data']['future_sample_ticks'] should cover that and then some

    def __getitem__(self, idx):
        try:
            batch_id = idx * self.batch_size // len(self)
            real_idx = self.valid_indices[idx]
            # next_real_idx = real_idx + 1

            datetime_index_at_idx = self.data.df_dict[self.data.shortest_timeframe].index[real_idx]
            # next_datetime_index_at_idx = self.data.df_dict[self.data.shortest_timeframe].index[next_real_idx]

            # print(len(self.valid_indices))
            raw_data = None
            df_list = []
            # next_df_list = []

            for tf_key, delta in self.data.tf_tdelta_tuple:
                tf_index = self.data.df_dict[tf_key].index

                # **Process Current Data**
                # Check if datetime_index_at_idx exists in the index
                if datetime_index_at_idx in tf_index:
                    start_position = tf_index.get_loc(datetime_index_at_idx) + 1
                else:
                    # Find the index of the previous timestamp
                    closest_idx = tf_index.get_indexer([datetime_index_at_idx], method='pad')[0]
                    if closest_idx == -1:
                        closest_idx = 0
                        print("we broke it")
                        exit()
                    start_position = closest_idx + 1

                data = self.data.df_dict[tf_key].iloc[start_position - self.sequence_length - delta : start_position - delta]
                df_list.append(data)

                # **Process Next Data**
                # Check if next_datetime_index_at_idx exists in the index
                # if next_datetime_index_at_idx in tf_index:
                #     next_start_position = tf_index.get_loc(next_datetime_index_at_idx) + 1
                # else:
                #     # Find the index of the previous timestamp
                #     closest_idx = tf_index.get_indexer([next_datetime_index_at_idx], method='pad')[0]
                #     if closest_idx == -1:
                #         closest_idx = 0
                #         print("we broke it")
                #         exit()
                #     next_start_position = closest_idx + 1

                # next_data = self.data.df_dict[tf_key].iloc[next_start_position - self.sequence_length - delta : next_start_position - delta]
                # next_df_list.append(next_data)

                # **Additional Processing Based on Timeframe Key**
                # if tf_key == self.data.target_tf:
                #     next_n_opens = self.data.df_dict[tf_key].iloc[start_position - delta : start_position - delta + self.data.future_ticks]['open']
                #     volatility = data['volatility'].iloc[-1]

                # if tf_key == self.data.shortest_timeframe:
                #     lowest_tf_next_n_opens = self.data.df_dict[tf_key].iloc[start_position - delta : start_position - delta + self.data.future_ticks]['open']
                #     raw_data = data.iloc[-1][["open", "high", "low", "close"]]

            # **Process the Data**
            processed_data = self.preprocess_data(df_list)
            # next_processed_data = self.preprocess_data(next_df_list)

            # lowest_tf_and_target_tf_index_0_overlap = lowest_tf_next_n_opens.index[0].to_pydatetime().minute % 30 == 0 #CHANGE BACK TO THIS! parse target_tf to int rather than string. self.data.target_tf == 0 # MAKE 30 A CONFIG PARAMETER OF TARGET TF or somethin
            # ticks_till_next_target_tf = count_five_minute_ticks(lowest_tf_next.index[0].to_pydatetime())
            
            # add the target tf open if we are on a harmonic time of the target tf and not the lowest tf.
            # if lowest_tf_and_target_tf_index_0_overlap:
            #     self.tf_overlap_price[batch_id] = lowest_tf_next_n_opens.iloc[0]
            # elif not lowest_tf_and_target_tf_index_0_overlap and self.tf_overlap_price[batch_id] != 0:
            #     next_n_opens = np.insert(next_n_opens, 0, self.tf_overlap_price[batch_id])
            
            if self.reward_objects[batch_id].current_position is not None:
                current_position = 0 if self.reward_objects[batch_id].current_position.position_type == 'long' else 1
            else:
                current_position = 2
                
            # next_n_opens = np.array(next_n_opens[:self.data.future_ticks])
            # self.reward_objects[batch_id].next(lowest_tf_open_array=np.array(lowest_tf_next_n_opens), goal_tf_open_array=next_n_opens, target_tf_open=lowest_tf_next_n_opens.iloc[0], index=real_idx)
            
            
            response = {
                'input': torch.FloatTensor(processed_data),
                'raw_data': raw_data,
                # 'reward_object': self.reward_objects[batch_id], # comment me out?
                'current_positions': current_position,
            }
            
            return response

        except Exception as e:
            print(f"An error occurred: {e}")
            raise

class ParallelSequenceSampler(Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size
        self.total_length = len(data_source)
        self.chunk_size = self.total_length // batch_size

        # Create indices for each sequence (chunk)
        self.indices = []
        for i in range(batch_size):
            start_idx = i * self.chunk_size
            end_idx = start_idx + self.chunk_size
            self.indices.append(list(range(start_idx, end_idx)))

        # Interleave the indices
        self.interleaved_indices = []
        for idx_tuple in zip(*self.indices):
            for idx in idx_tuple:
                self.interleaved_indices.append(idx)

    def __iter__(self):
        return iter(self.interleaved_indices)

    def __len__(self):
        return self.chunk_size * self.batch_size  # Total number of indices

# def custom_collate(batch):
#     elem = batch[0]
#     batch = {key: [d[key] for d in batch] for key in elem}
    
#     for key in batch:
#         if isinstance(batch[key][0], torch.Tensor):
#             batch[key] = torch.stack(batch[key])
#         elif isinstance(batch[key][0], (int, float)):
#             batch[key] = torch.tensor(batch[key])
    
#     return batch

def custom_collate(batch):
    elem = batch[0]
    batch = {key: [d[key] for d in batch] for key in elem}
    
    for key in batch:
        if key != 'reward_object':
            first_elem = batch[key][0]
            if first_elem is None:
                batch[key] = batch[key]  # Keep the list as is or handle None values appropriately
            elif isinstance(first_elem, torch.Tensor):
                batch[key] = torch.stack(batch[key])
            elif isinstance(first_elem, (int, float)):
                batch[key] = torch.tensor(batch[key])
            elif isinstance(first_elem, np.ndarray):
                batch[key] = torch.stack([torch.from_numpy(x) for x in batch[key]])
            elif isinstance(first_elem, (pd.DataFrame, pd.Series)):
                if key == 'lowest_tf_next_n_opens': # This should go into this if statment if key != 'reward_object':
                    batch[key] = torch.stack([torch.tensor(x.values) for x in batch[key]])
                else:
                    batch[key] = batch[key]
            else:
                # Handle other data types or raise an error
                raise TypeError(f"Unsupported data type in batch for key '{key}': {type(first_elem)}")
        else:
            # Keep list of reward_objects as is
            pass
        
    return batch

def get_data_loaders(data, config, epoch):
    total_samples = len(data)
    train_size = int(0.8 * total_samples)
    
    train_dataset = TimeSeriesDataset(data, config['model']['sequence_length'], config['model']['num_outputs'], epoch, range(train_size), data_type='train') # data_type... really dood?
    val_dataset = TimeSeriesDataset(data, config['model']['sequence_length'], config['model']['num_outputs'], epoch, range(train_size, total_samples), data_type='val')
    
    batch_size = config['training']['batch_size']

    train_sampler = ParallelSequenceSampler(train_dataset, batch_size)
    val_sampler = SequentialSampler(val_dataset) #, batch_size=1)  # For validation, you might use batch_size=1

    # train_sampler = RandomSampler(train_dataset, replacement=False, num_samples=len(train_dataset))
    # val_sampler = RandomSampler(val_dataset, replacement=False, num_samples=len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        # prefetch_factor=None, #It seems that having a worker causes some issue with creating all 4 self.future_states. I think it may be because the worker is trying to access the same data as the main process
        # For debugging this odd issue, see DataPrefetcher preload()
        # IF I'M UNABLE TO FIX THIS I NEED TO STOP THE FOR ACTION IN RANGE(4) LOOP! and instead just run/call the simulate_environment_step from the main loop.
        
        batch_size=batch_size,  # Now specify the batch size here
        sampler=train_sampler,
        shuffle=False,
        pin_memory=True,
        # num_workers=config['training']['num_workers'],
        # persistent_workers=True,
        collate_fn=custom_collate
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        sampler=val_sampler,
        shuffle=False,
        pin_memory=True,
        # num_workers=config['training']['num_workers'],
        # persistent_workers=True,
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
                    # df.join(macd)
                    df = df.join(macd) #don't think we need to do df = part...
                    
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
    def __init__(self,
                 config,
                 interpolate_to_target=True,
                 start_date=None,
                 end_date=None,
                 scaler=MinMaxScaler(), # Got to find a way to just pass this as a text object.
                 live_trading=False,
                 dataframe=None,
                 ):
        """interpolate_to_target if true will fill higher level timeframes to the target timeframe in an interpolated manner.
        For instance, if True and target_tf is '30min', the 4-hour timeframe will have 4hr values on a rolling 30-minute basis.
        """
        self.live_trading = live_trading
        self.sequence_length = config['model']['sequence_length']
        self.interpolate_to_target = interpolate_to_target
        self.timeframes = config['data']['resample_timeframes']
        self.target_tf = config['data']['target_tf']
        self.highest_timeframe = self.get_highest_timeframe()
        self.shortest_timeframe = self.get_shortest_timeframe()
        self.tf_tdelta_tuple = self.get_timedeltas()
        self.future_ticks = config['data']['future_sample_ticks']
        
        self.rsi_period = range(7, 15, 7)
        self.macd_period = range(9, 10, 3)
        self.macd_fast_periods = range(12, 13, 4)
        self.macd_slow_periods = range(26, 27, 12)
        
        self.df_dict = self.get_data(
            start_date=start_date,
            end_date=end_date,
            interpolate_to_target=interpolate_to_target,
            scaler=scaler,
            dataframe=dataframe,
            )
        
        self.data_types = len([col for col in self.df_dict[self.target_tf].columns if col not in ['leverage_target', 'position_class', 'weights']])
        self.valid_indices = self.get_valid_indices()
    
    def get_valid_indices(self):
        # Calculate timedelta_high and round it up to the nearest integer
        timedelta_high = ceil(pd.Timedelta(self.highest_timeframe) / pd.Timedelta(self.shortest_timeframe))
        # timedelta_high = ceil(pd.Timedelta(self.highest_timeframe) / pd.Timedelta(self.target_tf))
        
        # Multiply the timedelta_high by the sequence length to get the first index
        # Subtract the sequence length of the target timeframes from the first index to get the last index
        first_index = self.sequence_length if self.interpolate_to_target else timedelta_high + self.sequence_length # * or + here?
        last_index = ceil(len(self.df_dict[self.shortest_timeframe]))
        # last_index = ceil(len(self.df_dict[self.target_tf]))
        
        return range(first_index, last_index)
    
    def get_highest_timeframe(self):
        # Convert the timeframes to Timedelta objects
        timedeltas = [pd.Timedelta(t) for t in self.timeframes]
        # Find the maximum Timedelta and its corresponding original timeframe
        max_timedelta = max(timedeltas)
        return self.timeframes[timedeltas.index(max_timedelta)]
    
    def get_shortest_timeframe(self):
        # Convert the timeframes to Timedelta objects
        timedeltas = [pd.Timedelta(t) for t in self.timeframes]
        # Find the minimum Timedelta and its corresponding original timeframe
        min_timedelta = min(timedeltas)
        return self.timeframes[timedeltas.index(min_timedelta)]
    
    def get_timedeltas(self):
        "Justification: We need to do this crap because when there are timeframes greater than the target, the close time will be leaking information!"
        " For example, if a timeframe is 4 hours and we have a 30-minute target tf, and the latest token is midnight, "
        " The time is 00:00:00, and the close time will be 03:59:59... but the target of 30-minutes will be looking to predict 00:30:00!"
        " To solve this we will use the delta between the target tf and each timeframe, and subtract that delta for the given timeframe indice(s)."
        tf_tdelta_tuple = []
        if self.interpolate_to_target:
            for timeframe in self.timeframes:
                tf_td = pd.Timedelta(timeframe)
                target_tf_td = pd.Timedelta(self.target_tf)
                delta = 0 if tf_td <= target_tf_td else int(tf_td // target_tf_td)
                tf_tdelta_tuple.append((timeframe, delta))
        else:
            for timeframe in self.timeframes:
                tf_td = pd.Timedelta(timeframe)
                target_tf_td = pd.Timedelta(self.target_tf)
                delta = 0 if tf_td <= target_tf_td else 1 # Move back 1 tf to prevent leaking information/look ahead. See above comment for explanation.
                tf_tdelta_tuple.append((timeframe, delta))
        return tf_tdelta_tuple
    
    def get_data(self,
                 filepath='data/backdata/sorted_data_1min3.parquet',
                 start_date=None,
                 end_date=None,
                 interpolate_to_target=True,
                 scaler=MinMaxScaler(),
                 dataframe=None,
                 ):
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
            interaval = 1 if interaval < 1 or not interpolate_to_target else interaval
            
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
            resampled_df.dropna(inplace=True)
            
            if self.target_tf == timeframe:
                resampled_df['volatility'] = resampled_df['open'].pct_change().rolling(window=config['data']['volatility_window']).std()
                # exit()
            
            resampled_df = calculate_rsi(resampled_df, periods=self.rsi_period)
            resampled_df.dropna(inplace=True)
            
            resampled_df = calculate_macd(resampled_df, fast_periods=self.macd_fast_periods, slow_periods=self.macd_slow_periods, signal_periods=self.macd_period)
            resampled_df.dropna(inplace=True)
            
            # open_pct_change = resampled_df["open"].pct_change()
            # normalizes range [0.8, 1.0], where the smallest value becomes 0.8 and the largest becomes 1.0.
            # resampled_df['weights'] = 0.8 + (open_pct_change - open_pct_change.min()) / (open_pct_change.max() - open_pct_change.min()) * 0.2
            
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
            
            # region disabled scaling & ohlc pct
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
            # scaler = StandardScaler()
            # scaler = MinMaxScaler()
            
            for column in columns_to_scale: # 4 loop is dumb, but oh well.
                print(resampled_df[column].head())
                # reshaped_data = resampled_df[column].values.reshape(-1, 1)
                # print(reshaped_data)
                resampled_df[column] = scaler.fit_transform(resampled_df[column].values.reshape(-1, 1))
                
            for col in resampled_df.columns:
                if col not in ['leverage_target', 'position_class'] and col not in columns_to_scale and col not in ['open', 'high', 'low', 'close', 'volume']:
                    resampled_df[col] = scaler.fit_transform(resampled_df[col].values.reshape(-1, 1))
            # endregion
            
            resampled_df.dropna(inplace=True) # drop nan again
            
            old_timestamp = resampled_df.index.min()
            if newest_old_timestamp is None or old_timestamp > newest_old_timestamp:
                newest_old_timestamp = old_timestamp
            resampled_df_tfs_dict[f"{timeframe}"] = resampled_df#.astype(np.float32)
            
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
    
    def __len__(self):
        " Return the length of the shortest (longest) timeframe DataFrame "
        return len(self.df_dict[self.shortest_timeframe])

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

if __name__ == '__main__':
    config = load_config('config.yaml')
    logging.info(f"Starting training with configuration: {config}")
    
    # Load and preprocess data
    data_30min, data_4hr = Data() # get_data(config['data']['filepath']) #load_processed_data('data/RSI-n-MACD_30min_4hr_combined.parquet')   #get_data(config['data']['filepath']) #, start_date='2020-07-01', end_date='2021-07-01')
    train_data_loader, val_data_loader = get_data_loaders(data_30min, data_4hr, config)