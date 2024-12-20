import os
import sys
import numpy as np
import copy

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from config import load_config
config = load_config('config.yaml')

import random
from datetime import datetime
import torch
import pickle
import matplotlib.pyplot as plt

from Model.model_data import *
from Model.reward import *
from model import *
from model_trainer import CumulativeReturnBatch, DataPrefetcher, Portfolio

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class Trade:
    # if model can learn, delete the rest of this comment : implement a batch size thing here to take in more than one trade at a time. Have internal way to handle being None.
    def __init__(self, entry_price, position_type, leverage, index, highest_future_price, lowest_future_price, last_goal_tf_highest_index_first):
        self.entry_price = entry_price
        self.position_type = position_type  # 'long' or 'short'
        self.leverage = leverage
        self.exit_price = None
        self.profit_pct = None
        self.effective_leverage = None
        self.hold_time = 0
        self.entry_index = index
        self.exit_index = None
        self.passed_max_profit = False
        self.indices_flipped = False
        self.last_goal_tf_highest_index_first = last_goal_tf_highest_index_first
        
        # last_goal_tf_highest_index_first = self.goal_tf_highest_index_first if not None else None
        # self.goal_tf_highest_index_first = self.goal_tf_highest_price_index < self.goal_tf_lowest_price_index
        # self.indices_flipped = last_goal_tf_highest_index_first != self.goal_tf_highest_index_first
        
        # self.goal_tf_highest_index_first = None
        # last_goal_tf_highest_index_first = self.goal_tf_highest_index_first if not None else None
        # self.goal_tf_highest_index_first = self.goal_tf_highest_price_index < self.goal_tf_lowest_price_index
        # self.indices_flipped = last_goal_tf_highest_index_first != self.goal_tf_highest_index_first
        
        # Max profit that can be made given the current window(s) of future prices, could be greater if future ticks reveal more favorable prices.
        self.max_profit_pct = self.calculate_profit_pct(highest_future_price) if position_type == 'long' else self.calculate_profit_pct(lowest_future_price)
        self.max_loss_pct = 0.0
        self.effective_leverage_list = []

    def next(self, goal_tf_highest_index_first, highest_price, lowest_price):
        # Update indices flipped
        self.indices_flipped = self.last_goal_tf_highest_index_first != goal_tf_highest_index_first
        self.last_goal_tf_highest_index_first = goal_tf_highest_index_first
        
        # Update max profit pct
        if self.position_type == 'long':
            max_profit = self.calculate_profit_pct(highest_price)
        else:
            max_profit = self.calculate_profit_pct(lowest_price)
        self.max_profit_pct = max_profit if max_profit > self.max_profit_pct else self.max_profit_pct
        
        
    def close(self, exit_price, index):
        self.exit_price = exit_price
        self.exit_index = index
        if self.position_type == 'long':
            self.profit_pct = ((self.exit_price - self.entry_price) / self.entry_price) * self.leverage * 100
        elif self.position_type == 'short':
            self.profit_pct = ((self.entry_price - self.exit_price) / self.entry_price) * self.leverage * 100

    def update_hold_time(self):
        self.hold_time += 1

    def calculate_effective_leverage(self, current_price):
        trade_size = 1000 * self.leverage  # Assuming initial balance of 1000
        if self.position_type == 'long':
            position_value = (current_price / self.entry_price) * trade_size
            profit = (current_price - self.entry_price) * trade_size / self.entry_price
            # position_num = 0
        elif self.position_type == 'short':
            position_value = (self.entry_price / current_price) * trade_size
            profit = (self.entry_price - current_price) * trade_size / self.entry_price
            # position_num = 1
        new_equity = 1000 + profit
        if new_equity != 0:
            self.effective_leverage = position_value / new_equity
        else:
            self.effective_leverage = 0.0
        self.effective_leverage_list.append(self.effective_leverage)
        return self.effective_leverage# delete:, position_num

    def calculate_profit_pct(self, current_price):
        if self.position_type == 'long':
            return ((current_price - self.entry_price) / self.entry_price) * self.leverage * 100
        elif self.position_type == 'short':
            return ((self.entry_price - current_price) / self.entry_price) * self.leverage * 100
        else:
            return 0.0

class Reward_Simple:
    def __init__(self):#, dims):
        self.profit_reward_threshold = 0.80
        self.current_position = None
    
    def next(self, lowest_tf_open_array, goal_tf_open_array, target_tf_open, index):
        self.lowest_tf_open_array = lowest_tf_open_array
        self.goal_tf_open_array = goal_tf_open_array
        self.lowest_tf_next_open = self.lowest_tf_open_array[0]#.iloc[0]
        self.target_tf_open = target_tf_open
        self.index = index

        # Find the index of the lowest and highest prices in future_prices
        self.lowest_tf_lowest_price = min(self.lowest_tf_open_array)
        self.lowest_tf_highest_price = max(self.lowest_tf_open_array)
        self.lowest_tf_lowest_price_index = np.argmin(self.lowest_tf_open_array)#.values)
        self.lowest_tf_highest_price_index = np.argmax(self.lowest_tf_open_array)#.values)
        
        self.goal_tf_lowest_price = min(self.goal_tf_open_array)
        self.goal_tf_highest_price = max(self.goal_tf_open_array)
        self.goal_tf_lowest_price_index = np.argmin(self.goal_tf_open_array)
        self.goal_tf_highest_price_index = np.argmax(self.goal_tf_open_array)
        
        self.highest_price = max(self.goal_tf_highest_price, self.lowest_tf_highest_price)
        self.lowest_price = min(self.goal_tf_lowest_price, self.lowest_tf_lowest_price)
        
        self.goal_tf_highest_index_first = None
        
        # Create deep copies of self and simulate environment steps
        # self.future_states = []

    #     for action in range(4):
    #         state_copy = copy.deepcopy(self)
    #         state_copy.simulate_environment_step(action=action)
    #         self.future_states.append(state_copy)

    # def load_future_state_and_get_environment_step(self, action, i):
    #     future_state = self.future_states[action]
    #     self.__dict__.update(future_state.__dict__)
    #     return self.environment_step
        
    def simulate_environment_step(self, action=None): #, reward_weights=(1.0, 0.1)): #, current_price, future_prices, hold_thyme): #simulate_environment_step_new
        last_goal_tf_highest_index_first = self.goal_tf_highest_index_first if not None else None
        self.goal_tf_highest_index_first = self.goal_tf_highest_price_index < self.goal_tf_lowest_price_index
        # Determine if the values have changed.
        self.indices_flipped = (
            last_goal_tf_highest_index_first is not None
            and last_goal_tf_highest_index_first != self.goal_tf_highest_index_first
        )
        
        # Determine if the nearest peak (lowest or highest price) is within the array of self.lowest_tf_next_open.
        # We know this is true if the goal_tf_highest_price_index or goal_tf_lowest_price_index is 0, as the lowest tf will always be between 'this' time and the next index.
        self.nearest_peak_within_lowest_tf = (
            self.goal_tf_lowest_price_index == 0  or
            self.goal_tf_highest_price_index == 0
        )
        
        if self.current_position is not None:
            # current_position.passed_max_profit = current_position.exit_index < self.goal_tf_highest_price_index
            self.current_position.next(self.goal_tf_highest_index_first, self.highest_price, self.lowest_price)
        
        # pct_change = None
        profit = 0.0
        hold_time = 0
        effective_leverage = 0.0
        
        # Compute rewards for all actions
        long_reward  = self.calculate_long_reward() #copy.deepcopy(current_position), current_price, index, future_prices[0])
        short_reward = self.calculate_short_reward()
        hold_reward  = self.calculate_hold_reward()
        close_reward = self.calculate_close_reward()

        rewards = [long_reward, short_reward, hold_reward, close_reward]
        
        # Get the clipped reward for the chosen action
        reward = rewards[action]
        # Compute effective leverage
        if self.current_position is not None:
            # Use next price for effective leverage
            effective_leverage = self.current_position.calculate_effective_leverage(self.lowest_tf_next_open)
        else:
            effective_leverage = 0.0

        # Simulate the action taken
        if action == 0:  # Buy/Long
            if self.current_position is None or self.current_position.position_type != 'long':
                if self.current_position is not None:
                    # Close existing trade
                    self.current_position.close(self.lowest_tf_next_open, self.index)
                    profit = self.current_position.profit_pct
                    hold_time = self.current_position.hold_time
                    # save here the time of entry . NVMD NO THANKS
                # Open new long trade
                self.current_position = Trade(entry_price=self.lowest_tf_next_open, position_type='long', leverage=10, index=self.index, highest_future_price=self.highest_price, lowest_future_price=self.lowest_price, last_goal_tf_highest_index_first=self.goal_tf_highest_index_first)
                
            else:
                # Redundant action
                pass
        elif action == 1:  # Sell/Short
            if self.current_position is None or self.current_position.position_type != 'short':
                if self.current_position is not None:
                    # Close existing trade
                    self.current_position.close(self.lowest_tf_next_open, self.index)
                    profit = self.current_position.profit_pct
                    hold_time = self.current_position.hold_time
                # Open new short trade
                self.current_position = Trade(entry_price=self.lowest_tf_next_open, position_type='short', leverage=10, index=self.index, highest_future_price=self.highest_price, lowest_future_price=self.lowest_price, last_goal_tf_highest_index_first=self.goal_tf_highest_index_first)
            else:
                # Redundant action
                pass
        elif action == 2:  # Hold
            if self.current_position is not None:
                # Update hold time
                self.current_position.hold_time = self.index - self.current_position.entry_index
        elif action == 3:  # Close
            if self.current_position is not None:
                self.current_position.close(self.lowest_tf_next_open, self.index)
                profit = self.current_position.profit_pct
                hold_time = self.current_position.hold_time
                self.current_position = None
            else:
                # No position to close
                pass
            
        # self.environment_step = (rewards, self.current_position, profit, hold_time, effective_leverage)
        return rewards, self.current_position, profit, hold_time, effective_leverage#, pct_change
    
    def get_ideal_action(self):#, position, lowest_tf_next, target_tf_open):
        """
        Action key: 0: Long, 1: Short, 2: Hold, 3: Close
        """
        # hour_top_or_bot = lowest_tf_next.index[0].to_pydatetime().minute % 30 == 0 # MAKE 30 A CONFIG PARAMETER OF TARGET TF or somethin
        
        # if not hour_top_or_bot:
        #     self.goal_tf_open_array = np.insert(self.goal_tf_open_array, 0, target_tf_open)
                
        if self.current_position == None:
            if self.lowest_tf_highest_price_index == 0 and self.goal_tf_highest_price_index == 0:
                return 1
            elif self.lowest_tf_lowest_price_index == 0 and self.goal_tf_lowest_price_index == 0:
                return 0
        else:
            if self.lowest_tf_highest_price_index == 0 and self.goal_tf_highest_price_index == 0 and self.current_position.position_type != 'short':
                return 1 #3
            elif self.lowest_tf_lowest_price_index == 0 and self.goal_tf_lowest_price_index == 0 and self.current_position.position_type != 'long':
                return 0 #3
            
        return 2
    
    def calculate_long_reward(self):
        reward = 0
        if self.current_position is not None and self.current_position.position_type == 'long':
            # Penalize redundant choice
            return -1
        else:
            if self.current_position is not None and self.current_position.position_type == 'short':
                reward = self.calculate_close_reward()#self.current_position, current_price, index, future_prices, volatility)
            reward += 0.25 * (1 - 2 * ((self.lowest_tf_next_open - self.lowest_price) / (self.highest_price - self.lowest_price + 1e-6))) #1 - (self.lowest_tf_next_open - self.lowest_price) / (self.highest_price - self.lowest_price + 1e-6)

            return reward

    def calculate_short_reward(self):
        reward = 0
        if self.current_position is not None and self.current_position.position_type == 'short':
            # Penalize redundant choice
            return -1
        else:
            if self.current_position is not None and self.current_position.position_type == 'long':
                reward = self.calculate_close_reward()
            reward += 0.25 * (1 - 2 * ((self.highest_price - self.lowest_tf_next_open) / (self.highest_price - self.lowest_price + 1e-6)))
            
            return reward

    def calculate_hold_reward(self):
        reward = 0
        if self.current_position is None:
            """
            Calculate the reward for holding no trade.

            Reward is highest (1) at the midpoint between p_low and p_high,
            and decreases linearly to 0 as price approaches either extreme.
            """
            p_mid = (self.lowest_price + self.highest_price) / 2
            return 0.25 * (1 - 2 * abs((self.lowest_tf_next_open - p_mid) / (self.highest_price - self.lowest_price + 1e-6)))
        else:
            if self.current_position.position_type == 'long':
                if self.goal_tf_highest_price_index == 0 and self.lowest_tf_highest_price_index == 0:
                    # Highest index (profit) is next tick. Incentivize selling.
                    reward = -0.25
                    return reward
                elif self.goal_tf_lowest_price_index < self.goal_tf_highest_price_index: # and not self.current_position.indecies_flipped:
                    # Highest index (profit) is yet to come. (good).
                    reward += 0.25
                    return reward
                else:
                    if self.nearest_peak_within_lowest_tf: # May have entered early, don't discourage holding.
                        reward += 0.125
                        return reward
                    # Missed max profit, penalize. Low is coming now. Incentivize selling.
                    reward = -0.25
                    return reward
            else: # short
                if self.goal_tf_lowest_price_index == 0 and self.lowest_tf_lowest_price_index == 0:
                    reward = -0.25
                    return reward
                elif self.goal_tf_highest_price_index < self.goal_tf_lowest_price_index: # and not self.current_position.indecies_flipped:
                    reward += 0.25
                    return reward
                else:
                    if self.nearest_peak_within_lowest_tf:
                        reward += 0.125
                        return reward
                    reward = -0.25
                    return reward

    def calculate_close_reward(self):
        if self.current_position is None:
            # No position to close
            return -1  # Penalize trying to close when there's no position
        else:
            # Realized profit or loss from the trade
            realized_profit = self.current_position.calculate_profit_pct(self.lowest_tf_next_open)
            return realized_profit



class Reward:
    def __init__(self):
        self.profit_reward_threshold = 0.80
    
    def next(self, current_position, lowest_tf_open_array, goal_tf_open_array, target_tf_open, index):
        # self.w1 = reward_weights[0]
        # self.w2 = reward_weights[1]
        self.current_position = current_position
        self.lowest_tf_open_array = lowest_tf_open_array
        self.goal_tf_open_array = goal_tf_open_array
        self.lowest_tf_next_open = self.lowest_tf_open_array[0]#.iloc[0]
        self.target_tf_open = target_tf_open
        self.index = index
        
        # self.hour_top_or_bot = self.lowest_tf_open_array.index[0].to_pydatetime().minute % 30 == 0 # MAKE 30 A CONFIG PARAMETER OF TARGET TF or somethin
        
        # # BULLSHIT MAY LURK HERE:
        # # add the target tf open if we are on a harmonic time of the target tf and not the lowest tf.
        # if not self.hour_top_or_bot:
        #     self.goal_tf_open_array = np.insert(self.goal_tf_open_array, 0, self.target_tf_open)
        #     # self.lowest_tf_open_array = np.insert(self.lowest_tf_open_array, 0, self.target_tf_open)
        
        # Find the index of the lowest and highest prices in future_prices
        self.lowest_tf_lowest_price = min(self.lowest_tf_open_array)
        self.lowest_tf_highest_price = max(self.lowest_tf_open_array)
        self.lowest_tf_lowest_price_index = np.argmin(self.lowest_tf_open_array)#.values)
        self.lowest_tf_highest_price_index = np.argmax(self.lowest_tf_open_array)#.values)
        
        self.goal_tf_lowest_price = min(self.goal_tf_open_array)
        self.goal_tf_highest_price = max(self.goal_tf_open_array)
        self.goal_tf_lowest_price_index = np.argmin(self.goal_tf_open_array)
        self.goal_tf_highest_price_index = np.argmax(self.goal_tf_open_array)
        
        self.highest_price = max(self.goal_tf_highest_price, self.lowest_tf_highest_price)
        self.lowest_price = min(self.goal_tf_lowest_price, self.lowest_tf_lowest_price)
        
        self.goal_tf_highest_index_first = None
    
    def simulate_environment_step(self, hold_thyme, action=None): #, reward_weights=(1.0, 0.1)): #, current_price, future_prices, hold_thyme): #simulate_environment_step_new
        last_goal_tf_highest_index_first = self.goal_tf_highest_index_first if not None else None
        self.goal_tf_highest_index_first = self.goal_tf_highest_price_index < self.goal_tf_lowest_price_index
        # Determine if the values have changed.
        self.indices_flipped = (
            last_goal_tf_highest_index_first is not None
            and last_goal_tf_highest_index_first != self.goal_tf_highest_index_first
        )
        
        
        # Determine if the nearest peak (lowest or highest price) is within the array of self.lowest_tf_next_open.
        # We know this is true if the goal_tf_highest_price_index or goal_tf_lowest_price_index is 0, as the lowest tf will always be between 'this' time and the next index.
        self.nearest_peak_within_lowest_tf = (
            self.goal_tf_lowest_price_index == 0  or
            self.goal_tf_highest_price_index == 0
        )
                
        if self.current_position is not None:
            # current_position.passed_max_profit = current_position.exit_index < self.goal_tf_highest_price_index
            self.current_position.next(self.goal_tf_highest_index_first, self.highest_price, self.lowest_price)
            
            
        pct_change = None
        profit = 0.0
        hold_time = 0
        effective_leverage = 0.0
        

        # Compute rewards for all actions
        long_reward  = self.calculate_long_reward() #copy.deepcopy(current_position), current_price, index, future_prices[0])
        short_reward = self.calculate_short_reward()
        hold_reward  = self.calculate_hold_reward(hold_thyme)
        close_reward = self.calculate_close_reward()

        rewards = [long_reward, short_reward, hold_reward, close_reward]
        
        # Get the clipped reward for the chosen action
        reward = rewards[action] #clipped_rewards[action]
        # Compute effective leverage
        if self.current_position is not None:
            # Use next price for effective leverage
            effective_leverage = self.current_position.calculate_effective_leverage(self.lowest_tf_next_open)
        else:
            effective_leverage = 0.0

        # Simulate the action taken
        if action == 0:  # Buy/Long
            if self.current_position is None or self.current_position.position_type != 'long':
                if self.current_position is not None:
                    # Close existing trade
                    self.current_position.close(self.lowest_tf_next_open, self.index)
                    profit = self.current_position.profit_pct
                    hold_time = self.current_position.hold_time
                    # save here the time of entry . NVMD NO THANKS
                # Open new long trade
                self.current_position = Trade(entry_price=self.lowest_tf_next_open, position_type='long', leverage=10, index=self.index, highest_future_price=self.highest_price, lowest_future_price=self.lowest_price, last_goal_tf_highest_index_first=self.goal_tf_highest_index_first)
                
            else:
                # Redundant action
                pass
        elif action == 1:  # Sell/Short
            if self.current_position is None or self.current_position.position_type != 'short':
                if self.current_position is not None:
                    # Close existing trade
                    self.current_position.close(self.lowest_tf_next_open, self.index)
                    profit = self.current_position.profit_pct
                    hold_time = self.current_position.hold_time
                # Open new short trade
                self.current_position = Trade(entry_price=self.lowest_tf_next_open, position_type='short', leverage=10, index=self.index, highest_future_price=self.highest_price, lowest_future_price=self.lowest_price, last_goal_tf_highest_index_first=self.goal_tf_highest_index_first)
            else:
                # Redundant action
                pass
        elif action == 2:  # Hold
            if self.current_position is not None:
                # Update hold time
                self.current_position.hold_time = self.index - self.current_position.entry_index
        elif action == 3:  # Close
            if self.current_position is not None:
                self.current_position.close(self.lowest_tf_next_open, self.index)
                profit = self.current_position.profit_pct
                hold_time = self.current_position.hold_time
                self.current_position = None
            else:
                # No position to close
                pass
        return rewards, self.current_position, profit, hold_time, effective_leverage, pct_change
    
    def get_ideal_action(self):#, position, lowest_tf_next, target_tf_open):
        """
        Action key: 0: Long, 1: Short, 2: Hold, 3: Close
        """
        # hour_top_or_bot = lowest_tf_next.index[0].to_pydatetime().minute % 30 == 0 # MAKE 30 A CONFIG PARAMETER OF TARGET TF or somethin
        
        # if not hour_top_or_bot:
        #     self.goal_tf_open_array = np.insert(self.goal_tf_open_array, 0, target_tf_open)
                
        if self.current_position == None:
            if self.lowest_tf_highest_price_index == 0 and self.goal_tf_highest_price_index == 0:
                return 1
            elif self.lowest_tf_lowest_price_index == 0 and self.goal_tf_lowest_price_index == 0:
                return 0
        else:
            if self.lowest_tf_highest_price_index == 0 and self.goal_tf_highest_price_index == 0 and self.current_position.position_type != 'short':
                return 1 #3
            elif self.lowest_tf_lowest_price_index == 0 and self.goal_tf_lowest_price_index == 0 and self.current_position.position_type != 'long':
                return 0 #3
            
        return 2
    
    def calculate_long_reward(self):
        reward = 0
        if self.current_position is not None and self.current_position.position_type == 'long':
            # Penalize redundant choice
            return -1
        else:
            if self.current_position is not None and self.current_position.position_type == 'short':
                reward = self.calculate_close_reward()#self.current_position, current_price, index, future_prices, volatility)
            reward += 0.25 * (1 - 2 * ((self.lowest_tf_next_open - self.lowest_price) / (self.highest_price - self.lowest_price + 1e-6))) #1 - (self.lowest_tf_next_open - self.lowest_price) / (self.highest_price - self.lowest_price + 1e-6)

            return reward

    def calculate_short_reward(self):
        reward = 0
        if self.current_position is not None and self.current_position.position_type == 'short':
            # Penalize redundant choice
            return -1
        else:
            if self.current_position is not None and self.current_position.position_type == 'long':
                reward = self.calculate_close_reward()
            reward += 0.25 * (1 - 2 * ((self.highest_price - self.lowest_tf_next_open) / (self.highest_price - self.lowest_price + 1e-6)))
            
            return reward

    def calculate_hold_reward(self, hold_thyme):
        reward = 0
        if self.current_position is None:
            """
            Calculate the reward for holding no trade.

            Reward is highest (1) at the midpoint between p_low and p_high,
            and decreases linearly to 0 as price approaches either extreme.
            """
            p_mid = (self.lowest_price + self.highest_price) / 2
            return 0.25 * (1 - 2 * abs((self.lowest_tf_next_open - p_mid) / (self.highest_price - self.lowest_price + 1e-6)))
        else:
            if self.current_position.position_type == 'long':
                if self.goal_tf_highest_price_index == 0 and self.lowest_tf_highest_price_index == 0:
                    # Highest index (profit) is next tick. Incentivize selling.
                    reward = -0.25
                    return reward
                elif self.goal_tf_lowest_price_index < self.goal_tf_highest_price_index: # and not self.current_position.indecies_flipped:
                    # Highest index (profit) is yet to come. (good).
                    reward += 0.25
                    return reward
                else:
                    if self.nearest_peak_within_lowest_tf: # May have entered early, don't discourage holding.
                        reward += 0.125
                        return reward
                    # Missed max profit, penalize. Low is coming now. Incentivize selling.
                    reward = -0.25
                    return reward
            else: # short
                if self.goal_tf_lowest_price_index == 0 and self.lowest_tf_lowest_price_index == 0:
                    reward = -0.25
                    return reward
                elif self.goal_tf_highest_price_index < self.goal_tf_lowest_price_index: # and not self.current_position.indecies_flipped:
                    reward += 0.25
                    return reward
                else:
                    if self.nearest_peak_within_lowest_tf:
                        reward += 0.125
                        return reward
                    reward = -0.25
                    return reward

    def calculate_close_reward(self): #, curr_trade, current_price, index, future_prices, volatility):
        reward = 0
        if self.current_position is None:
            # No position to close
            return -1
        else:
            # Compute the profit for closing the current trade now
            current_profit_pct = self.current_position.calculate_profit_pct(self.lowest_tf_next_open)  # Profit at current price

            # # Compute potential future profits by looking ahead at future prices
            # potential_profits_pct = []
            # for future_price in future_prices:
            #     future_profit_pct = curr_trade.calculate_profit_pct(future_price)
            #     potential_profits_pct.append(future_profit_pct)

            if self.current_position.position_type == 'long':
                # max_future_profit_pct = curr_trade.calculate_profit_pct(self.highest_price)
                min_future_profit_pct = self.current_position.calculate_profit_pct(self.lowest_price)
            else:
                # max_future_profit_pct = curr_trade.calculate_profit_pct(self.lowest_price)
                min_future_profit_pct = self.current_position.calculate_profit_pct(self.highest_price)

            # Adjust reward based on whether closing early was beneficial
            if current_profit_pct < 0:
                # It was a losing trade and wasn't about to become profitable (and not self.nearest_peak_within_lowest_tf)   ###and it we didn't miss the max profit (and not self.indices_flipped)
                if self.current_position.position_type == 'long' and self.goal_tf_lowest_price_index < self.goal_tf_highest_price_index or self.current_position.position_type == 'short' and self.goal_tf_lowest_price_index > self.goal_tf_highest_price_index: # and not self.nearest_peak_within_lowest_tf and not self.indices_flipped:
                    # Closed trade was about to become profitable: negative reward for closing early
                    if self.nearest_peak_within_lowest_tf:
                        reward = -0.125
                        return reward
                    # We missed the max profit
                    # if self.indices_flipped:
                    #     #BREAKPOINT! ENSURE THIS SHIT IS CORRECT!
                    #     reward = current_profit_pct / 100 + (self.current_position.max_profit_pct - current_profit_pct) / 100
                    #     # reward = -0.25
                    #     return reward
                # - this is shit - if min_future_profit_pct < current_profit_pct and not self.nearest_peak_within_lowest_tf:
                    # The trade would have become worse; reward for closing early
                    additional_reward = (current_profit_pct - min_future_profit_pct) / 100  # Normalize
                    temp_reward = current_profit_pct / 100 + additional_reward
                    min_reward = 0.125
                    max_reward = 0.25
                    reward = min_reward + temp_reward * (max_reward - min_reward)
                # It was going to recover eventually, but we locked in losses
                else:
                    return -0.25
                    # reward = current_profit_pct / 100 this needs to be scaled based on the pct of the max profit, or something like that.
            else:
                # It was a profitable trade
                # We missed the max profit
                if self.indices_flipped:
                    #BREAKPOINT! ENSURE THIS SHIT IS CORRECT!
                    reward = current_profit_pct / 100 + (self.current_position.max_profit_pct - current_profit_pct) / 100
                    # reward = -0.25
                    return reward
                
                profit_pct = current_profit_pct/(self.current_position.max_profit_pct + 1e-6) #epsilon to prevent /by zero
                # print(profit_pct)
                # self.profit_reward_threshold = 0.80 #0.66 # how much from max profit the model must acheive before receiving a reward for closing.
                
                
                
                # Check if the profit percentage is above the reward threshold
                if profit_pct >= self.profit_reward_threshold:
                    # Calculate how far the profit percentage is between the threshold and 100%
                    # Use linear scaling to increase reward progressively from min_reward to max_reward
                    min_reward = 0.5
                    max_reward = 1
                    progress = (profit_pct - self.profit_reward_threshold) / (1 - self.profit_reward_threshold)
                    reward += min_reward + progress * (max_reward - min_reward)
                else:
                    # If the threshold is not met, return zero reward (or a small penalty if needed)
                    # return 0
                    reward += 0.05 #0 #-0.1

            return reward
    

"""TESTING PORPOISES"""

def epoch_loop(data_loader):
    nan_detected = False
    # Variables for tracking profits
    cumulative_profit = []
    epoch_pct_changes = []
    
    # Initialize tracking variables for hold times and trades
    open_trade_time = None
    profitable_trade_hold_times = []
    profitable_trade_pcts = []
    unprofitable_trade_hold_times = []
    unprofitable_trade_pcts = []
    # Initialize results list for logging
    results = []
    
    # process_batch = ProcessBatch(model) # look at this
    prefetcher = DataPrefetcher(data_loader, device)
    reward_object = Reward_Simple()
    # Initialize Portfolio
    portfolio = Portfolio(init_bal=1000, leverage=10)
    cum_return = CumulativeReturnBatch()
    current_trade = None
    # trades_list = []
    effective_leverages = []

    # Initialize lists to collect data
    timestamps = []
    cumulative_profit_over_time = []
    open_prices = []
    
    # Initialize batch_loss with a default value
    epoch_reward = 0
    current_position = 2
    
    nothing_hold_time = 0
    
    thirty_min_price = 0
    
    for index in range(len(data_loader)):
        if index == len(data_loader) - 1 or cum_return.get_done_tensor(-0.95):
            done = True
        else:
            done = False
        batch = prefetcher.next() # is batch supposed to be here? Are we prefetching in the optimal location?
        
        if batch is None:
            print(f"Reached end of data at index {index}")
            break
            
        raw_data = batch['raw_data'][0]
        current_price = raw_data['open']
        portfolio.next(current_price)
        
        # Update the current trade and calculate effective leverage
        if current_trade is not None:
            current_trade.update_hold_time()
            current_effective_leverage = current_trade.calculate_effective_leverage(current_price)
        else:
            current_effective_leverage = 0.0
        effective_leverages.append(current_effective_leverage)
        
        if current_trade is None:
            current_position = 2  # No position
        elif current_trade.position_type == 'long':
            current_position = 0  # Long position
        elif current_trade.position_type == 'short':
            current_position = 1  # Short position
            
        # Prepare model input, including effective leverage
        state = batch['input'].to(device)
        next_state = batch['next_input']
        
        lowest_tf_next = batch['lowest_tf_next_n_opens'][0] #.values
        goal_tf_next = np.array(batch['next_n_opens'][0]) #.values
        
        hour_top_or_bot = lowest_tf_next.index[0].to_pydatetime().minute % 30 == 0 # MAKE 30 A CONFIG PARAMETER OF TARGET TF or somethin
        # ticks_till_next_target_tf = count_five_minute_ticks(lowest_tf_next.index[0].to_pydatetime())
        lowest_tf_next = np.array(batch['lowest_tf_next_n_opens'][0]) #.values
        
        
        # BULLSHIT MAY LURK HERE:
        # add the target tf open if we are on a harmonic time of the target tf and not the lowest tf.
        if hour_top_or_bot:#not hour_top_or_bot:
            # goal_tf_next = np.insert(goal_tf_next, 0, current_price)
            thirty_min_price = current_price
        elif not hour_top_or_bot and thirty_min_price != 0:
            goal_tf_next = np.insert(goal_tf_next, 0, thirty_min_price)
        
        reward_object.next(current_trade, lowest_tf_open_array=lowest_tf_next, goal_tf_open_array=goal_tf_next, target_tf_open=current_price, index=index)
        
        ideal_action = reward_object.get_ideal_action()#current_trade, lowest_tf_next, current_price) #batch['raw_data'][0]['open'])
        
        action = ideal_action
        
        if action == 2: #hold
            nothing_hold_time += 1
        else:
            nothing_hold_time = 0
        # Simulate environment response
        rewards, new_current_trade, profit, hold_time, end_tick_effective_leverage, pct_change = reward_object.simulate_environment_step(
            action=action, hold_thyme=nothing_hold_time #batch['volatility'],
        )
        
        reward = rewards[action]
        epoch_reward += reward
        
        # Update the trade status: was a trade closed or a new trade opened?
        if current_trade != new_current_trade:
            if current_trade is not None and current_trade.exit_index is not None:
                # Trade was closed
                # trades_list.append(current_trade)
                # Update trade statistics
                if current_trade.profit_pct > 0:
                    profitable_trade_hold_times.append(current_trade.hold_time)
                    profitable_trade_pcts.append(current_trade.profit_pct)
                else:
                    unprofitable_trade_hold_times.append(current_trade.hold_time)
                    unprofitable_trade_pcts.append(current_trade.profit_pct)
                cumulative_profit.append(current_trade.profit_pct)
                last_batch_pct_change = current_trade.profit_pct
            current_trade = new_current_trade
            
            # Collect data for plotting
        timestamps.append(raw_data.name)  # Replace with actual timestamp if available
        # cumulative_profit_over_time.append(sum(cumulative_profit))
        
        # if len(cumulative_profit) > 0:
        #     # See trainer logic here.
        #     # # compound_return = calculate_cumulative_return_list(cumulative_profit)
        #     # cum_return = sum(cumulative_profit) / len(cumulative_profit) #calculate_cumulative_return_list(cumulative_profit)
        # else:
        #     cum_return = 0
        #     compound_return = 0
        # cumulative_profit_over_time.append(compound_return)
        open_prices.append(raw_data['open'])  # Assuming raw_data[0] corresponds to current time step
        # print(np.round(rewards, 4))
        boberts = np.round(rewards, 4)
        # Collect results for logging
        result_entry = {
            'index': index,
            'timestamp': raw_data.name, #batch.get('timestamp', index),  # Replace with actual timestamp if available
            'action': action,
            # 'cumulative_profit': compound_return,
            'reward': np.round(rewards[action], 4),
            'rewards': boberts, #np.round(rewards, 4),
            'current_price': current_price,
            'next_prices': goal_tf_next, #batch['next_n_opens'],
            'lowest_tf_next_prices': lowest_tf_next, #batch['lowest_tf_next_n_opens'],
            'current_position': current_position,
            'profit': profit,
            'hold_time': hold_time,
            'effective_leverage': current_trade.calculate_effective_leverage(lowest_tf_next[0]) if current_trade is not None else 0.0, #portfolio.effective_leverage,
            'trade_balance': portfolio.trade_balance,
            'pct_change': pct_change,
        }
        results.append(result_entry)
        if done:
            break
        
    # logging.info(f"{'Training' if is_training else 'Validation'} loop time: {end_loop - loop_start}")
    
    filepath = f"data/backtest/{timestamp}/"
    
    # Create the directory if it doesn't exist
    os.makedirs(filepath, exist_ok=True)
    
    # Save the Portfolio object
    portfolio_path = os.path.join(filepath, 'portfolio.pkl')
    with open(portfolio_path, 'wb') as f:
        pickle.dump(portfolio, f)
    print(f"Portfolio object saved to {portfolio_path}")
    
    # Generate and save the plot
    plt.figure(figsize=(12, 6))

    # Plot open prices
    ax1 = plt.gca()
    ax1.plot(timestamps, open_prices, label='Open Price', color='blue', alpha=0.6)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Open Price', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Plot cumulative profits on a secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(timestamps, cumulative_profit_over_time, label='Cumulative Profit', color='green')
    ax2.set_ylabel('Cumulative Profit', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    # Plot effective leverage on a third axis
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 0.5))
    ax3.plot(timestamps, effective_leverages, label='Effective Leverage', color='red')
    ax3.set_ylabel('Effective Leverage', color='red')
    ax3.tick_params(axis='y', labelcolor='red')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left')

    plt.title(f'Profit, Open Price, and Effective Leverage over Time')
    plot_path = os.path.join(filepath, f'profits_and_open_price.png')
    plt.savefig(plot_path)
    plt.close()
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(filepath, "log.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Compute and log statistics
    action_counts = results_df['action'].value_counts()
    average_reward = results_df['reward'].mean()
    total_profit = results_df['profit'].sum()
    average_profit = results_df['profit'].mean()
    average_hold_time = results_df['hold_time'].mean()

    print("\nEPOCH Training statistics:")
    print(f"Total reward: {results_df['reward'].sum():.4f}")
    print(f"Average reward: {average_reward:.4f}")
    print(f"Total profit: {total_profit:.4f}")
    print(f"Average profit per trade: {average_profit:.4f}")
    print(f"Action counts:\n{action_counts}")
    print(f"Average hold time: {average_hold_time:.2f} steps")
    
    print()
    # print(compound_return)
    
    return nan_detected, epoch_reward/index


if __name__ == "__main__":
    data = Data(config, interpolate_to_target=False, scaler=StandardScaler(), end_date='2020-01-01')
    train_data_loader, val_data_loader = get_data_loaders(data, config)

    _, reward = epoch_loop(train_data_loader)

    print(reward)
    
    
    
    
    
# import os
# import sys
# import numpy as np
# # from model_trainer import Trade
# import copy

# # Add the parent directory to sys.path#+
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(parent_dir)
# from config import load_config
# config = load_config('config.yaml')

# import re
# import gc
# import random
# import psutil
# # from torch.utils.data.dataloader import default_collate
# from datetime import datetime
# from pympler import asizeof
# import torch.optim as optim
# import torch
# import time
# from numba import njit
# from memory_profiler import profile
# import math
# from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR
# from torch.profiler import profile, record_function, ProfilerActivity
# from torch.cuda import Stream
# import torch.amp as amp
# from torch.nn import MSELoss
# from grokfast_pytorch import GrokFastAdamW
# from Model.model_data import *
# # from model3 import *
# # from Backtest.tradetest import Portfolio, calculate_returns
# # import torch.nn.functional as F  # Import F for loss computation
# import pickle
# import matplotlib.pyplot as plt
# from collections import deque, namedtuple
# import copy
# from Model.reward import * #, Trade
# from Model.model import *
# from Model.model_trainer import *

# from config import load_config
# config = load_config('config.yaml')

# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')









        #     position = self.current_position.position_type
        #     prices = future_prices #list(future_prices)
        #     reward = calculate_reward(position, curr_trade.entry_price, prices, volatility)
        #     # Need to ensure the calculate_reward wont surrender too much secured value
            
        #     if curr_trade.position_type == 'long':
        #         peak_profit_idx = future_prices.idxmax()
        #     else:
        #         peak_profit_idx = future_prices.idxmin()
            
        #     if future_prices.index[0] < peak_profit_idx and reward > 0:
        #         # max profit still to come, reward for holding
        #         reward += 0.05
        #     elif future_prices.index[0] == peak_profit_idx:
        #         # missed max profit
        #         reward = -0.05 #was -1
        # return reward
        
        
        
        
        
        
        
                            # if max_future_profit_pct * curr_trade.profit_reward_threshold < current_profit_pct:
                    #     # The trade would have become more profitable; penalize for closing early
                    #     additional_penalty = (max_future_profit_pct - current_profit_pct) / 100
                    #     reward = current_profit_pct / 100 - additional_penalty
                    # else:
                    #     reward = current_profit_pct / 100

            # Reward for closing if profit would decrease next tick
            # next_price = future_prices.iloc[1]  # The next price # 1, because 0 will be the close price, 1 is subsequent price
            # next_profit_pct = curr_trade.calculate_profit_pct(next_price)
            # if next_profit_pct < current_profit_pct:
            #     # Profit would decrease next tick; reward the model
            #     reward += (current_profit_pct - next_profit_pct) / 100  # Normalize reward

            # Clip reward to [-1, 1]
            # reward = np.clip(reward, -1, 1)