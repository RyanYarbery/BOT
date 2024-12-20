# model_trainer.py
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# To do, implement target network. Monitor Q value estimates.
import re
import gc
import random
# import psutil
# from torch.utils.data.dataloader import default_collate
from datetime import datetime, timedelta
# from pympler import asizeof
import torch.optim as optim
from Model.model import *
import torch
import time
import logging
from numba import njit
# from memory_profiler import profile
import math
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR
from torch.profiler import profile, record_function, ProfilerActivity
from torch.cuda import Stream
import torch.amp as amp
from torch.nn import MSELoss
# from grokfast_pytorch import GrokFastAdamW
from Model.model_data import *
# from model3 import *
# from Backtest.tradetest import Portfolio, calculate_returns
# import torch.nn.functional as F  # Import F for loss computation
import pickle
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import copy
from Model.reward import * #simulate_environment_step_new, get_ideal_action #, Trade

# Add the parent directory to sys.path#+
from config import load_config
import pandas as pd
from statistics import mean
config = load_config('config.yaml')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers= [
                        logging.FileHandler('data/logs/trainer.log'),
                        logging.StreamHandler()
                    ]) # For some reason, we ain't logging to trainer.log... it is going to model.log

# torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)


#     /home/andrew/Programming/TradeBot/Backtester/Model/model_trainer.py:494: RuntimeWarning: overflow encountered in scalar multiply
#   cumulative_return *= (1 + (pct/100))
# /home/andrew/Programming/TradeBot/Backtester/Model/model_trainer.py:494: RuntimeWarning: overflow encountered in scalar multiply
# 2024-11-23 21:08:36,239 - INFO - Index: 328382
# 2024-11-23 21:08:36,239 - INFO - GPU memory usage: 0.03 GB
# 2024-11-23 21:08:36,239 - INFO - CPU Memory: 26.05 GB
# 2024-11-23 21:08:36,240 - INFO - Progress: 100.00% | Est. time: 0.00s | Time/index: 0.4360s | Cumulative Avg Loss: 0.024948120
# 2024-11-23 21:08:36,240 - INFO - Avg Batch Losses: 0.004388136
# 2024-11-23 21:08:36,240 - INFO - Last Avg Loss: 0.018402100
# 2024-11-23 21:08:36,240 - INFO - Last Profit: BATCH | AVG | COMPOUND --- 6.751096 | 24.851403 | inf
# 2024-11-23 21:08:36,240 - INFO - Trades: 8445 (100.00% profitable) | Avg. Profit 24.85%. Hold Time 38.88 ticks. | Avg. Loss 0.00%. Hold Time 1.00 ticks.
# 2024-11-23 21:08:36,240 - INFO - Current Learning Rate: 0.0000300000
# 2024-11-23 21:08:36,240 - INFO - Action taken: 2
# 2024-11-23 21:08:36,240 - INFO - Total Reward Received, per index: 43093.1658, 0.13123
# 2024-11-23 21:08:36,240 - INFO - Current position: 1
# 2024-11-23 21:08:36,240 - INFO - Epsilon: 0.9500
# 2024-11-23 21:08:36,240 - INFO - Training loop time: 143187.64328193665
# Portfolio object saved to data/models/20241122_052208/epoch_0/training/portfolio.pkl
# /opt/anaconda3/envs/vectorbtpro2/lib/python3.12/site-packages/matplotlib/ticker.py:2096: RuntimeWarning: overflow encountered in multiply
#   steps = self._extended_steps * scale
# /opt/anaconda3/envs/vectorbtpro2/lib/python3.12/site-packages/matplotlib/ticker.py:2130: RuntimeWarning: overflow encountered in scalar subtract
#   high = edge.ge(_vmax - best_vmin)
# /opt/anaconda3/envs/vectorbtpro2/lib/python3.12/site-packages/matplotlib/ticker.py:1956: RuntimeWarning: invalid value encountered in scalar divmod
#   d, m = divmod(x, self.step)
# 2024-11-23 21:08:37,686 - ERROR - An error occurred during training: arange: cannot compute length
# Traceback (most recent call last):
#   File "/home/andrew/Programming/TradeBot/Backtester/Model/model_trainer.py", line 1233, in <module>
#   File "/home/andrew/Programming/TradeBot/Backtester/Model/model_trainer.py", line 1195, in init_training
    
#   File "/home/andrew/Programming/TradeBot/Backtester/Model/model_trainer.py", line 379, in train
#     avg_train_loss, nan_detected, train_compound_return = epoch_loop(
#                                                           ^^^^^^^^^^^
#   File "/home/andrew/Programming/TradeBot/Backtester/Model/model_trainer.py", line 1137, in epoch_loop
#     d_models_joint=config['model']['process_layers'],
# ^^^^^^^^^^^^^^^^^^^^^^
#   File "/opt/anaconda3/envs/vectorbtpro2/lib/python3.12/site-packages/matplotlib/pyplot.py", line 1134, in savefig
#     res = fig.savefig(*args, **kwargs)  # type: ignore[func-returns-value]
#           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/opt/anaconda3/envs/vectorbtpro2/lib/python3.12/site-packages/matplotlib/figure.py", line 3390, in savefig
#     self.canvas.print_figure(fname, **kwargs)
#   File "/opt/anaconda3/envs/vectorbtpro2/lib/python3.12/site-packages/matplotlib/backends/backend_qtagg.py", line 75, in print_figure
#     super().print_figure(*args, **kwargs)
#   File "/opt/anaconda3/envs/vectorbtpro2/lib/python3.12/site-packages/matplotlib/backend_bases.py", line 2193, in print_figure
#     result = print_method(
#              ^^^^^^^^^^^^^
#   File "/opt/anaconda3/envs/vectorbtpro2/lib/python3.12/site-packages/matplotlib/backend_bases.py", line 2043, in <lambda>
#     print_method = functools.wraps(meth)(lambda *args, **kwargs: meth(
#                                                                  ^^^^^
#   File "/opt/anaconda3/envs/vectorbtpro2/lib/python3.12/site-packages/matplotlib/backends/backend_agg.py", line 497, in print_png
#     self._print_pil(filename_or_obj, "png", pil_kwargs, metadata)
#   File "/opt/anaconda3/envs/vectorbtpro2/lib/python3.12/site-packages/matplotlib/backends/backend_agg.py", line 445, in _print_pil
#     FigureCanvasAgg.draw(self)
#   File "/opt/anaconda3/envs/vectorbtpro2/lib/python3.12/site-packages/matplotlib/backends/backend_agg.py", line 388, in draw
#     self.figure.draw(self.renderer)
#   File "/opt/anaconda3/envs/vectorbtpro2/lib/python3.12/site-packages/matplotlib/artist.py", line 95, in draw_wrapper
#     result = draw(artist, renderer, *args, **kwargs)
#              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/opt/anaconda3/envs/vectorbtpro2/lib/python3.12/site-packages/matplotlib/artist.py", line 72, in draw_wrapper
#     return draw(artist, renderer)
#            ^^^^^^^^^^^^^^^^^^^^^^
#   File "/opt/anaconda3/envs/vectorbtpro2/lib/python3.12/site-packages/matplotlib/figure.py", line 3154, in draw
#     mimage._draw_list_compositing_images(
#   File "/opt/anaconda3/envs/vectorbtpro2/lib/python3.12/site-packages/matplotlib/image.py", line 132, in _draw_list_compositing_images
#     a.draw(renderer)
#   File "/opt/anaconda3/envs/vectorbtpro2/lib/python3.12/site-packages/matplotlib/artist.py", line 72, in draw_wrapper
#     return draw(artist, renderer)
#            ^^^^^^^^^^^^^^^^^^^^^^
#   File "/opt/anaconda3/envs/vectorbtpro2/lib/python3.12/site-packages/matplotlib/axes/_base.py", line 3070, in draw
#     mimage._draw_list_compositing_images(
#   File "/opt/anaconda3/envs/vectorbtpro2/lib/python3.12/site-packages/matplotlib/image.py", line 132, in _draw_list_compositing_images
#     a.draw(renderer)
#   File "/opt/anaconda3/envs/vectorbtpro2/lib/python3.12/site-packages/matplotlib/artist.py", line 72, in draw_wrapper
#     return draw(artist, renderer)
#            ^^^^^^^^^^^^^^^^^^^^^^
#   File "/opt/anaconda3/envs/vectorbtpro2/lib/python3.12/site-packages/matplotlib/axis.py", line 1387, in draw
#     ticks_to_draw = self._update_ticks()
#                     ^^^^^^^^^^^^^^^^^^^^
#   File "/opt/anaconda3/envs/vectorbtpro2/lib/python3.12/site-packages/matplotlib/axis.py", line 1275, in _update_ticks
#     major_locs = self.get_majorticklocs()
#                  ^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/opt/anaconda3/envs/vectorbtpro2/lib/python3.12/site-packages/matplotlib/axis.py", line 1495, in get_majorticklocs
#     return self.major.locator()
#            ^^^^^^^^^^^^^^^^^^^^
#   File "/opt/anaconda3/envs/vectorbtpro2/lib/python3.12/site-packages/matplotlib/ticker.py", line 2140, in __call__
#     return self.tick_values(vmin, vmax)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/opt/anaconda3/envs/vectorbtpro2/lib/python3.12/site-packages/matplotlib/ticker.py", line 2148, in tick_values
#     locs = self._raw_ticks(vmin, vmax)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/opt/anaconda3/envs/vectorbtpro2/lib/python3.12/site-packages/matplotlib/ticker.py", line 2131, in _raw_ticks
#     ticks = np.arange(low, high + 1) * step + best_vmin
#             ^^^^^^^^^^^^^^^^^^^^^^^^
# ValueError: arange: cannot compute length



# Define a namedtuple for storing experiences
Experience = namedtuple('Experience', 
                        field_names=['state', 'action', 'reward', 'next_state', 'done'])
BUFFER_SIZE = int(5e4)  # e.g., 50,000
BATCH_SIZE = config['training']['batch_size'] #16
MODEL_UPDATE_FREQ = config['training']['model_update_freq']

class DataPrefetcher:
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = Stream(device=device)
        self.next_batch = None
        self.preload()
    
    def preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return

        with torch.cuda.stream(self.stream):
            for key in self.next_batch:
                if isinstance(self.next_batch[key], torch.Tensor):
                    self.next_batch[key] = self.next_batch[key].to(self.device, non_blocking=True)
                elif isinstance(self.next_batch[key], list):
                    self.next_batch[key] = [item.to(self.device, non_blocking=True) if isinstance(item, torch.Tensor) else item
                                            for item in self.next_batch[key]]
    
    def next(self):
        if self.next_batch is None:
            return None
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        self.preload()
        return batch



# class DataPrefetcher:
#     def __init__(self, loader, device):
#         self.loader = iter(loader)
#         self.device = device
#         self.stream = Stream(device=device)
#         self.preload()

#     def preload(self):
#         try:
#             self.next_batch = next(self.loader)
#         except StopIteration:
#             self.next_batch = None
#             return

#         with torch.cuda.stream(self.stream):
#             self.next_batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
#                                for k, v in self.next_batch.items()}

#     def next(self):
#         torch.cuda.current_stream().wait_stream(self.stream)
#         batch = self.next_batch
#         self.preload()
#         return batch
    
    # def preload(self):
    #     try:
    #         self.next_batch = next(self.loader)
    #         # while True:
    #         #     next_batch = next(self.loader)
    #         #     # Create a mask for items where target is not 2
    #         #     mask = next_batch['target'].view(-1) != 2
    #         #     # Apply the mask to all tensors in the batch
    #         #     filtered_batch = {k: v[mask] if isinstance(v, torch.Tensor) and v.size(0) == mask.size(0) else v 
    #         #                     for k, v in next_batch.items()}
    #         #     if len(filtered_batch['target']) >= config['training']['batch_size']:
    #         #         self.next_batch = filtered_batch
    #         #         break
    #     except StopIteration:
    #         self.next_batch = None
    #         return

class ProcessBatch(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
# Try commenting out the reward def class. Just provide rewards based on earning, eliminate fixed rewards (ie +=0.5)

# can having a huge reward screw up the loss calculation bacause a reward of like 10, when the model only outputs at most about 1, will lead to a large MSEerror/loss. Yes, use cliping


# use reward clipping or a non-linear transformation like tanh(reward/scale) https://claude.ai/chat/f36a93a9-9185-44bd-a9a7-d335e337e4f0

# class RegularizedCrossEntropyLoss(nn.Module):
#     def __init__(self, l1_lambda=0.01):# , l2_lambda=0.01):
#         super().__init__()
#         self.base_loss = nn.BCEWithLogitsLoss()
#         self.l1_lambda = l1_lambda

#     def forward(self, outputs, targets, model):
#         base_loss = self.base_loss(outputs, targets)
#         l1_reg = torch.tensor(0., requires_grad=True)
        
#         for param in model.parameters():
#             l1_reg = l1_reg + torch.norm(param, 1)
        
#         total_loss = base_loss + self.l1_lambda * l1_reg
#         return total_loss

class RegularizedCrossEntropyLoss(nn.Module):
    def __init__(self, l1_lambda=0.01):
        super().__init__()
        self.base_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.l1_lambda = l1_lambda

    def forward(self, outputs, targets, model, weights=None):
        base_loss = self.base_loss(outputs, targets)
        
        if weights is not None:
            base_loss = base_loss * weights
        
        base_loss = base_loss.mean()
        
        l1_reg = torch.tensor(0., requires_grad=True)
        
        for param in model.parameters():
            l1_reg = l1_reg + torch.norm(param, 1)
        
        total_loss = base_loss + self.l1_lambda * l1_reg
        return total_loss

# Custom learning rate scheduler with warm-up + cosine decay
class WarmupCosineDecayLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, base_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        super(WarmupCosineDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1  # Current epoch/step
        
        if step < self.warmup_steps:
            # Linear warm-up phase
            lr = step / self.warmup_steps * self.base_lr
        else:
            # Cosine decay phase
            decay_steps = step - self.warmup_steps
            total_decay_steps = self.total_steps - self.warmup_steps
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * decay_steps / total_decay_steps))
        
        return [lr for _ in self.base_lrs]  # Apply the same LR to all param groups
"ADJUST THIS: Patience-based early stopping: The logic seems fine, though theres a minor issue in restoring the best model in early stopping. It currently reloads from the saved checkpoint model, which might need careful reloading of states."

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed=42):
        """
        Initialize the replay buffer.
        
        Parameters:
        - buffer_size (int): Maximum number of experiences to store.
        - batch_size (int): Number of experiences to sample for each training step.
        - seed (int): Random seed for reproducibility.
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to the buffer.
        """
        e = Experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        # Process state components
        state_tensors = np.array([e.state[0] for e in experiences if e is not None])
        state_positions = np.array([e.state[1] for e in experiences if e is not None])
        state_leverages = np.array([e.state[2] for e in experiences if e is not None])
        state_actions = np.array([e.state[3] for e in experiences if e is not None])

        states = (state_tensors, state_positions, state_leverages, state_actions)

        # Process actions, rewards, next_states, dones similarly
        actions = np.array([e.action for e in experiences if e is not None])
        rewards = np.array([e.reward for e in experiences if e is not None])

        next_state_tensors = np.array([e.next_state[0] for e in experiences if e is not None])
        next_state_positions = np.array([e.next_state[1] for e in experiences if e is not None])
        next_state_leverages = np.array([e.next_state[2] for e in experiences if e is not None])
        next_state_actions = np.array([e.next_state[3] for e in experiences if e is not None])

        next_states = (next_state_tensors, next_state_positions, next_state_leverages, next_state_actions)

        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8)

        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        """
        Return the current size of internal memory.
        """
        return len(self.memory)

class Portfolio:
    def __init__(self, init_bal, leverage):
        self.position = 2  # 0: LONG, 1: SHORT, 2: NONE
        self.entry_price = 0.0
        self.trade_balance = init_bal # * (1 - withdrawl_pct)
        # self.reserve_balance = init_bal * withdrawl_pct
        self.leverage = leverage
        self.effective_leverage = 0.0
        self.open_position_value = 0.0
        self.anomalous_data = 0
        self.idx_ = 0  # Not necessary unless tracking over time
    
    def next(self, current_price):
        if self.position != 2:
            self.update_open_position_value(current_price)
            self.calculate_effective_leverage(current_price)
        else:
            self.effective_leverage = 0.0  # No leverage if no position
    
    def short(self, current_price):
        if self.position == 0:  # Closing LONG position
            self.close_position(current_price)
        self.position = 1  # Enter SHORT position
        self.entry_price = current_price
        self.update_open_position_value(current_price)
        self.calculate_effective_leverage(current_price)

    def long(self, current_price):
        if self.position == 1:  # Closing SHORT position
            self.close_position(current_price)
        self.position = 0  # Enter LONG position
        self.entry_price = current_price
        self.update_open_position_value(current_price)
        self.calculate_effective_leverage(current_price)

    def close_position(self, exit_price):
        trade_size = self.trade_balance * self.leverage
        if self.position == 1:  # Closing SHORT
            profit = (self.entry_price - exit_price) * trade_size / self.entry_price
            self.trade_balance += profit
        elif self.position == 0:  # Closing LONG
            profit = (exit_price - self.entry_price) * trade_size / self.entry_price
            self.trade_balance += profit
        self.trade_balance = max(self.trade_balance, 0)
        self.position = 2  # No position
        self.entry_price = 0.0
        self.open_position_value = 0.0
        self.effective_leverage = 0.0

    def calculate_effective_leverage(self, current_price):
        if self.position == 2 or self.trade_balance == 0:
            self.effective_leverage = 0.0
        else:
            trade_size = self.trade_balance * self.leverage
            if self.position == 1:  # SHORT
                new_equity = self.trade_balance + (self.entry_price / current_price - 1) * trade_size
            elif self.position == 0:  # LONG
                new_equity = self.trade_balance + (current_price / self.entry_price - 1) * trade_size
            else:
                new_equity = self.trade_balance
            self.effective_leverage = self.open_position_value / new_equity if new_equity != 0 else 0.0

    def update_open_position_value(self, current_price):
        trade_size = self.trade_balance * self.leverage
        if self.position == 1:  # SHORT
            self.open_position_value = (self.entry_price / current_price) * trade_size
        elif self.position == 0:  # LONG
            self.open_position_value = (current_price / self.entry_price) * trade_size
        else:
            self.open_position_value = 0.0
        
def train(config, train_data_loader, val_data_loader, model, dicts, info_dicts):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Len train data loader, val data loader: {len(train_data_loader)}, {len(val_data_loader)}")
    model.to(device)
    
    # model = torch.compile(model)
    
    train_losses = []
    val_losses = []
    train_compound_returns = []
    val_rewards = []
    
    best_val_loss = float('inf')
    best_val_reward = -float('inf')
    avg_val_loss = 0
    patience_counter = 0
    best_model_state = None
    start_epoch = 0
    # Separate optimizers for each model
    #optim.AdamW
    epsilon_start = config['training']['epsilon_start']
    epsilon_end = config['training']['epsilon_end']
    epsilon_decay = config['training']['epsilon_decay']
    optimizer = 0 #GrokFastAdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=0.0001, normalize_lr=False) #GrokFastAdamW(model.parameters(), lr=config['training']['learning_rate']) #, weight_decay=0.1)
                                                                                                    # ! Try total steps = another value.
    # scheduler = WarmupCosineDecayLR(optimizer, warmup_steps=config['training']['warmup'], total_steps=(len(train_data_loader)*config['training']['num_epochs']/config['training']['effective_batch_size']), base_lr=config['training']['learning_rate'])
    scaler = amp.GradScaler()
    
    replay_buffer_train = ReplayBuffer(buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, seed=42)
    replay_buffer_val = ReplayBuffer(buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, seed=42)

    # region old scheduler
    # Learning rate decay setup
    lr_decay_method = config['training'].get('lr_decay_method', 'step')
    
    if lr_decay_method == 'step':
        scheduler = StepLR(optimizer, step_size=config['training'].get('lr_step_size', 30), 
                           gamma=config['training'].get('lr_gamma', 0.1))
    elif lr_decay_method == 'exponential':
        scheduler = ExponentialLR(optimizer, gamma=config['training'].get('lr_gamma', 0.95))
    elif lr_decay_method == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=config['training']['num_epochs'])
    else:
        raise ValueError(f"Unknown lr_decay_method: {lr_decay_method}")
    # endregion
    
    if dicts is not None:
        optimizer.load_state_dict(dicts['optimizer_state_dict'])
        scheduler.load_state_dict(dicts['scheduler_state_dict'])
        scaler.load_state_dict(dicts['scaler_state_dict'])
    if info_dicts is not None:
        best_val_loss = info_dicts['best_val_loss']
        best_val_reward = info_dicts['best_val_reward']
        train_losses = info_dicts['train_losses']
        val_losses = info_dicts['val_losses']
        start_epoch = info_dicts['start_epoch']
        epsilon_start = info_dicts['epsilon_start']
        timestamp = info_dicts['timestamp']
    
    # # Check if there's a checkpoint to resume from
    # if config['training'].get('resume_from_checkpoint'): Move this to data resume from checkpoint to model creation area, if  none, then do the default model creation. Pass all the surpurfluous values to this point in a tuple to avoid having to pass them all individually. This is going to allow for passing epoch to the creation of the TimeSeriese Dataset for branch prediction effeiciency.
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        logging.info(f"Starting TRAINING epoch {epoch}")
                    
        avg_train_loss, nan_detected, train_compound_return = epoch_loop(
            model,
            train_data_loader,
            optimizer,
            device,
            timestamp,
            epoch,
            epsilon_start,
            replay_buffer=replay_buffer_train,
            is_training=True,
            scheduler=scheduler,
            scaler = scaler
        )
        train_losses.append(avg_train_loss)
        train_compound_returns.append(train_compound_return)
        torch.cuda.empty_cache()
        
        if nan_detected:
            logging.info(f"NaN detected in epoch {epoch}. Stopping training.")
            break
        
        logging.info(f"Epoch {epoch}/{config['training']['num_epochs']}, Total Loss: {avg_train_loss}\n")
        
        # Validation step
        logging.info("Starting VALIDATION step")
        avg_val_loss, val_nan_detected, val_reward = epoch_loop(
            model,
            val_data_loader,
            optimizer,
            device,
            timestamp,
            epoch,
            epsilon_start,
            replay_buffer=replay_buffer_val,
            is_training=False,
            scaler = scaler
        )
        val_losses.append(avg_val_loss)
        val_rewards.append(val_reward)
        logging.info(f"Epoch {epoch}/{config['training']['num_epochs']}, Validation Loss: {avg_val_loss}")
        
        if val_nan_detected:
            logging.info(f"NaN detected during validation in epoch {epoch}. Stopping training.")
            break
        
        # Epsilon-greedy action selection
        epsilon_start = epsilon_end + (epsilon_start - epsilon_end) * \
                  math.exp(-1. * epoch / epsilon_decay) # replaced steps_done with index
        train_data_loader.epoch += 1
        val_data_loader.epoch += 1
        # Early stopping logic
        if val_reward > best_val_reward: #avg_val_loss < best_val_loss:
            best_val_reward = val_reward
            patience_counter = 0

            filename = f"{timestamp}/epoch_{epoch}/model_return_{val_reward:2f}.pth"
            model_path = os.path.join(os.path.dirname(config['model']['output']['model_directory']), filename)

            model.cpu()  # Move model to CPU before saving
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_reward': best_val_reward,
                'train_losses': train_losses,
                'val_losses': val_losses, # NEED TO ENSURE ALL BEYSIAN VALUES ARE SAVED HERE, SAVE SCALER VALUES HERE TOO.
                'sequence_length': config['model']['sequence_length'],
                'num_heads': config['model']['nhead'],
                'num_outputs': config['model']['num_outputs'],
                'preprocess_layers': config['model']['preprocess_layers'],
                'process_layers': config['model']['process_layers'],
                'model_type': model.model_type,
                'dropout': config['model']['dropout'],
                'epsilon_start': epsilon_start,
            }

            # Ensure the directory exists
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            torch.save(checkpoint, model_path) #ONLY SAVE THE BEST MODELS!!!!!!!!
            model.to(device)  # Move model back to the original device
            torch.cuda.empty_cache()  # Clear CUDA cache

            # logging.info(f"New best model saved with loss: {best_val_loss:.6f}")
            logging.info(f"New best model saved with reward: {best_val_reward:.6f}")
        else:
            scheduler.step()
            
            filename = f"{timestamp}/epoch_{epoch}/model_return_{val_reward:2f}.pth"
            model_path = os.path.join(os.path.dirname(config['model']['output']['model_directory']), filename)

            model.cpu()  # Move model to CPU before saving
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_reward': best_val_reward,
                'train_losses': train_losses,
                'val_losses': val_losses, # NEED TO ENSURE ALL BEYSIAN VALUES ARE SAVED HERE, SAVE SCALER VALUES HERE TOO.
                'sequence_length': config['model']['sequence_length'],
                'num_heads': config['model']['nhead'],
                'num_outputs': config['model']['num_outputs'],
                'preprocess_layers': config['model']['preprocess_layers'],
                'process_layers': config['model']['process_layers'],
                'model_type': model.model_type,
                'dropout': config['model']['dropout'],
                'epsilon_start': epsilon_start,
            }

            # Ensure the directory exists
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            torch.save(checkpoint, model_path) #ONLY SAVE THE BEST MODELS!!!!!!!!
            model.to(device)  # Move model back to the original device
            torch.cuda.empty_cache()  # Clear CUDA cache
            
            if patience_counter >= config['training']['patience']:
                # logging.info(f"Early stopping triggered. Best loss: {best_val_loss:.6f}")
                logging.info(f"Early stopping triggered. Best reward: {best_val_reward:.6f}")
                break
            patience_counter += 1
            # model = checkpoint['model'] #.to(device)  # Move model back to the original device
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)  # Move model back to the original device
        # logging.info(f"Patience counter: {patience_counter}/{config['training']['patience']}, Best loss: {best_val_loss:.6f}")
        logging.info(f"Patience counter: {patience_counter}/{config['training']['patience']}, Best reward: {best_val_reward:.6f}")
        # Learning rate decay step
        current_lr = scheduler.get_last_lr()[0]
        logging.info(f"Current learning rate: {current_lr:.10f}")
        logging.info(f"Current epsilon: {epsilon_start:.6f}\n\n\n")
        
        gc.collect()

    logging.info(f"Training completed. Best model saved at {config['model']['output']['model_directory']}")

    return train_losses, val_losses


# def calculate_cumulative_return_list(pct_changes):
#     if pct_changes is None:
#         return 0
#     cumulative_return = 1
#     for pct in pct_changes:
#         cumulative_return *= (1 + (pct/100))
#     return cumulative_return - 1
# class CumulativeReturn:
#     def __init__(self, pct_changes=None):
#         self.cumulative_product = 1
#         if pct_changes:
#             for pct in pct_changes:
#                 self.add_pct_change(pct)

#     def add_pct_change(self, pct):
#         self.cumulative_product *= (1 + pct / 100)

#     def get_cumulative_return(self):
#         return self.cumulative_product - 1

class CumulativeReturnBatch:
    def __init__(self, threshold, batch_size, total_steps, device=None):
        self.batch_size = batch_size
        self.device = device if device else torch.device('cpu')
        # Initialize cumulative products as a tensor of ones
        self.cumulative_product = torch.ones(batch_size, device=self.device)
        # Initialize a mask indicating active items (True = active, False = inactive)
        self.active_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        # Initialize done_tensor
        self.done_tensor = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        self.total_steps = total_steps
        self.threshold = threshold
        
        # New: Counter for how many times add_pct_change() is called per item
        self.count_tensor = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
    
    def add_pct_change(self, pct_change, batch_idx):
        """
        Updates the cumulative product with a batch of percentage changes for active items only.
        
        Parameters:
        pct_change (list, numpy array, or torch tensor): A batch of percentage changes.
        """
        # Convert pct_change to a tensor if it's not already one
        if not isinstance(pct_change, torch.Tensor):
            pct_change = torch.tensor(pct_change, device=self.device, dtype=torch.float32)
        else:
            pct_change = pct_change.to(self.device, dtype=torch.float32)
        
        # # Ensure the input is of the correct shape
        # if pct_changes.shape[0] != self.batch_size:
        #     raise ValueError(f"Expected pct_changes of size {self.batch_size}, got {pct_changes.shape[0]}")
        
        # Update only if the item is still active
        if self.active_mask[batch_idx]:
            self.cumulative_product[batch_idx] *= (1 + pct_change.item() / 100.0)
            self.count_tensor[batch_idx] += 1
        
        # # Update only active items
        # active_indices = self.active_mask.nonzero(as_tuple=True)[0]
        # if active_indices.numel() > 0:
        #     self.cumulative_product[active_indices] *= (1 + pct_change[active_indices] / 100)
    
    def get_cumulative_return(self):
        """
        Returns the cumulative returns for the batch.
        
        Returns:
        torch.Tensor: A tensor of cumulative returns.
        """
        return self.cumulative_product - 1
    
    def is_under_threshold(self):
        """
        Checks which cumulative returns are under the given threshold.
        
        Parameters:
        threshold (float): The threshold value (e.g., -0.95)
        
        Returns:
        torch.Tensor: A boolean tensor indicating which cumulative returns are under the threshold.
        """
        cumulative_returns = self.get_cumulative_return()
        under_threshold = cumulative_returns <= self.threshold
        return under_threshold
    
    def deactivate_under_threshold(self):
        """
        Deactivates items whose cumulative return is under the threshold.
        
        Parameters:
        threshold (float): The threshold value
        """
        under_threshold = self.is_under_threshold()
        # Update active_mask: deactivate items under threshold
        self.active_mask = self.active_mask & (~under_threshold)
    
    def get_active_indices(self):
        """
        Returns the indices of the active items.
        
        Returns:
        torch.Tensor: A tensor of indices of active items.
        """
        return self.active_mask.nonzero(as_tuple=True)[0]
    
    def should_halt_all_outputs(self, percentage=0.005):
        """
        Checks if the proportion of items under the threshold exceeds the given percentage.
        
        Parameters:
        threshold (float): The threshold value
        percentage (float): The fraction (between 0 and 1) of items under threshold required to halt outputs
        
        Returns:
        bool: True if the condition is met, False otherwise
        """
        under_threshold = self.is_under_threshold()
        num_under = under_threshold.sum().item()
        if num_under / self.batch_size >= percentage:
            return True
        else:
            return False
    
    def update_done_tensor(self, index):
        """
        Updates the done_tensor based on the current index and threshold condition.
        
        Parameters:
        index (int): Current step index
        total_steps (int): Total number of steps in the data loader
        threshold (float): The threshold value
        """
        if index == self.total_steps - 1:
            # If it's the last step, mark all active items as done
            self.done_tensor = self.done_tensor | self.active_mask
        else:
            # Otherwise, mark items under threshold as done
            under_threshold = self.is_under_threshold()
            self.done_tensor = self.done_tensor | under_threshold
    
    def get_done_tensor(self):
        """
        Returns the current done_tensor.
        
        Returns:
        torch.Tensor: A boolean tensor indicating which items are done.
        """
        return self.done_tensor
    
    def get_num_done(self):
        """
        Returns the number of done items.
        
        Returns:
        int: Number of items marked as done.
        """
        return self.done_tensor.sum().item()
    
    def get_done_indices(self):
        """
        Returns the indices of done items.
        
        Returns:
        torch.Tensor: A tensor of indices of done items.
        """
        return self.done_tensor.nonzero(as_tuple=True)[0]
    
    def get_average_return(self):
        """
        Returns the geometric average return across the entire batch as a single scalar value.
        
        The geometric average for an item i is:
        (cumulative_product[i] ^ (1 / count_tensor[i])) - 1, if count_tensor[i] > 0
        Otherwise, if count_tensor[i] == 0 (no additions), returns 0.
        
        The final returned value is the arithmetic mean of these geometric averages across all batch items.
        
        Returns:
            torch.Tensor: A scalar tensor representing the average geometric return across the batch.
        """
        avg = self.cumulative_product.clone()
        
        # Items with count = 0 have not been updated, so their average return is 0
        count_zero_mask = (self.count_tensor == 0)
        
        # For items with count > 0, compute geometric average
        nonzero_mask = ~count_zero_mask
        if nonzero_mask.any():
            avg[nonzero_mask] = avg[nonzero_mask].pow(1.0 / self.count_tensor[nonzero_mask]) - 1.0
        
        # For items with count = 0, set to 0
        avg[count_zero_mask] = 0.0
        
        # Compute the arithmetic mean of the geometric averages
        average_over_batch = avg.mean()
        
        return average_over_batch




def clear_lines(num_lines):
    """Clear the last num_lines lines in the terminal."""
    for _ in range(num_lines):
        sys.stdout.write('\033[F')  # Move cursor up one line
        sys.stdout.write('\033[K')  # Clear the line
    sys.stdout.flush()

# class FixedSizeQueue:
#     def __init__(self, batch_size, max_size, default_value=(2, 0, 2)): # current_position (0: long, 1: short, 2: none), end_tick_effective_leverage, action (0: long, 1: short, 2: none, 3:close)
#         implement batch size to support multiple sequences. Can we put this on the GPU?
        
#         self.max_size = max_size
#         self.queue = deque([default_value] * max_size, maxlen=max_size)  # Fill deque with default values

#     def add_item(self, item):
#         self.queue.append(item)  # Adds item to the end; oldest item automatically removed if maxlen is reached

#     def get_items(self):
#         return list(self.queue)  # Returns a list of items in the queue for easy viewing

class FixedSizeQueue:
    def __init__(self, batch_size, max_size, device=None, default_value=(2, 0, 2)):
        """
        Initializes a fixed-size queue for multiple sequences (batches).

        Parameters:
        - batch_size (int): Number of sequences in the batch.
        - max_size (int): Maximum size of the queue.
        - device (torch.device, optional): The device on which tensors are allocated.
        - default_value (tuple): The default value to initialize the queue with.
          Format: (current_position, end_tick_effective_leverage, action)
        """
        self.batch_size = batch_size
        self.max_size = max_size
        self.device = device if device else torch.device('cpu')
        
        # Initialize the queue tensor with the default values
        # Shape: (batch_size, max_size, 3)
        # self.queue = torch.full(
        #     (batch_size, max_size, 3),
        #     default_value,
        #     dtype=torch.int16, #.float32,
        #     device=self.device
        # )
        
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
          Each item should be a tuple/list of (current_position, end_tick_effective_leverage, action).
          Shape: (batch_size, 3)
        """
        if isinstance(items, list):
            items = torch.tensor(items, dtype=torch.float32, device=self.device)
        elif isinstance(items, torch.Tensor):
            items = items.to(self.device, dtype=torch.float32)
        else:
            raise TypeError("Items must be a list of tuples or a torch.Tensor")
        
        if items.shape != (self.batch_size, 3):
            raise ValueError(f"Expected items of shape ({self.batch_size}, 3), got {items.shape}")
        
        # Update the queue at the current pointers
        # self.queue[torch.arange(self.batch_size, device=self.device), self.pointers] = items
        batch_indices = torch.arange(self.batch_size, device=self.device)
        self.queue[batch_indices, self.pointers] = items
        
        # Increment pointers and wrap around using modulo for circular buffer
        self.pointers = (self.pointers + 1) % self.max_size # WTF is this doing?https://chatgpt.com/c/674cfab6-63fc-8006-9f43-f5e1c013e295
    
    def get_items(self):
        """
        Retrieves the queue items in the correct order for each sequence in the batch.

        Returns:
        - torch.Tensor: A tensor of shape (batch_size, max_size, 3) containing the queue items.
        """
        # Create an index tensor for each position in the queue
        indices = (self.pointers.unsqueeze(1) + torch.arange(self.max_size, device=self.device)) % self.max_size
        # Gather the items in the correct order
        ordered_queue = torch.gather(self.queue, 1, indices.unsqueeze(-1).expand(-1, -1, 3))
        return ordered_queue
    
    def reset(self):
        """
        Resets the queue to the default values.
        """
        default_tensor = torch.tensor((2, 0, 2), dtype=torch.float32, device=self.device)
        self.queue = default_tensor.unsqueeze(0).unsqueeze(0).repeat(self.batch_size, self.max_size, 1)
        self.pointers.zero_()

def parse_timeframe_to_minutes(timeframe):
    """
    Converts a timeframe string (e.g., '5min', '1h') into minutes.

    Args:
        timeframe (str): The timeframe string (e.g., '5min', '1h').

    Returns:
        int: The equivalent timeframe in minutes.
    """
    match = re.match(r"(\d+)([a-zA-Z]+)", timeframe)
    if not match:
        raise ValueError(f"Invalid timeframe format: {timeframe}")

    value, unit = match.groups()
    value = int(value)

    if unit.lower() in ['min', 'm']:
        return value
    elif unit.lower() in ['h', 'hour', 'hours']:
        return value * 60
    elif unit.lower() in ['d', 'day', 'days']:
        return value * 1440  # 24 * 60
    else:
        raise ValueError(f"Unknown unit '{unit}' in timeframe '{timeframe}'")

# region merge current count_five_minute_ticks with this function as one!
# def count_ticks_in_block(current_time, unit_timeframe='5min', block_timeframe='30min'):
#     """
#     Calculate the number of unit ticks remaining in a block from the current time.

#     Args:
#         current_time (datetime): The current time.
#         unit_timeframe (str): The unit interval as a string (e.g., '5min').
#         block_timeframe (str): The block interval as a string (e.g., '30min').

#     Returns:
#         int: The number of unit ticks remaining in the block.
#     """
#     # Convert timeframes to minutes
#     unit_minutes = parse_timeframe_to_minutes(unit_timeframe)
#     block_minutes = parse_timeframe_to_minutes(block_timeframe)

#     # Determine the start and end of the current block
#     current_minute = current_time.minute
#     start_of_block = current_time.replace(
#         minute=(current_minute // block_minutes) * block_minutes,
#         second=0,
#         microsecond=0
#     )
#     end_of_block = start_of_block + timedelta(minutes=block_minutes)

#     # Calculate total minutes remaining in the block
#     total_minutes = (end_of_block - current_time).total_seconds() / 60

#     if total_minutes <= 0:
#         return 0

#     # Calculate the number of ticks
#     num_ticks = int(total_minutes // unit_minutes)

#     # If there's any remainder, add one tick to include the current partial tick
#     if total_minutes % unit_minutes != 0:
#         num_ticks += 1

#     return num_ticks

def count_five_minute_ticks(current_time, lowest_tf=config['data']['lowest_tf'], target_tf=config['data']['target_tf']):
    """
    Calculate the number of five-minute ticks from current_time to the end of the block.

    Args:
        current_time (datetime): The current time.
        unit_minutes (int): The unit interval in minutes (default is 5).
        block_minutes (int): The block interval in minutes (default is 30).

    Returns:
        int: The number of five-minute ticks remaining in the block.
    """
    # Convert timeframes to minutes
    unit_minutes = parse_timeframe_to_minutes(lowest_tf)
    block_minutes = parse_timeframe_to_minutes(target_tf)
    
    # Determine the start and end of the current block
    current_minute = current_time.minute
    start_of_block = current_time.replace(
        minute=(current_minute // block_minutes) * block_minutes,
        second=0,
        microsecond=0
    )
    end_of_block = start_of_block + timedelta(minutes=block_minutes)

    # Calculate time until the next five-minute tick
    minute_offset = current_time.minute % unit_minutes
    second_offset = current_time.second
    microsecond_offset = current_time.microsecond

    if minute_offset == 0 and second_offset == 0 and microsecond_offset == 0:
        # Current time is exactly on a five-minute tick, so we exclude it
        minutes_to_next_tick = unit_minutes
    else:
        # Time remaining until the next five-minute tick
        minutes_to_next_tick = unit_minutes - minute_offset - (second_offset + microsecond_offset / 1_000_000) / 60

    # Total minutes from the next tick to the end of the block
    total_minutes = (end_of_block - current_time).total_seconds() / 60 - minutes_to_next_tick

    if total_minutes < 0:
        return 0

    # Calculate the number of five-minute ticks remaining
    num_ticks = int(total_minutes // unit_minutes) + 1

    return num_ticks
# endregion



def epoch_loop(model, data_loader, optimizer, device, timestamp, epoch, epsilon, replay_buffer, is_training=True, scheduler=None, scaler=None):
    if is_training:
        model.train()
        if scheduler is None:
            raise ValueError("Training mode requires a scheduler")
    else:
        model.eval()

    optimizer.zero_grad(set_to_none=True)
    
    # Parameters for epsilon-greedy policy
    gamma = config['training']['gamma'] # Gamma - 1 is higher future reward focused, 0 is immediate reward focussed
    
    criterion = nn.MSELoss() # RegularizedCrossEntropyLoss(l1_lambda=0.0001)#, l2_lambda=0.0001)
    nan_detected = False
    print_every_x_indicies = 100 #max(len(data_loader) / config['training']['accumulation_per_epoch'], 1)
    
    # Variables for tracking profits
    cumulative_profit = []
    
    # Initialize tracking variables for hold times and trades
    profitable_trade_hold_times = []
    profitable_trade_pcts = []
    unprofitable_trade_hold_times = []
    unprofitable_trade_pcts = []
    # Initialize results list for logging
    results = []
    
    process_batch = ProcessBatch(model) # look at this
    prefetcher = DataPrefetcher(data_loader, device)
    # reward_objects = prefetcher.reward_objects
    
    if is_training:
        cum_return = CumulativeReturnBatch(threshold=-0.95, batch_size=BATCH_SIZE, total_steps=len(data_loader), device=device)
        action_leverage_history = FixedSizeQueue(batch_size=BATCH_SIZE, max_size=config['model']['action_sequence_length'], device=device)
    else:
        cum_return = CumulativeReturnBatch(threshold=-0.95, batch_size=1, device=device)
        action_leverage_history = FixedSizeQueue(batch_size=1, max_size=config['model']['action_sequence_length'], device=device)
        
    current_trades = [None] * BATCH_SIZE
    new_current_trades = [None] * BATCH_SIZE

    # Initialize lists to collect data
    timestamps = []
    open_prices = []
    
    # Initialize loss trackers
    total_loss = 0.0
    epoch_loss = 0.0
    epoch_reward = 0.0
    batch_loss = torch.tensor(0.0, dtype=torch.float16, device=device)
    
    target_model = copy.deepcopy(model) # (Throw away model)
    
    batch = prefetcher.next_batch
    index = 0
    while batch is not None:
        loop_start = time.time()
        #Get batch device...
        if batch is None:
            logging.info(f"Reached end of data at index {index}")
            break
        cum_return.update_done_tensor(index)
        
        if index == len(data_loader) - 1:
            done = torch.ones(BATCH_SIZE, dtype=torch.bool, device=device)
        else:
            done = cum_return.get_done_tensor()
        
        # Need to know the shape of raw_data and current_price with batch_size > 1
        raw_data = batch['raw_data'] # Open, high, low, close
        current_prices = batch['lowest_tf_next_n_opens'][:, 0] #[0].iloc[0]
        current_positions = batch['current_positions']  # What object type is this? Need to know the shape of this tensor
        state = batch['input'].to(device)
        next_state = batch['next_input']
        reward_objects = batch['reward_object']
        
        # Use the passed scaler for training
        if is_training and scaler is None:
            raise ValueError("Training mode requires a scaler")

        # Convert action_leverage_history into tensors
        history_items = action_leverage_history.get_items()
        # Extract positions, leverages, and actions using tensor slicing
        positions_tensor = history_items[:, :, 0].long()       # Shape: [batch_size, max_size]
        leverages_tensor = history_items[:, :, 1].float()      # Shape: [batch_size, max_size]
        actions_tensor = history_items[:, :, 2].long()         # Shape: [batch_size, max_size]
        if epoch == 0:
            # move this to the prefetcher, if possible
            actions = batch['ideal_action'].to(device)
        else:
            if random.random() < epsilon and is_training:
                # Explore: select a random action
                actions = torch.randint(0, model.num_actions, (BATCH_SIZE, 1), device=device)
            else:
                # Exploit: select the best action                
                with torch.no_grad():
                    with amp.autocast(device_type='cuda', dtype=torch.float16):
                        # need to define the q_values shape in comments so i know what its supposed to be
                        q_values = process_batch( # 0: Long, 1: Sell, 2: Hold, 3: Close
                            tensors = state,
                            positions = positions_tensor, #torch.tensor([current_position], device=device),
                            effective_leverages = leverages_tensor, #effective_leverage_tensor,
                            actions = actions_tensor,
                        )
                # Exploit: select the best action
                actions = q_values.max(1)[1]
                
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        results = []
        # Update current trades and Simulate environment response for each batch item
        for i in range(BATCH_SIZE):
            # if new_current_trades[i] is not None and new_current_trades[i].exit_index is not None:
            #     print("BOBOS")
            if current_trades[i] is not None and new_current_trades[i] is not None and current_trades[i].entry_index != new_current_trades[i].entry_index:
            #     print("BOBOS")
            # if current_trades[i] != new_current_trades[i] and new_current_trades[i].exit_index is not None:
                # Trade was closed
                profit_pct = current_trades[i].calculate_profit_pct(current_prices[i]).item()
                hold_time = current_trades[i].hold_time
                if profit_pct > 0:
                    profitable_trade_hold_times.append(hold_time)
                    profitable_trade_pcts.append(profit_pct)
                else:
                    unprofitable_trade_hold_times.append(hold_time)
                    unprofitable_trade_pcts.append(profit_pct)
                cum_return.add_pct_change(profit_pct, i)
                cumulative_profit.append(profit_pct)
            current_trades[i] = new_current_trades[i]
            action = actions[i].item()
            result = reward_objects[i].simulate_environment_step(action)
            results.append(result)
        
        # Unpack results
        rewards_list, new_current_trades, profits, hold_times, end_tick_effective_leverages = zip(*results)
        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)  # Shape: [batch_size, 4]
        rewards = rewards[torch.arange(rewards.size(0)), actions]
        epoch_reward += rewards.sum().item()
        
        batch = prefetcher.next() # Update here AFTER all the other stuff is done above, needs to likely be here for multi-threading to work properly when it gets implemented.
        
        # Update action leverage history
        items_to_add = torch.stack([
            current_positions.long(),
            torch.tensor(end_tick_effective_leverages, device=device),
            actions.long()
        ], dim=1)  # Shape: [batch_size, 3]
        action_leverage_history.add_item(items_to_add)
        
        # Get the updated history for next_state
        next_history_items = action_leverage_history.get_items()
        next_positions_tensor = next_history_items[:, :, 0].long()     # Shape: [batch_size, sequence_length]
        next_leverages_tensor = next_history_items[:, :, 1].float()    # Shape: [batch_size, sequence_length]
        next_actions_tensor = next_history_items[:, :, 2].long()       # Shape: [batch_size, sequence_length]
        
        # Fix this first:
        for i in range(BATCH_SIZE):
            replay_buffer.add(
                state=(state[i,:,:,:].cpu().numpy(), positions_tensor[i,:].cpu().numpy(), leverages_tensor[i,:].cpu().numpy(), actions_tensor[i,:].cpu().numpy()),
                action=actions[i].cpu().numpy(),
                reward=rewards[i].cpu().numpy(),
                next_state=(next_state[i,:,:,:].cpu().numpy(), next_positions_tensor[i,:].cpu().numpy(), next_leverages_tensor[i,:].cpu().numpy(), next_actions_tensor[i,:].cpu().numpy()),
                done=done[i].cpu().numpy()
            )
        
        # Sample a mini-batch from replay buffer and train
        if len(replay_buffer) >= BATCH_SIZE and len(replay_buffer) >= 100:
            experiences = replay_buffer.sample()
            states, actions_batch, rewards_batch, next_states, dones_batch = experiences

            # Unpack states
            state_tensors, state_positions, state_effective_leverages, state_actions = states
            state_tensors = torch.tensor(state_tensors, dtype=torch.float32, device=device)
            state_positions = torch.tensor(state_positions, dtype=torch.long, device=device)
            state_effective_leverages = torch.tensor(state_effective_leverages, dtype=torch.float32, device=device)
            state_actions = torch.tensor(state_actions, dtype=torch.long, device=device)

            # Unpack next_states
            next_state_tensors, next_state_positions, next_state_effective_leverages, next_state_actions = next_states
            next_state_tensors = torch.tensor(next_state_tensors, dtype=torch.float32, device=device)
            next_state_positions = torch.tensor(next_state_positions, dtype=torch.long, device=device)
            next_state_effective_leverages = torch.tensor(next_state_effective_leverages, dtype=torch.float32, device=device)
            next_state_actions = torch.tensor(next_state_actions, dtype=torch.long, device=device)

            # Convert actions, rewards, dones
            actions_batch = torch.tensor(actions_batch, dtype=torch.long, device=device)
            rewards_batch = torch.tensor(rewards_batch, dtype=torch.float32, device=device)
            dones_batch = torch.tensor(dones_batch, dtype=torch.float32, device=device)
            
            with amp.autocast(device_type='cuda', dtype=torch.float16):
                with torch.set_grad_enabled(is_training):
                    # Get current Q-values
                    current_q_values = model(
                        tensors=state_tensors,
                        positions=state_positions,
                        effective_leverages=state_effective_leverages,
                        actions=state_actions
                    ).gather(1, actions_batch.unsqueeze(1))
            
                with torch.no_grad():
                    # Get target Q values
                    max_next_q_values = target_model(
                        tensors=next_state_tensors,
                        positions=next_state_positions,
                        effective_leverages=next_state_effective_leverages,
                        actions=next_state_actions
                    ).max(1)[0].unsqueeze(1)
                    target_q_values = rewards_batch.unsqueeze(1) + (gamma * max_next_q_values * (1 - dones_batch.unsqueeze(1)))
            
            # Compute loss
            batch_loss = criterion(current_q_values, target_q_values.to(torch.float16))
            total_loss += batch_loss.item()
            epoch_loss += batch_loss.item()
            
            if torch.isnan(batch_loss).any():
                logging.info(f"NaN detected in batch loss at index {index}")
                nan_detected = True
                break
            if is_training:
                scaler.scale(batch_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if index % MODEL_UPDATE_FREQ == 0 and is_training:
                # Update target network
                target_model.load_state_dict(model.state_dict())
        
        # Collect data for plotting and logging
        timestamps.extend([rd.name for rd in raw_data])  # Assuming rd.name is the timestamp
        open_prices.extend([rd['open'] for rd in raw_data])
        
        # Collect results for logging
        for i in range(BATCH_SIZE):
            result_entry = {
                'index': index,
                'timestamp': raw_data[i].name,
                'action': actions[i].item(),
                'cumulative_profit': cum_return.get_cumulative_return().cpu().numpy(),
                'reward': rewards[i].item(),
                'rewards': rewards_list[i],
                'current_price': current_prices[i].item(),
                'next_prices': batch['next_n_opens'][i],
                'lowest_tf_next_prices': batch['lowest_tf_next_n_opens'][i],
                'current_position': current_positions[i].item(),
                'profit': profits[i],
                'hold_time': hold_times[i],
                'effective_leverage': end_tick_effective_leverages[i],
                'epsilon': epsilon,
            }
            results.append(result_entry)

        if index % print_every_x_indicies == 0 or index == 0 or cum_return.should_halt_all_outputs():# """Need to fix references of done, in this case we want the final print in the event that being done haults further processing. NOTE: see 'if done:/n break' below."""
            current_time = time.time()
            elapsed_time = (current_time - loop_start)# * print_every_x_indicies
            time_per_index = (elapsed_time / print_every_x_indicies)# / (index + 1)
            progress_percent = ((index + 1) / (len(data_loader))) * 100
            remaining_iterations = len(data_loader) - (index + 1)
            estimated_remaining_time = remaining_iterations * time_per_index * print_every_x_indicies
            
            if len(profitable_trade_hold_times) > 0:
                avg_profitable_hold_time = sum(profitable_trade_hold_times) / len(profitable_trade_hold_times)
                avg_profitable_trade_pct = sum(profitable_trade_pcts) / len(profitable_trade_pcts)
                profitable_pct = len(profitable_trade_hold_times) / len(cumulative_profit) *100
            else:
                avg_profitable_hold_time = 1
                avg_profitable_trade_pct = 0
                profitable_pct = 0.0
            if len(unprofitable_trade_hold_times) > 0:
                avg_unprofitable_hold_time = sum(unprofitable_trade_hold_times) / len(unprofitable_trade_hold_times)
                avg_unprofitable_trade_pct = sum(unprofitable_trade_pcts) / len(unprofitable_trade_pcts)
            else:
                avg_unprofitable_hold_time = 1
                avg_unprofitable_trade_pct = 0
            
            if index == 0:
                print("\n"*(12))
            clear_lines(12)
            # if cum_return <= -1.0 and not negative_logged: # This wont be 100% accurate, because we aren't running it outside of thise print accumulation logic.
            #     negative_logged = True
            #     negative_index = index
            #     logging.info(f"Index: {index}. NEGATIVE AT THIS INDEX! (but not exactly tho)")
            # elif negative_logged:
            #     logging.info(f"Index: {index}. Negative @ index {negative_index}")
            # else:
            logging.info(f"Index: {index} / {len(data_loader)}")
            logging.info(f"GPU memory usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            logging.info(f"Progress: {progress_percent:.2f}% | Est. time: {estimated_remaining_time:.2f}s | Time/index: {time_per_index:.4f}s | Cumulative Avg Loss: {epoch_loss/(index+1):.9f}")
            logging.info(f"Avg Batch Losses: {batch_loss/(index+1):.9f}")
            logging.info(f"Last Avg Loss: {batch_loss:.9f}")
            logging.info(f"Profit %: AVG | COMPOUND --- {cum_return.get_average_return().cpu().numpy()} | {cum_return.get_cumulative_return().cpu().numpy()}")
            logging.info(f"Trades: {len(cumulative_profit)} ({profitable_pct:.2f}% profitable) | Avg. Profit {avg_profitable_trade_pct:.2f}%. Hold Time {avg_profitable_hold_time:.2f} ticks. | Avg. Loss {avg_unprofitable_trade_pct:.2f}%. Hold Time {avg_unprofitable_hold_time:.2f} ticks.")
            if is_training:
                logging.info(f"Current Learning Rate: {scheduler.get_last_lr()[0]:.10f}")
            else:
                logging.info(f"Current Learning Rate: 'N/A'")
            logging.info(f"Action taken: {action}")
            logging.info(f"Total Reward Received (per index): {epoch_reward:.4f}, ({epoch_reward/(index+1):.5f})")
            logging.info(f"Epsilon: {epsilon:.4f}")
        batch_loss = torch.tensor(0.0, dtype=torch.float16, device=device)
        index += 1
        if cum_return.should_halt_all_outputs():
            break
        
        
    end_loop = time.time()
    logging.info(f"{'Training' if is_training else 'Validation'} loop time: {end_loop - loop_start}")
    
    if is_training:
        filepath = f"data/models/{timestamp}/epoch_{epoch}/training/"
    else:
        filepath = f"data/models/{timestamp}/epoch_{epoch}/validation/"
    
    # Create the directory if it doesn't exist
    os.makedirs(filepath, exist_ok=True)
        
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(filepath, "log.csv")
    results_df.to_csv(csv_path, index=False)
    
    # Compute and log statistics
    # action_counts = results_df['action'].value_counts()
    # average_reward = results_df['reward'].mean()
    # total_profit = results_df['profit'].sum()
    # average_profit = results_df['profit'].mean()
    # average_hold_time = results_df['hold_time'].mean()

    # print("\nEPOCH Training statistics:")
    # print(f"Average reward: {average_reward:.4f}")
    # print(f"Total profit: {total_profit:.4f}")
    # print(f"Average profit per trade: {average_profit:.4f}")
    # print(f"Action counts:\n{action_counts}")
    # print(f"Average hold time: {average_hold_time:.2f} steps")
    
    return epoch_loss, nan_detected, epoch_reward

def softmax(x):
    e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return e_x / e_x.sum()

def check_nan(tensor, tensor_name):
    if torch.isnan(tensor).any():
        logging.info(f"NaN detected in {tensor_name}")
        return True
    return False

def generate_models(data): #dimension
    print(f"Data.datatypes amount: {data.data_types}")
    
    dicts = None
    info_dict = None
    
    model = TradingModel(
        num_types=data.data_types,
        timeframes=data.timeframes,
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
    # Check if there's a checkpoint to resume from
    if config['training'].get('resume_from_checkpoint'):
        checkpoint_path = config['training']['resume_from_checkpoint']
        # print(checkpoint_path)
        # exit()
        if os.path.isfile(checkpoint_path):
            logging.info(f"Loading checkpoint '{checkpoint_path}'")
            checkpoint = torch.load(checkpoint_path)
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model_state_dict'])
            # model = checkpoint['model'] #torch.load(checkpoint_path, map_location=device) # map location?
            # model.to(device)
            optimizer_state_dict = checkpoint['optimizer_state_dict']
            scheduler_state_dict = checkpoint['scheduler_state_dict']
            scaler_state_dict = checkpoint['scaler_state_dict']
            dicts = {
                'optimizer_state_dict': optimizer_state_dict,
                'scheduler_state_dict': scheduler_state_dict,
                'scaler_state_dict': scaler_state_dict,
            }
            
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            # scaler.load_state_dict(checkpoint['scaler_state_dict'])
            best_val_loss = checkpoint['best_val_loss']
            best_val_reward = checkpoint['best_val_reward']
            train_losses = checkpoint['train_losses']
            val_losses = checkpoint['val_losses']
            
            
            # best loss = get best loss and save it
            
            epsilon_start = checkpoint.get('epsilon_start', 0.1721)
            
            
            # Regular expression to match the timestamp in the path
            match = re.search(r'\d{8}_\d{6}', checkpoint_path)
            # Extract the timestamp if found
            timestamp = match.group(0) if match else None
            print("Timestamp:", timestamp)
            info_dict = {
                'start_epoch': start_epoch,
                'best_val_loss': best_val_loss,
                'best_val_reward': best_val_reward,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'epsilon_start': epsilon_start,
                'timestamp': timestamp,
            }
            if timestamp is None:
                exit()
            logging.info(f"Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']})")
        else:
            logging.info(f"No checkpoint found at '{checkpoint_path}'")
            exit()
    
    # if info_dict is None:
    #     info_dict = {'start_epoch': 0}
    return [model], dicts, info_dict

def init_training(config, train_data_loader, val_data_loader, model, dicts, info_dicts):
    logging.info(f"Training {model.model_type}:")
    # Call the train function
    train_losses, val_losses = train(config, train_data_loader, val_data_loader, model, dicts, info_dicts)

    # Print average losses per epoch
    logging.info("\nAverage Losses per Epoch:")
    logging.info("Epoch | Training Loss | Validation Loss")
    logging.info("-" * 40)
    for epoch, (train_loss_, val_loss_) in enumerate(zip(train_losses, val_losses), 1):
        logging.info(f"{epoch:5d} | {train_loss_:13.6f} | {val_loss_:14.6f}")

    try:
        # Print overall average losses
        if train_losses and val_losses:
            avg_train_loss = sum(train_losses) / len(train_losses)
            avg_val_loss = sum(val_losses) / len(val_losses)
            logging.info("\nOverall Average Losses:")
            logging.info(f"Average Training Loss: {avg_train_loss:.6f}")
            logging.info(f"Average Validation Loss: {avg_val_loss:.6f}")
        else:
            logging.info("\nNo training completed. Check for errors in the training process.")
    except Exception as e:
        logging.error(f"An error occurred while calculating average losses: {str(e)}")

    results_df = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'val_loss': val_losses
    })

    save_path = os.path.join(os.path.dirname(config['model']['output']['model_directory']), f"{timestamp}.csv")
    results_df.to_csv(save_path, index=False)
    
    # Clear GPU memory after each model training
    model.cpu()
    del model
    torch.cuda.empty_cache()
    
if __name__ == '__main__':
    try: #to continue, implement 2 timeframe model. Implement float16 for mixed precision training, but use float32 for loss, and implement time/position embedding
        # config = load_config('config.yaml')
        logging.info(f"Starting training with configuration: {config}")
                
        # Load and preprocess data
        data = Data(config, interpolate_to_target=False, scaler=StandardScaler())#, end_date='2022-09-30') #data_30min, data_4hr = get_data(config['data']['filepath'])#load_processed_data('data/RSI-n-MACD_30min_4hr_combined.parquet') #get_data(config['data']['filepath']) #, start_date='2020-07-01', end_date='2021-04-01')
        models, dicts, info_dicts = generate_models(data)
        if info_dicts is None:
            epoch = 0
        else:
            epoch = info_dicts['start_epoch']
        train_data_loader, val_data_loader = get_data_loaders(data, config, epoch=epoch)

        for i, model in enumerate(models):
            init_training(config, train_data_loader, val_data_loader, model, dicts, info_dicts)

    except Exception as e:
        logging.error(f"An error occurred during training: {str(e)}")
        raise

# now that I know weight decay is the problem of converging to mean value, change the loss fn back to using the MSE on the predicted and so on. This would allow me to use float 16s again.
# I could try to have the model output both the 4h and 30 min values, and get the loss for both. This may incentivize the model to understand both the 4h and 30min values and make better predictions.
# implement graph saving.

# Consider having the model calculate the graph again.

# Consider impleminting weight decay but much smaller than 0.1. Try 0.01

# try setting the leverages bounded to 100 or 25 in model_data.py. This way the model can't learn a huge leverage. See how that affects the loss and model accuracy.

# convert model to Categorical loss model.



# check where every save is, ensure that log goes to the correct directory and checkpoints too.