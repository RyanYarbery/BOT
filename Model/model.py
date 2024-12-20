# model.py

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch
import math
# from numba import jitclass, int32
from pympler import asizeof
from numba import njit
import logging
import pandas_ta as ta
# from ..config import load_config
import torch.nn.functional as F
from torch.nn import LeakyReLU
import torch.jit as jit
import sys
import os
import torch.nn.functional as F

# Add the parent directory to sys.path#+
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))#+
sys.path.append(parent_dir)#+
from config import load_config

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers= [
                        logging.FileHandler('log.log'),
                        logging.StreamHandler()
                    ])

class ModelType:
    PERCENTHIGH = 0
    PERCENTLOW = 1# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ how to embed model class into the saved model pth file? YES https://chatgpt.com/c/46dfddd8-8c4d-4503-aa0b-784d017dd49f
    LEVERAGE_RETURNS = 2

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        if isinstance(other, int):
            return self.value == other
        elif isinstance(other, ModelType):
            return self.value == other.value
        return False

    def __str__(self):
        if self.value == 0:
            return "PERCENTHIGH"
        elif self.value == 1:
            return "PERCENTLOW"
        elif self.value == 2:
            return "LEVERAGE_RETURNS"
        else:
            return "UNKNOWN"
        
    def get_loss_fn(self):
        if self.value == 0:  # PERCENTHIGH or ABSOLUTEHIGH
            return self.asymmetric_mse_loss_high
        elif self.value == 1:  # PERCENTLOW or ABSOLUTELOW
            return self.asymmetric_mse_loss_low
        elif self.value == 2:
            return self.my_loss_fn
        else:
            raise ValueError(f"Unknown loss function type: {self}")
    
    @staticmethod
    def my_loss_fn(prediction: torch.Tensor, target: torch.Tensor, target_max: float):
        # denormalized_values = prediction * target_max
        # loss = torch.square(denormalized_values - target).mean()
        # return denormalized_values, loss # temp removing the signal mismatch penalty
        
        # Denormalize the prediction
        denormalized_values = prediction * target_max

        # Calculate the base error
        delta = target - denormalized_values
        
        # Calculate the sign mismatch penalty
        sign_mismatch = (denormalized_values * target) < 0
        
        # Base penalty factor
        base_penalty = 1.45 #2.0
        
        # Calculate a continuous penalty based on the magnitude of the error
        continuous_penalty = torch.where(
            sign_mismatch,
            base_penalty * (1 + torch.abs(delta)),  # Increase penalty for larger errors
            torch.ones_like(delta)
        )
        
        # Apply the penalty
        penalized_delta = delta * continuous_penalty
        
        # Calculate the final loss
        loss = torch.square(penalized_delta).mean()
        
        return denormalized_values, loss
    
    @staticmethod
    def asymmetric_mse_loss_high(previous_close: float, predicted_high: torch.Tensor, actual_high: torch.Tensor):
        # print(f"previous_close: {previous_close}, actual_high: {actual_high}, predicted_high: {predicted_high}")
        max_leverage = torch.min((previous_close / (previous_close - actual_high)), torch.full_like(previous_close, 20))
        predicted_max_leverage = torch.min((previous_close / (previous_close - predicted_high)), torch.full_like(previous_close, 20))
        
        max_leverage_earnings = (1 + max_leverage * ((actual_high - previous_close) / previous_close)) * previous_close # PCT
        predicted_max_leverage_earnings = (1 + predicted_max_leverage * ((predicted_high - previous_close) / previous_close)) * previous_close # PCT

        earnings_delta = (predicted_max_leverage_earnings - max_leverage_earnings) / previous_close
        
        print(f"Potential Earnings Delta: {earnings_delta}\n Previous Close: {previous_close}\n Actual High: {actual_high}\n Predicted High: {predicted_high}\n Max Leverage: {max_leverage}\n Predicted Max Leverage: {predicted_max_leverage}\n Max Leverage Earnings: {max_leverage_earnings}\n Predicted Max Leverage Earnings: {predicted_max_leverage_earnings}")
        
        loss = torch.square(earnings_delta).mean() # Ensure 0 values are epsilon
        print(f"Loss: {loss:.3f}")
        exit()
        return loss
        
    @staticmethod
    def asymmetric_mse_loss_low(previous_close: float, predicted_high: torch.Tensor, actual_high: torch.Tensor): #predicted: torch.Tensor, actual: torch.Tensor, penalty: float = 1.1
        overestimation_penalty = actual_high
        # percentage_diff = (predicted - actual) / actual * 10000 #.detach()
        
        # loss = torch.where(percentage_diff > 0, 
        #                 overestimation_penalty * percentage_diff, 
        #                 percentage_diff)

        # loss = torch.square(loss)

        # return loss.mean() #.requires_grad_(True)

def positional_encoding(seq_len, d_model, device): # Scaling?:https://chatgpt.com/c/66d7ac81-be6c-8006-8976-12ebd6ed28fc
    position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)                # v Try 40_000 here.
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) * -(math.log(100_000.0) / d_model))
    pos_encoding = torch.zeros(seq_len, d_model, device=device)
    pos_encoding[:, 0::2] = torch.sin(position * div_term)
    pos_encoding[:, 1::2] = torch.cos(position * div_term)
    return pos_encoding

class CustomTransformer(nn.Module):
    def __init__(self, d_models, num_heads, input_ticks, num_types, dropout=0.1, is_pre_process=True, dim_feedforward=None):
        super(CustomTransformer, self).__init__()
        self.num_layers = len(d_models)
        self.is_preprocess = is_pre_process
        self.pos_encoder = nn.Parameter(
            positional_encoding(input_ticks * num_types, d_models[0], device='cuda'), #d_models[0]
            requires_grad=False
        )
        
        if not is_pre_process:
            self.attention = nn.Linear(d_models[-1], 1) # delete me
            # Multi-head attention layer
            self.multihead_attn = nn.MultiheadAttention(embed_dim=d_models[-1], num_heads=num_heads)
            # Output layer for the joint model
            self.output = nn.Linear(d_models[-1], 1)  # 2 outputs for long, short
            self.sigmoid = nn.Sigmoid()

        # Transformer encoder layers with varying d_model sizes
        self.transformer_layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer_d_model = d_models[i]
            next_d_model = d_models[i + 1] if i + 1 < self.num_layers else d_models[-1]
            self.transformer_layers.append(
                nn.TransformerEncoderLayer(
                    d_model=layer_d_model,
                    nhead=num_heads,
                    dim_feedforward=layer_d_model * 4 if dim_feedforward is None else dim_feedforward,
                    dropout=dropout,
                    batch_first=True,
                    activation=nn.LeakyReLU(negative_slope=0.05)
                )
            )

            # Add a projection layer if the next layer has a different d_model
            if i + 1 < self.num_layers and layer_d_model != next_d_model:
                self.transformer_layers.append(nn.Linear(layer_d_model, next_d_model))
                self.transformer_layers.append(nn.LayerNorm(next_d_model))
                
    def forward(self, x): #action_embeddings=None
        # x shape: (batch_size, seq_length, input_dim)
        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.pos_encoder[:seq_len, :].unsqueeze(0).expand(x.size(0), -1, -1)
        
        # Apply Transformer encoder layers
        for layer in self.transformer_layers:
            x = layer(x)

        return x # shape: (batch_size, seq_length, embedding_dim)
        
        # if self.is_preprocess:
        #     return x  # Return the processed sequence
        # else:
        #     # For the joint model, use the last sequence element
        #     return x[:, 0, :]  # shape: shape: (batch_size, seq_length, embedding_dim)
            
            # if action_embeddings is None:
            #     logging.warning("No original embeddings provided to CustomTransformer forward pass. Using random numbers for prediction. Exiting")
            #     exit()
            
            # # Step 1: Calculate cosine similarity between the example embedding and each original embedding
            # similarities = F.cosine_similarity(output.unsqueeze(0), action_embeddings)

            # # Step 2: Find the index of the maximum similarity (closest match)
            # predicted_action = similarities.argmax()
            # return predicted_action
            
            # attn_output, _ = self.multihead_attn(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
            # attention_weights = F.softmax(self.attention(x).squeeze(-1), dim=1)  # shape: (batch_size, seq_length)
            # output = torch.bmm(attention_weights.unsqueeze(1), x).squeeze(1)  # shape: (batch_size, d_model)
                        
            # logits = self.output(output)#attn_output)
            # sigmoid = self.sigmoid(logits)
            # return logits, sigmoid #, predicted_class
            # probabilities = F.softmax(logits, dim=1) #consider probability accuracy during evaluation. If it says long, but isn't confident, and its wrong, it would be good to know.
            # predicted_class = torch.argmax(probabilities, dim=1)
"""
Notes on this class: 10/11/24
I could get rid of the self.attention and self.multihead_attn as they are not used in the current implementation, i think.
I should also ensure that I'm passing the embeddings to the forward function correctly, specifically, the current_position token.
Ensure I'm properly indexing the embeddings correctly!
"""
class TradingModel(nn.Module):
    def __init__(self, num_types, timeframes, d_models_pre, d_models_joint, num_heads, model_type, max_seq_length, target_tf, action_sequence_length, dropout=0.1, pre_process=True, dim_feedforward=None):
        super(TradingModel, self).__init__()
        self.is_preprocess = pre_process
        self.timeframes = timeframes
        self.sequence_length = max_seq_length
        self.action_sequence_length = action_sequence_length
        if pre_process:
            embedding_dims = d_models_pre[0]
        else:
            embedding_dims = d_models_joint[0]
        self.type_embedding = nn.Embedding(num_types, embedding_dims) # Embedding for the type of each value.
        self.timeframe_embedding = nn.Embedding(len(timeframes), embedding_dims)
        self.effective_leverage_projection = nn.Linear(1, embedding_dims)
        
        # state and action embedding could be combined to 5, Long, Short, None/hold/na, buy, sell... consider merging ALL embeddings!
        # Update embeddings for actions and states
        self.num_actions = 4  # Number of possible actions: buy, sell, hold, close
        self.action_embedding = nn.Embedding(self.num_actions, embedding_dims)
        self.state_embedding = nn.Embedding(3, embedding_dims) # 3 states: long, short, none
        # self.current_state = nn.Parameter(self.state_embedding(torch.tensor([2])), requires_grad=False)  # 2 represents 'none' state
        # self.current_effective_leverage = nn.Parameter(self.effective_leverage_projection(torch.tensor([0.0])), requires_grad=False)
        
        self.time_projection = nn.Linear(8, embedding_dims) # 8 times, min, hour, day, month as both sin and cos.
        # self.current_action = nn.Parameter(self.action_embedding(torch.tensor([2])), requires_grad=False)  # 2 represents 'hold' action
        
        # Multi-head attention layer
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dims, num_heads=num_heads)
        
        # Create a unique linear layer for THE VALUE OF each type (ie, RSI (70), MACD(-0.1)...)
        self.value_type_layers = nn.Linear(1, embedding_dims)
        
        self.model_type = ModelType(model_type)
        self.target_tf = target_tf
        
        if pre_process:
            self.preprocess_model = nn.ModuleDict({
                f'pre_process_{timeframe}': CustomTransformer(d_models_pre, num_heads, max_seq_length, num_types, dropout=dropout, is_pre_process=True, dim_feedforward=dim_feedforward) for timeframe in self.timeframes
            })
        
        # Max_seq_length is twice the number of ticks (for 30-minute and 4-hr, post cat)
        self.joint_model = CustomTransformer(d_models_joint, num_heads, max_seq_length*len(timeframes), num_types, dropout=dropout, is_pre_process=False, dim_feedforward=dim_feedforward)
        
        # Output layer to produce Q-values for each action
        self.output_layer = nn.Linear(d_models_joint[-1], self.num_actions)
    
    def forward(self, tensors, positions=None, effective_leverages=None, actions=None): #removed action=None, # Tensors of shape torch.Size([3, *self.num_timeframes*, 1856, 11])
        # tensors shape: (batch_size, num_timeframes, num_values (open, high, rsi[5]"close", etc. (all permuatations of the data)), num_features)
        x_list = []  # Initialize an empty list to store end/resultant x values
        index_of_latest_action = 0
        
        # base_tokens = tensors.shape[2]
        divisor = int(tensors.shape[2]/self.sequence_length)
        
        for tf_idx, timeframe in enumerate(self.timeframes):
            x = tensors[:, tf_idx, :, :]
            feature_value = x[:, :, 0].unsqueeze(-1) # adding unsqueeze # First column, value.
            feature_times = x[:, :, 1:9] # Second to fifth columns, time. (min_sin, cos, hr_sin, hr_cos, day_sin, day_cos, month_sin, month_cos)
            timeframe_feature = x[:, :, -2].long() # Second to last column, convert to long for embedding
            type_feature = x[:, :, -1].long() # Last column, convert to long for embedding (rsi, macd...)
            
            # Apply the appropriate linear layer to each type
            value_embedded = self.value_type_layers(feature_value)
            times_linear = self.time_projection(feature_times)
            type_embedding = self.type_embedding(type_feature)
            timeframe_embedding = self.timeframe_embedding(timeframe_feature)
            
            if timeframe == self.target_tf: # CONSIDER ADDING TIME PROJECTION TO action, position, and effective_leverage tokens to enhance learning.
                unique_tf_features = feature_times[:,::divisor,:]
                unique_tf_features = unique_tf_features.flip(dims=[1]) # Flip because the deque that holds the historic information are in reverse order.
                unique_tf_features = unique_tf_features[:,:self.action_sequence_length,:]
                unique_tf_feature_embedding = self.time_projection(unique_tf_features)
                
                # Process position
                if positions is not None:
                    state_embedded = self.state_embedding(positions)
                else:
                    state_embedded = self.state_embedding(torch.zeros(tensors.size(0), dtype=torch.long, device=tensors.device))
                x_list.append(state_embedded+unique_tf_feature_embedding)
                
                if effective_leverages is not None:
                    leverage = effective_leverages / 20  # Normalize leverage
                    effective_leverage_proj = self.effective_leverage_projection(leverage.unsqueeze(-1))
                else:
                    leverage = torch.zeros(tensors.size(0), 1, device=tensors.device)
                    effective_leverage_proj = self.effective_leverage_projection(leverage)
                x_list.append(effective_leverage_proj+unique_tf_feature_embedding)
                
                if actions is not None:
                    actions_embedded = self.action_embedding(actions)
                else:
                    print("SHIT FUCK BREAK")
                    exit()
                x_list.append(actions_embedded+unique_tf_feature_embedding)
                
                # last action is at the end of the list from the deque object passed to this fn. So we get the next action from the latest provided action token's index
                index_of_latest_action = len(x_list)
            
            # x = value_embedded + times_linear + type_embedding + timeframe_embedding

            # Concatenate embeddings to form a combined embedding\, no reason for value_embedded used here over: times_linear, type_embedding, timeframe_embedding, just needed that shape they all share.
            # Kinda dumb way to do this tbh, using an attention mechanism is dumb, as it is not learning anything.
            # If you can figure out why i'm doing it like this, with bs and the for loop, let me know next time you figure it out dingus!
            # embeddings = torch.cat([value_embedded, times_linear, type_embedding, timeframe_embedding], dim=-1)
            # See https://chatgpt.com/c/67143cd9-227c-8006-b9b6-754e84851025?model=o1-preview to make it better, possibly. ctf+f: 4. Adjust the Model's forward Function
            bs = torch.empty_like(value_embedded)
            for input in range(value_embedded.shape[1]): # 1 is the dimension that has all the inputs/tokens.
                embeddings = torch.stack([
                    value_embedded[:, input, :],
                    times_linear[:, input, :],
                    type_embedding[:, input, :],
                    timeframe_embedding[:, input, :]
                    ], dim=0)

                # Apply attention (using combined embeddings as query, key, and value)
                x, _ = self.multihead_attn(embeddings, embeddings, embeddings)
                x = x.mean(dim=0)
                
                bs[:, input, :] = x  # Store the processed sequence in the correct position in the batch
            x = bs
            if self.is_preprocess:
                x = self.preprocess_model[f'pre_process_{timeframe}'](x)
            x_list.append(x)  # Add the processed x to the list

        # Concatenate all x values after the loop
        x_combined = torch.cat(x_list, dim=1)
        # Pass through the joint model
        # logits, sigmoid = self.joint_model(x_combined)
        
        # return the token value from the joint model as a scalar, 0-2, long, short, none.
        x = self.joint_model(x_combined) #, self.action_embedding)
        
        # Obtain the Q-values for each action
        q_values = self.output_layer(x[:, index_of_latest_action, :])  # Use the representation of the first token # index_of_latest_action was 0 previously
        return q_values
        
"""Global Average Pooling: https://claude.ai/chat/48c8e8d3-2369-4051-b0df-488dd5a3236e
This considers all time steps equally.
pythonCopyoutput = torch.mean(x, dim=1)

Attention Mechanism:
This allows the model to weigh different time steps differently.
pythonCopy# Simplified attention mechanism
attention_weights = F.softmax(self.attention(x), dim=1)
output = torch.sum(x * attention_weights.unsqueeze(-1), dim=1)

Use all time steps:
Pass the entire sequence to the final layer, letting it learn what's important.
pythonCopybatch_size, seq_len, hidden_size = x.shape
output = self.output(x.reshape(batch_size * seq_len, hidden_size)).view(batch_size, seq_len, -1)"""


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class FinancialEmbedding(nn.Module):
#     def __init__(self, num_indicator_types, num_timeframes, d_model, n_heads):
#         super(FinancialEmbedding, self).__init__()
#         self.indicator_embedding = nn.Embedding(num_indicator_types, d_model)
#         self.timeframe_embedding = nn.Embedding(num_timeframes, d_model)
#         self.time_linear = nn.Linear(8, d_model)  # 8 for sin/cos of min, hour, day, month
#         self.value_linear = nn.Linear(1, d_model)
#         # Multi-head attention layer
#         self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads)

#     def forward(self, indicator_type, value, time_encoding, timeframe):
#         indicator_emb = self.indicator_embedding(indicator_type)
#         timeframe_emb = self.timeframe_embedding(timeframe)
#         time_emb = self.time_linear(time_encoding)
#         value_emb = self.value_linear(value.unsqueeze(-1))
        
#         # Concatenate embeddings to form a combined embedding
#         combined_emb = torch.cat([indicator_emb, value_emb, time_emb, timeframe_emb], dim=-1)
        
#         # Apply attention (using combined embeddings as query, key, and value)
#         attn_output, _ = self.multihead_attn(combined_emb.unsqueeze(0), combined_emb.unsqueeze(0), combined_emb.unsqueeze(0))
        
#         return indicator_emb + timeframe_emb + time_emb + value_emb

# class CustomTransformer(nn.Module):
#     def __init__(self, d_models, num_heads, input_ticks, num_types, dropout=0.1, dim_feedforward=None):
#         super(CustomTransformer, self).__init__()
#         self.num_layers = len(d_models)
        
#         self.transformer_layers = nn.ModuleList()
#         for i in range(self.num_layers):
#             layer_d_model = d_models[i]
#             next_d_model = d_models[i + 1] if i + 1 < self.num_layers else d_models[-1]
#             self.transformer_layers.append(
#                 nn.TransformerEncoderLayer(
#                     d_model=layer_d_model,
#                     nhead=num_heads,
#                     dim_feedforward=layer_d_model * 4 if dim_feedforward is None else dim_feedforward,
#                     dropout=dropout,
#                     batch_first=True,
#                     activation=nn.LeakyReLU(negative_slope=0.05)
#                 )
#             )

#             if i + 1 < self.num_layers and layer_d_model != next_d_model:
#                 self.transformer_layers.append(nn.Linear(layer_d_model, next_d_model))
#                 self.transformer_layers.append(nn.LayerNorm(next_d_model))
        
#         self.output = nn.Linear(d_models[-1], 1)

#     def forward(self, x):
#         for layer in self.transformer_layers:
#             x = layer(x)
#         return self.output(x)

# class TradingModel(nn.Module):
#     def __init__(self, num_indicator_types, num_timeframes, d_models, num_heads, max_seq_length, dropout=0.1, dim_feedforward=None):
#         super(TradingModel, self).__init__()
#         self.financial_embedding = FinancialEmbedding(num_indicator_types, num_timeframes, d_models[0])
#         self.transformer = CustomTransformer(d_models, num_heads, max_seq_length * num_timeframes, num_indicator_types, dropout, dim_feedforward)

#     def forward(self, indicator_types, values, time_encodings, timeframes):
#         # Assume input shapes are (batch_size, seq_length, feature_dim)
#         embedded = self.financial_embedding(indicator_types, values, time_encodings, timeframes)
#         logits = self.transformer(embedded)
#         return logits

# # Example usage
# num_indicator_types = 3  # RSI, MACD, open_pct_change
# num_timeframes = 3  # 5min, 30min, 4hr
# d_models = [64, 128, 256]
# num_heads = 8
# max_seq_length = 100
# model = TradingModel(num_indicator_types, num_timeframes, d_models, num_heads, max_seq_length)

# # Dummy input data
# batch_size = 32
# seq_length = 100
# indicator_types = torch.randint(0, num_indicator_types, (batch_size, seq_length))
# values = torch.randn(batch_size, seq_length)
# time_encodings = torch.randn(batch_size, seq_length, 8)
# timeframes = torch.randint(0, num_timeframes, (batch_size, seq_length))

# logits = model(indicator_types, values, time_encodings, timeframes)
# print(logits.shape)  # Should be (batch_size, seq_length, 1)