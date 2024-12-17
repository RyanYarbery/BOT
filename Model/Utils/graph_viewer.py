# viewer.py, Allows for viewing the model graph

from torchview import draw_graph
from model import *
from config import load_config

config = load_config('config.yaml')

model = TransformerModel(
    input_dim=config['model']['input_dim'],
    # sequence_length=config['model']['sequence_length'],
    num_heads=config['model']['nhead'],
    d_models=config['model']['preprocess_layers'],
    #preprocess_layers=config['model']['preprocess_layers'], changed to d_models
    # process_layers=config['model']['process_layers'],
    model_type=ModelType(0)
)
batch_size = 4

# Assuming both inputs have the same shape
input_shape_30min = (batch_size, config['model']['sequence_length'], config['model']['input_dim'])
# input_shape_4hr = (batch_size, config['model']['sequence_length'], config['model']['input_dim'])

model_graph = draw_graph(model, input_size=[input_shape_30min], device='meta') # removed , input_shape_4hr
model_graph.visual_graph.view(directory='model_info1')
model_graph.visual_graph.render('model_info/model_graph1', format="png")