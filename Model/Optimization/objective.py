# objective.py
import optuna
from config import load_config
from Optimization.hyperparameters import sample_hyperparameters
import yaml

def objective(trial):
    # Sample hyperparameters
    hyperparams = sample_hyperparameters(trial)
    
    # Load and update config
    config = load_config('config.yaml', overrides=hyperparams)
    
    # Save the updated config for this trial (optional)
    trial_number = trial.number
    with open(f'trial_configs/config_trial_{trial_number}.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Train the model and get accuracy
    accuracy = train_model(config)
    
    return accuracy

def train_model(config):
    # Implement your training logic here
    # For example:
    # model = YourModel(config)
    # accuracy = model.train_and_evaluate()
    # Return the accuracy
    accuracy = ...  # Compute accuracy based on your model
    return accuracy
