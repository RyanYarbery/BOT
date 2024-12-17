# hyperparameters.py
def sample_hyperparameters(trial):
    hyperparams = {
        'sequence_length': trial.suggest_categorical('sequence_length', [1,2,4,8,16,32,64,128,256]),
        'nheads': trial.suggest_categorical('nheads', [4,8,16]),
        'process_layers': trial.suggest_categorical('process_layers', [16,32,64,128]),
        'dropout': trial.suggest_categorical('dropout', [0.0, 0.1, 0.2, 0.3]),
        'resample_timeframes': trial.suggest_categorical(
            'resample_timeframes', 
            get_timeframe_combinations()
        )
    }
    return hyperparams

def get_timeframe_combinations():
    import itertools
    timeframes = ['5min', '15min', '30min', '1hr', '2hr', '4hr']
    combinations = []
    for r in range(3, len(timeframes)+1):
        combinations.extend(list(itertools.combinations(timeframes, r)))
    return combinations
