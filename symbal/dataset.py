
import random
import pandas as pd


class Dataset:

    def __init__(self, dataset: pd.DataFrame, **kwargs):

        initial_size = kwargs['initial_size'] if 'initial_size' in kwargs else 20
        holdout_size = kwargs['holdout_size'] if 'holdout_size' in kwargs else 100
        seed = kwargs['seed'] if 'seed' in kwargs else None
        output_name = kwargs['output_name'] if 'output_name' in kwargs else None

        if seed is not None:
            random.seed(seed)

        if output_name is not None:
            dataset = dataset.rename(columns={output_name: 'output'})
        else:
            dataset = dataset.rename(columns={dataset.columns[0]: 'output'})

        initial_indices = random.sample(range(len(dataset)), k=initial_size)
        self.initial_set = dataset.iloc[initial_indices, :].reset_index(drop=True)
        dataset = dataset.drop(initial_indices, axis=0).reset_index(drop=True)

        holdout_indices = random.sample(range(len(dataset)), k=holdout_size)
        self.holdout_set = dataset.iloc[holdout_indices, :].reset_index(drop=True)
        dataset = dataset.drop(holdout_indices, axis=0).reset_index(drop=True)

        self.candidates = dataset
