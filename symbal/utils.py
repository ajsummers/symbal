
import numpy as np
from symbal.penalties import invquad_penalty
import pandas as pd
import random


def new_penalty(penalty_array, penalty_function, a, b, by_range, batch_size):
    """
    Selects maximum from given values and penalizes area around selection.

    Assumes first column in penalty_array is penalized value, other columns are
    independent variables.

    Returns: independent variables for selected point & new penalty_array w/ penalized values
    """

    max_index = np.nanargmax(penalty_array[:, 0])  # index for largest value
    penalty_array[max_index, 0] = np.nan
    max_pos = penalty_array[max_index, 1:]  # independent variable values for this index

    r_x = np.abs(penalty_array[:, 1:] - max_pos)  # Distance to selected point for each variable
    if by_range:
        s_x = np.ptp(penalty_array[:, 1:], axis=0) / batch_size  # Tune width of penalty by range / batch_size
    else:
        s_x = np.std(penalty_array[:, 1:],
                     axis=0)  # Tune width of penalty by standard deviation of each independent variable
    s_y = np.nanstd(penalty_array[:, 0], axis=0)  # Standard deviation of penalized value

    penalty = penalty_function(a, b, r_x, s_x, s_y)
    penalty_array[:, 0] -= penalty  # subtract penalty

    return max_index, penalty_array


def batch_selection(uncertainty_array, penalty_function=invquad_penalty, a=1, b=1, by_range=False, batch_size=10):

    captured_penalties = pd.DataFrame()
    selected_indices = []
    penalty_array = uncertainty_array

    for i in range(batch_size):

        selected_index, penalty_array = new_penalty(penalty_array, penalty_function, a, b, by_range, batch_size)

        captured_penalties[f'{i}'] = penalty_array[:, 0]
        selected_indices.append(selected_index)

    return selected_indices, captured_penalties


def get_score(input_df, pysr_model):

    predicted = pysr_model.predict(input_df)
    actual = np.array(input_df['output'])
    score = np.mean(np.abs(predicted - actual))

    return score


def get_metrics(pysr_model):

    best_index = np.argmax(pysr_model.equations_['score'])
    equation = pysr_model.equations_.loc[best_index, 'equation']
    loss = pysr_model.equations_.loc[best_index, 'loss']
    score = pysr_model.equations_.loc[best_index, 'score']

    other_equations = pysr_model.equations_.drop(best_index, axis=0)
    loss_other = np.mean(other_equations['loss'])
    score_other = np.mean(other_equations['score'])

    return equation, loss, score, loss_other, score_other


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
