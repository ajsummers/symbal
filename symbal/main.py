import copy

from symbal import TestFunction, Dataset
from symbal.utils import batch_selection as bs
from symbal.utils import get_score, get_metrics, get_all_gradients, get_predictions
import numpy as np
import pandas as pd
# import logging
import random
import re


class SymbalTest:

    def __init__(self, iterations, batch_size, pysr_model, function=None, min_vals=None, max_vals=None, 
                 testfunction=None, batch_config=None, acquisition='active', dataset=None, data_config=None):

        testfunction = dict() if testfunction is None else testfunction
        batch_config = dict() if batch_config is None else batch_config
        data_config = dict() if data_config is None else data_config

        self.captured_penalties = pd.DataFrame()
        self.selected_indices = []
        self.past_model = None

        if function is not None:
            datobj = TestFunction(function, min_vals, max_vals, **testfunction)
        else:
            datobj = Dataset(dataset, **data_config)

        self.initial_set = datobj.initial_set
        self.candidates = datobj.candidates
        self.datobj = datobj

        equations, extrap_scores, interp_scores, existing_scores = [[], [], [], []]
        losses, best_scores, losses_other, scores_other = [[], [], [], []]
        holdout_scores = []

        for i in range(iterations):

            x_train = datobj.initial_set.drop('output', axis=1)
            y_train = datobj.initial_set['output']

            if pysr_model.equation_file is not None:
                if i == 0:
                    pysr_model.equation_file = pysr_model.equation_file.replace('.csv', '') + f'-{i}.csv'
                else:
                    pysr_model.equation_file = re.sub(r'-\d+', f'-{i}', pysr_model.equation_file)

            pysr_model.fit(x_train, y_train)

            if self.past_model is not None:

                past_pred = self.past_model.predict(x_train)
                curr_pred = pysr_model.predict(x_train)
                true_y = np.array(y_train)

                past_mae = np.nanmean(np.abs(past_pred - true_y))
                curr_mae = np.nanmean(np.abs(curr_pred - true_y))

                if curr_mae > past_mae:
                    pysr_model = self.past_model
                    pysr_model.equation_file = re.sub(r'-\d+', f'-{i}', pysr_model.equation_file)

            if function is not None:
                extrap_scores.append(get_score(datobj.extrapolation_testset, pysr_model))
                interp_scores.append(get_score(datobj.interpolation_testset, pysr_model))
            else:
                holdout_scores.append(get_score(datobj.holdout_set, pysr_model))

            existing_scores.append(get_score(datobj.initial_set, pysr_model))

            equation, loss, score, loss_other, score_other = get_metrics(pysr_model)
            equations.append(equation)
            losses.append(loss)
            best_scores.append(score)
            losses_other.append(loss_other)
            scores_other.append(score_other)

            x_cand = datobj.candidates.drop('output', axis=1)

            if 'uncertainty' in acquisition:
                objective_array = get_predictions(x_cand, pysr_model)
            elif 'gradient' in acquisition:
                if 'difference' in batch_config:
                    objective_array = get_all_gradients(x_cand, pysr_model, batch_config['difference'])
                else:
                    objective_array = get_all_gradients(x_cand, pysr_model)
            elif 'mix' in acquisition:
                if 'difference' in batch_config:
                    objective_array1 = get_all_gradients(x_cand, pysr_model, batch_config['difference'])
                else:
                    objective_array1 = get_all_gradients(x_cand, pysr_model)
                objective_array2 = get_predictions(x_cand, pysr_model)

                range1 = np.ptp(objective_array1, axis=0)
                range2 = np.ptp(objective_array2, axis=0)
                range1[range1 == 0] = 1
                range2[range2 == 0] = 1
                objective_array1 = objective_array1 / range1
                objective_array2 = objective_array2 / range2
                objective_array = 0.5 * objective_array1 + 0.5 * objective_array2

            else:
                objective_array = get_predictions(x_cand, pysr_model)  # TODO - create other else case

            scores = np.array(pysr_model.equations_['score'])
            if 'score_reg' in batch_config:
                if batch_config['score_reg']:
                    objective_array = objective_array * scores

            objective = np.sum(np.abs(objective_array), axis=1)
            x_cand.insert(0, 'objective', objective)

            if ('active' in acquisition) or ('AL' in acquisition):

                selected_indices, captured_penalties = bs(np.array(x_cand), batch_size=batch_size, **batch_config)
                captured_penalties = captured_penalties.rename(columns={
                    column: f'{i+1}-{column}' for column in list(captured_penalties.columns)
                })
                self.captured_penalties = pd.concat([self.captured_penalties, captured_penalties], axis=1)

            elif 'random' in acquisition:

                if 'seed' in testfunction:
                    random.seed(testfunction['seed'])

                selected_indices = random.sample(range(np.array(x_cand).shape[0]), k=batch_size)

            else:

                if 'seed' in testfunction:
                    random.seed(testfunction['seed'])

                selected_indices = random.sample(range(np.array(x_cand).shape[0]), k=batch_size)

            self.selected_indices.append(selected_indices)

            initial_addition = datobj.candidates.iloc[selected_indices, :]
            datobj.initial_set = pd.concat([datobj.initial_set, initial_addition], axis=0, ignore_index=True)
            datobj.candidates = datobj.candidates.drop(selected_indices, axis=0).reset_index(drop=True)

            self.past_model = copy.deepcopy(pysr_model)

        if function is not None:
            scores_dict = {
                'equation': equations,
                'extrap': extrap_scores,
                'interp': interp_scores,
                'existing': existing_scores,
                'loss': losses,
                'score': best_scores,
                'loss_other': losses_other,
                'score_other': scores_other
            }
        else:
            scores_dict = {
                'equation': equations,
                'holdout': holdout_scores,
                'existing': existing_scores,
                'loss': losses,
                'score': best_scores,
                'loss_other': losses_other,
                'score_other': scores_other
            }
        self.scores = pd.DataFrame(scores_dict)
