import copy

from symbal import TestFunction, Dataset
from symbal.utils import batch_selection as bs
from symbal.utils import get_score, get_metrics, get_test, get_distances, get_class_scores, keep_old
from symbal.strategy import objective
from pysr import PySRRegressor
import numpy as np
import pandas as pd
# import logging
import random
import re


class SymbalTest:

    def __init__(self, iterations, batch_size, pysr_model, function=None, min_vals=None, max_vals=None, 
                 testfunction=None, batch_config=None, acquisition=None, dataset=None, data_config=None,
                 test_config=None, boundary=False):

        testfunction = dict() if testfunction is None else testfunction
        batch_config = dict() if batch_config is None else batch_config
        data_config = dict() if data_config is None else data_config
        acquisition = dict(uncertainty=0.5, curvature=0.5) if acquisition is None else acquisition

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

        equations, extrap_scores, interp_scores, existing_scores = [], [], [], []
        losses, best_scores, losses_other, scores_other, test_scores = [], [], [], [], []
        holdout_scores = []

        boundary_distances, point_distances, class_scores, mod_class_scores = [], [], [], []

        if 'seed' in testfunction:
            random.seed(testfunction['seed'])

        for i in range(iterations):

            x_train = datobj.initial_set.drop('output', axis=1)
            y_train = datobj.initial_set['output']

            if pysr_model.equation_file is not None:
                if i == 0:
                    pysr_model.equation_file = pysr_model.equation_file.replace('.csv', '') + f'-{i}.csv'
                else:
                    pysr_model.equation_file = re.sub(r'-\d+', f'-{i}', pysr_model.equation_file)

            if ('X_units' in batch_config) and ('y_units' in batch_config):
                pysr_model.fit(x_train, y_train, X_units=batch_config['X_units'], y_units=batch_config['y_units'])
            else:
                pysr_model.fit(x_train, y_train)

            x_cand = datobj.candidates.drop('output', axis=1)

            if self.past_model is not None:

                keep_old_bool = keep_old(self.past_model, pysr_model, datobj.initial_set, x_cand, batch_config)

                if keep_old_bool:
                    pysr_model = self.past_model
                    pysr_model.equation_file = re.sub(r'-\d+', f'-{i}', pysr_model.equation_file)

            if function is not None:
                extrap_scores.append(get_score(datobj.extrapolation_testset, pysr_model))
                interp_scores.append(get_score(datobj.interpolation_testset, pysr_model))
            else:
                holdout_scores.append(get_score(datobj.holdout_set, pysr_model))

            existing_scores.append(get_score(datobj.initial_set, pysr_model))

            if test_config is not None:
                test_scores.append(get_test(test_config, pysr_model))

            equation, loss, score, loss_other, score_other = get_metrics(pysr_model)
            equations.append(equation)
            losses.append(loss)
            best_scores.append(score)
            losses_other.append(loss_other)
            scores_other.append(score_other)

            if boundary:

                boundary_distance, point_distance = get_distances(datobj.initial_set, x_cand)
                class_score, mod_class_score = get_class_scores(datobj.initial_set, pysr_model)

                boundary_distances.append(boundary_distance)
                point_distances.append(point_distance)
                class_scores.append(class_score)
                mod_class_scores.append(mod_class_score)

            if isinstance(pysr_model, PySRRegressor):
                pysr_model.column_list = list(x_cand.columns)

            objective_array = objective(datobj.candidates, datobj.initial_set, pysr_model, acquisition, batch_config)
            x_cand.insert(0, 'objective', objective_array)

            selected_indices, captured_penalties = bs(np.array(x_cand), batch_size=batch_size, **batch_config)
            captured_penalties = captured_penalties.rename(columns={
                column: f'{i+1}-{column}' for column in list(captured_penalties.columns)
            })
            self.captured_penalties = pd.concat([self.captured_penalties, captured_penalties], axis=1)

            self.selected_indices.append(selected_indices)

            initial_addition = datobj.candidates.iloc[selected_indices, :]
            datobj.initial_set = pd.concat([datobj.initial_set, initial_addition], axis=0, ignore_index=True)
            datobj.candidates = datobj.candidates.drop(selected_indices, axis=0).reset_index(drop=True)

            self.past_model = copy.deepcopy(pysr_model)

        scores_dict = {'equation': equations, 'existing': existing_scores, 'loss': losses, 'score': best_scores,
                       'loss_other': losses_other, 'score_other': scores_other}

        if function is not None:
            scores_dict.update({'extrap': extrap_scores, 'interp': interp_scores})
        else:
            scores_dict.update({'holdout': holdout_scores})

        if test_config is not None:
            scores_dict.update({'test_error': test_scores})

        if boundary:
            scores_dict.update({'boundary_distance': boundary_distances, 'point_distance': point_distances,
                                'class_score': class_scores, 'mod_class_score': mod_class_scores})

        self.scores = pd.DataFrame(scores_dict)
