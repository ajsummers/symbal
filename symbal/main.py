
from symbal import TestFunction
from symbal.utils import batch_selection as bs
from symbal.utils import get_score, get_metrics
import numpy as np
import pandas as pd


class SymbalTest:

    def __init__(self, function, min_vals, max_vals, iterations, batch_size, pysr_model, testfunction=None,
                 batch_config=None):

        testfunction = dict() if testfunction is None else testfunction
        batch_config = dict() if batch_config is None else batch_config

        self.captured_penalties = pd.DataFrame()
        self.selected_indices = []

        tf = TestFunction(function, min_vals, max_vals, **testfunction)
        self.initial_set = tf.initial_set
        self.candidates = tf.candidates

        equations, extrap_scores, interp_scores, existing_scores = [[], [], [], []]
        losses, scores, losses_other, scores_other = [[], [], [], []]

        for i in range(iterations):

            x_train = tf.initial_set.drop('output', axis=1)
            y_train = tf.initial_set['output']

            pysr_model.fit(x_train, y_train)

            extrap_scores.append(get_score(tf.extrapolation_testset, pysr_model))
            interp_scores.append(get_score(tf.interpolation_testset, pysr_model))
            existing_scores.append(get_score(tf.initial_set, pysr_model))

            equation, loss, score, loss_other, score_other = get_metrics(pysr_model)
            equations.append(equation)
            losses.append(loss)
            scores.append(score)
            losses_other.append(loss_other)
            scores_other.append(score_other)

            x_cand = tf.candidates.drop('output', axis=1)
            predictions = np.empty((len(x_cand), len(pysr_model.equations_['equation'])))

            equation_best = pysr_model.predict(x_cand)

            for j in range(pysr_model.equations_['equation']):
                predictions[:, j] = pysr_model.predict(x_cand, j) - equation_best

            scores = np.array(pysr_model.equations_['score'])

            predictions_weight = predictions * scores
            uncertainty = np.sum(np.abs(predictions_weight), axis=1)

            x_cand.insert(0, 'uncertainty', uncertainty)

            selected_indices, captured_penalties = bs(np.array(x_cand), batch_size=batch_size, **batch_config)
            captured_penalties = captured_penalties.rename(columns={
                column: f'{i+1}-{column}' for column in list(captured_penalties.columns)
            })
            self.captured_penalties = pd.concat([self.captured_penalties, captured_penalties], axis=1)
            self.selected_indices.append(selected_indices)

            tf.inital_set = pd.concat([tf.inital_set, tf.candidates.loc[selected_indices, :]], axis=0)
            tf.candidates = tf.candidates.drop(selected_indices, axis=0)

        scores_dict = {
            'equation': equations,
            'extrap': extrap_scores,
            'interp': interp_scores,
            'existing': existing_scores,
            'loss': losses,
            'score': scores,
            'loss_other': losses_other,
            'score_other': scores_other
        }
        self.scores = pd.DataFrame(scores_dict)