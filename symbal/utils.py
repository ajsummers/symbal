import copy

import numpy as np
from symbal.penalties import invquad_penalty
import pandas as pd
import warnings
import itertools
import sympy

from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import GridSearchCV
from typing import Union, List
from pysr import PySRRegressor
from scipy.spatial.distance import cdist


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
        s_x = np.nanstd(penalty_array[:, 1:],
                        axis=0)  # Tune width of penalty by standard deviation of each independent variable
    s_y = np.nanstd(penalty_array[:, 0], axis=0)  # Standard deviation of penalized value

    penalty = penalty_function(a, b, r_x, s_x, s_y)
    penalty_array[:, 0] -= penalty  # subtract penalty

    return max_index, penalty_array


def batch_selection(uncertainty_array, penalty_function=invquad_penalty, a=1, b=1, by_range=False, batch_size=10,
                    **kwargs):

    captured_penalties = pd.DataFrame()
    selected_indices = []
    penalty_array = uncertainty_array

    for i in range(batch_size):

        selected_index, penalty_array = new_penalty(penalty_array, penalty_function, a, b, by_range, batch_size)

        captured_penalties[f'{i}'] = penalty_array[:, 0]
        selected_indices.append(selected_index)

    return selected_indices, captured_penalties


def get_score(input_df, pysr_model):  # TODO - NaN filtering?

    predicted = pysr_model.predict(input_df)
    actual = np.array(input_df['output'])
    score = np.nanmean(np.abs(predicted - actual))

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


def get_gradient(x_cand, pysr_model, num=None, difference=None):

    if difference is None:
        difference = 1e-8

    overall = copy.deepcopy(x_cand)
    diff_dict = {
        column: pd.concat([x_cand.loc[:, column] + difference, x_cand.drop(column, axis=1)], axis=1)
        for column in x_cand
    }
    for column in x_cand:
        if num is not None:
            overall.loc[:, f'fh__{column}'] = pysr_model.predict(diff_dict[column], num)
            overall.loc[:, f'd__{column}'] = (overall.loc[:, f'fh__{column}'] -
                                              pysr_model.predict(x_cand, num)) / difference
        else:
            overall.loc[:, f'fh__{column}'] = pysr_model.predict(diff_dict[column])
            overall.loc[:, f'd__{column}'] = (overall.loc[:, f'fh__{column}'] -
                                              pysr_model.predict(x_cand)) / difference

    grad_string = 'sqrt('
    for column in overall:
        if 'd__' in column:
            grad_string += f'{column}**2 + '
    grad_string = grad_string.rstrip(' + ') + ')'
    overall['grad'] = overall.eval(grad_string)

    return np.array(overall['grad'])


def get_curvature(x_cand, pysr_model, num=None, difference=None):

    if difference is None:
        difference = 1e-8

    overall = copy.deepcopy(x_cand)
    diff_dict = {
        column: pd.concat([x_cand.loc[:, column] + difference, x_cand.drop(column, axis=1)], axis=1)
        for column in x_cand
    }
    diff2_dict = {
        column: pd.concat([x_cand.loc[:, column] + 2 * difference, x_cand.drop(column, axis=1)], axis=1)
        for column in x_cand
    }
    for column in x_cand:
        if num is not None:
            overall.loc[:, f'fh__{column}'] = pysr_model.predict(diff_dict[column], num)
            overall.loc[:, f'f2h__{column}'] = pysr_model.predict(diff2_dict[column], num)
            overall.loc[:, f'd2__{column}'] = (pysr_model.predict(x_cand, num) - 2 * overall.loc[:, f'fh__{column}'] +
                                               overall.loc[:, f'f2h__{column}']) / difference ** 2
        else:
            overall.loc[:, f'fh__{column}'] = pysr_model.predict(diff_dict[column])
            overall.loc[:, f'f2h__{column}'] = pysr_model.predict(diff2_dict[column])
            overall.loc[:, f'd2__{column}'] = (pysr_model.predict(x_cand) - 2 * overall.loc[:, f'fh__{column}'] +
                                               overall.loc[:, f'f2h__{column}']) / difference ** 2

    lapl_string = ''
    for column in overall:
        if 'd2__' in column:
            lapl_string += f'abs({column}) + '
    lapl_string = lapl_string.rstrip(' + ')
    overall['lapl'] = overall.eval(lapl_string)

    return np.array(overall['lapl'])


def get_all_gradients(x_cand, pysr_model, difference=1e-8):

    gradients = np.empty((len(x_cand), len(pysr_model.equations_['equation'])))

    for j, _ in enumerate(pysr_model.equations_['equation']):
        gradients[:, j] = get_gradient(x_cand, pysr_model, num=j, difference=difference)

    return gradients


def get_uncertainties(x_cand, pysr_model):

    uncertainties = np.empty((len(x_cand), len(pysr_model.equations_['equation'])))
    equation_best = pysr_model.predict(x_cand)

    for j, _ in enumerate(pysr_model.equations_['equation']):
        uncertainties[:, j] = pysr_model.predict(x_cand, j) - equation_best

    return uncertainties


class CustomStrPrinter(sympy.printing.str.StrPrinter):
    def _print_Float(self, expr):
        return f'{expr:.3e}'


def get_equation(column_list: List[str], model: Lasso, scaler: Union[MinMaxScaler, StandardScaler],
                 polynomial: PolynomialFeatures):

    if isinstance(scaler, StandardScaler):
        warnings.warn('Equation identification not yet implemented for StandardScaler.')
        return ''

    if polynomial.interaction_only:
        warnings.warn('Equation identification not yet implemented for interaction_only option.')
        return ''

    variable_list = ['1', *column_list]
    variable_list.extend(list(itertools.combinations_with_replacement(column_list, polynomial.degree)))
    variable_list = np.array(['*'.join(var) if isinstance(var, tuple) else var for var in variable_list])

    lasso_coef = model.coef_
    scaler_range = scaler.data_range_
    scaler_min = scaler.data_min_
    intercept = model.intercept_

    variable_list = variable_list[lasso_coef > 0]
    scaler_range = scaler_range[lasso_coef > 0]
    scaler_min = scaler_min[lasso_coef > 0]
    lasso_coef = lasso_coef[lasso_coef > 0]

    unscaled_coef = lasso_coef * scaler_range
    adj_intercept = intercept + np.sum(lasso_coef * scaler_min)

    equation_terms = [f'{coef:.3e}*{feat}' for coef, feat in zip(unscaled_coef, variable_list)]
    equation = ' + '.join(equation_terms)
    equation += f' + {adj_intercept:.3e}'

    parsed_equation = CustomStrPrinter().doprint(sympy.parsing.sympy_parser.parse_expr(equation))

    return parsed_equation


def get_test(test_config, pysr_model):

    if test_config['test_file']:
        test_df = pd.read_csv(test_config['test_file'], index_col=0)
    elif test_config['test_df']:
        test_df = test_config['test_df']
    else:
        raise RuntimeWarning('Must set either test_file or test_df for test_config.')

    test_df['pred_y'] = pysr_model.predict(test_df)

    variances = test_df.groupby(test_config['groupby'])['pred_y'].var().mean()
    error = np.sqrt(np.mean((test_df[test_config['output_name']] - test_df['pred_y']) ** 2))

    total_err = error + variances

    return total_err


def get_fim(cand_set, pysr_model, single=True, num=None, difference=1e-8):

    if not isinstance(pysr_model, PySRRegressor):
        raise RuntimeWarning('Fischer information matrix (FIM) currently only defined for PySRRegressor.')

    cand_set = cand_set[pysr_model.column_list]

    if single:
        cand_set = np.array([cand_set])
    else:
        cand_set = np.array(cand_set)

    if num is None:
        jax = pysr_model.jax()
    else:
        jax = pysr_model.jax(num)

    parameter_set = np.array([jax['parameters'] for i, _ in enumerate(jax['parameters'])])
    diff_vector = parameter_set.diagonal() * (1 + difference)
    np.fill_diagonal(parameter_set, diff_vector)

    fh = np.zeros((len(cand_set), len(jax['parameters'])))
    f = np.zeros((len(cand_set), len(jax['parameters'])))

    for i, _ in enumerate(cand_set):
        for j, _ in enumerate(jax['parameters']):
            fh[i, j] = jax['callable'](np.array([cand_set[i, :]]), parameter_set[j])[0]
            f[i, j] = jax['callable'](np.array([cand_set[i, :]]), jax['parameters'])[0]

    q = (fh - f) / (parameter_set.diagonal() * difference)
    fim = np.outer(q, q.T)

    return fim


def get_distances(exist_df, x_cand):

    act_y = np.array(exist_df['output'])
    boundary_distance = np.mean(np.abs(act_y))

    x_exist = exist_df.drop('output', axis=1)

    exist_array = np.array(x_exist)
    cand_array = np.array(x_cand)

    exist_norm = (exist_array - np.min(cand_array, axis=0)) / np.ptp(cand_array, axis=0)

    point_distances = cdist(exist_norm, exist_norm)
    point_distance = np.mean(point_distances)

    return boundary_distance, point_distance


def get_class_scores(exist_df, pysr_model):

    y_act = np.array(exist_df['output'])
    act_x = exist_df.drop('output', axis=1)

    y_pred = pysr_model.predict(act_x)
    class_scores = np.abs(np.sign(y_act) - np.sign(y_pred)) / 2
    mod_class_scores = class_scores * np.exp(-np.sqrt(np.abs(y_act) / np.ptp(y_act)))

    class_score = np.mean(class_scores)
    mod_class_score = np.mean(mod_class_scores)

    return class_score, mod_class_score


def get_optimal_krr(x_exist_scaled, y_exist, krr_param_grid, grid_rounds, alpha_number):

    krr_model = GridSearchCV(KernelRidge(), param_grid=krr_param_grid)
    krr_model.fit(x_exist_scaled, y_exist)

    for _ in range(grid_rounds):

        best_alpha = krr_model.best_params_['alpha']
        alpha_space = krr_param_grid['alpha'][1] / krr_param_grid['alpha'][0]
        new_alphas = [alpha for alpha in np.geomspace(best_alpha/alpha_space, best_alpha*alpha_space, alpha_number)]
        krr_param_grid['alpha'] = new_alphas

        krr_model = GridSearchCV(KernelRidge(), param_grid=krr_param_grid)
        krr_model.fit(x_exist_scaled, y_exist)

    return krr_model


def get_optimal_gpr(x_exist_scaled, y_exist, gpr_param_grid, grid_rounds, alpha_number):

    gpr_model = GridSearchCV(GaussianProcessRegressor(), param_grid=gpr_param_grid)
    gpr_model.fit(x_exist_scaled, y_exist)

    for _ in range(grid_rounds):

        best_alpha = gpr_model.best_params_['alpha']
        alpha_space = gpr_param_grid['alpha'][1] / gpr_param_grid['alpha'][0]
        new_alphas = [alpha for alpha in np.geomspace(best_alpha / alpha_space, best_alpha * alpha_space, alpha_number)]
        gpr_param_grid['alpha'] = new_alphas

        gpr_model = GridSearchCV(GaussianProcessRegressor(), param_grid=gpr_param_grid)
        gpr_model.fit(x_exist_scaled, y_exist)

    return gpr_model
