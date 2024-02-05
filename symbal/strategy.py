import copy

from symbal.utils import get_gradient, get_curvature, get_uncertainties
from symbal.utils import get_fim, get_optimal_krr, get_optimal_gpr
import numpy as np
import scipy as sp
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RationalQuadratic
from scipy.interpolate import Rbf
from operator import itemgetter


def objective(cand_df, exist_df, pysr_model, acquisition, batch_config):

    x_cand = cand_df.drop('output', axis=1)
    x_exist = exist_df.drop('output', axis=1)

    function_dict = {
        'gradient':     {'func': _gradient,     'params': ['x_cand', 'pysr_model',  'batch_config'], 'scale': True},
        'curvature':    {'func': _curvature,    'params': ['x_cand', 'pysr_model',  'batch_config'], 'scale': True},
        'grad1':        {'func': _grad1,        'params': ['x_cand', 'pysr_model',  'batch_config'], 'scale': True},
        'curv1':        {'func': _curv1,        'params': {'x_cand', 'pysr_model',  'batch_config'}, 'scale': True},
        'uncertainty':  {'func': _uncertainty,  'params': ['x_cand', 'pysr_model',  'batch_config'], 'scale': True},
        'certainty':    {'func': _certainty,    'params': ['x_cand', 'pysr_model',  'batch_config'], 'scale': True},
        'Aopt':         {'func': _aopt,         'params': ['x_cand', 'pysr_model',  'batch_config'], 'scale': True},
        'Dopt':         {'func': _dopt,         'params': ['x_cand', 'pysr_model',  'batch_config'], 'scale': True},
        'Eopt':         {'func': _eopt,         'params': ['x_cand', 'pysr_model',  'batch_config'], 'scale': True},
        'boundary':     {'func': _boundary,     'params': ['x_cand', 'pysr_model',  'batch_config'], 'scale': True},
        'distance':     {'func': _distance,     'params': ['x_cand', 'x_exist',     'batch_config'], 'scale': False},
        'proximity':    {'func': _proximity,    'params': ['x_cand', 'x_exist',     'batch_config'], 'scale': False},
        'density':      {'func': _density,      'params': ['x_cand', 'x_exist',     'batch_config'], 'scale': False},
        'sparsity':     {'func': _sparsity,     'params': ['x_cand', 'x_exist',     'batch_config'], 'scale': False},
        'gaussian_unc': {'func': _gaussian_unc, 'params': ['cand_df', 'exist_df',   'batch_config'], 'scale': True},
        'know_grad':    {'func': _know_grad,    'params': ['cand_df', 'exist_df',   'batch_config'], 'scale': True},
        'leaveoneout':  {'func': _leaveoneout,  'params': ['cand_df', 'exist_df',   'batch_config'], 'scale': True},
        'LOOA':         {'func': _looa,         'params': ['cand_df', 'exist_df',   'batch_config'], 'scale': True},
        'GUGS':         {'func': _gugs,         'params': ['cand_df', 'exist_df',   'batch_config'], 'scale': True},
        'random':       {'func': _random,       'params': ['x_cand'],                                'scale': False},
        'rand1':        {'func': _rand1,        'params': ['x_cand'],                                'scale': False},
    }
    param_dict = {'cand_df': cand_df, 'exist_df': exist_df, 'x_cand': x_cand, 'x_exist': x_exist,
                  'pysr_model': pysr_model, 'batch_config': batch_config}

    objective_array = np.zeros((len(x_cand)))

    for method in acquisition:

        if method not in function_dict:
            raise KeyError(f'{method} does not exist as an acquisition strategy. Available methods:'
                           f'{function_dict.keys()}')

        values = function_dict[method]['func'](*itemgetter(*function_dict[method]['params'])(param_dict))
        values = _scale_objective(values, batch_config) if function_dict[method]['scale'] else values
        objective_array += acquisition[method] * values

    if 'debug' in batch_config:
        if batch_config['debug']:

            print_string = f'max: {np.max(objective_array)}, min: {np.min(objective_array)}, '
            print_string += f'avg: {np.mean(objective_array)}, std: {np.std(objective_array)}'
            print(print_string)

    return objective_array


def _gradient(x_cand, pysr_model, batch_config):

    difference = batch_config['difference'] if 'difference' in batch_config else 1e-8

    gradient_array = np.empty((len(x_cand), len(pysr_model.equations_['equation'])))
    for j, _ in enumerate(pysr_model.equations_['equation']):
        gradient_array[:, j] = get_gradient(x_cand, pysr_model, num=j, difference=difference)

    if 'score_reg' in batch_config:
        if batch_config['score_reg']:
            scores = np.array(pysr_model.equations_['score'])
            gradient_array = gradient_array * scores

    gradients = np.sum(np.abs(gradient_array), axis=1)
    return gradients


def _curvature(x_cand, pysr_model, batch_config):

    difference = batch_config['difference'] if 'difference' in batch_config else 1e-8

    curvature_array = np.empty((len(x_cand), len(pysr_model.equations_['equation'])))
    for j, _ in enumerate(pysr_model.equations_['equation']):
        curvature_array[:, j] = get_curvature(x_cand, pysr_model, num=j, difference=difference)

    if 'score_reg' in batch_config:
        if batch_config['score_reg']:
            scores = np.array(pysr_model.equations_['score'])
            curvature_array = curvature_array * scores

    curvatures = np.sum(np.abs(curvature_array), axis=1)
    return curvatures


def _distance(x_cand, x_exist, batch_config):

    distance_metric = batch_config['distance_metric'] if 'distance_metric' in batch_config else 'euclidean'

    cand_array = np.array(x_cand)
    exist_array = np.array(x_exist)
    cand_norm = (cand_array - np.min(cand_array, axis=0)) / np.ptp(cand_array, axis=0)
    exist_norm = (exist_array - np.min(cand_array, axis=0)) / np.ptp(cand_array, axis=0)

    dist_array = cdist(cand_norm, exist_norm, metric=distance_metric)
    dist_vector = np.min(dist_array, axis=1)
    return dist_vector


def _proximity(x_cand, x_exist, batch_config):
    dist_vector = _distance(x_cand, x_exist, batch_config)
    return -dist_vector


def _density(x_cand, x_exist, batch_config):

    distance_metric = batch_config['distance_metric'] if 'distance_metric' in batch_config else 'euclidean'

    cand_array = np.array(x_cand)
    exist_array = np.array(x_exist)
    cand_norm = (cand_array - np.min(cand_array, axis=0)) / np.ptp(cand_array, axis=0)
    exist_norm = (exist_array - np.min(cand_array, axis=0)) / np.ptp(cand_array, axis=0)

    dist_array = cdist(cand_norm, exist_norm, metric=distance_metric)
    dens_vector = np.mean(dist_array, axis=1)
    return dens_vector


def _sparsity(x_cand, x_exist, batch_config):
    dens_vector = _density(x_cand, x_exist, batch_config)
    return -dens_vector


def _uncertainty(x_cand, pysr_model, batch_config):

    uncertainty_array = get_uncertainties(x_cand, pysr_model)

    if 'score_reg' in batch_config:
        if batch_config['score_reg']:
            scores = np.array(pysr_model.equations_['score'])
            uncertainty_array = uncertainty_array * scores

    uncertainties = np.sum(np.abs(uncertainty_array), axis=1)
    return uncertainties


def _certainty(x_cand, pysr_model, batch_config):
    uncertainties = _uncertainty(x_cand, pysr_model, batch_config)
    return -uncertainties


def _random(x_cand):
    random_array = np.random.uniform(size=(len(x_cand),))
    return random_array


def _rand1(x_cand):

    random_array = np.random.normal(size=(len(x_cand),))
    random_array = (random_array - np.min(random_array)) / np.ptp(random_array)
    return random_array


def _grad1(x_cand, pysr_model, batch_config):

    difference = batch_config['difference'] if 'difference' in batch_config else 1e-8
    gradient_array = get_gradient(x_cand, pysr_model, difference=difference)
    gradients = np.abs(gradient_array)
    return gradients


def _curv1(x_cand, pysr_model, batch_config):

    difference = batch_config['difference'] if 'difference' in batch_config else 1e-8
    curvature_array = get_curvature(x_cand, pysr_model, difference=difference)
    curvatures = np.abs(curvature_array)
    return curvatures


def _gaussian_unc(cand_df, exist_df, batch_config):
    _, y_cand_std = __gaussian_fit(cand_df, exist_df, batch_config)
    return y_cand_std


def _know_grad(cand_df, exist_df, batch_config):

    y_cand_mean, y_cand_std = __gaussian_fit(cand_df, exist_df, batch_config)

    z = np.zeros(shape=y_cand_mean.shape)

    y_exist_max = np.max(exist_df['output'])
    z[y_cand_std != 0.] = (y_cand_mean[y_cand_std != 0.] - y_exist_max) / y_cand_std[y_cand_std != 0.]

    cdf = sp.stats.norm.cdf(z)
    pdf = sp.stats.norm.pdf(z)

    gradients = np.zeros(shape=y_cand_mean.shape)
    gradients[z != 0.] = y_cand_std[z != 0.] * cdf[z != 0.] + pdf[z != 0.]

    return gradients


def _aopt(x_cand, pysr_model, batch_config):

    difference = batch_config['difference'] if 'difference' in batch_config else 1e-8
    traces = np.zeros((len(x_cand),))

    if 'subset' in batch_config:
        x_cand = x_cand.sample(batch_config['subset'])

    for i, cand in enumerate(list(x_cand.index)):
        fim = get_fim(x_cand.loc[cand, :], pysr_model, single=True, difference=difference)
        traces[cand] = np.trace(fim)

    return traces


def _dopt(x_cand, pysr_model, batch_config):

    difference = batch_config['difference'] if 'difference' in batch_config else 1e-8
    dets = np.zeros((len(x_cand),))

    if 'subset' in batch_config:
        x_cand = x_cand.sample(batch_config['subset'])

    for i, cand in enumerate(list(x_cand.index)):
        fim = get_fim(x_cand.loc[cand, :], pysr_model, single=True, difference=difference)
        dets[cand] = np.linalg.det(fim)

    return dets


def _eopt(x_cand, pysr_model, batch_config):

    difference = batch_config['difference'] if 'difference' in batch_config else 1e-8
    mineigs = np.zeros((len(x_cand),))

    if 'subset' in batch_config:
        x_cand = x_cand.sample(batch_config['subset'])

    for i, cand in enumerate(list(x_cand.index)):
        fim = get_fim(x_cand.loc[cand, :], pysr_model, single=True, difference=difference)
        mineigs[cand] = np.min(np.linalg.eigvals(fim))

    return mineigs


def _boundary(x_cand, pysr_model, batch_config):

    power = batch_config['power'] if 'power' in batch_config else 1

    y_pred = pysr_model.predict(x_cand)

    if 'scale_pred' in batch_config:
        if batch_config['scale_pred']:
            y_pred = y_pred / np.ptp(y_pred)
    else:
        y_pred = y_pred / np.ptp(y_pred)

    values = np.exp(-np.abs(y_pred) ** power)
    return values


def _leaveoneout(cand_df, exist_df, batch_config):

    cand_dfc = copy.deepcopy(cand_df)
    exist_dfc = copy.deepcopy(exist_df)

    alpha_number = batch_config['alpha_number'] if 'alpha_number' in batch_config else 10

    if 'krr_param_grid' in batch_config:
        krr_param_grid = batch_config['krr_param_grid']
    else:
        krr_param_grid = {
            'alpha': [alpha for alpha in np.geomspace(1e-4, 1e4, alpha_number)],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [2, 3],
        }

    Scaler = batch_config['scaler'] if 'scaler' in batch_config else StandardScaler
    grid_rounds = batch_config['grid_rounds'] if 'grid_rounds' in batch_config else 2
    interp_func = batch_config['interp_func'] if 'interp_func' in batch_config else 'multiquadric'

    scaler = Scaler()
    x_exist_scaled = scaler.fit_transform(exist_dfc.drop('output', axis=1))
    y_exist = exist_dfc['output']

    krr_model = get_optimal_krr(x_exist_scaled, y_exist, krr_param_grid, grid_rounds, alpha_number)
    initial_score = krr_model.best_score_

    exist_dfc['score_diff'] = 0

    for i, _ in enumerate(exist_dfc.iterrows()):

        exist_minus = exist_dfc.drop(i)

        scaler = Scaler()
        x_minus_scaled = scaler.fit_transform(exist_minus.drop(['output', 'score_diff'], axis=1))
        y_minus = exist_minus['output']

        krr_model = get_optimal_krr(x_minus_scaled, y_minus, krr_param_grid, grid_rounds, alpha_number)
        minus_score = krr_model.best_score_
        score_diff = initial_score - minus_score
        exist_dfc.loc[i, 'score_diff'] = score_diff

    exist_dfc = exist_dfc.drop('output', axis=1)
    cand_dfc = cand_dfc.drop('output', axis=1)

    rbf = Rbf(*[np.array(exist_dfc[column]) for column in list(exist_dfc.columns)], function=interp_func)
    scorediffs = rbf(*[np.array(cand_dfc[column]) for column in list(cand_dfc.columns)])

    return scorediffs


def _looa(cand_df, exist_df, batch_config):
    scorediffs = _leaveoneout(cand_df, exist_df, batch_config)
    scorediffs = np.abs(scorediffs)
    return scorediffs


def _gugs(cand_df, exist_df, batch_config):  # Gaussian Uncertainty with Grid Search

    alpha_number = batch_config['alpha_number'] if 'alpha_number' in batch_config else 10

    if 'gpr_param_grid' in batch_config:
        gpr_param_grid = batch_config['gpr_param_grid']
    else:
        gpr_param_grid = {
            'kernel': [RationalQuadratic(alpha=0.1, length_scale=0.1)],
            'alpha': np.geomspace(1e-4, 1e4, alpha_number),
            'n_restarts_optimizer': [0, 5, 10]
        }

    Scaler = batch_config['scaler'] if 'scaler' in batch_config else StandardScaler
    grid_rounds = batch_config['grid_rounds'] if 'grid_rounds' in batch_config else 2

    scaler = Scaler()
    x_exist_scaled = scaler.fit_transform(exist_df.drop('output', axis=1))
    y_exist = exist_df['output']

    gpr_model = get_optimal_gpr(x_exist_scaled, y_exist, gpr_param_grid, grid_rounds, alpha_number)

    x_cand_scaled = scaler.transform(cand_df.drop('output', axis=1))
    _,  y_cand_std = gpr_model.best_estimator_.predict(x_cand_scaled, return_std=True)

    return y_cand_std


def _scale_objective(objective_array, batch_config):

    if 'standard' in batch_config:
        if batch_config['standard']:
            objective_array = (objective_array - np.mean(objective_array)) / np.std(objective_array)
            objective_array = (objective_array - np.min(objective_array)) / np.ptp(objective_array)
        else:
            objective_array = (objective_array - np.min(objective_array)) / np.ptp(objective_array)
    else:
        objective_array = (objective_array - np.min(objective_array)) / np.ptp(objective_array)

    return objective_array


def __gaussian_fit(cand_df, exist_df, batch_config):

    scaler = batch_config['scaler'] if 'scaler' in batch_config else StandardScaler()
    gpr = batch_config['gpr'] if 'gpr' in batch_config else GaussianProcessRegressor(kernel=Matern(nu=0.5))

    x_exist = np.array(exist_df.drop('output', axis=1))
    y_exist = np.array(exist_df['output'])
    x_cand = np.array(cand_df.drop('output', axis=1))

    x_exist_norm = scaler.fit_transform(x_exist)
    x_cand_norm = scaler.transform(x_cand)

    gpr.fit(x_exist_norm, y_exist)
    y_cand_mean, y_cand_std = gpr.predict(x_cand_norm, return_std=True)

    return y_cand_mean, y_cand_std
