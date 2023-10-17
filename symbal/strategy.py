
from symbal.utils import get_gradient, get_curvature, get_uncertainties
import numpy as np
import scipy as sp
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel


def objective(cand_df, exist_df, pysr_model, acquisition, batch_config):

    x_cand = cand_df.drop('output', axis=1)
    x_exist = exist_df.drop('output', axis=1)

    objective_array = np.zeros((len(x_cand)))

    if 'gradient' in acquisition:
        gradients = _gradient(x_cand, pysr_model, batch_config)
        objective_array += acquisition['gradient'] * gradients

    if 'curvature' in acquisition:
        curvatures = _curvature(x_cand, pysr_model, batch_config)
        objective_array += acquisition['curvature'] * curvatures

    if 'distance' in acquisition:
        distances = _distance(x_cand, x_exist, batch_config)
        objective_array += acquisition['distance'] * distances

    if 'proximity' in acquisition:
        proximities = _proximity(x_cand, x_exist, batch_config)
        objective_array += acquisition['proximity'] * proximities

    if 'density' in acquisition:
        densities = _density(x_cand, x_exist, batch_config)
        objective_array += acquisition['density'] * densities

    if 'sparsity' in acquisition:
        sparsities = _sparsity(x_cand, x_exist, batch_config)
        objective_array += acquisition['sparsity'] * sparsities

    if 'uncertainty' in acquisition:
        uncertainties = _uncertainty(x_cand, pysr_model, batch_config)
        objective_array += acquisition['uncertainty'] * uncertainties

    if 'certainty' in acquisition:
        certainties = _certainty(x_cand, pysr_model, batch_config)
        objective_array += acquisition['certainty'] * certainties

    if 'random' in acquisition:
        random_array = _random(x_cand)
        objective_array += acquisition['random'] * random_array

    if 'rand1' in acquisition:
        random_array = _rand1(x_cand)
        objective_array += acquisition['rand1'] * random_array

    if 'grad1' in acquisition:
        gradients = _grad1(x_cand, pysr_model, batch_config)
        objective_array += acquisition['grad1'] * gradients

    if 'curv1' in acquisition:
        curvatures = _curv1(x_cand, pysr_model, batch_config)
        objective_array += acquisition['curv1'] * curvatures

    if 'gaussian_unc' in acquisition:
        uncertainties = _gaussian_unc(cand_df, exist_df, batch_config)
        objective_array += acquisition['gaussian_unc'] * uncertainties

    if 'know_grad' in acquisition:
        gradients = _know_grad(cand_df, exist_df, batch_config)
        objective_array += acquisition['know_grad'] * gradients

    if 'debug' in batch_config:
        if batch_config['debug']:

            print_string = f'max: {np.max(objective_array)}, min: {np.min(objective_array)}, '
            print_string += f'avg: {np.mean(objective_array)}, std: {np.std(objective_array)}'
            print(print_string)

    return objective_array


def _gradient(x_cand, pysr_model, batch_config):

    if 'difference' in batch_config:
        difference = batch_config['difference']
    else:
        difference = 1e-8

    gradient_array = np.empty((len(x_cand), len(pysr_model.equations_['equation'])))
    for j, _ in enumerate(pysr_model.equations_['equation']):
        gradient_array[:, j] = get_gradient(x_cand, pysr_model, num=j, difference=difference)

    if 'score_reg' in batch_config:
        if batch_config['score_reg']:
            scores = np.array(pysr_model.equations_['score'])
            gradient_array = gradient_array * scores

    gradients = np.sum(np.abs(gradient_array), axis=1)
    gradients = __scale_objective(gradients, batch_config)

    return gradients


def _curvature(x_cand, pysr_model, batch_config):

    if 'difference' in batch_config:
        difference = batch_config['difference']
    else:
        difference = 1e-8

    curvature_array = np.empty((len(x_cand), len(pysr_model.equations_['equation'])))
    for j, _ in enumerate(pysr_model.equations_['equation']):
        curvature_array[:, j] = get_curvature(x_cand, pysr_model, num=j, difference=difference)

    if 'score_reg' in batch_config:
        if batch_config['score_reg']:
            scores = np.array(pysr_model.equations_['score'])
            curvature_array = curvature_array * scores

    curvatures = np.sum(np.abs(curvature_array), axis=1)
    curvatures = __scale_objective(curvatures, batch_config)

    return curvatures


def _distance(x_cand, x_exist, batch_config):

    if 'distance_metric' in batch_config:
        distance_metric = batch_config['distance_metric']
    else:
        distance_metric = 'euclidean'

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

    if 'distance_metric' in batch_config:
        distance_metric = batch_config['distance_metric']
    else:
        distance_metric = 'euclidean'

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
    uncertainties = __scale_objective(uncertainties, batch_config)

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

    if 'difference' in batch_config:
        difference = batch_config['difference']
    else:
        difference = 1e-8

    gradient_array = get_gradient(x_cand, pysr_model, difference=difference)

    gradients = np.abs(gradient_array)
    gradients = __scale_objective(gradients, batch_config)

    return gradients


def _curv1(x_cand, pysr_model, batch_config):

    if 'difference' in batch_config:
        difference = batch_config['difference']
    else:
        difference = 1e-8

    curvature_array = get_curvature(x_cand, pysr_model, difference=difference)

    curvatures = np.abs(curvature_array)
    curvatures = __scale_objective(curvatures, batch_config)

    return curvatures


def _gaussian_unc(cand_df, exist_df, batch_config):

    _, y_cand_std = __gaussian_fit(cand_df, exist_df, batch_config)
    y_cand_std = __scale_objective(y_cand_std, batch_config)

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
    gradients = __scale_objective(gradients, batch_config)

    return gradients


def __scale_objective(objective_array, batch_config):

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

    if 'scaler' in batch_config:
        scaler = batch_config['scaler']
    else:
        scaler = StandardScaler()

    if 'gpr' in batch_config:
        gpr = batch_config['gpr']
    else:
        kernel = Matern(nu=0.501) + WhiteKernel()
        gpr = GaussianProcessRegressor(kernel=kernel)

    x_exist = np.array(exist_df.drop('output', axis=1))
    y_exist = np.array(exist_df['output'])
    x_cand = np.array(cand_df.drop('output', axis=1))

    x_exist_norm = scaler.fit_transform(x_exist)
    x_cand_norm = scaler.transform(x_cand)

    gpr.fit(x_exist_norm, y_exist)
    y_cand_mean, y_cand_std = gpr.predict(x_cand_norm, return_std=True)

    # y_cand_mean = y_cand_mean.reshape(-1, 1)
    # y_cand_std = y_cand_std.reshape(-1, 1)

    return y_cand_mean, y_cand_std
