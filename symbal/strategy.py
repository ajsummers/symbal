
from symbal.utils import get_gradient, get_curvature, get_uncertainties
import numpy as np
from scipy.spatial.distance import cdist


def objective(x_cand, x_exist, pysr_model, acquisition, batch_config):

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
    gradients = (gradients - np.min(gradients)) / np.ptp(gradients)

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
    curvatures = (curvatures - np.min(curvatures)) / np.ptp(curvatures)

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
    uncertainties = (uncertainties - np.min(uncertainties)) / np.ptp(uncertainties)

    return uncertainties


def _certainty(x_cand, pysr_model, batch_config):

    uncertainties = _uncertainty(x_cand, pysr_model, batch_config)

    return -uncertainties


def _random(x_cand):

    random_array = np.random.uniform(size=(len(x_cand),))

    return random_array
