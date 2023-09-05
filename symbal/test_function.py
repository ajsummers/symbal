
import numpy as np
from scipy.stats import qmc
import pandas as pd
import itertools as it
import random


class RandomEngine(qmc.QMCEngine):

    def __init__(self, d, seed=None):
        super().__init__(d=d, seed=seed)

    def _random(self, n=1, *, workers=1):
        return self.rng.random((n, self.d))

    def reset(self):
        super().__init__(d=self.d, seed=self.rng_seed)
        return self

    def fast_forward(self, n):
        self.random(n)
        return self


class TestFunction:

    def __init__(self, function, min_val, max_val, fract_extr=0.1, num_inter=30, num_extr=30,
                 method_inter='MC', method_extr='MC', method_cand='FF', cand_grid_size=30, initial_size=30,
                 seed=None, noise_level=0.1):

        self.candidates = None
        self.initial_set = None
        self.extrapolation_testset = None
        self.interpolation_testset = None
        self.min_vals = None
        self.max_vals = None
        self.function = function

        if isinstance(min_val, list):
            self.min_vals = np.array(min_val)
        elif isinstance(min_val, np.ndarray):
            self.min_vals = min_val
        else:
            raise TypeError('min_val should be of type list or np.ndarray.')

        if isinstance(max_val, list):
            self.max_vals = np.array(max_val)
        elif isinstance(max_val, np.ndarray):
            self.max_vals = max_val
        else:
            raise TypeError('max_val should be of type list or np.ndarray.')

        assert self.min_vals.shape == self.max_vals.shape, 'min_vals and max_vals are different sizes'

        self.init_extrap_set(function, fract_extr, num_extr, method_extr, seed, noise_level)
        self.init_interp_set(function, fract_extr, num_inter, method_inter, seed, noise_level)
        self.init_candidate_set(function, fract_extr, method_cand, cand_grid_size, initial_size, noise_level)

    def init_extrap_set(self, function, fract_extr, num_extr, method_extr, seed, noise_level):

        if method_extr == 'MC':
            pass  # TODO - add different extrapolation sampling methods

        n_extrs = 0
        self.extrapolation_testset = np.empty((num_extr, self.min_vals.shape[0]), dtype=float)
        engine = RandomEngine(d=self.min_vals.shape[0], seed=seed)

        lower_bound = self.min_vals + (fract_extr / 2) * (self.max_vals - self.min_vals)
        upper_bound = self.max_vals - (fract_extr / 2) * (self.max_vals - self.min_vals)

        while n_extrs < num_extr:

            sample = engine.random(1).flatten()
            sample = (self.max_vals - self.min_vals) * sample + self.min_vals

            bool_extr = np.any((sample < lower_bound) | (sample > upper_bound))
            if bool_extr:
                self.extrapolation_testset[n_extrs, :] = sample
                n_extrs += 1

        variables = [f'x{i+1}' for i in range(self.min_vals.shape[0])]
        extr_df = pd.DataFrame(self.extrapolation_testset, columns=variables)
        extr_df['output'] = extr_df.eval(function)
        extr_df['output'] = extr_df['output'] + np.random.normal(scale=noise_level*np.std(extr_df['output']),
                                                                 size=(num_extr,))
        self.extrapolation_testset = extr_df

    def init_interp_set(self, function, fract_extr, num_inter, method_inter, seed, noise_level):

        if method_inter == 'MC':
            pass  # TODO - add different interpolation sampling methods

        n_inter = 0
        self.interpolation_testset = np.empty((num_inter, self.min_vals.shape[0]), dtype=float)
        engine = RandomEngine(d=self.min_vals.shape[0], seed=seed)

        lower_bound = self.min_vals + (fract_extr / 2) * (self.max_vals - self.min_vals)
        upper_bound = self.max_vals - (fract_extr / 2) * (self.max_vals - self.min_vals)

        while n_inter < num_inter:

            sample = engine.random(1).flatten()
            sample = (self.max_vals - self.min_vals) * sample + self.min_vals

            bool_extr = np.any((sample < lower_bound) | (sample > upper_bound))
            if not bool_extr:
                self.interpolation_testset[n_inter, :] = sample
                n_inter += 1

        variables = [f'x{i+1}' for i in range(self.min_vals.shape[0])]
        inter_df = pd.DataFrame(self.interpolation_testset, columns=variables)
        inter_df['output'] = inter_df.eval(function)
        inter_df['output'] = inter_df['output'] + np.random.normal(scale=noise_level*np.std(inter_df['output']),
                                                                   size=(num_inter,))
        self.interpolation_testset = inter_df

    def init_candidate_set(self, function, fract_extr, method_cand, cand_grid_size, initial_size, noise_level):

        if method_cand == 'FF':
            pass  # TODO - add different candidate sampling methods

        lower_bound = self.min_vals + (fract_extr / 2) * (self.max_vals - self.min_vals)
        upper_bound = self.max_vals - (fract_extr / 2) * (self.max_vals - self.min_vals)

        comb_list = []

        for lower, upper in zip(lower_bound, upper_bound):
            comb_list.append(np.linspace(lower, upper, cand_grid_size))

        candidate_array = np.array(list(it.product(*comb_list)))

        ind_initial = random.sample(range(len(candidate_array)), k=initial_size)
        initial_array = candidate_array[ind_initial, :]
        candidate_array = np.delete(candidate_array, ind_initial, axis=0)

        variables = [f'x{i + 1}' for i in range(self.min_vals.shape[0])]
        self.candidates = pd.DataFrame(candidate_array, columns=variables)
        self.candidates['output'] = self.candidates.eval(function)
        self.candidates['output'] += np.random.normal(scale=noise_level*np.std(self.candidates['output']),
                                                      size=(len(self.candidates['output']),))

        self.initial_set = pd.DataFrame(initial_array, columns=variables)
        self.initial_set['output'] = self.initial_set.eval(function)
        self.initial_set['output'] += np.random.normal(scale=noise_level*np.std(self.initial_set['output']),
                                                       size=(len(self.initial_set['output']),))
