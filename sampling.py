import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from scipy.interpolate import UnivariateSpline
from scipy import interpolate
import os
import h5py
from scipy.interpolate import interp1d
from tqdm import tqdm

from ga import ga


def main(npoints=1000,
         signal_function=np.sum,
         Xo=np.zeros((64, 2)),  # shape of X rows calculated from this
         sigma_min=0.01,
         sigma_max=1.0,
         nsigmas=40,
         nsamples=20,
         hist_nbins=5,
         sigma_nbins=100,
         y_nbins=100,
         fitness_exponent=2,
         population_size=500,
         parameters_per_individual=100, # sigmas space
         parameter_bounds=(0.01, 1), # sigmas space
         mutation_rate=0.02,
         crossover_rate=0.8,
         freq_stats=200,
         max_gens=1000,
         sample_y=True,
         Xseeds_only=True,
         save_in='.temp_sampling.h5'
         ):
    sampled_sets = uniformSigmasSampling(
        signal_function=signal_function,
        Xo=Xo,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        nsigmas=nsigmas,
        nsamples=nsamples,
        graph=True,
        save_in=save_in)

    sigmas, ycenters, hist2d = histogramFromSampledSets(
        sampled_sets=sampled_sets,
        nbins=hist_nbins, graph=True,
        save_in=save_in)

    new_sigmas, new_ycenters, new_2dhist = interpolateHistogram(
        sigmas=sigmas,
        ycenters=ycenters,
        histogram2d=hist2d,
        sigma_nbins=sigma_nbins,
        y_nbins=y_nbins,
        graph=True,
        save_in=save_in)

    weights = sigmaWeightsForUniformYSampling(
        hist2d=new_2dhist,
        fitness_exponent=fitness_exponent,
        population_size=population_size,
        parameters_per_individual=parameters_per_individual,
        parameter_bounds=parameter_bounds,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        freq_stats=freq_stats,
        max_gens=max_gens,
        graph=True,
        save_in=save_in)

    if sample_y:
        X, ysampled = weightedSigmasSampling(
            npoints=npoints,
            sigmas=new_sigmas,
            weights=weights,
            signal_function=signal_function,
            Xo=Xo,
            graph=True,
            Xseeds_only=Xseeds_only,
            save_in=save_in)
    else:
        X, sampling_sigmas = weighted_sigmas_X(
            npoints=npoints,
            sigmas=new_sigmas,
            weights=weights,
            Xo=Xo,
            Xseeds_only=Xseeds_only,
            save_in=save_in)

    print 'results saved to "{}"'.format(save_in)


def initialize(save_in='.temp_sampling.h5'):
    if os.path.exists(save_in):
        if save_in != '.temp_sampling.h5':
            raise ValueError("file '{}' already exists".format(save_in))
        else:
            os.remove('.temp_sampling.h5')
    else:
        folder, name = os.path.split(save_in)
        if not os.path.exists(folder): os.makedirs(folder)

    with h5py.File(save_in, 'w') as f:
        pass


def uniformSigmasSampling(signal_function=np.sum,
                          Xo=np.zeros((64, 2)),
                          sigma_min=0.01,
                          sigma_max=1.0,
                          nsigmas=20,
                          nsamples=20,
                          graph=True,
                          save_in='.temp_sampling.h5'):
    sampled_sets = []
    sigmas = np.linspace(sigma_min, sigma_max, nsigmas)
    xshape = (nsamples,) + np.shape(Xo)
    # nx = np.size(Xo)
    ymeans = []
    ystds = []
    all_ys = []

    for sigma in sigmas:
        X = np.random.normal(0, sigma, xshape)
        # X = np.random.normal(0, sigma, (nsamples, nx))
        X += np.array(Xo).copy()
        ys = []
        for x in X:
            current_signal = signal_function(x)
            ys.append(current_signal)
            all_ys.append(current_signal)
        subset = {'sigma': sigma,
                  'measurements': ys,
                  'X': X}
        sampled_sets.append(subset)

        ymeans.append(np.mean(ys))
        ystds.append(np.std(ys))


    if graph:
        title = 'signals histogram (uniform sigma sampling)'
        plt.figure(title)
        __ = plt.hist(all_ys)
        plt.title(title)
        plt.xlabel('signal')
        plt.ylabel('ocurrences')
        plt.show()

        title = 'means and stds (uniform sigma sampling)'
        plt.figure(title)
        fig, ax = plt.subplots(nrows=1)
        ax.errorbar(sigmas, ymeans, yerr=ystds, fmt='-o')
        plt.xlabel('sigma')
        plt.ylabel('average +/- std')
        plt.title(title)
        plt.show()

    if save_in:
        df = pd.DataFrame(data=sampled_sets)
        df.to_hdf(save_in, 'initial_sampling',
                  mode='a')

    return sampled_sets


def histogramFromSampledSets(sampled_sets={'sigma': [], 'measurements':[]},
                             nbins=5,
                             graph=True,
                             save_in='.temp_sampling.h5'):
    # finding ymax and ymin:
    all_ys = np.array([])
    for subset in sampled_sets:
        all_ys = np.append(all_ys, subset['measurements'])
    ymax, ymin = all_ys.max(), all_ys.min()

    bin_edges = np.linspace(ymin, ymax, nbins + 1)
    ycenters = (bin_edges[1:] + bin_edges[:-1]) / 2.
    sigmas = []
    hists = []

    for subset in sampled_sets:
        sigma = subset['sigma']
        data = subset['measurements']
        hist, __ = np.histogram(data, bin_edges)
        sigmas.append(sigma)
        hists.append(hist)

    sigmas = np.array(sigmas)
    hists = np.array(hists).T

    if graph:
        title = '2d histogram from sampled sets'
        plt.figure(title)
        plt.pcolormesh(sigmas, ycenters, hists)
        plt.xlabel('sigma')
        plt.ylabel('measurement value')
        plt.title(title)
        plt.colorbar()
        plt.show()

    if save_in:
        s = pd.Series(data=sigmas)
        s.to_hdf(save_in,
                 key='histogram2d_from_initial_sampling/sigmas',
                 mode='a')

        s = pd.Series(data=ycenters)
        s.to_hdf(save_in,
                 key='histogram2d_from_initial_sampling/ycenters',
                 mode='a')

        df = pd.DataFrame(data=hists)
        df.to_hdf(save_in,
                 key='histogram2d_from_initial_sampling/histogram2d',
                 mode='a')

    return sigmas, ycenters, hists


def interpolateHistogram(sigmas, ycenters, histogram2d,
                         sigma_nbins=100, y_nbins=100,
                         graph=True,
                         save_in='.temp_sampling.h5'):
    f = interpolate.interp2d(sigmas, ycenters, histogram2d, kind='cubic')
    ymin, ymax = np.min(ycenters), np.max(ycenters)
    ycenters_new = np.linspace(ymin, ymax, y_nbins)
    sigma_min, sigma_max = np.min(sigmas), np.max(sigmas)
    sigmas_new = np.linspace(sigma_min, sigma_max, sigma_nbins)
    hists_new = f(sigmas_new, ycenters_new)

    if graph:
        title = 'interpolated 2d histogram'
        plt.figure(title)
        plt.pcolormesh(sigmas_new, ycenters_new, hists_new)
        plt.xlabel('sigma')
        plt.ylabel('measurement value')
        plt.title(title)
        plt.colorbar()
        plt.show()

    if save_in:
        s = pd.Series(data=sigmas_new)
        s.to_hdf(save_in,
                 key='interpolated_histogram2d/sigmas',
                 mode='a')

        s = pd.Series(data=ycenters_new)
        s.to_hdf(save_in,
                 key='interpolated_histogram2d/ycenters',
                 mode='a')

        df = pd.DataFrame(data=hists_new)
        df.to_hdf(save_in,
                  key='interpolated_histogram2d/histogram2d',
                  mode='a')

    return sigmas_new, ycenters_new, hists_new


def sigmaWeightsForUniformYSampling(hist2d=[[]],
                                    fitness_exponent=2,
                                    population_size=80,
                                    parameters_per_individual=100,
                                    parameter_bounds=(0.01, 1),
                                    mutation_rate=0.02,
                                    crossover_rate=0.8,
                                    freq_stats=200,
                                    max_gens=1000,
                                    graph=True,
                                    save_in='.temp_sampling.h5',
                                    ):
    # from ga import ga
    # from scipy.interpolate import interp1d

    hist_nrows, hist_ncols = np.shape(hist2d)
    mat = np.array(hist2d)
    mat[mat < 0] = 0
    mat /= np.sum(mat)
    b_target = np.mean(parameter_bounds) * np.ones(hist_ncols)
    b_target = np.dot(mat, b_target)
    b_target = np.mean(b_target)

    def ga_fitness(X):
        b = np.dot(mat, weightsFromX(X))
        # return np.mean(np.abs(b - target_ny))**2
        # return 1e5 * np.mean(np.abs(b - b_target))
        return np.mean(np.abs((b - b_target)/b_target) ** fitness_exponent)
        # return np.sum(np.abs(np.diff(b / b_target))**2)

    def weightsFromX(X):
        if np.size(X) != hist_ncols:
            # make a hist2d_ncolumns cubic interpolated phase from X:
            n = np.size(X)
            x = np.linspace(0, hist_ncols-1, n)
            y = np.copy(X)
            f = interp1d(x, y, kind='cubic')
            xnew = np.arange(hist_ncols)
            weights = f(xnew)
        else:
            weights = np.copy(X)

        # weights should be a probability:
        weights[weights < 0] = 0
        # weights /= np.sum(weights)

        return weights


    ga_engine = ga.main(fitness_function=ga_fitness,
                        population_size=population_size,
                        parameters_per_individual=parameters_per_individual,
                        parameter_bounds=parameter_bounds,
                        mutation_rate=mutation_rate,
                        crossover_rate=crossover_rate,
                        freq_stats=freq_stats,
                        max_gens=max_gens,
                        callback_functions=[],
                        optimization_type='minimize',
                        temp_fname='.__fitness_history__.csv',
                        stop_fname='.stop')

    # weights = ga_engine.bestIndividual()
    Xbest = ga_engine.bestIndividual()
    Xbest = weightsFromX(Xbest)
    weights = Xbest / np.sum(Xbest)

    if graph:
        title = 'estimated optimum weights'
        plt.figure(title)
        plt.plot(weights)
        plt.xlabel('weight index')
        plt.ylabel('weight')
        plt.title(title)
        plt.show()

        title = 'estimated sampled signals histogram'
        plt.figure(title)
        xhist = np.arange(hist_nrows)
        est_ny = np.dot(hist2d, Xbest)
        plt.bar(xhist, est_ny)
        plt.xlabel('signal bin')
        plt.ylabel('occurences')
        plt.title(title)
        plt.show()

    if save_in:
        s = pd.Series(data=weights)
        s.to_hdf(save_in,
                 key='weights',
                 mode='a')

    return weights


def weightedSigmasSampling(npoints=1000,
                           sigmas=[],
                           weights=[],
                           signal_function=np.sum,
                           Xo=np.zeros((64, 2)),
                           graph=True,
                           Xseeds_only=True,
                           save_in='.temp_sampling.h5',
                           ):
    new_ys = []

    if Xseeds_only:
        X, new_sigmas = generate_Xseeds_and_sigmas_from_weights(
            npoints, sigmas, weights, Xo, save_in)

        for xseed, sigma in tqdm(zip(X, new_sigmas)):
            x = random_x_from_seed(xseed, Xo, sigma)
            new_ys.append(signal_function(x))

    else:
        X = generate_X_from_weights(npoints, sigmas, weights, Xo,
                                    save_in)

        for x in tqdm(X):
            new_ys.append(signal_function(x))

    if graph:
        title = 'signals histogram (weighted sigma sampling)'
        plt.figure(title)
        __ = plt.hist(new_ys)
        plt.xlabel('signal')
        plt.ylabel('ocurrences')
        plt.title(title)
        plt.show()

    if save_in:# todo: save with pd.to_hdf(.. fomat='Table') to enable pytables queries
        with h5py.File(save_in, 'a') as f:
            key = 'weighted_sigmas_sampling/y'
            if key in f: del f[key]
            dset = f.create_dataset(key, data=np.array(new_ys))

    return np.array(X), np.array(new_ys)


def weighted_sigmas_X(npoints=1000,
                      sigmas=[],
                      weights=[],
                      Xo=np.zeros((64, 2)),
                      Xseeds_only=True,
                      save_in='.temp_sampling.h5'):
    if Xseeds_only:
        X, sampling_sigmas = generate_Xseeds_and_sigmas_from_weights(
            npoints, sigmas, weights, Xo, save_in)

        if save_in:
            with h5py.File(save_in, 'a') as f:
                key = 'weighted_sigmas_sampling/sampling_sigmas'
                if key in f: del f[key]
                dset = f.create_dataset(
                    key, data=np.array(sampling_sigmas))
    else:
        X = generate_X_from_weights(npoints, sigmas, weights, Xo,
                                    save_in)
        sampling_sigmas = []

    return X, sampling_sigmas


def generate_X_from_weights(npoints=1000,
                            sigmas=[],
                            weights=[],
                            Xo=np.zeros((64, 2)),
                            save_in='',
                            ):
    xshape = np.shape(Xo)
    # npars = np.size(Xo)
    new_sigmas = np.random.choice(sigmas, npoints, p=weights)
    Xsampled = []

    for sigma in new_sigmas:
        X = np.random.normal(0, sigma, xshape)
        X += np.copy(Xo)
        Xsampled.append(X)

    if save_in:
        with h5py.File(save_in, 'a') as f:
            key = 'weighted_sigmas_sampling/X'
            if key in f: del f[key]
            dset = f.create_dataset(key, data=np.array(Xsampled))

    return np.array(Xsampled)


def generate_Xseeds_and_sigmas_from_weights(npoints=1000,
                                            sigmas=[],
                                            weights=[],
                                            Xo=np.zeros((64, 2)),
                                            save_in='',
                                            ):
    xshape = np.shape(Xo)
    # npars = np.size(Xo)
    new_sigmas = np.random.choice(sigmas, npoints, p=weights)
    Xseeds = []

    for __ in new_sigmas:
        np.random.seed(None)
        seed = np.random.choice(2 ** 32)
        Xseeds.append(seed)

    if save_in: # todo: save with pd.to_hdf(.. fomat='Table') to enable pytables queries
        with h5py.File(save_in, 'a') as f:
            key = 'weighted_sigmas_sampling/Xseeds'
            if key in f: del f[key]
            dset = f.create_dataset(key, data=np.array(Xseeds))

            key = 'weighted_sigmas_sampling/sigmas'
            if key in f: del f[key]
            dset = f.create_dataset(key, data=np.array(new_sigmas))

    return np.array(Xseeds), new_sigmas


def random_x_from_seed(seed=0, xo=np.zeros((64, 2)), sigma=0.1):
    np.random.seed(seed)
    xshape = np.shape(xo)
    return xo + np.random.normal(0, sigma, xshape)

