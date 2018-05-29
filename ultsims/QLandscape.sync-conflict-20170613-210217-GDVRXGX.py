import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from scipy import interpolate as scipy_interpolate # because of conflict with my function below...
import copy


def main(Etfield=[], tfield=[], wo=2.354,
         Ho=[[]], mu=[[]],
         u=[[[]]], psi0=[], target_state=[],
         teval=np.linspace(-1000, 1000, 1000),
         wavs=np.linspace(740, 860),
         graph=False):

    field_t = {'values': Etfield,
               't': tfield, 'wo': wo}
    ft = resample_field(field_t, teval)
    # ft = interpolate_complex(teval, tfield, Etfield)

    mu1_plus_hat, mu1_minus_hat, mu2_plus_hat, vec_i, vec_p =\
        get_hessian_requirements(ft, Ho, mu, u, psi0, target_state)

    grad, __ = calculate_gradient(mu1_plus_hat, vec_i, vec_p, ft,
                                  'wavelength', wavs)

    hess, __, __ = calculate_hessian(mu1_plus_hat, mu1_minus_hat,
                                     mu2_plus_hat, vec_i, vec_p,
                                     ft, wavs)
    # print np.shape(hess), np.max(hess)
    if graph:
        plot_gradient(grad, wavs=wavs)
        plot_hessian(hessian_wav=hess, wavs=wavs)


def calculate_gradient(mu1_plus_hat=np.array([[[]]]),
                       vec_i=np.array([]),
                       vec_p=np.array([]),
                       field_t={},
                       output_in='wavelength',
                       wavs=np.linspace(740, 860)):
    """
    mu1_plus_hat and field_t should have the same time.
    grad=-1/hbar*{[<i|p><p|mu1plus_hat|i>-<i|mu1plus_hat|p><p|i>]}
    """
    if np.shape(mu1_plus_hat)[0] != np.size(field_t['t']):
        print 'mu1_plus_hat and field_t must share the same time'
        raise ValueError

    hbar = constants.hbar / (constants.eV * constants.femto)  # 0.658212 eV*fs
    ip = np.dot(np.conj(vec_i), vec_p)

    gradt = 0 + 0j
    gradt += ip * sandwich_product(vec_p, mu1_plus_hat, vec_i)
    gradt -= sandwich_product(vec_i, mu1_plus_hat, vec_p) * np.conj(ip)
    gradt *= - 1 / hbar

    # to output corresponding times and wo:
    times = field_t['t'][:]
    wo = field_t['wo']

    if output_in == 'wavelength':
        # get the field at wavelengths:
        dt = times[1] - times[0];print dt, wo
        E, __ = ft2Flambda(field_t['values'], dt, wo, wavs)
        # E = interpolate_complex(wavs, xold, yold)

        gradwav, __ = ft2Flambda(gradt, dt, wo, wavs)
        # gradwav = interpolate_complex(wavs, xold, yold)
        # gradwav = -np.real(np.conj(E) * gradwav)
        gradwav = -np.imag(np.conj(E) * gradwav)

        # scalling:
        dwav = np.abs(wavs[1] - wavs[0])
        c = constants.c * 1e9 / 1e15  # nm/fs
        gradwav *= c * dwav / wavs ** 2
        # gradwav *= 2 * np.pi * c * dwav / wavs**2

        return gradwav, wavs
    else:
        return gradt, times, wo


# def calculate_gradient(mu1_plus_hat=np.array([[[]]]),
#                        vec_i=np.array([]),
#                        vec_p=np.array([]),
#                        field_t={},
#                        output_in='wavelength',
#                        wavs=np.linspace(740, 860)):
#     """
#     mu1_plus_hat and field_t should have the same time.
#     grad=-1/hbar*{[<i|p><p|mu1plus_hat|i>-<i|mu1plus_hat|p><p|i>]}
#     """
#     if np.shape(mu1_plus_hat)[0] != np.size(field_t['t']):
#         print 'mu1_plus_hat and field_t must share the same time'
#         raise ValueError
#
#     hbar = constants.hbar / (constants.eV * constants.femto)  # 0.658212 eV*fs
#     nlevels = np.size(vec_i)
#     hbar = hbar
#     ip = np.dot(np.conj(vec_i), vec_p)
#
#     term1 = np.dot(np.conj(vec_i),
#                    np.dot(mu1_plus_hat, np.reshape(vec_p, (nlevels, 1))))
#     term1 = term1.ravel()
#     term2 = np.dot(np.conj(vec_p),
#                    np.dot(mu1_plus_hat, np.reshape(vec_i, (nlevels, 1))))
#     term2 = term2.ravel()
#     gradt = - 1 / hbar * (ip * term1 + term2 * np.conj(ip))
#     # gradt = np.real( 1/hbar*( ip*term1 + term2*np.conj(ip) ) )
#
#     # to output corresponding times and wo:
#     times = field_t['t'][:]
#     wo = field_t['wo']
#
#     if output_in == 'wavelength':
#         # get the field at wavelengths:
#         dt = times[1] - times[0]
#         yold, xold = ft2Flambda(field_t['values'], dt, wo)
#         E = interpolate_complex(wavs, xold, yold)
#
#         yold, xold = ft2Flambda(gradt, dt, wo)
#         gradwav = interpolate_complex(wavs, xold, yold)
#         # gradwav = -np.real(np.conj(E) * gradwav)
#         gradwav = -np.imag(np.conj(E) * gradwav)
#
#         # scalling:
#         dwav = np.abs(wavs[1] - wavs[0])
#         c = constants.c * 1e9 / 1e15  # nm/fs
#         gradwav *= c * dwav / wavs ** 2
#         # gradwav *= 2 * np.pi * c * dwav / wavs**2
#
#         return gradwav, wavs
#     else:
#         return gradt, times, wo


def calculate_hessian(mu1_plus_hat=[[[]]],
                      mu1_minus_hat=[[[]]],
                      mu2_plus_hat=[[[]]],
                      vec_i=[],
                      vec_p=[],
                      field_t={},
                      wavelengths=[]):
    """
    for wavelengths l1 and l2:
    Hessian(l1,l2) = -1/2*Re[E(l1)conj(E(l2))H+-(l1,l2)-E(l1)E(l2)H++(l1,l2)]
    If wavs=[] will return hessian vs. freqs.
    """
    nt = np.size(field_t['t'])

    if np.shape(mu1_plus_hat)[0] != nt or np.shape(mu1_minus_hat)[0] != nt or np.shape(mu2_plus_hat)[0] != nt:
        print 'mu1_plus_hat and field_t must share the same time'
        raise ValueError

    times = field_t['t'][:]
    wo = field_t['wo']
    dt = times[1] - times[0]

    # calculate the time domain hpp and hpm:
    hpp_t = calculate_Hplusplus_t(mu1_plus_hat, mu2_plus_hat,
                                  vec_i, vec_p)
    hpm_t = calculate_Hplusminus_t(mu1_plus_hat, mu1_minus_hat,
                                   vec_i, vec_p)

    # # # symmetrize:
    # hpp_t = 0.5 * (hpp_t + hpp_t.T)
    # hpm_t = 0.5 * (hpm_t + hpm_t.T)

    # transform to frequency domain:
    # If wavelengths=[] will return vs. freqs.
    hpp, freqs, wavs =\
        Hplusplus2freq(hpp_t, times, wo, wavelengths)
    hpm, __, __ =\
        Hplusminus2freq(hpm_t, times, wo, wavelengths)

    # get the field at same wavelengths:
    E, __ = ft2Flambda(field_t['values'], dt, wo, wavs)
    # yold, xold = ft2Flambda(field_t['values'], dt, wo)
    # if wavelengths != []:
    #     E = interpolate_complex(wavelengths, xold, yold)
    # else:
    #     E = yold

    #Hessian(l1,l2) = 1/2*Re[E(l1)conj(E(l2))H+-(l1,l2)-E(l1)E(l2)H++(l1,l2)]
    N = np.size(E)
    H = 0 + 0j
    H -= (np.reshape(E, (N,1))*E.conj())*hpm
    H += (np.reshape(E, (N, 1))*E) * hpp

    H = 0.5 * np.real(H)

    # scaling (to compare with montecarlo hessian):
    c = constants.c * 1e9 / 1e15  # in nm/fs
    dwav = wavs[1] - wavs[0]
    # H *= (c * dwav / wavelengths ** 2)**2
    H *= (2 * np.pi * c * dwav / wavs ** 2) ** 2

    # symmetrize:
    H = 0.5 * (H + H.T)

    return H, freqs, wavs


def resample_field(field_t={},
                   teval=np.linspace(-1000, 1000, 1000)):
    ft = copy.deepcopy(field_t)

    if not np.array(ft['t'] == teval).all():
        Et = interpolate_complex(teval, field_t['t'], field_t['values'])
        ft['values'] = Et
        ft['t'] = copy.deepcopy(teval)

    return ft


def calculate_Hplusplus_t(mu1_plus_hat=[[[]]],
                          mu2_plus_hat=[[[]]],
                          vec_i=[],
                          vec_p=[],
                          graph=False):
    """
    for t: rows, t': columns
    H_{++}(t,t') = -1/hbar^2*sum_k (<i|p><p|muplus(t)|k><k|muplus(t')|i> +
                                    +<i|muplus(t')|k><k|muplus(t)|p><p|i>) +
                    +1/hbar^2*(<i|muplus(t)|p><p|muplus(t')|i>+
                                +<i|muplus(t')|p><p|muplus(t)|i>)
                    -i/hbar*(<p|mu2plus|i><i|p>+<p|i><i|mu2plus|p>)*delta(t-t')
    :return:
    """
    hbar = constants.hbar / (constants.eV * constants.femto)  # 0.658212 eV*fs
    nlevels = np.shape(vec_i)[0]
    ntimes = np.shape(mu1_plus_hat)[0]
    ip = np.dot(np.conj(vec_i), vec_p)

    h_plus_plus = 0 + 1j * 0

    for vec_k in np.eye(nlevels):
        # term: -<i|p><p|muplus(t)|k><k|muplus(t')|i>
        row_vec = np.dot(np.conj(vec_p), np.dot(mu1_plus_hat, np.reshape(vec_k, (nlevels, 1))))
        col_vec = np.dot(np.conj(vec_k), np.dot(mu1_plus_hat, np.reshape(vec_i, (nlevels, 1))))
        row_vec = np.reshape(row_vec, (1, ntimes))
        h_plus_plus -= ip * row_vec * col_vec

        # term: -<i|muplus(t')|k><k|muplus(t)|p><p|i>
        row_vec = np.dot(np.conj(vec_k), np.dot(mu1_plus_hat, np.reshape(vec_p, (nlevels, 1))))
        col_vec = np.dot(np.conj(vec_i), np.dot(mu1_plus_hat, np.reshape(vec_k, (nlevels, 1))))
        row_vec = np.reshape(row_vec, (1, ntimes))
        h_plus_plus -= row_vec * col_vec * np.conj(ip)

    # term: <i|muplus(t)|p><p|muplus(t')|i>
    row_vec = np.dot(np.conj(vec_i), np.dot(mu1_plus_hat, np.reshape(vec_p, (nlevels, 1))))
    col_vec = np.dot(np.conj(vec_p), np.dot(mu1_plus_hat, np.reshape(vec_i, (nlevels, 1))))
    row_vec = row_vec.reshape((1, ntimes))
    h_plus_plus += row_vec * col_vec

    # term: <i|muplus(t')|p><p|muplus(t)|i>
    row_vec = np.dot(np.conj(vec_p), np.dot(mu1_plus_hat, np.reshape(vec_i, (nlevels, 1))))
    col_vec = np.dot(np.conj(vec_i), np.dot(mu1_plus_hat, np.reshape(vec_p, (nlevels, 1))))
    row_vec = row_vec.reshape((1, ntimes))
    h_plus_plus += row_vec * col_vec

    h_plus_plus *= 1 / hbar ** 2

    # term -i/hbar*<p|mu2plus|i><i|p>+<p|i><i|mu2plus|p>)*delta(t-t'):
    diag_vec = 0 + 1j * 0

    # subterm: <p|mu2plus|i><i|p>
    d1 = np.dot(np.conj(vec_p), np.dot(mu2_plus_hat, np.reshape(vec_i, (nlevels, 1))))
    d1 *= ip
    # subterm: <p|i><i|mu2plus|p>
    d2 = np.dot(np.conj(vec_i), np.dot(mu2_plus_hat, np.reshape(vec_p, (nlevels, 1))))
    d2 *= np.conj(ip)

    diag_vec += 1j / hbar * (d1 + d2)
    h_plus_plus -= np.diag(diag_vec)

    # imposing the t>t' condition:
    h_plus_plus = np.triu(h_plus_plus)

    if graph:
        times = np.arange(ntimes)
        title = 'hplusplus vs. time index'
        plot_hessian(hessian_t=h_plus_plus, times=times,
                     figname=title)
        plt.title(title)

    return h_plus_plus


def calculate_Hplusminus_t(mu1_plus_hat=[[[]]],
                           mu1_minus_hat=[[[]]],
                           vec_i=[],
                           vec_p=[],
                           graph=False):
    """
    for t: rows, t': columns
    H_{+-}(t,t') = -1/hbar^2*sum_k (<i|p><p|muplus(t)|k><k|muminus(t')|i> +
                                    +<i|muminus(t')|k><k|muplus(t)|p><p|i>) +
                    1/hbar^2*(<i|muplus(t)|p><p|muminus(t')|i>+
                                +<i|muminus(t')|p><p|muplus(t)|i>)
    :return:
    """
    hbar = constants.hbar / (constants.eV * constants.femto)  # 0.658212 eV*fs
    nlevels = np.shape(vec_i)[0]
    ntimes = np.shape(mu1_plus_hat)[0]
    ip = np.dot(np.conj(vec_i), vec_p)

    h_plus_minus = 0 + 1j * 0

    for vec_k in np.eye(nlevels):
        # term: -<i|p><p|muplus(t)|k><k|muminus(t')|i>
        row_vec = np.dot(np.conj(vec_p), np.dot(mu1_plus_hat, vec_k.reshape((nlevels, 1))))
        col_vec = np.dot(vec_k.conj(), np.dot(mu1_minus_hat, np.reshape(vec_i, (nlevels, 1))))
        row_vec = row_vec.reshape((1, ntimes))
        h_plus_minus -= ip * row_vec * col_vec

        # term: -<i|muminus(t')|k><k|muplus(t)|p><p|i>
        row_vec = np.dot(vec_k.conj(), np.dot(mu1_plus_hat, np.reshape(vec_p, (nlevels, 1))))
        col_vec = np.dot(np.conj(vec_i), np.dot(mu1_minus_hat, vec_k.reshape((nlevels, 1))))
        row_vec = row_vec.reshape((1, ntimes))
        h_plus_minus -= row_vec * col_vec * np.conj(ip)

    # term: <i|muplus(t)|p><p|muminus(t')|i>
    row_vec = np.dot(np.conj(vec_i), np.dot(mu1_plus_hat, np.reshape(vec_p, (nlevels, 1))))
    col_vec = np.dot(np.conj(vec_p), np.dot(mu1_minus_hat, np.reshape(vec_i, (nlevels, 1))))
    row_vec = row_vec.reshape((1, ntimes))
    h_plus_minus += row_vec * col_vec

    # term: <i|muminus(t')|p><p|muplus(t)|i>
    row_vec = np.dot(np.conj(vec_p), np.dot(mu1_plus_hat, np.reshape(vec_i, (nlevels, 1))))
    col_vec = np.dot(np.conj(vec_i), np.dot(mu1_minus_hat, np.reshape(vec_p, (nlevels, 1))))
    row_vec = row_vec.reshape((1, ntimes))
    h_plus_minus += row_vec * col_vec

    h_plus_minus *= 1 / hbar ** 2

    # imposing the t>t' condition:
    h_plus_minus = np.triu(h_plus_minus)

    # h_plus_minus = h_plus_minus.T

    if graph:
        times = np.arange(ntimes)
        title = 'hplusminus vs. time index'
        plot_hessian(hessian_t=h_plus_minus, times=times,
                     figname=title)
        plt.title(title)

    return h_plus_minus


def Hplusplus2freq(hpp_t=[[]], times=[], wo=0,
                   wavelengths=[], graph=False):
    """
    If wavelengths=[] will return h_freq vs. freqs.
    :param hpp_t:
    :param times:
    :param wo:
    :param wavelengths:
    :return:
    """

    if np.size(hpp_t) is 0:
        ht, times, wo = calculate_Hplusplus_t()
    else:
        ht = hpp_t[...]

    # h_freq, freqs, wavs = Mt2Mfreq(ht, times, wo)
    h_sim = 0.5 * (ht[...] + ht.T[...])
    h_freq, freqs, wavs = Mt2Mfreq(h_sim, times, wo)

    if wavelengths != []:
        c = constants.c * 1e9 / 1e15  # nm/fs
        # interp_freqs = c / np.copy(wavelengths)
        # h_freq =\
        #     interpolate_complex_M(interp_freqs, interp_freqs,
        #                           freqs, freqs, h_freq,
        #                           kind='linear')
        # freqs = interp_freqs
        # wavs = np.copy(wavelengths)

        h_freq = interpolate_complex_M(wavelengths, wavelengths,
                                              wavs, wavs, h_freq)
        freqs = c / np.copy(wavelengths)
        wavs = np.copy(wavelengths)

    if graph:
        title = 'h_plus_plus_freq vs. wavelengths'
        plot_hessian(hessian_wav=h_freq, wavs=wavs,
                     figname=title)
        plt.title(title)

    return h_freq, freqs, wavs


def Hplusminus2freq(hpm_t=[[]], times=[], wo=0,
                    wavelengths=[], graph=False):
    """
    If wavelengths=[] will return h_freq vs. freqs.
    :param hpm_t:
    :param times:
    :param wo:
    :param wavelengths:
    :return:
    """

    if np.size(hpm_t) is 0:
        ht, times, wo = calculate_Hplusminus_t()
    else:
        ht = hpm_t[...]

    h_flipped = np.flipud(ht)
    # h_freq, freqs, wavs = Mt2Mfreq(h_flipped, times, wo)
    h_sim = 0.5 * (h_flipped[...] + h_flipped.T[...])
    h_freq, freqs, wavs = Mt2Mfreq(h_sim, times, wo)

    if np.size(wavelengths) is not 0:
        c = constants.c * 1e9 / 1e15  # nm/fs
        # interp_freqs = c / np.array(wavelengths)
        # h_freq_interp = \
        #     interpolate_complex_M(interp_freqs, interp_freqs,
        #                           freqs, freqs, h_freq)

        h_freq = interpolate_complex_M(wavelengths, wavelengths,
                                              wavs, wavs, h_freq)
        freqs = constants.c / np.copy(wavelengths)
        wavs = np.copy(wavelengths)

    if graph:
        title = 'h_plus_minus_freq vs. wavelengths'
        plot_hessian(hessian_wav=h_freq, wavs=wavs,
                     figname=title)
        plt.title(title)

    return h_freq, freqs, wavs


def get_hessian_requirements(field_t={}, Ho=[[]], mu=[[]],
                             u=[[[]]], psi0=[], target_state=[],
                             graph=False):
    """
    :return: mu1_plus_hat, mu1_minus_hat, mu2_plus_hat,
    i, p, vecs_k (list of basis vectors).
    """
    assert np.size(field_t['t']) == np.shape(u)[0]

    Der1_muI_plus, Der1_muI_minus = \
        calculate_Der1_VI(field_t, Ho, mu)
    Der2_muI_plus, Der2_muI_minus = \
        calculate_Der2_VI(field_t, Ho, mu)

    mu1_plus_hat = \
        calculate_mu_hat(Der1_muI_plus, u)
    mu1_minus_hat = np.swapaxes(mu1_plus_hat.conj(), 1, 2)
    mu2_plus_hat = \
        calculate_mu_hat(Der2_muI_plus, u)

    vec_i = copy.deepcopy(psi0[:])
    UT = u[-1, :, :]
    UT_dagger = np.conj(UT).T
    f = copy.deepcopy(target_state)
    nlev = np.size(f)
    vec_p = np.dot(UT_dagger, np.reshape(f, (nlev, 1)))
    vec_p = np.ravel(vec_p)

    if graph:
        wo = field_t['wo']
        t = field_t['t']
        dt = t[1] - t[0]

        title = 'Der1_muI_plus vs wavelength'
        print title
        tensor_freq, wavs = tensor_t2wav(Der1_muI_plus, dt, wo)
        plot_tensor(tensor=tensor_freq, x=wavs, figname=title)
        plt.title(title)
        plt.show()
        print
        print

        title = 'Der2_muI_plus vs wavelength'
        print title
        tensor_freq, wavs = tensor_t2wav(Der2_muI_plus, dt, wo)
        plot_tensor(tensor=tensor_freq, x=wavs, figname=title)
        plt.title(title)
        plt.show()
        print
        print

        title = 'mu1_plus_hat vs wavelength'
        print title
        tensor_freq, wavs = tensor_t2wav(mu1_plus_hat, dt, wo)
        plot_tensor(tensor=tensor_freq, x=wavs, figname=title)
        plt.title(title)
        plt.show()
        print
        print

        title = 'mu2_plus_hat vs wavelength'
        print title
        tensor_freq, wavs = tensor_t2wav(mu2_plus_hat, dt, wo)
        plot_tensor(tensor=tensor_freq, x=wavs, figname=title)
        plt.title(title)
        plt.show()
        print
        print

    return mu1_plus_hat, mu1_minus_hat, mu2_plus_hat, vec_i, vec_p


def get_gradient_requirements(field_t={}, Ho=[[]], mu=[[]],
                              u=[[[]]], psi0=[], target_state=[]):
    """
    :return: mu1_plus_hat, i, p.
    """
    assert np.size(field_t['t']) == np.shape(u)[0]

    Der1_muI_plus, Der1_muI_minus = \
        calculate_Der1_VI(field_t, Ho, mu)

    mu1_plus_hat = \
        calculate_mu_hat(Der1_muI_plus, u)

    vec_i = copy.deepcopy(psi0)
    UT = u[-1, :, :]
    UT_dagger = np.conj(UT).T
    vec_p = np.dot(UT_dagger, target_state)

    return mu1_plus_hat, vec_i, vec_p


def calculate_Der1_VI(field_t={}, Ho=[[]], mu=[[]], graph=False):
    """
    calculate the matrices mu'_{I+} and mu'_{I-} in the first derivative
    of VI:
    delta VI(t)/ delta E(t') = mu'_I*delta(t-t')
    with:
    mu'_I = - \partial VI(t)/ \partial E(t)
    (mu'_I)_{jk} = - mu_jk*n_jk*E(t)^(n_jk-1)*exp(i*wjk*t)
    and:
    mu'_I = mu'_{I+}*e^{i*w_o*t} +  mu'_{I-}*e^{-i*w_o*t}
    mu'_{I-} = mu'_{I+}^#
    mu'_{I+} = [wij>0]*mu_jk*n_jk*E#(t)^(n_jk-1)*exp(i*(wjk-n*wo)*t)
    mu'_{I-} = [wij<0]*mu_jk*n_jk*E(t)^(n_jk-1)*exp(i*(wjk+n*wo)*t)
    """
    E = np.array(copy.deepcopy(field_t['values']), dtype=np.complex)
    times = np.array(copy.deepcopy(field_t['t']))
    wo = field_t['wo']

    wij = calculate_wij(Ho)
    nphoton = calculate_nphoton(wij, wo, mu)

    # calculate mu'_I+ first and then from it mu'_I-:
    plus = np.array(wij > 0).astype(float)
    wij *= plus
    n = nphoton
    n *= plus

    n_minus1 = (n - 1) * np.array(n >= 1).astype(float)

    E_to_nminus1 = np.array([e.conj() ** n_minus1 * plus for e in E])
    exp_factor = np.array([np.exp(1j * (wij - n * wo) * t) * plus for t in times])

    #        Der1_muI_plus = self.dot_through_time(mu*n, E_to_nminus1*exp_factor)
    Der1_muI_plus = [mu * n * ft for ft in E_to_nminus1 * exp_factor]
    Der1_muI_plus = np.array(Der1_muI_plus)

    # self.Der1_muI_plus = Der1_muI_plus
    # self.Der1_muI_minus = np.array([m.conj().T for m in Der1_muI_plus])
    Der1_muI_minus = np.array([m.conj().T for m in Der1_muI_plus])

    if graph:
        dt = times[1] - times[0]

        title = 'Der1_muI_plus vs time'
        print title
        plot_tensor(tensor=Der1_muI_plus, x=times, figname=title)
        plt.title(title)
        plt.show()
        print
        print

        title = 'Der1_muI_minus vs time'
        print title
        plot_tensor(tensor=Der1_muI_minus, x=times, figname=title)
        plt.title(title)
        plt.show()
        print
        print

        title = 'Der1_muI_plus vs wavelength'
        print title
        tensor_freq, wavs = tensor_t2wav(Der1_muI_plus, dt, wo)
        plot_tensor(tensor=tensor_freq, x=wavs, figname=title)
        plt.title(title)
        plt.show()
        print
        print

        title = 'Der1_muI_minus vs wavelength'
        print title
        tensor_freq, wavs = tensor_t2wav(Der1_muI_minus, -dt, wo)
        plot_tensor(tensor=tensor_freq, x=wavs, figname=title)
        plt.title(title)
        plt.show()

    return Der1_muI_plus, Der1_muI_minus


def calculate_Der2_VI(field_t={}, Ho=[[]], mu=[[]]):
    """
    calculate the matrices mu''_{I+} and mu''_{I-} in the second
    derivative of VI:
    \delta^2 VI(t)/ \delta E^2(t') = mu''_I*\delta(t-t')
    with:
    mu''_I = - \partial^2 VI(t)/ \partial E^2(t)
    (mu''_I)_{jk} = - mu_jk*n_jk*(n_jk-1)*E(t)^(n_jk-2)*exp(\pm2i*wjk*t)
    and:
    mu''_I = mu''_{I+}*e^{2i*w_o*t} +  mu''_{I-}*e^{-2i*w_o*t}
    mu''_{I-} = mu''_{I+}^#
    for n_jk >= 2:
    mu''_{I+} =\
    [wij>0]*mu_jk*n_jk*(n_jk-1)*E#(t)^(n_jk-2)*exp(i*(wjk-n*wo)*t)
    mu'_{I-} =\
    [wij<0]*mu_jk*n_jk*(n_jk-1)*E(t)^(n_jk-2)*exp(i*(wjk+n*wo)*t)
    """
    E = np.array(copy.deepcopy(field_t['values']), dtype=np.complex)
    times = np.array(copy.deepcopy(field_t['t']))
    wo = field_t['wo']

    wij = calculate_wij(Ho)
    nphoton = calculate_nphoton(wij, wo, mu)

    # calculate mu''_I+ first and then from it mu''_I-:
    plus = np.array(wij > 0).astype(float)
    wij *= plus
    n = nphoton
    n *= plus

    n_minus1 = (n - 1) * np.array(n >= 1).astype(float)
    n_minus2 = (n - 2) * np.array(n >= 2).astype(float)

    E_to_nminus2 = np.array([e.conj() ** n_minus2 * plus for e in E])
    exp_factor = np.array([np.exp(1j * (wij - n * wo) * t) * plus for t in times])

    Der2_muI_plus = [mu * n * n_minus1 * ft for ft in E_to_nminus2 * exp_factor]
    Der2_muI_plus = np.array(Der2_muI_plus)

    # self.Der2_muI_plus = Der2_muI_plus
    # self.Der2_muI_minus = np.array([m.conj().T for m in Der2_muI_plus])
    Der2_muI_minus = \
        np.array([m.conj().T for m in Der2_muI_plus])

    return Der2_muI_plus, Der2_muI_minus


def calculate_nphoton(wij, wo, mu):
    """
    calculate the nphoton matrix (i.e., the matrix with the photon order
    for each transition). Using the w at the field's maximum as center,
    not wo since due to sampling it may lie even outside the spectrum.
    """
    allowed_trans = np.array(mu) != 0
    nphoton = \
        np.round(np.abs(wij / wo) * allowed_trans.astype(float))

    return nphoton


def calculate_wij(Ho=[[]]):
    hbar = constants.hbar / (constants.eV * constants.femto)  # 0.658212 eV*fs
    energies = np.array(Ho, dtype=np.float64).diagonal()
    wr, wc = np.meshgrid(energies / hbar, energies / hbar)
    wij = wr - wc  # a matrix with the frequency differences.
    return wij


def calculate_mu_hat(mu=[[]], u=[[[]]]):
    u_dagger = np.swapaxes(np.conj(u), 1, 2)
    mu_hat = np.matmul(mu, u)
    mu_hat = np.matmul(u_dagger, mu_hat)

    ntimes, nlevels, __ = np.shape(mu)

    result = np.zeros((ntimes, nlevels, nlevels), dtype=complex)
    for j in range(nlevels):
        for k in range(nlevels):
            result[:, j, k] = mu_hat[:, k, j]

    return result


def plot_gradient(gwavs=[], gt=[],
                  wavs=np.linspace(740, 860),
                  times=np.arange(-2000, 2000, 4000),
                  figname = "gradient",
                  figsize = (15,5)):
    plt.figure(figname, figsize=figsize)
    if gwavs != []:
        plt.plot(wavs, gwavs)
        plt.xlabel('$\lambda$(nm)')
        plt.ylabel('gradient($\lambda$)')
        plt.title("gradient($\lambda$)")

    else:
        plt.plot(times, gt)
        plt.xlabel('t (fs)')
        plt.ylabel('gradient(t)')
        plt.title('gradient(t)')


def plot_hessian(hessian_wav=[[]], hessian_t=[[]],
                 wavs=np.linspace(740, 860),
                 times=np.arange(-2000, 2000, 4000),
                 figsize=(5, 3.5),
                 vmax=0, vmin=0,
                 nlevels=100,
                 exponent=1,
                 show_colorbar=True,
                 figname = "hessian plot"):

    if np.size(hessian_wav) > 1e6:
        raise ValueError('hessian_wav too big to plot')

    if np.size(hessian_t) > 1e6:
        raise ValueError('hessian_t too big to plot')

    if np.size(hessian_wav) is not 0:
        bwr_2d_plot(hessian_wav, wavs, wavs,
                    figsize, vmax, vmin, nlevels, exponent, show_colorbar,
                    figname, '$\lambda_1$ (nm)', '$\lambda_2$ (nm)')
    elif np.size(hessian_t) is not 0:
        bwr_2d_plot(hessian_t, times, times,
                    figsize, vmax, vmin, nlevels, exponent, show_colorbar,
                    figname, 't$_1$ (fs)', 't$_2$ (fs)')
    else:
        print('input non-null values for hessian_wav or hessian_t')


##### mechanisms:


def bwr_2d_plot(matrix, x=[], y=[],
              figsize=(5,3.5),
              vmax=0, vmin=0,
              nlevels=100,
              exponent=1,
              show_colorbar=True,
              figname='bwr 2D plot',
              xlabel='xlabel',
              ylabel='ylabel'):
    import matplotlib.colors as mcolors

    plt.figure(figname, figsize=figsize)

    real_data = np.real(matrix)
    data_max = np.abs(np.max(real_data))
    data_min = np.abs(np.min(real_data))

    # calculate the proportion of positive and negative points:
    nminus = int(data_min/(data_max + data_min)*nlevels)
    nplus = int(data_max / (data_max + data_min)*nlevels)

    # calculate the colors to put white in the middle:
    zero_to_one = np.linspace(0, 1, nminus) ** exponent
    m = np.vstack((zero_to_one, zero_to_one, np.ones(nminus)))
    colors = [tuple(row) for row in m.T]

    one_to_zero = np.linspace(1, 0, nplus) ** exponent
    m = np.vstack((np.ones(nplus), one_to_zero, one_to_zero))
    colors += [tuple(row) for row in m.T]

    cmap = mcolors.LinearSegmentedColormap.from_list(name='red_white_blue',
                                                     colors=colors,
                                                     N=len(colors) - 1,
                                                     )

    if not vmax: vmax = np.max(real_data)
    if not vmin: vmin = np.min(real_data)
    ny, nx = np.shape(real_data)
    if x == []: x = np.arange(nx)
    if y == []: y = np.arange(ny)

    plt.pcolormesh(x, y, real_data, cmap=cmap,
                   shading='gouraud', vmax=vmax, vmin=vmin)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if show_colorbar: plt.colorbar()


def plot_tensor(tensor=[[[]]], x=[],
                figsize=(15, 10),
                figname='U(lambda)',
                xlim=(),
                ylim=()
                ):
    """
    plots the real part of a tensor matrix elements vs x.
    :param tensor:
    :param x:
    :param figsize:
    :param figname:
    :param xlim:
    :param ylim:
    :return:
    """
    nx, nlevels, __ = np.shape(tensor)
    plt.figure(figname, figsize=figsize)

    if x == []:
        x = np.arange(nx)

    for j in range(nlevels):
        for k in range(nlevels):
            ax = plt.subplot2grid((nlevels, nlevels),
                                  (j, k))
            # plt.subplot(nlevels, nlevels, j*nlevels + k)
            ax.plot(x, np.real(tensor[:, j, k]))
            ax.annotate('matrix element\n({}, {})'.format(j, k),
                        xy=(0.01, 0.89),
                        xycoords='axes fraction')
            plt.ylabel('real part')
            # plt.ylabel('Re[tensor$_{%d, %d}$($\lambda$)]' % (j, k))
            if xlim: plt.xlim(xlim)
            if ylim: plt.ylim(ylim)
    plt.tight_layout()


def interpolate(new_x, old_x, old_y,
                interpolation="smooth"):
    """
    from two equally sized vectors old_x and old_y, output new vectors
    new_x and new_y interpolated from x and y.
    Interpolations: 'steps' and 'smooth' (default).
    It also sorts old_x and old_y if needed (e.g., np.interp requires
    sorted old_x).
    """
    sorting_indxs = np.argsort(old_x)
    x_sorted = np.array(old_x)[sorting_indxs]
    y_sorted = np.array(old_y)[sorting_indxs]

    # interp assumes by default: f(0), for x<0, and f(N)
    # for x>N.
    if interpolation == "steps":
        indices = np.arange(0, np.size(x_sorted))
        indices = np.interp(new_x, x_sorted, indices)
        indices = np.round(indices)
        indices = indices.astype(int)
        new_y = y_sorted[indices]
    else:
        new_y = np.interp(new_x, x_sorted, y_sorted)

    return new_y


def interpolate_complex(new_x, old_x, old_y,
                        interpolation='smooth'):
    amps = np.abs(old_y)
    phases = np.unwrap(np.angle(old_y))

    new_amps = interpolate(new_x, old_x, amps,
                           interpolation)
    new_phases = interpolate(new_x, old_x, phases,
                             interpolation)

    return new_amps*np.exp(1j*new_phases)


def ft2Fw(ft, dt=1, wo=0):
    """
    get F(w) from f(t) using a fft. This is due to my convention:
    f(t) = \int F(w)*exp(+iwt)dw
    which coincides with the numpy expression for fft (not ifft).

    Inputs:

    ft: a complex 1D array.
    dt: the time step between two points of ft.
    wo: the central (carrier) frequency.

    Outputs:

    Fw: the transform ordered in increasing w.
    w: the angular frequency corresponding to dt and centered at wo.
    """
    twopi = 2 * constants.pi
    Nt = np.size(ft)

    f = np.fft.fftfreq(Nt) / dt
    f = np.fft.fftshift(f)
    w = twopi * f + wo

    Fw = np.fft.fft(np.fft.ifftshift(ft))
    Fw = np.fft.fftshift(Fw)
    Fw *= dt / twopi

    return Fw, w


def Fw2Flambda(fw, w, wavs=np.linspace(740, 860)):
    """
    Calculate a function of wavelengths F(wavs) from a given function of
    angular frequency f(w).

    Inputs:

    fw: a complex 1D array function of angular frequency.
    w: increasing 1D array representing angular frequency.
    wavs: the wavelengths at which the output will be interpolated.

    Output:

    Flambda: complex 1D array, interpolation of fw at the given wavs.
    """
    c = constants.c * 1e9 / 1e15  # in nm/fs
    twopi = 2 * constants.pi
    #    ws = fw.w
    xold = copy.deepcopy(w)
    yold = copy.deepcopy(fw)

    xnew = twopi * c / copy.deepcopy(wavs)
    ynew = interpolate_complex(xnew, xold, yold)

    return ynew, copy.deepcopy(wavs)


# def Fw2Flambda(fw, w):
#     """
#     Calculate a function of wavelengths F(wavs) from a given function of
#     angular frequency f(w).
#
#     Inputs:
#
#     fw: a complex 1D array function of angular frequency.
#     w: increasing 1D array representing angular frequency.
#     wavs: the wavelengths at which the output will be interpolated.
#
#     Output:
#
#     Flambda: complex 1D array, interpolation of fw at the given wavs.
#     """
#     c = constants.c * 1e9 / 1e15  # in nm/fs
#     twopi = 2 * constants.pi
#     #    ws = fw.w
#     xold = twopi * c / w[:]
#     yold = copy.deepcopy(fw)
#
#     xnew = np.linspace(np.min(xold), np.max(xold), np.size(xold))
#     ynew = interpolate_complex(xnew, xold, yold)
#
#     return ynew, xnew


def ft2Flambda(ft, dt=1, wo=2.3544637527662644,
               wavs=np.linspace(740, 860)):
    fw, w = ft2Fw(ft, dt, wo)
    flambda, __ = Fw2Flambda(fw, w, wavs)
    return flambda, np.copy(wavs)


def Mt2Mfreq(Mt, times, wo):
    """
    transform Mt to frequency domain
    :param Mt: complex 2D array in time domain
    :param times: 1D array of regularly spaced real numbers
    :param wo: double, carrier angular frequency
    :return:
    Mf: complex 2D array in frequency domain
    freqs: 1D array with corresponding frequencies in 1/fs units
    wavs: 1D array of not regularly spaced wavelengths (in nm) corresponding to freqs.
    """
    # Mf = np.fft.ifft2(np.fft.fftshift(np.conj(Mt)))
    # Mf = np.fft.fftshift(Mf)
    Mf = np.fft.fft2(np.fft.ifftshift(Mt))
    Mf = np.fft.ifftshift(Mf)

    dt = np.abs(times[1] - times[0])
    n = Mf.shape[0]
    freqs = np.fft.fftfreq(n, dt)
    freqs = np.fft.ifftshift(freqs)
    c = constants.c * 1e9 / 1e15  # nm/fs
    wavs = c / (wo / (2 * np.pi) + freqs)

    Mf *= ( dt/(2*np.pi) )**2

    return Mf, freqs, wavs


def tensor_t2wav(tensor=[[[]]], dt=1, wo=2.3544637527662644,
                wavs=np.linspace(740, 860)):
    sh = np.shape(tensor)
    wav_sh = (np.size(wavs), sh[1], sh[2])
    tensor_freq = np.zeros(wav_sh, dtype=complex)

    for j in range(sh[1]):
        for k in range(sh[2]):
            tensor_freq[:, j, k], __ = ft2Flambda(tensor[:, j, k],
                                                  dt, wo)

    return tensor_freq, np.copy(wavs)


def interpolate_M(new_x, new_y, old_x, old_y, old_M,
                  kind='cubic'):
    finterp = scipy_interpolate.interp2d(old_x, old_y, old_M, kind=kind)
    new_M =  finterp(new_x, new_y)
    return new_M


def interpolate_complex_M(new_x, new_y, old_x, old_y, old_M,
                          kind='cubic'):
    new_re = interpolate_M(new_x, new_y, old_x, old_y, np.real(old_M), kind=kind)
    new_imag = interpolate_M(new_x, new_y, old_x, old_y, np.imag(old_M), kind=kind)

    return new_re + 1j*new_imag


def sandwich_product(a=[], M=[[]], b=[]):
    nlevs = np.size(a)
    bcol = np.reshape(b, (nlevs, 1))

    prod = np.matmul(M, bcol)
    prod = np.matmul(np.conj(a), prod)

    return prod.ravel()