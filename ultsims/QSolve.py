import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
import copy


def main(Ho=[[]], mu=[[]],
         Etfield=[], tfield=[], wo=2.354,
         teval=[],
         system='multiphoton multilevel',
         plot_all=False):

    # wo = field_t['wo']
    # Et = resample_field(teval, field_t)
    Et = interpolate_complex(teval, tfield, Etfield)
    t = teval[:]

    wij = calculate_wij(Ho)
    nphoton = calculate_nphoton(wij, wo, mu)
    
    if system == 'multilevel':
        nphoton[nphoton > 1] = 1.0

    if not are_transitions_resolvable(t, wij, wo,
                                      nphoton,True):
        shp = (np.size(t),) + np.shape(Ho)
        return np.zeros(shp, dtype=np.complex)

    if system == 'multilevel':
        VI = VI_multilevel(mu, Et, t, wo, wij)

    elif system == 'multiphoton multilevel':
        VI = VI_multiphoton_multilevel(mu, Et, t,
                                       wo, wij, nphoton)

    dUt = calculate_dUt(VI, t)
    u = propagate_solution(dUt)

    if plot_all:
        plot_u(t, u)
        uf, wavs = ut2uf(u, t, wo)
        plot_u_vs_wavelength(wavs, uf)

    return u


def resample_field(tnew=[], old_field={}):
    told, etold = old_field['t'], old_field['values']
    return interpolate_complex(tnew, told, etold)


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
    #
    # amps = self.field.Ew.amplitudes
    # w = self.field.Ew.w
    # wc = w[np.argmax(amps)]  # the w of the maximum in Ew
    # allowed_trans = np.array(self.mu) != 0
    # self.nphoton = \
    #     np.round(np.abs(self.wij / wc) * allowed_trans.astype(float))


def are_transitions_resolvable(teval=[],
                               wij=[[]],
                               wo=2.354,
                               nphoton=[[]],
                               verbose=True):
    """
    compare the transition periods to the time resolution to see
    what transitions can be resolved. dependencies: need to have
    loaded the field (to determine the time resolution).
    """
    #get the carrier frequency and find the current time resolution
    # Et = self.field.Et
    # wo = Et.wo
    t = teval[:]
    dt = np.abs(t[1] - t[0])
    # get the matrix with the transition frequencies
    # wij = self.wij
    #compare dt with transition periods and determine if they can be
    # resolved:
    dim = np.shape(wij)[0]
    I = np.eye(dim)
    if np.size(nphoton):
        w_rwa = wo*(nphoton + I) - np.sign(wij)*wij
    else:
        w_rwa = wo * I - np.sign(wij) * wij
    #periods = np.zeros(np.shape(self.mu))
#        periods[f_rwa != 0] = np.abs(1/f_rwa[f_rwa != 0])
    periods = 1/(w_rwa/(2*np.pi) )
    # self.transition_periods = periods
    res_trans = np.abs(periods) >= 2*dt
    # self.resolvable_transitions = res_trans.astype(float)

    # # set the irrelevant diagonal terms to true
    # res_trans += I.astype(np.bool)

    # index of all relevant transitions
    active_transitions = nphoton != 0

    all_resolvable = (res_trans[active_transitions] == True).all()

    if not all_resolvable and verbose:
        print 'unresolvable transition detected:'
        print 'minimum period is {} fs and current dt is {} fs'.format(np.min(np.abs(periods)), dt)

    return all_resolvable


def calculate_wij(Ho=[[]]):
    hbar = constants.hbar / (constants.eV * constants.femto)  # 0.658212 eV*fs
    energies = np.array(Ho, dtype=np.float64).diagonal()
    wr, wc = np.meshgrid(energies / hbar, energies / hbar)
    wij = wr - wc  # a matrix with the frequency differences.
    return wij


def VI_multilevel(mu=[[]], Et=[], times=[], wo = 2.354,
                  wij=[[]]):
    nt = np.size(times)
    shp = (nt, 1, 1)

    E = np.array(np.abs(Et), dtype=np.complex).reshape(shp)
    phase = np.angle(Et).reshape(shp)
    t = np.array(times).reshape(shp)

    mut = np.array(nt * [np.array(mu)], dtype=np.complex)
    wijt = np.array(nt * [np.array(wij)])

    # forming VI in the RWA:
    VI = - mut* E * np.exp(1j * (wijt * t - np.sign(wijt) * (wo * t + phase)))

    return VI


def VI_multiphoton_multilevel(mu=[[]], Et=[],
                              times=[], wo = 2.354,
                              wij=[[]],
                              nphoton=[[]]):
    nt = np.size(times)
    shp = (nt, 1, 1)

    E = np.array(np.abs(Et), dtype=np.complex).reshape(shp)
    phase = np.angle(Et).reshape(shp)
    t = np.array(times).reshape(shp)

    mut = np.array(nt * [np.array(mu)], dtype=np.complex)
    wijt = np.array(nt * [np.array(wij)])
    act_trans = np.array(nphoton != 0).astype(float)
    act_trans = np.array(nt * [act_trans])

    # forming VI in the RWA:
    VI = - act_trans * mut * (E**nphoton) * np.exp(1j * (wijt * t - np.sign(wijt) * nphoton * (wo * t + phase)))

    return VI


    # def calculate_VI(self, Et_amp = 0.02,
    #                  Et_phase = 1, t = 2,
    #                  wo = 2.354):
    #     wij = self.wij
    #     E = Et_amp
    #     phase = Et_phase
    #     mu = self.mu
    #     n = self.nphoton
    #     # to remove zero-photon transitions:
    #     active_transitions = n != 0
    #     active_transitions = active_transitions.astype(float)
    #     # forming VI in the RWA:
    #     VI = - mu*(E**n)*np.exp( 1j*( wij*t - np.sign(wij)*n*(wo*t+phase) ) )
    #     VI *= active_transitions
    #     return VI


def calculate_dUt(VI=[[[]]], times=[]):
    nt, nlevs, __ = np.shape(VI)
    dt = times[1] - times[0] # can later change to irregular step
    hbar = constants.hbar / (constants.eV * constants.femto) #eV/fs
    du = -1j*dt/hbar*VI/2
    I = np.array(nt * [np.eye(nlevs)], dtype=np.complex)
    pade_numerator = I + du
    pade_denominator = I - du

    inv_denom = np.linalg.inv(pade_denominator)

    return np.matmul(pade_numerator, inv_denom)


def propagate_solution(dUt=[[[]]]):
    cum_prod = np.eye(np.shape(dUt)[1])
    u = []

    for m in dUt:
        cum_prod = np.dot(cum_prod, m)
        u.append(cum_prod)

    return np.array(u)


def calculate_psi(u=[[[]]], psi0=[]):
    return np.matmul(u, psi0)


def ut2uf(ut=[[[]]], t=[], wo=2.354):
    nlevs = np.shape(ut)[1]
    uf = np.zeros(np.shape(ut), dtype=np.complex)
    dt = t[1] - t[0]

    for i in range(nlevs):
        for j in range(nlevs):
            ufij, wavs = ft2Flambda(ut[:, i, j], dt, wo)
            uf[:, i, j] = ufij

    return uf, wavs


def plot_u(t=[], ut=[[[]]],
           figsize=(15, 10),
           figname='U(t)',
           xlim=(),
           ylim=()):
    nlevels = np.shape(ut)[1]
    plt.figure(figname, figsize=figsize)

    for j in range(nlevels):
        for k in range(nlevels):
            ax = plt.subplot2grid((nlevels, nlevels),
                                  (j, k))
            # plt.subplot(nlevels, nlevels, j*nlevels + k)
            ax.plot(t, np.real(ut[:,j, k]))
            ax.annotate('matrix element\n({}, {})'.format(j, k), xy=(0.01, 0.89),
                        xycoords='axes fraction')
            plt.ylabel('Re[U$_{%d, %d}$(t)]'%(j, k))
            if xlim: plt.xlim(xlim)
            if ylim: plt.ylim(ylim)
    plt.tight_layout()


def plot_u_vs_wavelength(wavs=[], uf=[[[]]],
                         figsize=(15, 10),
                         figname='U(lambda)',
                         xlim=(),
                         ylim=()
                         ):
    nlevels = np.shape(uf)[1]
    plt.figure(figname, figsize=figsize)

    for j in range(nlevels):
        for k in range(nlevels):
            ax = plt.subplot2grid((nlevels, nlevels),
                                  (j, k))
            # plt.subplot(nlevels, nlevels, j*nlevels + k)
            ax.plot(wavs, np.real(uf[:, j, k]))
            ax.annotate('matrix element\n({}, {})'.format(j, k), xy=(0.01, 0.89),
                        xycoords='axes fraction')
            plt.ylabel('Re[U$_{%d, %d}$($\lambda$)]' % (j, k))
            if xlim: plt.xlim(xlim)
            if ylim: plt.ylim(ylim)
    plt.tight_layout()


def plot_populations(t=[], psi=[[[]]],
                     figsize=(10, 5),
                     figname='Populations vs. time',
                     xlim=(),
                     ylim=()):
    pops = np.abs(psi)**2
    plt.figure(figname, figsize=figsize)
    k = 0

    for pop in pops.T:
        plt.plot(t, pop, label=r'|%d$\rangle$'%k)
        if xlim: plt.xlim(xlim)
        if ylim: plt.ylim(ylim)
        k += 1

    plt.xlabel('time (fs)')
    plt.ylabel('Population')
    plt.legend()



##### mechanisms

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
    # plt.plot(new_x, new_y)
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


def Fw2Flambda(fw, w):
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
    xold = twopi * c / w[:]
    yold = copy.deepcopy(fw)

    xnew = np.linspace(np.min(xold), np.max(xold), np.size(xold))
    ynew = interpolate_complex(xnew, xold, yold)

    return ynew, xnew


def ft2Flambda(ft, dt=1, wo=2.3544637527662644):
    fw, w = ft2Fw(ft, dt, wo)
    return Fw2Flambda(fw, w)

