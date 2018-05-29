import numpy as np
from scipy import constants
from scipy.interpolate import interp1d, UnivariateSpline
import matplotlib.pyplot as plt
import pandas as pd
import copy
import sys
# import dtcwt


def main(output_type='Et',
         fluence=1e-3, dt=3, dlambda=0.1,
         spec_offset=0,
         spec_center_wavelength=800,
         fwhm=40,
         spec_lambda_min=740,
         spec_lambda_max=860,
         spec_dlambda=0.1,
         spec_fname="",
         spec_data=np.array([[]]),
         slm_phases=np.array([]),
         slm_amplitudes=np.array([]),
         slm_central_wavelength=800,
         pix2nm=0.2,
         npixels=640,
         slm_fname='',
         graph=False
         ):
    """
    :param output_type: string: 'spectrum', 'Elambda',
    'Ew', 'Et', 'slm_mask'.
    :param fluence:
    :param dt:
    :param dlambda:
    :param spec_offset:
    :param spec_center_wavelength:
    :param fwhm:
    :param spec_lambda_min:
    :param spec_lambda_max:
    :param spec_dlambda:
    :param spec_fname:
    :param spec_data:
    :param slm_phases:
    :param slm_amplitudes:
    :param slm_central_wavelength:
    :param pix2nm:
    :param npixels:
    :param slm_fname:
    :param plot_all:
    :return:
    """

    spec = get_spectrum(spec_center_wavelength,
         fwhm,
         spec_lambda_min,
         spec_lambda_max,
         spec_dlambda,
         spec_fname,
         spec_data)

    if output_type == 'spectrum':
        if graph: spec.plot()
        return spec

    if slm_fname:
        slm_mask =\
            slm_mask_from_file(slm_fname,
                               slm_central_wavelength,
                               pix2nm,
                               npixels,
                               )
    else:
        slm_mask =\
            construct_slm_mask(slm_phases,
                               slm_amplitudes,
                               slm_central_wavelength,
                               pix2nm,
                               npixels,
                               )

    if output_type == 'slm_mask':
        if graph: slm_mask.plot()
        return slm_mask

    Elambda = Elambda_from_spectrum(spec, fluence,
                                    spec_offset,
                                    dt, dlambda)
    Elambda = shape_Elambda(Elambda, slm_mask)

    if output_type == 'Elambda':
        if graph: plot_field(Elambda)
        return Elambda

    Ew = Ew_from_Elambda(Elambda)

    if output_type == 'Ew':
        if graph: plot_field(Ew)
        return Ew

    Et = Et_from_Ew(Ew)

    if output_type == 'Et':
        if graph: plot_field(Et)
        return Et

    print "'{}' is not a valid output type".format(output_type)
    print "valid types: 'spectrum', 'Elambda', " \
          "'Ew', 'Et', 'slm_mask'"
    return None


# class Efield(object):#todo: adapt to changes in functions; to_Et, etc.
#     def __init__(self, fname='', is_slm=False,
#                  *args, **kwargs):
#         super(Efield, self).__init__()
#         if fname is not '':
#             self.load(fname, is_slm)
#
#     def hasattr(self, attr_name=""):
#         return hasattr(self, attr_name)
#
#     def copy(self):
#         return copy.deepcopy(self)
#
#     def copy_attributes(self, Eother=object):
#         # self = Efield()
#         for k, v in Eother.__dict__.items():
#             setattr(self, k, copy.deepcopy(v))
#
#     def field_type(self):
#         if self.hasattr('phases'):
#             is_complex = True
#         else:
#             is_complex = False
#         # is_complex = 'complex' in str(type(y[0]))
#
#         if self.hasattr('wavelengths') and not is_complex:
#             return 'spectrum'
#
#         elif self.hasattr('pix2nm'):
#             return 'slm_mask'
#
#         elif self.hasattr('wavelengths') and is_complex and not self.hasattr('pix2nm'):
#             return 'Elambda'
#
#         elif self.hasattr('w'):
#             return 'Ew'
#
#         elif self.hasattr('t'):
#             return 'Et'
#
#         else:
#             print 'field type not recognized'
#             return None
#
#     def fluence(self):
#         t, w, wavelengths = None, None, None
#         ftype = self.field_type()
#         if ftype is 'Et': t = self.t
#         elif ftype is 'Ew': w = self.w
#         elif ftype is 'Elambda': wavelengths = self.wavelengths
#         else: print 'invalid field for fluence calculation'
#         return calculate_fluence(self.amplitudes, t=t, w=w,
#                                  wavelengths=wavelengths)
#
#     # def fluence(self):
#     #     """
#     #     fluence in J/cm**2 (not in the standard units here)
#     #     renormalized_field = renormalize_field(field, fluence = 0.0008,
#     #     domain = 'time'):
#     #     Renormalize the field amplitude (field has 3 columns: x,
#     #     phase, and amplitude) such that it corresponds to the
#     #     given fluence.
#     #     three domains: 'time', 'frequency, or 'wavelength'
#     #     which yield ouput fields in units of V/Angstrom,
#     #     V*fs/Angstrom, and V*fs/Angstrom, respectively.
#     #     The default fluence = 0.0008 J/cm^2 corresponds to an
#     #     uncollimated beam with energy = 100 uJ and radius = 2 mm.
#     #     the normalization factor is inferred from
#     #     \sum_k(|Et_k|^2) = Z_0*fluence/(2*dt) or
#     #     \sum_k(|Ew_k|^2) = Z_0*fluence/(2*df), by finding the factor
#     #     kappa that multiplied to the field yields the real field.
#     #     """
#     #     Z0 = 377  # Ohms, the vaccum impedance
#     #     #        Z0 *= 1.6e-4 # now in our units V/(e*fs)
#     #     c = constants.c
#     #     #        c = 2997.9245799999999 #speed of light in Angstrom/fs
#     #     Ex = np.copy(self.amplitudes)#np.abs(np.array(E['values']))
#     #     summation = np.sum(Ex ** 2)
#     #     field_type = self.field_type()
#     #
#     #     if field_type is 'Et':
#     #         x = np.copy(self.t)#E['t']
#     #         dx = (x[1] - x[0]) * constants.femto
#     #         summation /= constants.angstrom ** 2  # now in (V/m)**2
#     #
#     #     elif field_type is 'Ew':
#     #         x = np.copy(self.w)#E['w']
#     #         dx = (x[1] - x[0]) / (2 * np.pi) / constants.femto
#     #         summation *= constants.femto ** 2 / constants.angstrom ** 2  # now in fs*(V/m)**2
#     #
#     #     elif field_type is 'Elambda':
#     #         x = np.copy(self.wavelengths)#E['lambda']
#     #         wavs = x[:] * constants.nano  # in m
#     #         dx = (wavs[1] - wavs[0])  # in m
#     #         Elambda = Ex * constants.femto / constants.angstrom  # in V*s/m
#     #         summation = np.sum(Elambda ** 2 * c / wavs ** 2)
#     #         # const.femto/const.angstrom**2 # now in fs*(V/m)**2
#     #
#     #     else:
#     #         print 'invalid field type for fluence calculation'
#     #         dx = 0
#     #
#     #     fluence = 2 * dx * summation / Z0
#     #     fluence /= 1e4  # now in J/cm**2
#     #
#     #     return fluence
#
#     def complex_amplitudes(self):
#         return self.amplitudes.copy()*np.exp(1j*self.phases.copy())
#
#     def get_amplitudes(self):
#         if hasattr(self, 'amplitudes'):
#             return self.amplitudes.copy()
#         else:
#             return None
#
#     def get_phases(self):
#         if hasattr(self, 'phases'):
#             return self.phases.copy()
#         else:
#             return None
#
#     def get_x(self):
#         if hasattr(self, 't'):
#             return self.t.copy()
#         elif hasattr(self, 'w'):
#             return self.w.copy()
#         elif hasattr(self, 'wavelengths'):
#             return self.wavelengths.copy()
#         else:
#             return None
#
#     def to_Elambda(self):
#         elambda = Efield()
#         ftype = self.field_type()
#         if ftype is 'Elambda': elambda = self.copy()
#         elif ftype is 'spectrum':
#             elambda = self.copy()
#             elambda.phases = np.zeros(np.size(self.amplitudes))
#         elif ftype is 'Ew':
#             wavs, phs, amps =\
#                 Elambda_from_Ew(self.w, self.phases, self.amplitudes)
#             elambda.wavelengths = wavs
#             elambda.phases = phs
#             elambda.amplitudes = amps
#         elif ftype is 'Et':
#             wavs, phs, amps = \
#                 Elambda_from_Et(self.w, self.phases,
#                                 self.amplitudes, self.wo)
#             elambda.wavelengths = wavs
#             elambda.phases = phs
#             elambda.amplitudes = amps
#         else: print 'field cannot be transformed to Elambda'
#         return elambda
#
#     def to_Ew(self):
#         ew = Efield()
#         elambda = self.to_Elambda()
#         ew.w, ew.phases, ew.amplitudes =\
#             Ew_from_Elambda(elambda.wavelengths,
#                             elambda.phases, elambda.amplitudes)
#         return ew
#
#     def to_Et(self):
#         et = Efield()
#         elambda = self.to_Elambda()
#         et.t, et.phases, et.amplitudes, et.wo =\
#             Et_from_Elambda(elambda.wavelengths,
#                             elambda.phases, elambda.amplitudes)
#         return et
#
#     def resample(self, xnew):
#         enew = Efield()
#         xold = self.get_x()
#         enew.phases, enew.amplitudes =\
#             resample_field(xnew, xold,
#                            self.phases, self.amplitudes)
#         ftype = self.field_type()
#         if ftype is 'Et':
#             enew.t = np.copy(xnew)
#             enew.wo = self.wo
#         elif ftype is 'Ew': enew.w = np.copy(xnew)
#         elif ftype is 'Elambda': enew.wavelengths = np.copy(xnew)
#         else: print 'invalid field for resampling'
#         return enew
#
#     def plot(self, figname='', figsize=(5, 5)):
#         y = np.copy(self.amplitudes)#E['values']
#
#         if self.hasattr('phases'):
#             y = np.array(y, dtype=complex)
#             y *= np.exp(1j*np.copy(self.phases))
#             is_complex = True
#         else:
#             is_complex = False
#         # is_complex = 'complex' in str(type(y[0]))
#
#         field_type = self.field_type()
#
#         if field_type is 'spectrum':
#             if not figname: figname = 'spectrum plot'
#             plt.figure(figname, figsize=figsize)
#             x = self.wavelengths#E['lambda']
#             plt.plot(x, y)
#             plt.xlabel(r"$\lambda$ (nm)")
#             plt.ylabel('spectrum (a.u.)')
#
#         elif field_type is 'Elambda':
#             if not figname: figname = 'Elambda plot'
#             plt.figure(figname, figsize=figsize)
#             x = self.wavelengths#E['lambda']
#             plt.plot(x, np.abs(y))
#             plt.xlabel(r"$\lambda$ (nm)")
#             plt.ylabel('|E($\lambda$)| (V$\\times$fs/$\dot{A}$)')
#             plt.twinx()
#             plt.plot(x, np.unwrap(np.angle(y)), 'g')
#             plt.ylabel('Phase (rad)')
#
#         elif field_type is 'Ew':
#             if not figname: figname = 'Ew plot'
#             plt.figure(figname, figsize=figsize)
#             x = self.w#E['w']
#             plt.plot(x, np.abs(y))
#             plt.xlabel(r"$\omega$ (1/fs)")
#             plt.ylabel('|E($\omega$)| (V$\\times$fs/$\dot{A}$)')
#             plt.twinx()
#             plt.plot(x, np.unwrap(np.angle(y)), 'g')
#             plt.ylabel('Phase (rad)')
#
#         elif field_type is 'Et':
#             if not figname: figname = 'Et plot'
#             plt.figure(figname, figsize=figsize)
#             x = self.t#E['t']
#             plt.plot(x, np.abs(y))
#             plt.xlabel('t (fs)')
#             plt.ylabel('|E(t)| (V/$\dot{A}$)')
#             plt.twinx()
#             plt.plot(x, np.unwrap(np.angle(y)), 'g')
#             plt.ylabel('Phase (rad)')
#
#         elif field_type is 'slm_mask':
#             if not figname: figname = 'SLM mask plot'
#             plt.figure(figname, figsize=figsize)
#             x = self.wavelengths#E['slm lambda']
#             plt.plot(x, np.abs(y))
#             plt.xlabel("$\lambda$ (nm)")
#             plt.ylabel('SLM amplitude')
#             plt.twinx()
#             plt.plot(x, np.unwrap(np.angle(y)), 'g')
#             plt.ylabel('SLM phase (rad)')
#
#         else:
#             print 'invalid field for plotting'
#
#     # def plot_wavelet(self,
#     #                  min_wav=740,
#     #                  max_wav=840,
#     #                  tmin=-1000,
#     #                  tmax=1000,
#     #                  dt=0.1,
#     #                  nwidths=100,
#     #                  figname='wavelets',
#     #                  show_colorbar=True):
#     #     from scipy import signal
#     #
#     #     if self.field_type() is None:
#     #         print "invalid field for wavelet plot"
#     #         return
#     #
#     #     # get Et:
#     #     e = self.copy()
#     #
#     #     if not e.hasattr('phases'):
#     #         e.phases = np.zeros(np.size(e.amplitudes))
#     #
#     #     if e.hasattr('wavelengths'):
#     #         e = Et_from_Elambda(e)
#     #     elif e.hasattr('w'):
#     #         e = Et_from_Ew(e)
#     #
#     #     t = np.linspace(tmin, tmax, int((tmax-tmin)/float(dt)))  # np.copy(teval)
#     #     e = resample_field(t, e)
#     #     wo = e.wo
#     #     # dt = t[1] - t[0]
#     #
#     #     c = constants.c * 1e9 / 1e15  # nm/fs
#     #     wmax = 2 * np.pi * c / min_wav
#     #     wmin = 2 * np.pi * c / (1.01 * max_wav)
#     #     wlet_max = 2 * np.pi / (dt * 4.34782608696);print wlet_max, wlet_max/wo
#     #     width_i = wlet_max / wmax
#     #     width_f = wlet_max / wmin;print width_i, width_f
#     #     # widths = np.linspace(width_i, width_f, nwidths)
#     #     widths = np.linspace(1, 10, 100)
#     #
#     #     # sig = e.complex_amplitudes()
#     #     # sig = smooth_curve(np.real(sig), s) + 1j*smooth_curve(np.imag(sig), s)
#     #     # sig *= np.exp(1j * (-wmin + wo) * t)
#     #
#     #     sig = e.complex_amplitudes() * np.exp(-1j * wo * t)
#     #     # sig = e.complex_amplitudes() * np.exp(1j * (wmin - wo) * t)
#     #
#     #     cwtmatr = signal.cwt(np.real(sig), signal.ricker, widths)
#     #     cwtmatr = np.array(cwtmatr, dtype=complex)
#     #     cwtmatr += 1j * signal.cwt(np.imag(sig), signal.ricker, widths)
#     #     cwtmatr = np.abs(cwtmatr)
#     #
#     #     wtperiods = dt * 4.34782608696 * widths
#     #     w_wt = 2 * np.pi / wtperiods
#     #     wavs = 2 * np.pi * c / w_wt;print wavs[0], wavs[-1]
#     #
#     #     plt.figure(figname)
#     #
#     #     # plt.pcolormesh(cwtmatr, cmap='PRGn')
#     #
#     #     plt.pcolormesh(t, wavs, cwtmatr,
#     #                    cmap='PRGn', vmax=abs(cwtmatr).max(), vmin=0)
#     #
#     #     # plt.ylim((min_wav, max_wav))
#     #     # plt.xlabel('time (fs)')
#     #     # plt.ylabel('$\lambda$ (nm)')
#     #
#     #     if show_colorbar: plt.colorbar()
#
#     # def plot_wavelet(self,
#     #                  min_wav=740,
#     #                  max_wav=840,
#     #                  tmin=-1000,
#     #                  tmax=1000,
#     #                  nt=2000,
#     #                  nwidths=100,
#     #                  figname='wavelets',
#     #                  show_colorbar=True):
#     #     from scipy import signal
#     #
#     #     if self.field_type() is None:
#     #         print "invalid field for wavelet plot"
#     #         return
#     #
#     #     # get Et:
#     #     e = self.copy()
#     #
#     #     if not e.hasattr('phases'):
#     #         e.phases = np.zeros(np.size(e.amplitudes))
#     #
#     #     if e.hasattr('wavelengths'):
#     #         e = Et_from_Elambda(e)
#     #     elif e.hasattr('w'):
#     #         e = Et_from_Ew(e)
#     #
#     #     t = np.linspace(tmin, tmax, nt)#np.copy(teval)
#     #     e = resample_field(t, e)
#     #     wo = e.wo
#     #     dt = t[1] - t[0]
#     #
#     #     c = constants.c * 1e9 / 1e15  # nm/fs
#     #     wmax = 2 * np.pi * c / min_wav
#     #     wmin = 2 * np.pi * c / (1.01*max_wav)
#     #     wlet_max = 2*np.pi/(dt * 4.34782608696)
#     #     width_i = wlet_max / (wmax - wmin)
#     #     width_f = wlet_max / (0.1 * np.abs(wmin-wo))
#     #     widths = np.linspace(width_i, width_f, nwidths)
#     #     # widths = np.linspace(5, 100, 100)
#     #
#     #     # sig = e.complex_amplitudes()
#     #     # sig = smooth_curve(np.real(sig), s) + 1j*smooth_curve(np.imag(sig), s)
#     #     # sig *= np.exp(1j * (-wmin + wo) * t)
#     #
#     #     sig = e.complex_amplitudes() * np.exp(1j * (- wmin + wo) * t)
#     #
#     #     cwtmatr = signal.cwt(np.real(sig), signal.ricker, widths)
#     #     cwtmatr = np.array(cwtmatr, dtype=complex)
#     #     cwtmatr += 1j * signal.cwt(np.imag(sig), signal.ricker, widths)
#     #     cwtmatr = np.abs(cwtmatr)
#     #
#     #     wtperiods = dt * 4.34782608696 * widths
#     #     w_wt = wmin + 2 * np.pi / wtperiods
#     #     wavs = 2 * np.pi * c / w_wt
#     #
#     #     plt.figure(figname)
#     #
#     #     plt.pcolormesh(t, wavs, cwtmatr,
#     #                    cmap='PRGn', vmax=abs(cwtmatr).max(), vmin=0)
#     #
#     #     plt.ylim((min_wav, max_wav))
#     #     plt.xlabel('time (fs)')
#     #     plt.ylabel('$\lambda$ (nm)')
#     #
#     #     if show_colorbar: plt.colorbar()
#
#     def plot_wavelet(self,
#                      figname='wavelets',
#                      show_colorbar=True,
#                      cmap='PRGn'
#                      ):
#         import dtcwt
#
#         if self.field_type() is None:
#             print "invalid field for wavelet plot"
#             return
#
#         # get Et:
#         e = self.copy()
#
#         if not e.hasattr('phases'):
#             e.phases = np.zeros(np.size(e.amplitudes))
#
#         if e.hasattr('wavelengths'):
#             e = Et_from_Elambda(e)
#         elif e.hasattr('w'):
#             e = Et_from_Ew(e)
#
#         times = e.t
#         dt = times[1] - times[0]
#         ecomplex = e.complex_amplitudes()
#
#         if np.remainder(np.size(times), 2) > 0:
#             times = times[:-1]
#             ecomplex = ecomplex[:-1]
#
#         ereal = np.real(ecomplex)
#         eimag = np.imag(ecomplex)
#         data = np.vstack((ereal, eimag)).T
#         nlevels = int(np.floor(np.log2(np.size(ereal))))
#
#         trans = dtcwt.Transform1d()
#         vec_t = trans.forward(data, nlevels=nlevels)
#
#         plt.figure(figname)
#         matcoeff = []
#         periods = []
#         level = 1
#
#         for peaks in vec_t.highpasses:
#             coeffs = np.sum(np.abs(peaks)**2, axis=1)
#             coeffs = np.sqrt(coeffs)
#             tlevel = times[::2**level]
#             coeffs = interpolate(times, tlevel,
#                                  coeffs,
#                                  kind='linear')
#             matcoeff.append(coeffs)
#             periods.append(dt*2**level)
#             level += 1
#
#         matcoeff = np.array(matcoeff)
#         plt.pcolormesh(times, periods, matcoeff,
#                        cmap=cmap,
#                        vmax=np.max(matcoeff),
#                        vmin=0,
#                        shading='gouraud')
#         ax = plt.gca()
#         ax.set_yscale('log')
#         if show_colorbar: plt.colorbar()
#         plt.ylim((np.min(periods), np.max(periods)))
#         plt.xlim((times[0], times[-1]))
#         plt.xlabel('t (fs)')
#         plt.ylabel('period (fs)')
#
#     def plot_wigner(self,
#                     ti=-1000,
#                     tf=1000,
#                     dt=10,
#                     wav_min=740,
#                     wav_max=860,
#                     figname='wigner',
#                     show_colorbar=True,
#                     cmap='PRGn'):
#
#         twigner, wavs, wigner =\
#             self.calculate_wigner_function(ti, tf, dt)
#
#         plot_wigner(twigner, wavs, wigner,
#                     ti=ti,
#                     tf=tf,
#                     wav_min=wav_min,
#                     wav_max=wav_max,
#                     figname=figname,
#                     show_colorbar=show_colorbar,
#                     cmap=cmap)
#
#     def calculate_wigner_function(self,
#                                   ti=-1000,
#                                   tf=1000,
#                                   dt=10,
#                                   ):#todo: use func instead (with self.toEt)
#         et = self.to_Et()
#         return calculate_wigner_function(et.t, et.phases,
#                                          et.amplitudes, et.wo,
#                                          ti=ti, tf=tf, dt=dt)
#         # field_type = self.field_type()
#         #
#         # # get Et:
#         # if field_type is 'Elambda':
#         #     e = Et_from_Elambda(self)
#         # elif field_type is 'Ew':
#         #     e = Et_from_Ew(self)
#         # elif field_type is 'Et':
#         #     e = self.copy()
#         # else:
#         #     print "invalid field for wigner plot"
#         #     return
#         #
#         # t2res = np.linspace(e.t[0], e.t[-1],
#         #                     2 * np.size(e.t))
#         # e = resample_field(t2res, e)
#         # eplus = e.complex_amplitudes()
#         # eminus = np.copy(np.conj(eplus))[::-1]
#         # nt2 = np.size(eplus)
#         # nt2_half = int(nt2 / 2.)
#         # dt2 = t2res[1] - t2res[0]
#         #
#         # tfull = np.linspace(-nt2 * dt2, nt2 * dt2,
#         #                     2 * nt2)
#         #
#         # twigner = np.linspace(ti, tf,
#         #                       int(float(tf - ti) / dt))
#         # wigner = []
#         #
#         # for t in twigner:
#         #     eplus_long = np.zeros(3 * nt2, dtype=complex)
#         #     ni = int(nt2 - float(t) / dt2)
#         #     eplus_long[ni: ni + nt2] += eplus
#         #     eminus_long = np.zeros(3 * nt2, dtype=complex)
#         #     ni = int(nt2 + float(t) / dt2)
#         #     eminus_long[ni: ni + nt2] += eminus
#         #
#         #     # yt = efull[2*nt2: 3 * nt2: 2]
#         #
#         #     yt = eplus_long * eminus_long
#         #     yt = yt[nt2: 2 * nt2: 2]
#         #     yw = np.fft.fft(np.fft.ifftshift(yt))
#         #     yw = np.fft.fftshift(yw)
#         #     yw = yw  # [nt2: 2 * nt2: 2]
#         #
#         #     wigner.append(yw)
#         #
#         # Nyt = np.size(yw)
#         # wo = e.wo
#         # f = np.fft.fftfreq(Nyt) / (dt2 * 2.)
#         # f = np.fft.fftshift(f)
#         # w = 2 * np.pi * f + wo
#         # c = constants.c * 1e9 / 1e15  # nm/fs
#         # wavs = 2 * np.pi * c / w  # ;plt.plot(wavs)
#         #
#         # wigner = np.array(wigner).T * dt2 * 2.
#         #
#         # return twigner, wavs, wigner
#
#     def load(self, fname='', is_slm=False):
#         df = pd.read_csv(fname)
#         cols = df.columns
#
#         if 'amplitudes' in cols:
#             self.amplitudes = df['amplitudes'].values
#
#         if 'phases' in cols:
#             self.phases = df['phases'].values
#
#         if 't' in cols:
#             self.t = df['t'].values
#             if 'wo' in cols:
#                 self.wo = df['wo'].values[0]
#
#         elif 'w' in cols:
#             self.w = df['w'].values
#
#         elif 'wavelengths' in cols:
#             self.wavelengths = df['wavelengths'].values
#
#         if is_slm:
#             wavs = self.wavelengths
#             self.pix2nm = wavs[1] - wavs[0]
#             self.npixels = np.size(wavs)
#             cwav = wavs[0] + self.pix2nm*(self.npixels/2.)
#             self.central_wavelength = cwav
#
#     def save(self, fname=''):
#         df = pd.DataFrame()
#
#         if hasattr(self, 'wavelengths'):
#             df['wavelengths'] = self.wavelengths
#
#         elif hasattr(self, 'w'):
#             df['w'] = self.w
#
#         elif hasattr(self, 't'):
#             df['t'] = self.t
#
#         if hasattr(self, 'phases'):
#             df['phases'] = self.phases
#
#         if hasattr(self, 'amplitudes'):
#             df['amplitudes'] = self.amplitudes
#
#         if hasattr(self, 'wo'):
#             df['wo'] = self.wo
#
#         df.to_csv(fname, index=False)


def get_spectrum(center_wavelength=800,
                 fwhm=40,
                 lambda_min=740,
                 lambda_max=860,
                 dlambda = 0.1,
                 fname="",
                 data=np.array([[]])):
    """
    modes: 'gaussian' (default), 'input data', and 'file'.
    the data (input or in a file) has to have two columns.
    """
    wavelength, amplitudes = np.array([]), np.array([])

    if fname:
        try:
            data = np.loadtxt(fname)
            shape = np.shape(data)
            nrows = shape[0]
            ncolumns = shape[1]
            # check if file has valid format:
            if (nrows == 0) or (ncolumns != 2):
                print "invalid spectrum file"
            else:
                wavelength = data[:, 0]
                amplitudes = data[:, 1]
        except:
            print "couldn't open file or file not valid"

    elif np.size(data):
        try:
            nrows, ncolumns = np.shape(data)
            # check if data has valid format:
            if (nrows == 0) or (ncolumns != 2):
                print "invalid spectral data"
            else:
                wavelength = data[:, 0]
                amplitudes = data[:, 1]
        except:
            print "invalid spectral data"

    else:
        N_points = (lambda_max - lambda_min) / dlambda
        # N_points = (g["lambda max"] - g["lambda min"]) / g["delta lambda"]
        wavelength = np.linspace(lambda_min, lambda_max, N_points)
        sigma = fwhm / (2 * np.sqrt(np.log(2)))
        amplitudes = \
            np.exp(-( (wavelength - center_wavelength) / sigma) ** 2)

    df = pd.DataFrame()
    df['wavelength'] = wavelength
    df['amplitudes'] = amplitudes

    return df


def Elambda_from_spectrum(spectrum=pd.DataFrame(),
                          fluence=1e-3,
                          offset=0,
                          dt=3, dlambda=0.1):
    # Elambda = spectrum.copy()
    yold = np.copy(spectrum.amplitudes.copy()) - offset
    # yold = Elambda.amplitudes - offset
    xold = np.copy(spectrum.wavelength.copy())#Elambda.wavelengths
    wavc = xold[ np.argmax(yold) ]
    xnew = wavelengths_from_dt_and_dlambda(wavc, dt,
                                           dlambda)
    ynew = interpolate(xnew, xold, yold)

    df = pd.DataFrame()
    df['wavelength'] = xnew
    df['phases'] = np.zeros(np.size(xnew))
    df['amplitudes'] = ynew
    df = renormalize_field(df, fluence=fluence)

    return df


def Ew_from_Elambda(elambda=pd.DataFrame()):
    wavelengths = elambda.wavelength.values
    phases = elambda.phases.values
    amplitudes = elambda.amplitudes.values

    c = constants.c * 1e9 / 1e15  # in nm/fs
    twopi = 2 * constants.pi
    wavs = np.copy(wavelengths)#el['lambda']
    xmax = twopi * c / np.min(wavs)
    xmin = twopi * c / np.max(wavs)
    Npoints = np.size(wavs)
    xold = twopi * c / wavs[:]
    xold = xold[::-1]
    yold = np.copy(amplitudes)*np.exp(1j*np.copy(phases))
    yold = yold[::-1]

    xnew = np.linspace(xmin, xmax, Npoints)

    ynew = interpolate_complex(xnew, xold, yold)

    fluence = calculate_fluence(elambda)

    df = pd.DataFrame()
    df['w'] = xnew
    df['phases'] =  np.unwrap(np.angle(ynew))
    df['amplitudes'] = np.abs(ynew)
    df = renormalize_field(df, fluence=fluence)

    return df


def Et_from_Ew(ew=pd.DataFrame()):
    """
    the convention throrought is:
    E(t) = \int E(w)exp[+i*w*t] dw
    that is, the sign in the exponential is positive, which then
    corresponds to ifft.
    the reverse applyes to the E(w)->E(t) transformation.
    """
    w = ew.w.values
    phases = ew.phases.values
    amplitudes = ew.amplitudes.values

    twopi = 2 * constants.pi
    f = np.copy(w) / twopi
    yw = np.copy(amplitudes)*np.exp(1j*np.copy(phases))
    df = f[1] - f[0]
    Nf = np.size(f)

    wo = (np.max(w) + np.min(w)) / 2.0

    t = np.fft.fftfreq(Nf) / df
    t = np.fft.fftshift(t)

    yt = np.fft.ifft(np.fft.ifftshift(yw))
    yt = np.fft.fftshift(yt)
    yt *= Nf * df

    fluence = calculate_fluence(ew)
    df = pd.DataFrame()
    df['t'] = t
    df['phases'] = np.unwrap(np.angle(yt))
    df['amplitudes'] = np.abs(yt)
    df['wo'] = wo
    df = renormalize_field(df, fluence=fluence)

    return df


def Ew_from_Et(et=pd.DataFrame()):
    """
    the convention throrought is:
    E(t) = \int E(w)exp[+i*w*t] dw
    that is, the sign in the exponential is positive, which then
    corresponds to ifft.
    the reverse applyes to the E(w)->E(t) transformation.
    """
    t = et.t.values.copy()
    phases = et.phases.values
    amplitudes = et.amplitudes.values
    wo = et.wo.values[0]

    twopi = 2 * constants.pi
    yt = np.copy(amplitudes)*np.exp(1j*np.copy(phases))
    dt = t[1] - t[0]
    Nt = np.size(t)

    f = np.fft.fftfreq(Nt) / dt
    f = np.fft.fftshift(f)
    w = twopi * f + wo

    yw = np.fft.fft(np.fft.ifftshift(yt))
    yw = np.fft.fftshift(yw)
    yw *= dt

    fluence = calculate_fluence(et)
    df = pd.DataFrame()
    df['w'] = w
    df['phases'] = np.unwrap(np.angle(yw))
    df['amplitudes'] = np.abs(yw)
    df = renormalize_field(df, fluence=fluence)

    return df


def Elambda_from_Ew(ew=pd.DataFrame()):
    w = ew.w.values
    phases = ew.phases.values
    amplitudes = ew.amplitudes.values

    c = constants.c * 1e9 / 1e15  # in nm/fs
    twopi = 2 * constants.pi
    ws = np.copy(w)#ew.w#ew['w']
    xmax = twopi * c / np.min(ws)
    xmin = twopi * c / np.max(ws)
    xold = twopi * c / ws[:]
    xold = xold[::-1]
    yold = np.copy(amplitudes)*np.exp(1j*np.copy(phases))
    yold = yold[::-1]
    npoints = np.size(ws)
    xnew = np.linspace(xmin, xmax, npoints)

    ynew = interpolate_complex(xnew, xold, yold)

    fluence = calculate_fluence(ew)
    df = pd.DataFrame()
    df['wavelength'] = xnew
    df['phases'] = np.unwrap(np.angle(ynew))
    df['amplitudes'] = np.abs(ynew)
    df = renormalize_field(df, fluence=fluence)

    return df


def Et_from_Elambda(elambda=pd.DataFrame()):
    return Et_from_Ew(Ew_from_Elambda(elambda))


def Elambda_from_Et(et=pd.DataFrame()):
    return Elambda_from_Ew(Ew_from_Et(et))


def construct_slm_mask(phases=np.array([]),
                       amplitudes=np.array([]),
                       central_wavelength=800,
                       pix2nm=0.2,
                       npixels=640):

    # slm wavelengths:
    lambda_i = central_wavelength - npixels / 2. * pix2nm
    lambda_f = lambda_i + (npixels - 1) * pix2nm
    wavs = np.linspace(lambda_i, lambda_f, npixels)

    # slm phases:
    phs = np.array( copy.deepcopy(phases) )
    nphases = np.size(phs)
    if not nphases:
        phs = np.zeros(npixels)
    elif nphases != npixels:
        new_pixels = np.arange(npixels)
        old_pixels = np.linspace(0, npixels, nphases)
        phs = np.unwrap(phs)
        phs = interpolate(new_pixels, old_pixels, phs,
                          'steps')

    # slm amplitudes:
    namps = np.size(amplitudes)
    amps = np.array(copy.deepcopy(amplitudes))
    if not namps:
        amps = np.ones(npixels)
    elif namps != npixels:
        new_pixels = np.arange(npixels)
        old_pixels = np.linspace(0, npixels, namps)
        amps = interpolate(new_pixels, old_pixels, amps,
                           'steps')

    df = pd.DataFrame()
    df['slm_wavelength'] = wavs
    df['phases'] = phs
    df['amplitudes'] = amps

    return df


def slm_mask_from_file(fname='',
                       central_wavelength=800,
                       pix2nm=0.2,
                       npixels=640):
    data = np.loadtxt(fname)

    if np.size(data.shape) > 1:
        if data.shape[1] > 2: data = data.T
        phs, amps = data[:, 0], data[:, 1]

    else:
        phs = data[:]
        amps = np.ones(np.size(phs))

    slm_mask =\
        construct_slm_mask(phs, amps,
                          central_wavelength,
                          pix2nm, npixels)
    return slm_mask


def shape_Elambda(elambda=pd.DataFrame(),
                 slm_mask=pd.DataFrame()):
    xold = np.copy(slm_mask.slm_wavelength.values)
    yold = np.copy(slm_mask.amplitudes.values)*np.exp(1j*np.copy(slm_mask.phases.values))
    xnew = np.copy(elambda.wavelength.values)#Eshaped['lambda']

    ynew = interpolate_complex(xnew, xold, yold)

    enew = elambda.copy()
    enew.phases.values[:] += np.angle(ynew)
    enew.phases.values[:] = np.unwrap(enew.phases.values)
    enew.amplitudes.values[:] *= np.abs(ynew)

    return enew


def slm_mask_from_Elambda(elambda=pd.DataFrame(),
                          elambda_ref=pd.DataFrame(),
                          central_wavelength=800,
                          pix2nm=0.2,
                          npixels=640):
    """
    calculate the SLM mask corresponding to the shaped and unshaped fields.
    """
    # put the ref on the same wavelengths as e
    xold = np.copy(elambda_ref.wavelength.values)#eref['lambda']
    yold = np.copy(elambda_ref.amplitudes.values)*np.exp(1j*np.copy(elambda_ref.phases.values))#eref['values']
    xnew = np.copy(elambda.wavelength.values)#e['lambda']
    yref = interpolate_complex(xnew, xold, yold)

    yslm = np.ones(np.size(xnew), dtype=np.complex)
    yshaped = np.copy(elambda.amplitudes.values)*np.exp(1j*np.copy(elambda.phases.values))
    indxs = yref > 0
    yslm[indxs] = yshaped[indxs] / yref[indxs]

    # calculate slm wavelengths
    lambda_i = central_wavelength - npixels / 2. * pix2nm
    lambda_f = lambda_i + (npixels - 1) * pix2nm
    slm_wavs = np.linspace(lambda_i, lambda_f, npixels)

    # interpolate to slm wavs
    yslm = interpolate_complex(slm_wavs, xnew, yslm)

    return construct_slm_mask(np.unwrap(np.angle(yslm)),
                              np.abs(yslm),
                              central_wavelength,
                              pix2nm, npixels)


def slm_parameters_from_slm_wavelength(slm_wavelength=[]):
    npixels = np.size(slm_wavelength)
    pix2nm = slm_wavelength[1] - slm_wavelength[0]
    wav_i = slm_wavelength[0]
    wav_f = slm_wavelength[-1] + pix2nm/2.
    central_wavelength = (wav_i + wav_f)/2.
    return central_wavelength, pix2nm, npixels



def field_type(efield=pd.DataFrame()):
    if 'phases' in efield: is_complex = True
    else: is_complex = False

    if ('wavelength' in efield) and not is_complex:
        return 'spectrum'
    elif 'slm_wavelength' in efield: return 'slm_mask'
    elif ('wavelength' in efield) and is_complex:
        return 'Elambda'
    elif 'w' in efield: return 'Ew'
    elif 't' in efield: return 'Et'
    else:
        print 'field type not recognized'
        return None


def plot_field(efield=pd.DataFrame(),
               figname=''):
    etype = field_type(efield)
    xlabel = {'spectrum': r"$\lambda$ (nm)",
              'slm_mask': 'SLM amplitude',
              'Elambda': r"$\lambda$ (nm)",
              'Ew': r"$\omega$ (1/fs)",
              'Et': 't (fs)'}
    y1label = {'spectrum': 'spectrum (a.u.)',
              'slm_mask': 'SLM amplitude',
              'Elambda': r'|E($\lambda$)| (V$\times$fs/$\dot{A}$)',
              'Ew': r'|E($\omega$)| (V$\times$fs/$\dot{A}$)',
              'Et': r'|E(t)| (V/$\dot{A}$)'}
    y2label = {'slm_mask': 'SLM Phase (rad)',
               'Elambda': r'Phase (rad)',
               'Ew': r'Phase (rad)',
               'Et': r'Phase (rad)'}
    x = efield.values[:, 0]
    if figname is '': plt.figure(etype)
    else: plt.figure(figname)
    plt.xlabel(xlabel[etype])

    if 'amplitudes' in efield:
        plt.plot(x, efield.amplitudes.values)
        plt.ylabel(y1label[etype])
    if 'phases' in efield:
        plt.twinx()
        plt.plot(x, efield.phases.values, 'g')
        plt.ylabel(y2label[etype])


def calculate_wigner_function(et=pd.DataFrame(),#t, phases, amplitudes, wo,
                              ti=-1000,
                              tf=1000,
                              dt=10,
                              ):
    # efield = et.copy()
    t, wo = et.t.values.copy(), et.wo.values[0]
    # phases, amplitudes = efield.phases.values, efield.amplitudes.values
    t2res = np.linspace(t[0], t[-1],
                        2 * np.size(t))
    eres = resample_field(t2res, et, kind='linear')
    phs, amps = eres.phases.values, eres.amplitudes.values
    # phs, amps = resample_field(t2res, t, phases, amplitudes)
    eplus = amps * np.exp(1j * phs)
    eminus = np.copy(np.conj(eplus))[::-1]
    nt2 = np.size(eplus)
    dt2 = t2res[1] - t2res[0]

    twigner = np.linspace(ti, tf,
                          int(float(tf - ti) / dt))
    wigner = []

    for t in twigner:
        eplus_long = np.zeros(3 * nt2, dtype=complex)
        ni = int(nt2 - float(t) / dt2)
        eplus_long[ni: ni + nt2] += eplus
        eminus_long = np.zeros(3 * nt2, dtype=complex)
        ni = int(nt2 + float(t) / dt2)
        eminus_long[ni: ni + nt2] += eminus

        yt = eplus_long * eminus_long
        yt = yt[nt2: 2 * nt2: 2]
        yw = np.fft.fft(np.fft.ifftshift(yt))
        yw = np.fft.fftshift(yw)
        yw = yw

        wigner.append(yw)

    Nyt = np.size(yw)
    f = np.fft.fftfreq(Nyt) / (dt2 * 2.)
    f = np.fft.fftshift(f)
    w = 2 * np.pi * f + wo
    c = constants.c * 1e9 / 1e15  # nm/fs
    wavs = 2 * np.pi * c / w

    wigner = np.array(wigner).T * dt2 * 2.

    return twigner, wavs, wigner


def plot_wigner(twigner, wavs, wigner,
                ti=-1000,
                tf=1000,
                wav_min=740,
                wav_max=860,
                figname='wigner',
                show_colorbar=True,
                cmap='bone'):
    plt.figure(figname)
    plt.pcolormesh(twigner, wavs, np.abs(wigner),
                   cmap=cmap, #'bone',#'PRGn',
                   vmax=np.max(np.abs(wigner)),
                   vmin=0)
    plt.ylim((wav_min, wav_max))
    plt.xlim((ti, tf))
    plt.xlabel('t (fs)')
    plt.ylabel('$\lambda$ (nm)')
    if show_colorbar: plt.colorbar()


def wavelet_transform(amplitudes, phases=[]):
    import dtcwt

    if phases == []: phases = np.zeros(np.size(amplitudes))
    ecomplex = np.copy(amplitudes) * np.exp(1j * np.copy(phases))

    if np.remainder(np.size(amplitudes), 2) > 0:
        ecomplex = ecomplex[:-1]

    ereal = np.real(ecomplex)
    eimag = np.imag(ecomplex)
    data = np.vstack((ereal, eimag)).T
    nlevels = int(np.floor(np.log2(np.size(ereal))))

    trans = dtcwt.Transform1d()
    vec_t = trans.forward(data, nlevels=nlevels)
    detail_coeffs = vec_t.highpasses
    complex_coeffs = ()

    for peaks in detail_coeffs:
        pks = np.array(peaks)
        if len(pks.shape) == 1: pks = pks[:, np.newaxis]
        complex_coeffs += (pks[:, 0] + 1j * pks[:, 1],)

    return complex_coeffs


def plot_wavelet_piramid(coeffs=(),
                         figname='wavelet piramid plot'):
    nlevels = len(coeffs)
    plt.figure(figname, figsize=(10, 2 * nlevels))
    k = 1

    for peaks in coeffs:
        nrows = np.size(peaks)
        plt.subplot(nlevels, 1, k)
        plt.bar(range(nrows), np.abs(peaks))
        plt.ylabel('level {}'.format(k-1))
        if k == nlevels: plt.xlabel('coefficient index')
        k += 1


def calculate_fluence(efield=pd.DataFrame()):
    """
    fluence in J/cm**2 (not in the standard units here)
    renormalized_field = renormalize_field(field, fluence = 0.0008,
    domain = 'time'):
    Renormalize the field amplitude (field has 3 columns: x,
    phase, and amplitude) such that it corresponds to the
    given fluence.
    three domains: 'time', 'frequency, or 'wavelength'
    which yield ouput fields in units of V/Angstrom,
    V*fs/Angstrom, and V*fs/Angstrom, respectively.
    The default fluence = 0.0008 J/cm^2 corresponds to an
    uncollimated beam with energy = 100 uJ and radius = 2 mm.
    the normalization factor is inferred from
    \sum_k(|Et_k|^2) = Z_0*fluence/(2*dt) or
    \sum_k(|Ew_k|^2) = Z_0*fluence/(2*df), by finding the factor
    kappa that multiplied to the field yields the real field.
    """
    Z0 = 377  # Ohms, the vaccum impedance
    #        Z0 *= 1.6e-4 # now in our units V/(e*fs)
    c = constants.c
    #        c = 2997.9245799999999 #speed of light in Angstrom/fs
    Ex = np.copy(efield.amplitudes.values)#np.abs(np.array(E['values']))
    summation = np.sum(Ex ** 2)

    if 't' in efield:
        x = np.copy(efield.t.values)#E['t']
        dx = (x[1] - x[0]) * constants.femto
        summation /= constants.angstrom ** 2  # now in (V/m)**2

    elif 'w' in efield:
        x = np.copy(efield.w.values)#E['w']
        dx = (x[1] - x[0]) / (2 * np.pi) / constants.femto
        summation *= constants.femto ** 2 / constants.angstrom ** 2  # now in fs*(V/m)**2

    elif 'wavelength' in efield:
        x = np.copy(efield.wavelength.values)#E['lambda']
        wavs = x[:] * constants.nano  # in m
        dx = (wavs[1] - wavs[0])  # in m
        Elambda = Ex * constants.femto / constants.angstrom  # in V*s/m
        summation = np.sum(Elambda ** 2 * c / wavs ** 2)
        # const.femto/const.angstrom**2 # now in fs*(V/m)**2

    else:
        print 'invalid field type for fluence calculation'
        dx = 0

    fluence = 2 * dx * summation / Z0
    fluence /= 1e4  # now in J/cm**2

    return fluence


def renormalize_field(efield=pd.DataFrame(), fluence=1e-3):
    wrong_fluence = calculate_fluence(efield)
    enew = efield.copy()
    enew.amplitudes.values[:] *= np.sqrt(fluence/wrong_fluence)
    return enew


def resample_field(xnew, eold=pd.DataFrame(),
                   kind='linear'):
    amplitudes = eold.amplitudes.values.copy()
    phases = eold.phases.values.copy()
    xold = eold.values[:, 0]
    amps = interpolate(xnew, xold, amplitudes, kind=kind)
    phs = interpolate(xnew, xold, np.unwrap(phases),
                      kind=kind)
    enew = pd.DataFrame()
    for name, value in zip(eold.columns, [xnew, phs, amps]):
        # print name, np.shape(value)
        enew[name] = value

    if 'wo' in eold:
        enew['wo'] = np.size(xnew) * [eold.wo.values[0]]
    return enew


def wavelengths_from_dt_and_dlambda(central_wavelength=800,
                                    dt=3,
                                    dlambda=0.1):
    # determine the central wavelength from the spectrum:

    # determining Elambda's wavelength span
    c = constants.c * 1e9 / 1e15  # in nm/fs
    F = 1. / dt # bandwidth related to dt
    fo = c / central_wavelength

    if fo > F / 2.:
        fmax = fo + F / 2.
        fmin = fo - F / 2.
        df = c / central_wavelength - c / (central_wavelength + dlambda)
        npoints = int( (fmax - fmin) / df)
        minwav, maxwav = c / fmax, c / fmin
        return np.linspace(minwav, maxwav, npoints)

    else:
        mess = "for given central wavelength and dlambda\n"
        mess += "dt has to be larger than {}".format(0.5 * fo)
        sys.exit(mess)


def interpolate(new_x, old_x, old_y,
                interpolation="smooth",
                kind='linear'):
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
    elif kind is 'cubic':
        f = interp1d(x_sorted, y_sorted)
        new_y = f(new_x)
    else:
        new_y = np.interp(new_x, x_sorted, y_sorted)

    return new_y


#def cubic_interp(xnew, xold, yold):
##    import pyximport
##    pyximport.install(reload_support=True)
##    import lagrange_spline
##    spline = lagrange_spline.interp.copy(order='C')
##    ynew = spline(xnew, xold, yold)
#    ynew = lagrange_spline.interp(xnew, xold, yold)
#    ynew[xnew > xold[-1]] = yold[-1]
#    ynew[xnew < xold[0]] = yold[0]
#    return np.ascontiguousarray(ynew)
##    return np.asarray(ynew, order='C')


def interpolate_complex(new_x, old_x, old_y,
                        interpolation='smooth',
                        kind='linear'):
    amps = np.abs(old_y)
    phases = np.unwrap(np.angle(old_y))

    new_amps = interpolate(new_x, old_x, amps,
                           interpolation, kind)
    new_phases = interpolate(new_x, old_x, phases,
                             interpolation, kind)

    return new_amps*np.exp(1j*new_phases)


# def smooth_curve(y, s=1):
#     x = np.arange(np.size(y))
#     spl = UnivariateSpline(x, y, s=s)
#     return spl(x)


if __name__ == '__main__':
    main()

