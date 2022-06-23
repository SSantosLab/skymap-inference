import pylab
from gwosc.datasets import event_gps
from gwpy.timeseries import TimeSeries as TS
import bilby
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters, generate_all_bbh_parameters
print(bilby.__version__)

def bbh_parameter_estimation():
    merger = "GW170814"
    gps = float(event_gps(merger))
    
    sampling_rate = 4096
    duration = 4. # length of data in seconds
    start_time = gps-duration/2
    
    strain_dict = dict(L1=TS.fetch_open_data('L1', start_time, start_time+duration, sample_rate=4096),
                       H1=TS.fetch_open_data('H1', start_time, start_time+duration, sample_rate=4096),
                       V1=TS.fetch_open_data('V1', start_time, start_time+duration, sample_rate=4096)
                      )

    psd_dict = dict(L1=TS.fetch_open_data('L1', gps-512, gps-384, sample_rate=4096),
                    H1=TS.fetch_open_data('H1', gps-512, gps-384, sample_rate=4096),
                    V1=TS.fetch_open_data('V1', gps-512, gps-384, sample_rate=4096)
                   )

    interferometers = bilby.gw.detector.InterferometerList([])
    ifo_names = ['L1', 'H1', 'V1']

    for ifo_name in ifo_names:
        ifo = bilby.gw.detector.get_empty_interferometer(ifo_name)
        ifo.set_strain_data_from_gwpy_timeseries(strain_dict[ifo_name])
        ifo.maximum_frequency = 2048
        interferometers.append(ifo)

    psd_alpha = 2 * interferometers[0].strain_data.roll_off / 4

    for i in range(len(ifo_names)):
        psd_dict[ifo_names[i]] = psd_dict[ifo_name].psd(fftlength=4, overlap=0, window=("tukey", psd_alpha), method="median")
        interferometers[i].power_spectral_density = bilby.gw.detector.PowerSpectralDensity(frequency_array=psd_dict[ifo_names[i]].frequencies.value,
                                                                                           psd_array=psd_dict[ifo_names[i]].value
                                                                                          )

    prior = bilby.gw.prior.BBHPriorDict()

    names = ['ra', 'dec']
    mu = [[0.79, -0.83]]
    cov = [[[0.03, 0.], [0., 0.04]]]
    mvg = bilby.core.prior.MultivariateGaussianDist(names, mus=mu, covs=cov)

    prior["ra"] = bilby.core.prior.MultivariateGaussian(mvg, 'ra')
    prior["dec"] = bilby.core.prior.MultivariateGaussian(mvg, 'dec')
    prior["geocent_time"] = bilby.core.prior.Uniform(gps-0.1, gps+0.1, name="geocent_time")

    waveform_arguments = dict(waveform_approximant='IMRPhenomPv2', reference_frequency=100., catch_waveform_errors=True)
    waveform_generator = bilby.gw.WaveformGenerator(frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
                                                    waveform_arguments=waveform_arguments,
                                                    parameter_conversion=convert_to_lal_binary_black_hole_parameters
                                                   )

    likelihood = bilby.gw.likelihood.GravitationalWaveTransient(interferometers,
                                                                waveform_generator,
                                                                priors=prior,
                                                                time_marginalization=True,
                                                                phase_marginalization=True,
                                                                distance_marginalization=False
                                                               )

    result_short = bilby.run_sampler(likelihood,
                                     prior,
                                     sampler='dynesty',
                                     outdir='short',
                                     label='GW170814',
                                     nlive=250,
                                     conversion_function=bilby.gw.conversion.generate_all_bbh_parameters
                                    )
    return None

if __name__ == "__main__":
    bbh_parameter_estimation()