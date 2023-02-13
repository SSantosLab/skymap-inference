import bilby
import os
import numpy as np
import healpy as hp
import pandas as pd
import pylab
from astropy.table import Table
import astropy.cosmology as cosmo

######################## Load skymap and cut z/rd/dec #########################

name = 'HL0_lite_test'
outdir = 'data/simlogs/'+name
bilby.core.utils.setup_logger(outdir=outdir)

duration = 4.0
sampling_frequency = 2048.0
minimum_frequency = 20

run_data = dict()

catalog = Table.read('data/catalog_fullsky_zsm1.fits.gz').to_pandas()

ra = catalog['RA'].values
dec = catalog['DEC'].values
z = catalog['REDSHIFT'].values

mask = (z<0.2) & (ra>0) & (ra<360) & (np.abs(dec)<90)

ra = ra[mask] * np.pi / 180
dec = dec[mask] * np.pi / 180
z = z[mask]

cosmology = cosmo.WMAP7
bilby.gw.cosmology.set_cosmology(cosmology=cosmology)

dl = bilby.gw.conversion.redshift_to_luminosity_distance(z)

###############################################################################



####################### Initialize Injection Parameters #######################

masses = np.random.uniform(25,35,2)
index = np.random.randint(0, len(z))
sindist = bilby.core.prior.Sine()
cosdist = bilby.core.prior.Cosine()

injection_parameters = dict(mass_1=np.max(masses),
                            mass_2=np.min(masses),
                            a_1=np.random.uniform(0, 0.99),
                            a_2=np.random.uniform(0, 0.99),
                            tilt_1=sindist.sample(),
                            tilt_2=sindist.sample(),
                            phi_12=np.random.uniform(0, 2*np.pi),
                            phi_jl=np.random.uniform(0, 2*np.pi),
                            luminosity_distance=dl[index],
                            theta_jn=sindist.sample(),
                            psi=np.random.uniform(0, np.pi),
                            phase=np.random.uniform(0, 2*np.pi),
                            geocent_time=np.random.uniform(1.1e9, 1.2e9),
                            ra=ra[index],
                            dec=dec[index]
                           )

run_data['index'] = index
run_data['id'] = name
for key in injection_parameters:
    run_data[key] = injection_parameters[key]

waveform_arguments = dict(waveform_approximant="IMRPhenomPv2",
                          reference_frequency=50.0,
                          minimum_frequency=minimum_frequency,
                          catch_waveform_errors=True
                         )

###############################################################################



######################### Initialize Interferometers ##########################

waveform_generator = bilby.gw.WaveformGenerator(duration=duration,
                                                sampling_frequency=sampling_frequency,
                                                frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
                                                parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
                                                waveform_arguments=waveform_arguments
                                               )

ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])

ifos.set_strain_data_from_power_spectral_densities(sampling_frequency=sampling_frequency,
                                                   duration=duration,
                                                   start_time=injection_parameters["geocent_time"] - 2
                                                  )

ifos.inject_signal(waveform_generator=waveform_generator, parameters=injection_parameters)

if os.path.isfile(".distance_marginalization_lookup.npz"):
    os.remove(".distance_marginalization_lookup.npz")
    print("Distance Lookup Tables Deleted")
else:
    print("No Distance Lookup Tables Found")

###############################################################################



######################## Priors, Likelihoods, and Runs ########################

prior = bilby.gw.prior.BBHPriorDict()

for key in ["mass_1",
            "mass_2",
            "a_1",
            "a_2",
            "tilt_1",
            "tilt_2",
            "theta_jn",
            "phi_12",
            "phi_jl",
            "psi",
            "phase",
           ]:
    prior[key] = injection_parameters[key]

prior.pop("chirp_mass")
prior.pop("mass_ratio")

prior["luminosity_distance"] = bilby.gw.prior.UniformSourceFrame(name='luminosity_distance', minimum=2e1, maximum=1.5e3, unit='Mpc', latex_label='$d_L$')
prior["geocent_time"] = bilby.core.prior.Uniform(injection_parameters['geocent_time']-0.1, injection_parameters['geocent_time']+0.1, name="geocent_time")

likelihood = bilby.gw.GravitationalWaveTransient(interferometers=ifos, waveform_generator=waveform_generator, priors=prior,
                                                 time_marginalization=True, phase_marginalization=False, distance_marginalization=True)

result = bilby.run_sampler(likelihood=likelihood,
                           priors=prior,
                           sampler="dynesty",
                           npoints=500,
                           injection_parameters=injection_parameters,
                           outdir=outdir,
                           label=name+'-injection-uniform',
                           conversion_function=bilby.gw.conversion.generate_all_bbh_parameters
                          )

###############################################################################



############################ Posterior Analysis ###############################

dl = result.posterior["luminosity_distance"].values
lower_bound = np.quantile(dl, 0.05)
upper_bound = np.quantile(dl, 0.95)
median = np.median(dl)
mean = np.mean(dl)
std = np.std(dl)

run_data['un_dl_med'] = median
run_data['un_dl_upper'] = upper_bound
run_data['un_dl_lower'] = lower_bound
run_data['un_dl_avg'] = mean
run_data['un_dl_std'] = std

ra = result.posterior["ra"].values
lower_bound = np.quantile(ra, 0.05)
upper_bound = np.quantile(ra, 0.95)
median = np.median(ra)
mean = np.mean(ra)
std = np.std(ra)

run_data['un_ra_med'] = median
run_data['un_ra_upper'] = upper_bound
run_data['un_ra_lower'] = lower_bound
run_data['un_ra_avg'] = mean
run_data['un_ra_std'] = std

dec = result.posterior["dec"].values
lower_bound = np.quantile(dec, 0.05)
upper_bound = np.quantile(dec, 0.95)
median = np.median(dec)
mean = np.mean(dec)
std = np.std(dec)

run_data['un_dec_med'] = median
run_data['un_dec_upper'] = upper_bound
run_data['un_dec_lower'] = lower_bound
run_data['un_dec_avg'] = mean
run_data['un_dec_std'] = std

result.save_posterior_samples()

###############################################################################



############################### HEALPix Run ###################################

waveform_arguments = dict(waveform_approximant="IMRPhenomPv2",
                          reference_frequency=50.0,
                          minimum_frequency=minimum_frequency
                         )

waveform_generator = bilby.gw.WaveformGenerator(duration=duration,
                                                sampling_frequency=sampling_frequency,
                                                frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
                                                parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
                                                waveform_arguments=waveform_arguments
                                               )

ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])

ifos.set_strain_data_from_power_spectral_densities(sampling_frequency=sampling_frequency,
                                                   duration=duration,
                                                   start_time=injection_parameters["geocent_time"] - 2
                                                  )

ifos.inject_signal(waveform_generator=waveform_generator, parameters=injection_parameters)

if os.path.isfile(".distance_marginalization_lookup.npz"):
    os.remove(".distance_marginalization_lookup.npz")
    print("Distance Lookup Tables Deleted")
else:
    print("No Distance Lookup Tables Found")

prior = bilby.gw.prior.PriorDict()

for key in ["mass_1",
            "mass_2",
            "a_1",
            "a_2",
            "tilt_1",
            "tilt_2",
            "theta_jn",
            "phi_12",
            "phi_jl",
            "psi",
            "phase",
           ]:
    prior[key] = injection_parameters[key]

prior["luminosity_distance"] = bilby.gw.prior.UniformSourceFrame(name='luminosity_distance', minimum=2e1, maximum=1.5e3, unit='Mpc', latex_label='$d_L$')
prior["geocent_time"] = bilby.core.prior.Uniform(injection_parameters['geocent_time']-0.1, injection_parameters['geocent_time']+0.1, name="geocent_time")

hp_prior = bilby.gw.prior.HealPixMapPriorDist('data/catalog_skymap.fits', names=['ra', 'dec'], bounds={'ra': (0, 2*np.pi), 'dec': (-np.pi/2, np.pi/2)})
prior["ra"] = bilby.gw.prior.HealPixPrior(hp_prior, 'ra')
prior["dec"] = bilby.gw.prior.HealPixPrior(hp_prior, 'dec')

likelihood = bilby.gw.GravitationalWaveTransient(interferometers=ifos, waveform_generator=waveform_generator, priors=prior,
                                                 time_marginalization=True, phase_marginalization=False, distance_marginalization=True)

result_hp = bilby.run_sampler(likelihood=likelihood,
                           priors=prior,
                           sampler="dynesty",
                           npoints=500,
                           injection_parameters=injection_parameters,
                           outdir=outdir,
                           label=name+'-injection-healpix',
                           conversion_function=bilby.gw.conversion.generate_all_bbh_parameters
                          )

###############################################################################



############################ Posterior Analysis ###############################

dl = result_hp.posterior["luminosity_distance"].values
lower_bound = np.quantile(dl, 0.05)
upper_bound = np.quantile(dl, 0.95)
median = np.median(dl)
mean = np.mean(dl)
std = np.std(dl)

run_data['hp_dl_med'] = median
run_data['hp_dl_upper'] = upper_bound
run_data['hp_dl_lower'] = lower_bound
run_data['hp_dl_avg'] = mean
run_data['hp_dl_std'] = std

ra = result_hp.posterior["ra"].values
lower_bound = np.quantile(ra, 0.05)
upper_bound = np.quantile(ra, 0.95)
median = np.median(ra)
mean = np.mean(ra)
std = np.std(ra)

run_data['hp_ra_med'] = median
run_data['hp_ra_upper'] = upper_bound
run_data['hp_ra_lower'] = lower_bound
run_data['hp_ra_avg'] = mean
run_data['hp_ra_std'] = std

dec = result_hp.posterior["dec"].values
lower_bound = np.quantile(dec, 0.05)
upper_bound = np.quantile(dec, 0.95)
median = np.median(dec)
mean = np.mean(dec)
std = np.std(dec)

run_data['hp_dec_med'] = median
run_data['hp_dec_upper'] = upper_bound
run_data['hp_dec_lower'] = lower_bound
run_data['hp_dec_avg'] = mean
run_data['hp_dec_std'] = std

result_hp.save_posterior_samples()

pd.DataFrame([run_data]).to_csv('data/sim_results.csv', mode='a', index=False, header=False)
