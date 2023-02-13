import bilby
import os
import numpy as np
import healpy as hp
import pandas as pd
import pylab
from astropy.table import Table
import astropy.cosmology as cosmo

######################## Load skymap and cut z/rd/dec #########################

name = 'HLV1'
outdir = 'data/simlogs/'+name
bilby.core.utils.setup_logger(outdir=outdir)

duration = 64.0
sampling_frequency = 2048.0
minimum_frequency = 20

run_data = dict()

catalog = Table.read('data/catalog_fullsky_zsm1.fits.gz').to_pandas()

ra = catalog['RA'].values
dec = catalog['DEC'].values
z = catalog['REDSHIFT'].values

mask = (z<0.08) & (ra>0) & (ra<360) & (np.abs(dec)<90)

ra = ra[mask] * np.pi / 180
dec = dec[mask] * np.pi / 180
z = z[mask]

cosmology = cosmo.WMAP7
bilby.gw.cosmology.set_cosmology(cosmology=cosmology)

dl = bilby.gw.conversion.redshift_to_luminosity_distance(z)

###############################################################################



####################### Initialize Injection Parameters #######################

data_check = pd.read_csv('data/injection_parameters.csv')

for key in data_check:
    if key == 'id':
        data_check[key] = np.array(data_check[key]).astype('str')
    elif key == 'index':
        data_check[key] = np.array(data_check[key]).astype('int')
    else:
        data_check[key] = np.array(data_check[key]).astype('float')

if name in data_check['id'].values:
    injection_parameters = dict(mass_1=float(data_check['mass_1'][data_check['id']==name]),
                                mass_2=float(data_check['mass_2'][data_check['id']==name]),
                                a_1=float(data_check['a_1'][data_check['id']==name]),
                                a_2=float(data_check['a_2'][data_check['id']==name]),
                                tilt_1=float(data_check['tilt_1'][data_check['id']==name]),
                                tilt_2=float(data_check['tilt_2'][data_check['id']==name]),
                                phi_12=float(data_check['phi_12'][data_check['id']==name]),
                                phi_jl=float(data_check['phi_jl'][data_check['id']==name]),
                                luminosity_distance=float(data_check['luminosity_distance'][data_check['id']==name]),
                                theta_jn=float(data_check['theta_jn'][data_check['id']==name]),
                                psi=float(data_check['psi'][data_check['id']==name]),
                                phase=float(data_check['phase'][data_check['id']==name]),
                                geocent_time=float(data_check['geocent_time'][data_check['id']==name]),
                                ra=float(data_check['ra'][data_check['id']==name]),
                                dec=float(data_check['dec'][data_check['id']==name])
                               )
    
    run_data['index'] = int(data_check['index'][data_check['id']==name])
    run_data['id'] = name
    for key in injection_parameters:
        run_data[key] = injection_parameters[key]
else:
    masses = np.random.uniform(1.5,10,2)
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

    pd.DataFrame([run_data]).to_csv('data/injection_parameters.csv', mode='a', index=False, header=False)

###############################################################################



######################### Initialize Interferometers ##########################

waveform_arguments = dict(waveform_approximant="IMRPhenomPv2",
                          reference_frequency=50.0,
                          minimum_frequency=minimum_frequency,
                          catch_waveform_errors=True
                         )

waveform_generator = bilby.gw.WaveformGenerator(duration=duration,
                                                sampling_frequency=sampling_frequency,
                                                frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
                                                parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
                                                waveform_arguments=waveform_arguments
                                               )

ifos = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])

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

prior = bilby.gw.prior.PriorDict()

prior["mass_ratio"] = bilby.core.prior.Uniform(name='mass_ratio', minimum=0.15, maximum=1)
prior["chirp_mass"] = bilby.core.prior.Uniform(name='chirp_mass', minimum=1.3, maximum=8.71)
prior["mass_1"] = bilby.core.prior.Constraint(name='mass_1', minimum=1.3, maximum=12)
prior["mass_2"] = bilby.core.prior.Constraint(name='mass_2', minimum=1.3, maximum=12)
prior["a_1"] = bilby.core.prior.Uniform(name='a_1', minimum=0, maximum=0.99)
prior["a_2"] = bilby.core.prior.Uniform(name='a_2', minimum=0, maximum=0.99)
prior["tilt_1"] = bilby.core.prior.Sine(name='tilt_1')
prior["tilt_2"] = bilby.core.prior.Sine(name='tilt_2')
prior["phi_12"] = bilby.core.prior.Uniform(name='phi_12', minimum=0, maximum=2 * np.pi, boundary='periodic')
prior["phi_jl"] = bilby.core.prior.Uniform(name='phi_jl', minimum=0, maximum=2 * np.pi, boundary='periodic')
prior["theta_jn"] = bilby.core.prior.Sine(name='theta_jn')
prior["psi"] = bilby.core.prior.Uniform(name='psi', minimum=0, maximum=np.pi, boundary='periodic')
prior["phase"] = bilby.core.prior.Uniform(name='phase', minimum=0, maximum=2 * np.pi, boundary='periodic')

prior["luminosity_distance"] = bilby.gw.prior.UniformSourceFrame(name='luminosity_distance', minimum=2e1, maximum=4e2, unit='Mpc', latex_label='$d_L$')
prior["geocent_time"] = bilby.core.prior.Uniform(injection_parameters['geocent_time']-0.1, injection_parameters['geocent_time']+0.1, name="geocent_time")

hp_prior = bilby.gw.prior.HealPixMapPriorDist('data/catalog_skymap.fits', names=['ra', 'dec'], bounds={'ra': (0, 2*np.pi), 'dec': (-np.pi/2, np.pi/2)})
prior["ra"] = bilby.gw.prior.HealPixPrior(hp_prior, 'ra')
prior["dec"] = bilby.gw.prior.HealPixPrior(hp_prior, 'dec')

likelihood = bilby.gw.GravitationalWaveTransient(interferometers=ifos, waveform_generator=waveform_generator, priors=prior,
                                                 time_marginalization=True, phase_marginalization=False, distance_marginalization=True)

result = bilby.run_sampler(likelihood=likelihood,
                           priors=prior,
                           sampler="dynesty",
                           npoints=1200,
                           injection_parameters=injection_parameters,
                           outdir=outdir,
                           label=name+'_fx-inj',
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

run_data['dl_med'] = median
run_data['dl_upper'] = upper_bound
run_data['dl_lower'] = lower_bound
run_data['dl_avg'] = mean
run_data['dl_std'] = std

ra = result.posterior["ra"].values
lower_bound = np.quantile(ra, 0.05)
upper_bound = np.quantile(ra, 0.95)
median = np.median(ra)
mean = np.mean(ra)
std = np.std(ra)

run_data['ra_med'] = median
run_data['ra_upper'] = upper_bound
run_data['ra_lower'] = lower_bound
run_data['ra_avg'] = mean
run_data['ra_std'] = std

dec = result.posterior["dec"].values
lower_bound = np.quantile(dec, 0.05)
upper_bound = np.quantile(dec, 0.95)
median = np.median(dec)
mean = np.mean(dec)
std = np.std(dec)

run_data['dec_med'] = median
run_data['dec_upper'] = upper_bound
run_data['dec_lower'] = lower_bound
run_data['dec_avg'] = mean
run_data['dec_std'] = std

run_data['id'] = name+'_fx'

result.save_posterior_samples()

pd.DataFrame([run_data]).to_csv('data/sim_results.csv', mode='a', index=False, header=False)
