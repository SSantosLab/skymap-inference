import bilby
import numpy as np
import pandas as pd
from astropy.table import Table
import astropy.cosmology as cosmo
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--label', type=str, required=True)
parser.add_argument('-n', '--nodes', type=int, required=True)
parser.add_argument('-t', '--time', type=str, required=True)
parser.add_argument('-d', '--detectors', type=str, nargs='+')
parser.add_argument('--nlive', type=int)
parser.add_argument('--nact', type=int)
args = parser.parse_args()

name = args.label
run_data = dict()

if args.detectors:
    detectors = '['
    for ifo in args.detectors:
        detectors = detectors+ifo+', '
    detectors = detectors[:len(detectors)-2]+']'
else:
    detectors = '[H1, L1]'

nodes = args.nodes
time = args.time

if args.nlive:
    nlive = args.nlive
else:
    nlive = 1500

if args.nact:
    nact = args.nact
else:
    nact = 10

catalog = Table.read('data/catalog_fullsky_zsm1.fits.gz').to_pandas()

ra = catalog['RA'].values
dec = catalog['DEC'].values
z = catalog['REDSHIFT'].values

mask = (z<0.2) & (ra>0) & (ra<360) & (np.abs(dec)<90)

# z = 0.08 ~ 375 Mpc
# z = 0.11 ~ 525 Mpc
# z = 0.15 ~ 750 Mpc
# z = 0.18 ~ 900 Mpc
# z = 0.20 ~ 1000 Mpc
# z = 0.25 ~ 1300 Mpc

ra = ra[mask] * np.pi / 180
dec = dec[mask] * np.pi / 180
z = z[mask]

cosmology = cosmo.WMAP7
bilby.gw.cosmology.set_cosmology(cosmology=cosmology)

dl = bilby.gw.conversion.redshift_to_luminosity_distance(z)

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

pd.DataFrame([run_data]).to_csv(name+'_injection_parameters.csv', index=False)

prior = bilby.gw.prior.PriorDict()

prior["mass_ratio"] = bilby.core.prior.Uniform(name='mass_ratio', minimum=0.125, maximum=1)
prior["chirp_mass"] = bilby.core.prior.Uniform(name='chirp_mass', minimum=25, maximum=80)
prior["theta_jn"] = bilby.core.prior.Sine(name='theta_jn')
prior["psi"] = bilby.core.prior.Uniform(name='psi', minimum=0, maximum=np.pi, boundary='periodic')
prior["phase"] = bilby.core.prior.Uniform(name='phase', minimum=0, maximum=2 * np.pi, boundary='periodic')
prior["a_1"] = bilby.core.prior.Uniform(name='a_1', minimum=0, maximum=0.99)
prior["a_2"] = bilby.core.prior.Uniform(name='a_2', minimum=0, maximum=0.99)
prior["tilt_1"] = bilby.core.prior.Sine(name='tilt_1')
prior["tilt_2"] = bilby.core.prior.Sine(name='tilt_2')
prior["phi_12"] = bilby.core.prior.Uniform(name='phi_12', minimum=0, maximum=2 * np.pi, boundary='periodic')
prior["phi_jl"] = bilby.core.prior.Uniform(name='phi_jl', minimum=0, maximum=2 * np.pi, boundary='periodic')
prior["luminosity_distance"] = bilby.core.prior.PowerLaw(alpha=2, name='luminosity_distance', minimum=2e1, maximum=2e3, unit='Mpc', latex_label='$d_L$')
prior["geocent_time"] = bilby.core.prior.Uniform(injection_parameters['geocent_time']-0.1, injection_parameters['geocent_time']+0.1, name="geocent_time")
prior["dec"] = bilby.core.prior.Cosine(name='dec')
prior["ra"] = bilby.core.prior.Uniform(name='ra', minimum=0, maximum=2 * np.pi, boundary='periodic')

prior.to_file(outdir='.', label=name+'_un')

ini_str = """
################################################################################
## Data generation arguments
################################################################################
detectors = {}
gaussian-noise = True
zero-noise = False
clean = True
no-plot = True
sampling-frequency = 2048
duration = 4
trigger_time = {}
injection-dict = {}

################################################################################
## Job submission arguments
################################################################################
label = {}
outdir = data/simlogs/{}

################################################################################
## Likelihood arguments
################################################################################
distance-marginalization=True
phase-marginalization=False
time-marginalization=True

################################################################################
## Prior arguments
################################################################################
prior-file = {}

################################################################################
## Waveform arguments
################################################################################
waveform_approximant = IMRPhenomPv2
frequency-domain-source-model = lal_binary_black_hole

###############################################################################
## Sampler settings
################################################################################
sampler = dynesty
nlive = {}
nact = {}

################################################################################
## Slurm Settings
################################################################################
nodes = {}
ntasks-per-node = 32
time = {}
""".format(
    detectors,
    injection_parameters["geocent_time"],
    str(injection_parameters),
    name+'_un',
    name+'_un',
    name+'_un.prior',
    nlive,
    nact,
    nodes,
    time
)

ini_file = open(name+"_un_injection_pbilby.ini", "w")
n = ini_file.write(ini_str)
ini_file.close()



prior = bilby.gw.prior.PriorDict()

prior["mass_ratio"] = bilby.core.prior.Uniform(name='mass_ratio', minimum=0.125, maximum=1)
prior["chirp_mass"] = bilby.core.prior.Uniform(name='chirp_mass', minimum=25, maximum=80)
prior["theta_jn"] = bilby.core.prior.Sine(name='theta_jn')
prior["psi"] = bilby.core.prior.Uniform(name='psi', minimum=0, maximum=np.pi, boundary='periodic')
prior["phase"] = bilby.core.prior.Uniform(name='phase', minimum=0, maximum=2 * np.pi, boundary='periodic')
prior["a_1"] = bilby.core.prior.Uniform(name='a_1', minimum=0, maximum=0.99)
prior["a_2"] = bilby.core.prior.Uniform(name='a_2', minimum=0, maximum=0.99)
prior["tilt_1"] = bilby.core.prior.Sine(name='tilt_1')
prior["tilt_2"] = bilby.core.prior.Sine(name='tilt_2')
prior["phi_12"] = bilby.core.prior.Uniform(name='phi_12', minimum=0, maximum=2 * np.pi, boundary='periodic')
prior["phi_jl"] = bilby.core.prior.Uniform(name='phi_jl', minimum=0, maximum=2 * np.pi, boundary='periodic')
prior["luminosity_distance"] = bilby.core.prior.PowerLaw(alpha=2, name='luminosity_distance', minimum=2e1, maximum=2e3, unit='Mpc', latex_label='$d_L$')
prior["geocent_time"] = bilby.core.prior.Uniform(injection_parameters['geocent_time']-0.1, injection_parameters['geocent_time']+0.1, name="geocent_time")

hp_prior = bilby.gw.prior.HealPixMapPriorDist('data/catalog_skymap.fits', names=['ra', 'dec'], bounds={'ra': (0, 2*np.pi), 'dec': (-np.pi/2, np.pi/2)})
prior["ra"] = bilby.gw.prior.HealPixPrior(hp_prior, 'ra')
prior["dec"] = bilby.gw.prior.HealPixPrior(hp_prior, 'dec')

prior.to_file(outdir='.', label=name+'_hp')

ini_str = """
################################################################################
## Data generation arguments
################################################################################
detectors = {}
gaussian-noise = True
zero-noise = False
clean = True
no-plot = True
sampling-frequency = 2048
duration = 4
trigger_time = {}
injection-dict = {}

################################################################################
## Job submission arguments
################################################################################
label = {}
outdir = data/simlogs/{}

################################################################################
## Likelihood arguments
################################################################################
distance-marginalization=True
phase-marginalization=False
time-marginalization=True

################################################################################
## Prior arguments
################################################################################
prior-file = {}

################################################################################
## Waveform arguments
################################################################################
waveform_approximant = IMRPhenomPv2
frequency-domain-source-model = lal_binary_black_hole

###############################################################################
## Sampler settings
################################################################################
sampler = dynesty
nlive = {}
nact = {}

################################################################################
## Slurm Settings
################################################################################
nodes = {}
ntasks-per-node = 32
time = {}
""".format(
    detectors,
    injection_parameters["geocent_time"],
    str(injection_parameters),
    name+'_hp',
    name+'_hp',
    name+'_hp.prior',
    nlive,
    nact,
    nodes,
    time
)

ini_file = open(name+"_hp_injection_pbilby.ini", "w")
n = ini_file.write(ini_str)
ini_file.close()