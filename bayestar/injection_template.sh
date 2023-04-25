#!/bin/bash
lalapps_inspinj \
	`# writes output to named xml file` \
	--output inj.xml \
	`# lower frequency bound (Hz)` \
	--f-lower 25 \
	--source-file "np1.txt" \
	`# type of waveform model used in sim` \
	--waveform TaylorF2threePointFivePN \
	\
	`# Time Distribution: One injection is made every time step until gps time ends.` \
	--t-distr uniform --time-step 315360 --gps-start-time 1000000000 --gps-end-time 1031536000 \
	\
	`# Mass Distribution: Choices from totalMass (uniform in total mass), componentMass (uniform in m1/m2), gaussian, log (log dist in component mass), totalMassRatio (uniform in total and mass ratio), logTotalMassUniformMassRatio (log total, uniform mass ratio), totalMassFraction (uniform in total and m1/(m1+m2)), fixMasses (fixed values)` \
	--m-distr fixMasses --fixed-mass1 1.4 --fixed-mass2 1.4 \
	`# --m-distr componentMass --min-mass1 1.2 --max-mass1 1.4 --min-mass2 1.2 --max-mass2 1.4` \
	`# --m-distr totalMassRatio --min-mtotal 2.2 --max-mtotal 2.8 --min-mratio 0.85 --max-mratio 1.0` \
	`# --m-distr gaussian --mean-mass1 1.4 --stdev-mass1 0.1 --mean-mass2 1.4 --stdev-mass2 0.1` \
	\
	`# Distance Distribution: Choices from uniform (d), distancesquared (d^2), volume (d^3), log10 (log10(d)), or source file. Source file format unclear. Distances in kpc.` \
	`# --d-distr volume --min-distance 1 --max-distance 600e3` \
	--d-distr source \
	\
	`# Localization Distribution: Choices from random (isotropic), fixed (set using --longitude (ra) and --latitude (dec) in degrees), or from source file. Source file format unclear.` \
	--l-distr source \
	`# --l-distr fixed --longitude 8.294086e-01 --latitude -8.197192e-01` \
	\
	`# Inclination Distribution: Choices from uniform (arccos(i)), gaussian in i, or fixed i. Angles in degrees.` \
	--i-distr uniform \
	`# max angle in degrees, for use with uniform distr` \
	`# --max-inc` \
	`# --i-distr fixed --fixed-inc 20` \
	`# --i-distr gaussain --incl-std 2` \
	`# --polarization 30` \
	\
	`# Spin Distribution: Different from other distr args, choice between guassian or uniform.` \
	--disable-spin \
	`# aligned forces the spins to align to the orbital angular momentum` \
	`# --enable-spin --min-spin1 0.0 --max-spin1 1.0 --min-spin2 0.0 --max-spin2 1.0 --aligned` \
	`# --enable-spin --spin-gaussian --mean-spin1 0.5 --stdev-spin1 0.1 --mean-spin2 0.5 --stdev-spin2 0.1` \
	--disable-milkyway

bayestar-sample-model-psd \
	`# Write output to psd.xml` \
	-o psd.xml \
	`# Specify noise models for desired detectors. The ones used here are for O4 design sensitivity, based on LIGO tech report T1800545.` \
	--H1=aLIGO175MpcT1800545 --L1=aLIGO175MpcT1800545 --V1=aLIGOAdVO4T1800545

bayestar-realize-coincs \
	`# Write output to coinc.xml` \
	-o coinc.xml \
	`# Use the injections and noise PSDs that we generated` \
	inj.xml --reference-psd psd.xml \
	`# Specify which detectors are in science mode` \
	--detector H1 L1 V1 \
	`# Optionally, add Gaussian noise (rather than zero noise)` \
	--measurement-error gaussian-noise \
	`# Optionally, adjust the detection threshold: single-detector SNR, network SNR, and minimum number of detectors above threshold to form a coincidence.` \
	--snr-threshold 1.0 \
	--net-snr-threshold 3.0 \
	--min-triggers 2 \
	`# Optionally, save triggers that were below the single-detector threshold` \
	`# --keep-subthreshold` \

bayestar-localize-coincs coinc.xml

ligolw_sqlite --preserve-ids --replace --database coinc.sqlite coinc.xml

ligo-skymap-stats \
	-o bayestar.tsv \
	--database coinc.sqlite \
	*.fits \
	--contour 50 90 \
	--area 10 100 \
	-j 8
