#+
# Name:
#		GRIDRAD Python Module
# Purpose:
#		This module contains three functions for dealing with Gridded NEXRAD WSR-88D Radar
#		(GridRad) data: reading (read_file), filtering (filter), and decluttering (remove_clutter).
# Author and history:
#		Cameron R. Homeyer  2017-07-03.
#                         2021-02-23. Updated to be compatible with v4.2 GridRad data and v3 Python.
# Warning:
#		The authors' primary coding language is not Python. This code works, but may not be
#      the most efficient or proper approach. Please suggest improvements by sending an email
#		 to chomeyer@ou.edu.
#-
from multiprocessing import Pool
from datetime import datetime, timedelta
import sys, os
import requests
import sys
import os
import numpy as np
import netCDF4
from netCDF4 import Dataset, Variable
import matplotlib.pyplot as plt
import  FileList4Download,FileListUndownloaded
import pandas as pd

ext_storage="U:/skartalml/"


# GridRad read routine
def read_file(infile):
	# Check to see if file exists
	if not os.path.isfile(infile):
		print('File "' + infile + '" does not exist.  Returning -2.')
		return -2

	# Check to see if file has size of zero
	if os.stat(infile).st_size == 0:
		print('File "' + infile + '" contains no valid data.  Returning -1.')
		return -1

	# Open GridRad netCDF file
	id = Dataset(infile, "r", format="NETCDF4")

	# Read global attributes
	Analysis_time           = str(id.getncattr('Analysis_time'          ))
	Analysis_time_window    = str(id.getncattr('Analysis_time_window'   ))
	File_creation_date      = str(id.getncattr('File_creation_date'     ))
	Grid_scheme             = str(id.getncattr('Grid_scheme'            ))
	Algorithm_version       = str(id.getncattr('Algorithm_version'      ))
	Algorithm_description   = str(id.getncattr('Algorithm_description'  ))
	Authors                 = str(id.getncattr('Authors'                ))
	Project_sponsor         = str(id.getncattr('Project_sponsor'        ))
	Project_name            = str(id.getncattr('Project_name'           ))

	# Read list of merged radar sweeps
	sweeps_list   = (id.variables['sweeps_merged'])[:]
	sweeps_merged = ['']*(id.dimensions['Sweep'].size)
	for i in range(0,id.dimensions['Sweep'].size):
		for j in range(0,id.dimensions['SweepRef'].size):
			sweeps_merged[i] += str(sweeps_list[i,j])

	# Read longitude dimension
	x = id.variables['Longitude']
	x = {'values'    : 360-x[:],             \
		  'long_name' : str(x.long_name), \
		  'units'     : str(x.units),     \
		  'delta'     : str(x.delta),     \
		  'n'         : len(x[:])}

	# Read latitude dimension
	y = id.variables['Latitude']
	y = {'values'    : y[:],             \
		  'long_name' : str(y.long_name), \
		  'units'     : str(y.units),     \
		  'delta'     : str(y.delta),     \
		  'n'         : len(y[:])}

	# Read altitude dimension
	z = id.variables['Altitude']
	z = {'values'    : z[:],             \
		  'long_name' : str(z.long_name), \
		  'units'     : str(z.units),     \
		  'delta'     : str(z.delta),     \
		  'n'         : len(z[:])}

	# Read observation and echo counts
	nobs  = (id.variables['Nradobs' ])[:]
	necho = (id.variables['Nradecho'])[:]
	index = (id.variables['index'   ])[:]

	# Read reflectivity at horizontal polarization
	Z_H  = id.variables['Reflectivity' ]
	wZ_H = id.variables['wReflectivity']

	# Create arrays to store binned values for reflectivity at horizontal polarization
	values    = np.zeros(x['n']*y['n']*z['n'])
	wvalues   = np.zeros(x['n']*y['n']*z['n'])
	values[:] = float('nan')

	# Add values to arrays
	values[index[:]]  =  (Z_H)[:]
	wvalues[index[:]] = (wZ_H)[:]

	# Reshape arrays to 3-D GridRad domain
	values  =  values.reshape((z['n'], y['n'] ,x['n']))
	wvalues = wvalues.reshape((z['n'], y['n'] ,x['n']))

	Z_H = {'values'     : values,              \
			 'long_name'  : str(Z_H.long_name),  \
			 'units'      : str(Z_H.units),      \
			 'missing'    : float('nan'),        \
			 'wvalues'    : wvalues,             \
			 'wlong_name' : str(wZ_H.long_name), \
			 'wunits'     : str(wZ_H.units),     \
			 'wmissing'   : wZ_H.missing_value,  \
			 'n'          : values.size}

	# Read velocity spectrum width
	SW  = id.variables['SpectrumWidth' ]
	wSW = id.variables['wSpectrumWidth']

	# Create arrays to store binned values for velocity spectrum width
	values    = np.zeros(x['n']*y['n']*z['n'])
	wvalues   = np.zeros(x['n']*y['n']*z['n'])
	values[:] = float('nan')

	# Add values to arrays
	values[index[:]]  =  (SW)[:]
	wvalues[index[:]] = (wSW)[:]

	# Reshape arrays to 3-D GridRad domain
	values  =  values.reshape((z['n'], y['n'] ,x['n']))
	wvalues = wvalues.reshape((z['n'], y['n'] ,x['n']))

	SW  = {'values'     : values,             \
			 'long_name'  : str(SW.long_name),  \
			 'units'      : str(SW.units),      \
			 'missing'    : float('nan'),       \
			 'wvalues'    : wvalues,            \
			 'wlong_name' : str(wSW.long_name), \
			 'wunits'     : str(wSW.units),     \
			 'wmissing'   : wSW.missing_value,  \
			 'n'          : values.size}

	if ('AzShear' in id.variables):
		# Read azimuthal shear
		AzShr  = id.variables['AzShear' ]
		wAzShr = id.variables['wAzShear']

		# Create arrays to store binned values for azimuthal shear
		values    = np.zeros(x['n']*y['n']*z['n'])
		wvalues   = np.zeros(x['n']*y['n']*z['n'])
		values[:] = float('nan')

		# Add values to arrays
		values[index[:]]  =  (AzShr)[:]
		wvalues[index[:]] = (wAzShr)[:]

		# Reshape arrays to 3-D GridRad domain
		values  =  values.reshape((z['n'], y['n'] ,x['n']))
		wvalues = wvalues.reshape((z['n'], y['n'] ,x['n']))

		AzShr = {'values'     : values,                \
				   'long_name'  : str(AzShr.long_name),  \
				   'units'      : str(AzShr.units),      \
				   'missing'    : float('nan'),          \
				   'wvalues'    : wvalues,               \
				   'wlong_name' : str(wAzShr.long_name), \
				   'wunits'     : str(wAzShr.units),     \
				   'wmissing'   : wAzShr.missing_value,  \
				   'n'          : values.size}

		# Read radial divergence
		Div  = id.variables['Divergence' ]
		wDiv = id.variables['wDivergence']

		# Create arrays to store binned values for radial divergence
		values    = np.zeros(x['n']*y['n']*z['n'])
		wvalues   = np.zeros(x['n']*y['n']*z['n'])
		values[:] = float('nan')

		# Add values to arrays
		values[index[:]]  =  (Div)[:]
		wvalues[index[:]] = (wDiv)[:]

		# Reshape arrays to 3-D GridRad domain
		values  =  values.reshape((z['n'], y['n'] ,x['n']))
		wvalues = wvalues.reshape((z['n'], y['n'] ,x['n']))

		Div = {'values'     : values,              \
				 'long_name'  : str(Div.long_name),  \
				 'units'      : str(Div.units),      \
				 'missing'    : float('nan'),        \
				 'wvalues'    : wvalues,             \
				 'wlong_name' : str(wDiv.long_name), \
				 'wunits'     : str(wDiv.units),     \
				 'wmissing'   : wDiv.missing_value,  \
				 'n'          : values.size}

	else:
		AzShr = -1
		Div   = -1


	if ('DifferentialReflectivity' in id.variables):
		# Read radial differential reflectivity
		Z_DR  = id.variables['DifferentialReflectivity' ]
		wZ_DR = id.variables['wDifferentialReflectivity']

		# Create arrays to store binned values for differential reflectivity
		values    = np.zeros(x['n']*y['n']*z['n'])
		wvalues   = np.zeros(x['n']*y['n']*z['n'])
		values[:] = float('nan')

		# Add values to arrays
		values[index[:]]  =  (Z_DR)[:]
		wvalues[index[:]] = (wZ_DR)[:]

		# Reshape arrays to 3-D GridRad domain
		values  =  values.reshape((z['n'], y['n'] ,x['n']))
		wvalues = wvalues.reshape((z['n'], y['n'] ,x['n']))

		Z_DR = {'values'     : values,               \
				  'long_name'  : str(Z_DR.long_name),  \
				  'units'      : str(Z_DR.units),      \
				  'missing'    : float('nan'),         \
				  'wvalues'    : wvalues,              \
				  'wlong_name' : str(wZ_DR.long_name), \
				  'wunits'     : str(wZ_DR.units),     \
				  'wmissing'   : wZ_DR.missing_value,  \
				  'n'          : values.size}

		# Read specific differential phase
		K_DP  = id.variables['DifferentialPhase' ]
		wK_DP = id.variables['wDifferentialPhase']

		# Create arrays to store binned values for specific differential phase
		values    = np.zeros(x['n']*y['n']*z['n'])
		wvalues   = np.zeros(x['n']*y['n']*z['n'])
		values[:] = float('nan')

		# Add values to arrays
		values[index[:]]  =  (K_DP)[:]
		wvalues[index[:]] = (wK_DP)[:]

		# Reshape arrays to 3-D GridRad domain
		values  =  values.reshape((z['n'], y['n'] ,x['n']))
		wvalues = wvalues.reshape((z['n'], y['n'] ,x['n']))

		K_DP = {'values'     : values,               \
				  'long_name'  : str(K_DP.long_name),  \
				  'units'      : str(K_DP.units),      \
				  'missing'    : float('nan'),         \
				  'wvalues'    : wvalues,              \
				  'wlong_name' : str(wK_DP.long_name), \
				  'wunits'     : str(wK_DP.units),     \
				  'wmissing'   : wK_DP.missing_value,  \
				  'n'          : values.size}

		# Read correlation coefficient
		r_HV  = id.variables['CorrelationCoefficient' ]
		wr_HV = id.variables['wCorrelationCoefficient']

		# Create arrays to store binned values for correlation coefficient
		values    = np.zeros(x['n']*y['n']*z['n'])
		wvalues   = np.zeros(x['n']*y['n']*z['n'])
		values[:] = float('nan')

		# Add values to arrays
		values[index[:]]  =  (r_HV)[:]
		wvalues[index[:]] = (wr_HV)[:]

		# Reshape arrays to 3-D GridRad domain
		values  =  values.reshape((z['n'], y['n'] ,x['n']))
		wvalues = wvalues.reshape((z['n'], y['n'] ,x['n']))

		r_HV = {'values'     : values,               \
				  'long_name'  : str(r_HV.long_name),  \
				  'units'      : str(r_HV.units),      \
				  'missing'    : float('nan'),         \
				  'wvalues'    : wvalues,              \
				  'wlong_name' : str(wr_HV.long_name), \
				  'wunits'     : str(wr_HV.units),     \
				  'wmissing'   : wr_HV.missing_value,  \
				  'n'          : values.size}

	else:
		Z_DR = -1
		K_DP = -1
		r_HV = -1

	# Close netCDF4 file
	id.close()

	# Return data dictionary
	return {'name'                    : 'GridRad analysis for ' + Analysis_time, \
			  'x'                       : x, \
			  'y'                       : y, \
			  'z'                       : z, \
			  'Z_H'                     : Z_H, \
			  'SW'                      : SW, \
			  'AzShr'                   : AzShr, \
			  'Div'                     : Div, \
			  'Z_DR'                    : Z_DR, \
			  'K_DP'                    : K_DP, \
			  'r_HV'                    : r_HV, \
			  'nobs'                    : nobs, \
			  'necho'                   : necho, \
			  'file'                    : infile, \
			  'sweeps_merged'           : sweeps_merged, \
			  'Analysis_time'           : Analysis_time, \
			  'Analysis_time_window'    : Analysis_time_window, \
			  'File_creation_date'      : File_creation_date, \
			  'Grid_scheme'             : Grid_scheme, \
			  'Algorithm_version'       : Algorithm_version, \
			  'Algorithm_description'   : Algorithm_description, \
			  'Authors'                 : Authors, \
			  'Project_sponsor'         : Project_sponsor, \
			  'Project_name'            : Project_name}

# GridRad filter routine
def filter(data0):


	#Extract year from GridRad analysis time string
	year = int((data0['Analysis_time'])[0:4])

	wthresh     = 1.5												# Set default bin weight threshold for filtering by year (dimensionless)
	freq_thresh = 0.6												# Set echo frequency threshold (dimensionless)
	Z_H_thresh  = 15.0											# Reflectivity threshold (dBZ)
	nobs_thresh = 2												# Number of observations threshold

	# Extract dimension sizes
	nx = (data0['x'])['n']
	ny = (data0['y'])['n']
	nz = (data0['z'])['n']

	echo_frequency = np.zeros((nz,ny,nx))					# Create array to compute frequency of radar obs in grid volume with echo

	ipos = np.where(data0['nobs'] > 0)						# Find bins with obs
	npos = len(ipos[0])											# Count number of bins with obs

	if (npos > 0):
		echo_frequency[ipos] = (data0['necho'])[ipos]/(data0['nobs'])[ipos]		# Compute echo frequency (number of scans with echo out of total number of scans)

	inan = np.where(np.isnan((data0['Z_H'])['values']))				# Find bins with NaNs
	nnan = len(inan[0])														# Count number of bins with NaNs

	if (nnan > 0): ((data0['Z_H'])['values'])[inan] = 0.0

	# Find observations with low weight
	ifilter = np.where((((data0['Z_H'])['wvalues'] < wthresh    ) & ((data0['Z_H'])['values'] < Z_H_thresh)) |
							  ((echo_frequency           < freq_thresh) &  (data0['nobs'] > nobs_thresh)))

	nfilter = len(ifilter[0])									# Count number of bins that need to be removed

	# Remove low confidence observations
	if (nfilter > 0):
		((data0['Z_H'])['values'])[ifilter] = float('nan')
		((data0['SW' ])['values'])[ifilter] = float('nan')

		if (type(data0['AzShr']) is dict):
			((data0['AzShr'])['values'])[ifilter] = float('nan')
			((data0['Div'  ])['values'])[ifilter] = float('nan')

		if (type(data0['Z_DR']) is dict):
			((data0['Z_DR'])['values'])[ifilter] = float('nan')
			((data0['K_DP'])['values'])[ifilter] = float('nan')
			((data0['r_HV'])['values'])[ifilter] = float('nan')

	# Replace NaNs that were previously removed
	if (nnan > 0): ((data0['Z_H'])['values'])[inan] = float('nan')

	# Return filtered data0
	return data0

def remove_clutter(data0, skip_weak_ll_echo=0):

	# Set fractional areal coverage threshold for speckle identification
	areal_coverage_thresh = 0.32

	# Extract dimension sizes
	nx = (data0['x'])['n']
	ny = (data0['y'])['n']
	nz = (data0['z'])['n']

	# Copy altitude array to 3 dimensions
	zzz = ((((data0['z'])['values']).reshape(nz,1,1)).repeat(ny, axis = 1)).repeat(nx, axis = 2)

	# Light pass at a correlation coefficient decluttering approach first
	if (type(data0['Z_DR']) is dict):
		ibad = np.where((((data0['Z_H'])['values'] < 40.0) & ((data0['r_HV'])['values'] < 0.9)) | \
                      (((data0['Z_H'])['values'] < 25.0) & ((data0['r_HV'])['values'] < 0.95) & (zzz >= 10.0)))
		nbad = len(ibad[0])

		if (nbad > 0):
			((data0['Z_H' ])['values'])[ibad] = float('nan')
			((data0['SW'  ])['values'])[ibad] = float('nan')
			((data0['Z_DR'])['values'])[ibad] = float('nan')
			((data0['K_DP'])['values'])[ibad] = float('nan')
			((data0['r_HV'])['values'])[ibad] = float('nan')

			if (type(data0['AzShr']) is dict):
				((data0['AzShr'])['values'])[ibad] = float('nan')
				((data0['Div'  ])['values'])[ibad] = float('nan')

	# First pass at removing speckles
	fin = np.isfinite((data0['Z_H'])['values'])

	# Compute fraction of neighboring points with echo
	cover = np.zeros((nz,ny,nx))
	for i in range(-2,3):
		for j in range(-2,3):
			cover += np.roll(np.roll(fin, i, axis=2), j, axis=1)
	cover = cover/25.0

	# Find bins with low nearby areal echo coverage (i.e., speckles) and remove (set to NaN).
	ibad = np.where(cover <= areal_coverage_thresh)
	nbad = len(ibad[0])
	if (nbad > 0):
		((data0['Z_H'])['values'])[ibad] = float('nan')
		((data0['SW' ])['values'])[ibad] = float('nan')

		if (type(data0['AzShr']) is dict):
			((data0['AzShr'])['values'])[ibad] = float('nan')
			((data0['Div'  ])['values'])[ibad] = float('nan')

		if (type(data0['Z_DR']) is dict):
			((data0['Z_DR'])['values'])[ibad] = float('nan')
			((data0['K_DP'])['values'])[ibad] = float('nan')
			((data0['r_HV'])['values'])[ibad] = float('nan')


	# Attempts to mitigate ground clutter and biological scatterers
	if (skip_weak_ll_echo == 0):
		# First check for weak, low-level echo
		inan = np.where(np.isnan((data0['Z_H'])['values']))				# Find bins with NaNs
		nnan = len(inan[0])															# Count number of bins with NaNs

		if (nnan > 0): ((data0['Z_H'])['values'])[inan] = 0.0

		# Find weak low-level echo and remove (set to NaN)
		ibad = np.where(((data0['Z_H'])['values'] < 10.0) & (zzz <= 4.0))
		nbad = len(ibad[0])
		if (nbad > 0):
			((data0['Z_H'])['values'])[ibad] = float('nan')
			((data0['SW' ])['values'])[ibad] = float('nan')

			if (type(data0['AzShr']) is dict):
				((data0['AzShr'])['values'])[ibad] = float('nan')
				((data0['Div'  ])['values'])[ibad] = float('nan')

			if (type(data0['Z_DR']) is dict):
				((data0['Z_DR'])['values'])[ibad] = float('nan')
				((data0['K_DP'])['values'])[ibad] = float('nan')
				((data0['r_HV'])['values'])[ibad] = float('nan')

		# Replace NaNs that were removed
		if (nnan > 0): ((data0['Z_H'])['values'])[inan] = float('nan')

		# Second check for weak, low-level echo
		inan = np.where(np.isnan((data0['Z_H'])['values']))				# Find bins with NaNs
		nnan = len(inan[0])															# Count number of bins with NaNs

		if (nnan > 0): ((data0['Z_H'])['values'])[inan] = 0.0

		refl_max   = np.nanmax( (data0['Z_H'])['values'],             axis=0)
		echo0_max  = np.nanmax(((data0['Z_H'])['values'] >  0.0)*zzz, axis=0)
		echo0_min  = np.nanmin(((data0['Z_H'])['values'] >  0.0)*zzz, axis=0)
		echo5_max  = np.nanmax(((data0['Z_H'])['values'] >  5.0)*zzz, axis=0)
		echo15_max = np.nanmax(((data0['Z_H'])['values'] > 15.0)*zzz, axis=0)

		# Replace NaNs that were removed
		if (nnan > 0): ((data0['Z_H'])['values'])[inan] = float('nan')

		# Find weak and/or shallow echo
		ibad = np.where(((refl_max   <  20.0) & (echo0_max  <= 4.0) & (echo0_min  <= 3.0)) | \
							 ((refl_max   <  10.0) & (echo0_max  <= 5.0) & (echo0_min  <= 3.0)) | \
							 ((echo5_max  <=  5.0) & (echo5_max  >  0.0) & (echo15_max <= 3.0)) | \
							 ((echo15_max <   2.0) & (echo15_max >  0.0)))
		nbad = len(ibad[0])
		if (nbad > 0):
			kbad = (np.zeros((nbad))).astype(int)
			for k in range(0,nz):
				((data0['Z_H'])['values'])[(k+kbad),ibad[0],ibad[1]] = float('nan')
				((data0['SW' ])['values'])[(k+kbad),ibad[0],ibad[1]] = float('nan')

				if (type(data0['AzShr']) is dict):
					((data0['AzShr'])['values'])[(k+kbad),ibad[0],ibad[1]] = float('nan')
					((data0['Div'  ])['values'])[(k+kbad),ibad[0],ibad[1]] = float('nan')

				if (type(data0['Z_DR']) is dict):
					((data0['Z_DR'])['values'])[(k+kbad),ibad[0],ibad[1]] = float('nan')
					((data0['K_DP'])['values'])[(k+kbad),ibad[0],ibad[1]] = float('nan')
					((data0['r_HV'])['values'])[(k+kbad),ibad[0],ibad[1]] = float('nan')


	# Find clutter below convective anvils
	k4km = ((np.where((data0['z'])['values'] >= 4.0))[0])[0]
	fin  = np.isfinite((data0['Z_H'])['values'])
	ibad = np.where((          fin[k4km         ,:,:]          == 0) & \
							 (np.sum(fin[k4km:(nz  -1),:,:], axis=0) >  0) & \
							 (np.sum(fin[   0:(k4km-1),:,:], axis=0) >  0))
	nbad = len(ibad[0])
	if (nbad > 0):
		kbad = (np.zeros((nbad))).astype(int)
		for k in range(0,k4km+1):
			((data0['Z_H'])['values'])[(k+kbad),ibad[0],ibad[1]] = float('nan')
			((data0['SW' ])['values'])[(k+kbad),ibad[0],ibad[1]] = float('nan')

			if (type(data0['AzShr']) is dict):
				((data0['AzShr'])['values'])[(k+kbad),ibad[0],ibad[1]] = float('nan')
				((data0['Div'  ])['values'])[(k+kbad),ibad[0],ibad[1]] = float('nan')

			if (type(data0['Z_DR']) is dict):
				((data0['Z_DR'])['values'])[(k+kbad),ibad[0],ibad[1]] = float('nan')
				((data0['K_DP'])['values'])[(k+kbad),ibad[0],ibad[1]] = float('nan')
				((data0['r_HV'])['values'])[(k+kbad),ibad[0],ibad[1]] = float('nan')

	# Second pass at removing speckles
	fin = np.isfinite((data0['Z_H'])['values'])

	# Compute fraction of neighboring points with echo
	cover = np.zeros((nz,ny,nx))
	for i in range(-2,3):
		for j in range(-2,3):
			cover += np.roll(np.roll(fin, i, axis=2), j, axis=1)
	cover = cover/25.0

	# Find bins with low nearby areal echo coverage (i.e., speckles) and remove (set to NaN).
	ibad = np.where(cover <= areal_coverage_thresh)
	nbad = len(ibad[0])
	if (nbad > 0):
		((data0['Z_H'])['values'])[ibad] = float('nan')
		((data0['SW' ])['values'])[ibad] = float('nan')

		if (type(data0['AzShr']) is dict):
			((data0['AzShr'])['values'])[ibad] = float('nan')
			((data0['Div'  ])['values'])[ibad] = float('nan')

		if (type(data0['Z_DR']) is dict):
			((data0['Z_DR'])['values'])[ibad] = float('nan')
			((data0['K_DP'])['values'])[ibad] = float('nan')
			((data0['r_HV'])['values'])[ibad] = float('nan')

	return data0

# GridRad sample image plotting routine
def plot_image(data,img_name):

	# Extract dimensions and their sizes
	x  = (data['x'])['values']
	y  = (data['y'])['values']
	nx = (data['x'])['n']
	ny = (data['y'])['n']

	r = [ 49, 30, 15,150, 78, 15,255,217,255,198,255,109,255,255,255]		# RGB color values
	g = [239,141, 56,220,186, 97,222,164,107, 59,  0,  0,  0,171,255]
	b = [237,192,151,150, 25,  3,  0,  0,  0,  0,  0,  0,255,255,255]

	refl_max = np.nanmax((data['Z_H'])['values'], axis=0)						# Column-maximum reflectivity

	img    = np.zeros((ny,nx,3))
	Z_H    = np.zeros((ny,nx))														# Create image for plotting
	img[:] = 200.0/255.0																	# Set default color to gray

	ifin = np.where(np.isfinite(refl_max))											# Find finite values
	nfin = len(ifin[0])																	# Count number of finite values

	for i in range(0,nfin):
		img[(ifin[0])[i],(ifin[1])[i],:] = (r[min(int(refl_max[(ifin[0])[i],(ifin[1])[i]]/5),14)]/255.0, \
														g[min(int(refl_max[(ifin[0])[i],(ifin[1])[i]]/5),14)]/255.0, \
														b[min(int(refl_max[(ifin[0])[i],(ifin[1])[i]]/5),14)]/255.0)

		Z_H[(ifin[0])[i],(ifin[1])[i]]=refl_max[(ifin[0])[i],(ifin[1])[i]]

	imgplot = plt.imshow(img[::-1,:,:], extent = [x[0],x[nx-1],y[0],y[ny-1]])
	plt.savefig(img_name+'.png')

#download data part
def check_file_status(filepath, filesize):
    sys.stdout.write('\r')
    sys.stdout.flush()
    size = int(os.stat(filepath).st_size)
    percent_complete = (size/filesize)*100
    sys.stdout.write('%.3f %s' % (percent_complete, '% Completed'))
    sys.stdout.flush()



# written functions by Serkan
#main functions

def download_data_4_2_Severe(start_year=2008,end_year=2021):
	# Authenticate
	dspath = 'https://rda.ucar.edu/data/ds841.6/'
	save_dpath="U:/skartalml/gridrad_4_2_raw_data/"
	# filelist=FileList4Download.filelist_GridRad_Severe_2010
	filelist=FileListUndownloaded.filelist_GridRad_Severe_2010

	data_dict_list=[]
	i=0
	for file in filelist:
		i=i+1
		if i%100==0:
			print(i)

		filepath = dspath + file
		file_base = os.path.basename(file)
		if os.path.exists(save_dpath + file_base):
			continue
		data_dict = {
			"save_dpath": save_dpath,
			"file": file,
			"filepath": filepath
		}
		data_dict_list.append(data_dict)

	p = Pool(6)
	p.map(download_data_4_2_Severe_parallel, data_dict_list)

def filter_dataset_location(longitude,latitude,data_threshold,size):
	raw_file_path = ext_storage+"/gridrad_4_2_raw_data/"
	plyfile_directory = os.listdir(raw_file_path)

	with open('./data/location_filtered_files.txt') as f:
		location_filtered_files = f.readlines()

	data_dict_list=[]
	counter=0
	for i, file in enumerate(plyfile_directory):

		if check_isFileProcessed(location_filtered_files,file):
			counter=counter+1
			continue
		if  os.path.exists("./data/data" + str(size) + "_max_thr_" + str(data_threshold) + "/"+file[:-2]+"npy"):
			continue
		data_dict = {
			"longitude": longitude,
			"latitude": latitude,
			"data_threshold": data_threshold,
			"size":size,
			"raw_file_path":raw_file_path,
			"file":file,
			"index":i
		}
		data_dict_list.append(data_dict)

	if len(data_dict_list)>0:
		print("İşenmiş dosya sayisi:"+str(counter))
		print("Kalan dosya sayisi:"+str(len(data_dict_list)))
		p = Pool(4)
		p.map(filter_dataset_location_parallel, data_dict_list)
	else:
		print("Data process completed")

def create_numpySubDataSet(source_size, target_size, longitude, latitude, source_pixel_thr, dbz_thr=30, target_pixel_thr=30,target_data_name="Rees"):
	range1 = 0.667
	source_min_longitude = longitude - range1 * (source_size / 64)
	source_max_longitude = longitude + range1 * (source_size / 64)
	source_min_latitude = latitude - range1 * (source_size / 64)
	source_max_latitude = latitude + range1 * (source_size / 64)

	target_min_longitude = longitude - range1 * (target_size / 64)
	target_max_longitude = longitude + range1 * (target_size / 64)
	target_min_latitude = latitude - range1 * (target_size / 64)
	target_max_latitude = latitude + range1 * (target_size / 64)

	start_x = round(((source_max_longitude - target_max_longitude) / (range1 * 2)) * 64)
	end_x = round(((source_max_longitude - target_min_longitude) / (range1 * 2)) * 64)
	start_y = round(((source_max_latitude - target_max_latitude) / (range1 * 2)) * 64)
	end_y = round(((source_max_latitude - target_min_latitude) / (range1 * 2)) * 64)

	if (end_x - start_x) > target_size:
		end_x = end_x - 1
		end_y = end_y - 1

	# max values
	target_folder_path="./data/data" + str(target_size) + "_pixel_thr_" + str(target_pixel_thr)+ "_dbz_thr_" + str(dbz_thr)+"_"+target_data_name
	if not os.path.exists(target_folder_path):
		os.makedirs(target_folder_path)

	npfile_directory = os.listdir("./data/data" + str(source_size) + "_max_thr_" + str(source_pixel_thr) + "/")
	for file in npfile_directory:
		data = np.load("./data/data" + str(source_size) + "_max_thr_" + str(source_pixel_thr) + "/" + file)
		new_data = data[start_x:end_x, start_y:end_y]

		if   len(new_data[new_data>=dbz_thr])< target_pixel_thr:
			continue

		np.save(target_folder_path+ "/" + file[:-3] + "npy",new_data)
	a = 3

def plot_filtered_np_folders(data_path):
	#max values
	if not os.path.exists(data_path+"_imgs/"):
		os.makedirs(data_path+"_imgs/")

	npfile_directory = os.listdir(data_path)
	for file in npfile_directory:
		data=np.load(data_path+"/"+file)
		plot_filtered_np_image(data,data_path+"_imgs/"+file[:-3])
	### lowest layer
	# if not os.path.exists("./data/data"+str(size)+"_lowest_layer_thr_"+str(data_threshold)+"_imgs/"):
	# 	os.makedirs("./data/data"+str(size)+"_lowest_layer_thr_"+str(data_threshold)+"_imgs/")
	#
	# npfile_directory = os.listdir("./data/data"+str(size)+"_lowest_layer_thr_"+str(data_threshold)+"/")
	# for file in npfile_directory:
	# 	data=np.load("./data/data"+str(size)+"_lowest_layer_thr_"+str(data_threshold)+"/"+file)
	# 	plot_filtered_np_image(data,"./data/data"+str(size)+"_lowest_layer_thr_"+str(data_threshold)+"_imgs/"+file[:-3])
	a=4

def create_TexasY_from_GridRad(grid_data_np_path,texas_data_path,nameof_location):
	# if not os.path.exists(data_path+"_imgs/"):
	# 	os.makedirs(data_path+"_imgs/")

	npfile_directory = os.listdir(grid_data_np_path)
	texas_cols=["File","Time","10_meterWindSpeedPeak"]
	texas_pd= pd.DataFrame(columns=texas_cols)
	for file in npfile_directory:
		df_texas_temp = pd.Series(index=texas_cols)

		year=int(file[15:19])
		month=int(file[19:21])
		day=int(file[21:23])
		hour=int(file[24:26])
		min=int(file[26:28])
		grid_time=datetime(year=year, month=month, day=day,hour=hour,minute=min)+ timedelta(hours=-6)

		data_texas = pd.read_csv(texas_data_path+"/"+nameof_location+str(grid_time.year-2000)+'{:02}'.format(int(grid_time.month))+".txt", sep=",")
		data_texas["TIME"] = pd.to_datetime(data_texas["TIME"])
		data_texas=data_texas[data_texas["TIME"] == (grid_time)]
		df_texas_temp["File"]=file[:-4]
		df_texas_temp["Time"] =	grid_time+ timedelta(hours=+6) #return back to erojinal time
		df_texas_temp["10_meterWindSpeedPeak"] =data_texas["10_meterWindSpeedPeak"].values[0]

		texas_pd = pd.concat([texas_pd, df_texas_temp.to_frame().T])


	pd.options.display.float_format = '${:,.2f}'.format
	file_base = os.path.basename(grid_data_np_path)
	texas_pd.to_csv("./data/texas_y/"+file_base+"_y.csv", float_format='%.2f', index=False,header=False)

def squeeze_Xvalues(datafolder_X):
	file_list=os.listdir(datafolder_X)

	full_data=None
	for file in file_list:
		temp_data=np.load(datafolder_X+"/"+file)
		if full_data is None:
			full_data=np.expand_dims(temp_data,axis=0)
		else:
			full_data=np.append(full_data,np.expand_dims(temp_data,axis=0) ,axis=0)

	np.save(datafolder_X,full_data)


#auxiliary functions -yardimci fonksiyonlar

def check_isFileProcessed(processed_list,file):
	for processed in processed_list:
		if processed.__contains__(file):
			return True

	return  False

def download_data_4_2_Severe_parallel(data_dict):
	url = 'https://rda.ucar.edu/cgi-bin/login'
	values = {'email': 'serkankartal01@gmail.com', 'passwd': "griddatapass1234", 'action': 'login'}

	ret = requests.post(url, data=values)
	if ret.status_code != 200:
		print('Bad Authentication')
		print(ret.text)
		exit(1)

	save_dpath=data_dict["save_dpath"]
	filepath = data_dict["filepath"]
	file = data_dict["file"]


	file_base = os.path.basename(file)
	# print('Downloading', file_base)
	try:
		req = requests.get(filepath, cookies=ret.cookies, allow_redirects=True, stream=True)
		filesize = int(req.headers['Content-length'])
		with open(save_dpath + file_base, 'wb') as outfile:
			chunk_size = 1048576
			for chunk in req.iter_content(chunk_size=chunk_size):
				outfile.write(chunk)
				if chunk_size < filesize:
					check_file_status(save_dpath + file_base, filesize)
		check_file_status(save_dpath + file_base, filesize)
	except:
		print("Dosya indirmede hata : "+filepath )

def filter_area(data, min_longitude,max_longitude,min_latitude, max_latitude):

	lat_index=(data["y"]["values"]>min_latitude) & (data["y"]["values"]<max_latitude)
	long_index=(data["x"]["values"]>min_longitude) & (data["x"]["values"]<max_longitude)
	# z_max_index=np.argmax(data["Z_H"]["values"], axis=0)
	# np.max(data["Z_H"]["values"], axis=0).reshape(1, data["Z_H"]["values"].shape[1], -1)
	# (data["Z_H"]["values"][:,lat_index,:])[:,:,long_index]
	# values  =  values.reshape((z['n'], y['n'] ,x['n'])) ((data["Z_H"]["values"][z_max_index,:,:])[:,lat_index,:])[:,:,long_index]

	x = {'values': data["x"]["values"][long_index], \
		 'long_name': data["x"]["long_name"], \
		 'units': data["x"]["units"], \
		 'delta': data["x"]["delta"], \
		 'n':  len(data["x"]["values"][long_index])
		 }

	# Read latitude dimension
	y = {'values': data["y"]["values"][lat_index], \
		 'long_name': data["y"]["long_name"], \
		'units': data["y"]["units"], \
		'delta': data["y"]["delta"], \
		'n': len(data["y"]["values"][lat_index])
		 }

	# Read altitude dimension


	Z_H = {'values'     : (data["Z_H"]["values"][:, lat_index, :])[:, :, long_index],              \
			 'long_name'  : data["Z_H"]["long_name"] ,  \
			 'units'      : data["Z_H"]["units"] ,      \
			 'missing'    : data["Z_H"]["missing"],        \
			 'wvalues'    : (data["Z_H"]["wvalues"] [:, lat_index, :])[:, :, long_index],             \
			 'wlong_name' : data["Z_H"]["wlong_name"] , \
			 'wunits'     : data["Z_H"]["wunits"]  ,     \
			 'wmissing'   : data["Z_H"]["wmissing"] ,  \
			 'n'		  : ((data["Z_H"]["values"][:, lat_index, :])[:, :, long_index]).size
		   }

	SW  = {'values'     : (data["SW"]["values"][:, lat_index, :])[:, :, long_index],             \
			 'long_name'  : data["SW"]["long_name"],  \
			 'units'      : data["SW"]["units"],      \
			 'missing'    : data["SW"]["missing"],       \
			 'wvalues'    : (data["SW"]["wvalues"][:, lat_index, :])[:, :, long_index] ,            \
			 'wlong_name' : data["SW"]["wlong_name"], \
			 'wunits'     : data["SW"]["wunits"] ,     \
			 'wmissing'   : data["SW"]["wmissing"],  \
			 'n'          : ((data["SW"]["values"][:, lat_index, :])[:, :, long_index]).size
		   }


	data_filtered={'name'                    : 'GridRad analysis for ' + data["Analysis_time"], \
			  'x'                       : x, \
			  'y'                       : y, \
			  'z'                       : data["z"], \
			  'Z_H'                     : Z_H,#np.max((data["Z_H"]["values"][:,lat_index,:])[:,:,long_index], axis=0)  , \
			  'SW'                      : SW, \
			  'AzShr'                   : data["AzShr"], \
			  'Div'                     : data["Div"], \
			  'Z_DR'                    : data["Z_DR"], \
			  'K_DP'                    : data["K_DP"], \
			  'r_HV'                    : data["r_HV"], \
			  'nobs'                    : (data["nobs"][:,lat_index,:])[:,:,long_index], \
			  'necho'                   : (data["necho"][:,lat_index,:])[:,:,long_index], \
			  'file'                    : data["file"], \
			  'sweeps_merged'           : data["sweeps_merged"], \
			  'Analysis_time'           : data["Analysis_time"], \
			  'Analysis_time_window'    : data["Analysis_time_window"], \
			  'File_creation_date'      : data["File_creation_date"], \
			  'Grid_scheme'             : data["Grid_scheme"], \
			  'Algorithm_version'       : data["Algorithm_version"], \
			  'Algorithm_description'   : data["Algorithm_description"], \
			  'Authors'                 : data["Authors"], \
			  'Project_sponsor'         : data["Project_sponsor"], \
			  'Project_name'            : data["Project_name"]
				   }
	return data_filtered

def filter_dataset_location_parallel(data_dict):
	range1=0.667
	try:
		data = read_file(data_dict["raw_file_path"] + data_dict["file"] )
	except:
		print("Hatali dosya :" + data_dict["file"] )
		return


	min_longitude = data_dict["longitude"] - range1 * (data_dict["size"] / 64)
	max_longitude=data_dict["longitude"]  + range1 * (data_dict["size"]  / 64)
	min_latitude=data_dict["latitude"]  - range1 * (data_dict["size"]  / 64)
	max_latitude= data_dict["latitude"] + range1 * (data_dict["size"]  / 64)

	print( str(data_dict["index"]) +" : "+data_dict["file"])
	if (min_longitude < data["x"]["values"].min()) or (max_longitude > data["x"]["values"].max()) or (min_latitude < data["y"]["values"].min()) or (max_latitude>data["y"]["values"].max()):
		return

	data = filter(data)
	data = remove_clutter(data, skip_weak_ll_echo=1)
	# plot_image(data, "./data/full_data_imgs/"+file[:-2]+"png")

	data_filtered = filter_area(data, min_longitude,max_longitude,	min_latitude, max_latitude)
	save_as_np_data(data_filtered, data_dict["size"] , data_dict["data_threshold"], data_dict["file"] ,min_longitude,max_longitude,	min_latitude, max_latitude)


def save_as_np_data(data,size,data_threshold,file,longitude_min,longitude_max,latitude_min,latitude_max):

	folder_max="./data/data"+str(size)+"_max_thr_"+str(data_threshold)+"/"
	folder_lowest_layer="./data/data"+str(size)+"_lowest_layer_thr_"+str(data_threshold)+"/"

	if not os.path.exists(folder_max):
		os.makedirs(folder_max)
	if not os.path.exists(folder_lowest_layer):
		os.makedirs(folder_lowest_layer)

	refl_max = np.nanmax((data['Z_H'])['values'], axis=0)  # Column-maximum reflectivity
	ifin = np.where(np.isfinite(refl_max))  # Find finite values
	nfin = len(ifin[0])  # Count number of finite values
	if nfin < data_threshold:
		return -1

	# Extract dimensions and their sizes
	x_start_long=data["x"]["values"].max()
	y_start_lat=data["y"]["values"].min()

	x_end_long=data["x"]["values"].min()
	y_end_lat=data["y"]["values"].max()

	x_start=int(((longitude_max - x_start_long) * size) / ((longitude_max - longitude_min)))
	x_end=(size)-int(((x_end_long - longitude_min) * size) / ((longitude_max - longitude_min)))
	y_start=int(((y_start_lat-latitude_min) * size) / ((latitude_max - latitude_min)))
	y_end=(size)-int(((latitude_max-y_end_lat) * size) / ((latitude_max - latitude_min)))

	np_data = np.zeros((size, size))  # Create image for plotting
	np_data[y_start:y_end,x_start:x_end]=refl_max.copy()
	np_data[np.isnan(np_data)] = 0
	np.save(folder_max+file[:-3],np_data)


	#save the lowest layer
	refl_lowest_layer = np.isfinite((data['Z_H'])['values'][0]) # Column-maximum reflectivity
	if refl_lowest_layer.sum() < data_threshold:
		return -1

	np_data = np.zeros((size, size))  # Create image for plotting
	np_data[y_start:y_end,x_start:x_end]=(data['Z_H'])['values'][0].copy()
	np_data[np.isnan(np_data)] = 0
	np.save(folder_lowest_layer+file[:-3],np_data)

def plot_filtered_np_image(data, img_path):
	# Extract dimensions and their sizes
	size = data.shape[0]

	r = [49, 30, 15, 150, 78, 15, 255, 217, 255, 198, 255, 109, 255, 255, 255]  # RGB color values
	g = [239, 141, 56, 220, 186, 97, 222, 164, 107, 59, 0, 0, 0, 171, 255]
	b = [237, 192, 151, 150, 25, 3, 0, 0, 0, 0, 0, 0, 255, 255, 255]

	# refl_max = np.nanmax((data['Z_H'])['values'], axis=0)  # Column-maximum reflectivity

	img = np.zeros((size, size, 3))
	Z_H = np.zeros((size, size))  # Create image for plotting
	img[:] = 200.0 / 255.0  # Set default color to gray

	for i in range(0, data.shape[0]):
		for j in range(0, data.shape[0]):
			if data[i][j]==0:
				continue
			img[i, j, :] = (r[min(int(data[i,j] / 5), 14)] / 255.0, g[min(int(data[i][j] / 5), 14)] / 255.0, b[min(int(data[i][j] / 5), 14)] / 255.0)

	imgplot = plt.imshow(img[::-1, :, :])
	plt.savefig(img_path + 'png')


""" Old functions """
def filter_dataset_old(longitude,latitude,data_threshold):
	raw_file_path="./data/raw_data/"
	plyfile_directory = os.listdir(raw_file_path)

	# data_rees=gridrad.filter_area(data,101,103.67,32,34.67) # for rees
	range1=0.667

	if not os.path.exists("./data/full_data_imgs/"):
		os.makedirs("./data/full_data_imgs/")

	for i, file in enumerate(plyfile_directory):
		data = read_file(raw_file_path+file)
		data = filter(data)
		data = remove_clutter(data, skip_weak_ll_echo=1)
		plot_image(data, "./data/full_data_imgs/"+file[:-2]+"png")

		data_32=filter_area(data,longitude-range1/2,longitude+range1/2,latitude-range1/2,latitude+range1/2) # for rees
		data_64=filter_area(data,longitude-range1,longitude+range1,latitude-range1,latitude+range1) # for rees
		data_128=filter_area(data,longitude-range1*2,longitude+range1*2,latitude-range1*2,latitude+range1*2) # for rees
		data_256=filter_area(data,longitude-range1*4,longitude+range1*4,latitude-range1*4,latitude+range1*4) # for rees
		data_512=filter_area(data,longitude-range1*8,longitude+range1*8,latitude-range1*8,latitude+range1*8) # for rees
		data_1024=filter_area(data,longitude-range1*16,longitude+range1*16,latitude-range1*16,latitude+range1*16) # for rees

		save_as_np_data_old(data_32,32,data_threshold,file,longitude-range1/2,longitude+range1/2,latitude-range1/2,latitude+range1/2)
		save_as_np_data_old(data_64,64,data_threshold,file,longitude-range1,longitude+range1,latitude-range1,latitude+range1)
		save_as_np_data_old(data_128,128,data_threshold,file,longitude-range1*2,longitude+range1*2,latitude-range1*2,latitude+range1*2)
		save_as_np_data_old(data_256,256,data_threshold,file,longitude-range1*4,longitude+range1*4,latitude-range1*4,latitude+range1*4)
		save_as_np_data_old(data_512,512,data_threshold,file,longitude-range1*8,longitude+range1*8,latitude-range1*8,latitude+range1*8)
		save_as_np_data_old(data_1024,1024,data_threshold,file,longitude-range1*16,longitude+range1*16,latitude-range1*16,latitude+range1*16)

def save_as_np_data_old(data,size,data_threshold,file,longitude_min,longitude_max,latitude_min,latitude_max):

	folder_max="./data/data"+str(size)+"_max_thr_"+str(data_threshold)+"/"
	folder_lowest_layer="./data/data"+str(size)+"_lowest_layer_thr_"+str(data_threshold)+"/"

	if not os.path.exists(folder_max):
		os.makedirs(folder_max)
	if not os.path.exists(folder_lowest_layer):
		os.makedirs(folder_lowest_layer)

	refl_max = np.nanmax((data['Z_H'])['values'], axis=0)  # Column-maximum reflectivity
	ifin = np.where(np.isfinite(refl_max))  # Find finite values
	nfin = len(ifin[0])  # Count number of finite values
	if nfin < data_threshold:
		return -1

	# Extract dimensions and their sizes
	x_start_long=data["x"]["values"].max()
	y_start_lat=data["y"]["values"].min()

	x_end_long=data["x"]["values"].min()
	y_end_lat=data["y"]["values"].max()

	x_start=int(((longitude_max - x_start_long) * size) / ((longitude_max - longitude_min)))
	x_end=(size)-int(((x_end_long - longitude_min) * size) / ((longitude_max - longitude_min)))
	y_start=int(((y_start_lat-latitude_min) * size) / ((latitude_max - latitude_min)))
	y_end=(size)-int(((latitude_max-y_end_lat) * size) / ((latitude_max - latitude_min)))

	np_data = np.zeros((size, size))  # Create image for plotting
	np_data[y_start:y_end,x_start:x_end]=refl_max.copy()
	np_data[np.isnan(np_data)] = 0
	np.save(folder_max+file[:-3]+"npy",np_data)


	#save the lowest layer
	refl_lowest_layer = np.isfinite((data['Z_H'])['values'][0]) # Column-maximum reflectivity
	if refl_lowest_layer.sum() < data_threshold:
		return -1

	np_data = np.zeros((size, size))  # Create image for plotting
	np_data[y_start:y_end,x_start:x_end]=(data['Z_H'])['values'][0].copy()
	np_data[np.isnan(np_data)] = 0
	np.save(folder_lowest_layer+file[:-3]+"npy",np_data)

def plot_filtered_np_folders_old(data_threshold):

	#max values
	if not os.path.exists("./data/data32_max_thr_"+str(data_threshold)+"_imgs/"):
		os.makedirs("./data/data32_max_thr_"+str(data_threshold)+"_imgs/")
	if not os.path.exists("./data/data64_max_thr_"+str(data_threshold)+"_imgs/"):
		os.makedirs("./data/data64_max_thr_"+str(data_threshold)+"_imgs/")
	if not os.path.exists("./data/data128_max_thr_"+str(data_threshold)+"_imgs/"):
		os.makedirs("./data/data128_max_thr_"+str(data_threshold)+"_imgs/")
	if not os.path.exists("./data/data256_max_thr_"+str(data_threshold)+"_imgs/"):
		os.makedirs("./data/data256_max_thr_"+str(data_threshold)+"_imgs/")
	if not os.path.exists("./data/data512_max_thr_"+str(data_threshold)+"_imgs/"):
		os.makedirs("./data/data512_max_thr_"+str(data_threshold)+"_imgs/")
	if not os.path.exists("./data/data1024_max_thr_"+str(data_threshold)+"_imgs/"):
		os.makedirs("./data/data1024_max_thr_"+str(data_threshold)+"_imgs/")

	npfile_directory = os.listdir("./data/data32_max_thr_"+str(data_threshold)+"/")
	for file in npfile_directory:
		data=np.load("./data/data32_max_thr_"+str(data_threshold)+"/"+file)
		plot_filtered_np_image(data,"./data/data32_max_thr_"+str(data_threshold)+"_imgs/"+file[:-3])

	npfile_directory = os.listdir("./data/data64_max_thr_"+str(data_threshold)+"/")
	for file in npfile_directory:
		data=np.load("./data/data64_max_thr_"+str(data_threshold)+"/"+file)
		plot_filtered_np_image(data,"./data/data64_max_thr_"+str(data_threshold)+"_imgs/"+file[:-3])

	npfile_directory = os.listdir("./data/data128_max_thr_"+str(data_threshold)+"/")
	for file in npfile_directory:
		data=np.load("./data/data128_max_thr_"+str(data_threshold)+"/"+file)
		plot_filtered_np_image(data,"./data/data128_max_thr_"+str(data_threshold)+"_imgs/"+file[:-3])

	npfile_directory = os.listdir("./data/data256_max_thr_"+str(data_threshold)+"/")
	for file in npfile_directory:
		data=np.load("./data/data256_max_thr_"+str(data_threshold)+"/"+file)
		plot_filtered_np_image(data,"./data/data256_max_thr_"+str(data_threshold)+"_imgs/"+file[:-3])

	npfile_directory = os.listdir("./data/data512_max_thr_"+str(data_threshold)+"/")
	for file in npfile_directory:
		data=np.load("./data/data512_max_thr_"+str(data_threshold)+"/"+file)
		plot_filtered_np_image(data,"./data/data512_max_thr_"+str(data_threshold)+"_imgs/"+file[:-3])

	npfile_directory = os.listdir("./data/data1024_max_thr_"+str(data_threshold)+"/")
	for file in npfile_directory:
		data=np.load("./data/data1024_max_thr_"+str(data_threshold)+"/"+file)
		plot_filtered_np_image(data,"./data/data1024_max_thr_"+str(data_threshold)+"_imgs/"+file[:-3])


	### lowest layer

	if not os.path.exists("./data/data32_lowest_layer_thr_"+str(data_threshold)+"_imgs/"):
		os.makedirs("./data/data32_lowest_layer_thr_"+str(data_threshold)+"_imgs/")
	if not os.path.exists("./data/data64_lowest_layer_thr_"+str(data_threshold)+"_imgs/"):
		os.makedirs("./data/data64_lowest_layer_thr_"+str(data_threshold)+"_imgs/")
	if not os.path.exists("./data/data128_lowest_layer_thr_"+str(data_threshold)+"_imgs/"):
		os.makedirs("./data/data128_lowest_layer_thr_"+str(data_threshold)+"_imgs/")
	if not os.path.exists("./data/data256_lowest_layer_thr_"+str(data_threshold)+"_imgs/"):
		os.makedirs("./data/data256_lowest_layer_thr_"+str(data_threshold)+"_imgs/")
	if not os.path.exists("./data/data512_lowest_layer_thr_"+str(data_threshold)+"_imgs/"):
		os.makedirs("./data/data512_lowest_layer_thr_"+str(data_threshold)+"_imgs/")
	if not os.path.exists("./data/data1024_lowest_layer_thr_"+str(data_threshold)+"_imgs/"):
		os.makedirs("./data/data1024_lowest_layer_thr_"+str(data_threshold)+"_imgs/")

	npfile_directory = os.listdir("./data/data32_lowest_layer_thr_"+str(data_threshold)+"/")
	for file in npfile_directory:
		data=np.load("./data/data32_lowest_layer_thr_"+str(data_threshold)+"/"+file)
		plot_filtered_np_image(data,"./data/data32_lowest_layer_thr_"+str(data_threshold)+"_imgs/"+file[:-3])

	npfile_directory = os.listdir("./data/data64_lowest_layer_thr_"+str(data_threshold)+"/")
	for file in npfile_directory:
		data=np.load("./data/data64_lowest_layer_thr_"+str(data_threshold)+"/"+file)
		plot_filtered_np_image(data,"./data/data64_lowest_layer_thr_"+str(data_threshold)+"_imgs/"+file[:-3])

	npfile_directory = os.listdir("./data/data128_lowest_layer_thr_"+str(data_threshold)+"/")
	for file in npfile_directory:
		data=np.load("./data/data128_lowest_layer_thr_"+str(data_threshold)+"/"+file)
		plot_filtered_np_image(data,"./data/data128_lowest_layer_thr_"+str(data_threshold)+"_imgs/"+file[:-3])

	npfile_directory = os.listdir("./data/data256_lowest_layer_thr_"+str(data_threshold)+"/")
	for file in npfile_directory:
		data=np.load("./data/data256_lowest_layer_thr_"+str(data_threshold)+"/"+file)
		plot_filtered_np_image(data,"./data/data256_lowest_layer_thr_"+str(data_threshold)+"_imgs/"+file[:-3])

	npfile_directory = os.listdir("./data/data512_lowest_layer_thr_"+str(data_threshold)+"/")
	for file in npfile_directory:
		data=np.load("./data/data512_lowest_layer_thr_"+str(data_threshold)+"/"+file)
		plot_filtered_np_image(data,"./data/data512_lowest_layer_thr_"+str(data_threshold)+"_imgs/"+file[:-3])

	npfile_directory = os.listdir("./data/data1024_lowest_layer_thr_"+str(data_threshold)+"/")
	for file in npfile_directory:
		data=np.load("./data/data1024_lowest_layer_thr_"+str(data_threshold)+"/"+file)
		plot_filtered_np_image(data,"./data/data1024_lowest_layer_thr_"+str(data_threshold)+"_imgs/"+file[:-3])


# not realted with severe. So not using now
def download_data_4_2_Hourly(start_year=2008,end_year=2021):
	url = 'https://rda.ucar.edu/cgi-bin/login'
	values = {'email': 'serkankartal01@gmail.com', 'passwd': "&X1sshe47r", 'action': 'login'}
	# Authenticate
	ret = requests.post(url, data=values)
	if ret.status_code != 200:
		print('Bad Authentication')
		print(ret.text)
		exit(1)
	dspath = 'https://rda.ucar.edu/data/OS/ds841.1/'

	startDate = datetime(start_year, month=4, day=1,hour=00)
	endDate =  datetime(end_year, month=9, day=1,hour=00)

	filelist=[]
	while startDate < endDate:
		filelist.append(str(startDate.year) + '{:02}'.format(int(startDate.month)) + "/nexrad_3d_v4_2_" + str(
			startDate.year) + '{:02}'.format(int(startDate.month)) + '{:02}'.format(
			int(startDate.day)) + "T" + '{:02}'.format(int(startDate.hour)) + "0000Z.nc")

		startDate=startDate+ timedelta(hours=1)

	for file in filelist:
		filename=dspath+file
		file_base = os.path.basename(file)
		print('Downloading',file_base)
		req = requests.get(filename, cookies = ret.cookies, allow_redirects=True, stream=True)
		filesize = int(req.headers['Content-length'])
		with open("./data/raw_data/"+file_base, 'wb') as outfile:
			chunk_size=1048576
			for chunk in req.iter_content(chunk_size=chunk_size):
				outfile.write(chunk)
				if chunk_size < filesize:
					check_file_status("./data/raw_data/"+file_base, filesize)
		check_file_status("./data/raw_data/"+file_base, filesize)
		print()


def create_numpySubDataSet_old_pixelanddbz_seperate(source_size, target_size, longitude, latitude, source_pixel_thr, dbz_thr=30, target_pixel_thr=30,target_data_name="Rees"):
	range1 = 0.667
	source_min_longitude = longitude - range1 * (source_size / 64)
	source_max_longitude = longitude + range1 * (source_size / 64)
	source_min_latitude = latitude - range1 * (source_size / 64)
	source_max_latitude = latitude + range1 * (source_size / 64)

	target_min_longitude = longitude - range1 * (target_size / 64)
	target_max_longitude = longitude + range1 * (target_size / 64)
	target_min_latitude = latitude - range1 * (target_size / 64)
	target_max_latitude = latitude + range1 * (target_size / 64)

	start_x = round(((source_max_longitude - target_max_longitude) / (range1 * 2)) * 64)
	end_x = round(((source_max_longitude - target_min_longitude) / (range1 * 2)) * 64)
	start_y = round(((source_max_latitude - target_max_latitude) / (range1 * 2)) * 64)
	end_y = round(((source_max_latitude - target_min_latitude) / (range1 * 2)) * 64)

	if (end_x - start_x) > target_size:
		end_x = end_x - 1
		end_y = end_y - 1

	# max values
	target_folder_path="./data/data" + str(target_size) + "_pixel_thr_" + str(target_pixel_thr)+ "_dbz_thr_" + str(dbz_thr)+"_"+target_data_name
	if not os.path.exists(target_folder_path):
		os.makedirs(target_folder_path)

	npfile_directory = os.listdir("./data/data" + str(source_size) + "_max_thr_" + str(source_pixel_thr) + "/")
	for file in npfile_directory:
		data = np.load("./data/data" + str(source_size) + "_max_thr_" + str(source_pixel_thr) + "/" + file)
		new_data = data[start_x:end_x, start_y:end_y]

		if np.count_nonzero(new_data) < target_pixel_thr:
			continue

		non_zero_idx=np.nonzero(new_data)
		max_dbz=new_data[non_zero_idx].max()
		if max_dbz< dbz_thr:
			continue

		np.save(target_folder_path+ "/" + file[:-3] + "npy",new_data)
	a = 3

#eksik veri bölgesi olada alıyorduk
def filter_dataset_location_old(longitude,latitude,data_threshold,size):
	raw_file_path = "O:/raw_data/"

	plyfile_directory = os.listdir(raw_file_path)
	data_dict_list=[]
	for i, file in enumerate(plyfile_directory):
		if  os.path.exists("./data/data" + str(size) + "_max_thr_" + str(data_threshold) + "/"+file[:-2]+"npy"):
			continue
		data_dict = {
			"longitude": longitude,
			"latitude": latitude,
			"data_threshold": data_threshold,
			"size":size,
			"raw_file_path":raw_file_path,
			"file":file
		}
		data_dict_list.append(data_dict)

	p = Pool(6)
	p.map(filter_dataset_location_parallel, data_dict_list)

