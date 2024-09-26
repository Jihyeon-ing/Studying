from astropy.io import fits
from sunpy.map import Map
from aiapy.calibrate import register, update_pointing, degradation, correct_degradation

# your fits file
file = 'aia.lev1_euv_12s.2020-01-01T000006Z.193.image_lev1.fits' 

fits_file = fits.open(file)
header = fits_file[1].header
fits_file.close()

# 0. quaility check: If quality != 0, the data is not fine to use
if header['QUAILTIY'] != 0:
  print("The data is not fine")

# 1. lev 1 -> 1.5: aiapy.calibrate
def prep(smap):
  # input: sunpy.map.Map  
  return register(update_pointing(smap))

M = prep(Map(file))

# 2. dividing into exposure time
meta = M.meta
exptime = meta['EXPTIME']
data = M.data / exptime

# 3. degradation correction
corrected_data = data / degradation(M.wavelength, M.date)

# 4. save
np.save("filename.npy", corrected_data)
