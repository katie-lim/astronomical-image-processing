from get_data import *
from show_image import *

cap = hist_togetcap(cleandata, 3700, 5)

image = deletebackground(cleandata, cap)


