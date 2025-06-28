import h5py
import numpy as np
from PIL import Image

# Assume `image` is a NumPy 2D array, e.g. loaded from MATLAB struct
# (e.g., image = cjdata['image'][()])

def normalize_to_uint8(image):
    image = image.astype(np.float64)  
    min_val = np.min(image)
    max_val = np.max(image)
    
    norm_image = 255 * (image - min_val) / (max_val - min_val)
    return norm_image.astype(np.uint8)

data = h5py.File('dataset/2.mat', 'r')

objs = data['cjdata']

label = objs['label'][0][0]
PID = objs['PID'][0][0]
image = objs['image']
tumorBorder = objs['tumorBorder'][0][0]
tumorMask = objs['tumorMask']

image_uint8 = normalize_to_uint8(image)
image = Image.fromarray(image_uint8)

image.show()

data.close()