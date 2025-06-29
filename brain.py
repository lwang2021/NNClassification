import math
import h5py
import numpy as np
from PIL import Image
from neural_network import NeuralNetwork

image_size = 128
hidden_layer_size = 16
conv = True
conv_layers = 1
classifications = 3
np.random.seed(1)

def normalize_to_uint8(image):
    image = image.astype(np.float64)  
    min_val = np.min(image)
    max_val = np.max(image)
    
    norm_image = 255 * (image - min_val) / (max_val - min_val)
    return norm_image.astype(np.uint8)

def get_scan(file):
    data = h5py.File(file, 'r')

    objs = data['cjdata']

    label = objs['label'][0][0]
    image = objs['image']

    image = Image.fromarray(normalize_to_uint8(image))
    image_resized = image.resize((image_size, image_size), resample=Image.Resampling.LANCZOS)  # or Image.BILINEAR
    image_array = np.array(image_resized)
    # image_downsampled_array = np.array(image_resized).flatten()

    one_hot_idx = int(label) - 1

    data.close()
    return (image_array, one_hot_idx)

result_size = image_size
if conv:
    result_size = int((image_size) / (2 ** conv_layers))

nn = NeuralNetwork([result_size * result_size, hidden_layer_size, hidden_layer_size, classifications], 
                    doConvolution=conv, 
                    convLayers=conv_layers)

permutation = np.random.permutation(np.arange(1, 3065))
train = permutation[:2500]
test = permutation[2500:]

for epoch in range(1):
    print("Epoch:", epoch)
    for i in train:
        file = f"data/brain_scans/{i}.mat"
        nn.train(*get_scan(file))

for i in test:
    file = f"data/brain_scans/{i}.mat"
    nn.infer(*get_scan(file))

