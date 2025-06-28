import h5py
import numpy as np
from PIL import Image
from neural_network import NeuralNetwork

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
    image_resized = image.resize((64, 64), resample=Image.Resampling.LANCZOS)  # or Image.BILINEAR
    image_downsampled_array = np.array(image_resized).flatten()

    one_hot_idx = int(label) - 1

    data.close()
    return (image_downsampled_array, one_hot_idx)


nn = NeuralNetwork([64 * 64, 16, 16, 3])

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

