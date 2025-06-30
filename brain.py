import math
import h5py
import numpy as np
from PIL import Image
from neural_network import NeuralNetwork

image_size = 64
hidden_layer_size = 16
conv = False
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

# for epoch in range(1):
#     print("Epoch:", epoch)
#     for i in train:
#         file = f"data/brain_scans/{i}.mat"
#         nn.train(*get_scan(file))

# for i in test:
#     file = f"data/brain_scans/{i}.mat"
#     nn.infer(*get_scan(file))

indices = np.random.permutation(np.arange(1, 3065))
N = 3065
k = 5
fold_size = N // k
all_accuracies = []

for fold in range(k):
    nn = NeuralNetwork([result_size * result_size, hidden_layer_size, hidden_layer_size, classifications], 
                    doConvolution=conv, 
                    convLayers=conv_layers)
    
    accuracy = 0
    print(f"Fold {fold+1}/{k}")

    # Define validation fold
    val_start = fold * fold_size
    val_end = val_start + fold_size if fold != k - 1 else N

    val_indices = indices[val_start:val_end]
    train_indices = np.concatenate((indices[:val_start], indices[val_end:]))

    # Train your model
    for i in train_indices:
        file = f"data/brain_scans/{i}.mat"
        nn.train(*get_scan(file))

    for i in val_indices:
        file = f"data/brain_scans/{i}.mat"
        accuracy = nn.infer(*get_scan(file))
    
    all_accuracies.append(accuracy)

for i in range(len(all_accuracies)):
    print(f"Fold {i}: {all_accuracies[i]}")
          
print("Average Accuracy:", np.mean(all_accuracies))