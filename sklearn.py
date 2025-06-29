import os
import random
import h5py
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

# === Configuration ===
random.seed(42)
DATASET_DIR = "dataset"
NUM_SAMPLES = 3064
FILE_TEMPLATE = "{}.mat"  # Change to "{:04d}.mat" if filenames are zero-padded
LABEL_OFFSET = 1  # MATLAB labels may be 1-indexed
random_list = random.sample(range(1, 3065), 3064)

# === Helper: Load and Normalize Image ===

TARGET_SIZE = (512, 512)  # width x height

def load_image_and_label(file_path):
    with h5py.File(file_path, 'r') as f:
        image = np.array(f['cjdata']['image'], dtype=np.float32)
        label = int(np.array(f['cjdata']['label'])[0][0]) - LABEL_OFFSET

    # Normalize to [0, 255]
    min_val, max_val = image.min(), image.max()
    norm_image = ((image - min_val) / (max_val - min_val + 1e-8)) * 255.0
    norm_image = norm_image.astype(np.uint8)

    # Resize using PIL
    pil_image = Image.fromarray(norm_image)
    resized_image = pil_image.resize(TARGET_SIZE, Image.BILINEAR)

    # Flatten to 1D
    flat_image = np.array(resized_image).flatten()

    return flat_image, label

# === Load Full Dataset ===
X, y = [], []
for idx in random_list:
    file_path = os.path.join(DATASET_DIR, FILE_TEMPLATE.format(idx))
    if not os.path.exists(file_path):
        continue  # Skip missing files
    image, label = load_image_and_label(file_path)
    X.append(image)
    y.append(label)

X = np.array(X)
y = np.array(y)

# === Split Dataset ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Train Neural Network ===
model = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu',
                      solver='adam', max_iter=10, random_state=42, verbose=True)
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# === Output Results ===
print(f"\nTest Accuracy: {accuracy:.4f}\n")
print("Classification Report:")
print(report)
