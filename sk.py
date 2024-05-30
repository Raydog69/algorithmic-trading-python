import pandas as pd
import numpy as np
import os
import cv2
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn import tree

image_folder = 'archive/Patterns'

# Function to load images and extract features
def load_images_and_extract_features(folder):
    data = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Load the image
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
            if img is not None:
                # Resize the image to a fixed size (e.g., 64x64)
                img_resized = cv2.resize(img, (64, 64))
                # Flatten the image to a 1D array
                img_flattened = img_resized.flatten()
                # Append the image data and label
                data.append(img_flattened)
                # Extract label from filename (assuming filename format: genre_XXXX.jpg)
                label = filename.split('_')[0]
                labels.append(label)
    return np.array(data), np.array(labels)

x, y = load_images_and_extract_features('archive/Patterns')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(x_train, y_train)
joblib.dump(model, 'music-recommender.joblib')

# model = joblib.load('music-recommender.joblib')

predictions = model.predict(x_test) 
score = accuracy_score(y_test, predictions) # [1, 1, 1, 0, 0, 1, 0] = [1, 1, 1, 0, 0, 1, 0]
print(score)

tree.export_graphviz(model, out_file='music-recommender.dot',
                     feature_names=[f'pixel{i}' for i in range(x.shape[1])], 
                     class_names=sorted(set(y)),
                     label='all',
                     filled=True)