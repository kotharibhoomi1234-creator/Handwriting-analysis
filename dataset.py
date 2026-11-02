import cv2
import numpy as np
from graphology_db import GRAPHOLOGY_DATABASE

def generate_synthetic_dataset(num_samples_per_type=10):
    images = []
    labels = []
    
    for letter_idx, (char, traits) in enumerate(GRAPHOLOGY_DATABASE.items()):
        for trait_idx in range(len(traits)):
            for _ in range(num_samples_per_type):
                canvas = np.zeros((64, 64), dtype=np.float32)
                cv2.putText(canvas, char, (15 + np.random.randint(-3, 3), 45 + np.random.randint(-3, 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255), 2)
                canvas += np.random.normal(0, 15, canvas.shape)
                canvas = np.clip(canvas, 0, 255) / 255.0
                images.append(canvas[np.newaxis, :, :])
                labels.append(letter_idx)
                
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64)
