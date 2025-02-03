# ANN-DL---RedCellClassifier

**Team Members:**  
- Alessandro Annechini
- Riccardo Fiorentini  
- Lorenzo Vignoli  

**Submission Date:** November 24, 2024  

## Project Overview

This project is the first assignment for the *Artificial Neural Networks and Deep Learning* (AN2DL) course. The goal is to develop a deep learning model capable of classifying blood cell images into eight categories based on their shape, contour, and color. The primary objective is to maximize the prediction accuracy:  

**Accuracy Formula:**  
\[ \text{accuracy} = \frac{\text{correctly predicted samples}}{\text{all samples}} \]

---

## Key Challenges

1. **Data Diversity:**  
   The dataset included images with a homogeneous color palette, limiting its generalizability to real-world scenarios with more diverse settings.

2. **Incorrect Samples:**  
   Some pre-labeled images in the dataset were incorrect and required removal to ensure effective training.

---

## Methodology

### Transfer Learning Model
We utilized DenseNet121 from TensorFlow's Keras library with pre-trained ImageNet weights. Layers included:
- Convolutional and ReLU layers
- Batch normalization for stabilization
- Dropout to reduce overfitting
- Dense layers with softmax activation for classification  

### Data Augmentation
To enhance generalization:
- **Transformations:** Translation, zoom, rotation, brightness, contrast, saturation, and hue adjustments.
- **Advanced Techniques:** MixUp and RandAugment for creating realistic variations.  

### Training Parameters
- **Optimizers:** Lion for transfer learning, Nadam for fine-tuning.
- **Learning Rate Adjustment:** ReduceLROnPlateau.  
- **Overfitting Prevention:** EarlyStopping.

### Fine Tuning
Implemented progressive unfreezing of DenseNet121 layers across four stages, culminating in full dataset augmentation with RandAugment.

### Test Time Augmentation (TTA)
Multiple predictions on modified input images were averaged to improve robustness and accuracy.

---

## Experiments

### Models Explored
- DenseNet121, InceptionResNetV2, NASNet, ResNet50, MobileNet, EfficientNetB0
- CustomModel: A simple architecture with four convolutional and three dense layers.  

### Data Preprocessing
- **Preprocessing Filters:** Gray scaling, background removal, Sobel filter for edge detection.
- Observed significant improvements in simple models like CustomModel.

---

## Results

- Best Model: **DenseNet121** with 92% accuracy on the test set.
- Lightweight Architecture: 7,037,504 weights, 26.85 MB storage.
- Incremental improvements were key, with test time augmentation boosting accuracy by 1-5%.

| Model Name         | Accuracy |
|--------------------|----------|
| DenseNet121        | 92%      |
| InceptionResNetV2  | 85%      |
| NASNet             | 71%      |
| ResNet50           | 63%      |
| CustomModel        | 57%      |

---

## Takeaways

- **Data Augmentation:** Crucial for improving generalization and achieving high accuracy.
- **Model Complexity:** Increasing model complexity yielded diminishing returns compared to data-centric techniques.
- **TTA:** Consistently enhanced final accuracy across all models.

---

## Individual Contributions

- **Alessandro Annechini:** Data preprocessing, augmentations, and test time augmentation.  
- **Riccardo Fiorentini:** DenseNet121 selection, transfer learning optimization, fine-tuning experiments.  
- **Lorenzo Vignoli:** Large transfer learning model experiments and augmentation layer design.  

---

## References

Please see the full report for a comprehensive list of references used in this project. The code in the repository is the final and most performant model we developed.
