<div align="center">
 
# Supervised Learning
### ML Pipeline · Chest X-Ray Classification
 
[![Python](https://img.shields.io/badge/Python-3.11-3776ab?style=flat-square&logo=python&logoColor=white)]()
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-f7931e?style=flat-square&logo=scikitlearn&logoColor=white)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-ff6f00?style=flat-square&logo=tensorflow&logoColor=white)]()
[![Institution](https://img.shields.io/badge/UPMH-Maestría%20IA-6a0dad?style=flat-square)]()
 
</div>
 
## Overview
End-to-end supervised learning pipeline for multi-class classification of chest X-ray images into **COVID-19**, **Pneumonia**, and **Normal** categories. Eight algorithms evaluated across classical ML, ensemble, and deep learning paradigms.
 
## Models Evaluated
 
| Model | Accuracy | AUC-ROC | Type |
|:---|:---:|:---:|:---|
| YOLOv8-cls | 0.9767 | 0.9980 | Deep Learning |
| ResNet50 (Transfer) | 0.9746 | 0.9973 | Deep Learning |
| MLP | 0.9682 | 0.9930 | Neural Network |
| **SVM (RBF)** | **0.9640** | **0.9955** | **Classical — Best** |
| KNN (k=7) | 0.9301 | 0.9796 | Classical |
| Random Forest | 0.8475 | 0.9614 | Ensemble |
| Naive Bayes | 0.6377 | 0.8237 | Probabilistic |
 
## Dataset
- **3,141 images** · 224×224 RGB · Classes: COVID / Pneumonia / Normal
- Split: 70% train / 15% val / 15% test (stratified)
- Class imbalance handled via `class_weight='balanced'`
- PCA: 300 components · 95% explained variance
 
## Tech Stack
`Python 3.11` · `scikit-learn` · `TensorFlow 2.x` · `PyTorch` · `OpenCV` · `NumPy` · `Pandas` · `Matplotlib`
 
---
*Maestría en Inteligencia Artificial · Universidad Politécnica Metropolitana de Hidalgo*
