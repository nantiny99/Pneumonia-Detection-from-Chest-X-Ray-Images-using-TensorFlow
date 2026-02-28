# Pneumonia Detection from Chest X-Ray Images using TensorFlow
This project investigates deep learning approaches for detecting pneumonia from chest X-ray images using TensorFlow/Keras. The objective is not only to achieve high predictive performance, but to critically analyze model behavior under class imbalance and understand the trade-offs between sensitivity and specificity, which are especially important in medical imaging tasks.

The work progresses from a simple convolutional neural network trained from scratch to a transfer learning approach using a pretrained DenseNet121 architecture. Each model is evaluated using accuracy, confusion matrices, class-wise precision/recall, ROC–AUC, and decision threshold analysis.

## Dataset Link
The dataset retrieved from Kaggle:
https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images

## Dataset Description
The dataset consists of labeled chest X-ray images with two classes: **NORMAL** and **PNEUMONIA**.

- Training set: 5,232 images  
- Test set: 624 images  
- Class distribution in test set:
  - NORMAL: 234
  - PNEUMONIA: 390

This imbalance reflects real-world clinical data but introduces challenges during model training and evaluation.

## Baseline Model: CNN Trained from Scratch
The first model is a custom convolutional neural network composed of two convolutional layers followed by max pooling, a dense hidden layer, and a sigmoid output. Images are resized to 224×224 and normalized to the [0,1] range.

During training, the model reached extremely high training accuracy (>99%) within a few epochs. However, validation performance fluctuated significantly, suggesting overfitting.

### Test Results (Threshold = 0.5)

**Confusion Matrix**

|               | Predicted NORMAL | Predicted PNEUMONIA |
|---------------|------------------|---------------------|
| Actual NORMAL | 87               | 147                 |
| Actual PNEUMONIA | 1            | 389                 |

**Key Metrics**
- Accuracy: 76%
- NORMAL recall: 0.37
- PNEUMONIA recall: ~1.00
- NORMAL precision: 0.99
- PNEUMONIA precision: 0.73

**Interpretation**
The model rarely misses pneumonia cases, which results in a very high recall for the positive class. However, it misclassifies the majority of normal X-rays as pneumonia. This means the model is highly sensitive but extremely poor at ruling out disease, producing a large number of false positives.

This behavior is largely driven by:
- Class imbalance in the dataset
- Overfitting due to limited model depth
- The model learning coarse intensity and texture cues rather than robust anatomical features

### Threshold Adjustment (Threshold = 0.75)
To evaluate whether decision threshold tuning could improve specificity, predictions were re-evaluated using a threshold of 0.75.

**Confusion Matrix**

|               | Predicted NORMAL | Predicted PNEUMONIA |
|---------------|------------------|---------------------|
| Actual NORMAL | 96               | 138                 |
| Actual PNEUMONIA | 2            | 388                 |

**Key Metrics**
- Accuracy: 78%
- NORMAL recall: 0.41
- PNEUMONIA recall: 0.99

**Interpretation**
Increasing the threshold slightly improves the model’s ability to recognize normal cases, but the overall behavior remains similar. False positives remain high, indicating that threshold tuning alone cannot compensate for weak feature extraction. The model remains unsuitable for practical use despite high sensitivity.

## ROC–AUC Analysis (Baseline CNN)
The ROC curve for the baseline CNN yielded an AUC of approximately **0.88–0.90**, indicating that while the model has some discriminative ability, its decision boundary is poorly calibrated. This further supports the observation that accuracy alone is misleading in this setting.


## Transfer Learning Model: DenseNet121
To overcome the limitations of the baseline CNN, a transfer learning approach was adopted using DenseNet121 pretrained on ImageNet. The convolutional backbone was frozen, and a custom classification head with dropout was added.

This approach leverages rich, hierarchical feature representations learned from large-scale image data and is particularly effective for medical imaging tasks with limited labeled samples.

Training was performed for five epochs using a low learning rate (1e-4) to stabilize convergence.

## DenseNet121 Test Results (Threshold = 0.5)

**Confusion Matrix**

|               | Predicted NORMAL | Predicted PNEUMONIA |
|---------------|------------------|---------------------|
| Actual NORMAL | 168              | 66                  |
| Actual PNEUMONIA | 5            | 385                 |

**Key Metrics**
- Accuracy: 89%
- NORMAL recall: 0.72
- PNEUMONIA recall: 0.99
- NORMAL precision: 0.97
- PNEUMONIA precision: 0.85
- Validation AUC: 0.97

**Interpretation**
The DenseNet-based model achieves a much better balance between sensitivity and specificity. While maintaining very high pneumonia recall, it significantly reduces false positives compared to the baseline CNN. Normal-class recall nearly doubles, and overall accuracy improves by more than 10 percentage points.

The high AUC indicates strong class separability and a well-formed decision boundary. This model demonstrates behavior that is far more consistent with real-world diagnostic support systems.

## Comparative Summary
- The baseline CNN achieved deceptively high training accuracy but failed to generalize
- Threshold tuning improved results marginally but could not fix poor feature learning
- Transfer learning dramatically improved robustness, calibration, and interpretability
- Confusion matrix analysis revealed issues hidden by accuracy metrics alone

## Key Lessons Learned
- Accuracy is insufficient for evaluating medical classifiers
- Confusion matrices and recall metrics are critical in imbalanced datasets
- Transfer learning is essential when labeled medical data is limited
- High sensitivity must be balanced against excessive false positives

## Tools and Technologies
- Python
- TensorFlow / Keras
- Scikit-learn
- NumPy, Matplotlib

## Disclaimer
This project is for educational and research purposes only and must not be used for medical diagnosis or clinical decision-making.
