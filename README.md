# Lung Cancer Detection: Survey Data and Scan Image Classification

A single Jupyter notebook that explores two separate lung-cancer datasets:

1. **Survey data** (tabular): predicting a `LUNG_CANCER` yes/no label from 15 symptom and lifestyle features with classical machine learning models.
2. **Scan images**: classifying images into three classes (benign, malignant, normal) with a small convolutional neural network, plus a local Gradio demo.

This is a learning and portfolio project. It is **not a medical device** and must not be used for diagnosis or clinical decisions.

## Results

### Survey data (`survey lung cancer.csv`, 309 rows)

Models trained with an 80/20 split (56 test rows: 12 "NO", 44 "YES") and evaluated with 5-fold cross-validation on the full set.

| Model | Test accuracy | 5-fold CV accuracy |
|---|---|---|
| Gaussian Naive Bayes | 0.91 | 0.902 |
| Random Forest | 0.88 | 0.895 |
| Multinomial Naive Bayes | 0.82 | 0.888 |
| Decision Tree | 0.80 | 0.822 |
| MLP Classifier | 0.79 | 0.869 |

The classes are imbalanced and the test set is small. Several models miss most "NO" cases (for example the MLP predicted "YES" for all 12 in one run), so accuracy alone overstates how well they work.

### Image classification (CNN)

- 1,097 grayscale images resized to 256 x 256, split 877 train / 220 validation (stratified).
- The training split was oversampled with SMOTE to balance the classes.
- A CNN (three convolution + pooling blocks followed by dense layers) was trained for 10 epochs with Adam.
- On the 220 validation images it reached **0.99 accuracy** (macro F1 0.98). Confusion matrix (rows = true class): benign 22/24 correct, malignant 113/113, normal 83/83.

Caveats: there is no separate test set (the validation split was used for monitoring during training), and the split is random per image, so results may be optimistic if one patient contributes several images. Treat 0.99 as a promising exploratory result, not evidence of clinical performance.

## What is in this repository

```
FINAL Project.ipynb      the full analysis: EDA, models, CNN, Gradio demo
survey lung cancer.csv   the survey dataset used in part 1
```

The scan images and the trained model file (`model1.keras`) are **not** included. To run the image section, put the images in three folders named `Bengin cases`, `Malignant cases` and `Normal cases` and set `directory` in the notebook to their parent folder.

## Running it

```bash
pip install numpy pandas seaborn matplotlib scikit-learn imbalanced-learn opencv-python imageio tensorflow gradio
jupyter notebook "FINAL Project.ipynb"
```

The survey section runs as-is. The image section needs the image folders described above.

## Possible next steps

- Hold out a real test set and report precision and recall per class.
- Split images by patient, if patient identifiers are available.
- Handle class imbalance in the survey models and report recall for the minority class.
- Add a short script for training and inference outside the notebook.
