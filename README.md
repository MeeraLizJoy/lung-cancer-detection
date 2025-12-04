### Lung Cancer Detection via Transfer Learning project:

## Project Pipeline

1. **Data Collection**
   - Download lung CT scan datasets from sources like Kaggle, TCIA.
   - Organize raw images and labels in the `data/raw/` folder.

2. **Data Preprocessing**
   - Load raw images, resize to standard input size (e.g., 224x224).
   - Normalize pixel values, apply data augmentation (flips, rotations).
   - Split data into training, validation, and test sets.
   - Save preprocessed images in `data/processed/` and splits in respective folders.

3. **Exploratory Data Analysis (EDA)**
   - Analyze dataset distribution, imbalance, visualize samples.
   - Conduct statistical analysis of image sizes, classes in notebooks.

4. **Model Selection and Setup**
   - Choose pretrained CNN architectures (e.g., ResNet50, DenseNet121).
   - Define transfer learning model, freeze base layers initially.
   - Add classification head for lung cancer categories.

5. **Training**
   - Set hyperparameters (learning rate, batch size, epochs) via `config.yaml`.
   - Train model on training set with validation monitoring.
   - Use callbacks like early stopping, model checkpointing.

6. **Evaluation**
   - Evaluate model on the test set using metrics: accuracy, precision, recall, F1-score, ROC-AUC.
   - Plot confusion matrix and ROC curves.
   - Perform error analysis on misclassified cases.

7. **Comparative Analysis**
   - Train multiple models (ResNet, DenseNet, EfficientNet).
   - Compare performance and resource usage.
   - Summarize findings and best model choice.

8. **Inference & Deployment**
   - Develop prediction script for single or batch inference.
   - (Optional) Build simple Flask or FastAPI app for demo or deployment.
   - Prepare a user guide or API documentation.

9. **Documentation and Reporting**
   - Write comprehensive README describing problem, approach, results.
   - Document code and scripts clearly.
   - Include model training curves, results summary, project insights.

This pipeline covers every essential stage ensuring a robust, reproducible, and presentable project suitable for a resume or portfolio.[1][2][3][4]







You're right! Here's an updated README.md with a detailed **ML Pipeline** section added:

***

# Lung Cancer Detection from CT Scans

## Project Overview

This project implements a deep learning solution for the classification of lung CT scans into three categories: **benign**, **malignant**, and **normal**. Using state-of-the-art convolutional neural networks, primarily **EfficientNet-B0**, the system achieves high accuracy in distinguishing between cancerous and non-cancerous cases.

***

## Features

- ✅ Data preprocessing and augmentation for robust training
- ✅ Hyperparameter tuning with grid search across multiple models
- ✅ 5-fold cross-validation for model validation
- ✅ Final model training on full dataset using optimal hyperparameters
- ✅ Interactive Streamlit web application for real-time inference
- ✅ Comprehensive evaluation with confusion matrix and metrics
- ✅ Batch prediction capabilities for test datasets

***

## Machine Learning Pipeline

### 1. **Data Preprocessing**
- **Input:** Raw CT scan images from Kaggle dataset
- **Process:**
  - Resize images to 224×224 pixels
  - Convert to RGB format (3 channels)
  - Normalize pixel values (mean=0.5, std=0.5)
  - Apply data augmentation (horizontal flip, rotation)
- **Output:** Preprocessed images saved in `data/preprocessed/`
- **Script:** `src/data_preprocessing.py`

### 2. **Model Selection & Comparison**
- **Architectures Tested:**
  - EfficientNet-B0
  - ResNet-18
  - DenseNet-121
  - VGG-16
- **Initial Training:** All models trained on preprocessed data
- **Best Model:** **EfficientNet-B0** selected based on validation accuracy
- **Scripts:** Model training and comparison scripts in `src/`

### 3. **Hyperparameter Tuning**
- **Method:** Grid search over hyperparameter combinations
- **Parameters Tuned:**
  - Learning Rate: [0.0001, 0.0005, 0.001]
  - Weight Decay: [0.0, 0.0001, 0.001]
  - Dropout: [0.0, 0.2, 0.5]
- **Total Configurations:** 18 combinations tested
- **Best Configuration:**
  - Learning Rate: **0.0005**
  - Weight Decay: **0.0001**
  - Dropout: **0.2**
- **Validation Accuracy:** 100% on held-out validation set
- **Script:** `src/hyperparameter_tuning.py`

### 4. **K-Fold Cross-Validation**
- **Folds:** 5-fold cross-validation
- **Purpose:** Evaluate model generalization and stability
- **Process:**
  - Split data into 5 folds
  - Train model on 4 folds, validate on 1 fold
  - Repeat for all fold combinations
- **Results:**
  - Mean Validation Accuracy: **~99.6%**
  - Consistent performance across all folds
  - Low standard deviation indicating stable model
- **Visualizations:**
  - Per-fold accuracy comparison
  - Box plots showing distribution
  - Training time analysis
  - Mean ± standard deviation plots
- **Scripts:** `src/kfold_validation.py`, `src/visualize_kfold.py`

### 5. **Final Model Training**
- **Dataset:** Full preprocessed dataset (1377 images)
- **Configuration:** Best hyperparameters from tuning
- **Training Details:**
  - Epochs: 15
  - Batch Size: 32
  - Optimizer: Adam
  - Loss Function: CrossEntropyLoss
- **Final Accuracy:** **99.42%** on full training data
- **Model Saved:** `models/efficientnet_b0_final.pth`
- **Training Time:** ~150 minutes on CPU
- **Script:** `src/train_final_model.py`

### 6. **Model Evaluation**
- **Validation Dataset:** 374 images with ground truth labels
- **Metrics:**
  - Overall Accuracy: **98.93%**
  - Precision: **98.96%**
  - Recall: **98.93%**
  - F1-Score: **98.93%**
- **Per-Class Performance:**
  - Benign: Precision=1.0, Recall=0.974, F1=0.987
  - Malignant: Precision=1.0, Recall=0.988, F1=0.994
  - Normal: Precision=0.969, Recall=1.0, F1=0.984
- **Outputs:**
  - Confusion matrix visualization
  - Per-class accuracy bar charts
  - Classification report
- **Script:** `src/evaluate.py`

### 7. **Inference & Deployment**
- **Command-Line Interface:**
  - Single image prediction: `python src/predict.py --image path/to/image.jpg`
  - Batch prediction: `python src/predict.py --folder data/test/`
- **Web Application:**
  - Built with Streamlit
  - Real-time image upload and prediction
  - Interactive probability visualization
  - Launch: `streamlit run app.py`
- **Scripts:** `src/predict.py`, `app.py`

***

## Pipeline Flowchart

```
Raw Data (Kaggle CT Scans)
         ↓
   Preprocessing
   (Resize, Normalize, Augment)
         ↓
   data/preprocessed/
         ↓
   Model Selection
   (4 architectures compared)
         ↓
   Hyperparameter Tuning
   (18 configurations tested)
         ↓
   K-Fold Cross-Validation
   (5 folds, validation)
         ↓
   Final Model Training
   (Best config on full data)
         ↓
   models/efficientnet_b0_final.pth
         ↓
   Evaluation & Inference
   (CLI + Streamlit Web App)
```

***

## Repository Structure

```
.
├── data/
│   ├── preprocessed/          # Cleaned and processed images
│   ├── valid/                 # Validation dataset with labels
│   └── test/                  # Test images (no labels)
├── models/
│   └── efficientnet_b0_final.pth  # Final trained model
├── src/
│   ├── data_preprocessing.py  # Data preprocessing script
│   ├── train.py               # Initial model training
│   ├── hyperparameter_tuning.py  # Grid search for best params
│   ├── kfold_validation.py    # K-fold cross-validation
│   ├── visualize_kfold.py     # Visualization of k-fold results
│   ├── train_final_model.py   # Final model training on full data
│   ├── predict.py             # Inference script
│   ├── evaluate.py            # Evaluation script
│   └── __init__.py
├── app.py                     # Streamlit web application
├── final_training_logs/       # Training logs
├── evaluation_results/        # Evaluation metrics and plots
├── tuning_checkpoints/        # Hyperparameter tuning checkpoints
├── kfold_results/             # K-fold validation results
└── README.md
```

***

## Setup & Installation

1. **Clone the repository**

2. **Create and activate virtual environment:**
   ```bash
   python -m venv image_env
   source image_env/bin/activate      # Mac/Linux
   image_env\Scripts\activate         # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install torch torchvision streamlit plotly scikit-learn matplotlib seaborn pillow
   ```

***

## Usage

### Training Pipeline

```bash
# 1. Preprocess data
python src/data_preprocessing.py

# 2. Hyperparameter tuning (optional - already done)
python src/hyperparameter_tuning.py

# 3. K-fold validation (optional - already done)
python src/kfold_validation.py

# 4. Train final model on full dataset
python src/train_final_model.py
```

### Inference

```bash
# Single image prediction
python src/predict.py --image path/to/image.jpg

# Batch prediction
python src/predict.py --folder data/test/
```

### Evaluation

```bash
python src/evaluate.py --data_dir data/valid/
```

### Web Application

```bash
streamlit run app.py
```

***

## Model Performance

| Metric | Value |
|--------|-------|
| Architecture | EfficientNet-B0 |
| Training Accuracy | 99.42% |
| Validation Accuracy | 98.93% |
| K-Fold Mean Accuracy | 99.6% |
| Parameters | 4,011,391 |
| Training Time | ~150 minutes (CPU) |

### Per-Class Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Benign | 1.000 | 0.974 | 0.987 | 78 |
| Malignant | 1.000 | 0.988 | 0.994 | 170 |
| Normal | 0.969 | 1.000 | 0.984 | 126 |

***

## Key Insights

1. **EfficientNet-B0** outperformed ResNet-18, DenseNet-121, and VGG-16
2. **Dropout regularization (0.2)** significantly improved generalization
3. **Low learning rate (0.0005)** with weight decay prevented overfitting
4. **K-fold validation** confirmed model stability across different data splits
5. **High precision on malignant class** (100%) critical for clinical applications

***

## Limitations & Future Work

- Model trained on limited dataset (1377 images)
- CPU-only training (GPU would accelerate significantly)
- No ensemble methods explored
- Potential for additional data augmentation techniques
- Integration with DICOM medical imaging standards

***

## Disclaimer

⚠️ This is a research and educational project. The model is **not approved for clinical diagnosis** and should not be used as a substitute for professional medical advice.

***

## Contact

For questions, collaboration, or feedback, please contact the project author.

***

