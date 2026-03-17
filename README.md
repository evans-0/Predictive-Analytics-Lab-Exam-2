# Predictive Analytics Lab Exam-2

Binary classification using Logistic Regression with Polynomial Features.

## Files

| File | Description |
|------|-------------|
| `lab_test_2.ipynb` | Main notebook with EDA, model building, and evaluation |
| `Lab_Exam_binary_classification_dataset.csv` | Dataset with Feature1, Feature2, and Target columns |

## Dataset

- **Features**: Feature1 (float64), Feature2 (int64)
- **Target**: Binary — `Yes` / `No`, mapped to `1` / `0`
- **Raw size**: 1020 rows, 3 columns
- **Missing values**: 20 null values in Target column — dropped using `dropna()`
- **Outliers**: Detected in Feature1 (max value: 10000) — removed using IQR method

## Steps

### 1. Exploratory Data Analysis (EDA)
- Dataset overview: `shape`, `head()`, `info()`, `describe()`
- Missing value detection and removal (`dropna()`)
- Label encoding: `Yes → 1`, `No → 0`
- Class distribution plot (`countplot`)
- Outlier removal on Feature1 using IQR
- Scatter plot of Feature1 vs Feature2 colored by Target class

### 2. Classification Model
- **Model**: Logistic Regression with Polynomial Features
- **Pipeline**:
  1. `StandardScaler` — normalizes Feature1 and Feature2
  2. `PolynomialFeatures(degree=2)` — generates non-linear features
  3. `LogisticRegression` — fits the classification boundary
- **Train/Test split**: 80/20 with `random_state=42`

### 3. Decision Boundary
- Mesh grid generated over the feature space
- Decision regions plotted using `contourf`
- Actual data points overlaid on the boundary plot

### 4. Model Evaluation
- Classification report (Precision, Recall, F1-score per class)
- Confusion matrix heatmap

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/evans-0/Predictive-Analytics-Lab-Exam-2.git
   cd Predictive-Analytics-Lab-Exam-2
   ```

2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```

3. Launch Jupyter:
   ```bash
   jupyter notebook
   ```

4. Open `lab_test_2.ipynb` and run all cells (`Kernel → Restart & Run All`)