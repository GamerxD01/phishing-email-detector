# Phishing Email Detector

A robust, end-to-end Machine Learning project to classify emails as **Safe** or **Phishing** using Natural Language Processing (NLP) and Classification algorithms. 

This repository implements a data science workflow—from Exploratory Data Analysis (EDA) and custom feature engineering to building text classification pipelines, visualizing performance, and running inference on custom sample emails.

---

## 📂 Codebase Structure & Files

The workspace contains the following files:

*   **[phishing_detector.ipynb](phishing_detector.ipynb)**: The main Jupyter Notebook containing the full data science pipeline, models, and execution outputs.
*   **[phishing_email.csv](phishing_email.csv)** *(11.8 MB)*: The dataset containing email texts and their corresponding labels (Safe or Phishing). A shortened, stratified version of 10,000 rows is included directly in this repository for easy execution and compatibility.
*   **[feature_comparison.png](feature_comparison.png)**: A bar chart visualizing the differences in average URL count and average keyword count between safe and phishing emails.
*   **[confusion_matrix.png](confusion_matrix.png)**: Heatmap representing the confusion matrix for the best-performing model (Random Forest).

---

## 🔍 Detailed Workflow & Analysis

Here is a step-by-step breakdown of the pipeline implemented in the notebook:

### 1. Library Imports & Environment Setup
The project leverages standard scientific and machine learning libraries:
*   **Data Handling**: `pandas`, `numpy`
*   **Visualization**: `matplotlib`, `seaborn`
*   **Modeling & Evaluation**: `scikit-learn` (specifically `train_test_split`, `TfidfVectorizer`, `MultinomialNB`, `RandomForestClassifier`, `Pipeline`, and metrics like `accuracy_score`, `confusion_matrix`, `classification_report`, `ConfusionMatrixDisplay`)

### 2. Exploratory Data Analysis (EDA)
*   **Dataset Dimensions**: The dataset contains **82,486 rows** and **2 columns**.
*   **Target Balance**: The labels are highly balanced, minimizing potential bias:
    *   **Phishing (Label 1)**: 42,891 emails (52.00%)
    *   **Safe (Label 0)**: 39,595 emails (48.00%)
*   **Data Quality**: Checked and confirmed that there are no missing (`NaN`) values or type mismatches.

### 3. Data Cleaning
*   Handled potential missing values.
*   Renamed columns to make them more descriptive:
    *   `text_combined` ➔ `email_text`
    *   `label` ➔ `is_phishing`
*   Cast the target variable `is_phishing` to integer format and reset index.

### 4. Custom Feature Engineering
Two new numeric features were engineered from the raw email text:
1.  **`url_count`**: Counts occurrence of web links starting with `http://` or `https://` using regular expressions. Phishing emails frequently contain multiple links.
2.  **`keyword_count`**: Scans the email body for common phishing red-flags/keywords:
    *   *Keywords checked*: `click here`, `verify`, `account`, `password`, `urgent`, `winner`, `bank`, `login`, `confirm`, `suspended`, `limited time`, `act now`, `free`, `prize`, `security alert`

### 5. Feature Visualization
A comparative analysis was plotted and saved to `feature_comparison.png`:
*   **URL Count Chart**: Shows the average number of URLs in Safe vs. Phishing emails.
*   **Keyword Count Chart**: Compares the average frequency of red-flag keywords in Safe vs. Phishing emails.
*   *Insight*: Phishing emails exhibit a significantly higher density of both URLs and urgency keywords.

### 6. Train-Test Split
*   Split the dataset into **80% Training Set** (65,988 samples) and **20% Testing Set** (16,498 samples).
*   Used **stratified splitting** (`stratify=y`) to ensure that both training and testing datasets maintain the exact 52.00% phishing ratio, preventing distribution shifts.

### 7. Pipeline Construction & Modeling
Two Scikit-Learn Pipelines were constructed to automate feature extraction (text vectorization) and model training.

#### Text Representation: TF-IDF
Both pipelines utilize `TfidfVectorizer` to convert text into numerical vectors:
*   `max_features=10000`: Considers the top 10,000 most significant vocabulary words.
*   `ngram_range=(1, 2)`: Uses both individual words (unigrams) and combinations of two adjacent words (bigrams) to capture context.
*   `stop_words='english'`: Filters out common English words (e.g., 'the', 'is', 'and') that carry little classification value.

#### Evaluated Models:
1.  **Pipeline 1: Multinomial Naive Bayes**
    *   *Training Accuracy*: **96.39%**
    *   *Test Accuracy*: **95.99%**
2.  **Pipeline 2: Random Forest Classifier** (100 estimators, all CPU cores utilized)
    *   *Training Accuracy*: **100.00%** (fully memorized training data)
    *   *Test Accuracy*: **98.65%**

---

## 📈 Model Performance & Evaluation

The **Random Forest Classifier** outperformed Naive Bayes and was selected as the final classifier.

### Confusion Matrix (`confusion_matrix.png`)
Out of 16,498 test samples, the Random Forest model made the following predictions:

*   **True Negatives (Safe correctly caught)**: **7,807**
*   **True Positives (Phishing correctly caught)**: **8,468**
*   **False Positives (Safe wrongly flagged as Phish)**: **112** (low false-positive rate, crucial for email filters)
*   **False Negatives (Phish missed by model)**: **111** (low leak rate)

### Classification Report
Detailed metric breakdown for the Random Forest model:

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Safe (0)** | 0.99 | 0.99 | 0.99 | 7,919 |
| **Phishing (1)** | 0.99 | 0.99 | 0.99 | 8,579 |
| **Accuracy** | | | **0.99** | **16,498** |

---

## 🔮 Running Custom Inference

The notebook demonstrates predicting raw, unseen emails with the trained model pipeline:

```python
predictions = best_model.predict([
    "URGENT: Your bank account has been suspended. Click here immediately to verify your password...",
    "Hi team, please find attached the notes from yesterday's product meeting."
])
# Predictions: [1, 0] -> ["🚨 PHISHING", "✅ SAFE"]
```

---

## 🚀 How to Run the Project

1.  **Clone the Repository**:
    ```bash
    git clone <your-repository-url>
    cd phishing-email-detector
    ```
2.  **Dataset Availability**:
    *   The repository already includes the shortened dataset `phishing_email.csv` (10,000 rows), so you can run the notebook directly without any extra downloads!
3.  **Install Dependencies**:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn notebook
    ```
4.  **Launch the Notebook**:
    ```bash
    jupyter notebook phishing_detector.ipynb
    ```
