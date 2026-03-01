# Toxic Comment Classification

Multi-label text classification system to identify toxic comments across 6 categories using classical NLP and machine learning models.

---

## Problem Statement

Online platforms need automated systems to detect toxic comments for moderation.  
This dataset presents a **multi-label classification problem** with severe class imbalance.

Only ~9.6% of comments contain toxic labels, and some categories (e.g., `threat`, `identity_hate`) represent less than 1% of the data.

This makes accuracy a misleading metric and requires precision–recall focused evaluation.

---

## Dataset

- **Source**: Jigsaw Toxic Comment Classification Challenge (Kaggle)
- **Size**: 159,571 comments
- **Labels (6)**:
  - toxic
  - severe_toxic
  - obscene
  - threat
  - insult
  - identity_hate
- **Type**: Multi-label (one comment can belong to multiple categories)
- **Challenge**: Extreme class imbalance

---

## Approach

### 1. Exploratory Data Analysis (EDA)
- Checked dataset structure using `.info()` and `.describe()`
- Analyzed label distribution per category
- Observed strong imbalance across all classes
- Dropped unnecessary `id` column

---

### 2. Text Vectorization

- **TF-IDF Vectorizer**
  - `ngram_range=(1,2)`
  - `max_features=50,000`
  - `sublinear_tf=True`
- No manual cleaning (no stemming/lemmatization)
- Relied on TF-IDF to handle token weighting

---

### 3. Multi-Label Strategy

Used `OneVsRestClassifier`, where each label is treated as an independent binary classification task.

---

## Models Compared

### 1. Logistic Regression (Best Model)

- `class_weight='balanced'`
- Linear classifier
- Stable for high-dimensional sparse TF-IDF features
- Strong performance on imbalanced data

Logistic Regression performs well with sparse TF-IDF features due to its linear decision boundary and stability in high-dimensional feature spaces.

---

### 2. Naive Bayes

- `MultinomialNB(alpha=0.1)`
- Classic NLP baseline
- Fast training
- More conservative predictions

---

### 3. XGBoost

- `scale_pos_weight=13`
- `eval_metric='aucpr'`
- Ensemble boosting method
- More flexible but heavier model

---

### 4. Dummy Baseline

- Predicts all comments as non-toxic
- Used to demonstrate why accuracy is misleading

---

## Evaluation Strategy

Since the dataset is highly imbalanced:

- Accuracy is misleading
- Used precision-recall focused metrics instead:
  - **Macro F1**
  - **Micro F1**
  - **Macro PR-AUC**

PR-AUC is the primary evaluation metric.

---

## Results

| Model | Macro F1 | Micro F1 | Macro PR-AUC | Key Insight |
|-------|----------|----------|--------------|-------------|
| Dummy (Baseline) | 0.00 | 0.00 | 0.037 | Predicts majority class only (fails to detect toxic labels) |
| Naive Bayes | 0.50 | 0.66 | 0.522 | More conservative, lower recall on rare labels |
| XGBoost | 0.56 | 0.65 | 0.602 | Balanced performance across labels |
| **Logistic Regression** | **0.57** | **0.69** | **0.646** | **Best overall – strongest PR-AUC & stable across labels** |

---

## Key Findings

### Best Model: Logistic Regression

- Highest Macro PR-AUC (0.646)
- Highest Micro F1 (0.69)
- Strong recall across most categories
- Stable performance on sparse high-dimensional features

---

### Why Accuracy is Misleading

A dummy classifier predicting all comments as non-toxic would achieve ~90% accuracy but detect 0% toxic comments.

This demonstrates why PR-AUC and F1 scores are more appropriate for imbalanced multi-label classification.

---

## Technical Implementation

- Used **Pipeline pattern** to combine TF-IDF + model
- Prevents data leakage
- Ensures reproducible workflow

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn  
  - TfidfVectorizer  
  - OneVsRestClassifier  
  - Pipeline  
  - LogisticRegression  
  - MultinomialNB  
- XGBoost
- Jupyter Notebook

---

## Key Techniques Demonstrated

- Multi-label classification
- Precision–Recall evaluation
- PR-AUC computation
- Handling imbalanced data
- TF-IDF with bigrams
- Model comparison & benchmarking
- Pipeline architecture

---

## How to Run

```bash
pip install numpy pandas scikit-learn xgboost
jupyter notebook nlp.ipynb
