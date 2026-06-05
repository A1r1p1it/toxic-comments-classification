


# Toxic Comment Classification

> If GitHub does not render the notebook properly, open the nbviewer link below for a clean static view.

>**Notebook View**: [nlp.ipynb on nbviewer](https://nbviewer.org/github/A1r1p1it/toxic-comments-classification/blob/main/nlp.ipynb)

## Live Demo

- **Streamlit UI**: [https://huggingface.co/spaces/Arpitkr/toxic-comment-ui](https://huggingface.co/spaces/Arpitkr/toxic-comment-ui)
- **FastAPI API Documentation (Swagger UI)**: [https://arpitkr-toxic-comment-api.hf.space/docs](https://arpitkr-toxic-comment-api.hf.space/docs)

The deployed application allows users to submit comments and receive toxicity predictions across all six categories with associated probability scores.

Multi-label toxic comment classification system built using classical NLP and machine learning techniques. The project includes model training, evaluation, explainability, error analysis, LLM-assisted evaluation, a FastAPI inference API, a Streamlit frontend, Docker containerization, and deployment on Hugging Face Spaces.

---00

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

## Model Interpretation

To improve interpretability, I analyzed the highest-weight TF-IDF features learned by the Logistic Regression classifier for each toxicity category.

Examples:

- `identity_hate` was associated with discriminatory language and targeted slurs.
- `threat` was strongly associated with violent intent expressions.
- `obscene` was driven by profanity-heavy token patterns.

This analysis helped explain model behavior and understand which language patterns influenced predictions across different toxicity categories.

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

## Error Analysis

Performed qualitative analysis of false positives and false negatives across rare toxic categories such as `threat` and `identity_hate`.

Observed challenges included:

- Context-dependent toxicity
- Sarcasm and implicit insults
- Label overlap between `toxic`, `insult`, and `obscene`
- Rare-category instability caused by extreme class imbalance

This analysis provided insight into model limitations beyond aggregate evaluation metrics and helped compare model robustness on difficult examples.

## LLM-Assisted Toxicity Analysis

To better understand ambiguous and context-dependent toxic comments, I used an LLM to review a sample of difficult cases identified during error analysis.

The objective was not to improve model performance but to compare classical TF-IDF predictions against contextual reasoning from LLaMA 3.3 70B.

The LLM was used to:

- Identify toxicity labels
- Distinguish explicit vs implicit toxicity
- Determine whether conversational context was required
- Explain classification outcomes
- Compare contextual reasoning with model predictions

This analysis highlighted limitations of lexical approaches when handling nuanced, context-dependent, or implicitly toxic language.

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

- Serialized trained model using `pickle`
- Built FastAPI inference service
- Implemented structured API schemas using Pydantic
- Created Streamlit frontend for interactive predictions
- Containerized application using Docker

---

## Deployment Architecture

The project was extended beyond model development into a deployable machine learning application.

### Components

- Trained Logistic Regression model serialized using `pickle`
- FastAPI backend for real-time inference
- Pydantic request/response validation
- Streamlit frontend for interactive predictions
- Docker containerization
- Hugging Face Spaces deployment

### Inference Workflow

User Input
→ Streamlit UI
→ FastAPI API
→ Serialized Model
→ Toxicity Predictions
→ Results Displayed to User

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
- LLaMA 3.3 70B (LLM-assisted analysis)
- Jupyter Notebook
- FastAPI
- Streamlit 
- Pydantic
- Docker
- Hugging Face Spaces
- Model Serialization (pickle)
- Requests

---

## Key Techniques Demonstrated

- Multi-label classification
- Precision–Recall evaluation
- PR-AUC computation
- Handling imbalanced data
- TF-IDF with bigrams
- Model comparison & benchmarking
- Error analysis
- Model interpretation
- LLM-assisted evaluation
- Pipeline architecture
- Model explainability
- Error analysis
- LLM-assisted evaluation
- Model serialization
- REST API development
- Frontend-backend integration
- Docker containerization
- ML model deployment

---

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/A1r1p1it/toxic-comments-classification.git
cd toxic-comments-classification
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Notebook

```bash
jupyter notebook nlp.ipynb
```

### 4. Run the FastAPI Inference Service

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

### 5. Run the Streamlit Frontend

```bash
streamlit run streamlit_app.py
```

### 6. Run with Docker

```bash
docker build -t toxic-comment-classifier .
docker run -p 7860:7860 toxic-comment-classifier
```

