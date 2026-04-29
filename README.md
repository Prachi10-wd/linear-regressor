# 🏥 Medical Insurance Cost Predictor

A Linear Regression model that predicts medical insurance charges based on patient demographics and lifestyle features.

---

## 📊 Results

| Dataset | R² Score |
|--------|----------|
| Training | 0.83 |
| Testing | 0.86 |

> Testing R² slightly higher than training — the model generalizes well with no signs of overfitting.

---

## 📁 Dataset

**Kaggle — Medical Insurance Cost Dataset**

| Feature | Type | Description |
|--------|------|-------------|
| `age` | Numerical | Age of the patient |
| `sex` | Categorical | Male / Female |
| `bmi` | Numerical | Body Mass Index |
| `children` | Numerical | Number of dependents |
| `smoker` | Categorical | Smoker / Non-smoker |
| `region` | Categorical | US region (dropped) |
| `charges` | Target | Medical insurance cost |

---

## ⚙️ Workflow

### 1. Exploratory Data Analysis
- Scatter plot of BMI vs Charges colored by smoking status
- Confirmed strong linear relationship between smoking and higher charges

### 2. Feature Engineering
- Encoded `sex` → `female: 1, male: 0`
- Encoded `smoker` → `yes: 1, no: 0`
- Created interaction feature: `age_smoker = age × smoker`
- Created interaction feature: `bmi_smoker = bmi × smoker`
- Dropped `region` (low predictive value)

### 3. Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
```

### 4. Model Training
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
```

### 5. Evaluation
```python
from sklearn.metrics import r2_score
print(f"R² Training : {round(r2_train, 2)}")
print(f"R² Testing  : {round(r2_test, 2)}")
```

---

## 🛠️ Tech Stack

- **Python**
- **pandas** — data loading & feature engineering
- **seaborn** — exploratory visualization
- **scikit-learn** — model training, splitting & evaluation

---

## 💡 Key Learnings

- Interaction features (`age × smoker`, `bmi × smoker`) significantly improved model performance
- Dropping low-signal features like `region` reduced noise
- A testing R² of **0.86** confirms the model generalizes well to unseen data
- This project followed a previous attempt on a house price dataset (R² = 0.09) — the right dataset matters as much as the right model

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/yourusername/insurance-cost-predictor

# Install dependencies
pip install pandas seaborn scikit-learn

# Run the notebook
jupyter notebook insurance_predictor.ipynb
```

---

## 📌 Part of My ML Journey

I'm documenting every step of my machine learning journey publicly — the failures, the fixes, and the wins.  
Connect with me on [LinkedIn](https://www.linkedin.com/in/prachi-ai1011/) to follow along.
