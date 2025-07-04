# Titanic Survival Analysis

A modern web application for exploring, analyzing, and predicting Titanic passenger survival using interactive data visualizations and machine learning.

![Titanic Header](assets/titanic-sinking.png)

---

## 🚢 Overview

**Titanic Survival Analysis** is an interactive Dash app that lets you:
- Explore the Titanic dataset with rich visualizations
- Filter and analyze survival patterns by class, gender, and more
- Predict survival probability for new passengers using a trained ML model
- View model insights, feature importance, and historical passenger stories
- Export filtered data and your prediction history

---

## ✨ Features

- **Interactive Visualizations:**
  - Age distributions, survival rates by class/gender, fare analysis, and more
  - Correlation matrix and statistical summaries
- **Advanced Filtering:**
  - Filter by passenger class, gender, and other demographics
- **Real-time Predictions:**
  - Input passenger details and get instant survival probability
  - Prediction history with confidence stats
- **Model Analysis:**
  - Feature importance and model performance metrics
- **Passenger Stories & Timeline:**
  - Learn about notable passengers and key Titanic events
- **Data Export:**
  - Download filtered datasets or your prediction history as CSV
- **Custom Styling:**
  - Responsive, modern UI with custom CSS and branding

---

## 🚀 Quickstart

1. **Clone the repository:**
   ```bash
   git clone https://github.com/hhnguyen-20/titanic-survival-analysis.git
   cd titanic-survival-analysis
   ```
2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the app:**
   ```bash
   python3 app.py
   ```
5. **Open your browser:**
   - Go to [http://127.0.0.1:8050](http://127.0.0.1:8050)

---

## 🧑‍💻 Usage Guide

### Data Exploration
- Use filters (class, gender, etc.) and click **Update** to refresh charts
- Explore histograms, bar plots, scatter plots, and tables

### Making Predictions
- Fill in the form in the **Predict Survival** section
- Click **Predict** to see survival probability and add to your prediction history

### Model & Advanced Analysis
- View feature importance and model metrics
- Explore correlation matrix and statistical summaries
- Read historical passenger stories and Titanic timeline

### Exporting Data
- Select data to export (filtered data or predictions)
- Click **Download Data** to save as CSV

---

## 📁 Project Structure

```
titanic-survival-analysis/
├── app.py                 # Main Dash application
├── model.pkl              # Trained ML model (scikit-learn pipeline)
├── test.csv               # Titanic test dataset (with features & predictions)
├── user_predictions.csv   # User prediction history (auto-generated)
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── assets/
    ├── style.css          # Custom styles
    ├── favicon.ico        # App favicon
    └── titanic-sinking.png  # Header image
```

---

## 📊 Data

- **test.csv:** Titanic passenger data with features like Gender, Age, Class, Fare, Embark town, etc., plus model predictions.
- **user_predictions.csv:** Your prediction history, including input features, predicted survival, probability, and confidence.

## 📚 Data Understanding

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1O8hWqvNMS2YY_Qe5KRHy8uR2P_X2m6Gc?usp=sharing)

---

## 🛠️ Technical Details

- **Framework:** Dash (Plotly)
- **ML Model:** Pre-trained scikit-learn pipeline (see `model.pkl`)
- **Core Libraries:**
  - dash, dash_daq, plotly, pandas, numpy, scikit-learn
- **Custom CSS:** Responsive, modern design in `assets/style.css`

---

## 📦 Requirements

Install all dependencies with:
```bash
pip install -r requirements.txt
```

- dash
- dash_daq
- plotly
- pandas
- numpy
- scikit-learn==1.6.1

---

## 🙏 Credits & License

- Titanic dataset: [Kaggle Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)
- App author: [hhnguyen-20](https://github.com/hhnguyen-20)
- Built with [Dash](https://dash.plotly.com/) and [scikit-learn](https://scikit-learn.org/)

MIT License. See [LICENSE](LICENSE) if available.
