# Titanic Survival Analysis

A comprehensive web application for analyzing the Titanic dataset, exploring passenger survival patterns, and predicting survival probabilities using machine learning.

## Features

### 🎯 Core Analysis
- **Interactive Data Visualization**: Multiple charts including age distributions, survival rates by class, probability histograms, and scatter plots
- **Advanced Filtering**: Filter data by passenger class, gender, and other demographics
- **Real-time Predictions**: Predict survival probability for new passengers with an interactive form

### 🚀 New Enhanced Features

#### 1. **Model Analysis Dashboard**
- **Feature Importance Analysis**: Visualize which factors most influence survival predictions
- **User Prediction History**: Track all predictions made by users with statistics
- **Interactive Model Insights**: Understand how the machine learning model makes decisions

#### 2. **Passenger Stories & Historical Context**
- **Notable Passengers**: Learn about famous passengers like "The Unsinkable Molly Brown"
- **Interactive Timeline**: Explore the Titanic's journey from departure to sinking
- **Historical Context**: Understand the events that shaped this tragic voyage

#### 3. **Data Export & Download**
- **Filtered Data Export**: Download filtered datasets as CSV files
- **Timestamped Downloads**: Automatic file naming with timestamps
- **Multiple Export Options**: Choose what data to export

#### 4. **Enhanced User Experience**
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Loading States**: Smooth loading animations for better UX
- **Error Handling**: Graceful handling of edge cases and errors

## Data Understanding

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1O8hWqvNMS2YY_Qe5KRHy8uR2P_X2m6Gc?usp=sharing)

## Setup

1. Clone the repository
```bash
git clone https://github.com/hhnguyen-20/titanic-survival-analysis.git
```

2. Change the working directory
```bash
cd titanic-survival-analysis
```

3. Create a virtual environment
```bash
python3 -m venv .venv
```

4. Activate the virtual environment
```bash
source .venv/bin/activate
```

5. Install the dependencies
```bash
pip install -r requirements.txt
```

6. Run the application
```bash
python3 app.py
```

7. Open your browser and navigate to `http://127.0.0.1:8050`

## Usage Guide

### Data Exploration
1. Use the filters to select specific passenger classes and genders
2. Click "Update" to apply filters and refresh all visualizations
3. Explore different charts to understand survival patterns

### Making Predictions
1. Scroll to the "Predict Survival" section
2. Fill in passenger details (class, gender, age, etc.)
3. Click "Predict" to get survival probability

### Advanced Analysis
1. **Model Analysis**: View feature importance and model performance
2. **Correlation Matrix**: Understand relationships between variables
3. **Statistical Summary**: Get detailed statistics for numerical features
4. **Passenger Stories**: Learn about notable passengers and historical events

### Data Export
1. Select the data you want to export using checkboxes
2. Apply any desired filters
3. Click "Download Data" to get a CSV file

## Technical Details

### Dependencies
- **Dash**: Web framework for building analytical web applications
- **Plotly**: Interactive plotting library
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning library
- **NumPy**: Numerical computing

### Model Information
- The application uses a pre-trained machine learning pipeline
- Model file: `model.pkl`
- Supports feature importance analysis (for compatible model types)
- Includes model evaluation metrics when training data is available

### File Structure
```
titanic-survival-analysis/
├── app.py              # Main Dash application
├── model.pkl           # Trained machine learning model
├── test.csv            # Test dataset
├── train.csv           # Training dataset (for model evaluation)
├── requirements.txt    # Python dependencies
├── README.md          # Project documentation
└── assets/
    ├── style.css      # Custom styling
    ├── favicon.ico    # Browser icon
    └── titanic-sinking.png  # Header image
```

## Contributing

Feel free to contribute to this project by:
- Adding new visualizations
- Improving the machine learning model
- Enhancing the user interface
- Adding more historical context and passenger stories

## License

This project is open source and available under the MIT License.
