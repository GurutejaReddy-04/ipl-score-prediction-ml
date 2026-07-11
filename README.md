# IPL Score Prediction (Academic Project)

This repository contains an academic college project developed for predicting the final score of Indian Premier League (IPL) cricket matches using machine learning and deep learning techniques. 

The project is hosted here for portfolio and archival purposes and represents the joint effort of our project team.

## Contributors
- Guruteja Reddy Nallachi
- Team Member 3
- Team Member 3

## Overview & Problem Statement
Predicting the score in a T20 cricket match is challenging due to the dynamic nature of the game. This project aims to build a robust predictive model that estimates the final inning score based on the current match state (runs, wickets, overs) and match context (venue, batting and bowling teams).

## Features
- **Deep Learning Model**: Utilizes a Keras/TensorFlow sequential neural network optimized with Huber loss to handle potential outliers in cricket scores.
- **Data Preprocessing**: Robust pipeline using Label Encoding for categorical variables and Min-Max Scaling for numerical variables.
- **Interactive UI**: Includes an IPyWidgets interface within the Jupyter Notebook (and accessible via script in supported environments) for dynamic predictions.

## Project Structure
```text
.
├── data/
│   └── ipl.csv                  # Dataset containing ball-by-ball IPL match data
├── docs/                        # Architecture notes, diagrams, and project documentation
├── notebooks/
│   └── IPL Score Prediction Project.ipynb  # Jupyter Notebook for EDA and Modeling
├── reports/
│   └── (Project reports, presentations, and documentation)
├── src/
│   └── ipl_score_prediction_project.py     # Core training and prediction script
├── .gitignore                   # Excluded files and directories
├── README.md                    # Project documentation
└── requirements.txt             # Python dependencies
```

## Technologies & Architecture
- **Data Processing**: `pandas`, `numpy`, `scikit-learn`
- **Model Training**: `keras`, `tensorflow`
- **Architecture**: A deep neural network with layers of size (512 -> 216 -> 1) utilizing ReLU activation for hidden layers and linear activation for the final regression output.
- **Evaluation**: Mean Absolute Error (MAE).

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/GurutejaReddy-04/ipl-score-prediction-ml.git
   cd ipl-score-prediction
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Dependency versions in `requirements.txt` are minimum recommended estimates based on the syntax used, as the original exact environment was not exported.*

## Usage

### 1. Jupyter Notebook
Launch Jupyter Notebook to interact with the IPyWidgets interface directly:
```bash
jupyter notebook "notebooks/IPL Score Prediction Project.ipynb"
```

### 2. Python Script
Run the source script directly to train the model and view test set metrics:
```bash
python src/ipl_score_prediction_project.py
```
*(Note: IPyWidgets require a Jupyter frontend to render interactively.)*

## Limitations & Future Improvements
- **Limitations**: The model relies entirely on historical data up to the date of the dataset creation and does not factor in real-time player form, weather, or pitch degradation.
- **Future Improvements**:
  - Incorporate individual player statistics and form.
  - Deploy the model as a web application using Flask, FastAPI, or Streamlit.
  - Explore alternative algorithms like XGBoost or LightGBM for comparison.

## License
*Licensing terms have not been finalized by the project contributors. All rights reserved until an open-source license is explicitly added.*
