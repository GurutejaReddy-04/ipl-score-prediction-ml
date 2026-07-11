"""
IPL Score Prediction Project

This script builds a neural network regression model using Keras and TensorFlow
to predict the total score of an Indian Premier League (IPL) cricket match based
on current match state parameters such as venue, batting team, bowling team,
current wickets, current overs, and current runs.

It also provides an interactive IPyWidgets interface to test the model.
"""

import warnings
from pathlib import Path
from typing import Dict, Any

import ipywidgets as widgets
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython.display import clear_output, display
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# Define dynamic paths using pathlib to ensure portability
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "ipl.csv"

# Load dataset
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at expected path: {DATA_PATH}")

ipl = pd.read_csv(DATA_PATH)

# Select relevant features and target variable
features = ['venue', 'bat_team', 'bowl_team', 'wickets', 'overs', 'runs']
target = 'total'

df = ipl[features + [target]]
X = df[features].copy()
y = df[target].copy()

# Encode categorical variables using LabelEncoder
label_encoders: Dict[str, preprocessing.LabelEncoder] = {}
categorical_features = ['venue', 'bat_team', 'bowl_team']

for feature in categorical_features:
    le = preprocessing.LabelEncoder()
    label_encoders[feature] = le
    X.loc[:, feature] = le.fit_transform(X[feature])

# Split the dataset into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features using MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the neural network architecture
model = keras.Sequential([
    keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(216, activation='relu'),
    keras.layers.Dense(1, activation='linear')
])

# Compile the model with Huber loss for robustness against outliers
huber_loss = tf.keras.losses.Huber(delta=1.0)
model.compile(optimizer='adam', loss=huber_loss)

# Train the model
model.fit(
    X_train_scaled, 
    y_train, 
    epochs=50, 
    batch_size=64, 
    validation_data=(X_test_scaled, y_test),
    verbose=1
)

# Evaluate the model
predictions = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error on test set: {mae:.2f}")


def predict_score(
    venue: str, 
    batting_team: str, 
    bowling_team: str, 
    current_wickets: int, 
    current_overs: float, 
    current_runs: int
) -> None:
    """
    Predicts the total IPL score based on current match parameters.
    
    Args:
        venue (str): The venue of the match.
        batting_team (str): The name of the batting team.
        bowling_team (str): The name of the bowling team.
        current_wickets (int): Number of wickets fallen so far.
        current_overs (float): Number of overs bowled so far.
        current_runs (int): Number of runs scored so far.
        
    Returns:
        None. Prints the predicted score to the global output widget.
    """
    with output:
        clear_output()

        # Encode categorical features
        venue_encoded = label_encoders['venue'].transform([venue])[0]
        batting_team_encoded = label_encoders['bat_team'].transform([batting_team])[0]
        bowling_team_encoded = label_encoders['bowl_team'].transform([bowling_team])[0]

        # Create input array
        input_data = np.array([
            venue_encoded, 
            batting_team_encoded, 
            bowling_team_encoded,
            current_wickets, 
            current_overs, 
            current_runs
        ]).reshape(1, -1)

        # Scale the input data
        input_scaled = scaler.transform(input_data)

        # Validate inputs
        if not (0 <= current_wickets <= 9):
            print("Please enter a value for wickets between 0 and 9.")
            return
        if not (0.0 <= current_overs <= 20.0):
            print("Please enter a value for overs between 0 and 20.")
            return
        if current_runs < 0 or current_runs > 470:
            print("Please enter valid runs.")
            return

        # Predict the score
        predicted_score = model.predict(input_scaled)

        # Ensure predicted score is not less than current runs
        predicted_score_val = max(int(predicted_score[0][0]), current_runs)

        print(f"Predicted Score: {predicted_score_val}")


# IPyWidgets UI Configuration
venue_widget = widgets.Dropdown(options=ipl['venue'].unique().tolist(), description='Venue:')
batting_team_widget = widgets.Dropdown(options=ipl['bat_team'].unique().tolist(), description='Batting Team:')
bowling_team_widget = widgets.Dropdown(options=ipl['bowl_team'].unique().tolist(), description='Bowling Team:')
wickets_widget = widgets.IntText(value=0, min=0, max=9, description='Current Wickets:')
overs_widget = widgets.FloatText(value=0.0, min=0.0, max=20.0, description='Current Overs:')
runs_widget = widgets.IntText(value=0, description='Current Runs:')
predict_button = widgets.Button(description="Predict Score", layout=widgets.Layout(width='200px'))

# Adjust widget styles
side_bar_labels = [
    venue_widget, batting_team_widget, bowling_team_widget,
    wickets_widget, overs_widget, runs_widget
]
for label in side_bar_labels:
    label.style.description_width = '100px'

# Output widget for displaying prediction
output = widgets.Output()

def on_predict_button_clicked(b: Any) -> None:
    """Event handler for the prediction button."""
    predict_score(
        venue_widget.value, 
        batting_team_widget.value, 
        bowling_team_widget.value,
        wickets_widget.value, 
        overs_widget.value, 
        runs_widget.value
    )

predict_button.on_click(on_predict_button_clicked)

# Render widgets
display(
    venue_widget, 
    batting_team_widget, 
    bowling_team_widget, 
    overs_widget,
    runs_widget, 
    wickets_widget, 
    predict_button, 
    output
)