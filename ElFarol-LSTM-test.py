import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import random

# Parameters
N_AGENTS = 100
BAR_CAPACITY = 60
WEEKS = 100
SEQUENCE_LENGTH = 5

# Generate dummy attendance history
def generate_initial_attendance(weeks=SEQUENCE_LENGTH):
    return [random.randint(40, 70) for _ in range(weeks)]

# Build LSTM predictor
def build_lstm_model():
    model = Sequential([
        LSTM(10, input_shape=(SEQUENCE_LENGTH, 1), return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Simple Q-learning Agent
class Agent:
    def __init__(self):
        self.q_table = np.zeros((2,))  # 0: stay home, 1: go
        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 0.2
        self.lstm = build_lstm_model()
        self.past_attendance = generate_initial_attendance()
        # Train the LSTM model with the dummy data
        X = np.array([self.past_attendance[i:i+SEQUENCE_LENGTH] for i in range(len(self.past_attendance) - SEQUENCE_LENGTH)])
        y = np.array(self.past_attendance[SEQUENCE_LENGTH:])
        if len(X) > 0:
            self.lstm.fit(X.reshape(-1, SEQUENCE_LENGTH, 1), y, epochs=20, verbose=0)

    def predict_attendance(self):
        sequence = np.array(self.past_attendance[-SEQUENCE_LENGTH:]).reshape(1, SEQUENCE_LENGTH, 1)
        prediction = self.lstm.predict(sequence, verbose=0)[0][0]
        return prediction

    def decide(self, predicted_attendance):
        if random.random() < self.epsilon:
            return random.choice([0, 1])
        else:
            return np.argmax(self.q_table)

    def update_q(self, action, reward):
        self.q_table[action] = self.q_table[action] + self.alpha * (reward + self.gamma * np.max(self.q_table) - self.q_table[action])

    def update_attendance_history(self, value):
        self.past_attendance.append(value)
        if len(self.past_attendance) > SEQUENCE_LENGTH + 1:
            self.past_attendance.pop(0)

# Initialize agents
agents = [Agent() for _ in range(N_AGENTS)]

# Simulation
history = generate_initial_attendance(weeks=SEQUENCE_LENGTH)
for week in range(WEEKS):
    decisions = []
    predictions = []
    for agent in agents:
        pred = agent.predict_attendance()
        predictions.append(pred)
        action = agent.decide(pred)
        decisions.append(action)
    
    attendance = sum(decisions)
    history.append(attendance)
    history = history[-(SEQUENCE_LENGTH + 1):]

    for i, agent in enumerate(agents):
        reward = 1 if (decisions[i] == 1 and attendance <= BAR_CAPACITY) else -1 if decisions[i] == 1 else 0.5
        agent.update_q(decisions[i], reward)
        agent.update_attendance_history(attendance)
