from sklearn.linear_model import LogisticRegression
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df_hybrid = pd.read_csv('data/results_hybrid.csv')
#df_thompson = pd.read_csv('data/results_thompson.csv')
df_ucb = pd.read_csv('data/results_ucb.csv')
df_rnn_human = pd.read_csv('data/simulation_trained_network_human.csv')
df_rnn_thompson = pd.read_csv('data/simulation_trained_network_thompson.csv')
df_rnn_ucb = pd.read_csv('data/simulation_trained_network_ucb.csv')
df_rnn_hybrid = pd.read_csv('data/simulation_trained_network_hybrid.csv')




def run_probit_regression(df):
    # Flip the action variable if needed
    # TODO: implement conditional, depending on structure of dataset
    df = df.copy()
    # add perseverance column
    df['Perseverance'] = (df['Action'] == df['Action'].shift(1)).astype(int)

    # Prepare the data
    X = df[['V_t', 'RU', 'Perseverance']].copy()
    X['V/TU'] = X['V_t'] / df['TU']
    y = df['Action']  # Binary outcome variable
    print(df.head(50))

    # Fit a probit regression using logistic regression on a normal CDF
    model = LogisticRegression()
    model.fit(X, 1- y) 
    # Extract coefficients
    w1, w2, w3, w4 = model.coef_[0]
    print(f'Coefficients: w1={w1}, w2={w2}, w3={w3}, w4={w4}')

    # Generate predictions
    df['Predicted_Prob'] = norm.cdf(w1 * X['V_t'] + w2 * X['RU'] + w3 * X['V/TU'] + w4 * X['Perseverance']) #sigmoid

    # Return updated DataFrame and coefficients
    return df, (w1, w2, w3, w4)

def plot_probit_regression(w1, w2, w3, w4, df, title):
    # Fix RU and TU to their mean values
    RU_fixed = 0#df['RU'].mean()
    TU_fixed = 2.5#df['TU'].mean()
    perseverance = 0
    print(title)
    print(f'RU: {RU_fixed}, TU: {TU_fixed}')

    # Generate a range of V_t values
    V_range = np.linspace(-100, 100, 1000)

    # Predict probabilities for each V_t
    predicted_probs = norm.cdf(
        w1 * V_range + w2 * RU_fixed + w3 * (V_range / TU_fixed) + w4 * perseverance
    )

    # Plot the probit regression curve
    plt.figure(figsize=(8, 6))
    plt.plot(V_range, predicted_probs, label="Probit Regression Curve", color='black')
    plt.xlabel('Expected Value Difference (V)')
    plt.ylabel('Choice Probability')
    plt.title(title)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_probit_regression_median(w1, w2, w3, w4, df, title, UCB):
  if UCB:
    U_median = df['RU'].median()
    U = df['RU']
  else:
    U_median = df['TU'].median()
    U = df['TU']

  # Split the data into low and high SD groups
  low_SD = df[U <= U_median]
  high_SD = df[U > U_median]
  perseverance = 0

  # Generate a range of V_t values
  V_range = np.linspace(-100, 100, 1000)

  RU_low_SD = low_SD['RU'].mean()
  RU_high_SD = high_SD['RU'].mean()
  TU_low_SD = low_SD['TU'].mean()
  TU_high_SD = high_SD['TU'].mean()

  # Predict probabilities for low SD group
  predicted_probs_low_SD = norm.cdf(
      w1 * V_range + w2 * RU_low_SD + w3 * (V_range / TU_low_SD) + w4 * perseverance
  )

  # Predict probabilities for high SD group
  predicted_probs_high_SD = norm.cdf(
      w1 * V_range + w2 * RU_high_SD + w3 * (V_range / TU_high_SD) + w4 * perseverance
  )

  # Plot the probit regression curves for low and high SD
  plt.figure(figsize=(8, 6))
  plt.plot(V_range, predicted_probs_low_SD, label="Low SD", color='black', linewidth=2)
  plt.plot(V_range, predicted_probs_high_SD, label="High SD", color='gray', linewidth=2)
  plt.xlabel('Expected Value Difference (V)')
  plt.ylabel('Choice Probability')
  plt.title(title)
  plt.ylim(0, 1)
  plt.legend()
  plt.grid(True)
  plt.show()

  # Run probit regression and plot results for each dataset
datasets = {
    "Hybrid Model": df_hybrid,
    #"Thompson Model": df_thompson,
    #"UCB Model": df_ucb,
    #"RNN Model": df_rnn_human,
    #"RNN Thompson": df_rnn_thompson,
    #"RNN UCB": df_rnn_ucb,
    "RNN Hybrid": df_rnn_hybrid
}

for name, df in datasets.items():
    print(name)
    df_result, coefficients = run_probit_regression(df)
    plot_probit_regression(*coefficients, df_result, f"Probit Regression - {name}")
    plot_probit_regression_median(*coefficients, df_result, f"Probit Regression (Median Split) - {name}", UCB = True)