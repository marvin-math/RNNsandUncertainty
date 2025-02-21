from sklearn.linear_model import LogisticRegression
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load datasets
df_hybrid = pd.read_csv('data/results_hybrid.csv')
df_thompson = pd.read_csv('data/results_thompson.csv')
df_ucb = pd.read_csv('data/results_ucb.csv')
df_rnn_human = pd.read_csv('data/simulation_trained_network_human.csv')
df_rnn_thompson = pd.read_csv('data/simulation_trained_network_thompson.csv')
df_rnn_ucb = pd.read_csv('data/simulation_trained_network_ucb.csv')
df_rnn_hybrid = pd.read_csv('data/simulation_trained_network_hybrid.csv')

def run_probit_regression(df, include_perseverance=True):
    """
    Fit a probit regression model.
    
    Parameters:
        df (DataFrame): The dataset.
        include_perseverance (bool): Whether to include the perseverance factor.
    
    Returns:
        df (DataFrame): The updated DataFrame with predictions.
        coeffs (tuple): The regression coefficients.
    """
    df = df.copy()
    
    # Prepare the predictor variables
    X = df[['V_t', 'RU']].copy()
    X['V/TU'] = X['V_t'] / df['TU']
    
    if include_perseverance:
        # Create the perseverance column (1 if previous action is the same)
        df['Perseverance'] = (df['Action'] == df['Action'].shift(1)).astype(int)
        X['Perseverance'] = df['Perseverance']
    
    y = df['Action']  # Binary outcome variable
    print("First 50 rows of predictors:")
    print(X.head(50))
    
    # Fit a probit regression using logistic regression on a normal CDF
    model = LogisticRegression()
    model.fit(X, 1 - y)  # Note: using 1 - y as in your original code
    
    # Extract coefficients based on whether perseverance was included
    if include_perseverance:
        w1, w2, w3, w4 = model.coef_[0]
        coeffs = (w1, w2, w3, w4)
        print(f'Coefficients (with perseverance): w1={w1}, w2={w2}, w3={w3}, w4={w4}')
    else:
        w1, w2, w3 = model.coef_[0]
        coeffs = (w1, w2, w3)
        print(f'Coefficients (without perseverance): w1={w1}, w2={w2}, w3={w3}')
    
    # Generate predicted probabilities using the normal CDF
    if include_perseverance:
        df['Predicted_Prob'] = norm.cdf(
            w1 * X['V_t'] + w2 * X['RU'] + w3 * X['V/TU'] + w4 * X['Perseverance']
        )
    else:
        df['Predicted_Prob'] = norm.cdf(
            w1 * X['V_t'] + w2 * X['RU'] + w3 * X['V/TU']
        )
    
    return df, coeffs

def plot_probit_regression(coeffs, df, title, include_perseverance=True, perseverance_val=0):
    """
    Plot the probit regression curve.
    
    Parameters:
        coeffs (tuple): Regression coefficients.
        df (DataFrame): The dataset.
        title (str): Plot title.
        include_perseverance (bool): Whether the perseverance factor is used.
        perseverance_val (numeric): Value to use for the perseverance variable during plotting.
    """
    # Fixed values for RU and TU
    RU_fixed = 0     # You can alternatively use: df['RU'].mean()
    TU_fixed = 2.5   # You can alternatively use: df['TU'].mean()
    
    print(title)
    print(f'RU_fixed: {RU_fixed}, TU_fixed: {TU_fixed}')
    
    # Generate a range of V_t values for predictions
    V_range = np.linspace(-100, 100, 1000)
    
    if include_perseverance:
        w1, w2, w3, w4 = coeffs
        predicted_probs = norm.cdf(
            w1 * V_range + w2 * RU_fixed + w3 * (V_range / TU_fixed) + w4 * perseverance_val
        )
    else:
        w1, w2, w3 = coeffs
        predicted_probs = norm.cdf(
            w1 * V_range + w2 * RU_fixed + w3 * (V_range / TU_fixed)
        )
    
    # Plot the regression curve
    plt.figure(figsize=(8, 6))
    plt.plot(V_range, predicted_probs, label="Probit Regression Curve", color='black')
    plt.xlabel('Expected Value Difference (V)')
    plt.ylabel('Choice Probability')
    plt.title(title)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_probit_regression_median(coeffs, df, title, UCB, include_perseverance=True, perseverance_val=0):
    """
    Plot probit regression curves for a median split of uncertainty (either RU or TU).
    
    Parameters:
        coeffs (tuple): Regression coefficients.
        df (DataFrame): The dataset.
        title (str): Plot title.
        UCB (bool): If True, split by RU; if False, split by TU.
        include_perseverance (bool): Whether the perseverance factor is used.
        perseverance_val (numeric): Value to use for the perseverance variable during plotting.
    """
    # Determine which variable to use for the median split
    if UCB:
        U = df['RU']
        U_median = U.median()
    else:
        U = df['TU']
        U_median = U.median()
    
    # Split the data
    low_SD = df[U <= U_median]
    high_SD = df[U > U_median]
    
    # Generate range of V_t values
    V_range = np.linspace(-100, 100, 1000)
    
    # Get average RU and TU for each group
    RU_low = low_SD['RU'].mean()
    RU_high = high_SD['RU'].mean()
    TU_low = low_SD['TU'].mean()
    TU_high = high_SD['TU'].mean()
    
    # Calculate predicted probabilities for each group
    if include_perseverance:
        w1, w2, w3, w4 = coeffs
        predicted_low = norm.cdf(
            w1 * V_range + w2 * RU_low + w3 * (V_range / TU_low) + w4 * perseverance_val
        )
        predicted_high = norm.cdf(
            w1 * V_range + w2 * RU_high + w3 * (V_range / TU_high) + w4 * perseverance_val
        )
    else:
        w1, w2, w3 = coeffs
        predicted_low = norm.cdf(
            w1 * V_range + w2 * RU_low + w3 * (V_range / TU_low)
        )
        predicted_high = norm.cdf(
            w1 * V_range + w2 * RU_high + w3 * (V_range / TU_high)
        )
    
    # Plot the median split regression curves
    plt.figure(figsize=(8, 6))
    plt.plot(V_range, predicted_low, label="Low SD", color='black', linewidth=2)
    plt.plot(V_range, predicted_high, label="High SD", color='gray', linewidth=2)
    plt.xlabel('Expected Value Difference (V)')
    plt.ylabel('Choice Probability')
    plt.title(title)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.show()

# Choose whether to include the perseverance factor
include_perseverance = True  # Change to False if you wish to exclude it
UCB = False

# Define your datasets for processing
datasets = {
    "Hybrid Model": df_hybrid,
    #"Thompson Model": df_thompson,
    # "UCB Model": df_ucb,
    # "RNN Model": df_rnn_human,
    #"RNN Thompson": df_rnn_thompson,
    # "RNN UCB": df_rnn_ucb,
    "RNN Hybrid": df_rnn_hybrid
}

# Process each dataset
for name, df in datasets.items():
    print(f"\n{name}")
    df_result, coefficients = run_probit_regression(df, include_perseverance=include_perseverance)
    plot_probit_regression(coefficients, df_result, f"Probit Regression - {name}",
                           include_perseverance=include_perseverance, perseverance_val=0)
    plot_probit_regression_median(coefficients, df_result, f"Probit Regression (Median Split) - {name}",
                                  UCB=UCB, include_perseverance=include_perseverance, perseverance_val=0)
