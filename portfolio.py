import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from io import StringIO

# Fetch S&P 500 tickers from Wikipedia
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
table = soup.find('table', {'id': 'constituents'})

# Wrap HTML table in StringIO to avoid FutureWarning
tickers = pd.read_html(StringIO(str(table)))[0]['Symbol'].tolist()

# Clean tickers (replace '.' with '-' for yfinance compatibility)
tickers = [ticker.replace('.', '-') for ticker in tickers]

# Randomly select 50 tickers
np.random.seed(42)  # For reproducibility
selected_tickers = np.random.choice(tickers, size=50, replace=False).tolist()

# Download adjusted closing prices
data = yf.download(selected_tickers, start='2021-01-01', end='2025-04-21', group_by='ticker', auto_adjust=False)

# Extract adjusted close prices
adj_close = pd.DataFrame({ticker: data[ticker]['Adj Close'] if 'Adj Close' in data[ticker] else data[ticker]['Close'] for ticker in selected_tickers})

# Handle missing values (forward fill, then backward fill)
adj_close.fillna(method='ffill', inplace=True)
adj_close.fillna(method='bfill', inplace=True)

# Save to CSV
adj_close.to_csv('stock_prices.csv')

print("Dataset saved as stock_prices.csv")
print("Selected tickers:", selected_tickers)
print("Dataset shape:", adj_close.shape)
print(adj_close.head())


import numpy as np
import pandas as pd
import os

# Parameters
NUMBER_OF_TRADING_DAYS = 252  # Approximate number of trading days in a year

# Calculate return
def calculate_return(data):
    return np.log(data / data.shift(1))[1:]

# Show statistics
def show_statistics(returns):
    mean = returns.mean() * NUMBER_OF_TRADING_DAYS
    vars = returns.cov() * NUMBER_OF_TRADING_DAYS  # Fixed typo: cov) to cov()
    return np.array(mean), np.array(vars)

# Load the Dataset
csv_file = [file for file in os.listdir() if file.endswith('.csv')]
if not csv_file:
    raise FileNotFoundError("No CSV files found in the directory")
Dataset = pd.read_csv(csv_file[0], index_col='Date', parse_dates=True)

# Compute returns and statistics
log_daily_returns = calculate_return(Dataset)
mean, co_v = show_statistics(log_daily_returns)

# Output results
print("Log Mean Returns (annualized):")
print(mean)
print("\nCovariance Matrix (annualized):")
print(co_v)
print("\nShapes:")
print("Mean Returns Shape:", mean.shape)
print("Covariance Matrix Shape:", co_v.shape)
print("Is Covariance Symmetric?", np.allclose(co_v, co_v.T))



import numpy as np
import json
# Parameters
n = 50 # Number of assets
gamma = 0.5  # Risk-aversion parameter

# Initialize QUBO matrix
Q = np.zeros((n, n))

# Linear terms (diagonal)
for i in range(n):
    Q[i, i] = -1* mean[i] + gamma * co_v[i, i] 

# Quadratic terms (off-diagonal, symmetric)
for i in range(n):
    for j in range(i + 1, n):
        Q[i, j] = gamma * co_v[i, j] 
        Q[j, i] = Q[i, j]  # Ensure symmetry

# Verify symmetry
is_symmetric = np.allclose(Q, Q.T)
print("Is QUBO matrix symmetric?", is_symmetric)
print("Expected Returns:", mean)
print("Covariance Matrix:\n", co_v)
print("Symmetric QUBO Matrix:\n", Q)
qubo_dict = {(i, j): Q[i][j] for i in range(n) for j in range(n) if Q[i][j] != 0}
with open("qubo_portfolio.json", 'w') as f:
    json.dump({f"{i},{j}": val for (i, j), val in qubo_dict.items()}, f, indent=4)
print("QUBO matrix saved to qubo_portfolio.json")




import numpy as np
from dimod import SimulatedAnnealingSampler
import dimod
bqm = dimod.BinaryQuadraticModel.from_qubo(qubo_dict)

# Use SimulatedAnnealingSampler to solve the QUBO
sampler = SimulatedAnnealingSampler()
sampleset = sampler.sample(bqm, num_reads=1000)

# Get the best solution
best_sample = sampleset.first.sample
best_energy = sampleset.first.energy

# Convert sample to binary vector
solution_vector = np.array([best_sample[i] for i in range(n)])

# Verify energy by computing x^T Q x
computed_energy = solution_vector.T @ Q @ solution_vector

# Output results
print(f"Optimal Energy (from sampler): {best_energy}")
print(f"Computed Energy (x^T Q x): {computed_energy}")
print(f"Binary Vector (x): {solution_vector}")
