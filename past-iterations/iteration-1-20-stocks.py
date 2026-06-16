#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 19:30:58 2026

@author: kshitizbhandari


Mean-Variance Portfolio Optimization (1st implementation - Idea testing)

Goal:
    - Simple long-only portoflio optimization
    - estimate expected returns and covariance from past data
    - find weights that maximize Sharpe ratio
    - Test if this constructed portfolio still outperforms the universe next year

Notes to do:
    - Using 20 equities for now, will expand later to S&P500
    - Haven't added risk-free rate (download from FRED later)    

"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

pd.set_option('display.float_format', '{:.4f}'.format)

###########
# 1. Define investable universe
###########

## To do: download the S&P 500 list once this works
# for now just putting 20 stocks as proxy

tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",
           "JPM", "JNJ", "XOM", "PG", "UNH", "HD", "MA", "V",
           "DIS", "ADBE", "NFLX", "KO", "PEP", "INTC"]



#########
# 2. Time period and portfolio constraints
#########

# testing for one year: 2023 train and optimize, 2024 test 

START_DATE = "2023-01-01"
END_DATE = "2025-01-01"

TRAIN_YEAR = "2023"
TEST_YEAR = "2024"

# each weight represents allocation to one stock

MAX_WEIGHT = 0.10               # max weight per stock of 10% of portfolio
MIN_WEIGHT = 0.0                # setting 0 for long-only implementation

TRADING_DAYS = 252              # starndard assumption


########
# 3. Download data from yfinance and clculate returns
########

prices = yf.download(tickers, start = START_DATE, end = END_DATE, auto_adjust = True)
# taking the adjusted close as a close approximator for total-return-index
prices = prices["Close"]


# in case there is any missing values, forward filling data
prices = prices.ffill().dropna()

# calculating daily percentage returns from adjusted close prices
returns = prices.pct_change().dropna()

# Splitting training window and test window
# for now, using 2023 as training sample and 2024 as test sample
train_returns = returns.loc[TRAIN_YEAR]
test_returns = returns.loc[TEST_YEAR]



###########
# 4. Sharpe ratio calculation
#######


## To do - import risk free rate of 1Y treasure bill on first trading day

def portfolio_performance(weights, mean_returns, cov_matrix):
    '''
    Computes portfolio Sharpe ratio
    
    Parameters:
        - weights: allocation vector (sum = 1)
        - mean_returns: expected returns of assets
        - cov_matrix: covariance matrix of asset returns
        
    Returns:
        - Sharpe ratio
    '''
    # expected porfolio return: w^T * mu
    ret = np.dot(weights, mean_returns)
    # portfolio volatility: sqrt(w^T Σ w)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # for now assumes risk-free rate of zero percent 
    sharpe = ret / vol
    
    return sharpe

def negative_sharpe(weights, mean_returns, cov_matrix):
    '''
    Optimization helper function.
    SciPy -> minimize
    so to maximize Sharpe ratio, we can minimize the -(Sharpe ratio)
    '''
    return -portfolio_performance(weights, mean_returns, cov_matrix)



########
#5. Optimization
########

# number of assets in universe    
n = len(tickers)

# constraint: weights sum to 1 
# i.e., fully invested portfolio - only equities from investable universe

def weight_sum_constraint(w):
    return np.sum(w) - 1

constraints = ({
    'type': 'eq',
    'fun': weight_sum_constraint
    })
   
# bounds: enforce maximum weight per stock constraint
# each stock constrained individually
bounds = [(MIN_WEIGHT, MAX_WEIGHT) for _ in range(n)]


## Initial guess for optimizer -> assume equal weight portfolio
init_guess = np.ones(n) / n      # a vector of 1/n weight for each stock


# Expected return and covariance matrix (calculated from training data)
mean_returns = train_returns.mean()
cov_matrix = train_returns.cov()


# implement optimization

result = minimize(
    negative_sharpe,            # objective function
    init_guess,                 # starting weights
    args = (mean_returns, cov_matrix),
    method = "SLSQP",            # constrained optimization method
    bounds = bounds,            # individual weight limits
    constraints = constraints   # portfolio sum constraint
    )

# Check if optimizer converged successfully
if not result.success:
    print("WARNING: Optimization did not converge")
    print("Message:", result.message)
    
    
# optimal portfolio weights
weights = result.x



###########
# 6. Output of weights per stock
###########

portfolio = pd.DataFrame({
    "Ticker": tickers,
    "Weight": weights
    })

print("\nOptimized Portfolio Weights: \n")

print(portfolio.sort_values("Weight", ascending = False))

print("\nSanity Checks:\n")
print(f"Sum of weights: {np.sum(weights):.4f}")
print(f"Max weight: {np.max(weights):.4f}")
print(f"Min weight: {np.min(weights):.4f}")


###########
# 7. Optimal portfolio vs Equal weighted portfolio for train and test years
###########

## Equal portfolio
equal_weights = np.ones(n)/n


# train returns
opt_train_returns = train_returns @ weights         # optimized
eq_train_returns = train_returns @ equal_weights    # equal weights

# test returns 
opt_test_returns = test_returns @ weights           # optimized
eq_test_returns = test_returns @ equal_weights     # equal weights


#################
# 8. Annualization
#################


# TRAIN YEAR (in-sample) realized performance 
opt_train_ret_annual = (1 + opt_train_returns).prod() - 1 
opt_train_vol_annual = opt_train_returns.std() * np.sqrt(TRADING_DAYS)
opt_train_sharpe = opt_train_ret_annual / opt_train_vol_annual    
# *** ADD risk-free rate later


eq_train_ret_annual = (1 + eq_train_returns).prod() - 1
eq_train_vol_annual = eq_train_returns.std() * np.sqrt(TRADING_DAYS)
eq_train_sharpe = eq_train_ret_annual / eq_train_vol_annual

# TEST YEAR (out-of-sample) realized performance
opt_test_ret_annual = (1 + opt_test_returns).prod() - 1
opt_test_vol_annual = opt_test_returns.std() * np.sqrt(TRADING_DAYS)
opt_test_sharpe = opt_test_ret_annual / opt_test_vol_annual    
# *** ADD risk-free rate later


eq_test_ret_annual = (1 + eq_test_returns).prod() - 1
eq_test_vol_annual =  eq_test_returns.std() * np.sqrt(TRADING_DAYS)
eq_test_sharpe = eq_test_ret_annual / eq_test_vol_annual




###########
# 9. Results comparision for training year
###########

print("\n=================== RESULTS ===================\n")
print(f"Train Year: {TRAIN_YEAR}")
print(f"Test Year: {TEST_YEAR}")

## Train
print("\n=================== TRAIN (In-Sample Optimization Period) ===================\n")

print("Optimized Portfolio:\n")
print(f"Annual Return: {opt_train_ret_annual*100:.2f}%")
print(f"Volatility: {opt_train_vol_annual*100:.2f}%")
print(f"Sharpe Ratio: {opt_train_sharpe:.3f}")
      

print("\nEqual-weight Portfolio:\n")
print(f"Annual Return: {eq_train_ret_annual * 100 :.2f}%")
print(f"Volatility: {eq_train_vol_annual * 100 :.2f}%")
print(f"Sharpe Ratio: {eq_train_sharpe:.3f}")


## Test
print("\n=================== TEST (Out-of-Sample Evaluation Period) ===================\n")

print("Optimized Portfolio:\n")
print(f"Annual Return: {opt_test_ret_annual*100:.2f}%")
print(f"Volatility: {opt_test_vol_annual*100:.2f}%")
print(f"Sharpe Ratio: {opt_test_sharpe:.3f}")
      

print("\nEqual-weight Portfolio:\n")
print(f"Annual: {eq_test_ret_annual * 100 :.2f}%")
print(f"Volatility: {eq_test_vol_annual * 100 :.2f}%")
print(f"Sharpe Ratio: {eq_test_sharpe:.3f}")


