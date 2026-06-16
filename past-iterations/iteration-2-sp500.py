#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 1 2026

@author: kshitizbhandari


Mean-Variance Portfolio Optimization (S&P 500 Version)

Goal:
    - Long-only portfolio optimization on current S&P 500 constituents
    - Estimate expected returns + covariance from training year 
    - Solve for weights that maximize Sharpe ratio (1Y Treasury used as r_f)
    - Evaluate performance of the optimized weights out-of-sample (the following year)
    - Compare against Equal-Weighted portfolio and SPY total return

Notes:
    - Uses current S&P 500 list (survivorship bias present)
    - Annual rebalancing at the end of December (close price of last trading day)***
    - Stocks with insufficient data in training year are excluded
    - 1Y US Treasury yield (DGS1) at rebalancing used as risk-free rate
    
    
Error realized:
    annual rebalancing was intended but realized that the portfolio math 
        actually assumes daily rebalancing
    - having weights @ daily returns -> would assume same weight for each day,
        hence daily rebalancing
"""

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from scipy.optimize import minimize
import logging
from pathlib import Path

# supress noisy yfinance warnings (e.g., delisted tickers)
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

pd.set_option('display.float_format', '{:.4f}'.format)


#####################
# 0. Load 1-year Treasury yield (DGS1) downloaded as local .CSV from FRED
#####################

def load_dgs1_data(csv_path = "DGS1.csv"):
    """
    Load DGS1 (1Y US Treasury yield series) from a local CSV within data/.
    File obtained from FRED has two columns:
        - observation_date
        - DGS1 (yield in percent, i.e. 3.01 instead of 3.01%)
    
    Resamples data to daily frequency and forward-fills missing values
    
    Returns: dataframe - containing a daily time series of 1Y Treasury yields
    """
    csv_path = Path(csv_path)
    # in case file doesn't exist
    if not csv_path.exists():
        raise FileNotFoundError(f"DGS1 file not found: {csv_path.resolve()}")
    
    # read csv file with two columns: "observation_date" and "DGS1"
    df = pd.read_csv(csv_path, parse_dates=['observation_date'])
    # set the date as index and sort
    df = df.set_index('observation_date').sort_index()
    
    # ensure values are numeric
    df['DGS1']= pd.to_numeric(df['DGS1'], errors = 'coerce')
    
    # create daily index and forward fill missing values (weekends/holidays)
    df = df.resample('D').ffill()
    
    return df


# folder containing this script
PROJECT_ROOT = Path(__file__).resolve().parent

# walk up until repo root marker
# assumes repo root contains "data/" folder
while not(PROJECT_ROOT / "data").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data"
DGS1_PATH = DATA_DIR / "DGS1.csv"

DGS1_DATA = load_dgs1_data(DGS1_PATH)


def get_risk_free_rate(year: int) -> float:
    """
    Extracts 1Y Treasury yield from the last trading day of previous year 
    - Because portfolio is formed at END of (year-1)
    
    Returns: risk-free rate in decimal form (i.e., 0.0123 returned means 1.23%)
    """

    last_year_rates = DGS1_DATA.loc[f'{year-1}-12-15':f'{year-1}-12-31']['DGS1'].dropna()
    
    if last_year_rates.empty:
        print(f'WARNING: Could not find DGS1 rate for end of {year-1}.')
        print('Using r_f = 0.0')
        return 0.0
    
    # extract and return value of the last trading day of previous year
    return float(last_year_rates.iloc[-1]) / 100.0
    


#################
# 1. Get the investable universe (Current S&P 500 members)
#################

def get_sp500_tickers():
    """
    Scrapes current S&P500 constituents from Wikipedia.

    Note: this only gives CURRENT composition
    For historical usage, it introduces survivorship bias.
    
    Returns: list - with tickers of S&P 500 from current Wikipedia webpage.
    """
    # Wikipedia URL for current S&P 500 constituents
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    
    # using requests to scrape because using read_html directly got blocked
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WIN64; x64)"
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
        }
    response = requests.get(url, headers = headers, timeout = 15)
    response.raise_for_status()     # raise error if request fails
    
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', id = 'constituents')
    
    tickers = []
    
    if table is None:
        print('ERROR: Could not find S&P 500 table.')
        return tickers                # empty list
    
    
    for row in table.find_all("tr")[1:]:        # skips header
        cols = row.find_all("td")
        if cols:
            ticker = cols[0].text.strip()
            
            # convert stocks like BRK.B to BRK-B for yfinance compatibility
            ticker = ticker.replace(".", "-")
            # update list
            tickers.append(ticker)
    
    print(f'Fetched {len(tickers)} S&P 500 tickers.')
    
    return tickers





###################
# 2. Core Helper Functions
###################

def calculate_metrics(daily_returns, rf_rate, trading_days = 252):
    """
    Converts a series of daily returns into annualized metrics.
    
    Note: uses 252 as the standard number of trading days per year
    
    Returns: (Annual Return, Annual volatility, Sharpe ratio), where:
    - Annual return: compounded product of daily returns
    - Annual volatility: standard deviation of daily returns scaled annually
    - Sharpe ratio: Excess return divided by volatility
    
    """    
    if len(daily_returns) == 0:
        return 0.0, 0.0, 0.0
    
    # compute the relevant metrics
    annual_return = (1 + daily_returns).prod() - 1
    annual_vol = daily_returns.std() * np.sqrt(trading_days)
    
    sharpe_ratio = (annual_return - rf_rate) / annual_vol
    
    return annual_return, annual_vol, sharpe_ratio


def negative_sharpe(weights, mean_returns, cov_matrix, rf_rate):
    """
    Optimization helper function.
    
    Since scipy performs minimization,
    to maximize sharpe ratio, we can minimize the -ve Sharpe ratio
    
    Inputs:
        - weights: allocation vector, sum = 1 when fully invested
        - mean_returns: Expected returns of asset
        - cov_matrix: Covariance matrix of asset returns
        - rf_rate: risk-free rate used as 1Y Treasury at portfolio formation
        
    Portfolio Math: 
        - Expected return = w^T * mu
        - Portfolio variance: w^T Σ w
        - Volatility = sqrt(variance)
        
    Returns: negative of portfolio's Sharpe ratio
    """
    
    ret = np.dot(weights, mean_returns)
    vol = np.sqrt( np.dot(weights.T, np.dot(cov_matrix, weights)) ) 
    
    sharpe_ratio = (ret - rf_rate) / vol
    
    return -sharpe_ratio


def weight_sum_constraint(w):
    """
    Enforcing constraint: sum of weights = 1 for the portfolio.
    
    i.e., portfolio is fully invested (only equities from investable universe)

    """
    return np.sum(w) - 1



#####################
# 3. Main portfolio optimization
###################

def portfolio_optimizer(TRAIN_YEAR, TEST_YEAR, MAX_WEIGHT = 0.10, MIN_WEIGHT = 0.0, verbose = True):
    """
    Core optimization function:
        First, maximizes Sharpe ratio for TRAIN_YEAR
        Second, tests performance of that portfolio for TEST_YEAR
        
    If MAX_WEIGHT and MIN_WEIGHT not specified, enforces long-only with 10% cap
    Portfolio is fully invested.
    
    Returns:
        Dictionary with weights and performance metrics for optimized portfolio,
        equal-weighted portfolio, and SPY
    """
    
    all_tickers = get_sp500_tickers()
    
    START_DATE = f'{TRAIN_YEAR}-01-01'
    END_DATE = f'{TEST_YEAR + 1}-01-01'
    
    if verbose:
        print(f'\n=== Train: {TRAIN_YEAR} ➡︎ Test: {TEST_YEAR} ===')
    

    # ==================== DATA DOWNLOAD ========================
    # yfinance returns multiple prices for each ticker for each date
    # extracting auto-adjusted close (to generate total return)
    prices = yf.download(
        all_tickers,
        start = START_DATE,
        end = END_DATE,
        auto_adjust = True,
        progress = False
    )["Close"]
    
    # forward-fill to handle missing trading days for all tickers
    prices = prices.ffill()
    
    # convert prices to daily returns
    returns = prices.pct_change()
    
    
    # ==================== DATA FILTERING ========================
    # restricting to training year
    train_returns_raw = returns.loc[str(TRAIN_YEAR)]
    
    # compute fraction of non-missing data per asset
    data_availability = train_returns_raw.notna().mean()
    
    # Keep only stocks with >= 70% data availability in training year
    valid_mask = data_availability >= 0.70
    valid_stocks = data_availability[valid_mask].index.tolist()
    
    
    if len(valid_stocks) == 0:
        raise ValueError(f'No stocks had sufficient data in {TRAIN_YEAR}')
        
    if verbose:
        print(f'Using {len(valid_stocks)} stocks with sufficient training data'
              f'(dropped {len(all_tickers) - len(valid_stocks)} stocks)')
        
    # filter returns to only valid stocks
    train_returns = train_returns_raw[valid_stocks].dropna(how = 'all')
    test_returns = returns.loc[str(TEST_YEAR)] [valid_stocks]
    
    
    # ==================== BENCHMARK (SPY) ========================
    # download from yahoo finance
    spy_prices = yf.download("SPY", start = START_DATE, end = END_DATE,
                             auto_adjust = True, progress = False)['Close']
    # forward fill any mising values
    spy_prices = spy_prices.ffill()
    # calculate daily returns of the benchmark for train and test year
    spy_train_returns = spy_prices.loc[str(TRAIN_YEAR)].pct_change().dropna()
    spy_test_returns = spy_prices.loc[str(TEST_YEAR)].pct_change().dropna()
    
    #yfinance kept returning DataFrame, causing errors in calculation later
    if isinstance(spy_train_returns, pd.DataFrame):
        spy_train_returns = spy_train_returns.iloc[:,0]
    if isinstance(spy_test_returns, pd.DataFrame):
        spy_test_returns = spy_test_returns.iloc[:,0]
        
    # Risk-free rates
    train_rf = get_risk_free_rate(TRAIN_YEAR)
    test_rf = get_risk_free_rate(TEST_YEAR)
    
    if verbose:
        print(f'r_f Train: {train_rf*100:.3f}% | r_f Test: {test_rf*100:.3f}%')
        # converted to percent from decimal
    
        
    # ==================== OPTIMIZATION ========================
    
    # number of assets in universe
    n = len(valid_stocks)
    
    # Expected return and covariance matrix calculated from training data
    mean_returns = train_returns.mean()
    cov_matrix = train_returns.cov()
    
    # Bounds: enforce maximum stock per stock constraint
    # each stock constrained individually
    bounds = [(MIN_WEIGHT, MAX_WEIGHT) for _ in range(n)]
    
    constraints = {
        'type': 'eq',
        'fun': weight_sum_constraint    # weights sum to 1 (fully invested)
        }
    
    # Initial guess for optimizer -> assume equal weight portfolio
    init_guess = np.ones(n) / n     # a vector of 1/n weight for each stock
    
    
    # implement optimization
    result = minimize(
        negative_sharpe,            # objective function
        init_guess,                 # starting weights
        args = (mean_returns, cov_matrix, train_rf),
        method = "SLSQP",           # Sequential Least Squares Programming
        bounds = bounds,            # individual weight limits
        constraints = constraints,  # sum of weight constraint
        options = {'maxiter':100, 'ftol': 1e-8}
        )
        
    if not result.success:
        print("WARNING: Optimization did not fully converge")
    
    # optimal portfolio weights
    weights = result.x
    
    # ==================== PORTFOLIO RETURNS ========================
    
    # equal-weight portfolio
    equal_weights = np.ones(n)/n
    
    # train returns
    opt_train_returns = train_returns @ weights         # optimized
    eq_train_returns = train_returns @ equal_weights    # equal weights
    
    opt_test_returns = test_returns @ weights           # optimized
    eq_test_returns = test_returns @ equal_weights      # equal weights
    
    # ==================== PERFORMANCE METRICS ========================
    
    # TRAIN_YEAR (in-sample) actual performance
    opt_train_metrics = calculate_metrics(opt_train_returns, train_rf)
    eq_train_metrics = calculate_metrics(eq_train_returns, train_rf)
    spy_train_metrics = calculate_metrics(spy_train_returns, train_rf)
    
    # TEST_YEAR (out-of-sample) realized performance
    opt_test_metrics = calculate_metrics(opt_test_returns, test_rf)
    eq_test_metrics = calculate_metrics(eq_test_returns, test_rf)
    spy_test_metrics = calculate_metrics(spy_test_returns, test_rf)
    
    if verbose:
        print('\n--- TEST PERIOD (Out-of-Sample) ---')
        print(f'Optimized: Return {opt_test_metrics[0]*100:6.2f}% | Vol {opt_test_metrics[1]*100:6.2f}% | Sharpe {opt_test_metrics[2]:.3f}')
        print(f'Equal-Wt : Return {eq_test_metrics[0]*100:6.2f}% | Vol {eq_test_metrics[1]*100:6.2f}% | Sharpe {eq_test_metrics[2]:.3f}')
        print(f'SPY      : Return {spy_test_metrics[0]*100:6.2f}% | Vol {spy_test_metrics[1]*100:6.2f}% | Sharpe {spy_test_metrics[2]:.3f}')
    

    return {
        'train_year' : TRAIN_YEAR,
        'test_year' : TEST_YEAR,
        'weights' : pd.Series(weights, index = valid_stocks),
        'opt_train' : opt_train_metrics,
        'eq_train' : eq_train_metrics,
        'spy_train' : spy_train_metrics,
        'opt_test' : opt_test_metrics,
        'eq_test' : eq_test_metrics,
        'spy_test' : spy_test_metrics,
        'n_stocks' : len(valid_stocks)
        }

# ===================== DEMO RUN =====================
if __name__ == "__main__":
    result = portfolio_optimizer(TRAIN_YEAR=2024, TEST_YEAR=2025, verbose=True)

     #Example inspection:
    weights_sorted = result['weights'].sort_values(ascending=False)
    print(weights_sorted.head(15))
    
    