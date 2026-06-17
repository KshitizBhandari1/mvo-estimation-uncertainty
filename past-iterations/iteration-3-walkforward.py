#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 4 2026

@author: kshitizbhandari


Mean-Variance Portfolio Optimization (S&P 500 Version)

Goal:
    - Long-only portfolio optimization on current S&P 500 constituents
    - Estimate expected returns + covariance from training year 
    - Solve for weights that maximize Sharpe ratio (1Y Treasury used as r_f)
    - Evaluate performance of the optimized weights out-of-sample (next year)
    - Compare against Equal-Weighted portfolio and SPY total return

Notes:
    - Uses current S&P 500 list (survivorship bias present)
    - Annual rebalancing at the end of December (close price, last trading day)
    - Stocks with insufficient data in training year are excluded
    - 1Y US Treasury yield (DGS1) at rebalancing used as risk-free rate

Error realized:
    - implemented a duration mismatch between returns/covariance and r_f.
        - Optimizer inputs daily returns/covariance but annual risk-free rate

"""

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from scipy.optimize import minimize
from pathlib import Path
import time
import os
import logging
import matplotlib.pyplot as plt



#####################
# 0. Global variables
#####################

MIN_WEIGHT = 0.00           # 0.0 -> long only
MAX_WEIGHT = 0.10           # max weight per stock

TRADING_DAYS = 252          # trading days per year
STARTING_CAPITAL = 1.0      # set initial investment
COST_RATE = 0.0025          # 25 bps cost


# suppress noisy yfinance warniings (e.g., delisted tickers)
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

pd.set_option('display.float_format', '{:.4f}'.format)


#####################
# 1. Load 1-year Treasury yield (DGS1) downloaded as local .CSV from FRED
#####################

def load_dgs1_data (csv_path = 'DGS1.csv'):
    """
    Load DGS1 (1Y US Treasury yield series) from a local CSV.
    File obtained from FRED has two columns:
        - observation_date
        - DGS1 (yield in percent, i.e. 3.04 instead of 3.04%)
        
    Resamples data to daily frequency and forward-fills missing values
    
    Returns: dataframe - containing a daily time series of 1Y Treasury yields
    """
    csv_path = Path(csv_path)
    # in case file doesn't exist, raise error early on
    if not csv_path.exists():
        raise FileNotFoundError(f"DGS1 file not found: {csv_path.resolve()}")
    
    # read csv file with two columns: 'observation_date' and 'DGS1'
    df = pd.read_csv(csv_path, parse_dates = ['observation_date'])
    # set the date as index and sort
    df = df.set_index('observation_date').sort_index()
    
    # ensure values are numeric
    df['DGS1'] = pd.to_numeric(df['DGS1'], errors = 'coerce')
    
    # create daily index and forward fill missing values (weekends/holidays)
    df = df.resample('D').ffill()
    
    return df


# initiazlie project root as folder containing this script
PROJECT_ROOT = Path(__file__).resolve().parent

# walk up until repo root marker
# assumes repo root contains "data/" folder
while not(PROJECT_ROOT / "data").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_DIR = PROJECT_ROOT / 'data'
DGS1_PATH = DATA_DIR / 'DGS1.csv'

DGS1_DATA = load_dgs1_data(DGS1_PATH)


def get_risk_free_rate(year: int) -> float:
    """
    Extracts 1Y Treasury yield from the last trading day of previous year
    - Because portfolio is formed at END of (year-1)
    
    Returns:
        float - risk-free rate in decimal (i.e. 0.0123 returned means 1.23%)
    """
    # using last ~2 weeks of December to ensure at least one valid trading day
    last_year_rates = DGS1_DATA.loc[f'{year-1}-12-15':f'{year-1}-12-31']['DGS1'].dropna()
    
    if last_year_rates.empty:
        print(f'WARNING: Could not find DGS1 rate for end of {year-1}.')
        print('Using r_f = 0.0')
        return 0.0
    
    # extract and return alst value of the last trading day of previous year
    return float(last_year_rates.iloc[-1]) / 100.0


#####################
# 2. Get S&P 500 Tickers
#####################
def get_sp500_tickers() -> list[str]:
    """
    Scrapes current S&P500 constituents from Wikipedia.

    Note: this only gives CURRENT composition.
    For historical usage, it introduces survivorship bias.
    
    Returns:
        list - with tickers of S&P500 from current Wikipedia webpage.
    """
    # Wikipedia URL for current S&P 500 constituents
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    
    # using requests to scrape because using read_html directly got blocked
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WIN64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
        }
    response = requests.get(url, headers = headers)
    response.raise_for_status()             # raise error if request fails
    
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', id = 'constituents')
    
    tickers = []
    
    if table is None:
        print('ERROR: Could not find S&P 500 table.')
        return tickers                      # empty list
    
    for row in table.find_all('tr')[1:]:    # skip header
        cols = row.find_all('td')
        
        if cols:
            ticker = cols[0].text.strip()
            
            # convert stocks like BRK.B to BRK-B for yfinance compatibility
            ticker = ticker.replace('.', '-')
            # update list
            tickers.append(ticker)
            
    print(f'Fetched {len(tickers)} S&P 500 tickers.')
    
    return tickers


#####################
# 3. Price Data Caching
#####################

# cache directory - repo-root anchored
CACHE_DIR = PROJECT_ROOT / 'price_cache'
os.makedirs(CACHE_DIR, exist_ok = True)

def get_cache_path(train_year, test_year):
    """
    Generate file path for cached price data.
    
    Note: Cache is keyed by train/test window to preserve exact data slice
    used in each run.
    
    Returns:
        Path - path to parquet cache file
    """
    return CACHE_DIR / f'prices_{train_year}_{test_year}.parquet'


def download_and_cache_prices(train_year, test_year, tickers, force_redownload = False):
    """
    For a given pair of train year and test year (i.e. one test-train window), 
    downloads adjusted-close prices for tickers.
    
    Uses per-ticker download with added delay to avoid Yahoo rate limits.
    
    Saves the result locally as parquet for fast future loading.
    
    If force_redownload = False (default) and cache file already exists,
    loads from the locally saved files.
    """
    cache_file = get_cache_path(train_year, test_year)
    
    # load local cache file if it exists and force redownload = False
    if cache_file.exists() and not force_redownload:
        print(f'Loading cached data for {train_year}-{test_year}')
        return pd.read_parquet(cache_file)
    
    print(f'Downloading {train_year}-{test_year} ...')
    
    start_date = f'{train_year}-01-01'
    end_date = f'{test_year+1}-01-01'
    
    prices_dict = {}
    
    for i, ticker in enumerate(tickers):
        if i % 10 == 0 and i > 0:
            time.sleep(0.25)      # added delay to avoid rate limits
        
        try:
            df = yf.download(ticker, start = start_date, end = end_date,
                             auto_adjust = True, progress = False)
            
            if 'Close' not in df:
                # print(f'  Skipped {ticker} (empty or no close)')              # for debugging
                continue
            
            # convert to one-dimensional series
            close_series = df['Close'].squeeze()
            
            if not isinstance(close_series, pd.Series):
                # print(f'  Skipped {ticker} not a Series after squeeze)')      # for debugging
                continue 
            
            if close_series.notna().sum() <= 1:
                # print(f'  Skipped {ticker} (insufficient data)')              # for debugging
                continue
            
            prices_dict[ticker]  = close_series
            
        except:
            continue # skip failed downloads silently
    
    price_df = pd.concat(prices_dict, axis = 1)
    
    # flatten columns
    price_df.columns = price_df.columns.get_level_values(0)
    
    # convert to cache file
    price_df.to_parquet(cache_file)
    
    return price_df


#####################
# 4. Core Helper Functions
#####################
def calculate_metrics(daily_returns: pd.Series,
                      rf_rate: float) -> tuple[float, float, float]:
    """
    Compute annual performance metrics from daily returns and risk-free rate:
        - annual return (compounded)
        - annualized volatility (scaled by trading_days in a year)
        - Sharpe ratio
     
    Returns: tuple: (annual_return, annual volatility, sharpe ratio)
    """
    if len(daily_returns) == 0:
        return 0.0, 0.0, 0.0
         
    annual_return = (1 + daily_returns).prod() - 1
    annual_volatility = daily_returns.std() * np.sqrt(TRADING_DAYS)
     
    if annual_volatility < 1e-8:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = (annual_return - rf_rate) / annual_volatility
    
    return annual_return, annual_volatility, sharpe_ratio


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
        - Expected return, E[R] = w^T * mu
        - Portfolio variance    = w^T Σ w
        - Volatility            = qrt(variance)
        - Sharpe ratio          = (E[R] - r_f) / volatility
        
    Returns: float - negative of portfolio's Sharpe ratio
    """
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_vol = np.sqrt( np.dot(weights.T, np.dot(cov_matrix,weights)) )
    
    if portfolio_vol < 1e-8:
        return 0.0          # avoids division-by-zero instability in optimizer
    
    return -(portfolio_return - rf_rate) / portfolio_vol


def weight_sum_constraint(weights):
    """
    Constraint ensuring portfolio is fully invested.
    
    Returns:
        float - should equal 0.0 when sum(weights) = 1.
    """
    return np.sum(weights) - 1


#####################
# 5. Portfolio Simulation
#####################

def simulate_portfolio(prices, weights, prev_weights,
                       capital, cost_rate = COST_RATE):
    """
    Simulate portfolio evolution over the test period
    
    Process:
        - Aligns weights with available tickers. 
        - Computes and applies turnover and transaction cost.
        - Allocate capital into shares and track portfolio value over time
        - Compute final weights
    
    Returns:
        tuple (daily_returns, end_value, end_weights, portfolio_values) where:
            daily returns (pd.Series) - daily portfolio returns over test period
            end_value (float) - final portfolio value at end of period
            end_weights (pd.Series) - portfolio weights at tend of period after drift
            portfolio_values (pd.Series) - time series of total portfolio value
    """
    # align weights to available assets
    weights = weights.reindex(prices.columns).fillna(0)
    prev_weights = prev_weights.reindex(prices.columns).fillna(0)
    
    # calculate turnover
    turnover = np.sum( np.abs(weights - prev_weights) )
    # calculate cost for total turnover for the given capital
    cost = capital * cost_rate * turnover
    
    capital_after_cost = capital - cost
    
    # convert weights into shares at start
    start_prices = prices.iloc[0]
    shares = (weights * capital_after_cost) / start_prices
    # basically, buy and hold these amount of shares
    
    # Portfolio value through time
    portfolio_values = (prices * shares).sum(axis = 1)
    
    daily_returns = portfolio_values.pct_change().dropna()
    
    # portfolio value at the end of the period
    end_value = float(portfolio_values.iloc[-1])
    
    # compute ending weights after drift 
    # (fixed shares but weights differ due to differing in prices)
    end_weights = (shares * prices.iloc[-1])
    end_weights = end_weights / end_weights.sum()
    
    return daily_returns, end_value, end_weights, portfolio_values



def max_drawdown(portfolio_series):
    """
    Compute max drawdown of a portfolio time series.

    Inputs:
        portfolio_series (pd.Series) - time series of portfolio values

    Returns
        float - maximum drawdown (negative value)

    """
    cumulative_max = portfolio_series.cummax()
    drawdown = portfolio_series / cumulative_max - 1
    
    # return maximum drawdown (drawdown itself is negative value)
    return drawdown.min()

#####################
# 6. Portfolio Optimizer
#####################   
def portfolio_optimizer(TRAIN_YEAR, TEST_YEAR, tickers,
                        prev_weights_opt, prev_weights_eq,
                        capital_opt, capital_eq, cost_rate = COST_RATE):
    """
    Core optimization + backtest step for a train-test pair

    Process:
        1. Load price data
        2. Split into train/test year
        3. Filter valid stocks
        4. Estimate mean and covariance using training year data
        5. Optimize Sharpe ratio with constraints
        6. Simulate optimized and equal-weight portfolios
        7. Compare against SPY benchmark
        
    Returns:
        dict - metrics, updated_capital, updated weights, portfolio values

    """
    
    print(f'\n=== Train {TRAIN_YEAR} -> Test {TEST_YEAR} ===')
    
    # DATA download and forward fill
    prices = download_and_cache_prices(TRAIN_YEAR, TEST_YEAR, tickers).ffill()
    # convert prices to simple returns
    returns = prices.pct_change()
    
    
    # === DATA FILTERING ===
    train_returns_raw = returns.loc[str(TRAIN_YEAR)]

    # compute fraction of non-missing data per asset
    data_availability = train_returns_raw.notna().mean()
    
    # Require at least 70% of trading days to avoid sparse data bias
    valid_mask = data_availability >= 0.70
    valid_stocks = data_availability[valid_mask].index
    
    if len(valid_stocks) == 0:
        raise ValueError(f'No stocks had sufficient data in {TRAIN_YEAR}')
    
    print(f'Using {len(valid_stocks)} stocks.')
    
    # apply filtering
    train_returns = train_returns_raw[valid_stocks].dropna()
    
    # for test year
    test_prices = prices.loc[str(TEST_YEAR)][valid_stocks]
    
    
    # === RISK-FREE RATES ===
    train_rf = get_risk_free_rate(TRAIN_YEAR)
    test_rf = get_risk_free_rate(TEST_YEAR)
    
    
    # === OPTIMIZATION ===
    n = len(valid_stocks)
    
    # estimate inputs for optimization
    # using simple historical estimates (no factor model)
    mean_returns = train_returns.mean()
    cov_matrix = train_returns.cov()
    
    # constraints
    bounds = [(MIN_WEIGHT, MAX_WEIGHT)] * n     # weight per asset
    constraints = {'type':'eq', 'fun': weight_sum_constraint}
    
    # initial guess -> start from equal weights
    init_guess = np.ones(n) / n
    
    # Optimize Sharpe ratio
    result = minimize(
        negative_sharpe,
        x0 = init_guess,
        args = (mean_returns, cov_matrix, train_rf),
        method = "SLSQP",               # Sequential Least Squares Programming
        bounds = bounds,
        constraints = constraints
    )
    
    optimized_weights = pd.Series(result.x, index = valid_stocks)
    
    if not result.success:
        print("WARNING: Optimization did not fully converge.")
        
    # Debugging: inspect concentration
    print('\n--- Optimizer Weights, Top 10 (Bounded, at rebalance) ---')
    print(optimized_weights.sort_values(ascending = False).head(10))
    
    print(f'Max weight (should be <= {MAX_WEIGHT:.0%}): {optimized_weights.max():.2%}')
    
    
    # === BENCHMARKS ===  
    # SPY benchmark - separate download
    spy_prices = yf.download(
        'SPY',
        start = f'{TRAIN_YEAR}-01-01',
        end = f'{TEST_YEAR+1}-01-01',
        auto_adjust = True,
        progress = False
        )['Close']
    
    spy_prices = spy_prices.ffill()
    
    # flatten if needed (yfinance sometimes returns DataFrame)
    if isinstance(spy_prices, pd.DataFrame):
        spy_prices = spy_prices.iloc[:, 0]
    

    spy_test_returns = spy_prices.loc[str(TEST_YEAR)].pct_change().dropna()
    
    # Equal-weight benchmark
    equal_weights = pd.Series(1/n, index = valid_stocks)
    
    # === SIMULATE PORTFOLIOS ===
    # 1. Optimized portfolio
    opt_returns, capital_opt, new_weights_opt, opt_values = simulate_portfolio(
        test_prices, optimized_weights, prev_weights_opt, capital_opt
        )
    # 2. Equal-weight portfolio
    eq_returns, capital_eq, new_weights_eq, eq_values = simulate_portfolio(
        test_prices, equal_weights, prev_weights_eq, capital_eq
        )
    
    spy_values = spy_prices.loc[str(TEST_YEAR)]

    # === PERFORMANCE METRICS ===
    opt_test_metrics = calculate_metrics(opt_returns, test_rf)
    eq_test_metrics = calculate_metrics(eq_returns, test_rf)
    spy_test_metrics = calculate_metrics(spy_test_returns, test_rf)
    
    print('\n--- TEST PERIOD (Out-of-Sample) ---')
    print(f'Optimized : Return {opt_test_metrics[0] : 6.2%} |'
          f' Vol {opt_test_metrics[1] : 6.2%} |'
          f' Sharpe {opt_test_metrics[2] : .3f}')
    print(f'Equal-Wt  : Return {eq_test_metrics[0] : 6.2%} |'
          f' Vol {eq_test_metrics[1] : 6.2%} |'
          f' Sharpe {eq_test_metrics[2] : .3f}')
    print(f'SPY       : Return {spy_test_metrics[0] : 6.2%} |'
          f' Vol {spy_test_metrics[1] : 6.2%} |'
          f' Sharpe {spy_test_metrics[2] : .3f}\n')
    
    return {
        'opt' : opt_test_metrics,
        'eq' : eq_test_metrics,
        'spy' : spy_test_metrics,
        'cap_opt': capital_opt,
        'cap_eq': capital_eq,
        'weights_opt' : new_weights_opt,
        'weights_eq' : new_weights_eq,
        'opt_values' : opt_values,
        'eq_values' : eq_values,
        'spy_values' : spy_values
        }


#####################
# 7. Walk Forward
#####################  
def run_walk_forward_backtest(start_year = 2010, end_year = 2024):
    """
    Execute full walk-forward backtest.
    
    Steps:
        - For each year:
            Train on year t
            Test on year t+1
        - Carry forward capital and weights
        - Track performance over time
    
    Outputs:
        - Year-by-year metrics
        - Win rates vs SPY and Eq-Wt benchmarks
        - Cumulative return plot
        
    Returns:
        tuple - (summary, opt_max_drawdown, eq_max_drawdown), where
            - summary: pd.DataFrame -> summary statistics
            - opt_max_drawdown: float -> max drawdown of optimized portfolio
            - eq_max_drawdown: float -> max drawdown of equal-weight portfolio
    """
    
    # get tickers - only current members of S&P 500 are included
    tickers = get_sp500_tickers()
    # assign starting capital for both optimized and equal-weight portfolios
    capital_opt = STARTING_CAPITAL
    capital_eq = STARTING_CAPITAL
    # initialize previous weights for both portfolios
    prev_weights_opt = pd.Series(dtype = float)
    prev_weights_eq = pd.Series(dtype = float)
    
    results = []
    
    all_opt_values = []
    all_eq_values = []
    all_spy_values = []
    
    # walk-forward loop
    
    for year in range(start_year, end_year + 1):
        
        result = portfolio_optimizer(
            TRAIN_YEAR = year,
            TEST_YEAR = year + 1,
            tickers = tickers,
            prev_weights_opt = prev_weights_opt,
            prev_weights_eq = prev_weights_eq,
            capital_opt = capital_opt,
            capital_eq = capital_eq
            )
        
        # update capital and weights for next iteration
        capital_opt = result['cap_opt']
        capital_eq = result['cap_eq']
        
        prev_weights_opt = result['weights_opt']
        prev_weights_eq = result['weights_eq']
        
        # append portfolio values and the final result dictionary
        all_opt_values.append(result['opt_values'])
        all_eq_values.append(result['eq_values'])
        all_spy_values.append(result['spy_values'])
        
        results.append(result)

    # stitch yearly portfolio paths into one continuous time series
    opt_portfolio_series = pd.concat(all_opt_values).sort_index()
    eq_portfolio_series = pd.concat(all_eq_values).sort_index()
    spy_portfolio_series = pd.concat(all_spy_values).sort_index()
        
    
    summary = pd.DataFrame( [
        {
            'Train Year': start_year + i,
            'Test Year': start_year + i + 1,
            'Opt Return': r['opt'][0],
            'Eq Return': r['eq'][0],
            'SPY Return': r['spy'][0],
            'Opt Sharpe': r['opt'][2],
            'Eq Sharpe': r['eq'][2],
            'SPY Sharpe': r['spy'][2],
            'Opt > Eq': r['opt'][0] > r['eq'][0],
            'Opt > SPY': r['opt'][0] > r['spy'][0],
            'Sharpe > Eq': r['opt'][2] > r['eq'][2],
            'Sharpe > SPY' : r['opt'][2] > r['spy'][2]
        } for i, r in enumerate(results)
    ] )
    
    total_years = len(summary)
    
    print('\n' + '='*33)
    print('Performance Metrics')
    print('='*33)
    
    # beat benchmarks by return
    win_eq_return = summary ['Opt > Eq'].mean()
    win_spy_return = summary ['Opt > SPY'].mean()
    
    print(f'Beat Equal-Weight (Return): {win_eq_return:5.1%}'
          f'  ({ summary["Opt > Eq"].sum() }/{ total_years } years)')
    print(f'Beat SPY (Return)         : {win_spy_return:.1%}'
          f'  ({ summary["Opt > SPY"].sum() }/{ total_years } years)')
    
    # beat benchmarks by Sharpe
    win_eq_sharpe = summary ['Sharpe > Eq'].mean()
    win_spy_sharpe = summary ['Sharpe > SPY'].mean()

    print(f'Beat Equal-Weight (Sharpe): {win_eq_sharpe:5.1%}'
          f'  ({ summary["Sharpe > Eq"].sum() }/{ total_years } years)')
    print(f'Beat SPY (Sharpe)         : {win_spy_sharpe:.1%}'
          f'  ({ summary["Sharpe > SPY"].sum() }/{ total_years } years)')
    
    # average excess returns against benchmarks
    avg_excess_spy = (summary['Opt Return'] - summary['SPY Return']).mean()
    avg_excess_eq = (summary['Opt Return'] - summary['Eq Return']).mean()
    
    print(f'\nAverage Excess Return vs SPY  : {avg_excess_spy:5.2%}')
    print(f'Average Excess Return vs Eq-Wt: {avg_excess_eq:5.2%}')
    
    print(f'Average Optimized Sharpe      : {summary["Opt Sharpe"].mean():.3f}')
    
    # max drawdowns
    opt_max_drawdown = max_drawdown(opt_portfolio_series)
    eq_max_drawdown = max_drawdown(eq_portfolio_series)
    
    print(f'\nMax Drawdown (Optimized)    : {opt_max_drawdown:.2%}')
    print(f'Max Drawdown (Equal-Weight) : {eq_max_drawdown:.2%}')
    
    
    
    
    opt_curve = opt_portfolio_series / opt_portfolio_series.iloc[0]
    eq_curve = eq_portfolio_series / eq_portfolio_series.iloc[0]
    spy_curve = spy_portfolio_series / spy_portfolio_series.iloc[0]
    
    
    print(f'\n=== Final Value of ${STARTING_CAPITAL} Invested ===')
    print(f'Optimized Portfolio : ${opt_curve.iloc[-1]*STARTING_CAPITAL:5.2f}')
    print(f'Equal-Weighted      : ${eq_curve.iloc[-1]*STARTING_CAPITAL:5.2f}')
    print(f'SPY                 : ${spy_curve.iloc[-1]*STARTING_CAPITAL:5.2f}')
    
    
    # === PLOT ===
    plt.figure(figsize=(12, 7))
    
    plt.plot(opt_curve.index, opt_curve,
             label = 'Optimized Portfolio', linewidth = 2.5)
    plt.plot(eq_curve.index, eq_curve,
             label = 'Equal-Weighted', linewidth = 2)
    plt.plot(spy_curve.index, spy_curve,
             label = 'SPY', linewidth = 2)
    
    plt.title(f'Walk-Forward Backtest (Max Weight = {MAX_WEIGHT:.1%})')
    plt.xlabel('Year')
    plt.ylabel('Growth of $1')
    plt.legend()
    plt.grid(True)
    
    plt.show()
    
    return summary, opt_max_drawdown, eq_max_drawdown
    
    
#####################
# 7. Execute
#####################  
if __name__ == '__main__':
    run_walk_forward_backtest(2010, 2024)
