#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 13:53:08 2026

@author: kshitizbhandari

Mean-Variance Optimization - Iteration 4:
    Covariance Shrinkage and Estimation Error analysis (S&P 500)

==================
Framework
==================
    Strategy: 
        Long-only, fully invested mean-variance optimized portfolio with 
        a maximum allocation of 10% per single stock
    Universe:
        Current S&P 500 constituents scraped from Wikipedia
    Methodology: 
        Walk-forward backtest using annual rebalancing
            - Train on year t
            - Calculate portfolio weights to maximize Sharpe ratio at year-end t
            - Hold portfolio through year t+1
            - Rebalance Annually
            
    Benchmarks:
        - S&P500 (SPY total-return adjusted prices)
        - Equal-weighted (Eq-Wt) portfolio of the same stock universe

==================
Iteration Log
==================
    Iteration 1: 
        Simple proof of concept. 20-stock static universe, r_f = 0.0.
    Iteration 2: 
        Expanded universe to current S&P500 members.
        Sourced 1Y Treasury yields from FRED as a dynamic risk-free rate
        Identified mathematical flaw that was assuming daily rebalancing
    Iteration 3: 
        Walk-forward multi-year backtest (train: 2010-2024 yearly)
        Simulated portfolio drift to ensure annual rebalancing
            - fixing previously found flaw of daily rebalancing
        Added transaction costs, maximum drawdown, and equity curves
        Identified daily/annual duration mismatch in Sharpe optimization inputs
    Iteration 4:
        Fixed duration mismatch by annualizing returns and covariance matrix
            for Sharpe optimization
        Implemented Ledoit-Wolf covariance shrinkage 
            to reduce covariance estimation noise
        Added portfolio structure metrics
            - turnover
            - effective N
            - number of holdings at maximum weight constraint
        Changed return data structures to return nested dictionary for smoother
            data handling
        Built helper functions to compare performance, risk, and structure
            between Sample μ - Sample Σ and Sample μ - Ledoit Σ implementations
    Iteration 5 (current):
        Added mean shrinkage towards cross-sectional mean.
        Fixed plot namings
        Expanded comparison of performance, risk, and structure metrics to:
            sample-sample, sample-Ledoit, shrunk-sample, shrunk-Ledoit
            mean-variance combinations.
        Refined calculate_metrics() function to:
            - defensively scale to support multi-year and fractional-year 
                return slices
        Added helper function to save plots

                  
==================
Notes
==================
    - Uses current S&P 500 list
        - survivorship bias present as stocks removed from index not included
        - Therefore, reported performance should be interpreted as an upper
            bound rather than fully realizable historical strategy.
        
    - To keep comparisons fair, Equal-Weighted portfolio uses the exact same
        survivorship-biased universe. Thus, true performance can be isolated
    - Annual rebalancing at the end of December (close price, last trading day)
    - Stocks with insufficient data in training year are excluded
    - 1Y US Treasury yield (DGS1) at rebalancing used as risk-free rate

==================
Results
==================

MAX WT = 20%
------------

                           Sample μ - Sample Σ Sample μ - Ledoit Σ Shrunk μ - Sample Σ Shrunk μ - Ledoit Σ
Final Wealth                            $17.92              $19.62              $13.71              $15.67
Avg Sharpe                               1.385               1.430               1.409               1.475
Avg Excess Return vs SPY                  9.2%               10.0%                6.5%                7.6%
Avg Excess Return vs Eq-Wt                5.9%                6.7%                3.1%                4.2%
Win Rate vs SPY (Return)         11/15 (73.3%)       11/15 (73.3%)       10/15 (66.7%)       10/15 (66.7%)
Win Rate vs Eq-Wt (Return)        8/15 (53.3%)        8/15 (53.3%)        8/15 (53.3%)        8/15 (53.3%)
Win Rate vs SPY (Sharpe)          7/15 (46.7%)        7/15 (46.7%)        9/15 (60.0%)        9/15 (60.0%)
Win Rate vs Eq-Wt (Sharpe)        7/15 (46.7%)        7/15 (46.7%)        7/15 (46.7%)        8/15 (53.3%)
Max Drawdown                            -37.0%              -35.5%              -36.1%              -36.2%
Avg Effective N                           10.3                12.5                11.1                14.1
Avg Turnover                            181.7%              181.2%              179.8%              178.8%
Avg Stocks at Cap                          0.6                 0.3                 0.6                 0.3


MAX WT = 10%
------------
                           Sample μ - Sample Σ Sample μ - Ledoit Σ Shrunk μ - Sample Σ Shrunk μ - Ledoit Σ
Final Wealth                            $19.51              $20.83              $15.51              $16.45
Avg Sharpe                               1.478               1.500               1.527               1.546
Avg Excess Return vs SPY                 10.1%               10.6%                7.5%                8.0%
Avg Excess Return vs Eq-Wt                6.7%                7.2%                4.1%                4.7%
Win Rate vs SPY (Return)         12/15 (80.0%)       12/15 (80.0%)       11/15 (73.3%)       11/15 (73.3%)
Win Rate vs Eq-Wt (Return)        8/15 (53.3%)        8/15 (53.3%)       10/15 (66.7%)       10/15 (66.7%)
Win Rate vs SPY (Sharpe)          8/15 (53.3%)        7/15 (46.7%)       10/15 (66.7%)       10/15 (66.7%)
Win Rate vs Eq-Wt (Sharpe)        7/15 (46.7%)        7/15 (46.7%)        9/15 (60.0%)        9/15 (60.0%)
Max Drawdown                            -36.3%              -35.9%              -34.8%              -34.8%
Avg Effective N                           14.0                15.2                14.9                16.7
Avg Turnover                            181.8%              181.6%              178.0%              177.7%
Avg Stocks at Cap                          4.2                 3.7                 3.9                 2.9
   

MAX WT = 5%
-------------

                           Sample μ - Sample Σ Sample μ - Ledoit Σ Shrunk μ - Sample Σ Shrunk μ - Ledoit Σ
Final Wealth                            $18.78              $19.26              $16.16              $16.36
Avg Sharpe                               1.533               1.534               1.610               1.604
Avg Excess Return vs SPY                  9.7%                9.9%                7.7%                7.9%
Avg Excess Return vs Eq-Wt                6.3%                6.5%                4.3%                4.5%
Win Rate vs SPY (Return)         12/15 (80.0%)       12/15 (80.0%)       11/15 (73.3%)       11/15 (73.3%)
Win Rate vs Eq-Wt (Return)        9/15 (60.0%)        8/15 (53.3%)        9/15 (60.0%)        9/15 (60.0%)
Win Rate vs SPY (Sharpe)         10/15 (66.7%)       10/15 (66.7%)       11/15 (73.3%)       11/15 (73.3%)
Win Rate vs Eq-Wt (Sharpe)        7/15 (46.7%)        7/15 (46.7%)        9/15 (60.0%)        9/15 (60.0%)
Max Drawdown                            -36.8%              -36.7%              -35.3%              -35.4%
Avg Effective N                           23.7                24.2                24.4                25.6
Avg Turnover                            176.4%              176.5%              169.5%              169.5%
Avg Stocks at Cap                         12.8                11.9                12.1                10.9


    === Final Value of $1.0 Invested ===
    Equal-Weighted      : $10.75
    SPY                 : $ 7.02

"""

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from pathlib import Path
import time
import os
import logging
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates



#####################
# 0. Global variables
#####################

MIN_WEIGHT = 0.00           # 0.0 -> long only
MAX_WEIGHT = 0.20           # max weight per stock

TRADING_DAYS = 252          # trading days per year
STARTING_CAPITAL = 1.0      # set initial investment
COST_RATE = 0.0025          # 25 bps cost

MU_SHRINKAGE_ALPHA = 0.5    # weight of the cross-sectional mean vs sample


# suppress noisy yfinance warniings (e.g., delisted tickers)
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

pd.set_option('display.float_format', '{:.3f}'.format)


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
    
    Returns: DataFrame - containing a daily time series of 1Y Treasury yields
    """
    csv_path = Path(csv_path)
    # in case file doesn't exist, raise error early on
    if not csv_path.exists():
        raise FileNotFoundError(f'DGS1 file not found: {csv_path.resolve()}')
    
    # read csv file with two columns: 'observation_date' and 'DGS1'
    df = pd.read_csv(csv_path, parse_dates = ['observation_date'])
    # set the date as index and sort
    df = df.set_index('observation_date').sort_index()
    
    # ensure values are numeric
    df['DGS1'] = pd.to_numeric(df['DGS1'], errors = 'coerce')
    
    # create daily index and forward fill missing values (weekends/holidays)
    df = df.resample('D').ffill()
    
    return df


# initialize project root as directory containing this script
PROJECT_ROOT = Path(__file__).resolve().parent

# walk up until repo root marker
# assumes repo root contains "data/" directory
while not (PROJECT_ROOT / 'data').exists():
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
        (input CSV values are in percent units)
    """
    # using last ~2 weeks of December to ensure at least one valid trading day
    last_year_rates = DGS1_DATA.loc[f'{year-1}-12-15':f'{year-1}-12-31']['DGS1'].dropna()
    
    if last_year_rates.empty:
        print(f'WARNING: Could not find DGS1 rate for end of {year-1}.')
        print('Using r_f = 0.0')
        return 0.0
    
    # extract and return last value of the last trading day of previous year
    return float(last_year_rates.iloc[-1]) / 100.0


#####################
# 2. Get S&P 500 Tickers
#####################
def get_sp500_tickers(verbose = False) -> list[str]:
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
    
    if verbose:        
        print(f'Fetched {len(tickers)} S&P 500 tickers.')
    
    return tickers


#####################
# 3. Price Data Caching
#####################
CACHE_DIR = PROJECT_ROOT / 'price-cache'
os.makedirs(CACHE_DIR, exist_ok = True)

def get_cache_path(train_year, test_year):
    """
    Generate file path for cached price data.
    
    Note: Cache is keyed by train/test window to preserve exact data slice
    used in each run.
    
    Returns:
        string - path to parquet cache file
    """
    return CACHE_DIR / f'prices_{train_year}_{test_year}.parquet'


def download_and_cache_prices(train_year, test_year, tickers,
                              force_redownload = False,
                              verbose = False):
    """
    For a given pair of train year and test year (i.e. one train-test window), 
    downloads adjusted-close prices for tickers.
    
    Uses per-ticker download with added delay to avoid Yahoo rate limits.
    
    Saves the result locally as parquet for fast future loading.
    
    If force_redownload = False (default) and cache file already exists,
    loads from the locally saved files.
    """
    cache_file = get_cache_path(train_year, test_year)
    
    # load local cache file if it exists and force redownload = False
    if cache_file.exists() and not force_redownload:
        if verbose:
            print(f'Loading cached data for {train_year}-{test_year}')
        return pd.read_parquet(cache_file)
    
    if verbose:    
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
                      rf_rate: float) -> dict[str, float]:
    """
    Compute mathematically consistent annual performance metrics
        from simple daily returns and an annual risk-free rate.
    
    Dynamically adjust compounding horizon if multi-year or 
        fractional-year return series are passed.
     
    Returns:
        dict[str, float]:
            Dictionary containing: 
            - 'return': exact annualized compounded return
            - 'volatility': annualized simple daily volatility
            - 'sharpe': Sharpe ratio
        
    """
    if len(daily_returns) == 0:
        return {
            'return': 0.0,
            'volatility': 0.0,
            'sharpe': 0.0
            }
    
    # dynamically scale compounding window
    # to handle multi-year / fractional-year return slices
    actual_days = len(daily_returns)
    scaling_factor = TRADING_DAYS / actual_days # 1 if one-year data is passed
    
    ## Return
    # realized annual compounded return    
    cumulative_return = (1 + daily_returns).prod()
    annual_return = (cumulative_return ** scaling_factor) - 1
    
    ## Volatility
    # realized daily volatility scaled to standard trading year
    annual_volatility = daily_returns.std(ddof = 1) * np.sqrt(TRADING_DAYS)
    
    ## Sharpe
    # out of sample Sharpe ratio calculation
    if annual_volatility < 1e-8:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = (annual_return - rf_rate) / annual_volatility
    
    return {
        'return': annual_return,
        'volatility': annual_volatility,
        'sharpe': sharpe_ratio
        }


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
        - Volatility            = sqrt(variance)
        - Sharpe ratio          = (E[R] - r_f) / volatility
        
    Returns: float - negative of portfolio's Sharpe ratio
    """
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_vol = np.sqrt( np.dot(weights.T, np.dot(cov_matrix,weights)) )
    
    if portfolio_vol < 1e-8:
        return 0.0
    
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
        tuple:
            daily returns (pd.Series) - daily portfolio returns over test period
            end_value (float) - final portfolio value at end of period
            end_weights (pd.Series) - portfolio weights at tend of period after drift
            portfolio_values (pd.Series) - time series of total portfolio value
            turnover (float) - rebalance turnover of the optimized portfolio
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
    
    return daily_returns, end_value, end_weights, portfolio_values, turnover


def max_drawdown(portfolio_series):
    """
    Compute max drawdown of a portfolio time series.

    Inputs:
        portfolio_series (pd.Series) - time series of portfolio values

    Returns:
        float - maximum drawdown (negative value)

    """
    cumulative_max = portfolio_series.cummax()
    drawdown = portfolio_series / cumulative_max - 1
    
    # return maximum drawdown (drawdown itself is negative value)
    return drawdown.min()


def effective_n(weights):
    """
    Computes the effective number of holdings.
    Higher values imply more diversification.
    
    N_eff = 1 / sum(w_i^2)
    
    Returns:
        float - effective number of holdings
    """
    return 1.0 / np.sum(weights ** 2)


def save_backtest_plot(opt_curve, eq_curve, spy_curve,
                       experiment_label,
                       filename,
                       verbose = False
                       ):
    """

    Generate, format, and save high-quality equity curve comparisons.
    
    Inputs:
        opt_curve, eq_curve, spy_curve (pd.Series):
            Cumulative growth time series
        experiment_label (str):
            Title string for the plot indicating MVO configuration
        filename (Path):
            Export path for saving plot
        verbose (bool):
            print export confirmation if True

    Returns:
        None.

    """
    # initialize plotting layout
    fig, ax = plt.subplots(figsize = (12, 7), dpi = 300)

    ax.plot(opt_curve.index, opt_curve,
            label = 'Optimized Portfolio', linewidth = 2.5)
    ax.plot(eq_curve.index, eq_curve,
           label = 'Equal-Weighted', linewidth = 2)
    ax.plot(spy_curve.index, spy_curve,
            label = 'SPY', linewidth = 2)

    ax.set_title(experiment_label, fontsize = 13, fontweight = 'bold', pad = 12)

    ax.set_xlabel('Year', labelpad = 10)
    ax.set_ylabel('Growth of $1.00 Invested', labelpad = 10)
    ax.legend(loc = 'upper left')
    ax.grid(True, linestyle = ':', alpha = 0.6)

    # enforce fixed Y-axis 
    ax.set_ylim(0, 25)
    ax.set_yticks(range(0, 26, 4))
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, pos: f'${x:,.0f}')
        )

    # fix x-axis clustering (dates writing over one another)
    ax.xaxis.set_major_locator(
        mdates.AutoDateLocator(minticks = 5,maxticks = 10)
        )
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.autofmt_xdate(bottom = 0.18, rotation = 30, ha = 'right')
    
    plt.savefig(filename, dpi = 300, bbox_inches = 'tight')

    if verbose:
        print(f'Exported plot to: {filename}')

    plt.show()
    plt.close(fig)
    

#####################
# 6. Portfolio Optimizer
#####################   
def portfolio_optimizer(TRAIN_YEAR, TEST_YEAR, tickers,
                        prev_weights_opt, prev_weights_eq,
                        capital_opt, capital_eq, cost_rate = COST_RATE,
                        cov_shrinkage = False, 
                        mean_shrinkage = False,
                        verbose = False):
    """
    Core optimization + backtest step for a train-test pair

    Process:
        1. Load price data
        2. Split into train/test year
        3. Filter valid stocks
        4. Estimate expected returns and covariance using selected estimators
        5. Optimize Sharpe ratio with constraints
        6. Simulate optimized and equal-weight portfolios
        7. Compare against SPY benchmark
        
    Returns:
        dict - metrics (performance, portfolio, structure, and timeseries)

    """
    if verbose:
        print(f'\n=== Train {TRAIN_YEAR} -> Test {TEST_YEAR} ===')
    
    # DATA download and forward fill
    prices = download_and_cache_prices(TRAIN_YEAR, TEST_YEAR,
                                       tickers,
                                       verbose = verbose
                                       ).ffill()
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
    
    if verbose:
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
    mean_returns_sample = train_returns.mean()
            
    
    # conditional covariance matrix (sample vs shrinkage using Leodit-Wolf)
    if cov_shrinkage:
        lw = LedoitWolf()
        lw.fit(train_returns)
        
        cov_matrix = pd.DataFrame(
            lw.covariance_,
            index = train_returns.columns,
            columns = train_returns.columns
            )
    
    else:
        cov_matrix = train_returns.cov()
        
    # conditional expected return vector
    # (cross-sectional mean-shrinkage vs sample mean)
    if mean_shrinkage:
        mu_bar = mean_returns_sample.mean()
        mean_returns = MU_SHRINKAGE_ALPHA * mu_bar + (1 - MU_SHRINKAGE_ALPHA) * mean_returns_sample
    else:
        mean_returns = mean_returns_sample

    
    # Annualize for Sharpe optimization
    annual_mean_returns = mean_returns * TRADING_DAYS
    annual_cov_matrix = cov_matrix * TRADING_DAYS
    
    # constraints
    bounds = [(MIN_WEIGHT, MAX_WEIGHT)] * n     # weight per asset
    constraints = {'type':'eq', 'fun': weight_sum_constraint}
    
    # initial guess -> start from equal weights
    init_guess = np.ones(n) / n
    
    # Optimize Sharpe ratio
    result = minimize(
        negative_sharpe,
        x0 = init_guess,
        args = (annual_mean_returns, annual_cov_matrix, train_rf),
        method = "SLSQP",               # Sequential Least Squares Programming
        bounds = bounds,
        constraints = constraints
    )
    
    optimized_weights = pd.Series(result.x, index = valid_stocks)
    
    # number of stocks at maximum weight cap
    num_stocks_at_cap = np.sum(
        optimized_weights >= (MAX_WEIGHT - 1e-5)
        )
    
    if not result.success:
        raise RuntimeError('Optimization did not fully converge.')
        
    # Debugging: inspect concentration
    if verbose:
        print('\n--- Optimizer Weights, Top 10 (Bounded, at rebalance) ---')
        print(optimized_weights.sort_values(ascending = False).head(10))
    
        print(f'Max weight (should be <= 10%): '
              f'{optimized_weights.max():.2%}')
    
    
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

    (
     opt_returns, 
     capital_opt,
     new_weights_opt,
     opt_values,
     turnover_opt
    ) = simulate_portfolio(
        test_prices, optimized_weights, prev_weights_opt, capital_opt
        )
    
    # calculate effective number of stocks for optimized portfolio
    effective_n_opt = effective_n(optimized_weights)
       
    # 2. Equal-weight portfolio
    (
     eq_returns,
     capital_eq,
     new_weights_eq,
     eq_values,
     _
     ) = simulate_portfolio(
             test_prices, equal_weights, prev_weights_eq, capital_eq
             )
    
    spy_values = spy_prices.loc[str(TEST_YEAR)]

    # === PERFORMANCE METRICS ===
    opt_test_metrics = calculate_metrics(opt_returns, test_rf)
    eq_test_metrics = calculate_metrics(eq_returns, test_rf)
    spy_test_metrics = calculate_metrics(spy_test_returns, test_rf)
    
    if verbose:
        print('\n--- TEST PERIOD (Out-of-Sample) ---')
        print(f'Optimized : Return {opt_test_metrics["return"] : 7.2%} |'
              f' Vol {opt_test_metrics["volatility"] : 7.2%} |'
              f' Sharpe {opt_test_metrics["sharpe"] : .3f}')
        print(f'Equal-Wt  : Return {eq_test_metrics["return"] : 7.2%} |'
              f' Vol {eq_test_metrics["volatility"] : 7.2%} |'
              f' Sharpe {eq_test_metrics["sharpe"] : .3f}')
        print(f'SPY       : Return {spy_test_metrics["return"] : 7.2%} |'
              f' Vol {spy_test_metrics["volatility"] : 7.2%} |'
              f' Sharpe {spy_test_metrics["sharpe"] : .3f}\n')
    
    return {
        
        'performance': {
            'optimized': opt_test_metrics,
            'equal_weight': eq_test_metrics,
            'spy': spy_test_metrics
            },
        
        'portfolio' : {
            'capital_opt': capital_opt,
            'capital_eq': capital_eq,
            'weights_opt': new_weights_opt,
            'weights_eq': new_weights_eq
            },
        
        'structure': {
            'effective_n': effective_n_opt,
            'turnover': turnover_opt,
            'num_stocks_at_cap': num_stocks_at_cap
            },
        
        'timeseries': {
            'optimized': opt_values,
            'equal_weight': eq_values,
            'spy': spy_values
            }
        }


#####################
# 7. Walk Forward
#####################  

PLOTS_DIR = PROJECT_ROOT / 'plots'
os.makedirs(PLOTS_DIR, exist_ok = True)


def run_walk_forward_backtest(start_year = 2010, end_year = 2024,
                              cov_shrinkage = False,
                              mean_shrinkage = False,
                              verbose = False):
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
        dict containing:
            - annual results: pd.DataFrame -> yearly out-of-sample metrics
            - performance: aggregate return and Sharpe statistics
            - risk: drawdown statistics
            - structure: portfolio construction metrics
                - effective N
                - turnover
            - wealth:
                ending wealth values for optimized, eq-wt, and SPY portfolios
    """
    
    # get tickers - only current members of S&P 500 are included
    tickers = get_sp500_tickers(verbose = verbose)
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
            capital_eq = capital_eq,
            cov_shrinkage = cov_shrinkage,
            mean_shrinkage = mean_shrinkage,
            verbose = verbose
            )
        
        # update capital and weights for next iteration
        capital_opt = result['portfolio']['capital_opt']
        capital_eq = result['portfolio']['capital_eq']
        
        prev_weights_opt = result['portfolio']['weights_opt']
        prev_weights_eq = result['portfolio']['weights_eq']
        
        # append portfolio values and the final result dictionary
        all_opt_values.append(
            result['timeseries']['optimized']
            )
        all_eq_values.append(
            result['timeseries']['equal_weight']
            )
        all_spy_values.append(
            result['timeseries']['spy']
            )
        
        results.append(result)

    # stitch yearly portfolio paths into one continuous time series
    opt_portfolio_series = pd.concat(all_opt_values).sort_index()
    eq_portfolio_series = pd.concat(all_eq_values).sort_index()
    spy_portfolio_series = pd.concat(all_spy_values).sort_index()
        
    
    summary = pd.DataFrame( [
        {
            'Train Year': start_year + i,
            'Test Year': start_year + i + 1,
            'Opt Return': 
                r['performance']['optimized']['return'],
            'Eq Return': 
                r['performance']['equal_weight']['return'],
            'SPY Return': 
                r['performance']['spy']['return'],
            'Opt Sharpe': 
                r['performance']['optimized']['sharpe'],
            'Eq Sharpe': 
                r['performance']['equal_weight']['sharpe'],
            'SPY Sharpe':
                r['performance']['spy']['sharpe'],
            'Opt > Eq': 
                r['performance']['optimized']['return'] > 
                r['performance']['equal_weight']['return'],
            'Opt > SPY':
                r['performance']['optimized']['return'] >
                r['performance']['spy']['return'],
            'Sharpe > Eq': 
                r['performance']['optimized']['sharpe'] >
                r['performance']['equal_weight']['sharpe'],
            'Sharpe > SPY' : 
                r['performance']['optimized']['sharpe'] >
                r['performance']['spy']['sharpe'],
            'Opt Effective N' : r['structure']['effective_n'],
            'Opt Turnover': r['structure']['turnover'],
            'Stocks at Cap': r['structure']['num_stocks_at_cap']
        } for i, r in enumerate(results)
    ] )
    
    total_years = len(summary)
    
    if verbose:
        print('\n' + '='*33)
        print('Performance Metrics')
        print('='*33)
        
    # beat benchmarks by return
    # total wins
    wins_eq_return = summary ['Opt > Eq'].sum()
    wins_spy_return = summary ['Opt > SPY'].sum()
    
    # win rates
    win_rate_eq_return = summary ['Opt > Eq'].mean()
    win_rate_spy_return = summary ['Opt > SPY'].mean()
    
    if verbose:
        print(f'Beat Equal-Weight (Return): {win_rate_eq_return:5.1%}'
              f'  ({ summary["Opt > Eq"].sum() }/{ total_years } years)')
        print(f'Beat SPY (Return)         : {win_rate_spy_return:.1%}'
              f'  ({ summary["Opt > SPY"].sum() }/{ total_years } years)')
    
    # beat benchmarks by Sharpe
    # total wins
    wins_eq_sharpe = summary ['Sharpe > Eq'].sum()
    wins_spy_sharpe = summary ['Sharpe > SPY'].sum()
    
    # win rate
    win_rate_eq_sharpe = summary ['Sharpe > Eq'].mean()
    win_rate_spy_sharpe = summary ['Sharpe > SPY'].mean()
    
    if verbose:
        print(f'Beat Equal-Weight (Sharpe): {win_rate_eq_sharpe:5.1%}'
              f'  ({ summary["Sharpe > Eq"].sum() }/{ total_years } years)')
        print(f'Beat SPY (Sharpe)         : {win_rate_spy_sharpe:.1%}'
              f'  ({ summary["Sharpe > SPY"].sum() }/{ total_years } years)')
    
    # average excess returns against benchmarks
    avg_excess_spy = (summary['Opt Return'] - summary['SPY Return']).mean()
    avg_excess_eq = (summary['Opt Return'] - summary['Eq Return']).mean()
    
    if verbose:
        print(f'\nAverage Excess Return vs SPY  : {avg_excess_spy:5.2%}')
        print(f'Average Excess Return vs Eq-Wt: {avg_excess_eq:5.2%}')
        
        print(f'Average Out-of-Sample Sharpe  : {summary["Opt Sharpe"].mean():.3f}')
    
    # max drawdowns
    opt_max_drawdown = max_drawdown(opt_portfolio_series)
    eq_max_drawdown = max_drawdown(eq_portfolio_series)
    
    if verbose:
        print(f'\nMax Drawdown (Optimized)    : {opt_max_drawdown:.2%}')
        print(f'Max Drawdown (Equal-Weight) : {eq_max_drawdown:.2%}')
    
    # equity curves 
    opt_curve = opt_portfolio_series / opt_portfolio_series.iloc[0]
    eq_curve = eq_portfolio_series / eq_portfolio_series.iloc[0]
    spy_curve = spy_portfolio_series / spy_portfolio_series.iloc[0]
    
    # final wealth values
    final_wealth_opt = opt_curve.iloc[-1] * STARTING_CAPITAL
    final_wealth_eq = eq_curve.iloc[-1] * STARTING_CAPITAL
    final_wealth_spy = spy_curve.iloc[-1] * STARTING_CAPITAL
    
    if verbose:    
        print(f'\n=== Final Value of ${STARTING_CAPITAL} Invested ===')
        print(f'Optimized Portfolio : ${final_wealth_opt:5.2f}')
        print(f'Equal-Weighted      : ${final_wealth_eq:5.2f}')
        print(f'SPY                 : ${final_wealth_spy:5.2f}')
        
    # average turnover, effective number of bets, and stocks at max weight
    avg_effective_n_opt = summary['Opt Effective N'].mean()
    avg_turnover_opt = summary['Opt Turnover'].mean()
    avg_stocks_at_cap = summary['Stocks at Cap'].mean()
    
    if verbose:
        print('\n=== Portfolio Structure Metrics ===')
        
        print(f'Average Effective N : {avg_effective_n_opt:7.4f}')
        print(f'Average Turnover    : {avg_turnover_opt:7.4f}')
    
    # === PLOT ===
    cov_label = "Ledoit-Wolf Σ" if cov_shrinkage else "Sample Σ"
    mean_label = "Shrunk μ" if mean_shrinkage else "Sample μ" 
    
    experiment_label = f"MVO | {mean_label} + {cov_label} | max wt = {MAX_WEIGHT:.1%}"
    
    file_mean = 'shrunk_mu' if mean_shrinkage else 'sample_mu'
    file_cov = 'ledoit_sigma' if cov_shrinkage else 'sample_sigma'
    filename = PLOTS_DIR / f'{file_mean}_{file_cov}_{int(MAX_WEIGHT*100)}pct.png'
    
    save_backtest_plot(opt_curve, eq_curve, spy_curve, 
                       experiment_label, filename, verbose)

    return {
        
        'annual_results': summary,
        
        'performance': {
            'avg_excess_spy': avg_excess_spy,
            'avg_excess_eq' : avg_excess_eq,
            
            'avg_sharpe': summary['Opt Sharpe'].mean(),
            
            'wins_spy_return': wins_spy_return,
            'wins_eq_return' : wins_eq_return,
            
            'wins_spy_sharpe': wins_spy_sharpe,
            'wins_eq_sharpe' : wins_eq_sharpe,
                
            'win_rate_spy_return': win_rate_spy_return,
            'win_rate_eq_return' : win_rate_eq_return,
                
            'win_rate_spy_sharpe': win_rate_spy_sharpe,   
            'win_rate_eq_sharpe' : win_rate_eq_sharpe,
            
            'total_years' : total_years
            },
            
        'risk': {
            'max_drawdown_opt': opt_max_drawdown,
            
            'max_drawdown_eq' : eq_max_drawdown
            },
        
        'structure': {
            'avg_effective_n': avg_effective_n_opt,
            'avg_turnover': avg_turnover_opt,
            'avg_stocks_at_cap': avg_stocks_at_cap
            },
        
        'wealth': {
            'optimized': final_wealth_opt,
            'equal_weight': final_wealth_eq,
            'spy': final_wealth_spy
            }
        }
    

#####################
# 8. Experiment Framework
##################### 

def build_experiment_summary(experiments):
    """
    Build a comparison table across multiple experiments.
    
    Inputs:
        experiments (dict):
            Mapping experiment name to backtest result dictionary
        
            Example:
                {
                    'sample-sample': result dict,
                    'sample-ledoit': result dict,
                    ...
                }

    Returns:
        pd.DataFrame -> Summary table where:
            - columns = experiemnts
            - rows = aggregate performance metrics

    """
    rows = {}
    
    for name, r in experiments.items():
        
        rows[name] = {
            'Final Wealth': r['wealth']['optimized'],
            
            'Avg Sharpe': r['performance']['avg_sharpe'],
            
            'Avg Excess Return vs SPY': r['performance']['avg_excess_spy'],
            
            'Avg Excess Return vs Eq-Wt': r['performance']['avg_excess_eq'],
            
            'Win Rate vs SPY (Return)': 
                f"{r['performance']['wins_spy_return']}/"
                f"{r['performance']['total_years']} "
                f"({r['performance']['win_rate_spy_return']:.1%})",
            
            'Win Rate vs Eq-Wt (Return)': 
                f"{r['performance']['wins_eq_return']}/"
                f"{r['performance']['total_years']} "
                f"({r['performance']['win_rate_eq_return']:.1%})",

            'Win Rate vs SPY (Sharpe)': 
                f"{r['performance']['wins_spy_sharpe']}/"
                f"{r['performance']['total_years']} "
                f"({r['performance']['win_rate_spy_sharpe']:.1%})",
            
            'Win Rate vs Eq-Wt (Sharpe)': 
                f"{r['performance']['wins_eq_sharpe']}/"
                f"{r['performance']['total_years']} "
                f"({r['performance']['win_rate_eq_sharpe']:.1%})",
                
            'Max Drawdown': r['risk']['max_drawdown_opt'],
            
            'Avg Effective N': r['structure']['avg_effective_n'],
            
            'Avg Turnover': r['structure']['avg_turnover'],
            
            'Avg Stocks at Cap': r['structure']['avg_stocks_at_cap']
            }
            
    return pd.DataFrame(rows)


def format_experiment_table(df):
    """
    Create a formatted copy of experiment results for display purposes.
    Leaves original DataFrame unchanged.
    """
    
    formatted = df.copy().astype(object)
    
    # dollar values
    formatted.loc['Final Wealth'] = (
        formatted.loc['Final Wealth'].map( lambda x: f'${x:,.2f}')
        )
    
    # percentage metrics
    percent_rows = [
        'Avg Excess Return vs SPY',
        'Avg Excess Return vs Eq-Wt',
        'Max Drawdown',
        'Avg Turnover'
        ]
    
    for row in percent_rows:
        formatted.loc[row] = (
            formatted.loc[row].map(lambda x: f'{x:.1%}')
            )
        
    # normal decimals
    decimal_rows = [
        'Avg Effective N',
        'Avg Stocks at Cap'
    ]
    
    for row in decimal_rows:
        formatted.loc[row] = (
            formatted.loc[row].map(lambda x: f'{x:.1f}')
            )
        
    # return the copy DataFrame which has been newly formatted DataFrame
    return formatted
    
    
    
#####################
# 9. Execute
#####################  
if __name__ == '__main__':
    
    verbose = False
    START_YEAR = 2010   # this is first training year
    END_YEAR = 2024     # this is last training year -> test year will be +1
    
    experiments = {
        
        'Sample μ - Sample Σ': run_walk_forward_backtest(
            start_year = START_YEAR,
            end_year = END_YEAR,
            cov_shrinkage = False,
            mean_shrinkage = False,
            verbose = verbose
            ),
            
        'Sample μ - Ledoit Σ': run_walk_forward_backtest(
            start_year = START_YEAR,
            end_year = END_YEAR,
            cov_shrinkage = True,
            mean_shrinkage = False,
            verbose = verbose
            ),
        
        'Shrunk μ - Sample Σ': run_walk_forward_backtest(
            start_year = START_YEAR,
            end_year = END_YEAR,
            cov_shrinkage = False,
            mean_shrinkage = True,
            verbose = verbose
            ),
            
        'Shrunk μ - Ledoit Σ': run_walk_forward_backtest(
            start_year = START_YEAR,
            end_year = END_YEAR,
            cov_shrinkage = True,
            mean_shrinkage = True,
            verbose = verbose)  
        }
    
    comparison_table = build_experiment_summary(experiments)
    
    print('\n')
    print('=' * 67)
    print('EXPERIMENT COMPARISON (Mean-Variance combination)')
    print('=' * 67)
    
    # format the DataFrame using helper function
    display_table = format_experiment_table(comparison_table)
    
    print(display_table.to_string())
    