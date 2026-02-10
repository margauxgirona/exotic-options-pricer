"""
Monte Carlo simulation for option pricing.
"""

import numpy as np
from scipy.stats import norm


def simulate_terminal_price(S: float, T: float, r: float, sigma: float, 
                            n_simulations: int, seed: int = None) -> np.ndarray:
    """
    Simulate terminal stock prices using Geometric Brownian Motion.
    
    This method jumps directly to maturity (no intermediate steps),
    which is efficient for vanilla European options.
    
    Parameters
    ----------
    S : float
        Current stock price
    T : float
        Time to maturity in years
    r : float
        Risk-free interest rate
    sigma : float
        Volatility
    n_simulations : int
        Number of price paths to simulate
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    np.ndarray
        Array of simulated terminal prices
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random normal numbers
    Z = np.random.standard_normal(n_simulations)
    
    # Calculate terminal prices using GBM formula
    S_T = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    return S_T


def simulate_price_paths(S: float, T: float, r: float, sigma: float,
                         n_simulations: int, n_steps: int, seed: int = None) -> np.ndarray:
    """
    Simulate full price paths using Geometric Brownian Motion.
    
    This method generates complete trajectories, needed for path-dependent
    options like barriers and Asians.
    
    Parameters
    ----------
    S : float
        Current stock price
    T : float
        Time to maturity in years
    r : float
        Risk-free interest rate
    sigma : float
        Volatility
    n_simulations : int
        Number of price paths to simulate
    n_steps : int
        Number of time steps in each path
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    np.ndarray
        Matrix of shape (n_simulations, n_steps + 1) containing price paths.
        First column is S (initial price), last column is S_T (terminal price).
    """
    if seed is not None:
        np.random.seed(seed)
    
    dt = T / n_steps
    
    # Initialize price matrix
    # Each row is a simulation, each column is a time step
    paths = np.zeros((n_simulations, n_steps + 1))
    paths[:, 0] = S  # All paths start at S
    
    # Generate all random numbers at once (more efficient)
    Z = np.random.standard_normal((n_simulations, n_steps))
    
    # Simulate step by step
    for t in range(1, n_steps + 1):
        paths[:, t] = paths[:, t-1] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t-1]
        )
    
    return paths


def monte_carlo_call(S: float, K: float, T: float, r: float, sigma: float,
                     n_simulations: int = 100000, seed: int = None) -> dict:
    """
    Price a European call option using Monte Carlo simulation.
    
    Parameters
    ----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to maturity in years
    r : float
        Risk-free interest rate
    sigma : float
        Volatility
    n_simulations : int
        Number of simulations (default 100,000)
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    dict
        Contains price, standard error, and 95% confidence interval
    """
    # Simulate terminal prices
    S_T = simulate_terminal_price(S, T, r, sigma, n_simulations, seed)
    
    # Calculate payoffs
    payoffs = np.maximum(S_T - K, 0)
    
    # Discount to present value
    discount_factor = np.exp(-r * T)
    discounted_payoffs = discount_factor * payoffs
    
    # Calculate statistics
    price = np.mean(discounted_payoffs)
    std_error = np.std(discounted_payoffs) / np.sqrt(n_simulations)
    
    # 95% confidence interval
    ci_lower = price - 1.96 * std_error
    ci_upper = price + 1.96 * std_error
    
    return {
        "price": price,
        "std_error": std_error,
        "ci_95": (ci_lower, ci_upper),
        "n_simulations": n_simulations
    }


def monte_carlo_put(S: float, K: float, T: float, r: float, sigma: float,
                    n_simulations: int = 100000, seed: int = None) -> dict:
    """
    Price a European put option using Monte Carlo simulation.
    
    Parameters
    ----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to maturity in years
    r : float
        Risk-free interest rate
    sigma : float
        Volatility
    n_simulations : int
        Number of simulations (default 100,000)
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    dict
        Contains price, standard error, and 95% confidence interval
    """
    # Simulate terminal prices
    S_T = simulate_terminal_price(S, T, r, sigma, n_simulations, seed)
    
    # Calculate payoffs
    payoffs = np.maximum(K - S_T, 0)
    
    # Discount to present value
    discount_factor = np.exp(-r * T)
    discounted_payoffs = discount_factor * payoffs
    
    # Calculate statistics
    price = np.mean(discounted_payoffs)
    std_error = np.std(discounted_payoffs) / np.sqrt(n_simulations)
    
    # 95% confidence interval
    ci_lower = price - 1.96 * std_error
    ci_upper = price + 1.96 * std_error
    
    return {
        "price": price,
        "std_error": std_error,
        "ci_95": (ci_lower, ci_upper),
        "n_simulations": n_simulations
    }


def convergence_analysis(S: float, K: float, T: float, r: float, sigma: float,
                         true_price: float, simulation_counts: list = None) -> list:
    """
    Analyze how Monte Carlo price converges to true price as simulations increase.
    
    Parameters
    ----------
    S, K, T, r, sigma : float
        Option parameters
    true_price : float
        The analytical Black-Scholes price for comparison
    simulation_counts : list
        List of simulation counts to test (default: 100 to 1M)
    
    Returns
    -------
    list
        List of dicts with n_simulations, mc_price, error, std_error
    """
    if simulation_counts is None:
        simulation_counts = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
    
    results = []
    for n in simulation_counts:
        mc_result = monte_carlo_call(S, K, T, r, sigma, n, seed=42)
        results.append({
            "n_simulations": n,
            "mc_price": mc_result["price"],
            "error": mc_result["price"] - true_price,
            "error_percent": abs(mc_result["price"] - true_price) / true_price * 100,
            "std_error": mc_result["std_error"]
        })
    
    return results


# Test the implementation
if __name__ == "__main__":
    # Same parameters as Black-Scholes test
    S = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.20
    
    # True prices from Black-Scholes (for comparison)
    bs_call = 10.450583572185565
    bs_put = 5.573526022256971
    
    print("Monte Carlo Option Pricing")
    print("=" * 50)
    print(f"Parameters: S={S}, K={K}, T={T}, r={r}, σ={sigma}")
    print(f"Black-Scholes call price: {bs_call:.4f} €")
    print(f"Black-Scholes put price:  {bs_put:.4f} €")
    print("=" * 50)
    
    # Price with Monte Carlo
    print("\nMonte Carlo Results (100,000 simulations):")
    print("-" * 50)
    
    mc_call = monte_carlo_call(S, K, T, r, sigma, n_simulations=100000, seed=42)
    print(f"Call price: {mc_call['price']:.4f} €")
    print(f"Std error:  {mc_call['std_error']:.4f}")
    print(f"95% CI:     [{mc_call['ci_95'][0]:.4f}, {mc_call['ci_95'][1]:.4f}]")
    print(f"Error vs BS: {mc_call['price'] - bs_call:.4f} €")
    
    print()
    
    mc_put = monte_carlo_put(S, K, T, r, sigma, n_simulations=100000, seed=42)
    print(f"Put price:  {mc_put['price']:.4f} €")
    print(f"Std error:  {mc_put['std_error']:.4f}")
    print(f"95% CI:     [{mc_put['ci_95'][0]:.4f}, {mc_put['ci_95'][1]:.4f}]")
    print(f"Error vs BS: {mc_put['price'] - bs_put:.4f} €")
    
    # Convergence analysis
    print("\n" + "=" * 50)
    print("Convergence Analysis (Call option)")
    print("-" * 50)
    print(f"{'Simulations':>12} | {'MC Price':>10} | {'Error':>10} | {'Error %':>8}")
    print("-" * 50)
    
    convergence = convergence_analysis(S, K, T, r, sigma, bs_call)
    for result in convergence:
        print(f"{result['n_simulations']:>12,} | {result['mc_price']:>10.4f} | "
              f"{result['error']:>+10.4f} | {result['error_percent']:>7.3f}%")