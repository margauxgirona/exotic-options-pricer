import numpy as np
from scipy.stats import norm


def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate d1 parameter in Black-Scholes formula.
    
    Parameters
    ----------
    S : float
        Current price of the underlying asset (spot price)
    K : float
        Strike price of the option
    T : float
        Time to maturity in years
    r : float
        Risk-free interest rate (annualized, as decimal)
    sigma : float
        Volatility of the underlying asset (annualized, as decimal)
    
    Returns
    -------
    float
        The d1 parameter
    """
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate d2 parameter in Black-Scholes formula.
    
    Parameters
    ----------
    S : float
        Current price of the underlying asset (spot price)
    K : float
        Strike price of the option
    T : float
        Time to maturity in years
    r : float
        Risk-free interest rate (annualized, as decimal)
    sigma : float
        Volatility of the underlying asset (annualized, as decimal)
    
    Returns
    -------
    float
        The d2 parameter
    """
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate the price of a European call option using Black-Scholes.
    
    Parameters
    ----------
    S : float
        Current price of the underlying asset (spot price)
    K : float
        Strike price of the option
    T : float
        Time to maturity in years
    r : float
        Risk-free interest rate (annualized, as decimal)
    sigma : float
        Volatility of the underlying asset (annualized, as decimal)
    
    Returns
    -------
    float
        The price of the European call option
    
    Example
    -------
    >>> call_price(100, 100, 1, 0.05, 0.20)
    10.450583572185565
    """
    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)
    
    return S * norm.cdf(d1_val) - K * np.exp(-r * T) * norm.cdf(d2_val)


def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate the price of a European put option using Black-Scholes.
    
    Parameters
    ----------
    S : float
        Current price of the underlying asset (spot price)
    K : float
        Strike price of the option
    T : float
        Time to maturity in years
    r : float
        Risk-free interest rate (annualized, as decimal)
    sigma : float
        Volatility of the underlying asset (annualized, as decimal)
    
    Returns
    -------
    float
        The price of the European put option
    
    Example
    -------
    >>> put_price(100, 100, 1, 0.05, 0.20)
    5.573526022256971
    """
    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)
    
    return K * np.exp(-r * T) * norm.cdf(-d2_val) - S * norm.cdf(-d1_val)


def call_put_parity_check(S: float, K: float, T: float, r: float, sigma: float) -> dict:
    """
    Verify the call-put parity: C - P = S - K * exp(-rT)
    
    Parameters
    ----------
    S, K, T, r, sigma : float
        Standard Black-Scholes parameters
    
    Returns
    -------
    dict
        Contains call price, put price, left side (C-P), right side (S - K*exp(-rT)),
        and whether parity holds (difference < 1e-10)
    """
    call = call_price(S, K, T, r, sigma)
    put = put_price(S, K, T, r, sigma)
    
    left_side = call - put
    right_side = S - K * np.exp(-r * T)
    
    return {
        "call": call,
        "put": put,
        "C - P": left_side,
        "S - K*exp(-rT)": right_side,
        "parity_holds": abs(left_side - right_side) < 1e-10
    }


# Test the implementation
if __name__ == "__main__":
    # Parameters from our example
    S = 100     # Spot price
    K = 100     # Strike price
    T = 1       # 1 year to maturity
    r = 0.05    # 5% risk-free rate
    sigma = 0.20  # 20% volatility
    
    print("Black-Scholes Option Pricing")
    print("=" * 40)
    print(f"Spot price (S):     {S}")
    print(f"Strike price (K):   {K}")
    print(f"Time to maturity:   {T} year")
    print(f"Risk-free rate:     {r:.1%}")
    print(f"Volatility:         {sigma:.1%}")
    print("=" * 40)
    print(f"d1 = {d1(S, K, T, r, sigma):.4f}")
    print(f"d2 = {d2(S, K, T, r, sigma):.4f}")
    print(f"Call price: {call_price(S, K, T, r, sigma):.2f} €")
    print(f"Put price:  {put_price(S, K, T, r, sigma):.2f} €")
    print("=" * 40)
    
    # Verify call-put parity
    parity = call_put_parity_check(S, K, T, r, sigma)
    print("Call-Put Parity Check:")
    print(f"C - P = {parity['C - P']:.4f}")
    print(f"S - K*exp(-rT) = {parity['S - K*exp(-rT)']:.4f}")
    print(f"Parity holds: {parity['parity_holds']}")