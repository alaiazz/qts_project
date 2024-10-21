import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def downside_beta(ret: pd.Series, mkt_ret: pd.Series,
                  ann_factor: int = 252) -> float:
    """
    Calculate downside beta = correlation of strategy return and market
    return when market return < 0

    Parameters
    ----------
    ret: pd.Series of strategy return;
    mkt_ret: pd.Series of market return, same interval as ret;
    ann_factor: annualizing factor.

    Returns
    -------
    downside beta value
    """
    idx = ret.index.intersection(mkt_ret.loc[mkt_ret < 0].index)
    return ret.loc[idx].corr(mkt_ret.loc[idx])


def sharpe_ratio(ret: pd.Series, rf_ret: pd.Series = None,
                 ann_factor: int = 252) -> float:
    """
    Calculate sharpe ratio

    Parameters
    ----------
    ret: pd.Series of strategy return;
    rf_ret: pd.Series of risk-free rate/return, same interval as ret;
    ann_factor: annualizing factor.

    Returns
    -------
    sharpe ratio
    """
    if rf_ret is not None:
        ret = ret - rf_ret
    return ret.mean() / ret.std() * np.sqrt(ann_factor)


def sortino_ratio(ret: pd.Series, rf_ret: pd.Series = None,
                  ann_factor: int = 252) -> float:
    """
    Calculate sortino ratio

    Parameters
    ----------
    ret: pd.Series of strategy return;
    rf_ret: pd.Series of risk-free rate/return, same interval as ret;
    ann_factor: annualizing factor.

    Returns
    -------
    sortino ratio
    """
    if rf_ret is not None:
        ret = ret - rf_ret

    return ret.mean() / ret.loc[ret < 0].std() * np.sqrt(ann_factor)


def var(s: pd.Series, ann_factor: int, q: float = 0.05) -> float:
    """
    Calculate VaR = n-th quantile

    Parameters:
        s (pd.Series):
            Return series of certain asset;
        ann_factor (float):
            Annualization factor to match return and Fama-French data;
        q (float):
            quantile used to calculate VaR.

    Returns:
        VaR value of input asset returns
    """
    return np.quantile(s, q=q) * np.sqrt(ann_factor)


def cvar(s: pd.Series, ann_factor: int, q: float = 0.05) -> float:
    """
    Calculate the mean of the returns at or below the q quantile

    Parameters:
        s (pd.Series):
            Return series of certain asset;
        ann_factor (float):
            Annualization factor to match return and Fama-French data;
        q (float):
            Quantile used to calculate CVaR.

    Returns:
        CVaR value of input asset returns
    """
    return s.loc[s < np.quantile(s, q=q)].mean() * np.sqrt(ann_factor)


# %%
def max_drawdown(s: pd.Series, return_dict: bool = False) -> float | pd.Series:
    """
    Calculate the maximum drawdown, peak date, trough date, and recovery date

    Parameters:
        s (pd.Series):
            Return series of certain asset

    Returns:
        pd.Series of all statistics of given asset
    """
    s_cum = (s + 1).cumprod()
    s_cum_max = s_cum.cummax()
    pct_to_peak = s_cum / s_cum_max - 1
    drawdown = min(pct_to_peak)

    if return_dict:
        trough_date = pct_to_peak[pct_to_peak == drawdown].index[0]
        peak_cum = s_cum_max[pct_to_peak == drawdown][0]
        peak_date = s_cum[s_cum == peak_cum].index[0]
        is_recovered = ((s_cum.index > trough_date) &
                        (s_cum >= peak_cum))
        recovery_date = s_cum.loc[is_recovered].index[0] \
            if any(is_recovered) else None

        return pd.Series({
            "drawdown": drawdown,
            "trough_date": trough_date,
            "peak_date": peak_date,
            "recovery_date": recovery_date
        })
    else:
        return drawdown


def ff_decomposition(s: pd.Series, df_ff: pd.DataFrame, ann_factor: int):
    """
    Decomposition return series with Fama-French 3-factor model

    Parameters
    ----------
    s: Return series of certain asset;
    df_ff: Fama-French factor data;
    ann_factor: annualization factor to match return and Fama-French data.

    Returns
    -------
    dictionary of alpha and beta
    """
    idx = s.index.intersection(df_ff.index)
    s = s.reindex(idx)
    df_ff = df_ff.reindex(idx)

    df_ff = df_ff / ann_factor  # Use daily data for FF & our returns
    ols = LinearRegression().fit(df_ff.loc[s.index] / ann_factor, s)
    return {
        "alpha": ols.intercept_ * ann_factor,
        **{f"beta_{k}": v for k, v in zip(df_ff.columns.values, ols.coef_)}
    }


def eval_return(cum_ret: pd.Series,
                df_ff: pd.DataFrame,
                ann_factor: float = 252):
    """
    Calculate metrics to compare the results with different parameters:
    1. cumulative total return
    2. Sharpe ratio (with risk-free rate / market return as benchmark)
    3. Sortino ratio
    4. Alpha
    5. Beta for 3 Fama-French factors
    6. Tail - VaR
    7. Tail - CVaR
    8. Tail - Max Drawdown
    9. Risk - Downside beta

    Parameters
    ----------
    cum_ret: pd.Series of cumulative returns from CalcTrade();
    df_ff: pd.DataFrame of daily Fama-French factor data;
    ann_factor: annualization factor.

    Returns
    -------
    dict of metrics
    """
    ret = cum_ret.diff().dropna()
    idx = ret.index.intersection(df_ff.index)
    ret = ret.reindex(idx)
    df_ff = df_ff.reindex(idx)

    mkt_ret = df_ff["Mkt-RF"] + df_ff["RF"]

    return {
        "return": cum_ret.iloc[-1],
        "mean": ret.mean(),
        "std": ret.std(),
        "skew": ret.skew(),
        "kurtosis": ret.kurt(),
        "sharpe": sharpe_ratio(ret, df_ff["RF"] / ann_factor, ann_factor),
        "sharpe_mkt": sharpe_ratio(ret, mkt_ret / ann_factor, ann_factor),
        "sortino": sortino_ratio(ret, df_ff["RF"] / ann_factor, ann_factor),
        "var": var(ret, ann_factor),
        "cvar": cvar(ret, ann_factor),
        "max_drawdown": max_drawdown(ret),
        "downside_beta": downside_beta(ret, mkt_ret / ann_factor),
    }
