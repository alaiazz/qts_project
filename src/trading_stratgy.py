from functools import partial
from datetime import datetime

import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.preprocessing import scale
from scipy.optimize import minimize

from .download_data import get_rate


def calc_position_by_rank(
        df_ratio: pd.DataFrame,
        df_price: pd.DataFrame,
        rank_formula: str,
        ub: float = 0.1,
        lb: float = 0,
        interval: str = "1m",
        use_change: bool = False,
) -> pd.DataFrame:
    """
    Calculate position by ranking stocks on each rebalance date
    (resample by "interval") and find lowest / highest (lb ~ ub)%
    to long / short.

    We calculate the aggregate score using "rank_formula".

    After selection, we simply allocates assets by 1/N policy.

    Parameters
    ----------
    df_ratio: dataframe for all ratios
        Index: DatetimeIndex
        Columns: (ratio, ticker);
    df_price: dataframe for all prices
        Index: DatetimeIndex
        Columns: (ratio, ticker);
    rank_formula: calculate ranking score with df.evel(), e.g. dm - roi;
    lb: find top / bottom (ub ~ lb)%, e.g. top decile: lb = 0;
    ub: find top / bottom (ub ~ lb)%, e.g. top decile: ub = 10%;
    interval: resample interval, should be str acceptable by pd.resample(),
        i.e, xMS for monthly, xW-MON for weekly.
    use_change: if True, calculate ratio change and rank;

    Returns
    -------
    pd.DataFrame:
        Index: DatatimeIndex of all resample date
        Column: All tickers
        Values: 1 means long position, -1 means short position.
    """

    close_date = df_ratio.index.max()
    if interval is not None:
        idx = pd.date_range(df_ratio.index.min(), close_date, freq=interval)
        df_ratio = df_ratio.loc[idx]
    else:
        idx = df_ratio.index[(df_ratio.index >= df_price.index.min()) &
                             (df_ratio.index <= df_price.index.max())]
        if datetime(2024, 2, 29) in idx:
            print(idx)

    if use_change:
        df_ratio = df_ratio.diff().iloc[1:]
        df_ratio = df_ratio.replace(np.inf, np.nan).replace(-np.inf, np.nan)

    # Calculate ranking score
    score = df_ratio.stack()
    if isinstance(score, pd.DataFrame):
        score = score.eval(f"score = {rank_formula}")["score"]
    score = score.unstack(level=0)

    # Find filter quantile
    lb_long_quantile = score[idx].quantile(1 - lb)
    ub_long_quantile = score[idx].quantile(1 - ub)
    lb_short_quantile = score[idx].quantile(lb)
    ub_short_quantile = score[idx].quantile(ub)

    # Filter price
    df_long = (
        (score[idx] >= ub_long_quantile).T &
        (score[idx] <= lb_long_quantile).T
    ).astype(int)
    df_short = -(
        (score[idx] >= lb_short_quantile).T &
        (score[idx] <= ub_short_quantile).T
    ).astype(int)

    return df_long + df_short


def portfolio_equal_weighted(
    df_price: pd.DataFrame,
    position_size: float = 1e6,
    position_indicator: pd.DataFrame = None,
    *args, **kwargs,
):
    """
    Calculate dollar position from position indicator allocated equal-weighted

    Parameters
    ----------
    df_price: dataframe for all prices
        Index: DatetimeIndex
        Columns: (ratio, ticker);
    position_size: gross notional of long / short position.
    position_indicator: pd.DataFrame
        Index: DatatimeIndex of all resample date
        Column: All tickers
        Values: 1 means long position, -1 means short position.
        If None, run and get calc_position_by_rank() returns.
    args, kwargs:
        Other Parameters for calc_position_by_rank function

    Returns
    -------
    pd.DataFrame:
        Index: DatatimeIndex of all resample date
        Column: All tickers
        Values: dollar value of position for each (ticker, date)
    """
    if position_indicator is None:
        position_indicator = calc_position_by_rank(
            df_price=df_price, *args, **kwargs,
        )

    df_price = df_price.loc[position_indicator.index]
    df_long = df_price.where(position_indicator == 1, np.nan)
    df_short = -df_price.where(position_indicator == -1, np.nan)

    # Allocate by 1/N policy
    long_size = position_size / df_long.count(axis=1)
    short_size = position_size / df_short.count(axis=1)

    # Calculate position by size / price
    position = ((long_size / df_long.T).fillna(0) +
                (short_size / df_short.T).fillna(0)).T

    # close all position in the end
    position.iloc[-1] = 0

    return position


def optimize_portfolio(
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float = 0,
        lambda_: float = 0.5,
        positive_weight: bool = True,
):
    """
     Optimize portfolio weights by minimize weighted portfolio
     Parameters
     ----------
     expected_returns: pd.Series
         Expected return series (N,)
     cov_matrix: pd.DataFrame
         Covariance metrix (N,N)
     risk_free_rate: float
         risk-free rate
     lambda_: float
         Risk averse parameter. Used to adjust non-diagonal value of cov matrix

     Returns
     -------
     pd.Series: weighting fo each ticker in expected_returns
     """
    # Adjust covariance matrix to avoid overfitting
    diagonal = np.eye(cov_matrix.shape[0], dtype=bool)
    adj_cov_matrix = cov_matrix.where(diagonal, cov_matrix * lambda_)

    # Portfolio return and volatility
    n = len(expected_returns)
    weights = cp.Variable(n)
    ret = expected_returns.values - risk_free_rate
    cov = adj_cov_matrix.values
    portfolio_return = cp.sum(cp.multiply(ret, weights))
    portfolio_volatility = cp.quad_form(weights, cov)
    objective = portfolio_return - portfolio_volatility

    # Constraints: weights sum to 1 and are all >= 0 (long-only)
    constraints = [cp.sum(weights) == 1]
    if positive_weight:
        constraints.append(weights >= 0)

    # Problem definition
    try:
        problem = cp.Problem(cp.Maximize(objective), constraints)
        problem.solve()
        return pd.Series(weights.value, index=expected_returns.index)
    except Exception as e:
        print(e)
        return pd.Series(1/n, index=expected_returns.index)


def portfolio_markovitz_optimized(
        df_ratio: pd.DataFrame,
        df_price: pd.DataFrame,
        estimation_month: int = 12,
        min_weight: float = 1e-3,
        position_size: float = 1e6,
        position_indicator: pd.DataFrame = None,
        optimize_kwargs: dict = {},
        *args, **kwargs,
):
    """
    Calculate position with CAPM optimization

    Parameters
    ----------
    df_ratio: pd.DataFrame
        Index: pd.DatetimeIndex (monthly)
        Columns: (ratio, ticker);
    df_price: pd.DataFrame
        Index: pd.DatetimeIndex (monthly)
        Columns: (ratio, ticker);
    df_ff: pd.DataFrame
        Fama French factor dataframe
        Index: pd.DatatimeIndex (monthly)
        columns: at least (Mkt-RF, RF)
    estimation_month: int
        Use past n monthly return to estimate market beta / covariance matrix
    min_weight: float
        Minimum portfolio weighting to hold, default = 1e-3
    optimize_kwargs: dict
        Dictionary of kwargs of optimize_portfolio(),
        e.g. lambda_ / positive_weight
    other parameters see portfolio_equal_weighted()

    Returns
    -------
    pd.Series: weighting fo each ticker in expected_returns
    """
    if position_indicator is None:
        position_indicator = calc_position_by_rank(
            df_price=df_price, *args, **kwargs,
        )

    all_position = {}
    for t, row in position_indicator.iterrows():
        all_position[t] = {}
        for i in [1, -1]:
            # Find list of ticker to long / short
            tic = row.where(lambda x: x == i, np.nan).dropna().index

            # Extra dataframe slice to calculate beta / corr matrix
            ret_df = df_ratio["ret_0_1"].loc[
                         (df_ratio.index.get_level_values("date") <= t) &
                         (df_ratio.index.get_level_values("ticker").isin(tic))
                         ].unstack().iloc[-estimation_month:]
            cov = ret_df.cov().dropna(how="all").dropna(how="all", axis=1)
            ret_df = ret_df[cov.columns]

            # Calculate optimized allocation
            weight = optimize_portfolio(
                expected_returns=ret_df.mean(),
                cov_matrix=cov,
                **optimize_kwargs
            ).sort_values(ascending=False)
            weight = weight.loc[weight.abs() > min_weight]

            # calculate position (# stocks)
            position = (i * weight * position_size
                        / df_price.loc[t, weight.index])
            all_position[t].update(position.to_dict())

    # close all position in the end
    all_position = pd.DataFrame(all_position).reindex(
        index=position_indicator.columns).T
    all_position.iloc[-1] = 0
    return all_position.fillna(0)


def portfolio_capm_optimized(
        df_ratio: pd.DataFrame,
        df_price: pd.DataFrame,
        df_ff: pd.DataFrame,
        estimation_month: int = 12,
        min_weight: float = 1e-3,
        position_size: float = 1e6,
        position_indicator: pd.DataFrame = None,
        *args, **kwargs,
):
    """
    Calculate position with CAPM optimization

    Parameters
    ----------
    df_ratio: pd.DataFrame
        Index: pd.DatetimeIndex (monthly)
        Columns: (ratio, ticker);
    df_price: pd.DataFrame
        Index: pd.DatetimeIndex (monthly)
        Columns: (ratio, ticker);
    df_ff: pd.DataFrame
        Fama French factor dataframe
        Index: pd.DatatimeIndex (monthly)
        columns: at least (Mkt-RF, RF)
    estimation_month: int
        Use past n monthly return to estimate market beta / covariance matrix
    min_weight: float
        Minimum portfolio weighting to hold, default = 1e-3
    other parameters see portfolio_equal_weighted()

    Returns
    -------
    pd.Series: weighting fo each ticker in expected_returns
    """
    if position_indicator is None:
        position_indicator = calc_position_by_rank(
            df_price=df_price, *args, **kwargs,
        )

    all_position = {}
    for t, row in position_indicator.iterrows():
        all_position[t] = {}
        for i in [1, -1]:
            # Find list of ticker to long / short
            tic = row.where(lambda x: x == i, np.nan).dropna().index

            # Extra dataframe slice to calculate beta / corr matrix
            ret_df = df_ratio["ret_0_1"].loc[
                         (df_ratio.index.get_level_values("date") <= t) &
                         (df_ratio.index.get_level_values("ticker").isin(tic))
                         ].unstack().iloc[-estimation_month:]
            cov = ret_df.cov().dropna(how="all").dropna(how="all", axis=1)
            ret_df = ret_df[cov.columns]

            mkt_df = df_ff.loc[ret_df.index, "Mkt-RF"] / 100

            # Calculate beta -> expected return
            beta = ret_df.corrwith(mkt_df) * mkt_df.std()
            excess_ret = beta * df_ff.loc[t, "Mkt-RF"]

            # Calculate optimized allocation
            weight = optimize_portfolio(
                expected_returns=excess_ret,
                cov_matrix=cov,
            ).sort_values(ascending=False)
            weight = weight.loc[weight.abs() > min_weight]

            # calculate position (# stocks)
            position = (i * weight * position_size
                        / df_price.loc[t, weight.index])
            all_position[t].update(position.to_dict())

    # close all position in the end
    all_position = pd.DataFrame(all_position).reindex(
        index=position_indicator.columns).T
    all_position.iloc[-1] = 0
    return all_position.fillna(0)


def calc_position_by_industry(
        s_sector: pd.Series,
        df_ratio: pd.DataFrame,
        df_price: pd.DataFrame,
        rank_formula: str,
        ub: float = 0.1,
        lb: float = 0,
        interval: str = "1m",
        use_change: bool = False,
) -> pd.DataFrame:
    """

    Parameters
    ----------
    s_sector: pd.Series
        Index: str tickers
        value: zacks_sector_code;
    other parameters refer to calc_position_by_rank().

    Returns
    -------
    pd.DataFrame:
        Index: DatatimeIndex of all resample date
        Column: All tickers
        Values: 1 means long position, -1 means short position.
    """
    position_indicator_list = []
    for s in sorted(s_sector.unique()):
        ticker = list(s_sector.loc[s_sector == s].index)
        position_indicator = calc_position_by_rank(
            df_price=df_price.filter(ticker),
            df_ratio=df_ratio["predicted_ret"].filter(ticker).dropna(how="all"),
            rank_formula=rank_formula,
            lb=ub,
            ub=lb,
            interval=interval,
            use_change=use_change,
        )
        position_indicator_list.append(position_indicator)
    return pd.concat(position_indicator_list, axis=1)


class CalcTrade:
    """
    Iterate daily position dataframe to evaluate position adjustment
    """

    def __init__(
            self,
            df_price: pd.DataFrame,
            cost: float = 0,
            init_cap: float = 0,
            adj_small_change: bool = False,
    ):
        """
        Parameters
        ----------
        df_price: pd.DataFrame
            pd.DataFrame of daily price data
            Columns: (tickers)
            Index: DatatimeIndex;
        cost: float
            dollar trading cost = gross_traded_cash * cost (ratio);
        init_cap: float
            initial capital to invest, used to calculate cash_balance /
            total_return;
        adj_small_change: bool
            If True, change position on every rebalance date, i.e. no long
            apply the rule to only change when ticker added / removed to
            long / short list.
        """
        self._df_price = df_price
        self._cost = cost
        self._init_cap = init_cap
        self._adj_small_change = adj_small_change
        self.__zero_position = pd.Series([0] * df_price.shape[1],
                                         index=df_price.columns)

    def __reset(self):
        """
        Reset cumulative values before each iteration
        """
        self.__position = {}
        self.__position_dollar = {}
        self.__trade = {}
        self.__trade_dollar = {}
        self.__detail = {}
        self._open_position = self.__zero_position.copy()

    @property
    def position(self):
        """
        position, (tickers): evaluated true position data;
        """
        return pd.DataFrame(self.__position).T

    @property
    def trade(self):
        """
        trade, (tickers): trade made on date;
        """
        return pd.DataFrame(self.__trade).T

    @property
    def position_dollar(self):
        """
        position_dollar, (tickers): $ true position;
        """
        return pd.DataFrame(self.__position_dollar).T

    @property
    def trade_dollar(self):
        """
        trade_dollar, (tickers):  $ trade amount,
        i.e. should be -position_dollar on traded days;
        """
        return pd.DataFrame(self.__trade_dollar).T

    def iter_dates(self, position: pd.DataFrame) -> pd.DataFrame:
        """
        Main function to iterate through dates

        Parameters
        ----------
        position:
            pd.DataFrame of calculate position (# shares) by spread from
            calc_position_by_spread() function;
        """
        self.__reset()

        df_price = self._df_price.loc[position.index]
        for date, pos in position.iterrows():
            price = df_price.loc[date]
            if self._adj_small_change:
                act_pos = pos
            else:
                act_pos = self._action_pos(pos)
            self._make_trade(price, act_pos, date)

        return self._final_df()

    def _action_pos(self, pos: pd.Series) -> pd.Series:
        """
        To reduce to turnover rate, we only add new position when
        ideal position has different sign, i.e.
        - pos / neg -> 0
        - pos <-> neg

        Returns
        -------
        pd.Series of actionable idea position
        """
        same_sign = np.sign(self._open_position) != np.sign(pos)
        return pos.where(same_sign, self._open_position).rename(None)

    def _final_df(self) -> pd.DataFrame:
        """
        Summary of balance / total return on each date

        Returns
        -------
        pd.DataFrame with Columns:
            trading_cost: $ trading cost;
            cash: cash balance at the end of each date;
            position: current position value at the end of each date;
            total: cash + position balance;
            total_ret: return = total balance / initial capital - 1;
        """

        df = pd.DataFrame({
            "trading_cost": self.trade_dollar.abs().sum(axis=1) * self._cost,
            "cash": self.trade_dollar.sum(axis=1).cumsum() + self._init_cap,
            "long_position": self.position_dollar.where(lambda x: x > 0).sum(axis=1),
            "short_position": self.position_dollar.where(lambda x: x < 0).sum(axis=1),
            "position": self.position_dollar.sum(axis=1),
        })
        df.loc[:, "cash"] -= df["trading_cost"]

        short_notional = -self.trade_dollar.iloc[0].where(lambda x: x < 0).sum()
        df.loc[:, "interest"] = calc_interest(df, short_notional=short_notional)
        df.loc[:, "total"] = df["cash"] + df["position"] + df["interest"]
        df.loc[:, "total_ret"] = df["total"] / self._init_cap - 1

        return df

    def _make_trade(
            self,
            price: pd.Series,
            pos: pd.Series,
            date: datetime,
    ):
        """
        Calculate trade made with open position and ideal position

        Parameters
        ----------
        price: pd.Series of current date price;
        pos: pd.Series of ideal position after trade;
        date: timestamp.
        """
        self.__position[date] = pos
        self.__position_dollar[date] = pos * price
        self.__trade[date] = (pos - self._open_position)
        self.__trade_dollar[date] = self.__trade[date] * price
        self._open_position = pos


def calc_interest(
        final_df: pd.DataFrame,
        short_buffer: float = 0.01,
        short_notional: float = 1e6,
) -> pd.Series:
    """
    Calculate interest earned during the period:
    - cash in hand interest = cash in hand (cash - repo cash) *
                              long rate (SOFR) * n_day / 365.25
    - reverse repo interest = reverse repo (position size) *
                              short rate (SOFR - short_buffer) * n_day / 365.25

    Why reverse repo = position size?
    - I target short position to maintain the same notional size. i.e. each
      ticker when enter into short position should have value =
      position_size / n_shares in short.
    - Since we also maintain same n_shares in short over entire testing period,
      reverse repo size = total short notional = position_size.

    Parameters
    ----------
    final_df: final summary df from CalcTrade();
    short_buffer: short rate = SOFR - short_buffer, default = 100bp;
    short_notional: target gross notional for long / short position.

    Returns
    -------
    pd.Series of interest amount
    """
    sofr = (get_rate(use_cache=True) / 100).rename("long_rate")
    final_df = final_df.merge(sofr, left_index=True, right_index=True,
                              how="left")
    final_df.loc[:, "n_day"] = -final_df.index.diff(-1).days

    final_df = final_df.eval(f"""
        short_rate = long_rate - {short_buffer}
        short_interest = {short_notional} * short_rate * n_day / 365.25
        long_interest = (cash - {short_notional}) * long_rate * n_day / 365.25
    """)

    return (final_df["short_interest"] + final_df["long_interest"]).fillna(0)

