from datetime import datetime
import sys

import nasdaqdatalink
import pandas as pd
import numpy as np

from .download_data import (cache_df, START_DATE, END_DATE, get_price,
                            IGNORE_SECTOR, get_zacks)

ratio_col = [
    'curr_ratio', 'free_cash_flow_per_share', 'ret_equity', 'gross_margin', 'ret_asset',
    'tot_debt_tot_equity', 'oper_profit_margin',
    'invty_turn', 'ret_invst', 'pretax_profit_margin',
    'lterm_debt_cap', 'day_sale_rcv', 'ret_tang_equity', 'free_cash_flow',
    'profit_margin', 'rcv_turn', 'asset_turn', 'ebit_margin',
    'oper_cash_flow_per_share', 'book_val_per_share'
] + [
    'pe', 'illiquidity', 'cash_profit', 'excl_exp', 'rev_growth', 'rd_bias', 'asset_growth',
    'fluc', 'macd', 'bias', 'win_pct', 'quantile_95', 'kurt_skew', 'ulcer', 'price_pos', 'cmo', 'z_score',
    'V20', 'K20', 'TURN10', 'VEMA12', 'VSTD20', 'AR', 'BR', 'AU', 'AD', 'NPR', 'RPPS', 'Altman_Z_Score',
    'BIASVOL', 'ERBE', 'ERBU', 'FI', 'HMA', 'PVO', 'DPO', 'DC', 'alpha6', 'alpha12'
] + [
    "ret_0_1", "ret_1_6", "ret_6_12", "skew", "volume_shock",
    "rs_volatility", "p_volatility", "yz_volatility", "ret_autocorr",
    # "ret_consist" # remove because none of dates can do 3-qcut
]
# Removed:
# 1. insurance company ratios: loss_ratio, exp_ratio, comb_ratio


def find_valid_universe() -> list[str]:
    """
    Find valid ticker list by filtering available tickers in ZACKS:
    1. Listed on NASDAQ / NYSE
    2. active as of now
    3. belonging to S&P500 as of now (look ahead bias)
    4. not belonging to Finance / Automobile sector

    Returns
    -------
    list of valid tickers
    """

    # Get all Zacks universe
    all_tickers = pd.read_csv("https://static.quandl.com/coverage/ZACKS_FC.csv")

    # Filter 1: Listed in NASDAQ / NYSE
    # Filter 2: Active
    all_tickers = all_tickers.loc[
        all_tickers["exchange"].isin(["NASDAQ", "NYSE"]) &
        all_tickers["is_active"]
        ]

    # Get master table details
    master = nasdaqdatalink.get_table(
        'ZACKS/MT',
        ticker=','.join(all_tickers["ticker"].astype(str).to_list()))

    # Filter 3: Is common share
    master = master.loc[(master["asset_type"] == "COM") &
                        (~master["zacks_x_sector_code"].isin(IGNORE_SECTOR)) &
                        (master["sp500_member_flag"] == "Y")]

    return master["ticker"].to_list()


def fill_price_dates(g: pd.DataFrame, droplevel: bool=True):
    """
    Reindex dates to ffill prices on non-trading-day

    Parameters
    ----------
    g: pd.DataFrame
        main dataframe from merging price / zacks data of certain ticker
    droplevel: bool
        Default = True, drop "ticker" columns when apply this function through
          groupby to avoid duplicated "ticker" index.
        If directly use, set as False.

    Returns
    -------
    pd.DataFrame:
        Index: pd.DatetimeIndex daily from first data
    """
    dates = g.index.get_level_values("date")
    full_period = pd.date_range(start=dates.min(), end=dates.max())
    if droplevel:
        g = g.droplevel("ticker")
    return g.reindex(full_period)


def _stack(results):
    """ Reformat momentum function outputs """

    return results if isinstance(results, pd.Series) else results.stack()


class MomentumFactor:
    """
    Calculate momentum factors from prices
    """

    def calc_momentum_factor(self, df: pd.DataFrame):
        """
        Returns
        -------
        pd.DataFrame:
            price dataframe from get_price
            Index:
                ticker: str
                date: pd.DatatimeIndex for all trading day
        """
        df = df.groupby("ticker").apply(fill_price_dates)
        df.loc[:, "skew"] = _stack(df.groupby('ticker').apply(self.get_skew))
        df.loc[:, "rs_volatility"] = _stack(df.groupby('ticker')
                                            .apply(self.get_rs_vola))
        df.loc[:, "p_volatility"] = _stack(df.groupby('ticker')
                                           .apply(self.get_p_vola))
        df.loc[:, "yz_volatility"] = _stack(df.groupby('ticker').apply(
            self.get_yz_vola, p_vola=df["p_volatility"],
            rs_vola=df["rs_volatility"]))
        df.loc[:, "volume_shock"] = _stack(df.groupby("ticker")
                                           .apply(self.get_volume))
        df = self._calc_historic_stock_returns(df)

        df.loc[:, "ret_autocorr"] = _stack(df.groupby("ticker")
                                           .apply(self.get_ret_autocorr))
        df.loc[:, "ret_consist"] = _stack(df.groupby("ticker")
                                          .apply(self.get_ret_consistency))
        df = df.replace([np.inf, -np.inf], np.nan)
        df.index.names = ("ticker", "date")

        return df

    @staticmethod
    def get_rs_vola(df, halflife: int = 45, days_in_year: int = 252):
        """
        Calculate Rogers-Satchell volatility:

        ohlc_vola = average over period from start to end:
            Log(High/Open) * Log(High/Close) + Log(Low/Open) * Log(Open/Close)

        annualized = sqrt(daily*256)

        Parameters
        ----------
        df: pd.DataFrame
            groupby result price dataframe for each ticker
        halflife: int
            Default = 45, i.e. 45 days ~ mean over past 3 months
        days_in_year: int
            Annualize factor to convert daily volatility

        Returns
        -------

        """
        df = df.droplevel(0)
        open_data, high_data, low_data, close_data = df['adj_open'].values, df[
            'adj_high'].values, df['adj_low'].values, df['adj_close'].values

        # Calculate daily volatility
        log_hc_ratio = np.log(high_data / close_data)
        log_ho_ratio = np.log(high_data / open_data)
        log_lo_ratio = np.log(low_data / open_data)
        log_lc_ratio = np.log(low_data / close_data)
        sum_ = pd.Series(
            log_hc_ratio * log_ho_ratio + log_lo_ratio * log_lc_ratio,
            index=df.index,
        ) * np.sqrt(days_in_year)   # Calculate annualized volatility

        return sum_.ewm(halflife=halflife).mean()

    @staticmethod
    def get_yz_vola(
            df: pd.DataFrame,
            rs_vola: pd.Series,
            p_vola: pd.Series,
            halflife: int = 45,
            days_in_year: int = 252
    ):
        """
        Calculate Yang-Zhang (2000) volatility annualized

        Parameters
        ----------
        df: pd.DataFrame
            groupby result price dataframe for each ticker
        rs_vola: pd.Series
            Series of all Rogers-Satchell volatility
        p_vola: pd.Series
            Series of all Parkinson volatility
        halflife: int
            Default = 45, i.e. 45 days ~ mean over past 3 months
        days_in_year: int
            Annualize factor to convert daily volatility

        Returns
        -------

        """
        ticker = df.index.get_level_values(0)[0]
        df = df.droplevel(0)

        # Overnight volatility component (Annualized)
        log_oc = np.log(df['adj_open'] / df['adj_close'].shift(1))
        night_vola = ((log_oc ** 2).ewm(halflife=halflife).mean() *
                      np.sqrt(days_in_year))

        # Open-to-close volatility component (Annualized)
        log_co = np.log(df['adj_close'] / df['adj_open'])
        co_vola = ((log_co ** 2).ewm(halflife=halflife).mean() *
                   np.sqrt(days_in_year))

        # Yang-Zhang volatility estimator
        k = 0.34 / (1.35 + (halflife * 2 + 1) / (halflife * 2 - 1))
        vola_bar = (2*np.log(2) * p_vola.loc[ticker] - rs_vola.loc[ticker]
                    ) / (2 * np.log(2) - 1)

        sum_ = pd.Series(
            night_vola + k * co_vola * (1 - k) * vola_bar,
            index=df.index,
        )

        return sum_

    @staticmethod
    def get_p_vola(df, halflife: int = 45, days_in_year: int = 252):
        """
        Calculate Parkinson (1980) volatility
        = (Log(High/Open) - Log(Low/Open))^2 / 4ln2

        annualized = sqrt(daily*256)

        Parameters
        ----------
        df: pd.DataFrame
            groupby result price dataframe for each ticker
        halflife: int
            Default = 45, i.e. 45 days ~ mean over past 3 months
        days_in_year: int
            Annualize factor to convert daily volatility

        Returns
        -------

        """
        df = df.droplevel(0)
        open_data, high_data, low_data, close_data = df['adj_open'].values, df[
            'adj_high'].values, df['adj_low'].values, df['adj_close'].values

        # Calculate daily volatility
        log_ho_ratio = np.log(high_data / open_data)
        log_lo_ratio = np.log(low_data / open_data)
        sum_ = pd.Series(
            (log_ho_ratio - log_lo_ratio) ** 2 / 4 / np.log(2),
            index=df.index,
        ) * np.sqrt(days_in_year)   # Calculate annualized volatility

        return sum_.ewm(halflife=halflife).mean()

    @staticmethod
    def get_skew(df, days_in_year: int = 252):
        """
        Calculate past 1yr daily return skewness
        """
        df = df.droplevel(0)
        return df['adj_close'].dropna().pct_change().rolling(
            days_in_year, min_periods=days_in_year).skew()

    @staticmethod
    def get_ret_autocorr(df):
        """
        Calculate return predictability =
        serial correlations of Short-term returns

        References:
        Jegadeesh, Narasimhan, 1990, "Evidence of predictable behavior of security returns", Journal of Finance 45, 881-898.
        """
        df = df.droplevel(0)
        return df["ret_0_1"].rolling(12).apply(lambda x: x.autocorr())

    @staticmethod
    def get_ret_consistency(df):
        """
        Calculate return consistency = # month with postive return in past 6m

        References:
        Watkins, Boyce, 2003, "Riding the wave of sentiment: An analysis of return consistency as a predictor of future returns", Journal of Behavioral Finance 4, 191-200.
        """
        df = df.droplevel(0)
        return df.rolling(6)["ret_0_1"].apply(lambda x: (x > 0).sum())

    @staticmethod
    def get_volume(df):
        """
        Calculate volume shock indicator =
        EWM mean volume decayed by 3 days / EWM decayed by 45 days
        i.e. ~7 days / 3 month
        """
        df = df.droplevel(0)
        volume_1w = df['volume'].ewm(3).mean()
        volume_3m = df['volume'].ewm(45).mean()
        return volume_1w / volume_3m

    @staticmethod
    def _calc_historic_stock_returns(df):
        """
        calculate weekly / monthly return as independent variables
        """
        c = df.groupby("ticker")['adj_close'].ewm(3).mean().droplevel(0)

        def calc_return(g: pd.Series, start_month: int, end_month: int):
            g = g.droplevel("ticker").ffill()
            start_idx = g.index - pd.DateOffset(months=start_month)
            end_idx = g.index - pd.DateOffset(months=end_month)
            return pd.Series(
                np.log(g.reindex(end_idx).values /
                       g.reindex(start_idx).values),
                index=g.index
            )

        df.loc[:, "forward_ret"] = c.groupby("ticker").apply(
            calc_return, start_month=0, end_month=-1)
        df.loc[:, "ret_0_1"] = c.groupby("ticker").apply(
            calc_return, start_month=0, end_month=1)
        df.loc[:, "ret_1_6"] = c.groupby("ticker").apply(
            calc_return, start_month=1, end_month=6)
        df.loc[:, "ret_6_12"] = c.groupby("ticker").apply(
            calc_return, start_month=6, end_month=12)

        return df


def find_mkt_val_date_close(df_price: pd.DataFrame,
                            df_funda: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate mkt_val on non period end date

    Parameters
    ----------
    df_price: pd.DataFrame
        price dataframe
        Index: pd.DatatimeIndex daily
    df_funda: pd.DataFrame
        zacks dataframe for all fundamental data

    Returns
    -------
    pd.DataFrame:
        Columns:
            mkt_val_date_close: adj_close on mkt_val date, i.e. period_end
            ...
    """
    df = df_funda.reset_index().merge(
        df_price["adj_close"].rename("mkt_val_date_close"),
        left_on=["ticker", "per_end_date"],
        right_index=True, how="left"
    ).set_index(["ticker", "date"])

    return df


@cache_df("ratio")
def calc_ratio(use_data_cache: bool = True,
               tickers: list[str] = None,
               start_date: str | datetime = START_DATE,
               end_date: str | datetime = END_DATE) -> pd.DataFrame:
    """
    Calculate selected financial ratios

    Parameters
    ----------
    tickers: calculate ratios for given tickers;
    use_data_cache: use cached price / fundamental df;
    start_date: get price data from start_date to end_date (both inclusive);
    end_date: get price data from start_date to end_date (both inclusive).

    Returns
    -------
    pd.DataFrame: ratios
        Index:
            date: pd.DatetimeIndex at each month end
            ticker: str
    """

    # Make sure we get fundamental on start_date &
    # prices for calculate past 12 ~ 2 month return
    start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)
    funda_start_date = start_date - pd.offsets.DateOffset(months=12)
    kws = dict(use_cache=use_data_cache, tickers=tickers,
               start_date=funda_start_date, end_date=end_date)

    def filter_ticker(df: pd.DataFrame, tickers: list[str]):
        nonlocal funda_start_date, end_date
        return df.loc[df.index.get_level_values("ticker").isin(tickers) &
                      (df.index.get_level_values("date") >= funda_start_date) &
                      (df.index.get_level_values("date") <= end_date)]

    # in case (s_price, df_funda) read from cache, i.e. contain other tickers
    df_price = get_price(**kws).sort_index()
    no_gap_ticker = df_price.groupby("ticker").apply(
        lambda g: g.index.get_level_values("date").diff().max().days < 10)
    tickers = list(set(tickers) & set(no_gap_ticker.loc[no_gap_ticker].index))
    df_price = filter_ticker(df_price, tickers=tickers)
    df_price = MomentumFactor().calc_momentum_factor(df_price)
    df_price.loc[:, ["adj_close", "volume"]] = df_price.groupby("ticker")[[
        "adj_close", "volume"]].ffill()

    df_funda = filter_ticker(get_zacks(**kws).sort_index(), tickers=tickers)
    # Get price / period_end date price
    df_funda = find_mkt_val_date_close(df_price, df_funda).sort_index()

    df = df_price.merge(df_funda, left_index=True, right_index=True, how="outer")

    def resample_ffill(g: pd.DataFrame, limit: int = 91):
        """ ffill data and resample at monthly interval """
        return g.droplevel("ticker").ffill(limit=limit).resample("M").last()

    df = df.groupby("ticker").apply(resample_ffill)  # ffill for all ratio data

    # Calculate annual lag value
    for col in ["tot_revnu", "tot_share_holder_equity", "res_dev_exp", "tot_asset"]:
        df.loc[:, f"{col}_1y"] = df.groupby("ticker")[col].shift(12)

    # Fill 0 Columns
    for col in ["change_acct_rcv", "change_invty", "change_acct_pay_accrued_liab"]:
        df.loc[:, col] = df[col].fillna(0)

    # Calculate ratios
    df = df.eval("""
        current_mkt_val = mkt_val * adj_close / mkt_val_date_close
        tot_debt_tot_equity = tot_debt_tot_equity * mkt_val_date_close / adj_close
        lterm_debt_cap = lterm_debt_cap * (tot_lterm_debt + mkt_val) / (tot_lterm_debt + current_mkt_val)
        ret_invst = ret_invst * (tot_lterm_debt + mkt_val) / (tot_lterm_debt + current_mkt_val)
        asset_turn = asset_turn * tot_asset / (tot_liab + current_mkt_val)
        ret_equity = ret_equity * mkt_val_date_close / adj_close
        ret_tang_equity = ret_tang_equity * mkt_val_date_close / adj_close
        ret_asset = ret_asset * tot_asset / (tot_liab + current_mkt_val)
        pe = adj_close / eps_diluted_net
        illiquidity = volume / current_mkt_val
        cash_profit = (oper_income - change_acct_rcv - change_invty - change_acct_pay_accrued_liab) / tot_share_holder_equity_1y
        excl_exp = tot_non_oper_income_exp / tot_share_holder_equity_1y
        rev_growth = tot_revnu / tot_revnu_1y - 1
        rd_bias =  res_dev_exp / res_dev_exp_1y - 1 - rev_growth
        asset_growth = tot_asset / tot_asset_1y - 1
    """)
    df = df.swaplevel().sort_index().loc[start_date:end_date]

    # Combine factor calculated separately
    extra1 = pd.read_pickle("data/factors_JS.pkl").swaplevel()
    extra2 = (pd.read_pickle("data/factor_new_construct_z.pkl")
              .reset_index().drop_duplicates(subset=["date", "ticker"])
              .set_index(["date", "ticker"]))
    extra3 = pd.read_pickle("data/factor_TP.pkl").swaplevel()
    df = pd.concat([extra1, extra2, extra3, df], axis=1).sort_index()

    def exclude_ticker(g: pd.DataFrame):
        # Remove ticker with invalid D/E ratio (Price adj. D/E may < 0.1 again)
        if g["tot_debt_tot_equity"].max() < 0.1:
            return True

        # Remove ticker with Net Debt + Market Value < 0
        # (causing problem in ROI calculating)
        elif (g["net_lterm_debt"] + g["current_mkt_val"]).min() < 0:
            return True
        return False

    invalid_ticker = df.reset_index().groupby("ticker").apply(exclude_ticker)
    invalid_ticker = invalid_ticker.loc[invalid_ticker].index.to_list()
    df = df.loc[~df.index.get_level_values("ticker").isin(invalid_ticker)]

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.unstack().resample("M").ffill().stack().dropna(how="all")
    return df[ratio_col + ["forward_ret"]]


class FactorReturn:

    def __init__(self, trim_prc: float = None):
        """
        Parameters
        ----------
        trim_prc: float
            Default = None, won't remove outlier
            If trim_prc = 0.05, keep samples with forward return within
              [0.05, 0.95] range for factor return calculation
        """

        self.trim_prc = trim_prc

    def get_factor_return(self, df: pd.DataFrame):
        """
        Calculate factor return for each ratio

        Parameters
        ----------
        df: pd.DataFrame
            ratio dataframe, including adj_close
        """

        f_ret = {}

        for r in ratio_col:
            df_r = df[["forward_ret", r]].dropna(how="any")
            if len(df_r) == 0:
                continue

            if self.trim_prc is not None:
                df_r = df_r.loc[
                    (df_r >= df_r.quantile(q=self.trim_prc)) &
                    (df_r <= df_r.quantile(q=1 - self.trim_prc))
                ]

            df_r.loc[:, "quantile"] = df_r.iloc[:, -1].groupby(
                "date").apply(self.qcut)
            f_ret[r] = df_r.groupby(["date", "quantile"])["forward_ret"].mean()

        f_ret_df = pd.DataFrame(f_ret).stack().unstack("quantile")
        return (f_ret_df[2] - f_ret_df[0]).unstack()

    @staticmethod
    def qcut(s, prc: float = 0.3) -> pd.DataFrame:
        """
        For each factor at certain period, use qcut to calculate factor return
        as Top 30%/20% - Bottom 30%/20%
        """
        s = s.droplevel("date")
        bins = [0, prc, 1-prc, 1]
        q = pd.qcut(s, bins, duplicates='drop', labels=False)
        if len(q.unique()) != 3:
            return s.map(lambda _: np.nan)
        else:
            return q


@cache_df("factor")
def calc_factor(ratio: pd.DataFrame = None) -> pd.DataFrame:
    factor = FactorReturn().get_factor_return(ratio)
    return factor

