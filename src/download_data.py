from datetime import datetime
import sys

import quandl

import nasdaqdatalink
import pandas as pd
import numpy as np


nasdaqdatalink.ApiConfig.api_key = 'DKEpSA2RKyvtpZKmVWGv'
START_DATE = "1998-01-01"
END_DATE = datetime.today()
IGNORE_SECTOR = [5, 13]


def cache_df(name: str):
    """
    Wrapper function to use locally stored data
    """

    def inner_func(func):
        def wrapper(use_cache: bool = False, *args, **kwargs):
            """
            use_cache: if True, try read local pickle file
            """
            try:
                assert use_cache
                df = pd.read_pickle(f"data/{name}.pkl")
            except (AssertionError, FileNotFoundError):
                df = func(*args, **kwargs)
                if not "pytest" in sys.modules:
                    df.to_pickle(f"data/{name}.pkl")
            return df

        return wrapper

    return inner_func


@cache_df(name="fx_price")
def get_fx(cur_list: list[str],
           start_date: str | datetime = START_DATE,
           end_date: str | datetime = END_DATE) -> pd.DataFrame:
    """
    Download required data from Quandl EDI/CUR and
    save to pickle file for future rerun.

    Parameters
    ----------
    cur_list: download price data of given currencies
    start_date: get price data from start_date to end_date (both inclusive)
    end_date: get price data from start_date to end_date (both inclusive)

    Returns
    -------
    pd.DataFrame of required price data for analysis
    """
    dates = pd.date_range(start_date, end_date).strftime("%Y-%m-%d")
    df = nasdaqdatalink.get_table(
        'EDI/CUR',
        date=','.join(dates.tolist()),
        code=','.join(cur_list),
        paginate=True
    )
    df = df.set_index(["date", "code"])["rate"].unstack()

    return df


@cache_df(name="price")
def get_price(
        tickers: list[str] = None,
        start_date: str | datetime = START_DATE,
        end_date: str | datetime = END_DATE
) -> pd.Series:
    """
    Download price from Quandl QUOTEMEDIA/PRICES and  save to pickle file
    for future rerun.

    Parameters
    ----------
    tickers: download price data of given tickers
    start_date: get price data from start_date to end_date (both inclusive);
    end_date: get price data from start_date to end_date (both inclusive).

    Returns
    -------
    pd.DataFrame of required price data for analysis
    """
    dates = pd.date_range(start_date, end_date).strftime("%Y-%m-%d")
    df = nasdaqdatalink.get_table('QUOTEMEDIA/PRICES',
                                  date=','.join(dates.tolist()),
                                  ticker=','.join(tickers),
                                  paginate=True)
    df = df.set_index(["ticker", "date"])[[
        "adj_close", "adj_open", "adj_high", "adj_low", "volume"
    ]]
    return df


@cache_df(name="ff")
def get_ff_data(start_date: str | datetime = START_DATE,
                end_date: str | datetime = END_DATE) -> pd.DataFrame:
    """
    Download Fama-French data and save to pickle file for future rerun

    Parameters
    ----------
    start_date: get price data from start_date to end_date (both inclusive)
    end_date: get price data from start_date to end_date (both inclusive)

    Returns
    -------
    pd.DataFrame of required price data for analysis
    """

    ff_link = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
               "ftp/F-F_Research_Data_Factors_daily_CSV.zip")
    ff = pd.read_csv(ff_link, compression='zip', header=None, sep='\t',
                     lineterminator='\r')
    ff = (ff.iloc[4:, 0].str.strip("\n").str.split(",", expand=True)
          .dropna(how="any"))
    df = pd.DataFrame(
        ff.iloc[1:, 1:].values,
        index=pd.to_datetime(ff.iloc[1:, 0], format="%Y%m%d"),
        columns=ff.iloc[0, 1:].to_list()
    )
    return df.loc[start_date:end_date].apply(pd.to_numeric)


@cache_df(name="zack")
def get_zacks(
        tickers: list[str] = None,
        start_date: str | datetime = START_DATE,
        end_date: str | datetime = END_DATE,
) -> pd.DataFrame:
    """
    Download required price data from Quandl ZACKS/.. and
    save to pickle file for future rerun.

    Parameters
    ----------
    tickers: download price data of tickers;
    start_date: get price data from start_date to end_date (both inclusive);
    end_date: get price data from start_date to end_date (both inclusive).

    Returns
    -------
    pd.DataFrame of required price data for analysis
    """
    dates = pd.date_range(start_date, end_date).strftime("%Y-%m-%d")
    date_filter = ','.join(dates.tolist())
    ticker_filter = ','.join(tickers)
    idx_col = ["ticker", "per_end_date"]

    # Market Value
    mkt = nasdaqdatalink.get_table('ZACKS/MKTV',
                                   per_end_date=date_filter,
                                   ticker=ticker_filter,
                                   paginate=True)
    mkt = mkt.set_index(idx_col)[["mkt_val"]]

    # Financial Indicator
    fc = nasdaqdatalink.get_table('ZACKS/FC',
                                  per_end_date=date_filter,
                                  ticker=ticker_filter,
                                  per_type="Q",
                                  paginate=True)

    fc.loc[:, ["net_lterm_debt"]] = fc["net_lterm_debt"].fillna(
        fc["tot_lterm_debt"])
    fc.loc[:, "eps_diluted_net"] = fc["eps_diluted_net"].fillna(
        fc["basic_net_eps"]).where(lambda x: x > 0, 0.001)
    fc = fc.set_index(idx_col)
    fc = fc[list(set(fc.columns) - {"mkt_val"})]

    # Financial Ratio
    fr = nasdaqdatalink.get_table('ZACKS/FR',
                                  per_end_date=date_filter,
                                  ticker=ticker_filter,
                                  per_type="Q",
                                  paginate=True)
    fr = fr.set_index(idx_col)
    fr = fr[list(set(fr.columns) - set(fc.columns) - {"mkt_val"})]

    # Combine data
    df = pd.concat([mkt, fc, fr], axis=1)

    # Match price data with filing date + 1 -> Filing happened after market close
    df.loc[:, "filing_date"] = df["filing_date"] + pd.offsets.Day(1)
    # df.loc[:, "per_end_date"] = df["filing_date"] + pd.offsets.Day(46)
    df.reset_index(inplace=True)
    df.set_index(["ticker", "filing_date"], inplace=True)
    df.index.names = ("ticker", "date")

    return df


@cache_df("rate")
def get_rate():

    df_list = []
    for i in range(2002, 2025):
        df = pd.read_csv(f"https://home.treasury.gov/resource-center/"
                         f"data-chart-center/interest-rates/"
                         f"daily-treasury-rates.csv/{i}/all?"
                         f"type=daily_treasury_bill_rates&"
                         f"field_tdr_date_value={i}&page&_format=csv")
        df_list.append(df)

    df = pd.concat(df_list)
    df.index = pd.to_datetime(df["Date"], format="%m/%d/%Y")

    return df["4 WEEKS COUPON EQUIVALENT"].rename("rate")
