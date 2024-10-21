from datetime import datetime

import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.download_data import (
    get_fx,
    get_price,
    get_ff_data,
    get_zacks,
    get_rate
)


def test_get_price():
    s = get_price(use_cache=False,
                  tickers=["SVOL", "FEZ", "BNO", "DBV", "EGPT", "EEM"],
                  start_date="2022-01-01",
                  end_date="2022-01-31")

    assert isinstance(s, pd.Series)
    assert s.name == "adj_close"


def test_get_fx():
    cur = ["GBP", "EGP", "HUF", "CRC", "RON"]
    df = get_fx(use_cache=False,
                cur_list=cur,
                start_date="2022-01-01",
                end_date="2022-01-31", )

    assert isinstance(df, pd.DataFrame)
    assert set(df.columns.to_list()) == set(cur)


def test_get_ff_data():
    start_date = "2023-01-09"
    end_date = "2023-01-24"

    df = get_ff_data(use_cache=False,
                     start_date=start_date,
                     end_date=end_date)

    assert isinstance(df, pd.DataFrame), "Wrong data format"
    assert df.columns.tolist() == ['Mkt-RF', 'SMB', 'HML', 'RF'], \
        "Wrong Columns"

    assert isinstance(df.index, pd.DatetimeIndex), "Wrong Index"
    assert df.index.min() == pd.Timestamp(start_date), "Wrong start_date"
    assert df.index.max() == pd.Timestamp(end_date), "Wrong end_date"
    assert len(df) == 11, "Missing dates"

    assert df.apply(is_numeric_dtype).all(), "Wrong data type"


def test_get_zacks():
    tickers = ["AAPL", "TSLA", "META"]
    start_date = "2022-01-01"
    end_date = "2022-12-31"

    df = get_zacks(use_cache=False,
                   tickers=tickers,
                   start_date=start_date,
                   end_date=end_date)

    assert isinstance(df, pd.DataFrame), "Wrong data format"
    assert {'mkt_val', 'per_end_date', 'zacks_sector_code'}.issubset(
        set(df.columns.tolist())), "Wrong Columns"

    assert isinstance(df.index, pd.MultiIndex), "Wrong IndexType"
    assert df.index.names == ["ticker", "date"], "Wrong Index names"


def test_get_rate():
    rate = get_rate(use_cache=True)

    assert isinstance(rate, pd.Series)
    assert isinstance(rate.index, pd.DatetimeIndex)
