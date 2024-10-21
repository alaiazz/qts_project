import numpy as np

from src.calculate_factor import *


def test_find_valid_universe():
    lst = find_valid_universe()

    assert isinstance(lst, list)


def test_calc_ratio():
    df = calc_ratio(use_cache=True, use_data_cache=True, tickers=["AKAM"])

    assert isinstance(df, pd.DataFrame), "Wrong data format"
    assert not df.empty, "DF shouldn't be empty"

    assert set(ratio_col).issubset(set(df.columns.tolist())), \
        "Missing columns"
    # assert df.groupby("ticker").first()[ratio_col].notnull().all().all()


def test_calc_factor():
    ratio = calc_ratio(use_cache=True, tickers=["LLY", "AAPL", "GM"])
    df = calc_factor(ratio=ratio)

    assert isinstance(df, pd.DataFrame), "Wrong data format"
    assert isinstance(df.index, pd.DatetimeIndex)
    assert set(df.columns.tolist()).issubset(set(ratio_col))