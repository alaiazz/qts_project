import pandas as pd
import numpy as np
from pytest import fixture
import pytest

from src.trading_stratgy import (
    calc_position_by_rank,
    CalcTrade,
    calc_interest
)

INIT_CAPITAL = 1e6


def test_calc_position_by_rank_single(df_price, df_ratio):
    position = calc_position_by_rank(
        df_price=df_price,
        df_ratio=df_ratio,
        rank_formula="+dm",
        lb=0,
        ub=0.1,
        interval="1MS",
        use_change=False,
        position_size=1e6
    )

    assert (position.iloc[-1] == 0).all(), "Close day position should be 0"
    assert (position.index == pd.DatetimeIndex(
        ['2016-01-01', '2016-02-01', '2016-03-01', '2016-04-01',
         '2016-04-09'])).all(), "Wrong resample"

    # Test for reasonable selection amount
    tickers = df_price.columns.to_list()
    n_select = len(tickers) * 0.1
    rebalance_pos = position.iloc[:-1]
    assert (n_select == (rebalance_pos > 0).sum(axis=1)).all(), \
        f"Long position should be {n_select}."
    assert (n_select == (rebalance_pos < 0).sum(axis=1)).all(), \
        f"Long position should be {n_select}."

    # Test for best/worse ticker to long short
    d1 = position.index.min()
    long_tic = df_ratio.loc[d1, "dm"].idxmin()
    short_tic = df_ratio.loc[d1, "dm"].idxmax()
    assert rebalance_pos.loc[d1, long_tic] > 0, "Min ratio = long < 0"
    assert rebalance_pos.loc[d1, short_tic] < 0, "Max ratio = short > 0"


def test_calc_position_by_rank_weekly(df_price, df_ratio):
    position = calc_position_by_rank(
        df_price=df_price,
        df_ratio=df_ratio,
        rank_formula="pe",
        lb=0,
        ub=0.1,
        interval="2W-MON",
        use_change=False,
        position_size=1e6
    )

    assert (position.index == pd.DatetimeIndex([
        '2016-01-04', '2016-01-18', '2016-02-01', '2016-02-15',
        '2016-02-29', '2016-03-14', '2016-03-28', '2016-04-09']
    )).all(), "Wrong weekly resample"


def test_calc_position_by_rank_multiple(df_price, df_ratio):
    position = calc_position_by_rank(
        df_price=df_price,
        df_ratio=df_ratio,
        rank_formula="pe - roi + dm",
        lb=0.1,
        ub=0.2,
        interval="1MS",
        use_change=False,
        position_size=1e6
    )

    # Test for reasonable selection amount
    tickers = df_price.columns.to_list()
    n_select = len(tickers) * 0.1       # Not 35 due to multiple quantile
    rebalance_pos = position.iloc[:-1]
    assert (n_select == (rebalance_pos > 0).sum(axis=1)).all(), \
        f"Long position should be {n_select}."
    assert (n_select == (rebalance_pos < 0).sum(axis=1)).all(), \
        f"Long position should be {n_select}."

    # Test for best/worse ticker to long short
    d1 = position.index.min()
    for r in ["pe", "roi", "dm"]:
        extreme_tic = [df_ratio.loc[d1, r].idxmin(),
                       df_ratio.loc[d1, r].idxmax()]
        assert (rebalance_pos.loc[d1, extreme_tic] == 0).all(), \
            f"Highest / lowest {r} has no position if select second quantile"


def test_calc_position_by_rank_change(df_price, df_ratio):

    # Create infinite change -> should replace +/-inf with 0
    df_ratio.loc['2016-02-01', ("dm", "AAPL")] = 0

    position = calc_position_by_rank(
        df_price=df_price,
        df_ratio=df_ratio,
        rank_formula="pe",
        lb=0.1,
        ub=0.2,
        interval="1MS",
        use_change=True,
        position_size=1e6
    )

    assert position.abs().sum(axis=1).iloc[0] == 0, \
        "Use change has no position on first date"

    assert (position.index == pd.DatetimeIndex(
        ['2016-01-01', '2016-02-01', '2016-03-01', '2016-04-01',
         '2016-04-09'])).all(), "Wrong resample"


@fixture
def position(df_price, df_ratio):
    # True dataframe index = datetime but here simply test with RangeIndex
    sample_pos = calc_position_by_rank(
        df_price=df_price,
        df_ratio=df_ratio,
        rank_formula="+dm",
        lb=0,
        ub=0.1,
        interval="1MS",
        use_change=False,
        position_size=1e6
    )
    # sample_pos.loc[:, "AAPL"] = [10, 20, 0, -10, 0]
    return sample_pos


def test_calc_trade(df_price, position):

    trade_cls = CalcTrade(
        df_price=df_price,
        init_cap=INIT_CAPITAL
    )
    summary_df = trade_cls.iter_dates(position=position)

    assert trade_cls.trade["AAPL"].to_list() == [10.0, 0.0, -10.0, -10.0, 10.0], \
        "No position change when if still select in the period"

    assert np.isclose(summary_df.eval(f"""
        check_total = cash + position + interest - total
        check_ret = total / {INIT_CAPITAL} - 1 - total_ret
    """).filter(regex="^check_"), 0, atol=1e-4).all().all(), \
        "Wrong total / total_ret"


def test_calc_interest(df_price, position):
    trade_cls = CalcTrade(
        df_price=df_price,
        init_cap=INIT_CAPITAL
    )
    final_df = trade_cls.iter_dates(position=position)

    interest = calc_interest(final_df)

    assert isinstance(interest, pd.Series), "Wrong data type"
    assert interest.iloc[-1] == 0, "No interest in the last day"

    assert np.isclose(
        interest.values,
        np.array([
            30519.42052820585,
            28579.65959012486,
            30555.305616290007,
            7888.10058269634,
            0.0]), atol=1e-4).all(), "Wrong interest calculated"
