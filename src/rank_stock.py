from sklearn.preprocessing import scale
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
from typing import Callable

def ols_1(
        ratio: pd.DataFrame,
        factor_chosen: dict[datetime, list[str]],
) -> pd.DataFrame:
    """
    Prediction next period single stock return

    Difference between OLS_1 and OLS_2:
    1. OLS 1 use customized standardization function: x - x.mean()) / x.std();
       OLS 2 use sklearn.preprocessing.scale.
    2. OLS 1 use fill missing with 0; OLS 2 drop ticker with any nan ratio.
    3. OLS 1 use test_date selected factors for testing;
       OLS 2 use train_date selected.

    Parameters
    ----------
    ratio: pd.DataFrame
        Index: (date, ticker)
        column: ratios + forward return
    factor_chosen: dict[datetime, list[str]]
        date -> list of factor selected for the period

    Returns
    -------
    pd.DataFrame:
        Index: (date, ticker)
        Columns: "predicted_ret"
    """
    ### get the selected factor value df ###
    df = ratio.copy()
    df.sort_index(level=['date'], ascending=True, inplace=True)
    dates = list(factor_chosen.keys())[:-1]
    df = df.loc[dates[0]:]

    ratio_new = pd.DataFrame(index=df.index, columns=list(range(1, 16)))
    for date in dates:
        factor_list = factor_chosen[date]
        ratio_new.loc[date, :] = df.loc[date, factor_list].values
    ratio_new = ratio_new.fillna(0)
    ratio_new['ret'] = df['ret_0_1']

    ### predict stock return ###
    predicted_returns = pd.DataFrame(index=ratio_new.index, columns=['predicted_ret'])

    for i in range(1, len(dates)):
        ratio_i = ratio_new.loc[dates[i - 1], :].fillna(0)

        ratio_i.iloc[:, :-1] = ratio_i.iloc[:, :-1].apply(lambda x: (x - x.mean()) / x.std())
        X_train = ratio_i.iloc[:, :-1].values
        y_train = ratio_i.iloc[:, -1].values

        model = LinearRegression()
        model.fit(X_train, y_train)

        X_test = ratio_new.loc[dates[i]].iloc[:, :-1].fillna(0).apply(lambda x: (x - x.mean()) / x.std())
        y_pred = model.predict(X_test)

        predicted_returns.loc[dates[i], 'predicted_ret'] = y_pred

    predicted_returns = predicted_returns.dropna()

    return predicted_returns.apply(pd.to_numeric)


def ols_2(
        ratio: pd.DataFrame,
        factor_chosen: dict[datetime, list[str]],
        min_ticker: int = 50,
        model_cls: Callable = LinearRegression,
        pred_col: str = "forward_ret",
        fillna: bool = False,
        return_model: bool = False
) -> (pd.DataFrame | dict):
    """
    Prediction next period single stock return

    Parameters
    ----------
    ratio: pd.DataFrame
        Index: (date, ticker)
        column: ratios + forward return
    factor_chosen: dict[datetime, list[str]]
        date -> list of factor selected for the period
    min_ticker: int
        minimum number of ticker predicted to generate prediction,
        If too many ticker has missing on any of the selected factor
        on certain date, we will not predict, i.e. change position on this date.
    model_cls: Callable
        Use which model to predict return, e.g. sklearn.LinearRegression;
    pred_col: str
        Regression on current return (ret_0_1) or future return (forward_ret);
    fillna: bool
        If True, fillna with 0; else, drop any dates with nan value.
    return_model: bool
        If True, additional return a dictionary of {date -> model fitted}.

    Returns
    -------
    model_dict: pd.DataFrame
        Index: (date, ticker)
        Columns: "predicted_ret"
    model_dict: dict[datetime, Model]
        dictionary of model fitted
    """
    pred = {}
    model_dict = {}
    for test_date, selected in factor_chosen.items():

        # Make sure both train / test date has ratio
        train_date = test_date - pd.offsets.MonthEnd(1)
        if any([x not in ratio.index.get_level_values("date").unique()
                for x in [train_date, test_date]]):
            print(f"Not trained on {train_date}: No ratio data on train/test date")
            continue

        # Filter training dataset (drop NaN)
        train_df = ratio.loc[train_date][list(set(selected + [pred_col]))]
        train_df = train_df.fillna(0) if fillna else train_df.dropna(how="any")
        if len(train_df) < min_ticker:
            print(f"Not trained on {train_date}: Too many tickers have missing")
            continue

        # Training with current month selected factor + return
        X_train = scale(train_df[selected])
        y_train = train_df[pred_col]
        model = model_cls()
        model_dict[test_date] = model
        model.fit(X_train, y_train)

        # Predicting next month
        test_df = ratio.loc[test_date][selected]
        test_df = test_df.fillna(0) if fillna else test_df.dropna(how="any")
        if len(test_df) > 0:
            pred[test_date] = pd.Series(
                model.predict(scale(test_df)),
                index=test_df.index
            )

    pred_df = pd.DataFrame(pred).unstack().dropna().rename(
        "predicted_ret").to_frame()

    if return_model:
        return pred_df, model_dict
    else:
        return pred_df
