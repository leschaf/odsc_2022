import multiprocessing

import joblib
import pandas as pd


@pd.api.extensions.register_dataframe_accessor("parallelize")
class _DataFrameAccessor:
    """Applys a function to groupby elements in parallel"""

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def groupby_apply(self, func, groupby_ls: list, **args) -> pd.DataFrame:
        """applys function to groupby
        Args:
            func ([type]): function name
            groupby_ls (list): list of parameters to groupby
        Returns:
            pd.DataFrame: concatenated result
        """
        ret_lst = joblib.Parallel(n_jobs=multiprocessing.cpu_count())(
            joblib.delayed(func)(name, group, **args)
            for name, group in self._obj.groupby(groupby_ls)
        )

        return pd.concat(ret_lst)
