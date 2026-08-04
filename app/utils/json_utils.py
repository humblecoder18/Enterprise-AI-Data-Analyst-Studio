import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionDtype


def make_json_serializable(obj):
    """
    Recursively convert pandas/numpy objects into JSON-serializable Python objects.
    """

    if isinstance(obj, dict):
        return {
            str(k): make_json_serializable(v)
            for k, v in obj.items()
        }

    if isinstance(obj, list):
        return [make_json_serializable(i) for i in obj]

    if isinstance(obj, tuple):
        return [make_json_serializable(i) for i in obj]

    # NumPy scalar types
    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    # NumPy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # Pandas Series
    if isinstance(obj, pd.Series):
        return obj.to_list()

    # Pandas Index
    if isinstance(obj, pd.Index):
        return obj.tolist()

    # Pandas Timestamp
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()

    # NumPy dtype
    if isinstance(obj, np.dtype):
        return str(obj)

    # Pandas ExtensionDtype (StringDtype, Int64Dtype, BooleanDtype, etc.)
    if isinstance(obj, ExtensionDtype):
        return str(obj)

    return obj