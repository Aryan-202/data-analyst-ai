import pandas as pd
import numpy as np
from typing import Any, Dict, List, Union
import json


def json_serialize(obj: Any) -> Any:
    """
    Recursively serialize Python objects to JSON-compatible types
    """
    if obj is None:
        return None
    elif isinstance(obj, (bool, int, float, str)):
        return obj
    elif isinstance(obj, dict):
        return {str(key): json_serialize(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [json_serialize(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif pd.isna(obj):  # Handle pandas NaN, NaT, etc.
        return None
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif hasattr(obj, 'isoformat'):  # Handle datetime objects
        return obj.isoformat()
    elif hasattr(obj, '__dict__'):  # Handle objects with __dict__
        return json_serialize(obj.__dict__)
    else:
        try:
            return str(obj)
        except:
            return None


def dataframe_to_json_safe(df: pd.DataFrame, orient: str = "records") -> List[Dict]:
    """
    Convert DataFrame to JSON-serializable list of dictionaries
    """
    # Convert DataFrame to dictionary first
    if orient == "records":
        data_dict = df.to_dict(orient="records")
    else:
        data_dict = df.to_dict(orient=orient)

    # Clean the data
    cleaned_data = json_serialize(data_dict)

    return cleaned_data