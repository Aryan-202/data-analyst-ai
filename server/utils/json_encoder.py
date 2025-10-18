import json
import numpy as np
import pandas as pd
from decimal import Decimal
from typing import Any

class SafeJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NaN, Infinity, and other non-serializable values"""
    
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        elif isinstance(obj, Decimal):
            return float(obj)
        elif hasattr(obj, 'isoformat'):  # Handle datetime objects
            return obj.isoformat()
        else:
            return super().default(obj)

def safe_json_serialize(obj: Any) -> str:
    """Safely serialize Python object to JSON string"""
    return json.dumps(obj, cls=SafeJSONEncoder, ensure_ascii=False)

def safe_json_serialize_dict(obj: dict) -> dict:
    """Recursively clean a dictionary for JSON serialization"""
    if isinstance(obj, dict):
        return {key: safe_json_serialize_dict(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_json_serialize_dict(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif pd.isna(obj):
        return None
    else:
        return obj