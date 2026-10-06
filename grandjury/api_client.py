import requests
from typing import Union, List, Dict, Any, Optional
import json
from pathlib import Path

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

try:
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

try:
    import msgspec
    HAS_MSGSPEC = True
except ImportError:
    HAS_MSGSPEC = False

API_BASE_URL = "https://api.humanjudge.com/api/v1"

class GrandJuryClient:
    def __init__(self, api_key: Optional[str] = None, base_url: str = API_BASE_URL):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        # Ensure we have the correct API base path if user provides a different URL
        if not self.base_url.endswith('/api/v1'):
            self.base_url = f"{self.base_url}/api/v1"
        
    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def _make_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request with efficient serialization."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = self._get_headers()
        
        # Use msgspec for faster serialization if available
        if HAS_MSGSPEC:
            json_data = msgspec.json.encode(payload).decode()
            response = requests.post(url, data=json_data, headers=headers)
        else:
            response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API Error ({response.status_code}): {response.text}")

    def _parse_input(self, data: Union[str, Path, List[Dict], Dict]) -> List[Dict[str, Any]]:
        """Parse various input formats into list of dictionaries."""
        
        # Handle file paths (CSV, Parquet)
        if isinstance(data, (str, Path)):
            file_path = Path(data)
            if file_path.suffix.lower() == '.csv':
                return self._parse_csv(file_path)
            elif file_path.suffix.lower() in ['.parquet', '.pq']:
                return self._parse_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Handle polars DataFrame (check first since it's more specific)
        elif HAS_POLARS and hasattr(data, '__module__') and 'polars' in str(data.__module__):
            return self._parse_polars(data)
        
        # Handle pandas DataFrame  
        elif HAS_PANDAS and hasattr(data, '__module__') and ('pandas' in str(data.__module__) or hasattr(data, 'to_dict')):
            return self._parse_pandas(data)
        
        # Handle list of dictionaries
        elif isinstance(data, list):
            return data
        
        # Handle single dictionary
        elif isinstance(data, dict):
            return [data]
        
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def _parse_csv(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse CSV file efficiently."""
        if HAS_PANDAS:
            df = pd.read_csv(file_path)
            return self._pandas_to_records(df)
        else:
            import csv
            records = []
            with open(file_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert string 'True'/'False' to boolean where appropriate
                    processed_row = {}
                    for k, v in row.items():
                        if v.lower() in ('true', 'false'):
                            processed_row[k] = v.lower() == 'true'
                        elif v.lower() == 'none' or v == '':
                            processed_row[k] = None
                        else:
                            # Try to convert to int/float if possible
                            try:
                                if '.' in v:
                                    processed_row[k] = float(v)
                                else:
                                    processed_row[k] = int(v)
                            except ValueError:
                                processed_row[k] = v
                    records.append(processed_row)
            return records
    
    def _parse_parquet(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse Parquet file efficiently."""
        if HAS_PANDAS:
            df = pd.read_parquet(file_path)
            return self._pandas_to_records(df)
        elif HAS_PYARROW:
            table = pq.read_table(file_path)
            return table.to_pylist()
        else:
            raise ImportError("Either pandas or pyarrow is required to read Parquet files")
    
    def _parse_pandas(self, df) -> List[Dict[str, Any]]:
        """Convert pandas DataFrame to records."""
        return self._pandas_to_records(df)
    
    def _pandas_to_records(self, df) -> List[Dict[str, Any]]:
        """Convert pandas DataFrame to list of dictionaries with type conversion."""
        # Convert to records, handling NaN values
        if HAS_PANDAS:
            records = df.where(pd.notna(df), None).to_dict('records')
        else:
            records = df.to_dict('records')  # Fallback for when pandas isn't available
        
        # Convert numpy types to native Python types for JSON serialization
        converted_records = []
        for record in records:
            converted_record = {}
            for k, v in record.items():
                if HAS_PANDAS and pd.isna(v):
                    converted_record[k] = None
                elif hasattr(v, 'item'):  # numpy scalar
                    converted_record[k] = v.item()
                else:
                    converted_record[k] = v
            converted_records.append(converted_record)
        
        return converted_records
    
    def _parse_polars(self, df) -> List[Dict[str, Any]]:
        """Convert polars DataFrame to records."""
        return df.to_dicts()

    # API Methods
    def evaluate_model(self, previous_score: float = 0.0, previous_timestamp: str = None, 
                      votes: List[float] = None, reputations: List[float] = None) -> Dict[str, Any]:
        """Evaluate model with scoring algorithm (requires API key)."""
        from datetime import datetime, timedelta
        
        if previous_timestamp is None:
            previous_timestamp = (datetime.now() - timedelta(hours=1)).isoformat()
        if votes is None:
            votes = []
        if reputations is None:
            reputations = []
            
        payload = {
            "previous_score": previous_score,
            "previous_timestamp": previous_timestamp,
            "votes": votes,
            "reputations": reputations
        }
        return self._make_request("evaluate", payload)
    
    def vote_histogram(self, data, duration_minutes: int = 60, gross: bool = True) -> Dict[str, Any]:
        """Get vote time histogram."""
        parsed_data = self._parse_input(data)
        payload = {
            "data": parsed_data,
            "duration_minutes": duration_minutes,
            "gross": gross
        }
        return self._make_request("verdict/histogram", payload)
    
    def vote_completeness(self, data, voter_list: List[int], inference_ids: Optional[List[int]] = None, 
                         gross: bool = False) -> Union[Dict[str, float], float]:
        """Get vote completeness analysis."""
        parsed_data = self._parse_input(data)
        payload = {
            "data": parsed_data,
            "voter_list": voter_list,
            "gross": gross
        }
        if inference_ids is not None:
            payload["inference_ids"] = inference_ids
        return self._make_request("verdict/completeness", payload)
    
    def population_confidence(self, data, voter_list: List[int], inference_ids: Optional[List[int]] = None) -> Union[Dict[str, float], float]:
        """Get population confidence analysis."""
        parsed_data = self._parse_input(data)
        payload = {
            "data": parsed_data,
            "voter_list": voter_list
        }
        if inference_ids is not None:
            payload["inference_ids"] = inference_ids
        return self._make_request("verdict/population-confidence", payload)
    
    def majority_good_votes(self, data, good_vote: Union[bool, str] = True, threshold: float = 0.5) -> Dict[str, int]:
        """Get majority good votes count."""
        parsed_data = self._parse_input(data)
        payload = {
            "data": parsed_data,
            "good_vote": good_vote,
            "threshold": threshold
        }
        return self._make_request("verdict/majority-good", payload)
    
    def votes_distribution(self, data, inference_ids: Optional[List[int]] = None) -> Dict[str, int]:
        """Get votes distribution per inference."""
        parsed_data = self._parse_input(data)
        payload = {"data": parsed_data}
        if inference_ids is not None:
            payload["inference_ids"] = inference_ids
        return self._make_request("verdict/votes-distribution", payload)

# Backward compatibility function
def evaluate_model(predictions, references, api_key=None):
    """Original evaluate_model function for backward compatibility."""
    # This maintains the original signature but maps to the new scoring endpoint
    client = GrandJuryClient(api_key=api_key)
    # Convert predictions/references to votes if they're lists of strings
    if isinstance(predictions, list) and isinstance(references, list):
        # Simple scoring: compare predictions vs references
        votes = [1.0 if p == r else 0.0 for p, r in zip(predictions, references)]
        reputations = [1.0] * len(votes)
        return client.evaluate_model(votes=votes, reputations=reputations)
    else:
        # Assume they're already numeric scores
        return client.evaluate_model(votes=predictions, reputations=references)
