from .api_client import GrandJuryClient, evaluate_model
from .result_set import ResultSet
from .sdk import GrandJury, Span

# Public alias
GJClient = GrandJury

__version__ = "2.3.4"
__all__ = ["GJClient", "GrandJury", "Span", "ResultSet", "GrandJuryClient", "evaluate_model"]
