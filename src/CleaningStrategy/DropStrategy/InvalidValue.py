from ..base import CleaningStrategy
import pandas as pd
class RemoveNumericAirportCodes(CleaningStrategy):
    def handle_data(self, df: pd.DataFrame) -> pd.DataFrame:
        
        try:
            return df[~df['ORIGIN_AIRPORT'].astype(str).str.isnumeric()]
        except Exception as e:
            raise Exception(f"Error removing numeric airport codes: {e}")
