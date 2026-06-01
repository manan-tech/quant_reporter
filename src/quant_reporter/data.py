import logging
from typing import List, Optional

import pandas as pd

from .providers import DataProvider, get_default_provider

logger = logging.getLogger(__name__)


def get_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    provider: Optional[DataProvider] = None,
) -> Optional[pd.DataFrame]:
    """
    Fetch adjusted close prices for *tickers* between *start_date* and *end_date*.

    By default uses the global YFinanceProvider. Pass a custom *provider* to use
    Bloomberg, Refinitiv, CSV data, or a test fixture instead.
    """
    p = provider if provider is not None else get_default_provider()
    return p.get_prices(tickers, start_date, end_date)
