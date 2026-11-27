import json
from decimal import Decimal
from typing import Any
from urllib.request import Request, urlopen


def parse_balance(payload: dict[str, Any], currency: str) -> Decimal:
    for info in payload.get("balance_infos", []):
        if info.get("currency") == currency:
            return Decimal(info["total_balance"])
    raise ValueError(f"balance for currency {currency!r} not found")


class DeepSeekBilling:
    """Account-level balance lookup for pre-run checks and end-of-run reconciliation.

    Per-call cost comes from the price table, not from balance deltas.
    """

    def __init__(
        self,
        api_key: str,
        *,
        currency: str = "CNY",
        base_url: str = "https://api.deepseek.com",
        timeout: float = 10.0,
    ):
        self.api_key = api_key
        self.currency = currency
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _fetch(self) -> dict[str, Any]:
        request = Request(
            f"{self.base_url}/user/balance",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def get_balance(self) -> Decimal:
        return parse_balance(self._fetch(), currency=self.currency)

    def is_available(self) -> bool:
        return bool(self._fetch().get("is_available"))
