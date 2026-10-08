import json
from decimal import Decimal
from typing import Any
from urllib.request import Request, urlopen

from coding_agent.contracts import Cost, Money


def parse_balance(
    payload: dict[str, Any],
    currency: str,
) -> Money:
    for info in payload.get("balance_infos", []):
        if info.get("currency") == currency:
            return Money(
                amount=Decimal(info["total_balance"]),
                currency=currency,
            )

    raise ValueError(f"balance for currency {currency!r} not found")


class DeepSeekBilling:
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

    def get_balance(self) -> Money:
        request = Request(
            f"{self.base_url}/user/balance",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

        with urlopen(request, timeout=self.timeout) as response:
            payload = json.load(response)

        return parse_balance(
            payload,
            currency=self.currency,
        )

    @staticmethod
    def cost_between(
        before: Money,
        after: Money,
    ) -> Cost:
        if before.currency != after.currency:
            raise ValueError("cannot calculate cost between different currencies")

        amount = before.amount - after.amount

        if amount < 0:
            raise ValueError("balance increased; cost cannot be attributed safely")

        return Cost(
            money=Money(
                amount=amount,
                currency=before.currency,
            ),
            source="balance_delta",
        )
