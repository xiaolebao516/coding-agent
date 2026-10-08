from decimal import Decimal

import pytest

from coding_agent.deepseek_billing import parse_balance

PAYLOAD = {
    "is_available": True,
    "balance_infos": [
        {
            "currency": "CNY",
            "total_balance": "12.3456",
            "granted_balance": "2.3456",
            "topped_up_balance": "10.00",
        }
    ],
}


def test_parse_balance():
    assert parse_balance(PAYLOAD, "CNY") == Decimal("12.3456")


def test_parse_balance_rejects_missing_currency():
    with pytest.raises(ValueError):
        parse_balance(PAYLOAD, "USD")
