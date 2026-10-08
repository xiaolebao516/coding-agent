from decimal import Decimal

import pytest

from coding_agent.contracts import Money
from coding_agent.deepseek_billing import (
    DeepSeekBilling,
    parse_balance,
)


def test_parse_balance():
    payload = {
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

    balance = parse_balance(payload, "CNY")

    assert balance == Money(
        amount=Decimal("12.3456"),
        currency="CNY",
    )


def test_cost_between_balances():
    before = Money(
        amount=Decimal("10.0000"),
        currency="CNY",
    )
    after = Money(
        amount=Decimal("9.9973"),
        currency="CNY",
    )

    cost = DeepSeekBilling.cost_between(
        before,
        after,
    )

    assert cost.money.amount == Decimal("0.0027")
    assert cost.money.currency == "CNY"
    assert cost.source == "balance_delta"


def test_cost_between_rejects_currency_mismatch():
    with pytest.raises(ValueError):
        DeepSeekBilling.cost_between(
            Money(Decimal("10"), "CNY"),
            Money(Decimal("9"), "USD"),
        )


def test_cost_between_rejects_balance_increase():
    with pytest.raises(ValueError):
        DeepSeekBilling.cost_between(
            Money(Decimal("10"), "CNY"),
            Money(Decimal("11"), "CNY"),
        )
