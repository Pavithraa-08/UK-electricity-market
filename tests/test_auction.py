from pathlib import Path

import numpy as np

from dispatch.auction import clear_hour, run_auction
from dispatch.data import load_data

DATA = Path(__file__).resolve().parent.parent / "data" / "data.xlsx"


def test_partial_marginal_unit():
    r = clear_hour([0, 10, 20], [100, 100, 100], demand=150, price_cap=999)
    assert r.price == 10
    np.testing.assert_allclose(r.dispatch, [100, 50, 0])


def test_demand_exactly_fills_a_block_sets_that_blocks_price():
    r = clear_hour([0, 10, 20], [100, 100, 100], demand=200, price_cap=999)
    assert r.price == 10
    np.testing.assert_allclose(r.dispatch, [100, 100, 0])


def test_surplus_zero_bid_offers_are_curtailed_pro_rata_and_price_is_zero():
    r = clear_hour([0, 0, 15], [100, 300, 100], demand=200, price_cap=999)
    assert r.price == 0
    np.testing.assert_allclose(r.dispatch, [50, 150, 0])


def test_shortfall_reports_unserved_energy_and_cap_price():
    r = clear_hour([0, 10], [100, 100], demand=500, price_cap=777)
    assert r.price == 777 and r.unserved == 300
    np.testing.assert_allclose(r.dispatch, [100, 100])


def test_same_bid_offers_share_the_margin_pro_rata():
    r = clear_hour([10, 10], [100, 300], demand=200, price_cap=999)
    np.testing.assert_allclose(r.dispatch, [50, 150])


def test_gas_bid_conversion():
    md = load_data(DATA)
    assert abs(md.gas_bids()[0, 0] - 70 / 100 * 34.121 / 0.35) < 1e-9
    assert (np.diff(md.gas_bids(), axis=1) < 0).all()


def _reference_price(bids, qty, demand, cap):
    order = np.argsort(bids, kind="stable")
    cum = np.cumsum(np.asarray(qty)[order])
    for i, c in zip(order, cum):
        if c >= demand - 1e-9:
            return bids[i]
    return cap


def test_matches_reference_implementation_on_random_hours():
    rng = np.random.default_rng(0)
    for _ in range(500):
        n = rng.integers(2, 9)
        bids = rng.choice(np.arange(0, 120, 7.5), n)
        qty = rng.uniform(10, 500, n)
        demand = rng.uniform(0, qty.sum() * 1.2)
        r = clear_hour(bids, qty, demand, price_cap=1000)
        assert abs(r.price - _reference_price(bids, qty, demand, 1000)) < 1e-9
        assert abs(r.dispatch.sum() + r.unserved - demand) < 1e-6
        assert (r.dispatch <= qty + 1e-9).all()


def test_real_data_balance_and_price_bounds():
    md = load_data(DATA)
    res = run_auction(md, price_cap=1000)
    served = res["wind_MWh"] + res["solar_MWh"] + res["gas_MWh"]
    np.testing.assert_allclose(served + res["unserved_MWh"], md.demand, atol=1e-6)
    ok = ~res["scarcity"]
    assert res.loc[ok, "price_GBP_per_MWh"].between(0, md.gas_bids().max() + 1e-9).all()
    assert (res.loc[res["scarcity"], "price_GBP_per_MWh"] == 1000).all()
    for name, cap in zip(md.gas_names, md.gas_capacity):
        assert (res[f"{name}_MWh"] <= cap + 1e-9).all()
