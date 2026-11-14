from pathlib import Path

import numpy as np

from dispatch.auction import run_auction
from dispatch.data import load_data
from dispatch.storage import StorageParams, consistency_report, run_with_storage

DATA = Path(__file__).resolve().parent.parent / "data" / "data.xlsx"
CAP = 1000.0
MD = load_data(DATA)
P = StorageParams()
RES, SCHED = run_with_storage(MD, P, CAP)


def test_power_and_energy_limits_respected():
    assert RES["storage_charge_MWh"].max() <= P.power_mw + 1e-6
    assert RES["storage_discharge_MWh"].max() <= P.power_mw + 1e-6
    assert RES["storage_soc_MWh"].min() >= P.min_soc_frac * P.energy_mwh - 1e-6
    assert RES["storage_soc_MWh"].max() <= P.max_soc_frac * P.energy_mwh + 1e-6


def test_energy_conservation_and_cyclic_end_state():
    chg, dis = RES["storage_charge_MWh"].sum(), RES["storage_discharge_MWh"].sum()
    assert abs(dis - chg * P.round_trip_efficiency) < 1e-6
    assert abs(RES["storage_soc_MWh"].iloc[-1] - P.initial_soc_frac * P.energy_mwh) < 1e-6


def test_never_charges_and_discharges_in_the_same_hour():
    both = (RES["storage_charge_MWh"] > 1e-6) & (RES["storage_discharge_MWh"] > 1e-6)
    assert not both.any()


def test_auction_delivers_the_planned_schedule():
    c = consistency_report(RES, P)
    assert c["max_gap_actual_vs_planned_discharge_MWh"] < 1e-6
    assert c["charging_hours_paying_above_LP_value"] == 0


def test_system_balance_with_storage():
    served = RES["wind_MWh"] + RES["solar_MWh"] + RES["gas_MWh"] + RES["storage_discharge_MWh"]
    need = MD.demand + RES["storage_charge_MWh"]
    np.testing.assert_allclose(served + RES["unserved_MWh"], need, atol=1e-6)


def test_storage_never_makes_the_shortfall_worse_and_is_profitable():
    base = run_auction(MD, price_cap=CAP)
    assert RES["unserved_MWh"].sum() < base["unserved_MWh"].sum()
    assert RES["storage_profit_GBP"].sum() > 0


def test_bigger_storage_reduces_unserved_energy_monotonically():
    base = run_auction(MD, price_cap=CAP)["unserved_MWh"].sum()
    prev = base
    for mw in (1000, 2000, 4000, 8000):
        res, _ = run_with_storage(MD, StorageParams(power_mw=mw, energy_mwh=4 * mw), CAP)
        assert res["unserved_MWh"].sum() <= prev + 1e-6
        prev = res["unserved_MWh"].sum()


def test_zero_size_storage_reproduces_baseline_prices():
    res, _ = run_with_storage(MD, StorageParams(power_mw=0.0, energy_mwh=0.0), CAP)
    base = run_auction(MD, price_cap=CAP)
    np.testing.assert_allclose(res["price_GBP_per_MWh"], base["price_GBP_per_MWh"], atol=1e-6)
