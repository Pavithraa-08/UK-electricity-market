from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import MarketData

DEFAULT_PRICE_CAP = 1000.0  
_EPS = 1e-9


@dataclass
class HourResult:
    price: float
    dispatch: np.ndarray   
    unserved: float        
    marginal_index: int   


def clear_hour(bids: np.ndarray, quantities: np.ndarray, demand: float,
               price_cap: float = DEFAULT_PRICE_CAP) -> HourResult:
    if demand < 0:
        raise ValueError("demand must be non-negative")
    bids = np.asarray(bids, float)
    quantities = np.asarray(quantities, float)
    dispatch = np.zeros_like(quantities)

    levels = np.unique(bids)  
    remaining = float(demand)

    if remaining <= _EPS:  
        return HourResult(float(levels[0]) if len(levels) else 0.0, dispatch, 0.0, -1)

    for level in levels:
        block = np.flatnonzero(bids == level)
        block_qty = quantities[block].sum()
        if block_qty <= _EPS:
            continue
        if remaining <= block_qty + _EPS:           
            dispatch[block] = quantities[block] * min(remaining / block_qty, 1.0)
            return HourResult(float(level), dispatch, 0.0, int(block[0]))
        dispatch[block] = quantities[block]        
        remaining -= block_qty

    return HourResult(float(price_cap), dispatch, max(remaining, 0.0), -1)


def run_auction(md: MarketData, net_demand: np.ndarray | None = None,
                price_cap: float = DEFAULT_PRICE_CAP,
                storage_offer: tuple[np.ndarray, np.ndarray] | None = None) -> pd.DataFrame:
    demand = md.demand if net_demand is None else np.asarray(net_demand, float)
    wind_av, solar_av, gas_bids = md.wind_available(), md.solar_available(), md.gas_bids()
    n_w, n_s, n_g = len(md.wind_names), len(md.solar_names), len(md.gas_names)

    rows = []
    for t in range(md.n_hours):
        bids = np.concatenate([np.zeros(n_w + n_s), gas_bids[t]])
        qty = np.concatenate([wind_av[t], solar_av[t], md.gas_capacity])
        if storage_offer is not None:
            bids = np.append(bids, storage_offer[0][t])
            qty = np.append(qty, storage_offer[1][t])
        res = clear_hour(bids, qty, demand[t], price_cap)
        d = res.dispatch
        wind, solar = d[:n_w].sum(), d[n_w:n_w + n_s].sum()
        gas = d[n_w + n_s:n_w + n_s + len(md.gas_names)].sum()
        row = {
            "hour": int(md.hours[t]),
            "demand_MWh": md.demand[t],
            "net_demand_MWh": demand[t],
            "price_GBP_per_MWh": res.price,
            "wind_MWh": wind,
            "solar_MWh": solar,
            "gas_MWh": gas,
            "unserved_MWh": res.unserved,
            "renewable_available_MWh": wind_av[t].sum() + solar_av[t].sum(),
            "curtailed_MWh": wind_av[t].sum() + solar_av[t].sum() - wind - solar,
            "scarcity": res.unserved > _EPS,
        }
        for k, name in enumerate(md.wind_names + md.solar_names + md.gas_names):
            row[f"{name}_MWh"] = d[k]
        if storage_offer is not None:
            row["storage_discharge_MWh"] = d[-1]
        row["marginal_unit"] = ("none (scarcity)" if res.marginal_index < 0 and res.unserved > _EPS
                                else _marginal_label(md, res, n_w, n_s))
        rows.append(row)
    return pd.DataFrame(rows)


def _marginal_label(md: MarketData, res: HourResult, n_w: int, n_s: int) -> str:
    if res.marginal_index < 0:
        return "none (zero demand)"
    names = md.wind_names + md.solar_names + md.gas_names + ["storage"]
    if res.marginal_index < n_w + n_s:
        return "renewables (price 0)"
    return names[res.marginal_index]


def technology_mix(result: pd.DataFrame) -> pd.DataFrame:
    """Share of net demand served by each technology, per hour (%)."""
    cols = [c for c in ("wind_MWh", "solar_MWh", "gas_MWh", "storage_discharge_MWh") if c in result]
    served = result[cols].sum(axis=1)
    mix = result[["hour"]].copy()
    for c in cols:
        mix[c.replace("_MWh", "") + "_share_%"] = np.where(served > 0, 100 * result[c] / served, 0.0)
    return mix


def _demo() -> None:
    from .data import load_data

    np.set_printoptions(suppress=True, linewidth=120, precision=2)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    print("=== PART A: a toy market with 3 plants (bids 0, 10, 20 GBP/MWh; 100 MWh each) ===")
    cases = [
        ("Demand 150 (needs part of plant 2)", 150),
        ("Demand 200 (exactly fills plant 2)", 200),
        ("Demand 50  (cheapest plant only partly used)", 50),
        ("Demand 500 (more than all plants can give)", 500),
    ]
    for label, demand in cases:
        r = clear_hour([0, 10, 20], [100, 100, 100], demand, price_cap=1000)
        print(f"{label}: price = {r.price:g}, dispatch = {r.dispatch}, unserved = {r.unserved:g}")
    r = clear_hour([0, 0, 15], [100, 300, 100], 200, price_cap=1000)
    print(f"Two zero-bid plants (100 and 300 MWh), demand 200: price = {r.price:g}, "
          f"dispatch = {r.dispatch}  <- shared pro rata")

    print("\n=== PART B: hour 1 of the real data, offer by offer (cheapest first) ===")
    md = load_data("data/data.xlsx")
    n_w, n_s = len(md.wind_names), len(md.solar_names)
    names = md.wind_names + md.solar_names + md.gas_names
    bids = np.concatenate([np.zeros(n_w + n_s), md.gas_bids()[0]])
    qty = np.concatenate([md.wind_available()[0], md.solar_available()[0], md.gas_capacity])
    demand = md.demand[0]
    print(f"Demand in hour 1: {demand:,.0f} MWh")
    order = np.argsort(bids, kind="stable")
    running = 0.0
    print(f"{'plant':8}{'bid':>8}{'can give':>10}{'running total':>15}")
    for i in order:
        if bids[i] == 0 and running > 0 and names[i].startswith("solar") and qty[i] == 0:
            continue  
        running += qty[i]
        print(f"{names[i]:8}{bids[i]:8.2f}{qty[i]:10.0f}{running:15.0f}")
        if running >= demand:
            break
    res = clear_hour(bids, qty, demand, 1000)
    marginal = names[res.marginal_index]
    print(f"-> demand is reached at {marginal}, so the price is its bid: GBP {res.price:.2f}/MWh")

    print("\n=== PART C: the full 168-hour auction (first 8 hours) ===")
    result = run_auction(md, price_cap=1000)
    cols = ["hour", "demand_MWh", "price_GBP_per_MWh", "wind_MWh", "solar_MWh", "gas_MWh",
            "unserved_MWh", "marginal_unit"]
    print(result[cols].head(8).round(1).to_string(index=False))

    print("\n=== PART D: what happens across the week ===")
    print("Hours by marginal (price-setting) unit:")
    print(result["marginal_unit"].value_counts().to_string())
    short = result.loc[result["scarcity"], "hour"].tolist()
    print(f"\nHours with not enough supply ({len(short)}): {short}")
    zero = result.loc[result["price_GBP_per_MWh"] == 0, "hour"].tolist()
    print(f"Hours with a surplus of wind and solar, price 0 ({len(zero)}): {zero}")
    print(f"Total unserved energy: {result['unserved_MWh'].sum():,.0f} MWh")


if __name__ == "__main__":
    _demo()
