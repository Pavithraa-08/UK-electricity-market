from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from .auction import DEFAULT_PRICE_CAP, run_auction
from .data import MarketData

TIE_BREAK = 1e-6


@dataclass(frozen=True)
class StorageParams:

    power_mw: float = 2000.0
    energy_mwh: float = 8000.0
    round_trip_efficiency: float = 0.85
    initial_soc_frac: float = 0.5
    min_soc_frac: float = 0.0
    max_soc_frac: float = 1.0
    throughput_cost: float = 1.0

    @property
    def eta_charge(self) -> float:
        return float(np.sqrt(self.round_trip_efficiency))

    @property
    def eta_discharge(self) -> float:
        return float(np.sqrt(self.round_trip_efficiency))


@dataclass
class StorageSchedule:
    charge: np.ndarray
    discharge: np.ndarray
    soc: np.ndarray
    lp_prices: np.ndarray
    discharge_bid: np.ndarray


def optimise_storage(md: MarketData, params: StorageParams,
                     price_cap: float = DEFAULT_PRICE_CAP) -> StorageSchedule:
    T = md.n_hours
    n_g = len(md.gas_names)
    gas_cost = md.gas_bids()
    ren_av = md.wind_available().sum(1) + md.solar_available().sum(1)

    i_gas, i_ren = 0, T * n_g
    i_uns, i_chg = i_ren + T, i_ren + 2 * T
    i_dis, i_soc = i_chg + T, i_chg + 2 * T
    n = i_soc + T

    c = np.zeros(n)
    c[i_gas:i_ren] = gas_cost.reshape(-1)
    c[i_uns:i_chg] = price_cap
    c[i_chg:i_dis] = params.throughput_cost
    c[i_dis:i_soc] = params.throughput_cost

    ub = np.full(n, np.inf)
    ub[i_gas:i_ren] = np.tile(md.gas_capacity, T)
    ub[i_ren:i_uns] = ren_av
    ub[i_chg:i_dis] = params.power_mw
    ub[i_dis:i_soc] = params.power_mw
    lb = np.zeros(n)
    lb[i_soc:] = params.min_soc_frac * params.energy_mwh
    ub[i_soc:] = params.max_soc_frac * params.energy_mwh

    A_bal = np.zeros((T, n))
    for t in range(T):
        A_bal[t, i_gas + t * n_g: i_gas + (t + 1) * n_g] = 1.0
        A_bal[t, i_ren + t] = 1.0
        A_bal[t, i_uns + t] = 1.0
        A_bal[t, i_dis + t] = 1.0
        A_bal[t, i_chg + t] = -1.0
    b_bal = md.demand

    soc0 = params.initial_soc_frac * params.energy_mwh
    A_soc = np.zeros((T, n))
    b_soc = np.zeros(T)
    for t in range(T):
        A_soc[t, i_soc + t] = 1.0
        A_soc[t, i_chg + t] = -params.eta_charge
        A_soc[t, i_dis + t] = 1.0 / params.eta_discharge
        if t == 0:
            b_soc[t] = soc0
        else:
            A_soc[t, i_soc + t - 1] = -1.0
    A_end = np.zeros((1, n))
    A_end[0, i_soc + T - 1] = 1.0

    res = linprog(c, A_eq=np.vstack([A_bal, A_soc, A_end]),
                  b_eq=np.concatenate([b_bal, b_soc, [soc0]]),
                  bounds=np.column_stack([lb, ub]), method="highs")
    if not res.success:
        raise RuntimeError(f"Storage LP failed: {res.message}")

    x = res.x
    soc_duals = res.eqlin.marginals[T:2 * T]
    discharge_bid = np.maximum(params.throughput_cost - soc_duals / params.eta_discharge - TIE_BREAK, 0.0)
    return StorageSchedule(
        discharge_bid=discharge_bid,
        charge=np.clip(x[i_chg:i_dis], 0, None),
        discharge=np.clip(x[i_dis:i_soc], 0, None),
        soc=x[i_soc:],
        lp_prices=res.eqlin.marginals[:T],
    )


def run_with_storage(md: MarketData, params: StorageParams,
                     price_cap: float = DEFAULT_PRICE_CAP) -> tuple[pd.DataFrame, StorageSchedule]:
    sched = optimise_storage(md, params, price_cap)
    result = run_auction(md, net_demand=md.demand + sched.charge, price_cap=price_cap,
                         storage_offer=(sched.discharge_bid, sched.discharge))
    result["storage_charge_MWh"] = sched.charge
    soc, level = np.zeros(md.n_hours), params.initial_soc_frac * params.energy_mwh
    for t in range(md.n_hours):
        level += params.eta_charge * sched.charge[t] - result["storage_discharge_MWh"].iloc[t] / params.eta_discharge
        soc[t] = level
    result["storage_soc_MWh"] = np.round(soc, 9) + 0.0
    result["storage_revenue_GBP"] = result["price_GBP_per_MWh"] * (result["storage_discharge_MWh"] - sched.charge)
    result["storage_profit_GBP"] = (result["storage_revenue_GBP"] - params.throughput_cost
                                    * (sched.charge + result["storage_discharge_MWh"]))
    result["lp_price_GBP_per_MWh"] = sched.lp_prices
    result["storage_planned_discharge_MWh"] = sched.discharge
    return result, sched


def consistency_report(result: pd.DataFrame, params: StorageParams, tol: float = 0.01) -> dict:
    p = result["price_GBP_per_MWh"].to_numpy()
    lp = result["lp_price_GBP_per_MWh"].to_numpy()
    chg = result["storage_charge_MWh"].to_numpy()
    dis = result["storage_discharge_MWh"].to_numpy()
    plan = result["storage_planned_discharge_MWh"].to_numpy()
    soc = result["storage_soc_MWh"].to_numpy()
    return {
        "max_gap_actual_vs_planned_discharge_MWh": float(np.abs(dis - plan).max()),
        "hours_auction_price_differs_from_LP_price": int((np.abs(p - lp) > tol).sum()),
        "charging_hours_paying_above_LP_value": int(((chg > 1e-6) & (p > lp + tol)).sum()),
        "soc_min_MWh": float(soc.min()), "soc_max_MWh": float(soc.max()),
        "final_soc_MWh": float(soc[-1]),
    }


def _demo() -> None:
    from .data import load_data
    from .report import at_cap, summarise

    np.set_printoptions(suppress=True, linewidth=120, precision=2)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    md = load_data("data/data.xlsx")
    p = StorageParams()
    cap = DEFAULT_PRICE_CAP

    print("=== PART A: the battery we assume ===")
    print(f"Max charge / discharge power : {p.power_mw:,.0f} MW")
    print(f"Energy it can hold           : {p.energy_mwh:,.0f} MWh  ({p.energy_mwh / p.power_mw:.0f} hours at full power)")
    print(f"Round-trip efficiency        : {p.round_trip_efficiency:.0%}  "
          f"(= {p.eta_charge:.3f} going in x {p.eta_discharge:.3f} coming out)")
    print(f"Starts (and must end) at     : {p.initial_soc_frac:.0%} full = {p.initial_soc_frac * p.energy_mwh:,.0f} MWh")
    print(f"Wear / O&M cost              : GBP {p.throughput_cost:.2f} per MWh charged or discharged")

    print("\n=== PART B: the plan the optimiser found for the whole week ===")
    sched = optimise_storage(md, p, cap)
    print(f"Hours charging    : {(sched.charge > 1e-6).sum()}    total charged    : {sched.charge.sum():,.0f} MWh from the grid")
    print(f"Hours discharging : {(sched.discharge > 1e-6).sum()}    total discharged : {sched.discharge.sum():,.0f} MWh to the grid")
    print(f"Discharged / charged = {sched.discharge.sum() / sched.charge.sum():.3f}  (should equal the round-trip efficiency, {p.round_trip_efficiency})")
    print(f"State of charge at the end of hour 168: {sched.soc[-1]:,.0f} MWh  (= where it started)")

    print("\n=== PART C: the auction re-run WITH the battery. First 24 hours ===")
    result, _ = run_with_storage(md, p, cap)
    base = run_auction(md, price_cap=cap)
    result["price_without_battery"] = base["price_GBP_per_MWh"]
    cols = ["hour", "demand_MWh", "price_without_battery", "price_GBP_per_MWh", "storage_charge_MWh",
            "storage_discharge_MWh", "storage_soc_MWh", "marginal_unit"]
    print(result[cols].head(24).round(1).to_string(index=False))

    print("\n=== PART D: the 8 hours where the battery earns the most ===")
    top = result.sort_values("storage_profit_GBP", ascending=False).head(8)
    print(top[["hour", "price_GBP_per_MWh", "storage_discharge_MWh", "storage_profit_GBP"]]
          .round(1).to_string(index=False))

    print("\n=== PART E: week summary, without vs with the battery ===")
    s0, s1 = summarise(base, "no battery", cap), summarise(result, "with battery", cap)
    rows = [("Hours with unserved energy", "hours_with_unserved_energy"), ("Unserved energy (MWh)", "unserved_MWh"),
            ("Hours priced at the cap", "hours_priced_at_cap"), ("Mean price (GBP/MWh)", "mean_price_GBP_per_MWh"),
            ("Mean price excl. cap hours", "mean_price_excl_cap_hours"), ("Curtailed wind+solar (GWh)", "curtailed_GWh"),
            ("Gas generation (GWh)", "gas_GWh")]
    print(f"{'':32}{'no battery':>14}{'with battery':>16}")
    for label, key in rows:
        print(f"{label:32}{s0[key]:14,.1f}{s1[key]:16,.1f}")
    print(f"{'Battery profit (GBP m)':32}{'':>14}{s1['storage_profit_GBP_m']:16,.1f}")
    cap_profit = result.loc[at_cap(result, cap), "storage_profit_GBP"].sum() / 1e6
    print(f"   of which earned in hours priced at the cap: GBP {cap_profit:,.1f}m  "
          f"(so the result depends on the GBP {cap:,.0f} cap assumption)")

    print("\n=== PART F: self-checks (all should look sensible) ===")
    for k, v in consistency_report(result, p).items():
        print(f"{k:50}: {v:,.4f}" if isinstance(v, float) else f"{k:50}: {v}")


if __name__ == "__main__":
    _demo()
