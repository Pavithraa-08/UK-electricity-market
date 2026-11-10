from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

THERMS_PER_MWH = 34.121  


@dataclass(frozen=True)
class MarketData:
    hours: np.ndarray            # (T,)
    demand: np.ndarray           # (T,) MWh
    gas_price_p_per_therm: np.ndarray  # (T,)
    wind_names: list[str]
    wind_capacity: np.ndarray    # (n_wind,) MW
    wind_lf: np.ndarray          # (T, n_wind) 0..1
    solar_names: list[str]
    solar_capacity: np.ndarray   # (n_solar,) MW
    solar_lf: np.ndarray         # (T, n_solar) 0..1
    gas_names: list[str]
    gas_capacity: np.ndarray     # (n_gas,) MW
    gas_efficiency: np.ndarray   # (n_gas,) 0..1

    @property
    def n_hours(self) -> int:
        return len(self.hours)

    def gas_bids(self) -> np.ndarray:
        gbp_per_mwh_thermal = self.gas_price_p_per_therm / 100.0 * THERMS_PER_MWH
        return gbp_per_mwh_thermal[:, None] / self.gas_efficiency[None, :]

    def wind_available(self) -> np.ndarray:
        return self.wind_lf * self.wind_capacity[None, :]

    def solar_available(self) -> np.ndarray:
        return self.solar_lf * self.solar_capacity[None, :]


def _check_hours(df: pd.DataFrame, name: str, hours: np.ndarray) -> None:
    if not np.array_equal(df["hour"].to_numpy(), hours):
        raise ValueError(f"Sheet '{name}' hour index does not match the demand sheet")


def load_data(path: str | Path) -> MarketData:
    sheets = pd.read_excel(path, sheet_name=None)
    needed = ["windplants", "wind_loadfactors", "solarplants", "solar_loadfactors",
              "gasplants", "demand", "gas_prices"]
    missing = [s for s in needed if s not in sheets]
    if missing:
        raise ValueError(f"Missing sheets: {missing}")

    demand = sheets["demand"].sort_values("hour").reset_index(drop=True)
    hours = demand["hour"].to_numpy()
    wl = sheets["wind_loadfactors"].sort_values("hour").reset_index(drop=True)
    sl = sheets["solar_loadfactors"].sort_values("hour").reset_index(drop=True)
    gp = sheets["gas_prices"].sort_values("hour").reset_index(drop=True)
    for df, nm in ((wl, "wind_loadfactors"), (sl, "solar_loadfactors"), (gp, "gas_prices")):
        _check_hours(df, nm, hours)

    wp, sp, gas = sheets["windplants"], sheets["solarplants"], sheets["gasplants"]
    wind_names, solar_names = wp["name"].tolist(), sp["name"].tolist()

    md = MarketData(
        hours=hours,
        demand=demand["demand"].to_numpy(float),
        gas_price_p_per_therm=gp["price"].to_numpy(float),
        wind_names=wind_names,
        wind_capacity=wp["capacity"].to_numpy(float),
        wind_lf=wl[wind_names].to_numpy(float),
        solar_names=solar_names,
        solar_capacity=sp["capacity"].to_numpy(float),
        solar_lf=sl[solar_names].to_numpy(float),
        gas_names=gas["name"].tolist(),
        gas_capacity=gas["capacity"].to_numpy(float),
        gas_efficiency=gas["efficiency"].to_numpy(float),
    )
    _validate(md)
    return md


def _validate(md: MarketData) -> None:
    arrays = {"demand": md.demand, "gas price": md.gas_price_p_per_therm,
              "wind lf": md.wind_lf, "solar lf": md.solar_lf}
    for name, arr in arrays.items():
        if np.isnan(arr).any():
            raise ValueError(f"NaNs found in {name}")
    for name, lf in (("wind", md.wind_lf), ("solar", md.solar_lf)):
        if lf.min() < 0 or lf.max() > 1:
            raise ValueError(f"{name} load factors must be within [0, 1]")
    if (md.demand < 0).any():
        raise ValueError("Negative demand")
    if (md.gas_efficiency <= 0).any() or (md.gas_efficiency > 1).any():
        raise ValueError("Gas efficiency must be in (0, 1]")
    if (md.gas_price_p_per_therm <= 0).any():
        raise ValueError("Gas prices must be positive")


def _demo(path: str = "data/data.xlsx") -> None:
    np.set_printoptions(suppress=True, linewidth=120, precision=4)  # plain numbers, no 1e-05 style
    md = load_data(path)

    print("=== 1. Time axis and demand ===")
    print("Number of hours        :", md.n_hours)
    print("First 5 hour numbers   :", md.hours[:5])
    print("Last hour number       :", md.hours[-1])
    print("Demand, first 5 hours  :", md.demand[:5], "MWh")
    print(f"Demand min / max       : {md.demand.min():,.0f} / {md.demand.max():,.0f} MWh")

    print("\n=== 2. Wind ===")
    print("Plant names            :", md.wind_names)
    print("Capacity (MW)          :", md.wind_capacity)
    print("Load factors, hours 1-3 (rows = hours, columns = wind1..wind6):")
    print(md.wind_lf[:3])
    print("Available MW, hours 1-3 (= load factor x capacity):")
    print(md.wind_available()[:3])

    print("\n=== 3. Solar ===")
    print("Plant names            :", md.solar_names)
    print("Capacity (MW)          :", md.solar_capacity)
    print("Total available solar MW, hours 1-12 (zero at night, rising with the sun):")
    print(md.solar_available()[:12].sum(axis=1))

    print("\n=== 4. Gas ===")
    print("Plant names            :", md.gas_names)
    print("Capacity (MW)          :", md.gas_capacity)
    print("Efficiency             :", md.gas_efficiency)
    print("Gas price, hours 1-5   :", md.gas_price_p_per_therm[:5], "pence per therm")

    print("\n=== 5. Gas bids (GBP per MWh of electricity) ===")
    p = md.gas_price_p_per_therm[0]
    step1 = p / 100.0 * THERMS_PER_MWH
    print(f"Worked example, hour 1: {p:.0f}p/therm -> GBP {p/100:.2f}/therm "
          f"-> x{THERMS_PER_MWH} = GBP {step1:.2f} per MWh of gas")
    for name, eff, bid in zip(md.gas_names, md.gas_efficiency, md.gas_bids()[0]):
        print(f"  {name}: efficiency {eff:.2f} -> GBP {step1:.2f} / {eff:.2f} = GBP {bid:.2f} per MWh of electricity")
    print("Shape of the full bids grid (hours x plants):", md.gas_bids().shape)
    cheapest = np.array(md.gas_names)[md.gas_bids().argmin(axis=1)]
    print("Cheapest gas plant in every hour:", sorted(set(cheapest)), "(the most efficient plant is always cheapest)")


if __name__ == "__main__":
    import sys

    _demo(sys.argv[1] if len(sys.argv) > 1 else "data/data.xlsx")
