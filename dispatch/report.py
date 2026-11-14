"""Summary tables and charts."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def at_cap(result: pd.DataFrame, price_cap: float) -> pd.Series:
    """Hours priced at the scarcity cap (unserved energy, or storage covering the gap at the cap)."""
    return result["price_GBP_per_MWh"] >= 0.999 * price_cap


def summarise(result: pd.DataFrame, label: str, price_cap: float) -> dict:
    """Headline numbers for one scenario."""
    served = result["demand_MWh"] - result["unserved_MWh"]
    price = result["price_GBP_per_MWh"]
    capped = at_cap(result, price_cap)
    out = {
        "scenario": label,
        "mean_price_GBP_per_MWh": price.mean(),
        "mean_price_excl_cap_hours": price[~capped].mean(),
        "hours_with_unserved_energy": int((result["unserved_MWh"] > 1e-6).sum()),
        "hours_priced_at_cap": int(capped.sum()),
        "unserved_MWh": result["unserved_MWh"].sum(),
        "hours_at_zero_price": int((price == 0).sum()),
        "wind_GWh": result["wind_MWh"].sum() / 1e3,
        "solar_GWh": result["solar_MWh"].sum() / 1e3,
        "gas_GWh": result["gas_MWh"].sum() / 1e3,
        "curtailed_GWh": result["curtailed_MWh"].sum() / 1e3,
        "consumer_cost_GBP_m": (price * served).sum() / 1e6,
    }
    if "storage_discharge_MWh" in result:
        out.update({
            "storage_discharge_GWh": result["storage_discharge_MWh"].sum() / 1e3,
            "storage_charge_GWh": result["storage_charge_MWh"].sum() / 1e3,
            "storage_profit_GBP_m": result["storage_profit_GBP"].sum() / 1e6,
        })
    return out


def plot_scenario(result: pd.DataFrame, path: str, title: str, price_cap: float, price_clip: float = 150.0) -> None:
    """Stacked generation mix + price (clipped so scarcity spikes do not hide the rest)."""
    h = result["hour"].to_numpy()
    fig, ax = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1.6]})
    layers = [("Wind", result["wind_MWh"], "#4C9F70"), ("Solar", result["solar_MWh"], "#F2B134"),
              ("Gas", result["gas_MWh"], "#6C7A89")]
    if "storage_discharge_MWh" in result:
        layers.append(("Storage discharge", result["storage_discharge_MWh"], "#7B5EA7"))
    layers.append(("Unserved", result["unserved_MWh"], "#D1495B"))
    ax[0].stackplot(h, [l[1] for l in layers], labels=[l[0] for l in layers], colors=[l[2] for l in layers])
    ax[0].plot(h, result["net_demand_MWh"], color="black", lw=1.2, ls="--", label="Demand (incl. storage charging)")
    ax[0].set_ylabel("MWh per hour")
    ax[0].set_title(title)
    ax[0].legend(loc="upper right", ncol=3, fontsize=8)
    price = result["price_GBP_per_MWh"]
    ax[1].step(h, price.clip(upper=price_clip), where="mid", color="#1f4e79")
    spikes = at_cap(result, price_cap).to_numpy()
    ax[1].scatter(h[spikes], np.full(spikes.sum(), price_clip), marker="^", color="#D1495B",
                  label=f"hour priced at the cap (GBP {price_cap:,.0f}/MWh; axis clipped at {price_clip:,.0f})", s=22)
    ax[1].set_ylabel("Price GBP/MWh")
    ax[1].set_xlabel("Hour")
    ax[1].legend(loc="upper left", fontsize=8)
    for a in ax:
        a.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_storage(result: pd.DataFrame, path: str, energy_mwh: float) -> None:
    h = result["hour"].to_numpy()
    fig, ax = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)
    ax[0].bar(h, result["storage_discharge_MWh"], color="#7B5EA7", label="Discharge", width=0.9)
    ax[0].bar(h, -result["storage_charge_MWh"], color="#4C9F70", label="Charge", width=0.9)
    ax[0].set_ylabel("MWh per hour")
    ax[0].legend(loc="upper right", fontsize=8)
    ax[0].set_title("Storage schedule and state of charge")
    ax[1].plot(h, result["storage_soc_MWh"] / energy_mwh * 100, color="#7B5EA7")
    ax[1].set_ylabel("State of charge (%)")
    ax[1].set_xlabel("Hour")
    for a in ax:
        a.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
