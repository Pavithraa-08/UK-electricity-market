from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from dispatch.auction import DEFAULT_PRICE_CAP, run_auction, technology_mix
from dispatch.data import load_data
from dispatch.report import at_cap, plot_scenario, plot_storage, summarise
from dispatch.storage import StorageParams, consistency_report, run_with_storage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/data.xlsx")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--price-cap", type=float, default=DEFAULT_PRICE_CAP,
                    help="GBP/MWh price when supply cannot meet demand")
    ap.add_argument("--storage-power", type=float, default=StorageParams.power_mw)
    ap.add_argument("--storage-energy", type=float, default=StorageParams.energy_mwh)
    ap.add_argument("--round-trip-eff", type=float, default=StorageParams.round_trip_efficiency)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    md = load_data(args.data)
    cap = args.price_cap

    base = run_auction(md, price_cap=cap)
    base.to_csv(out / "baseline_hourly_results.csv", index=False)
    technology_mix(base).round(2).to_csv(out / "baseline_technology_mix_pct.csv", index=False)
    plot_scenario(base, str(out / "baseline_mix_and_price.png"), "Baseline: generation mix and clearing price", cap)

    params = StorageParams(power_mw=args.storage_power, energy_mwh=args.storage_energy,
                           round_trip_efficiency=args.round_trip_eff)
    withs, _ = run_with_storage(md, params, cap)
    withs.to_csv(out / "storage_hourly_results.csv", index=False)
    technology_mix(withs).round(2).to_csv(out / "storage_technology_mix_pct.csv", index=False)
    plot_scenario(withs, str(out / "storage_mix_and_price.png"), "With storage: generation mix and clearing price", cap)
    plot_storage(withs, str(out / "storage_schedule.png"), params.energy_mwh)

    summary = pd.DataFrame([summarise(base, "No storage", cap),
                            summarise(withs, f"Storage {params.power_mw:.0f} MW / {params.energy_mwh:.0f} MWh", cap)])
    pd.Series(consistency_report(withs, params)).round(4).to_csv(out / "storage_consistency_checks.csv", header=False)
    summary.round(2).T.to_csv(out / "summary.csv", header=False)

    rows = []
    for p_mw in (0, 1000, 2000, 4000, 8000):
        for hrs in (2, 4):
            if p_mw == 0 and hrs == 4:
                continue
            pr = replace(params, power_mw=p_mw, energy_mwh=p_mw * hrs) if p_mw else None
            res = run_auction(md, price_cap=cap) if pr is None else run_with_storage(md, pr, cap)[0]
            s = summarise(res, f"{p_mw} MW / {hrs} h", cap)
            rows.append({"storage": s["scenario"], "hours_with_unserved_energy": s["hours_with_unserved_energy"],
                         "hours_priced_at_cap": s["hours_priced_at_cap"],
                         "unserved_MWh": s["unserved_MWh"], "mean_price": s["mean_price_GBP_per_MWh"],
                         "mean_price_excl_cap_hours": s["mean_price_excl_cap_hours"],
                         "gas_GWh": s["gas_GWh"], "curtailed_GWh": s["curtailed_GWh"],
                         "storage_profit_GBP_m": s.get("storage_profit_GBP_m", 0.0)})
    pd.DataFrame(rows).round(2).to_csv(out / "sensitivity_storage_size.csv", index=False)

    rows = []
    for c in (300, 1000, 4000):
        b = run_auction(md, price_cap=c)
        w, _ = run_with_storage(md, params, c)
        rows.append({"price_cap_GBP_per_MWh": c,
                     "cap_hours_base": int(at_cap(b, c).sum()),
                     "cap_hours_storage": int(at_cap(w, c).sum()),
                     "mean_price_base": b["price_GBP_per_MWh"].mean(),
                     "mean_price_storage": w["price_GBP_per_MWh"].mean(),
                     "storage_profit_GBP_m": w["storage_profit_GBP"].sum() / 1e6,
                     "storage_profit_excl_cap_hours_GBP_m": w.loc[~at_cap(w, c), "storage_profit_GBP"].sum() / 1e6})
    pd.DataFrame(rows).round(2).to_csv(out / "sensitivity_price_cap.csv", index=False)

    print(summary.round(2).T.to_string(header=False))
    print("\nOutputs written to", out.resolve())


if __name__ == "__main__":
    main()
