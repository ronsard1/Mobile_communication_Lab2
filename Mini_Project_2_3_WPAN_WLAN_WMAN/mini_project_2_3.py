"""
Mini Project 2.3 - WPAN / WLAN / WMAN comparison tool.

Loads tech_specs.csv (sourced specification table, sources S1-S8 listed in SOURCES below),
computes derived figures of merit, draws the range-throughput-power trade-off, and checks
three deployment scenarios against each technology with explicit numbered requirements.

    python Mini_Project_2_3_WPAN_WLAN_WMAN/mini_project_2_3.py
"""
import math
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
LOG = []

SOURCES = {
    "S1": "Bluetooth SIG, Understanding Bluetooth range: bluetooth.com/learn-about-bluetooth/key-attributes/range/",
    "S2": "MathWorks, Comparison of Bluetooth BR/EDR and Bluetooth LE: mathworks.com/help/bluetooth/gs/"
          "comparison-of-bluetooth-bredr-and-bluetooth-le.html",
    "S3": "Wikipedia, Zigbee: en.wikipedia.org/wiki/Zigbee",
    "S4": "RF Wireless World, Understanding IEEE 802.11 Wi-Fi standards: rfwireless-world.com/articles/"
          "understanding-ieee-802-11-wi-fi-standards",
    "S5": "Cisco, 802.11ac wireless throughput testing and expectations (document 212892): cisco.com",
    "S6": "Powertec technical library, 802.11ax Wi-Fi 6: portal.powertec.com.au/technical-library/wireless/"
          "wireless-technologies/wifi/80211ax-wifi-6",
    "S7": "RF Wireless World, Understanding IEEE 802.16 WiMAX standards: rfwireless-world.com/articles/"
          "understanding-ieee-802-16-wimax-standards",
    "S8": "SpeedtestHQ, Wi-Fi power output (mW and dBm): speedtesthq.com/guides/wifi/wifi-power-output-mw-dbm",
}


def say(s=""):
    print(s)
    LOG.append(s)


df = pd.read_csv(os.path.join(HERE, "tech_specs.csv"))
df["tx_power_mW"] = 10 ** (df.tx_power_dBm / 10)
df["rate_x_range"] = df.typical_rate_Mbps * df.range_typ_m                  # Mbps*m figure of merit
df["Mbps_per_mW"] = df.typical_rate_Mbps / df.tx_power_mW                  # rate delivered per mW radiated
df["coverage_m2"] = math.pi * df.range_typ_m ** 2
df["area_capacity_Mbps_per_km2"] = df.typical_rate_Mbps / (df.coverage_m2 / 1e6)

say("=" * 96)
say("Dataset (typical values used for decisions; peak values are standard maxima)")
cols = ["tech", "class", "peak_rate_Mbps", "typical_rate_Mbps", "range_typ_m", "range_max_m",
        "tx_power_dBm", "power_class", "rate_x_range", "Mbps_per_mW", "area_capacity_Mbps_per_km2"]
with pd.option_context("display.width", 200, "display.max_columns", 20):
    say(df[cols].to_string(index=False, float_format=lambda v: f"{v:,.3g}"))
df.to_csv(os.path.join(HERE, "results_2_3_dataset_with_metrics.csv"), index=False)

# ======================================================================= trade-off chart
CLASS_COL = {"WPAN": "#2ca02c", "WLAN": "#1f77b4", "WMAN": "#d62728"}
fig, ax = plt.subplots(figsize=(9.5, 6))
for _, r in df.iterrows():
    size = 30 + 12 * (r.tx_power_dBm + 5)            # marker area grows with transmit power
    ax.errorbar(r.range_typ_m, r.typical_rate_Mbps,
                xerr=[[r.range_typ_m - r.range_min_m], [r.range_max_m - r.range_typ_m]],
                fmt="none", ecolor=CLASS_COL[r["class"]], alpha=0.35, lw=1)
    ax.scatter(r.range_typ_m, r.typical_rate_Mbps, s=size, color=CLASS_COL[r["class"]],
               edgecolor="black", linewidth=0.6, zorder=3)
    ax.scatter(r.range_typ_m, r.peak_rate_Mbps, marker="_", s=120, color=CLASS_COL[r["class"]], zorder=3)
    ax.plot([r.range_typ_m] * 2, [r.typical_rate_Mbps, r.peak_rate_Mbps], ":", color=CLASS_COL[r["class"]], lw=1)
    off = {"BLE (LE 1M/2M)": (8, 8), "Zigbee": (8, -12), "Wi-Fi 5 (802.11ac)": (8, -14),
           "Wi-Fi 6 (802.11ax)": (8, 6), "Mobile WiMAX (802.16e)": (-10, -16),
           "WiMAX fixed (802.16d)": (8, 4), "WiMAX 2 (802.16m)": (8, 6), "Bluetooth Classic (BR/EDR)": (-8, 6)}.get(r.tech, (8, 4))
    ax.annotate(f"{r.tech}\n{r.tx_power_dBm:.0f} dBm", (r.range_typ_m, r.typical_rate_Mbps), xytext=off,
                textcoords="offset points", fontsize=7.5,
                ha="right" if off[0] < 0 else "left")
SCEN_PTS = {"S1 smart home": (15, 0.01), "S2 campus (per user)": (35, 5), "S3 metro link": (8000, 20)}
for name, (x, y) in SCEN_PTS.items():
    ax.scatter(x, y, marker="*", s=220, color="#ff7f0e", edgecolor="black", zorder=4)
    ax.annotate(name, (x, y), xytext=(8, 6) if name.startswith("S2") else (6, -12), textcoords="offset points", fontsize=8, color="#b35900",
                fontweight="bold")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("range (m, log scale): dot = typical, bar = min to max")
ax.set_ylabel("data rate (Mbps, log scale): dot = typical, dash = peak")
ax.set_title("Range versus throughput; marker size = transmit power; stars = scenario requirements", fontsize=10)
ax.grid(alpha=0.3, which="both")
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c, mec="black", label=k) for k, c in CLASS_COL.items()]
          + [Line2D([], [], marker="*", ls="", color="#ff7f0e", mec="black", ms=12, label="scenario requirement")],
          fontsize=8, loc="lower right")
ax.set_xlim(0.7, 5e4)
ax.set_ylim(5e-3, 2e4)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_tradeoff_scatter.png"), dpi=160)
plt.close(fig)

# ======================================================================= metric bars
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
order = df.sort_values(["class", "typical_rate_Mbps"]).reset_index(drop=True)
y = np.arange(len(order))
colors = [CLASS_COL[c] for c in order["class"]]
for ax, col, title, logx in ((axes[0], "tx_power_dBm", "transmit power (dBm)", False),
                             (axes[1], "rate_x_range", "typical rate x typical range (Mbps.m)", True),
                             (axes[2], "Mbps_per_mW", "typical rate per mW radiated (Mbps/mW)", True)):
    vals = order[col].values
    if logx:
        ax.barh(y, vals, color=colors, edgecolor="black", linewidth=0.4)
        ax.set_xscale("log")
    else:
        ax.barh(y, vals - (-5), left=-5, color=colors, edgecolor="black", linewidth=0.4)
    for yi, v in zip(y, vals):
        ax.text(v * (1.15 if logx else 1) + (0 if logx else 0.8), yi, f"{v:,.3g}", va="center", fontsize=7)
    ax.set_title(title, fontsize=9.5)
    ax.grid(axis="x", alpha=0.3, which="both")
axes[0].set_yticks(y)
axes[0].set_yticklabels(order.tech, fontsize=8)
axes[0].set_xlim(-5, 52)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_metrics.png"), dpi=160)
plt.close(fig)

# ======================================================================= scenarios
POWER_RANK = {"very low": 0, "low": 1, "medium": 2, "high": 3}
scenarios = [
    {
        "id": "S1", "name": "Smart home with many low-power sensors",
        "reqs": ["R1 coverage: every room of a 150 m2 house, longest sensor-to-hub path about 15 m",
                 "R2 devices: 40 battery sensors (door, temperature, motion)",
                 "R3 throughput: each sensor sends a 100-byte report every 10 s (80 bit/s), bursts below 10 kbps; "
                 "aggregate average 40 x 80 bit/s = 3.2 kbps; channel load must stay under 50%",
                 "R4 power: at least 2 years on a coin or AA cell, so very low or low power class",
                 "R5 cost/licensing: unlicensed band, cheap modules"],
        "range_m": 15, "link_rate_Mbps": 0.01, "max_power": "low", "mobility": None, "licensed": None,
        "aggregate_Mbps": 0.0032,
    },
    {
        "id": "S2", "name": "Campus network, wide coverage and high aggregate throughput",
        "reqs": ["R1 coverage: 40,000 m2 of indoor floor space in several buildings",
                 "R2 devices: 600 concurrent users (laptops and phones)",
                 "R3 throughput: 5 Mbps per user, 3 Gbps aggregate",
                 "R4 power: access points mains powered; client radios must stay within phone budgets (<= 20 dBm)",
                 "R5 cost/licensing: unlicensed, uses the laptops' and phones' built-in radios"],
        "range_m": 35, "link_rate_Mbps": 5, "max_power": "medium", "mobility": None, "licensed": None,
        "aggregate_Mbps": 3000, "area_m2": 40000,
    },
    {
        "id": "S3", "name": "Metro-area fixed wireless link of several kilometres",
        "reqs": ["R1 coverage: one point-to-point (or point-to-multipoint) hop of 8 km",
                 "R2 devices: one branch office router behind the subscriber unit",
                 "R3 throughput: 20 Mbps sustained",
                 "R4 power: both ends mains powered, high power class acceptable",
                 "R5 licensing: licensed spectrum preferred to avoid interference on a business link"],
        "range_m": 8000, "link_rate_Mbps": 20, "max_power": "high", "mobility": None, "licensed": "yes",
        "aggregate_Mbps": 20,
    },
]

matrix = {}
for sc in scenarios:
    say("=" * 96)
    say(f"{sc['id']}: {sc['name']}")
    for r in sc["reqs"]:
        say("   " + r)
    say(f"   {'technology':28s}{'range':>8s}{'rate':>8s}{'power':>8s}{'lic.':>6s}  units needed / notes")
    best = []
    for _, t in df.iterrows():
        ok_range = t.range_typ_m >= sc["range_m"]
        ok_range_max = t.range_max_m >= sc["range_m"]
        ok_rate = t.typical_rate_Mbps >= sc["link_rate_Mbps"]
        ok_power = POWER_RANK[t.power_class] <= POWER_RANK[sc["max_power"]]
        ok_lic = True if sc["licensed"] is None else (t.licensed == sc["licensed"])
        note = ""
        if sc["id"] == "S2":
            by_area = math.ceil(1.5 * sc["area_m2"] / t.coverage_m2)        # 50% overlap for handover/capacity
            by_rate = math.ceil(sc["aggregate_Mbps"] / t.typical_rate_Mbps)
            units = max(by_area, by_rate)
            note = f"{units:,} cells (coverage {by_area:,}, capacity {by_rate:,})"
        elif sc["id"] == "S1":
            util = sc["aggregate_Mbps"] / t.typical_rate_Mbps
            note = f"channel load {100 * util:.1f}% of typical rate"
        else:
            note = f"rate margin x{t.typical_rate_Mbps / sc['link_rate_Mbps']:.1f}, range margin x{t.range_max_m / sc['range_m']:.2f} (max)"
        ok_load = True
        if sc["id"] == "S1":
            ok_load = sc["aggregate_Mbps"] / t.typical_rate_Mbps <= 0.5
        verdict = ok_range and ok_rate and ok_power and ok_lic and ok_load
        rng_sym = "yes" if ok_range else ("max" if ok_range_max else "no")
        say(f"   {t.tech:28s}{rng_sym:>8s}{('yes' if ok_rate else 'no'):>8s}{('yes' if ok_power else 'no'):>8s}"
            f"{('yes' if ok_lic else 'no'):>6s}  {note}{'   <- meets all' if verdict else ''}")
        matrix[(sc["id"], t.tech)] = 2 if verdict else (1 if (ok_range_max and ok_rate and ok_power and ok_lic) else 0)
        if verdict:
            best.append(t)
    if best:
        # preference: lowest power class first, then highest typical rate
        best.sort(key=lambda t: (POWER_RANK[t.power_class], -t.typical_rate_Mbps))
        say(f"   tool ranking of feasible options: {', '.join(t.tech for t in best)}")

# ======================================================================= feasibility heat map
fig, ax = plt.subplots(figsize=(8.5, 3.2))
grid = np.array([[matrix[(sc["id"], t)] for t in df.tech] for sc in scenarios])
from matplotlib.colors import ListedColormap  # noqa: E402
ax.imshow(grid, cmap=ListedColormap(["#f4cccc", "#fff2cc", "#d9ead3"]), vmin=0, vmax=2, aspect="auto")
labels = {0: "fails", 1: "only at max range", 2: "meets all"}
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        ax.text(j, i, labels[grid[i, j]], ha="center", va="center", fontsize=7)
ax.set_xticks(range(len(df)))
ax.set_xticklabels(df.tech, rotation=30, ha="right", fontsize=7.5)
ax.set_yticks(range(len(scenarios)))
ax.set_yticklabels(["S1 smart home", "S2 campus network", "S3 metro fixed link"], fontsize=8)
ax.set_title("Which technologies meet each scenario's numeric requirements", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_scenario_feasibility.png"), dpi=160)
plt.close(fig)

say("=" * 96)
say("Sources")
for k, v in SOURCES.items():
    say(f"  [{k}] {v}")
with open(os.path.join(HERE, "results_2_3.txt"), "w") as fh:
    fh.write("\n".join(LOG) + "\n")
print("figures written to", FIG)
