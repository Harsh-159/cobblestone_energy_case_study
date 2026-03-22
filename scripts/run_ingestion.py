#!/usr/bin/env python3
"""
Entry point for the DE_LU power data ingestion pipeline.

Runs the ingestion pipeline and generates a comprehensive visual dashboard
of all key metrics using matplotlib.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# Ensure project root is on sys.path so `src` is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion import run_ingestion_pipeline, OUTPUTS_DIR

# ---------------------------------------------------------------------------
# Plot style configuration
# ---------------------------------------------------------------------------
COLORS = {
    "da_price": "#1f77b4",       # blue
    "wind": "#2ca02c",           # green
    "solar": "#ff7f0e",          # orange
    "load": "#d62728",           # red
    "gas": "#9467bd",            # purple
    "grid": "#cccccc",
    "highlight": "#e74c3c",
    "bg": "#fafafa",
}

PLOTS_DIR = OUTPUTS_DIR / "plots"


def setup_plot_style():
    """Configure a clean, professional matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": COLORS["bg"],
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.color": COLORS["grid"],
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "figure.dpi": 130,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


# ---------------------------------------------------------------------------
# Chart 1: Full time series overview (all 5 variables)
# ---------------------------------------------------------------------------
def plot_full_timeseries(df: pd.DataFrame):
    """5-panel stacked time series of every variable across the full 4 years."""
    fig, axes = plt.subplots(5, 1, figsize=(18, 16), sharex=True)
    fig.suptitle("DE_LU Power Market — Full 4-Year Overview (2021–2024)", fontsize=16, fontweight="bold", y=0.98)

    configs = [
        ("da_price_eur_mwh", "DA Price (EUR/MWh)", COLORS["da_price"]),
        ("wind_forecast_mw", "Wind Forecast (MW)", COLORS["wind"]),
        ("solar_forecast_mw", "Solar Forecast (MW)", COLORS["solar"]),
        ("load_forecast_mw", "Load Forecast (MW)", COLORS["load"]),
        ("gas_price_eur_mwh", "Gas Price TTF (EUR/MWh)", COLORS["gas"]),
    ]

    for ax, (col, label, color) in zip(axes, configs):
        ax.plot(df.index, df[col], color=color, linewidth=0.3, alpha=0.7)
        # Add 7-day rolling mean for clarity
        rolling = df[col].rolling(168, min_periods=24).mean()  # 168h = 7 days
        ax.plot(df.index, rolling, color=color, linewidth=1.5, alpha=0.9, label="7-day avg")
        ax.set_ylabel(label, fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
        ax.tick_params(axis="y", labelsize=8)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(PLOTS_DIR / "01_full_timeseries.png")
    plt.close(fig)
    print("  [1/10] Full time series overview saved")


# ---------------------------------------------------------------------------
# Chart 2: DA Price distribution per year (violin + box)
# ---------------------------------------------------------------------------
def plot_price_distribution_by_year(df: pd.DataFrame):
    """Yearly box plots of DA prices showing crisis impact."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("DA Price Distribution by Year", fontsize=14, fontweight="bold")

    df_clean = df[["da_price_eur_mwh"]].dropna().copy()
    df_clean["year"] = df_clean.index.year

    years = sorted(df_clean["year"].unique())
    data_by_year = [df_clean.loc[df_clean["year"] == y, "da_price_eur_mwh"].values for y in years]

    # Box plot
    bp = ax1.boxplot(data_by_year, tick_labels=years, patch_artist=True, showfliers=True,
                     flierprops={"marker": ".", "markersize": 1, "alpha": 0.3})
    year_colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]
    for patch, color in zip(bp["boxes"], year_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax1.set_ylabel("EUR/MWh")
    ax1.set_title("Box Plot (with outliers)")
    ax1.axhline(y=0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)

    # Histogram overlay
    for y, color in zip(years, year_colors):
        vals = df_clean.loc[df_clean["year"] == y, "da_price_eur_mwh"]
        ax2.hist(vals, bins=80, alpha=0.5, label=str(y), color=color, density=True)
    ax2.set_xlabel("EUR/MWh")
    ax2.set_ylabel("Density")
    ax2.set_title("Price Density by Year")
    ax2.legend()
    ax2.set_xlim(-100, 500)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "02_price_distribution_by_year.png")
    plt.close(fig)
    print("  [2/10] Price distribution by year saved")


# ---------------------------------------------------------------------------
# Chart 3: Average daily profile by season (hourly pattern)
# ---------------------------------------------------------------------------
def plot_daily_profiles(df: pd.DataFrame):
    """Hourly average profiles for price, wind, solar, and load by season."""
    df_copy = df.copy()
    df_copy["hour"] = df_copy.index.hour
    month = df_copy.index.month
    df_copy["season"] = np.select(
        [month.isin([12, 1, 2]), month.isin([3, 4, 5]),
         month.isin([6, 7, 8]), month.isin([9, 10, 11])],
        ["Winter", "Spring", "Summer", "Autumn"],
        default="Unknown"
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Average Hourly Profiles by Season (UTC)", fontsize=14, fontweight="bold")

    cols = [
        ("da_price_eur_mwh", "DA Price (EUR/MWh)", axes[0, 0]),
        ("wind_forecast_mw", "Wind Forecast (MW)", axes[0, 1]),
        ("solar_forecast_mw", "Solar Forecast (MW)", axes[1, 0]),
        ("load_forecast_mw", "Load Forecast (MW)", axes[1, 1]),
    ]
    season_colors = {"Winter": "#2980b9", "Spring": "#27ae60", "Summer": "#f39c12", "Autumn": "#c0392b"}
    season_order = ["Winter", "Spring", "Summer", "Autumn"]

    for col, ylabel, ax in cols:
        for season in season_order:
            mask = df_copy["season"] == season
            hourly_avg = df_copy.loc[mask].groupby("hour")[col].mean()
            ax.plot(hourly_avg.index, hourly_avg.values, label=season,
                    color=season_colors[season], linewidth=2, marker="o", markersize=3)
        ax.set_xlabel("Hour (UTC)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=8)
        ax.set_xticks(range(0, 24, 3))

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(PLOTS_DIR / "03_daily_profiles_by_season.png")
    plt.close(fig)
    print("  [3/10] Daily profiles by season saved")


# ---------------------------------------------------------------------------
# Chart 4: Monthly statistics heatmap
# ---------------------------------------------------------------------------
def plot_monthly_heatmap(df: pd.DataFrame):
    """Heatmap of average monthly DA prices across years."""
    df_copy = df[["da_price_eur_mwh"]].dropna().copy()
    df_copy["year"] = df_copy.index.year
    df_copy["month"] = df_copy.index.month

    pivot = df_copy.groupby(["year", "month"])["da_price_eur_mwh"].mean().unstack(level=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.set_yticklabels([month_names[m - 1] for m in pivot.index])

    # Annotate cells with values
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                text_color = "white" if val > 150 else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=9, color=text_color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, label="Avg DA Price (EUR/MWh)")
    ax.set_title("Monthly Average DA Price Heatmap (EUR/MWh)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "04_monthly_price_heatmap.png")
    plt.close(fig)
    print("  [4/10] Monthly price heatmap saved")


# ---------------------------------------------------------------------------
# Chart 5: Wind & Solar vs Price scatter
# ---------------------------------------------------------------------------
def plot_renewables_vs_price(df: pd.DataFrame):
    """Scatter plots: renewable generation vs DA price (merit order effect)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Renewable Generation vs DA Price (Merit Order Effect)", fontsize=14, fontweight="bold")

    sample = df.dropna().sample(n=min(15000, len(df.dropna())), random_state=42)

    # Wind vs Price
    ax = axes[0]
    ax.scatter(sample["wind_forecast_mw"] / 1000, sample["da_price_eur_mwh"],
               alpha=0.1, s=3, color=COLORS["wind"])
    ax.set_xlabel("Wind Forecast (GW)")
    ax.set_ylabel("DA Price (EUR/MWh)")
    ax.set_title("Wind vs Price")
    corr_w = df["wind_forecast_mw"].corr(df["da_price_eur_mwh"])
    ax.text(0.05, 0.95, f"r = {corr_w:.3f}", transform=ax.transAxes,
            fontsize=11, va="top", fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Solar vs Price
    ax = axes[1]
    daytime = sample[sample.index.hour.isin(range(6, 20))]
    ax.scatter(daytime["solar_forecast_mw"] / 1000, daytime["da_price_eur_mwh"],
               alpha=0.1, s=3, color=COLORS["solar"])
    ax.set_xlabel("Solar Forecast (GW)")
    ax.set_ylabel("DA Price (EUR/MWh)")
    ax.set_title("Solar vs Price (daytime only)")
    corr_s = df.loc[df.index.hour.isin(range(6, 20)), "solar_forecast_mw"].corr(
        df.loc[df.index.hour.isin(range(6, 20)), "da_price_eur_mwh"])
    ax.text(0.05, 0.95, f"r = {corr_s:.3f}", transform=ax.transAxes,
            fontsize=11, va="top", fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Combined renewables vs Price
    ax = axes[2]
    re_total = sample["wind_forecast_mw"] + sample["solar_forecast_mw"]
    ax.scatter(re_total / 1000, sample["da_price_eur_mwh"],
               alpha=0.1, s=3, color="#17a589")
    ax.set_xlabel("Wind + Solar (GW)")
    ax.set_ylabel("DA Price (EUR/MWh)")
    ax.set_title("Combined Renewables vs Price")
    re_full = df["wind_forecast_mw"] + df["solar_forecast_mw"]
    corr_re = re_full.corr(df["da_price_eur_mwh"])
    ax.text(0.05, 0.95, f"r = {corr_re:.3f}", transform=ax.transAxes,
            fontsize=11, va="top", fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(PLOTS_DIR / "05_renewables_vs_price.png")
    plt.close(fig)
    print("  [5/10] Renewables vs price scatter saved")


# ---------------------------------------------------------------------------
# Chart 6: Gas price vs DA price (fuel cost driver)
# ---------------------------------------------------------------------------
def plot_gas_vs_da_price(df: pd.DataFrame):
    """Dual-axis time series and scatter of gas vs electricity price."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Gas Price (TTF) vs DA Electricity Price", fontsize=14, fontweight="bold")

    # Time series — daily averages
    daily = df.resample("D").mean()
    ax1.plot(daily.index, daily["gas_price_eur_mwh"], color=COLORS["gas"],
             linewidth=1.2, label="Gas TTF", alpha=0.9)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(daily.index, daily["da_price_eur_mwh"], color=COLORS["da_price"],
                  linewidth=1.2, label="DA Price", alpha=0.9)
    ax1.set_ylabel("Gas TTF (EUR/MWh)", color=COLORS["gas"])
    ax1_twin.set_ylabel("DA Price (EUR/MWh)", color=COLORS["da_price"])
    ax1.set_title("Daily Averages Over Time")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # Add legends for both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    # Scatter
    sample = df.dropna().sample(n=min(15000, len(df.dropna())), random_state=42)
    ax2.scatter(sample["gas_price_eur_mwh"], sample["da_price_eur_mwh"],
                alpha=0.08, s=3, color=COLORS["gas"])
    corr = df["gas_price_eur_mwh"].corr(df["da_price_eur_mwh"])
    ax2.set_xlabel("Gas TTF (EUR/MWh)")
    ax2.set_ylabel("DA Price (EUR/MWh)")
    ax2.set_title(f"Scatter (r = {corr:.3f})")

    # Fit line
    valid = df[["gas_price_eur_mwh", "da_price_eur_mwh"]].dropna()
    z = np.polyfit(valid["gas_price_eur_mwh"], valid["da_price_eur_mwh"], 1)
    x_line = np.linspace(valid["gas_price_eur_mwh"].min(), valid["gas_price_eur_mwh"].max(), 100)
    ax2.plot(x_line, np.polyval(z, x_line), color=COLORS["highlight"], linewidth=2,
             linestyle="--", label=f"Fit: {z[0]:.2f}x + {z[1]:.1f}")
    ax2.legend(fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(PLOTS_DIR / "06_gas_vs_da_price.png")
    plt.close(fig)
    print("  [6/10] Gas vs DA price saved")


# ---------------------------------------------------------------------------
# Chart 7: Missing data visualization
# ---------------------------------------------------------------------------
def plot_missing_data(df: pd.DataFrame):
    """Heatmap showing data availability by month for each column."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("Data Quality — Missing Values", fontsize=14, fontweight="bold")

    # Bar chart of total missing per column
    missing_counts = df.isna().sum()
    missing_pct = (df.isna().sum() / len(df) * 100)
    colors_bar = [COLORS["da_price"], COLORS["wind"], COLORS["solar"], COLORS["load"], COLORS["gas"]]
    bars = ax1.bar(range(len(missing_counts)), missing_counts.values, color=colors_bar, alpha=0.8)
    ax1.set_xticks(range(len(missing_counts)))
    ax1.set_xticklabels([c.replace("_", "\n") for c in df.columns], fontsize=8)
    ax1.set_ylabel("Missing Count")
    ax1.set_title("Total Missing Values per Column")
    for bar, pct in zip(bars, missing_pct):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 f"{pct:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Monthly missing heatmap
    monthly_miss = df.isna().groupby([df.index.year, df.index.month]).mean() * 100
    monthly_miss.index = [f"{y}-{m:02d}" for y, m in monthly_miss.index]
    im = ax2.imshow(monthly_miss.values.T, cmap="Reds", aspect="auto", vmin=0, vmax=10)
    ax2.set_yticks(range(len(df.columns)))
    ax2.set_yticklabels([c.replace("_", "\n") for c in df.columns], fontsize=8)
    ax2.set_xticks(range(0, len(monthly_miss), 3))
    ax2.set_xticklabels([monthly_miss.index[i] for i in range(0, len(monthly_miss), 3)],
                        rotation=45, fontsize=7)
    ax2.set_title("Monthly Missing Data % (red = more missing)")
    fig.colorbar(im, ax=ax2, label="% Missing", shrink=0.6)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(PLOTS_DIR / "07_missing_data.png")
    plt.close(fig)
    print("  [7/10] Missing data visualization saved")


# ---------------------------------------------------------------------------
# Chart 8: Renewable penetration ratio vs price
# ---------------------------------------------------------------------------
def plot_renewable_penetration(df: pd.DataFrame):
    """Shows how renewable share of load affects prices."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Renewable Penetration Ratio (Wind+Solar / Load)", fontsize=14, fontweight="bold")

    valid = df.dropna().copy()
    valid["re_penetration"] = (valid["wind_forecast_mw"] + valid["solar_forecast_mw"]) / valid["load_forecast_mw"]

    # Time series of monthly average penetration
    monthly_pen = valid["re_penetration"].resample("ME").mean() * 100
    ax1.bar(monthly_pen.index, monthly_pen.values, width=25, color="#27ae60", alpha=0.7)
    ax1.set_ylabel("Avg Renewable Penetration (%)")
    ax1.set_title("Monthly Average Penetration")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    ax1.axhline(y=100, color="red", linestyle="--", alpha=0.5, label="100% penetration")
    ax1.legend(fontsize=8)

    # Scatter: penetration vs price
    sample = valid.sample(n=min(15000, len(valid)), random_state=42)
    scatter = ax2.scatter(sample["re_penetration"] * 100, sample["da_price_eur_mwh"],
                          c=sample["gas_price_eur_mwh"], cmap="plasma",
                          alpha=0.15, s=3)
    fig.colorbar(scatter, ax=ax2, label="Gas Price (EUR/MWh)", shrink=0.8)
    ax2.set_xlabel("Renewable Penetration (%)")
    ax2.set_ylabel("DA Price (EUR/MWh)")
    ax2.set_title("Penetration vs Price (colored by gas)")
    ax2.set_xlim(0, 200)
    ax2.set_ylim(-100, 500)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(PLOTS_DIR / "08_renewable_penetration.png")
    plt.close(fig)
    print("  [8/10] Renewable penetration saved")


# ---------------------------------------------------------------------------
# Chart 9: Weekly pattern and weekday vs weekend
# ---------------------------------------------------------------------------
def plot_weekly_patterns(df: pd.DataFrame):
    """Day-of-week price and load profiles."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Weekly Patterns", fontsize=14, fontweight="bold")

    df_copy = df.dropna().copy()
    df_copy["dow"] = df_copy.index.dayofweek  # 0=Mon, 6=Sun
    df_copy["hour"] = df_copy.index.hour
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    # Average price by day of week
    avg_by_dow = df_copy.groupby("dow")["da_price_eur_mwh"].mean()
    colors_dow = ["#3498db"] * 5 + ["#e74c3c"] * 2
    axes[0].bar(range(7), avg_by_dow.values, color=colors_dow, alpha=0.8)
    axes[0].set_xticks(range(7))
    axes[0].set_xticklabels(dow_names)
    axes[0].set_ylabel("Avg DA Price (EUR/MWh)")
    axes[0].set_title("Average Price by Day of Week")

    # Average load by day of week
    avg_load_dow = df_copy.groupby("dow")["load_forecast_mw"].mean() / 1000
    axes[1].bar(range(7), avg_load_dow.values, color=colors_dow, alpha=0.8)
    axes[1].set_xticks(range(7))
    axes[1].set_xticklabels(dow_names)
    axes[1].set_ylabel("Avg Load (GW)")
    axes[1].set_title("Average Load by Day of Week")

    # Hourly profiles: weekday vs weekend
    weekday_mask = df_copy["dow"] < 5
    for label, mask, color, ls in [("Weekday", weekday_mask, "#2980b9", "-"),
                                    ("Weekend", ~weekday_mask, "#e74c3c", "--")]:
        hourly = df_copy.loc[mask].groupby("hour")["da_price_eur_mwh"].mean()
        axes[2].plot(hourly.index, hourly.values, label=label, color=color,
                     linewidth=2.5, linestyle=ls)
    axes[2].set_xlabel("Hour (UTC)")
    axes[2].set_ylabel("Avg DA Price (EUR/MWh)")
    axes[2].set_title("Weekday vs Weekend Hourly Price")
    axes[2].legend()
    axes[2].set_xticks(range(0, 24, 3))

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(PLOTS_DIR / "09_weekly_patterns.png")
    plt.close(fig)
    print("  [9/10] Weekly patterns saved")


# ---------------------------------------------------------------------------
# Chart 10: Summary statistics dashboard
# ---------------------------------------------------------------------------
def plot_summary_dashboard(df: pd.DataFrame):
    """Single-page dashboard with key statistics and mini-charts."""
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("DE_LU Power Market — Data Summary Dashboard", fontsize=16, fontweight="bold", y=0.98)

    # --- Top row: key stats as text ---
    ax_text = fig.add_axes([0.02, 0.82, 0.96, 0.13])
    ax_text.axis("off")

    stats_text = (
        f"Dataset: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}  |  "
        f"Rows: {len(df):,}  |  "
        f"Missing: {df.isna().sum().sum():,} total  |  "
        f"DA Price: mean={df['da_price_eur_mwh'].mean():.1f}, "
        f"min={df['da_price_eur_mwh'].min():.1f}, "
        f"max={df['da_price_eur_mwh'].max():.1f} EUR/MWh  |  "
        f"Gas: mean={df['gas_price_eur_mwh'].mean():.1f} EUR/MWh"
    )
    ax_text.text(0.5, 0.5, stats_text, ha="center", va="center", fontsize=11,
                 fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#ecf0f1", alpha=0.8))

    # --- Row 2: Monthly averages ---
    ax1 = fig.add_subplot(3, 3, 4)
    monthly_price = df["da_price_eur_mwh"].resample("ME").mean()
    ax1.fill_between(monthly_price.index, monthly_price.values, alpha=0.4, color=COLORS["da_price"])
    ax1.plot(monthly_price.index, monthly_price.values, color=COLORS["da_price"], linewidth=1.5)
    ax1.set_title("Monthly Avg DA Price", fontsize=10)
    ax1.set_ylabel("EUR/MWh", fontsize=8)
    ax1.tick_params(labelsize=7)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax2 = fig.add_subplot(3, 3, 5)
    monthly_wind = df["wind_forecast_mw"].resample("ME").mean() / 1000
    monthly_solar = df["solar_forecast_mw"].resample("ME").mean() / 1000
    ax2.bar(monthly_wind.index, monthly_wind.values, width=25, color=COLORS["wind"], alpha=0.7, label="Wind")
    ax2.bar(monthly_solar.index, monthly_solar.values, width=25, color=COLORS["solar"],
            alpha=0.7, bottom=monthly_wind.values, label="Solar")
    ax2.set_title("Monthly Avg Renewables", fontsize=10)
    ax2.set_ylabel("GW", fontsize=8)
    ax2.legend(fontsize=7)
    ax2.tick_params(labelsize=7)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax3 = fig.add_subplot(3, 3, 6)
    monthly_load = df["load_forecast_mw"].resample("ME").mean() / 1000
    ax3.plot(monthly_load.index, monthly_load.values, color=COLORS["load"], linewidth=1.5)
    ax3.fill_between(monthly_load.index, monthly_load.values, alpha=0.3, color=COLORS["load"])
    ax3.set_title("Monthly Avg Load", fontsize=10)
    ax3.set_ylabel("GW", fontsize=8)
    ax3.tick_params(labelsize=7)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # --- Row 3: Correlation matrix + negative price hours + rolling volatility ---
    ax4 = fig.add_subplot(3, 3, 7)
    corr_matrix = df.corr()
    im = ax4.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1)
    short_labels = ["DA Price", "Wind", "Solar", "Load", "Gas"]
    ax4.set_xticks(range(5))
    ax4.set_xticklabels(short_labels, rotation=45, fontsize=7)
    ax4.set_yticks(range(5))
    ax4.set_yticklabels(short_labels, fontsize=7)
    for i in range(5):
        for j in range(5):
            ax4.text(j, i, f"{corr_matrix.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax4.set_title("Correlation Matrix", fontsize=10)
    fig.colorbar(im, ax=ax4, shrink=0.7)

    ax5 = fig.add_subplot(3, 3, 8)
    neg_prices = df["da_price_eur_mwh"] < 0
    monthly_neg = neg_prices.resample("ME").sum()
    ax5.bar(monthly_neg.index, monthly_neg.values, width=25, color=COLORS["highlight"], alpha=0.7)
    ax5.set_title("Negative Price Hours / Month", fontsize=10)
    ax5.set_ylabel("Hours", fontsize=8)
    ax5.tick_params(labelsize=7)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax6 = fig.add_subplot(3, 3, 9)
    rolling_vol = df["da_price_eur_mwh"].rolling(720).std()  # 30-day rolling std
    ax6.plot(rolling_vol.index, rolling_vol.values, color=COLORS["da_price"], linewidth=0.8)
    ax6.set_title("30-Day Rolling Price Volatility", fontsize=10)
    ax6.set_ylabel("Std Dev (EUR/MWh)", fontsize=8)
    ax6.tick_params(labelsize=7)
    ax6.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(PLOTS_DIR / "10_summary_dashboard.png")
    plt.close(fig)
    print("  [10/10] Summary dashboard saved")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def generate_all_visualizations(df: pd.DataFrame):
    """Generate all 10 chart sets from the ingested dataset."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(" Generating Visual Dashboard (10 charts)")
    print("=" * 60)

    plot_full_timeseries(df)
    plot_price_distribution_by_year(df)
    plot_daily_profiles(df)
    plot_monthly_heatmap(df)
    plot_renewables_vs_price(df)
    plot_gas_vs_da_price(df)
    plot_missing_data(df)
    plot_renewable_penetration(df)
    plot_weekly_patterns(df)
    plot_summary_dashboard(df)

    print("\n" + "=" * 60)
    print(f" All charts saved to: {PLOTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    df = run_ingestion_pipeline()

    print("\n--- Dataset Preview ---")
    print(df.head())
    print("\n--- Data Types ---")
    print(df.dtypes)
    print("\n--- Summary Statistics ---")
    print(df.describe())

    generate_all_visualizations(df)
