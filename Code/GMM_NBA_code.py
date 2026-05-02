import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nba_api.stats.endpoints import (
    leaguedashplayerstats,
    leaguehustlestatsplayer,
)

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

SEASON = "2024-25"
SEASON_TYPE = "Regular Season"


def fetch_base_stats():
    """Per-game traditional box score stats."""
    resp = leaguedashplayerstats.LeagueDashPlayerStats(
        season=SEASON,
        season_type_all_star=SEASON_TYPE,
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Base",
    )
    df = resp.get_data_frames()[0]
    return df


def fetch_advanced_stats():
    """Per-game advanced stats (TS%, USG%, ratings if available)."""
    resp = leaguedashplayerstats.LeagueDashPlayerStats(
        season=SEASON,
        season_type_all_star=SEASON_TYPE,
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Advanced",
    )
    df = resp.get_data_frames()[0]
    return df


def fetch_hustle_stats():
    """Hustle stats per game for the 2023-24 regular season."""
    resp = leaguehustlestatsplayer.LeagueHustleStatsPlayer(
        per_mode_time="PerGame",
        season=SEASON,
        season_type_all_star=SEASON_TYPE,
        league_id_nullable="00",
    )
    df = resp.get_data_frames()[0]
    return df


# -----------------------------
# Fetch data from NBA API
# -----------------------------
base = fetch_base_stats()
print("Base stats shape:", base.shape)

time.sleep(1.5)

adv = fetch_advanced_stats()
print("Advanced stats shape:", adv.shape)

time.sleep(1.5)

hustle = fetch_hustle_stats()
print("Hustle stats shape:", hustle.shape)

# -----------------------------
# Merge into a single DataFrame
# -----------------------------
adv_cols = ["PLAYER_ID"]
for cand in ["TS_PCT", "USG_PCT", "OFF_RATING", "DEF_RATING", "NET_RATING"]:
    if cand in adv.columns:
        adv_cols.append(cand)

adv_small = adv[adv_cols].copy()

hustle_cols = [
    "PLAYER_ID",
    "DEFLECTIONS",
    "CONTESTED_SHOTS",
    "CHARGES_DRAWN",
    "SCREEN_ASSISTS",
    "SCREEN_AST_PTS",
    "LOOSE_BALLS_RECOVERED",
    "BOX_OUTS",
]
hustle_small = hustle[[c for c in hustle_cols if c in hustle.columns]].copy()

df = (
    base.merge(adv_small, on="PLAYER_ID", how="left")
        .merge(hustle_small, on="PLAYER_ID", how="left")
)

print("Merged shape:", df.shape)

# -----------------------------
# Cleaning & feature engineering
# -----------------------------
# Filter to players with enough games / minutes
df = df[df["GP"] >= 30]
df = df[df["MIN"] >= 15]
df = df.reset_index(drop=True)

df["THREE_RATE"] = df["FG3A"] / df["FGA"].replace({0: np.nan})
df["AST_TOV"] = df["AST"] / df["TOV"].replace({0: np.nan})

df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows missing core stats (adjust as needed)
df = df.dropna(subset=["PTS", "AST", "REB", "THREE_RATE"])

# -----------------------------
# Select feature columns for clustering
# -----------------------------
feature_cols = []

# Traditional scoring / volume
for c in ["PTS", "FGA", "FG3A", "FG_PCT", "FG3_PCT"]:
    if c in df.columns:
        feature_cols.append(c)

# Playmaking
for c in ["AST", "TOV"]:
    if c in df.columns:
        feature_cols.append(c)

# Rebounding
for c in ["REB", "OREB", "DREB"]:
    if c in df.columns:
        feature_cols.append(c)

# Defense
for c in ["STL", "BLK"]:
    if c in df.columns:
        feature_cols.append(c)

# Advanced
for c in ["TS_PCT", "USG_PCT", "OFF_RATING", "DEF_RATING", "NET_RATING"]:
    if c in df.columns:
        feature_cols.append(c)

# Ratios
for c in ["THREE_RATE", "AST_TOV"]:
    if c in df.columns:
        feature_cols.append(c)

# Hustle
for c in [
    "DEFLECTIONS",
    "CONTESTED_SHOTS",
    "CHARGES_DRAWN",
    "SCREEN_ASSISTS",
    "SCREEN_AST_PTS",
    "LOOSE_BALLS_RECOVERED",
    "BOX_OUTS",
]:
    if c in df.columns:
        feature_cols.append(c)

feature_cols = sorted(set(feature_cols))
print("Using", len(feature_cols), "features:", feature_cols)

X = df[feature_cols].copy()

# Keep only rows with complete feature data for the model
X = X.dropna()
df = df.loc[X.index].copy()

# -----------------------------
# Standardize features
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Choose k using GMM (BIC/AIC + silhouette)
# -----------------------------
ks = range(3, 11)
bics, aics, silhouettes = [], [], []

cov_type = "diag"  # try: "full", "diag", "tied", "spherical"

for k in ks:
    print(f"Fitting k={k}...", flush=True)
    gmm = GaussianMixture(
        n_components=k,
        covariance_type=cov_type,
        random_state=42,
        n_init=3,
        reg_covar=1e-6,
    )
    gmm.fit(X_scaled)
    print(f"  done", flush=True)

    labels = gmm.predict(X_scaled)  # hard assignment for silhouette
    bics.append(gmm.bic(X_scaled))
    aics.append(gmm.aic(X_scaled))
    silhouettes.append(silhouette_score(X_scaled, labels))

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(list(ks), bics, marker="o", label="BIC")
plt.plot(list(ks), aics, marker="o", label="AIC")
plt.xlabel("k")
plt.ylabel("Score (lower is better)")
plt.title(f"GMM Model Selection ({cov_type} covariance)")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(list(ks), silhouettes, marker="o")
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.title("Silhouette vs k")

plt.tight_layout()
plt.savefig("k_selection_gmm.png", dpi=300)
plt.close()

print("k values:", list(ks))
print("BIC:", bics)
print("AIC:", aics)
print("Silhouettes:", silhouettes)

# Pick k by minimum BIC (common choice)
k_opt = list(ks)[int(np.argmin(bics))]
print("Chosen k (min BIC):", k_opt)

# -----------------------------
# Fit final GMM with chosen k
# -----------------------------
gmm = GaussianMixture(
    n_components=k_opt,
    covariance_type=cov_type,
    random_state=42,
    n_init=20,
    reg_covar=1e-6,
)
gmm.fit(X_scaled)

df["cluster"] = gmm.predict(X_scaled)

# Soft membership (probabilities) = great for “hybrid” archetypes
proba = gmm.predict_proba(X_scaled)
df["cluster_conf"] = proba.max(axis=1)

for j in range(k_opt):
    df[f"p_cluster_{j}"] = proba[:, j]

CLUSTER_NAMES = {
    0: "Developmental Wings",
    1: "Two-Way Forwards",
    2: "Backup Facilitators",
    3: "Hustle Bigs",
    4: "Primary Ball Handlers",
    5: "3-and-D Wings",
    6: "Rim Runners",
    7: "Secondary Scorers",
    8: "Dominant Bigs",
}

# Add a human-readable label column right after cluster assignment
df["cluster_name"] = df["cluster"].map(CLUSTER_NAMES)

# -----------------------------
# PCA for visualization
# -----------------------------
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

df["pc1"] = X_pca[:, 0]
df["pc2"] = X_pca[:, 1]

plt.figure(figsize=(8, 6))
for c in range(k_opt):
    mask = df["cluster"] == c
    # Make uncertain points more transparent
    alpha_vals = 0.2 + 0.8 * df.loc[mask, "cluster_conf"].values
    plt.scatter(
        df.loc[mask, "pc1"],
        df.loc[mask, "pc2"],
        alpha=alpha_vals,
        label=CLUSTER_NAMES[c],
    )

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("NBA Player Play-Style Clusters (GMM) - PCA 2D")
plt.legend()
plt.tight_layout()
plt.savefig("nba_clusters_gmm_pca.png", dpi=300)
plt.close()

# -----------------------------
# Cluster means (for interpretation)
# -----------------------------
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 2000)

cluster_means = df.groupby("cluster")[feature_cols].mean()
print(cluster_means.round(2))

# Save (optional, just uncomment):
# cluster_means.round(2).to_csv("cluster_means_gmm.csv", index=True)

# Example: show most confident players in each cluster
examples = (
    df.sort_values(["cluster", "cluster_conf"], ascending=[True, False])
    [["PLAYER_NAME", "TEAM_ABBREVIATION", "MIN", "PTS",
        "AST", "REB", "cluster", "cluster_conf"]]
)

for c in sorted(df["cluster"].unique()):
    print(f"\n=== Cluster {c} ({CLUSTER_NAMES[c]}) (most confident) ===")
    print(examples[examples["cluster"] == c].head(10).to_string(index=False))


# -----------------------------
# EXPERIMENTAL ADDITIONS
# -----------------------------

# -----------------------------
# Save full clustered player data
# -----------------------------
df.to_csv("players_clustered_gmm.csv", index=False)
print("\nSaved players_clustered_gmm.csv")

# -----------------------------
# Team cluster summary
# -----------------------------
team_cluster_counts = (
    df.groupby(["TEAM_ABBREVIATION", "cluster"])
      .size()
      .unstack(fill_value=0)
)

team_cluster_pct = team_cluster_counts.div(
    team_cluster_counts.sum(axis=1), axis=0)

# -----------------------------
# League average cluster profile
# -----------------------------
league_cluster_avg = team_cluster_pct.mean()

print("\nLeague average cluster distribution:")
print("This shows what percentage of a typical roster "
      "belongs to each archetype.")
print("e.g. 0.13 means ~13 percent of an average team's "
      "players fall into that cluster.\n")
print(league_cluster_avg.round(3))

# -----------------------------
# Team needs (cluster deficits)
# -----------------------------
team_needs = league_cluster_avg - team_cluster_pct

print("\nTeam needs (positive = needs more of that cluster, "
      "negative = already has too many):")
print("Values near 0 mean the team is close to the league "
      "average for that archetype.")
print("Values are above +0.10 are meaningful deficits. "
      "Values below -0.10 are memeaningful surpluses.\n")
print(team_needs.round(3))

team_needs.to_csv("team_needs.csv")
league_cluster_avg.to_csv("league_cluster_average.csv", header=["avg_pct"])
team_cluster_counts.to_csv("team_cluster_counts.csv")
team_cluster_pct.to_csv("team_cluster_percentages.csv")

print("\nTeam cluster counts:")
print(team_cluster_counts)

print("\nTeam cluster percentages:")
print(team_cluster_pct.round(2))

# -----------------------------
# Player/Team Fit Scoring
# -----------------------------


def rank_players_for_team(team_name, top_n=10):
    if team_name not in team_needs.index:
        print(f"Team {team_name} not found.")
        return

    needs = team_needs.loc[team_name].values

    prob_cols = [col for col in df.columns if col.startswith("p_cluster_")]

    player_probs = df[prob_cols].values

    # Add player quality component
    impact = df["PTS"] + 0.7 * df["AST"] + 0.7 * df["REB"]
    scores = player_probs.dot(needs) * (df["PTS"] + df["AST"] + df["REB"])

    df_copy = df.copy()
    df_copy["fit_score"] = scores

    # filter out players on same team
    df_copy = df_copy[df_copy["TEAM_ABBREVIATION"] != team_name]

    # Sort by best fit
    ranked = df_copy.sort_values("fit_score", ascending=False)

    print(f"\nTop {top_n} best fits for {team_name}:")
    print(
        "Fit score = how well a player fills this team's archetype gaps, weighted"
        " their overall production. Higher is better. A score of ~5 is excellent;"
        " ~2-3 is a solid role fit.\n"
    )
    print(
        ranked[
            ["PLAYER_NAME", "TEAM_ABBREVIATION", "PTS",
                "AST", "REB", "cluster_name", "fit_score"]
        ].head(top_n).to_string(index=False)
    )

    return ranked.head(top_n)


def summarize_team(team_name):
    if team_name not in team_needs.index:
        print(f"Team {team_name} not found.")
        return

    needs = team_needs.loc[team_name]
    counts = team_cluster_counts.loc[team_name]

    print(f"\n=== {team_name} Roster Breakdown ===")
    print(f"{'Archetype':<25} {'Players':>8} {'vs League Average':>14}")
    print("-" * 52)

    for cluster_id, name in CLUSTER_NAMES.items():
        count = counts.get(cluster_id, 0)
        need = needs.get(cluster_id, 0)

        if need > 0.10:
            note = f"   <- needs more (+{need:.2f})"
        elif need < -0.10:
            note = f"   <- surplus  ({need:.2f})"
        else:
            note = "      (balanced)"
        print(f"{name:<25} {int(count):>8} {note}")


# -----------------------------
# Run example team fits and summaries
# -----------------------------
rank_players_for_team("OKC")
rank_players_for_team("LAL")
rank_players_for_team("BKN")
rank_players_for_team("PHX")
summarize_team("OKC")
summarize_team("LAL")
summarize_team("BKN")
summarize_team("PHX")
