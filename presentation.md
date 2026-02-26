---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    background: #ffffff;
    color: #1a1a2e;
  }
  section.title {
    background: #1a1a2e;
    color: #ffffff;
    text-align: center;
    justify-content: center;
  }
  section.title h1 {
    font-size: 2em;
    color: #ffffff;
    border-bottom: 3px solid #e94560;
    padding-bottom: 12px;
  }
  section.title p {
    color: #cccccc;
    font-size: 0.95em;
  }
  h1 {
    font-size: 1.5em;
    color: #1a1a2e;
    border-bottom: 3px solid #e94560;
    padding-bottom: 8px;
  }
  h2 { font-size: 1.1em; color: #e94560; margin-bottom: 4px; }
  table {
    font-size: 0.8em;
    width: 100%;
    border-collapse: collapse;
  }
  th { background: #1a1a2e; color: white; padding: 6px 10px; }
  td { padding: 5px 10px; border-bottom: 1px solid #ddd; }
  tr:nth-child(even) td { background: #f5f5f5; }
  .highlight { color: #e94560; font-weight: bold; }
  .columns { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; align-items: start; }
  .columns3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; align-items: start; }
  .card {
    background: #f8f9fa;
    border-left: 4px solid #e94560;
    padding: 10px 14px;
    border-radius: 4px;
    font-size: 0.85em;
  }
  .metric { font-size: 2em; font-weight: bold; color: #e94560; }
  .metric-label { font-size: 0.8em; color: #666; }
  ul { margin-top: 6px; }
  li { margin-bottom: 4px; }
---

<!-- _class: title -->

# Predicting Football Player Market Value
## Position-Specific Machine Learning Pipeline

<br>

**DAMA Hackathon 2026**

MSc Data Science & Machine Learning

---

# The Problem

<div class="columns">
<div>

## Why is this hard?

- Transfer fees exceed **€200M** for top players
- Valuations are **subjective** — clubs, agents, media shape prices
- A striker's value depends on **different signals** than a goalkeeper's

## Our Goal

> Predict `market_value_in_eur` from **publicly available** performance statistics — **separately per position group**

</div>
<div>

![w:480](figures/02_target_distribution.png)

*Market value distribution — heavily right-skewed; log-transform applied*

</div>
</div>

---

# Data & Pipeline

<div class="columns">
<div>

## 7 Transfermarkt CSVs

| Source | Size |
|---|---|
| Players | 34,291 |
| Appearances | 1.7M rows |
| Valuations | 449K rows |
| Clubs / Transfers | ~85K |

**After cleaning:** 30,718 labelled players

**4 position groups:**
GK · DEF · MID · ATT

</div>
<div>

## 10-step pipeline

```
Raw CSVs
  ↓ 01 Sanity checks
  ↓ 02 EDA
  ↓ 03 Clean & merge
  ↓ 04 Feature engineering
  ↓ 05 Preprocessing
  ↓ 06 Baseline models
  ↓ 07 Advanced models
  ↓ 08 SHAP & evaluation
  ↓ 09 Undervalued players
  ↓ 10 Final report
```

</div>
</div>

---

# Feature Engineering

<div class="columns">
<div>

## Shared base features (all positions)

- `age` + `age²` — non-linear career arc
- `log_minutes`, `log_appearances` — experience
- Rate stats: goals, assists, contributions per 90
- `log_highest_mv` — career peak reputation
- `league_mean_value` — target-encoded league prestige
- `transfer_count`, `log_total_fees`

## Why separate models?

![w:420](figures/02_value_by_position.png)

</div>
<div>

## Position-specific additions

<div class="card">
<b>GK</b> · save percentage proxy, clean sheet rate, offensive contribution
</div>
<br>
<div class="card">
<b>DEF</b> · defensive solidity, discipline score, defensive attack rate
</div>
<br>
<div class="card">
<b>MID</b> · creative output per 90, attack-defense ratio
</div>
<br>
<div class="card">
<b>ATT</b> · goals per appearance, minutes per goal, assist-to-goal ratio
</div>

</div>
</div>

---

# Models & Results

<div class="columns">
<div>

## What we trained

1. **Ridge** (linear baseline, RidgeCV)
2. **Random Forest** (RandomizedSearchCV)
3. **XGBoost** (hist method, tuned)
4. **MLP** (Adam, early stopping, 3 layers)

**No data leakage:** split → encode → scale

## Test-set R² by position

| Position | Ridge | RF | MLP | **XGB** |
|---|---|---|---|---|
| GK | 0.712 | 0.781 | 0.775 | **0.788** |
| DEF | 0.764 | 0.826 | 0.815 | **0.827** |
| MID | 0.771 | 0.825 | 0.812 | **0.829** |
| ATT | 0.750 | 0.802 | 0.800 | **0.809** |

</div>
<div>

![w:480](figures/10_r2_improvement.png)

*XGBoost gains +5–8 R² points over Ridge baselines*

</div>
</div>

---

# SHAP: What Drives Value?

<div class="columns">
<div>

## Cross-position findings

- **`league_mean_value`** — top-3 in every position: *where* you play matters as much as *how*
- **`log_highest_mv`** — market anchors to career reputation
- **`age²`** — value peaks mid-20s, accelerates decline post-30

## Position-specific signals

- **ATT**: `goals_per_90` becomes the strongest driver
- **GK**: performance stats matter less; reputation & league dominate
- **MID**: `creative_per_90` and `contributions_per_90` provide lift
- **DEF**: `defensive_solidity` and `transfer_count` add beyond base

</div>
<div>

![w:480](figures/10_shap_beeswarm_grid.png)

</div>
</div>

---

# Application: Dream Team

<div class="columns">
<div>

## Scoring all 33,500+ players

Using serialised preprocessors — **no leakage**

**Undervaluation ratio** = predicted ÷ actual

Minimum 10 appearances filter applied

## 4-3-3 squad — cost vs. predicted value

| Metric | Value |
|---|---|
| Total actual cost | **€0.50M** |
| Total predicted value | **€13.4M** |
| Return ratio | **26.7×** |

Top picks: Björn Engels (DEF, 31, BE1, 33.7×) and Joris Gnagnon (DEF, 28, ES1, 25.5×)

</div>
<div>

![w:500](figures/09_dream_team_pitch.png)

</div>
</div>

---

# Conclusions

<div class="columns3">
<div class="card">

## What we achieved

- R² **0.79–0.83** across all positions
- XGBoost outperforms RF, MLP, and linear baselines
- SHAP provides actionable interpretation
- Full reproducible pipeline (10 notebooks)

</div>
<div class="card">

## Key insights

- League prestige = strongest cross-position signal
- Career peak value anchors market estimates
- Position-specific features add meaningful lift
- High-value players systematically underestimated (compression effect)

</div>
<div class="card">

## Future work

- Add injury history & international caps
- Temporal modelling (LSTM / Transformer) for career trajectory
- Multimodal: combine stats with video embeddings
- Real-time API for transfer window scouting

</div>
</div>

<br>

> **Dataset:** Transfermarkt open data · **Models:** XGBoost, RF, MLP · **Interpretation:** SHAP · **Code:** GitHub

---

<!-- _class: title -->

# Thank You

<br>

**Repository:** `football-player-value-predictor`

**Pipeline:** 10 notebooks · 7 raw CSVs → dream team

**Best model:** XGBoost · R² up to **0.829** (MID)

<br>

*Questions?*
