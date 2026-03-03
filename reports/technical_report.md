# Predicting Football Player Market Value with Position-Specific Machine Learning Models

**DAMA Hackathon 2026 — Technical Report**

 - Petridis Dimitrios — AM 170264
 - Yfantidis Ioannis — AM 168057

---

## Abstract

We address the problem of predicting the market value of professional football players from publicly available performance statistics and player profile data. Using the Transfermarkt open dataset (~33,500 players, 1.7M game appearances), we design separate predictive pipelines for four position groups (Goalkeeper, Defender, Midfielder, Attacker), motivated by empirically distinct value distributions across positions. A multi-model comparison spanning linear models, Random Forests, XGBoost, and a Multi-Layer Perceptron (MLP) demonstrates that gradient-boosted trees consistently outperform competitors, achieving R² values of 0.79–0.83 across all positions — a gain of 5–8 percentage points over linear baselines. We interpret model decisions using SHAP and apply the trained models to identify systematically undervalued players, culminating in a cost-optimised 4-3-3 "dream team" whose total market cost is €5.7M against a model-predicted value of €60.2M.

---

## 1. Introduction

Player market valuation sits at the intersection of sports analytics, economics, and machine learning. Accurate valuation tools benefit clubs in transfer negotiations, agents in contract discussions, and scouts seeking undervalued talent. While subjective assessments dominate in practice, data-driven approaches have demonstrated the potential to surface systematic mispricings.

The central challenge is that market value depends on heterogeneous factors — performance statistics, age trajectory, league prestige, transfer history, and positional role — which interact non-linearly. A goalkeeper's value is driven by fundamentally different metrics than a forward's, making a single global model suboptimal.

This work makes the following contributions: (i) an end-to-end reproducible pipeline from raw Transfermarkt data to trained models; (ii) position-specific feature engineering tailored to each role; (iii) a systematic comparison of linear and non-linear models including an MLP; (iv) SHAP-based model interpretation; and (v) a practical application identifying undervalued players and constructing a budget dream team.

---

## 2. Data

### 2.1 Dataset

We use seven CSV files from the Transfermarkt open dataset, covering players, appearances, valuations, club games, lineups, clubs, and transfers. Table 1 summarises the key sources.

**Table 1: Raw data sources**

| File | Records | Role |
|---|---|---|
| `players.csv` | 34,291 | Player profiles, position, club |
| `appearances.csv` | 1.7M | Per-game stats, aggregated to career totals |
| `player_valuations.csv` | 449K | Value history; only latest kept |
| `club_games.csv` | 156K | Clean sheet computation |
| `game_lineups.csv` | 2.7M | Links players to matches |
| `clubs.csv` | ~450 | League code and squad size |
| `transfers.csv` | 85K | Transfer history |

### 2.2 Target Variable

The target is `log_market_value = log1p(market_value_in_eur)`. The raw distribution is heavily right-skewed (median €0.2M, max €200M), making the log transformation essential for regression stability. Players without a recorded valuation are excluded from model training, yielding 30,710 labelled players.

### 2.3 Cleaning and Merging

All seven sources are merged via left joins on `player_id`. Key decisions:
- 49 players dropped for missing date of birth; 189 for `position = 'Missing'`
- Players aged outside 15–45 excluded as stale
- Career stats capped at the 99th percentile to dampen outlier influence
- Height values below 100 cm treated as measurement errors and median-imputed
- Missing foot preference filled as `'unknown'`; missing club statistics filled with medians

The dataset is split into four position groups: GK (3,313), DEF (9,867), MID (8,976), ATT (8,554).

---

## 3. Methodology

### 3.1 Feature Engineering

Fourteen shared base features capture experience and role-agnostic performance (Table 2). Each position group then receives 3–5 additional specialist features encoding role-specific attributes.

**Table 2: Shared base features (all positions)**

| Feature | Description |
|---|---|
| `age`, `age_squared` | Captures non-linear career trajectory |
| `log_minutes`, `log_appearances` | Experience (log-transformed for skew) |
| `has_appearances` | Binary: any recorded game |
| `goals_per_90`, `assists_per_90`, `contributions_per_90` | Rate statistics per 90 min |
| `yellow_per_90` | Disciplinary proxy |
| `clean_sheets_per_90` | Defensive contribution for all positions |
| `log_highest_mv` | Peak historical value (career signal) |
| `log_total_fees`, `transfer_count` | Transfer market history |
| `squad_size` | Club competition proxy |
| `league_mean_value` | Target-encoded league quality |

Position-specific features:
- **GK**: `save_pct_proxy` (goals conceded vs. appearances ratio), `gk_offensive` (goals + assists), `cs_rate`
- **DEF**: `defensive_solidity` (contribution weighted by clean sheets), `def_cs_rate`, `discipline_score`, `def_attack_per_90`
- **MID**: `creative_per_90`, `attack_defense_ratio`, `mid_attack_per_90`
- **ATT**: `goals_per_app`, `assists_per_app`, `minutes_per_goal`, `assist_to_goal_ratio`

### 3.2 Preprocessing

An 80/20 stratified train/test split is applied **before** any encoding or scaling to prevent data leakage. The categorical league feature (`current_club_domestic_competition_id`, 14 unique values) is target-encoded using training-set mean log-values. All numerical features are standardised with `StandardScaler` fit on the training set only. Both the scalers and league encoding maps are serialised so they can be applied identically to unseen players in the scoring phase.

### 3.3 Models

Four model families are trained and tuned per position group:

1. **Linear baselines**: `LinearRegression`, `RidgeCV` (α ∈ {0.01, …, 100}), `LassoCV` (50 α values) — 5-fold cross-validation
2. **Random Forest**: `RandomForestRegressor` with `RandomizedSearchCV` (n\_iter=20, 5-fold) over depth, n\_estimators, max\_features, and leaf size
3. **XGBoost**: `XGBRegressor` (`tree_method='hist'`) tuned over learning rate, depth, subsample, column sample, and L1/L2 regularisation
4. **MLP**: `MLPRegressor` (Adam solver, early stopping with patience=20, validation\_fraction=0.1) — architecture search over layer sizes ([64,32], [128,64,32], [256,128,64]), activation, L2 (α), and learning rate

The best model per position is selected by test-set R².

---

## 4. Results

### 4.1 Baseline vs. Advanced Models

**Table 3: Test-set R² by position and model**

| Position | Ridge (baseline) | Random Forest | MLP | **XGBoost (best)** |
|---|---|---|---|---|
| GK | 0.713 | 0.781 | 0.738 | **0.788** |
| DEF | 0.756 | 0.825 | 0.815 | **0.828** |
| MID | 0.770 | 0.825 | 0.819 | **0.830** |
| ATT | 0.750 | 0.802 | 0.802 | **0.809** |

XGBoost consistently achieves the best generalisation across all four positions, outperforming linear baselines by 5–8 R² points and Random Forests by a narrow margin. The MLP is competitive but shows slightly higher variance and requires longer training time despite early stopping.

The remaining unexplained variance reflects genuine noise inherent in market valuation: club-specific negotiations, agent influence, media attention, and injury history — factors not available in open statistics.

### 4.2 SHAP Interpretation

SHAP TreeExplainer values reveal distinct importance hierarchies per position (Figure 1):

- **GK**: `log_highest_mv` (peak career value) is the dominant predictor, followed by `league_mean_value` and `log_minutes`. The `save_pct_proxy` and `cs_rate` provide modest additional signal, reflecting that raw counting stats are less informative for goalkeepers.
- **DEF**: `log_highest_mv` and `league_mean_value` again lead, with `log_minutes` and `log_appearances` confirming that experience translates to value. `defensive_solidity` and `transfer_count` provide meaningful lift.
- **MID**: `log_highest_mv`, `league_mean_value`, and `contributions_per_90` are top features. `creative_per_90` and `age` show clear directional effects: creativity increases value while age past the peak depresses it.
- **ATT**: `goals_per_90` and `contributions_per_90` become significantly more important relative to other groups, confirming that raw offensive output is the primary market signal for attackers. `log_highest_mv` remains relevant but yields weight to performance metrics.

Cross-cutting insights: `league_mean_value` is consistently in the top-3 features for every position, underscoring that league prestige is a strong market driver independent of individual performance. `age_squared` captures the inverted-U career trajectory: value rises through the mid-20s and accelerates its decline post-30. `log_highest_mv` acts as a career reputation signal — the market anchors partly to what a player has historically commanded.

### 4.3 Error Analysis

Residual analysis reveals a systematic pattern: high-value players (>€10M) are underestimated, and low-value fringe players are slightly overestimated. This compression effect is expected: extreme valuations for elite players incorporate brand, marketing, and scarcity factors beyond statistical performance. Residuals are near-symmetric around zero in log space, confirming no directional bias. Error by age group shows elevated RMSE for players above 35, consistent with retirement uncertainty and reduced appearance counts inflating rate statistics.

---

## 5. Application: Undervalued Players and Dream Team

### 5.1 Scoring All Players

Using serialised preprocessors (scalers + league encodings), we score all 33,524 players — including those without a current valuation record — through each position's XGBoost model. An **undervaluation ratio** is computed as `predicted_market_value / actual_market_value`. Players with fewer than 10 career appearances are excluded to filter out data-sparse profiles.

### 5.2 Dream Team (4-3-3)

A cost-optimised 4-3-3 squad is assembled by selecting the top-undervalued eligible player per position slot. Table 4 presents the result.

**Table 4: Model-selected 4-3-3 dream team**

| Pos | Player | Age | Actual Value (€M) | Predicted (€M) | Ratio | League |
|---|---|---|---|---|---|---|
| GK | Pierluigi Gollini | 30 | 0.80 | 7.91 | 9.9× | IT1 |
| DEF | Björn Engels | 31 | 0.10 | 3.07 | 30.7× | BE1 |
| DEF | Francis Guerrero | 29 | 0.05 | 1.27 | 25.3× | ES1 |
| DEF | Joris Gnagnon | 28 | 0.30 | 7.39 | 24.6× | ES1 |
| DEF | Stephen Kingsley | 31 | 0.07 | 0.79 | 11.3× | SC1 |
| MID | Floriano Vanzo | 31 | 0.01 | 0.29 | 28.9× | BE1 |
| MID | Vincent Koziello | 30 | 0.25 | 2.75 | 11.0× | BE1 |
| MID | Matheus Índio | 29 | 0.03 | 0.19 | 7.7× | PO1 |
| ATT | Nana Ampomah | 29 | 0.10 | 0.95 | 9.5× | BE1 |
| ATT | Maxi Gómez | 29 | 2.00 | 18.15 | 9.1× | ES1 |
| ATT | Conrad Harder | 20 | 2.00 | 17.48 | 8.7× | L1 |
| **Total** | | | **5.71** | **60.24** | **10.5×** | |

**Total squad cost: €5.71M → predicted value: €60.24M (10.5× return).**

The selected squad consists entirely of prime-career players (ages 20–31) from established leagues (Serie A, La Liga, Bundesliga, Belgian Pro League, Scottish Premiership, Primeira Liga). The largest absolute undervaluation gaps are Maxi Gómez (ES1, €16.15M gap) and Conrad Harder (L1, €15.48M gap) — players with strong statistical profiles whose current market prices appear structurally depressed. Björn Engels (BE1, €2.97M gap) and Joris Gnagnon (ES1, €7.09M gap) round out the defensive line with meaningful discrepancies at reasonable ages.

---

## 6. Conclusion

We presented a position-stratified machine learning pipeline for football player market value prediction. Separate XGBoost models per position group achieve R² of 0.79–0.83, representing a substantial gain over linear baselines and confirming that gradient-boosted trees effectively capture the non-linear interactions between performance statistics, career history, and league context.

SHAP analysis reveals that league prestige (`league_mean_value`) and career peak value (`log_highest_mv`) are the strongest cross-positional drivers, while position-specific features (offensive rate for ATT, clean-sheet metrics for GK, creative output for MID) provide meaningful incremental signal. The undervaluation application demonstrates a practical use case: sourcing budget transfer targets by combining model predictions with actual market prices.

**Limitations and future work**: (i) The dataset lacks injury records, international appearances, and social media reach — all known market drivers. (ii) Temporal structure is ignored: sequential valuations could be modelled with LSTM or temporal transformers to capture career trajectories. (iii) Multimodal approaches combining match video embeddings with tabular features represent a natural extension aligned with deep learning capabilities.

