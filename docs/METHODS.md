# Research methods

[Home](../README.md) · [User guide](USER_GUIDE.md) · [Benchmark](BENCHMARK.md)

AlphaResearchOS runs bounded, model-driven experiments over daily OHLCV panels. A candidate is a complete strategy: a hypothesis, feature expressions, a model family, parameters, rationale, and parent trajectory references.

## Research loop

1. **Propose.** Supply the direction, executable constraints, selected parents, recent development trials, retrieved memory, development baselines, and trading settings to a model.
2. **Validate.** Parse the feature DSL, enforce complexity and history bounds, check causality, and compare supported formulas through two numerical evaluators.
3. **Evaluate.** Generate predictions with a fit frozen before each development fold, then measure returns, turnover, coverage, and baseline differences.
4. **Review.** A fresh session using the configured model receives a compact candidate, development baseline metrics, and a technical audit summary. It returns `approve`, `revise`, or `reject`, plus a critique and proposed next experiment.
5. **Evolve.** Select parents through UCB and propose a mutation, crossover, or model change. Approved candidates are eligible parents; reviewed outcomes contribute to stored feedback.
6. **Freeze and evaluate.** Select the highest development score among technically accepted, approved candidates. Fix its features and parameters before evaluating the holdout.

Proposal and review are separately counted requests. Rejected, failed, and timed-out attempts remain in the experiment ledger. Review approval records technical acceptance into the research comparison. The complete measured evidence remains available in the run artifacts.

An optional, default-off [Jev gate](JEV.md) adds one typed decision request after
independent approval. Only allowlisted development evidence enters this request.
Selection then requires both reviews to approve; uncertain or failed Jev checks
cannot override the existing reviewer or numeric constraints. The threshold and
provider configuration are frozen per run, and attempts share its request/time/token
budgets. This is an evidence-review mechanism; its incremental effect on strategy
performance has not been measured. The published ETF benchmark predates Jev.

## Features and predictive models

The feature language includes `open`, `high`, `low`, `close`, and `volume`, arithmetic, positive historical windows, cross-sectional transforms, and a bounded set of functions. Each expression is parsed and evaluated through the project's DSL. The proposal contains 1–6 features.

All model inputs are transformed to daily cross-sectional average percentile ranks. Fitted models then estimate normalization statistics from training samples.

| Model | Prediction method |
| --- | --- |
| `rank` | Equal-weight mean of feature ranks followed by causal smoothing |
| `ridge` | Pooled linear regression with an unpenalized intercept and L2 regularization |
| `hist_gbdt` | Histogram gradient boosting with fixed capacity, seed 42, and configurable L2 regularization |

The predictive parameters are `alpha`, `train_window`, `retrain_every`, `horizon`, and `smoothing`. Rank ensembles use smoothing as their effective model parameter. Strategy identity incorporates normalized expressions, model family, and effective parameters.

### Numerical checks

Feature causality checks compare prefixes and perturb future inputs. Supported expressions also pass through the vendored FactorMiner NumPy core and the local pandas evaluator. The comparison records missing-value agreement, numerical error, and `passed`, `failed`, or `skipped` status. The supported subset and numerical patches are documented in [UPSTREAM.md](UPSTREAM.md).

Prediction checks cover complete-data coverage, cross-sectional variation, and redundancy with previously evaluated candidates. A feature participates only where its required history is available.

## Training timeline

For feature row `i` and target horizon `h`, the predictive target is:

```text
close[i + 1 + h] / close[i + 1] - 1
```

This label becomes available at row `i + 1 + h`. A fit may use only labels already matured at its fitting date. Training starts with sufficient complete dates and pooled samples, expands toward the requested training window, and then rolls over past data on the configured fitting schedule.

Each development evaluation uses a fitting cutoff before its first executable signal. Its scored fold uses the resulting frozen model. The final holdout uses a fit frozen at the end of development. `retrain_every` controls the historical fitting schedule up to that cutoff. The training audit records fitting rows, label endpoints, sample counts, normalization, estimators, and prediction intervals.

## Development and holdout

The default agent protocol uses a 252-session warmup, three chronological development folds, a final 20% holdout, and a two-session separation between development and holdout. Candidate search and reviewer feedback use development evidence. The selected strategy is fixed before the holdout metrics are computed.

Candidate score:

```text
mean(fold Sharpe)
- 0.5 × std(fold Sharpe, ddof=0)
+ 2 × mean(fold total return - equal-weight total return)
- 0.002 × strategy complexity
```

Baselines are evaluated under the same portfolio rules and dates. Their recorded scores use the performance terms before the candidate complexity penalty. The development folds start from cash separately; aggregate descriptive returns chain the resulting fold curves.

The report includes an equal-weight baseline and fixed 20-day and 60-day momentum baselines. Holdout cost stress applies additional fee scenarios to the already selected strategy. The Sharpe bootstrap interval is computed from that saved strategy's return sequence after selection. The full candidate ledger records the development search that preceded selection.

## Portfolio evaluation

The default profile holds the three highest-scoring assets at equal weight, rebalances every five sessions, and applies 10 bps one-way cost to traded value. The equal-weight benchmark holds the full input universe with matching timing, costs, and terminal liquidation.

A signal observed at close `t` is executed at close `t+1` and earns the following close-to-close return. Between rebalances, weights drift with prices. Each evaluation begins with cash and charges final liquidation. Missing eligible scores move the portfolio to cash at a scheduled rebalance. Equal scores are ordered by asset identifier.

Annualized statistics use 252 sessions. Cash return and the risk-free rate used in Sharpe are zero. The execution profile is daily, long-only, and based on a fixed asset universe with proportional costs. Imported data retains the supplied adjustment convention and fixed asset universe.

## Trajectories and research memory

Every attempt receives an ID and action. Its report can include hypothesis, features, model, parameters, development outcomes, review, parents, and usage. Successfully normalized strategy trajectories are also persisted in SQLite; the experiment graph retains failed attempts that did not produce a strategy.

Research memory is separated by dataset, evaluation settings, and implementation version. Retrieval uses the development cutoff and ranks eligible records by word overlap, development score, and feature diversity, keeping one record per strategy. A run initially retrieves up to four trajectories. Their IDs stay fixed during the run, while feedback counters refresh before subsequent proposals. Recent development trials and selected parent records provide the evolving context.

Each trajectory has at most two already-existing parents in the same scope. Immutable nodes and insertion-time parent checks maintain an acyclic graph. Parent edges supply direct-child feedback to a one-step UCB scheduler. The reward maps approved development scores through `tanh` to 0–1; stored unsuccessful strategies consume a visit with zero reward. Two explored distinct strategies can be selected for crossover.

The SQLite payload projects fields onto a typed development-data schema. Date filtering applies to both returned nodes and the child feedback used for visits and rewards. Active run settings and model credentials belong to project configuration rather than trajectory memory.

## Budgets and run control

The run ledger counts attempted calls before dispatch and retains consumed budget through interruption. Proposal and review have separate usage records. Token reservations govern admission to the next call; actual returned usage is displayed alongside request status. Checkpoints and process locks preserve active experiments during pause and resume.

Each run retains its data, configuration, complete strategy, review decisions, development outcomes, and final evaluation. Reports provide readable HTML plus JSON and CSV exports for analysis.

The [sector ETF benchmark](BENCHMARK.md) applies this method to nine US sector ETFs and presents the selected strategy, baseline comparisons, development folds, and cost sensitivity.
