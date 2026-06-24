---
title: >-
  [Paper Note] TabArena: A Living Benchmark for Machine Learning on Tabular Data
description: >-
  [NeurIPS 2025 Spotlight][Self-Supervised Learning][tabular data benchmark] This paper introduces TabArena, the first continuously maintained "living" benchmark for tabular machine learning. From 1,053 candidate datasets, 51 are curated and 16 models are evaluated through large-scale experiments (~25 million model training runs). Key findings: under post-hoc ensembling, deep learning models match or surpass GBDTs; tabular foundation models excel on small datasets…
tags:
  - "NeurIPS 2025 Spotlight"
  - "Self-Supervised Learning"
  - "tabular data benchmark"
  - "living benchmark"
  - "gradient boosted trees"
  - "deep learning"
  - "tabular foundation models"
date: 2026-05-08
content_hash: 0bd9cbfad6ae57dc
---

# TabArena: A Living Benchmark for Machine Learning on Tabular Data

**Conference**: NeurIPS 2025 Spotlight  
**arXiv**: [2506.16791](https://arxiv.org/abs/2506.16791)  
**Code**: [Available (tabarena.ai)](https://tabarena.ai)  
**Area**: Tabular Data / Benchmarking / AutoML
**Keywords**: tabular data benchmark, living benchmark, gradient boosted trees, deep learning, tabular foundation models

## TL;DR
This paper introduces TabArena, the first continuously maintained "living" benchmark for tabular machine learning. From 1,053 candidate datasets, 51 are curated and 16 models are evaluated through large-scale experiments (~25 million model training runs). Key findings: under post-hoc ensembling, deep learning models match or surpass GBDTs; tabular foundation models excel on small datasets; and cross-model ensembles further advance the state of the art.

## Background & Motivation
**Background**: The number of tabular ML benchmarks continues to grow, yet most existing benchmarks are static—once published, they are not updated even when flaws are discovered, models improve, or new methods emerge.

**Limitations of Prior Work**:
   - Inconsistent dataset quality: many benchmark datasets are outdated, contain data leakage, do not represent genuine tabular tasks, or have licensing issues.
   - Inconsistent evaluation protocols: different benchmarks use different validation strategies (holdout vs. cross-validation), hyperparameter search budgets, and ensembling strategies, making conclusions incomparable.
   - Absence of post-hoc ensemble evaluation: most benchmarks do not assess peak model performance under ensembling, thereby underestimating the true capability of individual models.
   - Subsequent benchmarks replicate the flaws of predecessors and fail to compare against genuine SOTA methods.

**Key Challenge**: The community urgently needs reliable benchmarks to address key questions such as deep learning vs. GBDTs, yet static benchmarks cannot provide continuously trustworthy answers.

**Goal**: Establish the first continuously maintained, versioned, community-driven "living benchmark" system to make tabular ML evaluation reliable and sustainable.

**Key Insight**: Develop rigorous protocols across three dimensions—dataset curation, model implementation, and evaluation design—and assemble a cross-institutional maintenance team.

**Core Idea**: Apply software engineering principles to benchmarking—versioning, continuous maintenance, and community contribution—rather than a publish-and-abandon approach.

## Method

### Overall Architecture
TabArena is a living benchmark system built around three core protocols:
1. **Model & Hyperparameter Optimization Protocol**: standardizes model implementation, search spaces, and ensembling strategies.
2. **Dataset Protocol**: rigorous manual curation criteria, selecting 51 datasets from 1,053 candidates.
3. **Evaluation Design Protocol**: unified cross-validation and repetition strategies, with an Elo-based leaderboard.

### Key Designs
1. **Manual Dataset Curation (51/1,053)**:

    - Ten filtering criteria: uniqueness, IID nature, genuine tabular domain, real distribution (non-synthetic), genuine prediction task, size constraints (500–250K samples), no irreversible preprocessing or data leakage, compliant license, publicly downloadable, and no ethical concerns.
    - Only deduplication and size filtering can be automated; all remaining criteria require manual per-dataset review.
    - Review notes for each dataset are published openly, and community challenges and contributions are invited.

2. **Standardized Model Implementation (16 models)**:

    - All models are implemented within AutoGluon's `AbstractModel` framework (scikit-learn-compatible API).
    - Includes 5 tree-based models (RF, ExtraTrees, XGBoost, LightGBM, CatBoost), 6 neural networks (FastaiMLP, TorchMLP, RealMLP, TabM, ModernNCA, EBM), 3 foundation models (TabPFNv2, TabICL, TabDPT), and 2 baselines (Linear, KNN).
    - Search spaces are confirmed through direct communication with original authors; each model is evaluated with 1 default configuration plus 200 random hyperparameter configurations.

3. **Cross-Validation and Post-Hoc Ensembling**:

    - Default protocol: 8-fold inner cross-validation with cross-validation ensembling.
    - Weighted Post-hoc Ensembling: weighted ensemble over models produced by different hyperparameter configurations.
    - Foundation models do not use cross-validation ensembling; instead, they are refit on the combined training and validation set.

4. **Elo Rating System**:

    - Pairwise Elo rating (analogous to ChatBot Arena), calibrated to 1,000 Elo for a default RandomForest.
    - An Elo difference of 400 corresponds to approximately a 91% win rate; each dataset contributes equally.
    - 95% confidence intervals are obtained via 200 bootstrap rounds.
    - Classification uses ROC AUC / log-loss; regression uses RMSE.

5. **Repetition Strategy**: datasets with $\leq$2,500 samples use 10 repetitions of 3-fold cross-validation; all other datasets use 3 repetitions.

### Loss & Training
- Each hyperparameter configuration is time-limited to 1 hour.
- CPU: AWS M6i.2xlarge (8-core Intel Xeon); GPU: NVIDIA L40S 48 GB VRAM.
- Total compute: approximately 15 wall-clock years, ~25 million model training runs.

## Key Experimental Results

### Main Results (TabArena-v0.1 Leaderboard, Post-Hoc Ensembling)

| Rank | Model | Type | Elo (Ensembled) |
|---|---|---|---|
| 1 | TabM | Neural Network | Highest |
| 2 | LightGBM | Tree-based | 2nd |
| 3 | RealMLP | Neural Network | 3rd |
| 4 | CatBoost | Tree-based | 4th (1st without ensembling) |
| 5 | XGBoost | Tree-based | 5th |
| Ref. | AutoGluon (4h) | System | ~2nd tier |

- CatBoost ranks first under standard tuning (without ensembling), but is surpassed by TabM, LightGBM, and RealMLP after post-hoc ensembling.
- Among foundation models, TabPFNv2 substantially outperforms others on compatible datasets ($\leq$10K samples), even surpassing AutoGluon.

### Ablation Study

| Evaluation Dimension | Key Finding |
|---|---|
| Holdout vs. cross-validation | Holdout validation severely underestimates the performance of all models and favors models that already incorporate ensembling internally. |
| Effect of post-hoc ensembling | The top-3 models (TabM, LightGBM, RealMLP) all underperform CatBoost when ensembling is not applied. |
| Cross-model ensembling | An ensemble pipeline using all models outperforms every individual model and AutoGluon. |
| Ensemble weight distribution | Models ranked highest on the leaderboard do not necessarily receive the largest ensemble weights (validation set overfitting effect). |
| Inference efficiency Pareto frontier | EBM and CatBoost achieve the fastest inference; RealMLP requires ~100× more inference time to attain higher performance. |
| Foundation models on small data | TabPFNv2 performs strongly on datasets with $\leq$10K samples even without hyperparameter tuning. |

### Key Findings
- **The GBDT vs. deep learning dichotomy is a false framing**: the two model families are complementary within ensembles, and cross-model ensembles significantly outperform any single model family.
- **Post-hoc ensembling is key to unlocking the potential of deep learning**: without ensembling, DL models generally underperform GBDTs.
- **Foundation models suit small-data regimes**: the in-context learning of TabPFNv2 excels in low-data settings.
- **Validation strategy is critical**: holdout validation systematically distorts model rankings.
- **High-quality datasets suitable for benchmarking are surprisingly scarce**: only 51 of 1,053 candidates pass all filtering criteria.

## Highlights & Insights
- **Living benchmark paradigm**: treating a benchmark as "software" rather than a "paper"—introducing version control, maintenance protocols, and community contribution workflows—represents a paradigm shift in benchmark research.
- **Fair evaluation of peak performance**: post-hoc ensembling allows different models to demonstrate their best performance under equal conditions, free from interference by training strategy differences.
- **High practical utility**: all models are implemented within the AutoGluon framework and are directly usable in real applications; precomputed results are publicly shared, enabling new models to be compared at low cost.
- **Transparency in dataset curation**: publishing review notes for every dataset sets a rare standard of transparency in benchmark research.
- **Elo rating system**: borrowing Elo scoring from LLM leaderboards avoids the sensitivity of traditional average-rank metrics to extreme datasets.

## Limitations & Future Work
- The current scope is limited to IID, small-to-medium-scale (500–250K samples) classification and regression tasks; time-series, distribution shift, clustering, and anomaly detection scenarios are not covered.
- A fixed budget of 200 random hyperparameter configurations limits the study of more advanced HPO strategies such as Bayesian optimization.
- The per-configuration 1-hour time limit is hardware-dependent, affecting comparability of results across users in edge cases.
- Strict dataset filtering yields only 51 datasets, limiting statistical power.
- The impact of feature engineering is not considered, although it may alter model rankings.
- A public benchmark is susceptible to leaderboard gaming (dataset overfitting and potential data contamination of foundation models).

## Related Work & Insights
- **Analogous to ChatBot Arena / LiveBench for LLMs**: borrows the ideas of living leaderboards and Elo scoring, but involves fundamentally different model and evaluation designs for tabular data.
- **vs. prior tabular benchmarks (OpenML-CC18, TabZilla, AutoML Benchmark, etc.)**: TabArena is the first benchmark to integrate post-hoc ensemble evaluation, rigorous manual dataset curation, and a continuous maintenance protocol.
- **AutoGluon as a reference pipeline**: represents the performance level readily achievable by practitioners and provides a realistic baseline for model evaluation.
- **Precomputed results inspired by TabRepo**: TabArena extends result sharing to finer granularity (including predictions and metadata), enabling subsequent research to conduct comparisons at zero additional cost.

## Rating
- ⭐⭐⭐⭐⭐ (5/5)
- Rationale: This work is not merely a benchmark paper but a paradigmatic innovation in tabular ML benchmarking. Every aspect, from dataset curation to evaluation design, is executed with exceptional rigor. The experimental scale is unprecedented (~25 million training runs), the conclusions reshape the community's understanding of GBDT vs. deep learning, and the living maintenance philosophy carries lasting influence.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] TabSTAR: A Tabular Foundation Model for Tabular Data with Text Fields](tabstar_a_tabular_foundation_model_for_tabular_data_with_text_fields.md)
- [\[ICML 2025\] Towards Benchmarking Foundation Models for Tabular Data With Text](../../ICML2025/self_supervised/towards_benchmarking_foundation_models_for_tabular_data_with_text.md)
- [\[NeurIPS 2025\] Hybrid Autoencoders for Tabular Data: Leveraging Model-Based Augmentation in Low-Label Settings](hybrid_autoencoders_for_tabular_data_leveraging_model-based_augmentation_in_low-.md)
- [\[ICML 2026\] Towards One-for-All Anomaly Detection for Tabular Data](../../ICML2026/self_supervised/towards_one-for-all_anomaly_detection_for_tabular_data.md)
- [\[NeurIPS 2025\] Mitra: Mixed Synthetic Priors for Enhancing Tabular Foundation Models](mitra_mixed_synthetic_priors_for_enhancing_tabular_foundation_models.md)

</div>

<!-- RELATED:END -->
