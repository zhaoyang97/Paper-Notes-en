---
title: >-
  [Paper Note] Mix, Don't Pick: Why Synthetic Corpus Composition Matters for Time Series Foundation Model Pretraining
description: >-
  [ICML 2026][Time Series][Time Series Foundation Models] This paper performs a systematic comparative study using 11 synthetic time series generators and 2 time series foundation models trained from scratch. It finds that generator rankings are unstable across different architectures, and the forecasting error gap between the best and worst generators can be as large as 2. Rather than solving the difficult selection problem, simply mixing all generators with equal weights (Mix…
tags:
  - "ICML 2026"
  - "Time Series"
  - "Time Series Foundation Models"
  - "Synthetic Data"
  - "Corpus Composition"
  - "Pretraining"
  - "Zero-shot Forecasting"
date: 2026-05-08
content_hash: 4d95300601e965d7
---

# Mix, Don't Pick: Why Synthetic Corpus Composition Matters for Time Series Foundation Model Pretraining

**Conference**: ICML 2026  
**arXiv**: [2606.09912](https://arxiv.org/abs/2606.09912)  
**Code**: TBD  
**Area**: Time Series / Foundation Model Pretraining  
**Keywords**: Time Series Foundation Models, Synthetic Data, Corpus Composition, Pretraining, Zero-shot Forecasting

## TL;DR
This paper performs a systematic comparative study using 11 synthetic time series generators and 2 time series foundation models trained from scratch. It finds that generator rankings are unstable across different architectures, and the forecasting error gap between the best and worst generators can be as large as 2. Rather than solving the difficult selection problem, simply mixing all generators with equal weights (Mixed11) can match or exceed the best single generator. Combining this with real data yields the strongest corpus. The study concludes that synthetic pretraining is a "corpus composition" problem rather than a "generator selection" problem, and composition strategies must be validated for each specific model architecture.

## Background & Motivation

**Background**: Time Series Foundation Models (TSFMs) increasingly rely on synthetic data for pretraining, either as the primary source or as an augmentation for real corpora. This is motivated by the scarcity and licensing restrictions of real-world time series data. However, decisions regarding "which generators to use and how to combine them" are typically made based on convention or convenience.

**Limitations of Prior Work**: A natural intuition is to "select generators whose output most closely resembles real time series," an assumption implicit in many feature-based validation works (measuring whether synthetic sequences match real statistical structures). However, "resembling real data" and "being useful for pretraining" are two different things. Existing work on synthetic TSFM pretraining has never systematically examined: (1) whether fidelity-based selection translates to downstream performance; (2) whether generator rankings transfer across architectures; and (3) what the optimal ratio between synthetic and real data should be.

**Key Challenge**: These three questions are not isolated but are different facets of the same "corpus composition" problem. Analogous to the extensively studied data mixing problem in LLM pretraining, corpus design decisions directly impact downstream quality. Treating "generator selection" as the core issue is fundamentally asking the wrong question.

**Goal**: Instead of solving the unstable "which generator is best" puzzle, this paper reframes it as a corpus composition problem and systematically quantifies three things: the performance gaps between generators, whether mixing serves as a robust default, and how to balance real and synthetic data.

**Key Insight**: The authors start from a counter-intuitive observation: since the ranking of individual generators is unstable across architectures, one should not bet on a single generator. Mixing diverse generators may robustly capture the structural benefits of each source.

**Core Idea**: Mix, Don't Pick—replace generator selection with equal-weight mixing and replace pure sources with "real + synthetic" compositions. It emphasizes that composition strategies must be validated for each model family rather than assuming transferability.

## Method

### Overall Architecture

This is a systematic empirical study that proposes a methodology for corpus composition rather than a new model architecture. The design involves generating one corpus from each of 11 synthetic generator families (each with 1 million univariate windows of length 1024, totaling 11.2 billion time points). Two architectures, Chronos-T5-Mini and Moirai-Small, are **trained from scratch under matching budgets** on each single-generator corpus, an equal-weight mixture of all 11 (Mixed11), a real reference corpus, and "real + Mixed11" mixtures at 75-25 / 50-50 / 25-75 window ratios. Finally, all models are evaluated **zero-shot** on the 28-dataset GIFT-Eval benchmark.

The evaluation protocol is the foundation of the study's credibility: the real reference corpus is sampled from the GIFT-Eval pretraining pool with strict measures to avoid train-test leakage. Metrics include normalized CRPS (probabilistic error) and normalized MASE (point error), both relative to a seasonal-naive baseline (lower is better; values < 1 outperform the baseline). Since many datasets have multiple horizons, paired bootstrapping is used with "dataset-horizon tasks" as resampling units, yielding 97 task-level observations.

The study revolves around three questions: ① How much does generator selection matter under a fixed budget? ② Can a simple multi-generator mixture serve as a robust alternative? ③ When real data is available, is synthetic data a supplement or a dilution?

### Key Designs

**1. Reframing Synthetic Pretraining as "Corpus Composition" rather than "Generator Selection"**

This is the central thesis of the paper and its primary design motivation. Prior work assumes one should "pick a good generator," which collapses a combinatorial optimization problem into a single-choice task where the optimal solution is neither stable nor transferable. Drawing from LLM pretraining experiences where data mixing has a first-order impact on downstream quality, the authors unify generator selection, cross-family mixing, and real-data integration into a single corpus composition problem. All subsequent experiments are designed to "design a corpus recipe" rather than "find a champion generator."

**2. 11 Generator Families × Dual Architecture Comparison: Exposing Non-transferable Rankings**

To prove that "selecting generators" is unreliable, the authors constructed a strictly controlled comparison across 11 families: linear statistical models (ARIMA, ETS), stochastic processes (fBm, SDE, GARCH-type volatility), deterministic chaotic systems (Lorenz, Mackey-Glass), waveform/structural signals (TimeSynth, StepFunction, Waveform, TSI), and Gaussian Process kernel synthesis (KernelSynth). For each, the corpus scale, window length, and training budget are aligned. Results show that CRPS varies by $1.6\times$ (Moirai) to $2.1\times$ (Chronos) across generators, and rankings do not transfer: ETS is the second-best for Moirai but the worst for Chronos. This proves that single-generator selection is architecture-dependent.

**3. Mixed11 Equal-Weight Mixing: Bypassing Selection via Naive Mixing**

Given the instability of selection, the authors test Mixed11 as a robust alternative. The key finding is that mixing **does not average away the useful structures of strong generators**: for Moirai-Small, Mixed11 (CRPS $0.735$) is essentially tied with the best single generator, KernelSynth ($0.734$), and leads the runner-up ETS. For Chronos-T5-Mini, Mixed11 (CRPS $0.906$) outperforms both KernelSynth ($0.936$) and SDE ($0.981$), achieving the best MASE among synthetic corpora. Paired bootstrapping confirms that on Chronos, the MASE improvement of Mixed11 is statistically significant ($+10.2\%$, CI $[5.5\%, 15.2\%]$ strictly above 0). Equal-weight mixing thus provides a robust "synthetic default recipe."

**4. Real-Synthetic Proportional Mixing + Per-Architecture Validation: Composition, Not Replacement**

The authors mixed the real reference and Mixed11 at ratios of 75-25 / 50-50 / 25-75. Each configuration was trained with 3 independent initializations to reflect model-level variance. The conclusion is that composition outperforms any pure source, but **the optimal ratio depends on the architecture**: Moirai-Small performed best at 75-25 (CRPS $0.685$ / MASE $0.984$), outperforming both pure real ($0.830$) and pure Mixed11 ($0.733$), suggesting synthetic data provides complementary structures. Chronos-T5-Mini performed best at 50-50 (CRPS $0.772$), though its CRPS improvement over pure real ($0.779$) was marginal. Because optimal ratios vary by architecture, the authors argue that corpus composition must be validated per model family.

## Key Experimental Results

Evaluation: 28 datasets from GIFT-Eval, zero-shot, normalized CRPS / MASE (relative to seasonal-naive, lower is better, <1 outperforms baseline), paired bootstrap with 97 task-level observations.

### Single Generator Scan (Table 1, CRPS)

| Generator | Moirai-Small CRPS | Chronos-T5-Mini CRPS |
|-----------|-------------------|----------------------|
| KernelSynth | **0.734** (Best) | **0.936** (Best) |
| ETS | 0.820 (Runner-up) | 1.976 (Worst) |
| SDE | 0.910 | 0.981 (Runner-up) |
| Chaotic | 1.154 | 1.842 |
| TimeSynth | 1.194 | 1.153 |
| Real Reference | 0.814 | 0.791 |

Takeaway: CRPS differs by up to $2.1\times$ between generators; rankings flip between architectures (e.g., ETS).

### Mixture vs. Best Single Generator / Real-Synthetic Composition

| Configuration | Moirai CRPS | Moirai MASE | Chronos CRPS | Chronos MASE |
|---------------|-------------|-------------|--------------|--------------|
| Best Single (KernelSynth) | 0.734 | 1.049 | 0.936 | 1.290 |
| **Mixed11** | 0.735 | 1.069 | **0.906** | **1.171** |
| Pure Real Reference | 0.830 | 1.172 | 0.779 | 1.052 |
| Real-Synthetic 75-25 | **0.685** | **0.984** | 0.794 | 1.052 |
| Real-Synthetic 50-50 | 0.710 | 1.012 | **0.772** | **1.019** |
| Real-Synthetic 25-75 | 0.775 | 1.100 | 0.780 | 1.044 |

### Key Findings
- Generators are not interchangeable and rankings do not transfer: Selection is an architecture-specific decision.
- Mixed11 is a robust synthetic default: It ties with the best single generator for Moirai and significantly outperforms it for Chronos (MASE $+10.2\%$, CI above 0).
- The strongest corpora come from "Real + Synthetic" compositions, but optimal ratios vary: Moirai favors 75-25, while Chronos favors 50-50.
- Synthetic data is a supplement for Moirai: The 75-25 mix outperforms both pure real and pure synthetic endpoints.

## Highlights & Insights
- The "Mix, Don't Pick" insight is highly transferable: When a selection dimension (generators) is unstable across conditions (architectures), robust mixing is superior to optimization.
- Rigorous "from-scratch" training, matched budgets, leakage prevention, and paired bootstrapping establish composition as a first-order effect with causal credibility.
- Debunking the "fidelity equals utility" hypothesis: Physical resemblance to real data is not the same as pretraining utility, warning the community against using feature-matching to select generators.
- Analogizing TSFM corpus composition to LLM data mixing connects time series research to a mature set of methodologies for weight learning.

## Limitations & Future Work
- The study covers only two compact TSFM architectures and a fixed 1M window budget; results might shift with larger models or longer training.
- Domain-level results suggest optimal compositions may vary by domain or horizon rather than following a single global ratio.
- Equal-weight Mixed11 is robust but not necessarily optimal. Future work includes leave-one-generator-out marginal contribution estimation and learning conditional mixing weights.
- There is no established predictive link between a generator's feature-space diagnostics and its downstream utility; selection currently relies on empirical scanning.

## Related Work & Insights
- **vs. Fidelity-based Generator Validation (Bahrpeyma et al. 2021, etc.)**: These works use feature matching to select generators. This paper empirically shows "resemblance" does not equal "utility," invalidating that selection path.
- **vs. Existing Synthetic TSFM Pretraining (Chronos / Moirai / TimesFM, etc.)**: These works treat synthetic data as a single source or simple augmentation based on convention. This paper is the first to systematically test ranking transferability and real-synthetic ratios.
- **vs. LLM Data Mixing (Xie et al. 2023, DoReMi, etc.)**: This paper adopts the LLM framework that corpus design has a first-order impact but currently uses naive mixing rather than learned weights.

## Rating
- Novelty: ⭐⭐⭐⭐ Reframing "selection" as "composition" is a fresh and compelling perspective for the field.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive comparisons across generators and architectures with rigorous controls.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, honest conclusions, and high information density.
- Value: ⭐⭐⭐⭐ Provides actionable guidelines for TSFM pretraining (mixing is better than picking; validate per-architecture).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] CauKer: Classification Time Series Foundation Models Can Be Pretrained on Synthetic Data](../../ICLR2026/time_series/cauker_classification_time_series_foundation_models_can_be_pretrained_on_synthet.md)
- [\[ICLR 2026\] Adapt Data to Model: Adaptive Transformation Optimization for Domain-shared Time Series Foundation Models](../../ICLR2026/time_series/adapt_data_to_model_adaptive_transformation_optimization_for_domain-shared_time_.md)
- [\[ICML 2026\] OLIVIA: Harmonizing Time Series Foundation Models with Power Spectral Density](olivia_harmonizing_time_series_foundation_models_with_power_spectral_density.md)
- [\[ICLR 2026\] UniCA: Unified Covariate Adaptation for Time Series Foundation Model](../../ICLR2026/time_series/unica_unified_covariate_adaptation_for_time_series_foundation_model.md)
- [\[ICML 2026\] Do Time Series Foundation Model Benchmarks Hide Regime-Dependent Failures? Evidence from Traffic Speed Forecasting](do_time_series_foundation_model_benchmarks_hide_regime-dependent_failures_eviden.md)

</div>

<!-- RELATED:END -->
