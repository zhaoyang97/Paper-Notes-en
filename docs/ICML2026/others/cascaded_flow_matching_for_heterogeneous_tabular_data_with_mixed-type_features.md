---
title: >-
  [Paper Note] Cascaded Flow Matching for Heterogeneous Tabular Data with Mixed-Type Features
description: >-
  [ICML 2026][Flow matching] TabCascade decomposes tabular rows into two cascaded stages: "low-resolution (categorical + discretized version of numerical)" and "high-resolution (continuous numerical)". It first uses CDTD t…
tags:
  - "ICML 2026"
  - "Flow matching"
  - "Cascaded diffusion"
  - "Tabular data"
  - "Mixed-type features"
  - "Missing value generation"
date: 2026-05-08
content_hash: ffb55487e95feb64
---

# Cascaded Flow Matching for Heterogeneous Tabular Data with Mixed-Type Features

**Conference**: ICML 2026  
**arXiv**: [2601.22816](https://arxiv.org/abs/2601.22816)  
**Code**: https://github.com/muellermarkus/tabcascade  
**Area**: Diffusion Models / Tabular Data Generation / Generative Modeling  
**Keywords**: Flow matching, Cascaded diffusion, Tabular data, Mixed-type features, Missing value generation

## TL;DR
TabCascade decomposes tabular rows into two cascaded stages: "low-resolution (categorical + discretized version of numerical)" and "high-resolution (continuous numerical)". It first uses CDTD to learn the low-resolution joint distribution, then employs flow matching guided by the low-resolution stage to generate numerical details. By utilizing data-dependent coupling and learnable non-linear schedules, it tightens transport costs. It natively supports "mixed-type features" such as missing values and zero-inflation, achieving a 51.9% improvement in detection score over the SOTA on 12 datasets.

## Background & Motivation
**Background**: Tabular data generation is a core requirement in finance, medicine, and survey scenarios. Leading methods (TabDDPM, TabSyn, CDTD, TabDiff) typically bundle categorical and numerical features into a shared diffusion/flow objective.

**Limitations of Prior Work**: (1) Categorical and numerical structures are fundamentally different (discrete support vs. continuous density, probability mass vs. probability density); a single objective allows certain features to implicitly dominate training. (2) **Mixed-type features** (e.g., zero-inflated wages, missing values, censored values) represent distributions where discrete point masses superimpose on continuous densities. Existing diffusion models fail here—they can train on data with missing values but only output continuous values during generation, failing to produce exact NaNs or zeros. (3) Evidence suggests numerical features are significantly harder to learn than categorical ones (numerical detection scores are far lower than categorical)—yet current models apply the same capacity to both.

**Key Challenge**: The double difficulty of heterogeneous features and mixed-type distributions. A unified objective cannot simultaneously (a) balance the relative importance of categorical vs. numerical features, (b) precisely place point masses at specific discrete values (e.g., 0, NaN), and (c) provide sufficient expressivity for the remaining continuous parts of numerical features.

**Goal**: (1) Decouple "hard-to-learn numerical details" from "easy-to-learn categorical structures," using specialized models for each. (2) Explicitly encode discrete decisions (e.g., "is this position missing/zero-inflated/normal") in the generation pipeline. (3) Adopt the "low-resolution guided high-resolution" mechanism from image cascaded diffusion to reduce transport costs.

**Key Insight**: Image cascaded diffusion (e.g., Imagen) uses low-resolution versions to guide high-resolution detail generation. The authors port this metaphor to tabular data by modeling "categorical features = low-resolution" and "numerical features = high-resolution" separately.

**Core Idea**: Decompose $p_\theta(x_{cat}, x_{num}) = \sum_z p_\theta^{\text{high}}(x_{num} | z, x_{cat}) p_\theta^{\text{low}}(z, x_{cat})$, where $z$ is a discretized version of numerical features (obtained via decision trees or GMM). This $z$ serves both as a conditioning variable and a representation for the discrete states of mixed-type features.

## Method

### Overall Architecture
TabCascade expands a tabular row $x = (x_{cat}, x_{num})$ into $x_{low} = (x_{cat}, z)$ and $x_{num}$. **(1) Offline**: For each numerical feature $x^{(i)}_{num}$, a Distributional Regression Tree (DT) or GMM encoder $\text{Enc}_i$ is trained to output $z^{(i)} = \text{Enc}_i(x^{(i)}_1)$ as a coarse bin ID; each bin corresponds to a Gaussian component $\mathcal{N}(\mu_{z^{(i)}}, \sigma^2_{z^{(i)}})$. **(2) Low-resolution model**: CDTD is used to learn $p_\theta^{\text{low}}(z, x_{cat})$, generating categories and numerical bin IDs. **(3) High-resolution model**: Conditional on $z$ and $x_{cat}$, flow matching generates numerical details $x_{num}$. **(4) Sampling**: First sample $z, x_{cat} \sim p_\theta^{\text{low}}$. If $z^{(i)} = c_{miss}$, output NaN directly; if $z^{(i)} = c_{infl}$, output the inflated value directly; otherwise, use the high-res flow to generate the continuous value.

### Key Designs

1. **Cascaded Decomposition with $z$ as Categorical Surrogate**:
    - **Function**: Breaks down "difficult" numerical generation into "decide coarse bin (low-res)" and "refine intra-bin details (high-res)". $z$ also carries mixed-type information (missing/inflated values are unique categories in $z$).
    - **Mechanism**: The chain rule ensures $H(x_{num} | z, x_{cat}) < H(x_{num} | x_{cat})$ (when $z$ is not independent of $x_{num}$), meaning lower conditional entropy simplifies the high-res model's task. $z^{(i)}$ is obtained via DT encoders where leaves are Gaussian components. Missing values are assigned to an independent category $c_{miss}$, and bins with $\sigma_{z^{(i)}}^2 \approx 0$ are treated as inflated values.
    - **Design Motivation**: (a) Specialized models for categorical/numerical features eliminate the need for manual loss balancing. (b) Offloading discrete point masses to the low-res model allows the high-res model to focus exclusively on continuous distributions.

2. **Guided Conditional Probability Path + Data-Dependent Coupling**:
    - **Function**: Uses $z$ to guide the source distribution and time schedule of the high-res flow, bringing the source closer to the target and shortening transmission distance.
    - **Mechanism**: Standard flow matching uses $x_t = t x_1 + (1-t) x_0, x_0 \sim \mathcal{N}(0, I)$. TabCascade introduces two improvements: (1) Data-dependent coupling $x_0 = \mu(z) + \sigma(z) \varepsilon, \varepsilon \sim \mathcal{N}(0,I)$, effectively centering the source distribution near the target bin. (2) Feature-specific non-linear schedules $\gamma_t(x_{low}): t \to [0,1]^{K_{num}}$ parameterized by 5th-order polynomials. The final path is $x_t = \gamma_t(x_{low}) x_1 + (1-\gamma_t(x_{low}))[\mu(z) + \sigma(z) \varepsilon]$, with a vector field $u_t(x_t | x_1, x_{low}) = \dot{\gamma}_t(x_{low})(x_1 - x_t)/(1 - \gamma_t(x_{low}))$.
    - **Design Motivation**: Theorem 1 proves that DT-derived coupling strictly tightens the transport cost bound. Figure 3 shows $p_0$ is already very close to $p_1$, allowing model capacity to concentrate on details.

3. **Mixed-Type Loss Masking + Disentangled Cascade**:
    - **Function**: Removes discrete state handling from the high-res model.
    - **Mechanism**: During high-res training, CFM loss is masked for samples where $z = c_{miss}$ or $c_{infl}$, as these are fully determined by $p_\theta^{\text{low}}$. During generation, branches are followed based on $z$: discrete states return fixed values, while continuous states invoke the flow model.
    - **Design Motivation**: Traditional models must learn both "whether it is missing" and "what value it takes if not missing." TabCascade uses $z$ to explicitly separate these concerns, applying a divide-and-conquer strategy.

### Loss & Training
**Low-res**: Reuses the score interpolation loss from CDTD. **High-res**: CFM loss $\mathcal{L}_{\text{CFM}} = \mathbb{E} \| u_t^\theta(x_t | x_{low}) - \dot{\gamma}_t(x_{low})(x_1 - [\mu(z) + \sigma(z) \varepsilon]) \|^2$, where $u_t^\theta = \dot{\gamma}_t \cdot f^\theta(x_t, x_{low}, t)$. The two stages are trained independently, removing the need to balance cross-type loss weights—a significant engineering simplification over CDTD.

## Key Experimental Results

### Main Results
Average results across 12 standard tabular datasets with 7 SOTA comparisons:

| Metric | TabDDPM | TabSyn | TabDiff | CDTD | **Ours (DT)** |
|------|---------|--------|---------|------|-------|
| Detection Score↑ | 0.478 | 0.202 | 0.430 | 0.518 | **0.787** |
| Shape↑ | 0.938 | 0.927 | 0.954 | 0.970 | **0.984** |
| Shape (num)↑ | 0.943 | 0.918 | 0.952 | 0.962 | **0.985** |
| WD (num)↓ | 0.015 | 0.031 | 0.016 | 0.009 | **0.004** |
| JSD (cat)↓ | 0.083 | 0.063 | 0.030 | 0.020 | **0.018** |
| Trend↑ | 0.900 | 0.893 | 0.924 | 0.956 | **0.965** |

The detection score improved from CDTD's 0.518 to 0.787, a relative increase of ~52%. The Wasserstein Distance (WD) for numerical features was reduced by over half compared to CDTD (0.009 → 0.004).

### Ablation Study
Motivational experiments (from Figure 2):

| Configuration | Detection Score | Description |
|------|---------|------|
| Avg diffusion baseline on cat only | ~0.85 | Categorical is simple |
| Avg diffusion baseline on num only | ~0.55 | Numerical is difficult |
| Ours on cat / num / total | ~0.85+ all | Num no longer drags down total after decoupling |
| CDTD on adult (cat loss weight=1×→4×) | 0.72 → 0.76 | Manual balancing is required |

### Key Findings
- **Numerical generation quality is the true bottleneck**: After decoupling, numerical WD dropped from 0.009 to 0.004—a breakthrough previous models couldn't achieve. Categorical performance remains at SOTA levels.
- **Mixed-type support is a qualitative leap**: All baselines (including CDTD/TabDiff) cannot natively generate NaNs or exact zeros. TabCascade is the first natively supportive diffusion model.
- **DT Encoders outperform GMM**: DT leaf nodes account for dependencies between other features, providing more informative $z$ values.
- **No loss balance tuning needed**: While CDTD requires grid searches for relative weights, TabCascade avoids this via independent stage training.
- **Robustness**: Low variance across 12 datasets (detection std 0.243 vs. CDTD 0.296).

## Highlights & Insights
- **Clever porting of "Image Cascaded Diffusion"**: Tabular data lacks a natural concept of "resolution"; the authors cleverly mapped "categorical = low-res" and "numerical = high-res," proving theoretically that $H(x_{num} | z, x_{cat}) < H(x_{num} | x_{cat})$ simplifies the task.
- **Addressing Mixed-type Generation**: In real-world tables, missing values are often informative (e.g., medical data indicating risk). Generating precise NaNs is transformative for downstream imputation and counterfactual analysis.
- **Theorem 1 on Data-Dependent Coupling**: DT encoder-derived coupling strictly tightens the transport cost bound—aligning with recent mini-batch OT coupling research but with zero additional overhead.
- **Divide and Conquer**: The work suggests that for heterogeneous data, "divide and conquer" is often more engineering-feasible than a single "unified objective."
- The methodology (cascade + explicit discrete state modeling + guided paths) is transferable to time series with missingness and multimodal health records.

## Limitations & Future Work
- **Two-stage Training Complexity**: Requires training encoders → low-res → high-res, resulting in a longer pipeline compared to one-stage models.
- **Encoder Dependency**: If the DT binning is poor, $z$ provides insufficient guidance, causing the stage-2 cascade to fail. DTs may overfit on high-cardinality categories.
- **Risk to Correlations**: Coarse binning for $z$ might lose fine-grained correlations between features (Trend (mixed) is 0.928, slightly lower than categorical-only).
- **Sampling Latency**: Sequential sampling from two models is slower; the paper lacks speed comparison data.

## Related Work & Insights
- **vs TabDDPM / CoDi**: Those models combined multinomial diffusion and DDPM but didn't solve the numerical learning difficulty; TabCascade uses explicit stages.
- **vs TabSyn**: TabSyn maps features to a continuous latent space, but recent research (Mueller 2025) suggests tabular latent diffusion is inferior to data space models; TabCascade remains in the data space while employing levels.
- **vs CDTD / TabDiff**: These prior works by the same team used joint noise schedules to handle heterogeneity; TabCascade redefines heterogeneity as a resolution problem.
- **vs Cascaded Diffusion (Ho 2022) / Imagen**: Inherits the methodology from image cascades but reinterprets it through feature types.
- **vs Sahoo 2024**: That work used latent-dependent noise schedules, which is a weaker version of cascading; TabCascade's explicit $z$ generation is more interpretable.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First tabular cascaded diffusion + first method to natively generate mixed-type features (NaN/inflated). Excellent conceptual translation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 12 datasets, 7 SOTA, motivation, MIA/MLE, and full ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Data-driven motivation (Fig 2), rigorous theory (Thm 1), and clear flowcharts.
- **Value**: ⭐⭐⭐⭐⭐ Resolves two long-standing tabular pain points (mixed-type + numerical precision). The 52% jump in detection score is a disruptive development for synthetic data and imputation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Parsimonious Learning-Augmented Online Metric Matching](parsimonious_learning-augmented_online_metric_matching.md)
- [\[ICML 2026\] Active Tabular Augmentation via Policy-Guided Diffusion Inpainting](active_tabular_augmentation_via_policy-guided_diffusion_inpainting.md)
- [\[AAAI 2026\] Cash Flow Underwriting with Bank Transaction Data: Advancing MSME Financial Inclusion in Malaysia](../../AAAI2026/others/cash_flow_underwriting_with_bank_transaction_data_advancing_msme_financial_inclu.md)
- [\[NeurIPS 2025\] Radar: Benchmarking Language Models on Imperfect Tabular Data](../../NeurIPS2025/others/radar_benchmarking_language_models_on_imperfect_tabular_data.md)
- [\[AAAI 2026\] Bipartite Mode Matching for Vision Training Set Search from a Hierarchical Data Server](../../AAAI2026/others/bipartite_mode_matching_for_vision_training_set_search_from_a_hierarchical_data_.md)

</div>

<!-- RELATED:END -->
