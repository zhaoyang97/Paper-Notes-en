---
title: >-
  [Paper Note] Cascaded Flow Matching for Heterogeneous Tabular Data with Mixed-Type Features
description: >-
  [ICML 2026][Flow matching] TabCascade decomposes each table row into "low-resolution (categorical + discretized numerical)" and "high-resolution (continuous numerical)" cascaded stages: first…
tags:
  - "ICML 2026"
  - "Flow matching"
  - "cascaded diffusion"
  - "tabular data"
  - "mixed-type features"
  - "missing value generation"
date: 2026-05-08
content_hash: 5a43830f2868ffae
---

# Cascaded Flow Matching for Heterogeneous Tabular Data with Mixed-Type Features

**Conference**: ICML 2026  
**arXiv**: [2601.22816](https://arxiv.org/abs/2601.22816)  
**Code**: https://github.com/muellermarkus/tabcascade  
**Area**: Diffusion Models / Tabular Data Generation / Generative Modeling  
**Keywords**: Flow matching, cascaded diffusion, tabular data, mixed-type features, missing value generation

## TL;DR
TabCascade decomposes each table row into "low-resolution (categorical + discretized numerical)" and "high-resolution (continuous numerical)" cascaded stages: first, CDTD learns the low-res joint distribution; then, flow matching generates numerical details conditioned on the low-res output, with data-dependent coupling and a learnable nonlinear time schedule to tighten transport cost. It natively supports generation of mixed-type features such as missing values and zero-inflation, achieving a 51.9% improvement in detection score over SOTA on 12 datasets.

## Background & Motivation
**Background**: Tabular data generation is a core need in finance, healthcare, and survey scenarios. Mainstream methods (TabDDPM, TabSyn, CDTD, TabDiff) treat categorical and numerical features with a shared diffusion/flow objective.

**Limitations of Prior Work**: (1) Categorical and numerical features have fundamentally different structures (discrete support vs. continuous density, probability mass vs. probability density), so a single objective implicitly biases training toward certain features; (2) **Mixed-type features** (e.g., zero-inflated salaries, missing or censored values) are distributions combining continuous density and discrete point mass, which current diffusion models cannot handle—they can train on data with missing values but only generate continuous values, never NaN or exact zeros; (3) Literature shows numerical features are much harder to learn than categorical ones (detection score on num is much lower than cat)—yet current models use the same capacity for both.

**Key Challenge**: The dual challenge of heterogeneous features and mixed-type distributions—single unified objectives cannot simultaneously (a) balance the relative importance of categorical vs. numerical, (b) accurately place point mass at specific discrete values (e.g., 0, NaN), and (c) give sufficient expressiveness to the remaining continuous part of numerical features.

**Goal**: (1) Decouple "hard-to-learn numerical details" from "easy-to-learn categorical structure," assigning dedicated models to each; (2) Explicitly encode discrete decisions (missing/zero-inflated/normal) in the generation pipeline; (3) Introduce a "low-resolution guides high-resolution" mechanism inspired by image cascaded diffusion to reduce transport cost.

**Key Insight**: Image cascaded diffusion (Imagen) uses low-res versions to guide high-res detail generation. The authors transplant this metaphor to tabular data—modeling "categorical features = low-res" and "numerical features = high-res" separately.

**Core Idea**: Decompose $p_\theta(x_{cat}, x_{num}) = \sum_z p_\theta^{\text{high}}(x_{num} | z, x_{cat}) p_\theta^{\text{low}}(z, x_{cat})$, where $z$ is the discretized version of numerical features (obtained via decision tree or GMM), serving both as conditioning and as an explicit carrier of mixed-type discrete states.

## Method

### Overall Architecture
TabCascade expands each table row $x = (x_{cat}, x_{num})$ into $x_{low} = (x_{cat}, z)$ and $x_{num}$. **(1) Offline**: For each numerical feature $x^{(i)}_{num}$, train a Distributional Regression Tree (DT) or GMM encoder $\text{Enc}_i$ to output $z^{(i)} = \text{Enc}_i(x^{(i)}_1)$ as a coarse bucket id; each bucket corresponds to a Gaussian component $\mathcal{N}(\mu_{z^{(i)}}, \sigma^2_{z^{(i)}})$. **(2) Low-resolution model**: Use CDTD to learn $p_\theta^{\text{low}}(z, x_{cat})$, generating categorical + numerical bucket ids. **(3) High-resolution model**: Conditioned on $z, x_{cat}$, use flow matching to generate numerical details $x_{num}$. **(4) Sampling**: First sample $z, x_{cat} \sim p_\theta^{\text{low}}$; if $z^{(i)} = c_{miss}$, output NaN directly; if $= c_{infl}$, output the inflated value; otherwise, use the high-res flow to generate continuous values.

### Key Designs

1. **Cascaded Decomposition with $z$ as Categorical Surrogate**:

    - **Function**: Decomposes the "hard" numerical generation into two steps: "first decide coarse bucket (low-res), then refine within-bucket details (high-res)"; $z$ also serves as the carrier for mixed-type states (missing/inflated are special categories of $z$).
    - **Mechanism**: The chain rule ensures $H(x_{num} | z, x_{cat}) < H(x_{num} | x_{cat})$ (when $z$ and $x_{num}$ are not independent), so the conditional entropy is lower, simplifying the high-res model's task. $z^{(i)}$ is obtained via a DT encoder, with each leaf node corresponding to a Gaussian component; missing values are assigned a special category $c_{miss}$; buckets with $\sigma_{z^{(i)}}^2 \approx 0$ are treated as inflated values and output $\mu_{z^{(i)}}$ directly.
    - **Design Motivation**: (a) Assign dedicated models to categorical/numerical features, naturally eliminating the "loss weight balancing" issue (CDTD requires tuning relative weights for improvement, TabCascade does not); (b) Offload the discrete point mass of mixed-type features to the low-res model, letting the high-res model focus solely on the continuous part.

2. **Guided Conditional Probability Path + Data-Dependent Coupling**:

    - **Function**: Uses $z$ to guide the source distribution and time schedule of the high-res flow, making the source distribution naturally closer to the target and shortening the transport distance.
    - **Mechanism**: Standard flow matching uses $x_t = t x_1 + (1-t) x_0, x_0 \sim \mathcal{N}(0, I)$. TabCascade introduces two improvements: (1) Data-dependent coupling $x_0 = \mu(z) + \sigma(z) \varepsilon, \varepsilon \sim \mathcal{N}(0,I)$, so the source is directly near the target bucket; (2) Feature-specific nonlinear time schedule $\gamma_t(x_{low}): t \to [0,1]^{K_{num}}$ parameterized by a 5th-order polynomial, closed-form differentiable. The final path is $x_t = \gamma_t(x_{low}) x_1 + (1-\gamma_t(x_{low}))[\mu(z) + \sigma(z) \varepsilon]$, with the guided vector field $u_t(x_t | x_1, x_{low}) = \dot{\gamma}_t(x_{low})(x_1 - x_t)/(1 - \gamma_t(x_{low}))$.
    - **Design Motivation**: The authors prove (Theorem 1) that DT-derived coupling strictly tightens the transport cost bound, making it easier to learn than independent coupling; Figure 3 visually shows that under data-dependent coupling, $p_0$ is already very close to $p_1$, allowing model capacity to focus on details.

3. **Mixed-Type Loss Masking + Disentangled Cascade**:

    - **Function**: Ensures the high-res model does not handle discrete states, dedicating its capacity to continuous details.
    - **Mechanism**: When training high-res, samples with $z = c_{miss}$ or $c_{infl}$ are masked from the CFM loss (loss term not included), as they are fully determined by $p_\theta^{\text{low}}$. During generation, the branch is chosen based on $z$: discrete states output fixed values directly, continuous states use the flow.
    - **Design Motivation**: Traditional models must learn both "should this be missing" and "what is the value if not missing," with gradients interfering; TabCascade explicitly separates these via $z$, enabling divide-and-conquer.

### Loss & Training
Low-res: Reuses CDTD's score interpolation loss. High-res: CFM loss $\mathcal{L}_{\text{CFM}} = \mathbb{E} \| u_t^\theta(x_t | x_{low}) - \dot{\gamma}_t(x_{low})(x_1 - [\mu(z) + \sigma(z) \varepsilon]) \|^2$, where $u_t^\theta = \dot{\gamma}_t \cdot f^\theta(x_t, x_{low}, t)$. The two stages are trained independently, with no need to balance cross-type loss weights—this is a key engineering simplification over CDTD.

## Key Experimental Results

### Main Results
12 standard tabular datasets × 7 SOTA baselines, average results:

| Metric | TabDDPM | TabSyn | TabDiff | CDTD | **Ours (DT)** |
|--------|---------|--------|---------|------|--------------|
| Detection Score↑ | 0.478 | 0.202 | 0.430 | 0.518 | **0.787** |
| Shape↑ | 0.938 | 0.927 | 0.954 | 0.970 | **0.984** |
| Shape (num)↑ | 0.943 | 0.918 | 0.952 | 0.962 | **0.985** |
| WD (num)↓ | 0.015 | 0.031 | 0.016 | 0.009 | **0.004** |
| JSD (cat)↓ | 0.083 | 0.063 | 0.030 | 0.020 | **0.018** |
| Trend↑ | 0.900 | 0.893 | 0.924 | 0.956 | **0.965** |

Detection score improves from 0.518 (CDTD) to 0.787, a relative increase of about 52% (the 51.9% figure in the abstract); WD for numerical features is more than halved compared to CDTD (0.009→0.004).

### Ablation Study
Motivational experiments (from Figure 2):

| Configuration | Detection Score | Notes |
|---------------|----------------|-------|
| Mean diffusion baseline on cat only | ~0.85 | Categorical is easy |
| Mean diffusion baseline on num only | ~0.55 | Numerical is hard |
| Ours on cat / num / all | ~0.85+ all | After decoupling, num no longer drags down performance |
| CDTD adult with cat loss weight=1×→4× | 0.72 → 0.76 | Manual tuning required |

### Key Findings
- **Numerical generation quality is the real bottleneck for tabular diffusion**: After decoupling, WD for numerical features drops from 0.009 to 0.004, a leap not achieved by previous models; categorical scores were already near 1, so maintaining parity is SOTA.
- **Mixed-type support is a qualitative leap**: All baselines (including the strongest CDTD/TabDiff) cannot natively generate NaN/exact 0; TabCascade is the first diffusion model with native support.
- **DT encoder outperforms GMM** (see appendix), as DT leaf nodes can partition based on other features, providing more informative $z$.
- **No need to tune loss balance**: CDTD requires grid-searching relative weights for improvement, while TabCascade's independent two-stage training avoids this issue entirely.
- Variance across 12 datasets is low (detection std only 0.243 vs CDTD 0.296), indicating greater robustness.

## Highlights & Insights
- **Translating the "image cascaded diffusion" concept to tabular data** is highly creative: tables lack a "resolution" concept, but the authors operationalize "categorical = low-res" and "numerical = high-res," with $H(x_{num} | z, x_{cat}) < H(x_{num} | x_{cat})$ strictly proving the task is simplified.
- **Mixed-type generation directly addresses real pain points**: In real tables, missing values are often informative (e.g., survey refusals indicate personality, missing medical tests indicate risk); being able to generate precise NaN is transformative for downstream imputation/counterfactual analysis.
- **Data-dependent coupling (Theorem 1)**: DT-derived coupling strictly tightens the transport cost bound—aligned with recent mini-batch OT couplings (Tong 2024) but with zero extra cost, and can be transferred to any flow matching scenario.
- **Completely avoids "loss balance" headaches**: CDTD series spent much effort on cross-type loss balancing; TabCascade's cascade approach is a clean solution, suggesting that divide-and-conquer is often more practical than unified objectives for heterogeneous data.
- The overall approach (cascade + explicit discrete state modeling + guided path) can be transferred to: time series with missing data, multimodal EHR, and multimodal recommendation features.

## Limitations & Future Work
- **Two-stage training increases complexity**: Requires training encoders → low-res → high-res; the pipeline is longer and more complex than single-stage CDTD.
- **Encoder quality determines upper bound**: If DT bucketization is poor, $z$ cannot provide effective guidance and level 2 cascade fails; DT may overfit on high-cardinality categories.
- **Risk of correlation loss**: If discretization of numerical features into $z$ is too coarse, fine-grained cross-feature correlations may be lost; paper's Trend (mixed) is only 0.928, slightly lower than cat-only Trend.
- **Generation latency is doubled**: Two models sampled in series are slower than a single model; lacks speed comparison data.
- Future directions: (a) Make the encoder end-to-end trainable; (b) Explore more cascade levels (coarse → medium → fine); (c) Extend the mixed-type framework to censored survival data; (d) Integrate imputation models for unified "generation + completion."

## Related Work & Insights
- **vs TabDDPM / CoDi**: These combine multinomial diffusion and DDPM but do not address the difficulty of learning numerical features; TabCascade explicitly stages the process.
- **vs TabSyn**: Uses latent diffusion to embed all features into continuous space, but (Mueller 2025) shows latent diffusion underperforms data space for tables; TabCascade stays in data space and is hierarchical.
- **vs CDTD / TabDiff**: Previous work from the Mueller group; TabDiff/CDTD learn joint noise schedules to address heterogeneity; TabCascade redefines "heterogeneity" as a "resolution" problem, a more thorough approach.
- **vs Cascaded Diffusion (Ho 2022) / Imagen**: Methodologically derived from image cascade, but tables lack the physical meaning of "super resolution"; the authors reinterpret via feature types and introduce mixed-type generation as a unique contribution.
- **vs Sahoo 2024 (latent noise schedule)**: That work makes the noise schedule depend on latent variables, a weaker form of cascade; TabCascade explicitly generates $z$ rather than using latent variables, making it more interpretable.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First cascaded diffusion for tabular data + first method to generate mixed-type features (NaN/inflated), with elegant conceptual translation and solid theory.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 12 datasets, 7 SOTA baselines, comprehensive coverage of motivation/main results/MIA/MLE/ablation, with variance reported.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is data-driven (Figure 2), theory is rigorously proven (Theorem 1), and workflow diagrams are clear—rare "full-stack" quality for tabular generation papers.
- Value: ⭐⭐⭐⭐⭐ Solves two long-standing pain points in tabular generation (mixed-type + numerical precision), with a 52% leap in detection score; disruptive for synthetic data, privacy-preserving release, and imputation industries.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] The Coupling Within: Flow Matching via Distilled Normalizing Flows](the_coupling_within_flow_matching_via_distilled_normalizing_flows.md)
- [\[ICML 2026\] Exploring and Exploiting Stability in Latent Flow Matching](exploring_and_exploiting_stability_in_latent_flow_matching.md)
- [\[ICML 2025\] Elucidating Flow Matching ODE Dynamics via Data Geometry and Denoisers](../../ICML2025/image_generation/elucidating_flow_matching_ode_dynamics_with_respect_to_data_geometries_and_denoi.md)
- [\[ICML 2026\] Saving Foundation Flow-Matching Priors for Inverse Problems](saving_foundation_flow-matching_priors_for_inverse_problems.md)
- [\[ICML 2025\] Understanding and Mitigating Memorization in Diffusion Models for Tabular Data](../../ICML2025/image_generation/understanding_and_mitigating_memorization_in_diffusion_models_for_tabular_data.md)

</div>

<!-- RELATED:END -->
