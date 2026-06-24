---
title: >-
  [Paper Note] MIAM: Modality Imbalance-Aware Masking for Multimodal Ecological Applications
description: >-
  [ICLR 2026][Multimodal VLM][Modality Imbalance] The authors formalize the masking strategy as a probability distribution on a unit hypercube and propose MIAM—a hybrid product-beta distribution featuring full support and corner prioritization. It dynamically increases the masking probability for dominant modalities based on relative performance and learning speed, providing a unified mechanism to resolve robustness to missing data, modality imbalance…
tags:
  - "ICLR 2026"
  - "Multimodal VLM"
  - "Modality Imbalance"
  - "Dynamic Masking"
  - "Missing Modalities"
  - "Multimodal Fusion"
  - "Species Distribution Modeling"
date: 2026-05-08
content_hash: b66428bce9f413fa
---

# MIAM: Modality Imbalance-Aware Masking for Multimodal Ecological Applications

**Conference**: ICLR 2026  
**Code**: [https://github.com/zbirobin/MIAM](https://github.com/zbirobin/MIAM)  
**Area**: Multimodal Learning / Data Masking / Ecological Applications  
**Keywords**: Modality Imbalance, Dynamic Masking, Missing Modalities, Multimodal Fusion, Species Distribution Modeling  

## TL;DR
The authors formalize the masking strategy as a probability distribution on a unit hypercube and propose MIAM—a hybrid product-beta distribution featuring full support and corner prioritization. It dynamically increases the masking probability for dominant modalities based on relative performance and learning speed, providing a unified mechanism to resolve robustness to missing data, modality imbalance, and fine-grained contribution analysis in multimodal ecological data.

## Background & Motivation
**Background**: Ecological modeling inherently relies on heterogeneous multimodal data—satellite imagery, environmental time series, tabular predictors (elevation/soil), bioacoustics, etc. Recently, multimodal learning has advanced through "data masking": during training, parts of the input are randomly hidden according to a probability distribution (e.g., 4M, MultiMAE, MaskSDM). This simulates missing data to enhance robustness and supports Shapley-style feature contribution analysis.

**Limitations of Prior Work**: Ecological data exhibits two layers of missingness: modality-level (e.g., no satellite imagery for a location) and intra-modality (e.g., missing years in climate time series), requiring models to operate flexibly on arbitrary, incomplete input subsets. However, existing masking distributions have systematic flaws: (1) Shared probability $p\sim U(0,1)$ (MaskSDM) causes the probability of observing only non-dominant modalities to decay exponentially with the number of tokens, meaning the model almost always sees dominant modality tokens; (2) Symmetric Dirichlet (4M) constrains the visible token ratio near $1/M$, limiting input combination diversity; (3) Modality dropout and the aforementioned methods treat all modalities equally, ignoring modality competition; (4) While OPM adjusts probabilities by performance, it restricts $p\in\{0,1\}^M$ (all-or-nothing masking), making it nearly static during fractional training and preventing fine-grained masking.

**Key Challenge**: **Modality imbalance (modality competition)**—dominant modalities monopolize most prediction signals and gradient flows, suppressing the optimization of other complementary modalities. This leads to multimodal models performing worse than unimodal oracles when evaluated only on weak modalities (Figure 1). Existing masking strategies fail to address this due to fixed, uniform, or insufficiently exploratory distributions.

**Goal**: Design a masking strategy that simultaneously satisfies "handling arbitrary missing inputs + mitigating modality imbalance + supporting cross-modality and intra-modality contribution analysis" without requiring extra components like teacher networks or gradient re-weighting.

**Key Insight**: **Formalize the masking strategy as a probability distribution over the hypercube $[0,1]^M$**, distill three effective masking principles (full support, corner-prioritization, and imbalance-awareness), and construct a **corner-anchored hybrid product-beta distribution** that adaptively adjusts according to modality learning dynamics during training.

## Method

### Overall Architecture
Each sample contains $M$ modalities, where modality $m$ has $T_m$ tokens. All tokens within the same modality share a masking probability $p_m$. The resulting masking probability vector $p=(p_1,\dots,p_M)$ resides on an $M$-dimensional unit hypercube, and the masking strategy is defined as a distribution (which evolves during training) over this cube. MIAM employs a hybrid product-beta distribution to concentrate probability mass near the hypercube corners to ensure full support and corner prioritization. It then uses two coefficients for each modality—"relative performance $\rho_{s_m}$" and "learning speed $\rho_{d_m}$"—to dynamically adjust the sharpness parameters of the beta distribution, ensuring that strong and stable modalities are masked more frequently. Tokens are masked according to $p_m$ and fed into a transformer for fusion and prediction.

```mermaid
flowchart LR
    A[Multimodal Input<br/>M modalities, Tm tokens each] --> B[Sample mask vector p~MixProdBeta]
    B --> C[Mask tokens per modality by pm]
    C --> D[Transformer Fusion]
    D --> E[Prediction]
    E --> F[Validate per-modality performance sm and its derivative dm]
    F --> G[Calculate ρsm, ρdm]
    G -->|Adjust sharpness κ·(ρsm/ρdm)^λ| B
```

### Key Designs
**1. Three Effective Masking Principles**: The authors argue that a superior masking distribution should satisfy: **Full support** (assigning non-zero probability to any $p$ to ensure all mask combinations are possible); **Corner prioritization** (sampling more frequently near the corners of the hypercube, as ecological missingness often occurs at the modality level rather than token level—making "almost entirely present/absent" scenarios critical; specifically weighting the $(0,\dots,0)$ and $(1,\dots,1)$ corners corresponding to all modalities available versus minimal tokens remaining); and **Imbalance awareness** (assigning higher masking probabilities to dominant modalities identified by performance or learning speed). These three principles define MIAM's design objectives.

**2. Corner-Anchored Hybrid Product-Beta Distribution**: For each corner $c\in\{0,1\}^M$, a product-beta component is defined. If $c_m=0$, $\mathrm{Beta}(p_m;1,\kappa)$ is used (pushing mass toward 0); if $c_m=1$, $\mathrm{Beta}(p_m;\kappa,1)$ is used (pushing mass toward 1). The sharpness $\kappa>1$ controls the concentration. The overall distribution is a weighted mixture of $2^M$ corner components: $\mathrm{MixProdBeta}(p)=\sum_{c\in C}w_c f_c(p)$. Weights are assigned **asymmetrically** to highlight critical corners: $w_c=\tfrac{1}{4}$ for $(0,\dots,0)$ and $(1,\dots,1)$, with the remaining half of the mass distributed equally among the other $2^M-2$ corners. This implementation satisfies both full support and corner prioritization.

**3. Dynamic Sharpness Adjustment via Imbalance Coefficients**: To identify and suppress dominant modalities during training, MIAM introduces two modality-specific factors: $\rho_{s_m}$ derived from the unimodal performance score $s_m$, and $\rho_{d_m}$ derived from the absolute derivative $d_m$ of $s_m$ (learning speed). Both are normalized by the geometric mean: $\rho_{s_m}=s_m/(\prod_{m'}s_{m'})^{1/M}$ and $\rho_{d_m}=d_m/(\prod_{m'}d_{m'})^{1/M}$. A high $\rho_{s_m}/\rho_{d_m}$ ratio indicates a modality is "strong and stable" and should be masked more often. The beta sharpness is modified to $\kappa\cdot(\rho_{s_m}/\rho_{d_m})^{\pm\lambda}$ ($+\lambda$ for $c_m=1$, $-\lambda$ for $c_m=0$), where $\lambda>0$ controls the intensity of imbalance regulation. Dominant modalities are pushed toward high masking probabilities, forcing the model to optimize under-learned weak modalities. A key insight is that while $\rho_{s_m}$ is relatively stable, fluctuations in $\rho_{d_m}$ drive periodic shifts in training focus, which is beneficial for learning.

## Key Experimental Results
Two ecological benchmarks: **GeoPlant** (Species Distribution Modeling, 3 modalities: Tabular environment, Sentinel-2 imagery, Climate+Landsat time series, 1783 species, metric: AUC) and **TaxaBench** (Multimodal species classification, 5 modalities: Ground images, Satellite images, Audio, Environmental tables, Geolocation, 199 species, metric: Top-1). All methods follow the same training protocol, differing only in masking strategy. Models are evaluated on all input subsets without retraining.

### Main Results (Partial Subsets + Average)
GeoPlant (AUC %):

| Masking Strategy | Partial Unimodal (1st Col) | Avg. |
|---|---|---|
| Constant | 68.6 | 80.4 |
| Uniform | 73.3 | 83.2 |
| Dirichlet | 65.1 | 80.6 |
| Modality dropout | 48.7 | 81.5 |
| OPM | 68.0 | 83.8 |
| **MIAM (Ours)** | **78.4** | **86.1** |
| Oracle (Individual models) | 78.0 | 87.2 |

TaxaBench (Top-1 %):

| Masking Strategy | Avg. |
|---|---|
| Uniform | 37.7 |
| Dirichlet | 37.4 |
| Modality dropout | 35.9 |
| OPM | 31.2 |
| **MIAM (Ours)** | **38.7** |
| Oracle | 40.0 |

MIAM outperforms the second-best method by approximately **2.3%** (GeoPlant) on average and stays close to the "oracle" trained on individual subsets. OPM fails significantly on unseen partial unimodal subsets.

### Ablation Study
Ablation follows the three principles (validated on GeoPlant): Uniform $\rightarrow$ Uniform hypercube (Full support) $\rightarrow$ Beta hypercube (Corner prioritized) $\rightarrow$ MIAM (Imbalance aware). Each principle added continuously improves performance on weak modalities (especially satellite imagery) while maintaining performance on dominant time series and "full modality" scenarios.

Effect of asymmetric corner weights $w_c$ (Validation set):

| | GeoPlant (AUC) | TaxaBench (Top-1) |
|---|---|---|
| Uniform $w_c$ | 85.2 | 36.0 |
| Non-uniform $w_c$ (Critical corners) | 85.4 | **37.1** |

### Key Findings
- MIAM shows the largest improvement on satellite imagery, the modality most suppressed by imbalance, nearly closing the gap with the unimodal oracle.
- Fluctuations in $\rho_{d_m}$ create periodic training focus shifts (similar to cyclic learning rates), providing a key advantage over the static OPM.
- Fine-grained contribution analysis reveals ecological signals: Red+NIR bands are most important in satellite imagery (used for NDVI calculations); extending the time series history captures signals from past extreme events (e.g., the 2003 European heatwave).

## Highlights & Insights
- **Unified Perspective**: Standardizes disparate masking strategies as "probability distributions on a hypercube" and uses actionable principles to diagnose existing flaws.
- **Three-in-One**: A single masking distribution resolves missingness robustness, modality imbalance, and cross/intra-modality contribution analysis without extra modules.
- **Learning Speed Signal**: Utilizing the derivative $d_m$ instead of just performance $s_m$ identifies dominant modalities based on "still learning vs. already mastered" dynamics, which is more reasonable than static scoring.
- **Explainable Ecological Value**: Fine-grained tokenization allows the model to pinpoint specific bands, years, or patches, producing actual ecological insights (NDVI, heatwaves) rather than just accuracy metrics.

## Limitations & Future Work
- MIAM slightly underperforms modality dropout/uniform on the "all modalities available" subset, necessitating a tradeoff via $\lambda$; hyperparameter tuning ($\lambda, \kappa$) is dataset-dependent.
- The hybrid product-beta involves $2^M$ corner components; as $M$ increases, the number of corners grows exponentially, and sampling costs require further discussion.
- Validation is limited to ecological benchmarks; transferability to general multimodal scenarios (large-scale video/text/audio) remains to be tested.
- Evaluating $s_m$ and $d_m$ for every modality on the validation set each epoch introduces additional overhead.

## Related Work & Insights
- **Modality Imbalance**: Gradient Blending (Wang 2020), OGM (Peng 2022), Learning speed scheduling (Wu 2022), Unimodal teacher distillation (Du 2021)—most require extra components and complete inputs. MIAM uses masking as a minimal mechanism to handle both missingness and imbalance.
- **Masking/Self-Supervised**: MAE/BERT, MultiMAE, 4M (cross/intra-modality reconstruction), Covert 2023 (masking to estimate patch Shapley); in ecology, MaskSDM uses uniform masking while OPM adjusts dropout by performance. MIAM is a refinement of this line regarding distribution shape and dynamic imbalance awareness.
- **Inspiration**: Treating random augmentation/masking as a "learnable probability distribution" and using learning dynamics as feedback is a methodology applicable to any multimodal system with modality/feature competition.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Unifying masking as a hypercube distribution with product-beta corner priors and learning-speed-driven imbalance awareness is an elegant and clear mechanism.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Two real ecological benchmarks with 3/5 modalities compared against 5 baselines and an oracle; includes progressive ablation and contribution analysis, though limited to the ecological domain.
- **Writing Quality**: ⭐⭐⭐⭐ Logical progression from motivation (modality imbalance) to derivation (three principles $\rightarrow$ formulas) with intuitive illustrations.
- **Value**: ⭐⭐⭐⭐ Highly practical for ecological/scientific multimodal applications where missingness is common and interpretability is required; provides a transferable masking design methodology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Label What Matters: Modality-Balanced and Difficulty-Aware Multimodal Active Learning](../../CVPR2026/multimodal_vlm/label_what_matters_modality-balanced_and_difficulty-aware_multimodal_active_lear.md)
- [\[ICLR 2026\] ScaleCap: Scalable Image Captioning via Dual-Modality Debiasing](scalecap_scalable_image_captioning_via_dual-modality_debiasing.md)
- [\[ICLR 2026\] Modality Alignment across Trees on Heterogeneous Hyperbolic Manifolds](modality_alignment_across_trees_on_heterogeneous_hyperbolic_manifolds.md)
- [\[ICLR 2026\] RAG4DMC: Retrieval-Augmented Generation for Data-Level Modality Completion](rag4dmc_retrieval-augmented_generation_for_data-level_modality_completion.md)
- [\[CVPR 2026\] Enhance-then-Balance Modality Collaboration for Robust Multimodal Sentiment Analysis](../../CVPR2026/multimodal_vlm/enhance-then-balance_modality_collaboration_for_robust_multimodal_sentiment_anal.md)

</div>

<!-- RELATED:END -->
