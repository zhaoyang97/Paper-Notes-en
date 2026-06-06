---
title: >-
  [Paper Note] Time-series Forecasting Through the Lens of Dynamics
description: >-
  [ICML 2026][Time Series][Dynamical Systems Perspective] The authors propose the PRO-DYN nomenclature based on Allen’s Interval Algebra, decomposing any time-series forecasting (TSF) model into a "Preprocessing (PRO) → Dy…
tags:
  - "ICML 2026"
  - "Time Series"
  - "Dynamical Systems Perspective"
  - "PRO-DYN Nomenclature"
  - "LTSF-Linear"
  - "Predictor Placement"
  - "TSF Model Design Principles"
date: 2026-05-08
content_hash: 198d7d6fb2f14426
---

# Time-series Forecasting Through the Lens of Dynamics

**Conference**: ICML 2026  
**arXiv**: [2507.15774](https://arxiv.org/abs/2507.15774)  
**Code**: None  
**Area**: Time-series Forecasting / Model Analysis  
**Keywords**: Dynamical Systems Perspective, PRO-DYN Nomenclature, LTSF-Linear, Predictor Placement, TSF Model Design Principles

## TL;DR
The authors propose the PRO-DYN nomenclature based on Allen’s Interval Algebra, decomposing any time-series forecasting (TSF) model into a "Preprocessing (PRO) → Dynamics (DYN) → Postprocessing (PRO)" sequence. Two empirical laws are identified: (i) DYN must be **learnable and complete** to outperform LTSF-Linear, and (ii) DYN must be placed at the **end of the pipeline** (PRE-DYN configuration) to leverage long lookback windows. These laws are validated by adding a linear DYN layer to Informer/FEDformer/MICN/FiLM (yielding consistent gains) and moving DYN to the front in iTransformer/PatchTST/Crossformer (causing performance degradation).

## Background & Motivation
**Background**: TSF is dominated by Transformer-based models (Informer, FEDformer, PatchTST, iTransformer, etc.). However, since 2023, "shallow" baselines like LTSF-Linear and FITS, which consist almost entirely of a single linear mapping, have outperformed many complex deep models. Recently, top-tier models like iTransformer and PatchTST have surpassed NLinear again, leaving the relationship between complexity and performance ambiguous.

**Limitations of Prior Work**: There is a lack of a unified perspective to explain why some Transformers fail while others succeed. Zeng et al. (2023) attributed failures to the attention mechanism, but PatchTST and iTransformer use attention and perform well. Ke et al. (2025) analyzed attention without explaining successful cases. Each paper argues for its own modifications, but the field lacks a "model anatomy."

**Key Challenge**: The essence of time-series generation is a **dynamical system**—data evolves according to an evolution law $x(t_n) = F(x(t_{n-1}), \dots, x(t_{n-K}))$. When porting text models to TSF, the critical question is: does the model truly learn $F$? If prediction relies on zero-padding or non-learnable functions (e.g., the decoder initialization in Informer/FEDformer), it is not "learning dynamics," and thus cannot beat LTSF-Linear which explicitly learns linear mappings.

**Goal**: (i) Establish a language for how models process time to enable structural analysis of any TSF model; (ii) Identify key features that distinguish superior models; (iii) Verify the causal role of these features through minimal interventions (adding a linear DYN layer); (iv) Provide plug-and-play design principles for future TSF models.

**Key Insight**: Starting from the algebraic relations of time intervals, Allen (1983) categorized the relations between two intervals $T_E$ and $T_F$ into 13 basic relations. A function $f$ is classified by the relation between its input and output intervals: if the output interval remains within the input (contains/equals, etc.), $f$ is PRO (Pre/Post-processing); if the output interval moves into the future (starts/overlaps/meets/before), $f$ is DYN (Dynamics). This provides a unified analytical "scalpel."

**Core Idea**: Model predictive capability is formalized as a function of "DYN completeness + DYN placement within the pipeline," anchored by the theoretical interpretation of LTSF-Linear as a "relaxed version of a linear time-delay dynamical system."

## Method

### Overall Architecture
For any TSF model $M_\theta$, the input $X \in \mathbb R^{L\times D}$ in history interval $T_X$ and output $\hat Y \in \mathbb R^{H\times D}$ in future interval $T_Y$ can be decomposed as:
$$M_\theta: X \xrightarrow{f^{\text{pre}}_{\theta_{\text{pre}}}} X_{\text{pre}} \xrightarrow{f^{\text{dyn}}_{\theta_{\text{dyn}}}} \tilde Y \xrightarrow{f^{\text{post}}_{\theta_{\text{post}}}} \hat Y$$
Where $f^{\text{dyn}}$ is the DYN function (orange) responsible for projecting time from $T_X$ to $T_Y$, and $f^{\text{pre}}, f^{\text{post}}$ are PRO functions (blue) that perform feature extraction or up/down-sampling within the input time interval. Reversible normalization is excluded from the nomenclature. The authors classify 16 models based on "DYN completeness" and "PRO configuration" and find patterns aligned with TFB benchmark rankings.

### Key Designs

1.  **PRO-DYN Three-Stage Nomenclature (Unified Scalpel)**:
    - **Function**: Quantifies the time-processing structure of any TSF model into distinct features.
    - **Mechanism**: Based on Allen's interval algebra. PRO functions $f$ satisfy $T_E$ contains/started by/finished by/equals $T_F$ (no forward time progression). DYN functions satisfy $T_E$ starts/overlaps/meets/before $T_F$ (forward progression). Models are labeled as PRE-DYN (preprocessing + terminal dynamics, e.g., iTransformer), DYN-POST (initial dynamics + postprocessing), PRE-DYN-POST (dynamics sandwiched between processing, e.g., Informer), or DYN (single dynamics layer, e.g., NLinear).
    - **Design Motivation**: High-ranking models (denoted as "↑" in TFB benchmark) are almost exclusively PRE-DYN with complete learnable DYN, while low-ranking models ("↓") are often PRE-DYN-POST with non-learnable components.

2.  **LTSF-Linear as a Theoretical Anchor**:
    - **Function**: Explains why simple models beat complex Transformers and provides a reference for DYN design.
    - **Mechanism**: Assuming true dynamics satisfy $[x(t_n), \dots, x(t_{n-L+1})]^T = M[x(t_{n-1}), \dots, x(t_{n-L})]^T$, then $Y = (M^H)_{-H:,:} X$. The prediction layer of LTSF-Linear $\hat Y = W_\theta X_g + b_\theta$ matches the dimension of $(M^H)_{-H:,:}$, acting as "relaxed dynamical system identification."
    - **Design Motivation**: Connects the empirical "shallow model win" to 50 years of dynamical systems theory, guiding the minimal intervention of adding a linear DYN layer.

3.  **Learnable Linear DYN Injection Experiments (RQ1 & RQ2)**:
    - **Function**: Causally validates the PRO-DYN laws via minimal invasive surgery.
    - **Mechanism**: RQ1 (Adding DYN)—Adds a linear DYN layer to replace zero-padding/mean initializations in Informer, FEDformer, MICN, and FiLM without changing other hyperparameters. RQ2 (Moving DYN)—Converts PRE-DYN models (iTransformer, PatchTST, Crossformer) into DYN-POST by adding a DYN layer at the front. A PRO control group (using feed-forward layers without time mapping) isolates the effect of increased parameters.
    - **Design Motivation**: Correlation is insufficient; counterfactual interventions across 25 datasets, 4 horizons, and 7 models with Wilcoxon tests ensure the reliability of the "DYN position" law.

### Loss & Training
All training protocols follow the TFB benchmark. Only learning rate, epochs, and patience are manually tuned; architectural hyperparameters remain identical to the original models. Evaluation uses MSE and MAE. Statistical significance is determined via one-sided Wilcoxon tests ($p < 0.05$).

## Key Experimental Results

### Main Results
Classification of 16 TSF models by PRO-DYN nomenclature (Selection):

| Model | TFB Rank | Complete Learnable DYN | PRO-DYN Config | DYN Function |
|---|---|---|---|---|
| iTransformer | ↑ (Beats NLinear) | ✓ | PRE-DYN | Linear |
| PatchTST | ↑ | ✓ | PRE-DYN | Linear |
| Crossformer | ↑ | ✓ | PRE-DYN | Linear |
| NLinear | Baseline | ✓ | DYN | Linear |
| FITS | ↓ (Loses to NLinear) | ✗ | PRE-DYN | Linear+0-pad |
| FEDformer | ↓ | ✗ | PRE-DYN-POST | Mean+0-pad |
| MICN | ↓ | ✗ | PRE-DYN-POST | Linear+0-pad |
| FiLM | ↓ | ✗ | PRE-DYN-POST | Legendre disc. |
| Informer | ↓ | ✗ | PRE-DYN-POST | 0-padding |

RQ1 Results: Average normalized MSE/MAE scores after adding a linear DYN layer to "underperforming" models (closer to 0 is better):

| Model | DYN added | Vanilla |
|---|---|---|
| Informer | **−0.228** | −0.333 |
| FiLM | **0.006** | −0.036 |
| MICN | **−0.164** | −0.176 |
| FEDformer | **−0.360** | −0.398 |

Informer showed the most significant improvement as its dynamics were virtually non-existent in the vanilla version.

### Ablation Study

| Config | Key Findings | Implication |
|---|---|---|
| RQ1 vanilla → +DYN | Better/equal in 80%+ scenarios across 4 models | Complete learnable DYN is critical |
| RQ1 +DYN vs +PRO | +DYN significantly outperforms +PRO | Gains come from time progression, not parameters |
| RQ2 PatchTST/Crossformer PRE-DYN → DYN-POST | Significant performance drop | Terminal DYN position is essential |
| RQ2 iTransformer PRE-DYN → DYN-POST | Only one metric dropped; 51% scenarios tied | iTransformer treats time as latent, less position-sensitive |
| Data Length Analysis ($H>L$ vs $H<L$) | DYN gain is greater in long horizon scenarios | Complete dynamics help leverage long lookbacks |

### Key Findings
- Performance groupings almost perfectly overlap with PRO-DYN features (with Triformer being the exception), indicating the nomenclature captures dominant factors.
- DYN must be at the end: PRE blocks "linearize" history variables, allowing the final linear DYN to perform system identification directly. Front-loading DYN forces prediction before PRE learning is stable.
- Benefits are independent of parameter count; the "forward time progression" action itself is the key.

## Highlights & Insights
- **Paradigm-level "Scalpel"**: Introducing Allen's interval algebra to TSF analysis is an elegant cross-disciplinary approach that operates at a higher abstraction level than discussing attention vs. MLPs.
- **Physical Interpretation of Shallow Models**: The observation that LTSF-Linear $\approx$ linear time-delay dynamical system provides a theoretical bridge between deep learning and classical system identification.
- **Transferable Guidance**: For any new TSF model, one should ask: (i) Is DYN complete and learnable? (ii) Is DYN at the end? These two rules provide "free" performance boosts.

## Limitations & Future Work
- The nomenclature primarily covers Family 1 (pattern-recognition models); foundation models and pure dynamics-based models (Family 2/3) are reserved for future work.
- Only **linear** DYN is studied; whether non-linear dynamics (Neural ODE, Koopman) share the same position sensitivity remains unknown.
- Evaluation metrics are limited to MSE/MAE, without analysis of long-term distribution shifts or probabilistic forecasting.
- Future direction: Researching robust combinations of "Linear DYN + Non-linear PRO" vs. "Non-linear DYN + Linear PRO."

## Related Work & Insights
- **vs Zeng et al. (2023)**: They blamed failures on the attention mechanism; this work proves the issue lies in whether DYN is complete and placed at the end, not the attention block itself.
- **vs Ke et al. (2025)**: While they analyzed attention failures, this work provides a constructive solution—what DYN to add to make attention models perform better.
- **vs Koopa/Attraos (Family 3)**: These explicitly model dynamics; this work is more lightweight, requiring only a structural structure ("Linear DYN at the end") rather than physical operator identification.
- **vs FITS/LTSF-Linear**: This work generalizes the single-baseline success of LTSF-Linear into a design principle for all TSF models.

## Rating
- Novelty: ⭐⭐⭐⭐ (Elegant use of interval algebra + dynamical system perspective).
- Experimental Thoroughness: ⭐⭐⭐⭐ (Large-scale categorization and modification across 25 datasets).
- Writing Quality: ⭐⭐⭐⭐ (Clear definitions and rigorous logic).
- Value: ⭐⭐⭐⭐ (Provides a "model anatomy" language for the entire field).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Towards Generalizable PDE Dynamics Forecasting via Physics-Guided Invariant Learning](../../ICLR2026/time_series/towards_generalizable_pde_dynamics_forecasting_via_physics-guided_invariant_lear.md)
- [\[NeurIPS 2025\] IonCast: A Deep Learning Framework for Forecasting Ionospheric Dynamics](../../NeurIPS2025/time_series/ioncast_a_deep_learning_framework_for_forecasting_ionospheric_total_electron_con.md)
- [\[ICML 2026\] Ellipsoidal Time Series Forecasting](ellipsoidal_time_series_forecasting.md)
- [\[ICML 2026\] From Observations to States: Latent Time Series Forecasting](from_observations_to_states_latent_time_series_forecasting.md)
- [\[ICML 2026\] Nested Spatio-Temporal Time Series Forecasting](nested_spatio-temporal_time_series_forecasting.md)

</div>

<!-- RELATED:END -->
