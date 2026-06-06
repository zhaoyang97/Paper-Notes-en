---
title: >-
  [Paper Note] Time-series Forecasting Through the Lens of Dynamics
description: >-
  [ICML 2026][Time Series][Dynamical Systems Perspective] The authors propose the PRO-DYN nomenclature using Allen's interval algebra, decomposing any time-series forecasting (TSF) model into "Pre-processing PRO → Dynamics…
tags:
  - "ICML 2026"
  - "Time Series"
  - "Dynamical Systems Perspective"
  - "PRO-DYN Nomenclature"
  - "LTSF-Linear"
  - "Predictor Position"
  - "Temporal Model Design Principles"
date: 2026-05-08
content_hash: 6d6ff83c8468caf8
---

# Time-series Forecasting Through the Lens of Dynamics

**Conference**: ICML 2026  
**arXiv**: [2507.15774](https://arxiv.org/abs/2507.15774)  
**Code**: None  
**Area**: Time-series Forecasting / Model Analysis  
**Keywords**: Dynamical Systems Perspective, PRO-DYN Nomenclature, LTSF-Linear, Predictor Position, Temporal Model Design Principles

## TL;DR
The authors propose the PRO-DYN nomenclature using Allen's interval algebra, decomposing any time-series forecasting (TSF) model into "Pre-processing PRO → Dynamics DYN → Post-processing PRO" three stages. They discover two empirical rules: (i) DYN must be **learnable and complete** to outperform LTSF-Linear, (ii) DYN must be placed at the **very end of the pipeline** (PRE-DYN configuration) to fully leverage long lookback benefits. By adding a linear DYN layer to Informer/FEDformer/MICN/FiLM, performance consistently improves; moving DYN to the front in iTransformer/PatchTST/Crossformer degrades performance, experimentally validating both rules.

## Background & Motivation
**Background**: Time-series forecasting is dominated by Transformer-based models (Informer, FEDformer, PatchTST, iTransformer, etc.), but since 2023, "shallow" baselines like LTSF-Linear and FITS, which are almost just a single linear mapping, have outperformed many complex deep models. Recently, top models like iTransformer and PatchTST have again surpassed NLinear, making the relationship between complexity and performance elusive.

**Limitations of Prior Work**: There is a lack of a unified perspective to explain "why some Transformers fail while others succeed." Zeng et al. (2023) attribute failures to the attention mechanism, but PatchTST and iTransformer also use attention and perform well; Ke et al. (2025) only analyze attention, without explaining successful cases. Each paper justifies its own tweaks, but the field lacks a "model anatomy."

**Key Challenge**: The essence of time-series generation is a **dynamical system**—data evolves according to some law $x(t_n) = F(x(t_{n-1}), \dots, x(t_{n-K}))$. When text models are directly transferred, the key question is: does the model actually learn $F$? If prediction relies on zero-padding or non-learnable functions (e.g., decoder initialization in Informer/FEDformer), it is not "learning dynamics," and thus cannot outperform LTSF-Linear, which explicitly learns a linear mapping.

**Goal**: (i) Establish a language for "how models handle time," enabling structured analysis of any TSF model; (ii) Use this language to identify key features distinguishing good and bad models; (iii) Experimentally validate the causal role of these features via minimal interventions (adding a linear DYN layer without changing original architecture hyperparameters); (iv) Provide plug-and-play design principles for future TSF models.

**Key Insight**: Start from the algebraic relationships of time intervals—Allen (1983) classifies the relationship between two intervals $T_E, T_F$ into 13 basic types. A function $f$ is classified by the relationship between its input and output intervals: if the output interval remains within the input interval (contains/equals, etc.), $f$ is PRO (pre/post-processing); if the output interval moves into the future (starts/overlaps/meets/before), $f$ is DYN (dynamics). This provides a unified "scalpel."

**Core Idea**: Formalize "model predictive capability" as two factors: "completeness of the DYN function" and "position of DYN in the pipeline," anchored by LTSF-Linear as a "relaxed version of linear time-delay dynamical systems."

## Method

### Overall Architecture
For any TSF model $M_\theta$, input $X \in \mathbb R^{L\times D}$ over history interval $T_X$, output $\hat Y \in \mathbb R^{H\times D}$ over future interval $T_Y$, can be decomposed as:
$$M_\theta: X \xrightarrow{f^{\text{pre}}_{\theta_{\text{pre}}}} X_{\text{pre}} \xrightarrow{f^{\text{dyn}}_{\theta_{\text{dyn}}}} \tilde Y \xrightarrow{f^{\text{post}}_{\theta_{\text{post}}}} \hat Y$$
Here, $f^{\text{dyn}}$ is the DYN function (orange, solely responsible for "future prediction") that advances time from $T_X$ to $T_Y$; $f^{\text{pre}}, f^{\text{post}}$ are PRO functions (blue, operating within the input interval for feature extraction/upsampling/downsampling). Invertible normalization, etc., are excluded from the nomenclature. The authors classify 16 models by "whether DYN is complete and learnable + PRO configuration," and compare with TFB benchmark multivariate TSF rankings to identify patterns.

### Key Designs

1. **PRO-DYN Three-stage Nomenclature (Unified Scalpel)**:

    - **Function**: Concisely characterizes the temporal processing structure of any TSF model, making "why this model works/doesn't work" a quantifiable feature.
    - **Mechanism**: Based on Allen's interval algebra. PRO functions $f$ satisfy $T_E$ contains/started by/finished by/equals $T_F$, i.e., do not advance time; DYN functions satisfy $T_E$ starts/overlaps/meets/before $T_F$, i.e., advance into the future. Each TSF model is labeled as PRE-DYN (pre-processing + terminal dynamics, e.g., iTransformer/PatchTST), DYN-POST (initial dynamics + post-processing), PRE-DYN-POST (both pre- and post-processing, DYN in the middle, e.g., Informer/FEDformer), or DYN (single dynamics layer, e.g., NLinear).
    - **Design Motivation**: The authors find that TFB benchmark rankings split into two groups—"↑" group (outperforming NLinear) are almost all PRE-DYN with complete, learnable DYN; "↓" group (worse than NLinear) are mostly PRE-DYN-POST with partially non-learnable DYN (e.g., FEDformer uses mean+0-padding for decoder initialization). These two features directly predict performance grouping.

2. **LTSF-Linear as the Theoretical Anchor of "Relaxed Linear Time-delay Dynamical Systems"**:

    - **Function**: Explains why such a simple model can outperform complex Transformers and provides principled guidance for DYN design.
    - **Mechanism**: Assume the true dynamics satisfy $[x(t_n), \dots, x(t_{n-L+1})]^T = M[x(t_{n-1}), \dots, x(t_{n-L})]^T$, then $Y = (M^H)_{-H:,:} X$. The prediction layer of LTSF-Linear, $\hat Y = W_\theta X_g + b_\theta$, matches the dimensions of $(M^H)_{-H:,:}$—it can be seen as "relaxed system identification without coefficient constraints." LTSF-Linear wins because it **actually learns a linear dynamic**, whereas Informer/FEDformer, etc., let the decoder predict the future from zero-padding.
    - **Design Motivation**: Links the empirical rule "shallow models can win" to 50 years of dynamical systems theory, providing engineering guidance: "adding a linear DYN layer can boost performance."

3. **Injecting Learnable Linear DYN Experiments (RQ1 & RQ2)**:

    - **Function**: Causally validates the two rules of the PRO-DYN nomenclature via minimal intervention.
    - **Mechanism**: RQ1 (completing DYN)—for models with incomplete or non-learnable dynamics (Informer, FEDformer, MICN, FiLM), add a linear DYN layer to replace original zero-padding/mean initialization (keeping other hyperparameters unchanged) and observe if performance improves; RQ2 (moving DYN position)—for PRE-DYN models (iTransformer, PatchTST, Crossformer), add a linear DYN layer at the front to make them DYN-POST (original terminal linear becomes PRO), and observe if performance drops. A PRO control group (same parameter count, but feed-forward instead of temporal mapping) isolates the effect of "increased parameters."
    - **Design Motivation**: Observing correlation is insufficient; counterfactual intervention is necessary. On 25 datasets × 4 horizons × 7 models, 200 scores per model are collected, with Wilcoxon tests for statistical significance, making the "DYN position matters" rule truly credible.

### Loss & Training
All training protocols follow the TFB benchmark, with only manual tuning of learning rate / epoch / patience; architecture hyperparameters remain unchanged (ensuring the only change is "adding a DYN layer" rather than "changing hyperparameters"). Evaluation uses MSE and MAE. Statistical significance is assessed with one-sided Wilcoxon test ($p < 0.05$).

## Key Experimental Results

### Main Results
Classification of 16 TSF models by PRO-DYN nomenclature (excerpt):

| Model | TFB Rank | Complete Learnable DYN | PRO-DYN Config | DYN Function |
|---|---|---|---|---|
| iTransformer | ↑ (beats NLinear) | ✓ | PRE-DYN | Linear |
| PatchTST | ↑ | ✓ | PRE-DYN | Linear |
| Crossformer | ↑ | ✓ | PRE-DYN | Linear |
| NLinear | Baseline | ✓ | DYN | Linear |
| FITS | ↓ (loses to NLinear) | ✗ | PRE-DYN | Linear+0-pad |
| FEDformer | ↓ | ✗ | PRE-DYN-POST | Mean+0-pad |
| MICN | ↓ | ✗ | PRE-DYN-POST | Linear+0-pad |
| FiLM | ↓ | ✗ | PRE-DYN-POST | Legendre disc. |
| Informer | ↓ | ✗ | PRE-DYN-POST | 0-padding |

RQ1 experiment: After adding a linear DYN layer to four "underperformers," normalized MSE/MAE average scores vs. NLinear (closer to 0 is better):

| Model | DYN added | Vanilla |
|---|---|---|
| Informer | **−0.228** | −0.333 |
| FiLM | **0.006** | −0.036 |
| MICN | **−0.164** | −0.176 |
| FEDformer | **−0.360** | −0.398 |

FiLM DYN even slightly outperforms NLinear; Informer shows the largest improvement (dynamics almost from none to present).

### Ablation Study

| Configuration | Key Findings | Meaning |
|---|---|---|
| RQ1 vanilla → +DYN | In 80%+ scenarios, 4 models match or outperform, statistically significant | Completing learnable DYN indeed improves performance |
| RQ1 +DYN vs +PRO (same parameter count, non-temporal mapping) | +DYN significantly better than +PRO | Gains come from "learning dynamics," not just "adding parameters" |
| RQ2 PatchTST/Crossformer PRE-DYN → DYN-POST | Both significantly worse | Terminal PRE-DYN position is irreplaceable |
| RQ2 iTransformer PRE-DYN → DYN-POST | Only one metric significantly worse, 51% scenarios unchanged | iTransformer treats time as latent, less sensitive to position |
| Data length scenario analysis ($H>L$ vs $H<L$) | DYN boost is greater in long horizon scenarios | Complete dynamics help models fully utilize long lookback |

### Key Findings
- Performance grouping almost perfectly matches PRO-DYN features (only Triformer is an exception), indicating the nomenclature captures the dominant factors.
- DYN must be at the end: because the PRE block "linearizes" historical variables, enabling the final linear DYN to perform system identification directly; if DYN is at the front, prediction occurs before PRE is learned, and is then disturbed by subsequent nonlinear PRO.
- The gain from adding DYN does not come from parameter count; the PRO control group shows no gain, indicating that "advancing time" itself is key.

## Highlights & Insights
- **Paradigm-level "scalpel"**: Introducing Allen's interval algebra into TSF model analysis is an elegant interdisciplinary approach, a level above simply debating "whether attention is needed." This nomenclature can directly analyze any future TSF model and is a lasting contribution.
- **Physical explanation for "why shallow models win"**: The observation that LTSF-Linear ≈ linear time-delay dynamical system is profound, reconnecting deep learning empirical phenomena to 50 years of system identification theory.
- **Transferable engineering guidance**: For any new TSF model design, ask two questions: (i) Is DYN complete and learnable? (ii) Is DYN at the end? These two almost cost-free checks can improve design. Also insightful for multimodal/video forecasting and neural ODE work.

## Limitations & Future Work
- The nomenclature only covers family 1 (pattern-recognition models); foundation models (Chronos/Toto) and dynamics-based models (Koopa/Attraos) are left to the appendix and future work, so conclusions cannot be directly generalized.
- Only **linear** DYN is studied; whether nonlinear dynamics (Koopman, Neural ODE, neural operators) exhibit similar position sensitivity remains unknown.
- Evaluation metrics are limited to MSE/MAE; long-term distribution shift and probabilistic forecasting are not analyzed.
- Future directions: extend the nomenclature to families 2/3, study which combination—"linear DYN + nonlinear PRO" or "nonlinear DYN + linear PRO"—is more robust; use PRO-DYN to design a new generation of lightweight TSF models, treating end-to-end DYN as an inductive bias.

## Related Work & Insights
- **vs Zeng et al. (2023) "Are Transformers Effective?"**: They attribute deep TSF model failures to the attention mechanism; this work shows the issue is not attention itself (PatchTST/iTransformer both use attention and perform well), but whether DYN is complete, learnable, and at the end.
- **vs Ke et al. (2025)**: That work only analyzes attention failures; this work provides a constructive approach—what DYN to add to improve attention models.
- **vs Koopa/Attraos (family 3)**: Those explicitly model dynamics (e.g., Koopman operators); this work does not require physical system identification, only the structural feature of "linear DYN block at the end," making it lighter and more general.
- **vs FITS/LTSF-Linear**: LTSF-Linear is used directly as DYN; this work upgrades it to "every PRE-DYN model should have such a layer at the end," generalizing a single baseline into a design principle.

## Rating
- Novelty: ⭐⭐⭐⭐ Uses Allen's interval algebra + LTSF-Linear as a relaxed dynamical system perspective, providing a new scalpel for the field.
- Experimental Thoroughness: ⭐⭐⭐⭐ 16 model classification + 7 model modifications + 25 datasets × 4 horizons × multiple statistical tests, high coverage; lacks nonlinear DYN and family 2/3 extension experiments.
- Writing Quality: ⭐⭐⭐⭐ Clear concepts, rich visualizations, exemplary logic chain of definition–observation–hypothesis–validation.
- Value: ⭐⭐⭐⭐ Provides a "model anatomy" language for the field; future TSF model designers will benefit.

## Related Papers

- [\[ICLR 2026\] Towards Generalizable PDE Dynamics Forecasting via Physics-Guided Invariant Learning](../../ICLR2026/time_series/towards_generalizable_pde_dynamics_forecasting_via_physics-guided_invariant_lear.md)
- [\[NeurIPS 2025\] IonCast: A Deep Learning Framework for Forecasting Ionospheric Dynamics](../../NeurIPS2025/time_series/ioncast_a_deep_learning_framework_for_forecasting_ionospheric_total_electron_con.md)
- [\[ICML 2026\] CombinationTS: A Modular Framework for Understanding Time-Series Forecasting Models](combinationts_a_modular_framework_for_understanding_time-series_forecasting_mode.md)
- [\[ICML 2026\] DAG: A Dual Correlation Network for Time Series Forecasting with Exogenous Variables](dag_a_dual_correlation_network_for_time_series_forecasting_with_exogenous_variab.md)
- [\[NeurIPS 2025\] Human-Machine Ritual: Synergic Performance through Real-Time Motion Recognition](../../NeurIPS2025/time_series/human-machine_ritual_synergic_performance_through_real-time_motion_recognition.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Towards Generalizable PDE Dynamics Forecasting via Physics-Guided Invariant Learning](../../ICLR2026/time_series/towards_generalizable_pde_dynamics_forecasting_via_physics-guided_invariant_lear.md)
- [\[NeurIPS 2025\] IonCast: A Deep Learning Framework for Forecasting Ionospheric Dynamics](../../NeurIPS2025/time_series/ioncast_a_deep_learning_framework_for_forecasting_ionospheric_total_electron_con.md)
- [\[ICML 2026\] Ellipsoidal Time Series Forecasting](ellipsoidal_time_series_forecasting.md)
- [\[ICML 2026\] From Observations to States: Latent Time Series Forecasting](from_observations_to_states_latent_time_series_forecasting.md)
- [\[ICML 2026\] CombinationTS: A Modular Framework for Understanding Time-Series Forecasting Models](combinationts_a_modular_framework_for_understanding_time-series_forecasting_mode.md)

</div>

<!-- RELATED:END -->
