---
title: >-
  [Paper Note] From Parameter Dynamics to Risk Scoring: Quantifying Sample-Level Safety Degradation in LLM Fine-tuning
description: >-
  [ICML 2026][LLM/NLP][Safety alignment] The authors track the cumulative drift of parameters along "dangerous/safe directions" during LoRA fine-tuning. They discover that the fundamental mechanism through which benign dat…
tags:
  - "ICML 2026"
  - "LLM/NLP"
  - "Safety alignment"
  - "fine-tuning risk"
  - "parameter dynamics"
  - "task vectors"
  - "sample scoring"
date: 2026-05-08
content_hash: cdf83caf70bc7349
---

# From Parameter Dynamics to Risk Scoring: Quantifying Sample-Level Safety Degradation in LLM Fine-tuning

**Conference**: ICML 2026  
**arXiv**: [2605.04572](https://arxiv.org/abs/2605.04572)  
**Code**: https://github.com/(repo) (Available)  
**Area**: LLM Alignment / Safety / Model Compression  
**Keywords**: Safety alignment, fine-tuning risk, parameter dynamics, task vectors, sample scoring

## TL;DR
The authors track the cumulative drift of parameters along "dangerous/safe directions" during LoRA fine-tuning. They discover that the fundamental mechanism through which benign data undermines alignment is the monotonic drift of parameters toward dangerous directions. Consequently, they propose SQSD—a method that assigns a continuous risk score to each sample based on the projection difference of single-step gradients along these two directions. SQSD maintains monotonic ASR rankings across 3 models and 2 datasets and demonstrates transferability across architectures, scales, and from LoRA to Full Fine-tuning.

## Background & Motivation

**Background**: LLMs undergo alignment (RLHF/DPO) before deployment to reject harmful requests. However, alignment proves surprisingly fragile during downstream fine-tuning; safety performance can collapse even when using as few as 100 **completely harmless** benign instruction samples. This "benign-sample-induced alignment breakage" is more dangerous than direct attacks using harmful samples because toxicity classifiers cannot filter them out.

**Limitations of Prior Work**: Prior research into the causes of this collapse follows two lines: (i) analyzing embedding drift (e.g., Vaccine), and (ii) analyzing static parameter perturbations (e.g., Booster, PEFT safety analysis). However, these studies only provide snapshots of the pre- and post-fine-tuning states without tracking the process. Furthermore, they focus on perturbation magnitude rather than direction, making it difficult to distinguish "safety-related drift" from "task-related drift." Sample selection methods (Bi-Anchor / Self-Inf-N / LARF) follow an "extreme sample selection" route, picking only the most dangerous subsets while failing for intermediate-risk samples, leading to a "boundary collapse" problem.

**Key Challenge**: Safety degradation is a **dynamic process** rather than a static perturbation, and it is strongly correlated with the "direction of parameter drift" rather than simple perturbation magnitude. Measuring only the magnitude mixes the effects of "task learning" and "safety destruction," making them inseparable.

**Goal**: (RQ1) Provide a mechanistic explanation for why benign fine-tuning breaks alignment; (RQ2) Within this mechanism, calculate a continuous and interpretable safety risk score for **every sample**.

**Key Insight**: Drawing on the concept of Task Vectors, the authors define two semantic anchor vectors: $V_\text{safety} = \hat\theta_\text{aligned} - \theta_0$ and $V_\text{danger} = \hat\theta_\text{harmful} - \theta_0$. They then track the cumulative drift $\Delta\theta_t = \theta_t - \theta_0$ at each step of fine-tuning along these two directions, directly mapping "safety degradation" to "parameter trajectories."

**Core Idea**: Directional projections in parameter space are used both to explain the mechanism (cumulative dangerous drift drives safety collapse) and to score samples (single-step gradient projection along the dangerous direction − projection along the safety direction = the sample's SQSD risk score).

## Method

### Overall Architecture
The method consists of two parts. **Part 1 (Mechanism Analysis)**: Construct two semantic anchor vectors $\rightarrow$ track cumulative drift over fine-tuning steps $\rightarrow$ observe the phenomenon where "monotonic increase in dangerous direction projection + near-zero safety direction projection" accompanies a Safety Score drop from 5.0 to below 1.0. **Part 2 (SQSD Sample Scoring)**: Compute the single-step LoRA gradient for each candidate sample $z \rightarrow$ estimate the equivalent weight update $\Delta W(z)$ caused by the sample $\rightarrow$ calculate the normalized projection difference of $\Delta W(z)$ relative to $V_\text{danger}$ and $V_\text{safety}$ for each LoRA module $\rightarrow$ sum across all modules to obtain the final $\text{SQSD}(z)$.

### Key Designs

1.  **Construction and Validation of "Safe / Dangerous" Directions in Parameter Space**:
    - **Function**: Defines two interpretable, projectable semantic anchor vectors in the parameter space.
    - **Mechanism**: $V_\text{safety} = \arg\min_\theta \mathcal{L}_\text{dpo}(\theta_0, D_\text{aligned}) - \theta_0$ (using DPO on PKU-SafeRLHF); $V_\text{danger} = \arg\min_\theta \mathcal{L}_\text{sft}(\theta_0, D_\text{harmful}) - \theta_0$ (using SFT on Aegis-unsafe and BeaverTails-unsafe to construct Aegis / Beaver dangerous directions). Steering experiments $\theta(\alpha) = \theta_0 + \alpha V$ confirm that the Safety Score decreases monotonically along $V_\text{danger}$ and increases monotonically along $V_\text{safety}$, validating that these vectors encode safety-related parameter displacements.
    - **Design Motivation**: Task Vectors in literature are primarily used for task arithmetic rather than safety analysis. Using the "direction = training endpoint − starting point" logic personifies abstract "safety" and "danger" as computable vectors, forming the basis for subsequent projection analysis.

2.  **Tracking Cumulative Parameter Drift Reveals Two-Stage Degradation**:
    - **Function**: Answers the mechanistic question of "why collapse occurs" using an interpretable trajectory map.
    - **Mechanism**: Calculating $p_\text{safety}(t) = \langle\Delta\theta_t, \hat{V}_\text{safety}\rangle$ and $p_\text{danger}(t) = \langle\Delta\theta_t, \hat{V}_\text{danger}\rangle$ at each checkpoint reveals a **distinct nonlinear two-stage pattern**: in the early stage, parameters drift rapidly along the dangerous direction (projection from 0 to $\sim 6.0$) while the Safety Score only drops slowly from 5.0 to 4.0; in the later stage, the drift slows, but the Safety Score suddenly collapses to below 1.0.
    - **Design Motivation**: This two-stage curve supports the geometric intuition of a "safety basin" (locally robust to perturbations, collapsing once outside) and guides the selection of initialization for SQSD—gradients must be computed at parameter states sensitive to directional signals.

3.  **SQSD: Single-Step Gradient Projection Difference as Sample Risk Score**:
    - **Function**: Provides a continuous risk score for every sample, covering the entire risk spectrum rather than just extreme subsets.
    - **Mechanism**: For a sample $z$, the single-step gradient of LoRA parameters is calculated to derive the equivalent weight update $\Delta W(z) \approx -\eta(B_0 \nabla_A + \nabla_B A_0)$ (ignoring $\mathcal{O}(\eta^2)$ terms). For each LoRA module $m$, the score is computed as $\text{SQSD}_m(z) = \langle\frac{\Delta W_m(z)}{\|\Delta W_m(z)\|_2}, \hat{V}_{\text{danger},m}\rangle - \langle\frac{\Delta W_m(z)}{\|\Delta W_m(z)\|_2}, \hat{V}_{\text{safety},m}\rangle$. The final score is $\text{SQSD}(z) = \sum_m \text{SQSD}_m(z)$. A first-order Taylor expansion proves that this projection difference corresponds to the relative degree to which a sample pushes the model toward $\theta_\text{danger}$ versus $\theta_\text{safety}$.
    - **Design Motivation**: Module-wise normalization eliminates "response length bias," as gradient magnitude is naturally correlated with response length. Without normalization, short-response samples would be incorrectly ranked. The authors also emphasize that initialization must be at a "direction-sensitive parameter state" to ensure valid SQSD rankings.

### Loss & Training
SQSD itself is not a training objective. However, all LoRA fine-tuning uses $r=8, \alpha=16$. For direction construction, lr=5e-6; for SQSD evaluation, lr=5e-5 to induce more pronounced safety degradation. The first-order Taylor derivation: $\eta[\mathcal{L}(z, \theta_\text{ref}) - \mathcal{L}(z, \theta_\text{target})] \approx (\theta' - \theta_\text{ref})^\top (\theta_\text{target} - \theta_\text{ref})$ associates the "gradient update direction vs. target direction" inner product with loss changes, providing a preference-based explanation for SQSD.

## Key Experimental Results

### Main Results
The study uses 3 models (Qwen3-8B / Llama-3.1-8B-Instruct / Llama-2-7B-Chat) × 2 datasets (Dolly / Alpaca). Each dataset is split into five subsets S1-S5 (1,000 samples each) based on SQSD ranking (from highest to lowest risk). Fine-tuning is performed on each to check if ASR decreases monotonically.

| Model + Data | S1 ASR | S5 ASR | Δ (S1-S5) | Monotonic? |
|-------------|--------|--------|-----------|-------|
| Qwen3-8B / Dolly / SQSD(Beaver) | 71.27 | 2.55 | **+68.72** | ✓ |
| Qwen3-8B / Alpaca / SQSD(Beaver) | 50.91 | 3.27 | +47.64 | ✓ |
| Llama3.1-8B / Dolly / SQSD(Beaver) | 79.82 | 4.73 | +75.09 | ✓ |
| Llama2-7B / Dolly / SQSD(Beaver) | 45.27 | 0.36 | +44.91 | ✓ |
| Reward Model baseline (Dolly avg) | 57.27 | 8.00 | 49.27 | ✗ |
| LARF baseline (Dolly avg) | 48.91 | 4.61 | 44.30 | ✗ |

SQSD maintains **strict monotonicity in 10/12 configurations** (while other baselines achieve at most 1/6), and the average Δ of 49.86% exceeds the strongest baseline (Reward Model at 43.76%). This indicates SQSD can correctly rank intermediate samples where baselines typically exhibit disordered rankings.

### Ablation Study

| Configuration | S1 ASR | Δ | Monotonic? |
|------|--------|---|-------|
| SQSD (full) | 71.27 | 68.72 | ✓ |
| w/o module-wise normalization | 13.09 | 12.54 | ✗ (dominated by length) |
| Danger direction only | 68.36 | 64.54 | ✗ (lacks safety constraint) |
| Safety direction only | 27.09 | 20.91 | ✗ (no danger signal) |
| Insensitive initialization | 38.36 | 37.27 | ✗ (failed projection signal) |

| Transfer Experiment | S1 | S5 | Monotonic? |
|----------|-----|-----|-------|
| Llama → Qwen | 42.55 | 1.64 | ✓ |
| Qwen → Llama | 79.64 | 28.00 | ✓ |
| Qwen-8B → 14B | 55.09 | 7.09 | ✓ |
| Qwen-8B → 32B | 28.91 | 2.00 | ✓ |
| Qwen LoRA → Full FT | 10.73 | 2.55 | ✓ |

### Key Findings
- **Module-wise normalization is essential**: Omitting it allows response length bias to overwhelm the risk signal (Δ drops from 68.72 to 12.54 and monotonicity is lost).
- **Dual directions are necessary**: Using only danger (Δ=64.54, non-monotonic) or only safety (Δ=20.91, non-monotonic) is inferior to the full version. The directions calibrate each other: danger indicates "push toward risk," while safety indicates "push away from safety." The difference represents the net risk.
- **Initialization sensitivity is a key limitation**: SQSD must be calculated at "direction-sensitive parameter states" (high-sensitivity checkpoints found mid-fine-tuning). Ranking fails at insensitive states like $\theta_0$.
- **Generalization across architectures/scales/LoRA to Full FT**: This suggests that the "sample-level danger drift propensity" captured by SQSD is a relatively architecture-independent attribute, allowing for "small model scoring + large model deployment" data filtering.

## Highlights & Insights
- **The "parameter dynamics perspective" is a breakthrough from static perturbation analysis**: Instead of asking "how much parameters differ," the authors ask "which direction they drift and how far." This reveals the non-linearity of the "two-stage collapse."
- **Continuous risk scoring vs. extreme selection**: Unlike prior methods that only identify the top-k dangerous samples, SQSD provides a full-spectrum score, enabling future potential for "risk-weighted safety fine-tuning."
- **Safety application of Task Vectors**: A clean adaptation of task arithmetic to safety alignment, validated by steering experiments.
- **Cross-scale transfer offers practical paths**: SQSD can calculate scores on an 8B model and apply them to 32B or Full FT models, significantly reducing data filtering costs.

## Limitations & Future Work
- SQSD depends heavily on the "direction sensitivity" of initialization. The authors admit that in some cases, even the top-1 sensitive checkpoint doesn't guarantee perfect consistency, requiring fallbacks to top-3/4.
- Constructing safety/danger directions requires full DPO/SFT training, which is costly. Lightweight or zero-shot construction is a future direction.
- While LoRA-to-Full FT transfer is shown, SQSD behavior under full FT might be more complex due to fewer gradient constraints.
- SQSD has not been integrated into a training algorithm yet; its value is currently diagnostic and for data filtering.

## Related Work & Insights
- **vs. Bi-Anchor / Self-Inf-N / LARF**: These use embedding or raw gradient similarity without distinguishing direction; SQSD uses the projection difference of two parameter directions for better sensitivity and continuous scoring.
- **vs. Vaccine / Booster**: These look at fragility via static perturbations; this paper's two-stage trajectory observation significantly expands the understanding of collapse mechanisms.
- **vs. LESS (gradient-based data selection)**: LESS is for general influence estimation; SQSD is specialized for safety with an interpretable task-vector basis.
- **vs. Safety Basin Theory (Peng et al.)**: The two-stage drift phenomenon confirms the safety basin theory, and SQSD converts this geometric intuition into an actionable sample score.

## Rating
- Novelty: ⭐⭐⭐⭐ "Parameter dynamics + dual-direction projection difference" is a very clear new perspective for safety risk scoring.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive coverage across 3 models, 2 datasets, 5 baselines, 5 ablations, and various transfer settings.
- Writing Quality: ⭐⭐⭐⭐ Linear and clear structure (RQ1 $\rightarrow$ Mechanism $\rightarrow$ RQ2 $\rightarrow$ Method $\rightarrow$ Theory $\rightarrow$ Experiments).
- Value: ⭐⭐⭐⭐ Provides both an interpretable mechanistic framework and a practical tool for safety audits in production fine-tuning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Synthetic Eggs in Many Baskets: The Impact of Synthetic Data Diversity on LLM Fine-Tuning](../../ACL2026/llm_nlp/synthetic_eggs_in_many_baskets_the_impact_of_synthetic_data_diversity_on_llm_fin.md)
- [\[ACL 2026\] GRASS: Gradient-based Adaptive Layer-wise Importance Sampling for Memory-Efficient LLM Fine-tuning](../../ACL2026/llm_nlp/grass_gradient-based_adaptive_layer-wise_importance_sampling_for_memory-efficien.md)
- [\[NeurIPS 2025\] Synergy over Discrepancy: A Partition-Based Approach to Multi-Domain LLM Fine-Tuning](../../NeurIPS2025/llm_nlp/synergy_over_discrepancy_a_partition-based_approach_to_multi-domain_llm_fine-tun.md)
- [\[ICLR 2026\] How Catastrophic is Your LLM? Certifying Risk in Conversation](../../ICLR2026/llm_nlp/how_catastrophic_is_your_llm_certifying_risk_in_conversation.md)
- [\[NeurIPS 2025\] Sparse MeZO: Less Parameters for Better Performance in Zeroth-Order LLM Fine-Tuning](../../NeurIPS2025/llm_nlp/sparse_mezo_less_parameters_for_better_performance_in_zeroth-order_llm_fine-tuni.md)

</div>

<!-- RELATED:END -->
