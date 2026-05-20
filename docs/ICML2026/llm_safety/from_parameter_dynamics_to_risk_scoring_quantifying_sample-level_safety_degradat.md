---
title: >-
  [Paper Note] From Parameter Dynamics to Risk Scoring: Quantifying Sample-Level Safety Degradation in LLM Fine-tuning
description: >-
  [ICML 2026][LLM Safety][safety alignment] By tracking the cumulative parameter drift along "dangerous/safe directions" during LoRA fine-tuning…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "safety alignment"
  - "fine-tuning risk"
  - "parameter dynamics"
  - "task vector"
  - "sample scoring"
date: 2026-05-08
content_hash: ba3b2613d414231c
---

# From Parameter Dynamics to Risk Scoring: Quantifying Sample-Level Safety Degradation in LLM Fine-tuning

**Conference**: ICML 2026  
**arXiv**: [2605.04572](https://arxiv.org/abs/2605.04572)  
**Code**: https://github.com/(repo) (available)  
**Area**: LLM alignment / safety / model compression  
**Keywords**: safety alignment, fine-tuning risk, parameter dynamics, task vector, sample scoring

## TL;DR
By tracking the cumulative parameter drift along "dangerous/safe directions" during LoRA fine-tuning, the authors reveal that the fundamental mechanism behind benign data breaking alignment is the monotonic drift of parameters toward the dangerous direction during fine-tuning. They propose SQSD—assigning each sample a continuous risk score based on the difference in single-step gradient projections along these two directions. SQSD maintains monotonic ASR ranking across 3 models × 2 datasets and generalizes across architectures, scales, and LoRA→Full transfer.

## Background & Motivation

**Background**: Before deployment, LLMs are aligned (RLHF/DPO) to refuse harmful requests. However, alignment is surprisingly fragile during downstream fine-tuning—even with only 100 completely harmless, benign instruction samples, safety can collapse. This "benign sample breaking alignment" attack is more dangerous than directly feeding harmful samples, as toxicity classifiers cannot detect it.

**Limitations of Prior Work**: Previous studies on "why alignment collapses" follow two lines: (i) examining embedding drift (e.g., Vaccine), (ii) analyzing static parameter perturbations (Booster, PEFT safety analysis). Both only compare pre-/post-fine-tuning snapshots, without tracking the process, and focus on perturbation magnitude rather than direction, making it hard to distinguish "safety-related drift" from "task-related drift." High-risk sample selection methods (Bi-Anchor / Self-Inf-N / LARF) focus on "extreme sample selection," only picking the most dangerous subset and failing for intermediate-risk samples, leading to "boundary collapse."

**Key Challenge**: Safety degradation is a **dynamic process** rather than a static perturbation, and is strongly related to "which direction parameters drift," not just the magnitude. Measuring only the magnitude mixes "task learning" and "safety breaking" effects, making them inseparable.

**Goal**: (RQ1) Provide a mechanistic explanation for why benign fine-tuning breaks alignment; (RQ2) Under this mechanism, compute a continuous, interpretable fine-tuning safety risk score for **each sample**.

**Key Insight**: Inspired by the Task Vector approach, define $V_\text{safety} = \hat\theta_\text{aligned} - \theta_0$ and $V_\text{danger} = \hat\theta_\text{harmful} - \theta_0$ as two semantic anchor vectors. Then, track the cumulative drift at each fine-tuning step $\Delta\theta_t = \theta_t - \theta_0$ and its projection along these directions, directly linking "safety degradation" to "parameter trajectory."

**Core Idea**: Use directional projections in parameter space both to explain the mechanism (cumulative dangerous drift drives safety collapse) and to score samples (single-step gradient projection along dangerous direction minus safe direction = sample risk score SQSD).

## Method

### Overall Architecture
Two parts. **Part 1 (Mechanism Analysis)**: Construct two semantic anchor vectors → track cumulative drift at each fine-tuning step → observe the phenomenon where "dangerous direction projection monotonically increases + safe direction projection remains near zero," coinciding with Safety Score dropping from 5.0 to below 1.0. **Part 2 (Sample Scoring SQSD)**: For each candidate sample $z$, compute a single-step LoRA gradient → estimate the equivalent weight update $\Delta W(z)$ → for each LoRA module, compute the normalized projection difference of $\Delta W(z)$ onto $V_\text{danger}$ and $V_\text{safety}$ → sum across all modules to obtain final $\text{SQSD}(z)$.

### Key Designs

1. **Construction and Validation of "Safe / Dangerous" Directions in Parameter Space**:

    - **Function**: Define two interpretable, projectable semantic anchor vectors in parameter space.
    - **Mechanism**: $V_\text{safety} = \arg\min_\theta \mathcal{L}_\text{dpo}(\theta_0, D_\text{aligned}) - \theta_0$ (using PKU-SafeRLHF for DPO), $V_\text{danger} = \arg\min_\theta \mathcal{L}_\text{sft}(\theta_0, D_\text{harmful}) - \theta_0$ (using SFT on Aegis-unsafe and BeaverTails-unsafe to construct Aegis/Beaver dangerous directions). Then, perform steering experiments $\theta(\alpha) = \theta_0 + \alpha V$ sweeping $\alpha$, confirming that moving along $V_\text{danger}$ monotonically decreases Safety Score, while moving along $V_\text{safety}$ monotonically increases it, validating that these vectors encode safety-related parameter displacements.
    - **Design Motivation**: Previous Task Vector literature mainly used for task arithmetic, not safety analysis. Here, the same "direction = training endpoint − start point" idea concretizes abstract "safety" and "danger" into computable direction vectors, forming the basis for subsequent projection analysis.

2. **Tracking Cumulative Parameter Drift Reveals Two-Stage Degradation**:

    - **Function**: Answer the mechanistic question of "why collapse" with an interpretable trajectory plot.
    - **Mechanism**: At each fine-tuning checkpoint, compute $p_\text{safety}(t) = \langle\Delta\theta_t, \hat{V}_\text{safety}\rangle$ and $p_\text{danger}(t) = \langle\Delta\theta_t, \hat{V}_\text{danger}\rangle$, revealing a **distinct nonlinear two-stage pattern**: early parameters drift rapidly along the dangerous direction (projection from 0 to ~6.0) but Safety Score only drops from 5.0 to 4.0; later, drift slows but Safety Score suddenly collapses below 1.0.
    - **Design Motivation**: This two-stage curve supports the geometric intuition of a "safety basin" (locally robust to perturbations, but collapses outside the basin), and guides SQSD initialization—gradients must be computed at "direction-sensitive" parameter states, otherwise the same perturbation signal is meaningless in insensitive regions.

3. **SQSD: Single-Step Gradient Projection Difference as Sample Risk Score**:

    - **Function**: Assign each sample a continuous risk score, covering the entire risk spectrum rather than only extreme subsets.
    - **Mechanism**: For sample $z$, compute a single-step LoRA parameter gradient, derive the equivalent weight update $\Delta W(z) \approx -\eta(B_0 \nabla_A + \nabla_B A_0)$ (ignoring $\mathcal{O}(\eta^2)$ second-order terms). For each LoRA module $m$, after module-wise normalization, compute $\text{SQSD}_m(z) = \langle\frac{\Delta W_m(z)}{\|\Delta W_m(z)\|_2}, \hat{V}_{\text{danger},m}\rangle - \langle\frac{\Delta W_m(z)}{\|\Delta W_m(z)\|_2}, \hat{V}_{\text{safety},m}\rangle$, and finally $\text{SQSD}(z) = \sum_m \text{SQSD}_m(z)$. The authors use first-order Taylor expansion to show this projection difference corresponds to the relative extent to which the sample pushes the model toward $\theta_\text{danger}$ vs $\theta_\text{safety}$, thus interpretable.
    - **Design Motivation**: Module-wise normalization eliminates "response length bias"—gradient magnitude naturally correlates with answer length, so without normalization, short-answer samples would be incorrectly ranked as high risk. The authors also discuss that initialization must be at a "direction-sensitive parameter state" (i.e., the nonlinear sensitive region identified above) to ensure SQSD ranking is effective; otherwise, projection signals cannot distinguish sample risk.

### Loss & Training
Not a training objective itself. All LoRA fine-tuning uses $r=8, \alpha=16$; for direction construction, lr=5e-6; for SQSD evaluation, lr=5e-5 to induce more pronounced safety degradation. First-order Taylor derivation: $\eta[\mathcal{L}(z, \theta_\text{ref}) - \mathcal{L}(z, \theta_\text{target})] \approx (\theta' - \theta_\text{ref})^\top (\theta_\text{target} - \theta_\text{ref})$, linking the inner product of "gradient update direction vs target direction" to loss change, providing a preference-based interpretation for SQSD.

## Key Experimental Results

### Main Results
Three models (Qwen3-8B / Llama-3.1-8B-Instruct / Llama-2-7B-Chat) × two datasets (Dolly / Alpaca). Each dataset is split into S1-S5 five subsets (each 1000 samples, from highest to lowest risk) by SQSD ranking; fine-tune each and measure ASR to check for monotonic decrease.

| Model + Data | S1 ASR | S5 ASR | Δ (S1-S5) | Monotonic? |
|--------------|--------|--------|-----------|------------|
| Qwen3-8B / Dolly / SQSD(Beaver) | 71.27 | 2.55 | **+68.72** | ✓ |
| Qwen3-8B / Alpaca / SQSD(Beaver) | 50.91 | 3.27 | +47.64 | ✓ |
| Llama3.1-8B / Dolly / SQSD(Beaver) | 79.82 | 4.73 | +75.09 | ✓ |
| Llama2-7B / Dolly / SQSD(Beaver) | 45.27 | 0.36 | +44.91 | ✓ |
| Reward Model baseline (Dolly avg) | 57.27 | 8.00 | 49.27 | ✗ |
| LARF baseline (Dolly avg) | 48.91 | 4.61 | 44.30 | ✗ |

SQSD maintains **strict monotonicity in 10/12 configurations** (other baselines at most 1/6 monotonic), and the average Δ of 49.86% exceeds the strongest baseline (Reward Model 43.76%). This means SQSD not only identifies extreme high-risk samples but also correctly ranks intermediate samples—baselines generally fail in the middle range.

### Ablation Study

| Configuration | S1 ASR | Δ | Monotonic? |
|---------------|--------|---|------------|
| SQSD (full) | 71.27 | 68.72 | ✓ |
| w/o module-wise normalization | 13.09 | 12.54 | ✗ (dominated by response length) |
| Danger direction only | 68.36 | 64.54 | ✗ (lacks safety counterbalance) |
| Safety direction only | 27.09 | 20.91 | ✗ (no danger signal) |
| Insensitive initialization | 38.36 | 37.27 | ✗ (projection signal fails in insensitive state) |

| Transfer Study | S1 | S5 | Monotonic? |
|---------------|----|----|------------|
| Llama → Qwen | 42.55 | 1.64 | ✓ |
| Qwen → Llama | 79.64 | 28.00 | ✓ |
| Qwen-8B → 14B | 55.09 | 7.09 | ✓ |
| Qwen-8B → 32B | 28.91 | 2.00 | ✓ |
| Qwen LoRA → Full FT | 10.73 | 2.55 | ✓ |

### Key Findings
- **Module-wise normalization is essential**: Without normalization, response length bias overwhelms true risk signals, Δ drops from 68.72 to 12.54, and monotonicity is lost.
- **Both directions are necessary**: Using only danger (Δ=64.54 but not monotonic) or only safety (Δ=20.91 and not monotonic) is inferior to the full version. The two directions calibrate each other: danger indicates "how dangerous," safety indicates "how far from safe," and their difference is net risk.
- **Initialization sensitivity is a key limitation of SQSD**: SQSD must be computed at a "direction-sensitive parameter state" (mid-fine-tuning, high-sensitivity checkpoint); computing at $\theta_0$ (insensitive state) renders ranking ineffective. The authors formalize linear-path and drift-enhanced sensitivities and provide a top-5 checkpoint selection strategy.
- **Cross-architecture / cross-scale / LoRA→Full transferability**: Indicates that SQSD captures a "sample-level danger drift propensity" that is relatively architecture-agnostic, making it practical—scores can be computed on small models and deployed for data filtering on large models.

## Highlights & Insights
- **"Parameter dynamics perspective" is key to moving beyond static perturbation analysis**: Rather than asking "how much do parameters differ before and after fine-tuning," ask "in what direction and how far do parameters drift during fine-tuning." This perspective directly reveals the nonlinear "two-stage collapse" and separates "safety-related drift" from "task-related drift," providing a rare clear framework for safety mechanism analysis.
- **Continuous risk scoring vs. extreme selection is a crucial distinction**: Previous methods (Bi-Anchor / Self-Inf-N / LARF) only select the most dangerous top-k samples, so subsequent fine-tuning strategies can only "avoid extremes," with no judgment for intermediate samples; SQSD provides full-spectrum scoring, enabling future "risk-weighted safety fine-tuning algorithms."
- **"Safety use" of Task Vector is a valuable extension**: Originally, task vectors were used for task arithmetic (adding/subtracting capabilities); here, they are adapted for "adding/subtracting safety," with steering experiments validating direction effectiveness—a clean application of this family to safety alignment.
- **Cross-scale transfer enables "small model pre-filtering + large model deployment"**: In practice, SQSD does not require gradient computation on giant models; scores can be computed on 8B models and then used for 32B / Full FT, greatly reducing data filtering costs.

## Limitations & Future Work
- SQSD is highly dependent on the "direction sensitivity" of initialization; the authors acknowledge that even with the top1 sensitive checkpoint, ranking consistency is not always perfect, sometimes requiring fallback to top3/top4—this remains an unresolved robustness issue.
- Construction of safe/dangerous directions requires a full DPO/SFT training, which is not cheap; exploring lighter-weight (even zero-shot) direction construction is a promising direction.
- All experiments are limited to LoRA fine-tuning (though LoRA→Full transfer is shown), but LoRA itself constrains gradient directions; SQSD behavior under full FT may be more complex.
- SQSD is not integrated into the fine-tuning algorithm itself—only the diagnostic value of "fine-tuning on data split by SQSD" is shown, without proposing "risk-weighted SFT" or "risk-aware data filtering" as direct applications, leaving clear follow-up opportunities.

## Related Work & Insights
- **vs Bi-Anchor / Self-Inf-N / LARF**: These use embedding or raw gradient similarity for risk assessment, without directionality; SQSD uses projection difference along "dangerous vs safe" parameter directions, is more direction-sensitive, and provides continuous scoring rather than only selecting extremes.
- **vs Vaccine / Booster**: Both analyze safety vulnerability from a static perturbation perspective, without tracking dynamics; this work's two-stage trajectory observation significantly extends understanding of collapse mechanisms.
- **vs LESS (gradient-based data selection)**: The latter is a general influence estimator; SQSD is specialized for the safety dimension, with a task-vector interpretation and clearer directionality.
- **vs Safety Basin theory (Peng et al.)**: The two-stage drift phenomenon directly supports the safety basin concept—robust to perturbations within the basin, collapses outside; SQSD operationalizes this geometric intuition into a sample scoring method.

## Rating
- Novelty: ⭐⭐⭐⭐ "Parameter dynamics + two-direction projection difference" as a safety risk score is a notably clear new perspective, with mechanism analysis and scoring method forming a coherent package.
- Experimental Thoroughness: ⭐⭐⭐⭐ 3 models × 2 datasets + 5 baselines + 5 ablations + cross-architecture/scale/PEFT transfer, with sufficient coverage; but benchmarks are limited to CategoricalHarmfulQA/AdvBench/HEx-PHI, lacking broader multilingual/multimodal safety tests.
- Writing Quality: ⭐⭐⭐⭐ RQ1 → mechanism discovery → RQ2 → method → theory → experiments, with a clear linear structure; Figure 1's conceptual diagram is well done.
- Value: ⭐⭐⭐⭐ Provides both an interpretable safety mechanism framework and a practical sample scoring tool, with direct value for fine-tuning safety audits in production environments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Encoding Mismatch：为什么宽 ViT 的特征蒸馏到窄学生总会失败](from_per-image_low-rank_to_encoding_mismatch_rethinking_feature_distillation_in_.md)
- [\[ICML 2026\] Internalizing Agency from Reflective Experience](internalizing_agency_from_reflective_experience.md)
- [\[ICML 2026\] FedRot-LoRA: Mitigating Rotational Misalignment in Federated LoRA](fedrot-lora_mitigating_rotational_misalignment_in_federated_lora.md)
- [\[ICML 2026\] Decomposing the Basic Abilities of Large Language Models: Mitigating Cross-Task Interference in Multi-Task Instruct-Tuning](decomposing_the_basic_abilities_of_large_language_models_mitigating_cross-task_i.md)
- [\[ICML 2026\] Semantic Integrity Matters: Benchmarking and Preserving High-Density Reasoning in KV Cache Compression](semantic_integrity_matters_benchmarking_and_preserving_high-density_reasoning_in.md)

</div>

<!-- RELATED:END -->
