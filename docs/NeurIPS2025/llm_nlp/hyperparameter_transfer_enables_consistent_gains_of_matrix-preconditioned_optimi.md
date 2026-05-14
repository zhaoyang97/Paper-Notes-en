---
title: >-
  [Paper Note] Hyperparameter Transfer Enables Consistent Gains of Matrix-Preconditioned Optimizers Across Scales
description: >-
  [NEURIPS2025][LLM/NLP][optimizer scaling] This paper investigates the hyperparameter scaling rules for matrix-preconditioned optimizers (Shampoo/SOAP/Muon) with respect to model width and depth under the μP framework…
tags:
  - "NEURIPS2025"
  - "LLM/NLP"
  - "optimizer scaling"
  - "μP"
  - "Shampoo"
  - "SOAP"
  - "Muon"
  - "hyperparameter transfer"
  - "matrix preconditioning"
date: 2026-05-08
content_hash: f07bd395793ebeb6
---

# Hyperparameter Transfer Enables Consistent Gains of Matrix-Preconditioned Optimizers Across Scales

**Conference**: NEURIPS2025
**arXiv**: [2512.05620](https://arxiv.org/abs/2512.05620)
**Code**: To be confirmed
**Area**: LLM/NLP
**Keywords**: optimizer scaling, μP, Shampoo, SOAP, Muon, hyperparameter transfer, matrix preconditioning

## TL;DR
This paper investigates the hyperparameter scaling rules for matrix-preconditioned optimizers (Shampoo/SOAP/Muon) with respect to model width and depth under the μP framework, and demonstrates that correct hyperparameter scaling is the key to achieving consistent speedups. Using μP with $1/\text{width}$ weight decay, all three optimizers consistently achieve approximately $1.4\times$ speedup on Llama models ranging from 190M to 1.4B parameters.

## Background & Motivation
**Background**: Several matrix-preconditioned optimizers (Shampoo, SOAP, Muon) have demonstrated significant speedups over AdamW (1.5–2×) in small-scale experiments. Muon has already been adopted in OpenAI's trillion-parameter training runs.

**Limitations of Prior Work**: Reproduction attempts have reported highly inconsistent speedups—some teams report 2× acceleration, others only 1.1×, and some find that the gains vanish rapidly as scale increases. The root cause is the **lack of reliable hyperparameter scaling rules**.

**Key Challenge**: The loss-compute scaling exponent in language modeling is small (~0.05), meaning a 2% loss difference corresponds to a 40% compute difference. Grid search over hyperparameters is infeasible at large scale, making reliable hyperparameter transfer essential.

**Goal**: Derive μP learning rate scaling rules for Shampoo/SOAP/Muon and empirically verify that correct scaling is critical for consistent speedups across model scales.

**Key Insight**: Apply μP (Maximal Update Parameterization) theory to derive per-optimizer scaling rules for learning rate and weight decay as functions of model width and depth.

**Core Idea**: The speedups from matrix-preconditioned optimizers are genuine, but correct μP scaling is required for them to manifest reliably at large scale.

## Method

### Overall Architecture
For each matrix-preconditioned optimizer, the paper derives μP scaling rules (governing how learning rate, weight decay, and regularization parameters scale with model width/depth), and validates these rules on language models ranging from 190M to 1.4B parameters.

### Key Designs

1. **μP Derivation (General Procedure)**:

    - For each weight matrix $W \in \mathbb{R}^{n \times m}$, ensure that the RMS of the update $\Delta W$ scales consistently as width $\to \infty$.
    - Due to differences in preconditioning, the learning rate scaling differs across optimizers: Adam uses $1/d_{\text{in}}$, Muon uses $\sqrt{d_{\text{out}}/d_{\text{in}}}$, and Shampoo depends on exponents $e_L, e_R$.

2. **Mitigating Finite-Width Bias**:

    - While μP guarantees learning rate transfer in the infinite-width limit, the optimal learning rate may shift at practical finite widths.
    - **Blocking** (partitioning large weight matrices into smaller blocks) and **Spectral Normalization** effectively mitigate this shift.
    - Grafting (normalizing the update direction by the norm from another optimizer) also affects the scaling behavior.

3. **Weight Decay Scaling**:

    - The paper finds that an independent weight decay (decoupled from the learning rate) scaled as $1/\text{width}$ is near-optimal across all optimizers.
    - This is consistent with the recommendation of Xiao (2024).

### Loss & Training
- Llama-architecture language models (190M, 470M, 1B, 1.4B) are trained on the FineWeb dataset.
- Hyperparameters are tuned on small models and transferred to large models via the derived μP scaling rules.

## Key Experimental Results

### Main Results
Compute-equivalent comparisons on Llama models from 190M to 1.4B:

| Optimizer | μP Scaling | 190M Speedup | 470M Speedup | 1B Speedup | 1.4B Speedup |
|-----------|-----------|-------------|-------------|-----------|-------------|
| Muon | ✅ | ~1.4× | ~1.4× | ~1.4× | **~1.4×** |
| SOAP | ✅ | ~1.4× | ~1.4× | ~1.4× | **~1.4×** |
| Shampoo | ✅ | ~1.4× | ~1.4× | ~1.3× | **~1.3×** |
| Muon | ❌ (SP) | ~1.4× | ~1.2× | ~1.0× | **Vanishes** |

### Ablation Study

| Configuration | Result | Notes |
|--------------|--------|-------|
| μP vs. standard parameterization | Consistent speedup under μP; vanishes under SP as scale increases | Hyperparameter transfer is critical |
| Blocking (128) | Mitigates finite-width bias | Practical and effective |
| Spectral normalization | Reduces learning rate sensitivity | Complementary to μP |
| WD=$1/\text{width}$ vs. fixed WD | $1/\text{width}$ is near-optimal | Consistent across optimizers |
| Compute-optimal model size | Matrix-preconditioned optimizers favor larger models | Different scaling law from AdamW |

### Key Findings
- **Incorrect hyperparameter scaling is the primary cause of prior reproduction failures**: Under standard parameterization, the speedups from Muon/SOAP nearly vanish on models at the 1B+ scale.
- **A $1.4\times$ speedup is consistent and reliable**: Under correct μP scaling, all three optimizers consistently achieve approximately $1.4\times$ compute savings across all tested scales.
- **μP scaling differs across optimizers**: Directly applying Adam's μP rules to Muon is a common and consequential mistake.

## Highlights & Insights
- **Systematically resolves the community debate on whether matrix-preconditioned optimizers are beneficial**—the answer is yes, but correct scaling is required. Prior inconsistent results can be attributed to hyperparameter misconfiguration.
- **Generality of the μP derivation procedure**: The paper provides a simple, general workflow for deriving scaling rules for any new optimizer.
- **Practical guidance**: Complete scaling formulas for each optimizer are provided (Table 1) and are directly applicable.

## Limitations & Future Work
- **Largest model tested is only 1.4B**: Validation on models at 10B+ scale is absent.
- **Only language modeling is evaluated**: Other tasks such as vision and multimodal settings are not addressed.
- **Relatively conservative speedup ($1.4\times$)**: This is more modest than the $2\times$ reported in some prior work, possibly because the AdamW baseline is more carefully tuned.
- **Future directions**: Validation at larger scales; analysis of interactions with learning rate warmup/cooldown schedules; development of automated hyperparameter transfer tools.

## Related Work & Insights
- **vs. Liu et al. (2024, M-Muon)**: They report a $2\times$ speedup for Muon without μP; this may result from an insufficiently tuned AdamW baseline.
- **vs. Wen et al. (2024)**: They report that speedups vanish with scale—the present paper explains this as a consequence of hyperparameter drift under standard parameterization.
- **vs. Yang et al. (μP)**: The original μP framework covers only Adam/SGD; this paper extends it to matrix-preconditioned optimizers.

## Rating
- Novelty: ⭐⭐⭐⭐ The extension of μP to new optimizers is elegant, though the core framework builds on existing μP theory.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multiple optimizers × multiple scales × ablations; very comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear and systematic; Table 1 provides an at-a-glance summary of scaling rules.
- Value: ⭐⭐⭐⭐⭐ Directly resolves a community controversy and delivers actionable practical guidance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Don't Be Lazy: CompleteP Enables Compute-Efficient Deep Transformers](dont_be_lazy_completep_enables_compute-efficient_deep_transformers.md)
- [\[ICLR 2026\] Weight Decay may matter more than μP for Learning Rate Transfer in Practice](../../ICLR2026/llm_nlp/weight_decay_may_matter_more_than_mup_for_learning_rate_transfer_in_practice.md)
- [\[ICLR 2026\] Evaluating Text Creativity across Diverse Domains: A Dataset and Large Language Model Evaluator](../../ICLR2026/llm_nlp/evaluating_text_creativity_across_diverse_domains_a_dataset_and_large_language_m.md)
- [\[NeurIPS 2025\] Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models](nemotron-flash_towards_latency-optimal_hybrid_small_language_models.md)
- [\[NeurIPS 2025\] Q♯: Provably Optimal Distributional RL for LLM Post-Training](qsharp_provably_optimal_distributional_rl_for_llm_post-training.md)

</div>

<!-- RELATED:END -->
