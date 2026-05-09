---
title: >-
  [Paper Note] Scaling with Collapse: Efficient and Predictable Training of LLM Families
description: >-
  [ICLR 2026][Medical Imaging][training loss curve collapse] This paper demonstrates that the training loss curves (TLCs) of LLM families "collapse" onto a single universal curve when optimization hyperparameters are matched to the data budget, and leverages this phenomenon for two practical applications: (1) deviation from collapse as an early diagnostic signal for training pathologies, and (2) the predictability of the collapse curve enabling early stopping for large-scale hyperparameter tuning.
tags:
  - ICLR 2026
  - Medical Imaging
  - training loss curve collapse
  - hyperparameter scaling
  - training diagnostics
  - early stopping
  - Cerebras
date: 2026-05-08
content_hash: 9c8f371aab2c2d82
---

# Scaling with Collapse: Efficient and Predictable Training of LLM Families

**Conference**: ICLR 2026
**arXiv**: [2509.25087](https://arxiv.org/abs/2509.25087)
**Code**: None
**Area**: Medical Imaging
**Keywords**: training loss curve collapse, hyperparameter scaling, training diagnostics, early stopping, Cerebras

## TL;DR
This paper demonstrates that the training loss curves (TLCs) of LLM families "collapse" onto a single universal curve when optimization hyperparameters are matched to the data budget, and leverages this phenomenon for two practical applications: (1) deviation from collapse as an early diagnostic signal for training pathologies, and (2) the predictability of the collapse curve enabling early stopping for large-scale hyperparameter tuning.

## Background & Motivation

### State of the Field

**Background**: Scaling laws can predict final loss and μP enables learning rate transfer, but the predictability of complete TLCs has not been validated at practical LLM scales.

**Limitations of Prior Work**:

### Root Cause

**Key Challenge**: Qiu et al. discovered the loss curve collapse phenomenon but validated it only at small scale, without testing on practical LLM training recipes.

### Limitations of Prior Work

**Limitations of Prior Work**: Direct experimentation at frontier scale is infeasible — conclusions must be extrapolated from small-scale results.

### Approach

**Approach**: Diagnosing training pathologies (loss spikes) still relies on manual judgment.

**Core Finding**: The necessary and sufficient condition for loss curve collapse is that the optimization hyperparameters are optimal for a given data budget — collapse is a "fingerprint" of compute-optimal training.

**Key Insight**: When all models are trained with the same tokens-per-parameter ($\text{TPP} = D/N$) and the AdamW timescale $\tau$ is set optimally, TLCs of models of different sizes fall onto the same universal curve after simple normalization.

## Method

### Overall Architecture
Two practical applications: (1) **Deviation-from-collapse diagnostics**: online monitoring of the deviation between the current TLC and the universal collapse curve → anomalous spikes or drift can be detected earlier; (2) **Early-stopping hyperparameter tuning**: the collapse curve is predictable → extrapolate final loss from partial TLCs to early-terminate poorly performing configurations.

### Key Designs
1. **Collapse condition**: all models share the same TPP + optimal hyperparameters (learning rate, batch size, weight decay jointly scaled) → TLC collapse.
2. **Deviation diagnostics**: fit the universal curve from small models, compare against large models in real time → numerical stability issues manifest in the deviation signal earlier.
3. **Early stopping**: fit a parameterized model to the collapse curve, extrapolate final loss from the first 10–20% of training, saving 80%+ of hyperparameter tuning compute.

### Celerity LLM Family
- A competitive LLM family trained using insights from the collapse framework.
- All experiments conducted on Cerebras CS-3.

## Key Experimental Results

### Main Results

| Phenomenon | Result |
|------------|--------|
| Llama-2 (varying TPP) | TLC does not collapse |
| Celerity (identical TPP + optimal hyperparameters) | **TLC collapses perfectly** |
| Deviation diagnostics | Detects loss spikes earlier than manual inspection |
| Early-stopping hyperparameter tuning | Extrapolates final loss from 20% of TLC with <1% error |

### Key Findings
- **Collapse is the necessary and sufficient condition for compute-optimal training** — it appears only when hyperparameters are set optimally according to scaling laws.
- Deviation diagnostics can detect numerical stability issues earlier (e.g., insufficient bf16 precision).
- Early stopping saves 80%+ of hyperparameter search compute.

## Ablation Study

### Collapse Condition Verification

| Condition | Collapse? | Notes |
|-----------|-----------|-------|
| Fixed TPP + optimal $\tau$ + fixed LR schedule | ✓ Collapse | Celerity family |
| Varying TPP (e.g., Llama-2) | ✗ No collapse | Different $D/N$ ratios alter TLC shape |
| Fixed TPP + suboptimal $\tau$ | ✗ No collapse | $\tau$ deviating from optimum stretches or compresses TLC |
| Fixed TPP + different LR schedule | ✗ No collapse | LR decay shape directly affects TLC shape |

### Practical Case Study: Deviation Diagnostics
- During Celerity 1.8B training, cached loss values exhibited a mild upward trend.
- Collapse residual analysis (normalizing the TLC and comparing against the universal curve) detected the deviation hundreds of steps before any obvious anomaly appeared in the raw TLC.
- Diagnosis: gradient accumulation instability caused by bf16 numerical precision issues.
- After the fix, the TLC returned to the collapse curve.

### Early-Stopping Hyperparameter Tuning
- Over 20 hyperparameter configurations were trained for only the first 20% of tokens.
- A parameterized collapse curve model was fitted to the partial TLC to extrapolate final loss.
- The 80% of configurations with the worst projected final loss were eliminated; only the top 20% proceeded to full training.
- This saves 80%+ of hyperparameter search compute, with extrapolation error <1%.

### Celerity on the Efficiency Frontier

| Model | Parameters | Training Tokens | Average Accuracy |
|-------|-----------|----------------|-----------------|
| Typical same-scale model | Equivalent | Equivalent | Baseline |
| **Celerity** | Equivalent | Equivalent | **Efficiency frontier** |

## Highlights & Insights
- **Collapse as a "health indicator"** is a simple yet powerful engineering tool — if the TLC does not collapse, the hyperparameters or training recipe are likely flawed. This is more intuitive than any other metric.
- **Collapse ↔ compute-optimal training** as a necessary and sufficient relationship is the core theoretical contribution — connecting a visual phenomenon to optimization theory.
- **Practical value of deviation diagnostics**: conventional approaches require manual judgment on whether a loss spike warrants rollback; the collapse curve provides an objective reference.
- **Early-stopping hyperparameter tuning**: reliable extrapolation of final loss substantially reduces the cost of large-scale hyperparameter search.
- **The unifying role of $\tau$**: the AdamW EMA timescale $\tau = 1/(\eta\lambda)$ is an overlooked yet critically important hyperparameter — it unifies the effects of learning rate and weight decay.

## Limitations & Future Work
- The collapse condition requires identical TPP across all models — in practice, different models may have different optimal TPP (e.g., Chinchilla's ratio of 20 vs. other estimates).
- Validation is limited to pretraining loss — whether collapse extends to downstream task performance is unexplored (loss collapse does not guarantee downstream accuracy collapse).
- Early-stopping extrapolation relies on the accuracy of the parameterized collapse curve model — refitting may be necessary for substantially different training recipes.
- All experiments were conducted on Cerebras CS-3 — collapse behavior on different hardware (e.g., GPUs) may differ slightly due to precision and communication patterns.
- Collapse has only been validated under μP parameterization — whether it holds under other parameterizations (e.g., SP) remains unknown.

## Related Work & Insights
- **vs. Chinchilla (Hoffmann et al.)**: Chinchilla predicts the final loss via a scaling law (a scalar); this paper predicts the shape of the entire training curve (a curve) — a "time-series version" of scaling laws.
- **vs. Qiu et al. (2025) Supercollapse**: They discovered collapse in small-scale autoregressive tasks; this paper generalizes it to practical LLM training and identifies the necessary and sufficient conditions (identical TPP + optimal $\tau$).
- **vs. μP (Yang & Hu)**: μP enables learning rate transfer across scales; this paper finds that under μP the entire TLC shape transfers across scales — a stronger corollary of μP.
- **vs. Wang & Aitchison (2024) AdamW EMA**: They found $\tau$ to be stable across scales in vision tasks; this paper finds that the optimal value of $\tau$ depends on TPP and is the key control variable for TLC collapse in LLMs.
- **Inspiration**: Collapse theory may generalize to other sequential training settings — e.g., whether similar universal shapes exist in diffusion model or reinforcement learning training curves.

## Rating
- Novelty: ⭐⭐⭐⭐ — The discovery of collapse conditions and their practical applications offer unique insights.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Large-scale Cerebras experiments, validation across multiple model sizes, real training diagnostic case studies.
- Writing Quality: ⭐⭐⭐⭐⭐ — The three-column comparison in Figure 1 is highly intuitive; the writing is clear throughout.
- Value: ⭐⭐⭐⭐⭐ — Extremely high practical engineering value for large-scale LLM training.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Reverse Distillation: Consistently Scaling Protein Language Model Representations](reverse_distillation_consistently_scaling_protein_language_model_representations.md)
- [\[ICLR 2026\] DriftLite: Lightweight Drift Control for Inference-Time Scaling of Diffusion Models](driftlite_lightweight_drift_control_for_inference-time_scaling_of_diffusion_mode.md)
- [\[ICLR 2026\] ExpGuard: LLM Content Moderation in Specialized Domains](expguard_llm_content_moderation_in_specialized_domains.md)
- [\[ICLR 2026\] AFD-INSTRUCTION: A Comprehensive Antibody Instruction Dataset with Functional Annotations for LLM-Based Understanding and Design](afd-instruction_a_comprehensive_antibody_instruction_dataset_with_functional_ann.md)
- [\[AAAI 2026\] Neural Bandit Based Optimal LLM Selection for a Pipeline of Tasks](../../AAAI2026/medical_imaging/neural_bandit_based_optimal_llm_selection_for_a_pipeline_of_tasks.md)

<!-- RELATED:END -->
