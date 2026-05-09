---
title: >-
  [Paper Note] Distillation Robustifies Unlearning
description: >-
  [NeurIPS 2025][Model Compression][Robust Unlearning] This paper reveals the core finding that "distillation can robustify unlearning" — distilling an unlearned model into a randomly initialized student network effectively discards latent capabilities. Building on this insight, the paper proposes UNDO (Unlearn-Noise-Distill-on-Outputs), which applies weight perturbation to the unlearned model prior to distillation, establishing a tunable compute–robustness trade-off that approaches the gold standard of retraining from scratch on both synthetic tasks and the WMDP benchmark.
tags:
  - NeurIPS 2025
  - Model Compression
  - Robust Unlearning
  - Knowledge Distillation
  - UNDO
  - Capability Removal
  - Relearning Attacks
  - WMDP
date: 2026-05-08
content_hash: 1f49a7d838ce98ce
---

# Distillation Robustifies Unlearning

**Conference**: NeurIPS 2025
**arXiv**: [2506.06278](https://arxiv.org/abs/2506.06278)
**Code**: Public (GitHub)
**Area**: AI Safety / Machine Unlearning / Knowledge Distillation
**Keywords**: Robust Unlearning, Knowledge Distillation, UNDO, Capability Removal, Relearning Attacks, WMDP

## TL;DR
This paper reveals the core finding that "distillation can robustify unlearning" — distilling an unlearned model into a randomly initialized student network effectively discards latent capabilities. Building on this insight, the paper proposes UNDO (Unlearn-Noise-Distill-on-Outputs), which applies weight perturbation to the unlearned model prior to distillation, establishing a tunable compute–robustness trade-off that approaches the gold standard of retraining from scratch on both synthetic tasks and the WMDP benchmark.

## Background & Motivation
LLM pretraining on large-scale unfiltered data may cause models to acquire hazardous capabilities (e.g., knowledge pertaining to cyberweapon development). Existing mitigation approaches suffer from fundamental limitations:

- **Post-training methods such as RLHF**: These suppress behavioral outputs while leaving underlying capabilities intact, making them vulnerable to recovery via adversarial prompting or fine-tuning attacks.
- **Existing unlearning methods** (gradient ascent, maximum entropy, etc.): These similarly operate at the behavioral level and can be reversed within a few fine-tuning steps.
- **Data filtering + retraining from scratch**: The theoretical gold standard, but annotating and filtering pretraining data at scale is practically infeasible.

The core dilemma is that a model's input–output behavior can be substantially altered while its underlying capability structure remains intact in parameter space — achieving perfect behavioral alignment with an oracle (a model that has never seen the forget data) does not guarantee genuine capability removal.

## Core Problem
How can one achieve **robust removal** of unwanted capabilities from LLMs such that the unlearning effect cannot be easily reversed even under the strongest relearning attacks (adversarial fine-tuning)?

## Method

### Key Finding 1: Oracle Matching Does Not Guarantee Robust Unlearning

A reference model (possessing both retain and forget capabilities) is distilled to match the outputs of an oracle model via KL divergence, bringing the two into near-perfect behavioral alignment (KL loss approaching zero). Nevertheless:

- **Student (Reference)**: Initialized from the reference model and matched to the oracle, it **rapidly recovers** forgotten capabilities under relearning attacks.
- **Student (Random)**: Initialized randomly and matched to the oracle, its recovery speed under relearning attacks **approaches that of the oracle itself**.

This demonstrates that behavioral matching is not equivalent to capability removal in parameter space.

### Key Finding 2: Distillation Robustifies Unlearning

Three standard unlearning methods are each combined with distillation (Unlearn-and-Distill):

1. **GradDiff**: $\mathcal{L}(\theta) = -L_{\text{forget}}(\theta) + L_{\text{retain}}(\theta)$, applying gradient ascent on forget data and gradient descent on retain data.
2. **MaxEnt**: $\mathcal{L}(\theta) = L_{\text{uniform}}(\theta) + L_{\text{retain}}(\theta)$, pushing the output distribution on forget data toward a uniform distribution.
3. **RMU**: $\mathcal{L}(\theta) = L_{\text{misdirect}}(\theta) + \alpha \cdot L_{\text{preserve}}(\theta)$, steering activations of forget data toward random directions at the representation level.

After each unlearning step, the resulting model is distilled into a randomly initialized student of the same architecture. The result is that **post-distillation models are significantly more resistant to relearning across all attack intensities**, in some cases approaching the gold standard of data-filtered retraining.

### UNDO: Unlearn-Noise-Distill-on-Outputs

UNDO generalizes Unlearn-and-Distill into a three-step pipeline that introduces a tunable compute–robustness trade-off:

**Step 1 – Unlearn**: Apply a standard unlearning method to obtain a behavior-suppressed model (teacher).

**Step 2 – Noise**: Apply controlled perturbation to the teacher weights to produce the student initialization:

$$\theta_{\text{perturbed}} = (1 - \alpha) \theta_{\text{suppressed}} + \alpha \beta N$$

where $\alpha \in [0,1]$ is the interpolation coefficient, $\beta \in \mathbb{R}^+$ is the noise scale, and $N$ denotes noise sampled from Xavier initialization. The $(1-\alpha)$ term shrinks the original parameters while $\alpha\beta N$ injects noise. Setting $\alpha=0$ degenerates to no perturbation (no robustness gain), and $\alpha=1, \beta=1$ degenerates to full random initialization.

**Step 3 – Distill**: Distill the teacher outputs into the perturbed student using forward KL divergence to restore retain performance.

### Key Hyperparameter Analysis
- **$\alpha$ is the primary control knob**: Increasing $\alpha$ improves robustness at the cost of additional distillation compute, with a near-linear relationship.
- **$\beta$ has secondary influence**: For most values of $\alpha$, performance is similar across $\beta \in [0.05, 0.5]$; pure weight shrinkage ($\beta=0$) is also effective.
- **Recommended configuration**: $\alpha=0.6$, $\beta=0.1$, with a MaxEnt-based teacher.

## Key Experimental Results

### Experimental Setup
- Synthetic tasks: 100M-parameter Gemma 2 architecture; language task (English retain / Korean forget) and arithmetic task (addition–subtraction retain / multiplication–division forget).
- WMDP benchmark: Gemma-2-2B ($\alpha=0.25$, $\beta=0$); evaluation of bioweapon (WMDP-Bio) and cyberweapon (WMDP-Cyber) knowledge removal.

### Main Results (After 500-Step Relearning Attack)

| Method | Language Retain ↑ | Language Forget ↓ | Arithmetic Retain ↑ | Arithmetic Forget ↓ |
|---|---|---|---|---|
| GradDiff (unlearn only) | High | Rapid recovery | High | Rapid recovery |
| MaxEnt (unlearn only) | High | Rapid recovery | High | Rapid recovery |
| RepNoise | Medium | Partial recovery | Medium | Partial recovery |
| SAM | Medium | Partial recovery | Medium | Partial recovery |
| **UNDO (MaxEnt)** | **High** | **Minimal recovery** | **High** | **Minimal recovery** |
| Data-filtered retraining | High | Minimal | High | Minimal |

### Computational Efficiency of UNDO
- At $\alpha=0.6$, UNDO requires approximately **60–80%** of the compute needed for data-filtered retraining.
- Only **0.01%** of pretraining data needs to be annotated (for the unlearning phase).
- On WMDP, UNDO incurs an average MMLU drop of 4.88% (on the Pareto frontier).

### Effect of Perturbation Magnitude (Language Task, MaxEnt Teacher)

| $\alpha$ | Relative Robustness ↑ | Relative Compute ↑ |
|---|---|---|
| 0.2 | ~20% | ~15% |
| 0.4 | ~45% | ~35% |
| 0.6 | ~70% | ~55% |
| 0.8 | ~90% | ~75% |

Robustness is defined as $(P_{\text{UNDO}} - P_{\text{Unlearn Only}}) / (P_{\text{Data Filtering}} - P_{\text{Unlearn Only}})$.

## Highlights & Insights
- **Deep and actionable core insight**: The finding that "distillation discards latent capabilities" is both elegant and immediately practical — leading AI labs already employ distillation extensively to reduce inference costs, so robustness can be obtained simply by prepending an unlearning step before distillation.
- **Clear theoretical framing**: The distinction between behavioral suppression and genuine capability removal is made intuitive through the oracle-matching experiment, which directly illustrates the preservation of latent capabilities in parameter space.
- **The $\alpha$ knob in UNDO**: Provides a well-defined compute–robustness trade-off surface that can be flexibly navigated according to deployment requirements.
- **Rigorous experimental design**: Worst-case adversary evaluation across multiple learning rates, five random seeds, and validation across multiple tasks and domains.

## Limitations & Future Work
- **Retain performance degradation on WMDP**: An average MMLU drop of 4.88% may be unacceptable in production settings; the authors attribute this to using only 0.015% of pretraining data during distillation.
- **Distillation compute cost**: Although cheaper than data-filtered retraining, UNDO remains substantially more expensive than pure fine-tuning-based unlearning methods.
- **Limited model scale**: Synthetic experiments use only 100M parameters and WMDP experiments use only 2B parameters; behavior at 70B+ scale remains unvalidated.
- **Definition of unlearning targets**: The current framework assumes a clean partition between retain and forget sets, whereas capability boundaries are often blurry in practice.
- **Sensitivity to distillation data volume**: WMDP results suggest performance degrades when distillation data is insufficient, but the minimum data requirements have not been systematically studied.
- **Stronger attacks not considered**: Only relearning attacks (fine-tuning) are evaluated; more advanced capability recovery methods such as representation engineering and model surgery are not tested.

## Rating
- Novelty: ⭐⭐⭐⭐ The insight that distillation robustifies unlearning is novel and far-reaching; the noise interpolation design in UNDO is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across multiple tasks, methods, baselines, and attack intensities, with validation on the real-world WMDP benchmark.
- Writing Quality: ⭐⭐⭐⭐⭐ The paper is logically coherent, the progression from observations to method is natural and well-motivated, and the figures are well-designed.
- Value: ⭐⭐⭐⭐⭐ Provides a new technical pathway for capability removal in AI safety with direct industrial applicability.

## Related Work & Insights

## Highlights & Insights

## Rating

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] DRAGON: Guard LLM Unlearning in Context via Negative Detection and Reasoning](dragon_guard_llm_unlearning_in_context_via_negative_detection_and_reasoning.md)
- [\[ICLR 2026\] Reference-Guided Machine Unlearning](../../ICLR2026/model_compression/reference-guided_machine_unlearning.md)
- [\[NeurIPS 2025\] Single-Teacher View Augmentation: Boosting Knowledge Distillation via Angular Diversity](single-teacher_view_augmentation_boosting_knowledge_distillation_via_angular_div.md)
- [\[NeurIPS 2025\] Hyperbolic Dataset Distillation](hyperbolic_dataset_distillation.md)
- [\[NeurIPS 2025\] Optimizing Distributional Geometry Alignment with Optimal Transport for Generative Dataset Distillation](optimizing_distributional_geometry_alignment_with_optimal_transport_for_generati.md)

<!-- RELATED:END -->
