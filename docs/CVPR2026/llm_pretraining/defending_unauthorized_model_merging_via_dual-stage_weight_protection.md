---
title: >-
  [Paper Note] Defending Unauthorized Model Merging via Dual-Stage Weight Protection
description: >-
  [CVPR 2026][LLM Pretraining][Model Merging Defense] This paper proposes MergeGuard, a proactive dual-stage weight protection framework: Stage 1 disperses task-critical weights via L2 regularization, and Stage 2 injects structured perturbations to disrupt merging compatibility. The protected model retains <1.5% performance loss while causing merged model accuracy to drop by up to 90%.
tags:
  - CVPR 2026
  - LLM Pretraining
  - Model Merging Defense
  - Intellectual Property Protection
  - Weight Protection
  - Adversarial Perturbation
  - Model Security
date: 2026-05-08
content_hash: 367b6d98729b856c
---

# Defending Unauthorized Model Merging via Dual-Stage Weight Protection

**Conference**: CVPR 2026  
**arXiv**: [2511.11851](https://arxiv.org/abs/2511.11851)  
**Code**: N/A (not yet open-sourced)  
**Area**: LLM Pretraining  
**Keywords**: Model Merging Defense, Intellectual Property Protection, Weight Protection, Adversarial Perturbation, Model Security

## TL;DR
This paper proposes MergeGuard, a proactive dual-stage weight protection framework: Stage 1 disperses task-critical weights via L2 regularization, and Stage 2 injects structured perturbations to disrupt merging compatibility. The protected model retains <1.5% performance loss while causing merged model accuracy to drop by up to 90%.

## Background & Motivation
**Current Landscape**: The pretrain-finetune paradigm has become the cornerstone of modern AI, with open repositories such as Hugging Face and GitHub hosting thousands of publicly available models. Model merging techniques (WA, TA, TIES, AdaMerging, etc.) can directly construct multi-task models through parameter-level combination without additional training.

**Existing Pain Points**: Free-riders can download fine-tuned models under specific licenses and merge them to create new multi-capability models for redistribution or commercial use—parameter blending inherently conceals the origin of each weight, making intellectual property infringement difficult to trace and hold accountable.

**Core Tension**: Defenders can only modify their own model parameters and cannot anticipate the attacker's model, merging strategy, or target tasks, making defense highly uncertain. Simultaneously, a balance must be struck between preserving original task accuracy and suppressing merged model performance.

**Objective**: Design a proactive defense mechanism that causes any subsequent model merging to result in functional degradation while maintaining the original model's task performance.

**Approach**: The authors observe that modern merging methods (TIES, AdaMerging) rely on sparsification to disperse task parameters and reduce interference—MergeGuard exploits this in reverse by first dispersing and then perturbing weights to produce destructive interference during merging.

**Core Idea**: Through density-aware fine-tuning to disperse weights and adversarial weight negation to inject perturbations, the method reshapes parameter geometry in two stages so that merged models end up in incompatible curvature regions.

## Method

### Overall Architecture
MergeGuard processes the defender's model $\theta_{def}$ in two stages:

**Stage 1 (Density-Aware Finetuning)**: Training phase—L2 regularization uniformly disperses task-critical weights.

**Stage 2 (Adversarial Weight Negation)**: Training-free phase—selectively shifts task-relevant weight directions to disrupt merging compatibility.

### Key Designs
1. **Density-Aware Finetuning (Stage 1)**: An L2 regularization term is added on top of the standard cross-entropy loss:

$$L_{Total} = L_{CE} + \alpha \sum_{\ell=1}^{L} \|\theta^{(\ell)}\|_2^2$$

For vision tasks this is the classification loss + L2; for LLMs it is the next-token prediction loss + L2. The L2 regularization is applied independently per layer, effectively smoothing large weights and distributing important information evenly across layers—making subsequent merging operations unstable (dispersed weights are more easily amplified/diluted/interfered with).

2. **Adversarial Weight Negation (Stage 2)**: A training-free process with three steps:

    - Per-layer mask detection: mask each layer and measure accuracy drop; exclude the top $k'\%$ critical layers
    - Construct binary mask $M$: set critical layers and the least important $(1-k)(1-k')\%$ parameters to 0, the rest to 1
    - Shift along the task vector direction: $\hat{\theta}_{def} = \theta'_{def} - \beta \cdot M \odot \tau'_{def}$

   where $\tau'_{def} = \theta'_{def} - \theta_{pre}$ is the defender's task vector. In LLMs, embed_tokens, norm, and lm_head layers are excluded to prevent collapse.

3. **Theoretical Analysis**: The merging interference increment is approximated as:

$$\Delta \mathcal{L}_{merge} \approx \lambda_1 \lambda_2 \|\tau'_{def}\| \|\tau_{fr}\| (1 - \cos\phi)$$

Even a modest task vector rotation angle $\phi > 30°$ suffices to push the merged model out of the shared basin, producing destructive interference. Stage 1 decorrelates the eigenvectors corresponding to large Hessian eigenvalues, and Stage 2 further rotates the task vector direction.

### Hyperparameter Settings
All experiments use fixed $k'=10$, $k=0.1$, $\alpha=0.01$, $\beta=1$, requiring no task-specific tuning.

## Key Experimental Results

### Main Results: ViT-L-14 Image Classification—Accuracy Before/After Protection and After Merging

| Dataset | Finetuned $\theta_{def}$ | Protected $\hat{\theta}_{def}$ | Merged (TA) $\theta_{merge}$ | Merged (TA) $\hat{\theta}_{merge}$ |
|---------|----------------------|---------------------------|----------------------------|---------------------------------|
| RESISC45 | 97.37 | 97.25 | 86.6 | **56.50** |
| EuroSAT | 99.81 | 95.46 | 94.1 | **54.94** |
| GTSRB | 99.24 | 98.25 | 86.7 | **12.91** |
| MNIST | 99.69 | 99.27 | 98.9 | **11.35** |
| DTD | 84.15 | 82.16 | 65.6 | **46.65** |

The protected model incurs <4.4% self-accuracy loss, but merged accuracy plummets (GTSRB: 86.7→12.9, MNIST: 98.9→11.4).

### Baseline Comparison (Average Accuracy Drop Under TA Merging)

| Method | Protected Model Avg Acc | Avg Accuracy Drop After Merging |
|--------|------------------------|--------------------------------|
| PaRaMS (only baseline) | ~comparable to original | **30.76** |
| **MergeGuard (Ours)** | ~comparable (<1.5% loss) | **52.11** |

MergeGuard's average accuracy drop exceeds PaRaMS by **21.35 percentage points**.

### Key Findings
- The effect is even more pronounced on LLMs: Gemma2 drops from 69.6% to 1.52% on GSM8K, and from 64.02% to 21.34% on HumanEval
- Effective against all mainstream merging methods (WA/TA/TIES/ADA)—ADA is the hardest to defend against but still causes significant degradation
- Both adaptive attacks fail: (i) subtracting a scaled copy of the protected parameters; (ii) estimating the perturbation vector and applying orthogonal projection—both fail because task information is dispersed and the perturbation is unobservable
- Fixed hyperparameters work across tasks/architectures without task-specific tuning

## Highlights & Insights
- **Elegant defense strategy**: Exploits the sparsification assumption that merging methods rely on in reverse—first "anti-sparsify" to disperse weights, then inject directional adversarial perturbations
- **Strong theoretical support**: The curvature incompatibility analysis provides clear intuition (task subspace rotation + basin separation)
- **Highly practical**: Stage 2 is entirely training-free, hyperparameters are fixed, and it generalizes across ViT/Llama2/Gemma2/Mistral architectures
- **First systematic defense**: Besides PaRaMS, this is the only proactive defense work, and it significantly outperforms the prior art

## Limitations & Future Work
- Only targets full-parameter fine-tuning; not applicable to PEFT methods (LoRA, shallow tuning)—since PEFT does not expose a complete task vector
- L2 regularization may reduce model expressiveness on complex tasks (EuroSAT accuracy drops by 4.4%)
- The defense assumes the attacker uses parameter-level merging—if the attacker switches to knowledge distillation, the defense fails (acknowledged by the authors, who argue distillation is costly)
- Layer importance ranking in Stage 2 is based on global mask accuracy drop; finer-grained importance measures may be more effective

## Rating ⭐
- Novelty: ⭐⭐⭐⭐ — The dual-stage "disperse + rotate" defense concept is novel with clear theoretical analysis
- Experimental Rigor: ⭐⭐⭐⭐⭐ — Covers vision/language architectures, multiple merging methods, adaptive attacks, and SD image generation
- Writing Quality: ⭐⭐⭐⭐ — Clear adversarial scenario modeling and rigorous theoretical derivation
- Significance: ⭐⭐⭐⭐ — Important practical implications for model IP protection, filling the gap in proactive defense

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Model Merging in the Essential Subspace](model_merging_in_the_essential_subspace.md)
- [\[ICLR 2026\] Steering Language Models with Weight Arithmetic](../../ICLR2026/llm_pretraining/steering_language_models_with_weight_arithmetic.md)
- [\[CVPR 2026\] MXNorm: Reusing MXFP Block Scales for Efficient Tensor Normalisation](mxnorm_reusing_mxfp_block_scales_for_efficient_ten.md)
- [\[CVPR 2026\] LottieGPT: Tokenizing Vector Animation for Autoregressive Generation](lottiegpt_vector_animation_generation.md)
- [\[CVPR 2026\] Evidential Transformation Network: Turning Pretrained Models into Evidential Models for Post-hoc Uncertainty Estimation](evidential_transformation_network_post_hoc_uncertainty_estimation.md)

<!-- RELATED:END -->
