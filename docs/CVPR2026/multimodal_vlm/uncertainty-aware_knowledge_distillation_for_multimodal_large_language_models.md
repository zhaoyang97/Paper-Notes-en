---
title: >-
  [Paper Note] Uncertainty-Aware Knowledge Distillation for Multimodal Large Language Models
description: >-
  [CVPR2026][Multimodal VLM][Knowledge Distillation] Proposes Beta-KD, an uncertainty-aware knowledge distillation framework from a Bayesian perspective. By modeling teacher supervision as a Gibbs prior and deriving a closed-form solution via Laplace approximation, it automatically adjusts the balance between data and teacher signals, consistently improving distillation performance on multimodal VQA benchmarks.
tags:
  - "CVPR2026"
  - "Multimodal VLM"
  - "Knowledge Distillation"
  - "Uncertainty Weighting"
  - "Bayesian Inference"
  - "Gibbs Prior"
  - "Multi-task Balancing"
date: 2026-05-08
content_hash: a5e682a5c596b541
---

# Uncertainty-Aware Knowledge Distillation for Multimodal Large Language Models

**Conference**: CVPR2026  
**arXiv**: [2603.21426](https://arxiv.org/abs/2603.21426)  
**Code**: [github.com/Jingchensun/beta-kd](https://github.com/Jingchensun/beta-kd)  
**Area**: Multimodal VLM  
**Keywords**: Knowledge Distillation, Uncertainty Weighting, Bayesian Inference, Gibbs Prior, Multi-task Balancing

## TL;DR
Proposes Beta-KD, an uncertainty-aware knowledge distillation framework from a Bayesian perspective. By modeling teacher supervision as a Gibbs prior and deriving a closed-form solution via Laplace approximation, it automatically adjusts the balance between data and teacher signals, consistently improving distillation performance on multimodal VQA benchmarks.

## Background & Motivation
Knowledge Distillation (KD) is a core technology for compressing large models, but it faces specific challenges in Multimodal LLM (MLLM) distillation:

- **Multi-loss Balancing Difficulty**: Distillation losses involve multiple channels—cross-entropy (learning from data), KL divergence (learning from teacher distributions), and feature alignment—each with different scales, gradients, and optimization dynamics.
- **Capacity Gap**: The significant capacity difference between teacher and student models leads to inconsistent scales and variances in logits and hidden representations.
- **High Cost of Weight Search**: Grid searching for weights is impractical for large-scale LLMs.

**Core Problem**: How to automatically balance data supervision and teacher supervision without manual weight tuning?

## Method

### Overall Architecture

Distillation for MLLMs requires simultaneous learning from data (cross-entropy) and the teacher (KL/feature alignment). These losses vary in scale and gradient, making manual weight tuning expensive and difficult. Beta-KD takes a different approach: it treats "what the student should look like" as a MAP (Maximum A Posteriori) inference problem. Teacher information is injected as a Gibbs prior, and Laplace approximation is used to simplify the intractable partition function into a closed-form solution. Finally, a lightweight network predicts the balancing coefficient $\beta$, eliminating grid search and automatically weighting the data and teacher supervision paths.

### Key Designs

**1. Teacher-Informed Gibbs Prior: Writing "Teacher Trust" as an Adjustable Temperature**

To balance data and teacher automatically, teacher supervision is formalized. Beta-KD formulates the teacher's constraint on student activations $a^s$ as a Gibbs prior $p(a^s \mid a^t, \beta) = \frac{1}{Z_\beta(a^t)} \exp[-\beta\,\ell(a^s; a^t)]$, where the alignment energy $\ell$ can take any form such as FKL, RKL, Cosine, or MSE. The coefficient $\beta$ represents the "degree of trust in the teacher": a large $\beta$ indicates higher trust in the teacher's distribution, while a small $\beta$ indicates higher trust in the data itself—reducing the balancing problem to a learnable scalar.

**2. MAP Inference + Laplace Approximation: Naturally Deriving a Regularization Term**

Viewing distillation as a MAP inference over student activations, the objective is $\min_{a^s} -\log p(y\mid a^s) + \beta\ell(a^s; a^t) + \log Z_\beta(a^t)$. The difficulty lies in the partition function $Z_\beta$. Using Laplace approximation, $\log Z_\beta \approx -\frac{d}{2}\log \beta + \text{const}$ is obtained. Substituting this back yields the final objective: $\min \mathcal{L}_{CE} + \beta\ell - \frac{d}{2}\log\beta$. The term $-\frac{d}{2}\log\beta$ is a regularization term naturally derived from the theoretical framework, preventing $\beta$ from collapsing to 0 or infinity.

**3. Task-level and Instance-level Granularity: Sample-Specific Balancing**

A fixed global $\beta$ is too coarse, as different samples rely on the teacher differently. Beta-KD provides two granularities: task-level (homoscedastic), where $\beta$ is a shared learnable scalar per task; and instance-level (heteroscedastic), where a lightweight network predicts $\beta(x) = g_\phi(h(x)) > 0$ from the input. Experiments show that instance-level adaptation is significantly stronger, justifying the value of sample-specific balancing.

**4. Energy Function Design Space: Generative MLLMs Prefer Cosine-Probs**

The choice of $\ell$ is critical. The paper systematically explores the design space and finds that Cosine-Probs performs best—it is scale-invariant and focuses on directional alignment, avoiding issues with logit scale/variance inconsistencies caused by the capacity gap. Conversely, pre-softmax logit matching (MSE-Logits, Cosine-Logits) performs poorly on generative MLLMs, contradicting common findings in discriminative tasks.

### Loss & Training

Final optimization objective (Instance-level):
$$\min_{\theta,\phi} \mathcal{L}_{CE}(\theta) + g_\phi(h(x))\,\ell(\theta) - \frac{d}{2}\log g_\phi(h(x))$$
During training, the visual encoder and tokenizer are frozen, and only the language backbone is fine-tuned.

## Key Experimental Results

### Main Results

| Method | ScienceQA VQA-Acc | ScienceQA IMG-Acc | Gain |
|------|-----------------|------------------|------|
| CE+JS | 48.5 | 54.8 | Baseline |
| CE+JS w/ Beta-KD(Task) | 50.5(+1.1) | 58.1(+1.7) | Task-level |
| CE+JS w/ Beta-KD(Instance) | 53.3(+3.9) | 66.9(+10.6) | Instance-level |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Various losses (FKL/RKL/JS/TVD) | Consistent gains | Methodology is robust to energy function choice |
| Task-level vs. Instance-level | Instance-level is superior | Fine-grained adaptation is valuable |
| 2-loss vs. 3-loss | Effective in both | Applicable to any loss combination |

### Key Findings
- Instance-level uncertainty weighting improves ScienceQA performance by up to +4.7 absolute points.
- The improvement on IMG-Acc is even larger (+10.6), indicating greater assistance for vision-related questions.
- Logit-level matching fails in generative MLLMs, unlike in discriminative models.
- Training dynamics visualization shows faster convergence, smoother optimization, and closer teacher-student logit alignment.

## Highlights & Insights
- Elegant theoretical explanation of KD from a unified Bayesian perspective: teacher supervision as a Gibbs prior and distillation as MAP inference.
- Laplace approximation yields the $-\frac{d}{2}\log\beta$ regularization term, naturally preventing extreme values of $\beta$.
- Exploration of the energy function design space provides useful practical guidelines: Cosine-Probs is the most effective.
- Logical consistency from theoretical derivation to implementation.

## Limitations & Future Work
- The instance-level uncertainty network adds parameters and computational overhead.
- Experiments are primarily based on MobileVLM; validation with larger-scale teachers is limited.
- Laplace approximation assumes local quadratic approximation, which may be imprecise for non-convex losses.
- Integration with newer base models (e.g., Qwen2.5-VL) has not been verified.

## Related Work & Insights
- Related to multi-task uncertainty weighting by Kendall & Gal, but generalized to arbitrary distillation losses.
- Multimodal KD methods like LLaVA-KD and Align-KD could benefit from this approach.
- Unlike BayesKD, which focuses on model parameter uncertainty, Beta-KD focuses on activation uncertainty.

## Rating
- Novelty: ⭐⭐⭐⭐ Innovative theoretical framework using Gibbs prior and Laplace approximation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple loss combinations, two granularities, and 6 benchmarks.
- Writing Quality: ⭐⭐⭐⭐ Clear and rigorous theoretical derivation.
- Value: ⭐⭐⭐⭐ Automatic loss balancing is highly practical for large model KD.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Direction-aware 3D Large Multimodal Models](direction-aware_3d_large_multimodal_models.md)
- [\[CVPR 2026\] CC-VQA: Conflict- and Correlation-Aware Method for Mitigating Knowledge Conflict in Knowledge-Based Visual Question Answering](cc-vqa_conflict-_and_correlation-aware_method_for_mitigating_knowledge_conflict_.md)
- [\[CVPR 2026\] CoVFT: Context-aware Visual Fine-tuning for Multimodal Large Language Models](covft_context-aware_visual_fine-tuning_for_multimodal_large_language_models.md)
- [\[CVPR 2026\] Beyond Missing Modalities: Hypergraph Guided Diffusion for Uncertainty-Aware Multimodal Emotion Recognition](beyond_missing_modalities_hypergraph_conditioned_diffusion_for_uncertainty-aware.md)
- [\[ICML 2026\] TUR-DPO: Topology- and Uncertainty-Aware Direct Preference Optimization](../../ICML2026/multimodal_vlm/tur-dpo_topology-_and_uncertainty-aware_direct_preference_optimization.md)

</div>

<!-- RELATED:END -->
