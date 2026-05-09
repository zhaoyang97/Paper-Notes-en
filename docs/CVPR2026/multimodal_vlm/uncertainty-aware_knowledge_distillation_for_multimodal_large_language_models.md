---
title: >-
  [Paper Note] Uncertainty-Aware Knowledge Distillation for Multimodal Large Language Models
description: >-
  [CVPR2026][Multimodal VLM][Knowledge Distillation] This paper proposes Beta-KD, an uncertainty-aware knowledge distillation framework grounded in a Bayesian perspective. By modeling teacher supervision as a Gibbs prior and deriving a closed-form solution via Laplace approximation, Beta-KD automatically balances data and teacher signals, consistently improving distillation performance on multimodal VQA benchmarks.
tags:
  - CVPR2026
  - Multimodal VLM
  - Knowledge Distillation
  - Uncertainty Weighting
  - Bayesian Inference
  - Gibbs Prior
  - Multi-task Balancing
date: 2026-05-08
content_hash: bd9eddcca5aa12b6
---

# Uncertainty-Aware Knowledge Distillation for Multimodal Large Language Models

**Conference**: CVPR2026  
**arXiv**: [2603.21426](https://arxiv.org/abs/2603.21426)  
**Code**: [github.com/Jingchensun/beta-kd](https://github.com/Jingchensun/beta-kd)  
**Area**: Multimodal VLM  
**Keywords**: Knowledge Distillation, Uncertainty Weighting, Bayesian Inference, Gibbs Prior, Multi-task Balancing

## TL;DR
This paper proposes Beta-KD, an uncertainty-aware knowledge distillation framework grounded in a Bayesian perspective. By modeling teacher supervision as a Gibbs prior and deriving a closed-form solution via Laplace approximation, Beta-KD automatically balances data and teacher signals, consistently improving distillation performance on multimodal VQA benchmarks.

## Background & Motivation
Knowledge distillation (KD) is a core technique for compressing large models, yet it faces unique challenges in the context of multimodal LLM distillation:

- **Multi-loss balancing**: Distillation losses involve multiple components — cross-entropy (learning from data), KL divergence (learning teacher distributions), feature alignment losses, etc. — each with different scales, gradients, and optimization dynamics.
- **Capacity gap**: The large capacity disparity between teacher and student leads to inconsistent scales and variances in logits and hidden representations.
- **High cost of weight search**: Grid search over loss weights is impractical for large-scale LLMs.

**Core Problem**: How can one automatically balance data supervision and teacher supervision without manually tuning loss weights?

## Method

### Overall Architecture
KD is formulated as a MAP inference problem over student activations, where teacher information serves as a Gibbs prior. The partition function is simplified via Laplace approximation, and the inference parameter $\beta$ is parameterized by a neural network.

### Key Designs

1. **Teacher-Informed Gibbs Prior**:

    - $p(a^s | a^t, \beta) = \frac{1}{Z_\beta(a^t)} \exp[-\beta \ell(a^s; a^t)]$
    - $\ell$ can be any alignment energy (FKL, RKL, Cosine, MSE, etc.)
    - $\beta$ controls supervision strength: larger $\beta$ implies greater trust in the teacher; smaller $\beta$ implies greater trust in the data.

2. **MAP Inference and Laplace Approximation**:

    - MAP objective: $\min_{a^s} -\log p(y|a^s) + \beta\ell(a^s;a^t) + \log Z_\beta(a^t)$
    - After Laplace approximation: $\log Z_\beta \approx -d/2 \cdot \log\beta + \text{const}$
    - Final objective: $\min \mathcal{L}_{CE} + \beta \ell + \frac{d}{2}\log\beta$ (with natural regularization)

3. **Two Uncertainty Granularities**:

    - **Task-level (homoscedastic)**: $\beta$ is a shared learnable scalar per task.
    - **Instance-level (heteroscedastic)**: $\beta(x) = g_\phi(h(x)) > 0$, predicted by a lightweight network from the input.
    - Instance-level uncertainty allows each sample to have its own data–teacher balance.

4. **Energy Function Design Space Exploration**:

    - Cosine-Probs is found to perform best (scale-invariant, focuses on directional alignment).
    - Pre-softmax logit matching (MSE-Logits, Cosine-Logits) performs poorly for generative MLLMs.
    - This finding contrasts with conclusions drawn from discriminative tasks.

### Loss & Training
$$\min_{\theta,\phi} \mathcal{L}_{CE}(\theta) + g_\phi(h(x))\ell(\theta) - \frac{d}{2}\log g_\phi(h(x))$$

The visual encoder and tokenizer are frozen; only the language backbone is fine-tuned.

## Key Experimental Results

### Main Results

| Method | ScienceQA VQA-Acc | ScienceQA IMG-Acc | Gain |
|------|-----------------|------------------|------|
| CE+JS | 48.5 | 54.8 | Baseline |
| CE+JS w/ Beta-KD(Task) | 50.5(+1.1) | 58.1(+1.7) | Task-level |
| CE+JS w/ Beta-KD(Instance) | 53.3(+3.9) | 66.9(+10.6) | Instance-level |

### Ablation Study

| Configuration | Key Metric | Note |
|------|---------|------|
| FKL/RKL/JS/TVD and other losses | All show improvement | Method is robust to loss function choice |
| Task-level vs. Instance-level | Instance-level superior | Fine-grained adaptation is beneficial |
| 2-loss vs. 3-loss | Both effective | Compatible with arbitrary combinations |

### Key Findings
- Instance-level uncertainty weighting achieves up to +4.7 absolute points improvement on ScienceQA.
- Gains are larger on IMG-Acc (+10.6), suggesting greater benefit for visually grounded questions.
- Logit-level matching fails for generative MLLMs, contrary to findings in discriminative settings.
- Training dynamics visualization reveals faster convergence, smoother optimization, and closer teacher–student logit alignment.

## Highlights & Insights
- The Bayesian formulation provides an elegant theoretical interpretation of KD: teacher supervision as a Gibbs prior, distillation as MAP inference.
- The Laplace approximation naturally yields the $-\frac{d}{2}\log\beta$ regularization term, preventing $\beta$ from becoming extreme.
- The energy function design space exploration offers practical guidance: Cosine-Probs is the preferred choice.
- The method is elegantly designed, with a coherent logical flow from theoretical derivation to implementation.

## Limitations & Future Work
- The instance-level uncertainty network introduces additional parameters and computational overhead.
- Experiments are primarily conducted on MobileVLM; validation on larger-scale teachers is limited.
- The Laplace approximation assumes local quadratic structure, which may be inaccurate on non-convex losses.
- Integration with more recent backbone models (e.g., Qwen2.5-VL) has not been explored.

## Related Work & Insights
- Related to Kendall & Gal's multi-task uncertainty weighting, but generalized to arbitrary distillation losses.
- Multimodal KD methods such as LLaVA-KD and Align-KD can benefit from this framework.
- BayesKD targets uncertainty over model parameters, while Beta-KD targets uncertainty over activations — representing complementary perspectives.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The theoretical framework combining Gibbs prior and Laplace approximation is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multiple loss combinations, two uncertainty granularities, and six benchmarks.
- **Writing Quality**: ⭐⭐⭐⭐ Theoretical derivations are clear and rigorous.
- **Value**: ⭐⭐⭐⭐ Automatic loss balancing is highly practical for large-model KD.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Purify-then-Align: Towards Robust Human Sensing under Modality Missing with Knowledge Distillation from Noisy Multimodal Teacher](purify-then-align_towards_robust_human_sensing_under_modality_missing_with_knowl.md)
- [\[CVPR 2026\] MASQuant: Modality-Aware Smoothing Quantization for Multimodal Large Language Models](masquant_modality-aware_smoothing_quantization_for_multimodal_large_language_mod.md)
- [\[CVPR 2026\] CoVFT: Context-aware Visual Fine-tuning for Multimodal Large Language Models](covft_context-aware_visual_fine-tuning_for_multimodal_large_language_models.md)
- [\[AAAI 2026\] FT-NCFM: An Influence-Aware Data Distillation Framework for Efficient VLA Models](../../AAAI2026/multimodal_vlm/ft-ncfm_an_influence-aware_data_distillation_framework_for_efficient_vla_models.md)
- [\[ICLR 2026\] Detecting Misbehaviors of Large Vision-Language Models by Evidential Uncertainty Quantification](../../ICLR2026/multimodal_vlm/detecting_misbehaviors_of_large_vision-language_models_by_evidential_uncertainty.md)

</div>

<!-- RELATED:END -->
