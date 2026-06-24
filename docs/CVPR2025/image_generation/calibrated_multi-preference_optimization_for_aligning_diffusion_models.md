---
title: >-
  [Paper Note] Calibrated Multi-Preference Optimization for Aligning Diffusion Models
description: >-
  [CVPR 2025][Image Generation][Calibrated Preference Optimization] This paper proposes Calibrated Preference Optimization (CaPO), which unifies scores from different reward models into expected win-rates through win-rate calibration, and designs a frontier-based rejection sampling (FRS) strategy based on the Pareto frontier to handle conflicts between multiple reward signals, consistently outperforming DPO and IPO methods on SDXL and SD3-Medium.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Calibrated Preference Optimization"
  - "Multi-Reward Alignment"
  - "Pareto Frontier"
  - "Diffusion Model Fine-tuning"
  - "Win-rate Calibration"
date: 2026-05-08
content_hash: 172b79a58b1adfe9
---

# Calibrated Multi-Preference Optimization for Aligning Diffusion Models

**Conference**: CVPR 2025  
**arXiv**: [2502.02588](https://arxiv.org/abs/2502.02588)  
**Code**: [https://kyungmnlee.github.io/capo.github.io/](https://kyungmnlee.github.io/capo.github.io/)  
**Area**: RLHF Alignment / Diffusion Models  
**Keywords**: Calibrated Preference Optimization, Multi-Reward Alignment, Pareto Frontier, Diffusion Model Fine-tuning, Win-rate Calibration

## TL;DR

This paper proposes Calibrated Preference Optimization (CaPO), which unifies scores from different reward models into expected win-rates through win-rate calibration, and designs a frontier-based rejection sampling (FRS) strategy based on the Pareto frontier to handle conflicts between multiple reward signals, consistently outperforming DPO and IPO methods on SDXL and SD3-Medium.

## Background & Motivation

**Background**: Text-to-image (T2I) diffusion model alignment with human preferences is an important research direction. Diffusion-DPO successfully applies DPO to large-scale diffusion models, but requires expensive human-annotated preference datasets. An alternative is to use multiple reward models to simulate human preferences and generate training data, thereby avoiding manual annotation costs.

**Limitations of Prior Work**: There are three main issues when using multiple reward models for preference optimization. **First, underutilization of pairwise info**: current methods only use reward models to determine binary preferences (win/lose), discarding the rich information in reward scores (such as the magnitude of quality gaps). **Second, lack of multi-reward generalization**: existing methods are mainly designed for a single reward, and directly using weighted sums of rewards to handle multiple rewards leads to suboptimal results because of large differences in score ranges and distributions across different reward models. **Third, conflicts between rewards**: for example, optimizing aesthetic rewards often degrades prompt alignment quality, and the inconsistency between different reward signals causes simple linear combinations to fail.

**Key Challenge**: The black-box score distributions of multiple reward models are inconsistent—some have a range of [0, 1], while others might be [-10, 10]. The absolute values of the scores do not reflect the true quality of the samples (reward values under the Bradley-Terry model can be high, yet the actual quality might be poor).

**Goal**: 1) How to calibrate scores from different reward models to a unified scale? 2) How to select training sample pairs when multiple rewards conflict? 3) How to avoid reward hacking of a single reward?

**Key Insight**: The authors propose replacing raw reward values with the "expected win-rate against the reference model." The win-rate naturally lies within the range of [0, 1], providing a unified metric for different reward models. The win-rate is approximately calculated through pairwise comparisons to obtain calibrated reward signals. For multi-reward scenarios, non-dominated sorting is used to find the Pareto frontier for training pair selection.

**Core Idea**: Using win-rate calibration instead of raw reward values, and Pareto frontier sampling instead of weighted sum, to achieve multi-reward diffusion model alignment.

## Method

### Overall Architecture

The training pipeline of CaPO is as follows: (a) generate $N$ images for each prompt using a pre-trained T2I model, and score them using multiple reward models; (b) calculate the calibrated reward for each image—approximating the win-rate through pairwise comparisons with the other $N-1$ images; (c) select training pairs—choosing samples with the highest/lowest calibrated rewards for single reward, or using Pareto frontier sampling for multiple rewards; (d) minimize the gap between the calibrated reward difference and the implicit reward difference during training using a regression loss.

### Key Designs

1. **Win-Rate Calibrated Rewards**:

    - **Function**: Unify the scores of different reward models to the [0, 1] range to eliminate distribution inconsistency.
    - **Mechanism**: For a given prompt $c$, generate $N$ samples $\{x_i\}_{i=1}^N$ from the reference model. Define the win-rate of data $x$ against the distribution $p(\cdot|c)$ as $\mathbb{P}(x \succ p | c) = \mathbb{E}_{x' \sim p}[\mathbb{P}(x \succ x'|c)]$. Under the Bradley-Terry model, the pairwise win-rate is $\mathbb{P}(x \succ x'|c) = \sigma(R(x,c) - R(x',c))$. Therefore, the calibrated reward of sample $x_i$ is the average pairwise win-rate against all other samples: $R_{\text{ca}}(x_i, c) = \frac{1}{N-1}\sum_{j \neq i} \sigma(R(x_i, c) - R(x_j, c))$. When $N$ is sufficiently large, $R_{\text{ca}}$ approximates the true win-rate against the reference model. The win-rate is naturally bounded and comparable, resolving the issue of inconsistent score ranges from different reward models.
    - **Design Motivation**: Even if raw Bradley-Terry reward values have high prediction accuracy, they do not necessarily reflect the absolute quality of the sample. For instance, in a batch of low-quality samples, the "best" one might score high but still have poor actual quality. The win-rate more accurately reflects "how much better a sample is compared to the reference model."

2. **Frontier-based Rejection Sampling (FRS)**:

    - **Function**: Select training pairs that balance all reward signals in multi-reward scenarios.
    - **Mechanism**: Given calibrated scores from $L$ reward models, construct an $L$-dimensional vector for each of the $N$ generated samples of a prompt. Use the non-dominated sorting algorithm to find the upper Pareto frontier (positive set $X^+$) and the lower Pareto frontier (negative set $X^-$). Specifically, $x$ dominates $x'$ if and only if $R_{\text{ca}}^{(j)}(x) \geq R_{\text{ca}}^{(j)}(x')$ holds for all $j=1,...,L$. The upper Pareto set is the set of non-dominated points, and the lower Pareto set is the set of dominated points. During training, positive samples are sampled from $X^+$ and negative samples from $X^-$. The optimization objective uses the average calibrated score of the multiple rewards: $R_{\text{ca}}(x,c) = \frac{1}{L}\sum_{j=1}^{L} R_{\text{ca}}^{(j)}(x,c)$.
    - **Design Motivation**: Weighted sum methods require manual weight tuning, and fixed weights may not be appropriate for all prompts. The Pareto frontier is a classic concept in multi-objective optimization, which automatically identifies samples that are good/bad in all dimensions without requiring preset weights. Pushing the model away from the lower Pareto frontier and towards the upper Pareto frontier naturally achieves multi-objective balance.

3. **CaPO Regression Loss**:

    - **Function**: Utilize the calibrated reward difference as a dynamic target to guide preference learning of the diffusion model.
    - **Mechanism**: Unlike the log-sigmoid loss of standard DPO (which implicitly assumes $\Delta R = 1$), CaPO uses a regression loss to match the calibrated reward difference: $\mathcal{L}_{\text{CaPO}}(\theta) = \mathbb{E}_{t,\epsilon,\epsilon'}\left[\left(R_{\text{ca}}(x^+, c) - R_{\text{ca}}(x^-, c) - \beta(R_\theta(x_t^+, c, t) - R_\theta(x_t^-, c, t))\right)^2\right]$. Here, $R_\theta$ is the implicit reward of the diffusion model, defined by the difference in $\epsilon$-prediction loss. CaPO is a generalization of IPO (which fixes $\Delta R = 1$), avoiding over-optimization through a dynamic objective—when the calibrated reward difference is small, the model is not forced to learn an excessively large preference gap.
    - **Design Motivation**: IPO with a fixed $\Delta R = 1$ applies learning signals of the same intensity to all preference pairs, ignoring the magnitude of the quality gap between the pairs. CaPO's dynamic objective makes learning more fine-grained—large gap pairs provide stronger signals, and small gap pairs provide gentler signals.

### Loss Weighting

A monotonically decreasing sigmoid weight function $w_t = \sigma(-\lambda_t + b)$ is adopted, where $\lambda_t$ is the log-SNR and $b$ is a bias parameter. The weight is large at high noise (low $\lambda_t$) and small at low noise (high $\lambda_t$). This is equivalent to using a weighted ELBO instead of KL divergence as the regularization term.

## Key Experimental Results

### Single-Reward Experiments (SDXL, win-rate % against baseline)

| Fine-tuning Reward → | MPS | VQAscore | VILA |
|------------|-----|----------|------|
| DPO | 58.5 / 49.3 / 61.7 | 53.1 / 50.6 / 55.9 | 52.6 / 46.4 / 81.8 |
| IPO | 56.8 / 50.1 / 64.1 | 53.1 / 51.9 / 53.8 | 53.3 / 48.5 / 76.1 |
| **CaPO** | **61.1** / 49.7 / **64.9** | **55.5** / **53.2** / **58.7** | **54.1** / **49.6** / **83.1** |

### Multi-Reward Experiments (SDXL)

| Target | Method | MPS Win% | VQA Win% | VILA Win% |
|------|------|----------|----------|-----------|
| DPO | SUM | 57.2 | 52.1 | 71.9 |
| DPO | FRS | 58.1 | 52.9 | 78.6 |
| CaPO | SUM | 61.2 | 52.5 | 75.0 |
| **CaPO** | **FRS** | **61.2** | **54.6** | **79.2** |

### GenEval Benchmark (Text-To-Image Alignment)

| Model | Overall |
|------|---------|
| SDXL | 0.55 |
| CaPO+SDXL | **0.59** |
| SD3-M | 0.68 |
| CaPO+SD3-M | **0.71** |

### Key Findings

- CaPO consistently outperforms DPO and IPO in both single-reward and multi-reward settings, showing that win-rate calibration indeed provides better training signals.
- FRS (Pareto frontier sampling) is more effective than simple weighted sum (SUM) and model soup (SOUP) in multi-reward scenarios, with the most pronounced improvement in the VILA (aesthetic) dimension.
- CaPO+FRS achieves simultaneous improvements in all three reward dimensions on SDXL, avoiding the typical "see-saw" effect when optimizing a single reward.
- CaPO improves overall performance for both SDXL and SD3-M on GenEval, showing that prompt alignment is not sacrificed while improving visual quality.
- The performance on SD3-Medium is equally stable, proving the effectiveness of the method across different diffusion model architectures (U-Net vs. DiT).

## Highlights & Insights

- **Win-rate calibration is a simple yet powerful idea**: Converting incomparable reward scores into a unified "how much better than reference model" metric is elegant and theoretically grounded.
- **Pareto frontier avoids manual weight tuning**: Non-dominated sorting is a classic tool in multi-objective optimization, which was rarely used in the diffusion model preference optimization domain before, making it a valuable introduction.
- **CaPO as a generalization of IPO**: The dynamic objective $\Delta R$ is more reasonable than a fixed target. It avoids the instability of DPO and is more fine-grained than IPO.
- **No human-annotated data required**: It relies entirely on reward model scoring, significantly reducing data costs.

## Limitations & Future Work

- It relies on the quality of the multiple reward models—if the reward models themselves are inaccurate or systematically biased, the calibrated results may still be suboptimal.
- The data preparation costs (high GPU overhead) for generating $N$ samples and evaluating multiple reward scores are relatively high.
- The experiments are mainly validated on SDXL and SD3-M, and have not yet been tested on larger-scale models (such as Flux, DALL-E 3).
- The Pareto frontier may degenerate in high-dimensional reward spaces (when $L$ is very large)—where most points become non-dominated, causing the sampling strategy to fail.
- No direct comparison has been made with RLHF methods (such as DDPO, ReFL).
- There is a lack of large-scale human evaluation; the consistency between automatic metrics and true human preferences has not been fully verified.

## Related Work & Insights

- **Diffusion-DPO (Wallace et al., 2023)**: Introduces DPO into diffusion models. On top of this, CaPO addresses reward calibration and multi-reward issues.
- **IPO (Azar et al., 2024)**: Proposes a general preference optimization framework. CaPO is its extension (dynamizing the fixed $\Delta R=1$ objective).
- **Rewarded Soups (Rame et al., 2024)**: Achieves multi-reward optimization by merging single-reward fine-tuned models. CaPO's joint optimization approach is superior.
- **DPOK / DDPO**: Fine-tuning diffusion models based on policy gradients, which is computationally more expensive.
- The win-rate calibration concept of CaPO can be generalized to LLM alignment—any scenario involving multi-reward DPO can benefit from this unified calibration method.

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Practicality: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Curriculum Direct Preference Optimization for Diffusion and Consistency Models](curriculum_direct_preference_optimization_for_diffusion_and_consistency_models.md)
- [\[CVPR 2025\] Personalized Preference Fine-tuning of Diffusion Models](personalized_preference_fine-tuning_of_diffusion_models.md)
- [\[ICML 2025\] Smoothed Preference Optimization via ReNoise Inversion for Aligning Diffusion Models with Varied Human Preferences](../../ICML2025/image_generation/smoothed_preference_optimization_via_renoise_inversion_for_aligning_diffusion_mo.md)
- [\[CVPR 2025\] Aesthetic Post-Training Diffusion Models from Generic Preferences with Step-by-step Preference Optimization](aesthetic_post-training_diffusion_models_from_generic_preferences_with_step-by-s.md)
- [\[AAAI 2026\] Margin-aware Preference Optimization for Aligning Diffusion Models without Reference](../../AAAI2026/image_generation/margin-aware_preference_optimization_for_aligning_diffusion_models_without_refer.md)

</div>

<!-- RELATED:END -->
