---
title: >-
  [Paper Note] How to Set AdamW's Weight Decay as You Scale Model and Dataset Size
description: >-
  [ICML 2025][Recommender Systems][AdamW] By interpreting the weight updates of AdamW as an Exponential Moving Average (EMA), this work reveals that the EMA timescale $\tau = 1/(\eta\lambda)$ is a core hyperparameter. Its optimal value in terms of epochs remains stable across varying model and dataset scales, thereby providing clear scaling rules for weight decay.
tags:
  - "ICML 2025"
  - "Recommender Systems"
  - "AdamW"
  - "weight decay"
  - "hyperparameter transfer"
  - "µP"
  - "exponential moving average"
date: 2026-05-08
content_hash: 75c8037ac807b653
---

# How to Set AdamW's Weight Decay as You Scale Model and Dataset Size

**Conference**: ICML 2025  
**arXiv**: [2405.13698](https://arxiv.org/abs/2405.13698)  
**Code**: None  
**Area**: Recommendation Systems  
**Keywords**: AdamW, weight decay, hyperparameter transfer, µP, exponential moving average

## TL;DR

By interpreting the weight updates of AdamW as an Exponential Moving Average (EMA), this work reveals that the EMA timescale $\tau = 1/(\eta\lambda)$ is a core hyperparameter. Its optimal value in terms of epochs remains stable across varying model and dataset scales, thereby providing clear scaling rules for weight decay.

## Background & Motivation

In the common workflow of large-scale model training, researchers typically prototype on a small scale first, then transfer the optimal hyperparameters to large-scale training. For learning rates, works like µP (Yang et al., 2022) have provided clear scaling rules with respect to width ($\eta \propto 1/\text{fan\_in}$). However, when utilizing the AdamW optimizer, **how the weight decay hyperparameter $\lambda$ scales with model size and dataset size** remains an open question, lacking systematic theoretical understanding and experimental validation.

In current practice, weight decay is typically set to a constant (e.g., 0.1). However, in reality:

- As the dataset grows larger, a fixed $\lambda$ leads to performance degradation.
- As the model becomes wider, the learning rate scaling of µP breaks down due to a fixed $\lambda$.
- There is a lack of a unified theoretical framework to guide the scaling of $\lambda$.

The core motivation of this paper is to fill this gap, providing theoretically grounded and experimentally validated weight decay scaling rules.

## Method

### Overall Architecture

The core idea of this paper is: **the weights in AdamW can themselves be understood as an exponential moving average (EMA) of recent updates**. Note that this does not refer to the EMA used in Adam to compute the first moment $\hat{m}_t$ and second moment $\hat{v}_t$; rather, the entire weight update process itself constitutes an EMA.

The update formula for AdamW is:

$$w_t = (1 - \eta_t \lambda) w_{t-1} - \eta_t \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Comparing this with the standard EMA formula $\text{ema}_t = (1 - 1/\tau) \text{ema}_{t-1} + (1/\tau) q_t$, the following mapping can be established:

| EMA Variable | AdamW Equivalent | Meaning |
|:---|:---|:---|
| $1/\tau_{\text{iter}}$ | $\eta_t \lambda$ | EMA Forgetting Rate |
| $\text{ema}_t$ | $w_t$ | Current Weight |
| $q_t$ | $-\frac{1}{\lambda}\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}$ | Scaled Update Amount |

Therefore, the EMA timescale is $\tau_{\text{iter}} = 1/(\eta\lambda)$, which intuitively represents the number of recent iteration updates that the EMA averages over.

### Key Designs

#### 1. From Iteration Timescale to Epoch Timescale

To measure what fraction of the dataset the EMA averages over, the authors introduce the epoch timescale:

$$\tau_{\text{epoch}} = \tau_{\text{iter}} / M = \frac{1}{\eta \lambda N / B}$$

where $N$ is the dataset size, $B$ is the batch size, and $M = N/B$ is the number of iterations per epoch. $\tau_{\text{epoch}}$ measures how many epochs of past updates the EMA of AdamW averages over.

**Core Finding**: The optimal $\tau_{\text{epoch}}$ remains approximately constant as model and dataset scales change.

#### 2. Scaling of Weight Decay with Dataset Size

Under a fixed learning rate, from $\lambda = 1/(\eta M \tau_{\text{epoch}})$, we can deduce that:

- As the dataset grows, $M$ increases.
- If the optimal $\tau_{\text{epoch}}$ remains constant, the optimal $\lambda$ should decrease.

**Scaling Rule**: $\lambda \propto 1/N$ (the larger the dataset, the smaller the weight decay).

#### 3. Scaling of Weight Decay with Model Width

µP requires the learning rate to scale with model width: $\eta = \eta_{\text{base}} / s$, where $s = \text{fan\_in} / \text{fan\_in}_{\text{base}}$.

If $\lambda$ is kept constant (the standard µP practice), then:

$$\tau_{\text{iter}} = s / (\eta_{\text{base}} \lambda_{\text{base}}) = s \cdot \tau_{\text{iter;base}}$$

The timescale increases as the model grows, **which causes learning rate transfer in µP to fail**.

The solution is to scale weight decay with width as well:

$$\eta = \eta_{\text{base}} / s, \quad \lambda = s \cdot \lambda_{\text{base}}$$

In this way, $\tau_{\text{iter}} = 1/(\eta\lambda) = 1/(\eta_{\text{base}} \lambda_{\text{base}}) = \tau_{\text{iter;base}}$, keeping the timescale invariant.

**Scaling Rule**: $\lambda \propto \text{fan\_in}$ (the wider the model, the larger the weight decay).

#### 4. Theoretical Guarantees on Scale-Invariant Networks (Theorem 1)

For a scale-invariant network (i.e., $\text{net}(x; w) = \text{net}(x; w/c)$), under the same EMA timescale and the same initial learning rate/initialization ratio, two different configurations of AdamW $(\eta, \lambda, \sigma, \epsilon)$ will produce identical network output trajectories. This theoretically proves that $\tau_{\text{iter}}$ is the key control parameter of AdamW on scale-invariant networks.

### Loss & Training

This work does not propose a new loss function, but rather hyperparameter scaling strategies, which are summarized below:

| Scaling Dimension | Learning Rate $\eta$ | Weight Decay $\lambda$ | EMA Timescale $\tau$ |
|:---|:---|:---|:---|
| Dataset expansion ($N \uparrow$) | Fixed | $\lambda \propto 1/N$ | $\tau_{\text{epoch}}$ invariant |
| Model widening (µP) | $\eta \propto 1/s$ | $\lambda \propto s$ | $\tau_{\text{iter}}$ invariant |
| Batch size enlargement | $\eta \propto B$ or Fixed | $\lambda \propto B$ (if $\eta$ is fixed) | $\tau_{\text{epoch}}$ invariant |

During training, a cosine decay learning rate schedule is employed (decaying to 0.1 or 0 of the initial value), and weight decay is not applied to normalization layers.

## Key Experimental Results

### Main Results

Experiments cover three architectures and three datasets:

| Model | Dataset | Verification Content | Conclusion |
|:---|:---|:---|:---|
| ResNet-18 | ImageNet 32×32 subset (80K-1.28M) | Stability of $\tau_{\text{epoch}}$ across dataset sizes | Optimal $\tau_{\text{epoch}}$ is stable, whereas optimal $\lambda$ changes drastically |
| ViT | ImageNet 32×32 subset | Same as above | Similarly validates the stability of $\tau_{\text{epoch}}$ |
| NanoGPT 124M | OpenWebText (1/4, 1/2, full dataset) | Dataset scaling on language models | Optimal $\lambda$ decreases as the dataset grows, $\tau_{\text{epoch}}$ remains stable |
| ResNet-18 (Widened) | ImageNet 32×32 (320K) | $\lambda$ scaling with model width | $\lambda \propto s$ aligns the optimal $\lambda_{\text{base}}$ |
| 8-layer GPT (256-1024 width) | OpenWebText | LLM width scaling | Same as above, confirming the effectiveness of $\lambda \propto s$ |

### Ablation Study

| Configuration | Key Metric | Description |
|:---|:---|:---|
| Fixed $\lambda$ + µP (Standard practice) | Optimal $\eta_{\text{base}}$ changes drastically with width | µP learning rate transfer fails |
| $\lambda \propto s$ + µP (Ours) | Optimal $\eta_{\text{base}}$ is consistent across widths | Restores the transferability of µP |
| Constant LR schedule | $\tau_{\text{epoch}}$ is stable | Consistent conclusions under different decay schedules |
| Cosine decay to 0 | $\tau_{\text{epoch}}$ is stable | Same as above |
| $\lambda \propto 1/N$ + Fixed $\eta$ | Optimal $\eta$ is consistent across dataset sizes | Scaling $\lambda$ correctly avoids the need to tune $\eta$ |

### Key Findings

1. **Optimal $\tau_{\text{epoch}}$ is stable across dataset and model scales**: This is the most crucial experimental finding, consistently holding true across ResNet, ViT, and GPT.
2. **µP + fixed $\lambda$ fails**: On CIFAR-10 and ImageNet, the standard µP practice (fixed $\lambda$) causes the optimal learning rate to vary significantly with width.
3. **$\lambda \propto s$ restores µP**: By scaling the weight decay proportionally to the model width, the optimal learning rate recovers its consistency across widths.
4. **Subsequent large-scale validation**: Blake et al. (2024) and Dey et al. (2025) confirmed the transferability of $\tau_{\text{epoch}}$ in billion-parameter scale LLM pre-training.

## Highlights & Insights

1. **Elegant theoretical perspective**: Interpreting AdamW through the lens of EMA is natural and compelling; a simple reparameterization reveals the underlying structure.
2. **High practical value**: Provides clear, actionable scaling rules for weight decay, directly applicable to large-scale training.
3. **Explaining existing phenomena**: Unifying and explaining the decoupling of $\eta$ and $\gamma=\eta\lambda$ observed by Loshchilov & Hutter, as well as the failure of µP on AdamW discovered by Lingle (2024).
4. **Theoretical depth of Theorem 1**: Rigorously proving that $\tau_{\text{iter}}$ is the unique control parameter on scale-invariant networks.

## Limitations & Future Work

1. **Limited experimental scale**: Verification was primarily conducted on small-scale datasets (ImageNet 32×32, CIFAR-10) and medium models (124M NanoGPT), although subsequent works have confirmed these findings on a larger scale.
2. **Only considering multi-epoch training**: LLM pre-training of modern models often runs for only 1 epoch, under which the optimal value of $\tau_{\text{epoch}}$ is no longer constant but follows a power law (Bergsma et al., 2025a).
3. **Limitations of the EMA approximation**: Standard EMA assumes $q_t$ is independent, but in AdamW, $q_t$ depends on $w_t$, making it an approximation.
4. **Depth scaling unexplored**: The study primarily focuses on width scaling, leaving the scaling of $\lambda$ under changing depths (number of layers) unexplored.
5. **The exact relationship between $\lambda$ and $s$ might be superlinear** ($\lambda \propto s^\alpha, \alpha > 1$), requiring larger-scale experiments to confirm.

## Related Work & Insights

- **µP (Yang et al., 2022)**: This work serves as an important complement to µP, addressing the issue where µP fails due to not considering weight decay.
- **Loshchilov & Hutter (2018)**: The original AdamW paper observed that $(\eta, \gamma)$ is more decoupled than $(\eta, \lambda)$; this study provides a theoretical explanation.
- **Wortsman et al. (2024)**: Observed that when $\gamma=\eta\lambda$ is fixed, the validation loss is insensitive to $\eta$, supporting the view of $\tau_{\text{iter}}$ as a core hyperparameter.
- **D'Angelo et al. (2024)**: Posited that the role of weight decay is not regularization but controlling minibatch noise, which complements the EMA perspective.
- **Bergsma et al. (2025a)**: Extended the framework of this paper to batch size scaling and single-epoch scenarios, finding that $\tau_{\text{epoch}}^{\text{optimal}} \approx (\text{TPP})^{-0.527}$.

**Insight**: This paper suggests that when designing hyperparameter search spaces for large-scale training, we should search for $\tau_{\text{epoch}}$ (or equivalently, $\eta\lambda$) instead of searching for $\lambda$ in isolation. This can substantially narrow the search space and increase the success rate of cross-scale transfer.

## Rating

| Dimension | Rating | Description |
|:---|:---|:---|
| Novelty | ⭐⭐⭐⭐ | The EMA perspective is simple yet highly insightful |
| Theoretical Depth | ⭐⭐⭐⭐ | Theorem 1 is rigorous and elegant, though primarily relying on experiments |
| Experimental Thoroughness | ⭐⭐⭐⭐ | Validated across multiple models and datasets, but on a relatively small scale |
| Value | ⭐⭐⭐⭐⭐ | Directly guides large-scale training, widely validated by subsequent works |
| Writing Quality | ⭐⭐⭐⭐⭐ | Clear logic with well-explained intuition |
| **Overall** | **⭐⭐⭐⭐☆** | A concise and powerful study with exceptional practicality |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] TraveLLaMA: A Multimodal Travel Assistant with Large-Scale Dataset and Structured Reasoning](../../AAAI2026/recommender/travellama_a_multimodal_travel_assistant_with_large-scale_dataset_and_structured.md)
- [\[NeurIPS 2025\] The More You Automate, the Less You See: Hidden Pitfalls of AI Scientist Systems](../../NeurIPS2025/recommender/the_more_you_automate_the_less_you_see_hidden_pitfalls_of_ai_scientist_systems.md)
- [\[ICML 2025\] PARM: Multi-Objective Test-Time Alignment via Preference-Aware Autoregressive Reward Model](parm_multi-objective_test-time_alignment_via_preference-aware_autoregressive_rew.md)
- [\[ICML 2025\] SIMPLEMIX: Frustratingly Simple Mixing of Off- and On-policy Data in Language Model Preference Learning](simplemix_frustratingly_simple_mixing_of_off-_and_on-policy_data_in_language_mod.md)
- [\[NeurIPS 2025\] Estimating Hitting Times Locally At Scale](../../NeurIPS2025/recommender/estimating_hitting_times_locally_at_scale.md)

</div>

<!-- RELATED:END -->
