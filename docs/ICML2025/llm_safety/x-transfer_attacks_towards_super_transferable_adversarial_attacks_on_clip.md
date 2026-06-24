---
title: >-
  [Paper Note] X-Transfer Attacks: Towards Super Transferable Adversarial Attacks on CLIP
description: >-
  [ICML 2025][LLM Safety][adversarial attack] Proposed the X-Transfer attack method, which generates "super-transferable" universal adversarial perturbations (UAPs) through an efficient surrogate model scaling strategy (dynamic selection based on multi-armed bandits). A single perturbation can simultaneously attack various CLIP encoders and downstream VLMs across data, domains, models, and tasks.
tags:
  - "ICML 2025"
  - "LLM Safety"
  - "adversarial attack"
  - "CLIP"
  - "universal adversarial perturbation"
  - "transferability"
  - "surrogate scaling"
date: 2026-05-08
content_hash: 97b03808276aa9e4
---

# X-Transfer Attacks: Towards Super Transferable Adversarial Attacks on CLIP

**Conference**: ICML 2025  
**arXiv**: [2505.05528](https://arxiv.org/abs/2505.05528)  
**Code**: [GitHub](https://github.com/HanxunH/X-Transfer)  
**Area**: AI Safety / Adversarial Attacks  
**Keywords**: adversarial attack, CLIP, universal adversarial perturbation, transferability, surrogate scaling

## TL;DR

Proposed the X-Transfer attack method, which generates "super-transferable" universal adversarial perturbations (UAPs) through an efficient surrogate model scaling strategy (dynamic selection based on multi-armed bandits). A single perturbation can simultaneously attack various CLIP encoders and downstream VLMs across data, domains, models, and tasks.

## Background & Motivation

- CLIP models are widely integrated into large vision-language models (VLMs) (such as LLaVA, BLIP2, MiniGPT-4), making their vision encoders key targets for security attacks.
- Universal Adversarial Perturbations (UAPs) can attack different samples with the same perturbation, but existing methods fail to achieve **super transferability**—simultaneity transfer across data, domains, models, and tasks.
- Existing ensemble methods (e.g., Liu et al., 2017) use a fixed set of surrogate models, which incurs prohibitive computational costs when scaling to a large number of surrogates.
- **Two Core Problems**: (1) Can a single perturbation achieve four-dimensional super transferability simultaneously? (2) How can the number of surrogate models be scaled efficiently?

## Method

### Overall Architecture

The core of X-Transfer is an **efficient surrogate scaling strategy**: dynamically selecting a small number ($k$) of surrogates from a large search space ($N$ CLIP encoders) to generate UAPs that transfer across all dimensions.

### Adversarial Objectives

**Non-targeted attack** (minimizing the embedding similarity between the adversarial sample and the original sample):

$$\arg\min_{\delta} \mathbb{E}_{x \sim \mathbb{D}'} \frac{1}{k}\sum_{i=1}^{k} \text{sim}(f_I'(x'), f_I'(x))$$

**Targeted attack** (maximizing the embedding similarity between the adversarial sample and the target text):

$$\arg\max_{\delta} \mathbb{E}_{x \sim \mathbb{D}'} \text{sim}(f_I'(x'), f_T'(t_{adv}))$$

where $x' = x + \delta$, and $\|\delta\|_\infty < \epsilon$. The objective functions operate on the embedding space, making them independent of the encoder architecture and embedding dimensions—a key factor in achieving cross-model integration.

### Efficient Surrogate Scaling Strategy (Core Contribution)

Based on a **non-stationary Multi-Armed Bandit (MAB)**:

- Each candidate encoder is treated as an "arm".
- At each step, the top-$k$ encoders are selected for gradient computation.
- The **UCB (Upper Confidence Bound) strategy** is used to balance exploration and exploitation:

$$\text{UCB} = R_i + \sqrt{\frac{2\ln n}{n_i}}$$

**Reward Design**: The loss value $\mathcal{L}_i$ is used directly as the reward. A higher loss indicates that the encoder is more difficult to break with the current UAP, meaning it should be selected more frequently to enhance target perturbation universality. The reward is updated using an Exponential Moving Average (EMA): $R_i = (1-m) R_i + m \mathcal{L}_i$.

### Search Space Configuration

| Configuration | Number of Encoders (N) | Selection per Step (k) | Architecture Types |
|------|:--------:|:-------:|---------|
| Base | 16 | 4 | 4 each of RN, ConvNext, ViT-B, ViT-L |
| Mid | 32 | 8 | Diverse architectures |
| Large | 64 | 16 | Maximum diversity |

The search space **excludes** any target victim models to ensure a strict black-box setting.

## Key Experimental Results

### Zero-shot Classification ASR (Average of 9 Black-box Victim Models)

| Method | C-10 | C-100 | Food | ImageNet | Cars | STL | Average |
|------|:----:|:-----:|:----:|:-------:|:----:|:---:|:---:|
| Meta-UAP (Best Baseline) | 79.3 | 93.4 | 46.0 | 30.9 | 28.5 | 25.9 | 50.8 |
| C-PGC (ViT-B/16) | 63.7 | 82.9 | 51.3 | 40.4 | 38.1 | 28.2 | 51.6 |
| ETU (ViT-B/16) | 70.2 | 86.5 | 47.1 | 34.1 | 31.1 | 27.5 | 49.8 |
| Vanilla (N=1) | 72.7 | 88.3 | 49.9 | 31.2 | 26.3 | 19.2 | 48.4 |
| **X-Transfer Base** | **86.6** | **97.5** | **74.8** | **56.0** | **52.1** | **46.8** | **69.2** |
| **X-Transfer Large** | **87.6** | **97.8** | **80.1** | **63.4** | **64.6** | **57.1** | **75.6** |

### Image-Text Retrieval ASR (MSCOCO)

| Method | TR@1 | IR@1 |
|------|:----:|:----:|
| Best Baseline (C-PGC ViT-B/16) | 43.8 | 35.7 |
| **X-Transfer Large** | **71.8** | **65.8** |

### Large VLM Attacks (Image Captioning + VQA)

X-Transfer also exhibits powerful cross-task transferability on large VLMs in tasks such as OpenFlamingo-3B, LLaVA-7B, MiniGPT-4, and BLIP2, significantly outperforming all baselines.

### Key Findings

- **Vanilla baseline is already on par with specialized CLIP methods**: A simple embedding space attack objective is sufficiently powerful, indicating that the general design of the objective function is crucial.
- **Super-transferability scales with larger search spaces**: Base (69.2%) $\rightarrow$ Mid (73.6%) $\rightarrow$ Large (75.6%), with only a linear increase in computational cost.
- **Only a few surrogates are needed**: Selecting only $1$-$4$ surrogates per step ($k$ is much smaller than $N$) achieves performance close to full ensemble integration.
- **Sampling strategy is not the key factor**: Differences between UCB, random, and round-robin strategies are marginal; the scale and diversity of the search space are the actual core drivers.

## Highlights & Insights

1. **Clever integration of MAB and UAP**: Modelling surrogate model selection as a multi-armed bandit problem, and using reward signals (loss values) to guide the selection of hard-to-break encoders naturally enhances perturbation universality.
2. **Extremely high efficiency**: Under a search space of 64 surrogate encoders, only 16 are used per step to compute gradients, saving $4\times$ computation compared to a full ensemble.
3. **General design of adversarial objective**: Direct operation in the embedding space (rather than task-specific classification/retrieval losses) allows the UAP to naturally transfer across tasks.
4. **Reveals systemic vulnerability of CLIP**: A fixed $L_\infty$ perturbation ($\epsilon=12/255$) can break various CLIP variants and downstream VLMs simultaneously, presenting a significant security risk.

## Limitations & Future Work

- A perturbation of $\epsilon=12/255$ may be human-noticeable in some scenarios; the imperceptibility in practical deployment needs further evaluation.
- The search space only includes publicly available CLIP encoders; transferability to proprietary or highly architecturally distinct models (such as non-CLIP vision encoders) has not been tested.
- The UCB reward design directly uses loss values, which might exhibit slow convergence in highly non-stationary environments.
- Defense methods (such as adversarial training, input preprocessing) and their mitigation efficacy were not thoroughly discussed.
- Assessments were mainly conducted on static images; video or interactive scenarios remain unexplored.

## Related Work

- **CLIP Adversarial Attacks**: AdvCLIP (Zhou et al., 2023a) quasi-black-box UAP, ETU (Zhang et al., 2024) global+local features, C-PGC (Fang et al., 2024b) cross-model transfer.
- **Universal Adversarial Perturbations**: UAP (Moosavi-Dezfooli et al., 2017), GD-UAP (Mopuri et al., 2018), TRM-UAP (Liu et al., 2023b), Meta-UAP (Weng et al., 2024).
- **Transfer Attacks**: Ensemble methods (Liu et al., 2017; Dong et al., 2018), Surrogate Scaling (Liu et al., 2024) points out the computational bottlenecks of fixed ensembles.
- **VLM Attacks**: Sample-specific attacks on VLMs by Zhao et al. (2023), Schlarmann et al. (2024), etc.

## Rating

⭐⭐⭐⭐ — First to define and systematically validate "super-transferable" UAPs on CLIP, with a highly elegant and efficient surrogate scaling design. The experiments are extensive (spanning multiple datasets, models, and tasks), yielding impressive results. It is a substantial contribution to understanding the security risks of CLIP in the VLM ecosystem.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Transferable and Stealthy Adversarial Attacks on Large Vision-Language Models](../../ICLR2026/llm_safety/transferable_and_stealthy_adversarial_attacks_on_large_vision-language_models.md)
- [\[NeurIPS 2025\] ToxicTextCLIP: Text-Based Poisoning and Backdoor Attacks on CLIP Pre-training](../../NeurIPS2025/llm_safety/toxictextclip_text-based_poisoning_and_backdoor_attacks_on_clip_pre-training.md)
- [\[ICML 2025\] The Ripple Effect: On Unforeseen Complications of Backdoor Attacks](the_ripple_effect_on_unforeseen_complications_of_backdoor_attacks.md)
- [\[ICML 2025\] ICLShield: Exploring and Mitigating In-Context Learning Backdoor Attacks](iclshield_exploring_and_mitigating_in-context_learning_backdoor_attacks.md)
- [\[ICML 2025\] Revealing Weaknesses in Text Watermarking Through Self-Information Rewrite Attacks](revealing_weaknesses_in_text_watermarking_through_self-information_rewrite_attac.md)

</div>

<!-- RELATED:END -->
