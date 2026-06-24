---
title: >-
  [Paper Note] CleanSight: Test-Time Attention Purification for Backdoored Large Vision Language Models
description: >-
  [CVPR 2025][LLM Safety][backdoor attack] CleanSight reveals that the backdoor attack mechanism in LVLMs lies not in the pixel space but in the attention map—the trigger activates the backdoor via "attention stealing" (where trigger tokens hijack the attention of text tokens). Based on this, a training-free, plug-and-play test-time defense method is proposed: identifying poisoned inputs by detecting anomalies in cross-modal attention ratios, and neutralizing the backdoor by pr…
tags:
  - "CVPR 2025"
  - "LLM Safety"
  - "backdoor attack"
  - "LVLM defense"
  - "attention mechanism"
  - "test-time defense"
  - "visual token pruning"
date: 2026-05-08
content_hash: ca1da4d12d1727cc
---

# CleanSight: Test-Time Attention Purification for Backdoored Large Vision Language Models

**Conference**: CVPR 2025  
**arXiv**: [2603.12989](https://arxiv.org/abs/2603.12989)  
**Code**: To be confirmed  
**Area**: Multimodal VLM / AI Safety  
**Keywords**: backdoor attack, LVLM defense, attention mechanism, test-time defense, visual token pruning

## TL;DR
CleanSight reveals that the backdoor attack mechanism in LVLMs lies not in the pixel space but in the attention map—the trigger activates the backdoor via "attention stealing" (where trigger tokens hijack the attention of text tokens). Based on this, a training-free, plug-and-play test-time defense method is proposed: identifying poisoned inputs by detecting anomalies in cross-modal attention ratios, and neutralizing the backdoor by pruning high-attention visual tokens. This reduces the ASR to near 0% with almost no impact on model performance.

## Background & Motivation
**Background**: LVLMs (e.g., LLaVA) adapt to downstream tasks by fine-tuning adapters; however, the fine-tuning phase is susceptible to backdoor attacks, where attackers inject trigger-containing samples into the training data.

**Limitations of Prior Work**: Existing defense methods either require retraining with clean data (computationally expensive and damaging to downstream performance) or perturb inputs in the pixel space (e.g., image transformations). However, **pixel perturbations are almost completely ineffective against LVLM backdoors**.

**Key Challenge**: Unlike CLIP trained from scratch, the backdoor associations in LVLMs are not bound to low-level visual features but are hidden within cross-modal attention interaction patterns. Consequently, pixel perturbations fail to address anomalies at the attention layer.

**Goal**: How can one detect and neutralize LVLM backdoors at test time without modifying model parameters?

**Key Insight**: A key phenomenon is discovered: the mechanism of backdoor activation is "attention stealing." Specifically, visual tokens of poisoned inputs abnormally hijack attention weights from text tokens in the intermediate layers of the LVLM, suppressing the model's instruction-following capability.

**Core Idea**: Backdoor lies in attention rather than pixels $\rightarrow$ detecting attention ratio anomalies + pruning high-attention visual tokens = training-free test-time defense.

## Method

### Overall Architecture
CleanSight operates in two steps:
1. **Detection**: In selected intermediate cross-modal fusion layers, the visual-to-text attention ratio of each attention head is computed and compared against a clean reference distribution (using whitened $\ell_2$ distance) to determine whether the input is poisoned.
2. **Purification**: For inputs flagged as poisoned, anomalously high-attention visual tokens across all heads are aggregated and pruned, preventing them from "stealing" attention in subsequent layers and during decoding.

### Key Designs

1. **Discovery of the Attention Stealing Mechanism**:

    - **Function**: Unveils the true mechanism of backdoor activation in LVLMs.
    - **Mechanism**: In clean inputs, the attention of visual tokens in intermediate layers is much lower than that of text tokens. In poisoned inputs, however, the attention of visual tokens in trigger areas spikes dramatically, while text token attention decreases correspondingly—thereby "stealing" the text attention.
    - **Design Motivation**: This explains why pixel perturbations are ineffective (as the trigger pattern remains) whereas attention perturbations are effective (as evening out the attention eliminates the backdoor).

2. **Attention-Ratio-Based Detection**:

    - **Function**: Detects poisoned inputs using head-specific visual-to-text attention ratios.
    - **Mechanism**: In a selected intermediate layer $\ell$, the ratio of visual attention to total attention is computed for each head $h$, forming an attention ratio vector $\mathbf{r}$. A few clean samples are used to estimate the clean distribution $(\boldsymbol{\mu}, \boldsymbol{\Sigma})$. During detection, the whitened $\ell_2$ distance $d = \|(\mathbf{r} - \boldsymbol{\mu}) \boldsymbol{\Sigma}^{-1/2}\|_2$ is compared against a threshold.
    - **Design Motivation**: Head-specific ratios are more sensitive than a global average, and the whitened distance handles the variance differences across different heads.

3. **Visual Token Pruning for Purification**:

    - **Function**: Identifies and removes the trigger tokens that "steal" attention.
    - **Mechanism**: In all selected heads, visual token indices whose attention values exceed a threshold (e.g., top-k or percentile) are collected. After taking the intersection or union, these tokens are pruned (removed directly from the KV cache) so subsequent layers no longer perceive them.
    - **Design Motivation**: Pinpoint removal of trigger-associated tokens rather than uniform perturbation, preserving clean semantics to the maximum extent.

### Loss & Training
Completely training-free. It only requires a small number of clean samples (~100) to estimate the reference distribution of the attention ratio.

## Key Experimental Results

### Main Results (VQA + Image Captioning)

| Defense Method | BadNet ASR↓ | Blended ASR↓ | ISSBA ASR↓ | WaNet ASR↓ | TrojVLM ASR↓ | VLOOD ASR↓ |
|---------|------------|-------------|-----------|-----------|-------------|-----------|
| No defense | 100.00 | 100.00 | 99.22 | 99.61 | 100.00 | 100.00 |
| ST defense | 85.55 | 98.05 | 67.19 | 53.91 | 77.73 | 82.42 |
| BDMAE | 88.28 | 100.00 | 100.00 | 99.22 | 80.86 | 86.33 |
| ZIP | 80.47 | 84.77 | 74.22 | 7.03 | 85.94 | 95.31 |
| **CleanSight** | **0** | **0** | **0** | **0** | **3.14** | **0** |

Clean Utility (VQAv2 accuracy) is maintained at 62-68% under CleanSight, which is virtually on par with the no-defense baseline.

### Ablation Study

| Component | ASR↓ | Description |
|------|------|------|
| Detection only (flag + no action) | n/a | Detection only without purification |
| Pruning without detection | ASR↓ but CU↓ | Pruning all inputs degrades normal performance |
| Detection + Pruning (CleanSight) | ~0% | Combined detection and purification performs best |
| Uniform attention perturbation | ~0% ASR but CU↓ | Effective but degrades clean inputs |

### Key Findings
- **Pixel perturbations are largely ineffective against LVLM backdoors**: ST defense and Blur still result in ASR >80% under most attacks, whereas attention perturbation eliminates the backdoor even at an intensity of 1.
- **CleanSight achieves near 0% ASR across 6 attack types**: Significantly outperforming all baselines (where ZIP is the strongest but still yields 55-95% ASR).
- The attention stealing phenomenon consistently occurs across various trigger types (patch/global/WaNet/ISSBA).
- The critical layers for pruning are the intermediate fusion layers (rather than the deepest or shallowest layers).

## Highlights & Insights
- **Mechanism discovery takes precedence over methodology**: The discovery of "Attention Stealing" profoundly reveals how LVLM backdoors work, offering a novel conceptual framework for the field.
- **Training-free and plug-and-play practicality**: Extremely low deployment cost with no parameter modification, no retraining, and only lightweight interventions during feedforward propagation.
- **Unexpected connection**: Visual token pruning was originally designed for inference acceleration (e.g., FastV, LLaVA-PruMerge). This work shows it also enhances backdoor security—enabling simultaneous speedup and security.
- **Transferable approach**: Attention ratio analysis can be extended to detect other anomalous behaviors in VLMs (e.g., adversarial examples, input contamination).

## Limitations & Future Work
- Requires a small number of clean samples to estimate the reference distribution—while the count is minimal (~100), zero-shot scenarios still remain to be explored.
- Threshold settings for detection require tuning; different models and attacks may demand different thresholds.
- Validated only on the LLaVA series; generalizability to other LVLM architectures (e.g., Qwen-VL, InternVL) is yet to be tested.
- Adaptive attackers might design novel backdoors that do not rely on attention stealing.
- For semantic-preserving attacks like TrojVLM, the ASR is reduced to 3-5% but not entirely eliminated, as their backdoor behavior is tightly intertwined with normal task objectives.

## Related Work & Insights
- **vs ST defense**: Spatial transformations in the pixel space (rotation/flipping) are ineffective against LVLM backdoors (ASR >80%) because backdoors do not reside in the pixel layer.
- **vs BDMAE**: Purifying inputs via MAE reconstruction is effective against some attacks but remains unstable.
- **vs ZIP**: Searching for purification vectors in the input space via zeroth-order optimization is effective against WaNet, but other attacks still yield high ASR.
- **vs FastV**: FastV prunes low-attention tokens (for acceleration), whereas CleanSight prunes high-attention anomalous tokens (for security)—making them complementary.
- **vs BDMAE/SampDetox**: Generative purification methods based on MAE/diffusion models exhibit unstable performance on LVLMs because backdoors do not reside in the pixel space.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The discovery of the attention stealing mechanism is highly profound and inspiring.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extensive evaluations across 6 attack types, multiple datasets, and several baselines, coupled with thorough ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ Highly coherent logical flow: observation $\rightarrow$ mechanism $\rightarrow$ methodology $\rightarrow$ validation.
- Value: ⭐⭐⭐⭐⭐ The first test-time backdoor defense for LVLMs, offering extreme practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] TAPT: Test-Time Adversarial Prompt Tuning for Robust Inference in Vision-Language Models](tapt_test-time_adversarial_prompt_tuning_for_robust_inference_in_vision-language.md)
- [\[ICCV 2025\] LATTE: Collaborative Test-Time Adaptation of Vision-Language Models in Federated Learning](../../ICCV2025/llm_safety/latte_collaborative_test-time_adaptation_of_vision-language_models_in_federated_.md)
- [\[CVPR 2025\] Hyperbolic Safety-Aware Vision-Language Models](hyperbolic_safety-aware_vision-language_models.md)
- [\[NeurIPS 2025\] Buffer Layers for Test-Time Adaptation](../../NeurIPS2025/llm_safety/buffer_layers_for_test-time_adaptation.md)
- [\[NeurIPS 2025\] Attention! Your Vision Language Model Could Be Maliciously Manipulated](../../NeurIPS2025/llm_safety/attention_your_vision_language_model_could_be_maliciously_manipulated.md)

</div>

<!-- RELATED:END -->
