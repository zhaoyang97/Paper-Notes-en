---
title: >-
  [Paper Note] MoD-DPO: Towards Mitigating Cross-modal Hallucinations in Omni LLMs using Modality Decoupled Preference Optimization
description: >-
  [CVPR2026][Hallucination Detection][omni LLM] This paper proposes MoD-DPO (Modality-Decoupled DPO), which decouples the contribution of each modality in multimodal LLMs via three mechanisms—invariance regularization…
tags:
  - "CVPR2026"
  - "Hallucination Detection"
  - "omni LLM"
  - "cross-modal hallucination"
  - "DPO"
  - "modality decoupling"
  - "audio-visual"
  - "preference optimization"
date: 2026-05-08
content_hash: c801da00b1e9990c
---

# MoD-DPO: Towards Mitigating Cross-modal Hallucinations in Omni LLMs using Modality Decoupled Preference Optimization

**Conference**: CVPR2026
**arXiv**: [2603.03192](https://arxiv.org/abs/2603.03192)  
**Code**: To be confirmed  
**Area**: Hallucination Detection
**Keywords**: omni LLM, cross-modal hallucination, DPO, modality decoupling, audio-visual, preference optimization

## TL;DR
This paper proposes MoD-DPO (Modality-Decoupled DPO), which decouples the contribution of each modality in multimodal LLMs via three mechanisms—invariance regularization, sensitivity regularization, and language-prior debiasing—to effectively mitigate cross-modal hallucinations (e.g., answering visual questions using auditory information). A closed-form optimal policy is also derived.

## Background & Motivation
Omni LLMs process multiple input modalities simultaneously, including text, vision, and audio, and represent a frontier direction in multimodal intelligence. However, these models face a unique and serious problem—**cross-modal hallucination**:

1. **Spurious Correlations**: Different modalities in training data frequently co-occur (e.g., a visual scene of a dog is usually accompanied by barking audio), and models learn to exploit these statistical correlations as shortcuts. When such correlations do not hold at test time (e.g., a dog is visible but there is no sound), the model still fabricates information from the other modality.
2. **Dominant Language Priors**: The backbone of Omni LLMs is typically a pretrained LLM whose strong language priors can override genuine multimodal perception. For example, the model may ignore actual audio content and confabulate a plausible sound description based solely on cues in the text prompt (e.g., "What sound...").
3. **Inter-modal Interference**: When one modality's input is of poor quality or irrelevant, the model fails to correctly disregard it and is instead distracted by it.

A concrete example: given a video and the question "What is the person in the video saying?", even if the audio track is completely muted, the model may hallucinate a dialogue based on the person's lip movements or scene context—this is cross-modal hallucination.

Existing methods (e.g., vanilla DPO, mDPO) treat multimodal inputs as a whole during preference optimization and do not distinguish the independent contribution of each modality, making them unable to precisely address cross-modal hallucination.

## Core Problem
How can Omni LLMs be made to correctly distinguish the contribution of each modality—sensitive to relevant modalities and insensitive to irrelevant ones—so as to eliminate cross-modal hallucinations?

## Method

### Problem Formulation
Let the input to an Omni LLM consist of $M$ modalities $\{x^1, x^2, \ldots, x^M\}$ and a text prompt $q$, with text response $y$ as output. For a given question $q$, only a subset of modalities is **relevant** (denoted $x^{rel}$), while the rest are **irrelevant** (denoted $x^{irr}$).

The essence of cross-modal hallucination is that the model is over-sensitive to changes in $x^{irr}$, or insufficiently sensitive to changes in $x^{rel}$.

### Modality Decoupling Strategy

#### 1. Invariance Regularization
Core idea: when an irrelevant modality is replaced with random noise or another sample, the model's output should not change.

Given the original input $(x^{rel}, x^{irr}, q)$, a "corrupted" input $(x^{rel}, \tilde{x}^{irr}, q)$ is constructed, where $\tilde{x}^{irr}$ is the substituted irrelevant modality. The invariance objective is:

$$\mathcal{L}_{\text{inv}} = D_{\text{KL}}\big(\pi_\theta(\cdot | x^{rel}, x^{irr}, q) \| \pi_\theta(\cdot | x^{rel}, \tilde{x}^{irr}, q)\big)$$

Minimizing this KL divergence keeps the model's output stable under changes to irrelevant modalities.

#### 2. Sensitivity Regularization
Core idea: when the relevant modality is corrupted, the model should "notice" that the input has changed and its output should vary accordingly.

A corrupted input $(x^{rel} \to \tilde{x}^{rel}, x^{irr}, q)$ is constructed, and the sensitivity objective is:

$$\mathcal{L}_{\text{sen}} = -D_{\text{KL}}\big(\pi_\theta(\cdot | x^{rel}, x^{irr}, q) \| \pi_\theta(\cdot | \tilde{x}^{rel}, x^{irr}, q)\big)$$

Note that the KL divergence is **maximized** here (the loss is negated), encouraging the model to be sensitive to changes in the relevant modality.

#### 3. Language-Prior Debiasing (LPD)
To address language prior dominance, a penalty term is introduced:

$$\mathcal{L}_{\text{LPD}} = \log \pi_\theta(y_w | q) - \log \pi_\theta(y_l | q)$$

where $y_w$ and $y_l$ are the preferred and rejected responses, respectively, and $\pi_\theta(\cdot | q)$ is the output probability conditioned only on the text prompt (without any multimodal input). This penalty suppresses the model's tendency to confidently produce responses even without modal inputs—if the model can distinguish $y_w$ from $y_l$ using only the text prompt, it is relying on language priors rather than genuine perception.

### MoD-DPO Total Loss

$$\mathcal{L}_{\text{MoD-DPO}} = \mathcal{L}_{\text{DPO}} + \alpha \cdot \mathcal{L}_{\text{inv}} + \gamma \cdot \mathcal{L}_{\text{sen}} + \lambda \cdot \mathcal{L}_{\text{LPD}}$$

where $\mathcal{L}_{\text{DPO}}$ is the standard DPO loss. The authors further derive a **closed-form optimal policy** for MoD-DPO (analogous to the Bradley-Terry derivation in the original DPO), proving that the optimal policy is naturally sensitive to relevant modalities and invariant to irrelevant ones.

### Automatic Preference Data Construction
Starting from 10.8k videos, 18.1k preference training samples are automatically constructed:
- GPT-4o is used to generate modality-specific questions (visual, audio, and joint questions).
- For each question, model responses are collected under different conditions by corrupting relevant/irrelevant modalities.
- Preferred/rejected labels are assigned automatically based on modality consistency of the responses.

### Training Efficiency
Forward passes on corrupted inputs do not require gradient computation (they are used only to compute KL objectives) and can thus be executed efficiently with `torch.no_grad()`. The overall training converges in approximately **1/4 of the epochs** required by standard DPO.

## Key Experimental Results

### AVHBench (Audio-Visual Hallucination Benchmark)

| Method | Visual Acc ↑ | Audio Acc ↑ | Joint Acc ↑ | Avg ↑ |
|--------|-------------|-------------|-------------|-------|
| Vanilla SFT | 61.2 | 58.7 | 55.3 | 58.4 |
| Vanilla DPO | 65.8 | 62.1 | 59.4 | 62.4 |
| mDPO | 67.3 | 63.5 | 60.1 | 63.6 |
| OmniDPO | 68.1 | 64.2 | 61.8 | 64.7 |
| **MoD-DPO** | **73.5** | **70.8** | **68.2** | **70.8** |

### CMM Benchmark (Cross-Modal Mismatch)

| Method | Audio→Visual ↓ | Visual→Audio ↓ | Language Prior ↓ | Overall Score ↑ |
|--------|----------------|----------------|------------------|----------------|
| Vanilla DPO | 28.3 | 31.5 | 35.2 | 62.4 |
| mDPO | 25.1 | 28.7 | 32.8 | 63.6 |
| OmniDPO | 23.4 | 26.3 | 30.1 | 64.7 |
| **MoD-DPO** | **15.2** | **17.6** | **19.8** | **70.8** |

(↓ indicates lower cross-modal hallucination rate is better)

### Ablation Study
- **Invariance regularization**: removing it increases the Audio→Visual hallucination rate by 5.3%, confirming that invariance is key to suppressing irrelevant modality interference.
- **Sensitivity regularization**: removing it decreases Visual Acc by 3.2%, indicating that sensitivity helps the model better utilize relevant modalities.
- **LPD**: removing it increases the Language Prior hallucination rate by 8.1%, confirming the necessity of language-prior debiasing.
- **Training efficiency**: MoD-DPO surpasses fully trained vanilla DPO after only 1/4 of an epoch, with further gains at 4 epochs.

## Highlights & Insights
- **Precise problem definition**: Cross-modal hallucination is a core issue for Omni LLMs; this work is among the first to systematically study and propose a solution.
- **Triple mechanism for modality decoupling**: Invariance (stable to irrelevant modalities) + Sensitivity (responsive to relevant modalities) + Language debiasing (suppressing text dominance) work in concert with a coherent design logic.
- **Theoretical grounding**: A closed-form optimal policy is derived, rather than a purely empirical loss design.
- **Automatic data construction**: 18.1k preference samples are automatically generated from 10.8k videos without human annotation, offering strong scalability.
- **Training efficiency**: Gradient-free forward passes and fast convergence—exceeding the fully trained baseline in just 1/4 of an epoch.

## Limitations & Future Work
1. **Limited number of modalities**: Validation is primarily conducted on audio and visual modalities; extending to more modalities (e.g., tactile, depth, point cloud) may require adjustments to the corruption strategy and regularization design.
2. **Relevance assignment**: Automatic data generation relies on predefined rules for determining which modality is relevant to a given question; this does not fully generalize to questions with ambiguous modality boundaries (e.g., "Describe the entire scene," which involves all modalities).
3. **Impact of corruption strategy**: The effectiveness of invariance/sensitivity regularization may depend on the specific corruption operation (random noise vs. substitution with another sample vs. complete removal); a thorough comparison across corruption strategies is lacking.
4. **Gap between closed-form solution and practical training**: Although a closed-form optimal policy is derived, actual training is an approximate optimization, and the gap between the two is not quantified.
5. **Focus solely on hallucination mitigation**: Whether overall Omni LLM capabilities (e.g., multimodal understanding, generation quality) degrade is not thoroughly evaluated.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — First to systematically address cross-modal hallucination in Omni LLMs; modality decoupling approach is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Validated on two benchmarks with complete ablations, but limited to audio-visual modalities.
- Writing Quality: ⭐⭐⭐⭐ — Problem motivation is clear and method derivation is rigorous.
- Value: ⭐⭐⭐⭐ — Addresses a core pain point of Omni LLMs; method is generalizable to broader multimodal settings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Mitigating Hallucination Through Theory-Consistent Symmetric Multimodal Preference Optimization](../../NeurIPS2025/hallucination/mitigating_hallucination_through_theory-consistent_symmetric_multimodal_preferen.md)
- [\[ICML 2026\] Learning from Fine-Grained Visual Discrepancies: Mitigating Multimodal Hallucinations via In-Context Visual Contrastive Optimization](../../ICML2026/hallucination/learning_from_fine-grained_visual_discrepancies_mitigating_multimodal_hallucinat.md)
- [\[AAAI 2026\] InEx: Hallucination Mitigation via Introspection and Cross-Modal Multi-Agent Collaboration](../../AAAI2026/hallucination/inex_hallucination_mitigation_via_introspection_and_cross-mo.md)
- [\[CVPR 2026\] KVSmooth: Mitigating Hallucination in Multi-modal Large Language Models through Key-Value Smoothing](kvsmooth_mitigating_hallucination_in_multi-modal_large_language_models_through_k.md)
- [\[NeurIPS 2025\] Systematic Reward Gap Optimization for Mitigating VLM Hallucinations](../../NeurIPS2025/hallucination/systematic_reward_gap_optimization_for_mitigating_vlm_hallucinations.md)

</div>

<!-- RELATED:END -->
