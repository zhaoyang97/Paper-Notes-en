---
title: >-
  [Paper Note] Visual Language Models as Zero-Shot Deepfake Detectors
description: >-
  [ICML 2025][LLM Safety][Deepfake Detection] Proposes an image classification framework based on VLM token probability normalization, upgrading deepfake detection from binary decisions to probability estimation. Under zero-shot settings, InstructBLIP outperforms most dedicated deepfake detectors, and achieves near-perfect performance on DFDC-P after fine-tuning.
tags:
  - "ICML 2025"
  - "LLM Safety"
  - "Deepfake Detection"
  - "Vision-Language Models"
  - "Zero-shot Classification"
  - "VLM Probability Calibration"
  - "InstructBLIP"
date: 2026-05-08
content_hash: 7ff118788ffdd17e
---

# Visual Language Models as Zero-Shot Deepfake Detectors

**Conference**: ICML 2025  
**arXiv**: [2507.22469](https://arxiv.org/abs/2507.22469)  
**Code**: None  
**Area**: Image Generation  
**Keywords**: Deepfake Detection, Vision-Language Models, Zero-shot Classification, VLM Probability Calibration, InstructBLIP

## TL;DR
Proposes an image classification framework based on VLM token probability normalization, upgrading deepfake detection from binary decisions to probability estimation. Under zero-shot settings, InstructBLIP outperforms most dedicated deepfake detectors, and achieves near-perfect performance on DFDC-P after fine-tuning.

## Background & Motivation
**Background**: Most deepfake detection methods train dedicated classifiers (such as FaceForensics++, SBI, MAT), which rely heavily on labeled data and generalize poorly to novel deepfakes.

**Limitations of Prior Work**: (a) Existing detectors experience a sharp drop in performance on out-of-distribution data; (b) Existing VLM-based deepfake studies only perform binary yes/no decisions and cannot output confidence probabilities; (c) They lack support for practical deployment metrics such as FAR/FRR.

**Key Challenge**: Real-world deployment requires probabilistic outputs to adjust thresholds (balancing false acceptance and false rejection rates), whereas the argmax output of VLMs can only yield 0/1 binary decisions.

**Goal**: How to extract meaningful classification confidence from the token distribution of VLMs?

**Key Insight**: Utilize the probability ratio of "yes"/"no" tokens in response to "Is this photo real?" as the confidence score.

**Core Idea**: Normalize the yes/no token probabilities into $\tilde{P}_{\text{fake}} = P_{\text{no}} / (P_{\text{no}} + P_{\text{yes}})$ to obtain continuous confidence scores suitable for ROC analysis.

## Method

### Overall Architecture
Given an image and a prompt (e.g., "Is this photo real?"), the VLM performs a single forward pass to obtain the token distribution. The probabilities of tokens like "yes"/"Yes"/"no"/"No" are extracted, summed by group, and normalized to obtain the fake confidence score for downstream decision-making.

### Key Designs

1. **Token Probability Normalized Classification**:

    - Function: Extract classification confidence from the VLM's token distribution.
    - Mechanism: $P(I \in D) \approx \frac{P_{\text{no}}}{P_{\text{no}} + P_{\text{yes}}}$, where $P_{\text{no}} = p(\text{"no"}) + p(\text{"No"})$ and $P_{\text{yes}} = p(\text{"yes"}) + p(\text{"Yes"})$.
    - Design Motivation: Compared to argmax (0/1 outputs), normalized probabilities support AUC/EER evaluation and threshold adjustment.

2. **Multi-token/Multi-class Extension (Algorithm 1)**:

    - Function: Support multi-token answers (e.g., "Yes for sure!") and multi-class classification.
    - Mechanism: For all candidate answer strings $s \in \mathcal{S}_c$ of class $c$, compute the autoregressive probability $P(s|I,Q) = \prod_k p(t_k|I,Q,t_{1:k-1}) \cdot p(\text{EOS}|I,Q,s)$, followed by summation and normalization.
    - Design Motivation: Because tokenizer vocabularies differ across VLMs, it is necessary to cover all potential answer formats.

3. **Prompt Engineering**:

    - Function: Design customized prompts for different VLMs.
    - Mechanism: InstructBLIP only requires "Is this photo real?"; LLaVA needs an additional "Answer using a single word"; GPT-4o requires role-play-style long prompts.
    - Design Motivation: Ensure models consistently return answers in a yes/no format.

## Key Experimental Results

### Main Results (Zero-Shot vs. Dedicated Detectors, CelebA-HQ SimSwap Dataset)

| Method | AUC ↑ | ACC ↑ | EER ↓ |
|------|-------|-------|-------|
| FF++ (XceptionNet) | 58.9 | 59.2 | 44.5 |
| MAT | 49.0 | 50.0 | 50.6 |
| RECCE | 46.9 | 49.1 | 50.8 |
| SBI (SOTA Dedicated) | **93.6** | **85.2** | **14.0** |
| InstructBLIP (Zero-Shot) | 81.3 | 75.3 | 26.9 |
| **InstructBLIP FT** | **92.1** | **85.0** | **12.2** |

### Method Comparison (Normalization vs. Softmax vs. Binary)

| VLM | Binary ACC | Normalize AUC | Softmax AUC |
|------|-----------|---------------|-------------|
| InstructBLIP | 68.0 | **81.3** | 80.9 |
| Idefics2 | 74.2 | **80.6** | 75.2 |
| LLaVA-1.6 | 58.3 | **74.2** | 74.2 |

### Key Findings
- The normalization method outperforms binary argmax across all VLMs (with a maximum gain of ~16% AUC).
- Zero-shot InstructBLIP outperforms most dedicated detectors (only lagging behind SBI + CADDM).
- Fine-tuning InstructBLIP yields an AUC of 92.1%, approaching SBI's 93.6%.

## Highlights & Insights
- **Practical Framework**: The token probability normalization method is generally applicable to any classification task using VLMs, not limited to deepfake detection.
- **Demonstration of Zero-Shot Capability**: The pre-trained knowledge of VLMs is sufficient to achieve viable performance on novel deepfakes.
- **Multi-Token Extension**: The autoregressive cumulative probability multiplication in Algorithm 1 supports answers of arbitrary length.

## Related Work & Insights
- **vs AntifakePrompt**: AntifakePrompt fine-tunes soft prompts on InstructBLIP for deepfake VQA but only outputs 0/1; ours requires no fine-tuning and outputs continuous probabilities.
- **vs SHIELD/ChatGPT deepfake**: These works qualitatively evaluate the deepfake detection capabilities of GPT-4V/Gemini but do not systematically quantify token probabilities; ours proposes a complete probabilistic framework.
- **vs SBI (SOTA)**: SBI trains highly generalizable classifiers via self-blending data augmentation; ours is completely zero-shot, and although its AUC is slightly lower, it requires no deepfake training data.
- The proposed token probability normalization framework can be directly applied to other scenarios requiring classification confidence from VLMs (e.g., medical image analysis, content moderation).

## Limitations & Future Work
- Only face-swap deepfakes were tested, while other modalities such as full-face generation (StyleGAN) and expression manipulation (Face2Face) were not covered.
- Token probabilities cannot be accessed for GPT-4o, restricting it to binary evaluations and limiting the application of closed-source models.
- VLM inference speed is significantly slower than lightweight classifiers (e.g., EfficientNet), which presents latency challenges for real-world deployment.
- Evaluation on recent deepfake methods (e.g., full-body deepfakes generated by Flux) is missing.
- Although the multi-token answer extension is mathematically described, it has not been systematically validated in experiments.

## Rating
- Novelty: ⭐⭐⭐⭐ The token probability normalization classification is a simple yet effective innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparison across multiple VLMs and detectors.
- Writing Quality: ⭐⭐⭐⭐ The methodological derivation is clear, with complete prompt and algorithm details.
- Value: ⭐⭐⭐⭐ Opens up a new application paradigm for VLMs in security detection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Unlocking the Capabilities of Large Vision-Language Models for Generalizable and Explainable Deepfake Detection](unlocking_the_capabilities_of_large_vision-language_models_for_generalizable_and.md)
- [\[NeurIPS 2025\] Zero-Shot Robustness of Vision Language Models Via Confidence-Aware Weighting](../../NeurIPS2025/llm_safety/zero-shot_robustness_of_vision_language_models_via_confidence-aware_weighting.md)
- [\[CVPR 2026\] ⊘ Source Models Leak What They Shouldn't ↛: Unlearning Zero-Shot Transfer in Domain Adaptation Through Adversarial Optimization](../../CVPR2026/llm_safety/oslash_source_models_leak_what_they_shouldnt_nrightarrow_unlearning_zero-shot_tr.md)
- [\[ICML 2025\] CROW: Eliminating Backdoors from Large Language Models via Internal Consistency Regularization](crow_eliminating_backdoors_from_large_language_models_via_internal_consistency_r.md)
- [\[ICLR 2026\] Veritas: Generalizable Deepfake Detection via Pattern-Aware Reasoning](../../ICLR2026/llm_safety/veritas_generalizable_deepfake_detection_via_pattern-aware_reasoning.md)

</div>

<!-- RELATED:END -->
