---
title: >-
  [Paper Note] Test-Time Attention Purification for Backdoored Large Vision Language Models
description: >-
  [CVPR 2026][Multimodal VLM][backdoor attack defense] This work identifies that the essence of backdoor behavior in LVLMs is cross-modal attention stealing (trigger visual tokens hijack the attention weights of text tokens), and proposes CleanSight — the first training-free test-time backdoor defense framework — which eliminates backdoor effects by detecting and pruning high-attention trigger tokens.
tags:
  - CVPR 2026
  - Multimodal VLM
  - backdoor attack defense
  - attention purification
  - LVLM security
  - test-time defense
  - visual token pruning
date: 2026-05-08
content_hash: bb1f4aa17cddc56e
---

# Test-Time Attention Purification for Backdoored Large Vision Language Models

**Conference**: CVPR 2026
**arXiv**: [2603.12989](https://arxiv.org/abs/2603.12989)
**Code**: To be confirmed
**Area**: Multimodal VLM
**Keywords**: backdoor attack defense, attention purification, LVLM security, test-time defense, visual token pruning

## TL;DR
This work identifies that the essence of backdoor behavior in LVLMs is cross-modal attention stealing (trigger visual tokens hijack the attention weights of text tokens), and proposes CleanSight — the first training-free test-time backdoor defense framework — which eliminates backdoor effects by detecting and pruning high-attention trigger tokens.

## Background & Motivation
**Background**: Fine-tuning lightweight adapters to adapt LVLMs to downstream tasks has become mainstream, but this also introduces backdoor attack risks — adversaries can inject trigger samples into fine-tuning data, causing the model to produce attacker-specified outputs upon encountering triggers at inference time.

**Limitations of Prior Work**: Existing defenses are primarily training-time defenses — retraining backdoor-contaminated parameters with clean data — which incur high computational costs and often degrade downstream performance. The few test-time defense methods (e.g., pixel perturbation) are designed for models trained from scratch and are nearly ineffective against LVLMs.

**Key Challenge**: In LVLMs, backdoor associations reside not in low-level pixel features but in cross-modal attention interactions — a fundamentally different finding from traditional backdoor models (e.g., ViT, CLIP). Pixel perturbation cannot disrupt backdoor associations at the attention level.

**Goal**: Design the first test-time backdoor defense method for LVLMs that requires no retraining and is plug-and-play.

**Key Insight**: The discovery of the "attention stealing" phenomenon — visual tokens from poisoned inputs abnormally capture the attention weights of text tokens, and the high-attention regions precisely correspond to trigger regions.

**Core Idea**: Eliminate backdoors at test time without modifying model parameters by detecting anomalous attention ratios and pruning high-attention visual tokens.

## Method

### Overall Architecture
CleanSight operates at inference time: it first computes visual-to-text attention ratios at selected intermediate layers to detect whether an input is poisoned; if detected as poisoned, it prunes visual tokens with abnormally high attention.

### Key Designs

1. **Attention Stealing Detection**:

    - **Function**: At intermediate layers where cross-modal fusion occurs, compute the visual-to-text attention ratio for each attention head.
    - **Mechanism**: For each layer and head in the detection layer set $\mathcal{L}_{\text{det}}$, compute $S^{\ell,h} = \frac{\sum_{j\in\mathcal{I}_{\text{vis}}}\alpha_{q,j}^{\ell,h}}{\sum_{j\in\mathcal{I}_{\text{prm}}}\alpha_{q,j}^{\ell,h}}$; concatenate ratio vectors across all heads and compute the whitened $\ell_2$ distance against a clean reference distribution: $d(\hat{s}) = \|\frac{\hat{s}-\mu}{\sigma}\|_2$; inputs exceeding the 99th-percentile threshold $\gamma$ are classified as poisoned.
    - **Design Motivation**: Intermediate layers are the primary site of cross-modal fusion, making attention anomalies there most discriminative (AUROC near perfect). Retaining head-level granularity is more robust than averaging across heads.

2. **Selective Pruning**:

    - **Function**: Identify and suppress visual tokens controlled by the trigger.
    - **Mechanism**: At the last detection layer, take the union $\Omega$ of visual tokens whose attention exceeds threshold $\tau$ across all heads; in all subsequent layers, apply a large negative bias $b\ll 0$ to these positions to drive their attention weights toward zero.
    - **Design Motivation**: Taking the union rather than the intersection ensures anomalies in any single head are captured; the large negative bias approximates zero attention after softmax, effectively isolating trigger tokens.

3. **Reference Distribution Construction**:

    - **Function**: Estimate reference statistics of attention ratios on a small clean validation set.
    - **Mechanism**: Collect per-sample attention ratio vectors, compute per-dimension mean and standard deviation, and set the detection threshold using the 99th-percentile whitened distance.
    - **Design Motivation**: Requires minimal clean data (only sufficient for statistics estimation), making it suitable for service deployment scenarios.

### Loss & Training
CleanSight is a completely training-free test-time method and involves no parameter updates or loss functions.

## Key Experimental Results

### Main Results (ASR↓ / CU↑ on VQAv2)

| Attack | No Defense ASR | CleanSight ASR | No Defense CU | CleanSight CU |
|--------|---------------|----------------|--------------|---------------|
| BadNet | 100.0 | **0.0** | 62.89 | 62.63 |
| Blended | 100.0 | **0.0** | 67.06 | 65.50 |
| ISSBA | 98.83 | **2.34** | 65.49 | 64.71 |
| WaNet | 100.0 | **0.0** | 68.10 | 67.32 |
| TrojVLM | 100.0 | **1.56** | 68.36 | 67.97 |
| VLOOD | 100.0 | **0.0** | 53.65 | 53.26 |

### Comparison with Baseline Defenses

| Defense | BadNet ASR↓ | Blended ASR↓ | WaNet ASR↓ | Training Required? |
|---------|------------|-------------|------------|-------------------|
| ST Defense | 82.81 | 97.66 | 92.58 | No |
| BDMAE | 88.28 | 100.0 | 99.22 | No |
| ZIP | 80.47 | 84.77 | 7.03 | No |
| CleanSight | **0.0** | **0.0** | **0.0** | No |

### Key Findings
- CleanSight reduces ASR to near 0% on almost all attack types with negligible clean accuracy loss.
- Traditional pixel perturbation defenses (Blur, ST Defense) are nearly ineffective against LVLM backdoors, validating the attention stealing mechanism.
- The effectiveness of attention perturbation increases monotonically with perturbation strength; when attention is fully uniformized, the backdoor disappears entirely even with trigger pixels still present.
- Detection layers in the middle range (layers 10–24) are most effective, consistent with the location of cross-modal fusion.

## Highlights & Insights
- **Mechanistic Discovery**: Revealing that the essence of LVLM backdoors lies in attention allocation rather than pixels reframes the understanding paradigm of VLM backdoor attack and defense, and can guide the design of more targeted attacks and defenses in the future.
- **Zero Training Overhead**: As a plug-and-play inference-time method, CleanSight is well-suited for FTaaS (Fine-Tuning as a Service) scenarios where users cannot control the training pipeline but can control the inference stack.
- **Connection to Visual Token Pruning (e.g., FastV)**: It is notable that efficiency-oriented token pruning is repurposed here as a security-oriented defense mechanism.

## Limitations & Future Work
- A small clean validation set is required to estimate the reference distribution, making the method inapplicable in scenarios with no clean data available.
- The thresholds $\gamma$ and $\tau$ may require tuning for different models and attacks.
- Only adapter/LoRA-level backdoor attacks have been validated; applicability to full-parameter backdoors remains unknown.
- Detection is performed at the first token decoding step; the latency impact on streaming generation scenarios warrants further analysis.

## Related Work & Insights
- **vs FastV**: FastV prunes low-attention visual tokens to accelerate inference; CleanSight prunes high-attention tokens to eliminate backdoors — opposite directions, but similar mechanisms.
- **vs ZIP**: ZIP defends via pixel-level perturbation, achieving 80% ASR on BadNet; CleanSight reduces ASR to 0% through attention perturbation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to reveal the attention stealing mechanism underlying LVLM backdoors, opening a new direction for test-time defense.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 6 attack types, multiple datasets, and multiple comparison baselines.
- Writing Quality: ⭐⭐⭐⭐⭐ Logically coherent, with a seamless progression from mechanistic discovery to method design.
- Value: ⭐⭐⭐⭐⭐ Makes an important contribution to the field of LVLM security.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Scaling Test-Time Robustness of Vision-Language Models via Self-Critical Inference Framework](scaling_test-time_robustness_of_vision-language_models_via_self-critical_inferen.md)
- [\[CVPR 2026\] Activation Matters: Test-time Activated Negative Labels for OOD Detection with Vision-Language Models](activation_matters_test-time_activated_negative_labels_for_ood_detection_with_vi.md)
- [\[CVPR 2026\] Decoupling Stability and Plasticity for Multi-Modal Test-Time Adaptation](decoupling_stability_and_plasticity_for_multi-modal_test-time_adaptation.md)
- [\[CVPR 2026\] AVA-VLA: Improving Vision-Language-Action models with Active Visual Attention](ava_vla_improving_vision_language_action_models_with_active_visual_attention.md)
- [\[CVPR 2026\] Can Vision-Language Models Count? A Synthetic Benchmark and Analysis of Attention-Based Interventions](can_vision-language_models_count_a_synthetic_benchmark_and_analysis_of_attention.md)

<!-- RELATED:END -->
