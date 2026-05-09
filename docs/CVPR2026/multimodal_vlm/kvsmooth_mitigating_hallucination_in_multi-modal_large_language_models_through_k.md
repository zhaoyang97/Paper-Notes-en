---
title: >-
  [Paper Note] KVSmooth: Mitigating Hallucination in Multi-modal Large Language Models through Key-Value Smoothing
description: >-
  [CVPR2026][Multimodal VLM][Multimodal hallucination mitigation] This paper proposes KVSmooth, a training-free plug-and-play inference-time method that applies adaptive exponential moving average (EMA) smoothing to KV-Cache guided by attention row entropy, effectively suppressing semantic drift and hallucination generation caused by sink tokens during decoding in multimodal large language models (MLLMs). On LLaVA-1.5, CHAIR_S is reduced from 41.8 to 18.2 (a 56% reduction), while F1 improves from 77.5 to 79.2.
tags:
  - CVPR2026
  - Multimodal VLM
  - Multimodal hallucination mitigation
  - KV-Cache smoothing
  - attention entropy
  - exponential moving average
  - training-free inference
date: 2026-05-08
content_hash: 672116d9dbdc92d7
---

# KVSmooth: Mitigating Hallucination in Multi-modal Large Language Models through Key-Value Smoothing

**Conference**: CVPR2026
**arXiv**: [2602.04268](https://arxiv.org/abs/2602.04268)
**Code**: To be confirmed
**Area**: Multimodal VLM
**Keywords**: Multimodal hallucination mitigation, KV-Cache smoothing, attention entropy, exponential moving average, training-free inference

## TL;DR

This paper proposes KVSmooth, a training-free plug-and-play inference-time method that applies adaptive exponential moving average (EMA) smoothing to KV-Cache guided by attention row entropy, effectively suppressing semantic drift and hallucination generation caused by sink tokens during decoding in multimodal large language models (MLLMs). On LLaVA-1.5, CHAIR_S is reduced from 41.8 to 18.2 (a 56% reduction), while F1 improves from 77.5 to 79.2.

## Background & Motivation

1. **Pervasive multimodal hallucination**: MLLMs frequently generate objects, attributes, or relations inconsistent with visual content in tasks such as image captioning and VQA, severely hindering trustworthy deployment.
2. **Long-range decay of visual dependency**: As the decoding sequence grows, the influence of early visual tokens gradually diminishes in hidden representations, causing generated text to progressively deviate from image content.
3. **Cumulative semantic drift**: Minor inaccuracies at early generation steps accumulate and amplify over time, widening the gap between generated descriptions and visual facts.
4. **Sink tokens exacerbate hallucination**: Attention concentrates on a small number of "aggregation tokens" whose hidden states, produced via global averaging, deviate from the visual context and systematically inflate the logit scores of hallucinated objects.
5. **Precision–recall trade-off in existing methods**: Fine-tuning approaches require substantial data and computation; contrastive decoding incurs high computational overhead; attention redistribution methods often suppress correct object descriptions alongside hallucinations, sacrificing recall.
6. **Insufficient understanding of the sink-token hallucination mechanism**: Prior work focuses on reducing the occurrence of sink tokens or diminishing their attention weights, but does not explain why they trigger hallucinations.

## Method

### Overall Architecture

KVSmooth is a **training-free, plug-and-play** inference-time method. Its core mechanism applies adaptive EMA smoothing to the Key and Value vectors of the KV-Cache during decoding, stabilizing hidden-state evolution and suppressing hallucinations. The method consists of two core components.

### Three Key Observations

- **Obs1 (Logit dynamic divergence)**: The mean logit of ground-truth (GT) objects decreases monotonically with stable variance, whereas the mean logit of hallucinated objects rises continuously with increasing variance—indicating that hallucination candidates accumulate instability in the hidden representations.
- **Obs2 (Row entropy and sink behavior)**: **Attention row entropy** is proposed as a real-time metric for measuring token sink intensity. Tokens with high row entropy have diffuse attention distributions; their hidden states approximate historical averages and exhibit small angular distances from most other states, causing them to attract disproportionately high attention in subsequent steps and form attention sinks. The cosine similarity between row entropy and the conventional column-sum metric concentrates around 0.79.
- **Obs3 (Entropy–rank coupling)**: The logit rank of hallucinated objects is positively correlated with attention row entropy—the more uniform a token's attention distribution (higher entropy), the higher the logits of hallucinated objects; the opposite holds for GT objects. This reveals a direct causal chain through which sink tokens systematically inflate hallucination scores via global context averaging.

### Component 1: EMA Smoothing on KV-Cache

Based on the assumption that an ideal decoding trajectory should be smooth, hidden-state transitions are modeled as a Gaussian random walk $h_t = h_{t-1} + \epsilon_t$. Bayesian MAP estimation yields:

$$\hat{h_t} = (1-\lambda_t) o_t + \lambda_t h_{t-1}$$

which is exactly the EMA form. A key design choice is to **apply EMA simultaneously to both Key and Value** (rather than directly to hidden states), as this maximally regularizes both the mean and variance of logits, achieving the strongest hallucination suppression without degrading recall.

### Component 2: Entropy-Guided Adaptive Coefficient

- The attention row entropy $z_t^l$ of the current token at each layer is computed and maintained in a FIFO queue of length $M$.
- The smoothing coefficient is determined by the **percentile rank** of the row entropy: $\hat{\lambda}_t^l = k/M$, assigning larger smoothing coefficients to higher-entropy tokens.
- A clipping mechanism is introduced: centered at hyperparameter $\lambda_{\text{ref}}$, the coefficient is clipped to $[\lambda_{\text{ref}}-0.2, \lambda_{\text{ref}}+0.2]$, stabilizing generation while preserving representational diversity.

### Final Update Rule

For token $x_t$ at designated layer $l$:

$$\hat{K_t^l} = (1-\tilde{\lambda}_t^l) K_t^l + \tilde{\lambda}_t^l K_{t-1}^l$$
$$\hat{V_t^l} = (1-\tilde{\lambda}_t^l) V_t^l + \tilde{\lambda}_t^l V_{t-1}^l$$

Applied to layers 3–31, with FIFO queue length 15 and $\lambda_{\text{ref}}$ set to 0.9 / 0.5 / 0.7 for LLaVA-1.5 / MiniGPT-4 / InstructBLIP, respectively.

## Key Experimental Results

### CHAIR Benchmark (Image Captioning Hallucination)

| Method | LLaVA-1.5 CHAIR_S↓ | LLaVA-1.5 F1↑ | MiniGPT-4 CHAIR_S↓ | MiniGPT-4 F1↑ | InstructBLIP CHAIR_S↓ | InstructBLIP F1↑ |
|--------|---------------------|----------------|---------------------|----------------|------------------------|-------------------|
| Baseline | 41.8 | 77.5 | 31.8 | 69.9 | 61.4 | 71.6 |
| PAI | 22.6 | 75.5 | 24.6 | 71.0 | 63.4 | 71.1 |
| OPERA | 44.2 | 78.6 | 27.4 | 69.4 | 68.0 | 69.2 |
| MiddleLayer | 17.8 | 75.9 | 24.6 | 71.2 | 75.0 | 67.2 |
| **KVSmooth** | **18.2** | **79.2** | **17.0** | **71.7** | **42.2** | **75.1** |

- On LLaVA-1.5, CHAIR_S is reduced by 56% while F1 improves from 77.5 to 79.2, making KVSmooth the **only method to simultaneously improve both precision and recall**.
- On MiniGPT-4, CHAIR_S is reduced from 31.8 to 17.0 (a 47% reduction).

### Object HalBench

On LLaVA-1.5, CHAIR_SR is reduced from 45.3% to 16.7% (a 63.1% reduction). Sentence-level hallucination rates are reduced by 63.1%, 40.3%, and 41.6% across the three models, respectively.

### Ablation Study

| Smoothing Location | LLaVA-1.5 Cs↓ | F1↑ |
|--------------------|----------------|-----|
| Attention output $o_t$ only | 33.8 | 74.7 |
| Key $K_t$ only | 35.6 | 79.4 |
| **Key + Value** | **18.2** | **79.2** |

- Jointly smoothing Key and Value yields the best performance; smoothing only hidden states leads to severe recall degradation.
- The adaptive coefficient (Ada.) further reduces CHAIR_S on LLaVA-1.5 from 36.2 to 18.2 compared to a fixed coefficient, validating the precise identification capability of the entropy-guided mechanism.

## Highlights & Insights

- **Training-free plug-and-play**: Operates directly on the KV-Cache at inference time without fine-tuning or modifying model parameters, offering inherent generality.
- **Theory- and empirical-driven**: The EMA form is derived from Bayesian MAP estimation, and the three observations provide a clear causal explanation chain (logit divergence → row-entropy sink → entropy–hallucination coupling).
- **Breaking the precision–recall dilemma**: PR curve analysis demonstrates that KVSmooth is the only method that substantially reduces hallucinations while maintaining or improving F1.
- **Introducing the sink degree concept**: Row entropy serves as a continuous, real-time sink metric that is more efficient than the conventional column-sum approach, requiring no multi-step look-back.
- **Broad validation**: Evaluated across 3 models (LLaVA-1.5 / MiniGPT-4 / InstructBLIP) × 4 benchmarks (CHAIR / OPOPE / AMBER / Object HalBench) with consistent results.

## Limitations & Future Work

- **Model-dependent hyperparameters**: $\lambda_{\text{ref}}$ requires separate tuning for each model (0.9 / 0.5 / 0.7), with no automatic selection scheme.
- **Evaluation limited to 7B models**: Performance on larger scales (13B / 70B) or newer architectures (e.g., Qwen2.5-VL) has not been verified.
- **Generation length constraint**: Maximum generation is capped at 512 tokens; the effectiveness of EMA in ultra-long generation scenarios remains unexplored.
- **Fixed layer range**: Application to layers 3–31 is empirically determined without a systematic layer selection criterion.
- **Focus on object hallucination only**: Finer-grained hallucination types such as attribute and relation hallucinations are not evaluated.
- **High entropy is not always detrimental**: At semantic transition points, high entropy may reflect legitimate context switching; uniform smoothing carries a theoretical risk of information loss.

## Related Work & Insights

| Category | Representative Methods | Core Idea | KVSmooth Advantage |
|----------|------------------------|-----------|-------------------|
| Contrastive decoding | VCD, OPERA | Noise-augmented views for contrast / rollback penalty | Lower computational overhead; no multiple forward passes required |
| Attention redistribution | PAI, SPARC, MiddleLayer | Enhance attention on visual tokens | Does not sacrifice recall; superior PR curve |
| Fine-tuning alignment | POVID, RLHF | Fine-tune with preference data | Training-free; requires no additional data |
| KV-Cache pruning | PruneHal | Remove redundant visual tokens | Preserves information more completely; regulates via smoothing rather than deletion |

## Rating

- Novelty: ⭐⭐⭐⭐ — The row-entropy sink degree concept is novel; deriving EMA smoothing from a Bayesian perspective is theoretically elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 3 models × 4 benchmarks with detailed ablations and PR analysis, though evaluation on larger models and additional hallucination types is absent.
- Writing Quality: ⭐⭐⭐⭐ — The observation–method–experiment logical chain is clear; mathematical derivations are concise and accessible.
- Value: ⭐⭐⭐⭐ — The lightweight, training-free approach offers strong practical utility and introduces a new direction for inference-time hallucination mitigation.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Residual Decoding: Mitigating Hallucinations in Large Vision-Language Models via History-Aware Residual Guidance](residual_decoding_mitigating_hallucinations_in_large_vision-language_models_via_.md)
- [\[CVPR 2026\] VideoFusion: A Spatio-Temporal Collaborative Network for Multi-modal Video Fusion](videofusion_a_spatio-temporal_collaborative_network_for_multi-modal_video_fusion.md)
- [\[CVPR 2026\] BriMA: Bridged Modality Adaptation for Multi-Modal Continual Action Quality Assessment](brima_bridged_modality_adaptation_for_multi-modal_continual_action_quality_asses.md)
- [\[CVPR 2026\] UniMMAD: Unified Multi-Modal and Multi-Class Anomaly Detection via MoE-Driven Feature Decompression](unimmad_unified_multi-modal_and_multi-class_anomaly_detection_via_moe-driven_fea.md)
- [\[CVPR 2026\] Multi-Modal Representation Learning via Semi-Supervised Rate Reduction for Generalized Category Discovery](multi-modal_representation_learning_via_semi-supervised_rate_reduction_for_gener.md)

<!-- RELATED:END -->
