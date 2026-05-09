---
title: >-
  [Paper Note] ViSpec: Accelerating Vision-Language Models with Vision-Aware Speculative Decoding
description: >-
  [NeurIPS 2025][Multimodal VLM][Speculative decoding] To address the difficulty of draft models in handling redundant visual tokens during VLM speculative decoding, this paper proposes ViSpec, a framework that achieves significant acceleration (up to 3.22×) in VLM speculative decoding for the first time, via a visual adapter for image token compression, global visual feature injection, and synthetic training data generation.
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - Speculative decoding
  - VLM acceleration
  - image token compression
  - draft model
  - inference acceleration
date: 2026-05-08
content_hash: 60d3c3acde6f5777
---

# ViSpec: Accelerating Vision-Language Models with Vision-Aware Speculative Decoding

**Conference**: NeurIPS 2025
**arXiv**: [2509.15235](https://arxiv.org/abs/2509.15235)
**Code**: [GitHub](https://github.com/KangJialiang/ViSpec)
**Area**: Multimodal VLM
**Keywords**: Speculative decoding, VLM acceleration, image token compression, draft model, inference acceleration

## TL;DR

To address the difficulty of draft models in handling redundant visual tokens during VLM speculative decoding, this paper proposes ViSpec, a framework that achieves significant acceleration (up to 3.22×) in VLM speculative decoding for the first time, via a visual adapter for image token compression, global visual feature injection, and synthetic training data generation.

## Background & Motivation

Speculative decoding has achieved 3–4× speedups in text-only LLMs, yet only marginal acceleration (<1.5×) has been reported for VLMs (e.g., SpecLLaVA achieves only 1.5×). The authors conduct an in-depth analysis of the root cause of this gap:

**Fundamental difference between text and visual data**: Text has evolved over millennia into a highly abstract, information-dense medium, whereas images, despite their visual richness, contain substantial redundancy (e.g., uniform-color regions). This makes it difficult for shallow draft models to extract key information from large amounts of redundant content.

**Theoretical analysis**: The authors provide a formal proof. Consider $R+1$ tokens, where $R$ are identical redundant tokens (e.g., patches from a uniform-color image region) and 1 is a unique token. For a single-layer Transformer, as $R \to \infty$, the attention weight assigned to the unique token approaches zero:

$$\alpha_{iu} = \frac{\exp(B)}{R\exp(A) + \exp(B)} \to 0$$

The output degenerates into an average over the redundant tokens, entirely ignoring the valuable unique visual information. Theoretically, handling nested complexity $K$ requires a network with $K+1$ layers, yet draft models are typically very shallow.

**Lost-in-the-Middle problem**: As the generated text sequence grows longer, image tokens positioned in the middle become overwhelmed, and the U-shaped performance curve of shallow draft models leads to further loss of visual information.

## Method

### Overall Architecture

ViSpec consists of three core components: (1) a visual adapter that compresses image embeddings, (2) global visual feature injection, and (3) synthetic long-answer training data generation. The draft model receives compressed visual tokens and augmented text tokens to predict the next token sequence, which the target model verifies in parallel.

### Key Designs

1. **Vision Adaptor**: A lightweight Transformer encoder inspired by Q-Former. A fixed number of learnable query vectors serve as queries, while original visual features serve as keys and values. Through cross-attention, thousands of image embeddings are compressed into a small number (only 1 in experiments) of compact visual tokens, while preserving the positional information of the original image. Experiments show that a single compressed embedding suffices to capture visual information; increasing the number has negligible effect on the acceptance length $\tau$ while reducing the actual speedup ratio by increasing the computational load of the draft model.

2. **Global Visual Feature Integration**: A global feature vector $g$ is extracted from the final output of the visual adapter and injected into the hidden states of all subsequent text tokens via a learned projection matrix $W_g$:

$$f_t^{\text{aug}} = f_t + W_g g$$

This ensures that the draft model retains access to global visual context throughout long text generation, effectively mitigating the Lost-in-the-Middle effect.

3. **Training Data Generation and Multi-Token Prediction**: To address the scarcity of long-answer samples in public multimodal datasets, synthetic training data are constructed by prompting the target VLM with modified instructions (e.g., appending "Please answer in at least 1000 words") to generate long responses. A sampling strategy (non-greedy) is adopted to break the one-to-one correspondence between hidden states and embeddings; combined with multi-token prediction (inspired by DeepSeek-V3), this prevents shortcut learning in the draft model.

$$L = \text{CrossEntropy}(p_i, \hat{p}_i)$$

### Loss & Training

- Two-stage training: text fundamentals are first learned on 68K ShareGPT text-only data, followed by fine-tuning on multimodal data.
- The ViSpec draft model is a single-layer network mirroring the decoder layer structure of the target model.
- Learning rate: 3e-6; optimizer: AdamW; batch size: 8; training: 20 epochs.
- At inference, the context-aware dynamic draft tree from EAGLE-2 is used (30 draft tokens, depth 3, 8-node expansion).

## Key Experimental Results

### Main Results (Temperature=0, LLaVA-1.6-7B)

| Method | SQA | TextVQA | COCO Caps | GQA | Avg. Speedup | Avg. $\tau$ |
|--------|-----|---------|-----------|-----|-------------|------------|
| Medusa | 1.41× | 1.46× | 1.61× | 1.29× | 1.42× | 0.72 |
| EAGLE-2 | 2.14× | 1.25× | 1.80× | 1.64× | 1.62× | 1.31 |
| **ViSpec** | **2.37×** | **2.90×** | **3.22×** | **2.22×** | **2.58×** | **2.98** |

### Cross-Model Validation (Temperature=0, Average)

| Model | Medusa | EAGLE-2 | ViSpec |
|-------|--------|---------|--------|
| LLaVA-1.6-7B | 1.42× | 1.62× | **2.58×** |
| LLaVA-1.6-13B | 1.48× | 1.86× | **2.38×** |
| Qwen2.5-VL-3B | 1.14× | 1.39× | **1.87×** |
| Qwen2.5-VL-7B | 1.11× | 1.40× | **1.80×** |

### Ablation Study

| Component | COCO Caps | GQA | MME |
|-----------|-----------|-----|-----|
| EAGLE-2 baseline | 1.80× | 1.64× | 1.68× |
| +Image compression | 2.37× (+30%) | 1.92× | 1.83× |
| +Global visual injection | 2.42× (+7%) | 2.03× | 1.95× |
| +Dataset generation | **3.22×** (+30%) | **2.22×** | **2.55×** |

| # Compressed Embeddings | COCO $\tau$ | COCO Speedup | GQA $\tau$ | GQA Speedup |
|------------------------|------------|-------------|-----------|------------|
| 1 | 3.30 | **3.22×** | 2.88 | **2.22×** |
| 4 | 3.24 | 3.24× | 2.84 | 2.24× |
| 64 | 3.25 | 2.71× | 2.86 | 1.91× |

### Key Findings

- ViSpec consistently outperforms Medusa and EAGLE-2 across all models and tasks, with speedup ratios ranging from 1.37× to 3.22×.
- Longer output sequences yield higher speedup ratios (TextVQA: 353.58 tokens → 2.90×; GQA: 46.25 tokens → 2.22×).
- The LLaVA series benefits more from acceleration than Qwen2.5-VL, as the latter's larger vocabulary increases token prediction complexity.
- The visual adapter introduces no significant prefill latency overhead (within measurement noise levels).
- On video tasks (MSVD-QA, MVBench), speedups of 1.32×–1.46× are achieved without any video-specific training.

## Highlights & Insights

1. **Integration of theory and practice**: The paper clearly explains from an attention mechanism perspective why shallow draft models struggle with redundant visual tokens, providing well-grounded motivation.
2. **Minimalist design principle**: A single compressed visual embedding suffices to capture key information, embodying the "less is more" philosophy.
3. **Elegant global feature injection**: A simple yet effective solution to the visual forgetting problem in long text generation.
4. **First work to break the 2× speedup barrier in VLM speculative decoding**, establishing a new benchmark for this research direction.

## Limitations & Future Work

- Absolute speedup ratios still lag behind state-of-the-art methods for text-only speculative decoding.
- Training data relies on target model generation, limiting the quality and diversity of synthetic data.
- The visual encoder architecture remains unoptimized (e.g., dynamic patch reduction, neural compression).
- Acceleration gains are noticeably weaker for the Qwen2.5-VL series than for LLaVA, warranting further investigation in large-vocabulary settings.
- For high-resolution images, end-to-end speedup is diluted by increased prefill time.

## Related Work & Insights

- The EAGLE series (target-aware feature injection) inspired the hidden state input design of ViSpec's draft model.
- The query–key–value compression scheme from Q-Former (BLIP-2) is adopted for the visual adapter design.
- The multi-token prediction strategy from DeepSeek-V3 is employed to prevent shortcut learning during training.
- This work provides a direct extension direction for accelerating video VLMs.

## Rating

- Novelty: ⭐⭐⭐⭐ — First framework specifically designed for VLM speculative decoding, with complete theoretical analysis and solution design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 4 models × 8 datasets × 2 temperatures, with detailed ablation analysis.
- Writing Quality: ⭐⭐⭐⭐ — Theoretical derivations are clear, though notation is dense.
- Value: ⭐⭐⭐⭐⭐ — A directly deployable VLM inference acceleration solution with open-source code.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Balanced Token Pruning: Accelerating Vision Language Models Beyond Local Optimization](balanced_token_pruning_accelerating_vision_language_models_b.md)
- [\[NeurIPS 2025\] Enhancing Vision-Language Model Reliability with Uncertainty-Guided Dropout Decoding](enhancing_visionlanguage_model_reliability_with_uncertaintyg.md)
- [\[NeurIPS 2025\] Zero-Shot Robustness of Vision Language Models Via Confidence-Aware Weighting](zero-shot_robustness_of_vision_language_models_via_confidence-aware_weighting.md)
- [\[NeurIPS 2025\] Test-Time Spectrum-Aware Latent Steering for Zero-Shot Generalization in Vision-Language Models](test-time_spectrum-aware_latent_steering_for_zero-shot_generalization_in_vision-.md)
- [\[NeurIPS 2025\] Scene-Aware Urban Design: A Human-AI Recommendation Framework Using Co-Occurrence Embeddings and Vision-Language Models](scene-aware_urban_design_a_human-ai_recommendation_framework_using_co-occurrence.md)

<!-- RELATED:END -->
