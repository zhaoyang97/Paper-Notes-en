---
title: >-
  [Paper Note] AirCache: Activating Inter-Modal Relevancy KV Cache Compression for Efficient Large Vision-Language Model Inference
description: >-
  [ICCV 2025][Multimodal VLM][KV Cache compression] This paper proposes AirCache, a KV Cache compression method for LVLMs that evaluates visual token importance via an Elite Observation Window…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "KV Cache compression"
  - "large vision-language models"
  - "cross-modal attention"
  - "layer-wise budget allocation"
  - "inference acceleration"
date: 2026-05-08
content_hash: 800ad5fe2b6995c8
---

# AirCache: Activating Inter-Modal Relevancy KV Cache Compression for Efficient Large Vision-Language Model Inference

**Conference**: ICCV 2025
**arXiv**: N/A (CVF OpenAccess)
**Code**: Unavailable
**Area**: Multimodal VLM
**Keywords**: KV Cache compression, large vision-language models, cross-modal attention, layer-wise budget allocation, inference acceleration

## TL;DR

This paper proposes AirCache, a KV Cache compression method for LVLMs that evaluates visual token importance via an Elite Observation Window, combined with adaptive layer-wise budget allocation based on the intensity and skewness of importance score distributions. At only 10% visual KV Cache retention, performance degradation remains within 1%, while decoding latency is reduced by 29%–66%.

## Background & Motivation

### KV Cache Bottleneck
Large vision-language models (LVLMs) face severe KV Cache memory pressure during inference:
- **Explosion of visual tokens**: High-resolution inputs, multi-image scenarios, and video sequences cause exponential growth in visual token counts.
- **Long-context generation**: KV Cache grows linearly with sequence length, leading to unsustainable latency and memory consumption.

### Comparison of Two Acceleration Paradigms

| Category | Stage | Advantages | Disadvantages |
|----------|-------|------------|---------------|
| Token Pruning | Prefill | Reduces computation and KV Cache for subsequent layers simultaneously | Discards tokens before forward propagation, causing irreversible loss of visual information |
| KV Cache Compression | Decoding | Tokens have completed full forward propagation; attention mechanisms have differentiated importance | Does not accelerate prefill |

**Why is KV Cache compression superior?** Because all tokens have completed full forward propagation, causal attention has already established importance differences among tokens. Selectively removing certain tokens has minimal impact on model performance. Token pruning discards tokens before prefill, resulting in permanent loss of visual information.

### Limitations of Existing KV Cache Methods

The authors conduct an in-depth analysis of issues with existing methods in LVLM settings:

**Inappropriate observation window selection**: Methods such as SnapKV use all text tokens or contiguous local text tokens as the window for evaluating visual token importance. However, different text tokens assign vastly different scores to the same visual token, causing the voting mechanism to introduce substantial noise.

**Suboptimal uniform budget allocation**: Different layers exhibit significantly different levels of attention to visual tokens; uniform budget allocation is not optimal.

### Core Observations

Through experimental analysis (Figure 1), the authors find that:
- Using all text tokens to evaluate visual token importance yields poor consistency.
- Using the last 16 text tokens produces unstable results.
- Using the last 16 visual tokens lacks text-side guidance.
- Only approximately 10% of visual tokens have a significantly positive impact on the final output (head effect).

## Method

### Overall Architecture

AirCache performs a one-time KV Cache compression after the prefill stage, comprising two core components:

1. **Elite Observation Window**: Selects key text tokens to form an elite observation window for evaluating visual token importance.
2. **Layer-wise Budget Allocation**: Adaptively allocates compression budgets across layers based on the intensity and skewness of importance score distributions.

The method executes after prefill completion and is compatible with mainstream LVLM architectures.

### Key Designs 1: Elite Observation Window

**Design Motivation**: Different text tokens assign vastly different importance scores to the same visual token. When text tokens within the observation window disagree on the same visual token, the voting mechanism introduces substantial noise, which is further amplified by cross-modal discrepancies.

**Mechanism**: Key text tokens are selected using intra-text self-attention.

The hidden states of the input prompt are first decomposed into visual and text components:
$$X = \text{Concat}(X_v, X_t) \in \mathbb{R}^{(N_v + N_t) \times D}$$

The unimodal text attention matrix is computed as:
$$\text{Att} = \text{Softmax}\left(\frac{Q_t K_t^T}{\sqrt{D}}\right) \in \mathbb{R}^{N_t \times N_t}$$

Using the last text token as a reference, key text tokens that receive high attention scores are selected:
$$k = \{j | \text{Att}[N_t-1, j] \geq \alpha \cdot \max \text{Att}[N_t-1, :]\}$$

where $\alpha \in [0,1]$ is the relevancy threshold (set to 0.9 in experiments).

Cross-modal attention is then computed using the queries of these elite text tokens against the keys of visual tokens and elite text tokens:
$$A_{vtk} = \text{Softmax}\left(\frac{Q_{tk} K_{vtk}^T}{\sqrt{D}}\right)$$

The final visual token importance scores are obtained via average pooling along the text dimension:
$$I_v = \frac{1}{N_{tk}} \sum_{j=0}^{N_{tk}-1} A_{vtk}[j, :N_v]$$

**Why is the elite window superior?**
- Text tokens within the elite window tend to assign more consistent scores to the same visual token.
- Noise is reduced and the accuracy of importance ranking is improved.
- Computational complexity is lower, since the number of elite text tokens is far smaller than the total number of text tokens.

### Key Designs 2: Layer-wise KV Cache Budget Allocation

The authors observe that different layers exhibit significant variation in their attention to visual information, necessitating differentiated budget allocation. Two dimensions are used for quantification:

**Dimension 1: Distribution Strength**
$$s_t = \sum_{i=0}^{N_v-1} I_v[i]$$

This excludes intra-text attention and aggregates only the total attention scores from text tokens to all visual tokens. A larger value indicates that the layer pays greater attention to visual information, warranting a larger budget.

**Dimension 2: Distribution Skewness**
$$s_k = \frac{N_v}{(N_v-1)(N_v-2)} \sum_{i=1}^{N_v} \left(\frac{I_v[i] - \mu_{I_v}}{\sigma_{I_v}}\right)^3$$

Skewness measures the degree of "head effect" in the attention distribution. High skewness indicates that a small number of visual tokens receive disproportionately high attention, suggesting that the layer's attention allocation is more precise and informative; such layers should be allocated more budget to preserve these dominant tokens.

**Final budget formula**:
$$\hat{r} = \frac{1}{2}(s'_t + s'_k) \cdot r$$

where $s'_t$ and $s'_k$ are the cross-layer normalized strength and skewness, respectively, and $r$ is the base budget.

**Why combine strength and skewness?**
- Strength alone indicates which layers are more important, but not whether attention is precisely focused.
- Skewness alone indicates whether attention is focused, but not the layer's overall degree of visual engagement.
- Their combination accounts for both the quantity of visual attention and its precision, yielding a more comprehensive signal.

### Loss & Training

**No training is required.** AirCache executes in a single pass after prefill:
1. The complete KV Cache is saved after each layer completes prefill.
2. After all layers have completed, layer-wise budgets are computed and compression is applied uniformly.
3. The compressed KV Cache is used for subsequent decoding.

The relevancy threshold is set to $\alpha = 0.9$; experiments are conducted on 8×A100-80G GPUs.

## Key Experimental Results

### Main Results

**VQA Benchmarks (LLaVA-OV-7B) — Visual KV Cache Retention Ratio vs. Performance**:

| Method | ChatQA 10% | InfoVQA 10% | DocVQA 10% | TextVQA 10% | ChatQA 1% | DocVQA 1% |
|--------|-----------|------------|-----------|-----------|----------|----------|
| Full | 80.3 | 66.1 | 87.0 | 76.0 | 80.3 | 87.0 |
| H2O | 77.4 | 59.2 | 74.2 | 70.1 | 71.0 | 55.3 |
| SnapKV | 79.3 | 64.2 | 84.4 | 73.4 | 72.9 | 64.1 |
| PrefixKV | 78.2 | 61.1 | 80.5 | 72.7 | 70.9 | 55.4 |
| **AirCache** | **79.9** | **65.7** | **85.5** | **75.3** | **76.4** | **73.2** |

At 10% retention, performance degradation is within approximately 1%. At 1% retention, AirCache substantially outperforms all baselines (ChatQA: +3.5 vs. SnapKV; DocVQA: +9.1 vs. SnapKV).

**Multi-Model Generalization (10% Retention)**:

| Model | Method | ChatQA | InfoVQA | DocVQA | TextVQA |
|-------|--------|--------|---------|--------|---------|
| InternVL2-8B | SnapKV | 80.4 | 72.3 | 90.1 | 73.9 |
| InternVL2-8B | **AirCache** | **81.7** | **72.6** | **90.0** | **77.0** |
| Qwen2-VL-7B | SnapKV | 81.6 | 74.9 | 87.2 | 83.1 |
| Qwen2-VL-7B | **AirCache** | **82.3** | **75.2** | **92.9** | **83.4** |

**Inference Acceleration**:

| Batch Size | Prompt Length | Decoding Latency Reduction (50%) | Decoding Latency Reduction (10%) | Throughput Gain (10%) |
|------------|-------------|----------------------------------|----------------------------------|-----------------------|
| 8 | 2k | -19.0% | -29.3% | +41.6% |
| 8 | 32k | -38.7% | -65.7% | +192.1% |
| 16 | 16k | -37.7% | -65.3% | +188.3% |

Acceleration benefits become more pronounced with longer inputs. At batch size 16 and prompt length 16k, a 10% retention rate yields a throughput improvement of **188.3%**.

### Ablation Study

**Elite Observation Window vs. Alternative Windows (LLaVA-OV-7B, 1% Retention)**:

| Observation Window | ChatQA | InfoVQA | DocVQA | TextVQA |
|-------------------|--------|---------|--------|---------|
| Contiguous window (16) | 70.4 | 56.6 | 61.3 | 55.9 |
| Contiguous window (32) | 72.9 | 57.8 | 64.1 | 58.2 |
| All text tokens | 72.2 | 58.4 | 65.7 | 57.0 |
| Visual window (32) | 68.8 | 55.1 | 59.2 | 53.7 |
| **Elite window (Ours)** | **76.4** | **62.5** | **73.2** | **67.1** |

The elite window significantly outperforms all alternatives; the pure visual window performs worst due to the absence of text-side guidance.

**Layer-wise Budget Allocation Ablation**:

| Allocation Strategy | ChatQA | InfoVQA | DocVQA | TextVQA |
|--------------------|--------|---------|--------|---------|
| Uniform allocation | 72.2 | 57.5 | 69.9 | 62.4 |
| Pyramid allocation (PyramidKV) | 69.6 | 54.9 | 55.8 | 52.6 |
| Strength only ($s_t$) | 74.2 | 59.8 | 71.1 | 64.9 |
| Skewness only ($s_k$) | 74.7 | 61.4 | 71.9 | 63.6 |
| **Strength + Skewness (Ours)** | **76.4** | **62.5** | **73.2** | **67.1** |

PyramidKV's pyramid allocation, which is effective in LLMs, degrades performance in LVLMs, suggesting that multimodal models exhibit unique layer-wise characteristics.

**Comparison with Token Pruning Methods (1% Retention)**:

| Method | ChatQA | MMBench | MME | Prefill Latency | Decode Latency |
|--------|--------|---------|-----|-----------------|----------------|
| FastV | 16.9 | 33.6 | 786 | 5.4s | 11.7s |
| IVTP | 22.5 | 36.2 | 849 | 5.8s | 12.6s |
| **AirCache** | **76.4** | **82.3** | **1585** | 9.8s | 11.8s |

Token pruning methods collapse in performance under extreme compression ratios, while AirCache maintains strong performance.

### Key Findings

1. **10% visual KV Cache suffices**: Retaining only 10% of visual tokens incurs an average performance loss of less than 1%, validating the pronounced head effect in visual token importance distributions.
2. **Cross-modal relevancy is critical**: The elite observation window leverages intra-text self-attention to select key text tokens, which are then used to evaluate visual token importance — yielding more stable and accurate results than using all text tokens or visual tokens directly.
3. **Layer-wise heterogeneity cannot be ignored**: Different layers in LVLMs exhibit dramatic variation in visual attention; strategies effective in LLMs (e.g., PyramidKV) are actually harmful in LVLMs.
4. **KV Cache compression outperforms token pruning**: Cross-modal interactions during prefill consolidate visual information into key text tokens; even when large numbers of visual KV Cache entries are removed, critical information is preserved through the text tokens.

## Highlights & Insights

1. **The Elite Observation Window is the core innovation**: Selecting "representative" text tokens via intra-text self-attention to evaluate visual token importance directly addresses the consistency problem of observation windows.
2. **Dual-dimensional budget allocation**: The combination of strength (quantity of attention) and skewness (precision of attention) outperforms any single metric or heuristic rule.
3. **In-depth analysis of LVLM attention mechanisms**: The paper reveals layer-wise differences in the distributional characteristics of visual token importance, providing important reference points for future research.
4. **High practical deployment value**: A 10% retention rate, sub-1% performance loss, and up to 66% decoding latency reduction make AirCache well-suited for large-scale deployment.
5. **Thorough comparison of KV Cache compression vs. token pruning**: The paper clearly demonstrates the fundamental distinction between the two paradigms — information consolidation during prefill gives KV Cache compression a natural advantage over token pruning.

## Limitations & Future Work

1. **No prefill acceleration**: AirCache applies compression after prefill and does not reduce prefill time (which may even increase slightly by approximately 5–12%).
2. **One-shot compression**: KV Cache compression is applied uniformly across all layers after prefill completion; dynamic adjustment during decoding is not supported.
3. **Hyperparameter sensitivity**: The relevancy threshold $\alpha = 0.9$ may require task-specific tuning.
4. **Long-output scenarios**: The paper notes that VQA datasets (requiring longer outputs) better demonstrate the benefits of KV Cache compression, while short-output tasks (e.g., multiple choice) are less sensitive to KV Cache size.
5. **Integration with token pruning**: The two paradigms are not mutually exclusive; combining prefill acceleration with decoding acceleration may yield more comprehensive gains.

## Related Work & Insights

- **SnapKV**: Uses a local window strategy to evaluate token importance; AirCache's elite window constitutes a significant improvement over this approach.
- **PyramidKV**: Employs pyramid budget allocation in LLMs, but fails in LVLMs — indicating that multimodal models possess unique characteristics.
- **VL-Cache**: Exploits visual token sparsity for layer-wise budget allocation; AirCache further introduces the skewness dimension.
- **Core Insight**: KV Cache compression in LVLMs cannot straightforwardly adopt LLM-based methods; cross-modal differences require dedicated treatment. Intra-text self-attention can serve as an effective bridge for establishing cross-modal relevancy.

## Rating

- Novelty: ⭐⭐⭐⭐ (The Elite Observation Window and dual-dimensional budget allocation are novel contributions)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Three model architectures, multiple retention ratios, comprehensive ablations, measured latency and throughput)
- Writing Quality: ⭐⭐⭐⭐⭐ (Motivation is clearly articulated; ablation study is progressively structured)
- Value: ⭐⭐⭐⭐ (Strong practical utility, though the lack of prefill acceleration limits overall impact)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] FOLDER: Accelerating Multi-modal Large Language Models with Enhanced Performance](folder_accelerating_multi-modal_large_language_models_with_enhanced_performance.md)
- [\[ICCV 2025\] The Inter-Intra Modal Measure: A Predictive Lens on Fine-Tuning Outcomes in Vision-Language Models](the_inter-intra_modal_measure_a_predictive_lens_on_fine-tuning_outcomes_in_visio.md)
- [\[ICCV 2025\] Scaling Inference-Time Search with Vision Value Model for Improved Visual Comprehension](scaling_inferencetime_search_with_vision_value_model_for_imp.md)
- [\[ICCV 2025\] Enhancing Few-Shot Vision-Language Classification with Large Multimodal Model Features](enhancing_few-shot_vision-language_classification_with_large_multimodal_model_fe.md)
- [\[ICCV 2025\] SparseVILA: Decoupling Visual Sparsity for Efficient VLM Inference](sparsevila_decoupling_visual_sparsity_for_efficient_vlm_inference.md)

</div>

<!-- RELATED:END -->
