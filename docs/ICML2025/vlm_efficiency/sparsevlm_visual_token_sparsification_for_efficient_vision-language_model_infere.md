---
title: >-
  [Paper Note] SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference
description: >-
  [ICML 2025][Multimodal Efficiency][visual token pruning] SparseVLM proposes the first training-free, text-guided visual token sparsification framework. By selecting vision-related text tokens as "raters" to evaluate the importance of visual tokens, combined with an adaptive pruning ratio and a token recycling mechanism, it preserves 99.1% of the original performance on LLaVA while retaining only 192 tokens (a 66.7% reduction).
tags:
  - "ICML 2025"
  - "Multimodal Efficiency"
  - "visual token pruning"
  - "VLM efficiency"
  - "text-guided sparsification"
  - "token recycling"
  - "training-free"
date: 2026-05-08
content_hash: 5ceacddf7f022a81
---

# SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference

**Conference**: ICML 2025  
**arXiv**: [2410.04417](https://arxiv.org/abs/2410.04417)  
**Code**: [https://github.com/Gumpest/SparseVLMs](https://github.com/Gumpest/SparseVLMs)  
**Area**: Multimodal VLMs  
**Keywords**: visual token pruning, VLM efficiency, text-guided sparsification, token recycling, training-free

## TL;DR

SparseVLM proposes the first training-free, text-guided visual token sparsification framework. By selecting vision-related text tokens as "raters" to evaluate the importance of visual tokens, combined with an adaptive pruning ratio and a token recycling mechanism, it preserves 99.1% of the original performance on LLaVA while retaining only 192 tokens (a 66.7% reduction).

## Background & Motivation

**Background**: Current vision-language models (VLMs) encode high-resolution images into a massive number of visual tokens—for instance, a 672×672 image produces 2,304 visual tokens in LLaVA, occupying over half of the context length. In video-based tasks, the number of tokens can reach several thousands, resulting in enormous computational overhead.

**Limitations of Prior Work**: Existing visual token compression methods suffer from two types of limitations: (1) methods that modify the vision encoder or projection layer (such as Q-Former and DeCo) require additional training; and (2) methods that perform pruning during the decoding phase (such as FastV) ignore the guidance of text tokens, resulting in text-independent sparsification. When facing different questions, the model may need to focus on different regions of an image (foreground or background), and static pruning strategies inevitably cause loss of critical information.

**Key Challenge**: Image information is inherently sparser (containing massive redundancy) compared to text, but existing methods either require additional training overhead or fail to consider the semantic guidance of text queries during pruning, leading to an overly aggressive trade-off between accuracy and efficiency.

**Goal**: (1) To achieve highly efficient visual token sparsification without requiring any training; (2) to leverage textual questions to guide the pruning process, adaptively retaining question-relevant visual information based on specific queries; and (3) to mitigate the information loss of the pruned tokens.

**Key Insight**: The self-attention matrix of the VLM decoder naturally contains text-to-visual interaction information, which can be directly reused to determine the importance of visual tokens without requiring additional parameters.

**Core Idea**: Leveraging the existing self-attention matrix of the VLM to perform text-guided visual token filtering, adaptively determining the pruning ratio using the matrix rank, and recycling information from pruned tokens to minimize information loss.

## Method

### Overall Architecture

SparseVLM is embedded into VLM decoder layers as a plug-and-play module. Given an image and a textual question, it first pre-selects "text raters" (vision-related text tokens) before entering the decoder. Then, in each decoder layer, the text-visual interaction part of the self-attention matrix is extracted to compute importance scores for visual tokens. The number of pruned tokens in the current layer is adaptively determined based on the rank of the attention matrix. Finally, the information-rich parts of the pruned tokens are clustered and compressed into compact reconstructed tokens.

### Key Designs

1. **Text Rater Selection**:

    - **Function**: To identify text tokens in the question that are genuinely relevant to the visual content, excluding irrelevant words such as prepositions and pronouns.
    - **Mechanism**: Before entering the decoder, the cross-attention between visual embedding $H_v$ and text embedding $H_q$ is calculated as $r = \frac{1}{L_v}\sum_{j=1}^{L_v}(\text{Softmax}(H_v H_q^T))_j$. Text tokens with a correlation score exceeding the mean $m = \text{mean}(r)$ are selected as raters. For instance, for medicine-related questions, key terms like "Tylenol" and "Advil" are selected as the basis for visual filtering.
    - **Design Motivation**: Not all question words are related to visual content, and involving irrelevant words in the rating process would lead to inaccurate correlation calculation. Experiments show that this selection mechanism brings a 2.7% improvement on POPE compared to omitting the filtering step.

2. **Sparsification Level Adaptation**:

    - **Function**: To adaptively determine the number of tokens to be pruned in each layer based on its level of information redundancy.
    - **Mechanism**: The rank of the priority matrix $P$ (the attention sub-matrix between text queries and visual keys) is utilized to measure redundancy. The difference between the dimension and the rank indicates the redundancy level. The number of discarded tokens is determined using a scaling factor $\lambda$ as $N = \lambda \times (L_v - \text{rank}(P))$. A high matrix rank indicates that the visual tokens are highly linearly independent with low redundancy, meaning fewer tokens should be pruned; a low rank suggests high redundancy, allowing more tokens to be pruned. If $N = 0$ is calculated for a layer, the pruning for that layer is skipped.
    - **Design Motivation**: Different images exhibit varying levels of information density; static-ratio pruning strategies waste computation on simple images and lose essential information on complex ones.

3. **Visual Token Recycling and Reconstruction**:

    - **Function**: To salvage the relatively information-rich parts of the pruned tokens and compress them into compact reconstructed tokens.
    - **Mechanism**: The top $\tau$% highest-priority tokens among the discarded ones are selected for recycling. A Density Peak Clustering (DPC) algorithm is used for adaptive clustering: first, the local density $\rho_i$ of each token is computed (based on k-nearest neighbor distances), followed by calculating the distance metric $\delta_i$ (the minimum distance to any token of higher density). Tokens with high $\rho_i \times \delta_i$ are selected as cluster centers. Finally, tokens within the same cluster are compressed into a single reconstructed token through element-wise summation.
    - **Design Motivation**: Although pruned tokens have lower overall importance, the relatively higher-priority ones among them still contain useful information. The recycling mechanism is particularly effective under high compression rates—when pruning from 192 tokens to 64 tokens on POPE, the improvement brought by the recycling mechanism increases from 1.5% to 17.7%.

### Loss & Training

SparseVLM is a training-free method and does not introduce additional loss functions. All operations (attention matrix extraction, rater selection, rank computation, clustering-based recycling) are performed online during inference.

## Key Experimental Results

### Main Results

Performance of SparseLLaVA (LLaVA + SparseVLM) across 8 benchmarks (576 $\rightarrow$ 192 tokens, a 66.7% reduction):

| Method | GQA | MMB | MME | POPE | SQA | SEED | TextVQA | MMVet | Avg. Accuracy |
|---|---|---|---|---|---|---|---|---|---|
| Vanilla (576 tokens) | 61.9 | 64.6 | 1864 | 85.9 | 69.5 | 60.3 | 58.3 | 30.9 | 100% |
| FastV (192 tokens) | 52.6 | 61.0 | 1605 | 64.8 | 69.1 | 52.1 | 52.5 | 26.7 | 87.9% |
| SparseVLM (192 tokens) | 59.5 | 64.1 | 1787 | 85.3 | 68.7 | 58.7 | 57.8 | 33.1 | **99.1%** |

### Ablation Study

Effect of Token Recycling (TR) under different compression rates (LLaVA 7B):

| Benchmark | 64 tokens | 96 tokens | 128 tokens | 192 tokens | Average |
|---|---|---|---|---|---|
| GQA | 52.2→53.8 | 55.2→56.4 | 58.1→58.4 | 59.4→59.5 | +0.8 |
| POPE | 72.8→77.5 | 77.5→81.9 | 83.7→85.0 | 85.2→85.3 | +2.6 |

### Key Findings

- Under a 4.5$\times$ visual token compression rate (576 $\rightarrow$ 128), SparseVLM maintains 96.7% of the original performance, with only a 3.3% drop.
- CUDA latency is reduced by 37% (57.82ms $\rightarrow$ 36.50ms), and FLOPs are reduced by 53.7% (4.62T $\rightarrow$ 2.14T), while accuracy only decreases by 0.9%.
- On video tasks (VideoLLaVA, 90.5% pruning rate), SparseVLM achieves an overall average accuracy of 95.0%, outperforming FastV by 14.7 percentage points.
- By removing 54.5% of visual tokens on Qwen2-VL, it maintains 98.0% accuracy, validating the universality of the method on dynamic-resolution models.

## Highlights & Insights

- This method is the first training-free, text-guided visual token sparsification framework for VLMs. It is plug-and-play without requiring fine-tuning, offering extremely high practical value.
- The design of token recycling based on Density Peak Clustering is elegant: instead of simply discarding pruned tokens, it salvages and compresses information from them, yielding more pronounced benefits at higher compression rates.
- Utilizing the rank of the attention matrix to measure redundancy is an ingenious, parameter-free design choice.

## Limitations & Future Work

- It is required to extract the full attention matrix from FlashAttention; although a compatible solution is proposed in the paper, it still introduces extra overhead.
- The computational complexity of the matrix rank is $O(L_t \times L_v \times \min(L_t, L_v))$, which may become a bottleneck for very long sequences.
- The sensitivity analysis of hyper-parameters $\lambda$, $\tau$, and $\theta$ is insufficient.
- Validation is restricted to single-image scenarios; the applicability to multi-image interleaved dialogue has not been explored.
- The text rater selection might degenerate when there are very few text tokens (e.g., "Describe this image").
- The performance on modern large-scale VLMs, such as LLaVA-OneVision and InternVL 2.5, has not yet been verified.
- The overhead of layer-wise rank computation and clustering may become a bottleneck in extremely long sequence scenarios (e.g., high-resolution videos).

### Supplementary Analysis

Visualization results (Figure 6) clearly illustrate the attention focusing process of SparseVLM: as the decoder depth increases, the retained tokens progressively focus on question-relevant regions (e.g., focusing on colored areas when asked "what color"), demonstrating the effectiveness of text-guided selection.

## Related Work & Insights

- FastV (ECCV 2024) is the most direct baseline, which also prunes tokens during decoding but disregards text guidance; SparseVLM significantly outperforms it under all settings.
- ToMe (ICLR 2023) utilizes token merging, but direct merging causes a performance drop under extreme compression levels.
- Insight: The text-guided visual processing concept can be extended to other multimodal efficiency optimization scenarios, such as multimodal retrieval, frame selection in video understanding, etc.
- PDrop (CVPR 2025) offers more lightweight computation, but falls behind SparseVLM in both accuracy and latency, demonstrating that the extra cost of attention matrix calculation is well worth it.
- VocoLLaMA (Ye et al., 2025) requires training, which is complementary to the training-free approach of SparseVLM; exploring their combination is a promising future direction.
- The KV Cache reduction of 67% possesses crucial engineering value for edge-device deployment and long-context multimodal inference scenarios.

## Rating

⭐⭐⭐⭐ (7.5/10)

The method design is complete and highly practical—training-free, plug-and-play, generalization across different models, and suitable for both images and videos. It achieves high efficiency gains while maintaining excellent accuracy. The experiments are thorough (covering 3 VLM architectures, 8 image benchmarks, and 4 video benchmarks) with detailed ablation analyses. A minor weakness lies in the limited theoretical novelty (individual components like matrix rank and DPC clustering are combinations of existing works), and the hyper-parameter sensitivity analysis is insufficient.

Specifically, the real deployment value of SparseVLM is substantial: saving 67% of KV Cache memory (reducing from 302.4MB to 100.8MB) is of great significance for inference on edge devices. Its capability to retain 95% accuracy under a 90.5% pruning rate in video tasks is also impressive. Future research directions may include: combining this method with KV Cache compression techniques, optimization for multi-image multi-round dialogues, and extending the text-guidance framework to other modality combinations like audio-text.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Feather the Throttle: Revisiting Visual Token Pruning for Vision-Language Model Acceleration](../../ICCV2025/vlm_efficiency/feather_the_throttle_revisiting_visual_token_pruning_for_vision-language_model_a.md)
- [\[CVPR 2026\] TransPrune: Token Transition Pruning for Efficient Large Vision-Language Model](../../CVPR2026/vlm_efficiency/transprune_token_transition_pruning_for_efficient_large_vision-language_model.md)
- [\[ICCV 2025\] AirCache: Activating Inter-Modal Relevancy KV Cache Compression for Efficient Large Vision-Language Model Inference](../../ICCV2025/vlm_efficiency/aircache_activating_inter-modal_relevancy_kv_cache_compression_for_efficient_lar.md)
- [\[NeurIPS 2025\] FlowCut: Rethinking Redundancy via Information Flow for Efficient Vision-Language Models](../../NeurIPS2025/vlm_efficiency/flowcut_rethinking_redundancy_via_information_flow_for_effic.md)
- [\[ICML 2025\] CoreMatching: A Co-adaptive Sparse Inference Framework with Token and Neuron Pruning for Comprehensive Acceleration of Vision-Language Models](corematching_a_co-adaptive_sparse_inference_framework_with_token_and_neuron_prun.md)

</div>

<!-- RELATED:END -->
