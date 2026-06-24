---
title: >-
  [Paper Note] FlexAttention for Efficient High-Resolution Vision-Language Models
description: >-
  [ECCV 2024][Multimodal VLM][High-Resolution VLM] This paper proposes FlexAttention, which reduces computational costs by nearly 40% while maintaining or even exceeding the performance of existing high-resolution VLMs, achieved through dynamic high-resolution token selection based on attention maps and a hierarchical self-attention fusion mechanism.
tags:
  - "ECCV 2024"
  - "Multimodal VLM"
  - "High-Resolution VLM"
  - "Attention Mechanism"
  - "Dynamic Token Selection"
  - "Hierarchical Self-Attention"
  - "Computational Efficiency"
date: 2026-05-08
content_hash: 468ae628a0c968fc
---

# FlexAttention for Efficient High-Resolution Vision-Language Models

**Conference**: ECCV 2024  
**arXiv**: [2407.20228](https://arxiv.org/abs/2407.20228)  
**Code**: [https://vis-www.cs.umass.edu/flexattention](https://vis-www.cs.umass.edu/flexattention)  
**Area**: Multimodal VLM  
**Keywords**: High-Resolution VLM, Attention Mechanism, Dynamic Token Selection, Hierarchical Self-Attention, Computational Efficiency

## TL;DR

This paper proposes FlexAttention, which reduces computational costs by nearly 40% while maintaining or even exceeding the performance of existing high-resolution VLMs, achieved through dynamic high-resolution token selection based on attention maps and a hierarchical self-attention fusion mechanism.

## Background & Motivation

**Background**: Mainstream VLMs (e.g., LLaVA-1.5, InstructBLIP) often restrict input images to lower resolutions (e.g., 224×224 or 336×336) due to their reliance on fixed-resolution visual encoders like CLIP. This leads to poor performance in scenarios requiring the recognition of fine details, such as small text or small objects.

**Limitations of Prior Work**: While existing high-resolution VLMs (e.g., LLaVA-1.5-HD, CogAgent) can accept high-resolution images, they feed all high-resolution tokens into the attention modules, causing the computational cost to grow quadratically with the number of tokens. For example, scaling the resolution from 336 to 1008 increases the token count by 9 times, resulting in an approximately 81-fold increase in attention computation.

**Key Challenge**: There is a fundamental trade-off between the visual detail provided by high resolution and the computational overhead caused by $O(N^2)$ complexity of self-attention. Existing methods either sacrifice resolution or suffer from massive computational costs.

**Human Vision Inspiration**: Human visual processing does not memorize all pixel-level details at once. Instead, it maintains a coarse global representation and only focuses on regions of interest with finer granularity when stimulated (selective attention mechanism).

**Key Insight**: The authors observe that during the generation process of VLMs, the model's attention maps naturally reveal which image regions are currently important. This "free" signal can be leveraged to dynamically select regions requiring high-resolution details, rather than exhaustively processing all high-resolution tokens.

**Core Idea**: Dynamically select around 10% of critical high-resolution tokens using attention maps and integrate them into low-resolution representations via hierarchical self-attention, achieving high-resolution perception with sub-linear computational growth.

## Method

### Overall Architecture

FlexAttention can replace the self-attention modules of existing VLMs in a plug-and-play manner. The overall pipeline is as follows:

- **Input**: A high-resolution image $I_{HR}$ (e.g., 1008×1008) and text input $T$.
- **Dual-Path Encoding**: Simultaneously encode the high-resolution image into high-resolution tokens $f_{HR}$ and low-resolution tokens $f_{LR}$ (encoded after downsampling).
- **First Half ($N_{SA}$ layers)**: Utilize only low-resolution image tokens and text tokens for standard self-attention computation to build a coarse understanding of the overall image.
- **Second Half ($N_{FA}$ layers)**: Enable FlexAttention, where a small subset of high-resolution tokens is dynamically selected in each layer and high-resolution details are fused via hierarchical self-attention.
- **Output**: The hidden state of the final layer is projected to generate the text answer.

The key to this design is: the first few layers establish a global understanding at low cost, and then high-resolution details are introduced in later layers, with only a small fraction (~10%) of high-resolution tokens incorporated in each layer.

### Key Designs

#### 1. High-Resolution Feature Selection Module

- **Function**: Select a small subset of the most relevant tokens from all high-resolution tokens according to the current layer's attention map.
- **Mechanism**:
    - Extract the first $N_i$ values from the last column of the attention map $Map$ (i.e., the attention weights of all low-resolution image tokens relative to the last text token), which reflect the model's focus on different image regions when generating the next token.
    - Reshape this 1D attention vector into a 2D spatial attention map.
    - Normalize and binarize the attention map to obtain a binary mask.
    - Upsample the mask to the spatial dimensions of the high-resolution feature map to form a high-resolution selection mask.
    - Apply the mask to select the "activated" high-resolution tokens $f_{SHR}$, which account for approximately 10% of the total high-resolution tokens.
- **Design Motivation**: Self-attention maps naturally serve as a "free" region importance signal. Leveraging them for token selection avoids the computational overhead of an additional selection network. This selection dynamically changes layer by layer, allowing different layers to focus on different image regions.

#### 2. Hierarchical Self-Attention Module

- **Function**: Integrate the selected high-resolution token information into the original hidden state (low-resolution tokens + text tokens).
- **Mechanism**:
    - Query $Q$ is derived solely from the original hidden state $H$: $Q = HW_Q$.
    - Key and Value are constructed by concatenating the original hidden state and the selected high-resolution tokens:
    - $K_{all} = \text{Concat}(HW_K, f_{SHR}W_K')$
    - $V_{all} = \text{Concat}(HW_V, f_{SHR}W_V')$
    - Note that the high-resolution tokens use independent projection matrices $W_K'$ and $W_V'$ instead of shared key/value projection weights.
    - The output attention map $Map'$ has dimensions $N \times (N+M)$. The first $N \times N$ part is truncated and passed to the next layer for token selection.
- **Design Motivation**: This design ensures that the dimensionality of the hidden state remains unchanged (still $N \times D$). High-resolution tokens participate only in Key/Value calculations without introducing new Queries. Consequently, the computational complexity is reduced from $O((M+N)^2D)$ to $O((M+N)ND)$, scaling linearly rather than quadratically with high-resolution token count.

#### 3. Iterative Layer-by-Layer Selection Mechanism

- **Function**: High-resolution selection and hierarchical attention are executed alternately in each FlexAttention layer.
- **Mechanism**: The attention map $Map^i$ of the $i$-th layer is used to select the high-resolution tokens $f_{SHR}^i$ for the $(i+1)$-th layer, forming an iterative refinement process.
- **Design Motivation**: As network depth increases, the model's understanding of the image gradually deepens, and the regions of interest may vary across different layers. Layer-by-layer iteration allows the model to "focus" on different detailed regions at different layers.

### Loss & Training

- Initialized with pre-trained LLaVA-1.5-7b weights.
- Trained for 1 epoch on the fine-tuning dataset of LLaVA-1.5-7b.
- Batch size = 1152, learning rate = 2e-5, cosine scheduler.
- High-resolution input is set to 1008×1008 (3x the original resolution).
- All evaluations are zero-shot.

## Key Experimental Results

### Main Results: High-Resolution VQA Benchmarks

| Model | Resolution | V* Bench Overall | V* Bench Spatial | MagnifierBench | TextVQA | RSVQA Overall |
|------|--------|-----------------|------------------|----------------|---------|---------------|
| InstructBLIP | 224² | 34.0 | 47.4 | 5.6 | - | - |
| LLaVA-1.5-7b | 336² | 47.6 | 56.6 | 26.8 | 46.0 | 68.4 |
| LLaVA-HD | 448² | 51.8 | 61.8 | 35.0 | 45.6 | 68.4 |
| LLaVA-XAttn | 1008² | 48.2 | 56.6 | 32.2 | 45.5 | 71.1 |
| **LLaVA-FlexAttn** | **1008²** | **54.5** | **64.5** | **35.0** | **48.9** | **72.7** |
| GPT-4V | - | 55.0 | 60.5 | - | - | - |

FlexAttention improves overall performance by 6.9% on V* Bench compared to the base model, and by 2.7% compared to LLaVA-HD. It outperforms LLaVA-HD by 3.3% on TextVQA, and on RSVQA, it surpasses GeoChat (72.7% vs 72.3%), a model specifically designed for remote sensing. On the V* Bench Spatial category, it even surpasses GPT-4V (64.5% vs 60.5%).

### Computational Efficiency Comparison

| Model | MagnifierBench TFLOPs | MagnifierBench Time(s) | TextVQA TFLOPs | TextVQA Time(s) |
|------|----------------------|------------------------|----------------|-----------------|
| LLaVA-HD | 24.9 | 154 | 24.5 | 3273 |
| LLaVA-XAttn | 27.1 | 178 | 26.7 | 3741 |
| **LLaVA-FlexAttn** | **17.1** | **112** | **17.1** | **2839** |

The TFLOPs of FlexAttention are reduced by approximately 31% compared to LLaVA-HD and by 37% compared to LLaVA-XAttn. In actual inference time, it is about 28-37% faster on MagnifierBench and 13-24% faster on TextVQA.

### Ablation Study

| Selection Strategy | MagnifierBench | TextVQA |
|---------|----------------|---------|
| Random (select 10% randomly) | 31.4 | 44.5 |
| Center (select center region) | 30.7 | 45.9 |
| **Attn. Map (attention map selection)** | **35.0** | **48.9** |

| High-Resolution Size | MagnifierBench | TextVQA | TFLOPs |
|-------------|----------------|---------|--------|
| 672×672 (2x) | ~32 | ~45 | ~13 |
| **1008×1008 (3x)** | **35.0** | **48.9** | **17.1** |
| 1344×1344 (4x) | ~36 | ~48.9 | ~23 |

| RefCOCO Subset | LLaVA-1.5 | LLaVA-FlexAttn | Gain |
|------------|-----------|----------------|------|
| Large Objects | 75.9 | 78.8 | +2.9 |
| Small Objects | 41.3 | **51.3** | **+10.0** |
| Overall | 75.4 | 78.4 | +3.0 |

### Key Findings

- **Attention Map Selection Far Outperforms Random/Center Selection**: On TextVQA, attention map selection is 4.4% higher than random selection, demonstrating the effectiveness of attention-based dynamic selection.
- **Diminishing Marginal Returns in Resolution Scaling**: Scaling from 672 to 1008 yields a significant performance boost, but scaling from 1008 to 1344 shows almost no improvement on TextVQA. This is because the average resolution of TextVQA images is around 950×811; scaling beyond the original resolution yields minimal benefits.
- **Significant Gains on Small Objects**: On RefCOCO, the accuracy for small objects improves by 10.0% (much higher than the 2.9% gain for large objects), validating the value of high-resolution inputs for fine-grained visual reasoning.
- **No Loss in General Capabilities**: Performance on general benchmarks like POPE, GQA, and VQAv2 remains on par with the base model, indicating that the introduction of FlexAttention does not degrade the model's fundamental capabilities.

## Highlights & Insights

- **"Free Reuse" of Attention Maps**: FlexAttention elegantly utilizes the self-attention maps themselves as selection signals for high-resolution tokens, eliminating the need for extra selection networks or learnable gating. This design is highly concise and efficient. Since attention maps must be computed anyway, reusing them for selection incurs near-zero additional overhead.
- **Asymmetric Design of Hierarchical Attention**: High-resolution tokens only participate in the calculation of Keys and Values and do not generate Queries. This keeps the hidden state dimensionality unchanged while achieving linear computational growth. This design can be transferred to other scenarios requiring multi-granularity information fusion (e.g., video multi-frame fusion, multi-sensor fusion).
- **Computational Simulation of Human Vision**: Inspired by selective attention theory in cognitive science, the natural visual process of "coarse preview then fine focus" is translated into a concrete computational architecture (coarse global understanding in early layers + dynamic focusing in later layers). This top-down + bottom-up dual-path paradigm is a valuable design pattern.
- **Plug-and-Play Design**: Serving as a direct replacement for standard self-attention modules, FlexAttention can be integrated into various VLM architectures, beyond just LLaVA.

## Limitations & Future Work

- **Fixed Selection Ratio of ~10%**: The paper does not thoroughly discuss whether different tasks require different selection ratios. For tasks requiring global detail like OCR, 10% may be insufficient, whereas for tasks focusing on a single object, 10% may be redundant.
- **Evaluations Restricted to LLaVA-1.5-7b**: The generalizability of the method has not been validated on larger models (e.g., 13B/70B) or other architectures (e.g., QFormer-based, Fuyu-style).
- **Information Loss due to Binarized Selection**: Binarizing attention maps results in a hard selection (selected vs. not selected), discarding continuous gradient information from attention values. Soft or weighted selection mechanisms might further improve performance.
- **Limited to Static Images**: Although the paper mentions that the method can be extended to long-sequence modalities like video and audio, no experiments were conducted to verify this.
- **Fairness of Comparison with LLaVA-HD**: The input resolution of LLaVA-HD is 448×448 while FlexAttention uses 1008×1008. The difference in resolving capability makes the comparison slightly unfair, although FlexAttention achieves lower computational cost.

## Related Work & Insights

- **vs LLaVA-1.5-HD**: LLaVA-HD directly concatenates high-resolution tokens into the sequence for full attention computation, which is simple but computationally expensive. FlexAttention avoids full-sequence computation through dynamic selection and hierarchical attention.
- **vs CogAgent**: CogAgent uses cross-attention at each layer to compute dense correspondences between the hidden state and all high-resolution features, which remains computationally heavy (requiring Key/Value processing for all high-resolution tokens). FlexAttention is more efficient by selecting first and then computing.
- **vs Sparse Attention Methods (e.g., BigBird, Reformer)**: These methods reduce complexity by sparsifying the attention matrix but are domain-agnostic. FlexAttention selectively sparsifies attention by exploiting the spatial structure of visual tokens and the semantic information in attention maps.
- **Insights**: This "coarse-to-fine" dynamic selection paradigm can be transferred to scenarios such as video understanding (focusing first on keyframes and then on detailed frames) and long-document understanding (examining summaries first and then details in key paragraphs).

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of using attention maps for dynamic token selection is simple and elegant, and the hierarchical attention design is sound. However, the overall concept is not a disruptive innovation in the field of efficient attention.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple benchmarks (high-resolution, general, domain-specific) and includes comprehensive ablation studies (selection strategy, resolution, object size), but lacks validation on larger models and a wider variety of architectures.
- Writing Quality: ⭐⭐⭐⭐ The paper is well-structured with a clear motivation and intuitive illustrations, though the description of the comparative experimental setup with CogAgent in the related work section could be clearer.
- Value: ⭐⭐⭐⭐ Proposes an effective efficiency-performance trade-off for the practical bottleneck of high-resolution VLMs; the 40% reduction in computation along with performance improvements holds solid engineering value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ERGO: Efficient High-Resolution Visual Understanding for Vision-Language Models](../../ICLR2026/multimodal_vlm/ergo_efficient_high-resolution_visual_understanding_for_vision-language_models.md)
- [\[ECCV 2024\] Efficient Inference of Vision Instruction-Following Models with Elastic Cache](efficient_inference_of_vision_instruction-following_models_with_elastic_cache.md)
- [\[CVPR 2026\] SenseSearch: Empowering Vision-Language Models with High-Resolution Agentic Search-Reasoning via Reinforcement Learning](../../CVPR2026/multimodal_vlm/sensesearch_empowering_vision-language_models_with_high-resolution_agentic_searc.md)
- [\[ECCV 2024\] Vary: Scaling up the Vision Vocabulary for Large Vision-Language Models](vary_scaling_up_the_vision_vocabulary_for_large_visionlanguag.md)
- [\[ECCV 2024\] BRAVE: Broadening the Visual Encoding of Vision-Language Models](brave_broadening_the_visual_encoding_of_vision-language_models.md)

</div>

<!-- RELATED:END -->
