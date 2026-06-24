---
title: >-
  [Paper Note] LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models
description: >-
  [ICCV 2025][Multimodal Efficiency][visual token compression] By exploiting the sparsity of attention scores between the CLS token and spatial tokens in the visual encoder, this work adaptively prunes and merges visual tokens, maintaining comparable LMM performance while retaining only 5.5% of visual tokens.
tags:
  - "ICCV 2025"
  - "Multimodal Efficiency"
  - "visual token compression"
  - "large multimodal models"
  - "token pruning and merging"
  - "attention sparsity"
  - "efficient inference"
date: 2026-05-08
content_hash: b9c7dd98102b0d40
---

# LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models

**Conference**: ICCV 2025
**arXiv**: [2403.15388](https://arxiv.org/abs/2403.15388)  
**Code**: [https://github.com/yuzhangshang/LLaVA-PruMerge](https://github.com/yuzhangshang/LLaVA-PruMerge)  
**Area**: Multimodal VLM
**Keywords**: visual token compression, large multimodal models, token pruning and merging, attention sparsity, efficient inference

## TL;DR

By exploiting the sparsity of attention scores between the CLS token and spatial tokens in the visual encoder, this work adaptively prunes and merges visual tokens, maintaining comparable LMM performance while retaining only 5.5% of visual tokens.

## Background & Motivation

Large multimodal models (LMMs) achieve visual reasoning by connecting a visual encoder (e.g., CLIP-ViT) with a large language model. However, these models face a severe efficiency bottleneck: the computational complexity of the LLM backbone scales **quadratically** with the number of input tokens. In LLaVA-1.5, for instance, a single image requires 576 visual tokens as prefix input to the LLM; Video-LLaVA requires as many as 2048 visual tokens. As demand for high-resolution image and video understanding grows, the number of visual tokens continues to expand.

Existing approaches to improving LMM efficiency broadly follow two directions: substituting a smaller LLM backbone (e.g., Phi-2 in place of LLaMA-7B), which sacrifices reasoning capability, or applying quantization compression, which still fails to address the fundamental problem of excessively long input sequences. **The root cause** is that LMMs require a large number of visual tokens to fully represent visual content, yet an abundance of tokens introduces prohibitive computational overhead.

The authors' key observation is that **significant spatial redundancy** exists in the visual encoder. Specifically, in the final layer of the ViT, attention scores between the CLS token and the vast majority of spatial tokens are nearly zero (exhibiting a highly sparse distribution), indicating that only a small fraction of visual tokens carry critical visual information. This naturally motivates the question: can this sparsity be exploited to adaptively select important visual tokens and substantially reduce the LLM's input sequence length?

## Method

### Overall Architecture

PruMerge is a plug-and-play visual token compression module inserted between the visual encoder and the LLM. The pipeline consists of three steps: (1) important tokens are selected from the CLS-spatial attention scores using an outlier detection algorithm; (2) the remaining tokens are clustered via K-nearest neighbors based on Key vector similarity; and (3) the information of pruned tokens is aggregated back into the retained anchor tokens through a weighted average.

### Key Designs

#### 1. Adaptive Important Token Selection (AITS) — Outlier Detection-Based

- **Function**: Adaptively determines how many visual tokens to retain based on the content complexity of each image.
- **Mechanism**: Leverages CLS-spatial attention scores computed in the second-to-last ViT layer as $\mathbf{a}_{\text{cls}} = \text{softmax}\left(\frac{\mathbf{q}_{\text{cls}} \cdot \mathbf{K}^T}{\sqrt{d_k}}\right)$. These scores exhibit a highly sparse distribution — the attention values of the vast majority of tokens are close to zero, while only a small subset attains high values. An interquartile range (IQR)-based outlier detection method is applied: the upper fence = Q3 + 1.5 × IQR; tokens exceeding this threshold are treated as "outliers" (i.e., important tokens) and retained.
- **Design Motivation**: Simple images (e.g., a billboard against a blue sky) can be adequately represented with few tokens, whereas complex images containing dense text require more. The IQR method requires no manual threshold setting and automatically adjusts the number of retained tokens according to image complexity. Experiments show that the average token count differs substantially across benchmarks (e.g., ScienceQA requires only 16 tokens on average, while TextVQA requires 40), validating the necessity of adaptivity.

#### 2. Token Supplement (TS) — Similarity-Based Key Clustering

- **Function**: Merges information from pruned tokens into the retained anchor tokens to prevent information loss.
- **Mechanism**: Key vectors from the final ViT layer are used to compute inter-token similarity $\text{Sim}(\mathbf{y}_i, \mathbf{y}_j) = \mathbf{k}_i \cdot \mathbf{k}_j^T$. For each unpruned token, the K nearest neighbors among the pruned tokens are identified, and the anchor token's representation is updated via a weighted average using the CLS attention value $\mathbf{a}[i]$ as the weight.
- **Design Motivation**: When large objects occupy much of an image (e.g., a building in a panoramic shot), aggressive pruning may discard important spatial information. Merging rather than simply discarding ensures that the information of pruned tokens is preserved. Key vectors are chosen as the similarity metric because they already aggregate positional and semantic information during self-attention. The time complexity is $O(n)$, more efficient than CrossGet's $O(n^2)$.

#### 3. PruMerge+ — Spatially Uniform Sampling Augmentation

- **Function**: Supplements PruMerge by additionally sampling tokens from "unimportant" regions according to a spatially uniform distribution.
- **Mechanism**: Using the number of outlier tokens as a reference ratio, tokens are sampled at uniform spatial intervals from non-outlier regions, ensuring that overlooked areas also have representative tokens. The final token count is approximately 25% of the original (roughly 144 out of 576), far fewer than the original total, while substantially reducing performance degradation.
- **Design Motivation**: Pure outlier detection may overlook regions that, while not receiving high attention scores, are nonetheless useful for understanding spatial layout. Spatially uniform sampling compensates for this limitation, achieving a better balance between token compression rate and performance.

### Loss & Training

- PruMerge can be applied directly in a **training-free** manner, or further adapted via **LoRA fine-tuning**.
- Fine-tuning uses the original LLaVA-1.5 instruction-tuning data for a single epoch.
- Fine-tuning enables the LLM to adapt to the compressed visual token structure and yields further performance gains on most benchmarks.

## Key Experimental Results

### Main Results

| Method | LLM | VQAv2 | SQA-I | TextVQA | POPE | MME | MMB |
|--------|-----|-------|-------|---------|------|-----|-----|
| LLaVA-1.5 | Vicuna-7B | 78.5 | 66.8 | 58.2 | 85.9 | 1510.7 | 64.3 |
| + PruMerge (5.5% tokens) | Vicuna-7B | 72.0 | **68.5** | 56.0 | 76.3 | 1350.3 | 60.9 |
| + PruMerge+ (25% tokens) | Vicuna-7B | 76.8 | **68.3** | 57.1 | 84.0 | 1462.4 | 64.9 |
| LLaVA-1.5 | Vicuna-13B | 80.0 | 71.6 | 61.3 | 85.9 | 1531.3 | 67.7 |
| + PruMerge+ (25% tokens) | Vicuna-13B | 77.8 | 71.0 | 58.6 | 84.4 | 1485.5 | 65.7 |

PruMerge+ even surpasses the original LLaVA-1.5 on ScienceQA, suggesting that removing redundant tokens actually helps the model focus on critical information.

### Ablation Study

| Configuration | TextVQA | MME | POPE | Note |
|---------------|---------|-----|------|------|
| PruMerge (AITS only) | 54.8 | 1221.6 | 75.7 | Pruning only, no merging |
| PruMerge (AITS + TS) | 56.0 | 1350.3 | 76.3 | Adding token merging significantly recovers performance |
| Sequential Sampling (40 tokens) | 42.7 | 703.6 | 11.7 | Random sequential sampling, performance collapses |
| Spatial 5×8 (40 tokens) | 46.9 | 1180.2 | 69.8 | Uniform spatial sampling |
| PruMerge (40 tokens) | **54.0** | **1250.1** | **76.2** | Adaptive selection clearly outperforms fixed strategies |

Efficiency analysis: PruMerge reduces LLM Prefill FLOPs from 9.3 TB to 0.91 TB (a **10×** reduction), Prefill latency from 88.6 ms to 15.3 ms, and activation memory from 4.60 GB to 0.28 GB.

### Key Findings

- Compared with unimodal token compression methods (ToMe, EViT, ATS) applied to LMMs, PruMerge+ achieves a VQAv2 score of 76.8 at a 25% compression rate, substantially outperforming ToMe (66.0), ATS (66.7), and EViT (65.5).
- PruMerge can be applied to Video-LLaVA **without any training** and actually improves performance, indicating that video LLMs suffer from even more severe token redundancy.
- Attention sparsity is a **universal phenomenon across models** and is not limited to specific ViT architectures.

## Highlights & Insights

- **Sparsity as a signal**: The sparse distribution of CLS attention is not a deficiency but a natural indicator of token importance. Applying statistical outlier detection to visual token selection is a concise and effective innovation.
- **Plug-and-play**: PruMerge requires no modification to the visual encoder or the internal structure of the LLM; it simply inserts a lightweight module between the two.
- **Distinguishing LMM from ViT token compression**: The authors explicitly argue that the efficiency bottleneck in LMMs lies in the LLM rather than the ViT, so the goal of token compression should be to reduce the LLM's input length rather than the ViT's internal computation — a fundamentally different design philosophy from unimodal methods.

## Limitations & Future Work

- Non-trivial performance degradation remains on tasks requiring global spatial understanding, such as VQAv2 and POPE (PruMerge drops from 85.9 to 76.3 on POPE).
- The current IQR threshold is a fixed strategy and is not dynamically adjusted for different tasks or datasets.
- Validation is limited to LLaVA and Video-LLaVA; experiments on more advanced LMMs (e.g., Qwen-VL, InternVL) are absent.
- Token selection is performed as a one-shot decision at the final layer, without considering progressive multi-level filtering.

## Related Work & Insights

- This work shares conceptual roots with ToMe (Token Merging), but ToMe focuses on layer-by-layer token merging within the ViT, incurring higher time complexity and being ill-suited for the LMM setting.
- The approach has strong practical value for high-resolution LMMs (e.g., LLaVA-Next, Monkey) that routinely process thousands of tokens.
- A promising direction for future work: can token selection be jointly optimized with the LLM's internal attention mechanism — for example, enabling the LLM to dynamically request more or fewer visual tokens during inference?

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CoIn: Coverage and Informativeness-Guided Token Reduction for Efficient Large Multimodal Models](../../CVPR2026/vlm_efficiency/coin_coverage_and_informativeness-guided_token_reduction_for_efficient_large_mul.md)
- [\[ICCV 2025\] ShortV: Efficient Multimodal Large Language Models by Freezing Visual Tokens in Ineffective Layers](shortv_efficient_multimodal_large_language_models_by_freezing_visual_tokens_in_i.md)
- [\[CVPR 2026\] Rethinking Token Reduction for Large Vision-Language Models](../../CVPR2026/vlm_efficiency/rethinking_token_reduction_for_large_vision-language_models.md)
- [\[ICCV 2025\] FOLDER: Accelerating Multi-modal Large Language Models with Enhanced Performance](folder_accelerating_multi-modal_large_language_models_with_enhanced_performance.md)
- [\[ACL 2025\] Token Pruning in Multimodal Large Language Models: Are We Solving the Right Problem?](../../ACL2025/vlm_efficiency/token_pruning_in_multimodal_large_language_models_are_we_solving_the_right_probl.md)

</div>

<!-- RELATED:END -->
