---
title: >-
  [Paper Note] InfoCLIP: Bridging Vision-Language Pretraining and Open-Vocabulary Semantic Segmentation via Information-Theoretic Alignment Transfer
description: >-
  [AAAI 2026][Segmentation][Open-vocabulary semantic segmentation] This paper proposes InfoCLIP, which adopts an information-theoretic perspective to design two objectives—information bottleneck compression and mutual information distillation—to remove noise in pretrained pixel-text alignment and preserve semantic alignment knowledge during CLIP fine-tuning. InfoCLIP achieves state-of-the-art results across six open-vocabulary semantic segmentation benchmarks (A-847: 16.6, A-150: 38.5, PC-59: 63.5 mIoU) while introducing only 0.53M additional parameters and negligible computational overhead.
tags:
  - AAAI 2026
  - Segmentation
  - Open-vocabulary semantic segmentation
  - CLIP fine-tuning
  - information bottleneck
  - mutual information distillation
  - modality alignment
date: 2026-05-08
content_hash: 2fa8d2856400acb0
---

# InfoCLIP: Bridging Vision-Language Pretraining and Open-Vocabulary Semantic Segmentation via Information-Theoretic Alignment Transfer

**Conference**: AAAI 2026
**arXiv**: [2511.15967](https://arxiv.org/abs/2511.15967)
**Code**: [https://muyaoyuan.github.io/InfoCLIP-Page](https://muyaoyuan.github.io/InfoCLIP-Page)
**Area**: Semantic Segmentation / Multimodal VLM
**Keywords**: Open-vocabulary semantic segmentation, CLIP fine-tuning, information bottleneck, mutual information distillation, modality alignment

## TL;DR
This paper proposes InfoCLIP, which adopts an information-theoretic perspective to design two objectives—information bottleneck compression and mutual information distillation—to remove noise in pretrained pixel-text alignment and preserve semantic alignment knowledge during CLIP fine-tuning. InfoCLIP achieves state-of-the-art results across six open-vocabulary semantic segmentation benchmarks (A-847: 16.6, A-150: 38.5, PC-59: 63.5 mIoU) while introducing only 0.53M additional parameters and negligible computational overhead.

## Background & Motivation
CLIP is widely used in open-vocabulary semantic segmentation (OVSS); however, fine-tuning CLIP introduces a fundamental conflict: CLIP pretraining learns global image-text alignment, whereas segmentation requires local pixel-text alignment. Fine-tuning on limited category data narrows the modality alignment space (overfitting to seen classes), and even freezing most parameters is fragile—modifying a small subset suffices to disrupt feature distributions. Existing distillation methods (e.g., the MAFT family) distill visual features only at the image level, leaving the pixel-text alignment problem unaddressed and even degrading performance.

## Core Problem
How can fine-grained pixel-text alignment knowledge be extracted from a pretrained CLIP, noise removed from coarse-grained representations, and the cleaned knowledge effectively transferred to a fine-tuned model without overfitting to seen categories? This is a two-stage challenge: (1) extracting useful pixel-level alignment relationships from noise; and (2) transferring them to the downstream model while preserving modality alignment.

## Method

### Overall Architecture
InfoCLIP builds upon the CAT-Seg framework, using a frozen pretrained CLIP as the teacher and a fine-tuned CLIP as the student. The core consists of three components: the LPAM module extracts pixel-text alignment relationships → information bottleneck compression denoises the representations → mutual information maximization distillation transfers the knowledge. The total loss is: task cross-entropy loss + $\lambda_1$ compression loss + $\lambda_2$ distillation loss.

### Key Designs
1. **Learnable Pixel-Text Alignment Module (LPAM)**: Takes dense embeddings from the CLIP image encoder ($D_V$) and text encoder embeddings ($D_L$), and produces a semantic alignment map $R \in \mathbb{R}^{(H \times W) \times N_C}$ via learned attention. Scaled dot-product attention is combined with residual cosine similarity. Teacher and student share LPAM parameters and produce $R^T$ and $R^S$, respectively. LPAM contains only 0.53M parameters and is extremely lightweight.

2. **Semantic Compression via Information Bottleneck (compression loss $\mathcal{L}_c$)**: Minimizes the mutual information between the pretrained CLIP input embeddings $(D_V^T, D_L^T)$ and the alignment map $R^T$. Matrix-based Rényi $\alpha$-entropy is adopted; when $\alpha=2$, it can be approximated via the Frobenius norm, avoiding eigenvalue decomposition and reducing computational complexity from $O(n^3)$ to $O(n^2)$. Intuitively, the first term compresses redundant signals in the alignment features, while the second term preserves joint semantic information among image, text, and alignment, forming an information bottleneck that filters out noise.

3. **Alignment Transfer via Mutual Information (distillation loss $\mathcal{L}_d$)**: Maximizes the Rényi mutual information between the teacher alignment map $R^T$ and the student alignment map $R^S$. Compared to KL-divergence distillation, mutual information preserves structural information without requiring density estimation, providing a stable and differentiable optimization objective. The first two terms serve as regularization, and the third term enforces relational alignment consistency between teacher and student.

### Loss & Training
$\mathcal{L} = \mathcal{L}_{task} + \lambda_1 \mathcal{L}_c + \lambda_2 \mathcal{L}_d$, with $\lambda_1 = \lambda_2 = 1$ (determined via hyperparameter sensitivity analysis). AdamW optimizer; decoder and distillation module learning rate $= 2 \times 10^{-4}$; CLIP backbone learning rate $= 2 \times 10^{-6}$; batch size $= 4$; 80k iterations; trained on a single A800 GPU.

## Key Experimental Results

| Dataset | Metric (mIoU) | InfoCLIP (ViT-B) | CAT-Seg (ViT-B) | InfoCLIP (ViT-L) | CAT-Seg (ViT-L) |
|---------|--------------|------------------|-----------------|------------------|-----------------|
| A-847 | mIoU | **12.6** | 12.0 | **16.6** | 16.0 |
| PC-459 | mIoU | **19.5** | 19.0 | **24.6** | 23.8 |
| A-150 | mIoU | **32.1** | 31.8 | **38.5** | 37.9 |
| PC-59 | mIoU | **58.1** | 57.5 | **63.5** | 63.3 |
| PAS-20 | mIoU | **95.5** | 94.6 | **97.5** | 97.0 |
| PAS-20b | mIoU | **78.1** | 77.3 | **83.1** | 82.5 |

InfoCLIP outperforms MAFT+ (ViT-L) by 8.4% on PC-459 and 4.5% on PC-59. Training overhead is minimal: only +0.53M parameters, forward +0.08s, backward +0.03s.

### Ablation Study
- $\mathcal{L}_d$ alone: A-847 11.3 (moderate); $\mathcal{L}_c$ alone: A-847 11.8 (slightly better); combining both yields 12.6 (+0.6), demonstrating clear complementarity.
- Conventional distillation methods (KL divergence, MAFT, MAFT+) even degrade CAT-Seg performance, confirming that direct feature distribution matching fails for OVSS transfer.
- $\alpha=2$ achieves the best results across all benchmarks with a 56× speedup (0.5ms vs. 28.2ms per iteration).

## Highlights & Insights
- **The information-theoretic perspective is elegant**—information bottleneck denoising combined with MI maximization distillation offers theoretical guarantees and empirical effectiveness.
- **The two-stage "compress-then-distill" design is intuitive**: first remove noise from the pretrained CLIP's local representations, then transfer the clean alignment knowledge to the fine-tuned model.
- The Frobenius norm approximation at $\alpha=2$ is a highly practical trick—reducing $O(n^3)$ to $O(n^2)$ while improving performance.
- The method is orthogonal to model architecture (requiring only a lightweight LPAM) and can be grafted onto other OVSS methods.
- t-SNE visualizations clearly demonstrate feature disentanglement between seen and unseen classes.

## Limitations & Future Work
- Validation is limited to the CAT-Seg architecture; extension to other frameworks (e.g., SAN, FC-CLIP) has not been explored.
- LPAM uses fixed shared parameters; the possibility of using separate LPAM modules for teacher and student has not been investigated.
- The degree of information bottleneck compression is controlled by $\lambda_1$, but the optimal compression level may vary across different scenarios.
- Integration with larger VLMs (e.g., InternVL, LLaVA) remains unexplored.

## Related Work & Insights
- **vs. CAT-Seg**: InfoCLIP augments CAT-Seg with information-theoretic distillation, consistently outperforming it with minimal overhead (+0.53M parameters).
- **vs. MAFT/MAFT+**: The MAFT family distills visual features at the image level and is unsuited for pixel-based methods, even degrading performance. InfoCLIP performs distillation at the pixel-text alignment level, yielding significantly superior results.
- **vs. KL distillation**: Standard KL distillation performs worst in OVSS (A-847 drops from 12.0 to 5.7), demonstrating that direct distribution matching fails under heterogeneous task transfer.
- The information bottleneck paradigm is transferable to other VLM downstream tasks (e.g., grounding, referring segmentation).
- The "denoise-then-distill" two-stage paradigm also has implications for token compression—MI could potentially serve as a measure for identifying noisy tokens.
- The efficient computation of matrix-based Rényi entropy is broadly applicable to other scenarios requiring mutual information estimation.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First to introduce an information-theoretic framework into CLIP fine-tuning for OVSS; both theory and methodology are highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Six benchmarks, two backbone variants, comprehensive ablations, efficiency analysis, and visualizations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Theoretical derivations are clear, motivation figures are intuitive, and the argumentation is logically rigorous.
- **Value**: ⭐⭐⭐⭐ Highly valuable within the OVSS domain, though the scope of applicability is relatively narrow.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Target Refocusing via Attention Redistribution for Open-Vocabulary Semantic Segmentation: An Explainability Perspective](target_refocusing_via_attention_redistribution_for_open-vocabulary_semantic_segm.md)
- [\[AAAI 2026\] Bridging Granularity Gaps: Hierarchical Semantic Learning for Cross-Domain Few-Shot Segmentation](bridging_granularity_gaps_hierarchical_semantic_learning_for_cross-domain_few-sh.md)
- [\[CVPR 2026\] PEARL: Geometry Aligns Semantics for Training-Free Open-Vocabulary Semantic Segmentation](../../CVPR2026/segmentation/pearl_geometry_aligns_semantics_for_training-free_open-vocabulary_semantic_segme.md)
- [\[CVPR 2026\] Direct Segmentation without Logits Optimization for Training-Free Open-Vocabulary Semantic Segmentation](../../CVPR2026/segmentation/direct_segmentation_without_logits_optimization_for_training-free_open-vocabular.md)
- [\[NeurIPS 2025\] LangHOPS: Language Grounded Hierarchical Open-Vocabulary Part Segmentation](../../NeurIPS2025/segmentation/langhops_language_grounded_hierarchical_open-vocabulary_part_segmentation.md)

</div>

<!-- RELATED:END -->
