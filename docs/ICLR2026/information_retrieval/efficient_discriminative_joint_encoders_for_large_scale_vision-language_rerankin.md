---
title: >-
  [Paper Note] Efficient Discriminative Joint Encoders for Large Scale Vision-Language Re-ranking
description: >-
  [ICLR 2026][Information Retrieval & RAG][Vision-Language Retrieval] This paper proposes EDJE (Efficient Discriminative Joint Encoder), which offlines visual feature extraction and compresses visual tokens via a lightweig…
tags:
  - "ICLR 2026"
  - "Information Retrieval & RAG"
  - "Vision-Language Retrieval"
  - "Joint Encoder"
  - "Re-ranking"
  - "Token Compression"
  - "Efficient Inference"
date: 2026-05-08
content_hash: a1e66dd34905af0d
---

# Efficient Discriminative Joint Encoders for Large Scale Vision-Language Re-ranking

**Conference**: ICLR 2026
**arXiv**: [2510.06820](https://arxiv.org/abs/2510.06820)
**Code**: [GitHub](https://github.com/gitanony04-lab/Simple-Efficient-Fusion)
**Area**: Information Retrieval
**Keywords**: Vision-Language Retrieval, Joint Encoder, Re-ranking, Token Compression, Efficient Inference

## TL;DR
This paper proposes EDJE (Efficient Discriminative Joint Encoder), which offlines visual feature extraction and compresses visual tokens via a lightweight attention adapter, achieving a throughput of 50k image-text pairs per second. EDJE matches the retrieval performance of existing joint encoders on Flickr (zero-shot) and COCO (fine-tuned), requiring only 49 kB of storage per image.

## Background & Motivation
In large-scale multimodal retrieval, embedding-based models (e.g., CLIP) enable efficient search via vector similarity, but independently encoding the two modalities limits fine-grained cross-modal interaction. Joint encoders (e.g., BLIP, BLIP-2) process both modalities jointly and achieve stronger retrieval performance. Cross-encoder re-ranking is already a standard paradigm in text retrieval.

**Key Challenge**: Visual feature extraction is a severe bottleneck in existing joint encoders—processing a batch of 64 images with BLIP using ViT-B takes ~400 ms and ~1,400 ms with ViT-L, accounting for 83%–93% of total inference time. By contrast, the widely used MiniLM re-ranker in text retrieval has only 22 M parameters and processes an equivalent batch in ~60 ms. This explains why multimodal re-rankers are nearly absent from practical retrieval systems.

**Core Idea**: Offline visual feature extraction—images are encoded once and cached to disk; at inference time, only a compact joint encoder processes a small number of visual tokens together with text tokens. A token compression adapter further reduces storage requirements substantially.

## Method

### Overall Architecture
EDJE operates in two stages—offline and online:
- **Offline stage**: A ViT encodes images; an adapter compresses the resulting features into a compact token set, which is stored to disk.
- **Online stage**: A compact language model (MiniLM) jointly processes the compressed visual tokens and text tokens to produce re-ranking scores.

### Key Designs
1. **Paradigm shift via visual pre-computation**: Because the visual encoder operates solely on images, its outputs can be cached and reused. A ViT-B projects each 16×16 patch into an embedding of dimension $d=384$; stored in FP16, the footprint is comparable to that of the original 8-bit RGB image. Scaling up the visual encoder improves representation quality without increasing online cost. The key challenge is that raw storage is infeasible at web-scale (potentially reaching terabytes), necessitating a compression strategy.

2. **Token compression adapter**: $m$ learnable universal query tokens $\mathbf{Q} = [\mathbf{q}_1, \ldots, \mathbf{q}_m]$ aggregate information from $n$ visual tokens $\mathbf{X}$ via cross-attention:
$$\mathbf{H} = \text{MultiHeadAttention}(\mathbf{Q},\, \mathbf{X}\mathbf{W}_K,\, \mathbf{X}\mathbf{W}_V)$$
The output is passed through a residual block and linearly projected into the language model embedding space:
$$\mathbf{Y} = \bigl(\mathbf{H} + \text{MLP}(\text{LayerNorm}(\mathbf{H}))\bigr)\mathbf{W}_{proj}$$
This compresses 576 ViT tokens down to 64 tokens, reducing storage from 442 kB to 49 kB per image.

3. **Compact joint encoder**: Following the VLM paradigm, visual tokens are projected into the language embedding space, concatenated with text tokens, and processed jointly via self-attention for cross-modal interaction. The language model is MiniLM (33 M parameters), substantially smaller than BLIP's 139–167 M. This design is modular: any ViT-based visual encoder can be paired with any pre-trained language model.

### Loss & Training
Three objectives are jointly optimized:
- **Image-Text Matching (ITM)**: Binary classification of positive pairs versus in-batch hard negatives.
- **Masked Language Modeling (MLM)**: 50% of text tokens are masked and predicted, encouraging cross-modal dependency.
- **Text Embedding Recovery**: Minimization of the cosine distance between the projected CLS token and the text encoder embedding.
- **Local-to-Compressed Distillation**: The uncompressed Local model serves as a teacher; logit-level distillation is applied to the compressed model.

Pre-training uses 14 M image-text pairs from CC12M, CC3M, SBU, VG, and COCO; fine-tuning uses COCO only.

## Key Experimental Results

### Main Results

**Flickr30k Zero-Shot Retrieval (SigLIP2 ViT-L/16, 384²):**

| Method | T2I R@1 | I2T R@1 | Storage/Image | Params | Inference Time |
|--------|---------|---------|---------------|--------|----------------|
| BLIP ViT-L/16 | 86.7 | 96.7 | 2,359 kB | 139 M | 101.61 ms |
| BLIP-2 ViT-L/16 | 88.6 | 96.9 | 2,359 kB | 167 M | 98.64 ms |
| EDJE Local | 87.8 | 96.5 | 442 kB | 33 M | 4.14 ms |
| EDJE Compressed-64 | 86.9 | 96.4 | 49 kB | 33 M | 1.91 ms |

**EDJE Gains over Various Embedding Models (Flickr30k Zero-Shot T2I R@1):**

| Backbone | Base | +EDJE | Gain |
|----------|------|-------|------|
| CLIP ViT-B/16 | 62.1 | 76.8 | +14.7 |
| CLIP ViT-L/14 | 65.2 | 80.6 | +15.4 |
| SigLIP2 ViT-B/16 | 82.1 | 84.3 | +2.2 |
| SigLIP2 ViT-L/16 | 82.3 | 87.8 | +5.5 |

### Ablation Study
- **Token count**: Experiments with {32, 64, 128, 256} target tokens show that 64 tokens achieve the best efficiency–performance trade-off; 32 tokens exhibit a notable performance drop, while 256 tokens approach the performance of the Local variant with 576 tokens.
- **Re-ranking pool size**: Retrieval performance on R@1/5/10 remains stable as $k$ varies from 5 to 50, demonstrating EDJE's robustness to noisy candidates.
- **Training objectives**: Incrementally adding MLM and text embedding recovery on top of ITM each yield positive contributions; the full combination is strongest.
- **Local-to-Compressed distillation**: Provides additional discriminative gains for the compressed variant.
- **Semantic analysis**: The 64 compressed tokens map to semantically meaningful object/scene descriptors (e.g., *boulders*, *caves*), whereas a large proportion of the 576 uncompressed tokens map to meaningless special tokens (e.g., *unused80*), confirming substantial redundancy in raw ViT tokens.
- **Quantization**: Quantizing the compressed tokens incurs negligible performance loss, offering a further storage–performance trade-off.

### Key Findings
- EDJE serves as a plug-and-play re-ranker that improves retrieval across all tested embedding models (CLIP, DFN, MetaCLIP, SigLIP2).
- Inference is 53× faster than BLIP-2, with 48× storage reduction (49 kB vs. 2,359 kB per image).
- Quantizing compressed tokens causes minimal performance degradation, enabling further storage optimization.

## Highlights & Insights
- The paper precisely identifies visual feature extraction as the joint-encoder bottleneck (83%–93% of inference time); the proposed offline-plus-compression solution is elegant and principled.
- The semantic analysis of the token compression adapter is highly informative: most ViT tokens are indeed redundant, and 64 compressed tokens suffice to capture critical semantics.
- The modular architecture makes EDJE a highly practical drop-in re-ranker.
- The paper's narrative is exceptionally clear, progressing logically from bottleneck analysis → paradigm shift → concrete design → experimental validation.
- This work is the first to systematically transfer the mature cross-encoder re-ranking paradigm from text retrieval to multimodal retrieval, filling an important gap.

## Limitations & Future Work
- Coverage is limited to image-text retrieval; multilingual multimodal retrieval, audio, and video modalities are not addressed.
- The discriminative capacity of the joint encoder leaves room for improvement; replacing MiniLM with a larger or more capable language model is worth exploring.
- Performance degrades noticeably at 32 tokens; more aggressive compression methods deserve investigation.
- Gains over DFN and MetaCLIP are smaller than those over CLIP and SigLIP2 (DFN was fine-tuned as a filtering network on Flickr).
- Downstream applications where joint encoders excel—such as zero-shot classification and large-scale data filtering—are not explored.

## Related Work & Insights
- **Relation to BLIP family**: EDJE can be viewed as an efficient replacement for BLIP's re-ranking capability; the core contribution is shifting visual feature extraction from online to offline.
- **Connection to ColBERT**: The approach shares conceptual similarity with ColBERT's token-level offline storage in text retrieval, but adds a compression dimension.
- **Relation to Q-Former**: The token compression layer resembles BLIP-2's Q-Former but is lighter and focuses on compression rather than generation.
- **Comparison with LightningDOT**: LightningDOT performs re-ranking with region features but compresses each region to a single vector, making it closer in nature to an embedding model than a true joint encoder.

## Rating
- Novelty: ⭐⭐⭐⭐ The core idea (offline visual features + compressed tokens) is intuitively clear, though each component has precedents.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Validation spans multiple backbones, with detailed ablations, semantic visualization, and comprehensive efficiency analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem formulation is precise, motivation is exceptionally clear, and bottleneck analysis is data-driven.
- Value: ⭐⭐⭐⭐⭐ High practical value; fills the gap left by the absence of joint-encoder re-rankers in multimodal retrieval.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] TokMem: One-Token Procedural Memory for Large Language Models](tokmem_one-token_procedural_memory_for_large_language_models.md)
- [\[ACL 2026\] Region-R1: Reinforcing Query-Side Region Cropping for Multi-Modal Re-Ranking](../../ACL2026/information_retrieval/region-r1_reinforcing_query-side_region_cropping_for_multi-modal_re-ranking.md)
- [\[ICLR 2026\] Query-Level Uncertainty in Large Language Models](query-level_uncertainty_in_large_language_models.md)
- [\[ICLR 2026\] RAEE: A Robust Retrieval-Augmented Early Exit Framework for Efficient Inference](raee_a_robust_retrieval-augmented_early_exit_framework_for_efficient_inference.md)
- [\[ICLR 2026\] Beyond RAG vs. Long-Context: Learning Distraction-Aware Retrieval for Efficient Knowledge Grounding](beyond_rag_vs_long-context_learning_distraction-aware_retrieval_for_efficient_kn.md)

</div>

<!-- RELATED:END -->
