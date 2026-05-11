---
title: >-
  [Paper Note] NanoVDR: Distilling a 2B Vision-Language Retriever into a 70M Text-Only Encoder for Visual Document Retrieval
description: >-
  [CVPR 2026][Information Retrieval & RAG][Visual Document Retrieval] NanoVDR exploits the modality asymmetry between queries and documents, distilling the query encoding capability of a 2B VLM teacher into a 69M text-only…
tags:
  - "CVPR 2026"
  - "Information Retrieval & RAG"
  - "Visual Document Retrieval"
  - "Asymmetric Distillation"
  - "VLM Compression"
  - "Text-Only Query Encoding"
  - "Cross-Modal Transfer"
date: 2026-05-08
content_hash: 0f551faf34acfd3b
---

# NanoVDR: Distilling a 2B Vision-Language Retriever into a 70M Text-Only Encoder for Visual Document Retrieval

**Conference**: CVPR 2026
**arXiv**: [2603.12824](https://arxiv.org/abs/2603.12824)
**Code**: [HuggingFace Models](https://huggingface.co/nanovdr/NanoVDR-S-Multi)
**Area**: Information Retrieval
**Keywords**: Visual Document Retrieval, Asymmetric Distillation, VLM Compression, Text-Only Query Encoding, Cross-Modal Transfer

## TL;DR

NanoVDR exploits the modality asymmetry between queries and documents, distilling the query encoding capability of a 2B VLM teacher into a 69M text-only encoder via pointwise cosine alignment. On the ViDoRe benchmark, the student retains 95.1% of teacher performance while reducing query latency by 50× with a total training cost of only 13 GPU hours.

## Background & Motivation

Visual Document Retrieval (VDR) treats document pages directly as images, encoding both queries and documents into a shared embedding space using a VLM, thereby avoiding the information loss inherent in OCR pipelines. State-of-the-art systems (ColPali, DSE-Qwen2, Tomoro-8B) all employ the same billion-parameter VLM to encode both queries and document pages.

**Core Insight**: This symmetric design is unnecessary. Document pages contain complex visual information such as charts, formulas, and layouts, which genuinely requires strong visual understanding; however, queries are simply short text strings containing no visual information whatsoever. Deploying a 2B-parameter VLM to encode a text query wastes all visual processing capacity and leads to:
- **High query latency**: Single-query encoding on CPU takes 2.5–8.2 seconds
- **GPU dependency**: Online inference requires a GPU, precluding edge deployment
- **Large model footprint**: Query encoder checkpoints reach 8.8–35 GB

The central question of this paper: Can the query-is-pure-text asymmetry be exploited to distill a VLM's query encoding capability into a lightweight text-only model capable of real-time CPU inference?

## Method

### Overall Architecture

NanoVDR explicitly decouples the retrieval pipeline into two asymmetric paths:

1. **Offline document indexing (heavy)**: A frozen 2B VLM teacher (Qwen3-VL-Embedding-2B) encodes each document page image into a $d=2048$-dimensional single-vector embedding $\mathbf{v}_j^D = g(d_j) \in \mathbb{R}^d$; this step is performed offline on a GPU.
2. **Online query encoding (lightweight)**: The distilled text-only student model maps queries into the teacher's embedding space: $\mathbf{v}_s^Q = f_\theta(q) \in \mathbb{R}^d$.
3. **Retrieval**: Scoring via cosine similarity: $\text{score}(q, d_j) = {\mathbf{v}_s^Q}^\top \mathbf{v}_j^D$.

The student encoder architecture consists of a pretrained text backbone $h$ (e.g., DistilBERT) → mean pooling → two-layer MLP projector (768→768→2048, GELU activation) → L2 normalization. Three scale variants are provided:

| Variant | Backbone | Total Params | Projector Params |
|---------|----------|-------------|-----------------|
| NanoVDR-S | DistilBERT | 69M | 2M |
| NanoVDR-M | BERT-base | 112M | 2M |
| NanoVDR-L | ModernBERT-base | 151M | 2M |

Experiments show minimal performance differences across scales, indicating that query encoding per se does not require large model capacity.

### Key Designs: Query-Centric Distillation

Training proceeds in two stages:

**Stage 1 (Pre-caching)**: The frozen VLM teacher performs forward inference on all training queries in text-only mode, caching teacher query embeddings $\mathbf{v}_t^Q = g(q)$. This step processes only text and takes approximately 1 GPU hour.

**Stage 2 (Distillation training)**: The student text encoder is trained to align with the teacher query embeddings using an exceptionally simple loss:

$$\mathcal{L}_{\text{align}} = 1 - \frac{\mathbf{v}_s^Q \cdot \mathbf{v}_t^Q}{\|\mathbf{v}_s^Q\| \|\mathbf{v}_t^Q\|}$$

The underlying rationale: because the teacher maps both queries and documents into the **same** embedding space, once the student learns to align its query embeddings with the teacher's, it automatically acquires retrieval capability against teacher document embeddings—despite never observing any image.

Compared to conventional ranking distillation (KL divergence over score distributions), pointwise alignment offers two key advantages:
- **Superior performance**: Pure alignment outperforms pure ranking across all 9 evaluation points (3 backbones × 3 benchmarks).
- **Lower cost**: Training requires no document embeddings at all (ranking distillation requires an additional ~24 GPU hours to pre-cache document embeddings).

The paper systematically compares six distillation objectives, finding that NDCG@5 improves monotonically as the alignment weight increases. The authors' explanation is that a high-quality teacher embedding space encodes richer geometric structure (coordinate positions) than relative rankings. Supporting evidence: teacher quality is the strongest predictor of distillation success ($r=+0.607$), whereas student–teacher cosine similarity is nearly uncorrelated ($r=+0.094$).

### Multilingual Query Augmentation

Per-language analysis reveals that the **primary bottleneck is cross-lingual transfer rather than cross-modal transfer**: English query retention is 94.3% (training data share: 68.7%), while Portuguese (entirely absent from the training set) achieves only 75.6%. On the multilingual subset of ViDoRe v3, the retention gap between English and Portuguese queries over identical document corpora reaches 17.4 percentage points.

The solution is lightweight: Helsinki-NLP Opus-MT is used to translate ~489K English queries into five target languages (Portuguese, Spanish, German, French, Italian), with each language balanced to ~200K instances; the frozen teacher then encodes the translated queries to produce new target embeddings. Augmentation expands the dataset from 711K to 1.49M pairs, all involving only text—no images are required.

### Loss & Training

- OneCycleLR schedule, peak lr=2e-4, 3% warmup
- Batch size 256 with gradient accumulation over 4 steps (effective batch size 1024)
- 20 epochs (~13.9K steps), trained on a single H200 GPU for 10–12 hours
- Total training cost including teacher pre-caching: < 13 GPU hours

## Key Experimental Results

### Main Results: Retrieval Performance Comparison (NDCG@5 × 100)

| Model | Type | Params | v1 (10) | v2 (4) | v3 (8) | CPU Latency |
|-------|------|--------|---------|--------|--------|-------------|
| Tomoro-8B | Multi-vec VLM | 8.0B | 90.6 | 65.0 | 59.0 | 8,225 ms |
| ColPali | Multi-vec VLM | 3.0B | 84.2 | 54.7 | 42.0 | 7,284 ms |
| Teacher (Qwen3-VL-2B) | Single-vec VLM | 2.0B | 84.3 | 65.3 | 50.0 | — |
| DSE-Qwen2 | Single-vec VLM | 2.0B | 85.1 | 55.7 | 41.3 | 2,539 ms |
| NanoVDR-L | Text-only student | 151M | 82.4 | 61.5 | 44.2 | 109 ms |
| NanoVDR-M | Text-only student | 112M | 82.1 | 62.2 | 44.7 | 101 ms |
| NanoVDR-S | Text-only student | 69M | 82.2 | 60.5 | 43.5 | 51 ms |
| **NanoVDR-S-Multi** | **Text-only student** | **69M** | **82.2** | **61.9** | **46.5** | **51 ms** |

NanoVDR-S-Multi (69M) surpasses DSE-Qwen2 (2B) and ColPali (3B) on v2/v3 while using 32× fewer parameters and achieving 50× lower latency.

### Ablation Study: Distillation Loss Comparison (NDCG@5 × 100, averaged over 3 backbones)

| Loss Configuration ($\lambda_a$, $\lambda_r$) | v1 | v2 | v3 |
|------------------------------------------------|-----|------|------|
| Pure Align (1, 0) | **82.2** | **61.4** | **44.1** |
| Align + 0.5 Rank (1, 0.5) | 81.6 | 59.8 | 42.8 |
| Equal (1, 1) | 81.5 | 59.1 | 42.5 |
| 0.5 Align + Rank (0.5, 1) | 81.5 | 58.6 | 42.1 |
| Pure Rank (0, 1) | 81.1 | 57.4 | 41.6 |
| InfoNCE (hard labels) | 71.5 | 39.8 | 30.0 |

**Key Findings**:
1. **Alignment monotonically outperforms ranking**: As the alignment weight increases from 0 to 1, performance improves monotonically across all three benchmarks; pure alignment outperforms pure ranking by +1.1/+4.0/+2.5.
2. **Soft labels are critical**: InfoNCE (hard labels) causes a catastrophic performance drop (−10.7/−21.6/−14.1), demonstrating that the teacher's "dark knowledge"—the continuous geometric relationships in embedding space—is essential for cross-modal transfer.
3. **High data efficiency**: 25% of training data achieves 93% retention on v1; even 10% yields 79%.
4. **Multilingual augmentation**: Portuguese sees the largest gain (+9.3 NDCG), with zero degradation on English; after augmentation, retention exceeds 92% across all languages.

## Highlights & Insights

- **Extreme methodological simplicity**: The entire approach can be summarized in one sentence—run the frozen teacher forward once over text to cache query embeddings, then train a small model to perform cosine alignment. No negative samples, no images, and no complex distillation strategy are required.
- **Systematic exploitation of asymmetry**: The observation that queries are pure text while documents carry visual complexity is translated directly into an architectural design decision. Many existing systems implement symmetric encoders without ever questioning this assumption.
- **Alignment > Ranking finding**: This challenges the conventional wisdom that ranking losses (KL/MarginMSE) are optimal for retrieval distillation, demonstrating that direct coordinate alignment is more effective within a high-quality teacher embedding space.
- **Cross-lingual vs. cross-modal bottleneck analysis**: Through carefully designed controlled experiments (identical corpora, varying query languages), the paper establishes that the bottleneck lies in language rather than modality—a finding with broad implications for multimodal compression research.
- **13 GPU-hour training cost**: Compared to VLM training that routinely requires hundreds of GPU hours, this is highly practical.

## Limitations & Future Work

- The student's performance ceiling is bounded by the teacher; the student cannot surpass the teacher.
- Offline document indexing still requires the full 2B VLM; indexing-side costs are not reduced. Teacher compression or incremental indexing are natural future directions.
- Only text query scenarios are validated; multimodal queries (e.g., queries incorporating images) remain unexplored.
- Multilingual augmentation depends on machine translation quality and may introduce semantic drift in terminology-dense domains such as finance or physics.
- No end-to-end fair comparison with the concurrent ModernVBERT (250M vision-language encoder) is provided.

## Related Work & Insights

- **ColPali/Tomoro**: Multi-vector + MaxSim; high quality but 7–8 second latency and 256–819 GB/M index size. NanoVDR uses single-vector cosine similarity with 51 ms latency and an 8.2 GB/M index.
- **DSE-Qwen2**: Also single-vector but uses a 2B VLM for query encoding (2.5 s latency). NanoVDR nevertheless outperforms it on v2/v3 with 32× fewer parameters.
- **SERVAL**: Generates document descriptions with a VLM before text retrieval, incurring 72B+7B inference overhead. NanoVDR's direct embedding space distillation is substantially more efficient.
- **TAS-B/MarginMSE**: Classic methods for text retrieval distillation. This paper demonstrates that alignment outperforms these ranking-based losses.
- The asymmetric distillation paradigm is generalizable to recommendation systems (large model for offline item encoding, small model for online user encoding) and analogous settings.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The method itself is straightforward; the core contributions lie in the asymmetry insight and the systematic empirical demonstration that alignment outperforms ranking.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive ablation across 22 datasets × 3 benchmark versions × 6 losses × 3 backbones, with an elegantly designed cross-lingual analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear structure, every claim supported by data, detailed appendix, and strong reproducibility.
- **Value**: ⭐⭐⭐⭐⭐ A 69M model with CPU inference and 13-hour training directly addresses the core deployment barriers of VDR in practice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] RobustVisRAG: Causality-Aware Vision-Based Retrieval-Augmented Generation under Visual Degradations](robustvisrag_causality-aware_vision-based_retrieval-augmented_generation_under_v.md)
- [\[CVPR 2026\] CC-VQA: Conflict- and Correlation-Aware Method for Mitigating Knowledge Conflict in Knowledge-Based Visual Question Answering](cc-vqa_conflict-_and_correlation-aware_method_for_mitigating_knowledge_conflict_.md)
- [\[ACL 2026\] Prune-then-Merge: Towards Efficient Multi-Vector Visual Document Retrieval](../../ACL2026/information_retrieval/sculpting_the_vector_space_towards_efficient_multi-vector_visual_document_retrie.md)
- [\[CVPR 2026\] Beyond Global Similarity: Towards Fine-Grained, Multi-Condition Multimodal Retrieval](beyond_global_similarity_towards_fine-grained_multi-condition_multimodal_retriev.md)
- [\[CVPR 2026\] Explaining CLIP Zero-shot Predictions Through Concepts](explaining_clip_zero-shot_predictions_through_concepts.md)

</div>

<!-- RELATED:END -->
