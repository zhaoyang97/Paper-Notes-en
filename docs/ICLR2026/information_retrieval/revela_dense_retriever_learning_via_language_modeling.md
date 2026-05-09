---
title: >-
  [Paper Note] Revela: Dense Retriever Learning via Language Modeling
description: >-
  [ICLR2026][dense retrieval] This paper proposes Revela, which integrates retriever learning into language modeling via an in-batch attention mechanism. Next-token prediction (NTP) draws not only on within-sequence context but also on other sequences in the batch, weighted by retriever similarity scores, enabling training of a strong dense retriever without labeled query-document pairs.
tags:
  - ICLR2026
  - dense retrieval
  - self-supervised learning
  - language modeling
  - in-batch attention
  - retriever
date: 2026-05-08
content_hash: 4e296c74e11ee1d1
---

# Revela: Dense Retriever Learning via Language Modeling

**Conference**: ICLR2026  
**arXiv**: [2506.16552](https://arxiv.org/abs/2506.16552)  
**Code**: To be confirmed  
**Area**: Information Retrieval  
**Keywords**: dense retrieval, self-supervised learning, language modeling, in-batch attention, retriever

## TL;DR
This paper proposes Revela, which integrates retriever learning into language modeling via an in-batch attention mechanism. Next-token prediction (NTP) draws not only on within-sequence context but also on other sequences in the batch, weighted by retriever similarity scores, enabling training of a strong dense retriever without labeled query-document pairs.

## Background & Motivation

**Background**: Dense retrievers typically require labeled query-document pairs for training, making annotation expensive in specialized domains and complex scenarios.

**Limitations of Prior Work**: Self-supervised retrieval methods (e.g., Contriever) tend to overfit structural biases in data; auto-encoding approaches lack pairwise supervision.

**Key Challenge**: LMs learn inter-token dependencies via NTP (a successful self-supervised paradigm). The key question is how to extend a similar idea to learning inter-chunk dependencies.

**Key Insight**: Retrieval is framed as "sequence-level NTP"—NTP identifies the most relevant preceding tokens, while retrieval identifies the most relevant documents.

**Core Idea**: In-batch attention is introduced into Transformer blocks, allowing NTP to leverage both within-sequence context and other sequences in the batch, with cross-sequence weights provided by the retriever.

## Method

### Overall Architecture
Documents are split into chunks and placed in the same batch. The retriever computes inter-chunk similarity, which modulates in-batch attention within the LM's Transformer blocks. The NTP loss jointly optimizes both the LM and the retriever.

### Key Designs

1. **In-batch Attention**: The embedding of sequence $i$ can attend to the embeddings of other sequences $j$ in the batch, with weights modulated by the retriever-computed $\text{Sim}(D_i, D_j)$.
2. **Joint Optimization**: The retriever and LM are trained end-to-end under a shared NTP objective.
3. **Same-document Negatives**: Different chunks from the same document are placed in the same batch, functioning similarly to hard negatives.

### Loss & Training
- Training is conducted on Wikipedia (general retrieval) or code corpora (StackOverflow + documentation) for code retrieval.
- Both the retriever and LM are fine-tuned with LoRA (rank=256); temperature $\tau=10^{-4}$, learning rate $10^{-4}$.
- Training runs for only 1 epoch: ~10K steps on Wikipedia (~44 hours) and ~11K steps on code (~48 hours), using 4×A100 GPUs.
- At inference, queries and documents support up to 2048 tokens; the `<eos>` token embedding serves as the document representation.
- "Query:" and "Passage:" prefixes are prepended to distinguish queries from passages.

## Key Experimental Results

### Main Results

| Method | CoIR (nDCG@10) | BRIGHT | BEIR |
|------|------|--------|------|
| E5-Mistral-7B (supervised) | baseline | baseline | baseline |
| **Revela-3B** (unsupervised) | **+2.8** | **surpasses commercial APIs** | **unsupervised SOTA** |

### Key Findings
- Revela without labeled data outperforms a supervised model with 7B parameters.
- It achieves unsupervised SOTA on BEIR using ~1000× less data and ~10× less compute.
- Cross-domain generalization is stronger than contrastive learning methods.
- Performance scales consistently with batch size and model size.

## Ablation Study

| Ablation / Analysis | Finding |
|-----------|------|
| Batch size scaling | Performance improves monotonically with batch size; larger batches provide more negatives. |
| Retriever size scaling | Consistent improvement from 135M to 3B, following a scaling law. |
| LM size | A larger LM (1B→3B) provides a stronger learning signal for the retriever. |
| Mixed-domain training | Joint training on Wikipedia + Code does not hurt single-domain performance while improving cross-domain generalization. |
| vs. REPLUG | Revela outperforms REPLUG at all scales; joint optimization is superior to a frozen LM. |
| Cross-domain generalization | Revela trained on Wikipedia surpasses commercial APIs on unseen reasoning-intensive tasks in BRIGHT. |

### Detailed CoIR Results (nDCG@10)

| Method | Scale | Avg. nDCG@10 |
|------|------|-------------|
| UniXCoder (supervised) | 0.1B | baseline |
| Revela | 0.1B | +11.1 |
| E5-PT (weakly supervised, 270M pairs) | 0.3B | baseline |
| Revela | 0.5B | **+9.7** |
| BGE-M3 (supervised) | 0.6B | baseline |
| Revela | 0.5B | **surpasses** |
| E5-Mistral-7B (supervised) | 7B | baseline |
| **Revela** | **3B** | **+2.8** |

## Highlights & Insights
- **The NTP→retrieval analogy** is both natural and effective: inter-token dependencies ↔ inter-document dependencies.
- **Power of joint optimization**: REPLUG relies on perplexity from a frozen LM, which is often poorly calibrated; Revela's joint updates address this fundamental issue.
- **Remarkable data efficiency**: ~1000× less data and ~10× less compute suffice to achieve unsupervised SOTA on BEIR—demonstrating that method design matters more than data scaling.
- **Cross-domain generalization**: Stronger than traditional contrastive learning methods, as the NTP objective captures more general "semantic dependencies" rather than superficial co-occurrence.

## Limitations & Future Work
- Performance is sensitive to batch size; a sufficiently large batch (16+) is required, which may become a bottleneck under resource constraints.
- In-batch attention increases training-time computational overhead, as each sequence must attend to all other sequences in the batch.
- Only text and code retrieval have been validated; multimodal retrieval (image, audio, etc.) remains unexplored.
- The effect of chunk segmentation strategies (sentence boundaries, fixed length, etc.) on performance has not been thoroughly analyzed.
- For very long documents (>2048 tokens), the current chunking approach may lose long-range dependencies.

## Related Work & Insights
- **vs. Contriever (Izacard et al.)**: Contriever uses contrastive learning (same document = positive, cross-document = negative); Revela models cross-document conditional probabilities via NTP, capturing "why two documents are related" more precisely.
- **vs. Atlas (Izacard et al.)**: Atlas trains the retriever using cross-attention signals from an encoder-decoder architecture and requires periodic re-indexing; Revela uses a decoder-only model with in-batch attention, which is more efficient and requires no re-indexing.
- **vs. REPLUG (Shi et al.)**: REPLUG distills a frozen LM via perplexity; Revela trains jointly—experiments confirm joint training is significantly superior at all scales.
- **vs. E5-PT (Wang et al.)**: E5-PT trains on 270M weakly supervised pairs, whereas Revela learns from raw text alone—demonstrating that a well-designed objective can substitute for large volumes of labeled data.
- **Broader Inspiration**: The in-batch attention paradigm can generalize to any setting requiring modeling of "intra-set relationships," such as multi-document summarization and cross-modal retrieval.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The paradigm of learning retrieval via NTP is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three benchmarks, multi-scale evaluation, and scaling analysis.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clear and formulations are rigorous.
- Value: ⭐⭐⭐⭐⭐ Provides a powerful new paradigm for self-supervised retrieval.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] HiMo-CLIP: Modeling Semantic Hierarchy and Monotonicity in Vision-Language Alignment](../../AAAI2026/information_retrieval/himo-clip_modeling_semantic_hierarchy_and_monotonicity_in_vi.md)
- [\[AAAI 2026\] RRRA: Resampling and Reranking through a Retriever Adapter](../../AAAI2026/information_retrieval/rrra_resampling_and_reranking_through_a_retriever_adapter.md)
- [\[ICCV 2025\] aligning information capacity between vision and language via dense-to-sparse fe](../../ICCV2025/information_retrieval/aligning_information_capacity_between_vision_and_language_via_dense-to-sparse_fe.md)
- [\[CVPR 2026\] NanoVDR: Distilling a 2B Vision-Language Retriever into a 70M Text-Only Encoder for Visual Document Retrieval](../../CVPR2026/information_retrieval/nanovdr_distilling_a_2b_visionlanguage_retriever_i.md)
- [\[NeurIPS 2025\] Scaling Language-Centric Omnimodal Representation Learning](../../NeurIPS2025/information_retrieval/scaling_language-centric_omnimodal_representation_learning.md)

</div>

<!-- RELATED:END -->
