---
title: >-
  [Paper Note] URaG: Unified Retrieval and Generation in Multimodal LLMs for Efficient Long Document Understanding
description: >-
  [AAAI 2026][Multimodal VLM][Long document understanding] URaG identifies a human-like "coarse-to-fine" reasoning pattern in MLLMs processing long documents—shallow layers exhibit uniformly distributed attention while dee…
tags:
  - "AAAI 2026"
  - "Multimodal VLM"
  - "Long document understanding"
  - "unified retrieval-generation"
  - "cross-modal retrieval"
  - "Transformer layer attention analysis"
  - "token efficiency"
date: 2026-05-08
content_hash: 8d53217556ca2fbd
---

# URaG: Unified Retrieval and Generation in Multimodal LLMs for Efficient Long Document Understanding

**Conference**: AAAI 2026  
**arXiv**: [2511.10552](https://arxiv.org/abs/2511.10552)  
**Code**: [https://github.com/shi-yx/URaG](https://github.com/shi-yx/URaG)  
**Area**: Multimodal Large Language Models / Document Understanding  
**Keywords**: Long document understanding, unified retrieval-generation, cross-modal retrieval, Transformer layer attention analysis, token efficiency  

## TL;DR
URaG identifies a human-like "coarse-to-fine" reasoning pattern in MLLMs processing long documents—shallow layers exhibit uniformly distributed attention while deep layers concentrate on evidence pages. Motivated by this insight, a lightweight cross-modal retrieval module is inserted at layer 6 (comprising only 0.05% of total parameters) to select the Top-5 relevant pages and discard the remainder, achieving SOTA performance while reducing computation by 44–56%.

## Background & Motivation
**Background**: MLLMs perform well on single-page documents, but multi-page long document understanding faces two major challenges: interference from irrelevant content and the quadratic complexity of Transformers.

**Limitations of Prior Work**: Two categories of existing approaches each have distinct drawbacks. (1) Token compression methods (e.g., mPLUG-DocOwl2) uniformly compress visual tokens across all pages, inevitably losing fine-grained visual detail. (2) External retriever methods (e.g., CREAM, SV-RAG) introduce separate retrieval modules that increase system complexity; the retriever and the MLLM cannot be optimized end-to-end, making them prone to "coordination failure and error propagation."

**Key Challenge**: Only a small number of pages in a long document contain answer evidence, yet existing methods either compress all pages (sacrificing precision) or require additional retrieval systems (increasing complexity).

**Goal**: How can evidence-page retrieval and answer generation be unified efficiently within a single MLLM?

**Key Insight**: Systematic analysis reveals that MLLMs already exhibit a human-like "coarse-to-fine" reasoning pattern internally—shallow layers scan globally in a uniform manner, intermediate layers gradually focus, and deep layers concentrate on evidence pages. This implies that hidden states at shallow layers are already sufficient to distinguish relevant from irrelevant pages, enabling the insertion of a lightweight retrieval module at that point.

**Core Idea**: Transform the shallow layers of an MLLM into a retriever; after filtering evidence pages, allow the deep layers to process only the relevant content, thereby unifying retrieval and generation.

## Method

### Overall Architecture
A single MLLM processes the long document: all pages pass through a visual encoder and projection layer to obtain visual tokens → concatenated with the text query and fed into the LLM → a cross-modal retrieval module is inserted at layer 6 → Top-5 relevant pages are retrieved → hidden states of the remaining pages are discarded → subsequent layers process only the retained content → the answer is generated.

### Key Designs

1. **Core Analytical Insight — Coarse-to-Fine Reasoning Pattern**:

    - Function: Systematically analyze the attention distribution and embedding retrieval capability at each layer of the MLLM.
    - Key Findings: (1) Shallow layers (1–3): high attention entropy, low retrieval accuracy — uniform scanning. (2) Intermediate layers (3–20): decreasing entropy, increasing retrieval accuracy — progressive focusing. (3) Deep layers (20–34): low entropy, high retrieval accuracy — concentrated on evidence pages. (4) Last 2 layers: entropy rises again — analogous to humans re-reading the full document for confirmation.
    - Additional Key Finding: Embedding-based retrieval achieves high accuracy at shallower layers (around layer 12) and is more stable than attention-based retrieval; accordingly, embedding retrieval is adopted.

2. **Cross-Modal Retrieval Module**:

    - Function: Extract hidden states at layer 6, compute relevance between the query and each page, and select the Top-5.
    - Mechanism: Two linear layers with GELU project $H \in \mathbb{R}^{L \times D}$ to $H' \in \mathbb{R}^{L \times D'}$ ($D'=512$), followed by L2 normalization. Query text features $E_q$ and per-page visual features $E_v^{(p)}$ are extracted, and similarity is computed using ColBERT-style contextualized late interaction: $s_{q,v}^{(p)} = \sum_i \max_j E_{q_i} \cdot E_{v_j}^{(p)T}$
    - Design Motivation: Only two linear layers are required (2.5–4M parameters, 0.05–0.07% of the total model), making the module extremely lightweight. Late interaction preserves richer token-level matching information compared to global pooling.

3. **Two-Stage Training Strategy**:

    - Function: Pre-train the retrieval module first, then jointly fine-tune with the LLM.
    - Stage 1: Freeze all MLLM parameters; train only the retrieval module by optimizing the ColBERT retrieval loss $\mathcal{L}_{\text{retrieval}}$.
    - Stage 2: Add LoRA (rank=32) to both the LLM and the retrieval module; jointly optimize $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{retrieval}} + \mathcal{L}_{\text{generation}}$. During training, ground-truth evidence pages are retained together with the highest-scoring retrieved pages (up to 5 pages total).
    - Design Motivation: The two-stage approach ensures the retrieval module first adapts to the task before being co-optimized with the generation component.

## Key Experimental Results

### Main Results — Document Understanding

| Method | Params | MPDocVQA | SlideVQA | MMLongBench F1 |
|--------|--------|---------|----------|----------------|
| Qwen2.5-VL-3B | 3B | 84.4 | 59.1 | 24.1 |
| Qwen2.5-VL-7B | 7B | 87.2 | 66.4 | 25.1 |
| **URaG-3B** | 3B | 86.0 | 63.8 | 28.7 |
| **URaG-7B** | 7B | **88.2** | **72.1** | **32.8** |

### Retrieval Performance

| Method | SlideVQA Top1/Top5 | MMLongBench Top1/Top5 |
|--------|-------------------|---------------------|
| ColPali (3B) | 90.2/98.2 | 60.3/80.2 |
| SV-RAG (4B) | 90.6/98.8 | 64.8/84.8 |
| **URaG-7B** | **92.9/99.0** | **68.3/86.0** |

### Ablation Study

| Configuration | SlideVQA | MMLongBench F1 | Note |
|--------------|----------|---------------|------|
| Layer 2 | 63.7 | 31.0 | Too shallow; retrieval imprecise |
| Layer 6 (default) | **63.8** | **31.1** | Optimal balance |
| Layer 12 | 62.9 | 31.0 | Better retrieval but insufficient deep reasoning capacity |
| Layer 18 | 62.3 | 30.6 | Placing the module too deep degrades performance |

### Key Findings
- URaG outperforms Qwen2.5-VL-7B by 5.7 points on SlideVQA (66.4→72.1) and 7.7 points on MMLongBench F1 (25.1→32.8) — gains are most pronounced in long-document scenarios.
- Pre-training the retrieval module alone (without fine-tuning the LLM) already surpasses the fully fine-tuned baseline (62.1 vs. 61.9 on SlideVQA), demonstrating the dominant contribution of retrieval itself.
- Computational efficiency: FLOPs are reduced by 55.8%, inference time by 41.6%, and GPU memory by 51.3% on 100-page documents.
- The retrieval module accounts for only 0.05–0.07% of total parameters — effectively negligible overhead.
- Layer 6 is the optimal insertion point — Top-5 retrieval accuracy already reaches 98.9% while preserving the maximum deep reasoning capacity.

## Highlights & Insights
- **Discovery of the Coarse-to-Fine Reasoning Pattern**: The gradual transition in per-layer attention from uniform to concentrated provides a theoretical basis for determining "where to insert which operation" within a Transformer — this analytical methodology can be generalized to other scenarios requiring intermediate-layer interventions.
- **Extremely Lightweight Design**: Two linear layers suffice to convert the shallow layers of an MLLM into an effective retriever at only 0.05% parameter cost — demonstrating that MLLM hidden states already encode sufficient cross-modal semantics.
- **Unified Architecture Eliminates System Complexity**: No additional retrieval pipeline is required; the system is end-to-end optimizable and deployed as a single model.

## Limitations & Future Work
- The fixed Top-5 retrieval strategy lacks flexibility — evidence spread across more than 5 pages may be missed, while evidence concentrated on a single page introduces redundancy.
- Training data covers only 3 datasets, potentially limiting generalization — fine-tuning on LongDocURL leads to performance regression.
- Multi-page evidence retrieval remains weaker than single-page — the retrieval module is better suited to localizing isolated evidence.
- Experiments are conducted exclusively on the Qwen2.5-VL series; InternVL2.5 experiments cover only a single configuration.
- Discarding hidden states from non-relevant pages is irreversible if the retrieval module makes an error.

## Related Work & Insights
- **vs. SV-RAG**: SV-RAG performs retrieval using the last-layer hidden states of the MLLM, requiring two forward passes; URaG truncates at layer 6, requiring only one — substantially more efficient.
- **vs. mPLUG-DocOwl2**: Uniform compression vs. selective retention — URaG applies no compression to retained pages, preserving complete visual detail.
- **vs. CREAM**: CREAM employs a complex multi-round retrieval and LLM re-ranking pipeline; URaG accomplishes the same within the model using only two linear layers — considerably simpler.
- Inspiration: The paradigm of "making decisions at intermediate Transformer layers" is broadly applicable — for instance, token pruning, dynamic routing, or early-exit decisions at intermediate layers.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The discovery of the coarse-to-fine reasoning pattern and the idea of converting shallow layers into a retriever are highly elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Five benchmarks, dual-dimension evaluation of retrieval and generation, efficiency analysis, ablation studies, and visualization.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain from analytical insight to method design is exceptionally clear.
- Value: ⭐⭐⭐⭐⭐ The unified retrieval-generation paradigm carries significant implications for long document understanding, and the 0.05% parameter overhead is highly practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DocSeeker: Structured Visual Reasoning with Evidence Grounding for Long Document Understanding](../../CVPR2026/multimodal_vlm/docseeker_long_document_understanding.md)
- [\[AAAI 2026\] FinMMDocR: Benchmarking Financial Multimodal Reasoning with Scenario Awareness, Document Understanding, and Multi-Step Computation](finmmdocr_benchmarking_financial_multimodal_reasoning_with_scenario_awareness_do.md)
- [\[ICCV 2025\] Harmonizing Visual Representations for Unified Multimodal Understanding and Generation](../../ICCV2025/multimodal_vlm/harmonizing_visual_representations_for_unified_multimodal_un.md)
- [\[CVPR 2026\] PersonaVLM: Long-Term Personalized Multimodal LLMs](../../CVPR2026/multimodal_vlm/personavlm_long_term_personalized_multimodal_llms.md)
- [\[ICLR 2026\] Index-Preserving Lightweight Token Pruning for Efficient Document Understanding](../../ICLR2026/multimodal_vlm/index-preserving_lightweight_token_pruning_for_efficient_document_understanding_.md)

</div>

<!-- RELATED:END -->
