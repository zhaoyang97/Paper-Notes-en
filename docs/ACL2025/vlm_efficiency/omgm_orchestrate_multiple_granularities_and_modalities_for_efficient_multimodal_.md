---
title: >-
  [Paper Note] OMGM: Orchestrate Multiple Granularities and Modalities for Efficient Multimodal Retrieval
description: >-
  [ACL 2025][Multimodal Efficiency][Multimodal Retrieval] The authors propose OMGM, a multimodal RAG system for knowledge-bound visual question answering (KB-VQA). By orchestrating the matching between query and knowledge base across various granularities and modalities via a coarse-to-fine three-step retrieval strategy, OMGM achieves state-of-the-art retrieval performance and highly competitive VQA results on InfoSeek and E-VQA datasets.
tags:
  - "ACL 2025"
  - "Multimodal Efficiency"
  - "Multimodal Retrieval"
  - "Knowledge-bound VQA"
  - "RAG"
  - "Granularity Alignment"
  - "Multimodal Fusion Re-ranking"
  - "Coarse-to-fine Retrieval"
date: 2026-05-08
content_hash: 7611616d82aa06e9
---

# OMGM: Orchestrate Multiple Granularities and Modalities for Efficient Multimodal Retrieval

**Conference**: ACL 2025  
**arXiv**: [2505.07879](https://arxiv.org/abs/2505.07879)  
**Authors**: Wei Yang, Jingjing Fu, Rui Wang, Jinyu Wang, Lei Song, Jiang Bian (Microsoft Research Asia)
**Code**: Not released  
**Area**: Multimodal VLM  
**Keywords**: Multimodal Retrieval, Knowledge-bound VQA, RAG, Granularity Alignment, Multimodal Fusion Re-ranking, Coarse-to-fine Retrieval

## TL;DR

The authors propose OMGM, a multimodal RAG system for knowledge-bound visual question answering (KB-VQA). By orchestrating the matching between query and knowledge base across various granularities and modalities via a coarse-to-fine three-step retrieval strategy, OMGM achieves state-of-the-art retrieval performance and highly competitive VQA results on InfoSeek and E-VQA datasets.

## Background & Motivation

### Background
Knowledge-bound visual question answering (KB-VQA) requires systems to not only comprehend image content but also retrieve relevant knowledge about the image subject from an external knowledge base to answer questions. Retrieval-Augmented Generation (RAG) is a dominant solution for such tasks, but multimodal retrieval faces two core challenges:

**Multimodality**: Both the query and the knowledge base contain both image and text modalities, requiring flexible utilization of unimodal, cross-modal, and multimodal matching strategies.

**Multi-granularity**: Within the query, the image provides coarse-grained entity information while the question targets fine-grained knowledge; in the knowledge base, an entity article comprises a coarse-grained overview image and summary alongside fine-grained detailed sections.

### Limitations of Prior Work
- **Single-step retrieval methods** (e.g., PreFLMR, MuKA): Direct multimodal retrieval over the entire database requires expensive task-specific pre-training and suffers from high computational costs during full-database search at inference time.
- **Multi-step retrieval methods** (e.g., Wiki-LLaVA, EchoSight): Employ hierarchical strategies but neglect the coordinated design of modalities and granularities across retrieval steps, leaving similarity scores from individual steps underutilized and un-propagated.
- Existing approaches generally lack a systematic consideration of granularity alignment between queries and the knowledge base.

### Design Motivation
To design a coarse-to-fine multi-step retrieval retrieval flow that selects the appropriate modality and granularity at each step, achieving globally optimal entity localization and knowledge filtering via cross-step similarity propagation.

## Method

### Overall Architecture
OMGM adopts a three-tier coarse-to-fine retrieval strategy:
1. **Coarse-grained cross-modal entity search**: Matches query images with entity summaries to filter top-k candidate entities from a million-scale knowledge base.
2. **Mixed-granularity multimodal fusion re-ranking**: Re-ranks candidate entities using fused multimodal features, selecting the most relevant entity in combination with the similarity scores from the previous step.
3. **Fine-grained text-boosted generation**: Filters the most relevant sections within the selected entity using a text re-ranker, passing them as context to the generator to answer the question.

### Key Design 1: Granularity-Aligned Entity Search
The core idea is to align the query and retrieval index in terms of information granularity. Since the query image inherently carries coarse-grained entity identity information, the retrieval index should likewise be a coarse-grained overview of the entity, rather than the full article or just the title.

Specifically:
- **Offline summary generation**: Synthesizes concise summaries $s_i = M_s(P, a_i)$ for all entity articles in the knowledge base using a pre-trained language model, which serve as the retrieval index.
- **Image-to-summary matching**: Encodes query images and entity summaries using CLIP's vision and text encoders respectively, performing inner product similarity search via FAISS to preserve the top-k entities.

Ablation studies verify that the Image→Summary retrieval outperforms Image→Article, Image→Image, and Image→Title, demonstrating the critical importance of granularity alignment.

### Key Design 2: Multimodal Fusion Re-ranking and Cross-Step Similarity Propagation
Fine-grained ranking within the top-k candidate entities requires utilizing the full multimodal information of the query (image + question) alongside the multimodal information of the candidates (entity main images + section text).

**Multimodal fusion feature extraction**: Employs the Q-Former architecture to fuse images and text into unified feature matrices:

$$Q = E_m(I_q, T_q), \quad C_{sec_e^h} = E_m(I_e, sec_e^h)$$

**Late interaction fine-grained matching**: Computes token-level fine-grained similarity between the query and candidate via a Max-Sum operator:

$$sim_m^{sec_e^h} = \sum_{i=1}^{l_Q} \max_{j=1}^{l_C} Q_i {C_{sec_e^h}^j}^\top$$

**Cross-step similarity fusion**: Weighted fusion is applied to combine the entity similarity from the first step $sim_c^e$ and the current multimodal similarity, selecting the final entity:

$$e_{top1} = \arg\max_{e \in Ent_k} \left(\alpha \cdot sim_c^e + (1-\alpha) \cdot \max_h sim_m^{sec_e^h}\right)$$

During training, hard negatives from the first retrieval step are leveraged for contrastive learning. Positive samples consist of the correct entity's main image + the evidence section, while negative samples comprise candidate entities' main images + non-evidence sections.

### Key Design 3: Fine-grained Section Filtering
Following the identification of the top-1 entity, the scores of the text re-ranker and the multimodal re-ranker are combined to filter the most relevant sections:

$$sec_{e_{top1}}^{best} = \arg\max_{sec \in e_{top1}} \left(\beta \cdot sim_m^{sec} + (1-\beta) \cdot sim_t^{sec}\right)$$

Ultimately, the filtered sections are fed alongside the query as context into the downstream generator.

## Key Experimental Results

### Retrieval Performance Comparison

| Method | E-VQA R@1 | E-VQA R@20 | InfoSeek R@1 | InfoSeek R@20 |
|------|-----------|------------|--------------|---------------|
| CLIP I-T | 3.3 | 16.5 | 32.0 | 68.2 |
| Wiki-LLaVA | 3.3 | 13.2 | 36.9 | 71.9 |
| EchoSight (w. rerank) | 36.5 | 48.8 | 53.2 | 77.9 |
| ReflectiVA | 15.6 | 49.8 | 56.1 | 86.4 |
| **OMGM (w. rerank)** | **42.8** | **58.7** | **64.0** | **84.8** |

OMGM outperforms EchoSight by 6.3 percentage points in R@1 on E-VQA, and beats ReflectiVA by 7.9 percentage points in R@1 on InfoSeek. The re-ranking stage alone contributes to a R@1 gain of 23.7% on E-VQA and 11.4% on InfoSeek.

### VQA Performance Comparison

| Method | Generator | Gen. FT | E-VQA | InfoSeek Overall |
|------|--------|---------|-------|------------------|
| Wiki-LLaVA | LLaVA-1.5-7B | ✓ | 21.8 | 28.9 |
| mR2AG | LLaVA-1.5-7B | ✓ | - | 40.2 |
| ReflectiVA | LLaVA-MORE-8B | ✓ | 35.5 | 40.1 |
| **OMGM** | LLaVA-1.5-7B | ✓ | **50.17** | **43.49** |
| OMGM | InternVL-2.5-8B | ✗ | 48.72 | 36.1 |

Under fine-tuned LLaVA-1.5-7B, OMGM achieves 50.17 on E-VQA and 43.49 on InfoSeek, both establishing state-of-the-art entries. Notably, even without generator tuning (tuning only the retriever), OMGM still outperforms most of the existing approaches that fine-tune the generator.

### Efficiency Comparison

| Method | Avg. Retrieval Time (s) | Avg. Inference Time (s) | E-VQA VQA Result |
|------|---------------|---------------|--------------|
| LLaVA-1.5-7B | - | 1.432 | 17.00 |
| PreFLMR | 0.984 | 2.196 | 54.45 |
| **OMGM** | **0.402** | 2.023 | **63.39** |

The multi-step retrieval is paradoxically faster than the single-step PreFLMR (0.4s vs 0.98s), as each subsequent step only searches over a contracted candidate space.

## Key Findings

- **Granularity alignment is critical**: Image→Summary retrieval achieves a R@20 of 58.7%, significantly outperforming Image→Article (41.7%) and Image→Image (48.8%), illustrating that matching the information granularity between query and index is a prerequisite for effective retrieval.
- **Dual-end multimodal fusion surpasses single-end**: Utilizing the multimodal fused (I,T)→(I,T) mode at both the query and candidate ends achieves a R@1 of 40.2%, substantially exceeding the text-only T→T counterpart (30.7%).
- **Progressive improvement across retrieval steps**: The three distinct steps contribute incremental VQA performance gains—16.25% from the first, 14.36% from the second, and 2.0% from the third (on E-VQA)—with the second-step multimodal re-ranking providing the largest boost.
- **Effective cross-step similarity propagation**: Passing and fusing similarity scores from previous steps into subsequent steps proves more robust than computing each step independently.

## Highlights & Insights

- **Systematic granularity-modality coordination**: Rather than casually stacking multi-step retrieval blocks, OMGM carefully orchestrates the matching of query and index modalities alongside their information granularities at each step, forming a complementary retrieval pipeline.
- **Lightweight and efficient**: Only requires training a single Q-Former re-ranker, yielding significant retrieval and VQA performance gains without relying on extensive generator fine-tuning. Multi-step retrieval paradoxically speeds up inference over full single-step search by gradually narrowing down the candidate space.
- **Cross-step information flow**: Achieves synergy across retrieval stages via similarity score propagation, instead of isolating multiple retrievers in a naive sequence.
- **Insight on using summaries as retrieval indices**: Substituting raw documents or titles with LLM-generated entity summaries as indices compresses information while preserving semantic alignment with query images—an approach highly generalizable to other RAG settings.

## Limitations & Future Work

- **Underutilization of auxiliary images in KB**: Entity articles often contain fine-grained pictures tied to specific sections. The current approach only utilizes the main entity image, omitting these potential multimodal cues.
- **Lack of deep integration on the generator side**: Retrieved fused multimodal features are parsed solely as textual context into the generator; the potential of directly injecting fused representations into the generation process remains unexplored.
- **Reliance on summary generation quality**: Retrieval efficacy in the first step heavily hinges on the quality of LLM-generated summaries, which might degrade for niche or highly specialized entities.
- **Validation limited to Wikipedia-styled databases**: Despite testing generalization on OK-VQA, the applicability of this framework to unstructured or non-encyclopedic knowledge bases requires further study.
- **Scalability of the re-ranking range $k$**: Increasing $k$ from 20 to 100 yields sustained retrieval improvements but scales computational time linearly; how to scale the re-ranking range efficiently on larger-scale databases lacks in-depth discussion.

## Related Work & Insights

- **EchoSight** (Yan & Xie 2024): Also utilizes multi-step retrieval, yet lacks granularity alignment and cross-step propagation mechanisms. OMGM systematically refines the modality-granularity configuration at each step compared to it.
- **PreFLMR** (Lin et al. 2024): Single-step exhaustive multimodal retrieval carrying heavy encoding overheads. OMGM demonstrates that progressive retrieval concurrently improves performance and speed.
- **ReflectiVA** (Cocchi et al. 2024): Drives iterative retrieval with reflective tokens, focusing on the interaction between generator and retriever. In contrast, OMGM emphasizes the coordination of granularity and modality within the retrieval pipeline itself.
- **Insights**: The concept of granularity alignment can be transferred to document-level RAG—coarse-filtering with document summaries, fine-ranking with paragraph-level selection, and finally extracting with sentence-level tools, employing the most fitting representation and matching mode for each stage.

## Rating

- Novelty: ⭐⭐⭐⭐ — The systematic design targeting coarse-to-fine multi-granularity and multimodal coordination brings a clear innovation, albeit individual technical blocks (CLIP retrieval, Q-Former re-ranking, contrastive learning) rely on mature paradigms.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Solid validation across two mainstream datasets and OK-VQA generalization tests; ablation studies verify each design choice alongside complete efficiency analysis.
- Writing Quality: ⭐⭐⭐⭐ — Structured logically, enriched with data figures/tables, systematically describes methods, and thoroughly articulates motivations.
- Value: ⭐⭐⭐⭐ — Demonstrates strong results on KB-VQA retrieval; the core philosophy of granularity alignment is highly inspiring for general multimodal RAG.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] HotelMatch-LLM: Joint Multi-Task Training of Small and Large Language Models for Efficient Multimodal Hotel Retrieval](hotelmatch_llm_retrieval.md)
- [\[ACL 2025\] MadaKV: Adaptive Modality-Perception KV Cache Eviction for Efficient Multimodal Long-Context Inference](madakv_adaptive_modality-perception_kv_cache_eviction_for_efficient_multimodal_l.md)
- [\[ACL 2025\] Token Pruning in Multimodal Large Language Models: Are We Solving the Right Problem?](token_pruning_in_multimodal_large_language_models_are_we_solving_the_right_probl.md)
- [\[NeurIPS 2025\] ElasticMM: Efficient MLLM Serving with Elastic Multimodal Parallelism](../../NeurIPS2025/vlm_efficiency/elasticmm_efficient_multimodal_llms_serving_with_elastic_multimodal_parallelism.md)
- [\[NeurIPS 2025\] SCOPE: Saliency-Coverage Oriented Token Pruning for Efficient Multimodal LLMs](../../NeurIPS2025/vlm_efficiency/scope_saliency-coverage_oriented_token_pruning_for_efficient_multimodel_llms.md)

</div>

<!-- RELATED:END -->
