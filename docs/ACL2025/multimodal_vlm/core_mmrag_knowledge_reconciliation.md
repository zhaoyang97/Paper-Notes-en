---
title: >-
  [Paper Note] CoRe-MMRAG: Cross-Source Knowledge Reconciliation for Multimodal RAG
description: >-
  [ACL 2025][Multimodal VLM][multimodal RAG] CoRe-MMRAG proposes an end-to-end multimodal RAG framework that addresses Parametric-Retrieval Knowledge Inconsistency (PRKI) and Visual-Textual Knowledge Inconsistency (VTKI) through a four-stage pipeline (parametric knowledge generation → joint visual-textual reranking → external knowledge generation → internal-external knowledge integration), achieving improvements of 5.6% and 9.3% on InfoSeek and Encyclopedic-VQA, respectively.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "multimodal RAG"
  - "knowledge inconsistency"
  - "visual-textual reconciliation"
  - "KB-VQA"
date: 2026-05-08
content_hash: 45ae14a610aa3f7b
---

# CoRe-MMRAG: Cross-Source Knowledge Reconciliation for Multimodal RAG

**Conference**: ACL 2025  
**arXiv**: [2506.02544](https://arxiv.org/abs/2506.02544)  
**Code**: [https://github.com/TyangJN/CoRe-MMRAG](https://github.com/TyangJN/CoRe-MMRAG)  
**Area**: LLM Agent / Multimodal VLM / RAG  
**Keywords**: multimodal RAG, knowledge inconsistency, visual-textual reconciliation, KB-VQA

## TL;DR
CoRe-MMRAG proposes an end-to-end multimodal RAG framework that addresses Parametric-Retrieval Knowledge Inconsistency (PRKI) and Visual-Textual Knowledge Inconsistency (VTKI) through a four-stage pipeline (parametric knowledge generation → joint visual-textual reranking → external knowledge generation → internal-external knowledge integration), achieving improvements of 5.6% and 9.3% on InfoSeek and Encyclopedic-VQA, respectively.

## Background & Motivation

**Background**: Multimodal RAG (MMRAG) enhances MLLMs by retrieving external image-text knowledge, and is widely applied to knowledge-intensive Visual Question Answering (KB-VQA).

**Limitations of Prior Work**:
   - **PRKI (Parametric-Retrieval Knowledge Inconsistency)**: Retrieved information may contradict the model's internal parametric knowledge, making it difficult for the model to judge which source is more reliable—noisy retrieval may override correct parametric knowledge.
   - **VTKI (Visual-Textual Knowledge Inconsistency)**: The images and text of retrieved items may point to different entities, causing text-only reranking to select incorrect items.
   - Existing MMRAG methods (such as Wiki-LLaVA and RoRA-VLM) rely solely on textual similarity in the reranking stage, ignoring cross-modal inconsistency.

**Key Challenge**: MMRAG introduces external knowledge but also introduces both types of inconsistencies, and existing methods lack explicit mechanisms to reconcile them.

**Goal**: To design a framework that simultaneously solves both PRKI and VTKI, enabling MLLMs to reliably integrate multi-source multimodal knowledge.

**Key Insight**: Similar to the "internal first, external second, then integrate" philosophy of Astute RAG, but extended to multimodal scenarios and incorporating joint visual-textual evaluation.

**Core Idea**: First generate a reference answer using parametric knowledge, then select the most relevant retrieved items using joint visual-textual similarity, and finally compare the internal and external answers to perform reliable integration.

## Method

### Overall Architecture
An end-to-end four-stage pipeline: **(1)** Generate an internal answer $y^{int}$ using only parametric knowledge → **(2)** Evaluate joint visual and textual similarity to select the most relevant retrieved item → **(3)** Generate an external answer $y^{ext}$ based on the best retrieved item → **(4)** Compare $y^{int}$ and $y^{ext}$, integrating the most reliable information to generate the final answer $y^*$.

### Key Designs

1. **Joint Visual-Textual Knowledge Integration (Addressing VTKI)**:

    - **Function**: Simultaneously consider the relevance of both images and text to select the best retrieved item.
    - **Mechanism**: $I^{tv} = \arg\max_i r^\mathcal{M}(Q, \{V_i, T_i\}_{i=1}^k)$—instead of ranking images and texts separately, they are evaluated jointly.
    - **Design Motivation**: The individual rankings of images and text can be inconsistent ($I^v \neq I^t$). Joint ranking leverages cross-modal complementarity.

2. **Parametric-Retrieval Knowledge Integration (Addressing PRKI)**:

    - **Function**: Explicitly compare the internal and external answers, assess their reliability, and generate the final answer.
    - **Mechanism**: $y^* = \mathcal{M}(Q, y^{int}, y^{ext}, (V_{I^{tv}}, T_{I^{tv}}))$—the model simultaneously processes the internal reference, the external reference, and the retrieved evidence.
    - **Design Motivation**: Present the model with a "second decision" opportunity—after observing two potentially conflicting answers, it makes a final choice integrated with evidence.

3. **Inconsistency-Aware Training Paradigm**:

    - **Three training objectives**:
        - **Knowledge Source Selection**: Screen samples where the model is correct using only parametric knowledge versus using only external knowledge, training the model to distinguish when to trust which source.
        - **Multimodal Selection**: Train the model to perform joint visual-textual reranking.
        - **Unified Answer Generation**: Train the entire four-stage pipeline for end-to-end generation.
    - **Design Motivation**: Self-training (inspired by STaR) requires no extra annotations and automatically constructs training data leveraging the model's own "biases."

## Key Experimental Results

### Main Results

| Method | InfoSeek | Encyclopedic-VQA |
|------|----------|-----------------|
| Qwen2-VL-7B (baseline) | - | - |
| Wiki-LLaVA | Low | Low |
| RoRA-VLM | Mid | Mid |
| **CoRe-MMRAG** | **+5.6%** | **+9.3%** |

### Ablation Study

| Configuration | InfoSeek |
|------|----------|
| Full CoRe-MMRAG | **Highest** |
| w/o Step 1 (w/o parametric knowledge) | Decrease of ~3% |
| w/o Joint Ranking (text-only ranking) | Decrease of ~2% |
| w/o Step 4 (no integration, direct external) | Decrease of ~4% |
| w/o Training Paradigm (zero-shot) | Decrease of ~5% |

### Key Findings
- **Parametric knowledge reference (Step 1) is crucial to the final answer quality**: Removing it leads to a 3% decrease, as the model loses its "comparison anchor."
- **Joint visual-textual ranking outperforms single-modality ranking**: Textual ranking biases are corrected by visual information.
- **The training paradigm yields the largest contribution (~5%)**: This indicates that the model needs to learn how to judge the reliability of knowledge sources.

## Highlights & Insights
- **The formalization of PRKI and VTKI is a significant contribution**: It deconstructs the vague "noise" issue in multimodal RAG into two distinct types of inconsistencies, providing a clear problem formulation for future research.
- **The four-stage "internal first, external second, then integrate" design shares similarities with Astute RAG**: It validates the generalizability of this knowledge reconciliation paradigm in the multimodal domain.
- **The self-training paradigm constructs training data in a clever way**: It automatically generates labels based on performance discrepancies with versus without retrieval, bypassing the need for manual annotation.

## Limitations & Future Work
- **Tested only on KB-VQA**: Broader multimodal tasks are not yet covered.
- **Base model limited to Qwen2-VL-7B**: It remains uncertain whether the gains are consistent across larger or stronger models.
- **High end-to-end inference cost**: The four-stage pipeline requires multiple generation passes by the model, increasing inference latency.

## Related Work & Insights
- **vs Astute RAG**: Astute RAG solves PRKI in textual RAG, while CoRe-MMRAG extends to multimodal scenarios and introduces VTKI resolution.
- **vs Wiki-LLaVA**: Wiki-LLaVA simply injects retrieved text into the prompt without handling inconsistency.
- **vs EchoSight**: EchoSight uses visual retrieval but relies only on text reranking, whereas CoRe-MMRAG performs joint ranking.

## Rating
- Novelty: ⭐⭐⭐⭐ Clear formalization of VTKI+PRKI, with a well-designed four-stage reconciliation mechanism.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two benchmarks, ablation studies, and comparisons between zero-shot and fine-tuned settings.
- Writing Quality: ⭐⭐⭐⭐ Clear formal definitions and intuitive architecture diagrams.
- Value: ⭐⭐⭐⭐ Provides an explicit solution to the knowledge inconsistency issue in multimodal RAG.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Core Knowledge Deficits in Multi-Modal Language Models](../../ICML2025/multimodal_vlm/core_knowledge_deficits_in_multi-modal_language_models.md)
- [\[ACL 2025\] REAL-MM-RAG: A Real-World Multi-Modal Retrieval Benchmark](real-mm-rag_a_real-world_multi-modal_retrieval_benchmark.md)
- [\[ACL 2025\] Insight Over Sight: Exploring the Vision-Knowledge Conflicts in Multimodal LLMs](conflictvis_vision_knowledge_conflict.md)
- [\[CVPR 2026\] M4-RAG: A Massive-Scale Multilingual Multi-Cultural Multimodal RAG](../../CVPR2026/multimodal_vlm/m4-rag_a_massive-scale_multilingual_multi-cultural_multimodal_rag.md)
- [\[ACL 2025\] MAGIC-VQA: Multimodal and Grounded Inference with Commonsense Knowledge for Visual Question Answering](magic-vqa_multimodal_and_grounded_inference_with_commonsense_knowledge_for_visua.md)

</div>

<!-- RELATED:END -->
