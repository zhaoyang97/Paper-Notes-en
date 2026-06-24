---
title: >-
  [Paper Note] DRAG: Distilling RAG for SLMs from LLMs to Transfer Knowledge and Mitigate Hallucination
description: >-
  [ACL 2025][Hallucination Detection][RAG Distillation] DRAG proposes a framework to distill RAG capabilities from large language models (LLMs) to small language models (SLMs): utilizing an LLM (e.g., GPT-4o) to generate evidence and knowledge graph triples for a given question. After ranking and filtering, these are fed to SLMs (2B-9B) as structured contexts, boosting SLM performance on ARC-C by up to 27.7% without fine-tuning, while significantly mitigating hallucinations.
tags:
  - "ACL 2025"
  - "Hallucination Detection"
  - "RAG Distillation"
  - "Knowledge Graph"
  - "Small Language Model"
  - "Hallucination Mitigation"
  - "Evidence Extraction"
date: 2026-05-08
content_hash: 24542d73cfc7f908
---

# DRAG: Distilling RAG for SLMs from LLMs to Transfer Knowledge and Mitigate Hallucination

**Conference**: ACL 2025  
**arXiv**: [2506.01954](https://arxiv.org/abs/2506.01954)  
**Code**: [https://github.com/VILA-Lab/DRAG](https://github.com/VILA-Lab/DRAG)  
**Area**: Hallucination Detection  
**Keywords**: RAG Distillation, Knowledge Graph, Small Language Model, Hallucination Mitigation, Evidence Extraction  

## TL;DR
DRAG proposes a framework to distill RAG capabilities from large language models (LLMs) to small language models (SLMs): utilizing an LLM (e.g., GPT-4o) to generate evidence and knowledge graph triples for a given question. After ranking and filtering, these are fed to SLMs (2B-9B) as structured contexts, boosting SLM performance on ARC-C by up to 27.7% without fine-tuning, while significantly mitigating hallucinations.

## Background & Motivation

**Background**: RAG enhances the factual accuracy of LLM generation by retrieving external knowledge, which has been widely adopted. However, existing RAG systems are primarily designed for large models. Maintaining large-scale knowledge bases is highly costly and difficult to deploy on small-scale models in resource-constrained environments.

**Limitations of Prior Work**: (a) SLMs exhibit poor performance when directly utilizing RAG due to their limited ability to understand and integrate retrieved documents; (b) retrieval results in traditional RAG may contain noise and irrelevant information, which can easily distract SLMs and lead to hallucinations; (c) existing distillation methods (e.g., LLMQuoter) require fine-tuning, which is inefficient.

**Key Challenge**: SLMs have limited context windows and comprehension capabilities, preventing them from directly processing large volumes of retrieved documents, yet they require external knowledge to mitigate hallucinations.

**Goal**: How to efficiently distill the RAG capability of LLMs to SLMs without fine-tuning.

**Key Insight**: Instead of prompting LLMs to directly provide answers, LLMs can generate structured "evidence" and "knowledge graphs" to serve as auxiliary contexts during the reasoning phase of SLMs. The LLM acts as the "retriever and knowledge extractor," while the SLM works as the "reasoner."

**Core Idea**: Utilizing LLMs to generate and rank evidence, construct knowledge graph triples, and inject the distilled structured knowledge into the prompts of SLMs, thereby enabling tuning-free RAG capability transfer.

## Method

### Overall Architecture

A four-stage pipeline: given a question q → (1) the LLM generates N pieces of evidence → (2) the evidence is filtered via semantic similarity and LLM ranking → (3) the LLM extracts knowledge graph triples (entity, entity, relation) from the filtered evidence → (4) the top-ranked evidence and/or graph triples are concatenated into the prompt and fed into the SLM to generate the final answer.

### Key Designs

1. **Evidence Generation**:

    - **Function**: Generating N textual evidences for each question using the LLM.
    - **Mechanism**: Designing prompts to instruct $\mathcal{M}_{large}$ to generate N fact fragments $\mathcal{D} = \{d_1, ..., d_N\}$ related to the question. A key insight is that "a well-trained LLM is a stronger knowledge source than traditional retrievers" since it has already internalized a massive amount of world knowledge.
    - **Design Motivation**: Avoiding the maintenance of an external document index by leveraging the internal knowledge of the LLM as the "retrieval" source, which is more friendly to SLMs.

2. **Evidence Ranking**:

    - **Function**: Filtering the most relevant evidence through a dual-scoring mechanism.
    - **Mechanism**: Computing a comprehensive score $s_i = \text{cos}(\mathbf{e}_i, \mathbf{q}) + \text{rank}_{LLM}(d_i)$ for each piece of evidence, where the former is the cosine similarity encoded by a sentence-transformer, and the latter is the intrinsic relevance ranking provided by the LLM. The top-K evidences are retained.
    - **Design Motivation**: Semantic similarity alone might be insufficient; LLM ranking captures deeper contextual relevance.

3. **Graph RAG Generation**:

    - **Function**: Converting evidence into structured entity-relation triples.
    - **Mechanism**: Prompting the LLM to extract $(a, b, r)$ triples from each filtered piece of evidence to construct a knowledge graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$. A graph aggregation strategy is also introduced to merge identical entity pairs into unified representations.
    - **Design Motivation**: Long evidence texts impose a heavy computational burden on SLMs. Although graph representation incurs some information loss, it is more concise, efficient, and preserves crucial structured relations.

4. **Privacy Protection Framework**:

    - **Function**: Local SLMs rewrite queries to remove private information before sending them to the cloud LLM.
    - **Mechanism**: The SLM rewrites the question (a simple task) → the cloud LLM generates evidence and graphs → the local SLM generates final answers utilizing the structured knowledge.
    - **Design Motivation**: Preventing original private user data from being uploaded to the cloud.

### Loss & Training
DRAG is a completely training-free framework, injecting knowledge distilled from the LLM into the reasoning process of the SLM solely through prompt engineering.

## Key Experimental Results

### Main Results

Teacher: GPT-4o; Student: various 2B-9B SLMs; Benchmarks: MedMCQA, MMLU, ARC-C

| Model | Method | MedMCQA Acc | MMLU Acc | ARC-C Acc |
|------|------|-------------|----------|-----------|
| Llama-3.2-3B | MiniRAG | 52.7 | 69.1 | 65.3 |
| Llama-3.2-3B | **DRAG** | **73.6 (+20.9)** | **74.4 (+5.3)** | **93.0 (+27.7)** |
| Gemma-2-2B | MiniRAG | 48.5 | 57.3 | 68.6 |
| Gemma-2-2B | **DRAG** | **72.4 (+23.9)** | **71.2 (+13.9)** | **91.5 (+22.9)** |
| Phi-3.5-mini | MiniRAG | 61.1 | 72.7 | 82.7 |
| Phi-3.5-mini | **DRAG** | **74.4 (+13.3)** | **77.8 (+5.1)** | **94.1 (+11.4)** |
| Llama-3.1-8B | SimRAG | - | 67.5 | 81.4 |
| Llama-3.1-8B | **DRAG** | **74.2** | **75.7 (+8.2)** | **93.1 (+11.7)** |

### Ablation Study

Impact of the number of evidence pieces K on Phi-3.5-mini on ARC-C:

| Configuration | K=5 | K=10 | K=15 | K=20 |
|------|-----|------|------|------|
| Graph Only | 91.69 | 92.76 | 93.48 | 93.30 |
| Evidence Only | 92.31 | 93.74 | **94.01** | 94.10 |
| Graph + Evidence | 92.40 | 93.12 | 93.74 | 93.74 |
| Original Model (No RAG)| 78.55 | - | - | - |

Teacher model comparison (Student: Qwen2.5-3B, ARC-C):

| Teacher | Acc |
|---------|-----|
| GPT-4o | **93.03** |
| Claude 3.5 Sonnet | 91.60 |
| DeepSeek V3 | 89.97 |
| Llama 3.3 70B | 87.23 |
| Gemini 1.5 Flash | 84.95 |

### Key Findings
- **~15 pieces of evidence is the optimal trade-off**: too few leads to insufficient knowledge, while too many introduce noise.
- **Evidence-only outperforms the graph + evidence combination**: the combined approach introduces redundancy and increases inference overhead without accuracy gains.
- **Stronger LLMs $\neq$ better distillation performance**: GPT-4o performs the best, but Gemini 1.5 Flash performs the worst, indicating that the structured quality of evidence is more critical than raw model capability.
- **Graphs, despite information loss, are highly effective**: the graph-only mode achieves 93.48% on ARC-C, close to the 94.01% of the evidence-only mode.

## Highlights & Insights
- **Tuning-free RAG Distillation**: Requires zero training on the SLM, substantially boosting performance solely by injecting structured knowledge via prompts. This paradigm can be generalized to any scenario requiring SLM deployment.
- **LLM as Retriever**: Replaces traditional external retrievers with the internal knowledge of the LLM, eliminating the cost of maintaining document indexes. In SLM scenarios, the abstract, clean evidence generated by LLMs is more effective than raw retrieved documents.
- **Added Value of the Privacy Protection Framework**: The pipeline (local query rewriting → cloud LLM knowledge generation → local answer generation) is naturally suited for enterprise deployment scenarios requiring privacy preservation.

## Limitations & Future Work
- Reliance on API calls to LLMs (e.g., GPT-4o) introduces cost and latency issues, making it unsuitable for real-time applications.
- The quality of evidence hinges entirely on the LLM's knowledge; any errors within the LLM itself will propagate as incorrect information.
- Comparison with more advanced RAG methodologies (e.g., Active RAG with dynamic retrieval, community-based approaches in GraphRAG) is lacking.
- The graph aggregation strategy is relatively simple; more sophisticated graph reasoning could potentially yield further improvements.
- The privacy-preserving evaluation benchmark under the zero-shot setting is small, requiring larger-scale validation for practical deployment.

## Related Work & Insights
- **vs MiniRAG (Fan et al., 2025)**: MiniRAG also targets SLM RAG but performs retrieval directly on the SLM. DRAG distills structured knowledge from the LLM to the SLM, outperforming it by a wide margin (+27.7% on ARC-C).
- **vs SimRAG (Xu et al., 2024)**: SimRAG is based on self-retrieval with an 8B model, whereas DRAG yields an 11.7% enhancement on the same 8B model after LLM distillation.
- **vs Self-RAG**: Self-RAG trains the model to self-determine when to retrieve, while DRAG realizes distillation directly through prompts without training.

## Rating
- Novelty: ⭐⭐⭐ The framework design is reasonable but components are standardized (LLM evidence generation + ranking + graph extraction); the core innovation lies in the combination scheme.
- Experimental Thoroughness: ⭐⭐⭐⭐ Incorporates multiple benchmarks, various teacher/student combinations, and comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The framework description is clear, and flowcharts are intuitive.
- Value: ⭐⭐⭐⭐ Tuning-free RAG distillation shows strong practicality in real-world deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Removal of Hallucination on Hallucination: Debate-Augmented RAG](removal_of_hallucination_on_hallucination_debate-augmented_rag.md)
- [\[ACL 2025\] Mixture of Decoding: An Attention-Inspired Adaptive Decoding Strategy to Mitigate Hallucination in Multimodal LLMs](mixture_of_decoding_an_attention-inspired_adaptive_decoding_strategy_to_mitigate.md)
- [\[ACL 2025\] On-Policy Self-Alignment with Fine-grained Knowledge Feedback for Hallucination Mitigation](on-policy_self-alignment_with_fine-grained_knowledge_feedback_for_hallucination_.md)
- [\[NeurIPS 2025\] Teaming LLMs to Detect and Mitigate Hallucinations](../../NeurIPS2025/hallucination/teaming_llms_to_detect_and_mitigate_hallucinations.md)
- [\[ACL 2025\] Alleviating Hallucinations from Knowledge Misalignment in Large Language Models via Selective Abstention Learning](alleviating_hallucinations_from_knowledge_misalignment_in_large_language_models_.md)

</div>

<!-- RELATED:END -->
