---
title: >-
  [Paper Note] WikiSeeker: Rethinking the Role of Vision-Language Models in Knowledge-Based Visual Question Answering
description: >-
  [ACL 2026][Multimodal VLM][Knowledge-based VQA] WikiSeeker is proposed to redefine the role of VLMs in multimodal RAG—transitioning from simple answer generators to two specialized agents: a Refiner trained with RL for q…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Knowledge-based VQA"
  - "Multimodal RAG"
  - "Query Rewriting"
  - "Reinforcement Learning"
  - "Retrieval-Augmented Generation"
date: 2026-05-08
content_hash: c40e834c4d754279
---

# WikiSeeker: Rethinking the Role of Vision-Language Models in Knowledge-Based Visual Question Answering

**Conference**: ACL 2026  
**arXiv**: [2604.05818](https://arxiv.org/abs/2604.05818)  
**Code**: [https://github.com/zhuyjan/WikiSeeker](https://github.com/zhuyjan/WikiSeeker)  
**Area**: Multimodal VLM  
**Keywords**: Knowledge-based VQA, Multimodal RAG, Query Rewriting, Reinforcement Learning, Retrieval-Augmented Generation

## TL;DR

WikiSeeker is proposed to redefine the role of VLMs in multimodal RAG—transitioning from simple answer generators to two specialized agents: a Refiner trained with RL for query rewriting and an Inspector to verify the reliability of retrieved context. This approach achieves SOTA performance on EVQA, InfoSeek, and M2KR benchmarks.

## Background & Motivation

**Background**: Multimodal Retrieval-Augmented Generation (RAG) is the dominant paradigm for Knowledge-Based Visual Question Answering (KB-VQA). It involves retrieving relevant documents from an external knowledge base, concatenating them with the input query, and feeding them into a generative model to produce an answer.

**Limitations of Prior Work**: (1) **Visual-only Retrieval**: Most methods use only the query image as the retrieval key, ignoring semantic information in the text query, which leads to poor retrieval when visual content is ambiguous. (2) **VLM Role Misalignment**: VLMs are typically used only as final answer generators. However, experiments show that VLMs are less effective than text-only LLMs at extracting answers from retrieved context—image tokens often act as noise rather than useful signals during the extraction phase.

**Key Challenge**: The visual understanding capability of VLMs is valuable during retrieval and verification (understanding entities in the image, judging if results match), but becomes a burden during answer extraction (visual tokens interfere with reading comprehension).

**Goal**: Redesign the role of VLMs in multimodal RAG to fully utilize their visual understanding for improving retrieval and verification, while delegating answer extraction to text-only LLMs which excel at the task.

**Key Insight**: Authors found through experiments that when the ratio of correct information in the retrieved context increases, the VQA performance of text-only LLMs actually exceeds that of VLMs with image inputs (e.g., at Ratio=1.0, Qwen achieves 93.45% vs. QwenVL(I+T) at 88.46%).

**Core Idea**: Reposition the VLM as a Refiner (rewriting queries with visual cues to enhance retrieval) and an Inspector (verifying context reliability and routing decisions), while leaving answer generation to a text-only LLM.

## Method

### Overall Architecture

WikiSeeker consists of three stages: (1) **Retrieval**: The VLM Refiner expands the original question, and a multimodal retriever (visual + text embedding concatenation) retrieves candidate documents from the knowledge base. (2) **Rerank**: A multimodal reranker filters the most relevant passages. (3) **Generation**: The VLM Inspector evaluates if the retrieved context is sufficient—if it passes, the query is routed to a text-only LLM; if it fails, the VLM answers directly using its internal knowledge.

### Key Designs

1. **VLM as Refiner (RL-based Query Rewriting)**:
    - **Function**: Utilizes visual cues to rewrite and expand short user queries into more informative retrieval queries.
    - **Mechanism**: Qwen2.5-VL-3B-Instruct is used as the Refiner, trained via Reinforcement Learning using Group Relative Policy Optimization (GRPO). The model generates CoT reasoning (inside `<think>` tags) before outputting the rewritten query (inside `<answer>` tags). The reward function consists of: (1) Format Reward—checking XML tag adherence; (2) Retrieval Reward—discrete rewards based on the hit rank of correct entities (e.g., +4 for top-5, decreasing rewards within top-200, and -2.5 for a miss).
    - **Design Motivation**: KB-VQA user queries are often short and abstract, making direct retrieval noisy. RL training allows the Refiner to autonomously discover optimal rewriting strategies without expensive human-annotated pairs.

2. **Multimodal Dense Retrieval (Weighted Concatenation Strategy)**:
    - **Function**: Utilizes both visual and textual information for retrieval.
    - **Mechanism**: The knowledge base is constructed as <image, passage> pairs. EVA-CLIP-8B encodes visual features and Qwen3-Embedding-0.6B encodes text, which are concatenated into a unified vector. Retrieval uses weighted concatenation: $\mathbf{v}_q = \text{Concat}[\alpha \cdot \Phi_{vis}(I_q), (1-\alpha) \cdot \Phi_{text}(T_q)]$. A hyperparameter $\alpha$ controls the relative importance of modalities.
    - **Design Motivation**: Visual-only retrieval ignores text semantics; the concatenation strategy allows both modalities to participate, while $\alpha$ provides flexible balance control.

3. **VLM as Inspector (Decoupled Generation Strategy)**:
    - **Function**: Verifies the reliability of retrieved context and dynamically routes answer generation.
    - **Mechanism**: The Inspector (VLM) receives the image, question, and reranked passages to output a judgment $s \in \{\text{PASS}, \text{FAIL}\}$ and an internal knowledge answer $A_{internal}$. For PASS, the rewritten query and context are sent to a text-only LLM (e.g., LLaMA/Qwen); for FAIL, the VLM's internal answer is used.
    - **Design Motivation**: Evidence suggests VLMs are inferior to text-only LLMs at utilizing context due to visual noise, but their visual capability makes them excellent at judging if retrieval matches the image. Decoupling allows each component to perform its best task.

### Loss & Training

The Refiner is trained using GRPO with a total reward $r_i = r_{retrieval}(o_i) + r_{format}(o_i)$. Retrieval rewards are based on discrete mapping of hit ranks (top-5: +4, top-200: +0.1, miss: -2.5). Format rewards check XML tag correctness (+1/-4). Training sets involve 7,000 samples per benchmark, stratified by hit rank.

## Key Experimental Results

### Main Results

Retrieval results (R@1) on EVQA and InfoSeek:

| Method | EVQA R@1 | EVQA R@20 | InfoSeek R@1 | InfoSeek R@20 |
|---|---|---|---|---|
| EchoSight | 36.5 | 48.8 | 53.2 | 77.9 |
| OMGM | 42.8 | 58.7 | 64.0 | 84.8 |
| WikiSeeker (w/o Refiner) | 28.0 | 43.4 | 53.5 | 78.5 |
| WikiSeeker (w. Refiner) | **44.1** | **62.3** | **67.0** | **87.7** |

The Refiner boosts EVQA R@1 from 28.0 to 44.1 (+57.5%), outperforming all baselines.

### Ablation Study

| Configuration | Key Metric | Description |
|---|---|---|
| w/o Refiner | R@1 28.0 (EVQA) | Basic multimodal retrieval |
| w. Refiner | R@1 44.1 (EVQA) | Query rewriting significantly boosts retrieval |
| VLM Gen vs LLM Gen | 88.46% vs 93.45% (Ratio=1.0) | LLM is superior with reliable context |
| w/o Inspector | Decrease | LLM is misled by unreliable context |

### Key Findings

- VLMs are indeed inferior to text-only LLMs during the answer extraction phase: as the ratio of correct information in the context increases (Ratio=0.3→1.0), the LLM advantage becomes more pronounced.
- RL-trained Refiner significantly outperforms SFT: RL allows the model to automatically learn how to rewrite queries to maximize retrieval hit rates.
- The Inspector's routing strategy is critical in unreliable retrieval scenarios—the VLM's internal knowledge compensates for retrieval failures via the FAIL path.
- SOTA results were also achieved on the M2KR multi-task benchmark, proving the generalizability of the method.

## Highlights & Insights

- The empirical finding that **"VLMs are inferior to LLMs for answer extraction"** is significant and counter-intuitive—it stems from visual tokens becoming noise once the correct textual context is available. This suggests an "assign the right task to the right model" design principle for RAG.
- Training query rewriting with **RL** is an elegant self-supervised solution—using retrieval hit rank as a reward signal eliminates the need for human-annotated pairs. GRPO's group-based relative advantage estimation avoids the overhead of training a critic model.
- The Inspector's dual-path design achieves an elegant fusion of retrieval-augmented and parametric knowledge—it dynamically chooses between them based on reliability rather than simply "always retrieving" or "always using internal knowledge."

## Limitations & Future Work

- The Inspector's PASS/FAIL judgment is a hard decision, which may lead to misclassifications in marginal cases.
- Use of a smaller VLM (3B) for the Refiner; larger models might produce even better query rewrites.
- Knowledge base construction depends on LLM summarization of long passages; summary quality affects retrieval.
- Validation was limited to encyclopedic KB-VQA; the effectiveness on commonsense reasoning VQA remains unknown.

## Related Work & Insights

- **vs EchoSight/OMGM**: These models use VLMs for answer generation and visual-only retrieval. WikiSeeker repositions the VLM as Refiner+Inspector, delegates generation to an LLM, and upgrades retrieval to multimodal. It exceeds OMGM by 1.3 percentage points on EVQA R@1.
- **vs ReflectiVA**: ReflectiVA introduces a reflection mechanism to decide if external knowledge is needed but still uses a VLM for generation. WikiSeeker's decoupling strategy more fundamentally addresses the noise issue of VLMs during answer extraction.

## Rating

- Novelty: ⭐⭐⭐⭐ Insights on VLM role repositioning are valuable; the RL-trained Refiner is elegant, though the overall framework cleverly combines existing techniques.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive experiments across three benchmarks, multiple ablations, and systematic VLM vs. LLM comparisons.
- Writing Quality: ⭐⭐⭐⭐ Motivation and methods are clearly described; the design of experiments in Table 2 is persuasive.
- Value: ⭐⭐⭐⭐ Provides direct guidance for the role design of VLMs in multimodal RAG systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] ReasonVQA: A Multi-hop Reasoning Benchmark with Structural Knowledge for Visual Question Answering](../../ICCV2025/multimodal_vlm/reasonvqa_a_multi-hop_reasoning_benchmark_with_structural_knowledge_for_visual_q.md)
- [\[ICLR 2026\] Meta-Adaptive Prompt Distillation for Few-Shot Visual Question Answering](../../ICLR2026/multimodal_vlm/meta-adaptive_prompt_distillation_for_few-shot_visual_question_answering.md)
- [\[ACL 2026\] Benchmarking Deflection and Hallucination in Large Vision-Language Models](benchmarking_deflection_and_hallucination_in_large_vision-language_models.md)
- [\[AAAI 2026\] MacVQA: Adaptive Memory Allocation and Global Noise Filtering for Continual Visual Question Answering](../../AAAI2026/multimodal_vlm/macvqa_adaptive_memory_allocation_and_global_noise_filtering_for_continual_visua.md)
- [\[NeurIPS 2025\] Are Vision Language Models Ready for Clinical Diagnosis? A 3D Medical Benchmark for Tumor-centric Visual Question Answering](../../NeurIPS2025/multimodal_vlm/are_vision_language_models_ready_for_clinical_diagnosis_a_3d_medical_benchmark_f.md)

</div>

<!-- RELATED:END -->
