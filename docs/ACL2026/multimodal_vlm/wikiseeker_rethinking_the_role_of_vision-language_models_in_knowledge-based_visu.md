---
title: >-
  [Paper Note] WikiSeeker: Rethinking the Role of Vision-Language Models in Knowledge-Based Visual Question Answering
description: >-
  [ACL 2026][Multimodal VLM][Knowledge-based VQA] This paper proposes WikiSeeker, which redefines the role of VLMs in multimodal RAG—transforming them from mere answer generators into two specialized agents: a Refiner (tra…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Knowledge-based VQA"
  - "Multimodal RAG"
  - "Query Rewriting"
  - "Reinforcement Learning"
  - "Retrieval-Augmented Generation"
date: 2026-05-08
content_hash: cb8d816fb8a88f4f
---

# WikiSeeker: Rethinking the Role of Vision-Language Models in Knowledge-Based Visual Question Answering

**Conference**: ACL 2026
**arXiv**: [2604.05818](https://arxiv.org/abs/2604.05818)  
**Code**: [https://github.com/zhuyjan/WikiSeeker](https://github.com/zhuyjan/WikiSeeker)  
**Area**: Multimodal VLM
**Keywords**: Knowledge-based VQA, Multimodal RAG, Query Rewriting, Reinforcement Learning, Retrieval-Augmented Generation

## TL;DR

This paper proposes WikiSeeker, which redefines the role of VLMs in multimodal RAG—transforming them from mere answer generators into two specialized agents: a Refiner (trained with RL to rewrite queries) and an Inspector (to verify the reliability of retrieved contexts). WikiSeeker achieves state-of-the-art performance on three benchmarks: EVQA, InfoSeek, and M2KR.

## Background & Motivation

**Background**: Multimodal retrieval-augmented generation (RAG) is the dominant paradigm for knowledge-based visual question answering (KB-VQA), where relevant documents are retrieved from an external knowledge base and concatenated with the input query before being fed into a generative model to produce answers.

**Limitations of Prior Work**: (1) **Vision-only retrieval**: Most existing methods use only the query image as the retrieval key, ignoring semantic information in the user's textual query, which degrades retrieval quality when visual content is ambiguous. (2) **Misaligned VLM role**: VLMs are typically used only as the final answer generator; however, experiments show that VLMs are actually inferior to text-only LLMs for extracting answers from retrieved contexts—image tokens tend to act as noise rather than useful signals during answer extraction.

**Key Challenge**: VLMs' visual understanding capability is valuable during retrieval and verification (e.g., identifying entities in images, judging whether retrieval results match the image), but becomes a burden during answer extraction, where visual tokens interfere with text-based reading comprehension.

**Goal**: To redesign the role of VLMs in multimodal RAG, leveraging their visual understanding strengths to improve retrieval and verification, while delegating answer extraction to text-only LLMs that are better suited for this task.

**Key Insight**: Through controlled experiments, the authors find that as the proportion of correct information in the retrieved context increases, text-only LLM performance on VQA surpasses that of VLMs with image input (e.g., at Ratio=1.0, Qwen achieves 93.45% vs. QwenVL(I+T) at 88.46%).

**Core Idea**: Reposition VLMs as a Refiner (rewriting queries using visual cues to improve retrieval) and an Inspector (verifying retrieved context reliability and routing decisions), while delegating answer generation to a text-only LLM.

## Method

### Overall Architecture

WikiSeeker consists of three stages: (1) **Retrieval**: The VLM Refiner expands the original question, and a multimodal retriever (concatenated visual and text embeddings) retrieves candidate documents from the knowledge base. (2) **Reranking**: A multimodal reranker selects the most relevant passages. (3) **Generation**: The VLM Inspector assesses whether the retrieved context is sufficient—if it passes, the rewritten query and retrieved context are routed to a text-only LLM for answer generation; otherwise, the VLM answers directly using its internal knowledge.

### Key Designs

1. **VLM as Refiner (RL-based Query Rewriting)**:

    - **Function**: Uses visual cues to rewrite and expand the user's short query, generating more informative retrieval queries.
    - **Mechanism**: Employs Qwen2.5-VL-3B-Instruct as the Refiner, trained via GRPO (Group Relative Policy Optimization). The model first generates chain-of-thought reasoning (within `<think>` tags) and then outputs the rewritten query (within `<answer>` tags). The reward function comprises two components: (1) a format reward checking whether the output conforms to the XML schema; and (2) a retrieval reward that performs retrieval with the rewritten query and assigns a discrete reward based on the rank of the correct entity hit (top-5: +4, decaying within top-200, miss: −2.5).
    - **Design Motivation**: User queries in KB-VQA are typically short and abstract, making them noisy retrieval keys. RL training enables the Refiner to autonomously discover optimal query rewriting strategies without costly human-annotated query pairs.

2. **Multimodal Dense Retrieval (Weighted Concatenation Strategy)**:

    - **Function**: Leverages both visual and textual information simultaneously for retrieval.
    - **Mechanism**: The knowledge base is constructed as $\langle$image, passage$\rangle$ pairs. EVA-CLIP-8B encodes visual content and Qwen3-Embedding-0.6B encodes text, which are concatenated into a unified vector. Retrieval uses a weighted concatenation: $\mathbf{v}_q = \text{Concat}[\alpha \cdot \Phi_{vis}(I_q), (1-\alpha) \cdot \Phi_{text}(T_q)]$, where the hyperparameter $\alpha$ controls the relative importance of visual and textual features.
    - **Design Motivation**: Vision-only retrieval ignores textual semantics. The concatenation strategy enables both modalities to participate in retrieval, while $\alpha$ provides flexible modality balance control.

3. **VLM as Inspector (Decoupled Generation Strategy)**:

    - **Function**: Verifies the reliability of retrieved contexts and dynamically routes answer generation.
    - **Mechanism**: The Inspector (VLM) receives the image, question, and reranked passages, and outputs a judgment $s \in \{\text{PASS}, \text{FAIL}\}$ along with an internal knowledge answer $A_{internal}$. On PASS, the rewritten query and retrieved context are sent to a text-only LLM (e.g., LLaMA/Qwen) for answer generation; on FAIL, the VLM's internal knowledge answer is used.
    - **Design Motivation**: Experiments demonstrate that VLMs underperform text-only LLMs when generating answers from retrieved contexts (visual tokens act as noise), yet VLMs' visual understanding capability makes them well-suited for judging whether retrieval results are consistent with the image. The decoupled strategy allows each component to perform the task it does best.

### Loss & Training

The Refiner is trained with GRPO. The total reward is $r_i = r_{retrieval}(o_i) + r_{format}(o_i)$. The retrieval reward is a discrete mapping based on hit rank (top-5: +4, top-200: +0.1, miss: −2.5), and the format reward checks XML tag correctness (+1/−4). Training uses 7,000 samples per benchmark, with stratified sampling by hit rank.

## Key Experimental Results

### Main Results

Retrieval results (R@1) on EVQA and InfoSeek:

| Method | EVQA R@1 | EVQA R@20 | InfoSeek R@1 | InfoSeek R@20 |
|--------|----------|-----------|--------------|---------------|
| EchoSight | 36.5 | 48.8 | 53.2 | 77.9 |
| OMGM | 42.8 | 58.7 | 64.0 | 84.8 |
| WikiSeeker (w/o Refiner) | 28.0 | 43.4 | 53.5 | 78.5 |
| WikiSeeker (w. Refiner) | **44.1** | **62.3** | **67.0** | **87.7** |

The Refiner improves EVQA R@1 from 28.0 to 44.1 (+57.5%), surpassing all baselines.

### Ablation Study

| Configuration | Key Metric | Note |
|---------------|-----------|------|
| w/o Refiner | R@1 28.0 (EVQA) | Baseline multimodal retrieval |
| w. Refiner | R@1 44.1 (EVQA) | Query rewriting substantially improves retrieval |
| VLM generation vs. LLM generation | 88.46% vs. 93.45% (Ratio=1.0) | LLM is superior when reliable context is available |
| w/o Inspector | Performance drop | Without Inspector, LLM is misled by unreliable contexts |

### Key Findings

- VLMs are indeed inferior to text-only LLMs for answer generation: as the proportion of correct information in retrieved contexts increases (Ratio=0.3→1.0), the LLM advantage becomes increasingly pronounced.
- The RL-trained Refiner substantially outperforms SFT: RL enables the model to autonomously learn how to rewrite queries to maximize retrieval hit rate.
- The Inspector's routing strategy is especially critical in unreliable retrieval scenarios—the VLM's internal knowledge compensates for retrieval failures on the FAIL path.
- State-of-the-art results on the M2KR multi-task benchmark further confirm the generalizability of the approach.

## Highlights & Insights

- The empirical finding that **"VLMs underperform LLMs in answer extraction"** is both important and counterintuitive—visual tokens become noise once correct textual context has already been retrieved. This suggests that RAG systems should assign the right model to the right task.
- **Using RL to train query rewriting** is an elegant self-supervised solution—retrieval hit rank serves as the reward signal, eliminating the need for human-annotated query rewriting pairs. GRPO's group-relative advantage estimation avoids the additional cost of training a critic model.
- The Inspector's dual-path design achieves an elegant fusion of retrieval augmentation and parametric knowledge—rather than always relying on retrieval or always relying on internal knowledge, the system dynamically selects the appropriate source based on reliability.

## Limitations & Future Work

- The Inspector's PASS/FAIL judgment is a hard decision, which may produce errors in borderline cases.
- The Refiner uses a relatively small VLM (3B parameters); larger models may produce higher-quality query rewrites.
- Knowledge base construction relies on LLM-generated summaries of long passages, and summary quality affects retrieval effectiveness.
- Validation is limited to encyclopedic KB-VQA; effectiveness on commonsense reasoning VQA remains unexplored.

## Related Work & Insights

- **vs. EchoSight/OMGM**: These methods use VLMs for answer generation with vision-only retrieval. WikiSeeker repositions VLMs as Refiner and Inspector, delegates answer generation to a text-only LLM, and upgrades retrieval to multimodal. WikiSeeker surpasses OMGM by 1.3 percentage points on EVQA R@1.
- **vs. ReflectiVA**: ReflectiVA introduces a reflection mechanism to determine whether external knowledge is needed but still uses a VLM for answer generation. WikiSeeker's decoupled strategy more fundamentally addresses the noise problem introduced by VLMs during answer extraction.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The insight of repositioning VLMs is valuable, and the RL-based Refiner training scheme is elegant, though the overall framework is a clever combination of existing techniques.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three benchmarks, multiple ablations, and a systematic VLM-vs-LLM comparison provide comprehensive empirical support.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation and method descriptions are clear; the experimental design in Table 2 is particularly persuasive.
- **Value**: ⭐⭐⭐⭐ Provides direct guidance for VLM role design in multimodal RAG systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Benchmarking Deflection and Hallucination in Large Vision-Language Models](benchmarking_deflection_and_hallucination_in_large_vision-language_models.md)
- [\[ICCV 2025\] ReasonVQA: A Multi-hop Reasoning Benchmark with Structural Knowledge for Visual Question Answering](../../ICCV2025/multimodal_vlm/reasonvqa_a_multi-hop_reasoning_benchmark_with_structural_knowledge_for_visual_q.md)
- [\[ACL 2026\] Rethinking Jailbreak Detection of Large Vision Language Models with Representational Contrastive Scoring](rethinking_jailbreak_detection_of_large_vision_language_models_with_representati.md)
- [\[ICLR 2026\] Meta-Adaptive Prompt Distillation for Few-Shot Visual Question Answering](../../ICLR2026/multimodal_vlm/meta-adaptive_prompt_distillation_for_few-shot_visual_question_answering.md)
- [\[AAAI 2026\] MacVQA: Adaptive Memory Allocation and Global Noise Filtering for Continual Visual Question Answering](../../AAAI2026/multimodal_vlm/macvqa_adaptive_memory_allocation_and_global_noise_filtering_for_continual_visua.md)

</div>

<!-- RELATED:END -->
