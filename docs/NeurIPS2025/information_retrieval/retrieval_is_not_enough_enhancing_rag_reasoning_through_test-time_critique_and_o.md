---
title: >-
  [Paper Note] Retrieval is Not Enough: Enhancing RAG Reasoning through Test-Time Critique and Optimization
description: >-
  [NeurIPS 2025][Information Retrieval & RAG][RAG] This paper proposes AlignRAG, a framework that reframes RAG as "retrieval-augmented reasoning" and trains a dedicated Critic Language Model (CLM) to iteratively critique a…
tags:
  - "NeurIPS 2025"
  - "Information Retrieval & RAG"
  - "RAG"
  - "reasoning misalignment"
  - "critique-driven alignment"
  - "test-time refinement"
  - "retrieval-augmented reasoning"
date: 2026-05-08
content_hash: ef576a1ae003d5d1
---

# Retrieval is Not Enough: Enhancing RAG Reasoning through Test-Time Critique and Optimization

**Conference**: NeurIPS 2025
**arXiv**: [2504.14858](https://arxiv.org/abs/2504.14858)
**Code**: [GitHub](https://github.com/AlignRAG/AlignRAG)
**Area**: Information Retrieval
**Keywords**: RAG, reasoning misalignment, critique-driven alignment, test-time refinement, retrieval-augmented reasoning

## TL;DR

This paper proposes AlignRAG, a framework that reframes RAG as "retrieval-augmented reasoning" and trains a dedicated Critic Language Model (CLM) to iteratively critique and refine the reasoning process at test time, addressing the misalignment between reasoning chains and retrieved evidence. An 8B CLM surpasses a 72B standard CLM on out-of-distribution tasks.

## Background & Motivation

- RAG has become the dominant paradigm for knowledge-augmented LLMs, yet standard RAG pipelines frequently fail to ensure that reasoning is consistent with retrieved evidence.
- The paper identifies a critical and underexplored failure mode: **Reasoning Misalignment**.
    - Even when relevant documents are successfully retrieved, the model's reasoning trajectory may still deviate from the constraints imposed by the evidence.
    - This is distinct from simple retrieval failure or factual error; it represents a **structural defect in the evidence integration process**.
- Existing approaches focus primarily on improving retrieval quality or generation robustness, but **neglect the explicit alignment between reasoning steps and evidence**.
- Reflective methods such as Self-RAG require architectural modifications or task-specific fine-tuning, limiting their generalizability.
- Self-critique approaches (e.g., Self-Refine) suffer from self-preference bias, with critique quality bounded by the model's own capabilities.

## Method

### Overall Architecture

The core idea of AlignRAG is to treat RAG reasoning as an optimizable artifact and achieve dynamic alignment between reasoning and evidence through a test-time critique-and-refinement loop.

**Three stages of reasoning misalignment**:

1. **Relevance Assessment Stage**: The model fails to accurately judge the relevance of document passages to the query.
2. **Query-Evidence Mapping Stage**: The model cannot correctly establish mappings from query elements to supporting evidence.
3. **Evidence Integration and Synthesis Stage**: Generated reasoning steps are not logically supported by the relevant evidence.

**System Architecture**:
- $\mathcal{M}_{\text{gen}}$: Generator (any RAG model)
- $\mathcal{M}_{\text{critic}}$: Trained Critic Language Model (CLM)
- Iterative alignment: $y_0 \xrightarrow{\text{CDA}} y_1 \xrightarrow{\text{CDA}} \cdots \xrightarrow{\text{CDA}} y_T$

### Key Designs

**1. Contrastive Critique Synthesis (CCS)**

CCS generates evidence-grounded critiques by contrasting reasoning trajectories from strong and weak models:

- A weak model (e.g., Qwen2.5-0.5B) generates $y_{\text{unexp}} \sim P_{\text{weak}}(y|q, \mathcal{D})$ (prone to misalignment)
- A strong model (e.g., LLaMA3.1-8B) generates $y_{\text{exp}} \sim P_{\text{strong}}(y|q, \mathcal{D})$ (better aligned)
- A preference-augmented input is constructed: $\mathcal{X}_{\text{pref}} = (q, \mathcal{D}, y_{\text{exp}}, y_{\text{unexp}})$
- Critique function: $\Delta y_{\text{unexp}} = \mathcal{F}(\mathcal{X}_{\text{pref}})$

Two advantages of the contrastive design:
1. Constrains the CLM to focus on differences grounded in $\mathcal{D}$ (rather than generic critique), promoting evidence sensitivity.
2. Enables fine-grained diagnosis of specific misalignment types by analyzing trajectory divergences.

**2. Structured Training Corpus Construction**

Multi-level contextual granularity is adopted:

$$\mathbf{c}_i = (r_i, h_i, m_i) \in \{0,1\}^3$$

- $r_i$ (relevance): whether a relevant document is included
- $h_i$ (helpfulness): whether the document contains an answer span
- $m_i$ (completeness): whether the document set supports a complete reasoning path

The system simulates diverse evidence configurations to expose the CLM to a wide variety of evidence environments.

**3. Critique Fine-Tuning (CFT)**

The training objective maximizes the likelihood of correct critiques:

$$\mathcal{L}_{\text{CFT}}(\theta) = -\sum_{\mathcal{C}_i \in \mathcal{C}} \log p_\theta(\Delta y_{\text{unexp}} | \mathcal{I}_{\text{critic}})$$

where $\mathcal{I}_{\text{critic}} = (q, \mathcal{D}, y_{\text{unexp}}, y_{\text{exp}})$. By decoupling critique generation from the target model's output, self-preference bias is avoided.

**4. Test-Time Critique-Driven Alignment (CDA)**

At each step, the CLM generates an edit signal $\Delta y_t$ identifying problems in $y_t$ and proposing $\mathcal{D}$-grounded corrections:

$$y_{t+1} = \mathcal{M}_{\text{gen}}(y_t \oplus \Delta y_t)$$

$\Delta y_t$ acts as a "pseudo-gradient" in discrete space, steering the generator toward evidence-aligned outputs.

### AlignRAG-auto: Dynamic Autonomous Alignment

The CLM is trained to predict [Good]/[Bad] control tokens:

$$p_\theta([\text{Good/Bad}], \Delta y | q, \mathcal{D}, y_{\text{unexp}})$$

At inference time, iteration terminates dynamically based on the predicted token:

$$y_{t+1} = \begin{cases} y_t & \text{if CLM outputs [Good]} \\ \mathcal{M}_{\text{gen}}(y_t \oplus \Delta y_t) & \text{if CLM outputs [Bad]} \end{cases}$$

No manual specification of iteration count is required; only responses that need correction undergo multiple refinement rounds.

## Key Experimental Results

### Main Results: 5 In-Domain QA Benchmarks

| Method | NQ | MultiHop | TriviaQA | PopQA | ASQA | Avg |
|---|---|---|---|---|---|---|
| CoT (Qwen-7B, no retrieval) | 33.9 | 45.0 | 58.3 | 26.9 | 20.5 | 36.9 |
| Vanilla RAG (Qwen-7B) | 60.2 | 44.7 | 73.2 | 63.7 | 42.8 | 56.9 |
| InstructRAG (Qwen-7B) | 63.8 | 46.3 | 76.1 | 67.5 | 47.5 | 60.2 |
| Self-Refine (Qwen-7B) | 61.6 | 45.0 | 74.4 | 65.5 | 45.2 | 58.3 |
| **AlignRAG (Qwen-7B + CLM 8B)** | **65.9** | **49.5** | **77.8** | **68.4** | **48.9** | **62.1** |
| Self-Refine (Qwen-14B) | 65.1 | 46.1 | 78.0 | 67.0 | 47.3 | 60.7 |
| **AlignRAG (Qwen-14B + CLM 8B)** | **67.7** | **49.8** | **79.5** | **68.4** | **48.6** | **62.8** |

AlignRAG achieves the best results across all backbone models and benchmarks. An 8B CLM paired with a 7B generator surpasses the 14B Self-Refine baseline.

### AlignRAG-auto vs. AlignRAG-fixed

| Dataset | Fixed (8B) | Auto (8B) | Fixed (14B) | Auto (14B) |
|---|---|---|---|---|
| PopQA | 66.5 | 67.6 | 68.4 | 68.3 |
| TriviaQA | 77.0 | 77.6 | 79.5 | 79.9 |
| NQ | 65.3 | **66.8** | 67.7 | **69.0** |
| ASQA | 47.1 | **48.8** | 48.6 | **49.8** |
| Avg | 60.6 | **61.4** | 62.8 | **63.4** |

**The Auto variant requires no manual tuning and achieves performance on par with or slightly better than the Fixed variant**, while saving computational cost by eliminating unnecessary iterations.

### OOD Generalization

- The 8B CLM outperforms the Self-Refine baseline by **12.1%** on OOD tasks.
- The 8B AlignRAG CLM surpasses a standard 72B CLM by **2.2%**.
- When applied as a plug-in to InstructRAG, AlignRAG improves the OOD accuracy of Qwen2.5-14B by **9.4%**.

### Robustness to Noisy Retrieval

Strong robustness is maintained under both informative and noisy retrieval conditions—"when RAG retrieval fails, AlignRAG becomes even more prominent."

### Key Findings

1. All retrieval-augmented methods outperform retrieval-free CoT, confirming the importance of external knowledge.
2. Training-time refinement (InstructRAG) yields further gains over standard RAG.
3. AlignRAG's improvements are especially pronounced on multi-hop reasoning tasks requiring multi-step evidence integration.
4. Contrastive critique synthesis is critical—an 8B specialized CLM proves more effective than a 72B general-purpose CLM.
5. The dynamic stopping mechanism significantly reduces computation without sacrificing accuracy.

## Highlights & Insights

1. **Problem Redefinition**: Reframing RAG from "retrieval-augmented generation" to "retrieval-augmented reasoning" represents a profound shift in perspective.
2. **Reasoning Misalignment Taxonomy**: The three-stage decomposition (relevance assessment, query mapping, evidence synthesis) provides a structured framework for diagnosing RAG failures.
3. **Elegant Contrastive Critique Design**: Contrasting strong and weak models avoids self-preference bias and enables a small CLM to achieve outsized performance.
4. **Plug-and-Play**: No modification to the RAG architecture is required; the framework can enhance any RAG system as an external module.
5. **Practical Auto Variant**: Dynamic stopping eliminates manual hyperparameter tuning, a critical feature for production deployment.
6. **Small Model, Large Impact**: An 8B CLM outperforming a 72B standard CLM demonstrates that targeted training yields far greater returns than brute-force scaling.

## Limitations & Future Work

- Training the CLM introduces additional training cost and synthetic data generation overhead.
- Iterative refinement increases inference latency, making the approach unsuitable for real-time, low-latency scenarios.
- Contrastive critique synthesis depends on the choice of strong and weak models—an excessively large capability gap may degrade critique quality.
- The training corpus is relatively small at 10K samples (2K per dataset).
- The use of Qwen2.5-0.5B as the weak model may produce responses that are too poor to reflect realistic failure patterns.
- The binary [Good]/[Bad] judgment may be overly coarse-grained; additional control levels could be beneficial.
- There is no in-depth analysis of which types of reasoning misalignment are most prevalent or most difficult to correct.

## Related Work & Insights

- **vs. Self-RAG**: Self-RAG performs self-evaluation using special tokens but requires architectural modifications; AlignRAG uses an external CLM and is plug-and-play.
- **vs. Self-Refine**: Self-Refine's use of the model's own critique introduces self-preference bias; AlignRAG overcomes this through contrastive learning.
- The contrastive critique paradigm is generalizable to other feedback-driven settings (e.g., code generation, mathematical reasoning).
- Critique-driven alignment shares conceptual similarities with reward models in RLHF, but operates on reasoning chains rather than final outputs.

## Rating

- Novelty: ⭐⭐⭐⭐ The reasoning misalignment concept is novel and the contrastive critique synthesis is creative, though the iterative refinement framework has precedents.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 7 datasets, 3 model families, multiple baselines, OOD, and noisy retrieval settings.
- Writing Quality: ⭐⭐⭐⭐ Problem definition is clear and the framework is described systematically, though some notation is slightly redundant.
- Value: ⭐⭐⭐⭐⭐ Makes an important contribution to the reliability of RAG systems; the plug-and-play nature yields high practical deployment value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SymRTLO: Enhancing RTL Code Optimization with LLMs and Neuron-Inspired Symbolic Reasoning](symrtlo_enhancing_rtl_code_optimization_with_llms_and_neuron-inspired_symbolic_r.md)
- [\[NeurIPS 2025\] SeCon-RAG: A Two-Stage Semantic Filtering and Conflict-Free Framework for Trustworthy RAG](secon-rag_a_two-stage_semantic_filtering_and_conflict-free_framework_for_trustwo.md)
- [\[NeurIPS 2025\] Learning Task-Agnostic Representations through Multi-Teacher Distillation](learning_task-agnostic_representations_through_multi-teacher_distillation.md)
- [\[AAAI 2026\] Towards Inference-Time Scaling for Continuous Space Reasoning](../../AAAI2026/information_retrieval/towards_inference-time_scaling_for_continuous_space_reasoning.md)
- [\[NeurIPS 2025\] Think Straight, Stop Smart: Structured Reasoning for Efficient Multi-Hop RAG](think_straight_stop_smart_structured_reasoning_for_efficient_multi-hop_rag.md)

</div>

<!-- RELATED:END -->
