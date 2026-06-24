---
title: >-
  [Paper Note] Micro-Act: Mitigate Knowledge Conflict in QA via Actionable Self-Reasoning
description: >-
  [ACL 2025][Information Retrieval & RAG][RAG] This paper proposes the Micro-Act framework, which introduces a hierarchical action space (navigational, functional, and bridging actions) and adaptive granularity decomposition. It enables LLMs to automatically perceive context complexity and disassemble knowledge contrast layer by layer. Micro-Act outperforms state-of-the-art (SOTA) methods across 5 knowledge-conflict benchmarks while maintaining robustness in conflict-free scena…
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "RAG"
  - "knowledge conflict"
  - "hierarchical action space"
  - "adaptive decomposition"
  - "reasoning enhancement"
date: 2026-05-08
content_hash: 63a74ecfc4fb9c9d
---

# Micro-Act: Mitigate Knowledge Conflict in QA via Actionable Self-Reasoning

**Conference**: ACL 2025  
**arXiv**: [2506.05278](https://arxiv.org/abs/2506.05278)  
**Code**: [https://github.com/Nan-Huo/Micro-Act](https://github.com/Nan-Huo/Micro-Act)  
**Area**: Other  
**Keywords**: RAG, knowledge conflict, hierarchical action space, adaptive decomposition, reasoning enhancement  

## TL;DR
This paper proposes the Micro-Act framework, which introduces a hierarchical action space (navigational, functional, and bridging actions) and adaptive granularity decomposition. It enables LLMs to automatically perceive context complexity and disassemble knowledge contrast layer by layer. Micro-Act outperforms state-of-the-art (SOTA) methods across 5 knowledge-conflict benchmarks while maintaining robustness in conflict-free scenarios.

## Background & Motivation
**Background**: RAG systems enhance LLM answer quality by retrieving external knowledge, but the retrieved information may contain noise, outdated content, or misinformation, generating conflicts with the parametric knowledge of LLMs (knowledge conflict).

**Limitations of Prior Work**: Existing methods fall into two categories: (a) general methods (e.g., CoT) that only reason on the retrieved context, failing to handle implicit conflicts; and (b) methods that first prompt LLMs to generate parametric knowledge and then conduct a side-by-side comparison with retrieved knowledge (e.g., GKP). However, this coarse-grained side-by-side comparison is susceptible to interference from redundant context, and stronger models suffer from "over-rationalization"—trying to justify both contradictory sides simultaneously.

**Key Challenge**: Knowledge conflicts can be hidden beneath shallow semantics (such as temporal conflicts—"in 2010" vs "recently", or semantic conflicts—synonymous representations in different contexts). Coarse-grained comparisons fail to detect these fine-grained contradictions.

**Goal**: How can LLMs adaptively adjust comparison granularity and decompose complexity step-by-step until hidden knowledge conflicts are detected?

**Key Insight**: Knowledge conflict detection can be modeled as a hierarchical action execution process under the ReAct framework, where the core DECOMPOSE action recursively disassembles complex comparisons into finer-grained sub-comparisons.

**Core Idea**: Through the adaptive decomposition action (DECOMPOSE) in a hierarchical action space, Micro-Act, like a "microscope", progressively magnifies the granularity of knowledge comparison until the LLM can confidently detect and resolve conflicts.

## Method

### Overall Architecture
The input consists of the user query $q$ and a set of retrieved evidence, and the output is the QA answer after knowledge conflict detection and resolution. The procedure is as follows:
1. ELICIT: Prompt the LLM to generate its parametric knowledge $K^p(q)$
2. RETRIEVE: Obtain retrieved knowledge $K^r(\mathcal{E})$
3. Iteratively execute the Thought $\rightarrow$ Action $\rightarrow$ Observation loop to detect and resolve conflicts via REASON/ASSERT/DECOMPOSE actions
4. Generate the final answer based on the complete history

### Key Designs

1. **Hierarchical Action Space**:

    - **Navigational Actions**: ELICIT (extracting parametric knowledge from the LLM) and REASON (generating a reasoning path for a piece of knowledge)—responsible for "exploration and preparation"
    - **Functional Actions**: ASSERT—performing consistency checks on two pieces of knowledge, outputting $\delta \in \{0, 1\}$ to represent whether a conflict exists
    - **Bridging Actions**: DECOMPOSE—when ASSERT detects a conflict but the context is too complex, it splits the current comparison into multiple finer-grained sub-comparisons
    - Design Motivation: The three classes of actions perform their respective duties, with DECOMPOSE being the core innovation that allows the system to operate across different granularity levels

2. **Adaptive Granularity Adjustment**:

    - Function: Automatically decides if decomposition is necessary based on input complexity
    - Mechanism: Defines a complexity score $\mathcal{C}_t$, where each DECOMPOSE ensures $\mathcal{C}_{t+1} < \mathcal{C}_t$ (since the context becomes shorter and the semantic scope narrows). Decomposition stops when $\mathcal{C}_t \leq \tau$ (a threshold where the LLM can process it confidently)
    - Model-level Adaptation: GPT-4o-mini, being weaker, automatically triggers more DECOMPOSE actions, whereas GPT-4o, with stronger capabilities, requires fewer decompositions. This adaptation is automatic and does not need manual tuning

3. **ReAct-based Reasoning Agent**:

    - Function: At each step $t$, the LLM first generates Thought $T_t$, then selects Action $A_t$, and receives Observation $O_t$ upon execution
    - Mechanism: $T_t \sim \mathcal{M}_\Theta(T_t | H_{t-1})$, $A_t \sim \mathcal{M}_\Theta(A_t | H_{t-1}, T_t)$, history $H_t = H_{t-1} \cup \{T_t, A_t, O_t\}$
    - Design Motivation: The ReAct framework provides a structured thought-action-observation loop, which naturally accommodates the hierarchical action space

4. **Preventing Infinite Decomposition**:

    - Theoretical Guarantee: Each DECOMPOSE strictly reduces complexity $\mathcal{C}_{t+1} < \mathcal{C}_t$, ensuring termination in finite steps
    - Practical measure: Sets a maximum round budget $N$ as a hard constraint

### Loss & Training
- Pure zero-shot prompting without any fine-tuning or task-specific customization
- Temperature = 0, top-p = 1, maximum generated tokens = 512
- Extra overhead: Compared to the GKP baseline, it only introduces approximately 2.8x input tokens and 1.3x output tokens

## Key Experimental Results

### Main Results

| Method | GPT-4o (ConflictBank) | GPT-4o (KRE) | GPT-4o-mini (CB) | GPT-4o-mini (KRE) | LLaMA-8B (CB) | LLaMA-8B (KRE) |
|------|----------------------|--------------|------------------|-------------------|---------------|----------------|
| CoT | 6.43 | 44.35 | 3.00 | 36.50 | 2.13 | 24.50 |
| GKP (Prev. SOTA) | 15.40 | 55.30 | 17.53 | 44.45 | 6.83 | 32.75 |
| **Ours** | **22.30** (+6.9) | **59.50** (+4.2) | **26.93** (+9.4) | **51.10** (+6.7) | **18.30** (+11.5) | **46.60** (+13.9) |

### Ablation Study

| Configuration | Misinformation | Temporal Conflict | Semantic Conflict | Description |
|------|--------|----------|----------|------|
| Micro-Act (Full) | 26.1 | 27.9 | 24.9 | Full Model |
| w/o Navigational Actions | 18.4 (-7.7)| 18.5 (-9.4) | 15.7 (-9.2) | Unable to extract parametric knowledge |
| w/o Functional Actions | 13.8 (-12.3)| 15.2 (-12.7)| 13.3 (-11.6)| Unable to detect conflicts |
| **w/o DECOMPOSE** | **4.2 (-21.9)**| **4.5 (-23.4)**| **0.8 (-24.1)** | Core component; performance collapses without it |

### Key Findings
- **DECOMPOSE is Key**: Disabling it causes a performance drop of over 20%, proving that adaptive granularity adjustment is the critical innovation.
- **Largest Gains in Temporal and Semantic Conflicts**: These two conflict types are typically hidden beneath the surface, which is precisely where Micro-Act excels.
- **Robustness in Conflict-free Scenarios**: Compared to end-to-end baselines, it incurs less than a 2% loss in accuracy, whereas other conflict resolution methods significantly degrade conflict-free performance.
- **"Over-rationalization" Phenomenon**: Stronger models (GPT-4o) are more prone to over-rationalization—attempting to argue that both conflicting sides are correct—than weaker models.
- **Cross-model Adaptability**: Despite its smaller scale, LLaMA-8B compensates for capability limitations by utilizing more decomposition actions to maintain robust performance.

## Highlights & Insights
- **"Microscopic" progressive zoom-in design philosophy** is highly elegant—instead of a one-size-fits-all comparison, it progressively increases resolution as needed. This approach can be transferred to any scenario requiring multi-granularity reasoning.
- **Discovered the "Over-rationalization" phenomenon**: Stronger LLMs are ironically more prone to concluding that "both sides are correct" when faced with contradictions, serving as an important heads-up for RAG system design.
- **Zero-shot + cross-LLM adaptation**: No fine-tuning or manual adjustment is needed; LLMs of varying capabilities automatically adjust their decomposition strategies, demonstrating high practical utility.

## Limitations & Future Work
- Extra intermediate reasoning steps introduce computational overhead (approx. 2.8x input tokens), which might limit applicability in latency-sensitive scenarios.
- Evaluation suggests scope is confined to English context; the effectiveness of the decomposition strategy in multilingual scenarios remains unexplored.
- The stopping criterion for DECOMPOSE depends on the LLM's self-judgment, and theoretical guarantees require a strict complexity reduction assumption.

## Related Work & Insights
- **vs GKP (Liu et al. 2022)**: GKP generates parametric knowledge first and conducts side-by-side comparisons with a fixed granularity, whereas Micro-Act adaptively adjusts granularity via DECOMPOSE.
- **vs ReAct (Yao et al. 2023)**: Micro-Act extends the action space of ReAct by introducing hierarchical actions dedicated to knowledge conflict resolution.
- **vs Self-Ask**: Self-Ask decomposes queries into sub-questions but does not decompose the granularity of knowledge comparisons.

## Rating
- Novelty: ⭐⭐⭐⭐ The design of hierarchical action space + adaptive decomposition is novel, and the discovery of "over-rationalization" is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 5 benchmarks, 4 models, 3 conflict types, alongside ablation, robustness, and case studies.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with tight integration of theoretical analysis and experiments.
- Value: ⭐⭐⭐⭐ Solves key pain points of RAG systems with high zero-shot practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] FaithfulRAG: Fact-Level Conflict Modeling for Context-Faithful Retrieval-Augmented Generation](faithfulrag_fact_level_conflict.md)
- [\[ACL 2025\] SeaKR: Self-aware Knowledge Retrieval for Adaptive Retrieval Augmented Generation](seakr_self-aware_knowledge_retrieval_for_adaptive_retrieval_augmented_generation.md)
- [\[ACL 2025\] ARise: Towards Knowledge-Augmented Reasoning via Risk-Adaptive Search](arise_risk_adaptive_search.md)
- [\[ACL 2025\] Mitigating Lost-in-Retrieval Problems in RAG Multi-Hop QA](mitigating_lost-in-retrieval_problems_in_retrieval_augmented_multi-hop_question_.md)
- [\[ACL 2025\] SGIC: A Self-Guided Iterative Calibration Framework for RAG](sgic_a_self-guided_iterative_calibration_framework_for_rag.md)

</div>

<!-- RELATED:END -->
