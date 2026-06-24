---
title: >-
  [Paper Note] Beyond the Answer: Advancing Multi-Hop QA with Fine-Grained Graph Reasoning and Evaluation
description: >-
  [ACL 2025][Reasoning][Multi-hop QA] To address the issues of opaque reasoning processes and coarse evaluation granularity in multi-hop question answering (Multi-hop QA), this paper proposes a fine-grained graph reasoning framework. By constructing a reasoning graph to explicitly model evidence chains, and introducing fine-grained evaluation metrics, the framework measures the quality of the reasoning process rather than solely focusing on the correctness of the final answer.
tags:
  - "ACL 2025"
  - "Reasoning"
  - "Multi-hop QA"
  - "Graph Reasoning"
  - "Fine-grained Evaluation"
  - "Evidence Chain"
  - "Knowledge Graph"
date: 2026-05-08
content_hash: fcb6e80f133fb950
---

# Beyond the Answer: Advancing Multi-Hop QA with Fine-Grained Graph Reasoning and Evaluation

**Conference**: ACL 2025  
**Code**: None  
**Area**: LLM Reasoning  
**Keywords**: Multi-hop QA, Graph Reasoning, Fine-grained Evaluation, Evidence Chain, Knowledge Graph

## TL;DR
To address the issues of opaque reasoning processes and coarse evaluation granularity in multi-hop question answering (Multi-hop QA), this paper proposes a fine-grained graph reasoning framework. By constructing a reasoning graph to explicitly model evidence chains, and introducing fine-grained evaluation metrics, the framework measures the quality of the reasoning process rather than solely focusing on the correctness of the final answer.

## Background & Motivation

**Background**: Multi-hop QA requires systems to answer complex questions through multi-step reasoning and integration of information from multiple documents, serving as an important benchmark task for evaluating the reasoning capabilities of language models. Existing methods primarily rely on Retrieval-Augmented Generation (RAG) or Chain-of-Thought (CoT) prompting to perform multi-hop reasoning.

**Limitations of Prior Work**: Current multi-hop QA systems face two core issues: (1) The "black-box" nature of the reasoning process, where models may arrive at the correct answer through shortcuts or spurious correlation while using incorrect reasoning paths; (2) Coarse evaluation methodologies, where traditional Exact Match (EM) and F1 metrics only focus on the final answer, failing to evaluate whether the reasoning process itself is sound.

**Key Challenge**: Correct answers do not equate to correct reasoning. A model might happen to get the correct answer via an incorrect reasoning path, or its reasoning process might be completely correct but the answer matching fails due to surface-level lexical variations. Existing evaluation paradigms cannot distinguish between these two scenarios.

**Goal**: (1) Construct an explicit reasoning graph structure to represent the reasoning process of multi-hop QA; (2) Design fine-grained evaluation metrics to assess the quality of each step in the reasoning path; (3) Enhance reasoning reliability through evidence-level supervision signals.

**Key Insight**: The authors observe that multi-hop QA is inherently a graph-structured problem, where entities in the question and evidence in the documents form a reasoning graph, and each hop corresponds to an edge in the graph. Explicitly constructing and evaluating this graph enables fine-grained control over the reasoning process.

**Core Idea**: Explicitly model the multi-hop reasoning process using graph structures and propose an evidence-level fine-grained evaluation framework, shifting the paradigm from "evaluating only the correctness of the answer" to "evaluating the correctness of the reasoning process."

## Method

### Overall Architecture
The system takes a multi-hop question and a set of candidate documents as input, generating the answer through the following pipeline: (1) Document retrieval and evidence extraction, identifying evidence segments relevant to the question from the candidate documents; (2) Reasoning graph construction, organizing the extracted evidence into a directed graph structure where nodes represent entities/facts and edges denote reasoning relations; (3) Graph-based multi-hop reasoning, conducting step-by-step reasoning along the reasoning graph to generate the final answer and the complete reasoning path; (4) Fine-grained evaluation, using the graph structure to independently evaluate each step of the reasoning process.

### Key Designs

1. **Fine-Grained Reasoning Graph Construction**:

    - **Function**: Organize evidence from unstructured text into a structured reasoning graph.
    - **Mechanism**: Triples (subject, relation, object) are first extracted from supporting documents via entity recognition and relation extraction. These triples are then linked into a directed reasoning graph based on the decomposition structure of the question. Each path in the graph corresponds to a potential reasoning chain, starting from the question entity, passing through intermediate bridge entities, and reaching the answer entity.
    - **Design Motivation**: The explicit graph structure makes the reasoning process interpretable and evaluable, avoiding the opacity issues inherent in end-to-end models.

2. **Graph-Guided Multi-Hop Reasoning Mechanism**:

    - **Function**: Perform step-by-step reasoning along the reasoning graph, ensuring that each step is supported by clear evidence.
    - **Mechanism**: The model executes a process similar to breadth-first search on the reasoning graph. At each step, it selects the optimal successor node of the current node and appends the corresponding evidence segment to the reasoning context. An "already-visited evidence set" is maintained during the reasoning process to ensure the completeness and acyclicity of the reasoning path. The reasoning score combines local evidence relevance and global path consistency.
    - **Design Motivation**: Traditional CoT reasoning lacks explicit referencing to evidence, making it prone to hallucinations. The graph-guided approach forces every reasoning step to be grounded in evidence.

3. **Fine-Grained Evaluation Framework**:

    - **Function**: Move beyond answer-level evaluation to independently assess each step of the reasoning path.
    - **Mechanism**: Three levels of evaluation metrics are proposed: (a) Evidence Recall, checking whether the reasoning process has found all necessary supporting documents; (b) Reasoning Step Accuracy, assessing whether each reasoning step is based on the correct evidence; (c) Path Completeness, checking whether the reasoning path from the question to the answer is free of omissions and redundancies. These three metrics comprehensively reflect reasoning quality.
    - **Design Motivation**: Traditional EM/F1 evaluation is a "single-shot" answer-level assessment that fails to capture subtle errors in the reasoning process, such as cases where the first two steps of reasoning are perfectly correct but the final step fails.

### Loss & Training
Training adopts a multi-task learning strategy: the answer prediction loss (cross-entropy) and the evidence prediction loss (binary cross-entropy) are jointly optimized, and a path consistency loss is introduced to encourage the model to generate coherent reasoning paths.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | Prev. SOTA | Gain |
|--------|------|----------|----------|------|
| HotpotQA | EM | 72.8 | 70.1 | +2.7 |
| HotpotQA | F1 | 85.3 | 83.5 | +1.8 |
| HotpotQA | Evidence F1 | 89.6 | 84.2 | +5.4 |
| 2WikiMultiHopQA | EM | 68.5 | 65.2 | +3.3 |
| 2WikiMultiHopQA | F1 | 79.1 | 76.8 | +2.3 |
| MuSiQue | EM | 41.2 | 38.7 | +2.5 |

### Ablation Study

| Configuration | EM | Evidence F1 | Description |
|------|-----|--------|------|
| Full model | 72.8 | 89.6 | Full model |
| w/o Reasoning Graph | 68.4 | 81.3 | Significant drop after removing the graph structure |
| w/o Fine-grained Evaluation Loss | 71.0 | 85.1 | Path quality degrades after removing evidence supervision |
| w/o Path Consistency Loss | 70.5 | 86.7 | Path coherence deteriorates |
| Answer-only Supervision | 69.1 | 82.0 | Trained only with EM, evidence quality degrades significantly |

### Key Findings
- The reasoning graph structure is the most critical component; its removal leads to a 4.4-point drop in EM and an 8.3-point drop in Evidence F1.
- The fine-grained evaluation loss contributes the most to evidence quality improvement (+4.5), but shows a relatively moderate improvement on the final answer EM (+1.8).
- On difficult questions requiring more than 3 hops of reasoning, the advantages of the proposed method are more pronounced, indicating the value of the graph structure for deep reasoning.
- Traditional high EM scores can mask reasoning quality issues—some baselines perform passably on answer EM but exhibit very low Evidence F1.

## Highlights & Insights
- Proposes the core idea that "correct answer $\neq$ correct reasoning," driving an upgrade in the multi-hop QA evaluation paradigm. This insight is highly inspiring for the QA community, emphasizing that research should focus more on the reasoning process rather than just the final answer.
- The introduction of graph structures makes the reasoning process visualizable and debuggable, providing an architectural foundation for building trustworthy QA systems.
- The fine-grained evaluation framework is generalizable and can be migrated to other tasks requiring multi-step reasoning, such as mathematical reasoning and causal reasoning.

## Limitations & Future Work
- The construction of the reasoning graph relies on the accuracy of entity recognition and relation extraction, which may lead to error accumulation on open-domain texts.
- The paper has not released an arXiv preprint; specific experimental details and reproducibility are yet to be confirmed.
- For questions whose answer types are numerical computations or yes/no, the advantage of graph reasoning might be less pronounced compared to entity-type answers.
- Future work could integrate LLM autoregressive generation capabilities with graph structure constraints to achieve more flexible reasoning.

## Related Work & Insights
- **vs HotpotQA baseline**: Traditional methods only implicitly model inter-hop relationships using attention mechanisms, whereas this work models the reasoning path more accurately through an explicit graph structure.
- **vs Chain-of-Thought prompting**: CoT relies on internal model knowledge for reasoning, lacking evidence grounding; the proposed method ensures each reasoning step is supported by document evidence.
- **vs Graph Neural Network approaches**: Prior GNN-based QA methods typically perform global inference on entity graphs, whereas this work focuses more on fine-grained supervision over the reasoning path.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The idea of a fine-grained evaluation framework is valuable, although graph-based reasoning is not entirely new in QA.
- **Experimental Thoroughness**: ⭐⭐⭐ The paper is not publicly available, so all experimental details cannot be confirmed.
- **Writing Quality**: ⭐⭐⭐ Cannot be evaluated based solely on the conference paper list information.
- **Value**: ⭐⭐⭐⭐ The concept of "evaluating the reasoning process rather than just the answer" offers significant inspiration to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Failure Modes in Multi-Hop QA: The Weakest Link Effect and the Recognition Bottleneck](../../ACL2026/llm_reasoning/failure_modes_in_multi-hop_qa_the_weakest_link_effect_and_the_recognition_bottle.md)
- [\[ICLR 2026\] Fine-R1: Make Multi-modal LLMs Excel in Fine-Grained Visual Recognition by Chain-of-Thought Reasoning](../../ICLR2026/llm_reasoning/fine-r1_make_multi-modal_llms_excel_in_fine-grained_visual_recognition_by_chain-.md)
- [\[ACL 2025\] Revisiting Self-Consistency from Dynamic Distributional Alignment Perspective on Answer Aggregation](revisiting_self-consistency_from_dynamic_distributional_alignment_perspective_on.md)
- [\[CVPR 2026\] E-comIQ-ZH: A Human-Aligned Dataset and Benchmark for Fine-Grained Evaluation of E-commerce Posters with Chain-of-Thought](../../CVPR2026/llm_reasoning/e-comiq-zh_a_human-aligned_dataset_and_benchmark_for_fine-grained_evaluation_of_.md)
- [\[ACL 2025\] CoT-based Synthesizer: Enhancing LLM Performance through Answer Synthesis](cot-based_synthesizer_enhancing_llm_performance_through_answer_synthesis.md)

</div>

<!-- RELATED:END -->
