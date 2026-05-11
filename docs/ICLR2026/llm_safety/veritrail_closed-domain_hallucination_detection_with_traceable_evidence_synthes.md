---
title: >-
  [Paper Note] VeriTrail: Closed-Domain Hallucination Detection with Traceability
description: >-
  [ICLR2026][LLM Safety][hallucination detection] This paper proposes VeriTrail — the first closed-domain hallucination detection method that provides traceability for multi-generative-step (MGS) processes. It models the generation process as a DAG and performs layer-by-layer verification along paths, while also introducing the first MGS datasets that include all intermediate outputs with human annotations.
tags:
  - ICLR2026
  - LLM Safety
  - hallucination detection
  - faithfulness evaluation
  - traceability
  - multi-generative-step
  - DAG
date: 2026-05-08
content_hash: 6f20b5c0b2670988
---

# VeriTrail: Closed-Domain Hallucination Detection with Traceability

**Conference**: ICLR2026  
**arXiv**: [2505.21786](https://arxiv.org/abs/2505.21786)  
**Code**: [Dataset](https://aka.ms/veritrail-datasets)  
**Area**: LLM Safety  
**Keywords**: hallucination detection, faithfulness evaluation, traceability, multi-generative-step, DAG

## TL;DR
This paper proposes VeriTrail — the first closed-domain hallucination detection method that provides traceability for multi-generative-step (MGS) processes. It models the generation process as a DAG and performs layer-by-layer verification along paths, while also introducing the first MGS datasets that include all intermediate outputs with human annotations.

## Background & Motivation
- LLMs frequently generate unsupported content even when instructed to follow source materials — a phenomenon termed "closed-domain hallucination."
- Generation processes fall into two categories:
    - **Single-Generative-Step (SGS)**: e.g., standard RAG, where a single LLM call produces the final output.
    - **Multi-Generative-Step (MGS)**: e.g., hierarchical summarization and GraphRAG, where intermediate outputs serve as inputs to subsequent steps.
- MGS is more prone to hallucinations: errors can be introduced and propagated at each step.
- **Core Argument**: For MGS, detecting hallucinations only in the final output is insufficient. Two additional capabilities are required:
    - **Provenance**: understanding how an output is derived from source material.
    - **Error Localization**: identifying at which step a hallucination was introduced.
- Existing methods evaluate only the relationship between the final output and source material, without leveraging intermediate outputs, and thus cannot provide traceability.

## Core Contributions
1. A unified conceptual framework for generation processes (DAG representation).
2. VeriTrail: the first closed-domain hallucination detection method providing traceability for both MGS and SGS.
3. FABLES+ and DiverseSumm+: the first MGS datasets containing all intermediate outputs with human annotations.

## Method

### Conceptual Framework: DAG Representation of Generation Processes

The generation process is modeled as a directed acyclic graph $G = (V, E)$:
- **Nodes** $v \in V$: text segments (source documents / intermediate outputs / final output).
- **Directed edges** $(u, v) \in E$: $u$ is used as input for generating $v$.
- **Root nodes** $V_0$: source documents (no incoming edges).
- **Terminal node** $v^*$: final output (no outgoing edges).
- **Stage function** $\text{stage}: V \to \mathbb{N}$: reflects a node's position in the generation process.

### VeriTrail Detection Pipeline

Input: (1) a completed generation process DAG; (2) termination parameter $q$; (3) a set of factual claims $C$ extracted from $v^*$.

Each claim $c \in C$ is processed independently through the following steps:

#### Step 1: Sub-claim Decomposition
- The Claimify Decomposition module is applied to split compound claims into independently verifiable sub-claims.
- Example: "Company X acquired two startups in 2020 as part of its healthcare expansion" → (1) X acquired two startups in 2020; (2) the acquisition was part of a healthcare expansion.
- Decomposition is applied recursively, with a maximum of 20 iterations to avoid infinite loops.

#### Step 2: Evidence Selection
- Starting from the source nodes $\text{src}(v^*)$ of the terminal node.
- Sentences are segmented using NLTK and assigned unique IDs.
- An LLM selects sentences that support or contradict the claim and its sub-claims (returning sentence IDs).
- If the context window is exceeded, the input is split into multiple parallel prompts.
- **ID validation guarantee**: non-matching IDs are discarded to ensure evidence is not hallucinated.

#### Step 3: Verdict Generation
- If no sentences are selected → "Not Fully Supported."
- Otherwise, the LLM assigns one of three verdicts based on the evidence:
    - **Fully Supported**: the source text strongly implies the entire claim.
    - **Not Fully Supported**: at least part of the claim is not supported by the source text.
    - **Inconclusive**: the source text is ambiguous or contradictory.

**Context Handling**: Rather than using selected sentences directly (which may be ambiguous out of context):
- Root nodes: full content is included.
- Non-root nodes: summaries generated during the evidence selection step are used.

#### Step 4: Candidate Node Selection and Iterative Termination
Candidate nodes for the next verification round are selected based on the latest verdict:

| Latest Verdict | Candidate Node Selection Strategy |
|---|---|
| Fully Supported / Inconclusive | Source nodes of nodes with evidence in the current round |
| Not Fully Supported | Source nodes of all verified nodes in the current round (broader, to prevent missed detections) |

Termination conditions (any one sufficient):
1. Candidate nodes consist only of already-verified root nodes with evidence → adopt the latest verdict.
2. No candidate nodes (root nodes not reached, or root nodes have no evidence) → Not Fully Supported.
3. Not Fully Supported for $q$ consecutive iterations → Not Fully Supported.

### Traceability Output
For each claim, the method returns:
- **Final verdict** + LLM reasoning.
- **All intermediate verdicts**.
- **Evidence chain**: selected sentences (with node IDs) + evidence summaries from each round.

#### Provenance
- For Fully Supported claims: the evidence chain records the path from intermediate nodes to root nodes.

#### Error Localization
- The last iteration $n$ yielding a Fully Supported verdict is identified.
- The stage of non-root nodes with evidence in that iteration is designated as the error stage.
- $\{\text{stage}(v) | v \in V_e(n), v \notin V_0\}$

## Dataset Construction

### FABLES+ (Hierarchical Summarization)
- Based on the FABLES book summarization dataset.
- Hierarchical summaries were regenerated for 22 books (average 118K tokens), with all intermediate outputs retained.
- 734 claims were extracted; 48% reused original annotations, with the remainder annotated manually.

### DiverseSumm+ (GraphRAG)
- Based on the DiverseSumm news dataset.
- 148 stories, 1,479 articles, totaling 1.19M tokens.
- 20 questions were sampled, and answers were generated using GraphRAG.
- 560 claims were extracted and annotated by 4 Upwork annotators and 1 author.
- 87% of claims could be assessed from associated articles; 13% required consulting additional articles.

## Experimental Results

### Baselines

| Category | Method | Long-Document Strategy |
|---|---|---|
| NLI | INFUSE | Bidirectional entailment ranking |
| NLI | AlignScore | 350-token chunking |
| NLI | Bespoke-MiniCheck-7B | 32K-token chunking |
| RAG | Top-k Retrieval | Embedding retrieval + verdict |
| Direct Verification | Gemini 1.5 Pro / GPT-4.1 Mini | Long-context LM |

### Hard Prediction Results (Macro F1 / Balanced Accuracy)

| Method | FABLES+ F1 | FABLES+ Bal.Acc | DiverseSumm+ F1 | DiverseSumm+ Bal.Acc |
|---|---|---|---|---|
| **VeriTrail (q=3)** | **84.5** | **83.6** | **79.5** | 76.3 |
| **VeriTrail (q=1)** | 74.0 | **84.6** | 76.6 | **83.0** |
| RAG (k=15) | 69.6 | 76.5 | 75.1 | 74.0 |
| Bespoke-MiniCheck-7B | 62.2 | 69.0 | 72.1 | 69.4 |
| Gemini 1.5 Pro | 61.1 | 60.8 | 49.8 | 57.6 |
| GPT-4.1 Mini | 60.7 | 58.2 | 62.9 | 61.5 |
| AlignScore | 59.6 | 67.5 | 60.4 | 62.7 |
| INFUSE | 40.5 | 59.5 | 20.0 | 50.1 |

**Key Findings**:
- VeriTrail outperforms all baselines on both datasets (q=3 achieves the best F1; q=1 achieves the best Balanced Accuracy).
- Direct long-context verification (Gemini 1.5 Pro) underperforms, likely due to difficulty retrieving relevant information from extremely long documents.
- Classical NLI methods such as AlignScore and INFUSE show notably degraded performance on long documents.

### Trade-off of the $q$ Parameter
- q=1 (terminate after one NFS): high NFS recall (89.8%), low NFS precision (55.1%).
- q=3 (terminate after three NFS): more balanced (NFS precision 84.5%, recall 55.9%).
- Larger $q$ yields more thorough verification but produces more conservative NFS verdicts.

## Strengths and Limitations

### Strengths
- The first hallucination detection method providing traceability (provenance + error localization).
- The DAG framework unifies the representation of both SGS and MGS processes.
- Sentence-level evidence selection with ID validation guarantees that evidence is not hallucinated.
- Outperforms strong baselines on extremely long documents (>100K tokens).
- Cost-effective (analyzed in Appendix D).

### Limitations
- Relies on LLMs for evidence selection and verdict generation, subject to LLM capability constraints.
- Error localization cannot always determine the exact stage in certain scenarios.
- Dataset scale is limited (734 + 560 claims).
- Evaluation is restricted to the gpt-4o model.

## Personal Evaluation and Reflections

### Novelty ⭐⭐⭐⭐⭐
- The paradigm shift from "detection" to "detection + traceability" is highly valuable.
- DAG-based modeling of generation processes represents a fundamental rethinking of hallucination detection.
- The iterative evidence selection and candidate node propagation mechanism is elegantly designed.

### Practical Value ⭐⭐⭐⭐⭐
- Directly addresses real-world demands of MGS pipelines (e.g., GraphRAG, hierarchical summarization).
- Error localization is extremely valuable for system debugging and improvement.
- Sentence-level evidence chains significantly reduce the cost of manual auditing.

### Dataset Contribution ⭐⭐⭐⭐
- FABLES+ and DiverseSumm+ fill a critical gap in MGS hallucination detection data.
- The inclusion of complete intermediate outputs is a key innovation.
- However, dataset scale remains limited.

### Experimental Design ⭐⭐⭐⭐
- Comprehensive baseline coverage (NLI, RAG, long-context LM).
- Dual evaluation via hard and soft predictions.
- Ablation analysis and error case studies (appendix) enhance credibility.

### Overall Rating ⭐⭐⭐⭐⭐
A pioneering work that advances closed-domain hallucination detection from "judging correctness" to "tracing origins and localizing errors." The DAG framework elegantly unifies diverse generation processes, and VeriTrail's iterative verification mechanism demonstrates strong performance on extremely long documents. For increasingly complex MGS pipelines such as GraphRAG, this traceable approach to hallucination detection offers substantial practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Enhancing Hallucination Detection through Noise Injection](enhancing_hallucination_detection_through_noise_injection.md)
- [\[ACL 2026\] Enhancing Hallucination Detection via Future Context](../../ACL2026/llm_safety/enhancing_hallucination_detection_via_future_context.md)
- [\[ICLR 2026\] SABRE-FL: Selective and Accurate Backdoor Rejection for Federated Prompt Learning](sabre-fl_selective_and_accurate_backdoor_rejection_for_federated_prompt_learning.md)
- [\[ICLR 2026\] Understanding Sensitivity of Differential Attention through the Lens of Adversarial Robustness](understanding_sensitivity_of_differential_attention_through_the_lens_of_adversar.md)
- [\[ICLR 2026\] Resource-Adaptive Federated Text Generation with Differential Privacy](resource-adaptive_federated_text_generation_with_differential_privacy.md)

</div>

<!-- RELATED:END -->
