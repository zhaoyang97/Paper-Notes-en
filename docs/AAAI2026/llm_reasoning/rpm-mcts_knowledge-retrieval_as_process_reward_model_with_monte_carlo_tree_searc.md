---
title: >-
  [Paper Note] RPM-MCTS: Knowledge-Retrieval as Process Reward Model with Monte Carlo Tree Search for Code Generation
description: >-
  [AAAI 2026][LLM Reasoning][Process Reward Model] This paper proposes RPM-MCTS, which replaces a trained Process Reward Model (PRM) with knowledge base retrieval to guide MCTS search for code generation. Exploiting the homogeneity of correct implementations within the same algorithm family, the method retrieves reference algorithm steps from a knowledge base as evaluation signals, applies similarity-based filtering to prune redundant expansion nodes, and uses sandbox execution to localize errors—achieving approximately 15% token reduction while surpassing prior state-of-the-art.
tags:
  - AAAI 2026
  - LLM Reasoning
  - Process Reward Model
  - MCTS
  - Knowledge Base Retrieval
  - Code Generation
  - Algorithm Step Evaluation
date: 2026-05-08
content_hash: f5c5d2c013dab636
---

# RPM-MCTS: Knowledge-Retrieval as Process Reward Model with Monte Carlo Tree Search for Code Generation

**Conference**: AAAI 2026
**arXiv**: [2511.19895](https://arxiv.org/abs/2511.19895)
**Code**: Available
**Area**: LLM Reasoning / Code Generation
**Keywords**: Process Reward Model, MCTS, Knowledge Base Retrieval, Code Generation, Algorithm Step Evaluation

## TL;DR
This paper proposes RPM-MCTS, which replaces a trained Process Reward Model (PRM) with knowledge base retrieval to guide MCTS search for code generation. Exploiting the homogeneity of correct implementations within the same algorithm family, the method retrieves reference algorithm steps from a knowledge base as evaluation signals, applies similarity-based filtering to prune redundant expansion nodes, and uses sandbox execution to localize errors—achieving approximately 15% token reduction while surpassing prior state-of-the-art.

## Background & Motivation

**Background**: Tree search (MCTS) has demonstrated effectiveness for code generation by enabling multi-step search to produce higher-quality algorithmic code. However, the central bottleneck lies in evaluating the quality of intermediate algorithm steps.

**Limitations of Prior Work**:
- Training a PRM requires large quantities of step-level annotated data, incurring extremely high collection and labeling costs.
- Existing methods cannot effectively localize and promptly correct erroneous steps—significant search budget may already be wasted by the time an error is detected.
- Large numbers of redundant nodes arise during search, as different expansions yield semantically duplicate candidates.

**Key Challenge**: Evaluating intermediate algorithm step quality is necessary, yet training a PRM is prohibitively expensive—motivating a training-free alternative.

**Goal**: Replace trained PRMs with knowledge base retrieval while simultaneously addressing search redundancy and error localization.

**Key Insight**: Correct implementations within the same algorithm family exhibit *homogeneity*—the correct steps of, say, sorting algorithms are structurally similar across different problem instances. A knowledge base of correct implementations can therefore serve as a reference for evaluating the quality of the current step.

**Core Idea**: Correct algorithm steps in the knowledge base serve as training-free process reward signals, enabling more efficient MCTS search.

## Method

### Overall Architecture
The framework comprises three components: (1) **knowledge base construction** (collecting correct implementations across algorithm categories); (2) **RPM-MCTS search** (retrieving from the knowledge base at each expansion step to evaluate step quality, applying similarity-based filtering to remove redundancy, and using sandbox execution to localize errors); and (3) **optional data distillation** (fine-tuning the base model with high-quality code produced by RPM-MCTS).

### Key Designs

1. **Knowledge Base as Process Reward Model**

   - **Function**: Replaces a trained PRM with retrieval-based signals.
   - **Mechanism**: The algorithm step generated at each expansion is matched against correct steps in the knowledge base via similarity scoring; high similarity indicates high step quality, prompting MCTS to prioritize that branch.
   - **Design Motivation**: Correct implementations of the same algorithm type are highly similar in structure (e.g., every correct quicksort contains a partition step)—the method leverages this homogeneity directly.

2. **Similarity-Based Redundancy Filtering**

   - **Function**: Removes semantically duplicate candidate nodes during the MCTS expansion phase.
   - **Mechanism**: Newly generated expansion nodes are compared against existing nodes; those exceeding a similarity threshold are treated as redundant and discarded.
   - **Design Motivation**: Repeated sampling from LLMs frequently yields code that is semantically identical but syntactically varied—retaining such redundancy wastes the search budget.

3. **Sandbox Execution with Error Reflection**

   - **Function**: Executes code during the simulation phase to detect errors and provide targeted feedback.
   - **Mechanism**: Candidate code from the MCTS simulation phase is executed against test cases in a sandbox; upon failure, error information is extracted and fed back to the LLM for directed correction.
   - **Design Motivation**: Code generation uniquely permits actual execution for verification—enabling not only error detection but also precise localization of the faulty step.

### Loss & Training
- No training is required during the search phase—the method operates entirely at inference time.
- Optional distillation: code produced by RPM-MCTS can be used for supervised fine-tuning (SFT) to improve the base model.

## Key Experimental Results

### Main Results (Pass@1)

| Method | APPS-Intro | APPS-Interv | APPS-Comp | CodeContests | HumanEval+ | MBPP+ | Avg. |
|--------|-----------|------------|----------|-------------|-----------|-------|------|
| Qwen3-8B Baseline | 56.7 | 35.3 | 29.3 | 10.7 | 75.6 | 72.2 | 52.1 |
| Qwen3-8B + ToT | 69.3 | 54.0 | 41.3 | 17.3 | 82.3 | 70.4 | 59.0 |
| **Qwen3-8B + RPM-MCTS** | **77.3** | **60.0** | **43.6** | **23.0** | **83.5** | **76.2** | **64.0** |
| Qwen3-235B Baseline | 78.0 | 54.7 | 42.7 | 24.0 | 86.0 | 78.8 | 64.6 |
| **Qwen3-235B + RPM-MCTS** | **86.7** | **67.3** | **59.3** | **36.7** | **87.8** | **81.2** | **72.3** |

- RPM-MCTS improves Qwen3-8B by an average of +11.9 points (52.1→64.0), reaching the level of the Qwen3-235B baseline.
- Qwen3-235B + RPM-MCTS achieves the largest gains on competition-level tasks (CodeContests: +12.7).

### Ablation Study

| Configuration | Avg. Pass@1 | Notes |
|---------------|-------------|-------|
| w/o knowledge base evaluation | 63.5 | Imprecise search direction; degenerates to standard MCTS |
| w/o similarity filtering | Slightly lower | Redundant expansions waste search budget |
| w/o sandbox execution feedback | Notably lower | Cannot localize specific erroneous steps |
| **Full RPM-MCTS** | **64.0** | All three components operate synergistically |
| + SFT distillation to base model | Substantial base model gain | RPM-MCTS also serves as a high-quality SFT data generation method |

### Token Consumption
- RPM-MCTS reduces token consumption by approximately 15% compared to standard MCTS, attributed to similarity-based filtering and early truncation at erroneous steps.

### Key Findings
- **Knowledge base retrieval effectively replaces trained PRMs**—process-level evaluation signals are obtained at zero training cost.
- **Similarity filtering reduces token usage by ~15%**—pruning redundant nodes focuses search on high-value paths.
- **Error reflection enables precise correction**—the system identifies not only *that* an error occurred but *at which step*, enabling targeted truncation and re-search.
- **Distillation further improves the base model**—RPM-MCTS functions both as an inference-time method and as a data generation pipeline.
- An 8B model with RPM-MCTS (64.0) matches a 235B baseline (64.6), demonstrating that search can compensate for a 30× parameter gap.

## Highlights & Insights
- The concept of **"homogeneity of algorithm implementations"** is the central insight—structural similarity among correct implementations of the same algorithm type serves as a cost-free reward signal.
- The combination of knowledge base retrieval and MCTS generalizes to other domains where correct references are retrievable (e.g., mathematical proofs, logical reasoning).
- Each of the three components (knowledge base evaluation, redundancy filtering, error reflection) addresses a distinct aspect of search efficiency.

## Limitations & Future Work
- The coverage of the knowledge base determines the method's applicable scope—novel algorithm types with no retrievable reference cannot be handled.
- The similarity threshold requires careful tuning.
- Evaluation is limited to competitive programming tasks; effectiveness on general software engineering code remains unknown.
- Maintaining and updating the knowledge base introduces additional overhead.

## Related Work & Insights
- **vs. MCTS-SQL**: Both apply MCTS to code search; RPM-MCTS additionally incorporates a knowledge base evaluation dimension.
- **vs. PRM (Lightman et al.)**: Trained PRMs require large amounts of annotated data; RPM-MCTS requires no training whatsoever.
- The concept of using a knowledge base as a reward model also holds implications for PRM design in mathematical reasoning.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The idea of using knowledge base retrieval as a training-free PRM is concise and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Includes MCTS comparisons, ablations, and distillation validation.
- **Writing Quality**: ⭐⭐⭐⭐ — The three-component design is presented clearly.
- **Value**: ⭐⭐⭐⭐ — Contributes to both code generation and PRM research.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Intention Chain-of-Thought Prompting with Dynamic Routing for Code Generation](intention_chain-of-thought_prompting_with_dynamic_routing_for_code_generation.md)
- [\[NeurIPS 2025\] Unlocking Multimodal Mathematical Reasoning via Process Reward Model](../../NeurIPS2025/llm_reasoning/unlocking_multimodal_mathematical_reasoning_via_process_reward_model.md)
- [\[NeurIPS 2025\] DreamPRM: Domain-Reweighted Process Reward Model for Multimodal Reasoning](../../NeurIPS2025/llm_reasoning/dreamprm_domain-reweighted_process_reward_model_for_multimodal_reasoning.md)
- [\[AAAI 2026\] Jupiter: Enhancing LLM Data Analysis Capabilities via Notebook and Inference-Time Value-Guided Search](jupiter_enhancing_llm_data_analysis_capabilities_via_notebook_and_inference-time.md)
- [\[ACL 2026\] Efficient Process Reward Modeling via Contrastive Mutual Information](../../ACL2026/llm_reasoning/efficient_process_reward_modeling_via_contrastive_mutual_information.md)

<!-- RELATED:END -->
