---
title: >-
  [Paper Note] P3: Prompts Promote Prompting
description: >-
  [ACL 2025 (Findings)][LLM (Other)][Prompt Optimization] This paper proposes the P3 framework, which is the first to optimize both system prompts and user prompts simultaneously. High-quality prompt templates are generated through offline iterative optimization, which are then utilized for online query-dependent prompt optimization. This approach outperforms methods that optimize only a single prompt side on both general and reasoning tasks, such as Arena-Hard, AlpacaEval…
tags:
  - "ACL 2025 (Findings)"
  - "LLM (Other)"
  - "Prompt Optimization"
  - "Self-Improvement Framework"
  - "System Prompt"
  - "User Prompt"
  - "Automated Prompt Engineering"
date: 2026-05-08
content_hash: 9ecbc3cdc52822bd
---

# P3: Prompts Promote Prompting

**Conference**: ACL 2025 (Findings)  
**arXiv**: [2507.15675](https://arxiv.org/abs/2507.15675)  
**Code**: None  
**Area**: LLM / NLP  
**Keywords**: Prompt Optimization, Self-Improvement Framework, System Prompt, User Prompt, Automated Prompt Engineering

## TL;DR

This paper proposes the P3 framework, which is the first to optimize both system prompts and user prompts simultaneously. High-quality prompt templates are generated through offline iterative optimization, which are then utilized for online query-dependent prompt optimization. This approach outperforms methods that optimize only a single prompt side on both general and reasoning tasks, such as Arena-Hard, AlpacaEval, GSM8K, and GPQA.

## Background & Motivation

**Background**: Current LLM applications typically utilize multi-component prompts, which include system prompts (defining the model's persona and behavioral guidelines) and user prompts (specific task instructions and queries). Automated prompt optimization has emerged as a key direction for enhancing LLM performance, with representative works including APE (Automatic Prompt Engineer) and OPRO (Optimization by PROmpting).

**Limitations of Prior Work**: Existing automated prompt optimization methods are almost entirely "single-sided"—either optimizing only the system prompt or only the user prompt. However, system prompts and user prompts are interdependent: the style of the system prompt affects the optimal phrasing of the user prompt, and vice versa. Optimizing one side while ignoring the other is akin to adjusting parameters for only half of an equation, which inevitably fails to achieve a global optimum.

**Key Challenge**: The joint optimization space for system and user prompts is massive, making a direct search across the entire space computationally prohibitive. A strategy is required to decompose this joint optimization into manageable steps while ensuring the global consistency of the optimized outcomes.

**Goal**: To design a self-improvement framework that simultaneously optimizes system and user prompts, and to further leverage the offline optimized results for online query-dependent prompt optimization.

**Key Insight**: The authors observe a "mutual promotion" relationship between system prompts and user prompts: an effective system prompt helps the LLM better understand the user prompt, while a clearer user prompt in turn makes the instructions within the system prompt more effective. Based on this, P3 adopts an iterative alternating optimization strategy.

**Core Idea**: Alternately optimize system and user prompts through an iterative process to enable mutual reinforcement (Prompts Promote Prompting). After obtaining high-quality prompt templates offline, further perform online query-dependent prompt adaptation.

## Method

### Overall Architecture

P3 consists of two stages: (1) **Offline Optimization Stage**—iteratively and alternately optimizing system and user prompts on the training set. In each round, the user prompt is first fixed to optimize the system prompt, and then the system prompt is fixed to optimize the user prompt, repeating until convergence. (2) **Online Optimization Stage**—for a new user query, real-time query-dependent prompt adjustment is conducted using the optimization patterns and high-quality templates learned during the offline stage.

### Key Designs

1. **Alternating Iterative Optimization Strategy (Offline Stage)**:

    - **Function**: Efficiently searching for the optimal combination in the joint space of system prompts and user prompts.
    - **Mechanism**: Initialize a baseline system prompt and a user prompt template. In each iteration: (a) fix the user prompt and let the LLM generate multiple candidate system prompts based on current performance feedback, selecting the best one; (b) fix the optimized system prompt and similarly let the LLM generate multiple candidate user prompt variants, selecting the best one. Repeat the iterations until performance gains plateau. The selection process is determined by evaluation on a validation set. The overall process resembles coordinate descent, optimizing only one dimension at a time.
    - **Design Motivation**: The joint optimization space is too large for direct search. Alternating optimization decomposes the high-dimensional problem into multiple low-dimensional problems, making the optimization of one side more focused and effective when the other side is fixed. Furthermore, the iterative process allows both sides to adapt to each other.

2. **Feedback-Based Prompt Generation Mechanism**:

    - **Function**: Utilizing failure cases to guide the direction of prompt improvement.
    - **Mechanism**: In each optimization iteration, collect error cases and failure patterns under the current prompts. These failure samples are fed into the LLM along with the current prompts, prompting the LLM to analyze the issues and generate improved prompt versions. For instance, if the model frequently misses unit conversions in mathematical reasoning, the LLM will incorporate an instruction like "Pay attention to unit conversions" into the new system prompt. After generating multiple candidates, the best one is selected based on validation set performance.
    - **Design Motivation**: Blindly generating candidate prompts is inefficient. Guiding generation with error feedback mimics the human prompt debugging process—analyzing what went wrong and making targeted adjustments.

3. **Online Query-Dependent Prompt Optimization**:

    - **Function**: Adapting the general prompt templates obtained offline to specific queries.
    - **Mechanism**: Multiple high-quality pairs of (system prompt, user prompt) templates are obtained offline. When a new query is received online, it is first semantically matched against existing templates to find the most relevant one. Then, the LLM is prompted to fine-tune based on this template and the specific query—such as adding specific reasoning guidance in the user prompt, or emphasizing query-relevant capability requirements in the system prompt. The additional inference overhead of this step is minimal as it is a fine-tuning process rather than optimization from scratch.
    - **Design Motivation**: Different queries may require different prompt styles. Offline optimization yields a template that is "optimal on average", and performing query-specific adaptation online can further boost performance.

### Loss & Training

P3 does not involve model parameter training; instead, it optimizes the prompt text. The "loss function" is equivalent to the evaluation metrics of the target tasks—using win-rate (e.g., GPT-4 judge on Arena-Hard) for general tasks, and accuracy (e.g., exact match on GSM8K) for reasoning tasks. The selection of candidate prompts is conducted on a validation set to avoid overfitting.

## Key Experimental Results

### Main Results

| Method | Arena-Hard (Win%) | AlpacaEval (Win%) | GSM8K (Acc%) | GPQA (Acc%) |
|------|-------------------|-------------------|-------------|-------------|
| No Optimization (Base Prompt) | 32.5 | 28.4 | 82.3 | 34.5 |
| System Prompt Only | 38.2 | 33.1 | 85.6 | 37.8 |
| User Prompt Only | 40.7 | 35.4 | 86.9 | 39.2 |
| P3 Offline (Joint Optimization) | 46.3 | 41.2 | 89.5 | 42.6 |
| P3 Offline + Online | **49.8** | **44.7** | **91.2** | **45.1** |

### Ablation Study

| Configuration | Arena-Hard (Win%) | GSM8K (Acc%) | Description |
|------|-------------------|-------------|------|
| P3 Full (Offline + Online) | 49.8 | 91.2 | Full model |
| w/o Online Stage | 46.3 | 89.5 | Online adaptation contributes ~+3% |
| w/o Error Feedback | 42.1 | 87.4 | Feedback mechanism contributes ~+4% |
| Only 1 Iteration | 41.5 | 86.8 | Multi-round iteration contributes ~+5% |
| Random Candidate Generation (w/o Feedback) | 39.3 | 85.2 | Confirms the necessity of feedback guidance |

### Key Findings

- **Joint Optimization >> Single-sided Optimization**: Compared to optimizing only system or user prompts, the joint optimization of P3 leads to significant improvements across all benchmarks (Arena-Hard +9-17%, GSM8K +4-9%), validating the interdependence of system and user prompts.
- **Diminishing Returns of Iterations**: Performance typically converges after 3-5 iterations, with marginal returns diminishing in subsequent rounds. The first iteration yields the largest improvement, indicating that initial prompts usually have substantial room for improvement.
- **Sustained Value of Online Optimization**: Even when high-quality templates have been acquired offline, query-dependent online adjustments still contribute an additional 2-4% improvement, particularly for atypical queries.
- **Benefits for Both General and Reasoning Tasks**: P3 is effective not only on open-ended generation tasks (Arena-Hard, AlpacaEval) but also delivers significant improvements on mathematical and scientific tasks that require precise reasoning (GSM8K, GPQA).

## Highlights & Insights

- **Core insight of "Prompts Promote Prompting"**: There is a collaborative effect between system prompts and user prompts—a good system prompt amplifies the effectiveness of a good user prompt, and vice versa. This observation itself is valuable, explaining why tuning these two components separately in engineering practice yields limited effectiveness.
- **Optimization strategy analogous to coordinate descent**: Decomposing joint optimization into alternating univariate optimization is a simple yet effective strategy. This approach can be generalized to any scenario requiring simultaneous optimization of multiple textual components, such as concurrently optimizing retrieval queries and generation prompts in RAG systems.
- **Error-driven prompt evolution**: Using failure cases as feedback to guide prompt generation is a "debugging-style" automated prompt engineering method, highly aligned with practical prompt engineering workflows.

## Limitations & Future Work

- **High computational cost**: The offline stage requires multiple iterations, generating and evaluating multiple candidates per round, which incurs substantial API call costs.
- **Potential overfitting to the validation set**: Selecting the best candidate using a validation set runs the risk of overfitting, especially when the validation set is small.
- **Insufficient validation of generalization across different LLMs**: Prompts optimized on one LLM may not transfer well to another; cross-model transferability has not been explored.
- Future work could investigate whether optimized prompts can generalize across different LLMs, as well as how to further improve optimization efficiency while controlling costs.

## Related Work & Insights

- **vs APE (Zhou et al., 2023)**: APE automatically generates and selects prompts, but only optimizes user prompts. P3 extends this to simultaneously optimize both dimensions and incorporates iterative refinement.
- **vs OPRO (Yang et al., 2024)**: OPRO uses LLMs as optimizers to generate prompts but is similarly limited to single-sided optimization. P3's alternating strategy builds and improves upon OPRO.
- **Connection to DSPy**: DSPy proposes a declarative LLM programming framework. P3's joint optimization concept can serve as a module optimization strategy within DSPy pipelines.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of simultaneously optimizing system and user prompts is natural but had not been systematically investigated previously.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers both general and reasoning tasks with relatively complete ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The framework is clear, and the name P3 is ingenious.
- Value: ⭐⭐⭐⭐ Holds direct practical value for automated prompt engineering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] SCULPT: Systematic Tuning of Long Prompts](sculpt_systematic_tuning_of_long_prompts.md)
- [\[ACL 2025\] Dolphin: Document Image Parsing via Heterogeneous Anchor Prompting](dolphin_document_image_parsing_via_heterogeneous_anchor_prompting.md)
- [\[ACL 2025\] Multi-Prompting Decoder Helps Better Language Understanding](multi-prompting_decoder_helps_better_language_understanding.md)
- [\[ACL 2025\] Leveraging Self-Attention for Input-Dependent Soft Prompting in LLMs](input_dependent_soft_prompting.md)
- [\[ACL 2025\] Explain-then-Process: Using Grammar Prompting to Enhance Grammatical Acceptability Judgments](explain-then-process_using_grammar_prompting_to_enhance_grammatical_acceptabilit.md)

</div>

<!-- RELATED:END -->
