---
title: >-
  [Paper Note] On the Eligibility of LLMs for Counterfactual Reasoning: A Decompositional Study
description: >-
  [ICLR2026][Causal Inference][counterfactual reasoning] This paper proposes a decompositional evaluation framework grounded in Structural Causal Models (SCMs)…
tags:
  - "ICLR2026"
  - "Causal Inference"
  - "counterfactual reasoning"
  - "structural causal model"
  - "LLM evaluation"
  - "decompositional analysis"
  - "tool-augmented learning"
date: 2026-05-08
content_hash: e362c43d5730b0b5
---

# On the Eligibility of LLMs for Counterfactual Reasoning: A Decompositional Study

**Conference**: ICLR2026
**arXiv**: [2505.11839](https://arxiv.org/abs/2505.11839)  
**Code**: To be confirmed  
**Area**: Causal Reasoning
**Keywords**: counterfactual reasoning, structural causal model, LLM evaluation, decompositional analysis, tool-augmented learning

## TL;DR

This paper proposes a decompositional evaluation framework grounded in Structural Causal Models (SCMs), decomposing LLM counterfactual reasoning into four stages (causal variable identification → causal graph construction → intervention identification → outcome reasoning). It systematically diagnoses capability bottlenecks at each stage across 11 multimodal datasets, and introduces tool-augmented and advanced elicitation strategies to improve performance.

## Background & Motivation

- Counterfactual reasoning is a critical capability for evaluating LLM adaptability and reliability: given a hypothetical change in premises, can a model adjust its conclusions accordingly?
- Prior work has shown that LLMs perform poorly on counterfactual reasoning tasks, yet a **standardized framework** for systematically analyzing failure causes is lacking.
- Existing evaluation approaches are predominantly end-to-end "direct probing": the model is given a counterfactual intervention and asked to respond, without accounting for the **causal modeling foundations** underlying counterfactual reasoning—such as variable identification and causal dependency construction.
- A decompositional approach is needed to break counterfactual reasoning into independently assessable stages, enabling precise localization of LLM reasoning bottlenecks.

## Core Problem

1. How do LLMs perform across the decomposed stages of counterfactual reasoning (causal variable identification, causal graph construction, intervention identification, and outcome reasoning)?
2. Which auxiliary techniques can effectively improve LLMs' counterfactual reasoning capabilities?

## Method

### Causal Modeling Foundation

Based on Pearl's Structural Causal Model (SCM), four types of causal variables are defined:

- **Exposure (X)**: The treatment or intervention condition applied to the system.
- **Covariate (Z)**: Pre-treatment variables that may affect both X and Y.
- **Mediator (M)**: A variable lying on the X→Y causal pathway, where $M = f_M(X, Z)$.
- **Outcome (Y)**: The result variable influenced by the exposure, where $Y = f_Y(X, M, Z)$.

The essence of counterfactual reasoning: given observed facts $(x, z, m, y)$, if X is intervened upon to take value $x'$, the updated quantities $M' = f_M(x', z)$ and $Y' = f_Y(x', M', z)$ must be computed.

### Four-Stage Decompositional Evaluation

| Stage | Task | Evaluation Objective |
|-------|------|----------------------|
| Task I | Causal Variable Identification | Correctly identify X, Z, M, Y from factual information |
| Task II | Causal Graph Construction | Construct the correct DAG given the identified variables |
| Task III | Counterfactual Intervention Identification | Correctly identify the intervened value $X'$ from a counterfactual query |
| Task IV | Outcome Reasoning | Infer counterfactual mediator $M'$ and outcome $Y'$ given the causal graph and intervention |

Key design: each stage's evaluation **provides ground-truth outputs from preceding stages**, so as to isolate and measure each stage's independent capability.

### Benchmark Construction

Eleven datasets are collected and curated across four modalities:

- **Text**: CRASS (QA), CLOMO (logical parsing), RNN-Typology (syntactic parsing)
- **Vision-Language**: CVQA-Bool, CVQA-Count, COCO
- **Mathematical/Symbolic**: Arithmetic
- **Code**: HumanEval-Exe, Open-Critic, Code-Preference
- **Mixed**: MalAlgoQA

Each dataset is preprocessed: causal variables are annotated, reference DAGs are constructed, and intervention variables and counterfactual outcomes are extracted.

### Improvement Strategies

**Strategy 1: Tool-Augmented Execution** (for explicit variable identification)

Modality-specific tools are employed to assist entity recognition:

- Text/Math: `bert-base-NER` for candidate entity extraction
- Vision: `grounding-dino-base` for detecting relevant objects in images
- Code: `GraphCodeBERT` for extracting functions, variables, and control structures

Tool outputs are then filtered and refined by the LLM to produce the final set of explicit variables.

**Strategy 2: Advanced Elicitation Strategies** (for implicit variable reasoning)

- **CoT**: Step-by-step reasoning over mediator variables.
- **CoT-SC**: Generates $k=5$ reasoning paths and selects the answer via majority voting.
- **ToT**: $k=5$ branching reasoning paths, with intermediate results evaluated against task descriptions using BERTScore.

## Key Experimental Results

Seven mainstream LLMs are evaluated: GPT-5, GPT-o4-mini-high, Qwen3-VL-235B, Llama-4-Scout, Llama-4-Maverick, Gemini2.5-Pro, and DeepSeek-VL.

### Task I: Causal Variable Identification

- F1 for X reaches 87–92% on text datasets, but drops noticeably on visual and code modalities (e.g., <72% on Open-Critic).
- **Implicit variable M is the hardest to identify**: even on text datasets, F1 for M is 5–10 points lower than for X.
- Higher modality complexity leads to greater difficulty; the inferential nature of variables (explicit vs. implicit) constitutes an independent difficulty factor beyond modality.

### Task II: Causal Graph Construction

- Overall performance is strongest, with F1 > 0.9 in most cases.
- The causal graph structure is rule-governed, enabling LLMs to apply construction rules effectively once variables are given.

### Task III: Counterfactual Intervention Identification

- LLMs consistently and accurately identify the value of $X'$ across modalities.
- However, this task is relatively straightforward, as it does not require reasoning about the propagation of intervention effects.

### Task IV: Outcome Reasoning (Core Bottleneck)

- **LLMs perform worst when reasoning about counterfactual $M'$ and $Y'$.**
- GPT-5 achieves $M'$=92.1%, $Y'$=88.0% on CRASS, but drops to $M' \approx 75\%$, $Y' \approx 70\%$ on code datasets.
- Weaker models (e.g., DeepSeek-VL) reach $Y' \approx 54\%$ on visual datasets.
- This indicates that LLMs lack sufficient capacity for reasoning along causal chains.

### Improvement Effects

- Tool augmentation yields significant gains for explicit variables: Llama-4-Scout improves by +32.0% on X in CVQA-Count.
- Advanced elicitation strategies are effective but bounded for implicit variables: CoT-SC and ToT sometimes underperform simple CoT, as complex strategies may induce **overthinking**.
- Combining both strategies yields the largest overall gains, though not in a simple additive manner, as errors accumulate and propagate through the pipeline.

## Highlights & Insights

1. **Systematic decompositional framework**: The first work to decompose LLM counterfactual reasoning into four independently evaluable stages based on Pearl's SCM, enabling fine-grained diagnosis.
2. **Large-scale multimodal benchmark**: Covers 11 datasets across 4 modalities, with each instance annotated with causal variables and reference DAGs.
3. **Diagnostic findings**: Causal graph construction is not the bottleneck (F1 > 0.9); the true bottleneck lies in implicit variable reasoning, particularly for counterfactual $M'$ and $Y'$.
4. **Actionable improvement strategies**: Tool augmentation and elicitation strategies target the respective weaknesses of explicit and implicit variables, and the limitations of ToT due to overthinking are identified.

## Limitations & Future Work

- Causal variable annotation and DAG construction during preprocessing rely on manual or semi-automatic methods, limiting scalability.
- The isolated evaluation design (providing ground-truth inputs per stage) does not reflect error accumulation in real end-to-end settings.
- No effective mitigation is proposed for the overthinking problem observed in advanced elicitation strategies.
- Only a limited set of LLMs is evaluated; open-source small models are not analyzed.
- The tool selection in the tool-augmented strategy is relatively fixed, with no exploration of alternative configurations.

## Related Work & Insights

| Method | Evaluation Style | Causal Modeling | Multimodal | Decompositional |
|--------|-----------------|-----------------|------------|-----------------|
| CRASS (Frohberg et al., 2021) | End-to-end QA | None | Text only | No |
| DICE (Shrivastava et al., 2025) | Diagnostic QA | Partial | Text only | No |
| CausalProbe (Chi et al., 2024) | Probing | Partial | Text only | No |
| MalAlgoQA (Sonkar et al., 2024) | Multiple-choice QA | None | Text + symbolic | No |
| **Ours** | **Four-stage decomposition** | **Full SCM** | **4 modalities** | **Per-stage diagnosis** |

Core distinction: prior work primarily conducts end-to-end evaluation; this paper is the first to align evaluation with the structured steps of SCM, enabling per-module failure attribution.

The decompositional evaluation paradigm is generalizable to other complex reasoning tasks (e.g., mathematical proof, multi-step planning) by decomposing steps to precisely localize model weaknesses. The "tool augmentation + LLM high-level reasoning" pattern warrants further exploration in causal inference, particularly within multi-agent frameworks. The overthinking phenomenon suggests that elicitation strategy design must balance reasoning depth against the degree of evidential support available. Implicit variable reasoning constitutes a key bottleneck in LLM causal capabilities, and future work should address this at the training level rather than relying solely on inference-time strategies.

## Rating
- **Novelty**: 8/10 — The decompositional framework offers a novel entry point, though the task definitions at individual stages are relatively standard.
- **Experimental Thoroughness**: 8/10 — Comprehensive coverage with 11 datasets and 7 LLMs, but lacks analysis of open-source small models and ablation studies.
- **Writing Quality**: 7/10 — Structure is clear, but tables are data-dense and core insights could be distilled more concisely.
- **Value**: 8/10 — Provides a systematic diagnostic tool and actionable improvement directions for LLM counterfactual reasoning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Perturbations](rfeval_benchmarking_reasoning_faithfulness_under_counterfactual_perturbations.md)
- [\[NeurIPS 2025\] Counterfactual Reasoning for Steerable Pluralistic Value Alignment of Large Language Models](../../NeurIPS2025/causal_inference/counterfactual_reasoning_for_steerable_pluralistic_value_alignment_of_large_lang.md)
- [\[ACL 2026\] Evaluating Counterfactual Strategic Reasoning in Large Language Models](../../ACL2026/causal_inference/evaluating_counterfactual_strategic_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Parallel Universes, Parallel Languages: A Comprehensive Study on LLM-based Multilingual Counterfactual Example Generation](../../ACL2026/causal_inference/parallel_universes_parallel_languages_a_comprehensive_study_on_llm-based_multili.md)
- [\[NeurIPS 2025\] Few-Shot Knowledge Distillation of LLMs With Counterfactual Explanations](../../NeurIPS2025/causal_inference/few-shot_knowledge_distillation_of_llms_with_counterfactual_explanations.md)

</div>

<!-- RELATED:END -->
