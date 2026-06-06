---
title: >-
  [Paper Note] METER: Evaluating Multi-Level Contextual Causal Reasoning in Large Language Models
description: >-
  [ACL 2026][Interpretability][Contextual Causal Reasoning] METER is the first benchmark to systematically evaluate LLMs' three-level causal reasoning (discovery / intervention / counterfactual) under a **unified context**…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "Contextual Causal Reasoning"
  - "Ladder of Causation"
  - "Information Flow Analysis"
  - "Saliency Scores"
  - "LLM Evaluation"
date: 2026-05-08
content_hash: a207ba062374825c
---

# METER: Evaluating Multi-Level Contextual Causal Reasoning in Large Language Models

**Conference**: ACL 2026  
**arXiv**: [2604.11502](https://arxiv.org/abs/2604.11502)  
**Code**: https://github.com/SCUNLP/METER (Available)  
**Area**: Causal Reasoning Evaluation / Mechanistic Interpretability  
**Keywords**: Contextual Causal Reasoning, Ladder of Causation, Information Flow Analysis, Saliency Scores, LLM Evaluation

## TL;DR
METER is the first benchmark to systematically evaluate LLMs' three-level causal reasoning (discovery / intervention / counterfactual) under a **unified context** (comprising 4,145 samples co-constructed via human auditing and LLM collaboration). Through saliency-based information flow analysis, it is discovered that LLM performance drops from 93% to 73% as they ascend the causal ladder—the root cause being interference from irrelevant facts during the discovery phase and a significant decrease in faithfulness to the context during higher-level reasoning phases.

## Background & Motivation
**Background**: Causal reasoning is considered an essential capability for AGI, particularly in high-risk domains like medical diagnosis. Clinical reports often require answering: (1) why ischemia occurred (discovery); (2) what would happen if a PCI was performed (intervention); and (3) what would have happened if there was no initial blockage (counterfactual). Judea Pearl’s "Ladder of Causation" categorizes these into three levels of increasing difficulty. Existing benchmarks like ExpliCa, CRASS, WIKIWHY, CausalQA, and IfQA evaluate LLM causal capabilities, but each typically covers only a single level.

**Limitations of Prior Work**:
1. **Incomplete Coverage**: Existing benchmarks either evaluate only discovery (WIKIWHY, CausalQA, RECESS, CRAB) or only counterfactuals (CRASS, IfQA); none integrate all three levels into a single benchmark.
2. **Context Inconsistency**: Even when cross-level evaluations occur (CalQuest, CLADDER, CausalBench), different questions use different contexts, precluding a strict comparison of the performance gap across levels for the same set of facts.
3. **Lack of Mechanistic Analysis**: No current benchmark provides an internal mechanistic study of failure modes—scores alone do not reveal where the models are failing internally.
4. **Mixed Paradigms**: Existing works mix common sense, formal symbolic rules, and contextual (document-based) paradigms; notably, contextual causal reasoning (strictly following evidence text) has not been evaluated in isolation.

**Key Challenge**: Implementing a "fair multi-level evaluation" requires ensuring each context is paired with questions for all three levels, which incurs extremely high manual annotation costs and requires strict de-contamination to prevent LLMs from memorizing answers.

**Goal**: (i) Systematically evaluate three-level causal reasoning under a unified context; (ii) provide a large-scale (4K+), high-quality, and de-contaminated human-audited benchmark; (iii) use mechanistic interpretability (information flow analysis) to explain LLM failure modes.

**Key Insight**: Translate Pearl’s Ladder of Causation (originally in the numerical probability domain) into a **textual paradigm** and construct the benchmark using a "human-LLM collaboration + three-stage quality inspection" pipeline. Employ saliency-based information flow (Wang et al., 2023) to trace internal model focus.

**Core Idea**: (1) Derive questions for all three levels from the same context to control variables for fair comparison; (2) design four types of distractors (contradictory / unfounded / causal-reversal / irrelevant-fact) to systematically detect failure modes; (3) bridge behavioral failure and mechanistic failure through information flow analysis on erroneous samples.

## Method

### Overall Architecture
METER consists of two primary components: (A) **Benchmark Construction** — extracting cause-effect pairs from 4 source datasets (ESL, MAVEN-ERE, MECI, WIKIWHY), using Gemini-2.5-Pro to expand event descriptions and generate questions, answers, and four types of distractors. Following a three-stage human audit by 9 annotators with NLP backgrounds, 4,145 entries were finalized (each entry contains one context and three-level multiple-choice questions with 5 options). (B) **Evaluation & Mechanistic Analysis** — evaluating 12 LLMs (GPT-4o, GPT-5, Gemini-3 series, Qwen3 full sizes, Llama-3.3-70B) across 4 prompting strategies and performing saliency-based hierarchical information flow analysis on Qwen3-4B/8B and Llama-3.2-3B.

### Key Designs

1. **Linguistic Translation of Causal Definitions (Benchmark Theoretical Foundation)**:

    - **Function**: Map the three levels of Pearl’s Ladder of Causation from the probabilistic domain to natural language tasks, providing clear and actionable criteria for each level.
    - **Mechanism**: (L1) **Causal Discovery**: Identifying existing causal relationships in text via explicit keywords ("caused/because") or implicit semantics. Example: Identifying "thrombotic occlusion $\to$ myocardial ischemia" from a clinical report. (L2) **Intervention**: Predicting the consequences of "introducing a new action" into the context, requiring multi-step causal chain reasoning. Example: "What if a PCI is performed?" requires reasoning: perform PCI $\to$ clear occlusion $\to$ restore blood flow. (L3) **Counterfactual**: Reversing known conditions to infer outcomes had the past been different. Example: "If there had been no blockage, the damage would not have occurred."
    - **Design Motivation**: Applying Pearl’s mathematical definitions directly to LLM evaluation is non-trivial; this linguistic definition based on "verbs / temporal order / modification dimensions" allows for batch generation of question templates. The critical difference between L2 and L3 is: L2 involves forward reasoning on real facts, whereas L3 requires reasoning on hypotheses that conflict with reality, forcing LLMs to truly "simulate a counterfactual world" rather than rely on rote memorization.

2. **Controlled Data Construction with Unified Context & Four Distractor Types**:

    - **Function**: Challenge LLMs across all three levels using the same context and expose failure modes through carefully designed incorrect options.
    - **Mechanism**: Each entry strictly follows the "1 context + 3 questions (L1/L2/L3)" structure; contexts average 228.91 tokens. Each question features 1 correct answer and 4 distractors from predefined categories:
        - **Contradictory Statement**: Directly conflicts with facts in the context.
        - **Unfounded Statement**: Not mentioned or inferable from the context (typical hallucination).
        - **Causal Reversal**: The causal direction is flipped (treating effect as cause).
        - **Irrelevant Fact**: A true statement from the context that is irrelevant to the specific causal chain.
        
        The pipeline involves four stages: (i) data preparation (de-contaminating pairs where three LLMs—Gemini-2.5-Pro, GPT-5, Qwen3-235B—agree on the answer without context); (ii) manual annotation filtering (Fleiss $\kappa=0.78$); (iii) Gemini-2.5-Pro generation based on templates; (iv) two-stage manual editing and filtering by 9 annotators ($\kappa=0.71/0.75$).
    - **Design Motivation**: (1) Unified contexts eliminate the confounding variable of "task difficulty," attributing performance gaps to model capability. (2) The four distractors serve as diagnostic probes—e.g., selecting an "Irrelevant Fact" indicates interference by irrelevant details, while "Unfounded" reveals hallucinations. (3) The LLM ensemble and high filtering rate minimize data contamination and noise.

3. **Saliency-Based Hierarchical Information Flow Analysis (Mechanistic Innovation)**:

    - **Function**: Drill down from "behavioral accuracy" to "mechanistic information flow" to reveal internal causes of LLM failure at different causal levels.
    - **Mechanism**: Based on Wang et al. (2023), saliency is calculated for each token-pair $(i,j)$ as $I_l(i,j) = \sum_h |A_{h,l}^\top \frac{\partial \mathcal{L}}{\partial A_{h,l}}|$, defining inter-segment information flow as $S_{X \to Y} = \frac{\sum_{(i,j) \in \mathcal{C}_{X \to Y}} I_l(i,j)}{|\mathcal{C}_{XY}|}$. Prompts are segmented into: **E** (evidence span), **N** (non-evidence), **Q** (question), **O** (selected option), and **T** (target final token). Analysis tracks $S_{E\to O}$, $S_{N\to O}$, $S_{Q\to O}$, $S_{O\to T}$, and $S_{rest}$ across layers and compares Correct vs. Error subsets. Validation is performed via attention masking—masking $E\to O$ flow in shallow layers (1-24) to observe accuracy drops (Discovery accuracy fell from 0.827 to 0.579, confirming evidence aggregation occurs in early layers).
    - **Design Motivation**: Accuracy only shows *that* a model fails; saliency-based flow indicates *where* it fails. By transforming raw attention into "signal transmission strength" and aligning it with human-labeled evidence spans, one can precisely locate whether the model is actually attending to evidence.

### Loss & Training
**METER is an evaluation-only project and involves no training**. All closed-source LLMs were accessed via APIs; open-source LLMs used vLLM on 4x A100 GPUs. Parameters included decoding temperature=0; Four prompting strategies: Zero-shot, Zero-shot CoT, Few-shot, and Few-shot CoT. Reasoning-optimized models (GPT-5, Gemini-3-Pro, Qwen3-Next-Thinking) were tested only on Zero-shot and Few-shot. The primary metric is accuracy. All results are averaged over 3 independent runs.

## Key Experimental Results

### Main Results (4 prompting types × 12 LLMs × 3 Causal Levels)

**Zero-shot accuracy (%)**:

| Model | L1 Discovery | L2 Intervention | L3 Counterfactual | Drop |
| :--- | :---: | :---: | :---: | :---: |
| **Gemini3-Pro** | **93.50** | **81.92** | **73.05** | -20.45 |
| GPT-5 | 92.96 | 82.17 | 72.14 | -20.82 |
| Gemini3-Flash | 91.02 | 78.09 | 70.51 | -20.51 |
| Qwen3-Next-Thinking | 90.36 | 77.52 | 70.57 | -19.79 |
| Qwen3-Next-Instruct | 89.56 | 75.13 | 64.47 | -25.09 |
| GPT-4o | 87.92 | 77.96 | 67.42 | -20.50 |
| Llama-3.3-70B | 87.24 | 78.17 | 62.08 | -25.16 |
| Qwen3-32B | 87.26 | 71.54 | 61.12 | -26.14 |
| Qwen3-14B | 87.95 | 67.54 | 52.05 | -35.90 |
| Qwen3-8B | 86.27 | 64.48 | 51.40 | -34.87 |
| Qwen3-4B | 87.03 | 53.47 | 43.26 | -43.77 |
| Qwen3-0.6B | 64.46 | 31.27 | 25.88 | -38.58 |
| **Human** | **95.80** | **92.80** | **91.00** | **-4.80** |

**Key Findings**: (1) All LLMs show a performance decline as they ascend the causal ladder; even the strongest, Gemini3-Pro, drops by 20+ points, whereas humans only drop by 4.8. (2) **Reasoning-optimized models are significantly more robust at higher levels than instruction-tuned models**, with Qwen3-Next-Thinking outperforming Qwen3-Next-Instruct by over 6 points on counterfactuals.

### Error Distribution Analysis (Ratio of distractor types in Zero-shot error samples)

| Distractor Type | Discovery (Qwen3-4B) | Intervention | Counterfactual |
|:---|:---:|:---:|:---:|
| **Irrelevant Fact** | **55.41%** | 26.87% | 24.15% |
| Unfounded | 22.20% | 39.43% | 36.77% |
| Contradictory | 5.69% | 20.96% | **33.87%** |
| Causal Reversal | 16.70% | 12.74% | 5.22% |

**Key Findings**:
- **Discovery errors are primarily caused by interference from irrelevant facts** (55%), where the model selects details that are true in the context but causally irrelevant.
- **Higher-level errors are driven by a surge in Unfounded (hallucination) and Contradictory choices**, indicating a significant drop in faithfulness.
- **Counterfactual "Contradictory" errors reach 33.87%**, indicating models fail to maintain logical consistency within hypothetical scenarios.

### Information Flow Analysis (Average saliency for Qwen3-4B)

| Metric | Discovery (Cor / Err) | Intervention | Counterfactual |
|:---|:---:|:---:|:---:|
| $S_{E\to O}$ (evidence$\to$option) | **0.1690 / 0.1247** | 0.1144 / 0.0936 | 0.1095 / 0.0988 |
| $S_{N\to O}$ (noise$\to$option) | **0.0945 / 0.1262** | 0.0858 / 0.0858 | 0.0805 / 0.0884 |
| $S_{Q\to O}$ (question$\to$option) | 0.2508 / 0.2163 | 0.2657 / 0.2378 | 0.2666 / 0.2281 |
| $S_{O\to T}$ (option$\to$target) | 0.4685 / 0.4941 | 0.5039 / 0.5457 | 0.5071 / 0.5452 |

**Key Findings**:
- **Discovery Error Samples**: Evidence flow drops while noise flow increases, confirming that distracted attention is the mechanistic root cause.
- **Intervention/Counterfactual**: Evidence flow is low ($\sim 0.10$) regardless of correctness, suggesting models rely more on internal world knowledge than the context, leading to higher hallucination rates.

### Attention Masking (Qwen3-4B)

| Mask Scope | Discovery Acc | Intervention Acc | Counterfactual Acc |
|:---|:---:|:---:|:---:|
| Baseline | 0.827 | $\sim 0.53$ | $\sim 0.43$ |
| **Shallow 1-24 layer $E\to O$ mask** | **0.579** (-25 pt) | $\sim 0.53$ | $\sim 0.43$ |
| Deep 25-end $E\to O$ mask | 0.827 | $\sim 0.53$ | $\sim 0.43$ |

$\to$ Evidence aggregation in Discovery **occurs only within the first 24 layers**. For intervention/counterfactuals, masking evidence flow has almost no impact, indicating a lack of dependence on the provided evidence.

## Highlights & Insights
- **Unified-Context Rigorous Experimental Design**: By splitting the same set of facts into three-level questions, this study quantifies the LLM performance decay along the causal ladder for the first time.
- **Dual-Viewpoints (Behavioral + Mechanistic)**: Error distributions explain *what* is wrong, information flow explains *why*, and attention masking provides *causal verification*. This multi-evidence narrative sets a new standard for evaluation papers.
- **Transferable Distractor Design**: The four distractor types (Unfounded, Contradictory, Reversal, Irrelevant Fact) effectively cover most LLM failure modes in multiple-choice tasks and are applicable to other reasoning benchmarks.
- **Realistic Value of "+Evidence"**: Explicitly adding evidence spans to prompts improves accuracy by 2-3 points, suggesting that the primary value of RAG/retrieval is **strengthening information flow** rather than just providing external knowledge.

## Limitations & Future Work
- **Limitations**: (1) Mechanistic analysis is restricted to open-source models; (2) source data from public corpora cannot be entirely excluded from potential pre-training contamination; (3) de-contamination relies on black-box verification.
- **Future Directions**: (1) Incorporate "+evidence" findings into training objectives (e.g., SFT/RL losses that strengthen $S_{E\to O}$); (2) expand to multilingual and multi-domain settings; (3) move beyond MCQs to evaluate open-ended counterfactual generation.

## Related Work & Insights
- **Comparison with ExpliCa / CRASS**: Unlike those that rely on common sense, METER strictly requires textual evidence to distinguish between context understanding and prior knowledge activation.
- **Comparison with CLADDER / CausalBench**: These focus on formal do-calculus and probabilistic operations, whereas METER’s contextual paradigm is more aligned with practical NLP applications.
- **Insight**: Evaluation papers should evolve from just "reporting scores" to "scores + error distribution + mechanistic analysis + causal verification + improvement sketches."

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of unified-context multi-level causal evaluation and mechanistic information flow analysis is a clear innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ High coverage across 12 models, multiple prompting strategies, human baselines, and attention masking.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear definitions, intuitive pipeline illustrations, and a compelling narrative for mechanistic analysis.
- **Value**: ⭐⭐⭐⭐⭐ An indispensable resource for the causal reasoning community; provides actionable insights for RAG and agent design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Knowledge Vector of Logical Reasoning in Large Language Models](knowledge_vector_of_logical_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Jacobian Scopes: Token-Level Causal Attributions in LLMs](jacobian_scopes_token-level_causal_attributions_in_llms.md)
- [\[ACL 2026\] Sparse Feature Coactivation Reveals Causal Semantic Modules in Large Language Models](sparse_feature_coactivation_reveals_causal_semantic_modules_in_large_language_mo.md)
- [\[ACL 2026\] Compositional Steering of Large Language Models with Steering Tokens](compositional_steering_of_large_language_models_with_steering_tokens.md)
- [\[ACL 2026\] Tracing Relational Knowledge Recall in Large Language Models](tracing_relational_knowledge_recall_in_large_language_models.md)

</div>

<!-- RELATED:END -->
