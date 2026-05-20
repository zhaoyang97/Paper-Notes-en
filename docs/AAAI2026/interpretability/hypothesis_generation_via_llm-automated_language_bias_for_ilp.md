---
title: >-
  [Paper Note] Hypothesis Generation via LLM-Automated Language Bias for ILP
description: >-
  [AAAI 2026][Interpretability][Inductive Logic Programming] This paper proposes the first end-to-end framework in which a multi-agent LLM system (Actor/Critic) automatically constructs ILP language bias (predicate system…
tags:
  - "AAAI 2026"
  - "Interpretability"
  - "Inductive Logic Programming"
  - "Language Bias"
  - "Multi-Agent"
  - "Prolog"
date: 2026-05-08
content_hash: 70c6d852f9c1f1a4
---

# Hypothesis Generation via LLM-Automated Language Bias for ILP

**Conference**: AAAI 2026
**arXiv**: [2505.21486](https://arxiv.org/abs/2505.21486)  
**Code**: None  
**Area**: Interpretability
**Keywords**: Inductive Logic Programming, Language Bias, Multi-Agent, Prolog, Interpretability

## TL;DR
This paper proposes the first end-to-end framework in which a multi-agent LLM system (Actor/Critic) automatically constructs ILP language bias (predicate system, type declarations, and mode constraints) from raw text. A Translator agent converts text into Prolog facts, and the MAXSYNTH solver then induces a globally optimal rule set based on the MDL principle. The framework achieves 88.3% and 81.3% accuracy on the SHOES and ZENDO tasks, respectively, with variance below 5% across four LLMs.

## Background & Motivation
**Background**: Inductive Logic Programming (ILP) is a classical approach for discovering interpretable logical rules from data. Heuristic methods such as FOIL, Progol, and Aleph, as well as constraint-based solvers such as ILASP, Popper, and MAXSYNTH, have continued to advance rule-search algorithms. Recent LLMs have demonstrated strong performance in hypothesis generation (e.g., ChatRule, HypoGeniC, IHR).

**Limitations of Prior Work**: (a) ILP requires experts to manually define the "language bias"—predicate sets, types, and mode declarations—making it difficult to extend to new domains (e.g., protein interaction research requires defining predicates for atomic distances and amino acid properties); (b) pure LLM methods (HypoGeniC, IHP) are highly sensitive to noise—a 12.5% increase in data noise causes accuracy to plummet from 71.2% to 50.9%; (c) heuristic rule generation by LLMs cannot guarantee the global consistency and optimality of the induced rule set, whereas ILP solvers can.

**Key Challenge**: ILP requires expert knowledge to define the search space (limiting scalability), while LLMs do not require such knowledge but produce results that are neither robust nor globally optimal.

**Goal**: To automate the most labor-intensive component of ILP (language bias construction) while retaining the robustness and global optimality guarantees of ILP solvers.

**Key Insight**: LLMs excel at semantic understanding and conceptual abstraction (well-suited for designing predicate systems), while ILP solvers excel at constrained search and global optimization (well-suited for inducing rules)—making them ideal collaborators.

**Core Idea**: LLMs automatically construct the "language bias" (predicate system) for ILP, and the ILP solver searches for the globally optimal rule set within this space—an ideal division of labor between neural and symbolic components.

## Method

### Overall Architecture
The framework consists of three stages: (1) predicate system construction via a multi-agent LLM; (2) symbolic knowledge encoding in which an LLM translates text into Prolog facts; and (3) ILP learning in which the MAXSYNTH solver induces rules.

### Key Designs

1. **Predicate System Construction (Actor-Critic Multi-Agent)**:

    - **Function**: Automatically designs a complete ILP language bias from raw text samples.
    - **Mechanism**: The Actor receives a small set of text samples, few-shot predicate abstraction examples, and design principles, and outputs a complete predicate system (head predicates, body predicates, arity, type declarations, input/output modes, and global constraints such as `max_vars`/`max_body`). The Critic evaluates the output from both semantic (completeness, redundancy, task relevance) and syntactic (type coverage, arity correctness, solver compatibility) perspectives; if the output fails evaluation, feedback is returned to the Actor for iterative refinement (up to 5 rounds).
    - **Design Motivation**: Actor-Critic collaboration is more reliable than a single-agent approach—the Actor handles creative design while the Critic performs formal verification, thereby avoiding common LLM errors in formal constraints.

2. **Symbolic Knowledge Encoding (Translator Agent)**:

    - **Function**: Translates natural language samples into Prolog facts according to the predicate system.
    - **Mechanism**: Samples are processed in batches (to circumvent context length limitations); the LLM parses each sample according to predicate definitions and maps it to Prolog facts. For example, the text "Shoe_001 is a black formal shoe made of leather" is translated into the fact set `{black(shoe_001), formal_shoes(shoe_001), leather(shoe_001)}`. Translation failures trigger automatic retries (up to 2 times).
    - **Design Motivation**: Batch processing rather than full-context processing addresses context window limitations, and the retry mechanism improves stability.

3. **ILP Learning (MAXSYNTH Solver)**:

    - **Function**: Induces a globally optimal rule set within the predicate space constructed by the LLM.
    - **Mechanism**: MAXSYNTH is a constraint solver based on the Minimum Description Length (MDL) principle that balances rule complexity and noise coverage. It outputs logical rules in Horn clause form, e.g., `suitable_for_business(A) ← expensive(A) ∧ formal_shoes(A)`.
    - **Design Motivation**: The MDL principle renders the solver robust to label noise (remaining significantly better than HypoGeniC under 20% noise), and constraint solving guarantees the global consistency and optimality of the induced rule set.

### Loss & Training
- LLM temperature is set to 0 to reduce generation randomness.
- An 80%/20% train/test split is used; results are averaged over 3 independent dataset generations.
- Solver failures trigger a restart from the predicate design stage (up to 2 times).

## Key Experimental Results

### Main Results

| Method | Model | SHOES Acc | ZENDO Acc | Avg. Acc |
|--------|-------|-----------|-----------|----------|
| IHR | GPT-4o | 96.7% | 50.0% | 73.4% |
| IHR | Claude-3.7 | 98.3% | 60.0% | 79.2% |
| HypoGeniC | GPT-4o | 51.7% | 73.3% | 62.5% |
| HypoGeniC | Claude-3.7 | 75.0% | 68.3% | 71.7% |
| **Ours** | **GPT-4o** | **87.9%** | **76.7%** | **82.3%** |
| **Ours** | **Claude-3.7** | **88.3%** | **81.3%** | **84.8%** |
| **Ours** | **DeepSeek-V3** | **88.3%** | **81.3%** | **84.8%** |
| **Ours** | **Qwen3-32B** | **87.9%** | **80.0%** | **84.0%** |

### Ablation Study (Data Dimension Robustness)

| Variable | Ours | Comparison |
|----------|------|------------|
| No. of rules: 1→3 | Stable, minimal degradation | IHR/HypoGeniC degrade significantly |
| No. of templates: 1→3 | Robust | HypoGeniC most affected |
| Sample size: 50→200 | Excellent at 50 samples | Baselines require 2× more samples |
| Positive ratio: 20%→50% | Stable under class imbalance | Gap widens at low positive ratios |
| Noise: 0%→20% | Some degradation but still outperforms | MAXSYNTH's MDL principle provides robustness |

### Key Findings
- **Model-agnosticism** is the most notable finding: variance across 4 LLMs is below 5%, and Claude and DeepSeek yield identical results, demonstrating that the actual reasoning occurs in the ILP solver rather than the LLM.
- IHR performs near-perfectly on a simple task (SHOES, 98.3%) but collapses to 50–60% on relational reasoning (ZENDO), revealing the limitations of pure LLMs in binary relational reasoning.
- HypoGeniC is highly model-dependent—performance on the same task varies by more than 30% across different LLMs.
- The proposed method reaches near-optimal performance with only 50 samples, demonstrating substantially greater data efficiency than baselines.

## Highlights & Insights
- **Ideal neural-symbolic division of labor** is the core contribution: LLMs handle "understanding" (semantic-to-predicate abstraction and text-to-fact translation), while ILP handles "reasoning" (constrained search and global optimization). This is more robust than approaches such as ChatRule that have LLMs directly generate rules, and has a lower barrier to entry than traditional ILP's reliance on domain experts.
- **Model-agnosticism** is of significant practical value: insensitivity to the LLM backend avoids vendor lock-in and allows the method to benefit automatically from future improvements in LLM capabilities.
- **Automated language bias construction** unlocks ILP applicability in new domains: whereas traditional ILP requires domain experts to spend weeks defining predicates, LLMs can accomplish this in minutes.

## Limitations & Future Work
- Validation is limited to only 2 constructed classification tasks; real-world datasets (e.g., medical or financial domains) are lacking.
- The quality ceiling of the predicate system is bounded by the LLM's conceptual understanding—highly specialized domains may still require human assistance.
- The scalability of the MAXSYNTH solver is not discussed with respect to computational cost for large-scale fact sets and complex rules.
- The translation step may introduce inconsistencies—although a retry mechanism is in place, the translation error rate is not quantified.

## Related Work & Insights
- **vs. HypoGeniC**: HypoGeniC uses LLMs to directly generate natural language hypotheses with counterexample-based iteration, which is flexible but inconsistent (variance > 30%); the proposed method uses ILP to guarantee global optimality.
- **vs. IHR**: IHR employs a propose-select-refine pipeline in which LLMs generate code-based hypotheses, yielding strong performance on simple tasks but weak relational reasoning; the ILP solver in the proposed method is inherently well-suited for relational reasoning.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ LLM-automated ILP language bias is an entirely new direction, pioneering a new paradigm for neural-symbolic hybrid hypothesis generation.
- **Experimental Thoroughness**: ⭐⭐⭐ Limited to 2 constructed tasks; real-world validation is absent.
- **Writing Quality**: ⭐⭐⭐⭐ The pipeline is clearly presented and ablations are comprehensive.
- **Value**: ⭐⭐⭐⭐ Opens a new path for neural-symbolic integration; model-agnosticism is an important practical advantage.

## Additional Notes
- This work demonstrates a new role for LLMs in scientific discovery: rather than solving problems directly, they serve as a bridge that converts unstructured knowledge into formal representations amenable to symbolic reasoning systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Explainable Melanoma Diagnosis with Contrastive Learning and LLM-based Report Generation](explainable_melanoma_diagnosis_with_contrastive_learning_and_llm-based_report_ge.md)
- [\[AAAI 2026\] LLM Circuit Analyses Are Consistent Across Training and Scale](llm_circuit_analyses_consistent_across_training_and_scale.md)
- [\[NeurIPS 2025\] scPilot: Large Language Model Reasoning Toward Automated Single-Cell Analysis and Discovery](../../NeurIPS2025/interpretability/scpilot_large_language_model_reasoning_toward_automated_single-cell_analysis_and.md)
- [\[AAAI 2026\] iMAD: Intelligent Multi-Agent Debate for Efficient and Accurate LLM Inference](imad_intelligent_multi-agent_debate_for_efficient_and_accura.md)
- [\[ICLR 2026\] Semantic Regexes: Auto-Interpreting LLM Features with a Structured Language](../../ICLR2026/interpretability/semantic_regexes_auto-interpreting_llm_features_with_a_structured_language_of_re.md)

</div>

<!-- RELATED:END -->
