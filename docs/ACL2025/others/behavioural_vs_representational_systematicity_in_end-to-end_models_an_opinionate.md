---
title: >-
  [Paper Note] Behavioural vs. Representational Systematicity in End-to-End Models: An Opinionated Survey
description: >-
  [ACL 2025][Systematicity] This opinionated survey distinguishes between behavioural systematicity (whether a model can generalize correctly to new combinations) and representational systematicity (whether internal representations are structurally compositional). Using Hadley's three-level classification (weak, quasi, and strong), the authors review mainstream benchmarks in both the language and vision domains, revealing that most existing benchmarks only test weak or quasi-sy…
tags:
  - "ACL 2025"
  - "Systematicity"
  - "Compositionality"
  - "Behavioural vs. Representational"
  - "Hadley Classification"
  - "Mechanistic Interpretability"
date: 2026-05-08
content_hash: 6299c6b3031b3f08
---

# Behavioural vs. Representational Systematicity in End-to-End Models: An Opinionated Survey

**Conference**: ACL 2025  
**arXiv**: [2506.04461](https://arxiv.org/abs/2506.04461)  
**Code**: None  
**Area**: Cognitive Linguistics / Compositional Generalization  
**Keywords**: Systematicity, Compositionality, Behavioural vs. Representational, Hadley Classification, Mechanistic Interpretability

## TL;DR

This opinionated survey distinguishes between behavioural systematicity (whether a model can generalize correctly to new combinations) and representational systematicity (whether internal representations are structurally compositional). Using Hadley's three-level classification (weak, quasi, and strong), the authors review mainstream benchmarks in both the language and vision domains, revealing that most existing benchmarks only test weak or quasi-systematicity, and call for bridging the gap between behavioural and representational evaluation through mechanistic interpretability methods.

## Background & Motivation

**Background**: Compositional generalization (e.g., the ability to understand "the fish ate the octopus" after understanding "the octopus ate the fish") is a core capability of human cognition. The machine learning (ML) community has produced numerous benchmarks (such as SCAN, COGS, and BLiMP) and models to test and enhance systematic generalization capabilities, with many works claiming to address the challenges raised by Fodor & Pylyshyn (1988).

**Limitations of Prior Work**: A critical confusion has been widely overlooked: Fodor & Pylyshyn (F&P) argued for **representational systematicity** (internal representations must possess structured compositional operations), whereas existing benchmarks and evaluations primarily test **behavioural systematicity** (whether a model can produce the correct output for novel inputs). Behavioral success does not imply structured representations; models can pass tests via memorization, heuristics, or shortcuts without possessing genuine systematic representations.

**Key Challenge**: The ML community often equates success in behavioural tests to solving F&P's challenge of representational systematicity. This leads to overestimations of model generalization capabilities and conflicting research findings.

**Goal**: To clarify the distinction between behavioural and representational systematicity, analyze the actual testing scope of existing benchmarks, and outline a path toward strong systematicity evaluation.

**Key Insight**: Introducing Hadley's (1994) three-level systematicity classification as an analytical tool, while incorporating theoretical traditions of operationalization and the competence-performance distinction from psychology.

**Core Idea**: Correct model behavior does not imply "understanding"; behavioural evidence must be supplemented with representational evidence derived from mechanistic interpretability.

## Method

### Survey Content Overview

As an opinionated survey, this paper presents its arguments across three structured parts:

1. **Historical Evolution from Representation to Behaviour**:

    - Tracing the original argument of F&P (1988): Systematicity is an entailment relation between cognitive abilities (understanding "aRb" entails understanding "bRa"), which requires representations to have structure and structure-sensitive operations.
    - Introducing Hadley's (1994) operationalization framework, which categorizes systematicity into three progressive levels:
        - **Weak Systematicity**: Familiar words in new combinations but restricted to syntactic positions encountered during training.
        - **Quasi-Systematicity**: Weak systematicity plus recursion/embedded clauses (where simple sentences of similar structures were seen during training).
        - **Strong Systematicity**: Words appearing in syntactic positions never encountered during training (closest to human capabilities).
    - Discussing the classical distinction between competence and performance: behavioural failure does not necessarily imply a lack of competence (e.g., object permanence tests in infants), and success does not necessarily prove competence (e.g., animals might solve matching tasks through entropy detection rather than relational reasoning).

2. **Systematicity Level Analysis of Language and Vision Benchmarks**:

    - **SCAN**: Split 1 likely tests only weak systematicity (the training set already covers all commands in all positions); Split 2 tests productivity (requiring at least quasi-systematicity); Split 3 targets strong systematicity, but Bastings et al. (2018) demonstrated that simple modeling tricks are sufficient to solve it.
    - **PCFG SET**: The Systematicity split tests weak systematicity, while the Productivity split tests at least quasi-systematicity.
    - **COGS/ReCOGS/SLOG**: These target strong systematicity; however, ReCOGS significantly boosted the performance of the same model simply by removing irrelevant output tokens and mitigating spurious correlations, suggesting that behavioural test results on COGS are unreliable. SLOG stands as the most rigorous language benchmark currently available.
    - **Vision Benchmarks**: Methods disentangling generative factors (such as dSprites) can only test weak systematicity; abstract reasoning (such as ARC) might test strong systematicity but lacks a systematic construction process; evaluating vision-language models is hindered because pre-training data is unknown, making it impossible to determine the generalization level.

3. **Evidence and Counter-Evidence of Representational Systematicity**:

    - **Supporting Evidence**: Linear probes discovering linguistic concepts, OthelloGPT's "world model," the recombinability of function vectors, and binding vectors in large LLMs.
    - **Counter-evidence**: Features decoded by probing are not necessarily used causally by the model; Transformer "world models" can be far less coherent than they appear (e.g., taxi-cab navigation path experiments); even if weak systematic representations are detected, they are not necessarily utilized for OOD generalization.
    - There exists a tension between compression efficiency and complete compositionality: models might sacrifice fully systematic representations to optimize data encoding efficiency.

### Core Argumentative Framework

The authors present three analytical cases to clarify the relationships among behaviour, representation, and operationalization:

1. **Case 1**: Systematic behavior without valid operationalization → No conclusions can be drawn.
2. **Case 2**: Systematic behavior under valid operationalization → Constitutes evidence of representational systematicity if the F&P position is accepted, though mechanistic interpretability is required for confirmation.
3. **Case 3**: No systematic behavior under valid operationalization → May indicate a lack of representations, or representations may exist but execution is blocked by a non-systematic module within the pipeline.

## Key Experimental Results

### Systematicity Level Analysis of Language Benchmarks

| Benchmark | Target Level | Actual Tested Level | Limitations |
|------|---------|------------|------|
| SCAN Split 1 | Weak | Weak | Handled by simple engineering/modeling tricks |
| SCAN Split 3 | Strong | Strong (Targeted) | Solvable without structured representations |
| PCFG SET Systematicity | Weak | Weak | Match in grammar parameters between train and test |
| COGS | Strong | Uncertain | ReCOGS demonstrates vulnerability to spurious factors |
| SLOG | Strong | Strong | The most rigorous language benchmark currently available |

### Systematicity Level Analysis of Vision Benchmarks

| Method Category | Representative Benchmark | Systematicity Level | Limitations |
|---------|---------|----------|------|
| Generative factor disentanglement | dSprites variants | Weak | No hierarchical structure, no novel contexts |
| Abstract reasoning | ARC, RAVEN | Potentially Strong | Lack of systematic construction description |
| Vision-language | Winoground | Unassessable | Pre-training data is unknown |

### Key Findings
- Most benchmarks designed to "test systematicity" actually evaluate only weak or quasi-systematicity.
- The contrasting performance of the same model on COGS and ReCOGS demonstrates that behavioural test results are heavily skewed by spurious factors in dataset construction.
- Rankings of the same group of models can be inconsistent across different compositional generalization datasets.
- Learning curve analysis shows that non-systematic learners require data that scales exponentially with the number of concepts.

## Highlights & Insights
- **Conceptual Clarification of Behaviour vs. Representation**—Shattering the ML community's common assumption that "correct output equates to understanding." This distinction has profound implications for all NLP evaluation paradigms.
- **Modern Application of Hadley's Three-Level Classification**—Applying a theoretical framework from 1994 to 2020s benchmark evaluations, establishing a standardized analytical tool to categorize systematicity testing.
- **Integration of Cross-Disciplinary Perspectives**—Bringing psychological theories of operationalization and the competence-performance distinction into ML evaluation, providing a deeper theoretical foundation for evaluation methodologies.

## Limitations & Future Work
- The paper is a pure survey without new experiments, relying on reinterpreting existing works for its claims.
- It focuses heavily on phrase structure grammar frameworks, leaving alternative theoretical perspectives like Tree-Adjoining Grammar or Construction Grammar unexplored.
- Mechanistic interpretability methods are still in their infancy; the paper does not prescribe concrete operational evaluation protocols.
- It focuses solely on English benchmarks, whereas multilingual models might exhibit different generalization capacities.
- The scaling effect remains undiscussed: it is unclear whether the relationship between behaviour and representation changes as model size scales up.

## Related Work & Insights
- **vs Russin et al. (2024)**: Also reviews compositionality, but their core question centers on "whether models can replicate behavioural features," remaining on a behavioural plane. This work explicitly adds the representational dimension.
- **vs McCurdy et al. (2024)**: Defines "compositional behavior" but decouples it from the learning process, failing to determine systematicity. This work stresses that the training distribution must be strictly controlled.
- **vs Pavlick (2023)**: Argues that LLMs can already encode symbols and engage in symbolic processing. This work cautions that even if representations appear structured, there is no guarantee they are utilized systematically.

## Rating
- Novelty: ⭐⭐⭐⭐ The application of the behaviour/representation distinction and Hadley's classification to ML is a valuable conceptual contribution.
- Experimental Thoroughness: ⭐⭐ Pure survey with no new experiments, though the analysis is broad in scope.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear and compelling arguments, robust logical reasoning, and precise conceptual boundaries.
- Value: ⭐⭐⭐⭐ Offers crucial theoretical insights for compositional generalization research and LLM evaluation methodologies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] CyberGym-E2E: Scalable Real-World Benchmark for AI Agents' End-to-End Cybersecurity Capabilities](../../ICML2026/others/cybergym-e2e_scalable_real-world_benchmark_for_ai_agents_end-to-end_cybersecurit.md)
- [\[ACL 2025\] Hybrid Preferences: Learning to Route Instances for Human vs. AI Feedback](hybrid_preferences_learning_to_route_instances_for_human_vs_ai_feedback.md)
- [\[ACL 2025\] The Hidden Attention of Mamba Models](the_hidden_attention_of_mamba_models.md)
- [\[ACL 2025\] Advancing Sequential Numerical Prediction in Autoregressive Models](advancing_sequential_numerical_prediction_in_autoregressive_models.md)
- [\[ACL 2025\] Using Shapley Interactions to Understand How Models Use Structure](using_shapley_interactions_to_understand_how_models_use_structure.md)

</div>

<!-- RELATED:END -->
