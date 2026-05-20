---
title: >-
  [Paper Note] Auditing Meta-Cognitive Hallucinations in Reasoning Large Language Models
description: >-
  [NeurIPS 2025][Interpretability][hallucination] This paper systematically audits the generation and propagation mechanisms of hallucinations in reasoning large language models (RLLMs)…
tags:
  - "NeurIPS 2025"
  - "Interpretability"
  - "hallucination"
  - "reasoning LLM"
  - "Chain-of-Thought"
  - "metacognition"
  - "reflection"
date: 2026-05-08
content_hash: e81bda84e0af4127
---

# Auditing Meta-Cognitive Hallucinations in Reasoning Large Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2505.13143](https://arxiv.org/abs/2505.13143)  
**Code**: [https://github.com/](https://github.com/) (available)  
**Area**: Interpretability
**Keywords**: hallucination, reasoning LLM, Chain-of-Thought, metacognition, reflection

## TL;DR
This paper systematically audits the generation and propagation mechanisms of hallucinations in reasoning large language models (RLLMs), finding that reflection in long CoT amplifies hallucinations through metacognitive bias rather than correcting them. Even targeted interventions at the hallucination source fail to alter final outputs (chain disloyalty), exposing critical shortcomings of existing hallucination detection methods in multi-step reasoning scenarios.

## Background & Motivation

**Background**: Reasoning large language models (e.g., DeepSeek-R1, O1) have substantially improved multi-step reasoning through long CoT and self-reflection mechanisms, yet these same mechanisms render hallucinations more covert—reasoning traces appear coherent while factual errors accumulate silently across multiple steps.

**Limitations of Prior Work**: (1) Existing hallucination detection methods (perplexity, hidden-state analysis, self-verification) are surface-level and do not analyze how hallucinations arise and propagate within reasoning chains; (2) circuit tracing requires access to model parameters and is inapplicable to black-box models; (3) controlled experimental environments for systematically studying hallucinations are lacking.

**Key Challenge**: The reflection mechanism of RLLMs is designed to correct errors, yet in practice it may amplify them—because the model's metacognitive confidence is decoupled from factual correctness (high confidence does not imply correctness).

**Goal**: (1) How can hallucinations be reproduced and categorized in a controlled environment? (2) How does the reflection mechanism interact with hallucinations? (3) Can interventions effectively correct hallucinations? (4) Are existing detection methods reliable?

**Key Insight**: A controlled knowledge domain based on RFC documents is constructed; hallucinations are categorized into Type I (seen during training but not learned) and Type II (unseen during training). CoT traces are audited with respect to knowledge flow, reflection patterns, and metacognitive confidence to analyze hallucination mechanisms.

**Core Idea**: Interpretable long-chain hallucination attribution is achieved under a black-box setting through CoT trace auditing, revealing the dual role of the reflection mechanism in hallucination propagation.

## Method

### Overall Architecture
Construct a controlled knowledge domain (RFC documents) → Generate CoT samples with and without hallucinations → Model knowledge flow and reflection behavior within CoT → Audit hallucination generation, propagation, and amplification → Evaluate intervention effectiveness and detection methods.

### Key Designs

1. **Hallucination Taxonomy (Type I / Type II)**:

    - Function: Categorize hallucinations into two types based on knowledge provenance.
    - Mechanism: Type I—knowledge exists in the training data but was not correctly learned by the model (seen but unlearned); the model exhibits overconfidence in knowledge it has not internalized. Type II—knowledge is absent from the training data (unseen or incorrect); the model fabricates it. Formally: $k \in \mathcal{D}$ but $k \notin \mathcal{K}_\mathcal{M}$ (Type I) vs. $k \notin \mathcal{D}$ (Type II).
    - Design Motivation: Hallucinations of different origins may exhibit distinct propagation mechanisms and require different correction strategies.

2. **CoT Knowledge Flow Modeling**:

    - Function: Formalize long CoT as a reasoning graph to trace how knowledge flows between steps.
    - Mechanism: Each reasoning node $c_i$ is an atomic claim that may originate from internal generation or external knowledge injection $k_i \to ck_i$. Reflection links refl($c_p = c_q$) denote the model revisiting earlier reasoning steps. Reflection yields one of three outcomes: validation-and-retention, revision-and-update, or rejection-and-termination.
    - Design Motivation: Structured modeling is necessary to track where hallucinations originate, how they propagate, and what role reflection plays in this process.

3. **Metacognitive Confidence Model**:

    - Function: Model the model's subjective assessment of its own knowledge state during reflection (as opposed to factual correctness).
    - Mechanism: conf(c) is defined as the model's metacognitive confidence in claim $c$ (i.e., the model's belief that it knows $c$, regardless of whether $c$ is correct). The paper proposes Prompt-Aligned Belief Adaptation: during reflection, the model tends to adjust confidence in the direction semantically aligned with user input, causing the confidence of erroneous claims to increase rather than decrease. The confidence update is formulated as $\Delta\text{conf} = \alpha \cdot f(c_{p-1}, c_q) + (1-\alpha) \cdot g(c_q, \text{prompt})$.
    - Design Motivation: Explains why reflection does not necessarily correct errors—if an erroneous claim is semantically aligned with the prompt, reflection reinforces confidence in it.

4. **RFC Controlled Knowledge Domain**:

    - Function: Construct a knowledge environment satisfying bounded scope and verifiability based on RFC documents.
    - Mechanism: RFCs are technical specification documents with well-defined boundaries and verifiable ground truth. A dataset of 1,515 questions × 5 answers = 7,575 samples is constructed, including Type I/II hallucination groups and control groups. Samples are filtered through multi-round generation and consistency verification.
    - Design Motivation: An environment with clear knowledge boundaries and verifiable answers is necessary to rigorously control experimental variables.

### Loss & Training
This paper is an analytical study and does not involve model training. DeepSeek-R1 is used as the primary test model.

## Key Experimental Results

### Main Results: Hallucination Behavior Analysis

| Metric | Type I (Hallucination) | Type I Control | Type II (Hallucination) | Type II Control |
|--------|------------------------|----------------|--------------------------|-----------------|
| Sample count (questions) | 439 | 500 | 484 | 92 |
| Avg. CoT length (tokens) | 1409 | 1029 | 1173 | 1254 |
| Hallucinated claim ratio | 12.78% | 0.68% | 18.14% | — |
| Avg. depth of hallucinated claims | 38.10 | 11.53 | 24.42 | — |
| Avg. reflection count | 9.33 | 4.40 | 7.12 | — |
| Hedging word frequency | 37.14 | 16.92 | 25.67 | — |
| Hesitation word frequency | 27.85 | 12.73 | 15.83 | — |

### Ablation Study: Intervention Experiments

| Intervention Position | Intervention Accepted? | CoT Changed? | Answer Changed? | Still Hallucinated? |
|-----------------------|------------------------|--------------|-----------------|---------------------|
| Edit 1 (early) | 83.5% | 98.5% | 98.5% | 77.5% |
| Edit 2 (middle) | 65% | 97.5% | 95% | 70% |
| Edit 3 (late) | 65% | 99% | 90% | 85% |
| Control group | 53.3% | 96.6% | 23% | 20% |

### Key Findings
- **Reflection amplifies hallucinations rather than correcting them**: The reflection frequency in hallucination groups is 2.12× that of control groups; hedging word frequency is 220% higher and hesitation word frequency 219% higher, yet these reflections reinforce rather than correct errors.
- **Chain Disloyalty**: Even when interventions at the hallucination source are accepted (83.5%), 77.5% of cases still produce hallucinated outputs—the reasoning chain resists correction and maintains its erroneous trajectory.
- **Internally generated errors dominate**: In Type II, only 25.93% of externally injected errors are adopted, yet the model itself generates an average of 5.25 additional internal erroneous knowledge items—the model does not merely replicate errors but actively creates new ones.
- **Existing detection methods fail**: The best-performing detection method achieves only 78.95% accuracy at extremely high computational cost; alternative methods yield AUROC below 55%.
- **Over-alignment**: Across CoT traces exceeding 1,000 tokens, the hallucination pass-through rate reaches 62.54% (Type I) and 56.08% (Type II), while the proportion of cases successfully resisting misleading inputs is only 10.66%.

## Highlights & Insights
- **The decoupling of metacognitive confidence from factual correctness** is the paper's most profound insight: a model can be highly confident in an erroneous claim because its confidence derives from semantic alignment with the prompt rather than from factual verification. This explains many apparently paradoxical phenomena.
- **Chain Disloyalty** is a particularly illuminating finding: even after correcting an upstream error, the reasoning chain maintains its original erroneous trajectory. This may reflect a form of inertia or preference formed during the reasoning process that cannot be disrupted by single-point interventions.
- **The RFC controlled knowledge domain experimental design** offers a replicable template, providing a rigorous experimental framework for black-box analysis.

## Limitations & Future Work
- **Single model evaluated (DeepSeek-R1)**: Hallucination propagation patterns may differ across RLLMs.
- **Narrow domain coverage (RFC)**: Reasoning patterns in technical specification documents may not generalize to other domains (e.g., commonsense reasoning, mathematical reasoning).
- **Coarse-grained interventions**: Edits are applied only at the individual claim level; simultaneous multi-point or structurally coordinated interventions have not been explored.
- **Recommended directions**: Develop hallucination detection methods specifically targeting long CoT, particularly those capable of detecting metacognitive bias.

## Related Work & Insights
- **vs. Circuit Tracing**: Circuit tracing requires white-box access to model parameters; the CoT auditing approach proposed here is applicable to black-box models and thus offers broader generalizability.
- **vs. FActScore / Self-Verification**: These methods perform surface-level detection; this paper provides a deeper analysis of hallucination propagation dynamics within reasoning chains.
- **vs. Sparse Autoencoders**: SAEs identify features but do not establish causal relationships; this paper establishes causal links through intervention experiments.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Metacognitive perspective on RLLM hallucinations; Chain Disloyalty is a novel concept
- Experimental Thoroughness: ⭐⭐⭐⭐ Rigorous experimental design with well-controlled environment, but limited model and domain coverage
- Writing Quality: ⭐⭐⭐⭐ Formal modeling is clear, though notation is occasionally dense
- Value: ⭐⭐⭐⭐⭐ Reveals deep mechanisms of RLLM hallucinations with important implications for safety and reliability research

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Table as a Modality for Large Language Models](table_as_a_modality_for_large_language_models.md)
- [\[NeurIPS 2025\] The Trilemma of Truth in Large Language Models](the_trilemma_of_truth_in_large_language_models.md)
- [\[NeurIPS 2025\] scPilot: Large Language Model Reasoning Toward Automated Single-Cell Analysis and Discovery](scpilot_large_language_model_reasoning_toward_automated_single-cell_analysis_and.md)
- [\[NeurIPS 2025\] Cognitive Mirrors: Exploring the Diverse Functional Roles of Attention Heads in LLM Reasoning](cognitive_mirrors_exploring_the_diverse_functional_roles_of_attention_heads_in_l.md)
- [\[AAAI 2026\] Beyond Hallucinations: A Composite Score for Measuring Reliability in Open-Source Large Language Models](../../AAAI2026/interpretability/beyond_hallucinations_a_composite_score_for_measuring_reliability_in_open-source.md)

</div>

<!-- RELATED:END -->
