---
title: >-
  [Paper Note] MCTSr-Zero: Self-Reflective Psychological Counseling Dialogues Generation via Principles and Adaptive Exploration
description: >-
  [AAAI 2026][Medical Imaging][Monte Carlo Tree Search] This paper proposes the MCTSr-Zero framework, which combines MCTS with domain-principle-based self-evaluation and a meta-prompt adaptive exploration mechanism to gene…
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "Monte Carlo Tree Search"
  - "psychological counseling"
  - "dialogue generation"
  - "self-reflection"
  - "domain alignment"
date: 2026-05-08
content_hash: 767503fd8c730823
---

# MCTSr-Zero: Self-Reflective Psychological Counseling Dialogues Generation via Principles and Adaptive Exploration

**Conference**: AAAI 2026
**arXiv**: [2505.23229](https://arxiv.org/abs/2505.23229)  
**Code**: [github](https://github.com/JianChengXingYun/Mctsr-Zero)  
**Area**: Medical Imaging
**Keywords**: Monte Carlo Tree Search, psychological counseling, dialogue generation, self-reflection, domain alignment

## TL;DR
This paper proposes the MCTSr-Zero framework, which combines MCTS with domain-principle-based self-evaluation and a meta-prompt adaptive exploration mechanism to generate high-quality multi-turn psychological counseling dialogue data. The resulting PsyLLM, fine-tuned on this data, achieves state-of-the-art performance on the authors' PsyEval benchmark.

## Background & Motivation

### State of the Field
The integration of MCTS and LLMs has achieved significant breakthroughs on structured tasks such as mathematical reasoning. Meanwhile, the application of LLMs in the mental health domain has given rise to specialized models such as PsyChat, CPsyCounX, and PsyDT, which typically rely on synthetic multi-turn dialogue datasets.

### Limitations of Prior Work

**Difficulty in evaluating open-ended dialogue**: Unlike mathematical tasks with objectively correct answers, the success of psychological counseling depends on subjective factors such as empathic engagement, ethical compliance, and human preference, with no rigorous standard of "correctness."

**Incompatibility of existing MCTS methods**: Outcome-oriented MCTS methods use predefined terminal states as search targets; when applied to open-ended dialogue, they may produce responses misaligned with human expectations.

**Poor principle adherence in LLMs**: LLMs often struggle to deeply understand and consistently follow complex, abstract psychological counseling standards.

**Scarcity of real data**: Authentic psychological counseling dialogue data is extremely scarce, making the quality of synthetic data critical.

**Lack of standardized evaluation**: No benchmark specifically designed for multi-turn psychological counseling dialogue evaluation exists.

### Root Cause
How can the search and planning capabilities of MCTS be applied to open-ended dialogue generation that lacks objectively correct answers?

### Starting Point
The paper introduces the concept of "domain alignment"—shifting the search objective from predefined terminal states to dialogue trajectories that conform to target domain principles (e.g., empathy, ethics). A "regeneration" mechanism and a "meta-prompt adaptation" mechanism substantially expand the search space, enabling MCTS to explore fundamentally different initial dialogue strategies.

## Method

### Overall Architecture
An iterative workflow: (1) initialize meta-prompt and generate initial response → (2) UCT-driven selection: deepen existing paths or regenerate new starting points → (3) self-evaluation based on psychological counseling principles (scoring + critique + suggestions) → (4) backpropagation to update Q-values and meta-prompt → repeat until termination criteria are met.

### Key Designs

1. **Domain-Aligned Principled Self-Evaluation**:

    - Core innovation: Inspired by Constitutional AI, 16 psychological counseling criteria are defined as the AI's "constitution."
    - Each newly generated or refined response undergoes structured evaluation:
        - Constitution-based critique: analyzes compliance with the 16 criteria.
        - Scoring (0–10): based on critique results and degree of criteria adherence.
        - Actionable suggestions: providing directions for improvement.
    - Q-value computation: $Q(a) = \frac{1}{2}(\min R_a + \frac{1}{|R_a|}\sum_{i=1}^{|R_a|} R_a^i)$, balancing average quality with robustness against minimum scores.
    - Multiple sampling for evaluation to enhance robustness.
    - Design motivation: replacing implicit "correctness" standards with explicit principles enables effective MCTS search in open-ended dialogue.

2. **Meta-Prompt Adaptation**:

    - Triggered when root node $P$ is selected by UCT.
    - A candidate meta-prompt is synthesized using the current active meta-prompt and recent evaluation feedback: $m_{cand} \leftarrow \mathcal{M}(m_{activate} \| \mathcal{F}_n)$.
    - Conditional update: the active meta-prompt is updated to the candidate only when $Q(A_{t+1}) \geq Q(P)$.
    - Fundamental distinction from standard MCTS: rather than only deepening the search under a fixed strategy, this mechanism discovers and switches to better initial generation strategies.
    - The search space is expanded from a tree structure to a higher-order cross-distribution space.
    - Design motivation: avoiding local optima caused by commitment to a single initial strategy.

3. **Reflective Self-Refine**:

    - Executed when a response node (non-root $P$) is selected.
    - The specific critique and suggestions $\mathcal{F}$ from the standard evaluation, combined with the active meta-prompt, serve as guidance: $A_{t+1}' = \mathcal{M}(A_t \| \mathcal{F}_t \| m_{activate})$.
    - Design motivation: leveraging targeted feedback from principled evaluation for iterative improvement.

4. **UCT Selection and Search Space Expansion**:

    - UCT formula: $UCT_s = Q(s) + c\sqrt{\frac{\ln N(Parent(s))+1}{N(s)+\epsilon}}$
    - Selection scope includes all response nodes and root node $P$.
    - Selecting a response node → Self-Refine (deepening the path).
    - Selecting root node $P$ → Regeneration + Meta-Prompt Adaptation (broadening the search).
    - Backpropagation:
        - Response nodes: $Q'(p) = \frac{1}{2}(Q(p) + \max_{c \in Children} Q(c))$
        - Root node: $Q(P) = \frac{1}{|\mathcal{A}_{initial}|}\sum_{a \in \mathcal{A}_{initial}} Q(a)$
    - Design motivation: adaptive balance between deepening and broadening.

5. **PsyEval Benchmark**:

    - Systematic scenario generation: 16 categories of psychological distress × 4 scenarios = 64 cases.
    - 16-dimensional evaluation framework: integrating theories including TES, ESHCC, MI, and person-centered therapy.
    - Six newly added key dimensions: dialogue logical consistency, session continuity, resistance handling, ethical/prosocial guidance, summarization, and dialogue pacing.
    - "Fallacy avoidance" is redefined as a hallucination control evaluation.
    - AI Judge mechanism for evaluation, ensuring scalability and consistency.

### Loss & Training
- **MCTSr-Zero-Psy dataset**: 4,000 multi-turn counseling dialogues, 16 categories × ~250 entries, averaging 20 turns.
- **Two-stage PsyLLM training**:
    - SFT: based on GLM-4-32B/9B, 2 epochs, lr=1e-4, 0.1 warmup, AdamW.
    - SimPO alignment: 3 epochs, lr=5e-7, 0.1 warmup.
    - 4 × NVIDIA A800 GPUs.

## Key Experimental Results

### Main Results

| Model | Total Score | ESHCC-R | DLC | CC | RH | Sum. | EPG | DPPA |
|------|-----------|---------|-----|----|----|------|-----|------|
| **PsyLLM-Large** | **90.93** | 54.53 | 4.57 | 4.56 | 4.47 | 4.53 | 4.55 | - |
| **PsyLLM-Mini** | **90.72** | 54.46 | 4.58 | 4.57 | 4.43 | 4.47 | 4.51 | - |
| Claude-3-7-Sonnet | 88.89 | 53.13 | 4.51 | 4.44 | 4.28 | 4.56 | 4.49 | - |
| Gemini-2.5-Pro | 88.62 | 53.01 | 4.53 | 4.48 | 4.33 | 4.34 | 4.36 | - |
| GPT-4.1 | 85.65 | 50.87 | 4.44 | 4.44 | 4.04 | 4.32 | 4.38 | - |
| GPT-4o | 82.31 | 48.71 | 4.28 | 4.18 | 3.87 | 4.25 | 4.24 | - |
| CPsyCounX | 66.00 | 39.99 | 3.37 | 3.24 | 3.01 | 3.82 | 3.31 | - |

### Ablation Study

| Configuration | Iteration 0 | Iteration 1 | Iteration 2 | Iteration 4 |
|------|------------|------------|------------|------------|
| Baseline (gpt-4.1-mini) | 83.60 | - | - | - |
| Self-Refine | - | 86.39 | ~87 | ~88 |
| MCTSr-Zero (w/o meta) | - | ~87 | ~88 | ~89 |
| **MCTSr-Zero (Full)** | - | ~87.5 | ~89 | **90.18** |

### Key Findings

1. **PsyLLM achieves comprehensive superiority**: Both the Large and Mini variants outperform all general-purpose and domain-specific models, including Claude-3-7-Sonnet (88.89) and Gemini-2.5-Pro (88.62).
2. **Balanced capability profile**: PsyLLM leads not only in the empathy dimension but also develops uniformly across dimensions such as logical consistency, continuity, and resistance handling.
3. **Iterative improvement is effective**: Scores improve from baseline 83.60 → 86.39 after 1 iteration → 90.18 after 4 iterations, demonstrating the value of the search mechanism.
4. **Full MCTSr-Zero achieves best performance**: The complete framework consistently outperforms simplified variants and Self-Refine, validating the contributions of meta-prompt adaptation and principled evaluation.
5. **Alignment between training data and evaluation**: The 16 criteria used in MCTSr-Zero are consistent with the PsyEval evaluation dimensions, making the generated training data naturally suited to the evaluation.

## Highlights & Insights
1. The paper transforms MCTS from outcome-oriented search to principle-oriented search, addressing the core challenge of lacking objective evaluation criteria in open-ended dialogue.
2. The meta-prompt adaptation mechanism is a key innovation: it optimizes not only the response content but also the generation strategy itself, enabling higher-order search space exploration.
3. The PsyEval benchmark fills a gap in multi-turn psychological counseling dialogue evaluation; its 16-dimensional evaluation framework is well-designed.
4. A small model (9B) can achieve performance comparable to a large model (90.72 vs. 90.93), highlighting the importance of training data quality.
5. The idea of Constitutional AI is cleverly incorporated into the evaluation component of MCTS.

## Limitations & Future Work
1. **Circular evaluation**: The training data generation criteria and evaluation criteria are highly consistent, potentially introducing self-validation bias.
2. **AI Judge bias**: The framework relies entirely on AI-based evaluation without human validation.
3. **High computational cost**: The multi-iteration search and evaluation in MCTSr-Zero incurs significant overhead.
4. **Limited scenario coverage**: 64 case scenarios may be insufficient to cover the full diversity of psychological counseling.
5. **Safety yet to be validated**: Psychological counseling scenarios demand extremely high safety standards, requiring more rigorous human evaluation.
6. More efficient search strategies and more diverse evaluation dimensions remain avenues for future exploration.

## Related Work & Insights
- **MCTSr (Zhang 2024)**: MCTS for text refinement → extended in this paper to open-ended dialogue.
- **Constitutional AI (Bai 2022)**: principle-driven self-improvement → embedded into MCTS search in this paper.
- **Self-Refine (Madaan 2023)**: LLM self-improvement → this paper performs reflective refinement guided by explicit standards.
- **PsyDT**: personalized counseling style → this paper focuses on principle adherence.

## Rating
- Novelty: ⭐⭐⭐⭐ (domain alignment + meta-prompt adaptation are valuable innovations)
- Experimental Thoroughness: ⭐⭐⭐ (self-constructed evaluation benchmark + self-trained model introduce circular validation risks)
- Writing Quality: ⭐⭐⭐⭐ (clear structure, complete formalization)
- Value: ⭐⭐⭐⭐ (opens a new direction for MCTS applications in open-ended dialogue)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] qa-FLoRA: Data-free Query-Adaptive Fusion of LoRAs for LLMs](qa-flora_data-free_query-adaptive_fusion_of_loras_for_llms.md)
- [\[AAAI 2026\] Self-supervised Multiplex Consensus Mamba for General Image Fusion](self-supervised_multiplex_consensus_mamba_for_general_image_fusion.md)
- [\[AAAI 2026\] SPA: Achieving Consensus in LLM Alignment via Self-Priority Optimization](spa_achieving_consensus_in_llm_alignment_via_self-priority_optimization.md)
- [\[AAAI 2026\] Human-in-the-Loop Interactive Report Generation for Chronic Disease Adherence](human-in-the-loop_interactive_report_generation_for_chronic_disease_adherence.md)
- [\[AAAI 2026\] Shrinking the Teacher: An Adaptive Teaching Paradigm for Asymmetric EEG-Vision Alignment](shrinking_the_teacher_an_adaptive_teaching_paradigm_for_asymmetric_eeg-vision_al.md)

</div>

<!-- RELATED:END -->
