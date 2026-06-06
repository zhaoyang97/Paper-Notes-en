---
title: >-
  [Paper Note] Can Large Language Models Generalize Procedures Across Representations?
description: >-
  [ICML2026][Reinforcement Learning][Cross-representation generalization] This paper finds that procedural knowledge learned by LLMs on symbolic representations (code/graphs) cannot reliably transfer to natural language ta…
tags:
  - "ICML2026"
  - "Reinforcement Learning"
  - "Cross-representation generalization"
  - "RL curriculum"
  - "generative analogy"
  - "procedural knowledge transfer"
  - "GRPO"
date: 2026-05-08
content_hash: 261ce24a22600e02
---

# Can Large Language Models Generalize Procedures Across Representations?

**Conference**: ICML2026  
**arXiv**: [2602.03542](https://arxiv.org/abs/2602.03542)  
**Code**: Yes (Dataset and code included with paper)  
**Area**: LLM Reasoning  
**Keywords**: Cross-representation generalization, RL curriculum, generative analogy, procedural knowledge transfer, GRPO  

## TL;DR

This paper finds that procedural knowledge learned by LLMs on symbolic representations (code/graphs) cannot reliably transfer to natural language tasks. It proposes a "symbolic-first, natural-language-second" two-stage RL curriculum strategy, prompting a 1.5B Qwen model to approach zero-shot GPT-4o performance on asynchronous planning tasks. Furthermore, it demonstrates from a cognitive science perspective that successful cross-representation generalization can be interpreted as generative analogy.

## Background & Motivation

**Background**: LLMs extensively utilize symbolic data such as code and graphs during pre-training and post-training. It is widely expected that symbolic training enhances the natural language reasoning capabilities of these models. Recent works (e.g., code-augmented pre-training, graph reasoning) have attempted to augment LLMs with symbolic data, but results have been inconsistent.

**Limitations of Prior Work**: Existing research conflates surface representation factors with deep structural factors, failing to address a critical question: under what conditions does symbolic training actually benefit natural language tasks? Experiments reveal that models trained exclusively on code or graphs perform well in those specific formats but suffer significant performance degradation—often performing worse than untrained baselines—when transferred to natural language.

**Key Challenge**: While symbolic and natural language representations share the same underlying algorithms and procedural structures (isomorphism), LLMs appear to learn surface patterns rather than transferable procedural knowledge. Existing post-training methods (vanilla SFT, distillation, STaR, GRPO) fail to achieve reliable generalization in cross-representation settings.

**Goal**: (1) Isolate procedural generalization from spurious transfer using a strictly isomorphic experimental design; (2) propose a training strategy that enables models to generalize across representations; (3) analyze the mechanisms of generalization from a cognitive science perspective.

**Key Insight**: Drawing from Structure-Mapping Theory in cognitive science, the authors view cross-representation generalization as a generative analogy problem. If a model identifies shared structures across different representations as humans do, it should achieve zero-shot success on new representations.

**Core Idea**: A two-stage RL curriculum—first training on symbolic representations (Graph) to induce abstract procedures, followed by natural language training for adaptation—is used to significantly improve cross-representation generalization.

## Method

### Overall Architecture

The authors construct isomorphic data based on an **asynchronous planning** task: given a set of steps with dependencies (e.g., cooking procedures), the model must calculate the minimum time required to complete all steps with infinite resources. This task is formalized as a critical path calculation on a Directed Acyclic Graph (DAG). Each instance is represented in three isomorphic formats: Natural Language (NL), Graph adjacency lists (Graph), and Python code snippets (Code). The underlying algorithm is identical across formats, differing only in surface form.

The training process consists of two stages: **Stage 1 (Symbolic Induction)** utilizes GRPO on Graph data for 40 steps to allow the model to rapidly learn abstract graph search procedures; **Stage 2 (Natural Language Adaptation)** continues GRPO training on NL data for 40 steps to transfer the procedural knowledge. The total training budget is identical to 80 steps of pure NL training.

### Key Designs

1.  **Isomorphic Data Construction & Representation Isolation**:
    *   Function: Provides strict experimental control to ensure the surface representation is the only variable between formats.
    *   Mechanism: Each natural language planning instance from the AsyncHow dataset is converted into a DAG. These are represented as adjacency list dictionaries (Graph, including START/END sentinel nodes and time-weight dictionaries) and Python longest-path search code (Code, with randomized node indices and units normalized to minutes). All three representations share the exact same graph structure and ground-truth answer.
    *   Design Motivation: Prior work confuses surface differences with structural differences. Isomorphic design solves this causal inference problem by determining if transfer failure is due to "unlearnable procedures" or "surface form interference."

2.  **Two-Stage RL Curriculum (Graph → NL)**:
    *   Function: Maximizes cross-representation generalization performance within a fixed training budget.
    *   Mechanism: Stage 1 applies GRPO ($k=16$ sampling, reward of 1 for correct and format-compliant answers) on Graph data for 40 steps to establish a procedural inductive bias. Stage 2 continues GRPO on NL for 40 steps to adapt the procedure. A key finding is that the **order is irreversible**: the reverse curriculum (NL → Graph, 0.431) is significantly worse than pure NL training (0.698).
    *   Design Motivation: Symbolic training alone does not transfer to NL, and pure NL training is inefficient. Training on Graphs is faster (higher within-representation performance) and provides a superior initialization for subsequent NL training. The reward curve in Stage 2 resembles symbolic training rather than pure NL training, suggesting that symbolic warm-up alters the learning dynamics.

3.  **Analogy Analysis Framework via Structure-Mapping Theory**:
    *   Function: Explains the cognitive mechanism of cross-representation generalization and verifies why the curriculum is effective.
    *   Mechanism: Analogy strength is defined as $$AS(G_b, G_t) = \alpha \cdot sim_u(V_b, V_t) + (1-\alpha) \cdot sim_b(E_b, E_t)$$, where $\alpha=0.4$. Unary similarity ($sim_u$) measures node time distributions via histogram-Jaccard, and binary similarity ($sim_b$) uses Weisfeiler-Lehman subtree kernels (3 iterations). The authors compare the frequency hypothesis (number of related instances in training) against the analogy hypothesis (structural similarity of the most similar training instance) using Pearson $\rho$.
    *   Design Motivation: To distinguish if the model succeeds by aggregating multiple moderately similar instances (frequency learning) or by structural mapping of a few highly similar instances (analogy). Results prove the curriculum encourages analogical behavior.

## Key Experimental Results

### Main Results: Cross-Representation Generalization (Qwen2.5-1.5B-Instruct, GRPO)

| Training Representation | Test NL | Test Graph | Test Code | Description |
| :--- | :--- | :--- | :--- | :--- |
| NL only (80 steps) | 0.698 | - | - | Pure NL Baseline |
| Graph only | High | High | - | Fails to transfer to NL |
| Code only | - | - | High | Fails to transfer to NL |
| **Graph→NL Curriculum (40+40)** | **0.782** | - | - | Outperforms NL-only at same budget |
| NL→Graph (Reverse) | 0.431 | - | - | Order is critical |
| Graph+NL Interleaved | 0.382 | - | - | Inferior to curriculum |

### Curriculum vs. Baselines (NL Test Accuracy)

| Model/Method | NL Accuracy | NL-AAVE Accuracy | Description |
| :--- | :--- | :--- | :--- |
| Qwen-1.5B + Graph→NL Curriculum | **0.782** | **0.573** | Ours |
| Qwen-3B + NL only (40 steps) | 0.471 | 0.400 | 2× Parameters |
| Qwen-7B + NL only (40 steps) | 0.698 | 0.573 | 4.7× Parameters |
| GPT-4o-mini (zero-shot) | 0.440 | 0.289 | Commercial Model |
| GPT-4o (zero-shot) | 0.782 | 0.724 | Commercial Model |

### Ablation Study

| Configuration | NL Accuracy | Description |
| :--- | :--- | :--- |
| Graph(40)→NL(40) GRPO | **0.782** | Full Curriculum |
| NL only (80) GRPO | 0.698 | No symbolic warm-up, -8.4% |
| NL→Graph (Reverse) | 0.431 | Reversed order, -35.1% |
| Graph+NL Interleaved | 0.382 | No stage separation, -40% |
| Code(40)→NL(40) | 0.382 | Code Stage 1 is ineffective |
| Stage 2 Distillation SFT | 0.462 | RL outperforms SFT |
| Standard SFT Curriculum | 0.249 | Curriculum ineffective for SFT |

### Key Findings

*   **Cross-representation generalization generally fails**: Four post-training methods (vanilla SFT, distillation, STaR, GRPO) across three model families failed to transfer from symbolic training to NL, indicating LLMs typically learn surface patterns rather than procedural knowledge.
*   **Curriculum order is vital**: Graph→NL is significantly better than NL→Graph (0.782 vs. 0.431). High efficiency in Graph training establishes a strong procedural foundation; the reverse order forces learning the harder task first, damaging acquired knowledge.
*   **Analogy hypothesis outperforms frequency hypothesis**: In successful generalization settings, the analogy correlation coefficient $\rho_k$ was consistently higher than the frequency coefficient $\rho_p$ (e.g., NL test after curriculum: 0.265 vs. 0.245), suggesting generalization relies on structural mapping.
*   **1.5B model matches GPT-4o**: The curriculum-trained 1.5B Qwen reached 0.782 on NL, matching zero-shot GPT-4o and outperforming the 7B model of the same family, demonstrating efficient parameter utilization.

## Highlights & Insights

*   **Isomorphic experimental design is a methodological highlight**: By constructing isomorphic NL/Graph/Code data, the authors strictly isolated surface differences from structural differences, providing a clean causal inference framework for cross-representation studies. This design is applicable to any "form vs. content" research.
*   **The "easy-to-hard" principle of curriculum learning is effective in RL**: The high information density and training efficiency of the Graph format serve as a warm-up that establishes a procedural inductive bias, altering the optimization dynamics of the subsequent NL phase.
*   **Introduction of a cognitive science perspective is inspiring**: Aligning LLM generalization with human generative analogy using Structure-Mapping Theory provides quantitative depth. Interestingly, LLMs still require significant training to generalize, contrasting with human few-shot analogical abilities.

## Limitations & Future Work

*   **Limited task types**: Experiments focused on asynchronous planning (DAG critical paths). While some math and physics experiments were added, it remains unclear if these findings hold for more complex reasoning (e.g., multi-step games, causal inference).
*   **Only one effective curriculum (Graph→NL) tested**: Code→NL performed poorly, as did Graph+Code→NL, showing sensitivity to the choice of Stage 1 representation. Automated selection of optimal symbolic representations remains an open question.
*   **NL-AAVE gap remains significant**: Post-curriculum NL-AAVE accuracy (0.573) is much lower than standard NL (0.782), suggesting room for improvement in dialect robustness.
*   **Future Directions**: (1) Exploring multi-representation ladder curricula (e.g., Graph→Code→NL); (2) introducing representation diversity during the symbolic stage; (3) combining curriculum strategies with test-time scaling (e.g., best-of-N, self-consistency).

## Related Work & Insights

*   **AsyncHow** (Lin et al., 2024a): Provided the natural language asynchronous planning dataset, formalizing real-world tasks as DAG critical path problems.
*   **DeepSeek-R1** (DeepSeek-AI, 2025): The GRPO method and Qwen base models used here refer to the R1 training paradigm, highlighting RL's superiority in procedural learning.
*   **Structure-Mapping Theory** (Gentner, 1983): The theoretical foundation for the analysis framework, defining constraints for analogy strength (structural consistency, parallel connectivity, systematicity).
*   **SFT memorizes, RL generalizes** (Chu et al., 2025): Consistent with this paper's findings that RL significantly outperforms SFT in cross-representation generalization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Why Does Reinforcement Learning Generalize? A Feature-Level Mechanistic Study of Post-Training in Large Language Models](../../ACL2026/reinforcement_learning/why_does_reinforcement_learning_generalize_a_feature-level_mechanistic_study_of_.md)
- [\[ICML 2026\] The Shape of Reasoning: Topological Analysis of Reasoning Traces in Large Language Models](the_shape_of_reasoning_topological_analysis_of_reasoning_traces_in_large_languag.md)
- [\[ICLR 2026\] Post-training Large Language Models for Diverse High-Quality Responses](../../ICLR2026/reinforcement_learning/post-training_large_language_models_for_diverse_high-quality_responses.md)
- [\[ICML 2026\] Game of Thought: Robust Information Seeking with Large Language Models Using Game Theory](game_of_thought_robust_information_seeking_with_large_language_models_using_game.md)
- [\[ICML 2026\] Break the Block: Dynamic-size Reasoning Blocks for Diffusion Large Language Models via Monotonic Entropy Descent with Reinforcement Learning](break_the_block_dynamic-size_reasoning_blocks_for_diffusion_large_language_model.md)

</div>

<!-- RELATED:END -->
