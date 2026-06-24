---
title: >-
  [Paper Note] Curriculum Abductive Learning
description: >-
  [NeurIPS 2025][Reasoning][Abductive Learning] This paper proposes Curriculum Abductive Learning (C-ABL), which partitions a knowledge base into sub-knowledge-bases according to its dependency structure and introduces them progressively during training. This substantially reduces the abduction search space in ABL, significantly improving training stability, convergence speed, and final accuracy.
tags:
  - "NeurIPS 2025"
  - "Reasoning"
  - "Abductive Learning"
  - "Curriculum Learning"
  - "Knowledge Base Partitioning"
  - "Neuro-Symbolic Reasoning"
  - "Training Stability"
date: 2026-05-08
content_hash: c5ee62f3ab4bea26
---

# Curriculum Abductive Learning

**Conference**: NeurIPS 2025
**arXiv**: [2505.12275](https://arxiv.org/abs/2505.12275)  
**Code**: None  
**Area**: LLM Reasoning / Neuro-Symbolic Learning
**Keywords**: Abductive Learning, Curriculum Learning, Knowledge Base Partitioning, Neuro-Symbolic Reasoning, Training Stability

## TL;DR
This paper proposes Curriculum Abductive Learning (C-ABL), which partitions a knowledge base into sub-knowledge-bases according to its dependency structure and introduces them progressively during training. This substantially reduces the abduction search space in ABL, significantly improving training stability, convergence speed, and final accuracy.

## Background & Motivation

**Background**: Abductive Learning (ABL) is a framework that integrates machine learning with logical reasoning. A learning model predicts symbolic concept labels from raw inputs, while a reasoning module generates candidate revised labels consistent with the knowledge base via abduction, which are then fed back to retrain the model. This loop enables the system to progressively align predictions with logical constraints.

**Limitations of Prior Work**: Due to the inherent uncertainty of abduction, the abduction search space $|\mathbb{S}|$ can grow exponentially (up to $N^m$, where $N$ is the number of concept labels and $m$ is the sequence length) as the knowledge base becomes more complex. A large number of "plausible but incorrect" candidate labels mislead training, causing the model to oscillate among different hypotheses, resulting in slow and unstable convergence.

**Key Challenge**: Existing methods (ABL, A3BL, etc.) focus on improving candidate selection strategies (consistency optimization), but when the search space itself is too large, even better selection strategies cannot address the fundamental bottleneck.

**Goal**: To actively manage and reduce the size of the abduction search space, rather than merely making better selections within a large space.

**Key Insight**: Many knowledge bases possess an inherent staged or hierarchical structure (e.g., simple foundational clauses vs. complex exception clauses in legal domains). This structure can be exploited to introduce the knowledge base in stages of increasing complexity.

**Core Idea**: Partition the knowledge base into a sequence of sub-knowledge-bases of increasing complexity according to its dependency graph, and introduce them progressively during training, thereby substantially reducing the search space at each stage.

## Method

### Overall Architecture
C-ABL restructures the standard ABL training process into $P$ curriculum stages. Given the full knowledge base $\mathcal{KB}$, a partitioning algorithm first splits it into a sequence of sub-knowledge-bases $\mathcal{KB}_1, \ldots, \mathcal{KB}_P$. At stage $p$, only $\mathcal{KB}_p$ is used to guide abductive reasoning, and the scope is progressively expanded until the full knowledge base is in use.

### Key Designs

1. **Dependency Graph Construction and Knowledge Base Partitioning (Algorithm 1)**:

    - **Function**: Partition $\mathcal{KB}$ into an ordered sequence of sub-knowledge-bases.
    - **Mechanism**: A dependency graph $G=(V,E)$ is first constructed over rules, where each node represents a rule and an edge is added from $r_i$ to $r_j$ if the head predicate of $r_i$ appears in the body of $r_j$. Partitioning then follows three principles: (1) **dependency cohesion**—mutually dependent rules are grouped into the same stage; (2) **increasing complexity**—simpler rules are introduced first; (3) **self-contained inference**—each stage's sub-knowledge-base supports independent reasoning. The algorithm initializes clusters for each concept label, recursively expands them, merges duplicates, and applies topological sorting to form a cumulative sub-knowledge-base sequence.
    - **Design Motivation**: By exploiting the symbolic structure of the knowledge base (rather than treating it as a black box), the complexity of the search space can be actively controlled.

2. **Curriculum-Guided Training**:

    - **Function**: Replace the full $\mathcal{KB}$ with $\mathcal{KB}_p$ for ABL training at stage $p$.
    - **Mechanism**: The abduction space at stage $p$ is $\mathbb{S}_p = \{\mathbf{z} \in \mathcal{Z}_p^m \mid \mathbf{z} \wedge \mathcal{KB}_p \models y\}$, with an upper bound of $|\mathcal{Z}_p \setminus \mathcal{Z}_{p-1}|^m$, which is far smaller than the full space of $N^m$. Once the prediction accuracy of all concept labels in stage $p$ exceeds the random-guessing baseline $1/|\mathcal{Z}|$, training advances to the next stage.
    - **Design Motivation**: Each stage only needs to handle newly introduced concept labels, while previously learned concepts serve as a stable foundation, exponentially reducing the search space.

3. **Theoretical Guarantees for Self-Contained Inference**:

    - **Function**: Ensure that each stage's sub-knowledge-base is logically complete and consistent within its concept domain.
    - **Mechanism**: Theorem 3.2 proves that $\mathcal{KB}_p$ simultaneously satisfies soundness (no globally invalid conclusions are derived) and completeness (all valid conclusions can be derived) for any formula $\varphi$ over $\mathcal{Z}_p$. Corollary 3.3 proves that the final stage $\mathcal{KB}_P$ is logically equivalent to the full $\mathcal{KB}$ over all concept labels.
    - **Design Motivation**: Prevents logical inconsistencies introduced by staged training, ensuring that each stage's "simplified" reasoning is fully trustworthy.

### Loss & Training

- **Stage Transition Criterion**: All current concept label accuracies exceed $1/|\mathcal{Z}|$ (evaluated on a validation set only, not used for training).
- **Convergence Guarantee**: Theorem 4.5 proves that the iteration complexity of ABL is $\mathcal{O}(|\mathbb{S}|^2 \cdot \rho^2 / \varepsilon^2)$; reducing $|\mathbb{S}|$ directly accelerates convergence.
- **Smooth Transition**: Theorem 4.6 (logical consistency) and Theorem 4.7 (topological continuity via nested Stone spaces) jointly guarantee that catastrophic forgetting does not occur across stages.

## Key Experimental Results

### Main Results: Multi-Digit Addition

| Task | ABL | A3BL | C-ABL | Gain |
|------|-----|------|-------|------|
| Decimal $d=2$ | 74.49% | 75.62% | **76.74%** | +1.1% |
| Decimal $d=3$ | 74.50% | 74.65% | **77.65%** | +3.0% |
| Decimal $d=4$ | 73.05% | 73.28% | **76.30%** | +3.0% |
| Hexadecimal $d=1$ | 60.87% | 22.45% | **63.02%** | +2.2% |
| Hexadecimal $d=2$ | 61.75% | 60.75% | **64.25%** | +2.5% |
| Hexadecimal $d=3$ | 64.14% | 65.81% | **66.67%** | +0.9% |

In terms of training time, C-ABL requires only 132.6 minutes on decimal $d=4$ compared to 253.6 minutes for ABL (a 48% reduction).

### Ablation Study: Different Tasks

| Task | ABL | A3BL | C-ABL |
|------|-----|------|-------|
| Chess Attack (Accuracy) | 73.75% | 74.96% | **86.79%** |
| Chess Attack (Convergence Iterations) | 4560 | 3389 | **2553** |
| Judicial Sentencing 10% Labels (F1) | 0.895 | - | **0.904** |
| Judicial Sentencing 50% Labels (F1) | 0.910 | - | **0.920** |

### Key Findings
- **Greater gains under higher complexity**: C-ABL's advantage is more pronounced in hexadecimal and higher-digit settings, where the knowledge base is larger and the search space is greater.
- **Most significant improvement on the chess task** (+12% accuracy), as the binary target labels induce an extremely large abduction space, making C-ABL's partitioning strategy most effective.
- **Smoother training curves**: C-ABL achieves rapid improvement within the first few hundred iterations and approaches its final performance by iteration 1000, whereas ABL exhibits pronounced oscillation and stagnation.
- **No catastrophic forgetting across stages**: The prediction accuracy of previously learned concept labels remains stable after new stages are introduced.

## Highlights & Insights
- **Paradigm shift from "select better" to "search less"**: Prior work focuses on making better selections within a large search space; C-ABL directly reduces the search space itself. This idea is simple yet profound—rather than finding a needle in the ocean, one first shrinks the ocean.
- **The symbolic structure of knowledge bases is untapped free information**: Knowledge bases have traditionally been treated as black boxes in ABL, but their internal dependency structure naturally provides a basis for curriculum stage division. This observation transfers to any learning system that uses structured knowledge.
- **Comprehensive theoretical analysis**: The paper establishes a complete theoretical chain spanning search space reduction (Thm. 4.2), iteration complexity (Thm. 4.5), and stage-transition smoothness (Thm. 4.6–4.7).

## Limitations & Future Work
- The partitioning algorithm relies on the knowledge base having a well-defined hierarchical or staged structure; its effectiveness may be limited for highly coupled knowledge bases that cannot be partitioned effectively.
- The stage transition criterion (accuracy exceeding $1/|\mathcal{Z}|$) is relatively simple and may not be optimal—transitioning too early or too late can adversely affect performance.
- Experiments are conducted primarily on synthetic tasks (digit addition, chessboard); only one real-world task (judicial sentencing) is included, at a relatively small scale.
- No comparison with neuro-symbolic methods that integrate modern large language models.

## Related Work & Insights
- **vs. ABL/A3BL**: These methods improve selection strategies within a fixed, complete search space; C-ABL reduces the search space at its source, representing a more fundamental approach.
- **vs. Data-Level Curriculum Learning (Bengio 2009, etc.)**: Traditional curriculum learning applies easy-to-hard ordering on the data side; C-ABL applies ordering on the knowledge/logic side, introducing a new dimension.
- **vs. NeurASP/DeepProbLog**: These probabilistic neuro-symbolic methods perform inference via weighted model counting at high computational cost; C-ABL reduces cost by shrinking the inference space.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Applying curriculum learning to knowledge base partitioning is a novel perspective, though the core idea (reducing the search space) is fairly intuitive.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three tasks plus theoretical analysis, but real-world tasks are few and small in scale.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Writing is clear, with a smooth logical flow from motivation → method → theory → experiments.
- **Value**: ⭐⭐⭐⭐ A clear contribution to the ABL community, though ABL itself has a relatively limited application scope.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Curriculum Reinforcement Learning from Easy to Hard Tasks Improves LLM Reasoning](../../ICLR2026/llm_reasoning/curriculum_reinforcement_learning_from_easy_to_hard_tasks_improves_llm_reasoning.md)
- [\[ACL 2025\] Commonsense Abductive Reasoning using Knowledge from Multiple Sources](../../ACL2025/llm_reasoning/commonsense_abductive_reasoning_using_knowledge_from_multiple_sources.md)
- [\[ICLR 2026\] Overthinking Reduction with Decoupled Rewards and Curriculum Data Scheduling](../../ICLR2026/llm_reasoning/overthinking_reduction_with_decoupled_rewards_and_curriculum_data_scheduling.md)
- [\[ACL 2026\] GanitLLM: Difficulty-Aware Bengali Mathematical Reasoning through Curriculum-GRPO](../../ACL2026/llm_reasoning/ganitllm_difficulty-aware_bengali_mathematical_reasoning_through_curriculum-grpo.md)
- [\[AAAI 2026\] NeSTR: A Neuro-Symbolic Abductive Framework for Temporal Reasoning in Large Language Models](../../AAAI2026/llm_reasoning/nestr_a_neuro-symbolic_abductive_framework_for_temporal_reasoning_in_large_langu.md)

</div>

<!-- RELATED:END -->
