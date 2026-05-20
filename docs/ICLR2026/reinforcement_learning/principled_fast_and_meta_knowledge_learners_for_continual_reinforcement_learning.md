---
title: >-
  [Paper Note] Principled Fast and Meta Knowledge Learners for Continual Reinforcement Learning
description: >-
  [ICLR 2026][Reinforcement Learning][continual RL] Inspired by the hippocampus–neocortex interaction in the human brain, this paper proposes FAME…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "continual RL"
  - "catastrophic forgetting"
  - "knowledge transfer"
  - "dual-learner"
  - "meta learning"
date: 2026-05-08
content_hash: be625c2cb4dd37f9
---

# Principled Fast and Meta Knowledge Learners for Continual Reinforcement Learning

**Conference**: ICLR 2026
**arXiv**: [2603.00903](https://arxiv.org/abs/2603.00903)  
**Code**: [GitHub](https://github.com/datake/FAME)  
**Area**: Reinforcement Learning
**Keywords**: continual RL, catastrophic forgetting, knowledge transfer, dual-learner, meta learning

## TL;DR

Inspired by the hippocampus–neocortex interaction in the human brain, this paper proposes FAME, a dual-learner framework for continual reinforcement learning that employs a fast learner for knowledge transfer and a meta learner for knowledge consolidation, achieving efficient continual RL while principally minimizing catastrophic forgetting.

## Background & Motivation

**Background**: Traditional deep RL algorithms are designed for single-task settings. When confronted with a sequential stream of tasks, they must balance plasticity (rapid adaptation to new tasks) and stability (retention of prior knowledge).

**Limitations of Prior Work**: Existing continual RL methods largely rely on heuristics or independent designs from disparate perspectives, lacking a unified theoretical framework to analyze when knowledge transfer is beneficial and how forgetting can be quantified.

**Key Challenge**: Direct fine-tuning leads to negative transfer and performance degradation when tasks differ substantially, whereas training from scratch (reset) fails to exploit previously accumulated knowledge.

**Neuroscientific Motivation**: In the human brain, the hippocampus rapidly encodes new experiences while the neocortex performs gradual knowledge consolidation. This division of labor provides a biological basis for algorithm design.

**Limitations of Multi-Task RL**: Conventional multi-task RL shares knowledge by maximizing average return, but cannot explicitly control catastrophic forgetting.

**Lack of MDP Distance Metrics**: No principled method previously existed to quantify the similarity between different environments, making it difficult to determine whether knowledge transfer would be beneficial.

## Method

### Overall Architecture

FAME (**FA**st and **ME**ta knowledge learner) consists of two coupled learners:
- **Fast Learner**: Analogous to the hippocampus; responsible for rapid learning on new tasks via an adaptive meta warm-up strategy that leverages prior knowledge.
- **Meta Learner**: Analogous to the neocortex; responsible for incrementally consolidating new and old knowledge by minimizing a catastrophic forgetting objective.

The two learners operate alternately: upon the arrival of a new task, the meta learner guides the fast learner for knowledge transfer; after the task concludes, the knowledge acquired by the fast learner is consolidated into the meta learner.

### Key Designs

**Design 1: MDP Distance Definition (Foundation 1)**

- **Function**: Defines a distance metric between MDPs to quantify the similarity of different environments.
- **Mechanism**: The distance between two MDPs is measured by the divergence between their corresponding optimal Q-functions or optimal policies (e.g., $\ell_2$ loss or KL divergence).
- **Design Motivation**: Theoretical analysis requires quantifying environment similarity to determine when knowledge transfer is beneficial and to assess the degree of interference a new task imposes on previously learned tasks.

**Design 2: Formal Definition of Catastrophic Forgetting (Foundation 2)**

- **Function**: Provides a quantitative measure of catastrophic forgetting in continual RL.
- **Mechanism**: Catastrophic forgetting is defined as the weighted discrepancy between old and new policies/Q-functions under the state visitation distribution of the old task. Crucially, the old policy's state visitation distribution is used as a weighting factor, as it better reflects the important state–action pairs of the old task.
- **Design Motivation**: Transforms catastrophic forgetting from an intuitive notion into a mathematically optimizable objective, providing a principled optimization target for algorithm design.

**Design 3: Policy-Level Incremental Meta Learner Update (Proposition 1)**

- **Function**: Derives the incremental update rule for the meta learner.
- **Mechanism**: Minimizing policy-level catastrophic forgetting under KL divergence is equivalent to performing maximum likelihood estimation (MLE) on the meta learner, fitting it to the mixture of state–action distributions across all encountered environments.
- **Design Motivation**: Storing all historical Q-functions is not scalable, necessitating an incremental update rule. Policy-level definitions are more stable than Q-value-level ones and are applicable across tasks with different reward scales.

**Design 4: Adaptive Meta Warm-up**

- **Function**: Adaptively selects the optimal initialization strategy upon the arrival of a new task.
- **Mechanism**: A one-vs-all hypothesis test is conducted among three candidates—the meta learner, the previous fast learner, and random initialization—to select the best warm-up strategy. Policy evaluation is performed during early interactions, followed by a statistical test to identify the best-performing initialization.
- **Design Motivation**: Avoids negative transfer—when new and old tasks differ substantially, initializing from prior knowledge can be harmful. At the same time, the design preserves the ability to leverage prior knowledge to accelerate learning.

**Design 5: Policy Consolidation via Wasserstein Distance (FAME-WD)**

- **Function**: In continuous action spaces, employs the Wasserstein distance for knowledge consolidation.
- **Mechanism**: When policies are represented as Gaussian distributions, the 2-Wasserstein distance admits a closed-form solution, enabling efficient computation of the meta learner's incremental updates.
- **Design Motivation**: The Wasserstein distance accounts for the geometric structure of the data space and is more suitable than KL divergence for complex policy distributions.

### Loss & Training

- **Knowledge Consolidation Loss**: Minimizes the policy-level catastrophic forgetting objective (Eq. 4), equivalent to maximizing the log-likelihood of the meta learner over the mixture of all historical state–action distributions.
- **Behavioral Cloning Regularization**: For value-based methods, meta warm-up adopts a BC regularization term $\mathcal{L}(Q_k) = \mathcal{L}_0(Q_k) + \lambda \cdot \mathbb{E}[\mathrm{KL}(\pi_M \| \pi_Q)]$, using the meta policy as an expert to guide early exploration.
- **Meta Buffer**: State–action pairs collected from the final $N$ steps of each task (approximately 1–2% of training data) are stored to estimate the weighting function used in knowledge consolidation.

## Key Experimental Results

### Main Results

**MinAtar Results (10 sequences × 3 seeds)**:

| Method | Avg. Perf ↑ | FT ↑ | Forgetting ↓ |
|--------|------------|------|-------------|
| Reset | 6.51 ± 1.67 | 0.74 ± 0.38 | 1.31 ± 0.23 |
| Finetune | 10.62 ± 2.75 | 0.89 ± 0.49 | 1.26 ± 0.32 |
| LargeBuffer | 10.71 ± 2.84 | 1.16 ± 0.59 | 1.65 ± 0.33 |
| **FAME** | **14.54 ± 0.58** | **1.69 ± 0.17** | **0.72 ± 0.13** |

**Meta-World Results (3 sequences × 10 seeds)**:

| Method | Avg. Perf ↑ | FT ↑ | Forgetting ↓ |
|--------|------------|------|-------------|
| Reset | 0.093 ± 0.017 | 0.000 | 0.710 ± 0.030 |
| PackNet | 0.491 ± 0.025 | -0.194 | 0.000 |
| FAME-KL | 0.733 ± 0.026 | 0.022 | 0.073 ± 0.019 |
| **FAME-WD** | **0.767 ± 0.024** | -0.003 | **0.023 ± 0.015** |

### Ablation Study

**Selection Frequency of Adaptive Meta Warm-up**:

| Arriving Environment Type | Meta Warm-up | Reset | Finetune |
|--------------------------|-------------|-------|----------|
| Known environment | 95.1% | Low | Low |
| Unknown environment | Low | Relatively high | Low |

**Atari Results**:

| Method | Freeway Avg. Perf | SpaceInvader Avg. Perf |
|--------|------------------|----------------------|
| Reset | 0.16 | 0.10 |
| PackNet | 0.41 | 0.47 |
| ProgressiveNet | 0.39 | 0.61 |
| **FAME** | **0.90** | **0.96** |

### Key Findings

1. FAME significantly outperforms all baselines across benchmarks, achieving the highest average performance with the smallest variance, demonstrating superior stability.
2. The adaptive meta warm-up correctly identifies known versus unknown environments: it selects meta warm-up with 95.1% probability for known environments while favoring random initialization for novel environments.
3. FAME-WD marginally outperforms FAME-KL in continuous action spaces, validating the advantage of the Wasserstein distance for complex policy distributions.
4. Although PackNet achieves zero forgetting by storing model parameters, it requires prior knowledge of the number of tasks and task IDs, limiting its practical applicability.

## Highlights & Insights

1. **Solid Theoretical Contributions**: This work provides the first formal definitions of MDP distance and catastrophic forgetting for continual RL, grounding algorithm design in a rigorous theoretical framework.
2. **Natural and Effective Neuroscientific Inspiration**: The hippocampus–neocortex dual-system analogy is not merely a metaphor but maps directly onto the two algorithmic components and their interaction.
3. **Knowledge Consolidation = MLE**: The equivalence between minimizing policy-level catastrophic forgetting and maximum likelihood estimation establishes a deep connection between continual RL and multi-task RL.
4. **Hypothesis Testing for Warm-up Selection**: The use of a rigorous statistical framework to address negative transfer, rather than relying on heuristic rules, is a notable methodological contribution.
5. **Applicability to Both Value-Based and Policy Gradient Methods**: The generality of the framework is not limited to any specific RL algorithm family.

## Limitations & Future Work

1. **Assumption Constraints**: The framework requires identical state and action spaces and known task boundaries, which may not hold in real-world applications.
2. **Meta Buffer Size**: Although only a small amount of data is stored, the buffer grows linearly with the number of tasks.
3. **Scalability**: The hypothesis testing procedure may introduce computational overhead when task switches are frequent.
4. **Future Directions**: Learning latent representations to replace direct policy/value function storage; online hypothesis testing to handle unknown task boundaries; context embeddings to enhance knowledge transfer.

## Related Work & Insights

- **Compared to PackNet and ProgressiveNet**: These methods avoid forgetting by storing parameters or network structures, but are inflexible and lack knowledge transfer capabilities. FAME addresses both forgetting and transfer through principled knowledge consolidation.
- **Compared to PT-DQN**: PT-DQN's permanent value function has limited knowledge retention capacity. FAME's meta learner is more effective through explicit optimization of the catastrophic forgetting objective.
- **Insights**: Combining theory-driven approaches with neuroscientific inspiration can provide more principled guidance for RL algorithm design. The formalization of catastrophic forgetting is generalizable to other continual learning settings.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The theoretical contributions (MDP distance, forgetting metric) are innovative; the dual-learner framework, while inspired by CLS theory, is novel in its formalization for RL.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers both pixel-based and continuous control benchmarks, value-based and policy gradient algorithms, with statistically robust results (30 seeds).
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, rigorous theoretical derivations, and a complete logical chain from foundational definitions to algorithm design.
- **Value**: ⭐⭐⭐⭐ Provides both a theoretical foundation and a practical algorithm for continual RL, bridging the gap between theory and practice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Continual Knowledge Adaptation for Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/continual_knowledge_adaptation_for_reinforcement_learning.md)
- [\[ICLR 2026\] Self-Improving Skill Learning for Robust Skill-based Meta-Reinforcement Learning](self-improving_skill_learning_for_robust_skill-based_meta-reinforcement_learning.md)
- [\[ACL 2026\] Data Mixing Agent: Learning to Re-weight Domains for Continual Pre-training](../../ACL2026/reinforcement_learning/data_mixing_agent_learning_to_re-weight_domains_for_continual_pre-training.md)
- [\[NeurIPS 2025\] Temporal-Difference Variational Continual Learning](../../NeurIPS2025/reinforcement_learning/temporal-difference_variational_continual_learning.md)
- [\[AAAI 2026\] Scalable Multi-Objective and Meta Reinforcement Learning via Gradient Estimation](../../AAAI2026/reinforcement_learning/scalable_multi-objective_and_meta_reinforcement_learning_via_gradient_estimation.md)

</div>

<!-- RELATED:END -->
