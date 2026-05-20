---
title: >-
  [Paper Note] Learning to Better Search with Language Models via Guided Reinforced Self-Training
description: >-
  [NeurIPS 2025][Model Compression][search strategy learning] This paper proposes Guided-ReST, which progressively incorporates optimal solutions as subgoals into model-generated search trajectories to produce high-quality…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "search strategy learning"
  - "self-training"
  - "reinforcement learning"
  - "test-time compute efficiency"
  - "language model reasoning"
date: 2026-05-08
content_hash: bab6923d2ae418bf
---

# Learning to Better Search with Language Models via Guided Reinforced Self-Training

**Conference**: NeurIPS 2025
**arXiv**: [2410.02992](https://arxiv.org/abs/2410.02992)  
**Code**: [GitHub](https://github.com/snu-mllab/guided-rest)  
**Area**: Model Compression (LLM Inference Optimization)
**Keywords**: search strategy learning, self-training, reinforcement learning, test-time compute efficiency, language model reasoning

## TL;DR

This paper proposes Guided-ReST, which progressively incorporates optimal solutions as subgoals into model-generated search trajectories to produce high-quality training data and distill more efficient search strategies. The approach yields substantial improvements in search efficiency and accuracy on Countdown and code self-repair tasks.

## Background & Motivation

**Background**: Language models still struggle with complex reasoning tasks. Recent work (e.g., Stream of Search, SoS) shows that training models to imitate full search trajectories—including exploration and backtracking—generalizes better than training on final answers alone.

**Limitations of Prior Work**: Search trajectories used in SoS training are often noisy and suboptimal, causing test-time compute to be wasted on redundant exploration and repeated backtracking. Standard ReST simply filters successful trajectories for fine-tuning without exploiting the structural information of optimal solutions.

**Key Challenge**: Optimal solutions provide valuable guidance but suffer from poor generalization when used directly for behavioral cloning, whereas search trajectories generalize well but are inefficient. How can both advantages be combined?

**Goal**: To generate high-quality search trajectory data that enables models to internalize efficient search strategies, maximizing problem-solving accuracy under a limited token budget.

**Key Insight**: Drawing on the dual-policy framework of jump-start RL, the paper uses optimal solutions as waypoints to guide the search process.

**Core Idea**: Optimal solutions are ill-suited for direct imitation, but can serve as step-by-step waypoints that are progressively integrated into model-generated search trajectories to produce high-quality training data.

## Method

### Overall Architecture

Guided-ReST consists of two stages: (1) generating high-quality search trajectories via subgoal augmentation followed by supervised fine-tuning; and (2) further optimization through PPO under an operation-level MDP.

### Key Designs

1. **Subgoal Augmentation**:

    - **Function**: Decomposes optimal solutions into a sequence of subgoal nodes and progressively incorporates them into model-generated search trajectories.
    - **Design Motivation**: Directly prompting with partial optimal solutions improves success rates (from 55.96% to 100%) but causes the trajectory distribution to diverge from the model's own behavior distribution (cross-entropy increases from 0.0481 to 0.2539), which may impair search capability upon fine-tuning.
    - **Mechanism**: For a search trajectory that fails to reach the $t$-th subgoal, the algorithm randomly selects a failed child node of the $(t-1)$-th subgoal node, replaces it with the correct $t$-th subgoal, truncates the subsequent inconsistent search history, and allows the model to continue searching from the augmented subgoal. This process iterates until all subgoals are incorporated.
    - **Novelty**: Unlike ReST, which discards failed trajectories outright, Guided-ReST injects correction signals at failure points, teaching the model where to backtrack and how to recover from failures.

2. **Operation-level Reinforcement Learning**:

    - **Function**: Applies PPO on top of the Guided-ReST fine-tuned model for further optimization.
    - **Design Motivation**: Standard PPO operates under a token-level MDP, whereas the optimization target is the search strategy rather than token generation per se.
    - **Mechanism**: Each action $a_h$ is defined as the token sequence $(a_{h,1}, \ldots, a_{h,L})$ corresponding to a single tree operation (node generation / exploration / verification / backtracking), with the importance ratio redefined as: $\log r_h(\theta) = \sum_{\ell=1}^{L}(\log\pi_\theta(a_{h,\ell}|a_{h,1:\ell-1}, s_h) - \log\pi_{\theta_{old}}(a_{h,\ell}|a_{h,1:\ell-1}, s_h))$
    - **Novelty**: The operation-level MDP achieves approximately $2\times$ the token efficiency of its token-level counterpart.

3. **Episode-level Extension (Code Self-Repair)**:

    - **Function**: Extends the framework to multi-turn interactive settings for code self-repair.
    - **Design Motivation**: Code tasks produce longer responses, making fine-grained tree search infeasible.
    - **Mechanism**: A simplified variant truncates failed episodes at each turn, augments the user feedback with an optimal-solution prompt, and allows the model to continue generation. As the number of turns increases, the number of turns incorporating the optimal-solution prompt grows progressively.

### Loss & Training

- **Stage 1**: Supervised fine-tuning: $\max_\theta \mathbb{E}_{q,Z}[\log\pi_\theta(Z|q)]$, where $Z$ denotes search trajectories generated by Guided-ReST.
- **Stage 2**: PPO objective: $\max_\theta \mathbb{E}[\min(r_h(\theta)A_h, \text{clip}(r_h(\theta), 1-\epsilon, 1+\epsilon)A_h)]$
- Advantage functions are estimated via Monte Carlo returns without a KL penalty term.
- Countdown experiments use Llama-3.2-1B-Instruct (4K tokens); code tasks use Qwen2.5-7B-Instruct (16K tokens).

## Key Experimental Results

### Main Results

**Countdown accuracy at maximum token budget:**

| Method | Seen Targets | Unseen Targets |
|--------|-------------|----------------|
| BC | <40% | <40% |
| SoS | ~65% | ~63% |
| ReST | ~72% | ~70% |
| PPO (from SoS) | ~77% | ~75% |
| **Guided-ReST + PPO** | **87%** | **87%** |

**Code self-repair Pass@k accuracy (CodeContests):**

| Method | Pass@1 | Pass@4 | Pass@16 | Pass@32 |
|--------|--------|--------|---------|---------|
| Base | 4.5 | 11.3 | 20.6 | 25.8 |
| ReST | 9.4 | 17.0 | 25.9 | 30.4 |
| **Guided-ReST** | **10.5** | **19.4** | **28.9** | **33.9** |

### Ablation Study

| Ablation Dimension | Results |
|-------------------|---------|
| Partial solution length vs. accuracy | 0 steps: 55.96%, 1 step: 85.28%, 2 steps: 95.27%, 3 steps: 100% |
| Partial solution length vs. cross-entropy | 0 steps: 0.0481, 1 step: 0.1245, 2 steps: 0.2412, 3 steps: 0.2539 |
| Operation-level vs. token-level MDP | Operation-level: 87%, token-level: 83%; token efficiency improved by $1.5$–$2\times$ |
| PPO with dense subgoal rewards | Nearly ineffective and slightly degrades performance, confirming the necessity of Guided-ReST pretraining |
| Pass@k (Guided-ReST vs. ReST) | At $k{=}32$: Guided-ReST 96.8% vs. ReST 76.6%; gap widens as $k$ increases |

### Key Findings

- Guided-ReST achieves 87% accuracy on unseen targets, demonstrating that the model does not simply memorize optimal solutions.
- Guided-ReST achieves accuracy comparable to PPO using only approximately 50% of the tokens, representing a substantial gain in token efficiency.
- Guided-ReST and PPO exhibit strong synergy, whereas ReST + PPO does not; the root cause lies in the difference in pass@k coverage.
- Direct RL with dense rewards cannot substitute for the data generation role of Guided-ReST.

## Highlights & Insights

- **Deep Core Insight**: Optimal solutions are unsuitable for direct imitation but effective as waypoints—this insight is both elegant and thoroughly validated.
- **Subgoal Augmentation**: The algorithm is carefully designed to inject corrections at failure points rather than simply restarting, preserving high trajectory likelihood.
- **Thorough Analysis**: The pass@k analysis reveals the fundamental reason for the synergy between Guided-ReST and PPO (broader coverage of correct candidates).
- **Strong Practicality**: The method is straightforward and generalizes from formal tasks (Countdown) to code self-repair.

## Limitations & Future Work

- The method assumes access to optimal solutions, which may not always be available in practice (the paper proposes using strong model generations as a relaxation).
- The Countdown experiments assume an oracle that can precisely identify the first erroneous step and trigger backtracking, which is unrealistic for general language models.
- The improvements on code self-repair are less pronounced than on Countdown, possibly due to limited training data and incomplete subgoal augmentation.
- Experiments are conducted only on 1B and 7B models; the effectiveness at larger scales remains unknown.
- Variants of Guided-ReST for other search algorithms (e.g., MCTS) are not explored.

## Related Work & Insights

- The closest relationship is with Stream of Search (SoS), building upon it to address trajectory quality and efficiency.
- The dual-policy framework inspired by Jump-Start RL is worth generalizing to broader settings.
- The work offers useful reference for training reasoning models such as DeepSeek-R1 on how to more efficiently leverage test-time computation.
- The idea of teaching models "where to backtrack" has potential for generalization to more general reasoning error-correction scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ — Subgoal augmentation and the operation-level MDP are novel designs, though the overall framework represents a clever combination of existing techniques.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Ablations are comprehensive and analysis is in-depth (pass@k analysis, MDP comparison, dense reward comparison, etc.).
- Writing Quality: ⭐⭐⭐⭐⭐ — Logic is clear, motivation is well-established, and figures are intuitive.
- Value: ⭐⭐⭐⭐ — Provides practically meaningful guidance for improving training efficiency of search-based reasoning models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Revisiting Semi-Supervised Learning in the Era of Foundation Models](revisiting_semi-supervised_learning_in_the_era_of_foundation_models.md)
- [\[NeurIPS 2025\] Gated Integration of Low-Rank Adaptation for Continual Learning of Large Language Models](gated_integration_of_low-rank_adaptation_for_continual_learning_of_large_languag.md)
- [\[NeurIPS 2025\] VESSA: Video-based objEct-centric Self-Supervised Adaptation for Visual Foundation Models](vessa_video-based_object-centric_self-supervised_adaptation_for_visual_foundatio.md)
- [\[NeurIPS 2025\] VQToken: Neural Discrete Token Representation Learning for Extreme Token Reduction in Video Large Language Models](vqtoken_neural_discrete_token_representation_learning_for_extreme_token_reductio.md)
- [\[ACL 2026\] Training-Free Test-Time Contrastive Learning for Large Language Models](../../ACL2026/model_compression/training-free_test-time_contrastive_learning_for_large_language_models.md)

</div>

<!-- RELATED:END -->
