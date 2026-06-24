---
title: >-
  [Paper Note] Training Language Models to Reason Efficiently
description: >-
  [NeurIPS 2025][Reinforcement Learning][Efficient Reasoning] By incorporating a length penalty term into the RL reward—multiplying the correctness reward by $(1 - \alpha \cdot \sigma(\text{norm\_len}))$—and using a single hyperparameter $\alpha$ to control the token–accuracy trade-off curve, this work achieves a 50% reduction in token usage with less than 5% accuracy degradation on 7B reasoning models after only 100 RL training steps.
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "Efficient Reasoning"
  - "CoT Compression"
  - "Length Penalty"
  - "Test-Time Compute"
date: 2026-05-08
content_hash: 6049649e5f785a7c
---

# Training Language Models to Reason Efficiently

**Conference**: NeurIPS 2025
**arXiv**: [2502.04463](https://arxiv.org/abs/2502.04463)  
**Code**: [github.com/Zanette-Labs/efficient-reasoning](https://github.com/Zanette-Labs/efficient-reasoning)  
**Area**: Reinforcement Learning
**Keywords**: Efficient Reasoning, CoT Compression, Length Penalty, Reinforcement Learning, Test-Time Compute

## TL;DR
By incorporating a length penalty term into the RL reward—multiplying the correctness reward by $(1 - \alpha \cdot \sigma(\text{norm\_len}))$—and using a single hyperparameter $\alpha$ to control the token–accuracy trade-off curve, this work achieves a 50% reduction in token usage with less than 5% accuracy degradation on 7B reasoning models after only 100 RL training steps.

## Background & Motivation

**Background**: Reasoning models (o1, R1) substantially improve reasoning capabilities through long CoT, but at high inference cost—the quadratic complexity of Transformer attention makes the computational cost of long CoT grow rapidly.

**Limitations of Prior Work**: (1) Reasoning models severely "overthink" on simple problems (e.g., computing 1+1 requires a full page of CoT); (2) directly constraining length via prompt instructions is ineffective (distilled reasoning models do not follow length constraints); (3) existing compression methods (SFT on shortest correct solutions, DPO) underperform RL-based approaches and lack controllability.

**Key Challenge**: Token usage must be reduced without sacrificing reasoning quality, yet naive truncation leads to incomplete reasoning, and prompt-based control is ineffective for reasoning models.

**Goal**: How can reasoning models be systematically trained to reach correct answers using fewer tokens, while retaining the ability to spend more tokens on harder problems?

**Key Insight**: Introduce a length regularization term into the RL reward function, applying length penalties exclusively to **correct responses** (incorrect responses always receive a reward of 0), with per-prompt normalization to avoid over-penalizing harder problems.

**Core Idea**: Multiply the correctness reward by a length discount $(1 - \alpha \cdot f(\text{len}))$, using $\alpha$ to smoothly control the efficiency–accuracy trade-off, encouraging the model to be more concise on simple problems while maintaining deep reasoning on difficult ones.

## Method

### Overall Architecture
Given a reasoning model (e.g., DeepSeek-R1-Distill), the objective is $\mathbb{E}[\mathbb{1}\{y = y^\star(x)\} \cdot (1 - \alpha \cdot f(\text{len}(y)))]$, where $\alpha \in [0, 1)$ controls the strength of the length penalty. Optimization is performed using PPO with a RLOO advantage estimator. Training converges in only 100 RL steps (~200 gradient updates) and is feasible on academic-scale resources.

### Key Designs

1. **Length Penalty Function Design**:

    - Function: Penalizes correct but verbose responses.
    - Mechanism: $f(\text{len}(y)) = \sigma\left(\frac{\text{len}(y) - \text{mean}(x)}{\text{std}(x)}\right)$, where $\text{mean}(x)$ and $\text{std}(x)$ are the per-prompt mean and standard deviation of lengths among correct responses to the same question, and $\sigma$ denotes the sigmoid function.
    - Design Motivation: Per-prompt normalization ensures that harder problems (which inherently require longer CoT) are not over-penalized; the sigmoid constrains the output to $[0,1]$, guaranteeing that the reward for a correct response remains positive (since $\alpha < 1$), i.e., being correct always dominates being incorrect.

2. **No Advantage Normalization**:

    - Function: Deliberately omits standard deviation normalization of the advantage function (unlike GRPO).
    - Mechanism: When all sampled responses are correct, advantage normalization renders the effective gradient of the length penalty independent of $\alpha$ (derivation: after normalization, the advantage gap between the shortest and longest correct responses is always 6, regardless of $\alpha$), causing excessively aggressive length reduction.
    - Design Motivation: Preserves the controllability of $\alpha$ over training behavior.

3. **Adaptive Difficulty Adjustment (Emergent)**:

    - Function: The model autonomously learns to compress simple problems significantly while maintaining long CoT for hard problems.
    - Mechanism: No explicit difficulty labels are required—per-prompt normalization naturally achieves this effect. Correct solutions for simple problems exhibit higher variance in length (relatively more redundancy), yielding greater compression headroom.
    - Design Motivation: On GSM8K, the 7B model reduces tokens by 83% ($\alpha=0.2$), whereas on AIME the reduction is only 27%.

### Loss & Training
PPO + RLOO (Leave-One-Out baseline), sampling 8 responses per prompt, with 3.2K training prompts from Numina Math. Learning rate $2 \times 10^{-6}$ (7B), KL coefficient $10^{-3}$. Convergence is achieved in only 100 RL steps (~20 hours on 4–8 GPUs).

## Key Experimental Results

### Main Results
Base model: DeepSeek-R1-Distill-Qwen-7B.

| $\alpha$ | MATH500 Acc. | MATH500 Tokens | AIME24 Acc. | AIME24 Tokens | GSM8K Acc. | GSM8K Tokens |
|---------|-------------|---------------|-------------|-------------|-----------|-----------|
| 0 (R1) | ~86% | ~4000 | ~30% | ~13000 | ~95% | ~3500 |
| 0.1 | ~84% (−2.2%) | ~2600 (−36%) | ~28% | ~11000 | ~94% | ~1700 |
| 0.2 | ~82% | ~2200 | ~27% (−3.3%) | ~9000 (−27%) | ~93% (−1.7%) | ~600 (−83%) |
| 0.4 | ~78% | ~1500 | ~22% | ~7000 | ~90% | ~400 |

### Ablation Study

| Configuration | Effect | Notes |
|------|------|------|
| No advantage normalization | Controllable | $\alpha$ precisely controls compression degree |
| With advantage normalization | Uncontrollable | Length drops sharply; accuracy collapses |
| SFT on shortest correct solutions | Inferior | Lower accuracy than RL at equivalent token budgets |
| DPO (longest vs. shortest) | Inferior | Uncontrollable; weaker than RL |
| O1-Pruner | Below Ours | $\alpha=0.05,0.1$ achieves higher accuracy than O1-Pruner at equivalent token budgets |

### Key Findings
- **50% token compression with <5% accuracy loss**: Results aggregated across 5 datasets for the 7B model.
- **Adaptive difficulty response**: The same $\alpha$ value yields 83% compression on GSM8K but only 27% on AIME—the model automatically becomes more concise on simpler problems.
- **Convergence in only 100 RL steps**: Extremely low training cost (~20 hours on 4–8 GPUs), requiring only a few lines of code changes to a standard RL implementation.
- **Compression primarily reduces verification, backtracking, and exploration behaviors**: As $\alpha$ increases, the frequency of verification (4.6→1.2), backtracking (19.1→6.4), and exploration (26.3→2.1) consistently decreases.
- **CoT faithfulness slightly decreases but remains far above non-reasoning models**: The faithfulness score drops from 0.622 (R1) to ~0.5 ($\alpha=0.1$), but remains substantially higher than instruction-tuned models (0.301).

## Highlights & Insights
- **Minimalist yet highly effective**: Adding a single length penalty term to the reward function yields substantial gains; the simplicity of the method is its greatest strength.
- **Per-prompt normalization is the critical design choice**: Without it, hard and easy problems would be treated equally, unfairly penalizing long reasoning on difficult problems. This design enables the emergent behavior of "fast solutions for easy problems, deep reasoning for hard ones."
- **The finding on omitting advantage normalization is valuable**: It reveals a subtle issue in standard GRPO—normalization eliminates the control dimension provided by $\alpha$. This is an important reminder for all work employing RL rewards with length terms.

## Limitations & Future Work
- $\alpha$ controls the overall degree of compression rather than a precise target length, which is insufficient for applications with strict latency constraints.
- Compression is consistently accompanied by a small accuracy loss; whether compression can be achieved without any degradation remains an open question.
- Validation is limited to mathematical reasoning; effectiveness on other reasoning tasks such as code generation is unknown.
- Although RL training requires only 100 steps, it remains more complex to set up than pure SFT pipelines.

## Related Work & Insights
- **vs. Kimi k1.5**: Kimi also uses length penalties in RL, but without per-prompt normalization or a tunable $\alpha$ for generating a model family.
- **vs. O1-Pruner**: O1-Pruner employs offline RL (a DPO variant), whereas this work uses online RL to achieve better results. $\alpha=0.05$ and $0.1$ consistently outperform O1-Pruner in accuracy at equivalent token budgets.
- **vs. SFT on shortest correct solutions**: SFT achieves lower accuracy at equivalent token budgets, demonstrating that online RL training is more effective for CoT compression.
- **vs. prompt-based control**: Experiments confirm that distilled reasoning models completely ignore length-constraint prompts (actual generation lengths remain essentially identical across all token limits).

## Rating
- Novelty: ⭐⭐⭐⭐ — The idea is simple yet represents the first systematic application of RL length penalties for training reasoning efficiency; the findings on per-prompt normalization and omitting advantage normalization are insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 5 datasets, 2 model scales, multiple baselines, ablations, qualitative analysis, and faithfulness study.
- Writing Quality: ⭐⭐⭐⭐⭐ — Motivation is clear, method is concise, theoretical guarantees are well-developed, and experimental figures are informative.
- Value: ⭐⭐⭐⭐⭐ — Extremely high practical value—a few lines of code changes and 20 hours of training reduce inference compute by half for reasoning models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Learning to Reason Efficiently with Discounted Reinforcement Learning](../../ICLR2026/reinforcement_learning/learning_to_reason_efficiently_with_discounted_reinforcement_learning.md)
- [\[NeurIPS 2025\] RePIC: Reinforced Post-Training for Personalizing Multi-Modal Language Models](repic_reinforced_post-training_for_personalizing_multi-modal_language_models.md)
- [\[ICLR 2026\] Learn to Reason Efficiently with Adaptive Length-based Reward Shaping](../../ICLR2026/reinforcement_learning/learn_to_reason_efficiently_with_adaptive_length-based_reward_shaping.md)
- [\[NeurIPS 2025\] Note 5: ReSearch — Learning to Reason with Search](research_learning_to_reason_with_search_for_llms_via_reinforcement_learning.md)
- [\[NeurIPS 2025\] Checklists Are Better Than Reward Models For Aligning Language Models](checklists_are_better_than_reward_models_for_aligning_langua.md)

</div>

<!-- RELATED:END -->
