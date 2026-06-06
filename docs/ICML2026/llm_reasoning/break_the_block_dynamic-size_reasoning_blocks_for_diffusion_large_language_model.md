---
title: >-
  [Paper Note] Break the Block: Dynamic-size Reasoning Blocks for Diffusion Large Language Models via Monotonic Entropy Descent with Reinforcement Learning
description: >-
  [ICML 2026][LLM Reasoning][dLLM] To address the issue where "fixed block size" in diffusion language models (dLLM) during semi-autoregressive generation disrupts the logical chain of reasoning…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "dLLM"
  - "GRPO"
  - "dynamic block size"
  - "monotonic entropy descent"
  - "reasoning consistency"
date: 2026-05-08
content_hash: be04d177fd5a50a9
---

# Break the Block: Dynamic-size Reasoning Blocks for Diffusion Large Language Models via Monotonic Entropy Descent with Reinforcement Learning

**Conference**: ICML 2026  
**arXiv**: [2605.02263](https://arxiv.org/abs/2605.02263)  
**Code**: https://github.com/YanJiangJerry/Block-R1 (available)  
**Area**: LLM Reasoning / Diffusion Language Models / Reinforcement Learning  
**Keywords**: dLLM, GRPO, dynamic block size, monotonic entropy descent, reasoning consistency

## TL;DR
To address the issue where "fixed block size" in diffusion language models (dLLM) during semi-autoregressive generation disrupts the logical chain of reasoning, this paper proposes b1: using RL to learn a block-ending indicator token for generating dynamic-length blocks, and introducing a "block-level monotonic entropy descent (MED) reward" to drive coherent reasoning. This reward can be plugged into existing dLLM RL frameworks (Diffu-GRPO/GDPO/d1/wd1) as a plug-and-play component, boosting wd1 on Countdown from 39.45 to 58.98.

## Background & Motivation

**Background**: Diffusion language models (dLLM) such as LLaDA, d1, and wd1 adopt a "semi-autoregressive + intra-block parallel denoising" generation paradigm: the target sequence is split into multiple fixed-size blocks of $c$ tokens, with blocks generated sequentially from left to right, and $T$ denoising steps performed in parallel within each block. Building on this paradigm, recent RL post-training methods (Diffu-GRPO, GDPO, wd1) have begun to push dLLM towards mathematical reasoning by mimicking GRPO.

**Limitations of Prior Work**: Fixed block size leads to two observed issues: (i) The optimal block size varies greatly across datasets (Sudoku/Countdown/GSM8K/MATH500), making a "one-size-fits-all" approach suboptimal; (ii) Even within the same problem, rigid boundaries often split a complete operation—an example given is splitting "$71-66$" between Block 3 and Block 4, resulting in anomalously high-entropy (uncertainty) tokens and calculation errors.

**Key Challenge**: There is a conflict between the dLLM's parallel block assumption (intra-block token conditional independence) and the fact that reasoning is a sequence of logically connected semantic steps—fixed boundaries almost inevitably cut through the middle of reasoning steps.

**Goal**: Enable dLLM to learn to segment blocks at "semantically complete reasoning steps," while ensuring that the overall uncertainty of the reasoning process decreases progressively.

**Key Insight**: Empirically, it is observed that for correct reasoning traces in LLaDA/d1/wd1, the "block-level average entropy" decreases monotonically along the generation direction, while incorrect reasoning shows oscillation or increase. This suggests that "monotonic block entropy descent" is a proxy indicator for reasoning correctness.

**Core Idea**: Make "where to switch blocks" learnable—introduce an end-of-step indicator token, and use RL with a "neighboring block entropy descent" dense reward to teach the model to start new blocks at appropriate positions, aligning block boundaries with reasoning steps and maintaining monotonic entropy descent throughout the reasoning process.

## Method

### Overall Architecture
b1 is a reward/decoding plugin added to the existing dLLM GRPO framework, consisting of three components: (1) Dynamic block construction—insert a special token $\tau_{\text{end}}$ during reasoning generation; each occurrence closes the current block and starts a new one; (2) MED training objective—use a "neighboring block entropy descent" proxy reward $R_{\text{ent}}$ and an "encourage multi-step reasoning" indicator reward $R_{\text{ind}}$, combine them with the task reward $R_{\text{task}}$ (weighted sum), and feed into Diffu-GRPO; (3) Reasoning alignment—during decoding, strictly follow the training process, dynamically adjusting the next block's starting point upon encountering $\tau_{\text{end}}$.

### Key Designs

1. **Dynamic Block Boundaries + Indicator Token Reward $R_{\text{ind}}$**:

    - **Function**: Allow the model to decide the length $d$ of the $k$-th block, so each block covers exactly one complete reasoning step, rather than a fixed $c$ tokens.
    - **Mechanism**: Add a block-ending token $\tau_{\text{end}}$ (default implementation: string `\block`) to the vocabulary. During denoising of the $k$-th block, once $\hat{\mathbf{x}}_{t}[S_{k-1}+j]=\tau_{\text{end}}$ appears, $j$ is taken as the dynamic block size $d$, and the next block continues after $\tau_{\text{end}}$. To prevent the model from lazily generating only 1–2 blocks, the "number of blocks" is made into a log-form dense reward: when total blocks $K\geq K_{\text{target}}$, $R_{\text{ind}}=1$; otherwise, $R_{\text{ind}}=\log(K+1)/\log(K_{\text{target}}+1)$ (default $K_{\text{target}}=10$).
    - **Design Motivation**: Elevate "block segmentation position" to an RL decision variable, rather than a manually set hyperparameter; use log reward to avoid collapse into a few large blocks.

2. **Block-level Entropy + MED Proxy Reward $R_{\text{ent}}$**:

    - **Function**: Directly drive "monotonic decrease of average entropy between neighboring blocks," indirectly yielding more confident and coherent reasoning traces.
    - **Mechanism**: At the diffusion step $t^{*}$ when a block ends, compute the Shannon entropy for each token and average within the block to obtain block entropy $\mathcal{H}(\mathbf{b}_{k}^{d})$. The ideal goal is to maximize the negative Spearman rank correlation coefficient $r_{\text{SCC}}$ of the block entropy sequence, but Spearman is a global ranking with high reward variance and unstable training. The authors relax this to the "proportion of decreasing adjacent pairs": $R_{\text{ent}}=\frac{1}{K-1}\sum_{k=2}^{K}\mathbb{I}(\mathcal{H}(\mathbf{b}_{k-1}^{d})>\mathcal{H}(\mathbf{b}_{k}^{d}))$, and prove in the appendix that maximizing this relaxed term has the same global optimum as maximizing the Spearman coefficient (i.e., strict monotonic descent).
    - **Design Motivation**: Decompose the sparse global ranking signal into $K-1$ independent pairwise comparisons, providing dense, low-variance gradients while retaining the optimal solution of "strict monotonic descent."

3. **Plug-and-Play GRPO Total Reward**:

    - **Function**: Ensure b1 is not tied to a specific dLLM RL algorithm—Diffu-GRPO, GDPO, d1, wd1 can all directly incorporate it.
    - **Mechanism**: Total reward $R_{\text{total}}=\alpha R_{\text{ent}}+\beta R_{\text{ind}}+\gamma R_{\text{task}}$, with default $\alpha=\beta=\gamma=1$ (no tuning needed), and plug into the base diffusion-GRPO objective (policy ratio still approximated by $\exp(\phi^{\pi_{\theta}}-\phi^{\pi_{\text{old}}})$). Complexity is $\mathcal{O}(K\cdot T\cdot L+L)$, negligible compared to self-attention's $O(L^{2})$.
    - **Design Motivation**: Decouple b1 from the RL algorithm, contributing only "block-level" signals, making it easy to stack with existing SOTA methods (wd1).

### Loss & Training
The training dataset is identical to d1/wd1: LLaDA-8B-Instruct is RL post-trained on GSM8K/MATH/Sudoku/Countdown, sequence lengths 256/512, 4×AMD Mi300x, batch=12 per GPU. All weights $\alpha,\beta,\gamma$ are fixed at 1.

## Key Experimental Results

### Main Results

| Algo / Dataset | Sudoku-256 | Countdown-256 | GSM8K-256 | MATH500-256 |
|---|---|---|---|---|
| LLaDA-8B-Instruct (base) | 7.67 | 16.80 | 76.19 | 32.00 |
| + Diffu-GRPO | 13.53 | 19.92 | 76.35 | 33.60 |
| + Diffu-GRPO + b1 | **16.97** (+3.44) | **28.91** (+8.99) | **78.39** (+2.04) | **34.60** (+1.00) |
| + d1 | 15.06 | 25.39 | 77.03 | 33.40 |
| + d1 + b1 | 18.48 (+3.42) | 30.47 (+5.08) | 78.24 (+1.21) | 34.40 (+1.00) |
| + wd1 | 23.14 | 39.45 | 78.85 | 34.20 |
| + wd1 + b1 | **27.29** (+4.15) | **58.98** (+19.53) | **80.82** (+1.97) | **37.40** (+3.20) |

Maximum single-point improvement: wd1 + b1 achieves +19.53 points on Countdown-256.

### Ablation Study

| Configuration (based on wd1) | Countdown | GSM8K | MATH500 |
|---|---|---|---|
| Fixed-size (wd1) | 39.45 | 78.85 | 34.20 |
| b1 w/o MED ($R_{\text{ent}}$) | 44.14 | 79.23 | 35.00 |
| b1 w/o $R_{\text{ind}}$ | 55.86 | 80.06 | 36.60 |
| b1 (Full) | **58.98** | **80.82** | **37.40** |

### Key Findings
- Removing the MED reward causes Countdown to drop from 58.98 to 44.14, confirming that "entropy monotonic descent" is the core signal of b1; removing the indicator reward also causes a moderate drop, indicating that dynamic blocks themselves contribute significantly.
- When reasoning samples are bucketed by $r_{\text{SCC}}$, accuracy increases monotonically with $r_{\text{SCC}}$; wd1+b1 raises $r_{\text{MED}}$ (proportion of positive $r_{\text{SCC}}$) on Countdown from 91.41% to 97.66%, with accuracy rising from 39.45 to 58.98—providing the first quantitative correspondence between "entropy monotonic descent" and reasoning correctness.
- Training step time increases from 1.31s/step (wd1) to 1.68s/step, throughput drops from 28.57→27.03 tok/s, both nearly negligible; average accuracy rises from 43.91→51.12, yielding excellent cost-effectiveness.
- AdaBlock-dLLM (a rule-based method that segments at newlines during inference) does not improve scores under 0-shot reproduction, proving that "dynamic blocks" must be learned, not just rule-based.

## Highlights & Insights
- Transforms "block size," long treated as a hyperparameter, into a learnable policy, paired with a theoretically grounded reward (optimality of $R_{\text{ent}}$ matches global Spearman), representing a rare dual innovation in dLLM post-training that modifies both decoding and reward.
- "Block entropy monotonic descent" is a novel, observable signal—essentially transferring the intuition of "token entropy descent → more confidence" from AR models to the block level, and it is task-agnostic, requiring no ground truth.
- Designed as a pure plug-and-play reward plugin, reusing all base GRPO infrastructure with almost zero code intrusion; this "signal-only, algorithm-agnostic" paradigm can be directly transferred to coherence optimization in any segmented generation model (e.g., block-diffusion image models).

## Limitations & Future Work
- Current evaluation focuses on math/logic tasks; it remains untested whether the "entropy monotonic descent" assumption holds for more complex open-ended generation (code, long document summarization).
- The default $K_{\text{target}}=10$ is empirically set; longer reasoning tasks may require adaptive target block numbers.
- Block entropy uses a mean-field assumption (intra-block token independence), but in reality, tokens within a reasoning step are strongly dependent; future work could introduce structured entropy estimation (e.g., conditional/joint entropy) to further improve signal quality.

## Related Work & Insights
- **vs d1 / wd1**: The underlying RL framework is the same, but d1/wd1 still use fixed blocks; b1 adds MED+indicator rewards on top, boosting wd1 from 39.45 to 58.98 (Countdown), proving that "dynamic blocks" are an orthogonal improvement to "better GRPO objectives."
- **vs AdaBlock-dLLM**: AdaBlock segments at high-confidence newlines during inference, requiring no training; b1 learns segmentation ability into the weights and directly optimizes via RL, significantly outperforming AdaBlock in 0-shot settings.
- **vs StableMoE / Dynamic Computation Routing**: The RL-learned boundary generation approach of b1 can inspire similar "learnable decision hyperparameters" in MoE dynamic top-k and related problems.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to make "block size" an RL-learnable variable on dLLM, and proposes the novel and provable "block entropy monotonic descent" optimization signal.
- Experimental Thoroughness: ⭐⭐⭐⭐ Systematic validation on 4 datasets and 4 RL base algorithms, with ablation, $r_{\text{SCC}}$ correlation analysis, and efficiency comparison, but lacks open-ended generation tasks.
- Writing Quality: ⭐⭐⭐⭐ Clear storyline (observation → hypothesis → method → theory → validation), with highly convincing token entropy visualizations in Figures 2/3.
- Value: ⭐⭐⭐⭐⭐ Plug-and-play, almost zero-cost stacking on top of existing SOTA dLLM RL algorithms, +19.5 points on Countdown, directly advancing dLLM reasoning research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Revisiting Entropy in Reinforcement Learning for Large Reasoning Models](../../ACL2026/llm_reasoning/revisiting_entropy_in_reinforcement_learning_for_large_reasoning_models.md)
- [\[ICML 2026\] Prism: Efficient Test-Time Scaling via Hierarchical Search and Self-Verification for Discrete Diffusion Language Models](prism_efficient_test-time_scaling_via_hierarchical_search_and_self-verification_.md)
- [\[ICML 2026\] ResRL: Boosting LLM Reasoning via Negative Sample Projection Residual Reinforcement Learning](resrl_boosting_llm_reasoning_via_negative_sample_projection_residual_reinforceme.md)
- [\[AAAI 2026\] Incorporating Self-Rewriting into Large Language Model Reasoning Reinforcement](../../AAAI2026/llm_reasoning/incorporating_self-rewriting_into_large_language_model_reasoning_reinforcement.md)
- [\[ICML 2026\] Game of Thought: Robust Information Seeking with Large Language Models Using Game Theory](game_of_thought_robust_information_seeking_with_large_language_models_using_game.md)

</div>

<!-- RELATED:END -->
