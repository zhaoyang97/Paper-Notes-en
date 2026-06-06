---
title: >-
  [Paper Note] Break the Block: Dynamic-size Reasoning Blocks for Diffusion Large Language Models via Monotonic Entropy Descent with Reinforcement Learning
description: >-
  [ICML 2026][Reinforcement Learning][dLLM] Addressing the issue where "fixed block sizes" disrupt the logical reasoning chain in semi-autoregressive generation of Diffusion Large Language Models (dLLM)…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "dLLM"
  - "GRPO"
  - "Dynamic Block Size"
  - "Monotonic Entropy Descent"
  - "Reasoning Consistency"
date: 2026-05-08
content_hash: 6aacecf099211702
---

# Break the Block: Dynamic-size Reasoning Blocks for Diffusion Large Language Models via Monotonic Entropy Descent with Reinforcement Learning

**Conference**: ICML 2026  
**arXiv**: [2605.02263](https://arxiv.org/abs/2605.02263)  
**Code**: https://github.com/YanJiangJerry/Block-R1 (Available)  
**Area**: LLM Reasoning / Diffusion Language Models / Reinforcement Learning  
**Keywords**: dLLM, GRPO, Dynamic Block Size, Monotonic Entropy Descent, Reasoning Consistency

## TL;DR
Addressing the issue where "fixed block sizes" disrupt the logical reasoning chain in semi-autoregressive generation of Diffusion Large Language Models (dLLM), this paper proposes b1. By learning an end-of-step indicator token via RL to generate dynamic-length blocks and utilizing a "block-level Monotonic Entropy Descent (MED) reward" to drive coherent reasoning, b1 serves as a plug-and-play reward component for existing dLLM RL frameworks (Diffu-GRPO/GDPO/d1/wd1). It improves wd1 from 39.45 to 58.98 on the Countdown task.

## Background & Motivation

**Background**: Diffusion language models (dLLM) such as LLaDA, d1, and wd1 adopt a "semi-autoregressive + intra-block parallel denoising" generation paradigm. The sequence to be generated is partitioned into multiple blocks of fixed size $c$, where generation proceeds sequentially between blocks and in $T$ parallel denoising steps within a block. Based on this paradigm, recent RL post-training methods (Diffu-GRPO, GDPO, wd1) have begun pushing dLLMs toward mathematical reasoning by mimicking GRPO.

**Limitations of Prior Work**: Fixed block sizes lead to two observed problems: (i) The optimal block size varies significantly across different datasets (Sudoku, Countdown, GSM8K, MATH500), making a "one-size-fits-all" approach suboptimal. (ii) Rigid boundaries often split a complete operation—an example provided shows "$71-66$" split between Block 3 and Block 4, resulting in high-entropy (uncertainty) anomalous tokens and subsequent calculation errors.

**Key Challenge**: There is a conflict between the parallel block assumption of dLLMs (intra-block token conditional independence) and the nature of reasoning as a continuous semantic sequence of logical steps. Fixed boundaries almost inevitably sever the middle of a reasoning step.

**Goal**: To enable the dLLM to learn to partition blocks at "semantically complete reasoning steps" while ensuring that the overall uncertainty of the reasoning process decreases progressively.

**Key Insight**: The authors empirically discovered a pattern in LLaDA, d1, and wd1: the "block-level average entropy" of correct reasoning traces descends monotonically along the generation direction, while incorrect reasoning shows fluctuations or increases. This suggests that "block entropy monotonic descent" is a proxy indicator for reasoning correctness.

**Core Idea**: Transform "where to switch blocks" into a learnable decision. By introducing an end-of-step indicator token and using RL with a dense "adjacent block entropy descent" reward, the model is taught to initiate a new block at appropriate positions. This aligns block boundaries with reasoning steps and maintains monotonic entropy descent throughout the reasoning process.

## Method

### Overall Architecture
b1 is a reward/decoding plugin implemented on top of existing dLLM GRPO frameworks, consisting of three components: (1) Dynamic block construction—inserting a special token $\tau_{\text{end}}$ during reasoning; its appearance closes the current block and starts a new one. (2) MED training objective—using a proxy reward $R_{\text{ent}}$ for "adjacent block entropy descent" plus an indicator reward $R_{\text{ind}}$ to "encourage multi-step reasoning," weighted with the task reward $R_{\text{task}}$ for Diffu-GRPO. (3) Inference alignment—decoding follows the training procedure, dynamically adjusting the start of the next block upon encountering $\tau_{\text{end}}$.

### Key Designs

1.  **Dynamic Block Boundaries + Indicator Token Reward $R_{\text{ind}}$**:
    - **Function**: Allows the model to determine the length $d$ of the $k$-th block so that each block covers exactly one complete reasoning step instead of a fixed $c$ tokens.
    - **Mechanism**: Adds a block-end token $\tau_{\text{end}}$ (defaulted as `\block`) to the vocabulary. During the denoising of the $k$-th block, if $\hat{\mathbf{x}}_{t}[S_{k-1}+j]=\tau_{\text{end}}$ appears, $j$ is taken as the dynamic size $d$ of the current block, and the next block resumes after $\tau_{\text{end}}$. To prevent the model from generating too few blocks, a dense log-form reward is used: $R_{\text{ind}}=1$ if the total blocks $K \geq K_{\text{target}}$, otherwise $R_{\text{ind}}=\log(K+1)/\log(K_{\text{target}}+1)$ (default $K_{\text{target}}=10$).
    - **Design Motivation**: Promotes "block partition positions" to RL decision variables rather than manually set hyperparameters; the log reward avoids collapse into a few large blocks.

2.  **Block-level Entropy + MED Proxy Reward $R_{\text{ent}}$**:
    - **Function**: Directly drives the monotonic descent of average entropy across adjacent blocks, indirectly compelling more confident and coherent reasoning traces.
    - **Mechanism**: At the final diffusion step $t^{*}$ of a block, Shannon entropy is calculated for each token and averaged to obtain the block entropy $\mathcal{H}(\mathbf{b}_{k}^{d})$. While the ideal goal is to maximize the negative Spearman rank correlation coefficient $r_{\text{SCC}}$ of the entropy sequence, Spearman is a global rank with high variance. The authors relax this into an "adjacent pair descent ratio": $R_{\text{ent}}=\frac{1}{K-1}\sum_{k=2}^{K}\mathbb{I}(\mathcal{H}(\mathbf{b}_{k-1}^{d})>\mathcal{H}(\mathbf{b}_{k}^{d}))$. The appendix proves that maximizing this relaxation shares the same global optimum as the Spearman coefficient (strict monotonic descent).
    - **Design Motivation**: Decomposes a sparse global ranking signal into $K-1$ independent pairwise comparisons, providing dense, low-variance gradients while retaining the optimal solution of strict monotonic descent.

3.  **Plug-and-play GRPO Total Reward**:
    - **Function**: Decouples b1 from specific dLLM RL algorithms, allowing integration with Diffu-GRPO, GDPO, d1, or wd1.
    - **Mechanism**: Total reward $R_{\text{total}}=\alpha R_{\text{ent}}+\beta R_{\text{ind}}+\gamma R_{\text{task}}$, with default $\alpha=\beta=\gamma=1$. This is integrated into the base algorithm's diffusion-GRPO objective (policy ratio approximated via $\exp(\phi^{\pi_{\theta}}-\phi^{\pi_{\text{old}}})$). Complexity is $\mathcal{O}(K\cdot T\cdot L+L)$, which is negligible compared to the $O(L^{2})$ of self-attention.
    - **Design Motivation**: Ensures b1 contributes block-level signals without binding to a specific algorithm, facilitating stacking with state-of-the-art methods like wd1.

### Loss & Training
The training dataset is identical to d1/wd1: RL post-training of LLaDA-8B-Instruct on GSM8K, MATH, Sudoku, and Countdown. Sequence lengths are 256/512. Hardware: 4×AMD Mi300x with batch=12 per GPU. All weights $\alpha, \beta, \gamma$ are fixed at 1.

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

Maximum Gain: wd1 + b1 increases Countdown-256 performance by +19.53 points.

### Ablation Study

| Configuration (Based on wd1) | Countdown | GSM8K | MATH500 |
|---|---|---|---|
| Fixed-size (wd1) | 39.45 | 78.85 | 34.20 |
| b1 w/o MED ($R_{\text{ent}}$) | 44.14 | 79.23 | 35.00 |
| b1 w/o $R_{\text{ind}}$ | 55.86 | 80.06 | 36.60 |
| b1 (Full) | **58.98** | **80.82** | **37.40** |

### Key Findings
- Removing the MED reward causes Countdown performance to drop from 58.98 to 44.14, confirming that "monotonic entropy descent" is the core signal of b1. Removing the indicator reward also results in a drop, showing that dynamic blocks themselves contribute significantly.
- Binning reasoning samples by $r_{\text{SCC}}$ shows that accuracy rises monotonically with $r_{\text{SCC}}$. wd1+b1 raises the $r_{\text{MED}}$ (proportion of positive $r_{\text{SCC}}$) on Countdown from 91.41% to 97.66%, corresponding to an accuracy increase from 39.45 to 58.98—the first quantitative link between entropy descent and reasoning correctness.
- Training step time increases from 1.31s to 1.68s for wd1, with throughput slightly decreasing from 28.57 to 27.03 tok/s. This overhead is negligible given the average accuracy improvement from 43.91 to 51.12.
- AdaBlock-dLLM (a training-free method that truncates at newlines during inference) shows no gain in 0-shot replication, proving that dynamic block capability must be learned and cannot rely solely on rules.

## Highlights & Insights
- Transforming "block size," previously treated as a hyperparameter, into a learnable policy paired with a theoretically grounded reward ($R_{\text{ent}}$) is a rare dual design in dLLM post-training that modifies both decoding and rewards.
- "Block entropy monotonic descent" is a novel observable signal—it essentially migrates the intuition of "token entropy descent implies higher confidence" from AR models to block granularity and is task-agnostic as it does not require ground truth.
- Designed as a pure plug-and-play reward plugin, it reuses the infrastructure of the base GRPO framework with almost zero code intrusion. This paradigm can be extended to coherence optimization in any segmented generation models (e.g., block-diffusion image models).

## Limitations & Future Work
- Evaluation is currently focused on mathematical/logic problems; more complex open-ended generation (code, long document summarization) has not yet been verified for the "entropy monotonic descent" hypothesis.
- The default $K_{\text{target}}=10$ is empirical; long reasoning tasks might require an adaptive target for the number of blocks.
- Block entropy relies on a mean-field assumption (intra-block token independence). Since tokens within a reasoning step have strong dependencies, future work could introduce structured entropy estimation (e.g., conditional/joint entropy) to improve signal quality.

## Related Work & Insights
- **vs. d1 / wd1**: Uses the same base RL framework, but d1/wd1 still use fixed blocks. b1 stacks MED and indicator rewards on top of them, pushing wd1 from 39.45 to 58.98 on Countdown, proving dynamic blocks are an orthogonal dimension of improvement to better GRPO objectives.
- **vs. AdaBlock-dLLM**: AdaBlock truncates at high-confidence newlines during inference without training. b1 learns block-partitioning within the weights via direct RL optimization, significantly outperforming AdaBlock in 0-shot settings.
- **vs. StableMoE / Dynamic Computation Routing**: The idea of using RL to learn generation boundaries in b1 could be cross-applied to MoE problems like dynamic top-k, where decision hyperparameters can be made learnable.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to make "block size" an RL-learnable variable in dLLMs and provide the "monotonic entropy descent" signal.
- Experimental Thoroughness: ⭐⭐⭐⭐ Systematically verified across 4 datasets and 4 base RL algorithms with ablation, correlation analysis, and efficiency comparisons, though lacking open-ended generation tasks.
- Writing Quality: ⭐⭐⭐⭐ Clear narrative (Observation → Hypothesis → Method → Theory → Verification); the token entropy visualization in Figures 2/3 is highly persuasive.
- Value: ⭐⭐⭐⭐⭐ Plug-and-play, nearly zero-cost addition to the strongest current dLLM RL algorithms, yielding +19.5 points on Countdown; directly advances dLLM reasoning research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Learning Unmasking Policies for Diffusion Language Models](learning_unmasking_policies_for_diffusion_language_models.md)
- [\[ICML 2026\] d2: Improving Reasoning in Diffusion Language Models via Trajectory Likelihood Estimation](d2_improving_reasoning_in_diffusion_language_models_via_trajectory_likelihood_es.md)
- [\[ICML 2026\] The Shape of Reasoning: Topological Analysis of Reasoning Traces in Large Language Models](the_shape_of_reasoning_topological_analysis_of_reasoning_traces_in_large_languag.md)
- [\[ACL 2026\] d-TreeRPO: Towards More Reliable Policy Optimization for Diffusion Language Models](../../ACL2026/reinforcement_learning/d-treerpo_towards_more_reliable_policy_optimization_for_diffusion_language_model.md)
- [\[ICML 2026\] Coupled Variational Reinforcement Learning for Language Model General Reasoning](coupled_variational_reinforcement_learning_for_language_model_general_reasoning.md)

</div>

<!-- RELATED:END -->
