---
title: >-
  [Paper Note] D$^2$Evo: Dual Difficulty-Aware Self-Evolution for Data-Efficient Reinforcement Learning
description: >-
  [ICML 2026][Reinforcement Learning][GRPO] In each RL iteration, D$^2$Evo utilizes the current Solver to estimate difficulty and select medium-difficulty real samples as anchors. It then trains a Questioner to synthesize…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "GRPO"
  - "Difficulty-Aware"
  - "Self-Evolution"
  - "Question Generation"
  - "Data-Efficient RL"
date: 2026-05-08
content_hash: 836eff05062d87b6
---

# D$^2$Evo: Dual Difficulty-Aware Self-Evolution for Data-Efficient Reinforcement Learning

**Conference**: ICML 2026  
**arXiv**: [2605.17037](https://arxiv.org/abs/2605.17037)  
**Code**: No public link yet  
**Area**: LLM Reasoning / Reinforcement Learning / Self-Evolution Training  
**Keywords**: GRPO, Difficulty-Aware, Self-Evolution, Question Generation, Data-Efficient RL  

## TL;DR
In each RL iteration, D$^2$Evo utilizes the current Solver to estimate difficulty and select medium-difficulty real samples as anchors. It then trains a Questioner to synthesize new problems of equivalent difficulty around these anchors. Consequently, it outperforms the GRPO baseline trained on 19K real data in both mathematical and general reasoning using < 2K real math problems.

## Background & Motivation

**Background**: Group-level RL, represented by GRPO, has become a mainstream paradigm for enhancing LLM reasoning capabilities post-training. The approach involves sampling a group of responses for each problem and performing policy gradient updates using relative advantage.

**Limitations of Prior Work**: GRPO is extremely sensitive to the difficulty distribution of training samples. If problems are too easy, all responses in a group are correct; if too difficult, all are incorrect. When intra-group variance drops to zero, the advantage signal collapses, leading to zero gradients and wasted training steps. However, only a small portion of existing mathematics datasets (e.g., Math12K, OpenRs-7K) falls within the medium difficulty range. Furthermore, after one epoch, most previously "medium" problems are mastered and become "easy," while "hard" problems remains unsolvable, causing **effective signal samples to diminish as training progresses**.

**Key Challenge**: The authors categorize these issues as "Effective Data Scarcity" and "Dynamic Difficulty Shifts." The root cause is the mismatch between static training data difficulty and dynamic Solver capabilities, leading to rapid exhaustion of iterative gains. Existing self-synthesis solutions are either anchorless (R-Zero, Absolute-Zero), causing entropy collapse and out-of-distribution generation, or use anchors without difficulty control (SPICE), resulting in synthesized problems clustered at easy/hard extremes, still wasting gradients.

**Goal**: To enable the Questioner to generate new problems centered on anchors that are "just right" for the current Solver in every iteration, allowing both to co-evolve and guide each other.

**Key Insight**: Based on the conclusion by Bae et al.—under binary rewards, the lower bound of KL divergence and the pass rate $p$ satisfy $D_{\mathrm{KL}}(\pi_{\mathrm{init}}\|\pi^{*})\ge p(1-p)/(2\beta^{2})$, which is maximized at $p=0.5$—the authors prove that "medium difficulty" is a theoretically supported optimal learning signal zone. Combined with empirical observations that the proportion of medium-difficulty samples drops sharply after one epoch of GRPO, they **naturally propose a cyclical structure of "re-mining anchors based on the current Solver + generating new problems of equivalent difficulty" in each iteration.**

**Core Idea**: Employs "dual difficulty awareness"—the Questioner uses difficulty rewards to align with the target difficulty band, and the Solver continuously trains using a hybrid buffer (real anchors + synthetic problems of the same difficulty)—enabling problems and solvers to co-evolve across multiple iterations to maximize the utility of limited real data.

## Method

### Overall Architecture
D$^2$Evo is a multi-round iterative self-evolution RL loop. Each round $t$ consists of four steps:

1.  **Difficulty Estimation**: The previous Solver $\pi_S^{t-1}$ (frozen) performs $N=32$ rollouts per candidate real problem. Difficulty is estimated as $\text{Difficulty}(q)=(1-\text{correct}/N)\times 100$. A medium-difficulty subset $\mathcal{D}^{mid}_{real}$ is selected based on thresholds $[\textit{low}=0.4, \textit{high}=0.8]$ to serve as anchors.
2.  **Questioner Training**: Conditioned on anchors $(q_{\text{anc}}, y_{\text{anc}}, s)$, the $\pi_Q$ is trained using GRPO with difficulty-aware rewards to generate new problems $\tilde q$, ensuring their pass rates under the current Solver fall within the target band $[\tau_\ell, \tau_u]$.
3.  **Hybrid Buffer Construction**: Synthesized problems outside the difficulty range are filtered out. Pseudo-labels are generated via majority voting and verified by GPT. This forms $\mathcal{D}^{mid}_{gen}$, which is combined with anchors to create $\mathcal{D}_{hybrid}=\mathcal{D}^{mid}_{real}\cup\mathcal{D}^{mid}_{gen}$.
4.  **Solver Training**: The Solver is updated on $\mathcal{D}_{hybrid}$ using GRPO with the reward $R_{\mathrm{comp}}=\alpha R_{\mathrm{Acc}}+(1-\alpha) R_{\mathrm{Fmt}}$.

The updated Solver then serves as the difficulty evaluator for the next round, closing the loop. This cycle aligns data generation, filtering, and model updates within a single difficulty coordinate system, avoiding inter-round shifts.

### Key Designs

1.  **Medium-Difficulty Anchor Mining Based on Current Solver**:
    *   **Function**: Before each iteration, the frozen current Solver performs fine-grained difficulty estimation on candidate real data to select a subset that is neither too easy nor too hard as anchors.
    *   **Mechanism**: $N=32$ rollouts per problem are performed, filtering by $\text{Acc}_S(q)\in[\textit{low}, \textit{high}]$ so that anchors naturally fall into the zone with the richest advantage signals. As the Solver strengthens, former "medium" problems are promoted to "easy" and removed, while "hard" problems are demoted into the anchor pool. The **anchor pool automatically updates with capability shifts**.
    *   **Design Motivation**: Training on static datasets with pre-labeled difficulty fails as the model strengthens. R-Zero/AZR lead to aimless generation without anchors; this mechanism recalibrates the "learning frontier" of real problems in each round.

2.  **Difficulty-Aware Reward with Plateau Target Band (Questioner)**:
    *   **Function**: Encourages the Questioner to produce problems where the Solver pass rate falls exactly within $[\tau_\ell, \tau_u]$, rather than simply pursuing higher or lower difficulty.
    *   **Mechanism**: Pass rate $x=\text{Acc}_S(\tilde q)$ is obtained via $N_v$ Solver rollouts on generated problem $\tilde q$. A piecewise reward $r_{\text{diff}}(x)=1$ is applied if $\tau_\ell\le x\le\tau_u$; otherwise, it takes $(x/\tau_\ell)^a$ for $x<\tau_\ell$ or $((1-x)/(1-\tau_u))^a$ for $x>\tau_u$, where $a\ge 1$ determines the decay sharpness. This is combined with format constraints to form $R_{\mathrm{comp}}$.
    *   **Design Motivation**: Solver pass rates shift significantly after updates, making fixed labels obsolete. The plateau reward concentrates training signals on valuable intervals, preventing the Questioner from drifting to extremes due to diversity rewards.

3.  **Hybrid Buffer of Real Anchors + Synthetic Problems**:
    *   **Function**: The Solver trains on both synthetic problems and real labeled anchors of equivalent difficulty within a unified buffer.
    *   **Mechanism**: Synthetic problems use Solver majority voting for pseudo-labels $\tilde y$, requiring the pass rate to remain in $[\tau_\ell, \tau_u]$. GPT-5.2 provides secondary verification to reduce noise. These are merged with real anchors into $\mathcal{D}_{hybrid}$ for GRPO updates.
    *   **Design Motivation**: Purely synthetic data suffers from pseudo-label noise, while purely real data is scarce. The hybrid buffer uses real anchors for grounded supervision and synthetic data for a continuous stream of fresh signals.

### Loss & Training
Both sides utilize GRPO (Eq. 2), sharing LLM weights and distinguishing roles via prompts. The Questioner's reward is $R_{\mathrm{comp}}$ as defined above, while the Solver's reward is $\alpha R_{\mathrm{Acc}}+(1-\alpha)R_{\mathrm{Fmt}}$ (requiring `<think>...</think>` and `\boxed{}` structures). Difficulty thresholds are $\textit{low}=0.4, \textit{high}=0.8$, with $N=32$ rollouts and 3 self-evolution iterations.

## Key Experimental Results

### Main Results
Evaluation across 7 math benchmarks (AMC, Minerva, MATH-500, GSM8K, Olympiad-Bench, AIME-2024, AIME-2025) using Base, Full Data (19K GRPO), R-Zero, AZR, and SPICE as baselines:

| Model / Method | # Real Data | Math Avg. (7 metrics) | Gain vs. Base |
|---|---|---|---|
| Qwen3-4B-Base | – | 43.87 | – |
| + Full Data (GRPO) | 19K | 49.28 | +5.41 |
| + R-Zero (Iter 3) | – | 46.91 | +3.04 |
| + AZR | – | 46.36 | +2.49 |
| + SPICE | 20K | 50.59 | +6.72 |
| **Ours (D$^2$Evo Iter 3)** | **0.1K** | **51.35** | **+7.48** |
| Qwen3-8B-Base | – | 47.24 | – |
| + Full Data | 19K | 52.70 | +5.46 |
| + SPICE | 20K | 54.34 | +7.10 |
| **Ours (D$^2$Evo Iter 3)** | **0.4K** | **55.32** | **+8.08** |
| Llama-3.1-8B-Inst | – | 29.35 | – |
| + Full Data | 19K | 31.10 | +1.75 |
| **Ours (D$^2$Evo Iter 3)** | **0.4K** | **33.09** | **+3.74** |

In general reasoning (SuperGPQA, MMLU-Pro, BBEH average), D$^2$Evo also outperformed Full Data baselines across all backbones (Qwen3-4B, 8B, Llama-3.1-8B).

### Ablation Study

| Configuration | Math Avg. | General Avg. | Description |
|---|---|---|---|
| D$^2$Evo (full, Qwen3-4B) | 51.35 | 32.16 | Full Method |
| w/o Questioner | 47.94 | 30.75 | No synthesis, Solver trained only on real anchors |
| w/o share weight | 49.99 | 31.62 | Questioner and Solver use independent weights |
| w/o synthesis data | 48.71 | 31.65 | Solver trained only on anchors |
| w/ random anchor data | 49.22 | 31.93 | Random sampling instead of difficulty-based anchors |

### Key Findings
- Removing the Questioner leads to the largest drop (3.4 points in math), indicating "self-synthesized medium-difficulty problems" drive the performance ceiling.
- Weight sharing slightly outperforms independent weights (51.35 vs 49.99), suggesting "learning to pose problems" sharpens the model's understanding of structure, providing positive co-evolution feedback.
- Stability: D$^2$Evo improves consistently across 3 iterations, whereas R-Zero fluctuates or declines, highlighting the necessity of difficulty-aware anchors for multi-round stability.

## Highlights & Insights
- **Maximizing GRPO Advantage Signals as Anchor Selection Criteria**: By deriving the $p \approx 0.5$ range and using dynamic rollouts, "medium difficulty" becomes a moving target that tracks model capability, upgrading curriculum learning to "adaptive materials based on student status."
- **Co-evolution via Weight Sharing**: Training the same LLM to both solve and generate problems of specific difficulty creates implicit auxiliary task regularization, where questioning ability enhances structural representation for solving.
- **Extreme Sample Efficiency**: With only hundreds of anchors and generated problems per round (total $\le 2$K real samples), the framework surpasses 19K Full Data models. This is ideal for scenarios with expensive annotations or limited private data.

## Limitations & Future Work
- **Computational Overhead**: Estimating difficulty requires $N=32$ rollouts and GPT verification per round. The total FLOPs compared to standard training are not detailed, and cost-effectiveness at 70B+ scales remains unverified.
- **Task Scope**: Validation is limited to math and general reasoning, excluding longer-horizon tasks like coding, agents, or multi-step tool calls.
- **Diversity Risks**: Self-evolution risks amplifying bias in small loops; direct quantitative analysis of synthesized problem diversity is lacking.
- **Hyperparameters**: Thresholds and target bands are preset rather than adaptive.

## Related Work & Insights
- **vs R-Zero (Huang et al., 2025)**: R-Zero produces homogeneous problems due to lack of anchors; D$^2$Evo avoids drift via real medium-difficulty anchors.
- **vs Absolute-Zero (Zhao et al., 2025)**: AZR lacks difficulty control; D$^2$Evo uses plateau rewards to keep problems in the instructional mid-range.
- **vs SPICE (Liu et al., 2025a)**: SPICE relies on 20K document-level corpora without difficulty awareness; D$^2$Evo proves that "difficulty awareness" is more critical than raw data volume.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

## Rating
- Novelty: Under Review
- Experimental Thoroughness: Under Review
- Writing Quality: Under Review
- Value: Under Review

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Easy Samples Are All You Need: Self-Evolving LLMs via Data-Efficient Reinforcement Learning](../../ACL2026/reinforcement_learning/easy_samples_are_all_you_need_self-evolving_llms_via_data-efficient_reinforcemen.md)
- [\[ICML 2026\] The Surprising Difficulty of Search in Model-Based Reinforcement Learning](the_surprising_difficulty_of_search_in_model-based_reinforcement_learning.md)
- [\[AAAI 2026\] STELAR-Vision: Self-Topology-Aware Efficient Learning for Aligned Reasoning in Vision](../../AAAI2026/reinforcement_learning/stelar-vision_self-topology-aware_efficient_learning_for_aligned_reasoning_in_vi.md)
- [\[ICML 2026\] InftyThink+: Effective and Efficient Infinite-Horizon Reasoning via Reinforcement Learning](inftythink_effective_and_efficient_infinite-horizon_reasoning_via_reinforcement_.md)
- [\[ICML 2026\] CPMöbius: Iterative Coach–Player Reasoning for Data-Free Reinforcement Learning](cpmobius_iterative_coach-player_reasoning_for_data-free_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
