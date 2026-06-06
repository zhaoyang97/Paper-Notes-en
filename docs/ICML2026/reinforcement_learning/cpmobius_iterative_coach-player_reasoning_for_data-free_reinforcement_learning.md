---
title: >-
  [Paper Note] CPMöbius: Iterative Coach–Player Reasoning for Data-Free Reinforcement Learning
description: >-
  [ICML 2026][Reinforcement Learning][Data-free RL] Transforms self-play from "adversarial" to "collaborative": the Coach generates problems, the Player solves them…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Data-free RL"
  - "Coach-Player"
  - "Curriculum Generation"
  - "GRPO"
  - "Multi-agent Collaboration"
date: 2026-05-08
content_hash: ef18baa88b1e47a5
---

# CPMöbius: Iterative Coach–Player Reasoning for Data-Free Reinforcement Learning

**Conference**: ICML 2026  
**arXiv**: [2602.02979](https://arxiv.org/abs/2602.02979)  
**Code**: https://github.com/thunlp/CPMobius  
**Area**: LLM Reasoning / Reinforcement Learning / Self-play  
**Keywords**: Data-free RL, Coach-Player, Curriculum Generation, GRPO, Multi-agent Collaboration

## TL;DR
Transforms self-play from "adversarial" to "collaborative": the Coach generates problems, the Player solves them, and the Coach receives a reward equal to "Player improvement × Player solve rate." Without any external training data, Qwen2.5-Math-7B-Instruct achieves an average +4.9 and OOD +5.4 across six math benchmarks, surpassing existing unsupervised methods like RENT/R-Zero.

## Background & Motivation
**Background**: Mainstream approaches to improving LLM reasoning rely on SFT + RLVR (Reinforcement Learning with Verifiable Rewards), repeatedly fine-tuning on high-quality human-curated problem sets; models like OpenAI o1 and DeepSeek-R1 depend on massive math/code datasets. Self-play offers a new direction, enabling "data-free" training by generating training signals internally, with representative works like R-Zero and AbsoluteZero mostly adopting adversarial settings—one side generates problems to challenge the other.

**Limitations of Prior Work**: (1) Adversarial self-play is highly unstable; the problem setter, aiming to "defeat" the Player, gradually generates meaningless or unsolvable problems (collapse). R-Zero even fails to train on OpenMath-Nemotron. (2) Pure entropy minimization methods (RENT) use the model's own confidence as reward, lacking external progress signals and yielding limited improvement (only +3.4 on Qwen2.5-Math-7B). (3) Most self-play works lack explicit curriculum signals, leading to random difficulty drift.

**Key Challenge**: Self-play aspires to be "open, adaptive, and always in the zone of proximal development," but adversarial mechanisms inherently pit "problem setter's gain = solver's failure," worsening over time; fully unsupervised approaches lack progress metrics and are easily misled by self-confidence.

**Goal**: To find a self-play paradigm that does not rely on external data yet produces a stable, learnable, and monotonically increasing curriculum, enabling continuous improvement in the Player's mathematical reasoning.

**Key Insight**: Inspired by the human coach–player relationship (where a coach's reward comes from the player's growth, not defeating the player), the authors design the problem setter and solver as collaborators rather than adversaries—the Coach's reward is directly tied to the Player's "improvement," so generating unsolvable problems yields no reward, structurally preventing collapse.

**Core Idea**: Use a "multiplicative reward" $R^{\text{Coach}}_i = R^{\text{Player}}_i \cdot \Delta_t$ so the Coach simultaneously pursues "problems solved by the Player" and "overall Player improvement," shifting self-play from zero-sum to positive-sum.

## Method

### Overall Architecture
Two independent policies, $\pi_\theta^{\text{C}}$ (Coach) and $\pi_\phi^{\text{P}}$ (Player), co-evolve in a 4-stage loop: (1) Coach samples $m$ problems; (2) for each problem, Player samples $n$ solutions, majority voting yields pseudo-label $y_i^*$, computes binary reward and GRPO advantage for each solution, and updates $\phi$ via GRPO; (3) on a fixed small validation set $\mathcal{D}_{\text{val}}$ (AMC in experiments), the accuracy difference before and after update, $\Delta_t$, serves as "environment feedback"; (4) Coach updates $\theta$ via REINFORCE using $R^{\text{Coach}}_i = R^{\text{Player}}_i \cdot \Delta_t$. The entire loop uses no external problem sets; Coach is only pre-warmed with a one-time SFT.

### Key Designs
1. **Multiplicative Coach Reward + Collaboration over Adversarial**:

    - **Function**: Provides the Coach with a scalar reward that simultaneously incentivizes "being solved" and "driving global improvement," eliminating degenerate solutions like "unsolvable" or "already mastered" problems.
    - **Mechanism**: $R^{\text{Coach}}_i = R^{\text{Player}}_i \cdot \Delta_t$, where $R^{\text{Player}}_i = \frac{1}{n}\sum_j r_{i,j}$ is the Player's average solve rate for the problem, and $\Delta_t = \text{Acc}_{\text{val}}(\pi_{\phi_{t+1}}^{\text{P}}) - \text{Acc}_{\text{val}}(\pi_{\phi_t}^{\text{P}})$ is the accuracy gain on the validation set after one Player update. If either factor is zero or negative, the problem is "penalized": too hard ($R^{\text{Player}}=0$), or no improvement ($\Delta_t \le 0$).
    - **Design Motivation**: Directly addresses adversarial self-play collapse—unsolvable problems immediately yield zero reward for the Coach; $\Delta_t$ introduces a "true learning progress" signal, closer to ground-truth improvement than RENT's self-confidence reward.

2. **Difficulty-Filtered Batching**:

    - **Function**: Before submitting problems, the Coach performs a cheap rollout, retaining only those in the "optimal teaching zone" $0.2 \leq acc \leq 0.8$, ensuring each training batch is in the zone of proximal development.
    - **Mechanism**: For each candidate $x_i$, Player runs $n$ solutions, computes majority voting accuracy $acc_i = \frac{1}{n}\sum_j \mathbb{I}[y_{i,j} = y_i^*]$; problems outside $[0.2, 0.8]$ are discarded and resampled until $m$ are collected. This filter removes "already mastered" and "completely unsolvable" samples.
    - **Design Motivation**: GRPO's advantage vanishes and gradients are zero on all-0 or all-1 reward batches; difficulty filtering handles occasional extreme problems from the Coach, ensuring non-empty training signals and making "active attempts with partial success" the main learning scenario, aligning with human curriculum intuition.

3. **Coach SFT Warm-up (One-time, Not Data Leakage)**:

    - **Function**: Before co-evolution, Coach undergoes lightweight SFT on 4K PRIME Eurus-2-RL-Data to acquire basic "problem generation" ability.
    - **Mechanism**: No access to validation/test sets; only trains Coach's "question format and diagnostic ability." Subsequent co-evolution uses zero external data. The paper explicitly defines "data-free" as the "co-evolution phase."
    - **Design Motivation**: Experiments show that using the base model as Coach directly leads to ambiguous/unsolvable problems, causing $\Delta_t$ signal noise to explode; warm-up is a minimal-cost initialization of "teaching skill," not "math knowledge" transfer. Ablation w/o Coach Warm-up drops to 23.7 (vs 28.8), proving this step is necessary but not data leakage.

### Loss & Training
Player uses GRPO: for each problem, $n$ solutions, advantage $A_{i,j} = (r_{i,j} - \text{mean})/\text{std}$, updated within a trust region; Coach uses REINFORCE: $\nabla_\theta J = \frac{1}{m}\sum_i R^{\text{Coach}}_i \nabla_\theta \log \pi_\theta^{\text{C}}(x_i)$. Validation set is fixed as AMC (moderate difficulty, neither saturated nor sparse); Minerva/OlympiadBench are also used for robustness checks. All training is on the verl framework, 4–8 A800-80GB GPUs, batch=16, rollout=16.

## Key Experimental Results

### Main Results
On Qwen2.5-Math-1.5B / OpenMath-Nemotron-1.5B / OctoThinker-3B-Hybrid-Zero / Qwen2.5-Math-7B-Instruct bases, evaluated on AMC + AIME 2024/2025 + Minerva + MATH + Olympiad, totaling 6 benchmarks.

| Base / Method | Avg | OOD Avg | Minerva | MATH | Olympiad |
|---------------|-----|---------|---------|------|----------|
| Qwen2.5-Math-1.5B base | 23.3 | 19.8 | 16.3 | 56.2 | 23.4 |
| + R-Zero (Iter 3) | 27.1 | 24.7 | 19.3 | 62.4 | 26.8 |
| + RENT | 27.1 | 24.7 | 19.0 | 62.2 | 27.1 |
| **+ CPMöbius** | **28.8** | **26.8** | **28.0** | **63.1** | 26.9 |
| Qwen2.5-Math-7B-Instruct base | 35.8 | 33.0 | 34.6 | 78.0 | 37.4 |
| + RENT | 39.2 | 37.6 | 38.8 | 83.8 | 38.8 |
| **+ CPMöbius** | **40.7** | **38.4** | **44.9** | **84.2** | 38.3 |

Most significant improvement is on Minerva: Qwen-1.5B 16.3→28.0 (+71.8%), Qwen-7B 34.6→44.9 (+29.8%), indicating strong OOD transfer of AMC-trained capabilities. R-Zero completely fails on OpenMath-Nemotron-1.5B (marked "–" in table), while CPMöbius still improves it from 59.5 to 62.1.

### Ablation Study

| Configuration | Avg | OOD Avg | Key Findings |
|---------------|-----|---------|-------------|
| Full CPMöbius | 28.8 | 26.8 | Complete framework |
| w/o Coach Update | 25.3 | 23.1 | Static curriculum after Coach is fixed, -3.5 |
| w/o Coach Warm-up | 23.7 | 21.2 | Poor problem quality with base model as Coach, -5.1 |
| w/o Instruction Filter | 24.9 | 22.5 | No difficulty filtering leads to noisy GRPO gradients, -3.9 |

### Key Findings
- All three ablation modules are indispensable, with Coach Warm-up having the largest impact (–5.1); difficulty filtering (–3.9) and Coach Update (–3.5) follow, indicating each part of the collaborative mechanism contributes significantly.
- Training dynamics: Player answer consistency monotonically decreases (Coach generates harder problems), while problem length increases and Player response length decreases (Player solves more concisely)—curriculum difficulty and solution efficiency both improve.
- Replacing AMC with Minerva/OlympiadBench as validation set, CPMöbius still improves, indicating collaboration is not "AMC data leakage."
- Even for RL-optimized Qwen-7B-Instruct, CPMöbius achieves +4.9; for SFT-trained OpenMath-Nemotron (+5.5M samples), still +2.6, showing CPMöbius can break through existing training paradigms' ceilings.

## Highlights & Insights
- **Multiplicative reward = mathematically prevents collapse**: The paper's cleanest design—using a scalar product to simultaneously bind "local signal (problem solved)" and "global signal (improvement)," making both collapse and reward-free scenarios impossible. This idea can be directly transferred to code/reasoning or any "verifiable process ability" task.
- **AMC as $\Delta_t$ signal source is high ROI**: Its moderate, non-saturated, non-sparse difficulty provides the Coach with "high-signal, low-variance" feedback; using a benchmark as a reward proxy is itself noteworthy.
- **Honest definition of "data-free"**: The paper explicitly excludes warm-up from the "data-free" claim, a rare restraint in self-play literature; it reminds readers to check for "initialization data leakage" in self-play works.
- **No reward model**: Relies entirely on verifiable reward (math answer correctness) + validation accuracy difference, avoiding RLHF-style reward hacking, but also limiting applicability (must be programmatically verifiable).

## Limitations & Future Work
- Authors acknowledge reliance on "verifiable answers," currently only validated in math; extending to code, theorem proving, or long-form writing would require new verifiers.
- AMC validation set has only ~40 problems, so $\Delta_t$ signal is coarse and noisy; training curves for $\Delta_t$ are visibly volatile. Future work could ensemble multiple validation sets or use EWMA smoothing.
- Coach and Player are homogeneous (same base); potential of heterogeneous Coach (e.g., larger model as Coach) is unexplored, and collaboration architecture may not require symmetry.
- Training cost: 4–8 A800 GPUs to train both Coach and Player; whether this is more efficient than "using equivalent compute for SFT on high-quality data" is not directly addressed.

## Related Work & Insights
- **vs R-Zero (adversarial self-play)**: R-Zero uses a challenger to defeat the solver, but collapses on OpenMath-Nemotron; CPMöbius stably improves all 4 bases. Collaboration > adversarial is more robust for unsolvable problems.
- **vs RENT (entropy minimization)**: RENT uses model self-confidence as reward, yielding +3.4 on Qwen-7B, while CPMöbius achieves +4.9 and +0.8 OOD; external progress signals are more reliable than internal confidence.
- **vs SFT on curated data**: CPMöbius requires no external problem sets, yet outperforms base + RENT/R-Zero on all bases; a practical "zero-data RL" solution for compute-limited but GPU-rich labs.
- **vs RLHF**: Both optimize policies via RL, but CPMöbius needs no reward model or human preference data; it provides a self-play template for "verifiable task RL."

## Rating
- Novelty: ⭐⭐⭐⭐ Collaborative self-play + multiplicative reward is a clearly original design, though difficulty filter/progress metrics have precedents; integration is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 bases × 6 benchmarks + 3 ablations + validation robustness + training dynamics visualization, but lacks direct compute cost vs SFT comparison.
- Writing Quality: ⭐⭐⭐⭐ 4-stage loop + figure 2 clearly explain the architecture, formulas and ablations are well-matched; honest "data-free" boundary explanation.
- Value: ⭐⭐⭐⭐ Provides a practical "zero-data curriculum generation" solution for math reasoning RL; ideas transferable to any verifiable task, open-sourced code lowers reproduction barrier.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] How Reasoning Evolves from Post-Training Data: An Empirical Study Using Chess](how_reasoning_evolves_from_post-training_data_an_empirical_study_using_chess.md)
- [\[CVPR 2026\] See It, Say It, Sorted: An Iterative Training-Free Framework for Visually-Grounded Multimodal Reasoning in LVLMs](../../CVPR2026/reinforcement_learning/see_it_say_it_sorted_an_iterative_training-free_framework_for_visually-grounded_.md)
- [\[ICML 2026\] Trajectory-Level Data Augmentation for Offline Reinforcement Learning](trajectory-level_data_augmentation_for_offline_reinforcement_learning.md)
- [\[AAAI 2026\] Reasoning or Memorization? Unreliable Results of Reinforcement Learning Due to Data Contamination](../../AAAI2026/reinforcement_learning/reasoning_or_memorization_unreliable_results_of_reinforcement_learning_due_to_da.md)
- [\[NeurIPS 2025\] RoiRL: Efficient, Self-Supervised Reasoning with Offline Iterative Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/roirl_efficient_self-supervised_reasoning_with_offline_iterative_reinforcement_l.md)

</div>

<!-- RELATED:END -->
