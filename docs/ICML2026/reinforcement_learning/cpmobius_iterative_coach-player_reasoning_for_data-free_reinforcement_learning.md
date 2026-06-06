---
title: >-
  [Paper Note] CPMöbius: Iterative Coach–Player Reasoning for Data-Free Reinforcement Learning
description: >-
  [ICML 2026][Reinforcement Learning][Data-free RL] Transforming self-play from "adversarial" to "collaborative": The Coach generates problems, the Player solves them…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Data-free RL"
  - "Coach-Player"
  - "Curriculum Generation"
  - "GRPO"
  - "Multi-agent Collaboration"
date: 2026-05-08
content_hash: 4d07b4cd685b55b4
---

# CPMöbius: Iterative Coach–Player Reasoning for Data-Free Reinforcement Learning

**Conference**: ICML 2026  
**arXiv**: [2602.02979](https://arxiv.org/abs/2602.02979)  
**Code**: https://github.com/thunlp/CPMobius  
**Area**: LLM Reasoning / Reinforcement Learning / Self-play  
**Keywords**: Data-free RL, Coach-Player, Curriculum Generation, GRPO, Multi-agent Collaboration

## TL;DR
Transforming self-play from "adversarial" to "collaborative": The Coach generates problems, the Player solves them, and the Coach receives a reward defined as "Player's progress magnitude $\times$ Player's success rate." Without any external training data, Qwen2.5-Math-7B-Instruct achieves a +4.9 average increase across six math benchmarks and +5.4 on OOD tasks, outperforming existing unsupervised methods like RENT and R-Zero.

## Background & Motivation
**Background**: Enhancing LLM reasoning capabilities primarily relies on SFT + RLVR (Reinforcement Learning with Verifiable Rewards) through repeated fine-tuning on high-quality human-annotated datasets. Models like OpenAI o1 and DeepSeek-R1 depend on massive amounts of math and code problems. Self-play offers a new direction by generating training signals from the model itself to achieve "data-free" training, with representative works like R-Zero and AbsoluteZero mostly adopting adversarial settings where one party attempts to stump the other.

**Limitations of Prior Work**: (1) Adversarial self-play is highly unstable; the problem-setter tends to generate meaningless or unlearnable problems to "defeat" the Player, leading to collapse (e.g., R-Zero fails to train on OpenMath-Nemotron). (2) Pure entropy minimization methods (RENT) use the model's own confidence as a reward, lacking external progress signals and yielding limited improvements (+3.4 on Qwen2.5-Math-7B). (3) Most self-play works lack explicit curriculum signals, leading to random difficulty drift.

**Key Challenge**: Self-play aims for "open, adaptive, and consistently within the zone of proximal development," but adversarial mechanisms inherently create tension where "Setter's gain = Player's failure," often degrading training. Conversely, fully unsupervised methods lack progress metrics and are easily misled by their own confidence.

**Goal**: To discover a self-play paradigm that does not rely on external data yet remains stable, learnable, and generates a curriculum of monotonically increasing difficulty to continuously improve Player's mathematical reasoning.

**Key Insight**: Inspired by the human coach-player relationship (where a coach's reward comes from the player's growth rather than defeating the player), the authors design the setter and solver to be collaborative rather than adversarial. The Coach's reward is directly tied to the Player's "progress magnitude"; generating problems that simply "stump" the Player yields no reward, mechanically preventing collapse.

**Core Idea**: Utilizing a "multiplicative reward" $R^{\text{Coach}}_i = R^{\text{Player}}_i \cdot \Delta_t$ to encourage the Coach to pursue "problems solved by the Player + overall Player capability growth," shifting self-play from zero-sum to positive-sum.

## Method

### Overall Architecture
Two independent policies, $\pi_\theta^{\text{C}}$ (Coach) and $\pi_\phi^{\text{P}}$ (Player), co-evolve in a 4-stage cycle: (1) Coach samples $m$ problems; (2) For each problem, the Player samples $n$ solutions, obtains pseudo-labels $y_i^*$ via majority voting, calculates binary rewards and GRPO advantages for each solution, and updates $\phi$ using GRPO; (3) Accuracy improvement $\Delta_t$ before and after the update is calculated on a fixed small validation set $\mathcal{D}_{\text{val}}$ (AMC in experiments) as "environmental feedback"; (4) The Coach updates $\theta$ via REINFORCE using $R^{\text{Coach}}_i = R^{\text{Player}}_i \cdot \Delta_t$. The entire loop uses no external problem sets, and the Coach is initialized only via a one-time SFT warm-up.

### Key Designs
1.  **Multiplicative Coach Reward + Collaboration instead of Adversary**:
    - **Function**: Provides the Coach with a scalar reward that simultaneously encourages "solvability" and "global progress," eliminating degenerate solutions like "unsolvable problems" or "already mastered trivial problems."
    - **Mechanism**: $R^{\text{Coach}}_i = R^{\text{Player}}_i \cdot \Delta_t$, where $R^{\text{Player}}_i = \frac{1}{n}\sum_j r_{i,j}$ is the Player's average success rate for that problem, and $\Delta_t = \text{Acc}_{\text{val}}(\pi_{\phi_{t+1}}^{\text{P}}) - \text{Acc}_{\text{val}}(\pi_{\phi_t}^{\text{P}})$ is the accuracy gain on the validation set resulting from one Player training step. If either factor is zero or negative (too difficult to solve $R^{\text{Player}}=0$, or no progress made $\Delta_t \le 0$), the problem is "penalized."
    - **Design Motivation**: Directly resolves the collapse of adversarial self-play—generating unsolvable problems immediately results in a zero reward for the Coach. Simultaneously, $\Delta_t$ introduces a "real learning progress" signal, which is closer to ground-truth progress than RENT's self-confidence reward.

2.  **Difficulty-Filtered Batching**:
    - **Function**: Performs a cheap rollout before the Coach submits problems, retaining only those in the "optimal teaching zone" ($0.2 \le \text{acc} \le 0.8$) to ensure every training batch falls within the zone of proximal development.
    - **Mechanism**: For each candidate $x_i$, the Player runs $n$ rollouts to calculate majority voting accuracy $acc_i = \frac{1}{n}\sum_j \mathbb{I}[y_{i,j} = y_i^*]$. If $acc_i$ falls outside $[0.2, 0.8]$, the problem is discarded and instantly resampled until $m$ problems are collected.
    - **Design Motivation**: GRPO's advantages and gradients become zero in batches where rewards are all 0 or all 1. Difficulty filtering removes extreme cases from the Coach to ensure the training signal is never empty and aligns with human curriculum learning intuition by focusing on "active attempts with partial success."

3.  **Coach SFT warm-up (One-time, non-data leakage)**:
    - **Function**: Performs a lightweight SFT on the Coach using 4K PRIME Eurus-2-RL-Data before co-evolution to establish the basic ability to "generate math problems."
    - **Mechanism**: Does not access validation or test sets; only trains the Coach on "question formatting and diagnostics." Subsequently, co-evolution uses zero external data. The paper defines "data-free" specifically for the "co-evolution stage."
    - **Design Motivation**: Experiments showed that using the base model directly as a Coach results in ambiguous or unsolvable problems, causing $\Delta_t$ noise to explode. Warm-up is the minimum cost initialization for "teaching skills" rather than a transfer of "mathematical knowledge." Ablation without Coach Warm-up shows a drop to 23.7 (vs 28.8), proving its necessity.

### Loss & Training
The Player uses GRPO: for $n$ solutions per problem, advantages are $A_{i,j} = (r_{i,j} - \text{mean})/\text{std}$, updated within a trust region. The Coach uses REINFORCE: $\nabla_\theta J = \frac{1}{m}\sum_i R^{\text{Coach}}_i \nabla_\theta \log \pi_\theta^{\text{C}}(x_i)$. The validation set is fixed as AMC (moderate difficulty, neither saturated nor sparse). The choice is validated for robustness using Minerva and OlympiadBench. Training is conducted in the `verl` framework using 4–8 A800-80GB GPUs, with batch=16 and rollout=16.

## Key Experimental Results

### Main Results
Evaluated on four base models (Qwen2.5-Math-1.5B / OpenMath-Nemotron-1.5B / OctoThinker-3B-Hybrid-Zero / Qwen2.5-Math-7B-Instruct) across six benchmarks (AMC + AIME 2024/2025 + Minerva + MATH + Olympiad).

| Base / Method | Avg | OOD Avg | Minerva | MATH | Olympiad |
|---------------|-----|---------|---------|------|----------|
| Qwen2.5-Math-1.5B base | 23.3 | 19.8 | 16.3 | 56.2 | 23.4 |
| + R-Zero (Iter 3) | 27.1 | 24.7 | 19.3 | 62.4 | 26.8 |
| + RENT | 27.1 | 24.7 | 19.0 | 62.2 | 27.1 |
| **+ CPMöbius** | **28.8** | **26.8** | **28.0** | **63.1** | 26.9 |
| Qwen2.5-Math-7B-Instruct base | 35.8 | 33.0 | 34.6 | 78.0 | 37.4 |
| + RENT | 39.2 | 37.6 | 38.8 | 83.8 | 38.8 |
| **+ CPMöbius** | **40.7** | **38.4** | **44.9** | **84.2** | 38.3 |

The most significant improvement is on Minerva: Qwen-1.5B 16.3→28.0 (+71.8%) and Qwen-7B 34.6→44.9 (+29.8%), indicating that capabilities trained on AMC generalize well to OOD math domains. R-Zero failed on OpenMath-Nemotron-1.5B (marked as "–" in tables), while CPMöbius pushed it from 59.5 to 62.1.

### Ablation Study

| Configuration | Avg | OOD Avg | Key Finding |
|------|-----|---------|----------|
| Full CPMöbius | 28.8 | 26.8 | Full Framework |
| w/o Coach Update | 25.3 | 23.1 | Degrades to static curriculum if Coach is fixed, -3.5 |
| w/o Coach Warm-up | 23.7 | 21.2 | Poor problem quality from base model coach, -5.1 |
| w/o Instruction Filter | 24.9 | 22.5 | High GRPO gradient noise without difficulty filtering, -3.9 |

### Key Findings
- All three ablation modules are indispensable, with Coach Warm-up having the largest impact (–5.1). Difficulty filtering (–3.9) and Coach Update (–3.5) follow, showing each component of the collaborative mechanism contributes significantly.
- Training Dynamics: Player solution consistency decreases monotonically (Coach presents harder problems), while problem length increases and Player response length decreases (Player becomes more concise)—indicating simultaneous improvement in curriculum difficulty and solution efficiency.
- CPMöbius still improves when replacing the AMC validation set with Minerva or OlympiadBench, proving results are not due to "AMC data leakage."
- It provides a +4.9 gain for the already RL-optimized Qwen-7B-Instruct and +2.6 for OpenMath-Nemotron (trained on 5.5M SFT samples), demonstrating that CPMöbius can push beyond the ceilings of existing training paradigms.

## Highlights & Insights
- **Multiplicative Reward = Mathematical prevention of collapse**: This is the cleanest design in the paper—using a scalar product to tie "local signals (solved problem)" with "global signals (capability growth)," making both collapse and reward-free scenarios impossible. This approach can be transferred to any task with verifiable "process capabilities" like code or reasoning.
- **AMC as $\Delta_t$ source is a high-ROI choice**: Providing a difficulty band that is neither saturated nor sparse gives the Coach "high-signal, low-variance" feedback. Using a benchmark as a reward proxy is a noteworthy strategy.
- **Honest "Data-free" boundary**: The paper explicitly excludes warm-up from its "data-free" claim. This terminological restraint is rare in self-play papers and reminds readers to check for "initialization data leakage" when evaluating such work.
- **No reward model**: Relying entirely on verifiable rewards (correctness of math answers) + validation accuracy delta avoids reward hacking associated with RLHF-type methods, though it limits application to tasks where correctness can be programmatically judged.

## Limitations & Future Work
- The method relies on "verifiable answers." While it works for math, extending it to code, theorem proving, or long-form writing requires redesigning verifiers.
- The AMC validation set has only ~40 problems; the $\Delta_t$ signal is coarse and noisy, leading to visible fluctuations in training curves. Future work could consider ensemble validation sets or EWMA smoothing.
- Coach and Player are derived from the same base. The potential of heterogeneous Coaches (e.g., using a larger model as a Coach) was not explored; the collaborative architecture does not necessarily need to be symmetric.
- Compute Cost: Training both Coach and Player policies on 4–8 A800 GPUs was not explicitly compared to SFT on high-quality data using equivalent compute.

## Related Work & Insights
- **vs R-Zero (Adversarial self-play)**: R-Zero uses a challenger to stump the solver, which collapsed on OpenMath-Nemotron. CPMöbius provides stable gains across all 4 bases; collaboration is more robust than competition when faced with unlearnable problems.
- **vs RENT (Entropy minimization)**: RENT uses self-confidence as a reward (+3.4 on Qwen-7B), whereas CPMöbius (+4.9) and OOD (+0.8) prove that external progress signals are more reliable than internal confidence signals.
- **vs SFT on curated data**: CPMöbius outperforms base + RENT/R-Zero without any external problem sets, offering a viable "zero-data RL" solution for labs with ample GPUs but limited data.
- **vs RLHF**: Both use RL to optimize policies, but CPMöbius requires neither a reward model nor human preference data, providing a self-play template for verifiable tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ Collaborative self-play + multiplicative reward is a clear, original design, though difficulty filters and progress metrics have precedents.
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 bases × 6 benchmarks + 3 ablations + validation robustness + training visualization, though compute cost comparison vs SFT is missing.
- Writing Quality: ⭐⭐⭐⭐ The 4-step cycle and Figure 2 explain the architecture clearly; formulas and ablations are well-mapped; "data-free" boundaries are honestly stated.
- Value: ⭐⭐⭐⭐ Provides a practical "zero-data curriculum generation" solution for math reasoning RL. The concepts are transferable to any verifiable task, and open-source code reduces the reproduction barrier.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] How Reasoning Evolves from Post-Training Data: An Empirical Study Using Chess](how_reasoning_evolves_from_post-training_data_an_empirical_study_using_chess.md)
- [\[ICML 2026\] Single-Rollout Hidden-State Dynamics for Training-Free RLVR Data Selection](single-rollout_hidden-state_dynamics_for_training-free_rlvr_data_selection.md)
- [\[ICML 2026\] D$^2$Evo: Dual Difficulty-Aware Self-Evolution for Data-Efficient Reinforcement Learning](d2evo_dual_difficulty-aware_self-evolution_for_data-efficient_reinforcement_lear.md)
- [\[ICML 2026\] Revisiting Regularized Policy Optimization for Stable and Efficient Reinforcement Learning in Two-Player Games](revisiting_regularized_policy_optimization_for_stable_and_efficient_reinforcemen.md)
- [\[CVPR 2026\] See It, Say It, Sorted: An Iterative Training-Free Framework for Visually-Grounded Multimodal Reasoning in LVLMs](../../CVPR2026/reinforcement_learning/see_it_say_it_sorted_an_iterative_training-free_framework_for_visually-grounded_.md)

</div>

<!-- RELATED:END -->
