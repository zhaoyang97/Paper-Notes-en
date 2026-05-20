---
title: >-
  [Paper Note] Regret-Guided Search Control for Efficient Learning in AlphaZero
description: >-
  [ICLR 2026][Reinforcement Learning][AlphaZero] This paper proposes RGSC (Regret-Guided Search Control), a framework that trains a regret network to identify high-regret states and prioritizes restarting self-play from th…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "AlphaZero"
  - "search control"
  - "regret network"
  - "MCTS"
  - "board games"
date: 2026-05-08
content_hash: daf9eb417fb3d5a7
---

# Regret-Guided Search Control for Efficient Learning in AlphaZero

**Conference**: ICLR 2026
**arXiv**: [2602.20809](https://arxiv.org/abs/2602.20809)  
**Code**: [Project Page](https://rlg.iis.sinica.edu.tw/papers/rgsc)  
**Area**: Reinforcement Learning
**Keywords**: AlphaZero, search control, regret network, MCTS, board games

## TL;DR

This paper proposes RGSC (Regret-Guided Search Control), a framework that trains a regret network to identify high-regret states and prioritizes restarting self-play from these states, emulating the human learning strategy of repeatedly reviewing mistakes. RGSC outperforms AlphaZero by an average of 77 Elo across 9×9 Go, 10×10 Othello, and 11×11 Hex.

## Background & Motivation

**Learning Efficiency Gap**: AlphaZero requires millions of self-play games to reach superhuman strength, whereas human players achieve comparable proficiency with far fewer games. The key difference lies in the learning strategy.

**Human Learning Patterns**: Human players do not replay complete games from scratch each time; instead, they repeatedly revisit critical positions where mistakes were made until weaknesses are corrected. AlphaZero, by contrast, always begins from an empty board and updates all positions uniformly.

**The Concept of Search Control**: Sutton & Barto introduced search control in the Dyna framework — selecting informative states as starting points for simulated experience, rather than always beginning from the initial state.

**Limitations of Go-Exploit**: The prior work Go-Exploit implemented self-play restarts from historical states but used uniform sampling, which cannot distinguish the learning value of different states. As training progresses and most states are mastered, the efficiency of uniform sampling degrades sharply.

**Non-Stationarity Challenge**: The regret of high-regret states decreases after repeated visits, making direct regret value prediction difficult due to severe distributional imbalance and non-stationary targets.

## Method

### Overall Architecture

RGSC extends AlphaZero with three core components:
1. **Regret Definition**: Quantifies the regret of each state in a game trajectory (the discrepancy between the agent's evaluation and the actual outcome).
2. **Regret Network**: Comprising a ranking network and a value network to identify high-regret states.
3. **Prioritized Regret Buffer (PRB)**: Stores and manages high-regret states, sampling from a softmax distribution to determine self-play restart positions.

### Key Designs

**Design 1: Regret Definition**

- **Function**: Defines the regret value for each state in a game trajectory.
- **Mechanism**: The regret $R(s_t)$ is defined as the mean squared deviation between the MCTS evaluation of the selected action and the actual outcome, averaged from state $s_t$ to terminal state $s_T$: `R(st) = (1/(T-t)) Σ(V_selected(si) - z)²`.
- **Design Motivation**: Captures states where the agent's evaluation diverges most from the actual result — these are the critical positions not yet mastered by the agent, thus carrying the highest learning potential.

**Design 2: Regret Ranking Network**

- **Function**: Learns to rank states by regret value rather than directly predicting regret.
- **Mechanism**: Outputs an unnormalized ranking score $\gamma_s$, which is converted via softmax into a restart distribution $\rho(s|S)$. The optimization objective is to maximize $J_{\text{rank}} = \sum \rho(s|S) \cdot R(s)$, assigning high sampling probability to high-regret states. A surrogate loss is used in practice: $L_{\text{rank}} = -\log \sum \exp(\log \text{softmax}(\gamma_s) + R(s))$.
- **Design Motivation**: Direct regret prediction suffers from severe distributional imbalance (most states have near-zero regret) and non-stationarity (high-regret states decline in regret after correction). The ranking objective only needs to identify relatively high-regret states, substantially reducing learning difficulty.

**Design 3: Regret Value Network**

- **Function**: Estimates the regret of internal nodes within the MCTS search tree.
- **Mechanism**: Regret can be computed directly for states on the self-play trajectory, but internal tree nodes lack complete trajectory information. The value network provides regret estimates for these nodes.
- **Design Motivation**: The search tree may contain high-regret states that were explored by MCTS but never actually played, and incorporating these states yields more diverse restart positions.

**Design 4: Prioritized Regret Buffer (PRB)**

- **Function**: Maintains a fixed-capacity set of $K$ high-regret states to serve as self-play restart points.
- **Mechanism**: After each self-play game, the ranking network selects the highest-ranked state, which is added to the PRB only if its regret exceeds the minimum regret among existing entries. During sampling, a softmax distribution $P(s_i) \propto R(s_i)^{1/\tau}$ prioritizes high-regret states. Regret values in the PRB are updated via EMA: $R_{\text{new}} \leftarrow (1-\alpha) \cdot R_{\text{old}} + \alpha \cdot R$.
- **Design Motivation**: EMA updates prevent abrupt regret drops, ensuring that a state's regret decays gradually only after the agent has truly mastered it, emulating the human process of repeated review until full understanding.

### Loss & Training

- **Ranking Loss**: $L_{\text{rank}} = -\log \sum \exp(\log \text{softmax}(\gamma_s) + R(s))$ — preserves ranking order via exponential transformation, guiding the model to assign high probability to high-regret states.
- **Value Loss**: Standard MSE regression loss for predicting state regret values.
- **Self-Play Strategy**: With probability $1-\lambda$, self-play starts from an empty board; with probability $\lambda$, it restarts from a state sampled from the PRB.
- **Training Integration**: The ranking and value networks are implemented as additional output heads (regret heads) of the AlphaZero network, incurring negligible computational overhead.

## Key Experimental Results

### Main Results

**Elo improvement across three board games (300 iterations, ~150 A6000 GPU hours each)**:

| Game | AlphaZero | Go-Exploit | RGSC | RGSC vs AZ | RGSC vs GE |
|------|-----------|-----------|------|-----------|-----------|
| 9×9 Go | 1000 (ref) | +low | +76 Elo | +76 | +96 |
| 10×10 Othello | 1000 (ref) | +20 | +70 Elo | +70 | +50 |
| 11×11 Hex | 1000 (ref) | -38 | +84 Elo | +84 | +122 |

**Win rates against external strong programs**:

| Game | Opponent | AlphaZero | Go-Exploit | RGSC |
|------|------|-----------|-----------|------|
| 9×9 Go | KataGo | 45.5% | 49.5% | **53.6%** |
| 10×10 Othello | Ludii α-β | 51.7% | 52.9% | **57.8%** |
| 11×11 Hex | MoHex | 83.6% | 89.2% | **91.1%** |

### Ablation Study

**State selection quality: ranking network vs. value network**:

| Method | 9×9 Go avg regret | 10×10 Othello avg regret | Performance |
|------|-------------------|-------------------------|------|
| Go-Exploit (uniform) | Lowest | Lowest | Baseline |
| Regret Value Net | Medium | Medium | Sub-optimal |
| **Regret Ranking Net** | **Highest** | **Highest** | **Best** |

**Continued training on a pre-trained model (15-block, 9×9 Go, 40 iterations)**:

| Method | Win rate vs. KataGo |
|------|---------------|
| Baseline (before training) | 69.3% ± 2.6% |
| AlphaZero (continued) | 70.2% ± 2.7% (negligible gain) |
| Go-Exploit | 69.2% ± 2.7% (no gain) |
| **RGSC** | **78.2% ± 2.5%** (+8.9%) |

### Key Findings

1. **Go-Exploit Degrades in Later Training**: Go-Exploit is effective in early training when many states remain unmastered, but its efficiency drops sharply in later stages as uniform sampling becomes less discriminative, eventually underperforming AlphaZero.
2. **Ranking Outperforms Regression**: The ranking network consistently selects states with higher regret values, validating the advantage of ranking objectives over direct value regression under non-stationary, imbalanced distributions.
3. **Regret in PRB Decreases as Expected**: Across all games, the average regret of states upon entry into the PRB is significantly higher than upon removal (Go: 0.655 → 0.296), confirming that RGSC effectively corrects mistakes.
4. **Strong Models Can Still Improve**: RGSC improves win rate by 8.9% on a well-trained model, while AlphaZero and Go-Exploit both plateau.

## Highlights & Insights

1. **Elegant Emulation of Human Learning**: The human strategy of repeatedly reviewing errors is naturally formalized as regret-guided search control, with clear motivation and concise implementation.
2. **Clever Design of the Ranking Objective**: By circumventing the difficulty of directly predicting non-stationary targets, the ranking objective requires only relative ordering, substantially reducing learning complexity.
3. **Exploitation of Internal Search Tree Nodes**: Beyond trajectory states, RGSC leverages states explored by MCTS but never actually played, increasing the diversity of restart positions.
4. **Minimal Additional Overhead**: The regret network comprises only two extra output heads on the AlphaZero network; overhead becomes negligible as the number of residual blocks increases.
5. **Generalization Potential**: Preliminary experiments demonstrate that RGSC is applicable to MuZero (Pac-Man), suggesting broader applicability across RL settings.

## Limitations & Future Work

1. **Validated Only on Board Games**: Board games are deterministic, perfect-information environments; the effectiveness of RGSC in stochastic or imperfect-information settings requires further investigation.
2. **Limitations of the Regret Definition**: The current regret definition is based on the deviation between MCTS evaluations and outcomes; defining regret for continuous control tasks requires new formulations.
3. **Fixed PRB Capacity**: A fixed-size buffer may be insufficient to cover all critical states in more complex games.
4. **19×19 Go Not Explored**: Experiments are conducted on 9×9 Go; scalability to larger boards remains to be verified.

## Related Work & Insights

- **Go-Exploit**: The first work to systematically study search control in AlphaZero; however, its limitation of uniform sampling is addressed by RGSC's prioritized sampling.
- **KataGo**'s random opening strategy inspired the idea of starting training from non-initial states.
- **Prioritized Experience Replay**: The PRB in RGSC can be viewed as an extension of prioritized sampling in experience replay to the level of search control.
- **Inspiration**: The regret-guided paradigm can be generalized to other settings requiring focused learning on difficult samples, such as curriculum learning and active learning.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The regret ranking network design is novel, and the approach to handling non-stationary target prediction is clever; however, the overall idea is a natural extension of PER to search control.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive validation across three board games, including win-rate evaluation against strong open-source programs, ranking vs. value network ablation, and continued training experiments on pre-trained models.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clearly articulated (the human vs. machine learning comparison figure is intuitive), method derivation is complete, and experimental presentation is clear.
- **Value**: ⭐⭐⭐⭐ Provides a concise and effective approach for improving AlphaZero training efficiency, with potential for generalization to broader RL settings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Scalable Exploration for High-Dimensional Continuous Control via Value-Guided Flow](scalable_exploration_for_high-dimensional_continuous_control_via_value-guided_fl.md)
- [\[ICLR 2026\] WIMLE: Uncertainty-Aware World Models with IMLE for Sample-Efficient Continuous Control](wimle_uncertainty-aware_world_models_with_imle_for_sample-efficient_continuous_c.md)
- [\[ICLR 2026\] Towards Bridging the Gap between Large-Scale Pretraining and Efficient Finetuning for Humanoid Control](towards_bridging_the_gap_between_large-scale_pretraining_and_efficient_finetunin.md)
- [\[ACL 2026\] AttnPO: Attention-Guided Process Supervision for Efficient Reasoning](../../ACL2026/reinforcement_learning/attnpo_attention-guided_process_supervision_for_efficient_reasoning.md)
- [\[ICLR 2026\] BA-MCTS: Bayes Adaptive Monte Carlo Tree Search for Offline Model-based RL](bayes_adaptive_monte_carlo_tree_search_for_offline_model-based_reinforcement_lea.md)

</div>

<!-- RELATED:END -->
