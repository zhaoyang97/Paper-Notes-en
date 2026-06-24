---
title: >-
  [Paper Note] Local Reinforcement Learning with Action-Conditioned Root Mean Squared Q-Functions
description: >-
  [ICLR 2026][Reinforcement Learning][Backpropagation-free learning] Inspired by the "goodness function" of the Forward-Forward algorithm, this paper proposes **ARQ (Action-conditioned Root mean squared Q-functions)**—reading the scalar Q-value directly as the "root mean square after subtracting the mean (i.e., standard deviation)" of the hidden vector output from each cell in local RL. By conditioning the model on the action via one-hot concatenation at the input…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "Backpropagation-free learning"
  - "Forward-Forward"
  - "Local RL"
  - "Q-learning"
  - "Action-conditioning"
  - "Temporal Difference"
date: 2026-05-08
content_hash: 8e8decb8c76bf26c
---

# Local Reinforcement Learning with Action-Conditioned Root Mean Squared Q-Functions

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=pi4tbBMLsM](https://openreview.net/forum?id=pi4tbBMLsM)  
**Code**: TBD  
**Area**: Reinforcement Learning / Biologically Plausible Learning  
**Keywords**: Backpropagation-free learning, Forward-Forward, Local RL, Q-learning, Action-conditioning, Temporal Difference

## TL;DR
Inspired by the "goodness function" of the Forward-Forward algorithm, this paper proposes **ARQ (Action-conditioned Root mean squared Q-functions)**—reading the scalar Q-value directly as the "root mean square after subtracting the mean (i.e., standard deviation)" of the hidden vector output from each cell in local RL. By conditioning the model on the action via one-hot concatenation at the input, it removes the constraint in previous BP-free methods where the "output dimension must equal the number of actions." On MinAtar and DeepMind Control, it outperforms the SOTA local RL method AD and beats BP-trained DQN/SAC on most tasks.

## Background & Motivation
- **Background**: While backpropagation (BP) is the cornerstone of deep learning, its biological plausibility is heavily questioned (requiring synchronous computation and weight symmetry). Hinton's Forward-Forward (FF) replaces the forward+backward passes with two forward passes, greedily maximizing the "goodness" $G_z=\sum_i z_i^2$ of positive samples and minimizing it for negative samples. It is a lightweight, biologically plausible BP-free paradigm.
- **Limitations of Prior Work**: Most BP-free research focuses on finding gradient alternatives in **supervised learning**, ignoring RL as a more "natural" source of learning signals—the brain itself is a reward-driven evolving system suspected of implementing TD learning. The few works introducing local learning to RL (e.g., Artificial Dopamine, AD) use dot products of two sets of mappings to produce value estimates. However, the **dot product output dimension is strictly bound to the action space size $n_a$**, severely limiting the expressivity of each cell.
- **Key Challenge**: FF's goodness measures "compatibility between input and label," while RL's value measures "desirability of a state-action pair"—both are essentially metrics of the "desirability" of the current input. Can this analogy be used to read value from **arbitrarily dimensioned** hidden vectors using a goodness function, thereby liberating the capacity of local RL networks?
- **Goal**: Design a plug-and-play local value estimation mechanism that can replace standard Q-learning formulas while remaining BP-free and unrestricted by action space dimensions.
- **Core Idea**: **[Value as Goodness]** Use the standard deviation (RMS after subtracting the mean) of hidden vectors as the scalar Q-value, combined with **[Action-in-Input]** concatenating candidate actions to the input so the network produces a specific representation for each state-action pair.

## Method

### Overall Architecture
ARQ follows the multi-cell stacked structure of AD, where each cell simultaneously receives inputs from the layer below, the layer above (via temporal skip connections), original observations, and **candidate actions**. Through an attention-like mechanism, a hidden vector $y$ is obtained, and a goodness function (RMS after subtracting the mean) is applied to $y$ to yield a scalar $Q(s,a)$. Gradients are only propagated within cells, maintaining a BP-free structure. Two key modifications are made: changing the output readout from "dot product projection to $n_a$ dimensions" to "RMS of an arbitrary-dimensional hidden vector" and moving actions from "output-side indexing" to "input-side conditioning."

```mermaid
flowchart LR
    S[State s_t] --> X[Concatenated Input X]
    HB[Lower Layer h_t^{l-1}] --> X
    HT[Upper Layer h_{t-1}^{l+1}] --> X
    A[Action Candidate a_t<br/>one-hot/binary] --> X
    X --> ATT[Attention-like<br/>tanh(XᵀWatt2ᵀWatt1X)·h]
    ATT --> Y[Hidden Vector y<br/>Any Dimension d]
    Y --> RMS[Goodness: RMS after mean-subtraction]
    RMS --> Q[Scalar Q(s,a)]
```

### Key Designs
**1. RMS Goodness Function: Reading Value as the Standard Deviation of Hidden Vectors.** The essence of ARQ is that instead of using linear projections to dot-product hidden states into $n_a$ scalars like AD, it directly performs a statistical readout of the produced hidden vector $y$: first calculating the mean $\mu_y=\mathbb{E}_{y_i\in y}\,y_i$, then calculating the RMS after subtracting the mean $Q_\theta(s,a)=\sqrt{\mathbb{E}_{y_i\in y}(y_i-\mu_y)^2}$, which is exactly the standard deviation of $y$. Compared to the original squared sum $\sum_i z_i^2$ in FF, subtracting the mean before taking the RMS **prevents goodness from exploding as the number of units increases**—allowing the hidden dimension $d$ to scale freely without affecting numerical magnitude. This readout process is **parameter-free**, meaning intermediate vectors from any architecture can be interpreted as value estimates. Local RL networks thus gain the freedom of "arbitrarily wide outputs." The training objective still uses the DQN Bellman MSE $\mathcal{L}_\theta=\big(R_t+\gamma\max_{a'}Q_\theta(S_{t+1},a')-Q_\theta(S_t,A_t)\big)^2$, requiring no changes to the existing Q-learning pipeline.

**2. Input Action-Conditioning: State-Action Specific Representations.** Since the goodness function naturally outputs a single scalar, feeding the action into the input is the most natural choice. ARQ concatenates the candidate action $a_t$ within the input $X=\mathrm{concat}(s_t,h_t^{l-1},h_{t-1}^{l+1},a_t)$ (using one-hot for discrete tasks and bang-bang binary discretization for continuous tasks). The network produces a specific scalar for the "state + this action." This contrasts with DQN/AD, which "only take state and output $n_a$ dimensions for action indexing." The authors use PCA visualization to demonstrate that **without conditioning, hidden activations cluster almost entirely by action identity and have no correlation with Q-values**, with action-related variance dominating the representation space. **With input conditioning, representations become dominated by state and show a moderate positive correlation with Q-values**, allowing the model to allocate capacity to structures truly relevant to value rather than implicitly inferring action identity.

**3. Implementing on AD: Unbinding Dimensions to Release Attention Capacity.** To fairly benchmark against the SOTA, ARQ is implemented directly on the AD architecture. A single cell calculates $h_t^l=\mathrm{ReLU}(W_h X)$, then an attention-like $y_t^l=\tanh(X^\top W_{att2}^\top W_{att1}X)\,h_t^l$, and finally the RMS goodness of $y_t^l$. Here $Z_1=W_{att1}X$, $Z_2=W_{att2}X$, and $h_t^l$ play the roles of query, key, and value in self-attention. $Z_2^\top Z_1$ generates an interaction map **across feature dimensions** (rather than across tokens) to redistribute information. The key difference is that AD forces the width of $Z_2$ to be $n_a$, causing this interaction map to be throttled by the number of actions; ARQ allows the dimensions $d$ of $Z_2$ and $y$ to be selected freely. The authors infer that this "arbitrary hidden dimension" allows ARQ to fully exploit the non-linearity within each cell, which, combined with the state-action specific representations from action-conditioning, truly maximizes the capacity of the local RL attention mechanism.

## Key Experimental Results

### Main Results
Performance on MinAtar (discrete control) and DeepMind Control (continuous control), shown as mean ± 95% CI over 5 random seeds.

| MinAtar | Freeway | Breakout | SpaceInvaders | Seaquest | Asterix |
|---|---|---|---|---|---|
| DQN (w/ BP) | 55.86 | 27.09 | 188.03 | 37.96 | 13.60 |
| AD (w/o BP) | 57.12 | 63.76 | 363.49 | 27.83 | 22.01 |
| **ARQ (Ours)** | **60.74** | **87.84** | **544.99** | **96.45** | **35.32** |

| DMC | Walker Walk | Walker Run | Hopper Hop | Cheetah Run | Reacher Hard |
|---|---|---|---|---|---|
| TD-MPC2 (w/ BP) | 958.80 | 834.07 | 348.55 | 808.46 | 934.84 |
| SAC (w/ BP) | 980.43 | 895.02 | 319.46 | 917.40 | 980.01 |
| AD (w/o BP) | 975.30 | 762.51 | 470.95 | 831.57 | 955.93 |
| **ARQ (Ours)** | 976.33 | 771.15 | **516.23** | 880.61 | 973.66 |

ARQ consistently outperforms AD on all five MinAtar games and unexpectedly **outperforms DQN on all games**. On DMC, ARQ is overall superior to AD and even surpasses BP-based SAC/TD-MPC2 on tasks like Hopper Hop.

### Ablation Study
Comparison of goodness nonlinearity choices (MinAtar, mean ± 95% CI). RMS outperforms alternatives like Mean, Mean Square, or Variance.

| Nonlinear Function | Breakout | SpaceInvaders |
|---|---|---|
| **ARQ (RMS, Default)** | **87.84** | **544.99** |
| Mean | 79.84 | 500.13 |
| MS (Mean Square) | 82.10 | 434.88 |
| Var (Variance) | 81.34 | 416.46 |
| AD | 67.40 | 369.96 |

### Key Findings
- **Action-conditioning is nearly decisive for ARQ**: On Breakout, adding input-side conditioning increased average return from ~55 to ~85 (+50%), whereas the same change yielded only slight improvements for AD—indicating that only the RMS + action-conditioning combination makes ARQ effective.
- **Game Mechanism Analysis**: ARQ significantly outperforms DQN on Breakout/SpaceInvaders, which the authors attribute to the temporal top-down connections in AD providing the temporal coherence needed for "combos." On Seaquest, where the strategy is bimodal (attacking vs. oxygenating), AD lags behind DQN while ARQ surpasses it, showing that action-conditioning better captures multi-modal policy structures.
- **Inferred Stability Source**: Local TD updates + shorter gradient paths + variance reduction from layer-wise averaging. These factors combined allow ARQ to learn more stably and quickly than full BP networks.

## Highlights & Insights
- **Parameter-free readout unlocks dimensions**: Using "Value = Standard Deviation of Hidden Vector" as RMS goodness is parameter-free and plug-and-play, directly dismantling the hard constraint in AD where "output dimension = number of actions." The idea is minimalist yet sharp.
- **Mapping FF to RL**: The analogy between goodness (compatibility) and value (desirability) is not just rhetorical but engineered into a specific readout function that achieves SOTA, opening a door for biologically plausible learning in RL.
- **BP-free beating BP**: Beating BP methods like DQN/SAC on most tasks in low-dimensional benchmarks is a rare empirical proof that "biologically plausible no longer equals performance compromise."
- **Convincing PCA explanation**: The visualization of "activations clustering by action vs. by state" makes the reason for input-side action-conditioning intuitive and credible, rather than just relying on metric gains.

## Limitations & Future Work
- **Limited to low-dimensional benchmarks**: Experiments were restricted to MinAtar (10×10 grids) and DMC low-dimensional observations; the viability of local methods on high-dimensional raw pixels or large action spaces remains unverified.
- **Continuous control relies on discretization**: Continuous actions are handled via bang-bang binary discretization, essentially avoiding true continuous action-conditioning, which raises scalability concerns.
- **Contrastive training not yet utilized**: The authors explicitly leave "sampling positive/negative samples from the replay buffer for training in the original FF contrastive style" as future work. Currently, they still use the DQN MSE objective, meaning the contrastive essence of FF is not fully exploited.
- **Theoretical explanations remain speculative**: Key assertions such as "arbitrary dimensions release non-linear capacity" and "layer-wise averaging reduces variance" are mostly framed as conjectures/hypotheses without rigorous analysis.

## Related Work & Insights
- **Forward-Forward (Hinton, 2022)**: The direct source for the goodness function and the BP-free paradigm; ARQ's core innovation is "migrating FF's goodness from supervision to RL value estimation."
- **Artificial Dopamine (Guan et al., 2024)**: The most direct baseline and implementation base. ARQ modifies its attention-like cells, specifically addressing the bottleneck of restricted output dimensions.
- **DQN / TD Learning (Mnih et al., 2013; Sutton, 1988)**: The foundation for training objectives and value estimation paradigms. ARQ reuses Bellman MSE, making it plug-and-play.
- **Bang-bang Discretization (Seyde et al., 2021)**: Provides a feasible discretization bridge for action-conditioning in continuous control.
- **Insight**: When a module's "output readout method" becomes a bottleneck for expressivity, instead of adding parameters, it is sometimes better to switch to a parameter-free statistic with free dimensions—scale-invariant statistics like RMS/standard deviation are tools worth reusing in more plug-and-play scenarios.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Ingenious analogy between FF goodness and RL value implemented as parameter-free RMS readout with input action-conditioning; however, it is a refined modification of AD rather than a brand-new framework.
- **Experimental Thoroughness**: ⭐⭐⭐ — Solid results on two benchmarks with 5 seeds, plus ablations for action-conditioning/non-linearity and PCA analysis, but lacks verification on high-dimensional or large-scale environments.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivational chain (FF→RL→dimension bottleneck), well-aligned pseudo-code and diagrams, and honest labeling of key assertions as conjectures.
- **Value**: ⭐⭐⭐⭐ — Provides empirical evidence that biologically plausible BP-free learning can exceed BP baselines in RL; the method is plug-and-play and likely to be adopted by subsequent local RL research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Mean Flow Policy with Instantaneous Velocity Constraint for One-step Action Generation](mean_flow_policy_with_instantaneous_velocity_constraint_for_one-step_action_gene.md)
- [\[ICLR 2026\] Multistep Quasimetric Learning for Scalable Goal-Conditioned Reinforcement Learning](scaling_goal-conditioned_reinforcement_learning_with_multistep_quasimetric_dista.md)
- [\[ICLR 2026\] XQC: Well-Conditioned Optimization Accelerates Deep Reinforcement Learning](xqc_well-conditioned_optimization_accelerates_deep_reinforcement_learning.md)
- [\[ICLR 2026\] Geometric-Mean Policy Optimization](geometric-mean_policy_optimization.md)
- [\[ICLR 2026\] Occupancy Reward Shaping: Improving Credit Assignment for Offline Goal-Conditioned Reinforcement Learning](occupancy_reward_shaping_improving_credit_assignment_for_offline_goal-conditione.md)

</div>

<!-- RELATED:END -->
