---
title: >-
  [Paper Note] Relative Value Learning
description: >-
  [ICLR 2026][Reinforcement Learning][R-GAE] Addressing the observation that "control only cares about value differences while the absolute value scale is a redundant degree of freedom," this paper proposes Relative Value Learning (RV). The critic directly learns an antisymmetric function $\Delta_\theta(s_i,s_j)=V^\pi(s_i)-V^\pi(s_j)$ supported by a Pairwise Bell
tags:
  - ICLR 2026
  - Reinforcement Learning
  - R-GAE
  - PPO
date: 2026-05-08
content_hash: bcb42f277b11bc48
---
# Relative Value Learning

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=ulTRUwrzt9](https://openreview.net/forum?id=ulTRUwrzt9)  
**Code**: [https://github.com/Hauf3n/relative-value-learning](https://github.com/Hauf3n/relative-value-learning)  
**Area**: Reinforcement Learning / Value Functions / Policy Gradients  
**Keywords**: Relative Value, Antisymmetric Functions, Pairwise Bellman Operator, R-GAE, PPO

## TL;DR
Addressing the observation that "control only cares about value differences while the absolute value scale is a redundant degree of freedom," this paper proposes Relative Value Learning (RV). The critic directly learns an antisymmetric function $\Delta_\theta(s_i,s_j)=V^\pi(s_i)-V^\pi(s_j)$ supported by a Pairwise Bellman Operator (proven to be a $\gamma$-contraction with its unique fixed point equal to the true value difference). The method includes well-defined 1-step / n-step / λ-return targets and an unbiased advantage estimator, R-GAE, reconstructed from pairwise differences. Integrated with PPO, it performs comparably to or better than standard PPO across 49 Atari games.

## Background & Motivation
**Background**: Mainstream value-based RL (TD(λ), DQN, Rainbow, A2C/PPO, etc.) requires the critic to approximate **absolute** state values $V^\pi(s)$ or action values $Q^\pi(s,a)$, evaluating "how good a single state/action is," and then deriving value differences as needed.

**Limitations of Prior Work**: In control tasks, actions are selected through **comparison**—greedy selection uses $\max_a Q^\pi(s,a)$, and policy gradients use the advantage $A^\pi(s,a)$, both of which depend solely on value differences. Adding a constant $c$ to $V^\pi$ (accommodated by corresponding reward shaping) does not change any advantages or greedy choices. Consequently, the absolute scale is behaviorally **meaningless**, representing an unconstrained "gauge freedom."

**Key Challenge**: Absolute critics are forced to predict this behaviorally meaningless scalar. This redundant degree of freedom causes three types of issues: ① susceptibility to drift when reward shaping or baselines change; ② ambiguity or ill-posedness in scenarios with only comparisons or implicit feedback (e.g., preference RL, human-in-the-loop RL) where absolute scale is undefined; ③ a mismatch between the invariance of the function class and the invariance of the decision problem.

**Goal**: To treat value differences as the primary learning objective, ensuring the critic's function class inherently conforms to the "only differences matter" invariance, thereby eliminating gauge freedom.

**Key Insight**: Learn an antisymmetric function $\Delta_\theta:S\times S\to\mathbb{R}$ and enforce $\Delta_\theta(s_i,s_j)=-\Delta_\theta(s_j,s_i)$ (automatically implying $\Delta_\theta(s,s)=0$) to approximate $V^\pi(s_i)-V^\pi(s_j)$. In this way, gauge freedom is "constructively" removed, and advantages can be reconstructed from pairwise differences without knowing absolute values.

**Core Idea**: Replace the absolute critic with an antisymmetric pairwise value difference network and equip it with a self-consistent Bellman theory (contraction, targets, advantage reconstruction), establishing "relative value" as a first-class citizen with a clean analytical foundation.

## Method

### Overall Architecture
RV replaces the "absolute critic" in traditional actor-critic methods with a **pairwise value difference critic** $\Delta_\theta(s_i,s_j)$. The method is structured around four components: (1) Defining a **Pairwise Bellman Operator** $T_\pi$ over the space of antisymmetric functions, proving it is a $\gamma$-contraction with a unique fixed point equal to the true value differences; (2) Reformulating bootstrapping targets for 1-step / n-step / λ-returns into forms containing only observable rewards and non-terminal pairwise differences to solve ill-posedness at terminal states; (3) Reconstructing relative values along a trajectory via telescoping to derive the **R-GAE** advantage estimator, proving it is unbiased relative to the standard policy gradient; (4) Utilizing **trajectory ranking** to estimate offsets for each trajectory in a batch to suppress the additional variance of R-GAE. The critic is implemented using a shared CNN encoder and a Siamese difference head, and the loss replaces the standard GAE in PPO with R-GAE.

### Key Designs

**1. Pairwise Bellman Operator and Contraction: Providing a Theoretical Foundation**

To directly learn $\Delta^\pi(s_i,s_j)=V^\pi(s_i)-V^\pi(s_j)$, one must prove that this objective satisfies a self-consistent recursive equation and can be iteratively approximated. By subtracting the individual Bellman equations for two states, the Pairwise Bellman Identity is obtained:

$$\Delta^\pi(s_i,s_j)=r^\pi(s_i)-r^\pi(s_j)+\gamma\,\mathbb{E}_{s_i'\sim P^\pi(\cdot|s_i),\,s_j'\sim P^\pi(\cdot|s_j)}\big[\Delta^\pi(s_i',s_j')\big]$$

where two successors $s_i',s_j'$ are sampled independently. This equation depends only on observable single-step reward differences and successor pairwise differences, and remains invariant to any global shift of $V^\pi$. Defining the operator $(T_\pi\Delta)(s_i,s_j):=\Delta r^\pi(s_i,s_j)+\gamma(\hat P^\pi\Delta)(s_i,s_j)$ on the Banach space $\mathcal{F}$ of bounded antisymmetric functions, the authors prove (Theorem 3.1) that $\|T_\pi\Delta_1-T_\pi\Delta_2\|_\infty\le\gamma\|\Delta_1-\Delta_2\|_\infty$. By the Banach Fixed-Point Theorem, $T_\pi$ has a unique fixed point exactly equal to the true value difference $V^\pi(s_i)-V^\pi(s_j)$, ensuring that learning differences has convergence guarantees similar to standard value iteration.

**2. Well-defined Pairwise Value Targets: Handling Terminal States**

Directly applying the Pairwise Bellman Operator to construct bootstrapping targets fails at terminal states: when a successor is terminal (done flag $d_i=1$), the naive term $\Delta(s_{i+1},s_{j+1})=0-V(s_{j+1})=-V(s_{j+1})$ requires an **absolute value**, which is unavailable in RV's function class. This paper rearranges all bootstrapping targets to contain only observable rewards and non-terminal pairwise differences. The 1-step target is $y^{(1)}_{ij}=(r_i-r_j)+\gamma\delta_{ij}$, where the bootstrap term $\delta_{ij}$ is determined by the termination flags of two trajectories:

$$\delta_{ij}=\begin{cases}\Delta_\theta(s_{i+1},s_{j+1}),&d_i=0,d_j=0\\\Delta_\theta(s_{i+1},s_j)+r_j,&d_i=0,d_j=1\\\Delta_\theta(s_i,s_{j+1})-r_i,&d_i=1,d_j=0\\\Delta_\theta(s_i,s_j)+r_j-r_i,&d_i=1,d_j=1\end{cases}$$

When both terminate, $\delta_{ij}=0$ is used by default. The λ-return target $y^{(\lambda)}_{ij}$ is similarly constructed by exponentially weighting n-step returns and truncating at the first termination, ensuring robustness.

**3. R-GAE: Unbiased Advantage Estimation from Pairwise Differences**

To serve as a critic for PPO, RV must produce advantages. Relative values are reconstructed along a rollout $(s_0,\dots,s_T)$ via telescoping: let $\tilde V_\theta(s_0):=0$ and $\tilde V_\theta(s_t):=\sum_{k=0}^{t-1}\Delta_\theta(s_{k+1},s_k)$ (or directly calculated as $\Delta_\theta(s_t,s_0)$). If $\Delta_\theta=\Delta^\pi$, then $\tilde V_\theta(s_t)=V^\pi(s_t)-V^\pi(s_0)$, effectively anchoring the trajectory at zero. Defining relative TD residuals $\tilde\delta_t=r_t+\gamma\tilde V_\theta(s_{t+1})-\tilde V_\theta(s_t)$ and $\tilde A_t=\sum_{l=0}^{T-t}(\gamma\lambda)^l\tilde\delta_{t+l}$, a key theoretical result (Lemma 3.2) shows: $\tilde A_t=A_t+B_t$, where $A_t$ is the standard GAE and $B_t=(1-\gamma)V^\pi(s_0)\sum_{l=0}^{T-t}(\gamma\lambda)^l$ is a **trajectory constant**. Consequently (Corollary 3.3), the policy gradient remains **unchanged** when using $\tilde A_t$, because the expectation of $B_t$ times the score function is zero. This means the policy gradient remains unbiased when using RV as a critic.

**4. Trajectory Ranking Initialization: Reducing Variance in Multi-Trajectory Batches**

Unbiasedness does not imply zero cost: $B_t$ in $\tilde A_t$ is proportional to the unknown trajectory constant $V^\pi(s_0)$. If $|V^\pi(s_0)|$ is large, it inflates the magnitude of $\tilde A_t$ and increases gradient variance. When a training batch contains multiple trajectories, anchoring each to zero is suboptimal as their relative levels differ. This paper uses a data-dependent initialization: start states from the batch are used to construct an $N\times N$ pairwise difference matrix $\Delta_{ij}=\Delta_\theta(s^{(i)}_{\text{start}},s^{(j)}_{\text{start}})$. Offset estimates $O(s^{(n)}_{\text{start}})=\frac1N\sum_j\Delta_{nj}$ are computed, and a relative shift $\hat V_\theta(s^{(n)}_{\text{start}})=O(s^{(n)}_{\text{start}})-\min_\ell O(s^{(\ell)}_{\text{start}})$ is applied to each rollout. This reduces variance while keeping the estimation robust to noise.

### Loss & Training
The critic uses a shared CNN encoder $f_{\text{enc}}(s)\in\mathbb{R}^d$ and a projection head: $\Delta_\theta(s_i,s_j)=\Phi(f_{\text{enc}}(s_i)-f_{\text{enc}}(s_j))$. To strictly enforce antisymmetry, $\Phi$ is implemented as a single learnable vector $w\in\mathbb{R}^d$ **without a bias**, ensuring $\Delta_\theta(s_i,s_j)=-\Delta_\theta(s_j,s_i)$ and $\Delta_\theta(s_i,s_i)=0$ hold naturally. The total loss replaces standard GAE with R-GAE:

$$L(\theta)=-L_{\text{policy}}(\theta)+c_v L_{\text{critic}}(\theta)+c_e L_{\text{ent}}(\theta)$$

$L_{\text{critic}}$ is the MSE regression of the n-step target $y^{(n)}_{ij}$, and $L_{\text{policy}}$ uses the PPO clip objective with $\tilde A_t$.

## Key Experimental Results

### Main Results
PPO+RV was evaluated as a drop-in critic on 49 Atari games, compared against standard PPO and DAE over 40M frames with 10 seeds.

| Game (Selected) | PPO | DAE | PPO + RV (Ours) |
|--------------|-----|-----|-----------------|
| BattleZone | 17366.7 | 16302.0 | **21780.0** |
| RoadRunner | 25076.0 | 16146.3 | **43346.3** |
| Robotank | 5.5 | 6.9 | **19.5** |
| TimePilot | 4342.0 | 7252.7 | **10212.7** |
| VideoPinball | 37389.0 | 23958.6 | **138564.8** |
| Enduro | 758.3 | 0.0 | **1080.2** |
| Gravitar | 737.2 | 443.5 | **1441.0** |
| Centipede | **4386.4** | 3915.8 | 1226.1 |
| Pong | **20.7** | 20.7 | 16.8 |
| Zaxxon | 5008.7 | 5612.2 | 845.8 |

Aggregated metrics (human-normalized IQM, Median, Mean) show that PPO+RV outperforms both PPO and DAE, with a lower optimality gap, suggesting that the relative critic is an effective and superior alternative to the absolute critic.

### Ablation Study

| Setting | Approach | Results / Description |
|------|------|-------------|
| R-GAE vs $\Delta_\gamma$ variant | Using $\Delta_\gamma(s',s)=\gamma V(s')-V(s)$ to match TD residuals exactly | Performed **worse** in practice; R-GAE is preferred |
| $\delta_{ij}$ Target | Setting $\delta_{ij}=0$ for dual termination | Reduces variance at the end of episodes |
| Projection Head $\Phi$ | Single vector $w$ vs Antisymmetric MLP | Non-linear heads provided **no additional gain** |
| Trajectory Ranking | Estimating offsets so $\mathbb{E}_t[B_t]\approx0$ | Effectively reduces variance caused by the trajectory constant $B_t$ |

### Key Findings
- RV significantly outperforms PPO in games like VideoPinball and RoadRunner, though it underperforms in a few titles like Centipede and Zaxxon.
- The $\Delta_\gamma$ variant, though theoretically equivalent to GAE, was less effective than the antisymmetric R-GAE in practice.
- The method remains stable without target networks or stop-gradient operations, indicating the antisymmetric constraint provides beneficial regularization.
- Computational efficiency: 40M frames (~10M steps) takes about 65 minutes on 1 A100 GPU.

## Highlights & Insights
- **Turning "Gauge Freedom" into a methodological starting point**: While it is well-known that constant shifts in $V$ do not change behavior, this work is the first to directly eliminate this redundancy via an antisymmetric function class.
- **Elegant Proof of Pairwise Bellman Contraction**: The cancellation of reward terms makes the $\gamma$-contraction proof simple and provides strong theoretical grounding for learning differences.
- **Transferable Unbiasedness Logic**: The argument that "reconstructed advantages differ only by a constant, thus gradients are unbiased" generalizes to any relative learning setup (e.g., preference-based RL).
- **Detail-oriented Terminal Handling**: Explicitly defining the four cases for terminal state bootstrapping targets provides a robust roadmap for implementation.

## Limitations & Future Work
- **Limited Scope**: RV has only been validated on Atari with discrete actions and on-policy PPO; its efficacy in continuous control or off-policy settings remains unverified.
- **Overhead and Variance**: Constructing state pairs and trajectory ranking adds computational cost ($N \times N$ matrix operations), and variance control still depends on specific heuristics.
- **Performance Dips**: The cause of significant performance drops in specific games (e.g., Centipede, Pong) requires further investigation.
- **Missing Preference RL Validation**: Despite being highlighted in the motivation, the method was not tested in preference-based or human-in-the-loop RL scenarios where relative values are most applicable.

## Related Work & Insights
- **vs. Absolute Critics (DQN/Rainbow)**: These predict scalars in an arbitrary scale space. RV removes the shift freedom at the **model level**, aligning the function class with decision-making invariance.
- **vs. Direct Advantage Estimation (DAE)**: DAE learns $A^\pi(s,a)$ directly to bypass value learning. RV learns $\Delta(s_i,s_j)$ with an explicit Bellman operator and generally outperformed DAE in the experiments.
- **vs. Dueling Architecture**: Dueling decomposes $Q=V+A$ but stays in the absolute space. RV uses a Siamese head to enforce antisymmetry and zero self-difference.
- **vs. Inverse RL/Preference RL**: While pairwise objectives appear in those fields, RV formalizes them into a Bellman-consistent value critic.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Relative Entropy Pathwise Policy Optimization](relative_entropy_pathwise_policy_optimization.md)
- [\[ICLR 2026\] Value Flows](value_flows.md)
- [\[ICLR 2026\] Reinforcement Learning via Value Gradient Flow](reinforcement_learning_via_value_gradient_flow.md)
- [\[ICLR 2026\] Inter-Agent Relative Representations for Multi-Agent Option Discovery](inter-agent_relative_representations_for_multi-agent_option_discovery.md)
- [\[ICLR 2026\] Transitive RL: Value Learning via Divide and Conquer](transitive_rl_value_learning_via_divide_and_conquer.md)

</div>

<!-- RELATED:END -->
