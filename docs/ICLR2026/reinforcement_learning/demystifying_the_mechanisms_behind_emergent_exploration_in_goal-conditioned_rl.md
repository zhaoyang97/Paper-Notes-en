---
title: >-
  [Paper Note] Demystifying The Mechanisms Behind Emergent Exploration in Goal-Conditioned RL
description: >-
  [ICLR2026][Reinforcement Learning][Goal-Conditioned RL] This paper uses a cognitive science-inspired "Rational Analysis + Intervention + Minimal Modeling" triplet to deconstruct why reward-free Single-Goal Contrastive RL (SGCRL) exhibits spontaneous exploration. The conclusion is that the actor maximizes an implicit reward shaped by the critic's representation (state-goal $\psi$-similarity), and this exploration-exploitation dynamic emerges from **low-rank representations** l…
tags:
  - "ICLR2026"
  - "Reinforcement Learning"
  - "Goal-Conditioned RL"
  - "Emergent Exploration"
  - "Contrastive Learning"
  - "Implicit Reward"
  - "Low-rank Representation"
date: 2026-05-08
content_hash: f61d52f40e91fdd0
---

# Demystifying The Mechanisms Behind Emergent Exploration in Goal-Conditioned RL

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=mwgYORsqtv](https://openreview.net/forum?id=mwgYORsqtv)  
**Code**: Project Page https://mahsa-bastankhah.github.io/demystifying-single-goal-exploration/  
**Area**: Reinforcement Learning  
**Keywords**: Goal-Conditioned RL, Emergent Exploration, Contrastive Learning, Implicit Reward, Low-rank Representation

## TL;DR
This paper uses a cognitive science-inspired "Rational Analysis + Intervention + Minimal Modeling" triplet to deconstruct why reward-free Single-Goal Contrastive RL (SGCRL) exhibits spontaneous exploration. The conclusion is that the actor maximizes an implicit reward shaped by the critic's representation (state-goal $\psi$-similarity), and this exploration-exploitation dynamic emerges from **low-rank representations** learned via contrastive learning, rather than neural network function approximation.

## Background & Motivation
**Background**: Recent deep RL has seen various "emergent behaviors" where agents learn complex skills without explicit rewards or human-designed curricula. A representative case is SGCRL (Liu et al., 2025): by focusing on **a single fixed difficult goal** and training the critic with temporal contrastive learning, it learns manipulation and locomotion skills in long-horizon tasks, even surpassing methods based on "goal distributions" or "subgoal curricula."

**Limitations of Prior Work**: While this phenomenon has been observed, **why** it happens remains unclear. Common folklore attributes emergence to "scale," but this paper argues that this rule of thumb is insufficient here—much like word analogies are due to loss functions rather than architectures. Is SGCRL's exploration an inherent property of the algorithm or a byproduct of neural network generalization? Without understanding the driving factors, it is impossible to predict when, how, or why exploration appears, hindering the safe and reliable use of such systems.

**Key Challenge**: SGCRL uses **no external rewards** during training yet exhibits a clear exploration-exploitation switch. How does a reward-free algorithm know where to go and when to stop for exploitation? This "where does the reward come from" question is the core of this paper.

**Goal**: Answer a specific question—why does SGCRL explore efficiently without any obvious intrinsic/extrinsic rewards?—and decompose the answer into experimentally falsifiable predictions.

**Key Insight**: Instead of the standard ML route of "tuning performance on benchmarks," the authors borrow tools from cognitive science for studying intelligent behavior: **Rational Analysis** (what objective the agent optimizes), **Interventions** (how behavior changes when a variable is modified), and **Minimal Modeling** (reproducing behavior with the smallest possible model).

**Core Idea**: Reinterpret the SGCRL actor's objective as "maximizing an implicit reward," which is the $\psi$-similarity learned by the critic; then prove this mechanism persists in a **tabular** setting without neural networks, thereby attributing the emergent exploration to low-rank contrastive representations themselves.

## Method

### Overall Architecture
This is an **analytical paper**, where the "Method" is a chain of tools for understanding mechanisms rather than a new algorithm. The subject, SGCRL, is an actor-critic framework: the critic estimates the likelihood of $(s,a)$ leading to a future state $s_f$ via $\phi(s,a)^\top\psi(s_f)$, trained using a backward InfoNCE loss (positive samples are future states sampled $\Delta\sim\text{Geom}(1-\gamma)$ steps ahead, negative samples from the marginal distribution); all representations are $\ell_2$ normalized. After training, the critic is equivalent to a log-Q: $\phi(s,a)^\top\psi(s_f)=\log p^\pi_\gamma(s_f\mid s,a)-\log p(s_f)$. The actor selects actions to maximize the likelihood of reaching the goal $\phi(s,a)^\top\psi(g)+\tau H(\pi)$, which for discrete actions is a softmax over $\frac{1}{\tau}\phi(s,a)^\top\psi(g)$.

The analysis proceeds in three layers: first, **4.1** proves the actor maximizes an implicit reward ($\psi$-similarity); then, **4.2** analyzes how this implicit reward evolves with representation updates—suppressing explored regions before finding the goal and reinforcing successful paths after; finally, a **tabular SGCRL** model (removing neural networks) verifies that the dynamics stem from low-rank contrastive representations rather than network approximation. The logic revolves around the interaction between the actor and critic: the actor rushes toward high-similarity regions, while the critic suppresses similarity in those regions when the goal is not found.

### Key Designs

**1. Rational Analysis: Reinterpreting the reward-free actor objective as maximizing implicit rewards**

This step answers "where the reward comes from." Based on an alignment property of InfoNCE (Assumption 1: at convergence, state-action representations align with the expected future state representations, $\phi(s,a)=\mathbb{E}_{s_f\sim p^\pi_\gamma}[\psi(s_f)]$), the authors derive Theorem 1: although SGCRL nominally only maximizes the "probability of reaching goal $g$," it is **equivalent** to maximizing the following return:

$$Q_\psi(s,a):=\mathbb{E}_{p^\pi_t}\!\left[\sum_{t=0}^{\infty}\gamma^t\,\psi(s_t)^\top\psi(g)\;\Big|\;s_0=s,a_0=a\right].$$

Thus, the stepwise reward is the current state-goal representation similarity ($\psi$-similarity $:=\psi(s_f)^\top\psi(g)$). This can be viewed as the agent's current belief about "where the goal is," corrected by the critic as data increases, similar to maintaining a posterior over rewards/transitions in posterior sampling. This explains why there is directionality without external rewards—rewards are endogenous to the representation. Authors also note this resembles successor representations/features but differs fundamentally: the latter requires explicit training of features, whereas SGCRL's features and their successor representations emerge naturally from the InfoNCE objective.

**2. Before finding the goal: InfoNCE pushes explored states orthogonal to the goal, automatically pruning the search space**

Implicit rewards alone are insufficient; one must explain why exploration doesn't loop. Theorem 2 (informal) analyzes a simplified setting: $\psi(g)$ is fixed, and the agent acts in regions without the goal. Assuming initial representations have a shared component parallel to $\psi(g)$ plus noise, InfoNCE updates with a large batch and small learning rate will **converge with high probability** to $\phi(s_i,a_i)^\top\psi(g)=\psi(s_{f,i})^\top\psi(g)=0$. Intuitively, since InfoNCE is invariant to adding a common vector to all representations, the shared component parallel to $\psi(g)$ learns no temporal difference, and because normalized representation capacity is finite, the system **suppresses this redundant component**. Consequently, the similarity of frequently visited states that fail to reach the goal is continuously lowered, and the actor naturally stops returning to them, acting as an **automatically generated subgoal curriculum** that prunes the search space. This proof relies solely on InfoNCE fixed-point analysis and **does not depend on neural network approximation**.

**3. After finding the goal: Successful trajectories leave high-similarity "traces," triggering the switch to exploitation**

The other half of the mechanism is exploitation. Since contrastive learning aligns representations of positive samples (Assumption 1), once the goal $g$ is reached, it appears as a positive sample for states on the successful trajectory. Thus, representations on this path **align toward $\psi(g)$**, leaving a high $\psi$-similarity "trace." While full alignment isn't theoretically guaranteed, this trace appears stably in experiments and guides the agent back to the goal, marking the transition from exploration to exploitation. Unlike traditional intrinsic rewards (novelty, counts, etc.), SGCRL's reward is not manually added with hyperparameter balancing but **emerges directly from the actor's objective**.

**4. Minimal Modeling: Tabular SGCRL isolates "low-rank representations" as the true cause**

The final step is causal attribution. Since Theorem 2 does not require networks, the authors create a **tabular SGCRL**: each state $s$ has an embedding $\psi(s)$ in a lookup table updated via InfoNCE gradients; assuming deterministic transitions with ground-truth dynamics $s_{t+1}=p(s_t,a_t)$, and the policy uses softmax. This minimal model exhibits the same two-phase dynamics (negative correlation before finding the goal, positive after). More importantly, a **control ablation** shows that replacing vectorized representations with an $|S|\times|S|$ scalar similarity lookup table (updated with the same objective) causes exploration to fail, requiring $\sim$100x more samples. This proves that exploration is driven by the geometric constraints of **low-rank (low-dimensional normalized) representations** learned via contrastive learning, not the contrastive objective itself or network generalization.

## Key Experimental Results

### Main Results (RQ1: Clear two-phase representation evolution)

| Setting | Phenomenon | Description |
|------|------|------|
| Tabular SGCRL (Tower of Hanoi) | Pearson $r$ between state visits and goal similarity: **negative** before finding the goal, **positive** after | Validates the two-phase dynamics predicted by Theorem 2 without NNs |
| Standard SGCRL (Continuous navigation) | $\psi$-similarity of states on frequently traversed paths **systematically decreases** during training | Reproduces "explored regions pushed away from goal" in continuous settings |
| PCA projection with imaginary goal | Representations initially cluster near the goal, then drift toward an "equator" orthogonal to $\psi(g)$, while preserving local structure | Visualizes the correspondence between orthognalization and visitation order |

### Ablation Study / Intervention Experiments

| Configuration | Key Result | Description |
|------|---------|------|
| Vectorized low-rank (Full) | Efficient exploration | Complete mechanism |
| $|S|\times|S|$ scalar lookup table | Fails exploration, requires **~100x** samples | Proves low-rank representation is key, not the objective alone |
| Single-goal vs. Uniform multi-goal data | Uniform sampling **fails** to push visited states away from the goal | Actor actively shapes representations through data collection |
| Intervention: Patch representation to $\psi(g)$ | Agent is strongly attracted to the patch, increasing visits to that room | Behavior is driven by the $\psi$-similarity implicit reward |
| Intervention: Set room representation to $-\psi(g)$ | Agent **systematically avoids** the region, finding alternative paths | Representation design enables safety-aware exploration |

### Key Findings
- **Low-rank representation is the cause, not the effect**: The failure of scalar lookup tables proves that geometric constraints from "capacity-limited low-dimensional normalized embeddings" underly the exploration dynamics. While low-rankness is often viewed as a defect in SSL (rank collapse), it is a necessary condition for SGCRL's success.
- **Actor as both consumer and producer**: The single-goal data collection strategy actively pushes visited states away from the goal, which uniform sampling cannot achieve. This explains why single-goal setups can outperform subgoal curricula.
- **Intervenable and applicable to safety**: Directly editing representations (setting to $\psi(g)$ or $-\psi(g)$) allows for directing or repelling the agent, suggesting the possibility of "representation design" over "reward engineering."
- **Isomorphism to classic exploration algorithms**: The evolution of $\psi$-similarity resembles R-MAX—initially assuming all states yield maximum rewards (high initial similarity) and correcting visits; this provides a path to bringing the theoretical guarantees of R-MAX/PSRL to high-dimensional long-horizon tasks.

## Highlights & Insights
- **Translating "No Reward" to "Implicit Reward"**: Theorem 1 reinterprets the actor's behavior as maximizing $\psi$-similarity. This makes a self-supervised algorithm's behavior interpretable, predictable, and intervenable.
- **Causal isolation with tabular models**: Demonstrating the mechanism persists without NNs, but fails with scalar tables, cleanly attributes the success to low-rank representations. This is an elegant experimental design for RL attribution.
- **Cognitive science methodology in ML**: The Rational Analysis + Intervention + Minimal Modeling combination provides a paradigm for understanding emergent behavior without relying solely on benchmarking.
- **Inversion of the "low-rank" defect**: Re-framing representation capacity limits as a necessary condition for exploration provides a fresh perspective for representation learning.

## Limitations & Future Work
- **Reliance on idealized assumptions**: The core theory relies on the InfoNCE alignment assumption and fixed $\psi(g)$; continuous settings or shared encoders only provide approximations. Post-goal alignment along successful trajectories lacks rigorous theoretical proof beyond empirical evidence.
- **Small task scale**: Experiments are limited to point-maze navigation and Tower of Hanoi. Whether these mechanisms explain emergent exploration in high-dimensional pixels or real-world robotics remains to be verified.
- **Preliminary safety applications**: Representation intervention for safety-aware exploration is a proof-of-concept and far from a practical safety constraint method.
- **Future Directions**: Systematizing this analytical framework to explain other RL algorithms and exploring how to port the efficiency of R-MAX/PSRL to high-dimensional tasks using these insights.

## Related Work & Insights
- **vs. Transparency/Post-hoc interpretability**: Instead of adding auxiliary tasks or explaining NNs as black boxes, this paper explains behavior from the **algorithm's objective itself**.
- **vs. Successor Representation (SR)**: While $\psi(s)$ resembles state features and $\phi(s,a)$ resembles successor feature predictions, SGCRL's features emerge from InfoNCE and focus on how the **representations themselves drive exploration**, rather than just reward transfer.
- **vs. Traditional Intrinsic Rewards**: Unlike heuristic designs that require hyperparameter tuning, SGCRL's intrinsic reward emerges from the actor objective and has a "maximizing current belief of goal probability" interpretation.
- **vs. Provably Efficient Exploration (R-MAX / PSRL)**: Shares the "optimism under uncertainty" dynamic, linking classic theory with high-dimensional long-horizon tasks.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Interpreting the actor as maximizing implicit representation rewards and attributing success to low-rankness is a rare and solid mechanistic explanation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Predictions are backed by controlled/intervention experiments, though task scales are small and safety applications are preliminary.
- Writing Quality: ⭐⭐⭐⭐⭐ Excellent theory-to-experiment mapping and clear narrative using cognitive science methodology.
- Value: ⭐⭐⭐⭐⭐ Explains SGCRL's counter-intuitive success and provides a transferable paradigm for emergent behavior analysis.

## Related Papers

- [\[AAAI 2026\] First-Order Representation Languages for Goal-Conditioned RL](../../AAAI2026/reinforcement_learning/first-order_representation_languages_for_goal-conditioned_rl.md)
- [\[ICLR 2026\] Multistep Quasimetric Learning for Scalable Goal-Conditioned Reinforcement Learning](scaling_goal-conditioned_reinforcement_learning_with_multistep_quasimetric_dista.md)
- [\[CVPR 2026\] MangoBench: A Benchmark for Multi-Agent Goal-Conditioned Offline Reinforcement Learning](../../CVPR2026/reinforcement_learning/mangobench_a_benchmark_for_multi-agent_goal-conditioned_offline_reinforcement_le.md)
- [\[ICML 2026\] Direction-Conditioned Policies via Compositional Subgoal Scoring for Online Goal-Conditioned Reinforcement Learning](../../ICML2026/reinforcement_learning/direction-conditioned_policies_via_compositional_subgoal_scoring_for_online_goal.md)
- [\[ICLR 2026\] Occupancy Reward Shaping: Improving Credit Assignment for Offline Goal-Conditioned Reinforcement Learning](occupancy_reward_shaping_improving_credit_assignment_for_offline_goal-conditione.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] First-Order Representation Languages for Goal-Conditioned RL](../../AAAI2026/reinforcement_learning/first-order_representation_languages_for_goal-conditioned_rl.md)
- [\[ICLR 2026\] Multistep Quasimetric Learning for Scalable Goal-Conditioned Reinforcement Learning](scaling_goal-conditioned_reinforcement_learning_with_multistep_quasimetric_dista.md)
- [\[CVPR 2026\] MangoBench: A Benchmark for Multi-Agent Goal-Conditioned Offline Reinforcement Learning](../../CVPR2026/reinforcement_learning/mangobench_a_benchmark_for_multi-agent_goal-conditioned_offline_reinforcement_le.md)
- [\[ICML 2026\] Direction-Conditioned Policies via Compositional Subgoal Scoring for Online Goal-Conditioned Reinforcement Learning](../../ICML2026/reinforcement_learning/direction-conditioned_policies_via_compositional_subgoal_scoring_for_online_goal.md)
- [\[ICLR 2026\] Dual Goal Representations](dual_goal_representations.md)

</div>

<!-- RELATED:END -->
