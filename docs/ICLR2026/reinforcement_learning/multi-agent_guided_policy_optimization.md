---
title: >-
  [Paper Note] Multi-Agent Guided Policy Optimization
description: >-
  [ICLR 2026][Reinforcement Learning][CTDE] MAGPO utilizes an **autoregressive joint "guider" policy** for centralized coordinated exploration and constrains it via KL alignment to within the reach of decentralized "learner" policies. This preserves CTDE deployability while providing theoretical guarantees for monotonic policy improvement.
tags:
  - ICLR 2026
  - Reinforcement Learning
  - CTDE
date: 2026-05-08
content_hash: b4fe4349371268fb
---
# Multi-Agent Guided Policy Optimization

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=OT8beoc0W0](https://openreview.net/forum?id=OT8beoc0W0)  
**Code**: Implemented based on the JAX MARL library Mava (independent repository not public in the paper)  
**Area**: reinforcement_learning (Cooperative Multi-Agent Reinforcement Learning / MARL)  
**Keywords**: CTDE, Multi-Agent Reinforcement Learning, Autoregressive Joint Policy, Teacher-Student Distillation, Monotonic Policy Improvement, Policy Mirror Descent  

## TL;DR
MAGPO utilizes an **autoregressive joint "guider" policy** for centralized coordinated exploration and constrains it via KL alignment to within the reach of decentralized "learner" policies. This preserves CTDE deployability while providing theoretical guarantees for monotonic policy improvement.

## Background & Motivation

**Background**: In cooperative MARL, "Centralized Training, Decentralized Execution" (CTDE) is the dominant paradigm due to partial observability and communication constraints: global information is used during training, while each agent makes independent decisions based on local observations during execution. Traditional CTDE (QMIX, MAPPO, etc.) only leverages global information through a centralized value function, termed *vanilla CTDE* by the authors—failing to fully exploit the potential of centralized training.

**Limitations of Prior Work**: Recently proposed CTDS (Centralized Teacher with Decentralized Student) tries to utilize centralized coordination more aggressively by training a centralized teacher that observes the global state and outputs joint actions, then distilling it into decentralized students. However, CTDS suffers from two structural flaws:

- **Poor Scalability**: The teacher learns in the joint action space, which expands exponentially with the number of agents.
- **Imitation Gap**: The teacher is conditioned on the global state and joint context, while students rely on local observations—the teacher's policy might simply not exist within the decentralized policy space, leading to inevitable performance loss during distillation.

**Key Challenge**: The paper illustrates this conflict with a toy example: "Three agents each report an integer, and their sum must equal 10" (Figure 1). Each paradigm has a critical weakness:

- **Vanilla CTDE**: Three agents share a goal but decide independently. They might all adjust from 3 to 4 simultaneously (totaling 12 and failing) because they lack a coordination signal for "who should adjust," relying on random trial-and-error to find the right combination.
- **CTCE**: Letting agents decide sequentially where the next agent sees the previous actions makes coordination easy and stable, but it requires centralized execution, which is often undeployable in reality.
- **CTDS**: If the teacher learns a **stochastic and non-decomposable** coordination strategy (e.g., the first agent randomly picks 3 or 4, and the third agent picks $7-x$), forcing this into independent decentralized policies fails (e.g., resulting in combinations like [4,3,4]), dropping the success rate to 50%.

**Coordination patterns encoded in the joint policy are lost when compressed into decentralized representations**—this is the core failure mode addressed throughout the paper.

**Goal**: To truly utilize centralized coordination without sacrificing decentralized deployability, while providing theoretical guarantees.

**Core Idea**: **Constrained guidance rather than free distillation**—maintaining an autoregressive joint guider policy for coordinated exploration, but **constraining it throughout to remain close to the decentralized learner**. This captures the coordination benefits of joint exploration while ensuring the learned strategies are decentralized-realizable, fundamentally closing the imitation gap.

## Method

### Overall Architecture
MAGPO maintains two sets of policies: an **autoregressive joint guider** $\mu(a|s)=\prod_j \mu_{i_j}(a_{i_j}|s, a_{i_{1:j-1}})$ (agents decide sequentially, observing previous actions and global information) and a **decentralized learner** $\pi(a|s)=\prod_j \pi_{i_j}(a_{i_j}|s)$ (agents independent). Training iterates in a four-step loop: ① Sample trajectories using the guider for coordinated exploration; ② Update the guider using Policy Mirror Descent (PMD); ③ Align the learner to the guider via KL minimization; ④ **Guider backtracking**—resetting the guider to the current learner. This design originates from single-agent GPO but adds sequential joint action modeling and decentralized alignment updates for MARL.

```mermaid
flowchart LR
    A[Data Collection<br/>Guider μ_k samples trajectories] --> B[Guider Update<br/>PMD/PPO improves return<br/>+KL constraint near learner]
    B --> C[Learner Update<br/>KL alignment to μ̂_k<br/>+RL auxiliary term]
    C --> D[Guider Backtrack<br/>μ_{k+1} ← π_{k+1}]
    D --> A
```

### Key Designs

**1. Autoregressive Guider + Guider Backtracking: Decoupling "Improvement" from "Deployability".** The guider uses PMD in the full joint space to find an improved policy $\hat\mu_k=\arg\max_\mu\{\eta_k\langle Q_{\mu_k}(s,\cdot),\mu(\cdot|s)\rangle-D_{KL}(\mu(\cdot|s),\mu_k(\cdot|s))\}$, which the learner then projects back into the decentralized policy space via KL minimization. The critical backtracking step $\mu_{k+1}=\pi_{k+1}$ is always theoretically feasible—any decentralized policy $\pi$ can degenerate into a valid autoregressive joint policy by "ignoring conditioning on action history." This allows the authors to prove **Theorem 4.1 (Monotonic Improvement)**: $V_\rho(\pi_{k+1})\ge V_\rho(\pi_k),\forall k$. Intuitively, the guider finds an improvement direction in the joint space, and the learner projects it down; since the target was chosen via the projection gradient, the return still improves after projection.

**2. Sequential Update Perspective Parallelized with HARL.** Using the multi-agent advantage decomposition lemma, the authors prove that the MAGPO learner update is equivalent to a set of **sequential advantage-weighted updates** (Corollary 4.2): $\pi^{i_j}_{k+1}=\arg\max_{\pi^{i_j}}\mathbb{E}[A^{i_j}_\pi(s,a_{i_{1:j-1}},a_{i_j})]-\frac{1}{\eta_k}D_{KL}(\pi^{i_j},\pi^{i_j}_k)$. This links it to theoretically grounded methods like HATRPO/HAPPO but with a fundamental difference: HARL requires agents to be **heterogeneous and updated serially**, whereas MAGPO allows **all agents to update in parallel** and holds for both homogeneous and heterogeneous agents. Thus, it can leverage the benefits of parameter sharing—a key engineering advantage in large-scale MARL.

**3. Dual Clipping + Masking: Tethering the Guider near the Learner with hyperparameter $\delta$.** In practice, the guider loss (Eq. 9) introduces a **dual clipping** $\text{clip}(\cdot,\epsilon,\delta)$ and a **mask** $m^{i_j}_t(\delta)$ in addition to standard PPO clipping. Controlled by a new hyperparameter $\delta > 1$, this forces the probability ratio between the guider and learner to stay within $(1/\delta, \delta)$. The inner clip truncates gradients when the advantage signal tries to push the guider too far from the learner, while the mask ensures the KL loss is only applied when the ratio exceeds the bounds. $\delta$ is the **most sensitive knob** in the method: the more non-decomposable the teacher strategy (e.g., CoordSum), the tighter $\delta$ must be to force mimicry; if the teacher is already easy to imitate (e.g., medium-4ag-hard), over-tightening $\delta$ will hinder learning.

**4. RL Auxiliary Term: Learner "Back-Supervision" for the Guider.** The learner loss (Eq. 10) consists of behavior cloning KL towards the guider plus a PPO-style RL auxiliary term weighted by $\lambda$. Since the guider is constrained to stay close to the learner, sampling is approximately on-policy, allowing this term to directly improve returns from trajectories. More importantly, it provides "back-supervision": if the guider's RL objective points toward a non-decomposable direction while the learner pulls it back via imitation constraints, they might stagnate; the RL auxiliary term helps the learner help the guider find a direction that is **more decentralized-realizable**. Note that this term is nearly useless for CTDS—because the CTDS behavior policy is an unaligned teacher (off-policy data), the on-policy RL loss on the student doesn't help.

## Key Experimental Results

### Main Results
SOTA comparisons were conducted on 6 JAX multi-agent suites across 43 tasks: Sable / MAT for CTCE, MAPPO / HAPPO for CTDE, and vanilla CTDS (≈ MAGPO without dual clipping, masking, and RL auxiliary). Aggregated using min-max normalized IQM with 95% bootstrap confidence intervals across 10 seeds over 20 million environment steps.

| Comparison Metric | MAGPO Performance |
|---|---|
| Tasks exceeding **all CTDE baselines** | 32 / 43 |
| Tasks exceeding **all baselines** (incl. CTCE) | 20 / 43 |
| Comparison with SOTA CTCE (Sable) | Tied or outperformed in 3 suites |
| Comparison with CTDS | Significant lead in CoordSum and RWARE |

The substantial lead in CoordSum and RWARE confirms the hypothesis: in these environments, CTCE teachers easily learn "non-decomposable" strategies where direct distillation (CTDS) fails, while MAGPO's constraint mechanism recovers performance.

### Ablation Study

| Design Component | Conclusion |
|---|---|
| **Guider Selection** (Sable vs MAT) | MAGPO performance scales with the guider: MAT weak on simple_spread_10ag → MAGPO(MAT) weak; MAT strong on large-8ag → MAGPO(MAT) better. This is a "bridge" characteristic. |
| **Constraint Ratio $\delta$** | The most sensitive hyperparameter. Small $\delta$ is better for non-decomposable tasks (CoordSum-5x20-80); small $\delta$ is restrictive for easy tasks (medium-4ag-hard). |
| **RL Auxiliary Weight $\lambda$** | Tuning $\lambda$ offers improvements but is less critical than $\delta$; CTDS gains no benefit from the same RL auxiliary (due to off-policy data). |

### Key Findings
- **Bridging CTCE and CTDE**: MAGPO allows progress in CTCE to directly benefit CTDE scenarios requiring decentralized deployment, enabling the two paradigms to evolve together.
- **Observation Asymmetry** is also lethal: CTCE is conditioned on the union of all local observations, while individual policies only see their own—this gap causes CTDS to fail even when the joint policy is decomposable, while MAGPO mitigates this by controlling divergence via $\delta$.

## Highlights & Insights
- **Reframing the "Imitation Gap" as "Constrained Projection"**: Instead of training a strong teacher then distilling, it constrains the teacher throughout to stay close to the student, ensuring coordination strategies lie within the realizable set from the start—a precise response to CTDS failure modes.
- **Monotonic Improvement + Parallelization**: Rarely achieves both theoretical guarantees (Theorem 4.1) and engineering practicality (parallel updates, compatible with parameter sharing), filling the gap between vanilla CTDE (no guarantees) and HARL (guaranteed but serial).
- **CoordSum Toy Environment**: Elegantly designed to transform the abstract failure of "non-decomposable stochastic coordination" into a reproducible benchmark.

## Limitations & Future Work
- **Bounded by Guider Upper Bound**: MAGPO typically doesn't significantly exceed its underlying CTCE method; if CTCE is weak, MAGPO is weak (framed as a "bridge property," but a ceiling nonetheless).
- **Per-task $\delta$ Tuning**: The most critical hyperparameter lacks an adaptive mechanism and depends on prior knowledge of "imitability."
- **Unused Privileged Information**: Training often has "true global state" available beyond the union of local observations; this paper does not feed such privileged signals to the guider, a direction for further improvement noted by the authors.
- The experiments are concentrated on JAX simulation suites, lacking validation on real robots or physical systems.

## Related Work & Insights
- **CTDE / Value Decomposition** (VDN, QMIX, QTRAN, QPLEX) and **Policy-based CTDE** (COMA, MADDPG, MAPPO): MAGPO critiques these for only utilizing global info via value functions.
- **CTDS** (Zhao et al., 2024, etc.): The direct target of MAGPO, noted for scalability and imitation gap issues.
- **HARL** (HATRPO/HAPPO/HASAC): Provides theoretical guarantees for sequential updates; MAGPO proves equivalence to sequential updates while being parallelizable and compatible with homogeneous parameter sharing.
- **CTCE / Transformer Sequence Modeling** (MAT, Sable): Serves as the guider backbone for MAGPO, "transferring" coordination capabilities to decentralized policies.
- **Single-agent GPO** (Li et al., 2025): The ideological origin; the contribution here lies in autoregressive joint action modeling and decentralized alignment for MARL rather than simple porting.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Transfers "constrained guidance" from single-agent GPO to MARL with specific designs for autoregressive guiders and dual-clip constraints, hitting the core of the CTDS imitation gap with a clear original approach.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 43 tasks across 6 suites, 10 seeds, rigorous IQM + confidence interval evaluation, and ablations covering Guider/$\delta$/$\lambda$. Slightly lacks real-world system validation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The CoordSum toy example clearly explains abstract failure modes; theory (monotonic improvement, sequential equivalence) and implementation are naturally linked with a complete logical chain.
- **Value**: ⭐⭐⭐⭐ Provides both theoretical support and parallel implementation; as a bridge between CTCE and CTDE, it has clear positioning and direct reference value for cooperative MARL requiring decentralized deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Guided Policy Optimization under Partial Observability](guided_policy_optimization_under_partial_observability.md)
- [\[ICLR 2026\] Heterogeneous Agent Q-weighted Policy Optimization](heterogeneous_agent_q-weighted_policy_optimization.md)
- [\[ICLR 2026\] Correlated Policy Optimization in Multi-Agent Subteams](correlated_policy_optimization_in_multi-agent_subteams.md)
- [\[ICLR 2026\] Boosting Multi-Domain Reasoning of LLMs via Curvature-Guided Policy Optimization](boosting_multi-domain_reasoning_of_llms_via_curvature-guided_policy_optimization.md)
- [\[ICML 2026\] LLM-Guided Communication for Cooperative Multi-Agent Reinforcement Learning](../../ICML2026/reinforcement_learning/llm-guided_communication_for_cooperative_multi-agent_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
