---
title: >-
  [Paper Note] Dual-Robust Cross-Domain Offline Reinforcement Learning Against Dynamics Shifts
description: >-
  [ICLR 2026][Reinforcement Learning][Offline Reinforcement Learning] This work is the first to simultaneously address train-time robustness (source–target domain dynamics mismatch) and test-time robustness (deployment-env…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "Offline Reinforcement Learning"
  - "Cross-Domain Transfer"
  - "Dynamics Shift"
  - "Dual Robustness"
  - "Bellman Operator"
date: 2026-05-08
content_hash: aaf8eeacd1a4f3a1
---

# Dual-Robust Cross-Domain Offline Reinforcement Learning Against Dynamics Shifts

**Conference**: ICLR 2026
**arXiv**: [2512.02486](https://arxiv.org/abs/2512.02486)  
**Code**: [https://github.com/zq2r/DROCO](https://github.com/zq2r/DROCO)  
**Area**: AI Safety / RL Robustness
**Keywords**: Offline Reinforcement Learning, Cross-Domain Transfer, Dynamics Shift, Dual Robustness, Bellman Operator

## TL;DR
This work is the first to simultaneously address train-time robustness (source–target domain dynamics mismatch) and test-time robustness (deployment-environment dynamics shift) in cross-domain offline RL. The proposed DROCO algorithm centers on the Robust Cross-Domain Bellman (RCB) operator—applying a robust Bellman update to source-domain data and a standard in-sample update to target-domain data—and reformulates intractable dynamics uncertainty as state-space perturbations via dual reconstruction. On the D4RL benchmark, DROCO achieves a total score of 1105.2, surpassing the second-best method by 14%, while exhibiting performance degradation under hard-level dynamics perturbations that is only half that of the baselines.

## Background & Motivation

**Background**: The core setting of cross-domain offline RL is learning policies by leveraging abundant source-domain data when target-domain data is scarce—e.g., in robotic manipulation, the source domain is a simulator (plentiful data) and the target domain is a real robot (very limited data). The two domains share state space, action space, and reward function, but differ in transition dynamics $P$. Existing methods such as DARA (reward correction via a domain classifier), IGDF (source-domain data filtering via mutual information), and OTDF (dynamics alignment via optimal transport) all focus on handling the source–target dynamics mismatch, i.e., so-called *train-time robustness*.

**Limitations of Prior Work**: These methods implicitly assume that once the source–target domain discrepancy is handled during training, the policy will perform normally at deployment. In practice, however, the dynamics of the deployment environment itself can shift—robot component wear, joint loosening, and payload changes all cause the actual transition dynamics to deviate from those of the training target domain. The authors empirically confirm this: a policy trained with IGDF on the hopper task suffers performance drops of 40.9% and 72.4% under medium- and hard-level kinematic perturbations, respectively. The problem is further exacerbated when target-domain data is reduced to 10%, as the policy overfits more severely to the dynamics characteristics present in the limited data.

**Key Challenge**: Train-time robustness and test-time robustness are orthogonal requirements. The former concerns a known source–target discrepancy (observable in the training data), while the latter concerns resistance to unknown deployment-environment perturbations (invisible at training time). Existing cross-domain offline RL methods address only the former, whereas single-domain robust RL methods address only the latter and cannot handle cross-domain data fusion.

**Goal**: Design a unified framework that simultaneously guarantees: (1) safe utilization of source-domain data without Q-value overestimation caused by OOD dynamics (train-time robustness); and (2) a performance lower bound for the learned policy under deployment-environment dynamics shifts (test-time robustness).

**Key Insight**: The authors observe that applying a robust Bellman operator to source-domain data (taking the worst case within a Wasserstein uncertainty set) inherently produces a conservative estimation effect—suppressing Q-value inflation induced by OOD dynamics (resolving the train-time problem) while also endowing the policy with resistance to dynamics perturbations (resolving the test-time problem). Standard in-sample Bellman updates are applied to target-domain data to fully exploit ground-truth dynamics information.

**Core Idea**: A single RCB operator unifies dual robustness—performing robust Bellman backup on source-domain data and standard backup on target-domain data—and translates dynamics uncertainty into actionable state perturbations via Wasserstein duality.

## Method

### Overall Architecture
The complete DROCO pipeline consists of four steps: (1) train an ensemble dynamics model $\hat{P}_\psi = \{\hat{P}_{\psi_i}\}_{i=1}^N$ on target-domain data $\mathcal{D}_{\text{tar}}$ via MLE; (2) for each source-domain transition $(s,a,s')$, use the ensemble to generate $N$ predicted next states $\{s'_1, \ldots, s'_N\}$ and use the one with the minimum Q-value as the robust Bellman target; (3) apply standard in-sample Bellman targets to target-domain data; (4) incorporate a dynamic value penalty and Huber loss for training stability, and optimize the policy with IQL. The inputs are offline datasets from both domains; the output is a policy that is robust to dynamics shifts at deployment.

### Key Designs

1. **Robust Cross-Domain Bellman (RCB) Operator**

    - **Function**: Defines distinct Bellman update rules for source- and target-domain data to achieve dual robustness.
    - **Mechanism**: For target-domain transitions $(s,a,s') \in \mathcal{D}_{\text{tar}}$, the RCB operator reduces to the standard in-sample Bellman operator $r + \gamma \mathbb{E}_{s'}[\max_{a' \sim \hat{\mu}} Q(s',a')]$, fully utilizing ground-truth dynamics. For source-domain transitions $(s,a,s') \in \mathcal{D}_{\text{src}}$, the worst case is taken within the Wasserstein uncertainty set $\mathcal{M}_\epsilon$: $r + \gamma \inf_{\hat{\mathcal{M}} \in \mathcal{M}_\epsilon} \mathbb{E}_{s' \sim P_{\hat{\mathcal{M}}}}[\max_{a'} Q(s',a')]$. The authors prove that the RCB operator is a $\gamma$-contraction (Proposition 4.1) with a unique fixed point, guaranteeing convergence. Key theoretical results include Proposition 4.4 (train-time robustness): when $\epsilon$ is large enough for the uncertainty set to cover the support of $P_{\text{tar}}$, the learned Q-values will not be overestimated; and Proposition 4.5 (test-time robustness): when the dynamics shift of the deployment environment is within Wasserstein distance $c$, the policy's actual performance in the perturbed environment is no lower than the learned robust value function.
    - **Design Motivation**: Naively mixing source- and target-domain data with standard Bellman updates leads to Q-value overestimation, since the source dynamics $P_{\text{src}}$ may transition the agent to states that appear high-reward in the target domain but are in fact unreachable. By introducing robustness constraints on source-domain data, RCB simultaneously addresses OOD dynamics and deployment robustness.

2. **Wasserstein Dual Reconstruction — Dynamics-to-State-Perturbation Mapping**

    - **Function**: Transforms the intractable optimization over a dynamics uncertainty set in the RCB operator into actionable state-space perturbations.
    - **Mechanism**: Via the dual form of the Wasserstein distance (Proposition 4.2), $\inf_{\hat{\mathcal{M}} \in \mathcal{M}_\epsilon}$ is equivalently reformulated as $\mathbb{E}_{s' \sim P_{\mathcal{M}}}[\inf_{\bar{s}: d(s',\bar{s}) \leq \epsilon} \max_{a'} Q(\bar{s}, a')]$. This means that instead of enumerating all MDPs in the dynamics uncertainty set, one need only search for the Q-minimizing state $\bar{s}$ within the $\epsilon$-neighborhood of the observed next state $s'$. In practice, the $N$ ensemble predictions $\{s'_i\}$ replace explicit $\epsilon$-ball search, and $\min_i Q(s'_i, \pi(s'_i))$ serves as the robust target.
    - **Design Motivation**: The original RCB operator requires knowledge of the source-domain dynamics uncertainty set $\mathcal{M}_\epsilon$, but the source environment is a black box. The dual reconstruction circumvents this issue, and the ensemble model further adaptively calibrates uncertainty—regions where model uncertainty is high naturally produce more dispersed predictions—avoiding the excessive conservatism that a fixed $\epsilon$ might impose.

3. **Dynamic Value Penalty and Huber Loss for Training Stability**

    - **Function**: Mitigates Q-value over- or under-estimation caused by ensemble model approximation errors.
    - **Mechanism**: A value penalty term $u(s,a,s') = \mathbb{I}(s' \sim P_{\text{src}}) \cdot (V(s') - \min_i V(s'_i))$ is defined as the difference between the source-domain observed $V(s')$ and the minimum ensemble-predicted $V$ value. The penalty coefficient $\beta$ controls conservatism: $\beta=1$ recovers the original RCB; $\beta>1$ increases conservatism (suppressing overestimation); $\beta<1$ reduces conservatism (alleviating underestimation). Additionally, Huber loss replaces L2 loss for source-domain Bellman updates: when the TD error $|Q - \hat{\mathcal{T}}Q| < \delta$, L2 loss is used; beyond $\delta$, L1 loss is automatically applied, preventing abnormally large TD errors (from inaccurate model predictions) from destabilizing training.
    - **Design Motivation**: Proposition 4.6 proves that a TV-distance error of $\epsilon$ in the ensemble model causes Q-value overestimation of at most $(1-(1-2\epsilon)^N) \cdot r_{\max}/(1-\gamma)$. Furthermore, the inf operation itself tends to produce underestimation. The two techniques target these respective failure modes.

### Loss & Training
The overall Q-function loss is:
$$\mathcal{L}_Q = \mathbb{E}_{\mathcal{D}_{\text{src}}}[l_\delta(Q - \hat{\mathcal{T}}_{\text{RCB}} Q)] + \frac{1}{2}\mathbb{E}_{\mathcal{D}_{\text{tar}}}[(Q - \mathcal{T}Q)^2]$$
The source-domain term uses Huber loss $l_\delta$ with RCB targets; the target-domain term uses standard L2 loss with standard Bellman targets. The ensemble dynamics model is trained via MLE on target-domain data: $\mathcal{L}_{\psi_i} = \mathbb{E}_{\mathcal{D}_{\text{tar}}}[\log \hat{P}_{\psi_i}(s'|s,a)]$. Policy optimization follows the IQL framework. Training runs for 1M steps with an ensemble size of $N=7$.

## Key Experimental Results

### Main Results: Normalized Scores across 16 Tasks under Kinematic Shift

| Task | IQL* | CQL* | BOSA | DARA | IGDF | OTDF | **DROCO** |
|------|------|------|------|------|------|------|-----------|
| half-m | 45.2 | 37.7 | 39.6 | 44.1 | 45.2 | 42.2 | **45.3** |
| half-mr | 22.1 | 23.6 | 26.3 | 21.6 | 22.9 | 15.6 | **26.9** |
| half-me | 43.7 | 54.8 | 42.2 | 52.7 | 57.1 | 46.7 | **60.1** |
| hopp-m | 48.8 | 35.7 | 71.4 | 48.8 | 54.3 | 46.3 | **55.4** |
| hopp-mr | 40.2 | 43.2 | 29.5 | 41.6 | 30.0 | 26.2 | **47.3** |
| walk-m | 48.7 | 47.7 | 44.5 | 43.4 | 51.8 | 43.0 | **70.8** |
| walk-mr | 12.6 | 17.8 | 4.8 | 15.6 | 11.2 | 10.7 | **27.7** |
| walk-e | 90.1 | 83.8 | 41.9 | 85.5 | 93.7 | 98.9 | **106.0** |
| ant-me | 106.1 | 100.6 | 102.5 | 104.8 | 112.8 | 105.1 | **119.0** |
| ant-e | 111.0 | 94.3 | 57.6 | 115.1 | 119.2 | 111.6 | **120.0** |
| **Total (16 tasks)** | 925.4 | 789.9 | 774.5 | 923.0 | 964.3 | 969.8 | **1105.2** |

DROCO achieves the best performance on 9 of the 16 tasks, with a total score exceeding the second-best method (OTDF) by 14.0% (1105.2 vs. 969.8). Particularly large gains are observed on walker2d-medium (70.8 vs. 51.8) and walker2d-medium-replay (27.7 vs. 17.8). On a few tasks (e.g., half-expert: 67.4 vs. BOSA's 84.3), DROCO is suboptimal; the authors attribute this to an inherent robustness–performance trade-off.

### Test-Time Robustness: Performance Degradation under Different Perturbation Types and Intensities

| Perturbation Type | Intensity | DROCO Degradation | IGDF Degradation | OTDF Degradation |
|---|---|---|---|---|
| Kinematic | Easy | 19.3% | >50% | >50% |
| Kinematic | Medium | ~30% | ~65% | ~55% |
| Kinematic | Hard | ~45% | ~85% | ~75% |
| Morphological | Easy | 42.1% | 78.9% | 62.4% |
| Min-Q Attack | scale=0.2 | 37.9% | 84.0% | 73.6% |

DROCO exhibits substantially lower degradation than baselines across all perturbation types and intensities. Notably, under adversarial min-Q attacks (which deliberately seek state perturbations that minimize Q-values), DROCO remains consistently stable across attack scales, confirming that the robust design of the RCB operator is effective. Degradation under morphological perturbations is larger (42.1% vs. 19.3% for kinematic), as only kinematic shifts are present in the source domain during training, making morphological perturbations an out-of-distribution perturbation type at test time.

### Key Findings
- **Necessity of dual robustness**: Methods with only train-time robustness (IGDF, OTDF) perform reasonably in clean environments but degrade severely under deployment shifts—IGDF degrades by >85% under hard kinematic perturbations—directly validating the paper's core motivation.
- **Tuning patterns for $\beta$ and $\delta$**: $\beta \leq 1.0$ is appropriate for most tasks (suggesting that underestimation from the inf operation is more prevalent than overestimation), and $\delta = 30$ or $50$ is a robust default (L2 loss benefits training stability, switching to L1 only for extreme outliers).
- **Effect of target-domain data quantity**: As target-domain data is reduced from 100% to 10%, test-time robustness degrades substantially for all methods, but DROCO's relative advantage becomes more pronounced, indicating that the RCB operator is more effective under data scarcity.
- **Task-specific hyperparameter preferences**: The hopper task favors $\beta=0.1$ (requiring reduced conservatism), while walker2d favors $\beta=1.0$ (requiring full conservatism), suggesting that the direction and magnitude of value estimation bias are task-dependent.

## Highlights & Insights
- **Elegant unification via the RCB operator**: A single operator addresses both robustness requirements, with the trade-off between them controlled theoretically by the parameter $\epsilon$. This is more parsimonious than designing two separate mechanisms and more amenable to theoretical analysis. The key insight is recognizing that applying a robust Bellman update to source-domain data simultaneously achieves conservative estimation of OOD dynamics and deployment robustness.
- **Practical realization through dual reconstruction**: The optimization over a dynamics uncertainty set (requiring enumeration over infinitely many MDPs) is reduced to an $\epsilon$-ball search in state space (searching for the worst-case state in a finite-dimensional space), which is further approximated by discrete ensemble predictions. This pathway from "computationally intractable" to "practically feasible" is broadly instructive—other RL methods involving distributionally robust optimization could adopt similar dual-plus-ensemble approximation strategies.
- **Design philosophy of the dynamic penalty coefficient**: Rather than fixing the degree of conservatism, the penalty is made self-adaptive: $u(s,a,s')$ directly measures the discrepancy between the observed source-domain dynamics and the ensemble predictions, imposing larger penalties where discrepancies are larger. This data-driven conservatism adjustment is transferable to other offline RL methods that involve domain discrepancies.

## Limitations & Future Work
- **Lipschitz Q-function assumption**: The theoretical analysis (Propositions 4.4 and 4.5) relies on the assumption that the Q-function is Lipschitz continuous with respect to the state. This assumption is difficult to verify or enforce in high-dimensional state spaces or when Q is parameterized by a deep network, particularly when Q exhibits sharp value variations.
- **Bottleneck of ensemble model quality**: The core state-perturbation approximation relies entirely on the quality of the ensemble dynamics model. When target-domain data is extremely scarce, the model itself may severely overfit, rendering the generated "perturbed states" meaningless. This failure mode is not sufficiently discussed in the paper.
- **Hyperparameter tuning burden**: Although the authors provide empirical guidance ($\beta \leq 1.0$, $\delta = 30$), optimal values do vary across tasks. Tuning remains necessary for new tasks, and $\epsilon$ (implicitly determined through the ensemble model) is itself a latent hyperparameter.
- **Evaluation limited to MuJoCo**: Assessment on only four continuous control tasks is relatively narrow in scope. More complex cross-domain RL settings—including high-dimensional observations (image input), discrete action spaces, and multi-agent scenarios—are not explored.
- **Future directions**: Replacing the ensemble model with a diffusion model could more accurately capture the target-domain dynamics distribution. Adaptively adjusting $\epsilon$ based on state regions, rather than applying a single global value, is another promising direction.

## Related Work & Insights
- **vs. IGDF (Wen et al., 2024)**: IGDF filters unreliable source-domain data via mutual information, addressing only train-time robustness. DROCO modifies the Bellman update rule rather than filtering data, simultaneously achieving test-time robustness. DROCO outperforms IGDF by approximately 15% in total score, with an even larger advantage under deployment perturbations.
- **vs. OTDF (Lyu et al., 2025)**: OTDF aligns source–target domain dynamics via optimal transport, focusing solely on train-time robustness. OTDF outperforms DROCO on some expert-level data tasks (e.g., hopper-expert: 97.0 vs. 89.3), suggesting that when data quality is high, precise dynamics alignment may be more effective than conservative robust estimation.
- **vs. MICRO (Liu et al., 2024c)**: MICRO is a robust method for single-domain offline RL; the design of the RCB operator is inspired by MICRO but extends it to the cross-domain setting and handles source–target domain discrepancies.
- **vs. practical sim-to-real scenarios**: The DROCO framework directly corresponds to the three-level discrepancy problem in sim-to-real RL: simulator (source domain) → real robot (target domain) → actual deployment (potentially shifted environment), with a clear application-driven motivation.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The dual robustness problem formulation is novel and practically meaningful; the RCB operator's divide-and-conquer treatment of source and target domain data is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Coverage is reasonably comprehensive—16 standard D4RL tasks, 3 types of test-time perturbations, and hyperparameter sensitivity analysis—though validation on more complex environments is absent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The theoretical derivations are clear and complete; the logical chain from problem formulation → theoretical operator → dual reconstruction → practical algorithm is coherent and well-structured.
- **Value**: ⭐⭐⭐⭐ The work offers direct guidance for RL deployment in sim-to-real scenarios, and the concept of dual robustness is generalizable to other cross-domain learning problems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Robust Deep Reinforcement Learning against Adversarial Behavior Manipulation](robust_deep_reinforcement_learning_against_adversarial_behavior_manipulation.md)
- [\[ICLR 2026\] Cross-Embodiment Offline Reinforcement Learning for Heterogeneous Robot Datasets](cross-embodiment_offline_reinforcement_learning_for_heterogeneous_robot_datasets.md)
- [\[ICLR 2026\] Less is More: Clustered Cross-Covariance Control for Offline RL](less_is_more_clustered_cross-covariance_control_for_offline_rl.md)
- [\[ICLR 2026\] Offline Reinforcement Learning with Generative Trajectory Policies](offline_reinforcement_learning_with_generative_trajectory_policies.md)
- [\[ICLR 2026\] Distributionally Robust Cooperative Multi-Agent Reinforcement Learning via Robust Value Factorization](distributionally_robust_cooperative_multi-agent_reinforcement_learning_via_robus.md)

</div>

<!-- RELATED:END -->
