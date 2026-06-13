---
title: >-
  [Paper Note] Interaction-Breaking Adversarial Learning Framework for Robust Multi-Agent Reinforcement Learning
description: >-
  [ICML 2026][Reinforcement Learning][MARL] This paper characterizes the "mutual influence" between multiple agents using conditional mutual information from an information-theoretic perspective. It designs an attacker tha…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "MARL"
  - "CTDE"
  - "Mutual Information"
  - "Adversarial Training"
  - "Collaborative Robustness"
date: 2026-05-08
content_hash: 448bb9822213fa61
---

# Interaction-Breaking Adversarial Learning Framework for Robust Multi-Agent Reinforcement Learning

**Conference**: ICML 2026  
**arXiv**: [2605.18024](https://arxiv.org/abs/2605.18024)  
**Code**: https://sunwoolee0504.github.io/IBAL  
**Area**: Reinforcement Learning / Multi-Agent / Robust MARL  
**Keywords**: MARL, CTDE, Mutual Information, Adversarial Training, Collaborative Robustness  

## TL;DR
This paper characterizes the "mutual influence" between multiple agents using conditional mutual information from an information-theoretic perspective. It designs an attacker that simultaneously masks observations and perturbs actions to minimize cross-group mutual information. Based on this, the IBAL policy is trained to maintain stable decision-making even when collaborative components collapse. It significantly outperforms existing robust MARL methods under various attacks and "missing teammate" perturbations in SMAC, SMACv2, and LBF environments.

## Background & Motivation

**Background**: Cooperative MARL typically adopts the Centralized Training with Decentralized Execution (CTDE) framework, such as VDN and QMIX, which decompose the joint action-value $Q^{tot}$ into individual utilities $Q^i$. While this tight coupling allows agents to learn sophisticated collaborative strategies, it also makes policies extremely sensitive to "interaction patterns rarely seen during training."

**Limitations of Prior Work**: Existing robust MARL research (such as adversarial regularization in Lin et al. 2020, critical-moment action attacks in ROMANCE/EGA, serial targeted attacks in Wolfpack, and RL-based attackers in ATLA) mostly attributes robustness to "value-oriented" perturbations. Attackers directly select actions that minimize $Q^i$ or add FGSM noise to observations. The implicit assumption is that "perturbations only occur at the input of a single agent" and do not directly destroy the "dependency structure between agents." Consequently, when attacks truly sever collaboration links (e.g., making one group unable to see another), the defenses of these methods collapse.

**Key Challenge**: There is a mismatch in structural assumptions between the training and execution phases of CTDE. During training, value decomposition assumes agents can consistently "read" each other. In real-world deployment, collaboration may be interrupted by communication failures, visual occlusions, or unit deaths, leading to interaction breaks that the policy has almost never encountered in the training distribution.

**Goal**: (i) Quantify "cross-agent influence" in a manner independent of the $Q^{tot}$ structure; (ii) construct an attacker capable of directly eliminating this influence; (iii) learn a set of policies robust to "collaboration collapse" that can further generalize to non-parametric perturbations like "missing teammates."

**Key Insight**: The authors randomly partition $n$ agents into two groups, $G_1$ and $G_2$, using conditional mutual information $\mathcal{I}(\boldsymbol{o}_{t+1}^{G_1}, \boldsymbol{a}_t^{G_1}; \boldsymbol{a}_t^{G_2} \mid \boldsymbol{\tau}_t)$ to characterize the influence of $G_2$ on $G_1$. According to the chain rule for mutual information, this is decomposed into observation-level and action-level terms. By using an observation-masking attacker and an action-rewriting attacker to minimize these two terms respectively, an attack is obtained that does not rely on value estimation and specifically targets the breakdown of collaboration channels.

**Core Idea**: Use "interaction breaking," defined as the minimization of cross-group mutual information, as the adversarial training objective. It is proven that this is equivalent to standard MARL optimization on an induced Dec-POMDP with perturbed transitions, thereby reducing the "collaboration collapse robustness" problem to a standard value learning problem.

## Method

### Overall Architecture
IBAL wraps a "Grouping-Attacking-Training" loop around the CTDE training cycle. At the start of each episode, $k \sim \mathrm{Unif}(\{0,\dots,K\})$ is sampled along with a random subgroup $G_1 \subset \mathcal{N}$ (restricted to $|G_1| \le n/2$ to avoid excessive attack strength), where $G_2 = \mathcal{N}\setminus G_1$. At each environment step, the observation attacker $\boldsymbol{f}_{\mathrm{adv}}$ identifies the top $L$ dimensions of the observation components of $G_1$ directed toward $G_2$ via mutual information scoring and sets them to zero. The policy $\boldsymbol{\pi}$ then selects intended actions $\hat{\boldsymbol{a}}_t$ based on the masked observations $\tilde{\boldsymbol{o}}_t$. With probability $P_{\mathrm{act}}$, the action attacker $\boldsymbol{\pi}_{\mathrm{adv}}$ is triggered to replace the sub-actions of $G_1$ with actions $\tilde{\boldsymbol{a}}_t^{\mathrm{min},G_1}$ that minimize mutual information with $\hat{\boldsymbol{a}}_t^{G_2}$. The environment advances using the perturbed joint action $\tilde{\boldsymbol{a}}_t$. Transitions and rewards are generated according to real dynamics but are equivalently viewed as samples from a new "perturbed transition" Dec-POMDP $\tilde{\mathcal{M}}$, which can be directly fed into standard CTDE optimizers like QMIX or MAPPO.

The theoretical support is Theorem 4.2: If the "observation attack $\to$ policy $\to$ action attack" sequence is viewed as a composite policy $\boldsymbol{\pi}_{\mathrm{adv}} \circ \boldsymbol{\pi} \circ \boldsymbol{f}_{\mathrm{adv}}$, its value in the Joint-Adversarial Dec-POMDP $\mathcal{M}^J$ is exactly equal to the value $\tilde{V}_{\boldsymbol{\pi}}(s_t)$ of the original policy $\boldsymbol{\pi}$ in an induced Dec-POMDP $\tilde{\mathcal{M}}$ where the state is expanded to $\tilde{s}_t = (s_t, \tilde{\boldsymbol{a}}_t)$ and the transition becomes $\tilde{P}(\tilde{s}_{t+1}\mid \tilde{s}_t, \hat{\boldsymbol{a}}_t) := P(s_{t+1}\mid s_t, \hat{\boldsymbol{a}}_t)\cdot \boldsymbol{\pi}_{\mathrm{adv}}(\tilde{\boldsymbol{a}}_t\mid s_t, \hat{\boldsymbol{a}}_t)$. This equivalence transforms "learning the optimal policy under an attacker" back into "learning the optimal policy in a new environment," allowing the direct use of the QMIX loss without specialized minimax optimization.

### Key Designs

1. **Cross-group Influence Characterization and Chain Decomposition**:
    - **Function**: Measures "how much $G_2$ influences $G_1$" using a metric completely decoupled from value functions, further splitting it into two attackable components.
    - **Mechanism**: The influence of $G_2$ on $G_1$ is defined as $\mathcal{I}(\boldsymbol{o}_{t+1}^{G_1}, \boldsymbol{a}_t^{G_1}; \boldsymbol{a}_t^{G_2} \mid \boldsymbol{\tau}_t)$. By the chain rule of mutual information, this decomposes into an observation-level term $\mathcal{I}(\boldsymbol{o}_{t+1}^{G_1}; \boldsymbol{a}_t^{G_2} \mid \boldsymbol{a}_t^{G_1}, \boldsymbol{\tau}_t)$ and an action-level term $\mathcal{I}(\boldsymbol{a}_t^{G_1}; \boldsymbol{a}_t^{G_2} \mid \boldsymbol{\tau}_t)$. The former reflects to what extent the actions of $G_2$ change the future observations of $G_1$, while the latter reflects the synergy between the actions of $G_1$ and $G_2$.
    - **Design Motivation**: Value-oriented attacks only consider the reduction in value and cannot express the extent of collaboration severance. Mutual information directly corresponds to "information flow" and is independent of structural assumptions like QMIX monotonicity, providing consistent attack directions even when value estimates are unreliable.

2. **Observation Attack: Dimension-level MI Upper Bound + Zero-masking**:
    - **Function**: Selects $L$ dimensions from each agent's observation in $G_1$ that most inform them of $G_2$'s state or actions and masks them out.
    - **Mechanism**: Calculating mutual information for all possible $L$-dimensional subsets leads to combinatorial explosion. The authors prove Lemma 4.3: group-level observation mutual information is upper-bounded by the sum of dimension-level mutual informations plus a group redundancy term $\mathcal{R}(G_1;G_2)$. Empirical tests show the redundancy term is negligible. Thus, the optimal mask is approximated as $D^{i,*} = \arg\max_{D^i:|D^i|=L}\sum_{d\in D^i}\sum_{j\in G_2}\mathcal{I}(o^i_{d,t+1}; a^j_t \mid a^i_t, \boldsymbol{\tau}_t)$, outputting $\tilde{o}^i_{d,t}=0$ for $d\in D^{i,*}$. IBAL uses "all-zero" masking rather than Gaussian noise/FGSM based on the Data Processing Inequality $\mathcal{I}(f(X);Y\mid Z)\le \mathcal{I}(X;Y\mid Z)$; deterministic zeroing maximizes signal erasure.
    - **Design Motivation**: Reducing the combinatorial optimization to per-dimension scoring makes the attack cost linearly controllable while providing interpretable results regarding which observation components carry cross-group information.

3. **Action Attack + Adaptive Attack Intensity Scheduling**:
    - **Function**: Assuming the intended action $\hat{\boldsymbol{a}}_t$ is collaborative, it rewrites $G_1$'s actions to minimize mutual information with $\hat{\boldsymbol{a}}_t^{G_2}$, gradually increasing attack probability during training.
    - **Mechanism**: Let $\tilde{\boldsymbol{a}}_t^{\mathrm{min},G_1} := \arg\min_{\boldsymbol{a}_t^{G_1}} \mathcal{I}(\boldsymbol{a}_t^{G_1}; \hat{\boldsymbol{a}}_t^{G_2}\mid \boldsymbol{\tau}_t)$. The attacker outputs $\langle \tilde{\boldsymbol{a}}_t^{\mathrm{min},G_1}, \hat{\boldsymbol{a}}_t^{G_2}\rangle$ with probability $P_{\mathrm{act}}$. The attack probability is sampled from $P_{\mathrm{act}}\sim\mathrm{Unif}(1/K, P_{\mathrm{act}}^{\max})$. Whenever the average win rate $\bar\sigma$ exceeds a threshold $\eta$, the upper bound is scaled by $\alpha$: $P_{\mathrm{act}}^{\max}\leftarrow \min(1,\alpha P_{\mathrm{act}}^{\max})$.
    - **Design Motivation**: Fixed $P_{\mathrm{act}}$ decouples learning intensity from attack intensity. Adaptive scheduling embeds curriculum-based difficulty into training, while the $1/K$ lower bound ensures non-trivial attacks even for weak policies.

### Loss & Training
The value loss follows the standard objective of the chosen backbone (QMIX or MAPPO), with the only difference being that transitions are sampled from $\tilde{\mathcal{M}}$. The CLUB and KL estimators are updated online alongside the policy. All SMAC experiments are run for 10M steps on QMIX, initialized from a 1M-step pre-trained QMIX to ensure fair comparison. Observation masking is symmetrized to avoid training bias. The group upper limit $K\le n/2$ is a key hyperparameter searched by environment.

## Key Experimental Results

### Main Results
Baselines include Vanilla QMIX, Rand-Obs/Rand-Act, FGSM, ATLA, ERNIE, ROMANCE, and WALL. Attacks considered cover Nat. / Rand. / FGSM / EGA / Wolfpack / Ours.

| Evaluation Setting | Vanilla QMIX | ROMANCE / WALL (Strong Trend) | IBAL (Ours) |
| :--- | :--- | :--- | :--- |
| Natural (No Attack) | Mid-High | Close to Vanilla | At least as strong as Vanilla |
| FGSM / EGA / Wolfpack Attacks | Significant Drop | Good against own attack; collapses under Interaction-Breaking | Maintains high win rate; superior on 1c3s5z |
| Interaction-Breaking Attack | Most severe collapse | Most severe collapse | Significantly highest |
| Dis-1 / Dis-2 (Teammate Disabled) | Sharp decline | Generally drops significantly | Gap further widens |
| HP-15 (Initial HP -15%) | Degradation | Slightly better but still drops | Clearly leading |
| LBF / SMACv2 Performance | Hampered by stochasticity | Close to Vanilla | Higher, showing robustness to stochasticity |
| MAPPO backbone | — | — | Improved robustness; decoupled from structure |

### Ablation Study
| Configuration | 8m Dis-1 Win Rate (%) | Notes |
| :--- | :--- | :--- |
| IBAL Full | 88.4 ± 3.3 | Complete Method |
| w/o adaptive prob. (Fixed $P_{\mathrm{act}}=1/K$) | Significant drop | Curriculum intensity is crucial |
| w/ random masking (vs. MI-based) | Drop | MI guidance is more effective than random |
| w/o Observation Attack | Substantial drop | Observation and Action attacks are not substitutable |
| w/o Action Attack | Substantial drop | Same as above |
| IBAL + Gaussian noise | 78.1 ± 13.3 (8m) | Residual MI weakens the attack |
| IBAL + FGSM mask | 38.5 ± 7.4 (8m) | FGSM is strong for visual robustness but weak for severing info flow |

### Key Findings
- Visualizing trajectories shows that IBAL's "MI-minimizing actions" manifest as directional "retreats to decouple from $G_2$," whereas value-minimization attacks trigger unreliable estimates for non-optimal actions under QMIX monotonicity, resulting in oscillatory jitter.
- IBAL policies learn emergency behaviors for collaboration collapse: in 8m, a healthy teammate moves forward to replace a damaged front-line unit; in MMM, low-HP units actively approach distracted Medivacs. These behaviors are rare in standard training and are forced by frequent collaboration interruptions.
- A larger $K$ is not always better: $K=1$ is optimal for 2s3z, while $K=4$ is stable for 8m. Excessively large $K$ hinders learning, reflecting a trade-off between attack intensity and learnability.

## Highlights & Insights
- **Value-oriented vs. Information-oriented Attacks**: This paper extends robust MARL attacks from "value reduction" to "information flow severance," providing a new attack surface more relevant to real-world scenarios like communication hijacking or visual occlusion.
- **Dimensional MI Upper Bound**: Converts the MI attack from a theoretical concept into an engineering solution. Calculating each (agent, dim) pair only once and aggregating during group changes avoids the computational explosion typically associated with MI attacks.
- **JA-Dec-POMDP $\to$ Induced Dec-POMDP Equivalence**: Bypasses unstable minimax training, allowing attack-defense training to share a single CTDE implementation. This architectural decoupling enables easy integration with both QMIX and MAPPO.
- **Implicit Coverage of Missing Teammate Scenarios**: The "Interaction-Breaking" attack + random grouping during training effectively covers the extreme case of "teammate disappearance," leading to significant gains in Dis-$\ell$ settings.

## Limitations & Future Work
- Introduces several hyperparameters ($K, L, P_{\mathrm{act}}^{\max}, \alpha, \eta$) that require small-scale searching across tasks.
- Requires continuous training of CLUB/KL estimators, incurring a "moderate" additional computational cost compared to vanilla QMIX.
- Evaluation is focused on the SMAC family and LBF; validation in real-world communication-impaired or heterogeneous agent scenarios is needed.
- The current adversary is "information-severing"; it does not yet cover "active misinformation injection" (e.g., forged observations).

## Related Work & Insights
- **vs. ROMANCE / EGA**: These use RL-learned attackers to minimize value estimates. IBAL uses MI-minimization, making the attack independent of Q-network structure and more robust against unreliable value estimates.
- **vs. Wolfpack / WALL**: While Wolfpack attacks individuals sequentially, IBAL severs group-level dependencies.
- **vs. ATLA**: ATLA trains an RL observation attacker, while IBAL provides a closed-form MI attack with a group curriculum, avoiding adversarial RL instability.
- **vs. MI for Collaboration**: Unlike previous work using MI as a reward to promote collaboration, IBAL uses "MI-minimization" as a dual adversarial objective for robust training.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Systematizes "interaction breakage" as a robust MARL attack surface with a complete proof-of-concept.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Extensive coverage across SMAC, LBF, and SMACv2 with multiple backbones and ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Clear connection between theoretical derivation and engineering implementation.
- **Value**: ⭐⭐⭐⭐ Directly applicable to deployment scenarios where collaboration is prone to failure.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Vulnerable Agent Identification in Large-Scale Multi-Agent Reinforcement Learning](vulnerable_agent_identification_in_large-scale_multi-agent_reinforcement_learnin.md)
- [\[ICML 2026\] LLM-Guided Communication for Cooperative Multi-Agent Reinforcement Learning](llm-guided_communication_for_cooperative_multi-agent_reinforcement_learning.md)
- [\[AAAI 2026\] ChartEditor: A Reinforcement Learning Framework for Robust Chart Editing](../../AAAI2026/reinforcement_learning/charteditor_a_reinforcement_learning_framework_for_robust_chart_editing.md)
- [\[ICLR 2026\] Robust Deep Reinforcement Learning against Adversarial Behavior Manipulation](../../ICLR2026/reinforcement_learning/robust_deep_reinforcement_learning_against_adversarial_behavior_manipulation.md)
- [\[AAAI 2026\] MARS: A Meta-Adaptive Reinforcement Learning Framework for Risk-Aware Multi-Agent Portfolio Management](../../AAAI2026/reinforcement_learning/mars_a_meta-adaptive_reinforcement_learning_framework_for_risk-aware_multi-agent.md)

</div>

<!-- RELATED:END -->
