---
title: >-
  [Paper Note] Multi-Agent Decision-Focused Learning via Value-Aware Sequential Communication
description: >-
  [ICML 2026][Reinforcement Learning][Multi-Agent Communication] SeqComm-DFL treats "multi-agent communication" as a predictor and "joint policy selection" as a downstream optimizer. By employing value-aware message genera…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Multi-Agent Communication"
  - "Decision-Focused Learning"
  - "Stackelberg Sequential Decision"
  - "Bilevel Optimization"
  - "QMIX"
date: 2026-05-08
content_hash: 236f10d06b1c124a
---

# Multi-Agent Decision-Focused Learning via Value-Aware Sequential Communication

**Conference**: ICML 2026  
**arXiv**: [2604.08944](https://arxiv.org/abs/2604.08944)  
**Code**: None  
**Area**: Reinforcement Learning / Multi-Agent  
**Keywords**: Multi-Agent Communication, Decision-Focused Learning, Stackelberg Sequential Decision, Bilevel Optimization, QMIX

## TL;DR
SeqComm-DFL treats "multi-agent communication" as a predictor and "joint policy selection" as a downstream optimizer. By employing value-aware message generation, Stackelberg sequential conditioning, and implicit differentiation for bilevel optimization, it directly aligns communication learning with team return. This approach achieves a 4-6x cumulative reward improvement and over 13 percentage points increase in win rate on hospital scheduling and SMAC benchmarks.

## Background & Motivation

**Background**: Cooperative multi-agent reinforcement learning (MARL) predominantly adopts the "centralized training, decentralized execution" paradigm, exemplified by QMIX, MAPPO, and MADDPG, which use value decomposition or actor-critic methods to address non-stationarity and credit assignment. Communication learning methods (CommNet, DIAL, NDQ, SeqComm, MAIC, etc.) further alleviate coordination challenges under partial observability by enabling agents to exchange messages.

**Limitations of Prior Work**: Existing communication protocols are mostly optimized for **proxy objectives**—such as reconstruction accuracy, mutual information, or simple token prediction—rather than the actual information content that impacts downstream decision quality. This leads to bandwidth being wasted on "informative but action-irrelevant" features, mirroring the classic "objective mismatch" in model-based RL, where world models overfit pixel details at the expense of value-irrelevant errors.

**Key Challenge**: The optimization signal for the communication module (reconstruction/mutual information) is decoupled from the team’s ultimate objective (cumulative reward) in gradient direction. As a result, even if agents learn to "accurately relay their observations," this does not necessarily improve teammates’ decisions. Additionally, parallel action selection by multiple agents naturally leads to coordination and multiple equilibria issues.

**Goal**: (1) Supervise the communication module directly by "downstream decision quality"; (2) Break coordination ambiguity under parallel decision symmetry; (3) Extend decision-focused learning (DFL) from single-agent + exogenous uncertainty to multi-agent + endogenous uncertainty (where messages can alter other agents’ policies).

**Key Insight**: The authors view communication as the "predictor" in the Predict-and-Optimize framework and multi-agent policy selection as the "optimizer," naturally leading to an end-to-end paradigm where task loss is backpropagated to the communication module. The Stackelberg leader-follower structure is leveraged to break the symmetry in multi-agent action selection.

**Core Idea**: Replace "message mutual information" with "receiver Q-value improvement $\Delta Q_j(m_i)$" as the training signal for communication. Sequential conditional decision-making is constructed by sorting agents according to prosocial guidance potential, and gradients are backpropagated to communication parameters via the implicit function theorem in bilevel optimization.

## Method

### Overall Architecture
SeqComm-DFL decomposes the Dec-POMDP cooperative problem into three interdependent modules: (1) **Value-Aware Communication**—each agent encodes its local observation $o_i$ into a base message $m_i^{\text{base}}=\phi_\theta(o_i)$, then refines the message using a refinement network based on estimated receiver decision gain $\Delta\hat Q_i$; (2) **Stackelberg Sequential Action Selection**—all agents are prioritized by guidance potential $\pi=\text{argsort}(-\text{GP})$, and actions are selected sequentially $a_{\pi_k}=\arg\max_a Q_{\pi_k}(o_{\pi_k}, M_{1:\pi_k-1}, a)$, allowing followers to observe leaders’ messages; (3) **Decision-Focused World Model Bilevel Optimization**—the inner loop trains the critic using the world model, while the outer loop evaluates the critic on real environment data and backpropagates gradients to the world model and communication module via implicit differentiation.

### Key Designs

1. **Value-Aware Messaging**:

    - **Function**: Directly uses "how much a message improves teammates’ decisions" as the communication training objective, instead of traditional reconstruction error or mutual information.
    - **Mechanism**: Defines receiver decision gain as $\Delta Q_j(m_i) = \max_a Q_j(o_j, m_i, a) - \max_a Q_j(o_j, \emptyset, a)$, and the loss as $\mathcal{L}_{\text{VA}}(\theta) = -\frac{1}{B \cdot N(N-1)} \sum_b \sum_i \sum_{j\neq i} \Delta Q_j(m_i^{(b)})$, encouraging each message to maximize other agents’ optimal Q-values. Early in training, when the critic $Q_w$ is unreliable, Monte Carlo rollouts estimate $\Delta Q_j^{\text{MC}}$, and a schedule $\Delta\hat Q = (1-\beta_t)\Delta Q^{\text{MC}}+\beta_t \Delta Q_w$ is used for annealing. The paper proves via the envelope theorem that at the optimal critic, the true loss gradient with respect to the message is $\propto -\sum_{j\neq i} \nabla_{m_i} Q_j$, which matches the direction of $\Delta Q_j$, making this loss a natural consequence of DFL rather than a heuristic.
    - **Design Motivation**: Previous methods, even if messages perfectly reconstruct observations, are meaningless if these details do not alter teammates’ actions. Using $\Delta Q$ directly quantifies decision value, decoupling "what information is worth communicating" from "how many bits" in the information-theoretic sense.

2. **Stackelberg Sequential Conditioning + Guidance Potential Sorting**:

    - **Function**: Addresses relative overgeneralization and multiple equilibrium coordination in parallel multi-agent decision-making.
    - **Mechanism**: Coordination is divided into three stages. **Negotiation**: For each agent, compute the prosocial guidance potential $\text{GP}_i(s) = \mathbb{E}_{\mathbf{a}^*}[Q_{1:N}(s,\mathbf{a}^*|i^+) - Q_{1:N}(s,\mathbf{a}^*|i^-)]$, measuring the team benefit of making the agent the leader, and obtain a differentiable priority order $\pi=\text{argsort}(-\text{GP})$ via Gumbel-softmax. **Launching**: Agents select actions sequentially according to $\pi$, with the $k$-th agent conditioning on all higher-priority messages $M_{1:\pi_k-1}$. **Regularization**: A counterfactual influence loss $\mathcal{L}_{\text{inf}} = -\frac{1}{N(N-1)}\sum_i \sum_{j\neq i} D_{\text{KL}}[\pi_j(\cdot|m_i)\,\|\,\pi_j(\cdot|\emptyset)]$ ensures that messages actually change receiver policies rather than merely correlate. Theoretically, $\text{GP}_i \propto \sum_{j\neq i} I(M_i; a_j^*|o_j)$, so agents with the largest information gap naturally become leaders.
    - **Design Motivation**: Unlike SeqComm, which sorts by "willingness to act," guidance potential is prosocial—prioritizing agents holding key private information for coordination, enabling teammates to make decisions based on truly valuable priors and achieving Pareto-superior Stackelberg equilibria.

3. **Decision-Focused Bilevel World Model + Implicit Differentiation**:

    - **Function**: Ensures the world model $f_\theta$ is optimized for "maximizing final team return" rather than "predicting the next state most accurately," integrating communication and the world model into a unified outer optimizer.
    - **Mechanism**: The outer loop minimizes $\mathcal{L}_{\text{true}}(w^*(\theta);\theta)$ (critic evaluated on real environment data), while the inner loop $w^*(\theta)=\arg\min_w \mathcal{L}_{\text{model}}(w;\theta) + \lambda_{\text{aware}}\mathcal{L}_{\text{aware}}(w)$ trains the critic using model predictions. Here, $\mathcal{L}_{\text{aware}}$ is a hinge-style "message-aware regularizer": $\max(0, \epsilon_{\text{margin}} - |Q_w(s,a,M)-Q_w(s,a,\mathbf{0})|)$, forcing the critic to distinguish between message and zero-message inputs, preventing "inner loop apathy"—if the critic ignores $M$, the hypergradient to communication parameters vanishes. The implicit function theorem is used to expand at the inner fixed point: $\frac{dw^*}{d\theta}=-[\nabla^2_{ww}\mathcal{L}_{\text{model}}]^{-1}\nabla^2_{\theta w}\mathcal{L}_{\text{model}}$, avoiding backpropagation through $K_{\text{inner}}$ SGD steps (which can vanish/explode). The remaining inverse Hessian-vector product $H^{-1}b$ is approximated via conjugate gradient $(H+\lambda I)v^* = b$, requiring only two autodiff calls per step, keeping overall complexity manageable.
    - **Design Motivation**: In MARL, the world model and critic are inherently chained in a bilevel structure. OMD has shown that in single-agent settings, such decoupling yields a tighter bound $\|Q^* - \hat Q_{\text{DFL}}\|_\infty \le \epsilon/(1-\gamma)$ than MLE’s $\epsilon_R/(1-\gamma)+\gamma\epsilon_P r_{\max}/2(1-\gamma)^2$. This work extends it to multi-agent scenarios with communication and QMIX decomposition, adding message apathy regularization.

### Loss & Training

The overall outer objective is $\theta \leftarrow \theta - \eta(\frac{d\mathcal{L}_{\text{true}}}{d\theta}+\lambda_{\text{VA}}\nabla\mathcal{L}_{\text{VA}}+\lambda_{\text{inf}}\nabla\mathcal{L}_{\text{inf}})$; the inner loop trains $w$ with $K_{\text{inner}}$ steps of SGD; target networks use Polyak EMA $\bar w \leftarrow \tau_{\text{ema}}\bar w + (1-\tau_{\text{ema}})w$; during warmup, $\beta_t = \min(t/T_w, 1)$ smoothly transitions from MC-based to critic-based $\Delta Q$; Gumbel-softmax provides differentiable exploration for priority sorting. Theoretical convergence is $\frac{1}{T}\sum_t \mathbb{E}\|\nabla_\theta \mathcal{L}_{\text{true}}\|^2 \le O(1/\sqrt T)$.

## Key Experimental Results

### Main Results
Two environments: a custom hospital multi-specialty collaboration ($N=3$ specialists, $\mathcal P=100$ patients, specialty-gated hidden risks), and the classic SMAC benchmark.

| Environment | Metric | SeqComm-DFL | Prev. SOTA | Gain |
|-------------|--------|-------------|------------|------|
| Hospital Dec-POMDP | Cumulative Reward | 4-6× baseline | QMIX/MAPPO/SeqComm | Multiple-fold improvement |
| SMAC | Win Rate | +13pp | QMIX/MAIC | Significant outperformance |
| Hospital | Communication Value $\Delta V$ | Matches theoretical lower bound $\frac{L_R}{1-\gamma}\sum\sqrt{2\ln 2\cdot I_i\cdot\text{Var}(a_i^*)}$ | — | Validates Thm 5.1 |

### Ablation Study

| Configuration | Key Effect | Description |
|---------------|------------|-------------|
| Full SeqComm-DFL | Optimal | All modules enabled |
| w/o $\mathcal{L}_{\text{VA}}$ | Communication degrades to reconstruction | Messages no longer optimized for decisions |
| w/o Stackelberg ordering | Coordination multiple equilibria | Simultaneous decisions fall into suboptimal equilibria |
| w/o $\mathcal{L}_{\text{aware}}$ | Inner loop apathy | Critic ignores messages, hypergradient vanishes |
| w/o IFT / direct BPTT inner loop | Training diverges | Backpropagating $K_{\text{inner}}$ steps leads to vanishing gradients |

### Key Findings
- Value-aware loss and message-aware regularization are both necessary for end-to-end trainability; omitting either causes communication to be drowned out by environmental noise.
- Guidance potential sorting enables agents with large information gaps $\mathcal I_i$ to naturally become leaders; otherwise, SeqComm degenerates to intention-based sorting.
- Convergence rate $O(1/\sqrt T)$ is closely tied to the bias $\epsilon_{\text{bias}}=\epsilon_{\text{inner}}+\epsilon_{\text{CG}}$ from implicit differentiation and CG; too few CG iterations bias the outer gradient.

## Highlights & Insights
- **Communication as the DFL "predictor"**: The authors are the first to explicitly place multi-agent communication within the predict-and-optimize framework, and use the envelope theorem to show that $\Delta Q$ is the dual of the true loss gradient—an elegant "theory-to-engineering-loss" path.
- **Message apathy regularization**: The unique failure mode of bilevel + communication is the critic ignoring messages; the hinge regularizer enforces a margin between $Q$ with $M$ and $\mathbf 0$, a strategy transferable to any scenario where "auxiliary inputs are easily ignored" (e.g., weak conditioning in diffusion, retrieval results in RAG).
- **Stackelberg + prosocial sorting**: Treating "who speaks first" as a learnable problem, with priority based on team benefit rather than individual preference, is the key difference from SeqComm; Gumbel-softmax makes the ordering differentiable and is lightweight in practice.

## Limitations & Future Work
- Implicit differentiation + CG is sensitive to iteration count and damping $\lambda$ in high-dimensional $w$; the complexity analysis assumes well-conditioned $H$, which may not hold in continuous control.
- Sequential leader-follower decision-making introduces execution delay as agent count $N$ increases; experiments only cover medium-scale SMAC, leaving scalability to swarm-level unaddressed.
- The hospital Dec-POMDP is a custom environment, with specialty gating tailored to create "information gap" scenarios; cross-domain real-world evaluation (e.g., traffic lights, multi-vehicle coordination) is lacking.
- Communication remains as continuous vectors $m\in\mathbb R^{d_m}$; discrete symbols and practical bandwidth constraints are not considered.

## Related Work & Insights
- **vs SeqComm (Ding 2023)**: Both use sequential communication, but SeqComm sorts by intention value ("who most wants to act"), while this work uses prosocial guidance potential ("who can help teammates most") and optimizes message content end-to-end, extending local greediness to team optimality.
- **vs MAIC (Yuan 2022)**: MAIC treats messages as Q-value incentives; this work adopts this idea for $\mathcal{L}_{\text{aware}}$, but further binds incentives to decision-focused bilevel optimization.
- **vs OMD (Nikishin 2022)**: OMD addresses objective mismatch in single-agent model-based RL; this work extends to multi-agent, endogenous uncertainty, and communication, with QMIX decomposition for scalability.
- **vs DFL (Donti 2017 / Elmachtoub-Grigas)**: Classic DFL assumes predictions do not affect downstream optimization ground truth; this is the first work to explicitly handle endogenous uncertainty (messages alter other agents’ action distributions) in DFL.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to extend DFL to multi-agent + endogenous uncertainty + communication, elegantly unifying three domains.
- Experimental Thoroughness: ⭐⭐⭐⭐ SMAC + custom hospital environment cover both symmetric and asymmetric information, but lack large-scale swarm and real industrial scenarios.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear progression from concept → theory → algorithm → experiment, with theorems and engineering losses tightly matched.
- Value: ⭐⭐⭐⭐ Inspiring for communication learning, model-based MARL, and DFL communities, but implicit differentiation is a high barrier and engineering reproduction requires many tricks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Counterfactual Effect Decomposition in Multi-Agent Sequential Decision Making](../../ICML2025/reinforcement_learning/counterfactual_effect_decomposition_in_multi-agent_sequential_decision_making.md)
- [\[ICML 2026\] Vulnerable Agent Identification in Large-Scale Multi-Agent Reinforcement Learning](vulnerable_agent_identification_in_large-scale_multi-agent_reinforcement_learnin.md)
- [\[ICLR 2026\] Continuous-Time Value Iteration for Multi-Agent Reinforcement Learning](../../ICLR2026/reinforcement_learning/continuous-time_value_iteration_for_multi-agent_reinforcement_learning.md)
- [\[NeurIPS 2025\] Sequential Multi-Agent Dynamic Algorithm Configuration](../../NeurIPS2025/reinforcement_learning/sequential_multi-agent_dynamic_algorithm_configuration.md)
- [\[ICLR 2026\] Distributionally Robust Cooperative Multi-Agent Reinforcement Learning via Robust Value Factorization](../../ICLR2026/reinforcement_learning/distributionally_robust_cooperative_multi-agent_reinforcement_learning_via_robus.md)

</div>

<!-- RELATED:END -->
