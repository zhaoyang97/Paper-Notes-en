---
title: >-
  [Paper Note] Multi-Agent Decision-Focused Learning via Value-Aware Sequential Communication
description: >-
  [ICML 2026][Reinforcement Learning][Multi-agent Communication] SeqComm-DFL treats "multi-agent communication" as a predictor and "joint policy selection" as a downstream optimizer. It directly aligns communication learni…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Multi-agent Communication"
  - "Decision-Focused Learning"
  - "Stackelberg Sequential Decision Making"
  - "Bilevel Optimization"
  - "QMIX"
date: 2026-05-08
content_hash: 7ecfa28bda00f5c2
---

# Multi-Agent Decision-Focused Learning via Value-Aware Sequential Communication

**Conference**: ICML 2026  
**arXiv**: [2604.08944](https://arxiv.org/abs/2604.08944)  
**Code**: None  
**Area**: Reinforcement Learning / Multi-Agent  
**Keywords**: Multi-agent Communication, Decision-Focused Learning, Stackelberg Sequential Decision Making, Bilevel Optimization, QMIX

## TL;DR
SeqComm-DFL treats "multi-agent communication" as a predictor and "joint policy selection" as a downstream optimizer. It directly aligns communication learning with team rewards using value-aware message generation, Stackelberg sequential conditioning, and implicit differentiation for bilevel optimization. It achieves a 4-6x increase in cumulative rewards in hospital scheduling and over 13 percentage points win rate improvement in SMAC.

## Background & Motivation

**Background**: Collaborative Multi-Agent Reinforcement Learning (MARL) typically adopts the "Centralized Training and Decentralized Execution" (CTDE) paradigm, represented by QMIX, MAPPO, and MADDPG, which mitigate non-stationarity and credit assignment via value decomposition or actor-critic methods. Communication learning methods (CommNet, DIAL, NDQ, SeqComm, MAIC, etc.) further address coordination challenges under partial observability by allowing agents to exchange messages.

**Limitations of Prior Work**: Current communication protocol optimization targets are largely **proxy objectives**—such as reconstruction accuracy, mutual information, or simple token prediction—rather than the information volume that truly impacts downstream decision quality. Consequently, bandwidth is wasted on features that are "informative but action-irrelevant," mirroring the classic "objective mismatch" problem in model-based RL where world models sacrifice value-relevant error optimization to fit pixel-level details.

**Key Challenge**: The optimization signals for communication modules (reconstruction/mutual information) are decoupled from the ultimate team objective (cumulative reward) in terms of gradient direction. Even if an agent learns to "accurately relay its observations," it may not lead to better teammate decisions. Additionally, concurrent action selection among multiple agents naturally suffers from coordination ambiguity due to multiple equilibria.

**Goal**: (1) Enable the communication module to be supervised directly by "downstream decision quality"; (2) Break coordination ambiguity under the symmetry of concurrent decision-making; (3) Extend Decision-Focused Learning (DFL) from single-agent exogenous uncertainty to multi-agent endogenous uncertainty (where messages inversely alter other agents' policies).

**Key Insight**: The authors view communication as the "predictor" in a Predict-and-Optimize framework and multi-agent policy selection as the "optimizer." This leads to an end-to-end paradigm of "backpropagating task loss to the communication module." They also utilize a Stackelberg leader-follower structure to break the symmetry of multi-agent action selection.

**Core Idea**: Replace "message mutual information" with "receiver Q-value gain $\Delta Q_j(m_i)$" as the communication training signal. Sequential conditional decisions are constructed by ordering agents according to prosocial guidance potential. Finally, gradients from bilevel optimization are backpropagated to communication parameters via the Implicit Function Theorem.

## Method

### Overall Architecture
SeqComm-DFL partitions the collaborative Dec-POMDP problem into three coupled modules: (1) **Value-Aware Communication** — Each agent encodes local observations $o_i$ into base messages $m_i^{\text{base}}=\phi_\theta(o_i)$, refined by a network based on estimated receiver decision gains $\Delta\hat Q_i$. (2) **Stackelberg Sequential Action Selection** — Agents are prioritized using guidance potential $\pi=\text{argsort}(-\text{GP})$, selecting actions sequentially $a_{\pi_k}=\arg\max_a Q_{\pi_k}(o_{\pi_k}, M_{1:\pi_k-1}, a)$ so followers can observe messages emitted by leaders. (3) **Decision-Focused World Model Bilevel Optimization** — The inner loop trains a critic using world model predictions, while the outer loop evaluates the critic on real environment data, backpropagating gradients to world model and communication parameters via implicit differentiation.

### Key Designs

1. **Value-Aware Messaging**:

    - **Function**: Directly uses "how much the message improves teammate decisions" as the communication training objective, replacing traditional reconstruction error or mutual information.
    - **Mechanism**: Defines receiver decision gain as $\Delta Q_j(m_i) = \max_a Q_j(o_j, m_i, a) - \max_a Q_j(o_j, \emptyset, a)$. The loss $\mathcal{L}_{\text{VA}}(\theta) = -\frac{1}{B \cdot N(N-1)} \sum_b \sum_i \sum_{j\neq i} \Delta Q_j(m_i^{(b)})$ encourages messages to maximize the optimal Q-values of other agents. Since the critic $Q_w$ is unreliable early in training, $\Delta Q_j^{\text{MC}}$ is estimated using Monte Carlo rollouts and transitioned via annealing: $\Delta\hat Q = (1-\beta_t)\Delta Q^{\text{MC}}+\beta_t \Delta Q_w$. Using the envelope theorem, the paper proves the gradient of the true loss w.r.t. the message is $\propto -\sum_{j\neq i} \nabla_{m_i} Q_j$ at the optimal critic, showing that this loss is naturally derived from DFL.
    - **Design Motivation**: Previous methods made messages irrelevant if details did not alter teammate actions. Quantifying decision value with $\Delta Q$ decouples "what information is worth communicating" from "how many bits are in the information theory sense."

2. **Stackelberg Sequential Conditioning + Guidance Potential Sorting**:

    - **Function**: Resolves relative overgeneralization and multiple equilibria coordination issues in concurrent multi-agent decision-making.
    - **Mechanism**: Coordination is divided into three phases. **Negotiation**: Calculate prosocial guidance potential $\text{GP}_i(s) = \mathbb{E}_{\mathbf{a}^*}[Q_{1:N}(s,\mathbf{a}^*|i^+) - Q_{1:N}(s,\mathbf{a}^*|i^-)]$, measuring team gain if agent $i$ acts as leader; differentiable priority $\pi=\text{argsort}(-\text{GP})$ is obtained via Gumbel-softmax. **Launching**: Actions are selected sequentially by $\pi$; the $k$-th agent conditions on all higher-priority messages $M_{1:\pi_k-1}$. **Regularization**: A counterfactual influence loss $\mathcal{L}_{\text{inf}} = -\frac{1}{N(N-1)}\sum_i \sum_{j\neq i} D_{\text{KL}}[\pi_j(\cdot|m_i)\,\|\,\pi_j(\cdot|\emptyset)]$ ensures messages actually change the receiver's policy. Theoretically, $\text{GP}_i \propto \sum_{j\neq i} I(M_i; a_j^*|o_j)$, pushing agents with the largest information gaps to the leader position.
    - **Design Motivation**: Unlike SeqComm's "willingness to act" sorting, guidance potential is altruistic—prioritizing agents holding critical private information for coordination so teammates can make decisions based on valuable priors, reaching a Pareto-superior Stackelberg equilibrium.

3. **Decision-Focused Bilevel World Model + Implicit Differentiation**:

    - **Function**: Optimizes the world model $f_\theta$ for the "highest final team reward" rather than "most accurate next-state prediction," integrating communication and world model into the same outer optimizer.
    - **Mechanism**: The outer loop minimizes $\mathcal{L}_{\text{true}}(w^*(\theta);\theta)$ (evaluating the critic on real data). The inner loop $w^*(\theta)=\arg\min_w \mathcal{L}_{\text{model}}(w;\theta) + \lambda_{\text{aware}}\mathcal{L}_{\text{aware}}(w)$ trains the critic on model predictions. $\mathcal{L}_{\text{aware}}$ is a hinge-based "message-aware regularization": $\max(0, \epsilon_{\text{margin}} - |Q_w(s,a,M)-Q_w(s,a,\mathbf{0})|)$, forcing the critic to distinguish inputs with and without messages to prevent "inner-loop apathy"—where hypergradients vanish if the critic ignores $M$. Using the Implicit Function Theorem at the inner local optimum, $\frac{dw^*}{d\theta}=-[\nabla^2_{ww}\mathcal{L}_{\text{model}}]^{-1}\nabla^2_{\theta w}\mathcal{L}_{\text{model}}$ is computed to avoid backpropagating through $K_{\text{inner}}$ steps. The inverse Hessian-vector product $H^{-1}b$ is approximated using Conjugate Gradient (CG) via $(H+\lambda I)v^* = b$.
    - **Design Motivation**: Bilevel coupling of world models and critics in MARL is common. OMD proved this decoupling yields tighter bounds $\|Q^* - \hat Q_{\text{DFL}}\|_\infty \le \epsilon/(1-\gamma)$ than MLE. Ours extends this to multi-agent scenarios with communication and QMIX decomposition.

### Loss & Training
Total outer objective: $\theta \leftarrow \theta - \eta(\frac{d\mathcal{L}_{\text{true}}}{d\theta}+\lambda_{\text{VA}}\nabla\mathcal{L}_{\text{VA}}+\lambda_{\text{inf}}\nabla\mathcal{L}_{\text{inf}})$. Inner loop uses $K_{\text{inner}}$ SGD steps for $w$. Target networks use Polyak EMA $\bar w \leftarrow \tau_{\text{ema}}\bar w + (1-\tau_{\text{ema}})w$. Warmup phase $\beta_t = \min(t/T_w, 1)$ transitions MC-based $\Delta Q$ to critic-based. Gumbel-softmax enables differentiable exploration for priority sorting. Theoretical convergence is $O(1/\sqrt T)$.

## Key Experimental Results

### Main Results
Tested on two environments: a custom multi-specialty hospital collaboration ($N=3$ specialists, $P=100$ patients, specialty-gated hidden risks) and the SMAC benchmark.

| Environment | Metric | Ours | Prev. SOTA | Gain |
|------|------|-------------|-----------|------|
| Hospital Dec-POMDP | Cumulative Reward | 4-6× baseline | QMIX/MAPPO/SeqComm | Multiple Fold |
| SMAC | Win Rate | +13pp | QMIX/MAIC | Significant |
| Hospital | Comm Value $\Delta V$ | Consistent with lower bound $\frac{L_R}{1-\gamma}\sum\sqrt{2\ln 2\cdot I_i\cdot\text{Var}(a_i^*)}$ | — | Validates Thm 5.1 |

### Ablation Study

| Configuration | Key Effect | Description |
|------|---------|------|
| Full SeqComm-DFL | Optimal | All modules active |
| w/o $\mathcal{L}_{\text{VA}}$ | Comm decays to reconstruction | Messages not optimized for decisions |
| w/o Stackelberg ordering | Coordination equilibrium failure | Concurrent decisions trapped in sub-optimal equilibria |
| w/o $\mathcal{L}_{\text{aware}}$ | Inner loop apathy | Critic ignores messages, hypergradients vanish |
| w/o IFT / Direct BPTT Inner | Training divergence | Vanishing gradients through $K_{\text{inner}}$ steps |

### Key Findings
- Value-aware loss and message-aware regularization are both necessary for end-to-end training; otherwise, communication is drowned by environmental noise.
- Guidance potential sorting makes the "most knowledgeable" agent the leader in high information-gap $\mathcal{I}_i$ scenarios; otherwise, it degrades to intention-based sorting like SeqComm.
- Convergence rate $O(1/\sqrt T)$ is closely tied to the bias $\epsilon_{\text{bias}}=\epsilon_{\text{inner}}+\epsilon_{\text{CG}}$ from implicit differentiation and CG. Insufficient CG iterations lead to biased outer gradients.

## Highlights & Insights
- **Communication as a DFL "Predictor"**: This work is the first to explicitly define multi-agent communication within a predict-and-optimize framework, using the envelope theorem to prove $\Delta Q$ is the dual of the true loss gradient. This path from theoretical duality to engineering loss is elegant.
- **Message Apathy Regularization**: Addresses a failure mode specific to bilevel communication where critics learn to ignore messages. The hinge loss forces a margin between $Q(M)$ and $Q(\mathbf{0})$, a concept applicable to any scenario where auxiliary inputs are easily ignored (e.g., weak conditioning in diffusion, retrieval results in RAG).
- **Stackelberg + Prosocial Ordering**: Learning "who speaks first" based on team gain rather than individual preference is the fundamental difference from SeqComm. The Gumbel-softmax approach makes the permutation differentiable and lightweight.

## Limitations & Future Work
- Implicit differentiation and CG are sensitive to iterations and damping coefficient $\lambda$ in high-dimensional $w$. Complexity analysis relies on a well-conditioned $H$, which may not hold in continuous control.
- Sequential leader-follower execution introduces latency as $N$ increases. Scaling to swarm-level agents was not addressed.
- The hospital Dec-POMDP is a custom environment; specialty gating is a "perfect fit" for the method's information-gap handling. Cross-domain evaluation (e.g., traffic lights) is missing.
- Communication remains continuous vectors $m\in\mathbb R^{d_m}$; discrete symbols and actual bandwidth constraints were not considered.

## Related Work & Insights
- **vs SeqComm (Ding 2023)**: Both use sequential communication, but SeqComm sorts by intention value. Ours uses prosocial guidance potential and optimizes content end-to-end, extending local greed to team optimality.
- **vs MAIC (Yuan 2022)**: MAIC treats messages as Q-value incentives. Ours adopts this for $\mathcal{L}_{\text{aware}}$ but binds incentives to decision-focused bilevel optimization.
- **vs OMD (Nikishin 2022)**: OMD solves objective mismatch in single-agent model-based RL. Ours extends this to multi-agent endogenous uncertainty + communication with QMIX.
- **vs DFL (Donti 2017 / Elmachtoub-Grigas)**: Classic DFL assumes predictions do not affect the optimizer's ground truth. Ours is the first DFL work to handle endogenous uncertainty where messages alter other agents' action distributions.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Elegant unification of DFL, multi-agent communication, and endogenous uncertainty.
- Experimental Thoroughness: ⭐⭐⭐⭐ Good coverage of information asymmetry, but lacks swarm scaling and industrial benchmarks.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely clear logic from theory to engineering loss.
- Value: ⭐⭐⭐⭐ Inspiring for comm-learning and model-based MARL, though implicit differentiation requires significant engineering effort.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] LLM-Guided Communication for Cooperative Multi-Agent Reinforcement Learning](llm-guided_communication_for_cooperative_multi-agent_reinforcement_learning.md)
- [\[ICLR 2026\] Continuous-Time Value Iteration for Multi-Agent Reinforcement Learning](../../ICLR2026/reinforcement_learning/continuous-time_value_iteration_for_multi-agent_reinforcement_learning.md)
- [\[ICML 2026\] Vulnerable Agent Identification in Large-Scale Multi-Agent Reinforcement Learning](vulnerable_agent_identification_in_large-scale_multi-agent_reinforcement_learnin.md)
- [\[NeurIPS 2025\] Sequential Multi-Agent Dynamic Algorithm Configuration](../../NeurIPS2025/reinforcement_learning/sequential_multi-agent_dynamic_algorithm_configuration.md)
- [\[ICML 2026\] Learning Query-Aware Budget-Tier Routing for Runtime Agent Memory](learning_query-aware_budget-tier_routing_for_runtime_agent_memory.md)

</div>

<!-- RELATED:END -->
