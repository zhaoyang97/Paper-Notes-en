---
title: >-
  [Paper Note] Efficient Multiagent Planning via Shared Action Suggestions
description: >-
  [AAAI 2026][Reinforcement Learning][Dec-POMDP] This paper proposes the MCAS algorithm, which infers other agents' belief states by sharing only "suggested actions" within a decentralized POMDP framework…
tags:
  - "AAAI 2026"
  - "Reinforcement Learning"
  - "Dec-POMDP"
  - "multiagent planning"
  - "action suggestion communication"
  - "belief space pruning"
  - "human-agent teaming"
date: 2026-05-08
content_hash: c83ed223409b70ee
---

# Efficient Multiagent Planning via Shared Action Suggestions

**Conference**: AAAI 2026
**arXiv**: [2412.11430](https://arxiv.org/abs/2412.11430)  
**Code**: [github.com/sisl/MCAS](https://github.com/sisl/MCAS)  
**Area**: Reinforcement Learning
**Keywords**: Dec-POMDP, multiagent planning, action suggestion communication, belief space pruning, human-agent teaming

## TL;DR

This paper proposes the MCAS algorithm, which infers other agents' belief states by sharing only "suggested actions" within a decentralized POMDP framework, achieving coordination performance close to centralized methods while substantially reducing communication overhead and computational complexity.

## Background & Motivation

### Problem Definition

Cooperative decision-making among multiple agents under uncertainty is a central challenge in artificial intelligence. The Dec-POMDP (Decentralized Partially Observable Markov Decision Process) is the standard framework for modeling this problem; however, its finite-horizon complexity is NEXP-complete, rendering it intractable for most practical applications.

### Limitations of Prior Work

- **Communication-free Dec-POMDP**: Each agent makes decisions based solely on its own action-observation history, resulting in severe information incompleteness and coordination difficulties.
- **Fully communicating Dec-POMDP-Com**: If agents can perfectly share all observations and actions, the problem reduces to an MPOMDP (multiagent POMDP) with complexity PSPACE-complete. However, full observation sharing is often impractical due to limited bandwidth, latency, and noise.
- **Existing communication schemes**: Most prior work focuses on designing optimal communication policies (when and what to communicate), but when communication is neither lossless nor free, the finite-horizon Dec-POMDP-Com remains NEXP-complete.

### Core Motivation

The authors observe a fundamental pattern in human collaboration: people typically coordinate through **suggested actions** (e.g., "Let's go to that restaurant") rather than by sharing detailed observations or beliefs. Such action suggestions implicitly encode the suggester's understanding of the environment—recommending a restaurant implies a belief that it is open, suitable for the group's preferences, and reasonably priced. This motivates the key question: can efficient multiagent coordination be achieved by sharing only suggested actions?

## Method

### Overall Architecture

The MCAS (Multiagent Control via Action Suggestions) algorithm proceeds as follows:

1. **Offline phase**: For each agent $i$, solve an MPOMDP in which the agent receives individual observations but controls the full joint action space; additionally solve a joint policy $\tilde{\pi}$ under the assumption of shared observations.
2. **Online phase**: At each step, (a) each agent proposes a suggested action based on its local belief and policy; (b) a coordinating agent receives all suggestions and infers other agents' beliefs via belief pruning; (c) a joint belief estimate is constructed from the inferred beliefs; (d) the joint policy $\tilde{\pi}$ is used to select the final action, which is then broadcast.

This requires solving $n+1$ MPOMDPs ($n$ individual policies plus one joint policy), rather than solving the Dec-POMDP directly.

### Key Designs

#### 1. **Belief Subspace Inference**: Inferring Belief Regions from Suggested Actions

Core idea: if agent $j$ suggests action $a_s$ under policy $\pi^j$, its belief $b^j$ must lie in the subspace of beliefs for which $a_s$ is optimal:

$$\mathcal{B}^j_{a_s} = \{b \mid \pi^j(b) = a_s, \forall b \in \mathcal{B}^j\}$$

For alpha-vector policies, this subspace admits an exact characterization:

$$\mathcal{B}^j_{a_s} = \{\mathbf{b} \mid (\boldsymbol{\alpha}_i - \boldsymbol{\alpha}_j) \cdot \mathbf{b} \geq 0, \forall \boldsymbol{\alpha}_i \in \Gamma_{a_s}, \forall \boldsymbol{\alpha}_j \in \Gamma\}$$

**Design Motivation**: Although action suggestions carry less information than full observations, they suffice to substantially constrain the plausible belief space, thereby approximating the performance of centralized methods.

#### 2. **Belief Pruning**: Controlling Exponential Growth of the Belief Set

At each step, agent $i$ must maintain a reachable belief set $\hat{\mathcal{B}}^{i,j}$ for every other agent $j$. Without pruning, this set grows to $(|\mathcal{O}^j| \prod_{i \neq j} |\mathcal{A}^k|)^\ell$ after $\ell$ steps—precisely the source of Dec-POMDP's NEXP complexity.

MCAS controls this growth through three mechanisms:

- **Action consistency pruning**: For each candidate belief $b$, check whether $\hat{\pi}^{i,j}(b) = \sigma^{j,i}$ is consistent with the received suggestion; inconsistent beliefs are removed.
- **$\delta$-ball merging**: Exploiting the Lipschitz property of POMDP value functions ($|V(b) - V(b')| \leq \frac{\|R\|_\infty}{1-\gamma} \delta$ when $\|b-b'\|_1 \leq \delta$), beliefs within $L_1$ distance $\delta$ are merged.
- **Hard cap**: When the belief set exceeds $\bar{B}_{max}=200$, the lower-weight belief from the closest pair is iteratively removed.

#### 3. **Joint Belief Estimation**: Fusing Multi-Source Beliefs

Two approaches are proposed:

**Exact reconstruction**: Inferred observation and action sequences are used to directly update the joint belief—theoretically sound but computationally expensive, requiring maintenance of $\prod_{j \neq i} |\hat{\mathcal{B}}^{i,j}|$ joint belief combinations.

**Practical approximation—Conflation**:

$$\hat{\tilde{b}}^i(s) = \frac{b^i(s) \prod_{j \neq i} \hat{b}^{i,j}(s)}{\sum_{s' \in \mathcal{S}} b^i(s') \prod_{j \neq i} \hat{b}^{i,j}(s')}$$

Advantages of conflation: (1) minimizes Shannon information loss; (2) automatically assigns higher weight to more confident beliefs; (3) requires no manual weight specification; (4) its non-idempotent nature makes it well-suited for fusing independently obtained observations.

### Loss & Training

The proposed method is planning-based rather than learning-based; the core optimization objective is to maximize expected cumulative discounted reward. In practice, the SARSOP algorithm is used offline to compute alpha-vector policies, while online execution performs belief updates and pruning for real-time coordination.

**MCAS-α variant**: Shares alpha-vector indices rather than raw actions, providing more precise belief subspace information (a single action may correspond to multiple alpha vectors, whereas an index uniquely identifies a subspace region).

## Key Experimental Results

### Main Results

Evaluated on multiple standard Dec-POMDP benchmarks, with 2,000 runs per scenario and 95% confidence intervals reported.

| Problem | Variant | Agents | MPOMDP | MCAS-α | MCAS | Independent | Dec-POMDP |
|---------|---------|--------|--------|--------|------|-------------|-----------|
| Dec-Tiger | — | 2 | 59.5±0.9 | 58.5±0.9 | 58.5±0.8 | -68.1±3.5 | 13.5 |
| Dec-Tiger | — | 3 | 108.5±1.0 | 108.5±1.0 | 108.5±1.0 | -95.5±4.1 | — |
| Broadcast | — | 2 | 9.4±0.0 | 9.4±0.0 | 9.4±0.0 | 7.6±0.1 | 9.3 |
| Meet 3×3 | AG,UI,WP | 2 | 7.3±0.1 | 7.3±0.1 | 7.1±0.1 | 2.8±0.1 | — |
| Box Push | SO | 2 | 204.3±2.5 | 203.2±2.5 | 199.8±2.5 | 138.5±3.8 | — |
| Mars Rover | UI | 2 | 23.9±0.1 | 23.9±0.1 | 19.8±0.2 | 15.3±0.2 | — |
| Mars Rover | 55G,UI | 2 | 20.7±0.1 | 20.7±0.8 | 18.0±0.2 | 13.1±0.2 | — |

### Ablation Study

Analysis of belief set size (maximum $|\hat{\mathcal{B}}^{1,j}|$ per simulation):

| Problem | Variant | MCAS-α | MCAS |
|---------|---------|--------|------|
| Meet 3×3 | — | 1.0±0.0 | 2.5±0.0 |
| Meet 27 | UI,WP | 1.5±0.0 | 16.8±1.6 |
| Box Push | SO | 4.8±0.1 | 192.1±1.2 |
| Mars Rover | UI | 1.0±0.0 | 2.0±0.0 |

### Key Findings

1. **MCAS-α ≈ MPOMDP-C**: MCAS-α, which shares alpha-vector indices, matches the performance of MPOMDP-C (which uses true belief fusion) on nearly all problems, validating the effectiveness of belief pruning.
2. **MCAS remains competitive**: MCAS, sharing only actions, performs slightly below MCAS-α but still substantially outperforms independent execution and most known Dec-POMDP solvers.
3. **Belief pruning is efficient**: The hard cap is triggered in only a minority of problems (Meet 27: 3.2%; Box Push-SO: 87.8%); belief sets remain very small in most cases.
4. **Communication value is heterogeneous**: In some problems (e.g., Broadcast, Wireless), a single agent's observations are already sufficient, and sharing observations yields negligible additional benefit.

## Highlights & Insights

1. **Elegant problem reformulation**: The NEXP-hard Dec-POMDP is reduced to solving $n+1$ MPOMDPs (PSPACE) plus lightweight online inference, achieving an excellent balance between theoretical complexity and practical efficiency.
2. **Natural fit for human-agent teaming**: Action suggestions are the most natural form of communication in human collaboration, and the framework extends directly to human-agent team scenarios.
3. **Judicious use of conflation**: Selecting conflation rather than a simple mixture distribution for belief fusion leverages its mathematical properties of minimizing information loss and providing automatic weighting.
4. **MCAS-α as a performance upper bound**: The comparison between MCAS and MCAS-α reveals the difference in information content between action-level and alpha-vector-level granularity for belief inference.

## Limitations & Future Work

1. **Strong assumptions**: The method requires perfect, cost-free communication; assumes all agents share a consistent initial belief; and requires the surrogate policy to match the true policy.
2. **Scalability bottleneck in the offline phase**: Although online execution is lightweight, the complexity of solving MPOMDPs offline grows exponentially with the number of agents.
3. **Validation limited to offline policies**: The approach has not yet been integrated with online solvers (e.g., AdaOPS, BetaZero), limiting applicability to larger-scale problems.
4. **Heuristic belief selection requires improvement**: The current count-based belief selection is insufficiently precise in some problems; regret minimization strategies are worth exploring.
5. **No analysis of non-compliance scenarios**: The case in which agents do not always follow the suggested actions remains unaddressed.

## Related Work & Insights

- Complements the occupancy state approach (Dibangoye et al.): the latter reformulates Dec-POMDP as a continuous-state MDP, whereas this work reduces the search space through communication.
- Aligns with the action advisory philosophy in TCAS/ACAS X: aviation collision avoidance systems use action suggestions (e.g., "do not descend") to implicitly coordinate, and this paper formalizes and generalizes that idea.
- Provides a theoretical grounding for communication learning in multiagent reinforcement learning—suggested actions constitute a structured, low-bandwidth communication modality.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of action suggestion communication and belief pruning is original
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multiple classic benchmarks, diverse variants, and thorough baseline comparisons
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear structure, rigorous notation, complete algorithmic pseudocode
- **Value**: ⭐⭐⭐⭐ — Provides a practical new paradigm for human-agent teaming and multiagent planning

---

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Partial Action Replacement: Tackling Distribution Shift in Offline MARL](partial_action_replacement_tackling_distribution_shift_in_offline_marl.md)
- [\[AAAI 2026\] Intention-Guided Cognitive Reasoning for Egocentric Long-Term Action Anticipation](intention-guided_cognitive_reasoning_for_egocentric_long-term_action_anticipatio.md)
- [\[AAAI 2026\] Good-for-MDP State Reduction for Stochastic LTL Planning](good-for-mdp_state_reduction_for_stochastic_ltl_planning.md)
- [\[ICLR 2026\] One Model for All Tasks: Leveraging Efficient World Models in Multi-Task Planning](../../ICLR2026/reinforcement_learning/one_model_for_all_tasks_leveraging_efficient_world_models_in_multi-task_planning.md)
- [\[AAAI 2026\] DeepProofLog: Efficient Proving in Deep Stochastic Logic Programs](deepprooflog_efficient_proving_in_deep_stochastic_logic_programs.md)

</div>

<!-- RELATED:END -->
