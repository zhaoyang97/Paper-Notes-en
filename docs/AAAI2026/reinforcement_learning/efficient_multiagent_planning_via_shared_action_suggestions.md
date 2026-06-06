---
title: >-
  [Paper Note] Efficient Multiagent Planning via Shared Action Suggestions
description: >-
  [AAAI 2026][Reinforcement Learning][Dec-POMDP] This paper proposes the MCAS algorithm, which communicates only suggested joint actions (rather than shared observations) within a Dec-POMDP…
tags:
  - "AAAI 2026"
  - "Reinforcement Learning"
  - "Dec-POMDP"
  - "multiagent planning"
  - "action suggestion communication"
  - "belief inference"
  - "joint belief estimation"
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

# Efficient Multiagent Planning via Shared Action Suggestions

**Conference**: AAAI 2026
**arXiv**: [2412.11430](https://arxiv.org/abs/2412.11430)  
**Code**: [github.com/sisl/MCAS](https://github.com/sisl/MCAS)  
**Area**: Reinforcement Learning
**Keywords**: Dec-POMDP, multiagent planning, action suggestion communication, belief inference, joint belief estimation

## TL;DR

This paper proposes the MCAS algorithm, which communicates only suggested joint actions (rather than shared observations) within a Dec-POMDP, uses action information to infer other agents' belief states, achieves coordination performance close to centralized methods, and substantially reduces computational complexity.

## Background & Motivation

### State of the Field
Multiagent systems are critical in complex engineering projects, emergency response, and similar domains. The Dec-POMDP (Decentralized Partially Observable Markov Decision Process) is the standard framework for multiagent decision-making under uncertainty, yet its finite-horizon complexity is NEXP-complete, rendering it practically intractable.

### Limitations of Prior Work
- **Communication-free Dec-POMDP**: Agents must independently reason about the environment and other agents' behavior, resulting in prohibitive complexity.
- **Full communication (MPOMDP)**: Sharing all observations and actions reduces the problem to PSPACE-complete, but complete information sharing is often infeasible in practice.
- **Dec-POMDP-Com with communication**: When communication is neither lossless nor free, the finite-horizon problem remains NEXP-complete.

### Core Motivation
Inspired by human collaboration: humans naturally coordinate through action suggestions—for example, "Let's go to that restaurant" implicitly encodes judgments about whether the restaurant is open, appropriate, and affordable, packaging complex reasoning into a simple action proposal. This paper formalizes that intuition: **by sharing only suggested joint actions, agents can infer one another's beliefs without sharing observations, achieving effective coordination**.

## Method

### Overall Architecture

The MCAS (Multiagent Control via Action Suggestions) algorithm proceeds in three stages:
1. Offline: solve $n+1$ MPOMDP policies (one per agent plus one joint policy).
2. Online: agents communicate by sharing suggested joint actions.
3. Use action suggestions to prune the belief space, estimate the joint belief, and select optimal actions.

### Key Designs

#### 1. **Belief Subspace Inference**

Core insight: an action suggestion encodes the information that the suggester's belief lies within a specific subspace of the belief space.

If agent $i$ receives a suggested action $a_s$ from agent $j$ using policy $\pi^j$, then:

$$\mathcal{B}^j_{a_s} = \{b \mid \pi^j(b) = a_s, \forall b \in \mathcal{B}^j\}$$

Under an alpha-vector policy, this corresponds to the belief subspace dominated by a specific alpha vector:

$$\mathcal{B}^j_{a_s} = \{\mathbf{b} \mid (\boldsymbol{\alpha}_i - \boldsymbol{\alpha}_j) \cdot \mathbf{b} \geq 0, \forall \boldsymbol{\alpha}_i \in \Gamma_{a_s}, \forall \boldsymbol{\alpha}_j \in \Gamma\}$$

**Design Motivation**: Assuming agents are cooperative and optimal, an action suggestion directly reflects the agent's belief about the environment.

#### 2. **Belief Pruning**

After each time step, the reachable belief set that agent $i$ maintains for agent $j$ grows by a factor of $|\mathcal{O}^j| \cdot \prod_{j \neq i} |\mathcal{A}^j|$, expanding exponentially over time. MCAS manages this growth through two mechanisms:

**Action consistency pruning**: Beliefs inconsistent with the received action suggestion are removed:
$$\hat{\mathcal{B}}^{i,j}_t \leftarrow \{b \in \hat{\mathcal{B}}^{i,j}_t \mid \hat{\pi}^{i,j}(b) = \sigma^{j,i}\}$$

**Similarity merging**: Exploiting the Lipschitz condition on POMDP value functions established by Lee et al., when two beliefs have $L_1$ distance less than $\delta$, their value functions differ by at most $\frac{\|R\|_\infty}{1-\gamma}\delta$, so nearby beliefs can be safely merged.

A maximum belief set size of $\bar{B}_{max} = 200$ is enforced by iteratively removing the lower-weight belief from the closest pair.

#### 3. **Joint Belief Estimation**

Two methods for joint belief estimation are proposed:

**Exact reconstruction** (theoretically sound but computationally expensive): The joint belief is updated directly using inferred observations and actions:
$$\hat{\tilde{b}}^i_t(s') \propto O(\hat{\mathbf{o}} \mid \hat{\mathbf{a}}, s') \sum_s T(s' \mid s, \hat{\mathbf{a}}) \hat{\tilde{b}}^i_{t^-}(s)$$

Memory requirements scale as $\prod_{j \neq i} |\hat{\mathcal{B}}^{i,j}|$ joint belief combinations, growing exponentially in the number of agents.

**Conflation approximation** (adopted in practice): Probability distributions are combined using the conflation method of Hill:
$$\hat{\tilde{b}}^i(s) = \frac{b^i(s) \prod_{j \neq i} \hat{b}^{i,j}(s)}{\sum_{s' \in \mathcal{S}} b^i(s') \prod_{j \neq i} \hat{b}^{i,j}(s')}$$

Conflation minimizes Shannon information loss during merging, automatically favors more confident beliefs, and requires no manually specified weights.

### Loss & Training

MCAS does not involve conventional training losses; it follows an offline-online planning pipeline:

**Offline phase**: The SARSOP solver generates MPOMDP policies (represented as alpha vectors) for each agent, along with a joint policy assuming centralized execution.

**Online phase**:
- MCAS variant: shares actions as messages; pruning is based on action consistency.
- MCAS-α variant: shares alpha-vector indices (providing finer-grained belief subspace information) for more effective pruning.
- A count-based heuristic selects the joint belief: visit counts for each estimated belief are maintained, and the most frequently visited joint belief is selected.
- The coordinating agent uses the estimated joint belief together with the joint policy to select and broadcast the final action.

## Key Experimental Results

### Main Results

Evaluated on multiple standard Dec-POMDP benchmarks, with 2,000 runs per scenario and 95% confidence intervals:

| Problem | Agents | MPOMDP (upper bound) | MCAS-α | MCAS | MPOMDP-I | Independent | Dec-POMDP Optimal |
|---------|--------|----------------------|--------|------|----------|-------------|-------------------|
| Dec-Tiger (2) | 2 | 59.5±0.9 | 58.5±0.9 | 58.5±0.8 | 34.3±1.7 | -68.1±3.5 | 13.5 |
| Dec-Tiger (3) | 3 | 108.5±1.0 | 108.5±1.0 | 108.5±1.0 | 82.1±1.5 | -95.5±4.1 | — |
| Broadcast (2) | 2 | 9.4±0.0 | 9.4±0.0 | 9.4±0.0 | 9.4±0.0 | 7.6±0.1 | 9.3 |
| Meet 3×3 (2) | 2 | 5.8±0.1 | 5.8±0.1 | 5.7±0.1 | 3.6±0.1 | 3.7±0.1 | 5.8 |
| Box Push (2) | 2 | 222.9±2.2 | 223.4±2.1 | 223.0±2.2 | 199.6±2.6 | 163.6±3.4 | 224.4 |
| Mars Rover-UI (2) | 2 | 23.9±0.1 | 23.9±0.1 | 19.8±0.2 | 16.4±0.2 | 15.3±0.2 | — |
| Mars Rover-UI (3) | 3 | 25.2±0.1 | 25.2±0.1 | 23.8±0.2 | 19.7±0.1 | 16.6±0.1 | — |

### Ablation Study

Belief set size analysis (maximum estimated belief set size for MCAS vs. MCAS-α):

| Problem | MCAS-α | MCAS |
|---------|--------|------|
| Meet 3×3 | 1.0±0.0 | 2.5±0.0 |
| Meet 27 (UI,WP) | 1.5±0.0 | 16.8±1.6 |
| Box Push (SO) | 4.8±0.1 | 192.1±1.2 |
| Wireless | 1.0±0.0 | 18.0±0.9 |
| Mars Rover (UI) | 1.0±0.0 | 2.0±0.0 |
| Mars Rover (55G,UI) | 1.0±0.0 | 3.0±0.0 |

### Key Findings

1. **MCAS-α nearly matches MPOMDP-C**: Precise subspace pruning via alpha-vector indices recovers near-centralized performance across all problems.
2. **Action-level pruning is also effective**: Using only action information, MCAS maintains good joint belief estimates with minimal performance loss.
3. **Belief pruning is efficient**: The hard cap is triggered in 87.8% of Box Push-SO runs, but in only 3.2% of Meet 27 runs; most problems maintain very small belief sets.
4. **Single-agent observations suffice in some problems**: In Broadcast and Wireless, MPOMDP-I performs comparably to MPOMDP, indicating that one agent's observations are sufficient for good decisions.
5. **Conflation is an effective belief fusion method**: MPOMDP-C performs comparably to MPOMDP across all problems.

## Highlights & Insights

- **Communication mechanism inspired by human collaboration**: Action suggestions are the most natural form of human cooperative communication; formalizing this as a Dec-POMDP communication strategy is elegant.
- **Inverse inference from actions to beliefs**: By assuming optimal behavior, action suggestions are converted into constraints on the belief space—a broadly applicable insight.
- **Substantial reduction in computational complexity**: Complexity is reduced from NEXP-complete to solving $n+1$ MPOMDPs (each PSPACE-complete), and experiments run on a standard laptop.
- **Contrast between MCAS-α and MCAS reveals the value of information granularity**: Sharing alpha-vector indices provides finer-grained belief subspace information than sharing actions, with significant differences on complex problems.

## Limitations & Future Work

1. **Dependence on offline policy generation**: MPOMDP solving still scales exponentially in the number of agents, limiting large-scale application.
2. **Strong assumptions**: All agents are assumed to share the same surrogate policy, and communication is assumed to be noiseless and cost-free—conditions that are difficult to satisfy in practice.
3. **Applicable only to offline policies**: Integration with online solvers (e.g., AdaOPS, BetaZero) has not yet been explored.
4. **Limitations of the belief selection heuristic**: The count-based selection method is suboptimal in some problems; regret minimization strategies warrant future investigation.
5. **Theoretical analysis is lacking**: Convergence guarantees for belief estimation and performance bounds relative to centralized methods are absent.

## Related Work & Insights

- Complements the occupancy state approach (Dibangoye 2016): the latter reformulates Dec-POMDP as a continuous-state MDP, whereas this work reduces complexity through communication.
- TCAS/ACAS X aviation collision avoidance systems are a real-world application of action suggestion communication, coordinating via directives such as "do not descend."
- Provides a natural communication framework for human-agent hybrid teams that is more consistent with human interaction norms than sharing observations or beliefs.

## Rating

- **Novelty**: ⭐⭐⭐⭐ (The idea of action suggestion communication has prior precedents, but the formalization of belief inference and pruning is a new contribution)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (Multiple variants across 7 benchmark problems, reasonably comprehensive, though large-scale real-world validation is lacking)
- **Writing Quality**: ⭐⭐⭐⭐⭐ (Notation is clearly defined, algorithms are described in detail, and the restaurant metaphor is used consistently throughout)
- **Value**: ⭐⭐⭐⭐ (Provides a valuable framework for scalable multiagent planning and human-agent collaboration)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Object-Centric Latent Action Learning](object-centric_latent_action_learning.md)
- [\[AAAI 2026\] Good-for-MDP State Reduction for Stochastic LTL Planning](good-for-mdp_state_reduction_for_stochastic_ltl_planning.md)
- [\[AAAI 2026\] Partial Action Replacement: Tackling Distribution Shift in Offline MARL](partial_action_replacement_tackling_distribution_shift_in_offline_marl.md)
- [\[ICLR 2026\] One Model for All Tasks: Leveraging Efficient World Models in Multi-Task Planning](../../ICLR2026/reinforcement_learning/one_model_for_all_tasks_leveraging_efficient_world_models_in_multi-task_planning.md)
- [\[AAAI 2026\] Intention-Guided Cognitive Reasoning for Egocentric Long-Term Action Anticipation](intention-guided_cognitive_reasoning_for_egocentric_long-term_action_anticipatio.md)

</div>

<!-- RELATED:END -->
