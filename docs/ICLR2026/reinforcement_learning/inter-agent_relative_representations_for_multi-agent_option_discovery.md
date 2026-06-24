---
title: >-
  [Paper Note] Inter-Agent Relative Representations for Multi-Agent Option Discovery
description: >-
  [ICLR 2026][Reinforcement Learning][Multi-Agent RL] This paper proposes a **relative representation focused on inter-agent relationships** for joint state abstraction. It first estimates a "Fermat state" that minimizes team-wide alignment costs, then utilizes dimension-wise temporal distances from each agent to this state as a new representation. Upon this representation, Graph Laplacian eigenoption decomposition is performed to discover a smaller number of highly coordinated…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "Multi-Agent RL"
  - "option discovery"
  - "joint state abstraction"
  - "Fermat states"
  - "Graph Laplacian eigenoptions"
  - "temporal distance"
date: 2026-05-08
content_hash: 356299489f2e69c6
---

# Inter-Agent Relative Representations for Multi-Agent Option Discovery

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=Fte7TOqnQp](https://openreview.net/forum?id=Fte7TOqnQp)  
**Code**: To be confirmed  
**Area**: Reinforcement Learning / Multi-Agent Option Discovery  
**Keywords**: Multi-Agent RL, option discovery, joint state abstraction, Fermat states, Graph Laplacian eigenoptions, temporal distance  

## TL;DR
This paper proposes a **relative representation focused on inter-agent relationships** for joint state abstraction. It first estimates a "Fermat state" that minimizes team-wide alignment costs, then utilizes dimension-wise temporal distances from each agent to this state as a new representation. Upon this representation, Graph Laplacian eigenoption decomposition is performed to discover a smaller number of highly coordinated multi-agent joint options.

## Background & Motivation
**Background**: In single-agent RL, the option framework (Sutton et al., 1999) utilizes temporally extended actions to act as "shortcuts" between distant regions in the state space, significantly improving exploration and planning. Eigenoptions based on Graph Laplacian eigen-decomposition (Machado et al., 2017) are popular due to their task-agnostic nature and exploration guarantees.

**Limitations of Prior Work**: Applying this framework to multi-agent scenarios faces two critical issues. First, the **joint state space grows exponentially with the number of agents**, and since the number of eigenoptions discovered is roughly twice the number of states, the number of options explodes. Second, current neural approximators for Laplacian eigenvectors are only stable when estimating **a few** eigenvectors, often missing truly useful options that promote exploration across different time scales.

**Key Challenge**: Existing multi-agent option discovery methods (e.g., those based on Kronecker products, Chen et al., 2022) tend to decompose joint behaviors into **loosely coupled or fully independent** individual behaviors to bypass exponential explosion—coordination only occurs at the "option selection" level, while the option policies themselves do not express collaboration. This sacrifices coordination for computability, leaving the discovery of strongly coordinated behaviors unresolved.

**Goal**: To compress the joint state space while **focusing the discovery process on how relationships between agents change**, thereby controlling the number of options while enabling the emergence of strongly coordinated joint behaviors.

**Key Insight**: The authors bet on an inductive bias—**in the absence of explicit targets, "synchronization/alignment" between agent states is a natural foundation for coordination** (e.g., passing and positioning in team sports can be understood as multi-agent state synchronization). Accordingly, they redefine the state representation using a "team-wide maximum alignment point," making eigenvectors naturally point toward inter-agent relationships rather than absolute positions.

## Method

### Overall Architecture
The method focuses on "how to represent the joint state," a component overlooked by previous work: **before** performing Graph Laplacian eigen-decomposition, the original joint state is embedded into an **inter-agent relative representation**. The pipeline consists of three steps—first, use a Fermat encoder to estimate the team alignment point and calculate dimension-wise temporal distance representations; second, run eigenoption discovery on this relative representation to obtain joint options; finally, integrate these joint options into the MacDec-POMDP framework to support decentralized execution.

```mermaid
flowchart LR
    A[Joint state s_t<br/>Factorized into agent states s_t^i] --> B[Fermat Encoder φ<br/>Estimate Max Alignment Point φ(s_t)]
    B --> C[Temporal Distance d_θ<br/>Output n-distance per feature dim]
    C --> D[CMI Decoupling<br/>Discriminator D_ψ prevents collapse]
    D --> E[Multi-dim Relative Representation<br/>Reconstruct State Transition Graph]
    E --> F[ALLO Graph Laplacian<br/>Eigenvector μ]
    F --> G[Eigenvector as Intrinsic Reward<br/>Train Joint Option Policy]
    G --> H[MacDec-POMDP<br/>Vote/Sync/Terminate]
```

### Key Designs
**1. Fermat State: Re-anchoring the joint state using the "minimum team alignment cost point."** Drawing from the geometric concept of Fermat n-distance (the point minimizing the sum of distances to a set of points), the authors define the "spreadingness" of a joint state as the sum of distances from each agent's state to a virtual alignment point. For an individual state space $(S^*, d)$ and $N$ agents, the Fermat inter-agent distance is $d_F(s^1,\dots,s^N)=\min_{s\in S^*}\sum_{i=1}^N d(s^i, s)$. Since this minimization is intractable in large or continuous spaces, a parameterized **Fermat Encoder** $\phi: S\to S^*$ is used to approximate this alignment point. The training objective is to minimize the sum of squared distances to each agent state: $L_F(\phi,d)=\mathbb{E}_{\tau\sim\rho_\pi}\big[\tfrac{1}{N}\sum_{i=1}^N d(s_t^i,\phi(s_t))^2\big]$. After re-expressing states centered on this Fermat point, eigenvectors become **highly sensitive** to changes in joint states (re-centring effect), a property lacking when decomposing original joint states.

**2. Temporal distance as the underlying metric instead of Euclidean distance.** The choice of $d$ in the above formula is crucial. The authors select **temporal/successor distances** (Myers et al., 2024), representing the expected number of steps required by a policy between two states, as it is invariant to feature semantics and respects environmental dynamics (e.g., avoiding obstacles). Since temporal distance is a **quasi-metric** lacking symmetry (uphill is slower than downhill), the authors fix the Fermat state as the second input to the function, obtaining a directed metric interpreted as the "expected steps for the team to achieve full alignment." The Fermat encoder $\phi$ and distance approximator $d_\theta$ are trained jointly, using stop-gradients on $d_\theta$ within the $\phi$ objective.

**3. Multi-dimensional dimension-wise n-distance + MI decoupling to prevent representation collapse.** Compressing the joint state into a **single scalar** distance would erase information about "which dimension two agents are misaligned in," severely distorting the transition graph topology and limiting the diversity of discoverable alignment behaviors. Instead, the authors let the distance module output a distance for **each dimension** of the individual state, $d_F:S^*\times S^*\to\mathbb{R}^F$ ($F=\dim(S^*)$), then use a linear projection layer to reconstruct the total distance. To prevent unconstrained dimension-wise decomposition from collapsing (all dimensions outputting the same values or ignoring dimensions), they introduce a **Conditional Mutual Information (CMI) constraint**: the information one feature distance $Z_f^{i,j}$ carries about other features $S_{-f}^{i,j}$ should not exceed the inherent correlation of the features themselves, i.e., minimize $I(S_{-f}^{i,j}; Z_f^{i,j}\mid S_f^{i,j})$. Implementation utilizes a discriminator network $D_\psi$ (Dunion et al., 2023) trained to distinguish real triplets from fake ones with shuffled $s_{-f}$, acting as a decoupling penalty.

**4. Integrating Joint Options into MacDec-POMDP: Voting, Sync, and Consensus Termination.** After obtaining the relative representation, the authors follow the ROD loop and ALLO approximator for eigen-decomposition, using eigenvectors as intrinsic rewards $r_e(s,s')=e(s')-e(s)$ to train option policies. To execute joint options correctly under decentralization, they extend MacDec-POMDP with two assumptions: an **inter-agent information sharing mechanism** and a **synchronization mechanism** to ensure minimum participation. A joint option is defined as a tuple $W=\langle I_W,\pi_W,\beta_W\rangle$ over joint macro-histories. An agent selecting an option is treated as a "vote"; joint options require $N$ votes (team consensus) to initiate, while local options require only 1. Primitive actions are treated as instantly terminating local options. Termination requires all agents to select the termination action, with a hard 50-step limit.

## Key Experimental Results

### Experimental Setup
Two multi-agent domains: **Level-Based Foraging (LBF)** and **Overcooked** (re-implemented in JAX). LBF uses `15x15-4p-3f` / `15x15-4p-5f` configurations where each apple level equals the sum of team levels (forced cooperation). Overcooked uses `Forced Coordination` and `Counter Circuit`. N-distance encoders and ALLO are trained on 500k random policy transitions. The first 10 eigenvectors (20 options) are estimated for LBF, and the first 20 eigenvectors (40 options) for Overcooked. Option policies are trained using IQL for 1M steps (5-10% of total training).

### Main Results: Utility of Joint Options (H1) + Comparison with Other Frameworks (H2)
IQM with 95% confidence intervals across 10 seeds are reported.

| Comparison Dimension | Conclusion |
|---|---|
| IQL+IARO vs. IQL (No Options) | Consistent improvement: Higher apple collection ratio in LBF, more deliveries per round in Overcooked. |
| IQL+IARO vs. MAPPO / IPPO / IQL / VDN | Leading in most scenarios; only Forced Coordination is inferior to VDN (where VDN is inherently strong). |
| IQL+IARO vs. IQL+Kron (Chen 2022 Kronecker) | IARO is superior; Kron even **degrades** performance in LBF. |
| IQL+IARO vs. IQL+RJS (Original Joint State) | IARO is superior; RJS's first eigenvectors push agents to the edges of the state space, hurting apple collection. |

The authors note that IARO **does not rely on centralization (MAPPO) or value decomposition (VDN)** to foster collaboration, although it converges slightly slower in early training (a known challenge when training with options under global initiation sets).

### Ablation Study: Multi-dimensional vs. Scalar n-distance (H3)

| Representation | LBF (Simple State) | Overcooked (Multi-semantic Features) |
|---|---|---|
| IARO-Scalar | Comparable to multi-dim | Weaker |
| IARO-MultiDim | Comparable to scalar | **Superior** |

Conclusion: As state feature semantics become richer, dimension-wise decoupled representations allow agents to align on **specific subsets of dimensions**, resulting in richer collaborative behaviors.

### Analysis of Option Quantity
Scanning different numbers of options in complex scenarios (15 seeds, 64 evaluation episodes): LBF achieves maximum gains with only **2 options**, while Overcooked requires the first **10 options** for significant improvement—consistent with eigenoption theory where leading eigenvectors connect distant nodes in the graph to provide strong exploration behaviors. Excessive options lead to saturated gains or training instability.

### Key Findings
- After re-centring around the Fermat state, eigenvectors are highly sensitive to joint state changes, and **the same set of leading eigenvectors induces consistent alignment patterns across teams of 3 or 4 agents**, indicating transferability of behavioral patterns across team sizes.
- The first eigenvector aligns along one coordinate axis while its negative aligns along another; subsequent eigenvectors facilitate simultaneous alignment across multiple axes and more complex synchronization patterns.
- Visualization in a 15x15 three-agent grid shows that relative representations (right) better reflect changes in "relative positions between agents" compared to individual states (left) or raw joint states (middle), rather than pushing agents to the edges of the state space.
- In Overcooked, IQL often gets stuck in sub-optimal solutions, whereas joint options allow agents to systematically sweep through various coordination modes to search for better strategies, explaining why IARO's improvement is particularly pronounced in this domain.

## Highlights & Insights
- **State representation as a controllable knob for option discovery**: The authors astutely point out that eigenoptions are entirely dominated by the state representation. Instead of altering the decomposition algorithm, they replace the representation with one centered on inter-agent relationships, solving both option explosion and the lack of coordination.
- **The Fermat state as an elegant unified anchor**: Using the "minimum team alignment cost point" to compress $N$ agent relationships into a learnable center preserves critical information about "who is misaligned with whom and on which dimension" while reducing dimensionality.
- **Dimension-wise decomposition + CMI decoupling** upgrades "alignment" from a scalar to a composable dictionary of behaviors, allowing the high-level policy to choose specific dimensions for coordination, resulting in significantly stronger behavioral expressiveness.
- Utilizing temporal distance instead of Euclidean distance makes the metric invariant to feature semantics and aligned with dynamics, a key step in migrating single-agent skill discovery (like METRA) to multi-agent settings.

## Limitations & Future Work
- **Strong Team Consensus Assumption**: Joint options must be initiated by the whole team, limiting the range of expressible behaviors. The authors acknowledge the discovery method supports arbitrary agent subsets, but only team-level options were implemented; subset options are left for future work.
- **Dependency on Info Sharing/Sync**: The method assumes inter-agent information sharing and synchronization (implicit in Overcooked, explicit in LBF via relative distances and "near apple" flags) rather than general communication protocols. Plan include replacing direct sharing with communication protocols.
- **Slow Early Training Convergence**: Training with options under global initiation sets slows early convergence, a classic issue in option learning.
- **Limited Evaluation Scales**: Verified only on LBF and Overcooked with up to 4 agents; scalability to larger teams and more complex domains remains to be tested.

## Related Work & Insights
- **State Similarity Metrics**: Bisimulation metrics (Ferns et al., 2004) measure similarity based on reward/transition differences but are prone to representation collapse under sparse rewards. Temporal/successor distances are representation-invariant and dynamic-aware. Most similar to METRA (Park et al., 2024), but the current paper uses temporal distance to approximate the latent representation for **joint option** discovery.
- **State Representations in MARL**: Previous work often used GNNs to aggregate local observations into global representations; Utke et al. (2025) emphasized relative information but relied on manual spatial relationship edge features—this paper **automatically learns** inter-agent relationships without restricting feature semantics.
- **Temporally Extended Actions in MARL**: Asynchronous schemes (Amato et al., 2019, etc.) mostly use local options where coordination occurs only at the selection layer. Chen et al. (2022) used Kronecker products to construct joint behaviors but still synchronized independent behaviors, failing to capture strong inter-agent dependencies—Ours specifically targets this "discovery of strongly coordinated behaviors."

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — Re-representing the joint state as dimension-wise temporal distances centered on a Fermat alignment point for eigenoption decomposition is a unique and self-consistent entry point, stitching together state representation, temporal distance, MI decoupling, and the option framework.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three hypotheses (H1/H2/H3) are verified, covering no options, other option frameworks, scalar vs. multi-dim, and option quantity; however, only 2 domains and 4 agents were tested.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation and inductive biases (team sport analogy) are clear, derivations (Fermat distance, CMI upper bound) are rigorous, and visualizations of eigenvectors and option roll-outs are provided.
- **Value**: ⭐⭐⭐⭐ — Provides a new path for discovering coordinated multi-agent options without relying on centralization or value decomposition. The Fermat state + decoupled representation idea offers good transferable insights for MARL representation learning and skill discovery.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Multi-Agent Guided Policy Optimization](multi-agent_guided_policy_optimization.md)
- [\[ICLR 2026\] Correlated Policy Optimization in Multi-Agent Subteams](correlated_policy_optimization_in_multi-agent_subteams.md)
- [\[ICLR 2026\] Retaining Suboptimal Actions to Follow Shifting Optima in Multi-Agent RL](retaining_suboptimal_actions_to_follow_shifting_optima_in_multi-agent_reinforcem.md)
- [\[ICLR 2026\] Ego-Foresight: Self-supervised Learning of Agent-Aware Representations for Improved RL](ego-foresight_self-supervised_learning_of_agent-aware_representations_for_improv.md)
- [\[ICLR 2026\] When Is Diversity Rewarded in Cooperative Multi-Agent Learning?](when_is_diversity_rewarded_in_cooperative_multi-agent_learning.md)

</div>

<!-- RELATED:END -->
