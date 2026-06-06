---
title: >-
  [Paper Note] Structure-Induced Information for Rerooting Levin Tree Search
description: >-
  [ICML 2026][Levin Tree Search] Within the $\sqrt{\mathrm{lts}}$ framework, the authors propose three "rerooters"—global Leiden clustering, local heuristic cost-to-go…
tags:
  - "ICML 2026"
  - "Levin Tree Search"
  - "rerooting"
  - "Leiden clustering"
  - "heuristics"
  - "subtask decomposition"
date: 2026-05-08
content_hash: 1789f01cfb159164
---

# Structure-Induced Information for Rerooting Levin Tree Search

**Conference**: ICML 2026  
**arXiv**: [2605.30664](https://arxiv.org/abs/2605.30664)  
**Code**: Not yet public  
**Area**: Reinforcement Learning / Learning-guided Search / Planning  
**Keywords**: Levin Tree Search, rerooting, Leiden clustering, heuristics, subtask decomposition

## TL;DR
Within the $\sqrt{\mathrm{lts}}$ framework, the authors propose three "rerooters"—global Leiden clustering, local heuristic cost-to-go, and an additive mixture of both—to automatically distribute search effort to implicit subtasks based on state space structure and goal distance. This approach avoids the computationally expensive explicit subgoal generation models used in HIPS-$\varepsilon$ / SGPS, achieving SOTA in online training sample efficiency and test expansion counts across complex domains such as BoulderDash and CraftWorld.

## Background & Motivation

**Background**: Policy tree search utilizes a learned policy $\pi$ to concentrate probability mass on promising branches. Levin Tree Search (LTS, 2018) uses $\varphi_{\mathrm{LTS}}(n)=\tfrac{d(n)+1}{\pi(n)}$ as node cost, providing a strict upper bound: "at most $(d(n^*)+1)/\pi(n^*)$ nodes are expanded before finding the first solution node." PHS* (2021) incorporates a heuristic $h$ into this bound and learns a policy to minimize it.

**Limitations of Prior Work**: LTS/PHS* struggle in complex domains (e.g., BoulderDash, CraftWorld) because they lack a "subgoal decomposition" mechanism. HIPS-$\varepsilon$ (2024) and SGPS (2025) decompose problems and extend search radius through explicit subgoal generation (using high-capacity generative models like VQ-VAE). While effective, the computational overhead of invoking subgoal networks scales poorly with domain complexity; on BoulderDash at 30% difficulty, PHS*($\pi^{\mathrm{SG}}$) solves 11 instances, but fails entirely due to timeouts at 40%.

**Key Challenge**: Scaling to complex domains requires subtask decomposition; however, explicit subgoal reconstruction implies high-capacity generative models, which are expensive for both training and inference. A significant trade-off exists between computational cost and performance.

**Goal**: Based on the "implicit subtask" mechanism of $\sqrt{\mathrm{lts}}$ (Orseau et al. 2024), this work addresses an open question left by Orseau: how to automatically derive rerooting weights $w_t$ from the search tree structure itself **without** invoking a separate subgoal network?

**Key Insight**: $\sqrt{\mathrm{lts}}$ implicitly initiates an LTS subsearch at each node $n_t$, where the node cost is modified to $c^r(n)=\min_{n_t\prec n}\tfrac{1}{w_t}c_t^r(n)$, and $w_t$ determines the time share allocated to each subsearch. Orseau et al. proved that if the rerooter "correctly selects" subtask boundaries, $\sqrt{\mathrm{lts}}$ can be exponentially better than LTS. The authors observe that rerooting weights can be derived from **global state space connectivity** (using Leiden clustering to partition the state space into "rooms") or from **local heuristic cost-to-go** $h(n_t)$, as the two are naturally complementary.

**Core Idea**: Utilize three lightweight structural signals—(L) Leiden clustering, (H) softmax heuristics, and (LH) an additive mixture—to dynamically generate $w_t$. This replaces "explicit subgoal generation" with "implicit structural awareness." The authors also provide proofs showing that additive rerooters maintain the subtask decomposition bounds of $\sqrt{\mathrm{lts}}$.

## Method

### Overall Architecture
The search process follows the BFS framework of $\sqrt{\mathrm{lts}}$: the cost of node $n$ is
$$c^r(n)=\min_{n_t\prec n}\tfrac{1}{w_t}c_t^r(n),\quad c_t^r(n)=\sum_{n_t\prec n'\preceq n}\tfrac{1}{\pi(n'\mid n_t)}.$$

The only component redefined by the authors is "how to calculate the weight $w_t$ for ancestor nodes $n_t$." Three rerooters ($\sqrt{\mathrm{lts}}$-L / -H / -LH) provide their respective $w_t$ formulas, while the rest of the search (priority queue, policy/heuristic neural networks, bootstrap training loop) remains unchanged.

The training employs Bootstrap (Arfaee et al. 2011): policy and heuristic networks are randomly initialized, the training set is scanned with the current budget, and solved trajectories are used to update the networks. If no new problems are solved in a pass, the expansion budget is doubled for the next scan. Training stops once 95% of the validation set is solved.

### Key Designs

1.  **$\sqrt{\mathrm{lts}}$-L: Global Structural Rerooter based on Leiden Clustering**:
    - **Function**: Partitions the state space into "rooms" (clusters) based on graph clustering, making $w_t$ reflect "how much unexplored space remains in the current cluster," automatically pushing search effort toward new clusters.
    - **Mechanism**: Incrementally constructs an induced subgraph $G_0$ of the state space during search (nodes = visited states, edges = utilized transitions). Run the Leiden algorithm at search steps $t=\gamma^i$ (geometric scheduling, $\gamma>1+1/\epsilon$) to obtain hierarchical clustering $(G_1,\dots,G_N)$. Assign a color $c$ (cluster ID) to each tree node $n$ from the $k$-th level $G_k$. Let $M_{\tau,c}$ be the number of nodes with color $c$ after the $\tau$-th coloring, and $\delta_{\tau,c}$ be the number of additional nodes of the same color expanded since then. Then $w_t=\tfrac{1}{M_{\tau,c_t}+\delta_{\tau,c_t}}$. Smaller clusters $\to$ higher $w_t \to$ lower cost in that direction $\to$ prioritized expansion. Expanding more nodes of the same color increases the denominator and decays $w_t$, shifting focus to new clusters.
    - **Design Motivation**: In human planning, "entering a new room" or "obtaining a key" are typically key subgoal boundaries, which correspond to cluster transitions. Leiden uses modularity to find this structure automatically without external subgoal labels. Geometric scheduling, avoiding retracing weights of expanded nodes, and color inheritance for child nodes amortize clustering costs to $O(bN\log N + DN)$, keeping it in the same complexity class as BFS (Theorem 3.1).

2.  **$\sqrt{\mathrm{lts}}$-H: Local Rerooter based on Heuristic Cost-to-go**:
    - **Function**: Uses the heuristic $h(n_t)$ to directly determine $w_t$, rewarding ancestor nodes that "appear closer to the goal."
    - **Mechanism**: $w_1=1$, $w_t=\exp\!\left(-\alpha\,\tfrac{h(n_t)}{h(n_1)}\right)$. Normalizing by $h(n_1)$ makes the weights invariant to multiplicative scaling of heuristic values. The exponential ensures strictly positive weights for all nodes. $\alpha$ is the inverse temperature—small values are conservative, while large values concentrate weight on low-heuristic nodes; for a set of candidate nodes $I$, $\tfrac{w_t}{\sum_{i\in I}w_i}$ is exactly the softmax over $-\alpha h/h(n_1)$.
    - **Design Motivation**: Global clustering does not distinguish which of two "structurally symmetric" subtrees is closer to the goal, whereas heuristics naturally provide this information. The softmax form allows the rerooter in $\sqrt{\mathrm{lts}}$ to smoothly distribute search time to promising ancestors rather than making a "hard pick," making it more robust to heuristic noise.

3.  **$\sqrt{\mathrm{lts}}$-LH: Additive Mixture with Theoretical Guarantees**:
    - **Function**: $w_t=u_a\,\tfrac{1}{M_{\tau,c_t}+\delta_{\tau,c_t}}+u_b\,\exp\!\left(-\alpha\,\tfrac{h(n_t)}{h(n_1)}\right)$ (default $u_a=u_b=1$). Uses global structure for coarse-grained time allocation and heuristics to refine priorities within clusters.
    - **Mechanism**: The authors prove in Theorem 3.2 that any additive rerooter $w=u_a w_a+u_b w_b$ maintains the subtask decomposition bound $T\le 1+(C+1)\min_D\max_i\min\{\tfrac{w_{a,<T}}{w_{a,T_i}},\tfrac{w_{b,<T}}{w_{b,T_i}}\}c^r_{T_i}(n_{T_{i+1}})$, provided the cumulative weight ratio remains bounded by $1/C\le \tfrac{u_a w_{a,<T}}{u_b w_{b,<T}}\le C$. This means the search can fall back to whichever signal is more informative in a given region.
    - **Design Motivation**: Pure heuristic rerooters can misguide search when heuristics fail; pure clustering rerooters are indifferent to which "room" is closer to the end. The additive mixture use the $\min$ term to naturally follow the tighter bound, while $u_a, u_b$ control $C$ to mitigate the risk of one rerooter "consuming" the other's time share.

### Loss & Training
Policy $\pi$ and heuristic $h$ are neural networks initialized from scratch and trained via Bootstrap. Solved trajectories are used as supervised samples to update the networks. If a round yields no new solutions, the expansion budget is doubled. All experiments use $\ell(n)=1$ (cost = number of expansions). The training budget is $10^6$ seconds (~11.5 CPU-days). Key hyperparameters include Leiden frequency $\gamma$, clustering level $k$, $\alpha$, $u_a$, and $u_b$.

## Key Experimental Results

### Main Results
Evaluation across four domains comparing three $\sqrt{\mathrm{lts}}$ variants against LTS, SGPS variants, PHS*, and WA*, with a test budget of $5.12\times10^5$ expansions, averaged over 5 seeds.

| Domain | Algorithm | Solved | Expansions | Time (s) |
| :--- | :--- | :--- | :---: | :---: |
| BoulderDash | LTS | 10 | 195 451 | 119.86 |
| BoulderDash | PHS*($\pi^{\mathrm{SG}}$) | 100 | 359.86 | 2.70 |
| BoulderDash | **$\sqrt{\mathrm{lts}}$-H** | 100 | **92.37** | **0.60** |
| BoulderDash | **$\sqrt{\mathrm{lts}}$-LH** | 100 | 92.68 | **0.58** |
| CraftWorld | LTS | 100 | 306 224 | 373.44 |
| CraftWorld | PHS*($\pi^{\mathrm{SG}}$) | 100 | 1 413 | 8.67 |
| CraftWorld | **$\sqrt{\mathrm{lts}}$-LH** | 100 | **1 347.5** | **4.59** |
| Sokoban | PHS*($\pi^{\mathrm{SG}}$) | 1 000 | 1 630.6 | 1.56 |
| Sokoban | $\sqrt{\mathrm{lts}}$-LH | 1 000 | 1 736.0 | 1.10 |
| TSP (Gridworld) | PHS*($\pi^{\mathrm{SG}}$) | 100 | **46.31** | 0.49 |
| TSP (Gridworld) | $\sqrt{\mathrm{lts}}$-H | 100 | 55.44 | 0.37 |

### BoulderDash Difficulty Ramp (Online Training)
The wall fill rate was increased from 10% to 40%. The table shows the expansions/time required to solve 10,000 training problems.

| Difficulty | PHS*($\pi^{\mathrm{SG}}$) Exp / Time(h) | $\sqrt{\mathrm{lts}}$-H Exp / Time(h) | Notes |
| :--- | :---: | :---: | :--- |
| 10% | $3.07\times10^7$ / 10.63 | $1.77\times10^7$ / **5.00** | Slight lead |
| 20% | $3.00\times10^8$ / 137.12 | $1.99\times10^7$ / **5.99** | 15× compute advantage |
| 30% | $4.29\times10^8$ / 278.23 (11 solved) | $2.80\times10^7$ / **8.37** (9,996 solved) | Baseline collapsed |
| 40% | — (Timeout) | $3.85\times10^7$ / **11.94** (9,994 solved) | Only method to complete |

### Key Findings
- In high-difficulty domains like BoulderDash 30%/40%, PHS*($\pi^{\mathrm{SG}}$) (based on subgoal generation) fails (solving only 11 at 30%, timing out at 40%), while all $\sqrt{\mathrm{lts}}$ variants maintain a 99% solve rate. **This indicates that explicit subgoal reconstruction is the scalability bottleneck for SGPS**, which rerooting bypasses using implicit subtasks.
- $\sqrt{\mathrm{lts}}$-H (local heuristic) alone outperforms SGPS in BoulderDash. **This suggests that even without global structural signals, learned heuristics + softmax weights are sufficient for effective subtask decomposition**, contradicting the assumption that subgoals must be explicitly generated.
- $\sqrt{\mathrm{lts}}$-LH combines the robustness of -L and the speed of -H. In CraftWorld, it uses 1,347 expansions (5% fewer than PHS* and half the time), confirming the additive complementarity in Theorem 3.2.
- In Sokoban, where subgoals are less obvious and the problem is more combinatorial, rerooting gains are smaller (comparable to PHS*), suggesting that the benefits of "implicit structural awareness" correlate with the clusterability of the domain.

## Highlights & Insights
- **Transforming "subgoal generation" into "weight allocation"**: Instead of training VQ-VAEs to output subgoals, weights are derived from the search tree and existing heuristic signals. This eliminates the training and inference costs of a generative network, a significant engineering simplification.
- **Leiden Clustering + Geometric Scheduling + Color Inheritance**: These three techniques together reduce the $O(N^2)$ per-step clustering cost to $O(bN\log N + DN)$, making clustering-based rerooters practically viable. This "amortized dynamic graph clustering" pattern is transferable to any search algorithm requiring online community detection.
- **Softmax Heuristic Temperature $\alpha$ as a Practical Interface**: Previously, heuristics influenced costs in a "hard" manner. Here, inverse temperature allows for continuous interpolation between "trusting the heuristic" and "preserving exploration," providing a concise way to incorporate heuristic uncertainty into search.
- **Subtask Decomposition Bound for Additive Rerooters** (Theorem 3.2) is a theoretical highlight: it demonstrates that as long as the weight ratio $C$ is bounded, the mixed rerooter does not destroy the exponential advantage of $\sqrt{\mathrm{lts}}$. This provides a theoretical template for integrating more complementary rerooters (e.g., adversarial signals, model uncertainty).

## Limitations & Future Work
- The global rerooter depends on the incremental construction of the state space subgraph; Leiden clustering fails in settings where the state space is non-enumerable or the transition function is a black box. The paper does not propose "approximate clustering" or "latent space clustering" alternatives.
- The heuristic rerooter requires the heuristic to be at least weakly correlated with goal distance. In early stages of sparse-reward RL, -H may be misguided. The temperature $\alpha$ is manually tuned without an automatic annealing schedule.
- Domains with "room-key" structures like BoulderDash/CraftWorld are naturally suited for clustering signals; gains are marginal in combinatorial domains like Sokoban that lack clear spatial clustering.
- Experiments focused on uniform step costs $\ell(n)=1$. Whether $\sqrt{\mathrm{lts}}$ remains exponentially superior to LTS in real-world planning with non-uniform costs (e.g., energy optimization) requires further verification.
- While $u_a$ and $u_b$ control the constant $C$ in Theorem 3.2, an algorithm for adaptively tuning these coefficients online is left for future work.

## Related Work & Insights
- **vs LTS / PHS* (Orseau & Lelis)**: These are base policy/heuristic search methods without subtask decomposition. This work proves that simple structural rerooters can achieve exponential or order-of-magnitude gains over them without subgoal networks.
- **vs HIPS-$\varepsilon$ (Kujanpää 2024) / SGPS (Tuero 2025)**: These use VQ-VAEs for explicit subgoals, which are effective but expensive. They collapse in BoulderDash 30%+. This work achieves similar or better decomposition effects with an order of magnitude less overhead via "implicit subtasks."
- **vs Original $\sqrt{\mathrm{lts}}$ (Orseau et al. 2024)**: The original work provided a theoretical framework and generic bounds but did not specify how to define rerooters in practice. This paper is the first to provide a complete instantiation and empirical validation in complex domains.
- **vs Louvain/Leiden + RL (Evans & Şimşek 2023)**: They use Louvain to find "rooms" for option discovery. This work utilizes Leiden's clustering capability for rerooting weights, which is more lightweight and computed online during search.
- **vs WA* (Weighted A*)**: WA* is heuristic-only. It is significantly outperformed by the rerooter series (over 10× more expansions in TSP), confirming that the "policy + rerooting" route has more scaling potential than pure heuristics.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic attempt to automate rerooters for $\sqrt{\mathrm{lts}}$ with theoretical bounds for additive mixtures; individual components (Leiden/softmax) are clever applications of existing tools.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four domains, three variants vs five baselines, difficulty ramps, and 5 seeds; lacks continuous control domains or real-world robot planning.
- Writing Quality: ⭐⭐⭐⭐ Clear pipeline with dedicated sections and diagrams for each rerooter; some theorem details are relegated to the appendix.
- Value: ⭐⭐⭐⭐⭐ Replaces expensive subgoal generation networks with a simple weight formula, extending capability to difficulty levels where subgoal methods fail; provides immediate engineering value to the search community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] NonZero: Interaction-Guided Exploration for Multi-Agent Monte Carlo Tree Search](nonzero_interaction-guided_exploration_for_multi-agent_monte_carlo_tree_search.md)
- [\[AAAI 2026\] Extreme Value Monte Carlo Tree Search for Classical Planning](../../AAAI2026/others/extreme_value_monte_carlo_tree_search_for_classical_planning.md)
- [\[ICML 2026\] Decision Tree Learning on Product Spaces](decision_tree_learning_on_product_spaces.md)
- [\[ICML 2026\] Complexity as Advantage: A Regret-Based Perspective on Emergent Structure](complexity_as_advantage_a_regret-based_perspective_on_emergent_structure.md)
- [\[ICML 2026\] Networked Information Aggregation for Binary Classification](networked_information_aggregation_for_binary_classification.md)

</div>

<!-- RELATED:END -->
