---
title: >-
  [Paper Note] Securing Multi-Agent Systems Against Corruptions via Node Contribution Backpropagation
description: >-
  [ICML 2026][Multi-Agent][Multi-Agent System] BPD reconstructs multi-round interactions of LLM multi-agent systems (MAS) into a "signed Directed Acyclic Graph (DAG)," assigning $\{-1, 0, 1\}$ scores representing agreement…
tags:
  - "ICML 2026"
  - "Multi-Agent"
  - "Multi-Agent System"
  - "Corruption Attack"
  - "Signed DAG"
  - "Backpropagation"
  - "PageRank"
date: 2026-05-08
content_hash: 9e6d5a206c39ec11
---

# Securing Multi-Agent Systems Against Corruptions via Node Contribution Backpropagation

**Conference**: ICML 2026  
**arXiv**: [2510.19420](https://arxiv.org/abs/2510.19420)  
**Code**: https://github.com/ChengcanWu/BPD  
**Area**: Multi-Agent System Security / LLM Agent Defense  
**Keywords**: Multi-Agent System, Corruption Attack, Signed DAG, Backpropagation, PageRank

## TL;DR
BPD reconstructs multi-round interactions of LLM multi-agent systems (MAS) into a "signed Directed Acyclic Graph (DAG)," assigning $\{-1, 0, 1\}$ scores representing agreement, indifference, or opposition to each message. It then employs a PageRank-style single-pass backward topological propagation to calculate the contribution score of each agent to the final answer. Agents with outlier scores are identified as malicious and their outgoing edges are pruned. The method is training-free, operates on a per-query basis, and is inherently robust to dynamic topologies.

## Background & Motivation
**Background**: LLM agents have evolved from monolithic entities to Multi-Agent Systems (MAS), applied in fields such as software engineering, market analysis, and web automation. Common topologies include Flat (equal discussion) and Hierarchical (respondents + reviewers).

**Limitations of Prior Work**: MAS are more fragile than single LLMs because information flows "infectiously"—harmful content from a hijacked agent can cascade through the dialogue topology to contaminate all downstream agents (corruption attack). Existing defenses fall into three categories, each with critical weaknesses:
- Output Supervision (BlockAgents, AgentForest): Rely on multi-round debates or similarity comparisons, but are vulnerable to subtle text perturbations or direct attacks on the evaluator.
- Static Graph Approaches (Huang et al.): Evaluate topological robustness as fixed GNN configurations, but fail when the topology changes.
- Dynamic Training (G-Safeguard): Train classifiers to read internal agent states but rely on local signals, failing to capture how "corrupt information flows to the final decision."

**Key Challenge**: Existing defenses are either "global but static" or "dynamic but local." No current method performs "global influence tracing" per query to quantify each agent's actual contribution to the final answer without retraining.

**Goal**: (i) Establish a unified graph representation for MAS communication capable of describing arbitrary topologies, agent counts, and rounds; (ii) Design a training-free influence evaluation operator using single-pass backpropagation to make "agent contribution to final decision" a computable metric; (iii) Utilize statistical outlier detection to identify malicious agents and repair the graph, ensuring immunity to dynamic topologies and attacker identity switching.

**Key Insight**: Multi-round MAS dialogues naturally form a DAG when expanded by "time + agent"—edges only cross adjacent rounds, and no cycles exist. This provides a perfect closed-form solution for backward recursion. Drawing from PageRank's concept of "influence aggregation from downstream," the $\{-1, 0, +1\}$ agreement signs are treated as "signed transition probabilities." A single backward pass from the final answer retrieves the accumulated signed influence for each node.

**Core Idea**: MAS = signed temporal DAG; Agent contribution = signed PageRank backward propagation; Malicious agent = intra-group deviation outlier; Defense = prune all outgoing edges of outlier agents.

## Method

### Overall Architecture
Given a MAS dialogue with $T$ rounds and $n$ agents, BPD processes in four steps: (1) Reconstruct the session into a signed DAG $G = (V, E)$; (2) Utilize an independent LLM scorer to assign $g_{ij} \in \{-1, 0, +1\}$ (oppose/ignore/agree) to each message; (3) Use backward recursion from the terminal answer to calculate node contribution scores $S(C_i)$ layer by layer; (4) Identify the set of malicious agents $\mathcal{M}$ based on outlier scores and prune their outgoing edges to obtain a repaired $G'$. This process occurs during the inference phase of each query, is training-free, requires a single backward topological pass, and is highly interpretable.

### Key Designs

1.  **MAS → Signed Temporal DAG**:
    - **Function**: Unifies complex multi-round MAS communication into an acyclic graph, enabling a closed-form solution for backpropagation.
    - **Mechanism**: Agent $A(i)$ at round $t$ is expanded into a temporal node $A_t(i)$, with the node set $V = \{A_t(i) | t=1..T, i=1..n\}$ and size $N = nT$. A message $A(i) \to A(j)$ at round $t$ is denoted as a directed edge $e_t(i,j): A_t(i) \to A_{t+1}(j)$. Since edges only cross adjacent time steps, the graph is inherently acyclic. An independent LLM evaluator rates each edge $e_{ij}$: when receiver $C_j$ produces $s_j$ based on $s_i$ from $C_i$, the scorer computes $g_{ij} = f(s_i, s_j) \in \{-1, 0, +1\}$, where positive indicates agreement/adoption, zero indicates low contribution, and negative indicates opposition/refutation.
    - **Design Motivation**: The DAG structure ensures a unique topological order for "influence propagation," avoiding iterative convergence issues. The signing mechanism allows both "successful attacks" and "exposed attacks" to leave computable traces (the former amplifies positive scores, the latter amplifies negative scores), triggering outlier signals in either direction.

2.  **Backward Score Propagation Operator**:
    - **Function**: Calculates the cumulative signed contribution of each agent node to the final decision via a single topological backward pass.
    - **Mechanism**: Defining the signed adjacency matrix $\mathbf{G} \in \mathbb{R}^{N \times N}$ and the out-degree matrix $\mathbf{D} = \text{diag}(k_1, \ldots, k_N)$, the row-normalized signed propagation operator is $\mathbf{B} = \mathbf{D}^{-1} \mathbf{G}$. The terminal layer is initialized using the final answer: $S(A_T(i)) = +1$ if $A(i)$'s answer matches the MAS final answer $y_{\text{final}}$, otherwise $-1$. The backward recursion is $S(C_i) = \frac{1}{k_i} \sum_{C_j \in \mathcal{N}^+(C_i)} g_{ij} S(C_j) = \sum_j B_{ij} S(C_j)$. In vector form: $\mathbf{S}^{(t)} = \mathbf{P}^{(t)} \mathbf{S}^{(t+1)}$, where $\mathbf{P}^{(t)}_{ij} = g_{t,i \to j} / k_{t,i}$. Since $G$ is a DAG, a single backward multiplication yields a unique closed-form solution $\mathbf{S}^{(t)} = \mathbf{P}^{(t)} \mathbf{P}^{(t+1)} \cdots \mathbf{P}^{(T-1)} \mathbf{S}^{(T)}$. This is a signed, layer-wise DAG generalization of classic PageRank $\mathbf{r}^{(\ell+1)} = (1-d)\mathbf{1}/N + d\, \mathbf{W}^\top \mathbf{r}^{(\ell)}$, where $\mathbf{P}^{(t)}$ replaces $\mathbf{W}^\top$, signs are introduced, and boundary initialization replaces damping/teleportation.
    - **Design Motivation**: PageRank directly quantifies "node influence on the final result," but traditional versions assume strong connectivity and steady states. MAS as finite DAGs are more efficient—calculated in one pass with complexity linear to the number of edges, keeping overhead $<10\%$.

3.  **Outlier Detection & Communication Pruning**:
    - **Function**: Aggregates node scores to the agent level, identifies agents whose scores deviate significantly from the group as malicious, and repairs the MAS by deleting their outgoing edges.
    - **Mechanism**: Agent scores are averaged as $\hat{S}(A(i)) = \frac{1}{|\mathcal{T}(i)|} \sum_{t \in \mathcal{T}(i)} S(A_t(i))$, followed by calculating the mean of pairwise differences: $\Delta(i) = \frac{1}{n-1} \sum_{j \ne i} |\hat{S}(A(i)) - \hat{S}(A(j))|$. The malicious set $\mathcal{M} = \{A(i) | \Delta(i) \ge \epsilon\}$ is identified (empirically $\epsilon = 1.5$). Pruning defines $E_\mathcal{M} = \{e_{t, i \to j} | A(i) \in \mathcal{M}\}$, leading to the repaired graph $G' = (V, E \setminus E_\mathcal{M})$, which silences malicious agents without collapsing the MAS structure.
    - **Design Motivation**: Whether an attack succeeds or fails, malicious agents exhibit deviance in signed PageRank. Success amplifies positive contribution through "infection," while failure leads to refutation and significant negative scores. This dual-direction deviation makes detection insensitive to the specific attack morphology.

### Loss & Training
BPD is **training-free**. The scorer $f$ can be any third-party LLM, and backpropagation is reduced to matrix multiplication. The only hyperparameter is the outlier threshold $\epsilon$, set to $1.5$ based on ablation studies. The complexity of a backward pass and scoring is $O(|E|)$ LLM calls/multiplications, with total time overhead $<10\%$.

## Key Experimental Results

### Main Results
Using GPT-4o as the base LLM, with 5-agent Flat/Hierarchy topologies across five MMLU subdomains. The default attack is the corruption attack by Amayuelas et al. (2024). Results are means of 3 runs $\pm$ 1 std.

| Structure | Method | Algebra | Math | Chemistry | Computer | Security | Avg |
|------|------|---------|------|-----------|----------|----------|-----|
| Flat | No Attack | 95.0 | 94.7 | 75.3 | 92.0 | 85.0 | 88.4 |
| Flat | Attack (None) | 78.7 | 74.7 | 64.7 | 82.3 | 81.0 | 76.3 |
| Flat | G-Safeguard | 88.3 | 88.7 | 71.0 | 87.7 | 83.0 | 83.7 |
| Flat | AGENTXPOSED | 90.0 | 79.0 | 67.0 | 89.0 | 87.3 | 82.5 |
| Flat | Challenger | 88.7 | 87.3 | 68.3 | 84.0 | 75.3 | 80.7 |
| Flat | Inspector | 84.0 | 89.0 | 65.7 | 80.7 | 75.7 | 79.0 |
| Flat | **BPD (Ours)** | 92.3 | 93.0 | 73.3 | 87.3 | 81.0 | **85.4** |
| Hier | Attack (None) | 81.7 | 81.7 | 66.7 | 84.7 | 78.0 | 78.6 |
| Hier | G-Safeguard | 92.0 | 91.7 | 71.3 | 90.3 | 82.3 | 85.5 |
| Hier | **BPD (Ours)** | 93.3 | 95.7 | 73.7 | 91.0 | 83.7 | **87.5** |

BPD improves performance by 9–10 percentage points over no defense, outperforming G-Safeguard by 1.7% on Flat and 2% on Hier topologies. Monitoring Accuracy (Monitor ACC) reaches 90.7% on Flat.

### Ablation Study

| Configuration | Monitor ACC | Task ACC | Key Finding |
|------|----------|----------|----------|
| Full BPD | 90%+ | 85–88% | Complete signed DAG + Backprop + Pruning |
| No Signs ($\{0, 1\}$) | ↓↓ | ↓↓ | Loss of "refutation signals"; failed attackers undetected |
| No Backprop (Local $g_{ij}$) | ↓↓ | ↓ | Degenerates to local signals; fails on deep infection |
| $\epsilon = 1.0$ | High Rec. | ↓ | Over-killing normal agents |
| $\epsilon = 1.5$ (Default) | Balanced | Optimal | Selected for the paper |
| $\epsilon = 2.0$ | Low Rec. | ↓ | Missing stealthy attacks |
| Dynamic Topology | Stable | -0% | Baseline average drop of 3% |
| Semantic Perturbation | Gain ↑ | +~10% | Output supervision methods fail |

### Key Findings
- The global perspective of backpropagation is stronger than local signals: In Flat topologies with deep infection, BPD outperforms G-Safeguard by ~1.7%. Gains are more pronounced in Hierarchy topologies (~2%) as the infection chain lengthens.
- BPD excels in dynamic MAS scenarios (changing structures and attacker identities per query). Competitors drop by 3% while BPD sees almost zero loss, validating the "training-free + per-query calculation" design.
- The largest advantage appears in semantic perturbation attacks (~+10%), which deceive output supervision but leave clear traces in "contribution score deviation."
- Time overhead is $<10\%$, primarily from the LLM scorer; the backward multiplication is negligible.

## Highlights & Insights
- The abstraction of "multi-round dialogue = temporal DAG" is elegant—it applies to any MAS topology and ensures closed-form solutions, avoiding the iterative complexity of standard PageRank. This can extend to multi-tool agents or debate-based reasoning.
- The sign mechanism $\{-1, 0, +1\}$ is critical—standard PageRank only identifies "influence by majority agreement." Signs allow "refutation" to become a signal, expanding detection to "failed attack" scenarios.
- Using the mean of pairwise differences $\Delta(i)$ for outlier detection is robust even for small groups ($n \le 5$) without requiring normality assumptions.
- The isomorphism between backward PageRank and neural network backpropagation—propagating "error signals" through a calculation graph—is a sophisticated cross-domain adaptation for MAS security.

## Limitations & Future Work
- Scorer $f$ is an external LLM; its reliability determines $g_{ij}$. If the scorer is compromised (e.g., via prompt injection), BPD fails.
- The DAG assumption requires strict "round-based" progression. It is not directly applicable to asynchronous or cyclic dialogues without temporal expansion.
- The threshold $\epsilon = 1.5$ is empirical and may requires tuning across different tasks or topologies; it lacks an adaptive mechanism.
- In "Byzantine" scenarios where malicious agents are the majority (>50%), the outlier assumption collapses, and BPD may misidentify normal agents.
- Coordinated attacks where multiple attackers create a "consistent but wrong" consensus would produce positive signs, making them indistinguishable from the normal majority without an external ground-truth arbiter.

## Related Work & Insights
- **vs G-Safeguard**: Both are dynamic, but G-Safeguard relies on trained GNNs and local signals. BPD is global and training-free, outperforming it by 1.7%–2.0%.
- **vs Output Supervision (BlockAgents/AgentForest)**: These rely on debate/similarity and fail against text perturbations. BPD relies on contribution deviation and maintains a ~10% lead on such attacks.
- **vs Static Topology (Huang et al. 2025)**: Their fixed searching fails in dynamic environments, where BPD remains robust.
- **vs Collaborative Defense (Challenger/Inspector)**: These rely on dedicated reviewer agents; if reviewers are breached, they fail. BPD has no centralized "attackable" node.

## Rating
- Novelty: ⭐⭐⭐⭐ Elegantly injects PageRank into MAS security via signed layer-wise DAGs.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers various topologies, LLMs, and tasks, including dynamic scenarios and temporal overhead.
- Writing Quality: ⭐⭐⭐⭐ Clear derivation from MAS to DAG to backpropagation.
- Value: ⭐⭐⭐⭐ Provides the first "training-free + global + dynamic" MAS defense, readily deployable for any agent system.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] MASPO: Joint Prompt Optimization for LLM-based Multi-Agent Systems](maspo_joint_prompt_optimization_for_llm-based_multi-agent_systems.md)
- [\[ICLR 2026\] Stochastic Self-Organization in Multi-Agent Systems](../../ICLR2026/multi_agent/stochastic_self-organization_in_multi-agent_systems.md)
- [\[ACL 2026\] Conjunctive Prompt Attacks in Multi-Agent LLM Systems](../../ACL2026/multi_agent/conjunctive_prompt_attacks_in_multi-agent_llm_systems.md)
- [\[ACL 2026\] Towards Self-Improving Error Diagnosis in Multi-Agent Systems](../../ACL2026/multi_agent/towards_self-improving_error_diagnosis_in_multi-agent_systems.md)
- [\[AAAI 2026\] BAMAS: Structuring Budget-Aware Multi-Agent Systems](../../AAAI2026/multi_agent/bamas_structuring_budget-aware_multi-agent_systems.md)

</div>

<!-- RELATED:END -->
