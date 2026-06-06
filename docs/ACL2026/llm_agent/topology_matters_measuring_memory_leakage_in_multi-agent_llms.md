---
title: >-
  [Paper Note] Topology Matters: Measuring Memory Leakage in Multi-agent LLMs
description: >-
  [ACL 2026][LLM Agent][Multi-agent LLM] This paper systematically measures how communication topology affects the extent of Personally Identifiable Information (PII) leakage in multi-agent LLM systems using the MAMA frame…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Multi-agent LLM"
  - "Memory Leakage"
  - "Topology"
  - "Privacy Attack"
  - "PII Extraction"
date: 2026-05-08
content_hash: 52b247fe571abf3a
---

# Topology Matters: Measuring Memory Leakage in Multi-agent LLMs

**Conference**: ACL 2026  
**arXiv**: [2512.04668](https://arxiv.org/abs/2512.04668)  
**Code**: https://github.com/llll121/mama-eval  
**Area**: Multi-agent / LLM Security  
**Keywords**: Multi-agent LLM, Memory Leakage, Topology, Privacy Attack, PII Extraction

## TL;DR
This paper systematically measures how communication topology affects the extent of Personally Identifiable Information (PII) leakage in multi-agent LLM systems using the MAMA framework. It finds that densely connected topologies and the distance between the attacker and the target are critical factors determining leakage risk.

## Background & Motivation

**Background**: Multi-agent LLM systems are transitioning from prototypes to practical applications, yet their security remains under-investigated. Although prior work has demonstrated that network topology affects the adversarial robustness of multi-agent systems, a systematic understanding of privacy information leakage is still missing.

**Limitations of Prior Work**: Existing multi-agent security research primarily focuses on the propagation of adversarial prompts or task performance degradation. There is a lack of quantitative understanding regarding the dynamics of PII leakage within topological structures. Specifically, most work fails to systematically compare the impact of different topologies, agent positions, and interaction rounds on PII leakage under controlled conditions, leading to a lack of security-based topological guidance in system design.

**Key Challenge**: Network topology is a fundamental feature of distributed multi-agent systems, but its impact on privacy leakage remains unquantified. Studies on single-agent memory attacks (such as MEXTRA or AgentPoison) cannot be directly generalized to multi-agent scenarios because topological structures create new information propagation paths.

**Goal**: To systematically quantify the extent of PII leakage across six typical topologies under various team sizes, attacker-target positions, and interaction rounds, and to provide actionable guidance for the security design of multi-agent systems.

**Key Insight**: The authors borrow topological analysis methods from network science to design a two-stage controlled evaluation framework (Engram+Resonance). This framework systematically probes information leakage across different topologies using synthetic private documents. This controlled experimental environment ensures that any observed leakage originates from agent interactions rather than pre-training memory.

**Core Idea**: Through a carefully designed topology-attack-defense triangle evaluation, it is proven that connectivity, distance, and centrality in the graph structure directly determine the difficulty of PII leakage.

## Method

### Overall Architecture

The MAMA framework consists of three key components: (1) Data Synthesis—generating documents containing annotated PII entities and corresponding public tasks based on the Gretel synthetic dataset; (2) Two-stage Interaction—the Engram stage initializes agents and injects private information into the target agent, while the Resonance stage propagates information through multi-round interactions under a specified topology; (3) Leakage Measurement—evaluating the amount of PII extracted by the attacker using a two-stage recovery mechanism combining exact matching and LLM reasoning.

### Key Designs

1.  **Controlled Data Synthesis and PII Isolation**:
    *   **Function**: Constructs the SPIRIT dataset, which includes annotated PII entities, private documents containing PII, and public task background-question pairs with PII removed. This ensures any PII leakage must come from multi-agent interaction rather than the task itself.
    *   **Mechanism**: By enforcing the constraint $\mathrm{contains}(B_i \cup Q_i, \mathcal{S}_i) = 0$, it is ensured that only the target agent possesses PII information at initialization. This makes the source of leakage fully traceable and excludes confusion with pre-training memory.
    *   **Design Motivation**: Addresses the fundamental problem in previous single-agent PII research where it is difficult to distinguish between "model memory" and "leakage during interaction."

2.  **Two-stage Recovery Evaluation (Exact Matching + LLM Reasoning)**:
    *   **Function**: Uses exact string matching to capture explicitly leaked PII, followed by using DeepSeek-V3.1 as a "judge" to infer PII that is paraphrased or obscured but still inferable.
    *   **Mechanism**: The first stage is $\hat{S}_i^{\mathrm{EM}} = \mathrm{match}(A_i^{(r_i^{\star})}, S_i)$, the second stage is $\hat{S}_i^{\mathrm{INF}} = \mathcal{J}(A_i^{(r_i^{\star})}, S_i \setminus \hat{S}_i^{\mathrm{EM}})$, and the final result is $\hat{S}_i = \hat{S}_i^{\mathrm{EM}} \cup \hat{S}_i^{\mathrm{INF}}$.
    *   **Design Motivation**: Exact matching often misses leakage that has undergone semantic transformation but remains essentially the same (e.g., inferring a birth year from "my child was born in 2008"). LLM reasoning fills this gap, reflecting the actual threat more accurately.

3.  **Systematic Topology-Position Comparison Design**:
    *   **Function**: Evaluates leakage for six typical topologies (Chain, Ring, Star, Star-Ring Hybrid, Tree, Complete Graph) at team sizes $n \in \{4,5,6\}$, enumerating non-redundant attacker-target position pairs to precisely control graph distance and node centrality.
    *   **Mechanism**: For each topology-size combination, the leakage rates of all non-equivalent (attacker_index, target_index) pairs are calculated, and their mean and standard deviation are recorded, covering global distances from direct adjacency (distance 1) to maximum separation (distance $\geq 2$).
    *   **Design Motivation**: Captures the fundamental network security intuition that "position is key," avoiding the pitfall of only comparing topology averages while ignoring massive variation within the same topology.

## Key Experimental Results

### Main Results: Topology Comparison

| Topology | Llama-3.1-70B (n=4) | Llama-3.1-70B (n=6) | DeepSeek-V3.1 (n=4) | Leakage Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| Complete | 29.65% | 25.32% | 16.51% | Most dangerous; all nodes reachable within 1 hop |
| Ring | 24.36% | 16.99% | 15.39% | Moderate risk; cyclic paths provide multiple propagation routes |
| Star-Ring | 25.75% | 23.64% | 14.32% | High risk due to central node bypass + ring edges |
| Pure Star | 24.25% | 23.18% | 14.42% | High risk; central node becomes an information hub |
| Chain | 19.18% | 12.95% | 11.91% | Low risk; requires sequential propagation |
| Tree | 17.47% | 15.14% | 15.23% | Low risk; hierarchical barriers to propagation |

### Attacker-Target Position Sensitivity

| Topology | Position Pair (T-A) | Leakage Rate (Llama-3.1-70B, n=6) | Distance | Description |
| :--- | :--- | :--- | :--- | :--- |
| Ring | 0–1 | 29.49% | 1 | Adjacent nodes; highest risk |
| Ring | 0–2 | 15.38% | 2 | Moderate distance; risk halved |
| Ring | 0–3 | 6.09% | 3 | Opposite nodes; lowest risk |
| Chain | 0–1 | 21.80% | 1 | Adjacent |
| Chain | 0–5 | 1.28% | 5 | Far ends; almost no leakage |
| Star (hub=0) | 0–1 | 30.77% | 1 | Hub-Leaf; extremely high risk |
| Star | 1–2 | 12.82% | 2 | Leaf-Leaf (via hub); reduced risk |

### Ablation Study & Dynamics Analysis

A consistent "rapid rise-plateau" pattern was observed across all topological configurations:

*   **Rounds 1-2**: Leakage rate grows rapidly by 30-50% (relative growth) as information completes its first mix.
*   **Rounds 3-4**: Growth slows significantly with diminishing marginal returns.
*   **Rounds 5+**: Reaches a plateau with very little new leakage.

### Key Findings

*   **Density Dictates**: The leakage rate of a complete graph is on average 2-2.5 times higher than that of a chain or tree, validating the intuition from graph theory that "short paths accelerate diffusion" in LLM scenarios.
*   **Distance Differences are Significant**: Within the same topology, the difference in leakage rates between adjacent and farthest positions can reach 5-25 times (e.g., 21.8% for 0-1 vs. 1.28% for 0-5 in a chain).
*   **Centrality Matters**: In a star topology, leakage is highest when the central node acts as the attacker (30.77%) and remains high when it is the target (25.96%).
*   **PII Type Variance**: Spatiotemporal information (dates, coordinates) has leakage rates >40%, while highly regulated identifiers (SSN, biometric IDs) are near 0%.
*   **Model Choice Affects Magnitude, Not Ranking**: Llama-3.1-70B has a leakage rate 5-6 times higher than GPT-4o, but the topological ranking remains consistent.

## Highlights & Insights

*   **Methodological Breakthrough**: The use of synthetic data with enforced PII isolation solves the long-standing problem of identifying the source of leakage, allowing the causal relationship of topological impact to be scientifically isolated for the first time.
*   **LLM-ization of Network Science**: Precisely mapping the "connectivity → diffusion speed" empirical rule from graph theory onto LLM agents proves that topological intuitions for discrete networks still hold in continuous LLM reasoning.
*   **Insight into Position-Topology Separation**: The paper's brilliance lies in varying both topology and position, discovering that position variance (5-25x within a topology) exceeds topological variance (2-2.5x between topologies). For system design, this implies that "node placement" is more critical than "topology selection."
*   **Plateau Features of Leakage Dynamics**: The discovery that "progress stalls after 3-4 rounds" means the security cost of multi-round interaction diminishes, allowing system designers to make informed trade-offs regarding interaction rounds.

## Limitations & Future Work

*   **Data Scope Limitations**: Synthetic data may not fully capture the semantic complexity and PII density distribution of real-world documents.
*   **Limited Topology Coverage**: Only six typical topologies were tested, excluding small-world graphs, scale-free networks, or multi-layer graphs.
*   **Simplified Interaction Assumptions**: Resonance is fixed at 10 rounds, using a single-attacker model, and all communication is text-based.
*   **Confounding Factors in Safety Alignment**: Leakage rates are also influenced by the model's inherent safety alignment, making it difficult to fully separate "topology-induced risk" from "safety alignment protection."
*   **Missing Concrete Defenses**: The recommendations provided are largely passive. Design of topology-aware access control or encrypted communication routing while maintaining functionality remains future work.

## Related Work & Insights

*   **vs. Single-agent Memory Attacks (MEXTRA/AgentPoison)**: These works focus on memory leakage in a single LLM agent, ignoring the amplification effect of topological structures in multi-agent systems. Ours proves through multi-round interaction that single-agent leakage risk is significantly magnified in multi-agent settings.
*   **vs. Topology-centric Multi-agent Security (NetSafe/G-Safeguard)**: Existing work focuses on adversarial prompt propagation. Ours is the first to systematically quantify the impact of topology on **privacy leakage** rather than functional degradation.
*   **Insight: Topology-aware Defense**: Future designs should not be limited to passively choosing "sparse or hierarchical" topologies but should actively design dynamic topologies—such as temporarily disconnecting unnecessary edges during highly sensitive tasks or using encrypted routing to limit information flow.

## Rating

*   Novelty: ⭐⭐⭐⭐⭐ First systematic quantitative analysis of network topology applied to multi-agent LLM privacy security; methodological innovation with solid experimental design.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 6 topologies × 3 sizes × multiple position pairs × 4 base models × 3 repetitions; high data integrity and reproducibility.
*   Writing Quality: ⭐⭐⭐⭐ Overall logic is clear, though some technical details could be further elaborated.
*   Value: ⭐⭐⭐⭐⭐ Direct guidance value for the security design of multi-agent systems; provides a quantitative topological risk benchmark for practitioners.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] StructMem: Structured Memory for Long-Horizon Behavior in LLMs](structmem_structured_memory_for_long-horizon_behavior_in_llms.md)
- [\[ACL 2026\] Scaling External Knowledge Input Beyond Context Windows of LLMs via Multi-Agent Collaboration](scaling_external_knowledge_input_beyond_context_windows_of_llms_via_multi-agent_.md)
- [\[ACL 2026\] MemSearcher: Training LLMs to Reason, Search and Manage Memory via End-to-End RL](memsearcher_training_llms_to_reason_search_and_manage_memory_via_end-to-end_rein.md)
- [\[ACL 2026\] MAGMA: A Multi-Graph based Agentic Memory Architecture for AI Agents](magma_a_multi-graph_based_agentic_memory_architecture_for_ai_agents.md)
- [\[ACL 2026\] OCR-Memory: Optical Context Retrieval for Long-Horizon Agent Memory](ocr-memory_optical_context_retrieval_for_long-horizon_agent_memory.md)

</div>

<!-- RELATED:END -->
