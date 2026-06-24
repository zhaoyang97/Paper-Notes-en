---
title: >-
  [Paper Note] Efficient Training-Free Online Routing for High-Volume Multi-LLM Serving
description: >-
  [NeurIPS 2025][LLM Efficiency][LLM routing] This paper proposes PORT, the first training-free online LLM routing algorithm. PORT estimates query features via approximate nearest neighbor search (ANNS) and performs a one-shot optimization of dual variables as routing weights on a small set of initial queries. Under a limited token budget, PORT achieves near-offline-optimal routing performance with a $1-o(1)$ competitive ratio, delivering on average 3.55× performance improvemen…
tags:
  - "NeurIPS 2025"
  - "LLM Efficiency"
  - "LLM routing"
  - "online optimization"
  - "approximate nearest neighbor search"
  - "dual optimization"
  - "multi-model serving"
date: 2026-05-08
content_hash: 842043f2bb4b91bb
---

# Efficient Training-Free Online Routing for High-Volume Multi-LLM Serving

**Conference**: NeurIPS 2025
**arXiv**: [2509.02718](https://arxiv.org/abs/2509.02718)  
**Code**: [https://github.com/fzwark/PORT](https://github.com/fzwark/PORT)  
**Area**: LLM Efficiency
**Keywords**: LLM routing, online optimization, approximate nearest neighbor search, dual optimization, multi-model serving

## TL;DR
This paper proposes PORT, the first training-free online LLM routing algorithm. PORT estimates query features via approximate nearest neighbor search (ANNS) and performs a one-shot optimization of dual variables as routing weights on a small set of initial queries. Under a limited token budget, PORT achieves near-offline-optimal routing performance with a $1-o(1)$ competitive ratio, delivering on average 3.55× performance improvement, 1.85× cost efficiency, and 4.25× throughput over baselines.

## Background & Motivation
**Background**: LLM service providers face massive query volumes (e.g., OpenAI peak at 12k QPS). Different LLMs excel in different domains and vary in cost, making query routing to the most suitable model a key strategy for cost reduction.

**Limitations of Prior Work**: Existing routing methods either rely on trained model predictors (e.g., RoBERTa-based), incurring high computational overhead and requiring retraining upon deployment changes, or depend on exact KNN with complexity $O(|D|)$, which is incompatible with low-latency online serving requirements.

**Key Challenge**: All existing methods rest on an offline assumption—that all queries arrive simultaneously and can be globally optimized. In practice, queries arrive sequentially, future queries are unknown, token budgets are limited, and offline MILP formulations become intractable.

**Goal**: (a) How to perform efficient routing online without training any model? (b) How to maximize overall service quality under hard budget constraints? (c) How to adapt to dynamic changes in LLM deployment configurations?

**Key Insight**: The authors observe that the LP relaxation dual of the routing problem is fully parameterized at optimality by a single dual variable $\gamma$, which serves as the routing weight for each LLM. Consequently, learning $\gamma^*$ from a small set of initial queries suffices to guide all subsequent routing decisions.

**Core Idea**: ANNS-based query feature estimation + one-shot dual variable optimization = training-free, efficient, theoretically-guaranteed online routing.

## Method

### Overall Architecture
PORT operates in two phases: (1) **Exploration phase**: the first $\epsilon|Q|$ queries (approximately 250) are routed randomly; their features are recorded and used for a one-shot optimization to solve for dual routing weights $\gamma^*$. (2) **Exploitation phase**: for all subsequent queries, ANNS estimates per-LLM performance and cost, and each query is routed according to $\arg\max_i(\alpha \hat{d}_{ij} - \hat{g}_{ij}\gamma_i^*)$.

### Key Designs

1. **ANNS-Based Feature Estimation**:

    - **Function**: Rapidly estimate each query's performance $\hat{d}_{ij}$ and cost $\hat{g}_{ij}$ on each LLM.
    - **Mechanism**: A historical dataset $D = \{j, a_j, d_j, g_j\}$ is maintained. For each new query, HNSW retrieves $k$ nearest neighbors $R_j$ in the embedding space, and estimates are computed as: $\hat{d}_{ij} = \frac{1}{|R_j|}\sum_{q \in R_j} d_{iq}$
    - **Design Motivation**: Compared to trained predictors, ANNS requires no training and adapts to deployment changes simply by updating the historical dataset. Search complexity is only $O(\log|D|)$, far superior to KNN's $O(|D|)$.

2. **Dual-Driven Routing Weight Learning**:

    - **Function**: Relax the routing MILP to an LP, derive its dual, and learn optimal routing weights $\gamma^*$.
    - **Mechanism**: By complementary slackness, the optimal routing rule is $x_{ij} > 0 \Leftrightarrow j$ is assigned to $\arg\max_i(\alpha\hat{d}_{ij} - \hat{g}_{ij}\gamma_i)$. The dual objective is parameterized as $F(\gamma) = \sum_i \gamma_i B_i + \sum_j \max_i(\alpha\hat{d}_{ij} - \hat{g}_{ij}\gamma_i)$, and $\gamma^*$ is obtained by minimizing $F(\gamma, P)$ on the small exploration sample.
    - **Design Motivation**: The dual variable $\gamma$ is essentially the shadow price of each LLM's budget. Once learned, it converts the budget constraint into a simple linear scoring rule.

3. **Control Parameter $\alpha$ and PAC Random Routing**:

    - **Function**: Introduce control parameter $\alpha$ to scale the objective; employ random routing during exploration.
    - **Mechanism**: $\alpha$ preserves the structure of the optimal solution while controlling the performance-cost trade-off for future queries. Random routing ensures unbiased sampling so that $\gamma^*$ generalizes to subsequent queries (following the PAC learning principle).
    - **Design Motivation**: Addresses the generalization challenge in online learning—weights learned from a small sample must remain effective over a large number of unseen queries.

### Loss & Training
No model training is required. A single convex optimization is performed over $\epsilon|Q|$ (approximately 2.5%) initial queries to obtain $\gamma^*$, with negligible computational overhead.

## Key Experimental Results

### Main Results

| Algorithm | RouterBench Perf | RouterBench PPC | SPROUT Perf | SPROUT PPC | OpenLLM Perf | OpenLLM PPC |
|------|-----------------|----------------|-------------|------------|-------------|-------------|
| Random | 1384.3 | 3243.3 | 2827.6 | 3927.3 | 953.0 | 1284.4 |
| Greedy-Cost | 1626.3 | 3534.5 | 3934.7 | 4630.4 | 1051.0 | 1371.3 |
| BatchSplit | 1838.1 | 4005.9 | 3975.5 | 4784.5 | 1059.0 | 1392.1 |
| Roberta-Cost | 481.4 | 3738.9 | 3996.2 | 4709.2 | 1044.0 | 1362.5 |
| **PORT (Ours)** | **2718.6** | **6075.6** | **4513.1** | **5536.7** | **1465.0** | **2060.3** |
| Offline Approx. Optimal | 3211.4 | 6975.2 | 5939.0 | 6986.5 | 1910.0 | 2493.7 |

PORT achieves 84.66%, 75.99%, and 76.7% of the approximate offline optimum on the three benchmarks, respectively.

### Ablation Study

| Dimension | Key Findings |
|---------|---------|
| Query volume 4k→12k | PORT's advantage grows with scale, outperforming BatchSplit by up to 50% |
| 100 random query orderings | Consistently outperforms all baselines across runs with low variance |
| Number of LLMs 2→16 | Maintains optimality across different deployment configurations |
| Budget multiplier 0.25×→2× | Remains optimal even under extremely low budgets |
| Extreme budget allocation (80/20) | Achieves 2× over BatchSplit when one model receives 80% of the budget |
| Historical data size 5k→25k | Performance is robust to the amount of historical data |
| Search candidates 3→10 | As few as 3 candidates suffice to outperform all baselines |

## Highlights & Insights
- **Training-free online routing with theoretical guarantees**: PORT is the first online LLM routing algorithm that simultaneously satisfies "no training required" and "provable $1-o(1)$ competitive ratio." The dual perspective reduces the complex online MILP to a one-shot optimization followed by linear scoring.
- **Clever ANNS-based substitution for KNN**: Using HNSW graph indexing reduces search complexity from $O(n)$ to $O(\log n)$, and the historical dataset supports dynamic incremental updates, perfectly accommodating the evolving nature of LLM deployments.
- **PAC random routing for generalization**: The exploration phase employs random rather than greedy routing, sacrificing optimality on a small fraction (2.5%) of queries in exchange for an unbiased estimate of routing weights—a principled application of classical online learning theory.
- **Transferability**: The dual decomposition + ANNS framework is transferable to other online resource allocation problems under constraints (e.g., ad serving, cloud resource scheduling).

## Limitations & Future Work
- **Only models performance and cost**: Dynamic operational factors such as query traffic patterns and system load—relevant to load balancing—are excluded from the current MILP formulation.
- **Practical satisfaction of Assumption 1**: The assumption that queries with nearby embeddings exhibit similar LLM performance/cost may not hold for cross-domain queries or out-of-distribution scenarios.
- **Cold-start in the exploration phase**: The method requires an existing historical dataset $D$, making it unsuitable for entirely fresh deployment scenarios.
- **Single query to single LLM**: Ensemble or cascading strategies are not considered, potentially forgoing gains from multi-model collaboration.

## Rating
- Novelty: ⭐⭐⭐⭐ The dual perspective + ANNS combination for online routing is novel, though the underlying tools (ANNS, LP duality) are well-established.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three benchmarks, eight baselines, and an exceptionally comprehensive robustness analysis.
- Writing Quality: ⭐⭐⭐⭐ The theoretical exposition is clear and rigorous, though the dense notation raises the entry barrier.
- Value: ⭐⭐⭐⭐ Addresses a practically relevant online LLM routing problem with both theoretical and engineering contributions.

## Related Work & Insights

| Method Category | Representative Work | PORT Advantage |
|---------|---------|----------|
| Model-predictor routing | RouteLLM, CARROT (RoBERTa-based) | No training required; no retraining needed upon deployment changes; lower latency |
| KNN routing | RouterBench (exact KNN) | ANNS search $O(\log n)$ vs. KNN $O(n)$; latency reduced by orders of magnitude |
| Cascade / ensemble | FrugalGPT, AutoMix | Single routing call vs. multiple model invocations; lower latency and cost |
| Online MILP | Cerebrum, On-the-fly | Dual decomposition avoids per-batch LP solving; one-shot optimization followed by linear scoring |
| BatchSplit | Mini-batch LP | PORT leverages global dual information; 33% higher performance and 24% higher throughput |

Core distinction: PORT is the only routing algorithm that simultaneously satisfies all three conditions—"training-free," "online," and "theoretically guaranteed." All other methods lack at least one of these properties.

## Related Papers

- [\[NeurIPS 2025\] Hierarchical Balance Packing: Towards Efficient Supervised Fine-tuning for Long-Context LLM](hierarchical_balance_packing_towards_efficient_supervised_fine-tuning_for_long-c.md)
- [\[NeurIPS 2025\] Mozart: Modularized and Efficient MoE Training on 3.5D Wafer-Scale Chiplet Architectures](mozart_modularized_and_efficient_moe_training_on_35d_wafer-scale_chiplet_archite.md)
- [\[NeurIPS 2025\] OmniDraft: A Cross-Vocabulary Online Adaptive Drafter for On-Device Speculative Decoding](omnidraft_a_cross-vocabulary_online_adaptive_drafter_for_on-device_speculative_d.md)
- [\[CVPR 2026\] SparVAR: Exploring Sparsity in Visual Autoregressive Modeling for Training-Free Acceleration](../../CVPR2026/llm_efficiency/sparvar_exploring_sparsity_in_visual_autoregressive_modeling_for_training-free_a.md)
- [\[NeurIPS 2025\] DISC: Dynamic Decomposition Improves LLM Inference Scaling](disc_dynamic_decomposition_improves_llm_inference_scaling.md)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Efficient Training-Free Multi-Token Prediction via Embedding-Space Probing](../../ICML2026/llm_efficiency/efficient_training-free_multi-token_prediction_via_embedding-space_probing.md)
- [\[NeurIPS 2025\] OmniDraft: A Cross-Vocabulary Online Adaptive Drafter for On-Device Speculative Decoding](omnidraft_a_cross-vocabulary_online_adaptive_drafter_for_on-device_speculative_d.md)
- [\[ICLR 2026\] DASH: Deterministic Attention Scheduling for High-throughput Reproducible LLM Training](../../ICLR2026/llm_efficiency/dash_deterministic_attention_scheduling_for_high-throughput_reproducible_llm_tra.md)
- [\[ICLR 2026\] Universal Model Routing for Efficient LLM Inference](../../ICLR2026/llm_efficiency/universal_model_routing_for_efficient_llm_inference.md)
- [\[ACL 2026\] Task-Aware LLM Routing with Multi-Level Task-Profile-Guided Data Synthesis for Cold-Start Scenarios](../../ACL2026/llm_efficiency/task-aware_llm_routing_with_multi-level_task-profile-guided_data_synthesis_for_c.md)

</div>

<!-- RELATED:END -->
