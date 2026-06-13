---
title: >-
  [Paper Note] Anchor-guided Hypergraph Condensation with Dual-level Discrimination
description: >-
  [ICML 2026][Graph Learning][hypergraph condensation] AHGCDD reformulates Hypergraph Condensation (HGC) from a decoupled paradigm (training a structure generator followed by trajectory matching) into an end-to-end framewo…
tags:
  - "ICML 2026"
  - "Graph Learning"
  - "hypergraph condensation"
  - "HKPR diffusion"
  - "anchor-guided hyperedge"
  - "dual-level discrimination"
  - "MMD"
date: 2026-05-08
content_hash: 23111fbaf6ee17be
---

# Anchor-guided Hypergraph Condensation with Dual-level Discrimination

**Conference**: ICML 2026  
**arXiv**: [2605.10001](https://arxiv.org/abs/2605.10001)  
**Code**: Not disclosed  
**Area**: Graph Learning / Hypergraph Neural Networks / Graph/Hypergraph Condensation  
**Keywords**: hypergraph condensation, HKPR diffusion, anchor-guided hyperedge, dual-level discrimination, MMD

## TL;DR
AHGCDD reformulates Hypergraph Condensation (HGC) from a decoupled paradigm (training a structure generator followed by trajectory matching) into an end-to-end framework: it embeds structural information into initialized features via Heat-Kernel-PageRank, synthesizes sparse learnable hyperedges based on feature distances using an anchor-guided approach, and replaces expensive HNN retraining with a coarse-to-fine dual-level discriminative loss (class prototype MMD + instance-level contrastive). It achieves ≥ SOTA performance on five hypergraph benchmarks with up to 144× speedup.

## Background & Motivation

**Background**: Hypergraph Neural Networks (HNNs) excel at modeling high-order interactions in social analysis, biochemistry, and e-commerce. However, training on large-scale hypergraphs incurs massive computational costs. Graph Condensation (GC) compresses the original graph into a small synthetic one while maintaining downstream GNN performance. In 2025, HG-Cond extended this to hypergraphs by pre-training a Neural Hyperedge Linker (NHL) with variational inference to capture high-order connectivity and using GPSM to align training trajectories via repeated HNN retraining.

**Limitations of Prior Work**: HG-Cond suffers from two fundamental issues: (1) **Decoupling of structure generation and feature optimization**: The NHL is frozen during the amelioration phase; it was optimized to "reconstruct the original hypergraph" rather than jointly with synthetic features, leading to structural-node misalignment and degraded downstream accuracy. (2) **Resource-intensive trajectory matching**: Every amelioration round requires retraining HNNs, and the memory overhead of NHL variational pre-training makes it difficult to scale to massive hypergraphs.

**Key Challenge**: Incorporating "structure, features, and training trajectories" simultaneously into bi-level optimization inevitably lead to either retraining or complex alignment losses. Maintaining downstream accuracy without retraining requires a lightweight signal that can supervise both structure and features.

**Goal**: (1) Integrate the structure generator into end-to-end optimization to avoid misalignment; (2) Identify an alignment objective that does not require HNN retraining; (3) Encode high-order structural information into features during the initialization phase to provide a strong starting point for optimization.

**Key Insight**: First, apply a low-pass spectral filter using Heat Kernel PageRank on the original graph to "bake" multi-hop structural knowledge into node features. Second, allow each synthetic node to serve as an anchor and use an MLP to learn pairwise affinity between synthetic nodes to form differentiable sparse hyperedges. Finally, use a composite loss of prototype MMD and node-level InfoNCE to preserve both the global class distribution and local decision boundaries.

**Core Idea**: Replace the "structure by generator + features by matching" paradigm with "structure and features driven simultaneously by discriminative loss," and use HKPR to collapse expensive "repeated propagation" into a one-time initialization filter.

## Method

### Overall Architecture
Given a large hypergraph $\mathcal{T}=(\mathbf{X},\mathbf{H},\mathbf{Y})$, the goal is to synthesize a hypergraph $\mathcal{S}=(\mathbf{X}',\mathbf{H}',\mathbf{Y}')$ where $N'\ll N$ and $M'\ll M$. The pipeline consists of: (1) **HKPR Initialization**—performing truncated Heat Kernel PageRank diffusion on original node features to obtain $\tilde{\mathbf{X}}$, then mapping to $N'$ synthetic nodes $\mathbf{X}'$ via intra-class mean pooling; (2) **Anchor-guided Hyperedge Generation**—treating each $\mathbf{X}'_i$ as an anchor and using an MLP to learn association scores $\hat{\mathbf{H}}'_i$ with other synthetic nodes, refined by an adaptive threshold $\delta_i$ per edge for sparsity via ReLU; (3) **Dual-level Discriminative Loss**—simultaneously aligning synthetic and original class prototypes (coarse) and instance geometry (fine), with dynamic switching using cos/sin time weighting. Downstream HNNs are trained on $\mathcal{S}$ to achieve near full-graph accuracy.

### Key Designs

1. **HKPR-based Node Initialization (Baking structural knowledge into features)**:

    - **Function**: Infuses multi-scale high-order structural information into synthetic node features before condensation starts, providing a structure-aware starting point and a topological signal for feature-driven hyperedge generation.
    - **Mechanism**: Define the normalized hypergraph propagation operator $\mathbf{P}=\mathbf{D}_v^{-1/2}\mathbf{H}\mathbf{D}_e^{-1}\mathbf{H}^\top\mathbf{D}_v^{-1/2}$. HKPR diffusion is $\tilde{\mathbf{X}}=\sum_{k=0}^\infty \frac{e^{-\lambda}\lambda^k}{k!}\mathbf{P}^{(k)}\mathbf{X}$, equivalent to low-pass filtering $g(\mu)=e^{-\lambda\mu}$ in the hypergraph Fourier domain (Thm 3.1). Truncating at $K=\lceil\lambda+3\sqrt{\lambda}\rceil$ yields an exponentially decaying error bound (Lemma 3.2). Synthetic features $\mathbf{X}'_i$ are initialized by classroom mean pooling: $\mathbf{X}'_i=\frac{1}{|S_i|}\sum_{j\in S_i}\tilde{\mathbf{X}}_j$.
    - **Design Motivation**: Direct random initialization forces the optimizer to learn structure from scratch. HKPR embeds "K-hop neighborhood + global context" via one-time low-pass filtering, providing a strong prior and filtering out high-frequency noise.

2. **Anchor-guided Hyperedge Generation + Adaptive Sparsity Threshold**:

    - **Function**: Makes structure generation differentiable, enables joint training with features, and supports independent sparsity per edge.
    - **Mechanism**: Each synthetic node $v_i'$ acts as an anchor. A shared MLP computes pairwise associations: $\hat{h}'_{i,j}=\text{sigmoid}(\text{MLP}_\Phi([\mathbf{X}'_i;\mathbf{X}'_j]))$ to form the incident vector $\hat{\mathbf{H}}'_i$. A per-hyperedge adaptive threshold $\delta_i$ is learned to sparsify the structure: $\mathbf{H}'_i=\text{ReLU}(\hat{\mathbf{H}}'_i-\delta_i)$. Both $\mathbf{H}'$ and $\mathbf{X}'$ are differentiable with respect to the same loss.
    - **Design Motivation**: Methods like HG-Cond use global thresholds, leading to uniform edge density and lost expressiveness. The anchor perspective allows each hyperedge to be driven by a central node, aligning with the intuition that hypergraphs are high-order motifs around nodes. The MLP avoids pre-training a generator and allows structure to evolve with features under discriminative loss gradients.

3. **Dual-level Discriminative Loss + cos/sin Dynamic Weighting**:

    - **Function**: Aligns the synthetic and original data across global class distributions (coarse) and intra-class instance geometry (fine) as the condensation objective, without retraining HNNs.
    - **Mechanism**: (a) **Coarse-grained** $\mathcal{L}_c$: Based on prototypes $\mathbf{C}=\mathbf{Y}^\top\tilde{\mathbf{X}}$ and $\mathbf{C}'=\mathbf{Y}'^\top\tilde{\mathbf{X}}'$, it maximizes cosine similarity for same-class prototypes while minimizing different-class similarity. Thm 3.3 proves this is equivalent to minimizing MMD on the joint distribution of (normalized features, labels); Prop 3.5 provides a class-level margin lower bound. (b) **Fine-grained** $\mathcal{L}_f$: For each synth node $v_i'$, samples same-class original nodes as positives and different-class as negatives for an InfoNCE-style contrastive loss. Prop 3.8 proves this upper-bounds the mis-ranking probability $\Pr(\mathcal{E}_i)\leq\mathbb{E}[e^{l_i}-1]$. (c) **Dynamic Weighting**: $\mathcal{L}_{Disc}^{(t)}=\cos(\frac{\pi t}{2T})\mathcal{L}_c+\sin(\frac{\pi t}{2T})\mathcal{L}_f$, where $T$ is the total rounds. Early stages focus on global alignment, while later stages refine local decision boundaries.
    - **Design Motivation**: Coarse alignment ensures global separability but fails to refine crowded regions. Fine alignment handles instance discrimination but is sensitive to sampling noise. The weighted fusion theoretically optimizes both MMD and ranking margin; the cos/sin scheduler implements curriculum learning (Global $\rightarrow$ Local) without additional hyperparameters.

### Loss & Training
The final condensation objective is $\min_{\mathbf{X}', \Phi, \delta}\mathcal{L}_{Disc}^{(t)}$, which bypasses the HNN retraining step. Hyperparameters include HKPR path intensity $\lambda$, truncation order $K$, sample size $s$, negative sample count $N_{neg}$, and training epochs $T$. The time complexity is $\mathcal{O}(KM\delta_e d+T(L_\Phi N'^2 d^2+N'N_{neg}d))$, where the primary terms relate to original edge counts and synthetic size, significantly lower than the HNN training cost in trajectory matching.

## Key Experimental Results

### Main Results
The authors compared SOTA HGC (HG-Cond) and several GC methods across five hypergraph benchmarks (Cora, Pubmed, DBLP-CA, Walmart, Yelp) regarding downstream HNN accuracy after condensation:

| Dataset | Nodes | Hyperedges | Classes | Description |
|--------|--------|--------|------|------|
| Cora | 2,708 | 1,579 | 7 | co-citation |
| Pubmed | 19,717 | 7,963 | 3 | co-citation |
| DBLP-CA | 41,302 | 22,363 | 6 | co-authorship |
| Walmart | 88,860 | 69,906 | 11 | co-purchase |
| Yelp | 50,758 | 679,302 | 9 | co-occurrence |

| Method Category | Accuracy Trends | Condensation Speed |
|----------|----------|----------|
| GC (Jin et al. 2022; Zheng et al. 2023; ...) applied to HG | Lags behind on all HG data (lacks high-order modeling) | Moderate |
| HG-Cond (Trajectory Matching + NHL) | SOTA but requires multiple HNN retrains | Slow |
| **AHGCDD** | ≥ HG-Cond on all 5 datasets | **Up to 144× speedup** |

### Ablation Study

| Configuration | Phenomenon | Interpretation |
|------|------|------|
| w/o HKPR (Random Synth Init) | Significant drop in accuracy | Structure-aware initialization is a vital prior |
| Global threshold instead of anchor-adaptive $\delta_i$ | Structural homogenization, lower accuracy | Adaptive sparsity allows density on demand |
| $\mathcal{L}_c$ only (Coarse-grained) | Global classes clear, but intra-class crowding (Yelp/Walmart drop) | Lacks local ranking signals |
| $\mathcal{L}_f$ only (Fine-grained) | Prototype shift, unstable training | Lacks global distribution constraints |
| Fixed 50%/50% Weighting | Inferior to cos/sin dynamic scheduling | Curriculum learning is effective |
| Retraining with GPSM (HG-Cond style) | Time cost ↑↑, no significant gain in accuracy | Dual-level discrimination is sufficient |
| Replace Anchor Gen with pre-trained NHL | Accuracy drop | End-to-end optimization is critical |

### Key Findings
- HKPR initialization and anchor generation are orthogonal performance gains: the former enables "structure $\rightarrow$ feature" knowledge transfer, while the latter enables "feature $\rightarrow$ structure" end-to-end feedback.
- $\lambda$ controls average HKPR diffusion steps; smaller $\lambda$ (e.g., 2-3) suits small-diameter graphs, while larger $\lambda$ is better for Pubmed/Walmart—reflecting the hypergraph spectral radius.
- Efficiency gains scale with the original graph size: the 144× speedup on Yelp occurs because HG-Cond requires massive trajectory matching/HNN retraining on large graphs.
- The Coarse-to-Fine order in the dual-level loss is crucial for convergence stability; reversing the order causes the model to get stuck in local intra-class optima early on.

## Highlights & Insights
- **Dual Proofs of Method and Theory**: Thm 3.3 links $\mathcal{L}_c$ to MMD, and Prop 3.8 links $\mathcal{L}_f$ to mis-ranking bounds. This "Condensation Loss = Distribution Alignment + Ranking Guarantee" duality is rare in GC literature and provides a strong template for future work.
- **HKPR Filter Perspective**: Compressing "multi-hop propagation" into a one-time spectral filter is a transferable trick—any initialization requiring prior structural signals can use similar low-pass filters instead of repeated message passing.
- **Differentiable Hyperedges via Anchors**: Each node serves as both an anchor and a candidate member, naturally supporting high-order interactions of arbitrary arity; the threshold $\delta_i$ lets the optimizer decide edge density.
- **Discriminative Loss without Retraining**: This is the most valuable engineering contribution—liberating GC from "proxy task matching" to "direct discriminative alignment," allowing scalability to billion-scale graphs.

## Limitations & Future Work
- The maximum scale tested was Yelp (50K nodes / 679K edges); whether the 144× speedup holds for industrial-scale hypergraphs (millions) remains unverified.
- Anchor MLP complexity is $\mathcal{O}(L_\Phi N'^2 d^2)$, quadratic in $N'$; this becomes a bottleneck if users require high condensation ratios (e.g., 1%).
- HKPR intensity $\lambda$ requires manual tuning or grid search; no adaptive estimation is provided despite its sensitivity to dataset types.
- Evaluation is limited to node classification; it is unknown if dual-level discrimination holds SOTA for hypergraph link prediction or subgraph classification.
- The cos/sin scheduler depends on a fixed total epoch count $T$; varying training durations may shift the optimal schedule.

## Related Work & Insights
- **vs HG-Cond (Gong et al. 2025)**: HG-Cond uses NHL pre-training and GPSM trajectory matching for high-fidelity condensation but at a massive cost; AHGCDD end-to-endizes structure generation and uses discriminative loss to drastically speed up while maintaining or exceeding accuracy.
- **vs GCond / SFGC (Graph Condensation)**: These works handle pairwise graphs; AHGCDD extends the logic to high-order via anchors and adaptive sparsity and is the first to prove the equivalence of "MMD ↔ Prototype Alignment" in HGC.
- **vs DSL / GraphSAINT (Graph Sampling)**: Sampling preserves original sub-structures, while AHGCDD synthesizes new ones for better control; they target different scenarios (training acceleration vs. inference serving).
- **vs Dataset Distillation (Wang et al.; Cazenavette et al.)**: Traditional DD uses gradient/trajectory matching; AHGCDD provides an alternative path—using "distribution alignment + ranking guarantee" to prove high-quality distillation is possible without proxy tasks.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of HKPR initialization, anchor hyperedges, and dual-level discrimination is a first for HGC, with solid theoretical backing for each.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 5 datasets with multiple HNN backbones and complete ablations; however, larger graphs and diverse downstream tasks are missing.
- **Writing Quality**: ⭐⭐⭐⭐ Rigorous derivations and clear ablations; Theorems 3.1/3.3 and Propositions 3.5/3.8 are perfectly positioned.
- **Value**: ⭐⭐⭐⭐ The 144× speedup + ≥ SOTA accuracy makes this highly practical, providing a viable pre-processing solution for large-scale hypergraph training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Learnable Kernel Density Estimation for Graphs and Its Application to Graph-Level Anomaly Detection](learnable_kernel_density_estimation_for_graphs_and_its_application_to_graph-leve.md)
- [\[ICML 2026\] DTKG: Dual-Track Knowledge Graph-Verified Reasoning Framework for Multi-Hop QA](dtkg_dual-track_knowledge_graph-verified_reasoning_framework_for_multi-hop_qa.md)
- [\[ICLR 2026\] Pairwise is Not Enough: Hypergraph Neural Networks for Multi-Agent Pathfinding](../../ICLR2026/graph_learning/pairwise_is_not_enough_hypergraph_neural_networks_for_multi-agent_pathfinding.md)
- [\[AAAI 2026\] BugSweeper: Function-Level Detection of Smart Contract Vulnerabilities Using Graph Neural Networks](../../AAAI2026/graph_learning/bugsweeper_function-level_detection_of_smart_contract_vulnerabilities_using_grap.md)
- [\[ACL 2026\] TagRAG: Tag-guided Hierarchical Knowledge Graph Retrieval-Augmented Generation](../../ACL2026/graph_learning/tagrag_tag-guided_hierarchical_knowledge_graph_retrieval-augmented_generation.md)

</div>

<!-- RELATED:END -->
