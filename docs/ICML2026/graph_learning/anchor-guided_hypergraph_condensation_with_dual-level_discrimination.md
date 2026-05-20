---
title: >-
  [Paper Note] Anchor-guided Hypergraph Condensation with Dual-level Discrimination
description: >-
  [ICML 2026][Graph Learning][hypergraph condensation] AHGCDD rewrites hypergraph condensation (HGC) from the decoupled paradigm of "train structure generator first…
tags:
  - "ICML 2026"
  - "Graph Learning"
  - "hypergraph condensation"
  - "HKPR diffusion"
  - "anchor-guided hyperedge"
  - "dual-level discrimination"
  - "MMD"
date: 2026-05-08
content_hash: 093e2b4f9e59b09b
---

# Anchor-guided Hypergraph Condensation with Dual-level Discrimination

**Conference**: ICML 2026  
**arXiv**: [2605.10001](https://arxiv.org/abs/2605.10001)  
**Code**: Not released  
**Area**: Graph Learning / Hypergraph Neural Networks / Dataset Condensation (Graph/Hypergraph Condensation)  
**Keywords**: hypergraph condensation, HKPR diffusion, anchor-guided hyperedge, dual-level discrimination, MMD

## TL;DR
AHGCDD rewrites hypergraph condensation (HGC) from the decoupled paradigm of "train structure generator first, then match training trajectories" into an end-to-end framework: Heat-Kernel-PageRank injects structural information into initial features, an anchor-guided approach synthesizes sparse, learnable hyperedges based on feature distances, and a dual-level discrimination loss (class prototype MMD + instance-level contrastive) replaces expensive HNN retraining. On five hypergraph benchmarks, it achieves ≥SOTA accuracy with up to 144× speedup.

## Background & Motivation

**Background**: Hypergraph neural networks (HNNs) excel at modeling higher-order interactions in domains such as social analysis, biochemistry, and e-commerce, but large-scale hypergraph training is computationally expensive. Graph condensation (GC) compresses the original graph into a small synthetic graph while preserving downstream GNN performance. In 2025, HG-Cond extended this to hypergraphs—pretraining a Neural Hyperedge Linker (NHL) to capture higher-order connectivity via variational inference, then using GPSM to repeatedly retrain HNNs for trajectory alignment.

**Limitations of Prior Work**: HG-Cond faces two fundamental issues—(1) **Decoupled structure generation and feature optimization**: NHL is frozen during the amelioration phase, optimized only for "reconstructing the original hypergraph" and not jointly trained with synthetic features, leading to misalignment between structure and nodes and degraded downstream accuracy; (2) **Resource-intensive trajectory matching**: Each amelioration round requires retraining the HNN, and the memory cost of NHL variational pretraining makes scaling to large hypergraphs difficult.

**Key Challenge**: Simultaneously encoding "structure / features / training trajectory" in bi-level optimization inevitably requires either retraining or complex alignment losses; to maintain downstream accuracy without retraining, a lightweight signal that supervises both structure and features is needed.

**Goal**: (1) Integrate the structure generator into end-to-end optimization to avoid misalignment; (2) Find an alignment target that does not require HNN retraining; (3) Encode higher-order structural information into features at initialization, providing a strong starting point for subsequent optimization.

**Key Insight**: First, use Heat Kernel PageRank to perform low-pass spectral filtering on the original graph, "baking" multi-hop structural knowledge into node features; then, let each synthetic node act as an anchor in turn, using an MLP to learn pairwise association strengths among synthetic nodes to form differentiable sparse hyperedges; finally, use a composite loss of prototype MMD + node-level InfoNCE to preserve both global class distribution and local decision boundaries.

**Core Idea**: Replace "structure by generator + features by matching" with "structure and features jointly driven by discrimination loss," and use HKPR to fold expensive "repeated propagation" into a one-time initialization filter.

## Method

### Overall Architecture
Given a large hypergraph $\mathcal{T}=(\mathbf{X},\mathbf{H},\mathbf{Y})$, the target synthetic hypergraph $\mathcal{S}=(\mathbf{X}',\mathbf{H}',\mathbf{Y}')$ satisfies $N'\ll N, M'\ll M$. The pipeline: (1) **HKPR Initialization**—apply truncated Heat Kernel PageRank diffusion to original node features to obtain $\tilde{\mathbf{X}}$, then map to $N'$ synthetic nodes $\mathbf{X}'$ via class-wise mean pooling; (2) **Anchor-guided Hyperedge Generation**—each $\mathbf{X}'_i$ acts as an anchor, an MLP learns its association scores $\hat{\mathbf{H}}'_i$ with other synthetic nodes, and an adaptive threshold $\delta_i$ sparsifies via ReLU to yield structure $\mathbf{H}'_i$; (3) **Dual-level Discrimination Loss** aligns class prototypes (coarse) and instance geometry (fine) between synthetic and original graphs, with dynamic cos/sin weighting. Downstream, training HNN on $\mathcal{S}$ achieves accuracy close to the full graph.

### Key Designs

1. **HKPR-based Node Initialization (Baking Structural Knowledge into Features)**:

    - **Function**: Before condensation, multi-scale higher-order structural information is embedded into synthetic node features, providing a structure-aware starting point for subsequent optimization and topological signals for feature-driven hyperedge generation.
    - **Mechanism**: Define the normalized hypergraph propagation operator $\mathbf{P}=\mathbf{D}_v^{-1/2}\mathbf{H}\mathbf{D}_e^{-1}\mathbf{H}^\top\mathbf{D}_v^{-1/2}$, HKPR diffusion as $\tilde{\mathbf{X}}=\sum_{k=0}^\infty \frac{e^{-\lambda}\lambda^k}{k!}\mathbf{P}^{(k)}\mathbf{X}$, equivalent to low-pass filtering $g(\mu)=e^{-\lambda\mu}$ in the hypergraph Fourier domain (Thm 3.1). Truncate at $K=\lceil\lambda+3\sqrt{\lambda}\rceil$—Lemma 3.2 provides a Poisson tail bound ensuring exponentially decaying error. Each class's synthetic node feature is obtained by mean pooling over original nodes of the same class: $\mathbf{X}'_i=\frac{1}{|S_i|}\sum_{j\in S_i}\tilde{\mathbf{X}}_j$.
    - **Design Motivation**: Randomly initializing synthetic features forces subsequent optimization to learn structure from scratch; HKPR injects "K-hop neighborhood + global context" via one-shot low-pass filtering, providing a strong prior and denoising high-frequency noise.

2. **Anchor-guided Hyperedge Generation + Adaptive Sparsity Threshold**:

    - **Function**: Enables differentiable structure generation jointly trained with features, supporting per-edge sparsity.
    - **Mechanism**: Each synthetic node $v_i'$ acts as an anchor, and for all other synthetic nodes $j$, a shared MLP computes pairwise association: $\hat{h}'_{i,j}=\text{sigmoid}(\text{MLP}_\Phi([\mathbf{X}'_i;\mathbf{X}'_j]))$, forming the incidence vector $\hat{\mathbf{H}}'_i$; an adaptive threshold $\delta_i$ is learned for each hyperedge, and final sparsification is $\mathbf{H}'_i=\text{ReLU}(\hat{\mathbf{H}}'_i-\delta_i)$. Thus, both structure $\mathbf{H}'$ and features $\mathbf{X}'$ are differentiable under the same loss.
    - **Design Motivation**: Using a global threshold (as in HG-Cond) leads to uniform hyperedge density and reduced expressiveness; the anchor perspective lets each hyperedge be driven by a central node, aligning with the intuition that "a hypergraph is essentially a high-order motif around a node." The MLP association avoids pretraining a generator, allowing structure to co-evolve with features under the discrimination loss.

3. **Dual-level Discrimination Loss + Cos/Sin Dynamic Weighting**:

    - **Function**: Aligns the global class distribution (coarse) and intra-class instance geometry (fine) between synthetic and original data without retraining HNN, serving as the condensation objective.
    - **Mechanism**: (a) **Coarse** $\mathcal{L}_c$: Based on class prototypes $\mathbf{C}=\mathbf{Y}^\top\tilde{\mathbf{X}}, \mathbf{C}'=\mathbf{Y}'^\top\tilde{\mathbf{X}}'$, maximize cosine similarity for same-class prototypes (→1), minimize for different classes (→0); Thm 3.3 shows this is equivalent to minimizing MMD over the joint distribution of normalized features and labels; Prop 3.5 provides a class-level margin lower bound. (b) **Fine** $\mathcal{L}_f$: For each synthetic node $v_i'$, sample same-class original nodes as positives and different-class as negatives, applying an InfoNCE-style contrastive loss; Prop 3.8 shows this directly upper-bounds the probability of mis-ranking events $\Pr(\mathcal{E}_i)\leq\mathbb{E}[e^{l_i}-1]$. (c) **Dynamic Weighting** $\mathcal{L}_{Disc}^{(t)}=\cos(\frac{\pi t}{2T})\mathcal{L}_c+\sin(\frac{\pi t}{2T})\mathcal{L}_f$, with $T$ as total epochs; early stages emphasize global distribution alignment, later stages refine local decision boundaries.
    - **Design Motivation**: Coarse loss ensures global class separability but cannot refine crowded intra-class regions; fine loss resolves instances but is sensitive to anchor sampling noise if used alone. Weighted fusion theoretically optimizes both MMD and ranking margin; cos/sin scheduling implements "global → local" curriculum learning without introducing new hyperparameters.

### Loss & Training

The final condensation objective is $\min_{\mathbf{X}', \Phi, \delta}\mathcal{L}_{Disc}^{(t)}$, with no HNN retraining. Main hyperparameters are HKPR path strength $\lambda$, truncation order $K$, sample count $s$, number of negatives $N_{neg}$, and training epochs $T$. Overall time complexity is $\mathcal{O}(KM\delta_e d+T(L_\Phi N'^2 d^2+N'N_{neg}d))$, dominated by original edge count and synthetic size, far lower than trajectory matching methods requiring HNN retraining.

## Key Experimental Results

### Main Results
The authors compare SOTA HGC (HG-Cond) and several GC methods on five hypergraph benchmarks (Cora, Pubmed, DBLP-CA, Walmart, Yelp) for downstream HNN accuracy after condensation:

| Dataset | #Nodes | #Hyperedges | #Classes | Description |
|---------|--------|-------------|----------|-------------|
| Cora | 2,708 | 1,579 | 7 | co-citation |
| Pubmed | 19,717 | 7,963 | 3 | co-citation |
| DBLP-CA | 41,302 | 22,363 | 6 | co-authorship |
| Walmart | 88,860 | 69,906 | 11 | co-purchase |
| Yelp | 50,758 | 679,302 | 9 | co-occurrence |

| Method Category | Accuracy Trend | Condensation Speed |
|-----------------|---------------|-------------------|
| GC methods (Jin et al. 2022; Zheng et al. 2023; ...) direct to hypergraph | Inferior on all HG data (no higher-order modeling) | Medium |
| HG-Cond (trajectory matching + NHL) | SOTA but requires multiple HNN retrainings | Slow |
| **AHGCDD** | ≥ HG-Cond on all 5 datasets | **Up to 144× speedup** |

### Ablation Study

| Configuration | Phenomenon | Interpretation |
|---------------|------------|----------------|
| w/o HKPR (random synthetic feature initialization) | Significant drop in downstream accuracy | Structure-aware initialization is a crucial prior |
| Global threshold instead of anchor-adaptive $\delta_i$ | Homogenized structure, accuracy drop | Adaptive sparsity allows per-hyperedge density |
| Only $\mathcal{L}_c$ (coarse) | Clear inter-class separation but intra-class crowding, accuracy drop on Yelp/Walmart | Lacks local ranking signal |
| Only $\mathcal{L}_f$ (fine) | Prototype drift, unstable training | Lacks global distribution constraint |
| Fixed 50%/50% weighting | Worse than cos/sin dynamic scheduling | Curriculum learning is effective |
| GPSM (HG-Cond style) retraining | ↑↑ time cost, no significant accuracy gain | Dual-level discrimination is sufficient |
| Replace anchor generation with pretrained NHL | Accuracy drop | End-to-end optimization is key |

### Key Findings
- HKPR initialization and anchor generation provide orthogonal gains: the former delivers "structure→feature" knowledge transfer, the latter "feature→structure" end-to-end feedback—both are indispensable.
- $\lambda$ controls HKPR average diffusion steps; smaller $\lambda$ (e.g., 2-3) suits small-diameter graphs, larger $\lambda$ benefits large graphs like Pubmed/Walmart—this correlates with hypergraph spectral radius.
- Condensation efficiency gain scales with original graph size: on Yelp, up to 144× speedup, mainly because HG-Cond requires extensive trajectory matching and HNN retraining on large graphs.
- The curriculum order from coarse to fine in dual-level loss greatly stabilizes convergence; reversing the order leads to early local optima within classes.

## Highlights & Insights
- **Dual Theoretical-Methodological Proofs**: Thm 3.3 links $\mathcal{L}_c$ to MMD, Prop 3.8 links $\mathcal{L}_f$ to a mis-ranking upper bound—this "condensation loss = distribution alignment + ranking guarantee" duality is rare in GC literature and provides a template for future work.
- **HKPR Filtering Perspective**: Compressing "multi-hop propagation" into a one-shot spectral filter is a transferable technique—any initialization needing prior structural signals can use similar low-pass filtering instead of repeated message passing.
- **Anchor-based Differentiable Hyperedges**: Each node is both anchor and candidate member, naturally supporting arbitrary-arity higher-order interactions; threshold $\delta_i$ lets the optimizer decide "edge density."
- **Discriminative Loss without HNN Retraining**: This is the most valuable engineering contribution—liberating GC from "proxy task matching" to "direct discriminative alignment," making the algorithm scalable to billion-scale graphs.

## Limitations & Future Work
- The largest experimental dataset is Yelp (50K nodes / 679K hyperedges); whether the 144× speedup holds for true industrial-scale hypergraphs (millions of nodes) is unverified.
- Anchor MLP complexity $\mathcal{O}(L_\Phi N'^2 d^2)$ is quadratic in $N'$; if users require larger synthetic graphs (e.g., 1% ratio), this becomes a bottleneck.
- HKPR path strength $\lambda$ requires manual tuning or grid search; no adaptive estimation is provided, and optimal $\lambda$ varies across hypergraphs.
- Evaluation is limited to node classification; whether dual-level discrimination remains SOTA for hypergraph link prediction, subgraph classification, etc., is unknown.
- The cos/sin scheduling of dual-level loss depends on a fixed total epoch count $T$; long/short training may shift the optimal schedule.

## Related Work & Insights
- **vs HG-Cond (Gong et al. 2025)**: HG-Cond achieves high-fidelity condensation via NHL pretraining + GPSM trajectory matching but at high cost; AHGCDD makes structure generation end-to-end and replaces trajectory matching with discrimination loss, greatly accelerating while maintaining accuracy.
- **vs GCond / SFGC (Graph Condensation)**: These works handle only pairwise graphs; AHGCDD extends the approach to higher-order via anchor + adaptive sparsity, and is the first to propose the equivalence between "MMD ↔ class prototype alignment" in hypergraph condensation.
- **vs DSL / GraphSAINT (Graph Sampling)**: Sampling preserves original substructures, while AHGCDD synthesizes new graphs with greater controllability; their application scenarios differ (the former for training acceleration, the latter for inference service).
- **vs Dataset Distillation (Wang et al.; Cazenavette et al.)**: Traditional DD uses gradient/trajectory matching; AHGCDD offers an alternative—"distribution alignment + ranking guarantee" instead of trajectory matching, showing that high-quality distillation is possible without proxy tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of HKPR initialization, anchor hyperedges, and dual-level discrimination appears for the first time in HGC, each with theoretical support.
- Experimental Thoroughness: ⭐⭐⭐⭐ 5 datasets + multiple backbone HNNs + complete ablation; but lacks validation on larger graphs and different downstream tasks.
- Writing Quality: ⭐⭐⭐⭐ Rigorous formula derivations, clear ablation correspondence; Theorem 3.1/3.3 + Prop 3.5/3.8 are well-placed theoretically.
- Value: ⭐⭐⭐⭐ The combination of 144× speedup and ≥SOTA accuracy is highly practical, providing a feasible preprocessing solution for large-scale hypergraph training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] BugSweeper: Function-Level Detection of Smart Contract Vulnerabilities Using Graph Neural Networks](../../AAAI2026/graph_learning/bugsweeper_function-level_detection_of_smart_contract_vulnerabilities_using_grap.md)
- [\[ICLR 2026\] Pairwise is Not Enough: Hypergraph Neural Networks for Multi-Agent Pathfinding](../../ICLR2026/graph_learning/pairwise_is_not_enough_hypergraph_neural_networks_for_multi-agent_pathfinding.md)
- [\[AAAI 2026\] Commonality in Few: Few-Shot Multimodal Anomaly Detection via Hypergraph-Enhanced Memory](../../AAAI2026/graph_learning/commonality_in_few_few-shot_multimodal_anomaly_detection_via_hypergraph-enhanced.md)
- [\[ECCV 2024\] Fine-Grained Scene Graph Generation via Sample-Level Bias Prediction](../../ECCV2024/graph_learning/fine-grained_scene_graph_generation_via_sample-level_bias_prediction.md)
- [\[ICLR 2026\] Entropy-Guided Dynamic Tokens for Graph-LLM Alignment in Molecular Understanding](../../ICLR2026/graph_learning/entropy-guided_dynamic_tokens_for_graph-llm_alignment_in_molecular_understanding.md)

</div>

<!-- RELATED:END -->
