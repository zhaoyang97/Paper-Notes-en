---
title: >-
  [Paper Note] Learnable Kernel Density Estimation for Graphs and Its Application to Graph-Level Anomaly Detection
description: >-
  [ICML 2026][Graph Learning][Graph Density Estimation] LGKDE embeds each graph as a "node distribution" using a learnable Deep MMD metric, overlays multi-scale Kernel Density Estimation (KDE) on this metric space…
tags:
  - "ICML 2026"
  - "Graph Learning"
  - "Graph Density Estimation"
  - "Kernel Density Estimation"
  - "Deep MMD"
  - "Spectral Perturbation"
  - "Graph-Level Anomaly Detection"
date: 2026-05-08
content_hash: 5bc4a0dbf9d5964e
---

# Learnable Kernel Density Estimation for Graphs and Its Application to Graph-Level Anomaly Detection

**Conference**: ICML 2026  
**arXiv**: [2505.21285](https://arxiv.org/abs/2505.21285)  
**Code**: To be confirmed  
**Area**: Graph Learning / Graph Anomaly Detection  
**Keywords**: Graph Density Estimation, Kernel Density Estimation, Deep MMD, Spectral Perturbation, Graph-Level Anomaly Detection  

## TL;DR
LGKDE embeds each graph as a "node distribution" using a learnable Deep MMD metric, overlays multi-scale Kernel Density Estimation (KDE) on this metric space, and trains end-to-end via a self-supervised contrastive signal where "original graph density is higher than its structure-aware perturbed version." This framework provides the first unified graph-level density estimation with theoretical guarantees—including consistency, convergence rates, robustness, and generalization bounds—while consistently surpassing GNN, contrastive, and one-class baselines across over ten graph anomaly detection benchmarks.

## Background & Motivation

**Background**: Mainstream graph-level anomaly detection follows two paths. The first is "Graph Kernels + KDE"—mapping graphs to similarity matrices using kernels like WL subtree, shortest path, or propagation kernels, followed by classical KDE. The second is "Deep Representation + Distance/One-class Boundary"—learning embeddings via Graph Neural Networks (GNNs) and applying proxy targets like SVDD (OCGIN), contrastive learning (CVTGAD), information bottleneck (SIGNET), or reconstruction (VAE) for anomaly scoring.

**Limitations of Prior Work**: Kernels in the first path are handcrafted with fixed bandwidths, struggling to capture both local substructures and global topology simultaneously. The second path replaces "density modeling" with "shape priors"—either forcing normal regions into a hypersphere or skipping explicit density entirely. This leads to a lack of theoretical guarantees, sensitivity to graph size heterogeneity, and a tendency to misclassify graphs that are semantically distinct but geometrically similar as normal.

**Key Challenge**: Graphs are non-Euclidean, discrete, and permutation-invariant structured data. "Density" must be sensitive to both structure and semantics while remaining isomorphism-invariant, differentiable, and trainable end-to-end. Classical kernel methods are provable but inflexible, while deep methods are flexible but lose provability; these two paths have remained separate.

**Goal**: Construct an end-to-end trainable graph density estimator $\hat f(G)$ that: (i) encodes structure and node features under permutation invariance; (ii) supports multi-scale adaptive bandwidths; (iii) possesses consistency, convergence rates, robustness, and generalization bounds; and (iv) naturally adapts to graph-level anomaly detection where the anomaly score $s(G)=-\hat f(G)$.

**Key Insight**: The authors noted that directly maximizing density for all training graphs would cause the model to collapse to a single embedding. However, by constructing a "structure-aware perturbed version" for each normal graph and using the relative difference "original density > perturbed density" as the optimization objective, one can avoid collapse and explicitly inject geometric information about "what constitutes a deviation from normal" into the KDE bandwidth and MMD metric.

**Core Idea**: Use Deep MMD to treat graphs as points in a metric space, learn a multi-scale KDE mixture density in that space, and jointly optimize embeddings, bandwidths, and mixture weights via a "density contrastive loss + dual spectral-feature perturbation." This is the first systematic attempt to truly "weave" provable KDE into GNNs.

## Method

The LGKDE framework consists of three parts: (1) Encoding each graph into a set of node embeddings via GNN and calculating inter-graph distances using Deep MMD; (2) Overlaying multi-scale Gaussian KDE on this MMD distance, with mixture weights learned via softmax; (3) Generating two types of contrastive samples—"node feature perturbation + energy spectral perturbation"—for each training graph, using the "relative density difference between original and perturbed graphs" as the sole training signal. During inference, $s(G)=-\hat f(G)$ is used, with a threshold set at the $\gamma_{TH}$ quantile (empirically 0.1) of the reference set densities.

### Overall Architecture
The input is a set of graphs $\mathcal{G}=\{G_1,\dots,G_N\}$, where each $G_i=(V_i,E_i,\mathbf{X}_i)$. A GNN $\mathrm{GNN}_{\bm\theta}$ maps the adjacency $\mathbf{A}_i$ and node features $\mathbf{X}_i$ to a node embedding matrix $\mathbf{Z}_i\in\mathbb{R}^{n_i\times d_{out}}$. Each graph is treated as an empirical distribution of $n_i$ points, and the distance to other graphs $d_{MMD}(G_i,G_j)$ is computed via Deep MMD. Finally, this distance is fed into a Gaussian KDE with a set of bandwidths $\{h_k\}_{k=1}^{M}=\{10^{-2},10^{-1},1,10,10^{2}\}$, where softmax weights $\pi_k(\bm\alpha)$ fuse the multi-scale densities. During training, only parameters $\{\bm\theta,\bm\alpha\}$ are optimized to maximize the relative difference "normal density - perturbed density."

### Key Designs

1. **Structure-Aware Dual Perturbation (Node Feature Swap + Energy Spectral Perturbation)**:
    - **Function**: Creates a set of "anomaly-like" contrastive versions $\tilde G^{(j)}$ for each normal graph $G$ to serve as the sole negative supervision.
    - **Mechanism**: Node-side—randomly select $r_{swap}|V|$ nodes and swap their features while keeping the structure intact. Structure-side—perform SVD $\mathbf{A}=\mathbf{U}\bm\Sigma \mathbf{V}^\top$, categorize singular values into high $\mathcal{S}_h$, medium $\mathcal{S}_m$, and low $\mathcal{S}_l$ based on cumulative energy thresholds $\tau_1=0.5, \tau_2=0.75$. Divide high-energy components by an adaptive ratio $r=\min(\mu_h/\mu_l, r_{max})$ to simulate "removing backbone edges" and multiply low-energy components by $r$ to simulate "adding noise edges," then reconstruct $\tilde{\mathbf{A}}=\mathbf{U}\tilde{\bm\Sigma}\mathbf{V}^\top$.
    - **Design Motivation**: Random edge addition/deletion in classical contrastive learning can destroy the core topology, making perturbed samples either "too anomalous" or "too similar." Energy spectral perturbation ensures controllable perturbation magnitude (Theorem 4.4 gives a closed-form upper bound for $\|\Delta_{\mathbf{A}}\|_2$), allowing $\tilde G$ to have a "slightly lower" density than $G$. Combined with the Lipschitz property in Theorem 4.3, this proves an upper bound on the false alarm rate—the key to embedding provability into data augmentation.

2. **Deep MMD Graph Distance**:
    - **Function**: Transforms graph distance into a differentiable, permutation-invariant, end-to-end metric that propagates to the GNN.
    - **Mechanism**: Represents graph $G_i$ as an empirical distribution of node embeddings $\{\mathbf{z}_p^{(i)}\}_{p=1}^{n_i}$. The distance is the Maximum Mean Discrepancy (MMD) over a family of Gaussian kernels $\Gamma_{emb}=\{\gamma_1,\dots,\gamma_S\}$: $d_{MMD}(G_i,G_j)=\sup_{\gamma}\left(\frac{1}{n_i^2}\sum k_\gamma(\mathbf{z}_p^{(i)},\mathbf{z}_q^{(i)})+\frac{1}{n_j^2}\sum k_\gamma(\mathbf{z}_p^{(j)},\mathbf{z}_q^{(j)})-\frac{2}{n_i n_j}\sum k_\gamma(\mathbf{z}_p^{(i)},\mathbf{z}_q^{(j)})\right)^{1/2}$, where $k_\gamma(\mathbf{u},\mathbf{v})=\exp(-\gamma\|\mathbf{u}-\mathbf{v}\|^2)$.
    - **Design Motivation**: Traditional graph kernels use fixed features or have complexity that explodes with node counts. MMD naturally averages over node counts $n_i, n_j$, is invariant to node permutations, and is differentiable. The multi-bandwidth $\sup$ automatically captures multi-scale structural differences, aligning with the requirements of downstream multi-scale KDE for "distance quality."

3. **Multi-scale KDE + Density Contrastive Loss**:
    - **Function**: Estimates $\hat f(G)$ over MMD distances and provides a unique training objective.
    - **Mechanism**: Define $\hat f(G)=\sum_{k=1}^{M}\pi_k(\bm\alpha)\phi_k(G)$, where $\pi_k(\bm\alpha)=\mathrm{softmax}(\alpha_k)$ and each component $\phi_k(G)=\frac{1}{N}\sum_i K_{KDE}(d_{MMD}(G,G_i),h_k)$. Here $K_{KDE}(d,h)=\frac{1}{C_{d_{int}}h^{d_{int}}}\exp(-\tfrac{d^2}{2h^2})$. Since MMD induces an intrinsic dimension $d_{int}=1$, the constant $C_{d_{int}}=\sqrt{2\pi}$. The loss minimizes $-\sum_{i,j}\frac{\hat f(G_i)-\hat f(\tilde G_i^{(j)})}{\hat f(G_i)}$.
    - **Design Motivation**: A single KDE bandwidth is either too smooth or too sharp when graph scales vary significantly. Softmax-mixed multi-bandwidths allow the data to determine the dominant scale without discrete selection. The relative density difference (rather than absolute density or hard negative contrast) prevents representation collapse and avoids treating perturbed graphs as a hard "anomaly class"—a fundamental difference from contrastive learning like GraphCL that treats augmentations as strict positive/negative pairs.

### Loss & Training
The final objective $\min_{\bm\theta,\bm\alpha}\mathcal{L}=-\sum_{i=1}^{N}\sum_{j=1}^{N_{pert}}\frac{\hat f(G_i)-\hat f(\tilde G_i^{(j)})}{\hat f(G_i)}$ uses denominator normalization to balance contributions from graphs of different density magnitudes. Theoretically: Theorem 4.1 establishes the consistency $\hat f\xrightarrow{p}f^\ast$ under the $L_1$ norm; Theorem 4.2 shows that with optimal bandwidth $h^\ast\sim N^{-1/(4+d_{int})}$, MISE reaches $O(N^{-4/(4+d_{int})})=O(N^{-0.8})$, matching the non-parametric minimax optimal rate. Theorems 4.3 + 4.4 + Corollary 4.5 chain MMD robustness to density robustness, proving bounded false alarm rates for controllable perturbations. Theorem 4.6 gives a generalization bound for unseen graphs, noting that $\alpha$ grows linearly only with $\sqrt{n}$, indicating insensitivity to graph size variance. Complexity is $O(L(md+nd^2)+NSn^2 d)$, reducible to $O(L(md+nd^2)+QSn^2 d)$ with $Q\ll N$ via the Nyström-like technique in Appendix E.4.3.

## Key Experimental Results

### Main Results
Evaluated on 12 graph anomaly detection benchmarks (MUTAG, PROTEINS, DD, ENZYMES, DHFR, BZR, COX2, AIDS, IMDB-B, NCI1, COLLAB, REDDIT-B) against four categories of over ten baselines (PK-SVM/iF, WL-SVM/iF, OCGIN, OCGTL, GLocalKD, iGAD, CVTGAD, SIGNET, etc.) using AUROC (%) and average ranking.

| Dataset / Metric | Ours (LGKDE) | Prev. SOTA Representative | Remarks |
|--------|------|----------|------|
| Average AUROC (12 sets) | Significantly highest | OCGIN / GLocalKD etc. | LGKDE ranked 1st on average, Top-3 on most datasets |
| MUTAG AUROC | Massive lead over PK/WL | WL-iF 65.71 | PK-SVM at 46.06 indicates manual kernel failure |
| DD AUROC | Surpasses OCGIN 79.08 | OCGIN 79.08 | Validates multi-scale KDE for large graphs |
| Synthetic ER Density | Peak aligns with Beta(2,2) $p=0.5$ | Trad. kernels struggle | Directly validates density estimation accuracy |

> Note: Original Table 1 highlights Top-3 for each dataset; LGKDE occupies Gold or Silver in most cases.

### Ablation Study
| Configuration | Key Metric Change | Explanation |
|------|---------|------|
| Full LGKDE | Baseline AUROC | MMD + Multi-scale KDE + Dual Perturbation |
| w/o Spectral Perturbation | Significant Drop | Validates necessity of structural perturbation for boundaries |
| w/o Feature Perturbation | Moderate Drop | Node semantic deviations are also critical |
| Single Bandwidth (M=1) | Drop, especially on large graphs | Multi-scale adaptation is key |
| Replace MMD with WL/PK | Large Drop, reverts to trad. kernel performance | Deep MMD is the source of expressivity |

### Key Findings
- Dual perturbation is indispensable: swapping node features alone (preserving structure) risks learning only feature noise, while spectral perturbation alone might ignore semantic differences. They complement each other to cover both "structural + semantic" anomalies.
- Bandwidth set $\{10^{-2},\dots,10^{2}\}$ spans 5 orders of magnitude; learned softmax weights vary significantly by dataset—small molecule data favors small bandwidths (structure-sensitive), while social networks favor large bandwidths (coarse-grained patterns).
- Spectral perturbation produces higher quality contrastive samples than random edge deletion/addition, consistent with the analytical control of $\|\Delta_{\mathbf{A}}\|_2$ in Theorem 4.4.
- Complexity acceleration (using $Q\ll N$ anchors) shows almost no performance loss, enabling scaling to large graph collections.

## Highlights & Insights
- The minimalist self-supervised objective "original density > perturbed density" solves three issues at once: preventing representation collapse, avoiding hard negative assumptions, and injecting provability into training—an achievement missing in GraphCL-style contrastive learning, which essentially treats OOD detection as density estimation rather than binary classification.
- Energy spectral perturbation is a rare design that links "graph operation controllability" with "theoretical Lipschitz bounds," using SVD energy ratios as a perturbation dial that directly feeds into the inequality of Theorem 4.4.
- The observation that MMD allows $d_{int}=1$ is crucial—it prevents the KDE optimal convergence rate from suffering the curse of dimensionality, which is why LGKDE can achieve a $O(N^{-0.8})$ MISE bound.
- Multi-bandwidth softmax mixing is far more elegant than manual selection and provides a diagnostic tool for the "structural scale" of a dataset, serving as an interpretable graph analysis method.

## Limitations & Future Work
- Calculating the MMD matrix for each batch is computationally heavy for extremely large graph sets (millions of graphs); online or bucketed anchor selection could be explored.
- The model currently only considers undirected graphs with continuous node features; extensions to heterogeneous, dynamic, or hypergraphs would require redefining spectral perturbation semantics.
- Perturbation parameters ($r_{max}=10$, energy thresholds $\tau_1,\tau_2$) are analyzed for sensitivity in the appendix but still require manual tuning; adaptive versions based on spectral entropy are worth exploring.
- The anomaly score $-\hat f(G)$ assumes anomalies fall in low-density regions, which might false-positive in multimodal distributions (e.g., "two normal clusters + sparse middle").

## Related Work & Insights
- **vs OCGIN / OCGTL (One-class GNN)**: These assume normal regions are hyperspheres; LGKDE uses KDE to characterize arbitrary density contours with fewer geometric priors and higher expressivity.
- **vs GLocalKD / CVTGAD (Distillation/Contrastive)**: These use proxy targets; LGKDE explicitly estimates density ratios, providing more direct interpretability and provability.
- **vs SIGNET / iGAD (Information Bottleneck/Dual Discriminator)**: These focus on interpretable subgraph discovery; LGKDE focuses on density estimation. They are complementary, as LGKDE density gradients could be used to find critical substructures.
- **vs Classical Graph Kernels + KDE**: By replacing "fixed kernels + fixed bandwidth" with "learned MMD + learned multi-scale mixture," LGKDE retains the theoretical framework of classical KDE while removing the bottleneck of handcrafted features.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First end-to-end provable KDE framework for GNNs; unique combination of spectral perturbation and density contrast.
- Experimental Thoroughness: ⭐⭐⭐⭐ 12 real datasets + synthetic graphs + ablation; 6 theorems. Only missing extremely large-scale experiments.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation-method-theory-experiment loop; rigorous derivations, though high density in the main text.
- Value: ⭐⭐⭐⭐⭐ Provides a new provable + trainable baseline for graph anomaly/OOD detection; spectral perturbation and density contrast components are independently transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] ProMoS: Generalist Graph Anomaly Detection via Prototype-Based Distillation](generalist_graph_anomaly_detection_via_prototype-based_distillation.md)
- [\[ICML 2026\] Rethinking Feature Alignment in Generalist Graph Anomaly Detection: A Relational Fingerprint-based Approach](rethinking_feature_alignment_in_generalist_graph_anomaly_detection_a_relational_.md)
- [\[AAAI 2026\] BugSweeper: Function-Level Detection of Smart Contract Vulnerabilities Using Graph Neural Networks](../../AAAI2026/graph_learning/bugsweeper_function-level_detection_of_smart_contract_vulnerabilities_using_grap.md)
- [\[ICML 2026\] Anchor-guided Hypergraph Condensation with Dual-level Discrimination](anchor-guided_hypergraph_condensation_with_dual-level_discrimination.md)
- [\[ICML 2026\] MedCoG: Maximizing LLM Inference Density in Medical Reasoning via Meta-Cognitive Regulation](medcog_maximizing_llm_inference_density_in_medical_reasoning_via_meta-cognitive_.md)

</div>

<!-- RELATED:END -->
