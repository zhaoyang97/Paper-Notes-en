---
title: >-
  [Paper Note] Towards Multiple Missing Values-Resistant Unsupervised Graph Anomaly Detection
description: >-
  [AAAI 2026][Object Detection][Graph Anomaly Detection] This paper proposes M2V-UGAD, the first framework to address unsupervised graph anomaly detection under simultaneous node attribute and graph topology missingness. T…
tags:
  - "AAAI 2026"
  - "Object Detection"
  - "Graph Anomaly Detection"
  - "Missing Values"
  - "Unsupervised Learning"
  - "Pseudo-Anomaly Generation"
  - "Sinkhorn Divergence"
date: 2026-05-08
content_hash: 54559e8f118501cd
---

# Towards Multiple Missing Values-Resistant Unsupervised Graph Anomaly Detection

**Conference**: AAAI 2026
**arXiv**: [2511.09917](https://arxiv.org/abs/2511.09917)  
**Code**: N/A  
**Area**: AI Security
**Keywords**: Graph Anomaly Detection, Missing Values, Unsupervised Learning, Pseudo-Anomaly Generation, Sinkhorn Divergence

## TL;DR

This paper proposes M2V-UGAD, the first framework to address unsupervised graph anomaly detection under simultaneous node attribute and graph topology missingness. Through three core mechanisms—dual-pathway independent imputation, hyperspherical latent space fusion, and pseudo-anomaly generation—the framework overcomes cross-view interference and imputation bias, consistently outperforming existing methods across 7 benchmark datasets.

## Background & Motivation

Graph Anomaly Detection (GAD) aims to identify nodes or substructures that deviate from normal patterns in graph-structured data, with applications in social network fake review detection, financial fraud detection, and other critical scenarios. Due to the extreme rarity of real anomalies and the high cost of annotation, **unsupervised GAD** has become a primary research focus.

Existing unsupervised GAD methods fall into two main categories:

**Reconstruction-based**: Train graph autoencoders to reconstruct node attributes or structure, then use reconstruction error to identify anomalies (e.g., ADA-GAD, DiffGAD).

**Contrastive learning-based**: Distinguish anomalies by measuring consistency or inconsistency between node representations and their local neighborhoods (e.g., CoLA, PREM).

**However, all existing methods implicitly assume that the input graph is complete**—i.e., the node attribute matrix is fully observed and the adjacency matrix captures all topological relationships. In practice, node attributes may be missing due to privacy constraints or collection errors, and edge relationships may be absent due to new node arrivals or unrecorded interactions.

Directly applying imputation followed by anomaly detection faces two fundamental challenges:

### Challenge 1: Cross-view Interference

When both node attributes and edges are missing simultaneously, performing attribute imputation based on an incomplete topology introduces erroneous neighborhood context; conversely, predicting missing edges based on incomplete attributes propagates reconstruction errors. The errors across the two views mutually contaminate each other, forming a vicious cycle.

### Challenge 2: Imputation Bias

Because anomalous nodes are extremely rare, imputation models are trained almost entirely on normal nodes. At inference time, missing anomalous attributes or edges are filled with typical normal values, effectively **"repairing" anomalous nodes to resemble normal ones**, thereby suppressing the anomaly signal. Paradoxically, the more accurate the imputation, the more severe the bias.

## Method

### Overall Architecture

M2V-UGAD comprises three key modules trained in a two-stage pipeline of pre-training followed by fine-tuning:

1. **Dual-pathway incomplete graph imputation**: Independently reconstructs attributes and topology to prevent cross-view contamination.
2. **Joint latent space modeling**: Fuses both views on a shared hypersphere, using Sinkhorn divergence to enforce compact normal node distributions and a reconstruction decoder to preserve semantics.
3. **Graph pseudo-anomaly generation**: Samples from the shell outside the normal region and decodes realistic pseudo-anomalous nodes and subgraphs as hard negative samples to fine-tune the decision boundary.

### Key Designs

#### 1. Dual-pathway Incomplete Graph Imputation

**Attribute reconstruction pathway**: A self-supervised MLP is used to impute missing attributes. The MLP operates independently of graph topology, remaining robust to input sparsity and avoiding the errors introduced by message passing on an incomplete topology.

$$\mathcal{L}_{feat} = \text{MSE}(\mathbf{M}_X \odot \hat{\mathbf{X}}, \mathbf{M}_X \odot \mathbf{X})$$

The MSE loss is computed only at observed attribute positions.

**Structure reconstruction pathway**: Deterministic Personalized PageRank (PPR) diffusion is applied to recover missing topology. PPR propagates information over multiple hops at a global scale to fill in missing edge information, alleviating node isolation caused by structural sparsity.

$$\mathbf{P}^{(t+1)} = \beta \tilde{\mathbf{A}} \mathbf{P}^{(t)} + (1-\beta)\mathbf{I}$$

The final augmented adjacency matrix: $\hat{\mathbf{A}} = \mathbf{A}_{obs} + \mathbf{A}_{ppr}$

**Design Motivation**: The attribute and structure pathways operate in complete isolation, fundamentally blocking cross-view error propagation. The attribute pathway does not depend on topology, and the structure pathway does not depend on attributes, achieving true decoupling.

#### 2. Joint Latent-Space Modeling

Although dual-pathway decoupling isolates the imputation processes, the inherent co-dependency between attributes and structure (neighbor attributes can inform missing features; node feature similarity can infer missing edges) must still be exploited.

**Hyperspherical fusion**: An $L$-layer GCN projects the imputed $(\hat{\mathbf{X}}, \hat{\mathbf{A}})$ into a shared latent space $\mathbf{Z}$. Sinkhorn divergence aligns the empirical distribution of $\mathbf{Z}$ to a truncated Gaussian prior $\mathcal{N}_r(\mathbf{0}, \mathbf{I})$, concentrating normal embeddings within a hypersphere of radius $r$ so that anomalies naturally reside outside.

$$\mathcal{L}_{dist} = \text{Sinkhorn}(\mathbf{Z}, \mathcal{N}_r(\mathbf{0}, \mathbf{I}))$$

**Why Sinkhorn instead of SVDD?** SVDD minimizes point-to-center distances and can cause all embeddings to collapse to a single point. Sinkhorn performs global distribution matching: it not only pulls normal nodes together but also encourages meaningful dispersion within the sphere, preventing trivial collapse.

**Semantic-preserving reconstruction**: An auxiliary MLP decoder $r_\omega$ reconstructs node attributes from embeddings:

$$\mathcal{L}_{recon} = \text{MSE}(\mathbf{M}_X \odot \tilde{\mathbf{X}}, \mathbf{M}_X \odot \mathbf{X})$$

Joint optimization of the Sinkhorn loss and reconstruction loss strikes a balance between compactness and semantic fidelity.

#### 3. Graph Pseudo-Anomaly Generation

This is the core mechanism for combating imputation bias.

**Latent space sampling**: $M$ latent vectors are sampled uniformly from an annular shell $\mathcal{S}_{shell} = \{r_a < \|z\|_2 < r_b\}$ outside the normal sphere $r$ (with $r_a=1.2r,\ r_b=2r$), ensuring that sampled points are:
- Close enough to the normal manifold—decoded features are realistic.
- Near the decision boundary—providing the most informative hard negative samples.

**Synthetic graph construction**:
1. The trained decoder maps latent vectors to node attributes $\tilde{x}^{(a)}$.
2. An intra-subgraph adjacency $\mathbf{A}^{(a)}$ for pseudo-anomalies is constructed based on cosine similarity of decoded features.
3. A block-diagonal augmented graph is formed—the pseudo-anomaly subgraph is disconnected from the real graph, avoiding spurious mixed edges:

$$\mathbf{A}_{aug} = \begin{bmatrix} \mathbf{A}_{obs} & \mathbf{0} \\ \mathbf{0} & \mathbf{A}^{(a)} \end{bmatrix}$$

**Design Motivation**: In conventional methods, anomaly signals are suppressed by imputation bias. By explicitly generating pseudo-samples situated near the normal–anomalous boundary, the model learns a well-defined decision boundary rather than being overwhelmed by the abundance of normal samples.

### Loss & Training

**Pre-training stage** (on the original incomplete graph):

$$\mathcal{L}_{pretrain} = \mathcal{L}_{dist} + \alpha \mathcal{L}_{feat} + \lambda \mathcal{L}_{recon}$$

**Fine-tuning stage** (on the augmented graph including pseudo-anomalies):

$$\mathcal{L}_{finetune} = \mathcal{L}'_{dist} + \alpha \mathcal{L}_{feat} + \lambda \mathcal{L}_{recon}$$

where the mixed Sinkhorn loss $\mathcal{L}'_{dist}$ simultaneously constrains normal embeddings inside the sphere and pseudo-anomaly embeddings within the annular shell:

$$\mathcal{L}'_{dist} = \text{Sinkhorn}(\mathbf{Z}, \mathcal{N}_r) + \text{Sinkhorn}(\mathbf{Z}_{pseudo}, \mathcal{N}_{\{r_a < \|z\| \leq r_b\}})$$

- Default hyperparameters: $\alpha=0.01,\ \lambda=0.001$; 100 epochs for each stage.
- Optimizer: Adam; learning rate selected from $\{0.0001, 0.001, 0.0005\}$.
- Inference anomaly score: $s_i = \|z_i\|_2$ (nodes farther from the origin are more anomalous).

## Key Experimental Results

### Main Results

**AUROC comparison at 30% missing rate (best imputation method + best detector vs. M2V-UGAD)**

| Dataset | Best Baseline | M2V-UGAD | Gain |
|---------|--------------|----------|------|
| Cora | 0.86 (ASD-VAE+ADA-GAD) | **0.93** | +0.07 |
| Citeseer | 0.90 (Mean/GAIN+ADA-GAD) | **0.92** | +0.02 |
| Books | 0.70 (ASD-VAE+ADA-GAD) | **0.63** | — |
| Disney | 0.78 (ASD-VAE+ADA-GAD) | **0.81** | +0.03 |
| Flickr | 0.75 (GAIN+ADA-GAD) | **0.93** | +0.18 |
| ACM | 0.82 (GAIN+ADA-GAD) | **0.92** | +0.10 |
| Reddit | 0.56 (ASD-VAE+PREM) | **0.58** | +0.02 |

Among 25 baseline combinations (imputation method × detector), M2V-UGAD achieves a significant lead on the majority of datasets.

**AUROC on Cora under varying missing rates**

| Missing Rate | Best Baseline | M2V-UGAD |
|-------------|--------------|----------|
| 10% | 0.86 | **0.93** |
| 20% | 0.85 | **0.93** |
| 30% | 0.84 | **0.93** |
| 40% | 0.83 | **0.92** |
| 50% | 0.83 | **0.92** |

M2V-UGAD's performance remains nearly constant (0.93→0.92) across missing rates from 10% to 50%, demonstrating exceptional robustness.

### Ablation Study

| Variant | Cora | Citeseer | Books | Disney |
|---------|------|----------|-------|--------|
| Full model | 0.93 | 0.92 | 0.63 | 0.81 |
| w/o $\mathcal{L}_{feat}$ | ~0.91 | ~0.90 | ~0.61 | ~0.77 |
| w/o $\mathcal{L}_{recon}$ | ~0.91 | ~0.90 | ~0.61 | ~0.77 |
| w/o attribute imputation pathway | ~0.90 | ~0.89 | ~0.53 | ~0.74 |
| w/o structure imputation pathway | ~0.83 | ~0.82 | ~0.53 | ~0.46 |
| **w/o pseudo-anomaly generation** | **~0.58** | **~0.57** | **~0.53** | **~0.60** |

- Removing pseudo-anomaly generation causes the most severe degradation (10–35%), confirming that the anti-imputation-bias mechanism is the most critical component.
- Removing the structure pathway has a larger negative impact than removing the attribute pathway, indicating that topological information is more essential for anomaly detection.
- The two regularization losses contribute more modestly but consistently.

### Key Findings

1. **Pseudo-anomaly generation is the primary driver**: Its removal causes AUROC to drop by 10–35%, confirming that imputation bias is the primary bottleneck in incomplete-graph GAD.
2. **Dual-pathway decoupling is essential**: Removing either pathway—especially the structure pathway—significantly degrades performance, validating the cross-view interference hypothesis.
3. **Insensitivity to missing rate**: AUROC fluctuates by only 1% across missing rates from 10% to 50%, demonstrating genuine robustness.
4. **Hyperparameter insensitivity**: Performance remains stable across a wide range of $\alpha,\ \lambda,\ \eta,\ r$.
5. **t-SNE visualization**: M2V-UGAD embeddings exhibit clear separation between normal and anomalous nodes, whereas baseline methods show severe overlap.

## Highlights & Insights

- **Novel problem formulation**: This is the first work to study unsupervised GAD under simultaneous attribute and topology missingness, representing a more realistic and practically important problem setting.
- **Elegant identification of and remedy for imputation bias**: By sampling pseudo-anomalies at the boundary of the latent normal region, the method cleverly transforms imputation bias into a learnable contrastive signal.
- **Theoretically grounded choice of Sinkhorn over SVDD**: Global distribution matching is superior to point-to-center distance minimization, avoiding trivial collapse.
- **Attention to detail in the block-diagonal augmented graph**: Disconnecting the pseudo-anomaly subgraph from the real graph provides contrastive signal without introducing topological noise.

## Limitations & Future Work

1. **Limitations of anomaly injection experiments**: Anomalies in 4 datasets are synthesized following the CoLA protocol (structural anomalies + contextual anomalies), which may not reflect real-world anomaly patterns.
2. **Computational efficiency is insufficiently discussed**: The computational overhead of Sinkhorn divergence (1000 iterations) and PPR diffusion may become a bottleneck on large-scale graphs.
3. **Missing mechanism assumption**: Experiments adopt Missing Completely At Random (MCAR), whereas real-world missingness may be non-random (e.g., anomalous nodes may be more likely to have missing attributes).
4. **Node-level anomaly detection only**: The framework has not been extended to edge-level or subgraph-level anomaly detection.
5. **Underperforms on the Books dataset** (0.63 vs. 0.70 for the best baseline), suggesting a potential overfitting risk on small-scale, low-dimensional datasets.

## Related Work & Insights

- This work fills a gap at the intersection of incomplete graph learning and unsupervised anomaly detection.
- The pseudo-anomaly generation idea is transferable to other detection tasks with class imbalance.
- The dual-pathway decoupling with downstream fusion paradigm offers broadly applicable insights for multi-view learning.
- The application of Sinkhorn divergence on hyperspheres provides a new perspective for latent space design in anomaly detection.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Novel problem formulation; three core mechanisms interlock elegantly with sophisticated design.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive evaluation across 7 datasets, 25 baseline combinations, multiple missing rates, and full ablation and sensitivity analyses.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, persuasive motivation, and consistent notation throughout.
- Value: ⭐⭐⭐⭐ — Addresses a practical and important problem; larger-scale validation is needed for industrial deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] RcAE: Recursive Reconstruction Framework for Unsupervised Industrial Anomaly Detection](rcae_recursive_reconstruction_framework_for_unsupervised_industrial_anomaly_dete.md)
- [\[AAAI 2026\] FDP: A Frequency-Decomposition Preprocessing Pipeline for Unsupervised Anomaly Detection in Brain MRI](fdp_a_frequency-decomposition_preprocessing_pipeline_for_unsupervised_anomaly_de.md)
- [\[AAAI 2026\] Correcting False Alarms from Unseen: Adapting Graph Anomaly Detectors at Test Time](correcting_false_alarms_from_unseen_adapting_graph_anomaly_detectors_at_test_tim.md)
- [\[ICLR 2026\] Towards Anomaly-Aware Pre-Training and Fine-Tuning for Graph Anomaly Detection](../../ICLR2026/object_detection/towards_anomaly-aware_pre-training_and_fine-tuning_for_graph_anomaly_detection.md)
- [\[AAAI 2026\] When Trackers Date Fish: A Benchmark and Framework for Underwater Multiple Fish Tracking](when_trackers_date_fish_a_benchmark_and_framework_for_underwater_multiple_fish_t.md)

</div>

<!-- RELATED:END -->
