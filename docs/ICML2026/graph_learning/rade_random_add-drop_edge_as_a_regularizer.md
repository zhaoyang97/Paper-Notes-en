---
title: >-
  [Paper Note] RADE: Random Add-Drop Edge as a Regularizer
description: >-
  [ICML 2026][Graph Learning][GNN Regularization] RADE performs simultaneous random edge dropping and adding during GNN training, utilizing "Expectation-Preserving" aggregation correction to align training and inference. I…
tags:
  - "ICML 2026"
  - "Graph Learning"
  - "GNN Regularization"
  - "Overfitting"
  - "Over-squashing"
  - "Random Graph Augmentation"
  - "Training-Inference Alignment"
date: 2026-05-08
content_hash: 592f2c4bf7bb92b9
---

# RADE: Random Add-Drop Edge as a Regularizer

**Conference**: ICML 2026  
**arXiv**: [2606.00757](https://arxiv.org/abs/2606.00757)  
**Code**: https://github.com/Danial-sb/RADE  
**Area**: Graph Learning / GNN Regularization  
**Keywords**: GNN Regularization, Overfitting, Over-squashing, Random Graph Augmentation, Training-Inference Alignment

## TL;DR
RADE performs simultaneous random edge dropping and adding during GNN training, utilizing "Expectation-Preserving" aggregation correction to align training and inference. It further employs GradNorm to adaptively adjust the drop/add rates, allowing a single augmentation to mitigate both overfitting and over-squashing.

## Background & Motivation

**Background**: The Message Passing Neural Network (MPNN) mechanism in GNNs naturally faces two pain points: first, **overfitting**, which is shared with all deep models; and second, **over-squashing**, which is unique to MPNNs—where the receptive field expands exponentially but representation dimensions remain finite, leading to the collapse or compression of long-range information. Mature solutions exist for each: random graph augmentations (e.g., DropEdge) use perturbations for implicit regularization against overfitting, while rewiring, virtual nodes, or Graph Transformers mitigate over-squashing by adding edges.

**Limitations of Prior Work**: These two solutions are often used in isolation. DropEdge-style edge-dropping augmentations are unrelated to over-squashing (dropping edges only exacerbates bottlenecks) and suffer from **training-inference mismatch**—the sparse perturbed graph is used during training while the original graph is used during inference, leading to inconsistent aggregation expectations. Conversely, rewiring methods construct a deterministic dense graph without randomness, thus providing no regularization. Both approaches also suffer from an engineering burden: the perturbation intensity $p$ must be meticulously fine-tuned for each dataset.

**Key Challenge**: Randomness and connectivity enhancement are perceived as separate tasks, but "adding edges" can both inject randomness (→ regularization) and create shortcuts (→ mitigating over-squashing). The problem is: if the perturbed graph changes constantly during training while a fixed graph is seen during inference, the aggregation expectations do not match. This distribution shift can harm generalization.

**Goal**: (1) Design a unified graph augmentation that incorporates both the "regularization of dropping edges" and the "connectivity enhancement of adding edges"; (2) Derive explicit aggregation correction rules so that perturbed aggregation during training equals a target aggregation during inference in an expected sense; (3) Eliminate manual hyperparameter tuning for perturbation rates $p$ and $q$.

**Key Insight**: The authors characterize graph augmentation as a conditional distribution $\mathcal{Q}(\cdot \mid \mathcal{G})$ and propose "Expectation-Preserving Aggregation" as the alignment criterion: requiring $\mathbb{E}_{\mathcal{G}'}[\widetilde{\mathbf{a}}_i^{(\mathcal{G}',\ell)}] = \mathbf{a}_i^{(\mathcal{G},\ell)}$. Once this equality holds, random perturbations only contribute aggregation noise with a zero mean, falling into the "variance-type implicit regularization" framework of Fang et al.: $\mathbb{E}[\tilde{L}_{\mathrm{BCE}}] \approx L_{\mathrm{BCE}} + \tfrac{1}{2}\sum_i z_i(1-z_i)\mathrm{Var}(\delta_i)$.

**Core Idea**: Randomly add edges while dropping others, and perform explicit correction for each edge/non-edge message. The correction target can be the original graph aggregation (addressing only overfitting) or a modified aggregation with "added-edge expectation" (addressing both), thereby unifying both types of augmentation within a single view-sampling framework.

## Method

### Overall Architecture
RADE consists of two components: **perturbation** (how the perturbed graph $\mathcal{G}'$ is sampled) and **aggregation correction** (how messages are adjusted during training to align with the inference aggregation). The process is as follows: in each epoch, $\mathbf{A}'$ is obtained via Bernoulli sampling (existing edges are kept with $1-p$, non-edges are added with $q$). Weighted sum aggregation is performed on $\mathcal{G}'$ as $\widetilde{\mathbf{a}}_i^{(\mathcal{G}',\ell)} = \sum_{j\neq i} \alpha_{ij}^{\mathcal{G}'} \widetilde{\mathbf{m}}_{ij}^{(\ell-1)}$, where $\widetilde{\mathbf{m}}_{ij}$ is given by the correction rules. During inference, $\mathcal{G}$ or a modified aggregation with added-edge expectations is used. The drop/add rates themselves are updated online via GradNorm, so users do not need to specify $p$ or $q$.

### Key Designs

1. **Random Add-Drop Perturbation**:
    - **Function**: Executes edge dropping and adding simultaneously in a single perturbation to construct a graph that provides both regularization and enhanced connectivity.
    - **Mechanism**: Each element of the adjacency matrix is sampled independently—if $A_{ij}=1$, then $A_{ij}' \sim \mathrm{Bernoulli}(1-p)$; if $A_{ij}=0$, then $A_{ij}' \sim \mathrm{Bernoulli}(q)$. The matrix is symmetrized as $A_{ji}'=A_{ij}'$ with no self-loops. $q=0$ reduces to DropEdge, and $p=0$ reduces to pure random edge addition. To avoid enumerating all non-edges, $K = q|\overline{E}|$ non-edges are sampled without replacement via a hypergeometric distribution.
    - **Design Motivation**: Drop and add operations are asymmetric (Proposition 4.4 proves that Drop-only and Add-only are generally not interchangeable as they scale different node-level statistics). Thus, they are complementary primitives that must be introduced together to achieve different variance spectra and open long-range communication.

2. **Expectation-Preserving Aggregation Correction (RADE-OF / RADE-OFS)**:
    - **Function**: Converts training-inference alignment into a derivable equality constraint and provides two sets of correction rules for different alignment targets.
    - **Mechanism**: **RADE-OF** (Fixing overfitting only) makes the training aggregation equal to the original graph aggregation in expectation. Correction messages are $\widetilde{\mathbf{m}}_{ij} = \frac{\alpha_{ij}^{\mathcal{G}}}{\mathbb{E}[\alpha_{ij}^{\mathcal{G}'}]} \mathbf{m}_{ij}$ (rescaling existing edges) and $\widetilde{\mathbf{m}}_{ij} = \mathbf{m}_{ij} - \boldsymbol{\mu}_i$ (subtracting the weighted mean of non-neighbors $\boldsymbol{\mu}_i = \tfrac{1}{Z_i}\sum_{j:A_{ij}=0}\mathbb{E}[\alpha_{ij}^{\mathcal{G}'}]\mathbf{m}_{ij}$), making the expected contribution of added edges zero. **RADE-OFS** (Fixing both) corrects only the edge-drop expectation and keeps the added-edge expectation, incorporating it into the modified inference aggregation $\widehat{\mathbf{a}}_i^{(\mathcal{G},\ell)} = \mathbf{a}_i^{(\mathcal{G},\ell)} + \sum_{j:A_{ij}=0}\mathbb{E}[\alpha_{ij}^{\mathcal{G}'}]\mathbf{m}_{ij}$, effectively allowing the model to see a "soft-densified" graph during inference. Propositions 4.1/4.2 prove both rules are expectation-preserving. For GIN (sum), the expectation terms are analytically solved as $\mathbb{E}[\alpha_{ij}^{\mathcal{G}'}] = 1-p$ or $q$. For GCN (symmetric normalization) and GAT (attention), where the perturbed degree $d_i'$ enters the denominator, the authors use the delta-method expansion for approximation or empirical estimation.
    - **Design Motivation**: Existing DropEdge only satisfies expectation preservation for global rescaling of sum aggregations; it does not automatically hold for weighted aggregations. Explicit correction provides a unified framework beyond sum and makes "whether added edges are utilized during inference" depend on the choice of correction target—this is key to the coexistence of OF and OFS.

3. **GradNorm Adaptive Rate Tuning**:
    - **Function**: Adjusts $(p, q)$ online to make RADE truly "hyperparameter-free," avoiding per-dataset rate sweeps.
    - **Mechanism**: The variance-type regularization term $R(B, p, q) = \tfrac{1}{2}\sum_{i\in B} z_i(1-z_i)\mathrm{Var}(\delta_i)$ is treated as an implicit loss. The required gradient magnitude on shared parameters $\boldsymbol{\theta}$, $G_{\mathrm{reg}}^B = \|\nabla_{\boldsymbol{\theta}} R\|_2$, is matched with the supervised loss gradient magnitude $G_{\mathrm{data}}^B$. After each mini-batch, one Adam step is taken on $\mathcal{J}(p, q) = \left[\log\frac{G_{\mathrm{reg}}^B + \epsilon}{G_{\mathrm{data}}^B + \epsilon}\right]^2 + \lambda \left(\frac{q}{D(\mathcal{G})}\right)^2$. The first term performs log-scale matching (avoiding absolute difference scale issues), and the second term penalizes excessive edge addition normalized by graph density $D(\mathcal{G})$. In practice, $\rho = q/D(\mathcal{G})$ is reparameterized for optimization to prevent density explosion from small $q$.
    - **Design Motivation**: Traditional augmentation methods are highly sensitive to $p$—too weak is ineffective, too strong causes distortion, and the optimal value varies by dataset. The GradNorm approach binds "regularization strength" to gradients of the same scale as the supervised loss, providing a principled rather than empirical self-adjustment. $\lambda$ is fixed at the task level (node/graph classification) and does not introduce dataset-level hyperparameters.

### Loss & Training
Supervised loss uses standard BCE / Cross-Entropy. Augmentation is not explicitly added to the loss but implicitly contributes a variance-type regularization term (Eq. 5) via randomly sampled perturbed graphs and corrected messages. The optimizer is Adam with a learning rate of 0.001, using a 2-layer backbone (node classification) or following standard protocols for each task (graph classification). RADE is consistently initialized with $p=0.5$ and $q = D(\mathcal{G})$, followed by adaptive updates with GradNorm, requiring no sweeping.

## Key Experimental Results

### Main Results

Node Classification (GCN backbone, 8 datasets, average of 5 runs, Micro-F1 for Flickr, Accuracy % for others):

| Method | Cora | CiteSeer | Computer | Physics | Flickr | ogbn-arxiv |
|------|------|----------|----------|---------|--------|------------|
| GCN | 80.10 | 69.12 | 89.63 | 96.52 | 51.89 | 70.51 |
| Dropout | 80.76 | 70.04 | 89.71 | 96.54 | 52.02 | 70.77 |
| DropEdge | 80.28 | 69.42 | 89.79 | 96.48 | 52.08 | 70.38 |
| DropMessage | 80.65 | 67.94 | 89.06 | 96.50 | 52.23 | 70.45 |
| **RADE-OF** | **81.08** | **70.20** | **90.22** | **96.58** | **52.25** | **71.09** |
| **RADE-OFS** | 80.82 | 70.12 | 89.94 | 96.53 | 51.92 | 70.95 |

Graph Classification (GCN backbone, Accuracy % for TU, ROC-AUC % for ogbg-molhiv, AP % for peptides-func):

| Method | MUTAG | PROTEINS | IMDB-B | ogbg-molhiv | peptides-func |
|------|-------|----------|--------|-------------|---------------|
| GCN | 84.00 | 75.37 | 75.70 | 75.21 | 55.14 |
| DropEdge | 84.50 | 75.29 | 75.70 | 76.01 | 55.29 |
| DropMessage | 86.08 | 75.45 | 75.40 | 74.21 | 54.98 |
| RADE-OF | 86.16 | 75.62 | 76.20 | 76.09 | 56.12 |
| **RADE-OFS** | **86.24** | **75.84** | **76.60** | **76.28** | **56.97** |

### Ablation Study (Node Classification, RADE-OF / GCN backbone)

| Configuration | Cora | Flickr | ogbn-arxiv | Description |
|------|------|--------|------------|------|
| RADE-OF | **81.08** | **52.25** | **71.09** | Full model |
| w/o GN | 80.90 | 51.42 | 70.52 | Remove GradNorm adaptivity → Flickr drops 0.83 |
| w/o GN & EPC | 78.94 | 49.35 | 70.33 | Remove expectation-preserving correction → Cora drops 2.14, Flickr 2.90 |
| RADE-OF-Lin | 80.42 | 50.65 | 69.92 | Linearized backbone |
| RADE-OF-Lin w/o GN & EPC | 77.80 | 50.09 | 68.72 | All removed → Significant degradation across datasets |
| GCN-Lin (SGC) | 79.60 | 50.12 | 68.67 | Baseline reference |

### Key Findings
- **Expectation-Preserving Correction (EPC) makes the largest contribution**: Removing EPC causes Cora to drop ~2 points and Flickr nearly 3 points, demonstrating that training-inference alignment is not a luxury but the actual mechanism through which RADE takes effect.
- **GradNorm adaptivity outperforms fixed tuning**: On large graphs like Flickr and ogbn-arxiv, the "w/o GN" configuration consistently decreases performance by 0.5–0.8 points, while saving the cost of sweeping $p$.
- **RADE-OFS excels in long-range tasks**: On peptides-func (an LRGB long-range benchmark), RADE-OFS improves over GCN by 1.83 AP and over RADE-OF by 0.85 AP, validating the effectiveness of "preserving added-edge expectations into inference" for mitigating over-squashing.
- **Add and Drop are complementary, not interchangeable**: Proposition 4.4 provides a theoretical strong constraint for interchangeability; ablations show both "add-only" and "drop-only" are weaker than joint perturbation.

## Highlights & Insights
- **"Expectation-Preserving Aggregation" as an alignment criterion**: It serves as both an alignment criterion and a constructive principle for deriving correction rules from any weighted sum aggregation rule. It is more general than the DropEdge "global rescale" special case and explains why simple rescaling fails for GCN/GAT.
- **The dual identity of adding edges**: The authors rediscover that "random edge addition" is both a noise source (regularization) and a shortcut (connectivity). By using OF/OFS correction targets, they explicitly control which aspect is utilized during inference—this design pattern of "embedding inductive bias in correction rules" can be transferred to any weighted sum graph operator.
- **Using GradNorm for "Gradient Matching between Implicit and Explicit Loss"**: While the original GradNorm was for multi-task balancing, here it is adapted to match "supervised loss vs. variance regularization proxy," providing a clean automatic tuning paradigm for any random regularization with difficult-to-tune intensities.

## Limitations & Future Work
- The expectation term $\mathbb{E}[\alpha_{ij}^{\mathcal{G}'}]$ lacks a closed-form solution for GCN/GAT and must be approximated via the delta-method or sampling, which may amplify errors for small graphs or high-variance nodes.
- The theoretical derivation for non-linear GNNs is based on the approximation of "linear MPNN + ignoring bias introduced by non-linearity" (Footnote for Eq. 5); no strict bounds are provided for deep non-linear stacks.
- The characterization of non-interchangeability is limited to sum aggregation; the authors do not fully characterize the variance spectra achievable by joint add-drop perturbations for other common aggregations.
- In practice, each mini-batch requires GradNorm + re-sampling + EPC recalculation; the trade-off between complexity and marginal regularization gains remains to be verified on massive graphs (>10M nodes).
- The potential for combination with explicit long-range modeling like Graph Transformers or virtual nodes was not discussed. Whether RADE-OFS can generalize to longer-range tasks like PCQM-Contact beyond small LRGB datasets remains to be answered.

## Related Work & Insights
- **vs DropEdge (Rong et al., 2020)**: DropEdge only drops edges, lacks explicit expectation alignment (except where sum happens to allow global rescaling), and is unrelated to over-squashing. RADE-OF strictly encompasses DropEdge as a special case when $q=0$ and adds correction rules for general weighted aggregations.
- **vs DropMessage (Fang et al., 2023)**: DropMessage perturbs messages rather than edges, sharing the variance regularization perspective but not addressing over-squashing or allowing added-edge expectations to carry into inference as RADE-OFS does.
- **vs Rewiring / Virtual Node (Alon & Yahav, 2021; Topping et al., 2022)**: These methods perform one-time structural modifications on a deterministic graph without training randomness; RADE-OFS achieves the "added-edge expectation" indirectly through a random process, equivalent to soft rewiring while gaining regularization.
- **vs GradNorm (Chen et al., 2018)**: Transferring multi-task gradient matching to "explicit loss vs. implicit regularization" matching is the key to RADE's "hyperparameter-free" nature and provides an inspiring paradigm for other random regularization methods.

## Rating
- Novelty: ⭐⭐⭐⭐ Unifying add-drop, expectation alignment, and adaptive tuning into a clean framework with a clear theoretical basis that solves long-standing generalization issues of DropEdge.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 8 node and 6 graph classification datasets across GCN/GIN/GAT backbones; ablation studies clearly decouple GN/EPC/linearization. Lacks comparison with Graph Transformers on PCQM benchmarks.
- Writing Quality: ⭐⭐⭐⭐ Proposition statements and proofs are well-organized; the comparison table for OF and OFS (Table 1) is clear, and approximation strategies are well-cited.
- Value: ⭐⭐⭐⭐ Provides a principled and hyperparameter-free solution to the common "random perturbation vs. deterministic graph" discrepancy; the GradNorm tuning paradigm is transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Evaluating LLMs on Large-Scale Graph Property Estimation via Random Walks](../../ACL2026/graph_learning/evaluating_llms_on_large-scale_graph_property_estimation_via_random_walks.md)
- [\[AAAI 2026\] Kernelized Edge Attention: Addressing Semantic Attention Blurring in Temporal Graph Neural Networks](../../AAAI2026/graph_learning/kernelized_edge_attention_addressing_semantic_attention_blurring_in_temporal_gra.md)
- [\[ICML 2026\] T-GINEE: A Tensor-Based Multilayer Graph Representation Learning](t-ginee_a_tensor-based_multilayer_graph_representation_learning.md)
- [\[ICML 2026\] View Space: Representation Learning Across Arbitrary Graphs](view_space_learning_representation_across_arbitrary_graphs.md)
- [\[ICML 2026\] L2G-Net: Local to Global Spectral Graph Neural Networks via Cauchy Factorizations](l2g-net_local_to_global_spectral_graph_neural_networks_via_cauchy_factorizations.md)

</div>

<!-- RELATED:END -->
