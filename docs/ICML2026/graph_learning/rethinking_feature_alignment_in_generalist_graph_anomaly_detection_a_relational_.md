---
title: >-
  [Paper Note] Rethinking Feature Alignment in Generalist Graph Anomaly Detection: A Relational Fingerprint-based Approach
description: >-
  [ICML 2026][Graph Learning][Generalist Graph Anomaly Detection] To address the negative transfer issue in generalist graph anomaly detection caused by "PCA alignment only unifying dimensions but not semantics…
tags:
  - "ICML 2026"
  - "Graph Learning"
  - "Generalist Graph Anomaly Detection"
  - "Relational Fingerprint"
  - "Cross-domain Alignment"
  - "SNR Recalibration"
  - "Few-shot"
date: 2026-05-08
content_hash: a605fa093b4fe975
---

# Rethinking Feature Alignment in Generalist Graph Anomaly Detection: A Relational Fingerprint-based Approach

**Conference**: ICML 2026  
**arXiv**: [2605.25429](https://arxiv.org/abs/2605.25429)  
**Code**: https://github.com/Yujingcn/REFI-GAD-code (Available)  
**Area**: Graph Learning / Graph Anomaly Detection  
**Keywords**: Generalist Graph Anomaly Detection, Relational Fingerprint, Cross-domain Alignment, SNR Recalibration, Few-shot  

## TL;DR
To address the negative transfer issue in generalist graph anomaly detection caused by "PCA alignment only unifying dimensions but not semantics," this paper uses a set of 5-dimensional "relational fingerprints" (neighborhood position/direction/global direction consistency + degree + clustering coefficient) to explicitly extract anomaly-indicative clues as cross-domain universal features. Combined with a Transformer domain-shared encoder and an SNR-guided domain-adaptive recalibration module, it achieves SOTA performance with "universal positive transfer" across 14 datasets.

## Background & Motivation

**Background**: The mainstream path of Graph Anomaly Detection (GAD) is shifting from the traditional "one-graph-one-training" approach to generalist GAD: pre-training a universal scorer $f_\theta$ on multi-source graphs and transferring it directly to target graphs via a few-shot support set without retraining. Representative methods like ARC, UNPrompt, AnomalyGFM, and IA-GGAD follow the paradigm of "aligning features first, then learning anomaly patterns."

**Limitations of Prior Work**: The authors conducted a counter-intuitive experiment: comparing the same architecture pre-trained on large-scale source graphs vs. training-free. The results showed that on 14 datasets, pre-training actually led to performance drops in most scenarios, with some representative methods even exhibiting average negative transfer. This implies that so-called "universal knowledge" is rarely learned, and generalization capability stems from architectural inductive bias rather than pre-training.

**Key Challenge**: The root cause is the inherent "feature heterogeneity" of graph data—Cora uses high-dimensional sparse bag-of-words, while YelpChi uses low-dimensional dense statistics. Existing methods use PCA/SVD for linear dimensionality reduction to force alignment, aiming to "maximally preserve the original distribution." However, this inherits semantic differences between datasets intact: t-SNE visualizations show that after PCA alignment, different datasets still form distinct clusters—dimensions are aligned, but semantics are not.

**Goal**: (i) Find a cross-domain universal and semantically consistent feature space; (ii) learn truly transferable anomaly knowledge within this feature space; (iii) retain the capability for lightweight adaptation to the target domain distribution.

**Key Insight**: The authors noticed that existing generalist GAD methods can generalize even without training, indicating that "transferable" knowledge lies in their architectural inductive biases—primarily the shared principle of "neighborhood consistency" (ARC uses ego-neighbor residuals, UNPrompt uses node-neighborhood similarity). By explicitly extracting these hand-coded biases as features, one can bypass the semantic chasm of raw features.

**Core Idea**: Replace original heterogeneous features with a cross-domain universal, low-dimensional (5D), and semantically aligned "Relational Fingerprint" (ReFi) as a unified representation. Then, train a lightweight anomaly detection head consisting of a Transformer and SNR recalibration in this fingerprint space.

## Method

ReFi-GAD splits the generalist GAD pipeline into two stages: first, compressing heterogeneous raw features into 5D semantically aligned relational fingerprints; and second, performing "domain-shared encoding + domain-adaptive recalibration + distance-based scoring" on the fingerprints.

### Overall Architecture
Input: A graph $\mathcal{G}=(\mathcal{V},\mathcal{E},\mathbf{X})$ and a few-shot support set ($k$ labeled nodes per class). Process: (1) Extract a 5D ReFi vector $\mathbf{p}_i$ for each node to form the fingerprint matrix $\mathbf{P}$; (2) Use MLP projection + context-aware Transformer to map $\mathbf{P}$ into the latent space $\mathbf{H}^{(1)}$, with an attention mask ensuring "support see each other, query only see support"; (3) The SNR-guided module calculates discriminative scores for each latent dimension based on the support set, multiplying them by the representation for domain-adaptive recalibration to obtain $\mathbf{H}$; (4) Use cosine similarity between query and support with softmax weighting to calculate the anomaly score $\tilde y_i$. Training uses BCE; at inference, the support set is used without parameter updates (training-free).

### Key Designs

1.  **5D Relational Fingerprint (ReFi) — Explicitly Encoding "Neighborhood Consistency" as Universal Features**:
    - **Function**: Compresses raw node features of any heterogeneous graph into 5 semantically unified scalars as a cross-domain universal representation.
    - **Mechanism**: Context patterns ($\text{NP}_i, \text{ND}_i, \text{GD}_i$) + structural patterns ($d_i, \text{LC}_i$). $\text{NP}_i = \frac{1}{|\mathcal{N}_i|}\sum_{v_j \in \mathcal{N}_i} \|\bar{\mathbf{x}}_i - \bar{\mathbf{x}}_j\|_2$ is neighborhood Euclidean distance; $\text{ND}_i$ is the average cosine similarity between a node and its neighbors; $\text{GD}_i = \frac{\hat{\mathbf{x}}_i \cdot \mathbf{c}_g}{\|\hat{\mathbf{x}}_i\| \|\mathbf{c}_g\|}$ measures global direction deviation; $d_i$ is degree; $\text{LC}_i = \frac{2T_i}{d_i(d_i-1)}$ is the local clustering coefficient. Before NP/ND calculation, a similarity-aware graph convolution $\bar{\mathbf{A}} = \hat{\mathbf{A}} \odot (\mathbf{X}\mathbf{X}^\top)$ is performed to weight similar neighbors, preventing GCN from "smoothing out" anomalies. Finally, a rank-based transformation $r(m_i) = \text{rank}(m_i)/n$ converts the 5 scalars into relative percentiles for cross-dataset scale unification.
    - **Design Motivation**: The semantic gap in raw features is the cause of negative transfer, but the notion of "consistency with neighbors/global" is semantically equivalent across all graphs, corresponding to the essence of anomalies—deviating from the majority. Rank transformation is more robust than z-score, avoiding feature collapse caused by scale differences.

2.  **Context-aware Transformer Encoder + Asymmetric Attention Mask**:
    - **Function**: Learns non-linear interactions between the 5D fingerprint dimensions and injects "domain context" from the support set into query node representations.
    - **Mechanism**: MLP projects $\mathbf{P}$ to $d'$ dimensions to get $\mathbf{H}^{(0)}$, concatenated into a sequence $\mathbf{Z}^{(0)} \in \mathbb{R}^{(2k+n_b) \times d'}$ [normal support, anomalous support, query batch]. A mask $m_{ij}=0$ if $j \le 2k$ or $i=j$, else $-\infty$ is applied. This allows support nodes full mutual visibility (as a stable global background) while query nodes can only attend to the support set, preventing cross-query interference within a batch.
    - **Design Motivation**: The original 5 dimensions are too low; a Transformer is needed for non-linear interactions. However, inter-dimension dependencies vary with the domain. An asymmetric mask ensures each query is "conditioned under a fixed background," equivalent to injecting few-shot information via in-context learning rather than gradient updates.

3.  **SNR-guided Domain-adaptive Dimensional Recalibration**:
    - **Function**: Online importance weighting for each latent dimension based on the discriminative signals of the current target domain's support set.
    - **Mechanism**: Estimate the normal center $\mathbf{h}_n$ (support normal + query) and anomalous center $\mathbf{h}_a$ (support anomalous) in latent space. Calculate Signal-to-Noise Ratio (SNR) per dimension $\mathbf{s} = \frac{(\mathbf{h}_a - \mathbf{h}_n)^2}{\sigma_n^2 + \epsilon}$, pass through a learnable sigmoid gate $\mathbf{m} = \sigma(\lambda \mathbf{s} + \beta)$ for weights, yielding $\mathbf{H} = \mathbf{W}'(\mathbf{H}^{(1)} \odot \mathbf{m})$. Scoring uses temperature-scaled ($\tau$) cosine similarity with softmax: $\tilde y_i = \frac{1}{2}(\sum_{S_a}\alpha_{i,j} - \sum_{S_n}\alpha_{i,j} + 1) \in [0,1]$.
    - **Design Motivation**: Although 5D fingerprints are semantically unified, the most discriminative dimension varies by domain. SNR is a classic feature selection metric (Golub 1999), and estimation via support/query fits training-free inference.

### Loss & Training
Trained only on the source domain in an episodic manner using BCE loss $\mathcal{L} = -\frac{1}{|\mathcal{Q}|}\sum_i [y_i \log \tilde y_i + (1-y_i)\log(1-\tilde y_i)]$. Each episode randomly samples a source dataset, $2k$ support nodes, and $n_b$ query nodes, forcing the model to learn context-based scoring rather than memorizing source statistics. Inference is forward-only, $\theta$ is not updated.

## Key Experimental Results

### Main Results
14 real-world datasets split into Group 1 (Cite/CS/ACM/Blog/Amz/Photo/Weibo) and Group 2 (Cora/Pubmed/Flickr/FB/Yelp/Quest/Reddit) for cross-domain training-testing. AUROC (%) is reported.

| Method | Cite | CS | Weibo | Cora | Pubmed | Yelp | Reddit | Avg Rank |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| GCN | 48.3 | 56.2 | 46.4 | 32.4 | 33.7 | 51.2 | 46.8 | 11.86 |
| CoLA | 73.8 | 66.0 | 41.1 | 66.0 | 70.1 | 52.4 | 50.6 | 8.29 |
| ARC | High | High | Mid | High | High | Mid | Mid | ~5–6 |
| **ReFi-GAD** | **Best** | **Best** | **Best** | **Best** | **Best** | **Best** | **Best** | **1st** |

Among all generalist baselines, ReFi-GAD ranks first on average and is the only method showing consistent positive pre-training gains across almost all datasets.

### Ablation Study
| Configuration | Avg AUROC | Description |
| :--- | :--- | :--- |
| Full ReFi-GAD | Highest | Full model |
| w/o ReFi (PCA alignment) | Sig. Drop | Validates ReFi is core, not just the Transformer |
| w/o SNR Recalibration | Med. Drop | Domain-adaptive weights are effective for cross-domain |
| w/o Sim-aware GCN | Low Drop | Impact of anomaly smoothing by standard GCN |
| 5D ReFi + Distance Scoring Only | High | ReFi alone carries sufficient anomaly priors |

### Key Findings
- **ReFi, not architecture, is the core contribution**: Even training-free 5D fingerprints + cosine distance outperform many baselines, indicating that "unified semantics" matters more than "model size."
- **PCA alignment failure is visible via t-SNE**: Different datasets remain clearly clustered after PCA alignment, whereas ReFi-aligned fingerprints provide strong visual evidence for "feature alignment $\neq$ semantic alignment."
- **SNR recalibration is few-shot friendly**: Stable dimension weights can be calculated from just a few support nodes, avoiding backpropagation.
- **Negative transfer is essentially feature space misalignment**: Once the feature space is replaced with semantically unified fingerprints, pre-training gains naturally become positive.

## Highlights & Insights
- **"Reverse distilling architectural inductive bias into features"**: The authors identified that existing generalist GAD methods work because of hand-coded neighborhood consistency and explicitly extracted these priors as 5D features. This "reverse distillation" is rare in transfer learning and applicable to other cross-domain tasks.
- **Rank-based transformation as a low-cost hero**: Replacing z-score with percentiles eliminates scale differences without feature collapse, being far simpler than adversarial alignment or domain-invariant learning.
- **Asymmetric attention mask formalizes ICL**: The mask design literally encodes in-context learning into the attention structure, providing an elegant solution for few-shot inference to avoid cross-query contamination.

## Limitations & Future Work
- ReFi relies on only 5 dimensions; it may struggle with anomalies depending on complex semantics (e.g., text content or temporal patterns).
- ReFi depends on topology and homophily; purely heterophilic or dynamic graphs may require redesigning fingerprints.
- Scalability for industrial-scale graphs (millions of nodes) was not fully discussed; rank transformation is $O(n \log n)$.
- The choice of these specific 5 dimensions is empirical; future work could explore information-theoretic optimized dimension search.

## Related Work & Insights
- **vs ARC (NeurIPS 2024)**: ARC implicitly learns ego-neighbor differences; ReFi-GAD explicitly extracts them. ARC is limited by PCA; ReFi-GAD uses unified fingerprints to ensure positive transfer.
- **vs UNPrompt (2024)**: UNPrompt aligns surface forms via prompts; ReFi-GAD aligns the "anomaly-indicative semantics" at the root.
- **vs SmoothGNN / CHRN**: Traditional GAD performs feature smoothness intra-domain; this work elevates similar priors to cross-domain transferable fingerprint dimensions.
- **Transferable Insight**: Perform an architectural attribution analysis to see "why existing methods work," then distill those biases into explicit features.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] ProMoS: Generalist Graph Anomaly Detection via Prototype-Based Distillation](generalist_graph_anomaly_detection_via_prototype-based_distillation.md)
- [\[ICML 2026\] Learnable Kernel Density Estimation for Graphs and Its Application to Graph-Level Anomaly Detection](learnable_kernel_density_estimation_for_graphs_and_its_application_to_graph-leve.md)
- [\[ICML 2026\] Polynomial Neural Sheaf Diffusion: A Spectral Filtering Approach on Cellular Sheaves](polynomial_neural_sheaf_diffusion_a_spectral_filtering_approach_on_cellular_shea.md)
- [\[ICML 2026\] Generative Representation Learning on Hyper-relational Knowledge Graphs via Masked Discrete Diffusion](generative_representation_learning_on_hyper-relational_knowledge_graphs_via_mask.md)
- [\[AAAI 2026\] Commonality in Few: Few-Shot Multimodal Anomaly Detection via Hypergraph-Enhanced Memory](../../AAAI2026/graph_learning/commonality_in_few_few-shot_multimodal_anomaly_detection_via_hypergraph-enhanced.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICML 2026\] ProMoS: Generalist Graph Anomaly Detection via Prototype-Based Distillation](generalist_graph_anomaly_detection_via_prototype-based_distillation.md)
- [\[ICML 2026\] Learnable Kernel Density Estimation for Graphs and Its Application to Graph-Level Anomaly Detection](learnable_kernel_density_estimation_for_graphs_and_its_application_to_graph-leve.md)
- [\[ICML 2026\] Polynomial Neural Sheaf Diffusion: A Spectral Filtering Approach on Cellular Sheaves](polynomial_neural_sheaf_diffusion_a_spectral_filtering_approach_on_cellular_shea.md)
- [\[ICML 2026\] Generative Representation Learning on Hyper-relational Knowledge Graphs via Masked Discrete Diffusion](generative_representation_learning_on_hyper-relational_knowledge_graphs_via_mask.md)
- [\[AAAI 2026\] Commonality in Few: Few-Shot Multimodal Anomaly Detection via Hypergraph-Enhanced Memory](../../AAAI2026/graph_learning/commonality_in_few_few-shot_multimodal_anomaly_detection_via_hypergraph-enhanced.md)

</div>

<!-- RELATED:END -->
