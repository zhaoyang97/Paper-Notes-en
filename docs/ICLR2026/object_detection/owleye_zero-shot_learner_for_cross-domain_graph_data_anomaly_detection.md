---
title: >-
  [Paper Note] OwlEye: Zero-Shot Learner for Cross-Domain Graph Data Anomaly Detection
description: >-
  [ICLR 2026][Object Detection][graph anomaly detection] This paper proposes OwlEye, a framework that aligns heterogeneous graph embeddings into a shared space via pairwise-distance-statistics-based cross-domain feature al…
tags:
  - "ICLR 2026"
  - "Object Detection"
  - "graph anomaly detection"
  - "zero-shot learning"
  - "cross-domain feature alignment"
  - "dictionary learning"
  - "continual learning"
date: 2026-05-08
content_hash: cf96c53d60088f65
---

# OwlEye: Zero-Shot Learner for Cross-Domain Graph Data Anomaly Detection

**Conference**: ICLR 2026
**arXiv**: [2601.19102](https://arxiv.org/abs/2601.19102)  
**Code**: N/A  
**Area**: Other
**Keywords**: graph anomaly detection, zero-shot learning, cross-domain feature alignment, dictionary learning, continual learning

## TL;DR

This paper proposes OwlEye, a framework that aligns heterogeneous graph embeddings into a shared space via pairwise-distance-statistics-based cross-domain feature alignment, extracts attribute-level and structure-level normal patterns from multiple graphs into an extensible dictionary, and detects anomalous nodes in unseen graphs under fully zero-shot conditions through a truncated attention-based reconstruction mechanism. OwlEye achieves an average AUPRC of 36.17% across 8 datasets, surpassing the strongest baseline ARC by approximately 5.4 percentage points.

## Background & Motivation

**Background**: Graph anomaly detection (GAD) is widely applied in financial fraud detection, network intrusion detection, and social network misinformation identification. Traditional methods follow a "one model for one dataset" paradigm, training independently on each graph, with notable progress achieved by DOMINANT, SLGAD, TAM, and CARE. More recently, ARC and UNPrompt have pioneered "one-for-all" universal detection frameworks that aim to train a single model and directly transfer it to unseen graphs.

**Limitations of Prior Work**: Cross-domain universal detection faces three core challenges. First, graph features across different domains are completely heterogeneous in dimension and semantics—node features in citation networks are text embeddings, whereas those in social networks are user profile attributes; simple PCA/SVD dimensionality reduction followed by normalization cannot maintain semantic consistency. Second, existing universal frameworks are static and do not support incremental integration of new graph knowledge after training—each newly added training graph requires retraining from scratch. Third, methods such as ARC assume a small number of labeled nodes in the target graph for few-shot learning during inference, yet anomaly annotation is extremely costly in practice and requires domain expertise.

**Key Challenge**: Through visualization experiments, the authors reveal specific failure modes of existing methods. ARC's cross-domain processing tends to push different graphs apart in feature space rather than aligning them (clearly visible in t-SNE visualizations where clusters from two graphs are separated), which directly contradicts the goal of cross-domain alignment. Although UNPrompt's normalization can merge the distributions of two graphs, it severely disrupts critical distance patterns—on the Weibo dataset, the distance density of Normal–Normal pairs is greater than that of Normal–Anomaly pairs in the original feature space (an important signal for distinguishing normal from anomalous nodes), but this relationship is reversed after UNPrompt's processing, effectively erasing the anomaly detection signal.

**Goal**: (1) How to unify the feature spaces of heterogeneous graphs without destroying semantic patterns? (2) How to design a knowledge accumulation mechanism that supports continual learning and can be extended on the fly? (3) How to reliably detect anomalies under a fully unlabeled zero-shot setting?

**Key Insight**: The authors observe that the pairwise distance distribution between node pairs is an invariant that can be preserved during normalization, and that normal behavioral patterns can be shared across different graphs—provided an appropriate alignment strategy is used. This observation motivates a solution based on "learning a dictionary of normal patterns + detecting anomalies via reconstruction error."

**Core Idea**: Align features across domains using a scaling factor derived from median pairwise distances; store normal patterns in a dual-branch attribute/structure dictionary; and achieve truly zero-shot detection via truncated attention that filters potentially anomalous support nodes.

## Method

### Overall Architecture

OwlEye's pipeline consists of three sequentially cascaded modules. The input is a collection of labeled training graphs $\mathcal{T}_{train}$ from multiple domains, and the output is an anomaly score for each node in an unseen test graph. The overall flow is: (1) the cross-domain feature alignment module maps heterogeneous features from all graphs into a shared space of uniform dimensionality while preserving pairwise distance patterns; (2) the multi-domain multi-pattern dictionary learning module extracts representative normal node patterns at both the attribute level and structure level from each training graph, storing them in two dictionaries $\text{Dict}_H$ and $\text{Dict}_R$; (3) the truncated attention-based reconstruction module reconstructs node representations in the test graph using normal patterns from the dictionary—normal nodes are reconstructed accurately while anomalous nodes incur large reconstruction errors, which serve as anomaly scores. The training phase optimizes a reconstruction loss combined with a triplet contrastive loss; the inference phase requires only a single forward pass and no labeled data whatsoever.

### Key Designs

1. **Cross-Domain Feature Alignment**:

    - Function: Unify graph features of varying dimensionality and semantics into a shared $d$-dimensional space, while ensuring that normalization does not destroy the pairwise distance patterns distinguishing normal from anomalous nodes.
    - Mechanism: The procedure consists of two steps. First, PCA reduces the $d_i$-dimensional features of each graph to a uniform $d$ dimensions. Second, a critical cross-domain normalization is applied: the average L2 norm $N^i$ of all nodes in graph $i$ is computed, along with the average pairwise distances $\text{dist}^i$ and $\text{dist}_N^i$ before and after normalization. Using the median distances $\text{dist}^{\text{med}}$ and $\text{dist}_N^{\text{med}}$ across all training graphs, a scaling factor $f = \sqrt{\frac{\text{dist}^{\text{med}} \cdot \text{dist}_N^i}{\text{dist}^i \cdot \text{dist}_N^{\text{med}}}}$ is computed, and the final normalization is $\tilde{X}^i \leftarrow \frac{\tilde{X}^i}{N^i} \cdot \max(f, \tau)$, where $\tau=1$ is a lower-bound temperature. The median is chosen over the mean to prevent any single extreme graph (with unusually large inter-node distances) from dominating the global statistics.
    - Design Motivation: Direct comparison with ARC and UNPrompt motivates this design—ARC causes different graphs to be pushed apart in t-SNE space, while UNPrompt reverses the Normal–Normal vs. Normal–Anomaly distance density relationship on Weibo. This module aims to align distributions while preserving these critical distance patterns.

2. **Multi-Domain Multi-Pattern Dictionary Learning**:

    - Function: Extract representative normal patterns along both the attribute and structure dimensions from training graphs, store them in an extensible dictionary, and support inference and continual learning.
    - Mechanism: A dual-branch GNN extracts features. The attribute branch takes the aligned features $\tilde{X}^i$ as input, passes them through multiple GCN layers to obtain $H_{\text{attr}}^{i,l}$, and concatenates multi-hop residual information $H^i = [H_{\text{attr}}^{i,2} - H_{\text{attr}}^{i,1}, \ldots, H_{\text{attr}}^{i,l+1} - H_{\text{attr}}^{i,1}]$ to capture node content semantics. The structure branch replaces all input features with an all-ones vector $\mathbf{1} \in \mathbb{R}^{n_i \times d}$ and learns a purely topological representation $R^i$ using separate GNN weights, thereby excluding interference from attribute information. For each training graph, $n_{sup}=2000$ nodes are randomly sampled, and their dual-branch representations are stored in dictionaries $\text{Dict}_H^j$ and $\text{Dict}_R^j$. Cross-graph similarity is computed solely based on structure-level representations: $\text{sim}(\mathcal{G}^i, \text{Dict}_R^j) = \max(\text{softmax}(R^i W_1 (R^j[\text{idx}])^T))$.
    - Design Motivation: The key reason for using only structure-level representations for cross-graph similarity matching is that camouflaged anomalous nodes deliberately mimic the attribute features of normal nodes; using attribute-level representations for matching would assign high similarity scores to these camouflaged nodes, allowing them to evade detection. The most significant engineering advantage of the dictionary design is its support for continual learning—patterns from new graphs can simply be extracted via a forward pass and appended to the dictionary without any parameter updates or retraining.

3. **Truncated Attention-Based Reconstruction**:

    - Function: Reconstruct node representations in the test graph using normal patterns from the dictionary, detect anomalies via reconstruction error, and filter out potentially anomalous nodes that might be mistakenly selected as "normal support nodes" through the truncation mechanism.
    - Mechanism: Attention scores between queries (test graph nodes) and keys (dictionary patterns) are computed as $\alpha = \sqrt{\frac{(W^Q H^i)(W^K (H^j)^T)}{\sqrt{ld}}}$. The critical step is truncation: the scores of the $k$ patterns with the lowest attention scores are set to $-\infty$, driving their softmax contributions to zero. The intuition is that patterns with low attention scores likely correspond to anomalous nodes (since anomalous nodes are dissimilar to normal patterns). The truncated attention-weighted dictionary patterns are then fused with similarity weights via Hadamard product to yield the final reconstruction $\hat{H}^i = \frac{1}{m} \sum_{j=1}^{m} \text{sim}(\mathcal{G}^i, \text{Dict}_H^j) \odot (\alpha_H^{ij} \text{Dict}_H^j)$. The structure level is handled analogously.
    - Design Motivation: In the zero-shot setting, no labels are available, and a naive approach would randomly sample "pseudo support nodes" from the test graph as normal references. However, since anomalous nodes objectively exist (albeit at a low ratio), random sampling inevitably includes anomalous nodes, contaminating the reconstruction reference. The truncation mechanism forms a self-filtering safety net—low attention score → dissimilar to known normal patterns → likely anomalous → excluded—with an extremely low temperature $\tau_a = 0.001$ set to amplify attention score gaps.

### Loss & Training

The total loss is $\mathcal{L} = \mathcal{L}_{\text{triplet}} + \mathcal{L}_{\text{recon}}$:

- **Reconstruction Loss $\mathcal{L}_{\text{recon}}$**: For attribute-level representations, this loss maximizes the cosine similarity between normal nodes and their reconstructions ($\frac{H_{v_j}^i (\hat{H}_{v_j}^i)^T}{|H_{v_j}^i||\hat{H}_{v_j}^i|}$) and minimizes the cosine similarity for anomalous nodes. The intent is to enable accurate reconstruction of normal nodes from dictionary patterns.
- **Triplet Loss $\mathcal{L}_{\text{triplet}}$**: Considering both the attribute and structure dimensions, for each pair (anomalous node $v_j$, normal node $v_k$), it computes $\max(\|\hat{H}_{v_j}^i - H_{v_j}^i\|^2 - \|\hat{H}_{v_j}^i - \hat{H}_{v_k}^i\|^2 + \lambda, 0)$, with margin $\lambda = 0.2$ and structure branch weight $\beta = 0.01$. The triplet loss provides additional pairwise contrastive signals to enhance discriminability.

During inference, the anomaly score is $\mathcal{S}_{v_j} = \|\hat{H}_{v_j}^i - H_{v_j}^i\|^2 + \beta \|\hat{R}_{v_j}^i - R_{v_j}^i\|^2$, integrating reconstruction errors from both the attribute and structure dimensions.

## Key Experimental Results

### Main Results: Zero-Shot AUPRC (%) Comparison

| Dataset | OwlEye | ARC | CARE | UNPrompt | DOMINANT | Notes |
|--------|--------|-----|------|----------|----------|------|
| Cora | 43.94±0.46 | **45.20**±1.08 | 35.12±0.23 | 9.84±2.90 | 31.77±0.34 | ARC slightly better; ours second |
| Flickr | **37.69**±0.25 | 35.13±0.20 | 25.64±0.16 | 25.21±1.84 | 28.76±1.52 | Large margin over ARC (+2.56) |
| ACM | 39.75±0.13 | 39.02±0.08 | 37.76±0.35 | 11.18±1.67 | 32.49±4.97 | Methods closely matched |
| BlogCatalog | **34.99**±0.31 | 33.43±0.15 | 25.06±0.10 | 18.24±13.05 | 29.51±3.44 | Consistently outperforms all baselines |
| Facebook | 5.62±1.17 | 4.25±0.47 | 5.52±0.34 | 4.32±0.55 | 3.42±0.86 | All methods perform poorly (hard dataset) |
| Weibo | 60.90±0.21 | **64.18**±0.68 | 40.70±0.74 | 20.58±5.62 | 29.63±0.86 | ARC stronger on social networks |
| Reddit | 4.25±0.11 | 4.20±0.25 | 3.17±0.17 | 3.77±0.32 | 3.28±0.37 | All low (severely imbalanced data) |
| Amazon | **62.20**±3.18 | 20.48±6.89 | 56.76±1.44 | 9.41±2.69 | 36.80±8.37 | ARC collapses; ours far ahead |
| **Average (8 datasets)** | **36.17**±0.73 | 30.74±1.23 | 28.72±0.44 | 12.82±3.58 | 24.46±3.11 | **+5.43 vs ARC** |

### Ablation Study & Continual Learning Analysis

| Configuration | Key Metric | Notes |
|---------|---------|------|
| OwlEye (full model) | Highest average AUPRC | All three modules cooperate |
| OwlEye-N (no feature normalization) | Average AUPRC decreases | Feature alignment is critical for cross-domain generalization |
| OwlEye-S (no structure branch) | Average AUPRC decreases | Structure and attribute information are complementary |
| OwlEye-T (standard attention replacing truncation) | Average AUPRC decreases | Truncation improves zero-shot robustness |
| Dictionary $n_{sup}$=10 → 200 | 35.46 → 36.01 | +0.55%; larger dictionary is better |
| Dictionary $n_{sup}$=200 → 2000 | 36.01 → 36.17 | Diminishing returns; 200 nearly saturates |
| Continual learning: add 0→3 auxiliary graphs (no retraining) | 31.29 → 32.27 | Improves without any gradient updates |
| Continual learning: add 3 auxiliary graphs (with retraining) | 31.33 | Worse than not retraining; training fails to converge |

### Key Findings

- **Feature alignment is critical for cross-domain generalization**: The OwlEye-N ablation demonstrates that removing cross-domain normalization leaves distributional gaps between different-domain graphs that GNNs cannot bridge on their own; the pairwise-distance-based alignment is therefore the cornerstone of the framework.
- **ARC collapses on Amazon** (20.48 vs. 62.20), exposing the fatal consequence of pushing different graphs apart rather than aligning them—when the distribution gap between the test graph and training graphs is large, ARC's in-context learning fails completely.
- **Dictionary-based continual learning outperforms fine-tuning**: The comparison between Case Study 1 and Case Study 2 shows that directly appending new patterns to the dictionary (without modifying parameters) outperforms fine-tuning model parameters on new graphs, as training on more graphs makes convergence harder.
- **Marginal returns on dictionary size**: Performance increases from 35.46 to 36.01 as $n_{sup}$ grows from 10 to 200, and from 36.01 to 36.17 as it grows from 200 to 2000, indicating that a small number of representative patterns is sufficient to capture the main distribution of normal behavior.
- **Advantage persists in the 10-shot setting**: Even when baselines are given 10 labeled nodes, OwlEye's average AUPRC (36.73) still surpasses all methods equipped with 10-shot information (ARC: 31.68, CARE: 30.74).

## Highlights & Insights

- **Dictionary-based continual learning** is the most significant engineering contribution of this work. Traditional methods require full retraining whenever new training data are added, whereas OwlEye only needs a single forward GNN pass over the new graph to extract patterns and append them to the dictionary, incurring zero additional training cost. This design philosophy is transferable to any scenario based on "pattern matching + anomaly detection" (e.g., temporal anomaly detection, log anomaly detection).
- **The dual-branch design's use of all-ones input to extract purely structural features** is particularly elegant. By eliminating attribute information, the GNN is forced to learn representations solely from the topological structure encoded in the adjacency matrix, yielding purely structural embeddings in an end-to-end manner—superior to prior works that rely on hand-crafted structural features such as degree or clustering coefficients.
- **The self-filtering mechanism of truncated attention** resolves the problem of "pseudo support sets potentially being contaminated by anomalies" in the zero-shot setting. Setting an extremely low temperature $\tau_a = 0.001$ makes the attention distribution extremely sharp, so that a small number of high-similarity patterns dominate the reconstruction while the contribution of the majority of patterns approaches zero, effectively isolating potential anomalies.
- **Using the median rather than the mean to compute pairwise distance statistics for cross-domain alignment** is a seemingly minor but critically important design choice. When some training graphs contain large numbers of high-dimensional nodes (leading to extremely large average pairwise distances), the mean would allow the scaling factor to be dominated by extreme graphs; the median provides a stable global reference.

## Limitations & Future Work

- **All methods perform poorly on Facebook and Reddit** (AUPRC of only 4–7%), indicating that there remains substantial room for improvement on certain specific domains. The authors do not analyze in depth why these two datasets are difficult—whether due to extremely low anomaly ratios or because anomalous patterns are hard to distinguish from normal ones.
- **Dictionary patterns are randomly sampled**, with no guarantee of representativeness. Clustering methods such as k-medoids could be used to select more representative dictionary atoms, potentially achieving better performance with a smaller dictionary.
- **Training still requires labeled data**: Although inference is zero-shot, training still requires normal/anomaly labels for every node to compute the triplet loss and reconstruction loss. Adopting a fully unsupervised training paradigm (e.g., a pure reconstruction objective) would broaden the range of applicable scenarios.
- **The structure and attribute branches share the same GNN architecture and depth**, without separate optimization for the characteristics of each signal type. Structure information may require deeper aggregation to capture global topological patterns, whereas attribute information may already be sufficient at shallower layers.
- **Efficiency on large-scale graphs is not analyzed**: Pairwise distance computation has complexity $O(n^2)$, which is infeasible for graphs with millions of nodes. Sampling or approximation strategies would need to be introduced.

## Related Work & Insights

- **vs. ARC**: ARC encodes higher-order affinity and heterophily information through in-context learning but handles cross-domain features crudely, tending to push graphs from different domains apart rather than aligning them. OwlEye's feature alignment module directly addresses this problem. ARC's AUPRC on Amazon is only 20.48 (vs. OwlEye's 62.20), revealing its fragility under large distribution shifts.
- **vs. UNPrompt**: UNPrompt predicts attributes via generalized neighborhood prompts and uses the result as anomaly scores, but its normalization reverses critical pairwise distance patterns. Its average AUPRC across 8 datasets is only 12.82, far below OwlEye's 36.17.
- **vs. CARE**: CARE is the strongest unsupervised baseline (28.72) and performs affinity-based detection. OwlEye's dictionary reconstruction paradigm provides a more explicit notion of a "normal pattern library" and supports continual learning, a capability that CARE lacks.

## Rating

- Novelty: ⭐⭐⭐⭐ The combined design of three modules is innovative, especially the dictionary-based continual learning and truncated attention ideas, although individual components (PCA alignment, GNN encoding, attention-based reconstruction) are not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐ The evaluation covers 8 test datasets, 3 ablation groups, 3 case studies, and visualization analysis, though efficiency comparisons on large-scale graphs are absent.
- Writing Quality: ⭐⭐⭐⭐ Motivation is compellingly argued (Figure 1 intuitively visualizes the failure modes of existing methods); mathematical derivations are clear.
- Value: ⭐⭐⭐⭐ Zero-shot cross-domain graph anomaly detection is a highly practical capability, and the engineering value of dictionary-based continual learning is substantial.

- Performance remains low on datasets such as Facebook and Reddit (AUPRC < 7%), indicating that cross-domain transfer is still difficult in certain domains.
- PCA projection discards high-dimensional information; more advanced dimensionality alignment methods may be needed.
- The dictionary size $n_{sup}$ is fixed at 2000 without in-depth analysis of its impact on performance.
- Efficiency on large-scale graphs (millions of nodes) is not evaluated.

## Related Work & Insights

OwlEye is directly compared against universal GAD methods such as ARC and UNPrompt. The dictionary learning idea draws from sparse coding, and truncated attention has conceptual connections to top-k attention. Key insight: introducing a "knowledge base" paradigm into graph learning enables elegant continual learning.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of cross-domain alignment and dictionary-based continual learning is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 8 test datasets and 3 case studies.
- Writing Quality: ⭐⭐⭐⭐ Problem-driven structure with clearly delineated modules.
- Value: ⭐⭐⭐⭐ Provides a practical zero-shot solution for universal graph anomaly detection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Remedying Target-Domain Astigmatism for Cross-Domain Few-Shot Object Detection](../../CVPR2026/object_detection/remedying_target-domain_astigmatism_for_cross-domain_few-shot_object_detection.md)
- [\[CVPR 2026\] Learning Multi-Modal Prototypes for Cross-Domain Few-Shot Object Detection](../../CVPR2026/object_detection/learning_multi-modal_prototypes_for_cross-domain_few-shot_object_detection.md)
- [\[ICLR 2026\] FSOD-VFM: Few-Shot Object Detection with Vision Foundation Models and Graph Diffusion](fsod-vfm_few-shot_object_detection_with_vision_foundation_models_and_graph_diffu.md)
- [\[ICLR 2026\] Towards Anomaly-Aware Pre-Training and Fine-Tuning for Graph Anomaly Detection](towards_anomaly-aware_pre-training_and_fine-tuning_for_graph_anomaly_detection.md)
- [\[ICLR 2026\] Dual Distillation for Few-Shot Anomaly Detection](dual_distillation_for_few-shot_anomaly_detection.md)

</div>

<!-- RELATED:END -->
