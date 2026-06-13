---
title: >-
  [Paper Note] GIIM: Graph-based Learning of Inter- and Intra-view Dependencies for Multi-view Medical Image Diagnosis
description: >-
  [AAAI 2026][Medical Imaging][Graph Neural Networks] This paper proposes GIIM, a Multi-Heterogeneous Graph (MHG)-based framework that simultaneously models intra-view dependencies among lesions and inter-view dynamic vari…
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "Graph Neural Networks"
  - "Multi-view Learning"
  - "Heterogeneous Graphs"
  - "Missing Views"
  - "Medical Image Classification"
date: 2026-05-08
content_hash: 916dd0679b54125c
---

# GIIM: Graph-based Learning of Inter- and Intra-view Dependencies for Multi-view Medical Image Diagnosis

**Conference**: AAAI 2026
**arXiv**: [2603.09446](https://arxiv.org/abs/2603.09446)  
**Code**: N/A  
**Area**: Medical Image Analysis / Multi-view Diagnosis
**Keywords**: Graph Neural Networks, Multi-view Learning, Heterogeneous Graphs, Missing Views, Medical Image Classification

## TL;DR

This paper proposes GIIM, a Multi-Heterogeneous Graph (MHG)-based framework that simultaneously models intra-view dependencies among lesions and inter-view dynamic variations via graph structures. Four missing-view representation strategies are introduced. GIIM achieves consistent and significant improvements over existing multi-view methods across three imaging modalities: liver CT, breast X-ray, and breast MRI.

## Background & Motivation

Computer-aided diagnosis (CADx) plays a critical role in medical imaging, yet existing automated systems struggle to replicate the complex reasoning processes of clinicians. Clinical diagnosis requires integrating relationships among abnormal regions across different views and time points — including tumor size, location, and spatial relationships among multiple lesions — all of which are essential for cancer diagnosis.

Existing multi-view CADx methods suffer from two fundamental limitations: (1) they ignore dependencies among multiple lesions within a single view (intra-view), such as co-occurrence patterns among multiple hepatic lesions within the same CT phase; and (2) they fail to model dynamic inter-view changes of lesions (inter-view), such as the distinct enhancement patterns a tumor exhibits across the arterial, venous, and delayed phases of CT. Current CNN- and Transformer-based methods require fixed-size inputs and cannot flexibly handle a variable number of lesions with complex connectivity. Furthermore, missing views are common in clinical practice due to technical failures, patient refusal, or protocol variations, yet existing methods lack robust mechanisms to address this.

The core starting point of this paper is to reformulate diagnosis as a relational modeling problem, leveraging the natural advantages of graph neural networks in handling variable-length inputs and complex relationships, to construct a unified framework that simultaneously captures intra-view and inter-view dependencies.

## Method

### Overall Architecture

GIIM adopts a two-stage pipeline. In the first stage, a ConvNeXt feature extractor is trained independently for each view to learn feature representations from each observation angle or time point. In the second stage, all lesion features across all views for each patient are organized into a Multi-Heterogeneous Graph (MHG), and heterogeneous message passing is applied to learn interactions among lesions and across views. Classification predictions are produced via five-layer SAGEConv.

### Key Designs

1. **Node Representation Design**:

    - Function: Two types of nodes are constructed for each lesion — single-view nodes and multi-view nodes.
    - Mechanism: A single-view node $N_{single_v}$ represents features extracted from the $v$-th view; a multi-view node $M_{multi} = \|_{v=1}^{V}(N_{single_v})$ is the concatenation of all single-view features, serving as a global summary representation for the lesion.
    - Design Motivation: Single-view nodes preserve independent diagnostic information from each view (e.g., enhancement features in a specific CT phase), while multi-view nodes aggregate global cues. This dual-layer node design enables information propagation at different levels of abstraction.

2. **Four-Type Edge Design**:

    - Function: Four types of relational edges are defined to fully model intra-lesion and inter-lesion dependencies across views.
    - Mechanism:
        - Intra-lesion cross-view edges $E_{intra}$: connect nodes of different views for the same lesion, capturing temporal dynamics (e.g., changes in enhancement patterns over time).
        - Single-view to multi-view edges $E_{s-m}$: connect single-view nodes to their corresponding multi-view summary nodes.
        - Inter-lesion same-view edges $E_{inter-s}$: connect different lesions within the same view, modeling their spatial relationships at a specific time point.
        - Inter-lesion multi-view edges $E_{inter-m}$: connect multi-view summary nodes of different lesions, capturing high-level contextual relationships.
    - Design Motivation: These four edge types cover all lesion–view relational combinations. In particular, $E_{inter-m}$ is especially beneficial for recognizing small lesions, which can leverage their distance relationships to larger tumors as additional diagnostic cues.

3. **Heterogeneous Message Passing Mechanism**:

    - Function: Aggregates information separately according to neighbor type and then fuses it to update node features.
    - Mechanism: For each node $n$ at layer $k$, aggregated features from single-view neighbors and multi-view neighbors are computed separately: $h_{N_{single}(n)}^{k} = \frac{1}{|N_{single}(n)|}\sum_{u} \mathbf{W}_{single}^{k} h_u^{k-1}$, and analogously for multi-view aggregation. The node is then updated by concatenating its own features with both aggregated representations and passing through a fully connected layer: $h_n^k = \sigma(\mathbf{W}^k \cdot \text{CONCAT}(h_n^{k-1}, h_{N_{single}}^{k}, h_{M_{multi}}^{k}))$.
    - Design Motivation: Using separate weight matrices $\mathbf{W}_{single}^k$ and $\mathbf{W}_{multi}^k$ for different neighbor types enables more fine-grained transformation learning compared to treating all neighbors uniformly.

4. **Four Missing-View Handling Strategies**:

    - **Constant (zero vector)**: Replaces missing view features with a zero vector $[0.0]^{1\times c}$, allowing the graph to automatically ignore that node and rely on other views.
    - **Learnable (trainable parameters)**: Treats missing view features as trainable parameters, optimized during training and normalized via the Frobenius norm.
    - **RAG-based (retrieval-augmented)**: Merges available view features and retrieves the most similar sample from a database, using its corresponding missing-view features as a substitute.
    - **Covariance-based (covariance similarity)**: Constructs a covariance matrix of the multi-view feature difference space and computes covariance similarity via $s_j = (\Delta^q)^T \Sigma \Delta_j$, borrowing missing features from the most similar sample.
    - Design Motivation: RAG and Covariance strategies outperform others under complete-view testing (as their generated representations are closer to the true distribution), whereas the Constant strategy performs better when all views are missing — since the zero vector explicitly signals the absence of a view to the graph, encouraging greater reliance on available nodes.

### Loss & Training

In the first stage, a standard classification loss is used to train the ConvNeXt extractor for each view independently, which is then frozen as a feature extractor. In the second stage, graphs are constructed per patient and classification loss $\mathcal{L}(Z_i, Y_i)$ is computed at the graph node level. During training, views are randomly dropped at varying missing rates $\eta$ to improve robustness.

## Key Experimental Results

### Main Results

| Dataset | Method | Acc (%) | AUC (%) |
|--------|------|---------|---------|
| Liver CT | NN-based | 75.45 | 89.09 |
| Liver CT | ML-based (LightGBM) | 73.63 | 88.00 |
| Liver CT | Attention-based | 73.41 | 88.53 |
| Liver CT | **GIIM (ours)** | **78.20** | **91.05** |
| VinDr-Mammo | NN-based | 67.48 | 82.21 |
| VinDr-Mammo | ML-based | 66.87 | 80.86 |
| VinDr-Mammo | Attention-based | 68.09 | 81.00 |
| VinDr-Mammo | **GIIM (ours)** | **71.17** | **82.54** |
| BreastDM (MRI) | NN-based | 80.85 | 87.35 |
| BreastDM (MRI) | Attention-based | 85.10 | 76.37 |
| BreastDM (MRI) | **GIIM (ours)** | **87.23** | **89.02** |

### Ablation Study (Missing-View Robustness, Liver Dataset)

| Method | η=0.0 (all-missing test) | η=0.5 | η=1.0 | η=0.0 (complete test) |
|------|---------------------|-------|-------|-------------------|
| NN-based | 70.00 | 70.23 | 72.50 | 75.45 |
| Attention-based | 67.50 | 71.36 | 72.73 | 73.41 |
| GIIM (constant) | 72.27 | 72.73 | 73.41 | 78.20 |
| GIIM (learnable) | 73.64 | 73.86 | 70.91 | 78.20 |
| GIIM (RAG-based) | 74.31 | 72.95 | 72.50 | 78.20 |
| GIIM (Covariance) | 70.91 | 72.95 | 72.73 | 78.20 |

### Key Findings
- Multi-view methods improve over single-view methods by approximately 12% in accuracy and 8.3% in AUC on the liver dataset.
- GIIM consistently outperforms all baselines across three datasets and three imaging modalities (CT / X-ray / MRI).
- Under missing-view conditions, GIIM (constant) is the most stable strategy when all views are absent, whereas RAG/Covariance perform better under complete-view testing — revealing an interesting trade-off.
- The performance gap between complete-view and missing-view testing is only approximately 4–5%, demonstrating strong tolerance of the graph-based approach to missing data.

## Highlights & Insights
- Reformulating multi-view medical image diagnosis as a relational modeling problem is a novel perspective; the four-type edge design comprehensively covers all meaningful lesion–view relationships.
- The comparative experiments on four missing-view strategies reveal a counterintuitive finding: the simple zero-vector strategy outperforms more complex retrieval-based strategies in missing-view test scenarios, suggesting that "informing the model of data absence" is more effective than "attempting to impute missing data."
- The natural capability of GNNs to handle variable-length inputs is fully exploited — regardless of how many lesions a patient has, a graph structure can be constructed naturally.

## Limitations & Future Work
- The graph structure is constructed based on handcrafted rules (same-view connections, cross-view connections, etc.), without considering finer-grained relationships such as spatial distances between lesions.
- The current approach handles only single-view missing scenarios; more complex cases involving simultaneous absence of multiple views remain underexplored.
- The liver dataset is proprietary, which limits reproducibility.
- The four missing-view strategies each have distinct trade-offs, yet no unified adaptive selection mechanism is proposed.

## Related Work & Insights
- Compared to traditional CNN-based multi-view fusion (addition / multiplication / averaging), graph-based methods can model more flexible many-to-many relationships.
- Compared to Transformer attention mechanisms, graph-based methods do not require fixed input sizes, making them better suited for clinical scenarios with variable numbers of lesions.
- The RAG-based and Covariance-based missing-view handling strategies are generalizable to other multimodal missing-data scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MAPI-GNN: Multi-Activation Plane Interaction Graph Neural Network for Multimodal Medical Diagnosis](mapi-gnn_multi-activation_plane_interaction_graph_neural_network_for_multimodal_.md)
- [\[AAAI 2026\] Rethinking Bias in Generative Data Augmentation for Medical AI: a Frequency Recalibration Approach](rethinking_bias_in_generative_data_augmentation_for_medical_ai_a_frequency_recal.md)
- [\[AAAI 2026\] PulseMind: A Multi-Modal Medical Model for Real-World Clinical Diagnosis](pulsemind_a_multi-modal_medical_model_for_real-world_clinical_diagnosis.md)
- [\[AAAI 2026\] MedEyes: Learning Dynamic Visual Focus for Medical Progressive Diagnosis](medeyes_learning_dynamic_visual_focus_for_medical_progressive_diagnosis.md)
- [\[AAAI 2026\] DW-DGAT: Dynamically Weighted Dual Graph Attention Network for Neurodegenerative Disease Diagnosis](dw-dgat_dynamically_weighted_dual_graph_attention_network_for_neurodegenerative_.md)

</div>

<!-- RELATED:END -->
