---
title: >-
  [Paper Note] GTP-4o: Modality-Prompted Heterogeneous Graph Learning for Omni-Modal Biomedical Representation
description: >-
  [ECCV 2024][Medical Imaging][Multimodal Learning] The authors propose GTP-4o, an omni-modal biomedical representation learning framework based on heterogeneous graphs. It explicitly models cross-modal relationships through heterogeneous graph embeddings, utilizes a graph prompting mechanism to complete missing modalities, and designs knowledge-guided hierarchical cross-modal aggregation. It achieves SOTA on glioma grading and survival prediction tasks.
tags:
  - "ECCV 2024"
  - "Medical Imaging"
  - "Multimodal Learning"
  - "Heterogeneous Graph"
  - "Modality Missing Completion"
  - "Biomedical Representation"
  - "Graph Prompt"
date: 2026-05-08
content_hash: fa9722c42198bec2
---

# GTP-4o: Modality-Prompted Heterogeneous Graph Learning for Omni-Modal Biomedical Representation

**Conference**: ECCV 2024  
**arXiv**: [2407.05540](https://arxiv.org/abs/2407.05540)  
**Code**: Yes ([https://gtp-4-o.github.io/](https://gtp-4-o.github.io/))  
**Area**: Medical Image  
**Keywords**: Multimodal Learning, Heterogeneous Graph, Modality Missing Completion, Biomedical Representation, Graph Prompt

## TL;DR

The authors propose GTP-4o, an omni-modal biomedical representation learning framework based on heterogeneous graphs. It explicitly models cross-modal relationships through heterogeneous graph embeddings, utilizes a graph prompting mechanism to complete missing modalities, and designs knowledge-guided hierarchical cross-modal aggregation. It achieves SOTA on glioma grading and survival prediction tasks.

## Background & Motivation

### Problem Background

In biomedical diagnosis, multiple clinical modalities (genomics, pathological images, cell atlas, diagnostic texts) exist. Integrating multimodal data provides a comprehensive understanding of the patient's condition from macro, micro, and molecular levels, enabling more accurate disease diagnosis. However, extending existing multimodal learning methods to multiple clinical modalities faces two core challenges:

### Challenge 1: Huge Semantic Heterogeneity Across Modalities

In natural images, a "dog" shares similar object semantics with its sound, but the semantic relationship between genomic data and pathological images is highly ambiguous. Relationships between different modality pairs have distinct semantic attributes—for example, the image-to-genomic relationship can be abstracted as "express", the image-to-text as "depict", and the image-to-cell-atlas as "atomize". Existing methods (optimal transport, cross-attention, etc.) ignore the heterogeneity of relationships themselves in high-order spaces.

### Challenge 2: Modality Missingness is Pervasive in Clinical Practice

Due to privacy concerns, ethical considerations, and data acquisition constraints, certain modalities are frequently missing in clinical scenarios (e.g., about 40% of patients in the TCGA-GBMLGG dataset lack RNA-Seq data). Most multimodal methods assume data completeness; once a modality is missing, multimodal fusion fails.

### Motivation

These observations inspire the authors to introduce a unified non-Euclidean representation—a heterogeneous graph—to explicitly capture the heterogeneous attributes of multimodal features (nodes) and cross-modal relationships (edges), and adaptively complete the feature spaces of missing modalities through a graph prompting mechanism.

## Method

### Overall Architecture

The GTP-4o framework consists of four steps: (1) data processing and feature extraction yield embeddings for the four modalities; (2) modality features are transformed into a heterogeneous graph embedding $\mathcal{G}=\{\mathcal{V}, \mathcal{E}, \mathcal{A}, \mathcal{R}\}$; (3) modality-prompted completion $g_\phi(\mathcal{G})$ handles missing modalities; (4) knowledge-guided hierarchical aggregation $\mathcal{M}$ fuses cross-modal information, which is finally fed into task heads for prediction.

### Key Designs

#### 1. **Heterogeneous Graph Embedding**

**Function**: Maps multimodal features into a unified heterogeneous graph space, explicitly distinguishing different modality nodes and cross-modal relationships.

**Mechanism**: Defines a heterogeneous graph $\mathcal{G}=\{\mathcal{V}, \mathcal{E}, \mathcal{A}, \mathcal{R}\}$, where the node attribute set $\mathcal{A}=\{G, I, C, T\}$ corresponds to genomics, pathology image, cell atlas, and text, and the edge relationship attribute set $\mathcal{R}=\{\text{express}, \text{depict}, \text{atomize}, \text{intra-modal}\}$ is defined by domain knowledge. The node mapping function is $\tau(v)=a \in \mathcal{A}$, and the edge mapping function is $\varphi(e)=r \in \mathcal{R}$.

**Design Motivation**: Unlike traditional methods that project all modalities into the same Euclidean space, a heterogeneous graph can explicitly encode semantic differences at both the node and edge levels, laying a foundation for subsequent selective cross-modal information interaction.

#### 2. **Modality-Prompted Completion**

**Function**: When a modality is missing, a graph prompting mechanism generates hallucinated nodes to complete the graph embedding, so that the missing representation approximates the complete representation.

**Mechanism**: Includes two levels of prompting:

- **General Prompting**: For a missing modality $M_\varnothing$, distribution priors are extracted from the features of non-missing samples of this modality, initializing $N_P$ prompt nodes $\mathcal{V}^P \in \mathbb{R}^{N_P \times d}$ through Gaussian sampling.
- **Entity-dependent Prompting**: Introduces a prompt bank $\mathcal{V}^{P_B} \in \mathbb{R}^{N_B \times d}$. Each prompt node $v^P$ generates weights $w$ via a linear mapping and softmax, weighted-combining components from the prompt bank:

$$v^P \leftarrow v^P + \sum_{i=1}^{N_B} w_i \cdot v_i^{P_B}$$

After completion, the node and edge sets are updated to restore the graph topology and relationships disrupted by the missing modality.

**Design Motivation**: Different samples of the same modality share similar distributions, allowing modality priors from existing samples to guide the completion of missing ones. Introducing a prompt bank renders the completion context-aware, avoiding static, sample-independent completion.

#### 3. **Knowledge-guided Hierarchical Aggregation**

**Function**: Performs global neighborhood discovery and local multiple-relation feature aggregation on the completed heterogeneous graph.

**Mechanism**: Divided into global and local levels:

- **Global Meta-path Neighbouring**: Employs domain knowledge to define a set of meta-paths $\Phi$, such as $G \xrightarrow{\text{express}} I \xrightarrow{\text{atomize}} C$, representing a 2-hop path from gene to image via "expressing" and then to cell atlas via "atomizing". A random walk strategy searches for the optimal meta-path in the semantic relationship space.
- **Local Multi-Relation Aggregation**: For each target node $v_t$, Multi-Head Attention (MHA) aggregates information over its meta-path neighborhood $\mathcal{N}^{\Phi}_{v_t}$. The attention score considers both node and edge features:

$$\text{SHA}(e,j) = \frac{v_s^{K,j} \cdot e^K_{v_s \to v_t} \cdot v_t^{Q,j}}{\sqrt{d}}$$

Finally, graph-level features are obtained via modality-specific pooling and mean readout layers.

**Design Motivation**: Meta-paths encode domain knowledge (gene expression $\to$ image phenotype $\to$ cell morphology) as channels for information propagation, achieving semantically meaningful cross-modal interaction rather than naïve fully-connected aggregation.

### Loss & Training

- Standard Negative Log-Likelihood (NLL) loss is used: $\min_{\mathcal{M},\mathcal{H},\phi} \mathbb{E}_{\mathcal{G}} \mathcal{L}(\mathcal{H}^{\mathcal{T}} \circ \mathcal{M} \circ \phi(\mathcal{G}), y^{\mathcal{T}})$
- The learning rate for the graph aggregator and task head is $1 \times 10^{-3}$, while the learning rate for prompt parameters is lower at $2 \times 10^{-4}$.
- Graph embedding dimension $d=512$, number of prompt nodes and prompt bank components are $N_P=N_B=5$.
- Data augmentation: Random edge/node dropping, adding Gaussian noise to node/edge features.
- Adam optimizer, 150 epochs, early stopping.

## Key Experimental Results

### Main Results

Evaluating glioma grading (AUC/ACC) and survival prediction (C-Index) on two cancer datasets, TCGA-GBMLGG and TCGA-KIRC.

| Method | Modalities | GBMLGG Grade AUC | GBMLGG Grade ACC | GBMLGG Survival C-Idx | KIRC Survival C-Idx |
|------|------|-------|-------|--------|--------|
| SNN | G | 0.8527 | 0.6583 | 0.7974 | 0.6639 |
| TransMIL | I | 0.9149 | 0.7683 | 0.8017 | 0.6876 |
| HEAT | I | 0.9289 | 0.8057 | 0.8223 | 0.7059 |
| Pathomic | G+I | 0.9172 | 0.7618 | 0.8101 | 0.7152 |
| MCAT | G+I | 0.9288 | 0.7929 | 0.8274 | 0.7235 |
| **GTP-4o** | G+I | **0.9256** | **0.8036** | **0.8296** | **0.7273** |
| TransFusion | G+I+C+T | 0.9245 | 0.7986 | 0.8296 | 0.7289 |
| **GTP-4o** | **G+I+C+T** | **0.9389** | **0.8126** | **0.8351** | **0.7336** |

When using all four modalities, GTP-4o reaches a glioma grading AUC of 0.9389 (1.44% higher than TransFusion) and a survival prediction C-Index of 0.8351.

### Ablation Study

Ablating components on TCGA-GBMLGG (using all four modalities):

| Configuration | Grade AUC | Grade ACC | Survival C-Idx | Description |
|------|---------|---------|-----------|------|
| w/o Heterogeneous Graph Embedding | 0.9232 | 0.8030 | 0.8168 | Degenerates to a homogeneous graph, performance significantly drops |
| w/o Heterogeneous Relations | 0.9259 | 0.8048 | 0.8201 | Edges lack heterogeneous attribute differentiation |
| Zero initialization for missing | 0.9087 | 0.7875 | 0.7946 | No completion, filled directly with zero, worst performance |
| Direct dropping for missing | 0.9288 | 0.8061 | 0.8233 | Dropping patients with missing modalities |
| w/o Prompt Bank | 0.9275 | 0.8081 | 0.8280 | Using only general prompting |
| w/o Knowledge-guided Aggregation | 0.9350 | 0.8071 | 0.8342 | Using random meta-paths |
| **Full GTP-4o** | **0.9389** | **0.8126** | **0.8416** | All components |

### Key Findings

1. **Modality Missingness Handling Differs Significantly**: Direct zero initialization (AUC 0.9087) vs. graph prompt completion (AUC 0.9389) shows a 3.02% gap, demonstrating that handling missingness is crucial.
2. **Modalities Contribute Differently to Tasks**: Genomics is more valuable for survival prediction, while pathology images are more effective for glioma grading.
3. **Performance Continues to Improve as Modalities Increase**: A monotonic increasing trend is observed from dual modalities to quad modalities.
4. **Visual Verification of Completion Quality**: The edge similarity pattern of the completed graph is highly consistent with the original complete graph.

## Highlights & Insights

1. **First to Introduce Heterogeneous Graphs to Multimodal Biomedical Representation**: Unlike simple concatenation or attention-based fusion, it explicitly models cross-modal semantic relationships (express/depict/atomize), making cross-modal interactions more meaningful.
2. **Novel Graph Prompt Completion**: Adapts prompt learning concepts to graph completion, utilizing learnable hallucinated nodes to adaptively compensate for missing information during training.
3. **Meta-paths Embed Domain Knowledge**: Biological causal paths (e.g., "gene $\to$ image $\to$ cell") are encoded as graph info propagation paths.

## Limitations & Future Work

1. **Noise in Synthesized Text Descriptions**: Since the dataset lacks real clinical reports, MiniGPT-4 was used to generate text descriptions, potentially introducing noise.
2. **Small Dataset Scales**: Only validated on two sub-datasets of TCGA (769 and 417 patients), requiring verification on larger-scale datasets for generalization.
3. **Tabular Data Excluded**: Clinical metadata such as age and gender are not incorporated.
4. **Computational Overhead**: Heterogeneous graph construction and meta-path search increase computational complexity; efficiency statistics were not reported.

## Related Work & Insights

- **PathomicFusion**: Foundational work in dual-modal fusion of genomics and pathological images.
- **MCAT**: Multimodal fusion introducing cross-attention, but does not model relationship heterogeneity.
- **HEAT**: Employs heterogeneous graphs within pathological images, but does not address cross-modal settings.
- **Insight**: The graph prompt completion method can scale to more scenarios with missing modalities (e.g., missing sequences in multimodal MRI).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of heterogeneous graphs and graph prompt completion is pioneering in biomedical multimodal learning.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive ablation study, modality combination analysis, and missing rate analysis; however, evaluated on only two datasets.
- **Writing Quality**: ⭐⭐⭐⭐ — Logically clear with well-articulated biological motivations, though the notation system is occasionally verbose.
- **Value**: ⭐⭐⭐⭐ — Handling missing modalities addresses a genuine pain point in clinical settings, though the generalizability of the approach requires further verification in more domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] OmniFM: Toward Modality-Robust and Task-Agnostic Federated Learning for Heterogeneous Medical Imaging](../../CVPR2026/medical_imaging/omnifm_toward_modality-robust_and_task-agnostic_federated_learning_for_heterogen.md)
- [\[ECCV 2024\] Improving Medical Multi-modal Contrastive Learning with Expert Annotations](improving_medical_multi-modal_contrastive_learning_with_expert_annotations.md)
- [\[ICCV 2025\] SimMLM: A Simple Framework for Multi-modal Learning with Missing Modality](../../ICCV2025/medical_imaging/simmlm_a_simple_framework_for_multi-modal_learning_with_missing_modality.md)
- [\[ICLR 2026\] sleep2vec: Unified Cross-Modal Alignment for Heterogeneous Nocturnal Biosignals](../../ICLR2026/medical_imaging/sleep2vec_unified_cross-modal_alignment_for_heterogeneous_nocturnal_biosignals.md)
- [\[ECCV 2024\] Energy-induced Explicit Quantification for Multi-modality MRI Fusion](energy-induced_explicit_quantification_for_multi-modality_mri_fusion.md)

</div>

<!-- RELATED:END -->
