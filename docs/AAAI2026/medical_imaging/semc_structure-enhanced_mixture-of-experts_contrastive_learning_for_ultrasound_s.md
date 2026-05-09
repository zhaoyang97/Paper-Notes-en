---
title: >-
  [Paper Note] SEMC: Structure-Enhanced Mixture-of-Experts Contrastive Learning for Ultrasound Standard Plane Recognition
description: >-
  [AAAI 2026][Medical Imaging][Ultrasound standard plane recognition] This paper proposes the SEMC framework, which aligns shallow structural cues with deep semantic representations via a **Semantic-Structure Fusion Module (SSFM)**, and performs hierarchical contrastive learning over multi-level features through a **Mixture-of-Experts Contrastive Recognition Module (MCRM)**, thereby enhancing fine-grained discriminability for ultrasound standard plane recognition. A new liver ultrasound dataset, LP2025, is also introduced.
tags:
  - AAAI 2026
  - Medical Imaging
  - Ultrasound standard plane recognition
  - mixture of experts
  - contrastive learning
  - semantic-structure fusion
  - liver ultrasound
date: 2026-05-08
content_hash: 31e9323f23bf72ff
---

# SEMC: Structure-Enhanced Mixture-of-Experts Contrastive Learning for Ultrasound Standard Plane Recognition

**Conference**: AAAI 2026
**arXiv**: [2511.12559](https://arxiv.org/abs/2511.12559)
**Code**: [https://github.com/YanGuihao/SEMC](https://github.com/YanGuihao/SEMC)
**Area**: Medical Imaging / Ultrasound Imaging
**Keywords**: Ultrasound standard plane recognition, mixture of experts, contrastive learning, semantic-structure fusion, liver ultrasound

## TL;DR

This paper proposes the SEMC framework, which aligns shallow structural cues with deep semantic representations via a **Semantic-Structure Fusion Module (SSFM)**, and performs hierarchical contrastive learning over multi-level features through a **Mixture-of-Experts Contrastive Recognition Module (MCRM)**, thereby enhancing fine-grained discriminability for ultrasound standard plane recognition. A new liver ultrasound dataset, LP2025, is also introduced.

## Background & Motivation

1. **State of the Field**: Ultrasound standard plane recognition is critical for disease screening, organ assessment, and biometric measurement. Deep learning approaches such as SonoNet have demonstrated notable progress, yet recognition performance remains limited.
2. **Limitations of Prior Work**: (a) Ultrasound images exhibit large intra-class variation (the same plane may appear substantially different due to varying acquisition angles and probe pressure) and small inter-class variation (different planes share similar visual patterns), demanding fine-grained discrimination; (b) existing methods rely predominantly on deep semantic features while neglecting shallow structural cues (e.g., anatomical boundaries and textures), resulting in insufficient structural awareness; (c) contrastive learning constructs positive/negative pairs via data augmentation, which fails to capture the inherent fine-grained semantic differences in ultrasound images.
3. **Root Cause**: The low contrast and blurred boundaries in ultrasound images make it infeasible to distinguish similar planes using deep features alone, while shallow features, although rich in structural information, operate at a low semantic level.
4. **Paper Goals**: How can multi-scale structural information be fused to enhance the model's structural awareness? How can effective contrastive learning over multi-level features improve inter-class separability?
5. **Starting Point**: Shallow structural cues are adaptively compressed and expanded to align with deep expert features before fusion; a MoE mechanism then performs hierarchical contrastive learning over the multi-level fused features.
6. **Core Idea**: Structure-aware feature fusion + expert-guided hierarchical contrastive learning = stronger discriminability for ultrasound standard plane recognition.

## Method

### Overall Architecture

Built on a ResNet backbone, the first three blocks share parameters to extract shallow features $\{F_1, F_2, F_3\}$, while the fourth block is split into three independent deep expert branches $\{D_1, D_2, D_3\}$. The SSFM aligns and fuses shallow features with deep expert features; the resulting representations are fed into the MCRM for hierarchical contrastive learning and classification.

### Key Designs

1. **Semantic-Structure Fusion Module (SSFM)**

   - **Function**: Aligns and fuses shallow structural cues (edges, textures) with deep semantic features to enhance structural awareness.
   - **Mechanism**: Consists of two sub-modules — (a) **Adaptive Compression-Expansion Block (ACE)**: progressively downsamples and adjusts channel dimensions of shallow features $\{F_1, F_2, F_3\}$ via stride-based depthwise convolutions to match the spatial and channel dimensions of the deep features, then performs element-wise addition $M_i = F_i' + D_i$ (avoiding channel redundancy introduced by concatenation); (b) **Structure-Aware Multi-Context Block (SAMC)**: computes channel-wise adaptive attention $\mathbf{C}_i$ (GAP + GMP + FC) and spatial attention $\mathbf{S}_i$ (Mean + Max + Conv), extracts multi-receptive-field features via parallel multi-scale convolutions, and applies channel shuffle for compression and output.
   - **Design Motivation**: Shallow features preserve fine-grained structural information but are misaligned with deep features in spatial resolution and channel dimensionality. The lightweight ACE alignment avoids information loss from upsampling/downsampling; the dual channel–spatial attention in SAMC highlights anatomically relevant regions.

2. **Mixture-of-Experts Contrastive Recognition Module (MCRM)**

   - **Function**: Performs hierarchical contrastive learning over multi-level features to enhance inter-class separability, while improving recognition accuracy through MoE-based classification.
   - **Mechanism**: Comprises two branches — (a) **MoE Contrastive Branch**: among the three expert outputs $\{\mathbf{O}_1, \mathbf{O}_2, \mathbf{O}_3\}$, $\mathbf{O}_1$ serves as the anchor (query) and $\mathbf{O}_2, \mathbf{O}_3$ as positive keys; a momentum memory queue $\mathcal{Q}$ stores historical representations, and the branch jointly optimizes a supervised contrastive loss $\mathcal{L}_{sup}$ and a self-supervised contrastive loss $\mathcal{L}_{self}$; (b) **MoE Recognition Branch**: employs a Gumbel-Softmax gating mechanism to adaptively select the most relevant experts, computing a weighted fusion prediction $\mathbf{z}_{fused} = \sum_n w_n \cdot \mathbf{z}_n$ trained with cross-entropy loss. The two branches are balanced in an end-to-end manner via a sample-adaptive weight $\alpha$ predicted by an adaptive network.
   - **Design Motivation**: Positive/negative pairs generated by data augmentation cannot approximate the fine-grained semantic differences inherent to ultrasound images. Hierarchical contrastive learning over multi-expert, multi-level features directly constructs more informative contrastive pairs within a classification-relevant feature space.

3. **LP2025 Liver Ultrasound Dataset**

   - **Function**: Fills the gap in publicly available standard plane datasets for liver ultrasound.
   - **Mechanism**: Contains 9,369 high-quality clinically validated images covering 6 standard planes (first hepatic hilum, second hepatic hilum, left lobe, right lobe, left portal vein sagittal plane, and hepatorenal interface) plus non-standard planes; independently annotated by multiple senior sonographers with over 5 years of experience and subjected to multi-stage quality control.

### Loss & Training

$L_{total} = \alpha \cdot L_{moe} + (1-\alpha) \cdot L_{mc}$, where $\alpha = g(\mathbf{O})$ is dynamically adjusted by the adaptive network based on sample difficulty. $L_{mc} = L_{sup} + \lambda L_{self}$.

## Key Experimental Results

### Main Results

Results on the FPUS23 and CAMUS public datasets:

| Method | FPUS23 Acc↑ | FPUS23 F1↑ | CAMUS Acc↑ | CAMUS F1↑ |
|--------|------------|-----------|-----------|----------|
| Diffmic | 95.29 | 81.08 | 80.91 | 79.69 |
| Metaformer | 95.52 | 94.53 | 81.52 | 80.49 |
| Area | 95.20 | 94.40 | 81.59 | 80.88 |
| Supmin | 95.28 | 94.34 | 81.13 | 79.71 |
| **SEMC** | **95.78** | **95.06** | **82.13** | **80.93** |

### Ablation Study

| Configuration | Effect | Description |
|---------------|--------|-------------|
| w/o SSFM | Accuracy drop | Loss of shallow structural information |
| w/o ACE | Accuracy drop | Insufficient feature alignment |
| w/o SAMC | Accuracy drop | Loss of multi-context structural awareness |
| w/o contrastive branch | Accuracy drop | Reduced intra-class compactness / inter-class separability |
| w/o recognition branch | Accuracy drop | Reduced classification capacity |
| w/o momentum queue | Weakened contrastive learning | Insufficient negative sample pool |

### Key Findings

- The shallow structural fusion in SSFM yields the largest gains for non-dominant-class planes (i.e., planes with subtle appearance differences).
- The three-expert design outperforms single- and dual-expert variants, though further increasing the number of experts yields diminishing returns.
- Gumbel-Softmax gating outperforms simple averaging and Softmax in sample-level feature selection.
- SEMC also achieves state-of-the-art performance on the 7-class classification task using the proposed LP2025 dataset.

## Highlights & Insights

- **Shallow–Deep Multi-Granularity Fusion**: The progressive downsampling and channel adjustment in ACE constitutes an elegant feature alignment strategy that avoids the channel redundancy of naive concatenation. This design is generalizable to other medical imaging tasks requiring multi-scale feature fusion.
- **Hierarchical Contrastive Learning**: Using multi-expert outputs as anchors and positive keys produces semantically more meaningful contrastive pairs than augmentation-based approaches.
- **Adaptive Loss Balancing**: Predicting $\alpha$ via a network to automatically balance classification and contrastive losses eliminates the need for manual hyperparameter tuning.

## Limitations & Future Work

- The LP2025 dataset originates from a single hospital; cross-center generalizability remains unvalidated.
- Non-standard plane (NSP) samples substantially outnumber standard plane samples (4,626 vs. <1,100), and the class imbalance handling strategy is not sufficiently described.
- Only ResNet is used as the backbone; whether stronger backbones (e.g., Swin Transformer) yield further improvements is unexplored.
- The method is primarily designed for classification tasks; extension to plane quality assessment or standard plane detection (localization) requires additional design effort.

## Related Work & Insights

- **vs. SonoNet**: SonoNet is VGG-based and relies solely on deep features; SEMC incorporates shallow structural fusion and achieves significantly better fine-grained discrimination.
- **vs. MoCo/SimCLR**: General contrastive learning methods rely on augmentation-generated positive/negative pairs; SEMC constructs semantically richer pairs via multiple experts.
- **vs. Diffmic**: The diffusion model achieves acceptable accuracy on this task but substantially lower F1 than SEMC, demonstrating that fine-grained discrimination still requires task-specific design.

## Rating

- Novelty: ⭐⭐⭐⭐ The combined design of SSFM and MoE-based contrastive learning is innovative
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on three datasets with comprehensive ablations; cross-center experiments are absent
- Writing Quality: ⭐⭐⭐⭐ Well-structured with sufficient illustration
- Value: ⭐⭐⭐⭐ Clinically relevant for ultrasound standard plane recognition; LP2025 dataset constitutes a meaningful contribution

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] S2Drug: Bridging Protein Sequence and 3D Structure in Contrastive Representation Learning for Virtual Screening](s2drug_bridging_protein_sequence_and_3d_structure_in_contrastive_representation_.md)
- [\[CVPR 2026\] Ultrasound-CLIP: Semantic-Aware Contrastive Pre-training for Ultrasound Image-Text Understanding](../../CVPR2026/medical_imaging/ultrasound-clip_semantic-aware_contrastive_pre-training_for_ultrasound_image-tex.md)
- [\[NeurIPS 2025\] Dual Mixture-of-Experts Framework for Discrete-Time Survival Analysis](../../NeurIPS2025/medical_imaging/dual_mixture-of-experts_framework_for_discrete-time_survival_analysis.md)
- [\[AAAI 2026\] Dual-Path Knowledge-Augmented Contrastive Alignment Network for Spatially Resolved Transcriptomics](dual-path_knowledge-augmented_contrastive_alignment_network_for_spatially_resolv.md)
- [\[AAAI 2026\] MAPI-GNN: Multi-Activation Plane Interaction Graph Neural Network for Multimodal Medical Diagnosis](mapi-gnn_multi-activation_plane_interaction_graph_neural_network_for_multimodal_.md)

<!-- RELATED:END -->
