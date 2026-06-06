---
title: >-
  [Paper Note] D2Dewarp: Dual Dimensions Geometric Representation Learning Based Document Image Dewarping
description: >-
  [CVPR 2026][Self-Supervised Learning][Document Dewarping] This paper proposes D2Dewarp—the first document dewarping method that learns geometric representations from both horizontal and vertical dimensions. A UNet with d…
tags:
  - "CVPR 2026"
  - "Self-Supervised Learning"
  - "Document Dewarping"
  - "Dual Dimension"
  - "Geometric Lines"
  - "UNet"
  - "HV Fusion"
date: 2026-05-08
content_hash: aab685c3af02747d
---

# D2Dewarp: Dual Dimensions Geometric Representation Learning Based Document Image Dewarping

**Conference**: CVPR 2026
**arXiv**: [2507.08492](https://arxiv.org/abs/2507.08492)  
**Code**: [Available](https://github.com/xiaomore/D2Dewarp)  
**Area**: Self-Supervised Learning / Document Image Understanding
**Keywords**: Document Dewarping, Dual Dimension, Geometric Lines, UNet, HV Fusion

## TL;DR

This paper proposes D2Dewarp—the first document dewarping method that learns geometric representations from both horizontal and vertical dimensions. A UNet with dual decoders predicts horizontal lines (top/bottom boundaries of documents, tables, and text lines) and vertical lines (left/right boundaries) respectively. An HV Fusion Module cross-fuses features from both directions via mixed attention. The authors also introduce the DocDewarpHV dataset containing 114K images with dual-dimension annotations.

## Background & Motivation

### 1. State of the Field

Document image dewarping aims to rectify distorted or wrinkled document photographs into a flat, canonical form, serving as a critical preprocessing step for OCR and document analysis. Existing methods fall into three broad categories: (i) 3D coordinate regression-based methods (e.g., DewarpNet); (ii) 2D optical flow / displacement field-based methods (e.g., DocTr); and (iii) geometric line-based methods (e.g., RDGR), which impose geometric constraints by predicting text-line boundary curves in the document.

### 2. Limitations of Prior Work

- **Exclusive focus on horizontal lines**: Methods such as RDGR exploit only horizontal geometric lines (top/bottom boundaries of text lines), entirely neglecting structural information in the vertical direction.
- **Vertical distortion overlooked**: Vertical bending—prominent in book-spine folds, table column boundaries, and multi-column layouts—is never explicitly modeled.
- **Insufficient feature fusion**: Even when horizontal and vertical features are extracted simultaneously, the absence of an effective cross-fusion mechanism prevents full exploitation of their complementary information.
- **Missing annotations**: Existing datasets (Doc3D, DocUNet) provide no vertical line annotations, limiting the feasibility of dual-dimension learning.

### 3. Root Cause

Document distortion is inherently a two-dimensional spatial deformation, yet existing methods apply geometric constraints from only one dimension (horizontal), resulting in a fundamental information deficit.

### 4. Paper Goals

Leverage geometric structure information from both horizontal and vertical dimensions to guide document dewarping, accompanied by a corresponding dataset.

### 5. Starting Point

The paper approaches the problem from the perspective of dual-dimension geometric representation learning: separately learning structural features for horizontal and vertical lines, then fusing their complementary information via an attention mechanism to produce high-quality deformation mappings.

## Method

### Overall Architecture

D2Dewarp adopts a UNet backbone comprising three core modules:

1. **Shared Encoder**: Extracts general multi-scale features from the input document image.
2. **Dual Decoders**: Two parallel decoder branches dedicated to predicting the Horizontal Line Map (H-Line Map) and the Vertical Line Map (V-Line Map), respectively.
3. **HV Fusion Module**: Cross-fuses horizontal and vertical features at intermediate decoder layers to enhance geometric perception.

The final output is a 2D backward mapping (displacement field) that maps each pixel in the distorted image back to its corresponding position in the flattened document.

### Key Designs

#### Design 1: Dual-Decoder Architecture

- **Function**: After the shared encoder, the network branches into an H-Decoder and a V-Decoder, each predicting geometric lines in its respective direction.
- **H-Lines (Horizontal)**: Defined as the top and bottom boundary curves of horizontal structural elements (text lines, table rows, images, paragraphs), reflecting the document's bending pattern in the vertical direction.
- **V-Lines (Vertical)**: Defined as the left and right boundary curves of vertical structural elements (table columns, paragraph side margins, binding gutters), reflecting the document's bending pattern in the horizontal direction.
- **Design Motivation**: Decoupling the two directional decoders allows the model to specialize in the geometric features of each direction, preventing horizontal and vertical information from interfering with each other in a shared decoding process.

#### Design 2: HV Fusion Module

- **Function**: Fuses intermediate-layer features from the H-Decoder and V-Decoder so that each direction can provide contextual cues to the other.
- **Core Structure**:
    - **Direction-Aware Pooling**: Applies AvgPool along the X-axis on horizontal features (preserving vertical spatial information) and along the Y-axis on vertical features (preserving horizontal spatial information).
    - **Mixed Attention**: Concatenates the pooled H/V features and applies cross-attention, enabling the horizontal branch to perceive vertical structures and vice versa.
    - **Directional Self-Attention**: After fusion, X-Self Attention and Y-Self Attention are applied separately to restore spatial resolution in each direction.
    - **Sigmoid Re-weighting**: A Sigmoid gate weights the fused features and adds them back to the original decoder features.
- **Design Motivation**: Simple concatenation or addition cannot effectively model inter-directional dependencies. Direction-aware pooling first compresses the irrelevant dimension to reduce computation, while cross-attention explicitly models cross-direction correlations.

#### Design 3: DocDewarpHV Dataset

- **Scale**: 114,000 training images plus validation and test splits, at resolution 512×512.
- **Annotations**: Each image provides four types of ground truth: 3D coordinates, UV mapping, H-Line map, and V-Line map.
- **Generation**: Built upon the Blender 3D rendering engine, which maps flat documents onto curved 3D meshes and automatically extracts horizontal and vertical boundary curves.
- **Diversity**: Covers both Chinese and English documents, including plain text, tables, and mixed text-figure layouts.
- **Comparison with Doc3D**: Doc3D contains only 102K images and provides no V-Line annotations; DocDewarpHV surpasses it in both scale and annotation richness.

### Loss & Training

The total loss is a weighted sum of a reconstruction loss and a line prediction loss:

$$\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{rec}} + \mathcal{L}_{\text{line}}$$

- **$\mathcal{L}_{\text{rec}}$ (Reconstruction Loss)**: L1 loss measuring pixel-level differences between the predicted backward mapping and the ground-truth displacement field.
- **$\mathcal{L}_{\text{line}}$ (Line Prediction Loss)**: A weighted BCE loss analogous to that used in RDGR, computed separately for the H-Line and V-Line prediction maps. Positive samples (line pixels) are up-weighted to alleviate the severe class imbalance (line pixels constitute less than 5% of all pixels).
- **Weight $\alpha$**: A hyperparameter balancing the two loss terms.

Training configuration: Adam optimizer, learning rate 1e-4, batch size 16, 300 epochs. Models are trained on DocDewarpHV and evaluated on three real-world benchmarks: DocUNet, DIR300, and WarpDoc.

## Key Experimental Results

### Main Results

**Table 1: Comparison on DocUNet Benchmark (130 real distorted documents)**

| Method | MS-SSIM↑ | LD↓ | CER↓ |
|---|---|---|---|
| DewarpNet | 0.4735 | 8.39 | 0.4210 |
| DocTr | 0.5105 | 7.76 | 0.3561 |
| DocGeoNet | 0.5040 | 7.71 | 0.3806 |
| RDGR | 0.5224 | 7.61 | 0.3343 |
| RecDocNet | 0.5198 | 7.42 | 0.3482 |
| **D2Dewarp (Ours)** | **0.5387** | **7.18** | **0.3127** |

**Table 2: Comparison on DIR300 Benchmark (300 documents)**

| Method | MS-SSIM↑ | LD↓ |
|---|---|---|
| DewarpNet | 0.4868 | 8.98 |
| DocTr | 0.5241 | 7.94 |
| RDGR | 0.5356 | 7.63 |
| **D2Dewarp (Ours)** | **0.5521** | **7.28** |

**Table 3: Comparison on WarpDoc Benchmark (1,020 documents)**

| Method | MS-SSIM↑ | LD↓ | CER↓ |
|---|---|---|---|
| DocTr | 0.6842 | 5.31 | 0.1987 |
| RDGR | 0.7015 | 5.08 | 0.1842 |
| **D2Dewarp (Ours)** | **0.7234** | **4.76** | **0.1653** |

### Ablation Study

**Table 4: Ablation of core components (DocUNet Benchmark)**

| Configuration | MS-SSIM↑ | LD↓ |
|---|---|---|
| Baseline (single decoder + H-Line only) | 0.5224 | 7.61 |
| + V-Line branch (dual decoder, no fusion) | 0.5298 | 7.42 |
| + Simple concatenation fusion | 0.5315 | 7.36 |
| + HV Fusion Module (Full) | **0.5387** | **7.18** |

**Table 5: Internal ablation of HV Fusion Module**

| Configuration | MS-SSIM↑ | LD↓ |
|---|---|---|
| Without direction-aware pooling (direct cross-attention) | 0.5341 | 7.31 |
| Without Sigmoid re-weighting (direct addition) | 0.5328 | 7.35 |
| Without directional self-attention | 0.5352 | 7.29 |
| Full HV Fusion | **0.5387** | **7.18** |

### Key Findings

1. **Dual-dimension substantially outperforms single-dimension**: Adding only the V-Line branch (without fusion) already raises MS-SSIM from 0.5224 to 0.5298, confirming the importance of vertical geometric information.
2. **Fusion mechanism is critical**: The HV Fusion Module contributes an additional 0.72% MS-SSIM gain over simple concatenation, and direction-aware design outperforms naive attention.
3. **Consistent superiority across all three benchmarks**: D2Dewarp achieves state-of-the-art MS-SSIM, LD, and CER on DocUNet, DIR300, and WarpDoc simultaneously.
4. **Notable OCR improvement**: CER drops from 0.3343 (RDGR) to 0.3127 (D2Dewarp) on DocUNet—a 6.5% relative reduction—demonstrating that dewarping quality directly benefits downstream text recognition.
5. **DocDewarpHV dataset effectiveness**: Training on DocDewarpHV yields 1.2% higher MS-SSIM than training on Doc3D, attributable to V-Line annotations and larger dataset scale.

## Highlights & Insights

- **A long-overlooked but intuitive insight**: Document distortion is clearly a two-dimensional problem, yet all prior methods relied solely on horizontal lines. D2Dewarp is the first to identify and address this blind spot.
- **End-to-end dual-dimension learning**: The shared encoder + dual decoders + HV Fusion design achieves both computational efficiency and bidirectional feature complementarity.
- **Dataset contribution**: DocDewarpHV (114K images with H/V-Line annotations) is a significant community contribution that resolves the bottleneck of missing vertical line annotations.
- **Direction-aware attention design**: Compressing along the corresponding axis via pooling before applying cross-attention is more efficient than naive global attention and carries clearer geometric semantics.

## Limitations & Future Work

- The dataset is generated via 3D rendering; the resulting domain gap from real-world distributions may limit generalization.
- The definitions of H-Lines and V-Lines depend on regular document structures (text lines, tables) and may be less effective for handwritten documents or irregular layouts.
- The dual-decoder architecture increases parameter count and computational cost by approximately 40%, potentially restricting deployment on mobile or edge devices.
- HV Fusion is applied at fixed decoder layers; multi-scale fusion has not been explored.
- No comparison is made against recent Transformer-based global-attention methods (e.g., DocFormerv2).
- The paper focuses exclusively on backward mapping output and does not explore combining 3D coordinate reconstruction for richer geometric priors.

## Related Work & Insights

- **RDGR [Li et al.]**: A dewarping method based on horizontal text-line curves and the direct predecessor of D2Dewarp. This paper extends its "single-dimension" approach to "dual-dimension."
- **DewarpNet [Das et al.]**: The first end-to-end deep learning dewarping method, predicting 3D coordinate mappings. D2Dewarp demonstrates that geometric line representations are more effective than 3D coordinates.
- **DocTr [Feng et al.]**: Introduces Transformers for document dewarping, leveraging global attention to capture long-range dependencies. D2Dewarp's direction-aware attention offers a more targeted alternative.
- **Doc3D [Das et al.]**: A 100K-scale rendered dataset that serves as the standard training set in document dewarping. DocDewarpHV extends it with V-Line annotations and additional samples.
- **Insight**: The concept of directional decoupling is transferable to other document analysis tasks—layout analysis could simultaneously attend to row segmentation (horizontal) and column segmentation (vertical), and table recognition could separately model row lines and column lines.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic introduction of dual-dimension line modeling; the HV Fusion Module design is intuitive and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation on three benchmarks with full ablation studies and failure case analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, rich figures and tables, intuitive visual comparisons.
- Value: ⭐⭐⭐⭐ The DocDewarpHV dataset is a lasting contribution; the dual-dimension paradigm generalizes to other document analysis tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Text-Phase Synergy Network with Dual Priors for Unsupervised Cross-Domain Image Retrieval](text-phase_synergy_network_with_dual_priors_for_unsupervised_cross-domain_image_.md)
- [\[ICCV 2025\] A Token-level Text Image Foundation Model for Document Understanding (TokenFD/TokenVL)](../../ICCV2025/self_supervised/a_tokenlevel_text_image_foundation_model_for_document_unders.md)
- [\[ICML 2026\] The Geometric Mechanics of Contrastive Representation Learning: Alignment Potentials, Entropic Dispersion, and Cross-modal Divergence](../../ICML2026/self_supervised/the_geometric_mechanics_of_contrastive_representation_learning_alignment_potenti.md)
- [\[CVPR 2026\] Suppressing Non-Semantic Noise in Masked Image Modeling Representations](suppressing_non-semantic_noise_in_masked_image_modeling_representations.md)
- [\[CVPR 2026\] Representation Learning for Spatiotemporal Physical Systems](representation_learning_for_spatiotemporal_physica.md)

</div>

<!-- RELATED:END -->
