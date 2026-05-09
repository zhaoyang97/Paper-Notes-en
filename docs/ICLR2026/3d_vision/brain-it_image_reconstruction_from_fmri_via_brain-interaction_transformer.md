---
title: >-
  [Paper Note] Brain-IT: Image Reconstruction from fMRI via Brain-Interaction Transformer
description: >-
  [ICLR 2026][3D Vision][fMRI brain decoding] This paper proposes Brain-IT, a framework that employs a brain-inspired Brain Interaction Transformer (BIT) to cluster functionally similar brain voxels into cross-subject shared Brain Tokens, from which localized semantic and structural image features are predicted, enabling high-fidelity reconstruction of images from fMRI signals. With only 1 hour of data, Brain-IT achieves performance comparable to prior methods trained on 40 hours of data.
tags:
  - ICLR 2026
  - 3D Vision
  - fMRI brain decoding
  - image reconstruction
  - Brain-Interaction Transformer
  - cross-subject transfer
  - diffusion models
  - Deep Image Prior
date: 2026-05-08
content_hash: c4a08f9dbc620626
---

# Brain-IT: Image Reconstruction from fMRI via Brain-Interaction Transformer

**Conference**: ICLR 2026
**arXiv**: [2510.25976](https://arxiv.org/abs/2510.25976)
**Code**: [Project Page](https://amitzalcher.github.io/Brain-IT/)
**Area**: 3D Vision
**Keywords**: fMRI brain decoding, image reconstruction, Brain-Interaction Transformer, cross-subject transfer, diffusion models, Deep Image Prior

## TL;DR

This paper proposes Brain-IT, a framework that employs a brain-inspired Brain Interaction Transformer (BIT) to cluster functionally similar brain voxels into cross-subject shared Brain Tokens, from which localized semantic and structural image features are predicted, enabling high-fidelity reconstruction of images from fMRI signals. With only 1 hour of data, Brain-IT achieves performance comparable to prior methods trained on 40 hours of data.

## Background & Motivation

Reconstructing visual experiences from fMRI brain signals is a core challenge in neuroscience and brain–computer interface research. Although the introduction of diffusion models has brought significant progress, existing methods still suffer from notable **fidelity** deficiencies—generated images may appear visually plausible but frequently deviate from the actually perceived stimuli, manifesting as:

- **Structural bias**: incorrect position, color, and spatial layout
- **Semantic distortion**: missing or distorted semantic content
- **Root cause**: over-reliance on the generative prior of diffusion models, which can produce "realistic" images even when fMRI guidance is insufficient

The authors attribute the problem to three levels: (1) inappropriate fMRI representation extraction—existing methods compress all voxels into a single global embedding, discarding distributed information across the visual cortex; (2) the mapping strategy to image features—fully connected layers fail to exploit the distributed nature of brain regions; (3) feature integration in the generative model—lacking structural guidance.

Furthermore, fMRI data acquisition is costly and time-consuming (approximately 40 hours of scanning per subject), making it an important practical challenge to transfer models to new subjects with minimal data.

## Method

### Overall Architecture

Brain-IT consists of two stages: **image feature prediction** (BIT model) → **image reconstruction** (dual-branch generation).

### 1. Voxel-to-Cluster Mapping (V2C)

Voxel embeddings are obtained using the brain encoder from Beliy et al. (2024) (capturing the functional role of each voxel), and Gaussian Mixture Model (GMM) clustering is applied to voxel embeddings across all subjects, mapping approximately 40,000 voxels per subject into **128 functional clusters**. Key properties:
- Clusters are shared across subjects, capturing functionally similar brain regions across individuals
- Information aggregation is compressed from the voxel level to the cluster level, substantially reducing complexity

### 2. Brain Interaction Transformer (BIT)

The core model that transforms fMRI signals into **localized image features**:

**Brain Tokenizer**: Converts fMRI activations into Brain Tokens
- **Voxel embeddings** (512-dim): learnable per-voxel vectors capturing functional properties, multiplied by fMRI activation values for modulation
- **Cluster embeddings** (512-dim): learnable per-cluster vectors serving as information selection bottlenecks
- Aggregation via a single-head graph attention layer: cluster embeddings as Query, modulated voxel activations as Key/Value, with attention restricted by the V2C mapping
- Outputs 128 Brain Tokens of 512 dimensions each

**Cross-Transformer**:
- Self-attention layers model inter-cluster interactions
- Cross-attention layers directly map information from Brain Tokens to localized image features
- Each query token corresponds to one output image feature location, enabling direct information flow from functional clusters to local image features

### 3. Dual-Branch Image Reconstruction

**Semantic branch** (high-level):
- BIT predicts 256 spatial OpenCLIP ViT-bigG/14 tokens
- Training proceeds in two stages: feature alignment first (L2 loss), followed by joint training of BIT + diffusion model (diffusion loss)
- Joint training allows BIT outputs to deviate from original CLIP representations, forming representations better suited for fMRI-conditioned generation

**Low-level branch** (structural):
- BIT predicts multi-layer VGG features (trained with InfoNCE loss)
- Inverted via Deep Image Prior (DIP): a randomly initialized CNN outputs an image optimized so that its VGG features match BIT predictions
- The convolutional inductive bias of DIP provides a strong image prior, generating coarse but structurally correct layouts

**Dual-branch fusion** (at inference):
- The low-level branch generates a coarse image → noise is added and used as initialization for the diffusion process
- The semantic branch provides conditional guidance → the diffusion model refines the coarse structure into a detailed image
- Exploiting the diffusion model's "coarse-to-fine" generation property, the low-level image supplies reliable global structure

### 4. Training Data Augmentation

The image-to-fMRI encoder from Beliy et al. (2024) is used to predict fMRI responses for ~120,000 unlabeled COCO images, providing additional training pairs that are particularly important for transfer learning.

## Key Experimental Results

**Dataset**: NSD dataset (7T fMRI), 4 subjects (S1/2/5/7), ~9,000 images per subject, 1,000 shared test images.

**Main Results with 40-hour full data** (Table 1, averaged over 4 subjects):

| Metric | MindEye2 | MindTuner | **Brain-IT** |
|--------|----------|-----------|-------------|
| PixCorr ↑ | 0.322 | 0.322 | **0.386** |
| SSIM ↑ | 0.431 | 0.421 | **0.486** |
| Alex(2) ↑ | 96.1% | 95.8% | **98.4%** |
| Alex(5) ↑ | 98.6% | 98.8% | **99.5%** |
| Incep ↑ | 95.4% | 95.6% | **97.3%** |
| CLIP ↑ | 93.0% | 93.8% | **96.4%** |
| Eff ↓ | 0.619 | 0.612 | **0.564** |
| SwAV ↓ | 0.344 | 0.340 | **0.320** |

→ **State-of-the-art on 7 out of 8 metrics**, with substantial margins on low-level metrics (PixCorr, SSIM)

**1-hour transfer learning**:

| Metric | MindEye2 (1h) | MindTuner (1h) | **Brain-IT (1h)** |
|--------|---------------|----------------|------------------|
| PixCorr | 0.195 | 0.224 | **0.331** |
| SSIM | 0.419 | 0.420 | **0.473** |
| Alex(2) | 84.2% | 87.8% | **97.1%** |

→ **Brain-IT with 1 hour of data matches prior methods trained on 40 hours**
→ Meaningful reconstructions are obtained with as little as 15 minutes of data

**Ablation on branch contribution**:
- Low-level branch only: SSIM=0.505 (best structural fidelity), CLIP=85.8% (weak semantics)
- Semantic branch only: SSIM=0.431, CLIP=95.2% (strong semantics)
- Dual-branch fusion: SSIM=0.486, CLIP=96.4% (complementary enhancement)

## Highlights & Insights

1. **Brain-inspired design**: The functional clustering and Brain Token design directly correspond to the distributed organization of the visual cortex and retinotopic structure, offering a more principled alternative to global compression.
2. **Localized feature prediction**: Predicting localized image features directly from Brain Tokens (rather than a global embedding) preserves spatial information; cross-attention maps exhibit clear contralateral organization and semantic selectivity.
3. **Novel DIP low-level branch**: Inverting VGG features via Deep Image Prior is a pioneering signal-to-image approach that leverages CNN inductive biases without training, effectively capturing structural information such as color and contour.
4. **Highly efficient transfer learning**: Only voxel embeddings need to be fine-tuned (network weights are frozen); 1 hour ≈ prior methods' 40 hours, with meaningful results at 15 minutes—enabled by the shared cluster and weight design.
5. **Interpretable attention maps**: Different Brain Tokens correspond to specific spatial locations and semantic concepts (faces, limbs, text), offering valuable neuroscientific insights.

## Limitations & Future Work

1. **Imperfect reconstruction**: Semantic and fine-grained details are sometimes inaccurate (acknowledged in the paper), potentially limited by the inherent resolution of fMRI signals.
2. **Dependence on pretrained encoder**: The V2C mapping relies on the quality of the brain encoder from Beliy et al.; cluster quality affects the entire pipeline.
3. **DIP inference overhead**: Low-level reconstruction for each image requires independent DIP network optimization, resulting in longer inference times.
4. **Single dataset evaluation**: Validation is primarily conducted on the NSD dataset; although an OOD test on NSD Synthetic is included, generalization to other fMRI datasets has not been verified.
5. **Limited subject count**: Only 4 subjects (S1/2/5/7) are evaluated; generalizability across individual differences awaits validation at a larger scale.

## Related Work & Insights

- **Global embedding methods**: MindEye/MindEye2 (Scotti et al.)—linear/MLP mapping from fMRI to global CLIP embeddings, discarding spatial information
- **Cross-subject methods**: MindTuner (Gong et al.), MindBridge (Wang et al.)—scan-level fMRI alignment, exploiting only scan-level shared representations
- **Voxel grouping**: NeuroPictor (Huo et al.), NeuroVLA (Shen et al.)—voxel grouping in anatomical space, yet still predicting global representations
- **Brain-IT advantage**: functional clustering + localized prediction + dual-branch fusion, preserving spatial information from voxels through to image features

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (brain-inspired functional clustering, localized feature prediction, and DIP low-level branch are all pioneering contributions)
- Experimental Thoroughness: ⭐⭐⭐⭐ (comprehensive metric comparisons and thorough transfer learning analysis, but limited to a single dataset)
- Writing Quality: ⭐⭐⭐⭐⭐ (clear structure, excellent figures, intuitive method exposition)
- Value: ⭐⭐⭐⭐⭐ (substantially advances fMRI image reconstruction SOTA; 1-hour transfer learning has significant clinical implications)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Modeling Spatiotemporal Neural Frames for High Resolution Brain Dynamics](../../CVPR2026/3d_vision/modeling_spatiotemporal_neural_frames_for_high_resolution_brain_dynamic.md)
- [\[CVPR 2026\] Meta-learning In-Context Enables Training-Free Cross Subject Brain Decoding](../../CVPR2026/3d_vision/meta-learning_in-context_enables_training-free_cross_subject_brain_decoding.md)
- [\[CVPR 2026\] Human Interaction-Aware 3D Reconstruction from a Single Image](../../CVPR2026/3d_vision/human_interaction-aware_3d_reconstruction_from_a_single_image.md)
- [\[NeurIPS 2025\] Scalable Diffusion Transformer for Conditional 4D fMRI Synthesis](../../NeurIPS2025/3d_vision/scalable_diffusion_transformer_for_conditional_4d_fmri_synthesis.md)
- [\[ICLR 2026\] NOVA3R: Non-pixel-aligned Visual Transformer for Amodal 3D Reconstruction](nova3r_non-pixel-aligned_visual_transformer_for_amodal_3d_reconstruction.md)

</div>

<!-- RELATED:END -->
