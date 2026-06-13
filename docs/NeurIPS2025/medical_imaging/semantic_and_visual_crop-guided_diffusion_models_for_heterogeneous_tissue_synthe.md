---
title: >-
  [Paper Note] Semantic and Visual Crop-Guided Diffusion Models for Heterogeneous Tissue Synthesis in Histopathology
description: >-
  [NeurIPS 2025][Medical Imaging][Pathology Image Synthesis] This paper proposes HeteroTissue-Diffuse (HTD), a dual-conditioned Latent Diffusion Model that generates heterogeneous pathology images by simultaneously conditi…
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "Pathology Image Synthesis"
  - "Diffusion Models"
  - "Dual-Condition Generation"
  - "Heterogeneous Tissue"
  - "Self-Supervised Clustering"
date: 2026-05-08
content_hash: e0a1e5dc2bb9b799
---

# Semantic and Visual Crop-Guided Diffusion Models for Heterogeneous Tissue Synthesis in Histopathology

**Conference**: NeurIPS 2025
**arXiv**: [2509.17847](https://arxiv.org/abs/2509.17847)  
**Code**: [Project Page](https://kimialabmayo.github.io/hetero_tissue_diffuse_page/)  
**Area**: Medical Imaging / Computational Pathology
**Keywords**: Pathology Image Synthesis, Diffusion Models, Dual-Condition Generation, Heterogeneous Tissue, Self-Supervised Clustering

## TL;DR

This paper proposes HeteroTissue-Diffuse (HTD), a dual-conditioned Latent Diffusion Model that generates heterogeneous pathology images by simultaneously conditioning on semantic segmentation maps and real tissue crops (visual crops). On Camelyon16, the method reduces Fréchet Distance from 430 to 72 (a 6× improvement). DeepLabv3+ segmentation IoU trained on synthetic data falls within 1–2% of models trained on real data. The approach is further extended to 11,765 unannotated TCGA whole-slide images via self-supervised clustering.

## Background & Motivation

**Background**: AI-based diagnosis in pathology is constrained by data scarcity, expensive annotation, and privacy concerns. With the evolution from GANs to diffusion models, image quality and training stability have substantially improved. However, existing methods mostly generate homogeneous tissue (single tissue type) and fail to reflect the multi-tissue co-existence observed in real clinical specimens.

**Limitations of Prior Work**: Conditional control mechanisms suffer from three categories of deficiencies. Unconditional generation lacks control over tissue type. Text-guided methods are limited by inter-observer variability in pathological terminology (kappa only 0.48). Visual embedding methods (e.g., CLIP, RNA-seq embeddings) lose critical diagnostic features such as nuclear texture and staining patterns during dimensionality reduction. None of these approaches simultaneously achieves spatially precise control and morphological fidelity for heterogeneous tissue.

**Key Challenge**: A fundamental tension exists between spatial precision and morphological fidelity. Semantic segmentation maps provide accurate spatial layout control but carry no information about actual tissue appearance; visual embeddings encode appearance but lose fine-grained details. More fundamentally, large-scale pathology datasets (e.g., 11,765 WSIs from TCGA) lack pixel-level annotations and thus cannot be directly used for conditional generation training.

**Goal**: (1) How can heterogeneous tissue images be generated with simultaneous spatial layout accuracy and morphological fidelity? (2) How can the method scale to large unannotated datasets?

**Key Insight**: The key observation is that using raw tissue crops directly as visual conditions—rather than abstract embeddings extracted by an encoder—preserves staining patterns and cellular morphology without information loss. Foundation model embeddings are additionally leveraged for automatic pseudo-annotation of unannotated data via clustering.

**Core Idea**: Replace text/embedding-based conditioning with a dual-conditioning mechanism of "semantic segmentation map + real tissue crops" to guide a Latent Diffusion Model in generating heterogeneous pathology images.

## Method

### Overall Architecture

HeteroTissue-Diffuse (HTD) is built upon a Latent Diffusion Model. The input is a pathology image patch together with its corresponding semantic segmentation map; the output is a synthetic pathology image with accurate region annotations. The overall framework consists of three components: (a) unsupervised tissue clustering on unannotated data (TCGA) to generate pseudo semantic maps; (b) online sampling of heterogeneous regions; and (c) dual-conditioned LDM generation. For annotated datasets (Camelyon16, Panda), existing annotations are used directly in step (c).

### Key Designs

1. **Dual-Conditioning Mechanism**:

    - **Function**: Simultaneously exploit semantic spatial information and real tissue appearance to precisely guide image generation.
    - **Mechanism**: For a segmentation map $M$ with $K$ tissue classes, a square crop $p_i$ of size $d \times d$ (where $d \in [50, 200]$ pixels) is randomly sampled from the corresponding semantic region for each class $i$ and placed into a sparse tensor $C_i$ of the same spatial dimensions as the original image. The final conditioning signal is $c = \text{concat}(M, C_1, ..., C_K)$, i.e., the concatenation of the segmentation map channel and the per-class visual crop channels. This condition is injected into the UNet denoising network via a ControlNet-style mechanism.
    - **Design Motivation**: The semantic map provides spatial layout ("which region is which tissue"), while the visual crop provides morphological reference ("what that tissue looks like"). Compared to embedding-based methods, using raw pixels directly avoids information loss; compared to text-based methods, it avoids terminological ambiguity.

2. **Heterogeneous Patch Sampling Strategy**:

    - **Function**: Ensure that training samples contain meaningful tissue diversity.
    - **Mechanism**: For annotated data, patches are extracted with the constraint that tissue composition falls within 20%–80% (i.e., at least two tissue types co-exist). For unannotated TCGA data, tissue diversity entropy $H(r) = -\sum_i p_i(r) \log p_i(r)$ is computed for each region, and high-entropy regions (e.g., tumor–stroma interfaces) are preferentially sampled. Crop size adapts to tissue complexity: $d_i = d_{\text{base}} \cdot (1 + \alpha \cdot \text{ComplexityScore}(i))$.
    - **Design Motivation**: Prevents training samples from being dominated by homogeneous regions and enables the model to learn realistic tissue transitions.

3. **Self-Supervised Tissue Type Discovery (TCGA Extension)**:

    - **Function**: Automatically generate pseudo semantic segmentation maps for 11,765 unannotated TCGA WSIs.
    - **Mechanism**: A three-stage pipeline is employed. (1) Foundation models such as UNI are used to extract embeddings for all patches (634 million patches total); diversity-aware sampling selects 1,000 representative patches per WSI. (2) Hierarchical K-means clusters patches into 100 tissue phenotypes, with high-variance clusters further sub-clustered. (3) Multi-scale pseudo semantic maps are generated at granularities $k \in \{5, 10, 20, 50, 100\}$; during training, curriculum learning progressively increases granularity from coarse to fine: $k'(t) = k_{\min} + (k_{\max} - k_{\min}) \cdot \min(1, t/T_{\text{warmup}})$.
    - **Design Motivation**: TCGA covers 33 cancer types but lacks segmentation annotations. Self-supervised clustering allows the framework to scale to large-scale diverse data without manual annotation.

### Loss & Training

The standard LDM noise prediction loss is used: $\mathcal{L} = \mathbb{E}\|\epsilon - \epsilon_\theta(z_t, t, c)\|_2^2$. At inference time, a lightweight ViT-small classifier is additionally trained to replace computationally expensive foundation models for tissue type classification, reducing computational overhead by approximately 85%. Training incorporates tissue-aware augmentations including stain variation, rotation, and brightness perturbation.

## Key Experimental Results

### Main Results

**Fréchet Distance** (lower is better; evaluated with 8 different encoders):

| Dataset | Condition | RN50-BT | DINOv2 | UNI2-H | UNI |
|--------|------|---------|--------|--------|-----|
| Camelyon16 | Unconditional | 430.1 | 122.0 | 139.8 | 70.0 |
| Camelyon16 | Embedding-conditioned | 183.0 | 289.6 | 141.6 | 841.1 |
| Camelyon16 | **Visual crop-conditioned** | **72.0** | **52.7** | **85.2** | 481.4 |
| Panda | Unconditional | 150.0 | 352.4 | 113.6 | 650.5 |
| Panda | **Visual crop-conditioned** | **22.8** | **61.4** | **52.4** | 299.9 |

**Downstream Segmentation IoU**:

| Training Data | Camelyon16 IoU | Panda IoU |
|----------|---------------|-----------|
| Unconditional synthetic | 0.63 | 0.86 |
| Embedding-conditioned synthetic | 0.69 | 0.88 |
| Visual crop-conditioned synthetic | **0.71** | **0.95** |
| Real data | 0.72 | 0.96 |

### Ablation Study

| Configuration | Effect | Notes |
|------|------|------|
| Unconditional → Visual crop-conditioned | 6× FD reduction (RN50-BT) | Visual conditioning is the primary driver of quality improvement |
| Embedding-conditioned → Visual crop-conditioned | Additional 2–3× FD reduction | Raw pixels outperform abstract embeddings |
| Synthetic data vs. Real data | IoU gap of only 1–2% | Near-complete substitution of real data |

### Key Findings

- Visual crop conditioning consistently outperforms embedding-based and unconditional methods across all encoders and datasets, validating the core hypothesis of using raw pixels as conditions.
- RN50-BT and DINOv2 encoders are most sensitive to visual conditioning, showing the largest FD improvements.
- In a blinded evaluation by board-certified pathologists on 120 images, synthetic images were rated as indistinguishable from real images, with some synthetic images rated as equal or higher in quality than real ones.
- On the Panda dataset, segmentation models trained on synthetic data achieve an IoU of 0.95 versus 0.96 for real data, nearly constituting a complete replacement for real data.

## Highlights & Insights

- **Visual crops as a simple yet effective replacement for embeddings**: Directly placing real tissue patches into the conditioning channel preserves diagnostically critical subtle features (nuclear morphology, staining patterns) better than any learned abstract representation. This insight—that the simplest approach is often the best—is worth generalizing to other conditional generation tasks.
- **Self-supervised clustering enables unannotated-data scaling**: Using foundation model embeddings and K-means to automatically discover 100 tissue phenotypes, then generating pseudo semantic maps for training, elegantly addresses the challenge of leveraging large-scale unannotated data. The curriculum learning strategy from coarse to fine granularity is also worth referencing.
- **Synthetic data nearly replaces real data**: An IoU gap of only 1–2% represents an important milestone, indicating that diagnostic models can be trained without sharing patient data—a significant contribution to privacy-preserving AI and data-scarce scenarios such as rare cancers.

## Limitations & Future Work

- **Inference efficiency**: Extracting embeddings for 634 million TCGA patches required 3 months of computation on a single A100 GPU. Although replacing foundation models with ViT-small reduces inference overhead by 85%, the initial clustering cost remains substantial.
- **Limited clustering classifier accuracy**: The ViT-small classifier achieves only 47% accuracy across 100 tissue phenotypes, which may compromise condition quality at inference time.
- **Resolution and multi-scale consistency**: The current approach operates at a fixed patch scale and does not address structural consistency across magnification levels in WSIs (despite citing URCDM and DifInfinite, these approaches are not adopted).
- **Absence of slide-level global structure evaluation**: Both FD and IoU are patch-level metrics; the structural coherence of generated images at the whole-slide scale is not assessed.
- **Validation limited to segmentation**: The utility of synthetic data is not evaluated on other downstream tasks such as classification or detection.

## Related Work & Insights

- **vs. NASDM / Konz et al.**: These methods use semantic segmentation map conditioning but focus on homogeneous single-tissue-type generation. The present work extends to heterogeneous multi-tissue generation and adds visual crop conditioning to preserve morphological detail.
- **vs. URCDM**: URCDM employs cascaded diffusion models for multi-resolution generation, whereas the present work focuses on patch-level conditional control rather than multi-scale consistency. The two approaches are complementary.
- **vs. Text-guided methods (PathLDM, etc.)**: Text conditioning is limited by terminological ambiguity and inter-observer variability in pathology; visual crops entirely bypass the bottleneck of linguistic description.
- **Inspiration**: The visual crop conditioning paradigm is transferable to other medical imaging domains requiring precise generation control (e.g., radiology, dermatology), and could be explored in combination with text conditioning for multimodal fusion.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The dual-conditioning mechanism (semantic map + visual crop) is a novel and practical design for pathology image synthesis, though the LDM backbone and ControlNet mechanism are not original contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive validation across three datasets, FD evaluation with 8 encoders, downstream segmentation, and pathologist blind evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with detailed method descriptions, though some technical details (e.g., the ControlNet injection scheme) are not fully elaborated in the main text.
- **Value**: ⭐⭐⭐⭐ The near-complete substitution of real data by synthetic data carries significant implications for privacy-preserving medical AI; the TCGA extension demonstrates strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] STARC-9: A Large-scale Dataset for Multi-Class Tissue Classification for CRC Histopathology](starc-9_a_large-scale_dataset_for_multi-class_tissue_classification_for_crc_hist.md)
- [\[NeurIPS 2025\] Posterior Sampling by Combining Diffusion Models with Annealed Langevin Dynamics](posterior_sampling_by_combining_diffusion_models_with_annealed_langevin_dynamics.md)
- [\[NeurIPS 2025\] Scalable Diffusion Transformer for Conditional 4D fMRI Synthesis](scalable_diffusion_transformer_for_conditional_4d_fmri_synthesis.md)
- [\[NeurIPS 2025\] SynBrain: Enhancing Visual-to-fMRI Synthesis via Probabilistic Representation Learning](synbrain_enhancing_visual-to-fmri_synthesis_via_probabilistic_representation_lea.md)
- [\[NeurIPS 2025\] Surf2CT: Cascaded 3D Flow Matching Models for Torso 3D CT Synthesis from Skin Surface](surf2ct_cascaded_3d_flow_matching_models_for_torso_3d_ct_synthesis_from_skin_sur.md)

</div>

<!-- RELATED:END -->
