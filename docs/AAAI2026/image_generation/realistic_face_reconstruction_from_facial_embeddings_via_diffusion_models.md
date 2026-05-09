---
title: >-
  [Paper Note] Realistic Face Reconstruction from Facial Embeddings via Diffusion Models
description: >-
  [AAAI 2026][Image Generation][Face Reconstruction] This paper proposes the FEM (Face Embedding Mapping) framework, which employs a KAN-based network to map embeddings from arbitrary face recognition (FR) or privacy-preserving face recognition (PPFR) systems into the embedding space of a pretrained identity-preserving (ID-Preserving) diffusion model, enabling high-resolution realistic face reconstruction for evaluating privacy leakage risks in FR systems.
tags:
  - AAAI 2026
  - Image Generation
  - Face Reconstruction
  - Facial Embeddings
  - Privacy Attack
  - Kolmogorov-Arnold Network
  - Diffusion Models
date: 2026-05-08
content_hash: b1fcfee6822e2ead
---

# Realistic Face Reconstruction from Facial Embeddings via Diffusion Models

**Conference**: AAAI 2026
**arXiv**: [2602.13168](https://arxiv.org/abs/2602.13168)
**Code**: N/A
**Area**: Image Generation
**Keywords**: Face Reconstruction, Facial Embeddings, Privacy Attack, Kolmogorov-Arnold Network, Diffusion Models

## TL;DR

This paper proposes the FEM (Face Embedding Mapping) framework, which employs a KAN-based network to map embeddings from arbitrary face recognition (FR) or privacy-preserving face recognition (PPFR) systems into the embedding space of a pretrained identity-preserving (ID-Preserving) diffusion model, enabling high-resolution realistic face reconstruction for evaluating privacy leakage risks in FR systems.

## Background & Motivation

### State of the Field

Face recognition (FR) systems generate facial embedding vectors as identity templates via black-box CNN/DNN models. To enhance privacy, privacy-preserving face recognition (PPFR) systems (e.g., DCTDP, HFCF, PartialFace, MinusFace) and embedding protection algorithms (e.g., PolyProtect, MLP-Hash, SlerpFace) have been proposed.

### Limitations of Prior Work

**Poor reconstruction quality of CNN-based methods**: NbNet and end-to-end CNN approaches produce blurry, noisy facial images, typically limited to low resolution.

**Significant limitations of GAN-based methods**: FaceTI relies on StyleGAN3 and requires intensive training resources (51 hours/epoch, 25 GB VRAM); MAP2V requires no training but is extremely slow at inference (111 seconds/image).

**Insufficient research targeting PPFR**: Existing methods focus primarily on standard FR systems, with limited investigation into embedding attacks against privacy-preserving systems.

**Lack of generalizability**: Existing methods struggle to handle realistic attack scenarios involving partially leaked or protected embeddings.

### Root Cause

How can a lightweight mapping network uniformly map the embedding spaces of diverse FR/PPFR systems—each with distinct characteristics—into the embedding space of a high-quality face generation model, achieving efficient and high-quality face reconstruction?

### Starting Point

By leveraging the pretrained ID-Preserving diffusion model IPA-FaceID, which already possesses the capability to generate high-quality faces from embeddings, the core problem is reformulated as **learning a mapping between embedding spaces**. A KAN (Kolmogorov-Arnold Network) is introduced to capture complex nonlinear relationships between embedding spaces.

## Method

### Overall Architecture

The FEM framework consists of two phases: training and inference.

**Training Phase**:
1. A public face dataset is fed into both the target FR/PPFR model $\Gamma'(\cdot)$ and the default FR model of IPA-FaceID $\Gamma(\cdot)$.
2. Two embedding distributions $\mathcal{D}'(e'_i)$ and $\mathcal{D}(e_i)$ are obtained.
3. The FEM model $\mathcal{M}(\cdot)$ is trained so that the mapped embedding $\hat{e}_i = \mathcal{M}(e'_i)$ approximates the corresponding $e_i$ as closely as possible.

**Inference Phase**:
1. The leaked target system embedding $e'$ is fed into the trained FEM.
2. FEM produces the mapped embedding $\hat{e}$.
3. IPA-FaceID directly generates a high-resolution identity-preserving facial image.

### Key Designs

#### 1. **FEM-KAN: KAN-Based Embedding Mapping**

**Mechanism**: The Kolmogorov-Arnold theorem states that any continuous function can be represented as a finite composition of univariate continuous functions. The mapping relationship between facial embeddings can thus be decomposed into compositions of univariate function operations.

$$f(x) = \sum_q \Phi_q\left(\sum_i \phi_{q,i}(x_i)\right)$$

**Distinction from FEM-MLP**:
- FEM-MLP uses fixed activation functions (GELU) with a 3-layer MLP and 1D batch normalization.
- FEM-KAN employs **learnable activation functions** placed on edges in a 3-layer KAN architecture, enabling more accurate capture of nonlinear mappings.

**Design Motivation**: Although facial embeddings are high-dimensional, they possess inherent structure. The univariate function decomposition in KAN is better suited to capturing complex nonlinear relationships between embedding spaces. UMAP visualizations confirm that FEM effectively maps target-domain embeddings into the target domain or boundary regions of IPA-FR.

#### 2. **Loss Function Design**

Mean Squared Error (MSE) is used as the reconstruction loss:

$$\mathcal{L}_{MSE}(e_i, \hat{e}_i) = \frac{\sum_{i=0}^{N-1}(e_i - \hat{e}_i)^2}{N}$$

where $e_i$ is the target embedding (output of IPA-FR) and $\hat{e}_i = \mathcal{M}(e'_i)$ is the FEM-mapped embedding.

#### 3. **Exploiting the ID-Preserving Capability of IPA-FaceID**

IPA-FaceID injects facial embeddings into a pretrained T2I diffusion model via decoupled cross-attention. The text prompt is fixed as "front portrait of a person" to generate frontal portraits. Once FEM maps the embedding into the target domain, IPA-FaceID can directly generate identity-preserving facial images.

### Loss & Training

- 90% of the FFHQ dataset is used for training; testing is performed on 1,000 unseen identities from CelebA-HQ.
- AdamW optimizer with an initial learning rate of $10^{-2}$ and exponential decay rate of 0.8.
- Batch size of 128; trained for 20 epochs.
- Training conducted on a Tesla V100 32 GB GPU.

## Key Experimental Results

### Main Results

Attack Success Rate (ASR) on the CelebA-HQ dataset:

| Target Model | Method | MF | EF | GF | AF | Avg. |
|---|---|---|---|---|---|---|
| IRSE50 (FR) | FaceTI | 93.4 | 80.8 | 49.6 | 66.8 | 72.7 |
| | MAP2V | 94.0 | 86.2 | 59.3 | 72.0 | 77.9 |
| | FEM-MLP | 98.0 | 91.8 | 62.6 | 73.4 | 81.5 |
| | **FEM-KAN** | **99.2** | **93.8** | **65.7** | **76.1** | **83.7** |
| HFCF (PPFR) | MAP2V | 76.3 | 15.4 | 5.3 | 14.8 | 28.0 |
| | **FEM-KAN** | **98.3** | **90.7** | **66.5** | **76.9** | **83.1** |
| MinusFace (PPFR) | MAP2V | 68.0 | 4.8 | 2.3 | 5.6 | 20.2 |
| | **FEM-KAN** | **96.5** | **71.3** | **44.5** | **58.1** | **67.6** |

FEM-KAN achieves the highest average ASR across all FR and PPFR target models, especially surpassing MAP2V by a large margin on PPFR models such as HFCF and MinusFace (83.1 vs. 28.0 and 67.6 vs. 20.2, respectively).

### Ablation Study

| Configuration | Key Metric | Notes |
|---|---|---|
| FEM Training Time | 3 hrs vs. FaceTI 51 hrs | **17×** faster |
| FEM GPU Memory | 4325 MiB vs. FaceTI 25383 MiB | **5.8×** more efficient |
| FEM Inference Time | 2.6s vs. MAP2V 111s | **42×** faster |
| 50% Embedding Leakage | FEM-KAN ASR 53.2% vs. FaceTI 50.8% | FEM more robust |
| 30% Embedding Leakage | FEM-KAN ASR 32.5% | Attack viable at very low leakage |
| Makeup Scenario (LADN-M) | FEM-KAN Avg. ASR 85.1% vs. FaceTI 56.4% | FEM more robust to makeup |

### Protected Embedding Attacks

| Protection Algorithm | FEM-KAN MF/EF/GF/AF | MAP2V MF/EF/GF/AF |
|---|---|---|
| MLP-Hash | 82.1/54.7/56.5/71.6 | 48.1/0.6/0.3/1.5 |
| SlerpFace | 79.4/9.3/7.8/15.4 | 11.4/0.0/0.1/0.1 |
| PolyProtect | 50.3/7.1/5.6/15.4 | 28.6/4.4/3.6/4.3 |

FEM achieves ASR under MLP-Hash protection close to the unprotected baseline, indicating a critical security vulnerability in this embedding protection algorithm.

### Key Findings

- **KAN outperforms MLP**: FEM-KAN surpasses FEM-MLP in nearly all scenarios, validating the advantage of learnable activation functions for embedding mapping.
- **PPFR systems are not secure**: Even after privacy-preserving transformations such as frequency-domain processing, embeddings retain sufficient identity information for high-quality face reconstruction.
- **Partial embeddings remain exploitable**: FEM-KAN achieves 32.5% ASR even with only 30% of the embedding vector available.
- **Makeup has minimal impact on FEM**: Makeup causes an 18.1% ASR drop for FaceTI but only a 6.4% drop for FEM-KAN.
- **Face anti-spoofing systems can be bypassed**: The reconstructed high-quality faces can pass FASNet detection.

## Highlights & Insights

1. **First application of KAN to embedding mapping**: This work demonstrates that learnable activation functions in KAN genuinely outperform conventional MLP for nonlinear mapping of high-dimensional structured data (facial embeddings).
2. **Framework-level thinking**: The face reconstruction problem is elegantly reformulated as a two-stage pipeline of "embedding space mapping + pretrained generation," enabling an extremely lightweight mapping network.
3. **Comprehensive security evaluation**: The framework covers standard FR, PPFR, partial leakage, protected embeddings, and protected images, offering practical value for real-world security assessment.
4. **High training efficiency**: A 3-layer network trained in 3 hours with 2.6-second inference significantly outperforms existing methods.

## Limitations & Future Work

- The framework depends on IPA-FaceID as the generation backend; it would become ineffective if this model is updated or deprecated.
- The text prompt is fixed as "front portrait of a person," which may limit reconstruction accuracy for non-frontal faces.
- FEM requires black-box query access to the target FR system to construct training data.
- ASR drops noticeably for low-resolution faces (LFW 112×112).
- When the embedding leakage rate falls below 30%, reconstructed faces exhibit visible artifacts.
- Only MSE loss is employed; alternative distance metrics (e.g., cosine distance, contrastive loss) are not explored.

## Related Work & Insights

- FEM can serve as an **evaluation tool** for the privacy security of FR/PPFR systems, quantifying the privacy leakage risk of different protection algorithms.
- The results inspire security research: embedding protection algorithms, particularly MLP-Hash, require redesign to resist mapping-based attacks.
- KAN may have application potential in other embedding space alignment tasks, such as cross-modal retrieval and domain adaptation.
- This work underscores the "embedding-as-privacy" security paradigm: even when images are protected, embedding vectors may still leak identity information.

## Rating

- Novelty: ⭐⭐⭐⭐ (Novel application of KAN to embedding mapping; framework design is elegant and concise)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Covers diverse FR/PPFR systems, multiple attack scenarios, and resource comparisons)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure with detailed experimental setup)
- Value: ⭐⭐⭐⭐ (Practically significant for privacy security research; exposes security risks in PPFR systems)

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] High-Fidelity Diffusion Face Swapping with ID-Constrained Facial Conditioning](../../CVPR2026/image_generation/high-fidelity_diffusion_face_swapping_with_id-constrained_facial_conditioning.md)
- [\[AAAI 2026\] Diffusion Reconstruction-Based Data Likelihood Estimation for Core-Set Selection](diffusion_reconstruction-based_data_likelihood_estimation_for_core-set_selection.md)
- [\[CVPR 2026\] ExpressEdit: Fast Editing of Stylized Facial Expressions with Diffusion Models in Photoshop](../../CVPR2026/image_generation/expressedit_fast_editing_of_stylized_facial_expressions_with_diffusion_models_in.md)
- [\[CVPR 2026\] Face2Scene: Using Facial Degradation as an Oracle for Diffusion-Based Scene Restoration](../../CVPR2026/image_generation/face2scene_using_facial_degradation_as_an_oracle_for_diffusion-based_scene_resto.md)
- [\[ICCV 2025\] MoFRR: Mixture of Diffusion Models for Face Retouching Restoration](../../ICCV2025/image_generation/mofrr_mixture_of_diffusion_models_for_face_retouching_restoration.md)

<!-- RELATED:END -->
