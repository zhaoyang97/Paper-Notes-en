---
title: >-
  [Paper Note] Generate, Refine, and Encode: Leveraging Synthesized Novel Samples for On-the-Fly Fine-Grained Category Discovery
description: >-
  [ICCV 2025][on-the-fly category discovery] This paper proposes DiffGRE, a diffusion-model-based framework for on-the-fly category discovery. It synthesizes novel samples containing virtual category information via Attribute Composition Generation (ACG), filters low-quality samples through Diversity-Driven Refinement (DDR), and injects additional category knowledge via Semi-supervised Leader Encoding (SLE). DiffGRE achieves substantial performance gains over existing OCD methods across 6 fine-grained datasets (average ACC-ALL improvement of 6.5%).
tags:
  - ICCV 2025
  - on-the-fly category discovery
  - diffusion models
  - attribute composition generation
  - fine-grained recognition
  - online clustering inference
date: 2026-05-08
content_hash: d91ac06d852a4bbb
---

# Generate, Refine, and Encode: Leveraging Synthesized Novel Samples for On-the-Fly Fine-Grained Category Discovery

**Conference**: ICCV 2025
**arXiv**: [2507.04051](https://arxiv.org/abs/2507.04051)
**Code**: [github.com/XLiu443/DiffGRE](https://github.com/XLiu443/DiffGRE)
**Area**: Other
**Keywords**: on-the-fly category discovery, diffusion models, attribute composition generation, fine-grained recognition, online clustering inference

## TL;DR

This paper proposes DiffGRE, a diffusion-model-based framework for on-the-fly category discovery. It synthesizes novel samples containing virtual category information via Attribute Composition Generation (ACG), filters low-quality samples through Diversity-Driven Refinement (DDR), and injects additional category knowledge via Semi-supervised Leader Encoding (SLE). DiffGRE achieves substantial performance gains over existing OCD methods across 6 fine-grained datasets (average ACC-ALL improvement of 6.5%).

## Background & Motivation

### Problem Definition

On-the-fly Category Discovery (OCD) is a practical yet challenging task:
- **Training phase**: The model is trained exclusively on labeled data from known categories (support set $\mathcal{D}_S$).
- **Test phase**: The model processes streaming query data ($\mathcal{D}_Q$), which may belong to either known or unknown categories.
- **Objective**: Provide immediate, instance-level feedback for each new instance, rather than offline batch clustering.

The key distinction from Generalized Category Discovery (GCD): GCD requires access to a predefined query set during training and assigns labels to batches via offline clustering, whereas OCD requires neither query set access nor batch-level processing, instead providing per-instance online inference for streaming data.

### Limitations of Prior Work

**Insufficient knowledge**: Existing OCD methods extract transferable knowledge solely from labeled data. In fine-grained scenarios with few labeled samples or categories, the knowledge encoded in known classes is often insufficient to support novel category discovery.

**Hash inference limitations**: Methods such as SMILE use hash codes as category descriptors; the binarization inevitably reduces the representational capacity of high-dimensional features.

**Ineffective data augmentation**: Conventional category-preserving augmentation (e.g., generating known-class images via T2I models) offers limited benefit for novel category discovery—experiments confirm that augmenting only known-class data fails to improve unknown category discovery.

### Core Problem

**Key insight**: Unknown categories can be synthesized by recombining semantic attributes from known categories. For example, performing latent-space interpolation between two known bird species within a diffusion model can produce images visually similar to an unseen bird category. This "attribute recombination" paradigm opens the possibility of generating synthetic data containing additional category information directly from known classes for the OCD task.

## Method

### Overall Architecture

DiffGRE consists of three stages:
1. **Generate**: ACG performs cross-category interpolation in the diffusion model's latent space to synthesize virtual-category images.
2. **Refine**: DDR filters synthetic images that are overly similar to known categories.
3. **Encode**: SLE assigns reliable pseudo-labels to synthetic data and uses Leader features for online clustering inference.

### Key Designs

#### 1. Attribute Composition Generation (ACG)

- **Function**: Samples two images from different known categories and performs cross-category interpolation in the diffusion model's latent space to synthesize images belonging to "virtual categories."
- **Mechanism**: Spherical interpolation is performed simultaneously across three embedding spaces—Stable Diffusion's latent space, CLIP visual space, and CLIP text space:

$$\bar{z}^* = \frac{\sin((1-\lambda_*)\theta)}{\sin(\theta)} z_1^* + \frac{\sin(\lambda_*\theta)}{\sin(\theta)} z_2^*, \quad * \in \{t, v, l\}$$

where $\theta = \arccos(z_1^* \cdot z_2^*)$ and $\lambda_* \in [0,1]$ is the interpolation coefficient.
- The VAE encoder extracts latent embeddings $z^l$.
- The CLIP visual encoder extracts visual embeddings $z^v$.
- A pretrained image captioning model converts images to text, from which the CLIP text encoder extracts text embeddings $z^t$.

The denoising process follows the standard diffusion objective:

$$\mathcal{L}_{DM} = \mathbb{E}_{t, \bar{z}_0^l, \epsilon} \left[\|\epsilon - \epsilon_\theta(\bar{z}^l, t, \bar{z}^t)\|^2\right]$$

- **Design Motivation**: Stable Diffusion's latent space primarily captures visual details, while CLIP space optimizes for high-level semantic understanding. Combining both allows the model to recombine semantic attributes while preserving visual quality, yielding visually plausible and semantically novel virtual-category images.

#### 2. Diversity-Driven Refinement (DDR)

- **Function**: Selects synthetic images that carry rich additional category information and filters those overly similar to known categories.
- **Mechanism**:
  1. Compute cosine similarity between each synthetic image and all known class centers: $s(z_{\text{gen}}, c_k) = \frac{z_{\text{gen}} \cdot c_k}{\|z_{\text{gen}}\| \|c_k\|}$
  2. Compute the mean similarity of each synthetic sample across all class centers: $s_{\text{mean}}(z_{\text{gen}}) = \frac{1}{K}\sum_{k=1}^{K} s(z_{\text{gen}}, c_k)$
  3. Apply threshold $\gamma$ to filter high-similarity samples: $\mathcal{F}(I_{\text{gen}}) = \mathbf{1}(s_{\text{mean}}(z_{\text{gen}}) \leq \gamma) \cdot I_{\text{gen}}$

- **Design Motivation**: The stochasticity of the diffusion process means not every synthetic image benefits OCD. Experiments show that using all synthetic images indiscriminately disrupts the learning of discriminative features for known categories. DDR uses class-center-based mean similarity—rather than per-image comparisons—for greater stability and robustness to long-tail distributions.

#### 3. Semi-supervised Leader Encoding (SLE)

- **Function**: Assigns reliable pseudo-labels to synthetic images and generates class-level Leader features for online inference.
- **Mechanism**:
  1. **Virtual category assignment**: Merges labeled and synthetic data into a proxy training set $\mathcal{D}_A = \mathcal{D}_S \cup \mathcal{D}_G$, applies clustering to generate initial category labels, and corrects alignment with known labels via the Hungarian algorithm.
  2. **Leader feature generation**: Computes the mean feature of each virtual category as its Leader feature.
  3. **Leader contrastive learning**: Maximizes inter-class distance and minimizes intra-class distance:

$$\mathcal{L}_{sle}(x_n, y_n) = -\log \frac{\exp(f(x_n) \cdot l_{y_n}^T / \tau)}{\sum_{m \neq y_n} \exp(f(x_n) \cdot l_m^T / \tau)}$$

  4. **Online Clustering Inference (OCI)**: At test time, a dynamic Leader memory is initialized; an adaptive threshold determines whether a new instance belongs to a known category or triggers the creation of a new one.

- **Design Motivation**: Naively treating images interpolated from the same pair as belonging to the same new category causes severe performance degradation. SLE assigns more principled category labels to synthetic data through clustering combined with Hungarian alignment. Compared to hash-based inference, OCI operates directly in the high-dimensional feature space, avoiding information loss from binarization.

### Loss & Training

The total loss is:

$$\mathcal{L} = \mathcal{L}_{sup} + \mathcal{L}_{reg} + \alpha \cdot \mathcal{L}_{sle} + \beta \cdot \mathcal{L}_{c}$$

- $\mathcal{L}_{sup}$: Supervised contrastive loss from SMILE.
- $\mathcal{L}_{reg}$: Regularization loss on the hash head output.
- $\mathcal{L}_{sle}$: Leader contrastive loss ($\alpha=0.3$).
- $\mathcal{L}_c$: Classification cross-entropy loss ($\beta=1.0$).

## Key Experimental Results

### Main Results

**DiffGRE improvements over three baselines under hash inference (average across 6 fine-grained datasets)**:

| Method | ACC-ALL | ACC-OLD | ACC-NEW |
|--------|---------|---------|---------|
| BaseHash | 25.5 | 36.3 | 19.7 |
| BaseHash + DiffGRE | **32.0** | **47.8** | **23.9** |
| SMILE | 32.7 | 49.1 | 24.3 |
| SMILE + DiffGRE | **36.3** | **56.7** | **25.9** |
| PHE | 38.9 | 61.3 | 26.7 |
| PHE + DiffGRE | **40.0** | **62.9** | **28.0** |

**Comparison under Online Clustering Inference (OCI)**:

| Method | ACC-ALL | ACC-OLD | ACC-NEW |
|--------|---------|---------|---------|
| SMILE + SLE-based | 41.9 | 52.6 | 37.0 |
| SMILE + DiffGRE | **43.4** | **53.4** | **38.7** |
| PHE + SLE-based | 39.7 | 57.1 | 30.8 |
| PHE + DiffGRE | **42.3** | **59.1** | **33.5** |

### Ablation Study

**Training component ablation (Arachnida / Mollusca / CUB datasets, hash inference)**:

| Configuration | Arachnida-ALL | Mollusca-ALL | CUB-ALL |
|--------------|--------------|-------------|---------|
| SMILE baseline | 27.9 | 33.5 | 32.2 |
| w/o $\mathcal{L}_{sle}$ | 29.3 | 33.9 | 33.2 |
| w/o $\mathcal{L}_{c}$ | 34.5 | 36.0 | 33.6 |
| w/o DDR | 33.5 | 34.5 | 32.4 |
| **SMILE + DiffGRE** | **35.4** | **36.5** | **35.4** |

**Comparison of synthesis methods (CUB dataset)**:

| Synthesis Method | Type | ACC-ALL |
|-----------------|------|---------|
| CutMix | Pixel mixing | Lower |
| MixUp | Pixel mixing | Lower |
| Da-Fusion | T2I (known-class text) | Limited gain |
| Diff-Mix | T2I (known-class text) | Limited gain |
| **ACG (Ours)** | **Latent-space attribute recombination** | **Best** |

### Key Findings

1. **DiffGRE is a general plug-and-play framework**: It consistently improves all three baselines—BaseHash, SMILE, and PHE.
2. **DDR is indispensable**: Removing DDR causes an average 4.6% drop in ACC-OLD, confirming that low-quality synthetic samples disrupt discriminative feature learning for known categories.
3. **SLE substantially outperforms hash inference**: Online clustering inference surpasses hash inference on all datasets, with an average ACC-NEW improvement of 16.5%.
4. **Attribute recombination outperforms category-preserving augmentation**: Conventional T2I methods can only generate images of known categories, offering limited benefit for novel category discovery; ACG synthesizes images visually similar to unknown categories through attribute recombination.
5. **Synthetic sample count should match the scale of labeled data**: DDR achieves the best results when the optimal threshold $\gamma$ retains a number of synthetic samples comparable to the labeled data volume.

## Highlights & Insights

1. **Novelty of the attribute recombination paradigm**: Rather than generating faithful augmentations of known categories, DiffGRE synthesizes "virtual categories" via cross-category latent-space interpolation—a novel application of diffusion models to open-world discovery tasks.
2. **Tri-space joint interpolation**: Simultaneous interpolation in the diffusion latent space, CLIP visual space, and CLIP text space balances low-level visual fidelity with high-level semantic coherence.
3. **Inference paradigm shift from hashing to online clustering**: SLE's Leader features preserve high-dimensional information, and OCI's adaptive thresholding enables more robust online inference.
4. **Simplicity and effectiveness of DDR**: Using only the mean cosine similarity to class centers is sufficient to filter low-quality samples—a design that is both computationally efficient and conceptually clean.

## Limitations & Future Work

1. **Dependence on pretrained diffusion model quality**: The quality of synthetic images is bounded by Stable Diffusion's performance, which may limit effectiveness in low-resource domains such as medical imaging.
2. **Sensitivity of interpolation parameters**: The three interpolation coefficients $\lambda_t$, $\lambda_v$, $\lambda_l$ require dataset-specific tuning.
3. **Noise in cluster assignment**: SLE's virtual category assignment relies on clustering quality and may introduce noise when the number of categories is unknown or class distributions are imbalanced.
4. **Computational overhead**: Synthetic image generation requires additional diffusion model inference time, particularly on large-scale datasets.
5. **Limited gains on PHE**: When the baseline employs an independently optimized feature extractor (e.g., PHE), the improvements from DiffGRE are comparatively modest.

## Related Work & Insights

- **Distinction from GCD/NCD**: OCD requires neither query set access during training nor batch-level processing, demanding per-instance online inference.
- **Distinction from Diff-Mix**: Diff-Mix generates interpolations between two known categories and remains a form of known-class augmentation; in contrast, DiffGRE's ACG explicitly aims to synthesize virtual categories.
- **Inspiring finding**: When performing latent-space interpolation between known categories, nearest-neighbor search in the feature space retrieves samples from unknown categories, empirically validating the feasibility of synthesizing virtual categories through attribute recombination.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The concept of synthesizing virtual categories via attribute recombination is original; the tri-space interpolation and OCI inference design are innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive evaluation across 6 fine-grained datasets and 3 baselines, with detailed ablation analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Method descriptions are clear and motivation figures are intuitive.
- **Value**: ⭐⭐⭐⭐ — The plug-and-play framework offers strong practical utility and opens a new direction for applying diffusion models to category discovery.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] A Hidden Stumbling Block in Generalized Category Discovery: Distracted Attention](a_hidden_stumbling_block_in_generalized_category_discovery_d.md)
- [\[ICCV 2025\] Intra-view and Inter-view Correlation Guided Multi-view Novel Class Discovery](intra-view_and_inter-view_correlation_guided_multi-view_novel_class_discovery.md)
- [\[NeurIPS 2025\] MiCADangelo: Fine-Grained Reconstruction of Constrained CAD Models from 3D Scans](../../NeurIPS2025/others/micadangelo_fine-grained_reconstruction_of_constrained_cad_models_from_3d_scans.md)
- [\[ICCV 2025\] Membership Inference Attacks with False Discovery Rate Control](membership_inference_attacks_with_false_discovery_rate_control.md)
- [\[AAAI 2026\] MF-Speech: Achieving Fine-Grained and Compositional Control in Speech Generation via Factor Disentanglement](../../AAAI2026/others/mf-speech_achieving_fine-grained_and_compositional_control_in_speech_generation_.md)

<!-- RELATED:END -->
