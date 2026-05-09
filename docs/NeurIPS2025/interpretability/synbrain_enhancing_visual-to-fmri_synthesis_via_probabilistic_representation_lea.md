---
title: >-
  [Paper Note] SynBrain: Enhancing Visual-to-fMRI Synthesis via Probabilistic Representation Learning
description: >-
  [NeurIPS 2025][visual-to-fMRI synthesis] This paper proposes SynBrain, a framework that models fMRI responses as visual-semantic-conditioned probability distributions via BrainVAE, and employs an S2N Mapper for one-step semantic-to-neural-space mapping. SynBrain substantially outperforms MindSimulator on visual-to-fMRI synthesis (65% reduction in MSE, 96% improvement in Pearson correlation), and the synthesized fMRI signals effectively enhance few-shot cross-subject decoding performance.
tags:
  - NeurIPS 2025
  - visual-to-fMRI synthesis
  - variational autoencoder
  - probabilistic representation learning
  - brain encoding
  - few-shot adaptation
date: 2026-05-08
content_hash: 3bf406f2b3a682aa
---

# SynBrain: Enhancing Visual-to-fMRI Synthesis via Probabilistic Representation Learning

**Conference**: NeurIPS 2025
**arXiv**: [2508.10298](https://arxiv.org/abs/2508.10298)
**Code**: [GitHub](https://github.com/MichaelMaiii/SynBrain)
**Area**: Interpretability
**Keywords**: visual-to-fMRI synthesis, variational autoencoder, probabilistic representation learning, brain encoding, few-shot adaptation

## TL;DR

This paper proposes SynBrain, a framework that models fMRI responses as visual-semantic-conditioned probability distributions via BrainVAE, and employs an S2N Mapper for one-step semantic-to-neural-space mapping. SynBrain substantially outperforms MindSimulator on visual-to-fMRI synthesis (65% reduction in MSE, 96% improvement in Pearson correlation), and the synthesized fMRI signals effectively enhance few-shot cross-subject decoding performance.

## Background & Motivation

Understanding how visual stimuli are transformed into cortical responses is a core challenge in computational neuroscience. fMRI, as the dominant brain imaging modality, indirectly reflects neural activity by measuring BOLD signals. **Visual-to-fMRI encoding** aims to establish a functional mapping from external visual perception to spatially distributed neural responses.

Existing encoding methods primarily adopt regression or deterministic generative strategies, yet face a fundamental contradiction: **the visual-to-neural mapping is inherently one-to-many**. Large-scale neuroimaging studies (e.g., the NSD dataset) clearly demonstrate that repeated presentations of identical visual stimuli elicit substantially different fMRI responses across trials and subjects, reflecting trial-level noise, attentional fluctuations, and individual differences.

Three core limitations of prior methods:

**Deterministic modeling**: MindSimulator, for instance, uses a deterministic AutoEncoder that produces a unique latent representation for each input, collapsing diverse neural patterns into an uninformative average response.

**Lack of functionally consistent variability**: Existing methods cannot simultaneously model the "pattern variability" and "functional encoding consistency" of neural responses.

**Limited utility of synthesized data**: The absence of cross-subject transfer capability restricts the use of synthesized signals as a data augmentation source.

The core mechanism of SynBrain is to model fMRI responses as **semantically conditioned continuous probability distributions**, capturing biologically grounded neural variability through probabilistic learning while preserving functional consistency.

## Method

### Overall Architecture

SynBrain follows a two-stage training plus inference pipeline:
- **Stage 1**: Train BrainVAE to learn a probabilistic latent distribution over fMRI signals, conditioned on CLIP visual embeddings.
- **Stage 2**: Train the S2N Mapper to map CLIP embeddings into the BrainVAE latent space.
- **Inference**: The frozen S2N Mapper performs a one-step mapping from CLIP embeddings to the latent space; the BrainVAE decoder then generates the fMRI signal.

### Key Designs

1. **BrainVAE**: A variational autoencoder specifically designed for fMRI data. The encoder maps fMRI input $y_{\text{fMRI}} \in \mathbb{R}^{1 \times n}$ to a posterior distribution $q(z|y)$, parameterized by mean $\mu$ and log-variance $\log \sigma^2$, with latent samples drawn via the reparameterization trick as $z \sim \mathcal{N}(\mu, \sigma^2)$.

   **Architectural innovation**: The authors observe that MLP-based VAEs (MLP-VAE) suffer from training instability (diverging MSE) due to the lack of spatial inductive bias in MLPs. BrainVAE integrates **convolutional layers** (for local voxel feature extraction) and **attention layers** (for capturing long-range inter-voxel dependencies), yielding a smoother latent space. Experiments confirm that BrainVAE substantially outperforms MLP-AE and MLP-VAE in both convergence speed and semantic expressiveness.

   Training objective:
   $$\mathcal{L}_{\text{BrainVAE}} = \mathcal{L}_{\text{MSE}} + \lambda_{\text{KL}} \mathcal{L}_{\text{KL}} + \lambda_{\text{CLIP}} \mathcal{L}_{\text{CLIP}}$$

   - $\mathcal{L}_{\text{MSE}} = \|D(z) - y_{\text{fMRI}}\|_2^2$: voxel-level reconstruction fidelity
   - $\mathcal{L}_{\text{KL}} = D_{KL}(q(z|y_{\text{fMRI}}) \| \mathcal{N}(0,I))$: latent space regularization, $\lambda_{\text{KL}}=0.001$
   - $\mathcal{L}_{\text{CLIP}} = \text{SoftCLIP}(z, z_{\text{CLIP}})$: semantic alignment contrastive loss, $\lambda_{\text{CLIP}}=1000$

2. **S2N Mapper (Semantic-to-Neural Mapper)**: A lightweight Transformer module consisting of stacked multi-head self-attention layers and feed-forward networks. It implements a nonlinear transformation $f_{\text{S2N}}: \mathbb{R}^{m \times d} \rightarrow \mathbb{R}^{m \times d}$, directly mapping CLIP visual embeddings into the BrainVAE latent space. The training objective is an MSE loss:

   $$\mathcal{L}_{\text{S2N}} = \text{MSE}(f_{\text{S2N}}(z_{\text{CLIP}}), z)$$

   Compared to the diffusion-based alignment used in MindSimulator, the S2N Mapper achieves **one-step mapping**, eliminating the need for iterative denoising and avoiding the train-inference distribution mismatch.

3. **Few-shot cross-subject adaptation**: The entire BrainVAE is fine-tuned using only one hour of data from a new subject, while the S2N Mapper updates only the MLP sub-modules within the Transformer, enabling parameter-efficient adaptation.

### Loss & Training

- OpenCLIP ViT-bigG/14 is used as a frozen visual encoder.
- AdamW optimizer with lr=1e-4 and weight decay=0.05.
- BrainVAE employs early stopping to prevent overfitting; S2N Mapper is trained for 50K steps.
- Training completes within 2 hours on 4 A100 GPUs.

## Key Experimental Results

### Main Results: Subject-Specific fMRI Synthesis (Average over 4 Subjects)

| Method | MSE↓ | Pearson↑ | Incep↑ | CLIP↑ | Syn Retrieval↑ |
|--------|------|----------|--------|-------|----------------|
| MindSimulator (Trials=1) | .403 | .346 | 92.1% | 90.4% | - |
| MindSimulator (Trials=5) | .385 | .357 | 93.1% | 91.2% | - |
| **SynBrain (Trials=1)** | **.139** | **.687** | **95.7%** | **94.3%** | **92.5%** |

SynBrain with a single sample surpasses MindSimulator averaged over five samples. Notably, the raw fMRI retrieval accuracy is 84.8%, whereas SynBrain's synthesized fMRI achieves 92.5%, indicating that synthesized signals preserve semantic information more faithfully than raw signals.

### Ablation Study (Subject 1)

| Configuration | MSE↓ | Pearson↑ | CLIP↑ | Syn Retrieval↑ | Note |
|---------------|------|----------|-------|----------------|------|
| SynBrain | .079 | .715 | 95.9% | 99.3% | Full model |
| w/o variational sampling | .086 | .687 | 86.7% | 88.4% | Deterministic AE |
| w/o contrastive learning | .127 | .635 | 84.5% | 0.4% | No CLIP loss |
| w/o S2N Mapper | .105 | .564 | 75.0% | 50.5% | Direct contrastive alignment |

### Few-Shot Adaptation + Data Augmentation

| Method | CLIP↑ | Eff↓ | Brain Retrieval↑ |
|--------|-------|------|-----------------|
| MindEye2 (1h) | 80.8% | .798 | 77.6% |
| MindAligner (1h) | 81.8% | .800 | 86.9% |
| MindEye2+DA(1h) | **84.7%** | .770 | 82.0% |

Adding just one hour of synthesized data yields a 3.9% improvement in CLIP similarity, validating the effectiveness of synthesized fMRI as a data augmentation source.

### Key Findings

- Probabilistic modeling is critical: removing variational sampling reduces semantic alignment by ~9%, indicating that distribution-level learning captures functional consistency better than deterministic modeling.
- Contrastive learning is the foundation of semantic space alignment: its removal causes retrieval accuracy to collapse from 99.3% to 0.4%.
- The S2N Mapper bridges the modality gap: its removal degrades CLIP similarity from 95.9% to 75.0%.
- Cross-trial functional consistency: category-selective regions (e.g., fusiform face area) maintain consistent activation patterns across trials.
- Cross-subject functional consistency: only one hour of adaptation data suffices to produce activation patterns comparable to those from full-data training.

## Highlights & Insights

- Modeling neural responses as probability distributions rather than deterministic mappings accurately reflects the fundamental biological property of neural variability.
- The architectural design of BrainVAE (convolution + attention replacing pure MLP) resolves the training instability of VAEs on high-dimensional fMRI data.
- One-step mapping vs. diffusion models: the former is simpler and more efficient, and avoids distribution mismatch issues.
- The retrieval accuracy of synthesized fMRI exceeds that of raw fMRI, suggesting that the model learns to "denoise" and extract the semantic core of neural signals.

## Limitations & Future Work

- Reliance on the CLIP visual encoder may introduce representational biases that do not fully align with neural processing.
- The model cannot capture all sources of neural variability (e.g., attentional state fluctuations, neuromodulatory effects).
- Validation is limited to the NSD dataset; generalizability requires further investigation.
- The benefit of data augmentation plateaus or degrades with increasing amounts of synthesized data, necessitating optimization of the quality–diversity trade-off.

## Related Work & Insights

- The most direct comparison is with MindSimulator, whose stochasticity is introduced only at inference time via diffusion sampling, while the core generative process remains deterministic.
- The "probabilistic + semantic conditioning" paradigm of BrainVAE is generalizable to other neuroimaging modalities (EEG, MEG).
- The paradigm of using synthesized fMRI as data augmentation offers a promising new direction for addressing the scarcity of brain imaging data.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Probabilistic neural encoding model combined with one-step mapping, with thorough biological justification.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multi-subject evaluation, few-shot and data augmentation experiments, ablation studies, and brain functional analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation with tightly coupled method, experiments, and analysis.
- Value: ⭐⭐⭐⭐⭐ Direct contributions to both neuroscience and BCI research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Probabilistic Token Alignment for Large Language Model Fusion](probabilistic_token_alignment_for_large_language_model_fusion.md)
- [\[NeurIPS 2025\] Time-Evolving Dynamical System for Learning Latent Representations of Mouse Visual Cortex](time-evolving_dynamical_system_for_learning_latent_representations_of_mouse_visu.md)
- [\[NeurIPS 2025\] Toward Real-world Text Image Forgery Localization: Structured and Interpretable Data Synthesis](toward_real-world_text_image_forgery_localization_structured_and_interpretable_d.md)
- [\[NeurIPS 2025\] SHAP Values via Sparse Fourier Representation](shap_values_via_sparse_fourier_representation.md)
- [\[NeurIPS 2025\] VL-SAE: Interpreting and Enhancing Vision-Language Alignment with a Unified Concept Set](vlsae_interpreting_and_enhancing_visionlanguage_alignment_wi.md)

</div>

<!-- RELATED:END -->
