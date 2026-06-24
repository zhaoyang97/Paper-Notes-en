---
title: >-
  [Paper Note] FedDifRC: Unlocking the Potential of Text-to-Image Diffusion Models in Heterogeneous Federated Learning
description: >-
  [ICCV 2025][Image Generation][Federated Learning] This paper is the first to introduce the internal representations of a pretrained text-to-image diffusion model (Stable Diffusion) into federated learning, proposing the FedDifRC framework. Through two complementary modules—Text-Driven Diffusion Contrastive Learning (TDCL) and Noise-Driven Diffusion Consistency Regularization (NDCR)—the framework effectively mitigates data heterogeneity and achieves significant performance imp…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Federated Learning"
  - "Data Heterogeneity"
  - "Diffusion Model Representations"
  - "Contrastive Learning"
  - "Consistency Regularization"
date: 2026-05-08
content_hash: a06eff1b7ae4f8fc
---

# FedDifRC: Unlocking the Potential of Text-to-Image Diffusion Models in Heterogeneous Federated Learning

**Conference**: ICCV 2025
**arXiv**: [2507.06482](https://arxiv.org/abs/2507.06482)  
**Code**: [https://github.com/hwang52/FedDifRC](https://github.com/hwang52/FedDifRC)  
**Area**: Image Generation
**Keywords**: Federated Learning, Data Heterogeneity, Diffusion Model Representations, Contrastive Learning, Consistency Regularization

## TL;DR

This paper is the first to introduce the internal representations of a pretrained text-to-image diffusion model (Stable Diffusion) into federated learning, proposing the FedDifRC framework. Through two complementary modules—Text-Driven Diffusion Contrastive Learning (TDCL) and Noise-Driven Diffusion Consistency Regularization (NDCR)—the framework effectively mitigates data heterogeneity and achieves significant performance improvements on global models across diverse non-iid settings.

## Background & Motivation

One of the core challenges in federated learning (FL) is data heterogeneity (non-iid): divergent local data distributions across clients lead to inconsistent local optimization directions, causing slow and unstable global model convergence.

**Limitations of Prior Work**:
- **Client-side optimization methods** (FedProx, SCAFFOLD, etc.): Reduce gradient inconsistency by constraining local updates relative to the global model, but cannot fundamentally prevent local models from overfitting to local distributions.
- **Server-side aggregation methods** (FedNova, etc.): Improve global aggregation strategies, yet parameter drift continues to accumulate.
- **Synthetic data-based methods**: Use diffusion models to generate synthetic data for augmentation, but synthetic data can still cause local models to overfit local domain distributions, leaving the heterogeneity problem unresolved.

**Core Insight**: Pretrained Stable Diffusion models encode rich visual-semantic representations. Through t-SNE visualization (Fig. 2), the authors observe that even without task-specific training, the UNet decoder of an SD model can naturally cluster samples of different categories at appropriate timesteps and layer depths. This motivates two key observations:

**The broad general knowledge in diffusion models can enhance local semantic diversity in FL** (→ TDCL)

**The smooth semantic correspondences encoded in diffusion models serve as natural guidance signals for FL** (→ NDCR)

## Method

### Overall Architecture

FedDifRC augments the standard FL pipeline (FedAvg) by adding two pretrained SD-based regularization modules to each client's local training. The SD model parameters are **completely frozen** throughout training and used solely for feature extraction. The overall loss function is:

$$\mathcal{L} = \mathcal{L}_{TDCL} + \mathcal{L}_{NDCR} + \mathcal{L}_{CE}$$

### Key Designs

1. **Conditional Diffusion Representations**:

    - **Function**: Leverages conditional generative feedback from the SD model to construct rich class-relevant semantic representations for each sample.
    - **Mechanism**: The feature encoding $\mathbf{c}_i = h_k(x_i)$ of sample $x_i$ is injected as a condition into the SD model, paired with a text prompt $\mathcal{P}_{y_i}$ = "a photo of a [class name]". Features are extracted from layers 2–4 of the UNet decoder, reduced via PCA, and concatenated into a fused representation $\widetilde{\mathcal{F}}_i$.
    - **Design Motivation**: K-Means clustering (Fig. 3) and PCA visualization (Fig. 4) show that fusing multi-layer features captures both high-level semantics and low-level texture, yielding more comprehensive representations than any single layer.

2. **Text-Driven Diffusion Contrastive Learning (TDCL)**:

    - **Function**: Constructs cross-modal contrastive learning based on diffusion representations to enhance the local model's class discriminability.
    - **Mechanism**: For each sample embedding $z_i$, the conditional diffusion representation $\widetilde{\mathcal{F}}_i$ generated with the matching text prompt serves as a positive pair, while representations $\widetilde{\mathcal{F}}_{\mathcal{N}_i}$ generated with non-matching prompts serve as negative pairs. A modified InfoNCE loss is applied: $\mathcal{L}_{TDCL} = \log(1 + \frac{\sum_j \exp(s(z_i, \widetilde{\mathcal{F}}_j)/\tau)}{\exp(s(z_i, \widetilde{\mathcal{F}}_i)/\tau)})$
    - Similarity is computed using normalized cosine similarity, with normalization factor $\mathcal{U}$ defined as the mean distance between all sample embeddings and the current diffusion representation.
    - **Design Motivation**: Positive and negative pairs are derived from diffusion model feedback conditioned on the same input but different text conditions, providing rich cross-domain variation signals that guide local models toward more generalizable class-discriminative representations.

3. **Noise-Driven Diffusion Consistency Regularization (NDCR)**:

    - **Function**: Uses denoising diffusion representations as stable convergence targets to constrain the feature space of local models.
    - **Mechanism**: Noise is added to input $x_i$ for $t$ steps before being passed to the SD model for denoising; UNet decoder features are extracted and fused into $\widetilde{\mathcal{H}}_i$, and aligned via an L2 loss: $\mathcal{L}_{NDCR} = \sum_{q=1}^{d}(z_{i(q)} - \widetilde{\mathcal{H}}_{i(q)})^2$
    - **Design Motivation**: The conditional diffusion representations in TDCL depend on dynamically generated conditions that vary each round and cannot provide stable convergence signals. Denoising diffusion representations, derived from the fixed denoising process of a frozen SD model, act as a "virtual teacher" offering consistent feature-level alignment targets.

### Loss & Training

- Total loss: $\mathcal{L} = \mathcal{L}_{TDCL} + \mathcal{L}_{NDCR} + \mathcal{L}_{CE}$, where $\mathcal{L}_{CE}$ is standard cross-entropy.
- The SD model is completely frozen and serves only as a representation extractor.
- The framework is extensible to a self-supervised setting: TDCL uses "a photo of a similar object" as the positive prompt and randomly sampled Tiny-ImageNet class names for negative prompts; NDCR uses "a photo of a visual object" to generate denoising representations.
- The authors provide theoretical convergence analysis under non-convex objectives, deriving conditions on the number of communication rounds $R$ and learning rate $\eta$.

## Key Experimental Results

### Main Results (CIFAR-10, Various Non-iid Settings, Accuracy %)

| Method | NID1_0.05 | NID1_0.2 | NID1_0.5 | NID2 | AVG |
|--------|-----------|----------|----------|------|-----|
| FedAvg | 78.27 | 84.65 | 86.11 | 72.60 | 80.41 |
| FedProx | 78.42 | 84.59 | — | 72.81 | — |
| MOON | 80.79 | 86.10 | — | 73.35 | — |
| FedNH | 80.25 | — | — | — | — |
| **FedDifRC** | **83.14** | **88.27** | **89.31** | **76.45** | **84.29** |

### Ablation Study (CIFAR-10, NID1_0.2 and Varying Layer Configurations)

| Configuration | NID1_0.05 | NID1_0.2 | NID1_0.5 | NID2 | Notes |
|---------------|-----------|----------|----------|------|-------|
| Baseline (FedAvg) | 78.27 | 84.65 | 86.11 | 72.60 | No diffusion model assistance |
| + TDCL only | 81.39 | 86.03 | 88.16 | 75.67 | Contrastive learning is effective |
| + NDCR only | 80.35 | 86.40 | 87.54 | 75.33 | Consistency regularization is effective |
| + TDCL + NDCR | **83.14** | **88.27** | **89.31** | **76.45** | Two modules are complementary |
| Layer L=2 only | — | 87.28 | — | 75.73 | High-level semantics |
| Layer L=3 only | — | 87.81 | — | 75.61 | Low-level texture |
| Fused L={2,3,4} | — | **88.27** | — | **76.45** | Best performance (+0.46) |

### Key Findings
- TDCL and NDCR are complementary: each module alone yields ~2% improvement, while their combination yields ~4%.
- Fusing multi-layer features (L={2,3,4}) consistently outperforms any single-layer configuration, though the margin is modest (+0.28–0.72%).
- Denoising timestep $t=300$ is optimal (Fig. 6 left); excessively large values (e.g., $t=999$) produce ambiguous and inseparable representations.
- The framework generalizes effectively to long-tailed distributions, domain shift, and other heterogeneity scenarios.
- The self-supervised variant (without labeled data) also demonstrates competitive performance.

## Highlights & Insights
- This is the first work to systematically explore the use of pretrained diffusion model internal representations to enhance federated learning, opening a new direction in FL research.
- Detailed t-SNE and K-Means analyses (Figs. 2–4) provide intuitive empirical validation that diffusion models are effective representation learners.
- The paper theoretically demonstrates that the SD model's denoising process is equivalent to a linear autoencoder learning the principal component space of the data (Eq. 6), providing a theoretical foundation for exploiting diffusion representations.
- TDCL and NDCR address two distinct challenges—positive/negative sample construction in contrastive learning and convergence stability—resulting in a conceptually clean and modular design.

## Limitations & Future Work
- Deploying a pretrained SD model on each client for inference increases local computational and storage overhead.
- The number of PCA components (256/128) must be predefined and may not be optimal across all datasets.
- The text prompt template is fixed as "a photo of a [class]", which may lack expressiveness for fine-grained categories.
- Validation is limited to image classification; applicability to downstream tasks such as object detection and semantic segmentation remains unexplored.
- The potential of more recent diffusion architectures (e.g., DiT) as representation extractors has not been investigated.

## Related Work & Insights
- DIFT and Diffusion Hyperfeatures have established that intermediate features of diffusion models are powerful visual representations; this paper is the first to apply such findings to FL.
- Contrastive learning in FL has been explored in prior work (MOON, FedCR); FedDifRC's novelty lies in using a diffusion model as the "anchor" for contrastive learning.
- The broader paradigm of leveraging representations from large generative models to assist discriminative task training is transferable to other distributed learning scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Unlocking the Potential of Diffusion Priors in Blind Face Restoration](unlocking_the_potential_of_diffusion_priors_in_blind_face_restoration.md)
- [\[ICLR 2026\] Bridging Generalization Gap of Heterogeneous Federated Clients Using Generative Models](../../ICLR2026/image_generation/bridging_generalization_gap_of_heterogeneous_federated_clients_using_generative_.md)
- [\[ICCV 2025\] PLA: Prompt Learning Attack against Text-to-Image Generative Models](pla_prompt_learning_attack_against_text-to-image_generative_models.md)
- [\[ICCV 2025\] REPA-E: Unlocking VAE for End-to-End Tuning with Latent Diffusion Transformers](repa-e_unlocking_vae_for_end-to-end_tuning_of_latent_diffusion_transformers.md)
- [\[ICCV 2025\] Golden Noise for Diffusion Models: A Learning Framework](golden_noise_for_diffusion_models_a_learning_framework.md)

</div>

<!-- RELATED:END -->
