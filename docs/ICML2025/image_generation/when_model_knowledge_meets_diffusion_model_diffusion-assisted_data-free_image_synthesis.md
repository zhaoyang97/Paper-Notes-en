---
title: >-
  [Paper Note] DDIS: When Model Knowledge Meets Diffusion Model
description: >-
  [ICML2025][Image Generation][Data-Free Image Synthesis] This work proposes DDIS, the first data-free image synthesis method that leverages text-to-image (T2I) diffusion models as image priors. By aligning Batch Normalization (BN) layer statistics during diffusion sampling via Domain Alignment Guidance (DAG), and encoding class-specific attributes through a Class Alignment Token (CAT), DDIS significantly outperforms existing DFIS methods on ImageNet-1k and multi-domain PACS.
tags:
  - "ICML2025"
  - "Image Generation"
  - "Data-Free Image Synthesis"
  - "Diffusion Model"
  - "Domain Alignment"
  - "Class Alignment Token"
  - "Knowledge Distillation"
date: 2026-05-08
content_hash: c9cbecbe05497823
---

# DDIS: When Model Knowledge Meets Diffusion Model

**Conference**: ICML2025  
**arXiv**: [2506.15381](https://arxiv.org/abs/2506.15381)  
**Code**: None  
**Area**: Diffusion Models / Data-Free Learning  
**Keywords**: Data-Free Image Synthesis, Diffusion Model, Domain Alignment, Class Alignment Token, Knowledge Distillation

## TL;DR
This work proposes DDIS, the first data-free image synthesis method that leverages text-to-image (T2I) diffusion models as image priors. By aligning Batch Normalization (BN) layer statistics during diffusion sampling via Domain Alignment Guidance (DAG), and encoding class-specific attributes through a Class Alignment Token (CAT), DDIS significantly outperforms existing DFIS methods on ImageNet-1k and multi-domain PACS.

## Background & Motivation

**Background**: Open-source pre-trained models are widely available (e.g., via the Hugging Face platform), but their training data is often inaccessible due to privacy or copyright concerns. Data-Free Image Synthesis (DFIS) addresses this issue by synthesizing surrogate data from the internal knowledge of pre-trained models, enabling downstream tasks (e.g., knowledge distillation, model pruning) to proceed without the original dataset.

**Limitations of Prior Work**: Traditional DFIS methods like DeepInversion directly optimize noise inputs in the high-dimensional pixel space. Without natural image priors, they face an immense search space. The synthesized images often contain unnatural artificial artifacts and deviate severely from the training data distribution, limiting the utility of synthetic data in downstream tasks.

**Key Challenge**: DFIS requires generating images that match the training distribution without any direct information about the original training dataset. However, the search space (the entire natural image space) is too vast, and generation easily deviates without guidance.

**Goal**: (1) How to leverage the powerful natural image prior of diffusion models to reduce the search space? (2) How to extract domain and class knowledge from pre-trained models to guide the diffusion generation process? (3) How to resolve semantic ambiguity in class labels?

**Key Insight**: The running statistics of BN layers encode the domain distribution of the training set, while a learnable pseudo-word embedding can capture fine-grained class attributes not explicitly expressed by class label names. These two knowledge sources can be injected into the sampling process of T2I diffusion models.

**Core Idea**: Utilizing the BN statistics of pre-trained models as domain guidance and optimization-based token embeddings as class guidance to drive T2I diffusion models to generate synthetic data highly aligned with the training distribution.

## Method

### Overall Architecture
DDIS is built upon Stable Diffusion 2.1 and introduces a two-tier guidance mechanism during the diffusion sampling process. The inputs are class labels and a pre-trained classifier, and the outputs are synthetic images aligned with the training distribution. The pipeline consists of: (1) constructing a text prompt containing the CAT; (2) performing step-by-step denoising from Gaussian noise, applying DAG domain guidance to correct the noisy latent variables at each step; (3) passing the final image through the classifier to compute CE loss and optimize the CAT embedding.

### Key Designs

1. **Domain Alignment Guidance (DAG)**:

    - **Function**: Guides the noisy latent variables at each diffusion sampling step so that the internal feature statistics of the generated images align with the running statistics of the BN layers in the pre-trained model.
    - **Mechanism**: The running mean $\mu_l$ and running variance $\sigma_l^2$ of BN layers encode the domain feature distribution of the entire training set. DAG decomposes the conditional score function into an unconditional score plus a statistical alignment gradient term. Specifically, at each timestep $t$, the latent variable $\mathbf{z}_t$ is first decoded into a pixel-space image $\hat{\mathbf{x}}_t = \mathcal{D}(\mathbf{z}_t)$. The image is passed through the classifier to compute the discrepancy loss between the image feature statistics and the BN statistics: $\mathcal{L}_{BN} = \sum_l (\|\mu_l(\hat{\mathbf{x}}_t) - \mu_l\|^2 + \|\sigma_l^2(\hat{\mathbf{x}}_t) - \sigma_l^2\|^2)$. The latent variable is then updated using the gradient of this loss: $\tilde{\mathbf{z}}_t = \mathbf{z}_t - \eta \nabla_{\mathbf{z}_t} \mathcal{L}_{BN}$.
    - **Design Motivation**: Directly employing Classifier Guidance requires training time-dependent classifiers for each timestep (which is impossible under data-free settings), whereas BN statistics are plug-and-play, stable guiding signals across different timesteps.

2. **Class Alignment Token (CAT)**:

    - **Function**: Learns a pseudo-word embedding to encode fine-grained class attributes that are not explicitly represented by class label names.
    - **Mechanism**: A new token $S_c$ is defined for each class $c$ to construct the prompt "A/An $\{S_c\}$ $\{$class label$\}$". The embedding $v_c$ of $S_c$ is optimized by minimizing the CE loss of the classifier: $\mathcal{L}_{CE}(f(\hat{\mathbf{x}}_0; \theta^*), \mathbf{c})$, where $\hat{\mathbf{x}}_0$ is the final generated image guided by DAG. Only the image from the final step is optimized (since $p(\mathbf{x}) \approx p(\hat{\mathbf{x}}_0)$), freezing all SD parameters and updating only a single token embedding.
    - **Design Motivation**: Class labels like "dog" lack specific breed details, and "crane" can mean either a bird or a construction machine. By interacting with the classifier, CAT learns the true visual attributes of the class in the training data, while simultaneously resolving semantic ambiguities.

3. **Integrated Sampling of DAG and CFG**:

    - **Function**: Unifies domain guidance and text-conditioned generation into a single sampling pipeline.
    - **Mechanism**: First, DAG is used to modify the latent variable to $\tilde{\mathbf{z}}_t$, and then Classifier-Free Guidance is performed based on the modified latent variable: $\tilde{\epsilon}_t = \epsilon_\theta(\tilde{\mathbf{z}}_t; \varnothing, t) + s(\epsilon_\theta(\tilde{\mathbf{z}}_t; \tau_\phi(\mathbf{y}), t) - \epsilon_\theta(\tilde{\mathbf{z}}_t; \varnothing, t))$. Generation is completed in 30 steps using a DDIM sampler.
    - **Design Motivation**: CFG can only control semantic consistency with text, but cannot provide domain knowledge guidance. Treating DAG as a preprocessing step prior to CFG achieves orthogonal and complementary benefits between the two.

### Loss & Training
For the optimization of CAT embeddings, the Adam optimizer is used with a learning rate of 0.005, running for at most 30 epochs with 20 gradient accumulations per epoch. A gradient skipping strategy is adopted—gradients are backpropagated only for the final denoising step to save GPU memory. Early stopping is triggered when more than 70% of the samples in a batch are correctly predicted by the classifier. All parameters of SD are frozen during the entire process, with only a single 1×784 dimensional token embedding being optimized.

## Key Experimental Results

### Main Results: Synthetic Image Quality (IS↑/FID↓/Precision↑/Recall↑)

| Dataset | Domain | DDIS IS/FID | DeepInversion IS/FID | PlugInInversion IS/FID |
|--------|------|------------|---------------------|----------------------|
| ImageNet-1k | Photo | **15.92/30.31** | 9.52/187.63 | 3.51/220.62 |
| PACS | Art Painting | **4.12/133.37** | 4.00/188.53 | 2.53/208.73 |
| PACS | Cartoon | **4.04/85.41** | 3.91/148.94 | 2.81/275.86 |
| Style-Aligned | Caricature | **3.94/139.75** | 3.58/195.25 | 2.51/293.58 |
| Style-Aligned | Manga | **3.87/145.82** | 3.32/206.57 | 2.36/295.14 |

### Ablation Study (PACS Art Painting)

| Configuration | IS↑ | FID↓ | Precision↑ | Recall↑ |
|------|-----|------|-----------|---------|
| Vanilla SD | 2.88 | 193.57 | 0.6429 | 0.2572 |
| SD + CAT (w/o DAG) | 3.29 | 174.31 | 0.6995 | 0.3074 |
| SD + DAG (w/o CAT) | 3.95 | 166.22 | 0.6871 | 0.2843 |
| **DDIS (DAG + CAT)** | **4.12** | **133.37** | **0.7742** | **0.3213** |

### Key Findings
- DAG and CAT are complementary: DAG mainly contributes to domain alignment (greatly reducing FID), while CAT mainly contributes to class accuracy (obviously improving Precision). The combination of both yields the best results.
- CAT resolves semantic ambiguity: On ambiguous classes such as "tiger cat", "beach wagon", and "mail bag", vanilla SD generates incorrect concepts, whereas DDIS learns the correct visual semantics through CAT.
- In data-free knowledge distillation, using synthetic data from DDIS for ResNet-34 $\to$ ResNet-18 distillation achieves 41.68% Top-1 accuracy on ImageNet-1k (vs. DeepInversion 4.67% and PlugInInversion 2.01%).
- Generation Efficiency: Synthesizing 100k ImageNet images requires only 30k iterations (vs. 8M iterations in DeepInversion), with a total training duration of 126 hours vs. 18,444 hours.

## Highlights & Insights
- Utilizing diffusion models as image priors for DFIS represents a paradigm shift—moving from "searching in pixel space" to "guiding generative models," which shrinks the search space from infinity to the natural image manifold of diffusion models. This idea is transferrable to other tasks requiring data-free or few-shot generation.
- Employing BN statistics as a bridge for domain knowledge is highly clever—running mean/variance serves as the "fingerprint" of the training set and remains stable across diffusion timesteps, making it an ideal guidance signal.
- The semantic disambiguation capability of CAT is an unexpected byproduct—a single token embedding can encode sufficient class semantics to distinguish homophones.

## Limitations & Future Work
- Reliance on BN Layers: The method is only applicable to CNN classifiers containing BN (e.g., ResNet/VGG), and cannot be easily applied to models using LayerNorm (e.g., ViTs), limiting its generalizability.
- Failure in the Sketch Domain: The abstract sketch style deviates too far from the natural image prior of SD, preventing DAG from providing effective guidance.
- Independent CAT Optimization per Class: 1,000 classes in ImageNet requires 1,000 independent optimization processes. Although each run takes only 7.5 minutes, the total overhead is non-trivial.
- Biases inherent in SD itself might propagate into the synthesized data.
- For scenarios where the training data distribution is highly different from SD pre-training data (e.g., medical images), the performance might be limited.
- DAG requires decoding the latent variable back to pixel space and performing a classifier forward pass to compute gradients at each step, which increases sampling time.

## Related Work & Insights
- **vs DeepInversion**: DeepInversion directly optimizes pixels alongside BN regularization, whereas DDIS guides generative diffusion models. The latter possesses a much stronger natural image prior, achieving 4-6x improvement in terms of FID.
- **vs Textual Inversion (Gal et al.)**: Both optimize pseudo-word embeddings, but Textual Inversion relies on reference images, whereas DDIS is completely data-free—driving the optimization via the classifier's CE loss instead of image reconstruction loss.
- **vs NaturalInversion (Kim et al. 2022)**: NI only handles small-scale datasets, whereas DDIS is the first to achieve high-quality synthesis at the scale of ImageNet-1k.
- Insights: DAG can be extended into a model-agnostic statistical alignment guidance (without relying on BN), or paired with LoRA as an alternative to CAT for stronger model adaptation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to introduce diffusion models to DFIS, pioneering a new paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covering multiple domains (photo/art/cartoon/manga/caricature), multiple tasks (synthesis/distillation/pruning), and extensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clearly structured with coherent logic flowing from motivation and method to experiments.
- Value: ⭐⭐⭐⭐⭐ Significantly advances the field of data-free learning, delivering a monumental improvement in the quality of synthesized data.
- Overall: ⭐⭐⭐⭐⭐ A milestone study in the DFIS field, paving a promising new direction of leveraging diffusion models for data-free learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Broadband Ground Motion Synthesis by Diffusion Model with Minimal Condition](broadband_ground_motion_synthesis_by_diffusion_model_with_minimal_condition.md)
- [\[CVPR 2025\] See Further When Clear: Curriculum Consistency Model](../../CVPR2025/image_generation/see_further_when_clear_curriculum_consistency_model.md)
- [\[CVPR 2025\] Using Powerful Prior Knowledge of Diffusion Model in Deep Unfolding Networks for Image Compressive Sensing](../../CVPR2025/image_generation/using_powerful_prior_knowledge_of_diffusion_model_in_deep_unfolding_networks_for.md)
- [\[ICML 2025\] Stealix: Model Stealing via Prompt Evolution](stealix_model_stealing_via_prompt_evolution.md)
- [\[NeurIPS 2025\] LLM Meets Diffusion: A Hybrid Framework for Crystal Material Generation](../../NeurIPS2025/image_generation/llm_meets_diffusion_a_hybrid_framework_for_crystal_material_generation.md)

</div>

<!-- RELATED:END -->
