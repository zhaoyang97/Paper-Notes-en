---
title: >-
  [Paper Note] Community Forensics: Using Thousands of Generators to Train Fake Image Detectors
description: >-
  [CVPR 2025][Image Generation][AI-generated image detection] This work constructs the Community Forensics dataset containing 4,803 generative models and 2.7 million images. It reveals that scaling up the number of models, even those with similar architectures, significantly enhances the generalization of fake image detection, achieving a state-of-the-art average mAP of 0.966 across multiple benchmarks.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "AI-generated image detection"
  - "data diversity"
  - "community models"
  - "generalization detection"
  - "large-scale dataset"
date: 2026-05-08
content_hash: 32f8ca31123fa462
---

# Community Forensics: Using Thousands of Generators to Train Fake Image Detectors

**Conference**: CVPR 2025  
**arXiv**: [2411.04125](https://arxiv.org/abs/2411.04125)  
**Code**: [https://jespark.net/projects/2024/community_forensics](https://jespark.net/projects/2024/community_forensics) (Dataset)  
**Area**: Image Generation  
**Keywords**: AI-generated image detection, data diversity, community models, generalization detection, large-scale dataset

## TL;DR
This work constructs the Community Forensics dataset containing 4,803 generative models and 2.7 million images. It reveals that scaling up the number of models, even those with similar architectures, significantly enhances the generalization of fake image detection, achieving a state-of-the-art average mAP of 0.966 across multiple benchmarks.

## Background & Motivation

**Background**: AI-generated image detection faces severe generalization challenges—detectors perform well on the generative models used in the training set but experience sharp performance drops when facing unseen models. Each generator possesses unique architectures, loss functions, training data, and image processing pipelines, leading to highly distinct model-specific fingerprints.

**Limitations of Prior Work**: Existing detection datasets suffer from a severe lack of model diversity—even the largest, RED140, contains only 140 models. Wang et al. trained a detector using only 1 GAN, which relied heavily on data augmentation hyperparameters and generalized poorly to new models. Even datasets like Synthbuster and GenImage, which use more models, still feature no more than 20, falling far short of covering the diversity of generators encountered in the wild.

**Key Challenge**: While the open-source community (such as Hugging Face) hosts thousands of generative models, detector training data has only utilized a handful of them, resulting in a massive train-test distribution gap. A large-scale, systematically collected multi-model dataset is urgently needed.

**Goal**: To study how model diversity affects detection generalization by constructing a detection dataset of unprecedented scale and diversity.

**Key Insight**: Thousands of text-to-image models on Hugging Face share a unified interface via the `diffusers` library, enabling automated batch downloading and sampling. Although most of them are latent diffusion variants, each model's fine-tuning data, LoRA configurations, and image processing details differ, collectively covering a vast range of subtle variations.

**Core Idea**: Systematically download 4,763 open-source diffusion models + 19 hand-selected models + 11 commercial models to construct a dataset of 2.7 million images, proving that increasing the number of models (even with similar architectures) log-linearly scales up detection generalization.

## Method

### Overall Architecture
The dataset consists of three parts. **Systematic Collection**: 4,763 latent diffusion models were downloaded from Hugging Face based on download counts, with approximately 403 images sampled per model using captions from real datasets as prompts, totaling 1.9 million images. **Hand-selected Open-source Models**: 19 models of various architectures (GANs, pixel-space diffusion, autoregressive, etc.) with an average of 40K images each, totaling 774K images. **Commercial Models**: 11 models (DALL·E, Midjourney, FLUX, etc.) totaling 15K images, used solely for evaluation. The detector is trained end-to-end using standard CNN/ViT classifiers.

### Key Designs

1. **Large-scale Automated Model Sampling Pipeline**

    - **Function**: Systematically collects images from thousands of generative models in the model community.
    - **Mechanism**: Leverages the unified interface of the Hugging Face `diffusers` library to automatically download models, extract hyperparameters (inference steps, guidance scale, pipeline configurations), and sample images using text prompts derived from real datasets (such as LAION, ImageNet, COCO). Images are stored in PNG format to avoid JPEG compression bias. Only a few hundred images are sampled per model (as sampling more yields diminishing returns). Models incompatible with the automated pipeline (e.g., pixel-space diffusion models) are manually processed and reserved for the test set.
    - **Design Motivation**: Model diversity is more critical than the amount of images per model (a key finding). The automated pipeline makes collecting 4,763 models feasible.

2. **Log-Linear Relationship Between Model Quantity and Generalization**

    - **Function**: Proves that expanding the number of models during training is the core factor for enhancing detection generalization.
    - **Mechanism**: Fixing the total number of training images while varying only the number of models (1, 10, 100, 1,000). The results show that detection performance scales log-linearly with the number of models. This improvement is not only effective on same-type (latent diffusion) models but also exhibits even greater improvements on out-of-distribution models (GANs, pixel-space diffusion). This suggests that different fine-tuned versions of latent diffusion models capture distinct generative artifact patterns, allowing the detector to learn more general real-vs-fake distinguishing clues.
    - **Design Motivation**: No prior work has systematically investigated the impact of "model quantity" as a dimension of diversity on detection.

3. **Additional Gains from Cross-Architecture Diversity**

    - **Function**: Further enhances generalization by incorporating models across different architectures.
    - **Mechanism**: In addition to the 4,763 latent diffusion models, hand-selected models (GANs, autoregressive models, pixel-space diffusion, etc.) are included. Experiments demonstrate that architectural diversity brings significant additional gains—classifiers do not generalize perfectly across different architectures, and each new architecture provides unique detection signals.
    - **Design Motivation**: In-the-wild scenarios present generators with diverse architectures; relying solely on latent diffusion variants is insufficient.

### Loss & Training
Standard binary classification (real/fake) cross-entropy loss. CNN (ResNet) and ViT architectures are used for end-to-end training. Results indicate that simple end-to-end training is sufficient to achieve strong generalization (without requiring specialized architectures or pre-training), contrary to prior work concluding that CLIP features are necessary.

## Key Experimental Results

### Main Results (Average mAP across Benchmarks)

| Training Data | Wang↑ | Ojha↑ | Synthbuster↑ | GenImage↑ | Ours(Comp)↑ | Average |
|---------|-------|-------|-------------|----------|------------|------|
| Wang et al. | 0.897 | 0.696 | 0.516 | 0.642 | 0.537 | 0.648 |
| Ojha et al. | 0.939 | 0.957 | 0.620 | 0.797 | 0.592 | 0.760 |
| GenImage | 0.929 | 0.984 | 0.813 | 0.999 | 0.912 | 0.934 |
| **Ours** | **0.964** | **0.991** | **0.904** | **0.990** | **0.971** | **0.966** |
| Ours (High res) | 0.967 | 0.996 | 0.974 | 0.998 | 0.987 | 0.986 |

### Ablation Study

| Number of Training Models | Same-Architecture Test mAP | Cross-Architecture Test mAP |
|-------------|-------------|-------------|
| 1 Model | ~0.7 | ~0.5 |
| 100 Models | ~0.85 | ~0.7 |
| 1000 Models | ~0.92 | ~0.85 |
| **4763 Models** | **~0.97** | **~0.95** |

(Performance scales log-linearly with model quantity)

### Key Findings
- Even when all newly added models are latent diffusion variants, detection performance consistently improves across all generator types, including GANs and pixel-space diffusion (the most counter-intuitive finding).
- End-to-end CNN/ViT training is sufficient to obtain strong generalization, without requiring CLIP features or specialized architectures (contrary to the conclusions of Ojha et al.).
- High-resolution input (without resizing) further boosts the mAP to 0.986.
- The Synthbuster benchmark is the most challenging (mAP 0.904) because it utilizes Dresden real-world images (without JPEG compression), thereby eliminating compression bias.

## Highlights & Insights
- **The finding that "model quantity is more important than model type"** is highly practical—there is no need to wait for new architectures; continuously collecting latent diffusion variants from the community can steadily enhance detection performance.
- **The log-linear relationship** implies a clear scaling law for performance gains, allowing predictions on the benefits of collecting more models in the future.
- **The dataset construction methodology** itself is a reusable pipeline—any model compatible with the `diffusers` library can be automatically incorporated.

## Limitations & Future Work
- The dataset is heavily biased towards latent diffusion (which accounts for almost all of the 4,763 models), with only 19 hand-selected models for other architectures (GANs, autoregressive).
- Pixel-space diffusion models are incompatible with the automated pipeline and require manual processing.
- Essentially a data-driven approach—it requires continuously collecting new models to retain detection capability and does not learn architecture-agnostic generalized features.
- The robustness against adversarial attacks (such as post-processing evasion techniques) has not been investigated.

## Related Work & Insights
- **vs Wang et al.**: Trained with only 1 ProGAN model, generalizing poorly (average mAP 0.648 vs 0.966). This proves that single-model training is utterly insufficient.
- **vs Ojha et al.**: Used CLIP features + a linear classifier trained on 4 models. Ours achieves significantly superior performance via end-to-end training and 4,803 models.
- **vs GenImage**: Used 8 models with 1.4M images. Ours scales the number of models by 600× and improves average mAP from 0.934 to 0.966.
- **vs RED116/140**: Uses 116-140 models with only 1K images per model. Ours significantly leads in both the number of models and total image count.

## Rating
- Novelty: ⭐⭐⭐⭐ The systematic large-scale collection of community models is a novel angle, and the scaling law discovery is highly valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive with 5 evaluation benchmarks, detailed scaling analysis, and cross-architecture ablations.
- Writing Quality: ⭐⭐⭐⭐ Detailed dataset construction descriptions and in-depth experimental analyses.
- Value: ⭐⭐⭐⭐⭐ The dataset and methodology have a major impact on the AI safety domain, and the dataset has been publicly released.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Modeling Thousands of Human Annotators for Generalizable Text-to-Image Person Re-identification](modeling_thousands_of_human_annotators_for_generalizable_text-to-image_person_re.md)
- [\[ICLR 2026\] Why Adversarially Train Diffusion Models?](../../ICLR2026/image_generation/why_adversarially_train_diffusion_models.md)
- [\[ICML 2026\] Coarse-Grained Boltzmann Generators](../../ICML2026/image_generation/coarse-grained_boltzmann_generators.md)
- [\[NeurIPS 2025\] Flatten Graphs as Sequences: Transformers are Scalable Graph Generators](../../NeurIPS2025/image_generation/flatten_graphs_as_sequences_transformers_are_scalable_graph_generators.md)
- [\[CVPR 2026\] Erasing Thousands of Concepts: Towards Scalable and Practical Concept Erasure for Text-to-Image Diffusion Models](../../CVPR2026/image_generation/erasing_thousands_of_concepts_towards_scalable_and_practical_concept_erasure_for.md)

</div>

<!-- RELATED:END -->
