---
title: >-
  [Paper Note] MPA: Multimodal Prototype Augmentation for Few-Shot Learning
description: >-
  [AAAI 2026][Medical Imaging][Few-Shot Learning] This paper proposes MPA, a framework that enhances prototype quality through three components: LLM-based Multi-Variant Semantic Enhancement (LMSE) for enriching semantic information, Hierarchical Multi-View Augmentation (HMA) for diversifying visual features, and an Adaptive Uncertain Class Absorber (AUCA) for modeling inter-class uncertainty. MPA achieves significant improvements over existing methods on 4 single-domain and 6 c…
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "Few-Shot Learning"
  - "Multimodal Prototype"
  - "LLM Semantic Enhancement"
  - "Data Augmentation"
  - "CLIP"
date: 2026-05-08
content_hash: 01c44376c0d972f7
---

# MPA: Multimodal Prototype Augmentation for Few-Shot Learning

**Conference**: AAAI 2026
**arXiv**: [2602.10143](https://arxiv.org/abs/2602.10143)  
**Code**: [GitHub](https://github.com/ww36user/MPA)  
**Area**: Few-Shot Learning / Multimodal Learning
**Keywords**: Few-Shot Learning, Multimodal Prototype, LLM Semantic Enhancement, Data Augmentation, CLIP

## TL;DR

This paper proposes MPA, a framework that enhances prototype quality through three components: LLM-based Multi-Variant Semantic Enhancement (LMSE) for enriching semantic information, Hierarchical Multi-View Augmentation (HMA) for diversifying visual features, and an Adaptive Uncertain Class Absorber (AUCA) for modeling inter-class uncertainty. MPA achieves significant improvements over existing methods on 4 single-domain and 6 cross-domain few-shot learning benchmarks, surpassing the second-best method by 12.29% and 24.56% under the 5-way 1-shot setting for single-domain and cross-domain scenarios, respectively.

## Background & Motivation

Few-shot learning (FSL) aims to recognize novel categories from only a small number of labeled samples. Prototype-based metric learning methods have attracted widespread attention due to their simplicity and efficiency. Such methods typically compute class prototypes from support set images and classify query samples based on their distance to prototypes.

However, existing methods suffer from two key limitations: (1) **Reliance on visual modality alone** — with limited support samples, visual information is insufficient to capture complete class representations, particularly when visually similar instances belong to different categories (e.g., different bird species with similar color and morphology but differing in beak shape and wing patterns); (2) **Lack of multi-view representation** — single-view images fail to adequately express the multi-dimensional characteristics of objects.

The core starting point of this paper is to incorporate multimodal information (semantic text + multi-view visual) and uncertainty modeling into the prototype construction process. Large language models are leveraged to generate rich class-level semantic descriptions to complement visual information; multi-level data augmentation is employed to obtain multi-view features for enhanced representational diversity; and an uncertain class is constructed to absorb boundary samples that are difficult to classify.

## Method

### Overall Architecture

The MPA framework is built upon CLIP and consists of three core modules: LMSE leverages LLMs to generate multi-variant semantic features embedded via the CLIP text encoder; HMA enriches visual diversity of support images through natural and geometric transformations; and AUCA constructs an adaptive uncertain class via interpolation and Gaussian sampling to absorb ambiguous samples. A logistic regression classifier is ultimately applied to the augmented multimodal features for classification.

### Key Designs

1. **LLM-based Multi-Variant Semantic Enhancement (LMSE)**:

    - Function: Leverages large language models to generate multi-variant semantic descriptions for each class, enriching the semantic information of the support set.
    - Mechanism: Given a class name $c$, an LLM (e.g., GPT-4.0) is prompted to generate an appearance description along with 4 paraphrased variants, yielding a semantic set $\{t_m\}_{m=1}^{M}$. The semantic variants are projected into the embedding space via the CLIP text encoder $h(\cdot)$: $F_t = h(\mathcal{G}_\text{LLM}(c))$. Support set image features are simultaneously extracted via the CLIP image encoder as $F_i = f(\{I_n\}_{n=1}^{N})$.
    - Design Motivation: Semantic descriptions focus on key discriminative attributes (without interference from noisy backgrounds), and LLM-generated variants introduce contextual diversity and implicit world knowledge, substantially improving generalization. Unlike fixed or hand-crafted class names, LLM variants provide broader coverage of the semantic space.

2. **Hierarchical Multi-View Augmentation (HMA)**:

    - Function: Generates multi-view features through a hierarchical data augmentation strategy to enhance visual diversity of the support set.
    - Mechanism: Two levels of augmentation are applied to each support image $I$: (a) natural-view augmentation $I_a = \{\tau_n(I) | \tau_n \in \mathcal{T}\}$, including center cropping (120, 170, 200 pixels), rotation (45°, 90°, 180°, 270°, 315°), and color jitter (brightness/contrast/saturation 0.5, hue 0.2); (b) geometric-view augmentation via horizontal flipping to produce complementary perspectives. All augmented images are encoded by the CLIP image encoder to extract features $F_a \in \mathbb{R}^{M \times d}$.
    - Design Motivation: This simulates natural variations in viewing distance, camera angle, and lighting conditions in real-world scenarios, increasing feature diversity without altering class labels and alleviating over-reliance on a small number of support samples.

3. **Adaptive Uncertain Class Absorber (AUCA)**:

    - Function: Dynamically constructs an uncertain class to absorb samples near class decision boundaries, reducing inter-class interference.
    - Mechanism: Mixed features are first generated via inter-class prototype interpolation: $D_i = [\alpha, 1-\alpha] \cdot [F_j; F_k]$ ($\alpha \in [0.2, 0.8]$), and random features are sampled from a standard normal distribution $D_n \sim \mathcal{N}(0,1)$. The cosine similarity matrix $\mathbf{S}$ is computed across all class prototype pairs, and the normalized average pairwise difference $\lambda = 1 - \frac{2}{\binom{C}{2}} \sum_{j<k} S'_{j,k}$ is obtained. The uncertain class data is then formed by mixing the two sources with probability $\lambda$: $\mathbb{E}[D_u] = (1-\lambda) \cdot D_n + \lambda \cdot D_i$.
    - Design Motivation: $\lambda$ adaptively reflects data characteristics — in cross-domain scenarios, features are more clustered and $\lambda$ is smaller (more randomness); in single-domain scenarios, features are more separable and $\lambda$ is larger (more interpolation). The uncertain class provides the classifier with negative-class information, enabling more robust decision boundaries.

### Loss & Training

- The feature extractor uses pretrained CLIP (ViT-L/14) with frozen weights
- LLM semantic generation is performed as an offline preprocessing step
- Final classification uses a logistic regression classifier trained on the augmented multimodal features
- 100 episodes are randomly sampled per epoch, each containing a support set and query set over 5 classes

## Key Experimental Results

### Main Results

**5-way 1-shot Single-Domain Datasets:**

| Method | miniImageNet | tieredImageNet | CIFAR-FS | FC100 | Average |
|--------|-------------|----------------|----------|-------|---------|
| SPM (AAAI'24) | 93.70 | 88.79 | 82.40 | 68.35 | 83.31 |
| MLVLM (AAAI'25) | 98.24 | 98.06 | 95.02 | - | - |
| **MPA (Ours)** | **98.87** | **98.57** | **97.47** | **87.47** | **95.60** |

**5-way 1-shot Cross-Domain Datasets:**

| Method | CUB | Cars | Places | Plantae | EuroSAT | CropDisease | Average |
|--------|-----|------|--------|---------|---------|-------------|---------|
| SVasP (AAAI'25) | 85.56 | 40.51 | 75.93 | 56.25 | 75.51 | 83.98 | 69.62 |
| SPM (AAAI'24) | 84.39 | 41.71 | 72.35 | 53.85 | 74.97 | 84.43 | 68.62 |
| **MPA (Ours)** | **98.95** | **98.51** | **93.55** | **91.73** | **87.05** | **95.28** | **94.18** |

MPA achieves an average improvement of 24.56% under the cross-domain 1-shot setting, with particularly remarkable gains on the Cars dataset (from 41.71% to 98.51%).

### Ablation Study

| LMSE | HMA | AUCA | EuroSAT | Places | CIFAR-FS |
|------|-----|------|---------|--------|----------|
| ✗ | ✗ | ✗ | 76.41 | 87.24 | 93.69 |
| ✓ | ✗ | ✗ | 83.03 | 93.43 | 95.36 |
| ✗ | ✓ | ✗ | 79.44 | 84.71 | 94.17 |
| ✓ | ✓ | ✗ | 85.69 | 92.64 | 96.32 |
| ✓ | ✓ | ✓ | **87.05** | **93.55** | **97.47** |

### Key Findings

- **LMSE contributes most**: When used alone, it improves EuroSAT by 6.62% (76.41→83.03) and Places by 6.19%
- **Strong complementarity among three modules**: Each successive module addition yields consistent gains; the full framework achieves the best performance across all datasets
- **Backbone-agnostic**: Consistently outperforms baselines across four CLIP backbones: ViT-L/14, ViT-B/32, ViT-B/16, and ResNet101
- **Robust to LLM choice**: Although GPT-4.0 performs best, the performance gap among 10 LLMs including GPT-3.5, DeepSeek, Claude-4, and Gemini-2.5 is minimal (97.79–98.57% on tieredImageNet)
- **Remarkable cross-domain performance**: MPA achieves 99.63% on Cars under the 5-shot setting, surpassing the second-best method by 33.16%

## Highlights & Insights

- **Multimodal prototypes constitute an effective paradigm for FSL**: Joint augmentation with LLM semantics and multi-view visual features substantially improves prototype quality, with particularly pronounced advantages in fine-grained tasks
- **Adaptive mechanism of AUCA is elegantly designed**: $\lambda$ automatically adjusts the composition of the uncertain class based on inter-class similarity, requiring no manual hyperparameter tuning
- **Performance gains are remarkably large**: Especially in cross-domain and fine-grained scenarios (Cars: ~40% → ~99%)
- **Method is simple and practical**: Built on frozen CLIP features with a logistic regression classifier, requiring no complex training pipelines

## Limitations & Future Work

- Strong reliance on the quality of CLIP pretraining and LLM generation capability; performance may be limited in specialized domains not well covered by CLIP (e.g., medical imaging with domain-specific terminology)
- Logistic regression may not be the optimal classifier; more sophisticated classification heads could potentially yield further improvements
- Data augmentation strategies (crop sizes, rotation angles, etc.) are manually specified; adaptive augmentation policies may be more effective
- A substantial portion of performance gains may be attributed to CLIP's inherently strong representational capacity, as the baseline (no modules, CLIP features + logistic regression only) already achieves high accuracy on several datasets
- LLM semantic generation is an offline step; for entirely novel classes, additional LLM inference is required, increasing deployment complexity

## Related Work & Insights

This work lies at the intersection of FSL and multimodal learning. The idea behind LMSE is related to SPM (which uses semantic information as prompts) and SEVPro (semantically enhanced visual prototypes), but goes further by leveraging LLMs to generate multi-variant descriptions. The uncertain class concept in AUCA is novel and offers a new perspective for modeling decision boundaries in few-shot scenarios. The overall framework suggests that, in low-shot regimes, "supplementing limited data with more modalities and perspectives" may be more effective than "developing more complex meta-learning algorithms." Applying similar ideas to other low-resource learning tasks warrants further exploration.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Interpretable Cross-Domain Few-Shot Learning with Rectified Target-Domain Local Alignment](../../CVPR2026/medical_imaging/interpretable_cross-domain_few-shot_learning_with_rectified_target-domain_local_.md)
- [\[CVPR 2026\] Reclaiming Lost Text Layers for Source-Free Cross-Domain Few-Shot Learning](../../CVPR2026/medical_imaging/reclaiming_lost_text_layers_for_source-free_cross-domain_few-shot_learning.md)
- [\[CVPR 2026\] Mind the Discriminability Trap in Source-Free Cross-domain Few-shot Learning](../../CVPR2026/medical_imaging/mind_the_discriminability_trap_in_source-free_cross-domain_few-shot_learning.md)
- [\[CVPR 2026\] Universal-to-Specific: Dynamic Knowledge-Guided Multiple Instance Learning for Few-Shot Whole Slide Image Classification](../../CVPR2026/medical_imaging/universal-to-specific_dynamic_knowledge-guided_multiple_instance_learning_for_fe.md)
- [\[CVPR 2026\] MUSE: Harnessing Precise and Diverse Semantics for Few-Shot Whole Slide Image Classification](../../CVPR2026/medical_imaging/muse_harnessing_precise_and_diverse_semantics_for_few-shot_whole_slide_image_cla.md)

</div>

<!-- RELATED:END -->
