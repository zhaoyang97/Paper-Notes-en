---
title: >-
  [Paper Note] Image Generation Diversity Issues and How to Tame Them
description: >-
  [CVPR 2025][Image Generation][Generation Diversity] This work reveals that current diffusion models suffer from a severe lack of diversity (with state-of-the-art models covering only 77% of training data diversity). It proposes the Image Retrieval Score (IRS), a metric based on image retrieval that provides interpretable diversity measurement, and introduces Diversity-Aware Diffusion Models (DiADM) to enhance diversity without sacrificing generation quality.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Generation Diversity"
  - "Image Retrieval Score"
  - "Diffusion Model Evaluation"
  - "Diversity-Aware Generation"
  - "Feature Extractor Evaluation"
date: 2026-05-08
content_hash: a2a6896baa3458bc
---

# Image Generation Diversity Issues and How to Tame Them

**Conference**: CVPR 2025  
**arXiv**: [2411.16171](https://arxiv.org/abs/2411.16171)  
**Code**: [https://github.com/MischaD/beyondfid](https://github.com/MischaD/beyondfid)  
**Area**: Diffusion Models / Image Generation Evaluation  
**Keywords**: Generation Diversity, Image Retrieval Score, Diffusion Model Evaluation, Diversity-Aware Generation, Feature Extractor Evaluation

## TL;DR

This work reveals that current diffusion models suffer from a severe lack of diversity (with state-of-the-art models covering only 77% of training data diversity). It proposes the Image Retrieval Score (IRS), a metric based on image retrieval that provides interpretable diversity measurement, and introduces Diversity-Aware Diffusion Models (DiADM) to enhance diversity without sacrificing generation quality.

## Background & Motivation

**Background**: Generative models can now produce high-quality images nearly indistinguishable from real data, yet the issue of diversity has long been neglected. Unlike quality deficits, a lack of diversity is visually difficult to detect and requires specialized metrics. Existing metrics like FID primarily measure quality and distribution distance, while Precision/Recall, though touching upon diversity, rely heavily on hyperparameters and lack interpretability.

**Limitations of Prior Work**: (1) Feature extractors used for calculating existing metrics (such as Inception v3 and DINOv2) suffer from "collapse" in their feature spaces, failing to correctly measure diversity even for real data (leading to a "measurement gap"). (2) Existing diversity metrics lack interpretability—what does a one-point gap in FID actually mean? Does a Recall saturated at 0.8 guarantee sufficient diversity? (3) Methods to enhance diversity usually come at the cost of generation quality (e.g., through noise perturbation or fidelity degradation).

**Key Challenge**: Diversity and fidelity have long been considered a trade-off, and existing methods cannot decouple these two attributes. Meanwhile, there is a lack of a reliable and interpretable metric to quantify diversity.

**Goal**: (1) Design an interpretable diversity metric; (2) quantify the diversity gap in existing models; and (3) enhance diversity without sacrificing generation quality.

**Key Insight**: Diversity evaluation is framed as an image retrieval problem, where generated images are used to retrieve training images, and the fraction of retrieved training images represents diversity. This is combined with the probabilistic framework of the Coupon Collector's problem to provide statistical estimation and confidence intervals.

**Core Idea**: Measure the diversity of generative models using image retrieval, and decouple diversity and fidelity by utilizing pseudo-unconditional features as conditional inputs.

## Method

### Overall Architecture

The framework consists of two components: (1) The IRS metric: For each synthesized image, the nearest training image in the feature space is identified, and the deduplicated fraction of "retrieved" training images is counted. Based on the probabilistic model of the Coupon Collector's problem, the diversity upper bound under infinite sampling $\text{IRS}_\infty$ and its confidence intervals are inferred from a small sample size. (2) The DiADM model: Pseudo-labels are computed for training images using a pre-trained feature extractor and used as conditional inputs to replace placeholder labels, enabling the decoupling of diversity and fidelity.

### Key Designs

1. **Image Retrieval Score (IRS)**:

    - **Function**: Provide an interpretable, hyperparameter-free metric for generation diversity.
    - **Mechanism**: Define the "learned" image set as $\mathcal{X}_{learned} = \{x_t \in \mathcal{X} \mid \exists x_t' \in \mathcal{X}': x_t = \arg\min_{x_t} \mathcal{P}(x_t, x_t')\}$, where IRS is given by $N_{learned}/N_{train}$. The key innovation is using Stirling numbers and the probability distribution of the Coupon Collector's problem, $P(k,n,s) = \frac{\text{Stir}(n,k) \cdot s!}{(s-k)! \cdot s^n}$, to infer the maximum likelihood estimation $\text{IRS}_\infty$ and its upper/lower bounds from a small number of samples (much smaller than the training set size). To eliminate systemic bias caused by the collapse of feature extractors, an adjustment step $\text{IRS}_{\infty,a} = \text{IRS}_{\infty,snth}/\text{IRS}_{\infty,real}$ is introduced.
    - **Design Motivation**: Existing Recall metrics saturate at 0.8 as long as all classes are present, failing to distinguish between 80% and 100% diversity. Coverage depends heavily on hyperparameters and overestimates diversity in low-diversity scenarios. IRS correlates linearly with the ratio of actual classes in controlled experiments on ImageNet, providing an intuitive measurement.

2. **Feature Extractor Evaluation and Selection**:

    - **Function**: Find the feature space best suited for measuring diversity.
    - **Mechanism**: Evaluate the image retrieval performance of real data across 9 popular feature extractors (BYOL, CLIP, ConvNeXt, DINOv2, Inception, MAE, etc.). A pseudo ground truth is established via ensemble voting (where 5 or more models must agree on the same retrieval result). The consistency rate of each extractor with the ensemble consensus is measured, leading to the selection of SwAV as the default feature extractor.
    - **Design Motivation**: All feature extractors exhibit measurement errors (measurement gaps) even on real data, indicating that metrics like FID and Precision/Recall computed with them are inherently unreliable. By eliminating systemic bias through the adjustment step and selecting the extractor most consistent with the ensemble consensus, interpretability is maximized.

3. **Diversity-Aware Diffusion Models (DiADM)**:

    - **Function**: Enhance the generation diversity of unconditional diffusion models without degrading FID performance.
    - **Mechanism**: Extract features for each training image using a pre-trained Inception v3 to serve as pseudo-labels, replacing the placeholder labels commonly used in unconditional generation. Built upon the EDM-2-XS architecture, the label dimension is modified to match the feature dimension. During sampling, features of training images are directly used as conditional queries to ensure the model covers the entire training distribution. Essentially, this treats each training sample as its own "class", fulfilling the decoupling of diversity and fidelity.
    - **Design Motivation**: Unconditional generation lacks guiding signals, causing models to converge toward high-density regions of the distribution. Providing instance-level conditioning via pseudo-labels guides the model to cover long-tail samples.

### Loss & Training

DiADM is trained using the standard diffusion loss, with modifications only to the conditional inputs. The training budget is 574 A40 GPU hours (only 1/10 of the full EDM training). During sampling, training data features are used as conditions.

## Key Experimental Results

### Main Results

| Model | FID ↓ | IRS∞,a ↑(Conditional) | IRS∞,a ↑(Unconditional) |
|------|-------|----------------|------------------|
| ADM-256 | 6.01 | 0.44 | 0.20 |
| DiT-XL/2-256 | 22.15 | 0.23 | 0.33 |
| MAR-H-256 | 3.11 | 0.64 | 0.38 |
| EDM-2-XL-512 | 2.92 | **0.77** | 0.03 |
| EDM-2-XXL-512 | 2.87 | 0.75 | 0.05 |
| LDM-256 | 26.09 | 0.16 | 0.16 |

Even the best conditional model, EDM-2-XL, only recovers 77% of the training data diversity. The diversity of unconditional generation drops even more precipitously.

### Ablation Study

| Dataset | EDM FID | DiADM FID | EDM IRS∞,a | DiADM IRS∞,a |
|--------|---------|-----------|------------|--------------|
| ImageNet-512 | 51.59 | 22.28 | 0.09 | 0.15 |
| FFHQ | 40.92 | 6.24 | 0.23 | 1.51 |
| ChestX-ray14 | 24.29 | 6.76 | 0.19 | 1.08 |
| CelebV-HQ | 68.41 | 13.64 | 0.18 | 0.69 |

DiADM improves both FID and IRS across all datasets, demonstrating that diversity and quality can indeed be decoupled. On FFHQ, the IRS exceeds 1.0, suggesting that pseudo-conditioning guides the model to cover regions beyond the training distribution.

### Key Findings

- All existing feature extractors suffer from a severe measurement gap when gauging the diversity of real data, meaning that metrics like FID that rely on these extractors are inherently biased.
- Model scale correlates positively with diversity: scaling EDM from XS to XL increases IRS from 46% to 77%, while the FID gap is only 0.9 points.
- Conditional generation yields significantly better diversity than unconditional generation (IRS 0.77 vs 0.03), suggesting that guiding signals are crucial for diversity.
- Text-to-image models also exhibit severe diversity biases; for example, Deepfloyd yields only about 50% diversity when gender is left unspecified.

## Highlights & Insights

- **Framing diversity evaluation as an image retrieval problem is remarkably elegant**: The Coupon Collector's probabilistic model offers a mathematical foundation to estimate global diversity from small sample sizes, providing an output with intuitive physical meaning ("what percentage of the data the model has learned").
- **The discovery of the measurement gap is cautionary**: It uncovers systemic flaws in widely used feature extractors when evaluating diversity, thereby questioning the reliability of numerous conclusions based on FID or Recall.
- **The pseudo-label concept of DiADM is simple yet effective**: Utilizing pre-trained features as self-supervised "class labels" transforms unconditional generation into instance-level conditional generation, conceptually decoupling diversity and fidelity entirely. This approach can be extended to video and 3D generation.

## Limitations & Future Work

- Random fluctuations of IRS can be high when sample sizes are small, making minor IRS differences statistically insignificant.
- DiADM experiments were conducted under a constrained computational budget (1/10 of full training); the performance under full training remains unknown.
- The pre-trained Inception v3 used for pseudo-labels suffers from feature collapse itself, and employing better feature extractors could further enhance the policy.
- DiADM is only evaluated on unconditional generation; extending it to text-to-image generation is of great practical value but requires resolving potential conditional conflicts.
- IRS relies on nearest-neighbor search, which calls for optimization regarding computational efficiency on large-scale datasets.

## Related Work & Insights

- **vs FID/Precision/Recall**: These conventional metrics either fail to isolate diversity (FID conflates quality and diversity) or depend on hyperparameters and lack interpretability (Recall). IRS offers the only metric that directly maps to "data coverage."
- **vs Vendi Score**: Vendi Score operates without any reference set, thus failing to assess the relationship to training data. IRS uses the training set as a reference, making it better suited to evaluate whether a generative model has captured the full distribution.
- **vs Diversity Enhancement Methods (e.g., SDEdit noise, DDIM diversity)**: These methods trade off quality to obtain diversity. In contrast, DiADM achieves both simultaneously through decoupling.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Novel problem definition (quantifying the diversity gap), novel methodology (IRS + DiADM), and crucial discovery (measurement gap).
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluated across 9 feature extractors, 5 datasets, and over ten generative models, with a meticulously designed controlled validation.
- **Writing Quality**: ⭐⭐⭐⭐ Theoretical derivations are clear, though some symbol definitions are somewhat tedious.
- **Value**: ⭐⭐⭐⭐⭐ Highlights the widely neglected diversity issue in the community and provides a reliable measurement tool and optimization solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Trade-offs in Image Generation: How Do Different Dimensions Interact?](../../ICCV2025/image_generation/trade-offs_in_image_generation_how_do_different_dimensions_interact.md)
- [\[CVPR 2025\] IDEA-Bench: How Far are Generative Models from Professional Designing?](idea-bench_how_far_are_generative_models_from_professional_designing.md)
- [\[CVPR 2026\] One Algorithm to Align Them All](../../CVPR2026/image_generation/one_algorithm_to_align_them_all.md)
- [\[CVPR 2026\] DiverseGRPO: Mitigating Mode Collapse in Image Generation via Diversity-Aware GRPO](../../CVPR2026/image_generation/diversegrpo_mitigating_mode_collapse_in_image_generation_via_diversity-aware_grp.md)
- [\[CVPR 2025\] OmniGen: Unified Image Generation](omnigen_unified_image_generation.md)

</div>

<!-- RELATED:END -->
