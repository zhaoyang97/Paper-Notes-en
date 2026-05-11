---
title: >-
  [Paper Note] What Is Wrong with Synthetic Data for Scene Text Recognition? A Strong Synthetic Engine with Diverse Simulations and Self-Evolution
description: >-
  [CVPR2026][Scene Text Recognition] This paper systematically analyzes the deficiencies of existing rendered synthetic data in corpus, font, and layout diversity, and proposes the UnionST synthetic engine together with a Self-Evolution Learning (SEL) framework. Using only synthetic data, the approach substantially outperforms conventional synthetic sets; combined with SEL, only 9% of real labeled data is required to approach fully supervised performance.
tags:
  - CVPR2026
  - Scene Text Recognition
  - Synthetic Data
  - Data Engine
  - Self-Evolution Learning
  - Pseudo Labels
date: 2026-05-08
content_hash: a187f08612264ed5
---

# What Is Wrong with Synthetic Data for Scene Text Recognition? A Strong Synthetic Engine with Diverse Simulations and Self-Evolution

**Conference**: CVPR2026
**arXiv**: [2602.06450](https://arxiv.org/abs/2602.06450)
**Code**: [YesianRohn/UnionST](https://github.com/YesianRohn/UnionST)
**Area**: Others
**Keywords**: Scene Text Recognition, Synthetic Data, Data Engine, Self-Evolution Learning, Pseudo Labels

## TL;DR

This paper systematically analyzes the deficiencies of existing rendered synthetic data in corpus, font, and layout diversity, and proposes the UnionST synthetic engine together with a Self-Evolution Learning (SEL) framework. Using only synthetic data, the approach substantially outperforms conventional synthetic sets; combined with SEL, only 9% of real labeled data is required to approach fully supervised performance.

## Background & Motivation

**Background**: Scene text recognition (STR) relies on large-scale training data. Real-world annotation is expensive and class-imbalanced, making synthetic data a cost-effective alternative. However, a significant domain gap exists between existing synthetic data and real-world data.

**Limitations of Prior Work**: Rendering-based methods still outperform generative approaches — diffusion-model-based methods yield far lower text correctness (best editing accuracy of only 84.67%) and incur 10–10,000× higher computational cost. Existing synthetic sets are dominated by single semantically meaningful words, lacking non-semantic text (license plates, phone numbers), incomplete text, and multi-word phrases. Mainstream engines use only 1.2K–3.6K fonts, insufficient to cover artistic fonts encountered in the wild. Layouts are overly uniform, with characters arranged horizontally at uniform size, failing to simulate curved, multi-directional, or multi-scale text.

**Key Challenge**: Even state-of-the-art models trained on the largest real datasets show substantial room for improvement, indicating that the data dimension of STR remains far from solved and higher-quality synthetic data is needed.

## Method

### Overall Architecture

UnionST is a rendering-based synthetic data engine. The core pipeline is: (a) sample text from an augmented corpus and select compatible fonts → (b) render each character independently and compute position, orientation, and size parameters → (c) apply elastic distortion, perspective transformation, and border effects → (d) select a background, determine text color via a color correspondence table, and overlay random text instances, shadows, and embossing effects → output synthetic images and labels.

### Key Designs

- **Corpus Augmentation**: In addition to common semantic words, three categories of challenging text are introduced — (1) *Contextless*: random characters simulating non-semantic text such as license plates; (2) *Incomplete*: characters randomly removed to simulate occlusion or cropping; (3) *Multi-Words*: phrases and multi-word text segments.
- **Font Expansion**: 113.8K public fonts are collected (vs. 1.4K in MJ), with automatic filtering of fonts that cannot distinguish between upper- and lower-case letters.
- **Challenging Layout Modeling**: Each character is independently modeled with position $p_i$, orientation $o_i$, and size $s_i$; curvature is controlled by a quadratic parameter $a \in [20, 200]$, and multi-directional text is introduced via a global rotation angle $\phi \sim \text{Uniform}[0, 2\pi)$.
- **DTAug Online Augmentation**: Downsampling and transmission distortion augmentations are applied during training to simulate small-resolution and blurry samples.
- **SVTRv2-AR Model**: The CTC decoder of SVTRv2 is replaced with an attention-based autoregressive (AR) decoder, addressing the limitation of the monotonic alignment assumption for curved and multi-directional text.

### Self-Evolution Learning Framework (SEL)

- **Pseudo Corpus Augmentation**: A model trained on UnionST-S generates pseudo labels for unlabeled real data; these pseudo-labeled texts serve as the corpus for synthesizing UnionST-P (5M), which is merged with UnionST-S to form UnionST-SP (10M).
- **Iterative Self-Refinement (ISR)**: In each round, high-confidence pseudo-labeled samples (confidence ≥ 0.9) are selected to fine-tune the model. After two rounds, only the remaining low-confidence samples (~9%) require human annotation, enabling performance within a negligible margin of the fully supervised upper bound.

### Loss & Training

Standard STR training losses are employed (CTC loss / AR cross-entropy loss). The primary contributions are on the data side rather than in loss function design.

## Key Experimental Results

### Main Results

| Training Data | Volume | Common AVG | U14M-Bench AVG |
|---|---|---|---|
| ST-2D (all 2D synthetic sets merged) | 36.0M | 94.90% | 73.36% |
| UnionST-S | 5.0M | **95.32%** | **83.00%** |
| U14M-Filter (real data) | 3.22M | 96.56% | 87.22% |
| UnionST-SP | 10.0M | 96.07% | 84.86% |
| UnionST-SP + Real | 10.0M + 3.22M | **97.84%** | **91.39%** |

- UnionST-S (5M) surpasses ST-2D (36M) on U14M-Bench by **9.64%**, and even exceeds real data on the Multi-Words subset.
- UnionST-SP + Real pushes the U14M-Bench accuracy to **91.39%**, the first time this benchmark has exceeded 90%.

### SEL Results

| Stage | U14M-Bench AVG |
|---|---|
| UnionST-SP (synthetic only) | 84.86% |
| + Round 1 pseudo labels | 89.12% |
| + Round 2 pseudo labels | 89.81% |
| + 290K human-annotated hard samples | 91.23% |
| Fully supervised upper bound | 91.39% |

Annotating only 9% (290K / 3.22M) of real data achieves within 0.16% of the fully supervised upper bound.

### Ablation Study

1. **Curved layout** yields the most significant gain on the Curve subset (19.83% → 46.70%).
2. **Multi-directional variation** simultaneously benefits the Curve, MLO, and Salient subsets.
3. **Corpus augmentation** primarily benefits Contextless (+11%) and Multi-Words (+18%), with a slight drop on Common benchmarks, suggesting that evaluation on Common alone is prone to overfitting.
4. **Font diversity** shows limited effect at small scale but causes a notable drop in the Artistic subset when fonts are reduced at the 5M scale.
5. **DTAug** provides clear improvement on the General subset.
6. **Irreplaceability of pseudo corpus**: Using MJ+ST or ST-2D as the ISR base data achieves only 73.30% and 80.51%, respectively, far below the 89.12% achieved with UnionST-SP.

## Highlights & Insights

- Systematically diagnoses three key bottlenecks in rendered synthetic data (corpus / font / layout) and provides targeted solutions for each.
- 5M UnionST-S data outperforms 36M conventional synthetic data, demonstrating that data quality matters far more than quantity.
- The SEL framework reduces human annotation requirements by 91%, offering high practical value.
- First approach to surpass 90% accuracy on the Union14M-Benchmark.

## Limitations & Future Work

- The study is limited to English; extension to multilingual scenarios (Chinese, Arabic, etc.) has not been validated.
- Document OCR and handwriting recognition are not addressed.
- The visual realism of rendered synthetic images remains inferior to generative methods, imposing an upper bound on visual diversity.
- The font filtering strategy based on case distinguishability may inadvertently exclude some useful artistic fonts.
- No further gain is observed after the third ISR iteration; the issue of pseudo-label error accumulation warrants further investigation.

## Related Work & Insights

- **Rendering-based synthesis**: MJ, ST, CurvedST, SynthAdd, SynthTIGER, UnrealText, SynthText3D — none adequately combine challenging factors.
- **Generative synthesis**: MOSTEL, AnyText, TextCtrl, TextSSR, Flux.1 Kontext — insufficient correctness and high computational cost.
- **Data-centric analysis**: TRBA, STR-Fewer-Labels, Union14M — reveal data bottlenecks in STR.
- **Self/semi-supervised methods**: CCD, CCDPlus, ViSu — leverage synthetic data for semi-supervised learning.

## Rating

- Novelty: ⭐⭐⭐⭐ — Combination of systematic diagnosis and engineering improvements; the SEL framework is innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive comparisons, detailed ablations, covering diverse scenarios and baselines.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, well-defined problem formulation, rich figures and tables.
- Value: ⭐⭐⭐⭐⭐ — Highly practical; both the synthetic engine and the SEL framework directly advance the STR community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] BenDFM: A taxonomy and synthetic CAD dataset for manufacturability assessment in sheet metal bending](bendfm_a_taxonomy_and_synthetic_cad_dataset_for_ma.md)
- [\[NeurIPS 2025\] Private Evolution Converges](../../NeurIPS2025/others/private_evolution_converges.md)
- [\[CVPR 2026\] Next-Scale Autoregressive Models for Text-to-Motion Generation](next-scale_autoregressive_models_for_text-to-motion_generation.md)
- [\[CVPR 2026\] SimRecon: SimReady Compositional Scene Reconstruction from Real Videos](simrecon_simready_compositional_scene_reconstruction_from_real_videos.md)
- [\[AAAI 2026\] Finding Diverse Solutions Parameterized by Cliquewidth](../../AAAI2026/others/finding_diverse_solutions_parameterized_by_cliquewidth.md)

</div>

<!-- RELATED:END -->
