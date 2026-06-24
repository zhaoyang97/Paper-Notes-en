---
title: >-
  [Paper Note] Taste More, Taste Better: Diverse Data and Strong Model Boost Semi-Supervised Crowd Counting
description: >-
  [CVPR 2025][Image Generation][Semi-supervised learning] The TMTB framework is proposed. By enhancing background diversity via diffusion model inpainting, introducing a VMamba backbone, and leveraging an anti-noise classification branch, it reduces the MAE on JHU-Crowd++ to 67.0 using only 5% of labeled data, substantially surpassing the state-of-the-art in semi-supervised crowd counting.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Semi-supervised learning"
  - "crowd counting"
  - "data augmentation"
  - "state space models"
  - "pseudo-labeling"
date: 2026-05-08
content_hash: 8777b8025e25278e
---

# Taste More, Taste Better: Diverse Data and Strong Model Boost Semi-Supervised Crowd Counting

**Conference**: CVPR 2025  
**arXiv**: [2503.17984](https://arxiv.org/abs/2503.17984)  
**Code**: [https://github.com/syhien/taste_more_taste_better](https://github.com/syhien/taste_more_taste_better)  
**Area**: Image Generation  
**Keywords**: Semi-supervised learning, crowd counting, data augmentation, state space models, pseudo-labeling

## TL;DR
The TMTB framework is proposed. By enhancing background diversity via diffusion model inpainting, introducing a VMamba backbone, and leveraging an anti-noise classification branch, it reduces the MAE on JHU-Crowd++ to 67.0 using only 5% of labeled data, substantially surpassing the state-of-the-art in semi-supervised crowd counting.

## Background & Motivation
1. **Background**: Crowd counting is a fundamental computer vision task widely applied in public safety, traffic management, and disaster prevention. Fully supervised methods rely heavily on point annotations, which incur extremely high annotation costs (e.g., approximately 2000 person-hours for UCF-QNRF).
2. **Limitations of Prior Work**: Semi-supervised crowd counting methods (based on the Mean Teacher framework + pseudo-labels) reduce annotation demands but face two major bottlenecks: (a) existing data augmentation methods (Mixup, CutMix) destroy the spatial structure of crowds, leading to density map distortion; (b) CNN backbones focus excessively on local details and lack global context modeling capabilities.
3. **Key Challenge**: Density map regression for crowd counting heavily relies on the integrity of spatial structures, which standard augmentation methods inherently disrupt. Meanwhile, density map labels themselves are noisy, and the regression head is highly sensitive to such noise.
4. **Goal**: Design a data augmentation method tailored for crowd counting, a backbone capable of modeling long-range dependencies, and an anti-noise training strategy.
5. **Key Insight**: Modify only background regions (inpainting) while keeping foreground crowds intact, and use a classification task to provide "imprecise but accurate" supervisory signals.
6. **Core Idea**: Enhance diversity by inpainting backgrounds using Stable Diffusion, replace CNNs with VMamba to obtain a global receptive field, and resist annotation noise with an anti-noise classification branch.

## Method

### Overall Architecture
TMTB is built upon the Mean Teacher semi-supervised framework. The student and teacher models share the same architecture (VMamba backbone + regression head + classification head), and the teacher model's weights are updated via EMA. Given an input image, features are extracted by VMamba and then fed into two branches: the density map regression branch and the count-interval classification branch. The Inpainter module periodically performs background inpainting on the training set to enhance diversity. Labeled data are directly supervised, while unlabeled data are learned through consistency regularization.

### Key Designs

1. **Inpainting Data Augmentation**

    - **Function**: Enhance the diversity of training data without damaging foreground crowds.
    - **Mechanism**: Utilize the foreground/background segmentation mask $M^{\text{inp}}$ predicted by the classification branch to invoke Stable Diffusion for inpainting *only* on background regions, generating diverse backgrounds using randomized positive text prompts. All inpainted images are treated as unlabeled data. To filter low-quality inpainted regions, the teacher model computes the inconsistency of classification predictions between weakly and strongly augmented versions to generate a weighted mask $M^{\text{incon}}$, allowing the model to learn from unreliable regions with lower weights. The inconsistency weight gradually decays as training progresses: $\omega_l^t = \text{softmax}(e^{-l \cdot t / T^{\text{inpw}}})$.
    - **Design Motivation**: Mixup and CutMix disrupt the spatial structure of density maps (by cutting or mixing crowds), whereas inpainting only modifies the background and preserves the foreground intact, naturally fitting density regression tasks.

2. **VMamba (Visual State Space Model) Backbone**

    - **Function**: Replace CNNs/Transformers as the feature extraction backbone to model global long-range dependencies.
    - **Mechanism**: Employ the 2D-Selective-Scan (SS2D) module of VMamba, where each pixel integrates global information from different directions through complementary 1D traversal paths. This achieves a global receptive field while maintaining a linear time complexity of $O(n)$. The extracted features are then fed into both the regression and classification branches.
    - **Design Motivation**: CNNs are prone to overfitting local details in extremely congested, low-light, or severe weather scenes, whereas Transformers offer strong global modeling but suffer from $O(n^2)$ computational complexity. VSSM combines the advantages of both.

3. **Anti-Noise Classification Branch**

    - **Function**: Provide "imprecise but accurate" auxiliary supervision to resist point annotation noise.
    - **Mechanism**: Quantize pixel-level density values into predefined count interval bins, and the classification head predicts which bin each location belongs to. Supervision is performed using the cross-entropy loss $\mathcal{L}_{\text{cls}} = \frac{1}{N}\sum_i^N \mathcal{H}(p_i^{gt}, \hat{p}_i)$. The output of the classification head is simultaneously utilized to generate the inpainting mask and perform filtration of unreliable regions.
    - **Design Motivation**: Density map annotations suffer from location shifts (annotators disagree on the exact center of heads), making the regression targets inherently noisy. Classification into "a specific interval" is more robust than precise regression.

### Loss & Training
The total loss is formulated as $\mathcal{L} = \mathcal{L}^s + \lambda_w \cdot \mathcal{L}^u + \lambda_w \cdot \mathcal{L}^{inp}$:
- **Supervised loss** $\mathcal{L}^s = \mathcal{L}_{reg}^s + \mathcal{L}_{cls}^s$: The regression loss utilizes multi-scale SSIM + TV loss (CUT loss), while the classification loss employs cross-entropy.
- **Consistency loss** $\mathcal{L}^u$: MAE(Student Density, Teacher Density) + MAE(Student Classification, Teacher Classification), accompanied by patch-aligned random masking strong augmentation.
- **Inpainting loss** $\mathcal{L}^{inp}$: Shares the same structure as the consistency loss, but is multiplied by the inconsistency weighted mask $M^{\text{incon}}$ to filter out low-quality regions.
- $\lambda_w$ linearly warms up from 0 to 1.0 during the first 20 epochs.

## Key Experimental Results

### Main Results

| Dataset | Label Ratio | Metric (MAE) | TMTB | MRC-Crowd (Prev. SOTA) | Gain |
|--------|---------|-----------|------|-------------------|------|
| JHU-Crowd++ | 5% | MAE↓ | **67.0** | 76.5 | -12.4% |
| UCF-QNRF | 5% | MAE↓ | **96.3** | 101.4 | -5.0% |
| ShanghaiTech A | 5% | MAE↓ | **72.4** | 74.8 | -3.2% |
| ShanghaiTech B | 5% | MAE↓ | **10.6** | 11.7 | -9.4% |
| JHU-Crowd++ | 10% | MAE↓ | **66.3** | 70.7 | -6.2% |
| UCF-QNRF | 10% | MAE↓ | **91.7** | 93.4 | -1.8% |
| JHU-Crowd++ | 40% | MAE↓ | **60.0** | 60.0 | Flat |
| ShanghaiTech B | 40% | MAE↓ | **7.5** | 7.8 | -3.8% |

### Ablation Study

| Configuration | JHU 5% MAE | Description |
|------|-----------|------|
| Full TMTB | 67.0 | Full model |
| w/o Inpainting Aug | ~73 | Significant degradation after removing inpainting augmentation |
| w/o VMamba (using CNN) | ~75 | CNN backbone lacks global receptive fields |
| w/o Anti-Noise cls | ~71 | Removing the classification branch makes regression sensitive to noise |
| Replacing Inpainting with Mixup | ~80+ | Mixup destroys the spatial structure of density maps |

### Key Findings
- Inpainting augmentation contributes the most in ultra-low label ratio settings (5%), where data diversity is more critical.
- On JHU-Crowd++ with 5% labels, the MAE drops below 70 for the first time (67.0), representing a landmark breakthrough.
- The inconsistency filtration mechanism $M^{\text{incon}}$ is vital for controlling inpainting quality; inpainting without filtration introduces noise.
- In cross-dataset generalization experiments, TMTB even outperforms some fully supervised methods.

## Highlights & Insights
- **Inpainting as an augmentation scheme for density map estimation** is highly clever: traditional augmentation methods inevitably disrupt label correspondences, whereas modifying only the background perfectly circumvents this issue. This approach can be transferred to any task where labels are tightly bound to spatial locations (e.g., keypoint detection, instance segmentation).
- **The classification branch simultaneously serves three purposes**: anti-noise supervision, inpainting mask generation, and unreliable region detection. Reusing a single module for multiple purposes demonstrates a highly efficient design.
- **The first application of VSSM in density estimation** validates the value of linear-complexity global modeling in dense prediction tasks.

## Limitations & Future Work
- The inpainting process incurs extra inference overhead from Stable Diffusion, significantly increasing training time.
- The effectiveness in extremely dense scenes (over thousands of people) has not been analyzed separately.
- The settings for count-interval classification bins are hyperparameters and may require tuning for different datasets.
- Future work could explore more efficient generative models to replace SD for inpainting, or replace heuristic rules with learnable augmentation strategies.

## Related Work & Insights
- **vs MRC-Crowd**: Both employ a classification auxiliary task, but MRC-Crowd relies on a CNN backbone and lacks innovation in data augmentation. This work advances both aspects simultaneously.
- **vs DACount**: DACount utilizes Transformers to refine foreground features, whereas this work uses VMamba to achieve global modeling more efficiently.
- **vs DiffusionMix**: DiffusionMix also leverages diffusion models for data augmentation, but it requires predefined binary masks, making it unsuitable for scenarios where foreground locations are unpredictable (as in crowd counting).

## Rating
- Novelty: ⭐⭐⭐⭐ The inpainting augmentation concept is a first in density estimation, though individual modules build upon previous works.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across four datasets and three label ratios, with detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with a naturally motivated derivation.
- Value: ⭐⭐⭐⭐ Landmark breakthrough in semi-supervised crowd counting, achieving MAE < 70 on JHU 5% for the first time.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Boost Your Human Image Generation Model via Direct Preference Optimization](boost_your_human_image_generation_model_via_direct_preference_optimization.md)
- [\[CVPR 2026\] StableMaterials: Enhancing Diversity in Material Generation via Semi-Supervised Learning](../../CVPR2026/image_generation/stablematerials_enhancing_diversity_in_material_generation_via_semi-supervised_l.md)
- [\[CVPR 2025\] Training Data Provenance Verification: Did Your Model Use Synthetic Data from My Generative Model for Training?](training_data_provenance_verification_did_your_model_use_synthetic_data_from_my_.md)
- [\[CVPR 2025\] InsightEdit: Towards Better Instruction Following for Image Editing](insightedit_towards_better_instruction_following_for_image_editing.md)
- [\[CVPR 2025\] ViUniT: Visual Unit Tests for More Robust Visual Programming](viunit_visual_unit_tests_for_more_robust_visual_programming.md)

</div>

<!-- RELATED:END -->
