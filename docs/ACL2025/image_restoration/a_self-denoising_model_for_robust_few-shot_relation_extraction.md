---
title: >-
  [Paper Note] A Self-Denoising Model for Robust Few-Shot Relation Extraction
description: >-
  [ACL 2025][Image Restoration][Few-Shot Relation Extraction] This paper proposes a Self-Denoising Model (SDM) to address the issue of support set label noise in few-shot relation extraction. Through the co-training of a label correction module and a relation classification module, SDM automatically corrects noisy labels and achieves more robust relation prediction, significantly outperforming baselines even in noise-free scenarios.
tags:
  - "ACL 2025"
  - "Image Restoration"
  - "Few-Shot Relation Extraction"
  - "Noisy Labels"
  - "Prototypical Networks"
  - "Label Correction"
  - "Self-Denoising"
date: 2026-05-08
content_hash: 8c437db42cef4e23
---

# A Self-Denoising Model for Robust Few-Shot Relation Extraction

**Conference**: ACL 2025  
**Code**: None  
**Area**: Image Restoration  
**Keywords**: Few-Shot Relation Extraction, Noisy Labels, Prototypical Networks, Label Correction, Self-Denoising

## TL;DR
This paper proposes a Self-Denoising Model (SDM) to address the issue of support set label noise in few-shot relation extraction. Through the co-training of a label correction module and a relation classification module, SDM automatically corrects noisy labels and achieves more robust relation prediction, significantly outperforming baselines even in noise-free scenarios.

## Background & Motivation

**Background**: Few-Shot Relation Extraction (FSRE) aims to enable models to recognize new relation types using an extremely small number of annotated samples. Prevailing methods are based on Prototypical Networks (ProtoNet), which perform classification by calculating the distance between query instances and the prototype of each class. Each relation class in the support set typically contains only $K$ annotated samples ($K$-shot), and the model needs to quickly adapt to new relations from these samples.

**Limitations of Prior Work**: Existing FSRE studies almost universally assume that the labels in the support set are perfectly accurate, but this assumption does not hold in real-world applications. Relation annotation is highly professional and subjective; even carefully audited datasets inevitably contain erroneous labels. While standard training sets can dilute the impact of noisy labels through large volumes of data, in few-shot scenarios where each class has only a few samples, the impact of a single erroneous label is drastically amplified.

**Key Challenge**: The prototypes in Prototypical Networks are the average vectors of all samples in the support set. A single noisy sample (which belongs to another relation but is mislabeled as the current relation) will severely deviate the prototype's direction, leading to a sharp decline in downstream classification performance. Simply discarding suspected noisy samples is infeasible because every sample is precious in few-shot scenarios—even mislabeled samples may still contain useful relation information in their textual content itself.

**Goal**: (1) Systematically reveal the sensitivity of Prototypical Networks to noisy labels in the support set; (2) design a model that can automatically correct noisy labels rather than discarding them; (3) make full use of corrected samples by rendering their contributions to the correct prototypes.

**Key Insight**: The authors first conducted preliminary experiments by randomly injecting different proportions of noisy labels into the support set, finding that even 10% noise could cause prototypical networks' performance to plummet. However, further analysis revealed that if the erroneous labels could be corrected and the samples reassigned to their correct categories, performance was not only restored but also improved—which is equivalent to increasing the effective sample size of the support set.

**Core Idea**: Rather than detecting and discarding noisy samples, it is better to design a self-denoising mechanism to "correct" their labels, turning erroneous samples into correct, additional training data to turn a disadvantage into an advantage.

## Method

### Overall Architecture
The Self-Denoising Model (SDM) consists of two core modules: the Label Correction Module (LCM) and the Relation Classification Module (RCM). Given the support set and query set as inputs, the LCM first predicts and corrects potential noisy labels in the support set to generate a corrected support set. Then, the RCM constructs relation prototypes based on the corrected support set and classifies the query instances. The two modules are optimized collaboratively in an end-to-end manner using a feedback training strategy.

### Key Designs

1. **Label Correction Module (LCM)**:

    - **Function**: Automatically detect and correct noisy labels in the support set.
    - **Mechanism**: After mapping all samples from the support set into the embedding space, the distance between each sample and each relation class center is calculated. If a sample is far from the center of its annotated class but very close to the center of another class, it is considered potentially mislabeled. The LCM outputs a soft label correction probability distribution: for each sample, it predicts the probability of truly belonging to each relation class. When the predicted label is inconsistent with the original label, it is replaced by the predicted label. In embedding computation, LCM uses an independent encoder to prevent interference with the representation space of RCM.
    - **Design Motivation**: Distance-based correction leverages the homophily principle—instances of the same relation should be close to each other in the embedding space. This is more flexible than directly setting a threshold to discard suspicious samples, as it simultaneously determines which correct class the sample should belong to.

2. **Relation Classification Module (RCM)**:

    - **Function**: Perform robust relation classification based on the corrected support set.
    - **Mechanism**: The RCM receives the corrected support set from the LCM and constructs prototype vectors for each relation class using the corrected labels. Unlike the standard Prototypical Network, the RCM assigns a weight to each sample based on the correction confidence of the LCM—samples with higher correction confidence contribute more to the prototype. For query instances, the RCM calculates the distance between their embeddings and each prototype, taking the relation corresponding to the nearest prototype as the prediction.
    - **Design Motivation**: Even though the LCM performs label correction, some corrections may still be inaccurate. Through confidence-based weighting, the interference of uncertain corrected samples on the prototypes can be further mitigated.

3. **Feedback Training Strategy**:

    - **Function**: Enable LCM and RCM to co-evolve.
    - **Mechanism**: Training is conducted alternately in two phases. Phase 1: Freeze the RCM and train the LCM using the RCM's classification loss as a feedback signal—if the LCM's correction improves the performance of the RCM, the current correction strategy is reinforced. Phase 2: Freeze the LCM and train the RCM using the corrected support set output by the LCM. The two phases alternate periodically, allowing the LCM to better adapt to the needs of the RCM, while the RCM gradually learns to utilize the corrected samples.
    - **Design Motivation**: Direct joint training may cause gradients of the two modules to interfere with each other. Alternate training allows each module to optimize in a stable environment, resembling the alternative training strategy of GANs.

### Loss & Training
Loss of LCM: Cross-entropy loss, which compares the difference between the label distribution predicted by LCM and the true clean labels (known noise is constructed during training via a meta-learning strategy). Loss of RCM: Standard prototypical network loss, i.e., the negative log-likelihood of the distance between the query instance embedding and the correct prototype. The total loss is combined via alternating optimization. During the construction of training episodes, a certain proportion of noisy labels is randomly injected into the support set, forcing the model to learn to handle noise.

## Key Experimental Results

### Main Results

| Dataset | Setting | SDM | Proto-BERT | HCRP | TD-Proto | Gain |
|--------|------|-----|-----------|------|---------|------|
| FewRel 1.0 | 5-way 1-shot (No noise) | Significantly outperforms | Baseline | Baseline | Baseline | +2-4% |
| FewRel 1.0 | 5-way 5-shot (No noise) | Significantly outperforms | Baseline | Baseline | Baseline | +1-3% |
| FewRel 1.0 | 5-way 5-shot (20% noise) | Far outperforms rivals | Plummeted | Plummeted | Plummeted | +8-15% |
| FewRel 2.0 | 5-way 1-shot (No noise) | Significantly outperforms | Baseline | Baseline | Baseline | +2-5% |
| FewRel 2.0 | 5-way 5-shot (30% noise) | Far outperforms rivals | Plummeted | Plummeted | Plummeted | +10-18% |

### Ablation Study

| Configuration | 5-way 5-shot Acc (20% noise) | Description |
|------|---------------------------|------|
| Full SDM | Highest | Full self-denoising model |
| w/o LCM (Directly discarding suspected noise) | Significant drop | Decreased support set samples after discarding, leading to degraded prototype quality |
| w/o Feedback Training | Drop of 3-5% | Joint training replaces alternate training, leading to mutual gradient interference |
| w/o Confidence Weighting | Drop of 1-2% | Corrected samples treated with equal weight |
| LCM only (No RCM feedback) | Drop of 4-6% | LCM lacks task-oriented optimization signals |
| Proto-BERT Baseline | Much lower (dropped 10%+) | Without any denoising mechanism |

### Key Findings
- Most important finding: SDM significantly outperforms baselines even on completely noise-free clean data. This indicates that LCM is not merely a denoising tool; it also enhances prototype quality by re-evaluating the relationships between samples.
- The higher the noise ratio, the greater the advantage of SDM over baselines. At 30% noise, baseline performance is close to random, whereas SDM still maintains high accuracy.
- The feedback training strategy is critical to performance; its removal leads to a noticeable degradation in the correction capability of LCM.
- The performance on the cross-domain FewRel 2.0 is equally outstanding, demonstrating that the denoising ability of SDM possesses strong transferability.

## Highlights & Insights
- The "turning a disadvantage into an advantage" design concept is highly ingenious: instead of detecting and discarding noisy samples, it corrects the labels to let erroneous samples turn into useful data. In few-shot scenarios, this approach is far more rational than discarding strategies.
- The feedback training strategy tackles the classic challenge of multi-module collaborative optimization. The alternating training of LCM and RCM forms a virtuous cycle—the classification signals of RCM guide LCM to perform better correction, and the corrected data in turn enhances the performance of RCM.
- The denoising concept can be transferred to other few-shot learning scenarios: few-shot NER, few-shot text classification, etc., all face similar noisy label problems. The dual-module framework of LCM+RCM is highly versatile.

## Limitations & Future Work
- The experiments were validated only on English relation extraction datasets; the relation types and annotation characteristics of Chinese or other languages may differ.
- LCM relies on distances in the embedding space to determine whether labels are correct. If the embedding quality is poor, or the relation classes themselves are hard to separate in the embedding space, LCM's correction is prone to errors.
- The injection method of noisy labels is random flipping to other relations, but practical annotation errors may exhibit specific patterns (e.g., easily confused relation pairs); the model's robustness to structured noise remains to be verified.
- In terms of training cost, alternating training requires more epochs to converge, which could become a bottleneck in ultra-large-scale meta-learning training.

## Related Work & Insights
- **vs Proto-BERT (Gao et al., 2019)**: Standard Prototypical Networks assume that the support set is perfectly clean, with performance plummeting once noise is introduced. SDM is a noise-robust enhancement of Proto-BERT.
- **vs Noisy Label Learning Methods (DivideMix, C2D, etc.)**: These methods target a large amount of noisy data in conventional supervised learning, with the strategy of isolating clean and noisy samples and training them differentially. However, in few-shot scenarios, samples are too scarce for effective isolation, making SDM's correction strategy more suitable for few-shot learning.
- **vs TD-Proto**: TD-Proto improves prototype representation through triplet distance but does not handle noisy labels. SDM addresses a more fundamental issue—the correctness of the labels in the support set itself.

## Rating
- Novelty: ⭐⭐⭐⭐ First to systematically study the noisy label problem in FSRE, with an innovatively designed self-denoising framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparison across multiple noise ratios and clean scenarios, with detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The approach of using preliminary experiments to motivate the work is highly persuasive.
- Value: ⭐⭐⭐⭐ Addresses an overlooked yet important practical problem in few-shot learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Rotation-Equivariant Self-Supervised Method in Image Denoising](../../CVPR2025/image_restoration/rotation-equivariant_self-supervised_method_in_image_denoising.md)
- [\[NeurIPS 2025\] The Effect of Optimal Self-Distillation in Noisy Gaussian Mixture Model](../../NeurIPS2025/image_restoration/the_effect_of_optimal_self-distillation_in_noisy_gaussian_mixture_model.md)
- [\[CVPR 2025\] Classic Video Denoising in a Machine Learning World: Robust, Fast, and Controllable](../../CVPR2025/image_restoration/classic_video_denoising_in_a_machine_learning_world_robust_fast_and_controllable.md)
- [\[ICCV 2025\] Blind2Sound: Self-Supervised Image Denoising without Residual Noise](../../ICCV2025/image_restoration/blind2sound_self-supervised_image_denoising_without_residual_noise.md)
- [\[ICLR 2026\] Are Deep Speech Denoising Models Robust to Adversarial Noise?](../../ICLR2026/image_restoration/are_deep_speech_denoising_models_robust_to_adversarial_noise.md)

</div>

<!-- RELATED:END -->
