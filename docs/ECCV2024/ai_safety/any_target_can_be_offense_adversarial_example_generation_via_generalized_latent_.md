---
title: >-
  [Paper Note] Any Target Can Be Offense: Adversarial Example Generation via Generalized Latent Infection
description: >-
  [ECCV 2024][AI Safety][adversarial attack] GAKer is proposed as the first targeted adversarial attack generator that generalizes to unseen target classes. By injecting target features (latent infection) into the intermediate layers of a UNet and employing a class-agnostic cosine distance loss instead of cross-entropy, it outperforms HGN on unseen classes by 14.13% in attack success rate.
tags:
  - "ECCV 2024"
  - "AI Safety"
  - "adversarial attack"
  - "Targeted Attack"
  - "Generalization"
  - "Unknown Classes"
  - "Feature Infection"
date: 2026-05-08
content_hash: e97af7d1934151f1
---

# Any Target Can Be Offense: Adversarial Example Generation via Generalized Latent Infection

**Conference**: ECCV 2024  
**arXiv**: [2407.12292](https://arxiv.org/abs/2407.12292)  
**Code**: [https://github.com/VL-Group/GAKer](https://github.com/VL-Group/GAKer)  
**Area**: AI Safety / Adversarial Attacks  
**Keywords**: adversarial attack, Targeted Attack, Generalization, Unknown Classes, Feature Infection

## TL;DR
GAKer is proposed as the first targeted adversarial attack generator that generalizes to unseen target classes. By injecting target features (latent infection) into the intermediate layers of a UNet and employing a class-agnostic cosine distance loss instead of cross-entropy, it outperforms HGN on unseen classes by 14.13% in attack success rate.

## Background & Motivation
**Background**: Targeted adversarial attacks aim to mislead a classifier into classifying inputs as a specific target class. Existing generator-based methods (e.g., MAN, HGN, ESMA) efficiently generate adversarial examples but are restricted to attacking classes seen during training.

**Limitations of Prior Work**:
   - Generator-based methods memorize features of specific classes rather than learning transferable attack principles.
   - Attack success rates drop as the number of training classes increases.
   - Retraining is required when facing newly emerging classes, limiting practical deployment.

**Key Challenge**: Designing a "train-once, attack-any-class" generator, whereas existing methods couple the attack strategy with a fixed class label space.

**Goal**: How to train a class-agnostic adversarial attack generator that can generalize to entirely unseen target classes?

**Key Insight**: Shifting the target from "class labels" to "visual features"—guiding the attack using the features of target images instead of class IDs, and replacing cross-entropy loss with cosine distance.

**Core Idea**: During training, minimize the cosine distance between the features of the adversarial example and the target image (class-agnostic), and inject target features into the intermediate layers of the UNet generator via a Feature Transform Module.

## Method

### Overall Architecture
GAKer consists of a frozen feature extractor $\mathcal{F}_\psi$ and a UNet generator $\mathcal{G}_\theta$. Given source and target images as input, the generator receives target feature injections in each ResBlock and outputs a perturbation $\delta$ ($\ell_\infty \leq 16$).

### Key Designs

1. **Latent Infection Mechanism**:

    - **Function**: Injects target image features into the intermediate representations of the UNet generator.
    - **Mechanism**: Enhances target features with a Feature Transform Module (FTM, Linear-GELU-Linear) $\rightarrow$ aligns dimensions with a Dimension Matching Module (DMM, Linear-GELU) $\rightarrow$ injects into each ResBlock.
    - **Design Motivation**: Feature fusion in intermediate layers guides the attack direction better than simple input concatenation, allowing the generator to learn "how to move toward the target feature space" instead of "how to generate a specific class."

2. **Class-Agnostic Loss Function**:

    - **Function**: Replaces cross-entropy loss with cosine distance.
    - **Mechanism**: $\mathcal{L} = D_{cos}(\mathcal{F}(\mathcal{G}(x_s, x_t)), \mathcal{F}(x_t)) + 0.5 \cdot D_{cos}(\mathcal{F}(\delta), \mathcal{F}(x_t))$
    - **Design Motivation**: Cross-entropy requires class labels (limiting it to known classes), whereas cosine distance only compares feature directions (class-agnostic). The second term encourages the perturbation itself to carry target features.

3. **Known Class Selection Strategy**:

    - **Function**: Intelligently selects 200 known classes for training.
    - **Mechanism**: Uses a greedy algorithm to select classes with the largest discrepancies in the feature space, ensuring coverage of the feature space.
    - **Design Motivation**: Class selection significantly impacts attack generalization to unseen classes—greedy selection outperforms random selection by 16.52%.

### Loss & Training
- Trained only on 200 ImageNet classes (325 high-quality images per class).
- Perturbation budget $\ell_\infty \leq 16/255$.
- $\alpha = 0.5$ to balance the two cosine distance terms.

## Key Experimental Results

### Unseen Class Attack Success Rate (Res-50 Surrogate Model)

| Method | Res-50* | Res-152 | VGG-19 | Dense-121 | Inc-v3 | ViT | Avg |
|------|---------|---------|--------|-----------|--------|-----|-----|
| HGN | 0.05 | 0.14 | 0.15 | 0.06 | 0.04 | 0.05 | 0.08 |
| **GAKer** | **41.69** | **23.05** | **26.02** | **23.80** | **5.85** | **1.44** | **16.40** |

### Seen Class Attack Success Rate

| Method | Res-50* | Res-152 | VGG-19 | Dense-121 | Avg |
|------|---------|---------|--------|-----------|-----|
| ESMA | 95.60 | 83.22 | 81.98 | 82.54 | 61.37 |
| **GAKer** | **96.61** | **83.36** | **82.20** | **81.95** | **58.71** |

### Attacking Large Models (VLM, Unseen Classes)

| Method | Qwen-VL | LLaVA |
|------|---------|-------|
| HGN | 12.50% | 13.85% |
| **GAKer** | **52.60%** | **56.45%** |

### Ablation Study

| Ablation Dimension | Optimal Setting | Key Findings |
|---------|---------|---------|
| Number of Training Classes N | 200 | N=200 performs similarly to N=500, but N=200 trains faster. |
| Class Selection | Greedy vs Random | Greedy selection improves performance by +16.52% over random selection. |
| Number of Samples M | 325/class | M=325 is optimal, M=650 leads to performance degradation. |
| α | 0.5 | Balanced optimal. |

### Key Findings
- **Core Contribution to Unseen Class Generalization**: HGN almost entirely fails on unseen classes (0.08%), whereas GAKer achieves 16.40%—a hundred-fold increase.
- **No Performance Drop on Seen Classes**: GAKer remains comparable to ESMA on seen classes (58.71% vs 61.37%), indicating generalization does not come at the cost of performance on seen classes.
- **Effective Against VLMs**: Achieves >50% attack success rate on unseen classes for Qwen-VL/LLaVA, demonstrating threats to multi-modal models as well.
- **Crucial Class Selection**: The gap between greedy selection (covering the feature space) and random selection is 16.52%.

## Highlights & Insights
- **Paradigm Shift from "Label Space" to "Feature Space"**: Shifting the target of directed attacks from discrete class labels to continuous feature spaces is the key to achieving generalization. This concept is also inspiring for other generative tasks requiring generalization.
- **Latent Infection Mechanism**: Injecting target features into each ResBlock of UNet, similar to conditional injections in ControlNet but tailored for adversarial attacks—simple and effective.
- **Attacking Any Class with Only 200 Training Classes**: Significantly reduces the deployment cost of adversarial attacks and the difficulty of threat assessments.

## Limitations & Future Work
- **Large Gap Between Seen and Unseen Classes**: Unseen classes (16.40%) vs seen classes (58.71%), a gap of about 42%—indicating that feature space generalization is still imperfect.
- **Limited Cross-Architecture Transferability**: The attack success rate on ViT is only 1.44% (transferred from a CNN surrogate), indicating that CNN-to-Transformer transferability remains a challenge.
- **High Variance Among Unseen Classes**: The attack difficulty varies greatly across different unseen classes, lacking analysis on "which classes are easy or hard to attack."
- **Ethical Concerns**: Although the paper discusses its use for security assessments, the double-edged nature of the tool warrants attention.

## Related Work & Insights
- **vs HGN**: HGN uses class labels for supervision, failing completely to generalize to unseen classes. GAKer uses feature distance supervision to enable generalization.
- **vs ESMA**: ESMA is stronger on seen classes (61.37% vs 58.71%) but similarly fails to generalize to unseen classes.
- **vs TTP**: TTP requires retraining for every new class, which costs 100 times more computationally than GAKer.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The first targeted attack generator for unseen classes, with a novel feature infection concept.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluations across CNNs, ViTs, VLMs, and defense models, with detailed ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure and detailed ablation studies, though some notations are slightly redundant.
- **Value**: ⭐⭐⭐⭐ Highly significant for AI security assessment, revealing widespread vulnerabilities in models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Your Classifier Can Do More: Towards Balancing the Gaps in Classification, Robustness, and Generation](../../CVPR2026/ai_safety/your_classifier_can_do_more_towards_balancing_the.md)
- [\[ICML 2026\] Exposing Vulnerabilities in Explanation for Time Series Classifiers via Dual-Target Adversarial Attack](../../ICML2026/ai_safety/exposing_vulnerabilities_in_explanation_for_time_series_classifiers_via_dual-tar.md)
- [\[AAAI 2026\] Rethinking Target Label Conditioning in Adversarial Attacks: A 2D Tensor-Guided Generative Approach](../../AAAI2026/ai_safety/rethinking_target_label_conditioning_in_adversarial_attacks_a_2d_tensor-guided_g.md)
- [\[ICML 2026\] One Model to Translate Them All: Universal Any-to-Any Translation for Heterogeneous Collaborative Perception](../../ICML2026/ai_safety/one_model_to_translate_them_all_universal_any-to-any_translation_for_heterogeneo.md)
- [\[ECCV 2024\] CLIP-Guided Generative Networks for Transferable Targeted Adversarial Attacks](clip-guided_generative_networks_for_transferable_targeted_adversarial_attacks.md)

</div>

<!-- RELATED:END -->
