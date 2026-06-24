---
title: >-
  [Paper Note] Joint Out-of-Distribution Filtering and Data Discovery Active Learning
description: >-
  [CVPR 2025][AI Safety][active learning] Proposes the Open-Set Discovery Active Learning (OSDAL) scenario and designs the Joda algorithm. Through a three-phase workflow of training, filtering, and selection, Joda utilizes a single model to simultaneously filter OOD data and discover new categories without requiring auxiliary models, consistently achieving state-of-the-art accuracy across 18 configurations.
tags:
  - "CVPR 2025"
  - "AI Safety"
  - "active learning"
  - "OOD detection"
  - "category discovery"
  - "open-set"
  - "energy score"
  - "SISOMe"
date: 2026-05-08
content_hash: 61c5c1314a220b0c
---

# Joint Out-of-Distribution Filtering and Data Discovery Active Learning

**Conference**: CVPR 2025  
**arXiv**: [2503.02491](https://arxiv.org/abs/2503.02491)  
**Code**: [Project Page](https://www.cs.cit.tum.de/daml/joda/)  
**Area**: Others  
**Keywords**: active learning, OOD detection, category discovery, open-set, energy score, SISOMe

## TL;DR

Proposes the Open-Set Discovery Active Learning (OSDAL) scenario and designs the Joda algorithm. Through a three-phase workflow of training, filtering, and selection, Joda utilizes a single model to simultaneously filter OOD data and discover new categories without requiring auxiliary models, consistently achieving state-of-the-art accuracy across 18 configurations.

## Background & Motivation

**Background**: Active Learning (AL) reduces labeling costs by strategically selecting samples for annotation. In real-world scenarios, the unlabeled pool poses two major challenges: (1) OOD data contamination (previously studied in open-set AL), and (2) undiscovered new categories (independently studied in category discovery).

**Limitations of Prior Work**:
- Existing open-set AL methods (LfOSA, CCAL, MQNet, Pal) require 1-2 extra auxiliary models, and most require access to the unlabeled pool for training.
- These methods only handle OOD filtering and do not consider new category discovery.
- AGD considers category discovery but does not address OOD data.
- Both problems co-exist in real-world scenarios (such as autonomous driving), but no unified solution existed previously.

**Key Challenge**: New category data acts as near-OOD (semantically close to InD), whereas OOD data is far-OOD. Existing methods lack the granularity to make fine-grained distinctions among the three types of data (known InD, discoverable new categories, and irrelevant OOD).

**Goal**: Simultaneously achieve OOD data filtering and new category discovery in active learning without increasing model complexity.

## Method

### Overall Architecture

Joda consists of three phases:
1. **Training Phase (I)**: Jointly trains a single classification model using CrossEntropy + Outlier Exposure.
2. **Filtering Phase (II)**: Determines a threshold based on energy scores and ROC analysis on the labeled pool to filter out OOD samples.
3. **Selection Phase (III)**: Selects the most valuable samples using the SISOMe metric combined with a class-balancing factor.

### Key Designs

**1. Deeply Coupled Training Loss**
- **Function**: Separately handles InD samples and accidentally labeled OOD samples from the labeled pool: $\mathcal{L}(b) = \mathcal{L}_{CE}(b_{InD}) + \lambda_{OE} \cdot \mathcal{L}_{OE}(b_{OOD})$.
- **Core Idea**: The Outlier Exposure loss regularizes the model to output a uniform distribution for OOD samples, creating a clear separation in energy scores between InD and OOD data. This degenerates to standard CE when there is no OOD data in the initial labeled pool.
- **Design Motivation**: Instead of training an external, separate OOD detector, OE is directly integrated into the task model training, building a shared feature space to differentiate among the three data types.

**2. Energy-based OOD Filtering**
- **Function**: Computes $E(x) = -\log \sum_{i=1}^{c} \exp(f(x)_i)$ and performs ROC analysis on the labeled pool to find the optimal Youden's J threshold $t_{opt}$ for filtering.
- **Core Idea**: After OE training, the logits of OOD samples tend to be uniform (low energy), whereas InD and new category samples exhibit higher energy. As near-OOD, new categories are unaffected by OE regularization, positioning their energy distribution between InD and far-OOD.
- **Design Motivation**: Leverage the by-product of OE training to naturally achieve OOD filtering without requiring auxiliary models. Youden's J adaptively selects the threshold.

**3. SISOMe Selection + Class Balancing**
- **Function**: Combines energy score and the in-to-out feature space distance ratio using the SISOMe metric, augmented with a class-balancing factor $b_f(c) = -\sigma_{\hat{m}_l} (\frac{n(c) \cdot C}{|L|} - 1)$.
- **Core Idea**: The self-balancing mechanism of SISOMe trade-offs energy and diversity. New categories naturally receive higher selection scores (similar to near-OOD) since they are not regularized by OE. The class-balancing factor uses pseudo-labels to estimate class distributions, favoring under-sampled classes.
- **Design Motivation**: Closed-loop design—the OE loss in training, energy in filtering, and SISOMe in selection all revolve around the same energy/logit space, making the three phases deeply coupled.

### Loss & Training

- Loss: $\mathcal{L} = \mathcal{L}_{CE}(b_{InD}) + 0.5 \cdot \mathcal{L}_{OE}(b_{OOD})$, with $\lambda_{OE}=0.5$
- Model: ResNet-18, without auxiliary models
- In the first AL cycle, the labeled pool is clean, so the filtering step is bypassed
- Once the number of discovered new category samples reaches threshold $t_e$, they are integrated into the known classes for training

## Key Experimental Results

### Main Results

Evaluated on CIFAR-10, CIFAR-100, and TinyImageNet. OOD data sources include Random, MNIST, Places365, and ImageNetC-800.

| Method | Extra Models | CIFAR-100 Acc↑ | TinyImageNet Acc↑ | Selection Precision↑ |
|---|---|---|---|---|
| LfOSA | 1 | ~42% | ~32% | ~0.95 |
| CCAL | 2 | ~40% | ~30% | ~0.85 |
| MQNet | 2(+1) | ~38% | ~28% | ~0.80 |
| Pal | 2 | ~41% | ~31% | ~0.90 |
| Badge | 0 | ~39% | OOM | ~0.75 |
| **Joda** | **0** | **~46%** | **~35%** | **~1.0** |

### Ablation Study

| Setting | Accuracy | Category Discovery | Selection Precision |
|---|---|---|---|
| Full Joda | Highest | Fastest | ~1.0 |
| w/o OE (no Outlier Exposure) | Signif. Drop | Drop | Signif. Drop |
| w/o Filtering (no energy filtering) | Signif. Drop | Drop | Signif. Drop |
| Energy Exposure replacing OE | Drop | Drop | Drop |
| Different $\lambda$ (0.1-1.0) | Robust | Robust | Robust |

### Key Findings

1. **Deep Coupling of Three Phases is Key**: Removing any component (OE, filtering, SISOMe) leads to a significant performance drop.
2. **Near-Perfect Selection Precision**: Joda achieves selection precision close to 1.0 in 8 out of 10 configurations (almost never selecting OOD data).
3. **Fastest Category Discovery**: Joda discovers new categories the fastest across all configurations.
4. **Zero Extra Models**: Compared to competing methods requiring 1-2 auxiliary models, Joda utilizes only a single classification model.
5. **Robust Hyperparameters**: Performance varies minimally across $\lambda_{OE}$ values in the 0.1-1.0 range.

## Highlights & Insights

- Valuable Definition of the OSDAL Scenario: Formulates the joint problem of AL, OOD filtering, and category discovery for the first time.
- "Subtractive" Design Philosophy: While existing methods use 2+ extra models, Joda operates with 0 extra models yet delivers superior performance.
- Closed-Loop Design with Deeply Coupled Phases: Training, filtering, and selection share the exact same feature space and energy metrics.
- Clever Exploitation of OE Loss: It not only conducts OOD detection but also inherently distinguishes between near-OOD (new categories) and far-OOD.

## Limitations & Future Work

- Evaluated only on image classification tasks, without expansion to more complex tasks such as object detection or semantic segmentation.
- The threshold $t_e$ for discovering new categories needs to be predefined.
- Filtering precision decreases slightly when OOD data semantically overlaps with InD data (e.g., Places365 vs. TinyImageNet).
- No comparison with newer foundation-model-based OOD detection methods.
- Future work could explore extending Joda to stream-based AL scenarios.

## Related Work & Insights

- **SISOMe (Schmidt et al.)**: Core selection metric for Joda, combining energy and feature-space distances.
- **Outlier Exposure (Hendrycks & Gimpel)**: A classic OOD detection method that Joda cleverly integrates into the AL training loop.
- **LfOSA / CCAL / MQNet**: Open-set AL baselines, all relying on auxiliary models.
- **AGD (Ma et al.)**: AL focusing solely on new category discovery without handling OOD filtering.

## Rating

⭐⭐⭐⭐ — The problem definition has practical value. The method is simple and elegant (0 extra models, 0 extra data), and the experiments are comprehensive (18 configurations, 3 metrics). It has direct application value in practical deployment scenarios such as autonomous driving and robotic perception.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Detecting Out-of-Distribution through the Lens of Neural Collapse](detecting_out-of-distribution_through_the_lens_of_neural_collapse.md)
- [\[CVPR 2025\] A Simple Data Augmentation for Feature Distribution Skewed Federated Learning](a_simple_data_augmentation_for_feature_distribution_skewed_federated_learning.md)
- [\[CVPR 2025\] H2ST: Hierarchical Two-Sample Tests for Continual Out-of-Distribution Detection](h2st_hierarchical_two-sample_tests_for_continual_out-of-distribution_detection.md)
- [\[CVPR 2025\] Leveraging Perturbation Robustness to Enhance Out-of-Distribution Detection](leveraging_perturbation_robustness_to_enhance_out-of-distribution_detection.md)
- [\[CVPR 2025\] OODD: Test-time Out-of-Distribution Detection with Dynamic Dictionary](oodd_test-time_out-of-distribution_detection_with_dynamic_dictionary.md)

</div>

<!-- RELATED:END -->
