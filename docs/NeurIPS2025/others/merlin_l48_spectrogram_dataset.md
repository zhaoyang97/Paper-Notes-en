---
title: >-
  [Paper Note] Merlin L48 Spectrogram Dataset
description: >-
  [NeurIPS 2025][Single Positive Multi-Label Learning (SPML)] This paper introduces the L48 dataset — a fine-grained spectrogram multi-label classification benchmark derived from real-world bird recordings that naturally e…
tags:
  - "NeurIPS 2025"
  - "Single Positive Multi-Label Learning (SPML)"
  - "Bird Sounds"
  - "Spectrogram"
  - "Fine-Grained Classification"
  - "Ecological Priors"
date: 2026-05-08
content_hash: 533c3c17e5971522
---

# Merlin L48 Spectrogram Dataset

**Conference**: NeurIPS 2025
**arXiv**: [2511.00252](https://arxiv.org/abs/2511.00252)  
**Code**: [https://github.com/cvl-umass/l48-benchmarking](https://github.com/cvl-umass/l48-benchmarking)  
**Area**: Multi-Label Learning / Bird Sound Recognition / Dataset
**Keywords**: Single Positive Multi-Label Learning (SPML), Bird Sounds, Spectrogram, Fine-Grained Classification, Ecological Priors

## TL;DR

This paper introduces the L48 dataset — a fine-grained spectrogram multi-label classification benchmark derived from real-world bird recordings that naturally exhibits the Single Positive Multi-Label (SPML) setting. The dataset exposes critical shortcomings of existing SPML methods under realistic conditions, and proposes an intra-recording consistency regularization scheme to improve performance.

## Background & Motivation

Single Positive Multi-Label Learning (SPML) is an important yet underexplored real-world problem: each image is annotated with only one positive class label, while the presence or absence of all other classes remains unknown. Existing SPML research is primarily evaluated on synthetic datasets such as COCO and VOC, where the SPML setting is simulated by randomly dropping labels from fully annotated data.

This synthetic strategy, however, has two critical flaws:

**Distribution mismatch**: Synthetic SPML preserves the class distribution of the original dataset, whereas in real-world scenarios, the class distributions of training and test sets are often inconsistent.

**Lack of fine-grained challenges**: Object categories in COCO are visually distinct and easy to differentiate; however, real SPML scenarios (e.g., biodiversity monitoring) involve numerous easily confused fine-grained categories.

The authors observed that in the annotation workflow of the Merlin Bird ID app, asking experts to annotate all species in a recording is highly inefficient, whereas annotating only one target species is not. This practical workflow naturally produces SPML data, motivating the construction of the L48 dataset.

## Method

### Overall Architecture

The L48 dataset is sourced from the Merlin Sound ID system covering recordings from the contiguous 48 U.S. states (Lower 48). The overall design encompasses three components: dataset construction, benchmarking of SPML methods, and a regularization scheme tailored to the dataset's structure.

### Key Designs

1. **Dataset Construction**: 100 bird species are selected from the Merlin Sound ID data, with 100 recordings (assets) per species. Experts perform dense annotation on spectrograms by drawing bounding boxes. Recordings are segmented into 3-second clips, yielding spectrogram images. The training set comprises 82,081 images; the test set is fully annotated. The dataset spans all four seasons and diverse habitats across the country.

2. **Three Data Modes**:

    - **Target-only**: Only the target species label is retained; all others are unknown (strictest SPML setting).
    - **Geo Prior**: Species geographic range information is used to designate species outside the recording location as negative samples (averaging 42 negative labels).
    - **Checklist Prior**: eBird observation checklists are used to designate species absent from the checklist as negative samples (averaging 79 negative labels).

3. **Asset Regularization**: This approach exploits the structural property of L48 — multiple clips from the same recording should yield consistent species predictions. The regularization term is:
   $\mathcal{R}_P(\mathbf{x}_j^i) = \mathcal{L}_{BCE}(f_\theta(\mathbf{x}_j^i), \bar{y}_t^i)$
   where $\bar{y}_t^i$ is a running average of predictions across all clips from the same recording. Averaging over multiple "views" effectively disentangles misclassification from genuine background species occurrences.

### Loss & Training

- Base architecture: ImageNet-pretrained ResNet50
- Image preprocessing: resized to $448\times448$, normalized to ImageNet statistics
- Total loss: $\mathcal{L}_{SPML} + \alpha \mathcal{R}_P$, where $\alpha$ is a hyperparameter
- Training for 10 epochs on NVIDIA GTX 1080 Ti / 2080 Ti / Titan X
- Geo and Checklist priors are also simulated on COCO as control conditions

## Key Experimental Results

### Main Results

| Method | COCO (mAP%) | L48 (mAP%) | L48+Geo | L48+CL | L48+$\mathcal{R}_P$ |
|--------|-------------|------------|---------|--------|----------------------|
| BCE-Full | 76.4 | 62.4 | — | — | 66.4 |
| BCE-AN | 64.4 | 52.2 | — | — | 56.1 |
| LS | 67.3 | **56.4** | 57.1 | 58.4 | 56.4 |
| EM | 71.1 | 55.3 | 56.3 | 57.2 | 55.2 |
| LL-R | 71.4 | 50.1 | 51.3 | 52.6 | 55.0 |
| LL-Ct | 70.5 | 48.0 | 48.1 | 52.4 | 54.1 |
| LL-Cp | 69.8 | 43.8 | 45.8 | 50.6 | 44.4 |
| SPML Avg. | 68.4 | 51.5 | 52.2 | 54.0 | 53.9 |

### Ablation Study

| Comparison | COCO mAP | L48 mAP | Notes |
|------------|----------|---------|-------|
| BCE-Full vs BCE-AN | 76.4 vs 64.4 | 62.4 vs 52.2 | The SPML performance gap is larger on L48 |
| LL variants on COCO | 69.8–71.4 | — | Strong performance on COCO |
| LL variants on L48 | — | 43.8–50.1 | Fall below the BCE-AN baseline on L48 |
| Target-only vs CL Prior | +1.9 (COCO) | +2.5 (L48) | Additional negative labels consistently improve performance |
| Without vs with regularization | — | 51.5 vs 53.9 | Asset regularization yields an average gain of 2.4 points |

### Key Findings

1. **L48 is substantially harder than COCO**: The fully supervised baseline (BCE-Full) achieves 14 percentage points lower mAP on L48 than on COCO, attributable to high confusion rates among fine-grained species pairs.
2. **SPML methods break down on L48**: The LL-family methods, which perform best on COCO, fall below the simple BCE-AN baseline on L48.
3. **Misclassification and false negatives are indistinguishable**: In fine-grained settings, high-confidence predictions may reflect model confusion rather than genuine false negatives; LL methods erroneously correct these as positive labels.
4. **Asset regularization is effective**: Enforcing consistency across clips within the same recording improves performance for nearly all methods.
5. **Weak priors carry high value**: The average SPML performance under the Checklist Prior (54.0) approaches that of Asset Regularization (53.9), suggesting that a small amount of targeted supervision is equivalent to comprehensive negative labeling.

## Highlights & Insights

- The paper introduces a **naturally occurring SPML benchmark** rather than a synthetic construction, exposing blind spots of existing methods under distribution mismatch and fine-grained recognition.
- The regularization scheme cleverly exploits the recording–clip hierarchy in the data, which is essentially a form of **temporal consistency constraint**.
- The three data mode design introduces the research direction of **domain priors** into the SPML field.
- The dataset supports both multi-label classification and object detection paradigms.

## Limitations & Future Work

- Coverage is limited to 100 bird species from the contiguous 48 U.S. states, with constrained geographic and species diversity.
- The rich bounding box annotations are not fully exploited for semi-supervised learning.
- Asset regularization relies on the recording–clip hierarchical structure of the data, limiting its generalizability.
- The adaptation of SPML methods when incorporating negative labels is relatively straightforward; more effective utilization strategies may exist.
- Only ResNet50 is evaluated; the impact of stronger backbone networks remains unverified.

## Related Work & Insights

- L48 complements large-scale weakly labeled bird datasets such as iNatSounds and BirdSet by providing a densely annotated medium-scale benchmark.
- The idea behind Asset Regularization can be extended to scenarios with multi-view or temporal relationships, such as video action recognition and large-scale satellite imagery.
- The three data mode design can inspire active learning research — specifically, how to most effectively leverage domain expert annotation capacity.

## Rating
- Novelty: ⭐⭐⭐⭐ (primarily a dataset contribution; methodological novelty is moderate)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (multiple methods, multiple data modes, detailed analysis)
- Writing Quality: ⭐⭐⭐⭐⭐ (clear structure, in-depth and thorough analysis)
- Value: ⭐⭐⭐⭐ (provides a valuable real-world benchmark for the SPML field)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Bispectral OT: Dataset Comparison using Symmetry-Aware Optimal Transport](bispectral_ot_dataset_comparison_using_symmetry-aware_optimal_transport.md)
- [\[NeurIPS 2025\] Sign-In to the Lottery: Reparameterized Sparse Training from Scratch](sign-in_to_the_lottery_reparameterizing_sparse_training_from_scratch.md)
- [\[NeurIPS 2025\] MaxSup: Overcoming Representation Collapse in Label Smoothing](maxsup_overcoming_representation_collapse_in_label_smoothing.md)
- [\[NeurIPS 2025\] Continuous Thought Machines](continuous_thought_machines.md)
- [\[NeurIPS 2025\] Radar: Benchmarking Language Models on Imperfect Tabular Data](radar_benchmarking_language_models_on_imperfect_tabular_data.md)

</div>

<!-- RELATED:END -->
