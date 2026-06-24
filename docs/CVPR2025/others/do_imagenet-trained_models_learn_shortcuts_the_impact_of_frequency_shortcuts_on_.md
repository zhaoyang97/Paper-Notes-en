---
title: >-
  [Paper Note] Do ImageNet-trained Models Learn Shortcuts? The Impact of Frequency Shortcuts on Generalization
description: >-
  [CVPR 2025][Frequency shortcuts] This paper proposes a Hierarchical Frequency Shortcut Search (HFSS) method to efficiently discover frequency shortcuts learned by CNNs and Transformers at the ImageNet-1K scale for the first time (permitting correct classification with only 5% of frequencies). It reveals that frequency shortcuts are surprisingly beneficial in texture-preserving OOD tests but detrimental in stylized tests (IN-R/IN-S), pointing out that existing OOD evaluation f…
tags:
  - "CVPR 2025"
  - "Frequency shortcuts"
  - "OOD generalization"
  - "Fourier analysis"
  - "texture bias"
  - "model robustness"
date: 2026-05-08
content_hash: 29fcb4504acc5f2c
---

# Do ImageNet-trained Models Learn Shortcuts? The Impact of Frequency Shortcuts on Generalization

**Conference**: CVPR 2025  
**arXiv**: [2503.03519](https://arxiv.org/abs/2503.03519)  
**Code**: [https://github.com/nis-research/hfss](https://github.com/nis-research/hfss)  
**Area**: LLM Evaluation  
**Keywords**: Frequency shortcuts, OOD generalization, Fourier analysis, texture bias, model robustness

## TL;DR
This paper proposes a Hierarchical Frequency Shortcut Search (HFSS) method to efficiently discover frequency shortcuts learned by CNNs and Transformers at the ImageNet-1K scale for the first time (permitting correct classification with only 5% of frequencies). It reveals that frequency shortcuts are surprisingly beneficial in texture-preserving OOD tests but detrimental in stylized tests (IN-R/IN-S), pointing out that existing OOD evaluation frameworks overlook the impact of frequency shortcuts.

## Background & Motivation

**Background**: Deep models are prone to learning shortcut features (spurious correlations) rather than genuine semantic features from training data. Shortcuts in the frequency domain—a small subset of frequencies that models rely on—are more covert and cannot be detected via visual inspection. ImageNet models are known to exhibit texture bias, but a large-scale analysis of frequency shortcuts is still lacking.

**Limitations of Prior Work**: Existing frequency shortcut identification methods (e.g., Wang et al.) evaluate the impact by removing individual frequencies one by one. The computational complexity is proportional to the number of classes and image resolution, requiring approximately 354 days (8,500 hours) on ImageNet-1K, which is completely infeasible. Furthermore, individual frequency evaluation ignores the joint contributions among different frequencies.

**Key Challenge**: Performing large-scale frequency shortcut analysis is computationally intractable, leaving it unclear whether ImageNet-trained models actually learn frequency shortcuts and how these shortcuts affect generalization in diverse OOD scenarios.

**Goal**: To develop an efficient frequency shortcut search method, reveal the frequency shortcut learning behavior of ImageNet models, and analyze its impact on generalization across various OOD scenarios.

**Key Insight**: Replacing the exhaustive search with a hierarchical (coarse-to-fine) search that progressively narrows down the search space in the frequency domain from large to small patches over multiple stages, combined with random sampling to evaluate the joint contributions of frequency combinations.

**Core Idea**: Utilizing hierarchical Fourier frequency search to reveal at the ImageNet scale that models indeed learn frequency shortcuts, and that these shortcuts act as "good shortcuts" in texture-preserving OOD tests, while being detrimental only in stylized tests.

## Method

### Overall Architecture
HFSS consists of multiple search stages. In each stage: (1) the frequency spectrum is divided into patches, and $p\%$ of the patches are randomly sampled to form $B$ candidate frequency subsets; (2) the classification loss is evaluated using the model on images retaining only these frequencies to assess the shortcut information of each subset; (3) the top-$N$ subsets are selected to enter the next stage of finer-grained search. Ultimately, the Dominant Frequency Map (DFM) is output for each class. Classes are then categorized into shortcut and non-shortcut classes to analyze the performance differences between the two groups on various OOD datasets.

### Key Designs

1. **Hierarchical Frequency Search (HFSS)**:

    - **Function**: Efficiently identifying the frequency subsets that the model relies on.
    - **Mechanism**: The frequency spectrum is searched over 6 stages from large patches (56×56) to small patches (2×2). In each stage, 60% of the patches are randomly sampled to construct candidate subsets, and the category loss of the model on frequency-filtered images is calculated as the evaluation metric. Subsequent stages only search within the top-$N$ subsets from the previous stage, exponentially narrowing the search space. This reduces the search time on ImageNet-1K from 354 days to 9.2 days (a 38x speedup).
    - **Design Motivation**: Random sampling + hierarchical narrowing down = considering the joint contributions of frequencies while significantly reducing computational cost, distinct from exhaustive individual-frequency evaluation.

2. **Shortcut Metric and Classification**:

    - **Function**: Quantifying the degree of frequency shortcut learning by the model.
    - **Mechanism**: For each class, the TPR on original images and the $\text{TPR}^{DFM}$ on DFM-filtered images are calculated. If $\text{TPR}^{DFM} > t$ (threshold), the class is classified as a shortcut class. The average TPR of shortcut classes and non-shortcut classes on ID and OOD data are calculated separately, revealing the impact of shortcuts through the performance discrepancy between the two groups.
    - **Design Motivation**: Avoiding averaging over all classes (as the $\text{TPR}^{DFM}$ of non-shortcut classes is close to 0, which would drown out the analysis results); group-based analysis is more insightful.

3. **Differentiated Analysis of OOD Scenarios**:

    - **Function**: Revealing the positive/negative impacts of frequency shortcuts in different OOD scenarios.
    - **Mechanism**: In OOD tests that preserve texture information (IN-v2, IN-C, FGSM adversarial), shortcut classes outperform non-shortcut classes, showing that shortcuts are "beneficial". In stylized/sketch OOD tests (IN-R, IN-S), shortcut classes perform comparable to or worse than non-shortcut classes, because the texture information corresponding to frequency shortcuts does not exist in these datasets.
    - **Design Motivation**: Challenging the simplistic notion that "shortcuts are always harmful", pointing out that the impact depends on OOD data characteristics.

### Loss & Training
HFSS requires no training and only performs inference evaluation on pre-trained models. Standard cross-entropy loss is used to measure the classification relevance of frequency subsets. Experiments cover four architectures: ResNet-18/50, ViT-B, and CCT.

## Key Experimental Results

### Main Results

| Findings | Key Data |
|------|---------|
| ImageNet models indeed learn shortcuts | With DFM images of only 5% frequencies, shortcut classes (t=0.5) achieve TPR > 60% (ResNet-18) |
| Shortcuts are beneficial on IN-v2/IN-C | Shortcut class TPR > non-shortcut class TPR (consistent across all models) |
| Shortcuts are harmful on IN-R | Shortcut class TPR < non-shortcut class TPR (rendition changes destroy texture) |
| Shortcuts are neutral on IN-S | The TPRs of both groups are close (sketches preserve partial structural information) |
| Shortcuts are beneficial under FGSM adversarial attacks | Shortcut classes are more robust (adversarial noise is unlikely to alter texture information) |

### Ablation Study

| Configuration | Search Time | Number of Shortcut Classes Found | Description |
|------|---------|------------------|------|
| CF-1 (Most complete) | ~7.5h (CIFAR) | Baseline | Full search |
| CF-2.10 (Most efficient) | ~0.04h | ~90% coverage (low threshold) | 200x acceleration |
| Wang et al. [41] | 7.5h (CIFAR) | Only 2 strong shortcut classes | Individual-frequency evaluation |
| HFSS (Ours) | 0.5h (CIFAR) | 6 strong shortcut classes | Joint frequency contributions |

### Key Findings
- **Interesting Behavior of CCT**: CCT exhibits the highest degree of shortcut learning, yet it simultaneously learns other semantic features. This "learning shortcuts without hindering genuine learning" pattern paradoxically allows CCT to outperform ResNet-50 of similar accuracy on IN-C (57.73% vs. 48.85%).
- HFSS is more effective in finding shortcuts than existing methods (6 vs. 2 strong shortcut classes) because it accounts for the joint contributions of frequencies.
- Frequency shortcuts are directly associated with texture bias—shortcuts primarily correspond to texture rather than shape information.

## Highlights & Insights
- **Subversive Conclusion: Shortcuts Are Not Necessarily Harmful**: In texture-preserving OOD scenarios (including adversarial attacks), frequency shortcuts actually enhance model robustness. This poses a significant challenge to the paradigm of "eliminating all shortcuts."
- **Efficiency of Hierarchical Search**: Reducing execution time from 354 days to 9.2 days makes large-scale frequency analysis feasible for the first time. The coarse-to-fine + random sampling paradigm can be transferred to other large-scale search problems.
- **Insights on OOD Evaluation Benchmarks**: Existing OOD evaluation frameworks do not distinguish whether "shortcuts are present in test data," potentially yielding misleading generalization assessments. Future benchmarks should explicitly consider the impact of frequency shortcuts.

## Limitations & Future Work
- The search results of HFSS depend on sampling strategies and hyperparameters (patch size, sampling ratio), potentially missing certain frequency combinations.
- The analysis is limited to classification tasks; frequency shortcuts in tasks like detection and segmentation remain uninvestigated.
- Only post-training models were analyzed; the formation dynamics of frequency shortcuts during the training process have not been explored.
- Lacks the proposal of shortcut mitigation methods—problems are identified, but no solutions are provided.

## Related Work & Insights
- **vs. Wang et al. [41]**: Individual-frequency evaluation vs. joint-frequency search; HFSS discovers more and stronger shortcuts, yielding a 38x increase in computational efficiency.
- **vs. Geirhos et al. [8] (Texture Bias)**: Explaining the formation mechanism of texture bias from the frequency-domain perspective—frequency shortcuts are the root cause of texture bias.
- **vs. OOD Evaluation Benchmarks**: Revealing that the effectiveness of benchmarks like IN-R/IN-S partially stems from the removal of frequency shortcuts present in training data.

## Rating
- Novelty: ⭐⭐⭐⭐ The first large-scale frequency shortcut analysis; the finding that "shortcuts can be beneficial" is highly inspiring.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple models, diverse OOD scenarios, comparisons against existing methods, and configuration ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear experimental design and analytical logic with intuitive visualizations.
- Value: ⭐⭐⭐⭐ Holds significant guiding value for understanding model generalization and designing OOD benchmarks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Implicit Reasoning in Transformers is Reasoning through Shortcuts](../../ACL2025/others/implicit_reasoning_in_transformers_is_reasoning_through_shortcuts.md)
- [\[CVPR 2025\] On the Generalization of Handwritten Text Recognition Models](on_the_generalization_of_handwritten_text_recognition_models.md)
- [\[ICML 2025\] How Do Transformers Learn Variable Binding in Symbolic Programs?](../../ICML2025/others/how_do_transformers_learn_variable_binding_in_symbolic_programs.md)
- [\[NeurIPS 2025\] Impact of Layer Norm on Memorization and Generalization in Transformers](../../NeurIPS2025/others/impact_of_layer_norm_on_memorization_and_generalization_in_transformers.md)
- [\[CVPR 2025\] EBS-EKF: Accurate and High Frequency Event-based Star Tracking](ebs-ekf_accurate_and_high_frequency_event-based_star_tracking.md)

</div>

<!-- RELATED:END -->
