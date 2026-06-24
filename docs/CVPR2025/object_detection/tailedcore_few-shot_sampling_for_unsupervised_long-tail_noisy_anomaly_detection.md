---
title: >-
  [Paper Note] TailedCore: Few-Shot Sampling for Unsupervised Long-Tail Noisy Anomaly Detection
description: >-
  [CVPR 2025][Object Detection][Unsupervised Anomaly Detection] TailedCore addresses a realistic scenario in unsupervised anomaly detection where normal samples contain noisy defects and concurrently follow an unknown long-tail class distribution. It proposes TailSampler to predict class cardinality based on the symmetry assumption of embedding similarity, allowing for the independent sampling of tail-class specimens. This constructs a memory bank model that captures tail-class…
tags:
  - "CVPR 2025"
  - "Object Detection"
  - "Unsupervised Anomaly Detection"
  - "Long-Tail Distribution"
  - "Noise Robustness"
  - "Memory Bank"
  - "Few-Shot Sampling"
date: 2026-05-08
content_hash: 47c0c7de868fca71
---

# TailedCore: Few-Shot Sampling for Unsupervised Long-Tail Noisy Anomaly Detection

**Conference**: CVPR 2025  
**arXiv**: [2504.02775](https://arxiv.org/abs/2504.02775)  
**Code**: [https://github.com/YoonGyoJung/TailedCore](https://github.com/YoonGyoJung/TailedCore)  
**Area**: Others  
**Keywords**: Unsupervised Anomaly Detection, Long-Tail Distribution, Noise Robustness, Memory Bank, Few-Shot Sampling

## TL;DR
TailedCore addresses a realistic scenario in unsupervised anomaly detection where normal samples contain noisy defects and concurrently follow an unknown long-tail class distribution. It proposes TailSampler to predict class cardinality based on the symmetry assumption of embedding similarity, allowing for the independent sampling of tail-class specimens. This constructs a memory bank model that captures tail-class information while remaining robust to noise, outperforming SOTA in various settings.

## Background & Motivation

**Background**: Industrial anomaly detection typically assumes that the training set contains only "normal" samples. Memory-bank-based methods (such as PatchCore) detect anomalies during testing by storing a representative subset of normal features. These methods perform excellently under ideal conditions.

**Limitations of Prior Work**: In real-world industrial scenarios, normal training data faces two co-occurring challenges: (1) **noise pollution**—a small number of defective samples (pixel-level noise) are inevitably mixed into the collected "normal" data; (2) **long-tail distribution**—normal products have multiple classes or variants, but the class distribution is long-tailed and unknown. Existing methods fail when attempting to handle both problems simultaneously.

**Key Challenge**: There exists a trade-off of "tail class vs. noise". Noise-robust methods (e.g., denoising by filtering rare features) mistakenly delete rare normal features of tail classes; tail-class-friendly methods (e.g., retaining all rare features) fail to filter noise. The two cannot be optimized simultaneously.

**Goal**: To break the tail-vs-noise trade-off, enabling the model's memory bank to contain complete information of tail classes while remaining uncontaminated by noise features.

**Key Insight**: The authors propose to decouple the processing of tail-class samples and noise samples—first identifying which samples belong to the tail classes to process them separately (enhancing representation), while applying standard noise filtering to non-tail-class samples. Consequently, the two issues no longer interfere with each other.

**Core Idea**: TailSampler is used to predict the class cardinality of each sample based on the symmetry of embedding similarity distribution, thereby accurately sampling tail-class samples. These are processed separately from head-class samples to construct a split-and-conquer memory bank.

## Method

### Overall Architecture
TailedCore is a memory-bank-based anomaly detection model. The input is a set of unlabeled "normal" images (which may contain noise and long-tailed classes), and the output consists of anomaly scores and anomaly localization heatmaps for each test image. The workflow is divided into three steps: (1) use a pre-trained feature extractor to obtain image/patch embeddings; (2) TailSampler predicts the class cardinality of each sample, dividing samples into head and tail classes; (3) perform standard coreset sampling and noise filtering on the head classes, while fully retaining or performing enhanced sampling on the tail classes, merging them to construct the final memory bank.

### Key Designs

1. **TailSampler (Tail Class Sampler)**:

    - **Function**: Estimates the size of the class (class cardinality) to which each sample belongs, thereby identifying tail-class samples.
    - **Mechanism**: Based on a key observation—if a sample belongs to a large class, its embedding similarity distribution with other samples will have more samples in the high-similarity region; if it belongs to a small class, there will be very few samples in the high-similarity region. Specifically, the cosine similarity of each sample with all other samples is calculated, and the **Symmetric Assumption**—which assumes the embedding distribution of each class is symmetric around the class center—is leveraged to estimate the class cardinality. If sample $x$ has an average similarity of $s$ with its top $k$ nearest neighbors, its class cardinality is estimated as $\hat{n} = f(s, k)$, where $f$ is derived from the properties of the symmetric distribution.
    - **Design Motivation**: Directly using clustering methods to estimate classes requires pre-setting the number of clusters (which is unknown here), and clustering is highly sensitive to noise. TailSampler does not require explicit clustering and estimates class scale solely through local similarity statistics.

2. **Split Memory Construction**:

    - **Function**: Separately processes head-class and tail-class samples to construct a memory bank that is both comprehensive and clean.
    - **Mechanism**: After TailSampler outputs the class cardinality estimate for each sample, a threshold $\tau$ is set to split samples into a head-class set $\mathcal{H}$ and a tail-class set $\mathcal{T}$. **For head classes**: sample size is abundant, allowing standard coreset sampling (such as greedy coreset selection) and filtering of potential noise samples; **for tail classes**: samples are inherently scarce, so a more conservative strategy is deployed—increasing the sampling ratio to ensure tail features are sufficiently represented in the memory bank. The final memory bank is $\mathcal{M} = \mathcal{M}_H \cup \mathcal{M}_T$.
    - **Design Motivation**: This split-and-conquer strategy is key to breaking the tail-vs-noise trade-off. Head classes can be safely denoised (deleting a few samples does not compromise representation), while tail classes are preserved with higher priority (retaining a small amount of noise is preferable to losing rare normal patterns).

3. **Noise-Robust Coreset Sampling**:

    - **Function**: Performs noise-aware representative sampling within head-class samples.
    - **Mechanism**: Standard coreset sampling is extended by introducing a noise score. For each patch feature, consistency with its local neighborhood is calculated—if a patch feature's similarity with its $k$-NN is abnormally lower than the average level of the same class, it is flagged as suspected noise. During sampling, the priority of suspected noise patches is reduced.
    - **Design Motivation**: Standard coreset sampling intentionally selects samples furthest from already chosen features to maximize coverage, which unfortunately makes it prone to selecting noise samples (since they are often outliers) into the memory bank.

### Loss & Training
As a memory-bank-based method, TailedCore does not require additional training. Feature extraction utilizes a pre-trained backbone (e.g., WideResNet-50). During testing, the anomaly score is computed as the distance from the test patch feature to its nearest neighbor in the memory bank.

## Key Experimental Results

### Main Results

Evaluation is conducted on the MVTec-AD and VisA datasets under different configurations of imbalance ratios (IR) and noise ratios (NR).

| Method | MVTec (IR=100, NR=5%) | MVTec (IR=50, NR=10%) | VisA (IR=100, NR=5%) | VisA (IR=50, NR=10%) |
|------|----------------------|----------------------|---------------------|---------------------|
| PatchCore | 87.2 | 83.5 | 82.1 | 78.4 |
| SoftPatch | 89.1 | 86.3 | 84.5 | 81.2 |
| NoisyAD | 88.7 | 87.1 | 83.8 | 80.6 |
| **TailedCore** | **92.4** | **90.8** | **88.3** | **85.7** |

### Ablation Study

| Configuration | MVTec (AUROC) | Description |
|------|-------------|------|
| Full TailedCore | 92.4 | Full model |
| w/o TailSampler | 88.1 | No head/tail separation, unified processing, drops 4.3% |
| w/o Noise Filtering | 90.2 | No noise-aware sampling, drops 2.2% |
| w/o Split Memory | 88.9 | Unified sampling strategy, drops 3.5% |
| Random sampler replacing TailSampler | 88.5 | Randomly sample tail classes, drops 3.9% |

### Key Findings
- TailSampler is the most critical component, with its removal leading to the most severe degradation (-4.3%), demonstrating that accurately identifying tail classes is core to resolving the tail-vs-noise trade-off.
- Under high noise ratio (10%) settings, the advantages of TailedCore are more pronounced, indicating that the split-and-conquer strategy yields higher gains in difficult scenarios.
- The symmetry assumption of TailSampler holds in practical distributions: experiments verify that pre-trained features of industrial images indeed approximately satisfy intra-class symmetric distribution.
- Under noise-only without long-tail settings ($IR=1, NR>0$), TailedCore is on par with SoftPatch; under long-tail-only without noise settings ($IR>1, NR=0$), it outperforms PatchCore. This shows that TailedCore incurs no extra cost for implementing "split-and-conquer".

## Highlights & Insights
- **Estimating class cardinality via the symmetry assumption is highly elegant**: It requires neither clustering nor density estimation. By utilizing only $k$-NN similarity statistics, it can distinguish head and tail classes efficiently with no hyperparameter sensitivity issues.
- **The formulation of the problem itself is a significant contribution**: It is the first to formally introduce "long-tail + noise" as a joint challenge in anomaly detection, systematizing the analysis of the tail-vs-noise trade-off.
- **The split-and-conquer philosophy is transferable to other memory bank methods**: Any method based on representative subsets can incorporate a similar head-tail separation strategy.

## Limitations & Future Work
- TailSampler relies heavily on the quality of pre-trained features. If the backbone lacks discriminative capability for certain product types, the symmetry assumption may fail.
- The threshold $\tau$ still requires empirical tuning, and different datasets may necessitate different thresholds.
- Validation has only been performed on industrial anomaly detection scenarios; whether it is effective for long-tail noise scenarios in natural images or medical domains remains untested.
- The method does not handle cases with gradual or continuous distributions between classes (e.g., the same product with a color gradient), where the definition of "class" itself becomes ambiguous.

## Related Work & Insights
- **vs PatchCore**: PatchCore is a classic memory bank method that assumes a clean and balanced training set. The primary advantage of TailedCore lies in its capability to handle non-ideal data.
- **vs SoftPatch**: SoftPatch addresses noise using soft weights but overlooks the long-tail problem. While they are comparable under pure noise settings, TailedCore significantly outperforms SoftPatch under the joint long-tail + noise settings.
- **vs NoisyAD**: NoisyAD focuses on noise robustness but ignores class imbalance, showing performance degradation under high imbalance ratios.

## Rating
- Novelty: ⭐⭐⭐⭐ The problem definition is novel (joint long-tail + noise), and the design of TailSampler is ingenious, though the overall framework still operates within the PatchCore paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐ Systematic evaluation across various combinations of imbalance and noise ratios with comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear trade-off analysis and convincing motivational explanations.
- Value: ⭐⭐⭐⭐ Imperfect data is common in real-world industrial scenarios; this work directly addresses actual pain points.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Search and Detect: Training-Free Long Tail Object Detection via Web-Image Retrieval](search_and_detect_training-free_long_tail_object_detection_via_web-image_retriev.md)
- [\[CVPR 2025\] UniVAD: A Training-free Unified Model for Few-shot Visual Anomaly Detection](univad_a_training-free_unified_model_for_few-shot_visual_anomaly_detection.md)
- [\[AAAI 2026\] Commonality in Few: Few-Shot Multimodal Anomaly Detection via Hypergraph-Enhanced Memory](../../AAAI2026/object_detection/commonality_in_few_few-shot_multimodal_anomaly_detection_via_hypergraph-enhanced.md)
- [\[ICLR 2026\] Dual Distillation for Few-Shot Anomaly Detection](../../ICLR2026/object_detection/dual_distillation_for_few-shot_anomaly_detection.md)
- [\[CVPR 2025\] AA-CLIP: Enhancing Zero-Shot Anomaly Detection via Anomaly-Aware CLIP](aa-clip_enhancing_zero-shot_anomaly_detection_via_anomaly-aware_clip.md)

</div>

<!-- RELATED:END -->
