---
title: >-
  [Paper Note] Dataset Growth (InfoGrowth)
description: >-
  [ECCV2024][Multimodal VLM][Data Cleaning] InfoGrowth is proposed as an efficient online data cleaning and selection algorithm. By estimating the information gain of each sample through nearest neighbor search, it enables continuous dataset growth while maintaining cleanliness and diversity, outperforming full training on CC3M using only 1/6 of the data.
tags:
  - "ECCV2024"
  - "Multimodal VLM"
  - "Data Cleaning"
  - "Online Data Selection"
  - "Submodular Functions"
  - "Data Efficiency"
  - "Multimodal Pre-training"
date: 2026-05-08
content_hash: e72a80b5d5e7ec49
---

# Dataset Growth (InfoGrowth)

**Conference**: ECCV2024  
**arXiv**: [2405.18347](https://arxiv.org/abs/2405.18347)  
**Code**: [https://github.com/NUS-HPC-AI-Lab/InfoGrowth](https://github.com/NUS-HPC-AI-Lab/InfoGrowth)  
**Area**: Multimodal VLM  
**Keywords**: Data Cleaning, Online Data Selection, Submodular Functions, Data Efficiency, Multimodal Pre-training

## TL;DR
InfoGrowth is proposed as an efficient online data cleaning and selection algorithm. By estimating the information gain of each sample through nearest neighbor search, it enables continuous dataset growth while maintaining cleanliness and diversity, outperforming full training on CC3M using only 1/6 of the data.

## Background & Motivation
**Background**: Deep learning relies on large-scale data, but web data is growing exponentially (0.33 ZB generated per day in 2024). Ultra-large-scale datasets such as LAION-5B and Common Crawl are virtually uncleaned.

**Limitations of Prior Work**: Manual annotation/cleaning is no longer feasible at today's data scale. Existing cleaning methods (e.g., bootstrapping in BLIP) require training models on the full dataset, which cannot scale to larger volumes. Furthermore, most existing methods are designed for offline scenarios, addressing either noise or redundancy but not both.

**Key Challenge**: Data continuously flows in with highly variable quality, where noise and redundancy coexist. The "collect first, clean later" paradigm is highly inefficient and requires iterative execution.

**Goal**: How to simultaneously handle noise and redundancy in a single step as data arrives in streams, efficiently maintaining a growing and high-quality dataset.

**Key Insight**: Recursively formulate the problem. Compared to the existing dataset, each incoming data point belongs to one of three states: noise (needs filtering/re-labeling), redundancy (needs reduced sampling frequency), or new information (needs to be added normally). These states can be efficiently determined via nearest-neighbor relationships in the embedding space.

**Core Idea**: Online nearest neighbor search is leveraged to replace the full submodular function computation, reducing the complexity of estimating the information gain of each sample from $O(n)$ to $O(\log n)$, thereby enabling real-time quality evaluation of streaming data.

## Method

### Overall Architecture
InfoGrowth is a streaming data processing pipeline. Given streaming data pairs $(d_i, c_i)$, it outputs a clean and diverse dataset after routing through three modules:
1. **Cleaner**: Detects noisy samples and relabels them.
2. **Gain Calculator**: Computes information gain based on HNSW nearest neighbor search.
3. **Selector/Sampler**: Samples data according to the computed gain values.

All data are first mapped to the embedding space via a BLIP encoder, and subsequent operations are performed entirely within this embedding space.

### Key Designs

1. **Cleaner (Noise Detection and Correction)**:

    - **Function**: Detects noisy multimodal samples (mismatched image-text pairs) and attempts to correct them.
    - **Mechanism**: The BLIP encoder is used to calculate the cosine similarity between image and text embeddings. Samples below a threshold $\delta$ are identified as noise. For noisy samples, new captions are regenerated using MiniGPT-4 (recaptioning). After relabeling, the samples are checked again; if they still fail, they are discarded.
    - **Design Motivation**: Compared to the original BLIP method that requires an additional encoder for ITM classification, directly using cosine similarity is more efficient. Leveraging state-of-the-art LLMs (MiniGPT-4) for recaptioning ensures higher quality.

2. **Gain Calculator (Information Gain Computation)**:

    - **Function**: Quantifies the marginal informational contribution of each sample to the dataset.
    - **Mechanism**: HNSW online approximate nearest neighbor search is used to find the $k$ nearest neighbors. The average cosine distance between the sample and its neighbors is computed as the information gain: $Gain_{Info} = mean_i[cos\text{-}dis(d, neighbour_i)]$. A larger distance indicates a more "novel" sample, resulting in higher gain.
    - For single-modal classification tasks, an additional entropy gain is introduced: $Gain_{Entropy} = 1 - p$, where $p$ is the probability of correctly predicting the class of the current sample using its neighbors' classes. This assigns higher weights to samples near decision boundaries.
    - **Design Motivation**: Traditional submodular functions with $O(n)$ complexity cannot scale to billion-scale datasets. HNSW reduces both query and update complexities to $O(\log n)$ while preserving the submodular property ($U \subseteq V \Rightarrow f(x,U) \geq f(x,V)$).

3. **Sampler (Sampling Strategy)**:

    - **Function**: Selects high-value samples based on gain values.
    - Static Sampling: Samples up to the target quantity without replacement, using probabilities normalized by gain values: $\mathcal{P}(x_i) = G_i / \sum G_j$.
    - Dynamic Two-Stage Sampling: The first stage samples using gain-based probabilities to ensure global diversity. The second stage reverses the gain as $G'(x_i) = \max(0.1, 1-G_i)$ to sample for local diversity and generalization. Alternating between these two stages epoch-by-epoch saves over 45% of training overhead.
    - **Design Motivation**: Pure gain-based sampling may disregard "common but crucial" samples. The two-stage strategy strikes a balance between diversity and generalization.

### Loss & Training
InfoGrowth itself does not modify the training loss of downstream models. BLIP is trained using contrastive loss + ITM loss + LM loss. The value of InfoGrowth lies at the data level—providing higher-quality training data.

## Key Experimental Results

### Main Results

| Dataset/Task | Data Volume | COCO TR@1 (zero-shot) | COCO IR@1 (zero-shot) | Flickr TR@1 (finetune) | Flickr IR@1 (finetune) |
|------------|--------|----------------------|----------------------|----------------------|----------------------|
| CC3M (Original) | 2.71M | 34.7 | 28.0 | 89.2 | 75.1 |
| InfoGrowth | 0.40M | 37.1 (+2.4) | 28.1 (+0.1) | 84.6 | 69.5 |
| InfoGrowth | 0.68M | 41.1 (+6.4) | 32.8 (+4.8) | 89.0 | 74.6 |
| InfoGrowth | 1.35M | 47.7 (+13.0) | 38.1 (+10.1) | 91.5 | 77.7 |

Using only 0.4M data (15% of the original), the method outperforms the full CC3M dataset on COCO zero-shot retrieval, while showing improvements in downstream tasks like VQA, NLVR2, and image captioning.

### Ablation Study

| Configuration | TR@1 | IR@1 | Description |
|------|------|------|------|
| Random baseline | 21.8 | 18.7 | Randomly select 0.4M |
| + Recaption only | 34.1 | 27.1 | Noise cleaning only |
| + Sampling only | 23.9 | 21.0 | Gain-based sampling only |
| + Recaption + Sampling | 37.1 | 28.1 | Full InfoGrowth |
| Full CC3M (2.7M) | 34.7 | 28.0 | Full data reference |

### Key Findings
- Data cleaning (recaption) contributes the most, improving TR@1 from 21.8 to 34.1; gain-based sampling further increases it to 37.1.
- Number of nearest neighbors $k=4$ combined with arithmetic mean yields the best results; excessively large $k$ degrades performance.
- Multimodal gain (image + text) outperforms single-modal gain.
- On the ImageNet single-modal task, it achieves 75.8% accuracy with 50% of the data, outperforming GC (72.8%), EL2N (74.6%), and UCB (75.3%).
- The method is equally effective for the CLIP architecture, demonstrating independence of specific model structures.
- It exhibits robust noise tolerance under scenarios with 10%/25% artificially injected noise.

## Highlights & Insights
- **Online Streaming Design**: Eliminates the need to "collect all data before processing," making it suitable for continuous execution in real-world data engines, which is highly practical for large-scale data processing.
- **HNSW Replacing Full Submodular Computation**: Reduces the total complexity from $O(n^2)$ to $O(n \log n)$ while preserving the theoretical guarantees of submodularity, demonstrating an elegant balance between engineering and theory.
- **Two-Stage Dynamic Sampling**: Alternates between global diversity and local generalization, saving 45% of training overhead without performance degradation.
- **Unified Bayesian Perspective**: Unifies datasets for different tasks like classification, retrieval, and generation under the conditional distribution $P(d|c)$ perspective, providing a theoretical foundation for cross-task data selection.

## Limitations & Future Work
- Reliance on the quality of the pre-trained BLIP encoder: If the encoder performs poorly on a certain data category, both cleaning and gain estimation will be affected.
- Recaptioning depends on MiniGPT-4, introducing additional computational overhead and model dependencies.
- Evaluated only on CC3M (2.7M) and ImageNet-1K; not yet tested on larger scales (such as the LAION-5B level).
- The two-stage alternating strategy in dynamic sampling is somewhat heuristic, lacking theoretical analysis regarding the optimal alternation frequency.
- The selection of the threshold $\delta$ impacts results, but adaptive $\delta$ was not fully explored in the paper.

## Related Work & Insights
- **vs BLIP bootstrapping**: BLIP requires training models on the full dataset to perform filtering and recaptioning, whereas InfoGrowth processes streams in the embedding space without requiring full training.
- **vs DivideMix**: DivideMix uses co-teaching with dual models to detect noise, focusing solely on the noise problem, whereas InfoGrowth addresses both noise and redundancy simultaneously.
- **vs Traditional Coreset Selection (GC, EL2N, UCB)**: These methods require pre-training models to obtain gradients/loss information offline, while InfoGrowth runs online without pre-training.
- This data cleaning paradigm can be extended to other multimodal data processing tasks like video-text and audio-text.

## Rating
- Novelty: ⭐⭐⭐⭐ Combining submodular functions with HNSW for online data selection is novel, and the unified Bayesian perspective is inspiring.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on both multimodal and single-modal tasks with comprehensive ablations, though experiments on larger datasets are missing.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical analysis with a smooth transition from motivation to methodological derivation.
- Value: ⭐⭐⭐⭐ Addresses real-world problems with a practical and scalable approach, offering solid reference value for large-scale data engineering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] A Multimodal Benchmark Dataset and Model for Crop Disease Diagnosis](a_multimodal_benchmark_dataset_and_model_for_crop_disease_di.md)
- [\[NeurIPS 2025\] SmartWilds: Multimodal Wildlife Monitoring Dataset](../../NeurIPS2025/multimodal_vlm/smartwilds_multimodal_wildlife_monitoring_dataset.md)
- [\[ACL 2025\] ViGiL3D: A Linguistically Diverse Dataset for 3D Visual Grounding](../../ACL2025/multimodal_vlm/vigil3d_a_linguistically_diverse_dataset_for_3d_visual_grounding.md)
- [\[ICLR 2026\] Multimodal Dataset Distillation via Phased Teacher Models](../../ICLR2026/multimodal_vlm/multimodal_dataset_distillation_via_phased_teacher_models.md)
- [\[ICLR 2026\] ChartGalaxy: A Dataset for Infographic Chart Understanding and Generation](../../ICLR2026/multimodal_vlm/chartgalaxy_a_dataset_for_infographic_chart_understanding_and_generation.md)

</div>

<!-- RELATED:END -->
