---
title: >-
  [Paper Note] Sync from the Sea: Retrieving Alignable Videos from Large-Scale Datasets
description: >-
  [ECCV 2024][LLM Evaluation][Video Alignment] This paper proposes the task of Alignable Video Retrieval (AVR), which identifies and retrieves the most suitable videos for temporal alignment with a query video from a large-scale video database using the DRAQ alignment quality metric. It also introduces a feature contextualization method to improve alignment performance.
tags:
  - "ECCV 2024"
  - "LLM Evaluation"
  - "Video Alignment"
  - "Video Retrieval"
  - "Temporal Alignment"
  - "Dynamic Time Warping"
  - "Self-supervised Features"
date: 2026-05-08
content_hash: bcbc3a9ba3551ff3
---

# Sync from the Sea: Retrieving Alignable Videos from Large-Scale Datasets

**Conference**: ECCV 2024  
**arXiv**: [2409.01445](https://arxiv.org/abs/2409.01445)  
**Code**: [Yes](https://daveishan.github.io/avr-webpage/)  
**Area**: LLM Evaluation  
**Keywords**: Video Alignment, Video Retrieval, Temporal Alignment, Dynamic Time Warping, Self-supervised Features

## TL;DR

This paper proposes the task of Alignable Video Retrieval (AVR), which identifies and retrieves the most suitable videos for temporal alignment with a query video from a large-scale video database using the DRAQ alignment quality metric. It also introduces a feature contextualization method to improve alignment performance.

## Background & Motivation

Temporal video alignment aims to synchronize key events (such as action phase transitions, object interactions, etc.) across two videos, which is crucial for applications like video editing, audio track transfer, and exemplar-driven temporal remapping. However, existing methods suffer from two core limiting assumptions:

**Given video pairs**: Existing methods only focus on how to align a known pair of videos, neglecting the preliminary question of "how to find videos suitable for alignment".

**Limited to well-behaved action classes**: Such as baseball swings, which have a fixed sequence of action phases, ignoring the massive variability in general videos.

Taking "cutting a pineapple" as an example, different videos within the same action category can have completely different execution styles. Simply knowing the action category is insufficient to determine whether two videos can be aligned. Therefore, action-recognition-based retrieval methods are inadequate for identifying alignable video pairs, necessitating dedicated alignment quality evaluation and reranking schemes.

## Method

### Overall Architecture

The AVR pipeline consists of three phases:

| Phase | Goal | Method |
|------|------|------|
| 1. Candidate Retrieval | Retrieve top-k candidates from a large-scale database | k-NN retrieval based on clip-level features |
| 2. Reranking | Identify the most alignable videos from candidates | Reranking using the DRAQ metric |
| 3. Temporal Alignment | Align the best match with the query video | DTW based on contextualized features |

The system encodes both whole-video and frame-level features based on temporally self-supervised pretrained video representations (NMS features).

### Key Designs

**Contextualized frame-level features**: Each frame feature is concatenated with the cumulative mean of the features up to that moment, injecting global temporal context into the frame features. Specifically, for a video with $T$ frames, the contextualized feature is:

$$\bar{f}_j^{(i)} = f_j^{(i)} \oplus \frac{1}{T} \sum_{t=1}^{j} f_t^{(i)}$$

Each clip's features are further normalized via zero-centering. This design allows frame features to not only capture the current scene (e.g., human pose) but also encode the positional information of that moment within the entire action sequence (whether it is at the beginning or the end). This method is general, requires no extra training, and can be applied to any frame-level features.

**DRAQ: Dynamic Relative Alignment Quality evaluation**. Directly using the optimal path cost of DTW $D(n,m)$ to evaluate alignment quality is dominated by appearance similarity rather than temporal alignability. DRAQ eliminates the appearance bias by taking the ratio of the optimal alignment cost to a random alignment cost:

$$\text{DRAQ} = \frac{D(n,m)}{\text{Cost}_{\text{random}}}$$

The sampling strategy for random paths: starting from $(n,m)$, directions are sampled with position ratios $P_{\text{up}} = i/(i+j)$ and $P_{\text{left}} = j/(i+j)$, steering the random path toward the diagonal and increasing the "challenge". $k$ random paths are generated and their average cost is obtained.

A lower DRAQ indicates a greater improvement of the optimal alignment over the random alignment, meaning the two videos are more likely to be meaningfully aligned. Since the cost matrix $C$ only needs to be computed once, random path sampling is highly efficient, introducing almost zero extra overhead compared to DTW.

### Loss & Training

This method **requires no extra training**—it directly utilizes pretrained video feature representations. The core innovations lie in the feature design and reranking algorithm during the inference stage:

- Video representation employs NMS self-supervised pretrained features.
- Candidate retrieval uses approximate nearest neighbors with cosine similarity.
- Alignment uses the standard DTW algorithm.
- DRAQ uses $k=10$ random paths.

## Key Experimental Results

### Main Results

**AVR cycle-consistency evaluation (PennAction↺, Penn⇄UCF, Kinetics700↺)**:

| Candidate Source | Feature | DRAQ Reranking | PennAction FPE↓ | PennAction CPE↓ | Kinetics FPE↓ | Kinetics CPE↓ |
|---------|------|----------|----------------|----------------|---------------|---------------|
| NMS Retrieval | NMS | ✗ | 13.4 | 1.32 | 22.7 | 0.86 |
| NMS Retrieval | NMS | ✓ | **9.5** | **0.20** | **0.5** | **0.0** |
| Oracle | NMS | ✗ | 24.7 | 1.70 | 35.3 | 1.08 |
| Oracle | NMS | ✓ | **7.8** | **0.33** | **0.3** | **0.01** |

### Ablation Study

**Impact of feature contextualization on alignment quality (PennAction APA%↑)**:

| Feature | W/o Contextualization Avg | W/ Contextualization Avg | W/o Contextualization Top-DRAQ | W/ Contextualization Top-DRAQ |
|------|--------------|---------------|-------------------|-------------------|
| BYOL | 0.769 | Gain | 0.814 | Gain |
| CARL | 0.826 | Gain | 0.856 | Gain |
| NMS | Baseline | Gain | Baseline | Gain |

### Key Findings

1. **DRAQ reranking has a significant effect**: Across all datasets and feature combinations, DRAQ reranking drastically reduces the cycle-consistency error. On Kinetics, CPE drops from 0.86 to 0.0.
2. **Feature contextualization is universally effective**: Adding cumulative context to any frame-level features improves alignment performance.
3. **DRAQ outperforms DTW cost as a reranking metric**: Relative metrics eliminate the appearance bias.
4. **NMS features perform the best**: Features pretrained with temporal self-supervision are naturally suited for alignment tasks.
5. **Cross-dataset alignment (Penn⇄UCF) is more challenging**: Due to mismatched action categories and a restricted retrieval set.

## Highlights & Insights

- **Contribution of problem definition**: It defines the AVR task for the first time, expanding video alignment from "known video pairs" to real-world scenarios of "large-scale retrieval + alignment".
- **Simplicity and elegance of the method**: Both DRAQ and feature contextualization are plug-and-play, lightweight schemes that do not require extra training.
- **Innovation of the evaluation protocol**: It proposes an AVR evaluation method based on Cycle Consistency, avoiding the expensive cost of dense manual annotations.
- **Reflections on existing benchmarks**: It points out that the proxy metric on PennAction can "cheat" via positional encoding, and proposes the more direct Aligned Phase Agreement metric.

## Limitations & Future Work

1. Cross-dataset alignment (semantic alignment between different action classes) shows limited effectiveness.
2. The random path sampling strategy of DRAQ still has room for optimization (e.g., learnable sampling).
3. Feature contextualization only uses a simple cumulative mean; more complex temporal aggregation could be explored.
4. Partial alignment scenarios, where two videos are only partially alignable, are not considered.
5. The efficiency of large-scale retrieval (particularly as DRAQ requires computing DTW for each candidate) can be further optimized.

## Related Work & Insights

- The classic role of DTW in video alignment is revisited and enhanced in this work.
- The success of self-supervised temporal features (NMS) demonstrates that time-sensitive pretraining is crucial for alignment.
- The concept of cycle-consistency evaluation, adapted from visual correspondence and optical flow fields, is cleverly tailored for AVR.

## Rating

| Dimension | Rating (1-5) | Explanation |
|------|-----------|------|
| Novelty | 4 | Proposes the AVR task for the first time, and the DRAQ metric is novel and highly practical. |
| Technical Depth | 3.5 | The method is simple but not complex; the core contributions lie in the problem definition and evaluation protocol. |
| Experimental Thoroughness | 4 | Comprehensive evaluation across 3 datasets, multiple features, and various settings. |
| Value | 4 | Plug-and-play, training-free, and directly applicable to large-scale video databases. |
| Overall | 4 | Defines an important and practical new task, and proposes a simple yet effective baseline solution. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] PetFace: A Large-Scale Dataset and Benchmark for Animal Identification](petface_a_large-scale_dataset_and_benchmark_for_animal_identification.md)
- [\[ICML 2025\] PhantomWiki: On-Demand Datasets for Reasoning and Retrieval Evaluation](../../ICML2025/llm_evaluation/phantomwiki_on-demand_datasets_for_reasoning_and_retrieval_evaluation.md)
- [\[ICLR 2026\] The Open Proof Corpus: A Large-Scale Study of LLM-Generated Mathematical Proofs](../../ICLR2026/llm_evaluation/the_open_proof_corpus_a_large-scale_study_of_llm-generated_mathematical_proofs.md)
- [\[ACL 2025\] HellaSwag-Pro: A Large-Scale Bilingual Benchmark for Evaluating the Robustness of LLMs in Commonsense Reasoning](../../ACL2025/llm_evaluation/hellaswag-pro_a_large-scale_bilingual_benchmark_for_evaluating_the_robustness_of.md)
- [\[ICML 2026\] Authority, Truth, and Citation Bias: A Large-Scale Multi-Domain Benchmark for Studying Epistemic Susceptibility in Large Language Models](../../ICML2026/llm_evaluation/authority_truth_and_citation_bias_a_large-scale_multi-domain_benchmark_for_study.md)

</div>

<!-- RELATED:END -->
