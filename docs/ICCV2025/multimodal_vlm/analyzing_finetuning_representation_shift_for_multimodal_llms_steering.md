---
title: >-
  [Paper Note] Analyzing Finetuning Representation Shift for Multimodal LLMs Steering
description: >-
  [ICCV 2025][Multimodal VLM][MLLM Interpretability] A training-free framework that reveals representation shifts in multimodal large language models (MLLMs) during finetuning through concept-level analysis, and leverages shift vectors for lightweight model behavior steering (debiasing, safety control).
tags:
  - ICCV 2025
  - Multimodal VLM
  - MLLM Interpretability
  - Concept Drift
  - Representation Shift
  - Model Steering
  - Debiasing
date: 2026-05-08
content_hash: c4f6bf36e050b596
---

# Analyzing Finetuning Representation Shift for Multimodal LLMs Steering

**Conference**: ICCV 2025
**arXiv**: [2501.03012](https://arxiv.org/abs/2501.03012)
**Code**: [Project Page](https://pegah-kh.github.io/projects/lmm-finetuning-analysis-and-steering/)
**Area**: Multimodal VLM
**Keywords**: MLLM Interpretability, Concept Drift, Representation Shift, Model Steering, Debiasing

## TL;DR

A training-free framework that reveals representation shifts in multimodal large language models (MLLMs) during finetuning through concept-level analysis, and leverages shift vectors for lightweight model behavior steering (debiasing, safety control).

## Background & Motivation

MLLMs achieve strong performance on tasks such as image captioning and visual question answering, yet understanding their internal behavior remains challenging. Most existing work analyzes fully trained models in a post-hoc manner, overlooking the **dynamic changes in latent concepts during finetuning**. For example, after applying "place-focused" finetuning to an image captioning model, concepts originally associated with "people" may silently absorb place-related keywords, while some concepts disappear entirely and new ones emerge.

The authors identify two core issues:

**Finetuning introduces uncontrolled concept drift**: Different concepts are affected to varying degrees—some are refined while others are fundamentally reshaped. These changes may introduce biases or unsafe behaviors.

**Lack of interpretability and control mechanisms**: Existing interpretability methods are mostly designed for unimodal settings (vision or language) and are nearly absent for MLLMs; steering methods have been explored only for text-only LLMs.

The paper therefore aims to: (a) monitor finetuning-induced changes at a human-readable concept level, and (b) leverage discovered shift vectors to steer MLLM behavior at zero additional training cost.

## Method

### Overall Architecture

The framework consists of three components: **concept extraction and comparison** (Section 3.1) → **finetuning concept shift recovery** (Section 3.2) → **model steering applications** (Section 3.3).

### Key Designs

1. **Concept Extraction (K-Means Dictionary Learning)**
   Given a set of images, representations $\bm{Z} \in \mathbb{R}^{D \times M}$ are extracted from the residual stream of a specific MLLM layer and decomposed via K-Means as $\bm{Z} \approx \bm{U}\bm{V}$. Each column $\bm{u}_k$ of $\bm{U}$ constitutes a concept, interpreted in human-readable form via **image grounding** (maximum activation samples) and **text grounding** (mapping through the unembedding matrix to the vocabulary). Inter-concept similarity is measured by **Text Grounding Overlap (T-Overlap)**.

2. **Concept Shift Vectors**
   The representations of the original model $f^a$ and the finetuned model $f^b$ are compared on the same dataset. For each original concept $\bm{u}_k^a$, the associated sample set $\bm{A}_k$ is identified and the shift vector is computed as:
   $$\bm{\Delta}_k^{a \to b}(\bm{u}_k^a) = \frac{1}{|\bm{A}_k|} \sum_{m \in \bm{A}_k} (\bm{b}_m - \bm{a}_m)$$
   The shifted concept is then obtained via $\bm{u}_k^s = \bm{u}_k^a + \alpha \bm{\Delta}_k^{a \to b}$. A key finding is that **higher individual shift consistency correlates with better concept recovery** (statistically significant positive correlation).

3. **Coarse-grained and Fine-grained Model Steering**

   - **Coarse-grained steering**: A steering vector $\bm{s}_c$ is computed as the difference between the mean representations of a target sample set and the original sample set, and added to features at inference time: $\tilde{f_l}(x) = f_l(x) + \alpha \bm{s}_c$.
   - **Fine-grained steering**: After concept decomposition, concept-pair differences $\bm{s}_{ij}^f = \bm{u}_j - \bm{u}_i$ are computed and applied only to samples that activate specific concepts, enabling targeted modification (e.g., steering "yes" answers toward "no").

### Loss & Training

This method **requires no training**. All shift vectors and steering vectors are added directly to residual stream representations at inference time without modifying any model parameters. $\alpha$ defaults to 1. Steering is most effective at deeper layers (especially the last layer), where concepts are approximately linearly separable.

## Key Experimental Results

### Main Results

| Steering Direction | Yes/No Accuracy | Number Accuracy | Other Accuracy | Original Answer Change | Target Answer Change |
|--------------------|----------------|----------------|---------------|----------------------|---------------------|
| None               | 90.82          | 58.47          | 71.10         | 0                    | 0                   |
| Yes→No             | 69.03          | 56.82          | 68.99         | −828                 | +828                |
| 1→3                | 90.71          | 54.52          | 71.12         | −215                 | +144                |
| White→Black        | 90.40          | 58.42          | 58.36         | −98                  | +441                |

On VQAv2, steering substantially alters the target answer counts while **leaving the accuracy and distribution of other answer types largely unchanged**, demonstrating the targeted nature of the approach.

### Ablation Study

| Model      | Total Gender Expressions | Method                  | Gender→Neutral Conversions |
|------------|--------------------------|-------------------------|---------------------------|
| LLaVA-1.5  | 794                      | Coarse-grained steering | 232                       |
| LLaVA-1.5  | 794                      | Fine-grained steering   | **632**                   |
| Idefics2   | 815                      | Coarse-grained steering | 237                       |
| Idefics2   | 815                      | Fine-grained steering   | **315**                   |
| Qwen2-VL   | 926                      | Coarse-grained steering | 134                       |
| Qwen2-VL   | 926                      | Fine-grained steering   | **300**                   |

Fine-grained steering substantially outperforms coarse-grained steering on the gender debiasing task, with consistent effectiveness across all three MLLMs.

### Key Findings

- Finetuning progressively shifts concepts away from their original state (T-Overlap decreases monotonically with training iterations), with substantial variation in the degree of impact across concepts.
- Shift vectors can partially recover post-finetuning concepts ($\alpha = 1$ is generally optimal), and recovery quality correlates positively with individual shift consistency.
- Safety steering: the attack success rate (ASR) of Qwen2-VL is reduced from 45/100 to 5/100, a notably strong result.

## Highlights & Insights

- The linear assumption of **concepts as vectors** is empirically validated in the deep layers of MLLMs, providing a theoretical foundation for lightweight steering.
- Interpretability and controllability are unified within a single framework: first understand (analyze shifts), then intervene (apply shift vectors).
- The method is training-free and plug-and-play, generalizing to multiple MLLMs (LLaVA, Idefics2, Qwen2-VL).

## Limitations & Future Work

- Relies on the linear representation assumption, which may fail for features encoded in a non-linear manner.
- Concept matching uses cosine similarity combined with optimal transport; more sophisticated matching algorithms may improve results.
- As the steering strength $\alpha$ increases, output diversity degrades and may collapse into repetitive tokens.
- Validation is limited to image captioning and VQA scenarios; more complex multimodal interactions (video, dialogue) remain unexplored.

## Related Work & Insights

- Conceptually related to representation finetuning methods such as ReFT, but requires no additional training whatsoever.
- Complementary to the SAE (Sparse Autoencoder) line of work for LLM interpretability: SAEs offer finer granularity at higher computational cost, whereas the proposed method is considerably more lightweight.
- Activation addition / steering vector methods were previously validated only on text-only LLMs; this work extends them to the multimodal setting for the first time.

## Rating

- Novelty: ⭐⭐⭐⭐ (First work to systematically analyze concept shift in MLLM finetuning and enable concept-level steering)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Multiple models, tasks, and scenarios, including debiasing and safety applications)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure with rich figures and tables)
- Value: ⭐⭐⭐⭐ (Zero-cost steering has practical significance for MLLM safety and debiasing)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Hints of Prompt: Enhancing Visual Representation for Multimodal LLMs in Autonomous Driving](hints_of_prompt_enhancing_visual_representation_for_multimodal_llms_in_autonomou.md)
- [\[NeurIPS 2025\] Learning to Steer: Input-dependent Steering for Multimodal LLMs](../../NeurIPS2025/multimodal_vlm/learning_to_steer_input-dependent_steering_for_multimodal_llms.md)
- [\[ICLR 2026\] Steering and Rectifying Latent Representation Manifolds in Frozen Multi-Modal LLMs for Video Anomaly Detection](../../ICLR2026/multimodal_vlm/steering_and_rectifying_latent_representation_manifolds_in_frozen_multi-modal_ll.md)
- [\[ICCV 2025\] Controlling Multimodal LLMs via Reward-guided Decoding](controlling_multimodal_llms_via_rewardguided_decoding.md)
- [\[ICCV 2025\] Enrich and Detect: Video Temporal Grounding with Multimodal LLMs](enrich_and_detect_video_temporal_grounding_with_multimodal_llms.md)

</div>

<!-- RELATED:END -->
