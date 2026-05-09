---
title: >-
  [Paper Note] Retrv-R1: A Reasoning-Driven MLLM Framework for Universal and Efficient Multimodal Retrieval
description: >-
  [NeurIPS 2025][Multimodal VLM][Multimodal Retrieval] This paper proposes Retrv-R1, the first R1-style reasoning-based multimodal retrieval framework. It reduces token consumption via an Information Compression Module (ICM), preserves complete information for hard candidates through a Details Inspection Mechanism (DIM), and employs a curriculum-based RL reward to balance effectiveness and efficiency, achieving state-of-the-art performance on universal multimodal retrieval benchmarks.
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - Multimodal Retrieval
  - Reinforcement Learning
  - Reasoning MLLM
  - Information Compression
  - DeepSeek-R1
date: 2026-05-08
content_hash: b061e5da4acc31b5
---

# Retrv-R1: A Reasoning-Driven MLLM Framework for Universal and Efficient Multimodal Retrieval

**Conference**: NeurIPS 2025
**arXiv**: [2510.02745](https://arxiv.org/abs/2510.02745)
**Code**: Not available
**Area**: Multimodal VLM
**Keywords**: Multimodal Retrieval, Reinforcement Learning, Reasoning MLLM, Information Compression, DeepSeek-R1

## TL;DR

This paper proposes Retrv-R1, the first R1-style reasoning-based multimodal retrieval framework. It reduces token consumption via an Information Compression Module (ICM), preserves complete information for hard candidates through a Details Inspection Mechanism (DIM), and employs a curriculum-based RL reward to balance effectiveness and efficiency, achieving state-of-the-art performance on universal multimodal retrieval benchmarks.

## Background & Motivation

Universal multimodal retrieval requires a single model to handle diverse retrieval tasks, including text-to-image, image-to-text, and composed query retrieval. Existing approaches fall into two categories: (1) MLLM embedding-based similarity matching, which suffers from limited accuracy and robustness; and (2) retrieval formulated as a QA task (e.g., LamRA), where MLLMs directly generate retrieval results but **lack explicit reasoning processes**, limiting their capability on complex cases.

DeepSeek-R1 has demonstrated the remarkable potential of RL-trained chain-of-thought (CoT) reasoning, motivating a natural question: **Can the R1 paradigm enhance retrieval MLLMs?** However, directly applying GRPO to retrieval tasks faces two major obstacles:

**High computational cost**: Multi-candidate inputs combined with CoT reasoning generate a large number of tokens, potentially exceeding the context length limit.

**Training instability**: Direct RL training is difficult to converge, and models frequently generate erroneous reasoning chains, leading to suboptimal results.

This paper systematically addresses these challenges from two perspectives: model architecture (information compression) and training strategy (a two-stage activation-and-enhancement approach).

## Method

### Overall Architecture

Retrv-R1 adopts a two-stage retrieval pipeline. The first stage uses an embedding model $\phi$ to coarsely retrieve top-K candidates; the second stage employs a reasoning MLLM $\theta$ equipped with an ICM to re-rank and select the best match from the K candidates. The core innovations are concentrated in the architecture design and training strategy of the second-stage model.

### Key Designs

1. **Information Compression Module (ICM)**

   The core idea is to compress the token sequence of each candidate into 2 tokens, freeing up context space for CoT reasoning. The ICM is placed before the language model of the MLLM and generates two types of compressed tokens for each candidate $c_k$:

   - **Content token** $t_{con}^{c_k}$: A learnable embedding $e_{con}$ serves as the query, and the candidate token sequence $T_{c_k}$ serves as key/value. Compression is performed via two-layer attention:
   $t_{con}^{c_k} = \text{ATT}_1(\mathbf{Q}_{e_{con}}, \mathbf{K}_{T_{c_k}}, \mathbf{V}_{T_{c_k}})$

   - **Relation token** $t_{rel}^{c_k}$: Cross-attention is first applied using the candidate tokens to attend over query tokens, modeling a relational feature $R_{q,c_k}$, which is then compressed into a single token:
   $R_{q,c_k} = \text{ATT}_2(\mathbf{Q}_{T_{c_k}}, \mathbf{K}_{T_q}, \mathbf{V}_{T_q})$

   The ICM is pre-trained via a self-alignment strategy: the language model is frozen, and a cross-entropy loss constrains the LM outputs from compressed tokens to match those from the original full token sequences, ensuring that retrieval-critical information is preserved after compression.

2. **Details Inspection Mechanism (DIM)**

   Although compressed tokens suffice for most candidates, hard candidates require complete information. DIM introduces two special tokens, `<inspection-index-start>` and `<inspection-index-end>`, enabling the MLLM to **autonomously determine** during CoT reasoning which candidates warrant closer inspection, and to automatically retrieve their full token sequences as supplementary input. This allows the model to adaptively trade off efficiency against accuracy.

3. **Curriculum Efficiency Constraint**

   Training proceeds in three stages: (a) ICM pre-training via self-alignment; (b) SFT to activate reasoning capability — Qwen2.5-VL-72B is used to synthesize CoT annotation data with a four-phase structure (hypothesize → quick elimination → fine-grained verification → result generation); (c) GRPO-based reinforcement learning to further enhance reasoning.

   The RL reward function consists of a format reward $r_f$ and a result-efficiency reward $r_r$:

   $r_r = \mathbb{1}(\hat{c} = \hat{c}_{gt})(1 - \lambda \frac{N_{ins}}{K})$

   where $N_{ins}$ is the number of candidates for which full tokens are inspected and $K$ is the total number of candidates. The key innovation is a **curriculum schedule** $\lambda_i = i / N_{iter}$: early in training, the model is allowed to freely use complete information, while the efficiency constraint is progressively tightened in later stages.

### Loss & Training

- Pre-training stage: Cross-entropy self-alignment loss
- SFT stage: Cross-entropy loss on synthesized CoT data, with joint training of ICM and MLLM
- RL stage: GRPO objective + format reward + curriculum result-efficiency reward, trained on 10K hard samples
- The visual encoder is frozen throughout; the language model is fine-tuned with LoRA

## Key Experimental Results

### Main Results: M-BEIR Test Set (Average over 16 Sub-tasks)

| Method | Parameters | Avg. Recall |
|--------|------------|-------------|
| CLIP-L | - | 32.5 |
| UniIR-CLIP | - | 50.6 |
| MM-Embed-7B | 7B | 52.7 |
| LamRA-7B | 7B | 63.7 |
| **Retrv-R1-3B** | **3B** | **65.5** |
| **Retrv-R1-7B** | **7B** | **69.2** |

Note: The 3B model surpasses all 7B baselines.

### Efficiency Comparison (CIRR Task, K=50)

| Method | R@5 | Inference Time Ratio | Memory Ratio |
|--------|-----|----------------------|--------------|
| Qwen2.5-VL-7B | 55.1 | 4.79x | 2.44x |
| Vision-R1-7B | 57.7 | 7.23x | 3.28x |
| LamRA-Rank-L-7B | 66.2 | 4.98x | 2.46x |
| **Retrv-R1-7B** | **72.3** | **1.00x** | **1.00x** |

### Ablation Study

| Configuration | CIRR R@5 | OVEN R@5 | Note |
|---------------|----------|----------|------|
| Retrv-R1-3B (full) | 67.9 | 83.3 | Baseline |
| w/o ICM | 68.8 | 84.4 | Slight gain but 7.4x slower |
| w/o content token $t_{con}$ | 60.7 | 77.6 | −7.2 |
| w/o relation token $t_{rel}$ | 64.7 | 80.7 | −3.2 |
| w/o self-alignment pre-training | 64.3 | 80.7 | −3.6 |
| w/o DIM | 62.3 | 78.1 | −5.6 |
| w/o SFT stage | 63.0 | 79.4 | −4.9 |
| w/o RL stage | 61.1 | 78.4 | −6.8 |

### Key Findings

- The curriculum efficiency constraint outperforms any fixed $\lambda$; a fixed $\lambda=0.5$ results in a 1.3% drop in R@5.
- Strong generalization: the model achieves substantial improvements on unseen datasets (CIRCO, GeneCIS, etc.) and unseen task types.
- After fine-tuning on multimodal recommendation (Amazon Review), the model also achieves state-of-the-art performance.

## Highlights & Insights

- **First successful application of the R1 reasoning paradigm to retrieval**: This work demonstrates the value of CoT reasoning for retrieval tasks, going beyond simple embedding matching.
- The ICM design is elegant: dual-path compression via content tokens and relation tokens respectively captures candidate-intrinsic information and query-candidate relational features.
- The DIM endows the model with the ability to **autonomously determine the accuracy–efficiency trade-off**, representing a graceful adaptive design.
- The curriculum RL reward is intuitively well-motivated: the model first learns to be correct, then learns to be efficient.

## Limitations & Future Work

- The framework still depends on the recall quality of the first-stage embedding model; end-to-end optimization could yield further improvements.
- The two-layer attention structure of ICM is simple but may incur excessive information loss in extreme scenarios.
- Synthesizing CoT data requires a 72B teacher model, entailing substantial computational cost.
- Scalability to larger values of K beyond 50 remains to be validated.

## Related Work & Insights

- LamRA first formalized retrieval as a QA task for MLLMs; Retrv-R1 builds upon this by introducing explicit reasoning.
- DeepSeek-R1's GRPO and Vision-R1's exploration of MLLM reasoning serve as direct inspiration for this work.
- The pre-training alignment strategy from BLIP-2 is adapted for ICM self-alignment.
- Insight: R1-style reasoning is generalizable to other information retrieval sub-tasks such as re-ranking and recommendation.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First R1-style retrieval MLLM; ICM and DIM are highly original designs.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Full M-BEIR evaluation + generalization + efficiency + ablation + recommendation task.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with clear motivation.
- **Value**: ⭐⭐⭐⭐⭐ Opens a new reasoning-augmented paradigm for multimodal retrieval.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Video-R1: Reinforcing Video Reasoning in MLLMs](video-r1_reinforcing_video_reasoning_in_mllms.md)
- [\[NeurIPS 2025\] ElasticMM: Efficient MLLM Serving with Elastic Multimodal Parallelism](elasticmm_efficient_multimodal_llms_serving_with_elastic_multimodal_parallelism.md)
- [\[CVPR 2026\] HIVE: Query, Hypothesize, Verify — An LLM Framework for Multimodal Reasoning-Intensive Retrieval](../../CVPR2026/multimodal_vlm/hive_query_hypothesize_verify_an_llm_framework_for_multimodal_reasoning-intensiv.md)
- [\[ICLR 2026\] Shuffle-R1: Efficient RL Framework for Multimodal Large Language Models via Data-centric Dynamic Shuffle](../../ICLR2026/multimodal_vlm/shuffle-r1_efficient_rl_framework_for_multimodal_large_language_models_via_data-.md)
- [\[NeurIPS 2025\] Struct2D: A Perception-Guided Framework for Spatial Reasoning in MLLMs](struct2d_a_perception-guided_framework_for_spatial_reasoning_in_mllms.md)

</div>

<!-- RELATED:END -->
