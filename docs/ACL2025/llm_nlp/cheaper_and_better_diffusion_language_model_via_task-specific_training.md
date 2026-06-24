---
title: >-
  [Paper Note] Cheaper and Better Diffusion Language Model via Task-Specific Training
description: >-
  [ACL 2025][LLM (Other)][Diffusion Language Models] This paper proposes to optimize diffusion language models through task-specific training strategies, significantly reducing training and inference costs while maintaining generation quality, making diffusion models more practical for text generation tasks.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Diffusion Language Models"
  - "Task-Specific Training"
  - "Text Generation"
  - "Denoising Efficiency"
  - "Inference Acceleration"
date: 2026-05-08
content_hash: 20a4b71a4be7903d
---

# Cheaper and Better Diffusion Language Model via Task-Specific Training

**Conference**: ACL 2025  
**Area**: LLM/NLP  
**Keywords**: Diffusion Language Models, Task-Specific Training, Text Generation, Denoising Efficiency, Inference Acceleration

## TL;DR
This paper proposes to optimize diffusion language models through task-specific training strategies, significantly reducing training and inference costs while maintaining generation quality, making diffusion models more practical for text generation tasks.

## Background & Motivation

**Background**: Diffusion models have achieved immense success in image generation (e.g., DALL-E, Stable Diffusion) and have recently been introduced to the text generation domain. Text diffusion models generate text through an iterative denoising process, offering potential advantages over autoregressive models such as non-autoregressive generation (parallelizable) and better global consistency. Representative works include Diffusion-LM, SSD-LM, and MDLM.

**Limitations of Prior Work**: Current text diffusion models suffer from two core problems: (1) High training costs—training the denoising network requires operating across a massive range of noise levels, leading to slow convergence and high GPU resource consumption; (2) Slow inference—generating a piece of text requires hundreds of iterative denoising steps, which is far slower than the single forward pass of autoregressive models. Consequently, diffusion language models struggle to compete with autoregressive models like GPT in practical applications.

**Key Challenge**: General diffusion training (uniform training across all noise levels) leads to substantial computational waste on noise levels that contribute little to downstream task performance, while truly critical noise intervals do not receive sufficient attention.

**Goal**: To design task-specific training strategies that (1) identify the most critical noise level intervals for the target task, (2) concentrate training resources on these critical intervals, and (3) optimize inference scheduling to reduce denoising steps.

**Key Insight**: Different downstream tasks (summarization, translation, dialogue, etc.) possess varying sensitivities to different noise levels during the diffusion process. Intermediate noise levels are generally the most critical (excessively high noise approximates randomness, while extremely low noise approximates the original text), but the optimal interval varies by task.

**Core Idea**: Directing training resources toward the noise levels that contribute most to the target task via task-sensitive noise scheduling and importance sampling, thereby significantly improving both training efficiency and generation quality.

## Method

### Overall Architecture
The input consists of the embedding representations of the target text, which undergo a forward diffusion process that adds different levels of noise, followed by training a denoising network to reconstruct the original text. The core improvements lie in the noise scheduling during the training phase and the step optimization during the inference phase.

### Key Designs

1. **Task-Aware Noise Importance Sampling**:

    - **Function**: Adaptively allocating sampling probabilities to different noise levels during training, thereby distributing more training resources to the noise intervals most effective for the target task.
    - **Mechanism**: First, pre-train uniformly on all noise levels using a small-scale dataset to evaluate the denoising loss gradient variance $v(t)$ at each noise level $t$ as an informativeness metric. Then, the sampling probability is set as $p(t) \propto v(t)^\alpha$, where $\alpha$ is a temperature parameter. Noise levels with high informativeness (high gradient variance) are sampled more frequently, while those with low informativeness are downsampled.
    - **Design Motivation**: Uniform sampling wastes substantial computation on "easy" noise levels (extremely high and extremely low noise). Focusing on information-rich intermediate intervals allows for more efficient utilization of the training budget.

2. **Task-Specific Denoiser Adaptation**:

    - **Function**: Fine-tuning partial parameters of the denoising network for specific tasks instead of training from scratch.
    - **Mechanism**: Utilizing a pre-trained general diffusion language model as the base and fine-tuning only the layers related to noise conditioning (such as timestep embedding layers and self-attention layers) while freezing the remaining parameters. Fine-tuning is conducted using task-specific data and the aforementioned importance sampling strategy.
    - **Design Motivation**: Training diffusion models from scratch is computationally expensive. Constructing initialization from pre-trained models and fine-tuning only critical components reduces training costs by several orders of magnitude.

3. **Adaptive Inference Step Scheduling**:

    - **Function**: Automatically determining the optimal number of denoising steps during the inference phase to avoid unnecessary iterations.
    - **Mechanism**: Introducing a lightweight "convergence detector" to evaluate the change in text embeddings $\Delta_t = \|x_t - x_{t-1}\|$ after each denoising step. The denoising process is early-stopped when the change falls below a threshold $\epsilon$. The threshold $\epsilon$ is determined on the validation set using a quality-speed trade-off curve.
    - **Design Motivation**: A fixed-step inference schedule is sub-optimal—simple generations might converge in just a few steps, while complex generations require more. Adaptive scheduling can reduce the number of inference steps by 40-60% on average while maintaining text quality.

### Loss & Training
The standard diffusion denoising loss $L = \mathbb{E}_{t \sim p(t)} [\|x_0 - f_\theta(x_t, t)\|^2]$ is used, but the sampling distribution $p(t)$ is determined by the task-aware importance sampling rather than being a uniform distribution.

## Key Experimental Results

### Main Results

| Method | Training GPU Hours ↓ | Inference Steps ↓ | XSum ROUGE-L ↑ | WMT14 BLEU ↑ | CommonGen CIDEr ↑ |
|------|-------------|----------|--------------|-------------|-----------------|
| Diffusion-LM | 480 | 2000 | 28.3 | 22.1 | 112.5 |
| SSD-LM | 320 | 500 | 31.2 | 25.4 | 118.3 |
| MDLM | 256 | 256 | 33.5 | 27.8 | 122.7 |
| **Ours** | **64** | **50-80** | **35.1** | **29.2** | **126.4** |
| GPT-2 (Autoregressive) | 48 | N/A | 34.8 | 28.5 | 124.1 |

### Ablation Study

| Configuration | XSum ROUGE-L ↑ | Training GPU Hours | Description |
|------|--------------|------------|------|
| Full Method | 35.1 | 64 | All components |
| w/o Importance Sampling (Uniform Sampling) | 32.8 | 64 | Quality drops by 2.3 under the same budget |
| w/o Task Adaptation (From Scratch) | 33.2 | 256 | Trained 4x longer but yields lower quality |
| w/o Adaptive Inference (Fixed 256 steps) | 35.0 | 64 | Comparable quality but 3-5x slower inference |
| Importance sampling temperature $\alpha=0.5$ | 34.5 | 64 | Lower temperature, insufficient sampling bias |
| Importance sampling temperature $\alpha=2.0$ | 34.8 | 64 | Overly focused on a small number of noise levels |

### Key Findings
- Task-aware noise importance sampling contributes the most significantly (+2.3 ROUGE-L), proving the effectiveness of focusing on critical noise intervals.
- Compared to training from scratch, task-specific adaptation achieves better results using only 1/4 of the training time, which underscores the vital importance of pre-trained initialization.
- Adaptive inference steps have almost no negative impact on generation quality (only -0.1), while improving inference speed by 3-5 times.
- The final method approaches or even outperforms autoregressive baselines like GPT-2 in text generation quality, while retaining the benefits of parallel generation and better global consistency inherent to diffusion models.

## Highlights & Insights
- The concept of noise importance sampling is simple yet highly effective, revealing the key insight that "not all noise levels are equally important" in diffusion training.
- The design of the adaptive inference step detector is lightweight and practical, drastically accelerating inference at almost zero cost.
- The paradigm shift from "general-purpose diffusion training" to "task-specific diffusion training" can be generalized to other diffusion model applications.

## Limitations & Future Work
- Evaluation is primarily conducted on medium-scale text generation tasks, without testing on long-form text generation (e.g., story generation, paper writing).
- Task-specific training implies that each task requires separate fine-tuning, lacking a unified multi-task scheme.
- A performance gap still remains compared to state-of-the-art autoregressive LLMs (e.g., LLaMA, GPT-4); the quality ceiling of diffusion models for text generation has yet to be pushed further.
- The estimation phase of the importance sampling strategy still incurs some extra computational overhead.

## Related Work & Insights
- **vs Diffusion-LM (Li et al., 2022)**: The first text diffusion model; the proposed method significantly reduces training and inference costs on top of it.
- **vs SSD-LM (Han et al., 2023)**: SSD-LM introduces semi-autoregressive diffusion, whereas this work focuses on optimizing training efficiency, making the two approaches complementary.
- **vs MDLM (Sahoo et al., 2024)**: MDLM employs masked diffusion, while this work uses continuous diffusion but achieves more efficient training.

## Rating
- Novelty: ⭐⭐⭐⭐ Noise importance sampling is a valuable insight, but the overall technical combination is relatively incremental.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-task evaluation and detailed ablation studies, though lacking comparison with the latest large language models.
- Writing Quality: ⭐⭐⭐⭐ The motivation is clearly articulated, and the methodology is comprehensively described.
- Value: ⭐⭐⭐⭐ Plays an important role in driving the practical adoption of diffusion language models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Training Language Model to Critique for Better Refinement](training_language_model_to_critique_for_better_refinement.md)
- [\[ACL 2025\] Training-free LLM Merging for Multi-task Learning](training-free_llm_merging_for_multi-task_learning.md)
- [\[ACL 2025\] Token Prepending: A Training-Free Approach for Eliciting Better Sentence Embeddings from LLMs](token_prepending_training_free.md)
- [\[ACL 2025\] OLMoTrace: Tracing Language Model Outputs Back to Trillions of Training Tokens](olmotrace_tracing_language_model_outputs_back_to_trillions_of_training_tokens.md)
- [\[ACL 2025\] A Survey on Efficient Large Language Model Training: From Data-centric Perspectives](a_survey_on_efficient_large_language.md)

</div>

<!-- RELATED:END -->
