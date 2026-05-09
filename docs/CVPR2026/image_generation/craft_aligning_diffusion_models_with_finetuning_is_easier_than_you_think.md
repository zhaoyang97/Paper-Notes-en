---
title: >-
  [Paper Note] CRAFT: Aligning Diffusion Models with Fine-Tuning Is Easier Than You Think
description: >-
  [CVPR 2026][Image Generation][Diffusion Model Alignment] CRAFT proposes an ultra-lightweight alignment method for diffusion models: it automatically constructs high-quality training sets through a Compositional Reward Filtering (CRF) strategy and then performs an enhanced version of SFT. Theoretically, CRAFT optimizes the lower bound of Group Relative Policy Optimization (GRPO). It outperforms SOTA methods requiring thousands of preference pairs using only 100 samples, with training speeds 11-220 times faster.
tags:
  - CVPR 2026
  - Image Generation
  - Diffusion Model Alignment
  - Human Preference
  - Compositional Reward Filtering
  - Supervised Fine-Tuning
  - Data Efficiency
date: 2026-05-08
content_hash: 555657b44102fb1b
---

# CRAFT: Aligning Diffusion Models with Fine-Tuning Is Easier Than You Think

**Conference**: CVPR 2026  
**arXiv**: [2603.18991](https://arxiv.org/abs/2603.18991)  
**Code**: None  
**Area**: Image Generation / Diffusion Model Alignment  
**Keywords**: Diffusion Model Alignment, Human Preference, Compositional Reward Filtering, Supervised Fine-Tuning, Data Efficiency  

## TL;DR

CRAFT proposes an ultra-lightweight alignment method for diffusion models: it automatically constructs high-quality training sets through a Compositional Reward Filtering (CRF) strategy and then performs an enhanced version of SFT. Theoretically, CRAFT optimizes the lower bound of Group Relative Policy Optimization (GRPO). It outperforms SOTA methods requiring thousands of preference pairs using only 100 samples, with training speeds 11-220 times faster.

## Background & Motivation

1.  **Background**: Post-training alignment for diffusion models follows three main paths: SFT (requires high-quality data), DPO-style preference optimization (requires large-scale preference pairs), and online RL methods (computationally expensive).
2.  **Limitations of Prior Work**: SFT relies on high-quality images that are difficult to obtain; DPO methods like Diff-DPO rely on large-scale preference datasets with inconsistent quality; online methods like SPO require repeated sampling and evaluation, making them extremely expensive computationally.
3.  **Key Challenge**: The dual challenge of data efficiency and computational efficiency—existing methods either require massive amounts of data or massive computation, making it difficult to achieve both.
4.  **Goal**: Design a fine-tuning method that is both data-efficient and computationally lightweight.
5.  **Key Insight**: Eliminating the need for external high-quality data or preference pairs by having the model generate candidate images itself and filtering the best samples through multi-dimensional rewards.
6.  **Core Idea**: Compositional multi-reward model data filtering combined with advantage-weighted SFT, which is theoretically equivalent to optimizing the lower bound of grouped RL.

## Method

### Overall Architecture

CRAFT consists of three stages: (1) Data Construction: Sample 10,000 prompts from the HPD dataset, expand them into variants using Qwen-Plus, and generate candidate images using the base model; (2) Compositional Reward Filtering: Use multiple reward models to jointly filter high-quality samples; (3) Weighted SFT Fine-tuning: Calculate weights using grouped advantages and compute loss only for samples that pass the filter.

### Key Designs

1.  **Compositional Reward Filtering (CRF)**:
    -   **Function**: Automatically filtering high-quality training data from model-generated candidate images.
    -   **Mechanism**: Uses three complementary reward models—HPSv2.1 (human preference), PickScore (selection preference), and AES (aesthetic score). A multi-level filtering strategy is designed: Single-Reward Filtering $\mathcal{I}_\xi$ (keep if any reward increases), Dual-Reward Filtering $\mathcal{I}_{ha}$ (two must increase), and Triple Filtering $\mathcal{I}_{hpa}$ (all three must increase, most strict). For each original prompt, a batch of samples is kept if its expanded version produces an image superior to the original across all reward metrics.
    -   **Design Motivation**: Automated data curation avoids reliance on external high-quality datasets or strong model distillation; combining multi-dimensional rewards ensures data consistency.

2.  **Grouped Advantage-Weighted SFT**:
    -   **Function**: Adaptively adjusting gradient contributions based on sample quality.
    -   **Mechanism**: Calculates normalized advantages for each group of samples $\hat{A}^{(i,j)} = (r^{(i,j)}_{\text{total}} - \text{mean}) / (\text{std} + \epsilon)$, then weights the standard SFT loss $\|\epsilon_\theta(x^{(i,j)}_t, t, c) - \epsilon^{(i,j)}_t\|^2$ by the advantage value. An indicator function is used to compute gradients only for samples that pass the filter.
    -   **Design Motivation**: Better quality samples receive larger gradient contributions while poor samples are suppressed, achieving implicit reward guidance.

3.  **Theoretical Guarantee (Theorem 3.1)**:
    -   **Function**: Establishing a theoretical link between SFT and Reinforcement Learning (RL).
    -   **Mechanism**: Under the small learning rate assumption, it proves that the CRAFT loss actually optimizes the lower bound of the grouped RL objective $\hat{J}(\theta)$. Specifically, an exact mathematical relationship exists between the advantage-weighted SFT loss and the RL objective.
    -   **Design Motivation**: Providing a theoretical foundation for the idea that "SFT with selective data can achieve RL-level alignment," moving beyond purely empirical approaches.

### Loss & Training

The loss function is an advantage-weighted noise prediction MSE loss. Full-parameter fine-tuning is performed on the UNet using the AdamW optimizer. SD1.5 is trained for 120 steps, and SDXL is trained for 200 steps with a batch size of 128 and a learning rate of 5e-5. Total training takes approximately 4 GPU hours (SDXL on H100).

## Key Experimental Results

### Main Results

| Benchmark/Metric | SDXL Baseline | Diff-DPO | SPO | CRAFT | Gain vs SPO |
| :--- | :--- | :--- | :--- | :--- | :--- |
| HPDv2 HPSv2.1↑ | 27.93 | 29.76 | 32.32 | **32.67** | +0.35 |
| HPDv2 ImgReward↑ | 0.819 | 1.037 | 1.103 | **1.312** | +0.209 |
| HPDv2 MPS↑ | 14.35 | 14.70 | 15.36 | **15.62** | +0.26 |
| Parti HPS↑ | 27.32 | 28.74 | 30.54 | **31.10** | +0.56 |

CRAFT leads across all metrics and datasets; notably, ImageReward and MPS were not used during training, proving its generalization ability.

### Ablation Study

| Configuration | HPSv2.1 | Training Size | GPU Time |
| :--- | :--- | :--- | :--- |
| CRAFT ($\mathcal{I}_{hpa}$) | 32.67 | 100 | ~4h |
| CRAFT ($\mathcal{I}_{ha}$) | 32.45 | ~300 | ~4h |
| CRAFT ($\mathcal{I}_h$) | 32.12 | ~1000 | ~4h |
| Unfiltered SFT | 31.80 | 10000 | ~4h |

### Key Findings

-   The strictest triple filtering $\mathcal{I}_{hpa}$ performs best, indicating that data quality is far more important than quantity.
-   CRAFT surpasses SPO (which requires 4,000 samples) using only 100 samples, a 40x improvement in data efficiency.
-   Training speed is 19.7x faster than SPO (SDXL) and 60.1x faster than SmPO.
-   Performs excellently on the GenEval compositional reasoning benchmark, showing alignment capability transfers to instruction following.
-   Leads on un-trained reward metrics, showing it is not over-fitting to the training rewards.

## Highlights & Insights

-   **Extreme Data Efficiency**: 100 samples outperform methods using thousands of preference pairs, overturning the assumption that "alignment requires massive preference data."
-   **Self-Curated Data Pipeline**: No external data required; the model generates, filters, and trains on its own data, making it completely self-contained.
-   **Theoretical Elegance**: proving that selective SFT is equivalent to RL lower bound optimization bridges the theoretical gap between the two paradigms.
-   **Immediate Practical Value**: Aligning SDXL in just 4 GPU hours significantly lowers the barrier for post-training diffusion models.

## Limitations & Future Work

-   Depends on the quality of reward models; biases in reward models may propagate to the fine-tuned model.
-   Only validated on SD1.5 and SDXL; has not been tested on newer architectures (e.g., DiT/FLUX).
-   Theoretical proof assumes a small learning rate; it may not hold under large learning rates.
-   Future applications could be explored in video diffusion models or 3D generation.

## Related Work & Insights

-   **vs Diff-DPO**: DPO requires many preference pairs and is inefficient; CRAFT achieves better results with SFT.
-   **vs SPO**: SPO requires online sampling and evaluation; CRAFT is completely offline and 20x faster.
-   **vs RLHF/GRPO**: CRAFT is theoretically proven equivalent to RL but much simpler to implement.

## Rating

-   Novelty: ⭐⭐⭐⭐ Compositional reward filtering is novel; theoretical link is valuable.
-   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive comparison across multiple benchmarks, metrics, and baselines.
-   Writing Quality: ⭐⭐⭐⭐ Clear structure with a good balance of theory and experiments.
-   Value: ⭐⭐⭐⭐⭐ Extremely high practical value, significantly reducing the cost of diffusion model alignment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Aligning Text to Image in Diffusion Models is Easier Than You Think](../../NeurIPS2025/image_generation/aligning_text_to_image_in_diffusion_models_is_easier_than_you_think.md)
- [\[CVPR 2026\] Memory-Efficient Fine-Tuning Diffusion Transformers via Dynamic Patch Sampling and Block Skipping](memory-efficient_fine-tuning_diffusion_transformers_via_dynamic_patch_sampling_a.md)
- [\[CVPR 2026\] LeapAlign: Post-Training Flow Matching Models at Any Generation Step by Building Two-Step Trajectories](leapalign_post_training_flow_matching_models_at_any_generation_step.md)
- [\[CVPR 2026\] RewardFlow: Generate Images by Optimizing What You Reward](rewardflow_generate_images_by_optimizing_what_you_reward.md)
- [\[CVPR 2026\] Low-Resolution Editing is All You Need for High-Resolution Editing](low-resolution_editing_is_all_you_need_for_high-resolution_editing.md)

</div>

<!-- RELATED:END -->
