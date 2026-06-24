---
title: >-
  [Paper Note] Curriculum Direct Preference Optimization for Diffusion and Consistency Models
description: >-
  [CVPR 2025][Image Generation][DPO] This work introduces curriculum learning into DPO for the first time and adapts DPO to consistency models. By progressively training from "easy-to-distinguish preference pairs" to "hard-to-distinguish preference pairs", it comprehensively outperforms standard DPO and DDPO in text alignment, aesthetics, and human preferences, while requiring only 1/10 of the training data.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "DPO"
  - "curriculum learning"
  - "diffusion models"
  - "consistency model"
  - "preference optimization"
date: 2026-05-08
content_hash: 7959c89ada342740
---

# Curriculum Direct Preference Optimization for Diffusion and Consistency Models

**Conference**: CVPR 2025  
**arXiv**: [2405.13637](https://arxiv.org/abs/2405.13637)  
**Code**: [https://github.com/CroitoruAlin/Curriculum-DPO](https://github.com/CroitoruAlin/Curriculum-DPO)  
**Area**: LLM Alignment/RLHF  
**Keywords**: DPO, curriculum learning, diffusion models, consistency model, preference optimization

## TL;DR
This work introduces curriculum learning into DPO for the first time and adapts DPO to consistency models. By progressively training from "easy-to-distinguish preference pairs" to "hard-to-distinguish preference pairs", it comprehensively outperforms standard DPO and DDPO in text alignment, aesthetics, and human preferences, while requiring only 1/10 of the training data.

## Background & Motivation

**Background**: DPO has been successfully applied to LLM alignment, and Diffusion-DPO extends it to prompt-to-image generation alignment in diffusion models. However, current DPO training treats all preference pairs equally, ignoring the differences in the "difficulty" of preference pairs.

**Limitations of Prior Work**: Standard DPO feeds all preference pairs into training at once. However, some preference pairs show obvious differences (excellent vs. poor), while others show subtle differences (both are moderate but slightly different). Mixing them leads to inefficiency—the model may saturate on simple samples first and fail to effectively learn the subtle preferences of hard samples.

**Key Challenge**: Preference learning needs to extract progressive preference ranking information from rich training signals, but current methods flatten the signals into binary contrasts, losing the hierarchical structure in the ranking. Furthermore, DPO has not yet been extended to consistency models.

**Goal**: (1) How to leverage the difficulty gradient between preference pairs to improve DPO training? (2) How to generalize DPO to consistency models?

**Key Insight**: Drawing inspiration from the curriculum learning concept of humans learning "from easy to hard"—grouping preference pairs by ranking gaps, training on "easy" pairs with large gaps first, and gradually introducing "hard" pairs with small gaps.

**Core Idea**: Generate images using a reward model for ranking, and construct a curriculum hierarchically based on the ranking gap to progressively train DPO from easy to hard. Concurrently, a Consistency-DPO loss function is proposed to adapt to consistency models for the first time.

## Method

### Overall Architecture
A two-stage paradigm: (1) Ranking phase—generate M images for each prompt and rank them using a reward model; (2) Curriculum training phase—divide the preference pairs into B batches based on difficulty, and train chronologically and cumulatively from easy to hard. This applies to both Stable Diffusion and Latent Consistency Models.

### Key Designs

1. **Ranking and Curriculum Division**

    - Function: Rank $M$ generated images by reward and construct preference pairs of varying difficulties according to the ranking gap.
    - Mechanism: Generate $M$ images for prompt $c$ and rank them in descending order using reward model $r_\phi$. Create preference pairs $(x_0^w, x_0^l)$ and divide them into $B$ batches based on the ranking gap: $L_k = (M-1)(B-k)/B$, $R_k = (M-1)(B-(k-1))/B$.
    - Design Motivation: Batch 1 contains the "easiest" pairs with the largest ranking gaps, while batch B contains the "hardest" pairs with the smallest gaps.

2. **Cumulative Training Strategy**

    - Function: Retain all previous simple batches when incorporating a new batch.
    - Mechanism: Stage $k$ uses $P = \bigcup_{i=1}^k S_i$ for training.
    - Design Motivation: Prevent forgetting simple patterns, allowing difficult samples to be learned on top of existing basic knowledge.

3. **Consistency-DPO Loss Function (First of its kind)**

    - Function: Adapt the DPO loss from diffusion models to consistency models.
    - Mechanism: $\mathcal{L}_{\text{Con-DPO}}(\phi) = -\mathbb{E}[\log \sigma(-\beta(d^w - d^l))]$, where $d^*$ is a distance metric based on the consistency function.
    - Design Motivation: Consistency models do not use noise prediction, making it impossible to directly apply the $\epsilon$ loss from Diffusion-DPO.

4. **Diffusion-DPO Loss (Improved Version)**

    - Standard noise prediction loss, using LoRA for efficient fine-tuning.
    - $\beta=5000$ (Diff-DPO) vs. $\beta=200$ (Con-DPO).

### Loss & Training
- AdamW optimizer, learning rate of $3\times10^{-4}$.
- Curriculum batch $B=5$, with $H_i=400$ iterations per batch; 10,000 total iterations.
- LCM: LoRA rank=64, ~2 days on A100 (64GB); SD: LoRA rank=8, ~1 day on A100 (36GB).
- Reward models: Sentence-BERT (text alignment), LAION Aesthetics (aesthetics), HPSv2 (human preference).

## Key Experimental Results

### Main Results

**D1 Dataset — Latent Consistency Model:**

| Task | Baseline | DDPO | DPO | Curriculum DPO |
|------|---------|------|-----|----------------|
| Text Alignment| 0.7243 | 0.7490 | 0.7502 | **0.7548** |
| Aesthetic Score| 6.0490 | 6.3730 | 6.4741 | **6.6417** |
| Human Preference| 0.2912 | 0.2952 | 0.2990 | **0.3237** |

**Human Evaluation (1-5 scale, 11,520 annotations):**

| Setting | Baseline | DDPO | DPO | Curriculum DPO |
|------|---------|------|-----|----------------|
| LCM Text | 2.778 | 2.810 | 2.846 | **3.440** (p<0.005) |
| LCM Aesthetics | 2.718 | 2.765 | 2.782 | **3.006** |
| SD Text | 2.276 | 2.983 | 2.821 | **3.175** |

### Ablation Study

| Hyperparameter | Optimal Value | Description |
|--------|--------|------|
| $\beta$ (Con-DPO) | 200 | Range {50,100,200,300,500} |
| $\beta$ (Diff-DPO) | 5000 | — |
| K (Iterations per batch) | 300-400 | {100,200,300,400,500} |
| B (Curriculum batches) | 5 | {3,5,7}, all values outperform no-curriculum |
| M (Number of images) | 50 | Curriculum DPO with M=50 matches DPO with M=500 |

### Key Findings
- **10x Data Efficiency Improvement**: Curriculum DPO with M=50 achieves comparable performance to standard DPO with M=500.
- **Statistically Significant Human Evaluation**: LCM text alignment scores 3.440 vs. DPO 2.846 (p<0.005).
- Curriculum learning outperforms the no-curriculum baseline across all B values (3/5/7).
- LoRA alone degrades performance; it must be coupled with DPO to be effective.

## Highlights & Insights
- **Curriculum learning is highly intuitive in preference optimization**: Preference pairs naturally have difficulty gradients. Leveraging this structure is a sharp insight.
- **10x data efficiency**: Substantially reduces the computational overhead of reward model evaluations in practical applications.
- **Pioneering Consistency-DPO**: Extending DPO to consistency models unlocks new alignment pathways.
- The curriculum division strategy is generalizable and can be transferred to DPO training in LLMs.

## Limitations & Future Work
- The quality of the reward model directly impacts ranking reliability.
- Only validated on SD v1.5 and LCM; not tested on SDXL/SD3.
- B and K still require manual tuning.
- Online curriculum learning (dynamically adjusting difficulty based on model capacity) remains unexplored.

## Related Work & Insights
- **vs. Diffusion-DPO**: Standard DPO treats all preference pairs equally during training, whereas this study improves performance significantly via hierarchical curriculum learning.
- **vs. DDPO**: DDPO relies on RL-style training, which incurs higher computational overhead.
- **vs. SPO**: Step-wise preference optimization is orthogonal to curriculum learning and may be mutually complementary.

## Rating
- Novelty: ⭐⭐⭐⭐ Both curriculum learning + DPO and Consistency-DPO are valuable firsts.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multiple datasets, multiple models, automatic + human evaluation.
- Writing Quality: ⭐⭐⭐⭐ Clear logic, rigorous experiments.
- Value: ⭐⭐⭐⭐ 10x data efficiency yields strong practical application value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Rethinking Direct Preference Optimization in Diffusion Models](../../NeurIPS2025/image_generation/rethinking_direct_preference_optimization_in_diffusion_models.md)
- [\[CVPR 2025\] Boost Your Human Image Generation Model via Direct Preference Optimization](boost_your_human_image_generation_model_via_direct_preference_optimization.md)
- [\[CVPR 2025\] InPO: Inversion Preference Optimization with Reparametrized DDIM for Efficient Diffusion Model Alignment](inpo_inversion_preference_optimization_diffusion_alignment.md)
- [\[ICLR 2026\] Reinforcing Diffusion Models by Direct Group Preference Optimization](../../ICLR2026/image_generation/reinforcing_diffusion_models_by_direct_group_preference_optimization.md)
- [\[CVPR 2025\] See Further When Clear: Curriculum Consistency Model](see_further_when_clear_curriculum_consistency_model.md)

</div>

<!-- RELATED:END -->
