---
title: >-
  [Paper Note] LLaVA-Critic: Learning to Evaluate Multimodal Models
description: >-
  [CVPR 2025][Multimodal VLM][Multimodal Evaluator] LLaVA-Critic is the first open-source general-purpose multimodal evaluation model. By training on a carefully constructed 113k evaluation instruction dataset, it endows open-source LMMs with pointwise scoring and pairwise ranking capabilities close to the level of GPT-4o. It can also act as a reward model to provide effective preference signals for iterative DPO, surpassing the LLaVA-RLHF reward model trained on human feedback…
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Multimodal Evaluator"
  - "LMM-as-a-Judge"
  - "Preference Learning"
  - "Evaluation Instruction Data"
  - "Reward Signal"
date: 2026-05-08
content_hash: 4bd342ae97377bd8
---

# LLaVA-Critic: Learning to Evaluate Multimodal Models

**Conference**: CVPR 2025  
**arXiv**: [2410.02712](https://arxiv.org/abs/2410.02712)  
**Code**: [https://github.com/LLaVA-VL/LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT)  
**Area**: Multimodal VLM  
**Keywords**: Multimodal Evaluator, LMM-as-a-Judge, Preference Learning, Evaluation Instruction Data, Reward Signal

## TL;DR

LLaVA-Critic is the first open-source general-purpose multimodal evaluation model. By training on a carefully constructed 113k evaluation instruction dataset, it endows open-source LMMs with pointwise scoring and pairwise ranking capabilities close to the level of GPT-4o. It can also act as a reward model to provide effective preference signals for iterative DPO, surpassing the LLaVA-RLHF reward model trained on human feedback.

## Background & Motivation

As LMMs enter the post-training era, the ability of "learning to evaluate" has become crucial: (1) Many evaluation benchmarks rely on GPT-4V/4o as a judge, which is costly and non-customizable; (2) Preference learning (DPO/RLHF) requires reliable reward signals, but collecting human feedback is expensive and difficult to scale; (3) Test-time search (such as Best-of-N) requires an evaluator to select the optimal response.

Key Challenge: Although existing open-source LMMs have made great progress on various visual tasks, they have barely been trained on the discriminative ability of "judging response quality". Direct evaluation using LLaVA-OneVision tends to yield fixed scores (e.g., always outputting "Tie" on WildVision or "Score 6" on MMHal), lacking effective discriminative granularity.

Key Insight: Treat "evaluation" as a trainable instruction-following capability, and teach the model "how to be a good Judge" by constructing high-quality evaluation instruction data. Core Idea: A good evaluator should not only provide scores, but also offer well-reasoned justifications for those scores.

## Method

### Overall Architecture

The construction of LLaVA-Critic consists of two steps: (1) Data collection — constructing 113k evaluation instruction data for both pointwise scoring and pairwise ranking settings; (2) Model training — fine-tuning for 1 epoch on the LLaVA-OneVision pre-trained checkpoint. The trained model can be applied to two scenarios: acting as an evaluator to replace GPT-4o (Scenario 1), and serving as a reward model to provide preference signals (Scenario 2).

### Key Designs

1. **Pointwise Evaluation Data Construction**:
    - Function: Train the model to score a single response and provide justifications according to specific evaluation criteria.
    - Mechanism: Collect instructions from 8 multimodal instruction tuning datasets (covering general dialogue, complex reasoning, OCR, medicine, robotics, etc.), gather responses from 12 off-the-shelf LMMs (from VLFeedback), and use GPT-4o to generate high-quality reference answers. The key innovation is the construction of an **evaluation prompt pool** consisting of evaluation criteria from 7 mainstream evaluation benchmarks, including LLaVA-Bench, LLaVA-Wilder, MMVet, MMHal-Bench, etc. Using GPT-4o as a judge, pointwise scores and justifications are outputted for each (instruction, response, evaluation criteria) combination.
    - Design Motivation: Evaluation criteria vary significantly across different benchmarks (visual chat vs. detailed description vs. hallucination detection). The model needs to learn to understand and follow diverse evaluation prompts. This process finally yields 18,915 image-question pairs and 72,782 pointwise samples.

2. **Pairwise Ranking Data Construction**:
    - Function: Train the model to compare two responses and determine preference relationships.
    - Mechanism: Collect existing preference pair datasets from VLFeedback, LLaVA-RLHF, and RLHF-V. From VLFeedback, 20k pairs were filtered where the GPT-4V three-dimensional score difference was >0.6, along with 5k "Tie" samples to ensure diversity. LLaVA-RLHF and RLHF-V provided 9.4k and 5.7k human-annotated preference pairs, respectively. Thirty diverse evaluation prompt templates were designed, and a template was randomly assigned to each pair. GPT-4o was then used to generate justifications.
    - Design Motivation: Pairwise evaluation is extremely common in practice (Arena ranking, A/B testing) and requires handling ties. The 30 templates ensure the model does not overfit to specific evaluation formats. In total, 40.1k Pairwise samples were obtained.

3. **Iterative DPO Preference Learning (Scenario 2)**:
    - Function: Utilize LLaVA-Critic as a reward model to provide signals for preference learning.
    - Mechanism: For each question-image pair, randomly generate $K=5$ candidate responses using the policy model. Structure all $K\times(K-1)$ ordered pairs, and LLaVA-Critic outputs a relative rating $a_{ij}$ for each pair. Aggregating these relative ratings yields the reward score for each response: $r_i = \sum_{k \neq i} a_{ki} - \sum_{l \neq i} a_{il}$. Select the response with the highest score as $y^+$ and the lowest score as $y^-$ for DPO training. This is iterated for $M=3$ rounds.
    - Design Motivation: Scoring all ordered pairs and aggregating them symmetrically effectively mitigates potential position bias in LLaVA-Critic (where responses appearing first might be preferred). This "round-robin" scoring mechanism is more robust than single comparisons.

### Loss & Training

- LLaVA-Critic Training: Standard cross-entropy loss computed on both scores and justifications. Learning rate of $2\times10^{-6}$, batch size of 32, and trained for 1 epoch.
- Preference Learning: Standard DPO loss with a temperature of 0.7, of top-p 0.9 for sampling candidate responses.
- Complete dataset of 113k (72.8k Pointwise + 40.1k Pairwise); condensed version of 53k (42k + 11k).

## Key Experimental Results

### Main Results: Pointwise Scoring (Pearson-r correlation with GPT-4o)

| Evaluator | ImageDC | MMVet | WildVision | LLaVA-B | LLaVA-W | L-Wilder | MMHal | Avg |
|--------|---------|-------|-----------|---------|---------|----------|-------|-----|
| LLaVA-OV-7B | 0.056 | 0.349 | 0.251 | 0.335 | 0.533 | 0.592 | 0.433 | 0.364 |
| Qwen2-VL-7B | 0.199 | 0.463 | 0.096 | 0.208 | 0.476 | 0.694 | 0.329 | 0.352 |
| LLaVA-Critic-7B | **0.735** | **0.733** | **0.616** | **0.510** | **0.843** | **0.940** | **0.748** | **0.732** |
| LLaVA-OV-72B | 0.718 | 0.680 | 0.446 | 0.436 | 0.716 | 0.824 | 0.620 | 0.634 |
| LLaVA-Critic-72B | **0.802** | **0.723** | **0.705** | **0.524** | **0.782** | **0.951** | **0.790** | **0.754** |

### Main Results: Pairwise Ranking (WildVision Arena alignment with human preferences)

| Evaluator | Acc (w/ Tie) ↑ | Acc (w/o Tie) ↑ | Kendall's τ ↑ |
|--------|-----------|-------------|------------|
| GPT-4o | 0.617 | 0.734 | 0.819 |
| GPT-4V | 0.620 | 0.733 | 0.787 |
| LLaVA-OV-7B | 0.531 | 0.640 | 0.715 |
| LLaVA-Critic-7B | **0.596** | **0.722** | **0.763** |
| LLaVA-Critic-72B | **0.605** | **0.736** | **0.779** |

### Preference Learning Effectiveness

| Base Model | Reward Source | LLaVA-W | L-Wilder | WV-B | Live-B | V-DC | MMHal |
|---------|---------|---------|----------|------|--------|------|-------|
| OV-7B | None (Baseline) | 90.7 | 67.8 | 54.0 | 77.1 | 3.75 | 3.19 |
| OV-7B | LLaVA-RLHF | 97.5 | 70.3 | 64.1 | 83.1 | 3.84 | 4.01 |
| OV-7B | Critic-7B | **100.3** | **71.6** | **67.3** | 84.5 | **3.87** | 3.91 |
| OV-72B | LLaVA-RLHF | 103.2 | 75.2 | 65.2 | 86.2 | 3.85 | 3.67 |
| OV-72B | Critic-72B | **104.4** | **75.9** | **70.0** | **88.5** | **3.86** | **3.77** |

### Ablation Study

| Configuration | Average Pearson-r | Description |
|------|-------------|------|
| LLaVA-Critic-7B (v0.5, 53k data) | 0.712 | Fewer data and domains |
| LLaVA-Critic-7B (113k data) | 0.732 | Data scaling is effective |
| LLaVA-Critic-72B | 0.754 | Model scaling is effective |

### Key Findings

- **7B Critic close to 72B level**: The pointwise scoring gap between LLaVA-Critic-7B (0.732) and LLaVA-Critic-72B (0.754) is extremely small, and both significantly outperform Qwen2-VL-7B (0.352) and LLaMA3.2-V-11B (0.359). This suggests that evaluation capabilities can be efficiently acquired through a small amount of high-quality data.
- **Vastly weak native evaluation capacity of LLaVA-OV**: LLaVA-OV-7B without Critic training achieves an average Pearson-r of only 0.364, giving monotonous and fixed scores, failed to distinguish variations in response quality.
- **Critic reward outperforms human feedback reward**: In preference learning, LLaVA-Critic-7B outperforms LLaVA-RLHF (a reward model trained on human feedback) on 5 out of 6 benchmarks, using only 9.4k prompts.
- **Best-of-N test-time search is effective**: On models already trained with DPO, using Critic-7B for Best-of-5 selection yields additional gains of +1.7 (LLaVA-W) and +3.2 (L-Wilder).
- **Cross-modal generalization**: Training preference alignment solely with image data also brings improvements to the video detailed description (Video-DC) task.

## Highlights & Insights

- **First open-source general-purpose multimodal Judge**: Fills the gap in the open-source community for LMM evaluators. Evaluating once on GPT-4o for iterative DPO costs ~$690, whereas LLaVA-Critic is entirely free.
- **Design Philosophy of the Evaluation Prompt Pool**: Instead of training a "one-size-fits-all evaluation criterion", the model is trained to "understand and follow different evaluation criteria". This allows Critic to adapt to any user-defined evaluation dimensions.
- **Symmetric aggregation to eliminate position bias**: By scoring all ordered pairs and adopting the $r_i = \sum a_{ki} - \sum a_{il}$ design, the position bias issue common in pairwise evaluation is elegantly solved.
- **Dual training of Score + Justification**: Training the model not only to assign scores but also to provide reasons makes the evaluation process transparent and verifiable.

## Limitations & Future Work

- The training data relies on GPT-4o to generate scores and justifications, presenting a distillation bottleneck.
- In pointwise evaluation, scoring criteria are inconsistent across different benchmarks (1-10 vs 1-5). Generalization across benchmarks requires a more unified design.
- The dataset size of 113k is relatively small, which might be insufficient for evaluations involving highly specialized fields (e.g., science, medicine).
- Using evaluation data generated by LLaVA-Critic itself to iteratively train a stronger Critic (self-improving evaluator) was not explored.

## Related Work & Insights

- **vs Prometheus-Vision**: The first VLM evaluator but only supports user-defined scoring criteria; it is not a general-purpose evaluator. LLaVA-Critic covers 7 evaluation scenarios and has much stronger generalizability.
- **vs RLAIF-V (2405.17220)**: RLAIF-V uses a divide-and-conquer strategy to calculate atomic claim-level rewards, while LLaVA-Critic trains a specialized evaluator to provide response-level rewards. In preference learning comparisons on LLaVA-v1.5-7B, LLaVA-Critic achieves performance comparable to RLAIF-V with 33.8k prompts using only 9.4k prompts, showing higher efficiency.
- **vs CriticGPT**: CriticGPT focuses on code evaluation, whereas LLaVA-Critic is a multimodal general-purpose version that can evaluate various tasks such as visual chat, detailed description, and hallucination detection.

## Rating

- Novelty: ⭐⭐⭐⭐ The positioning of the first open-source general-purpose multimodal evaluator is highly valuable, but the core methodology (using GPT-4o to generate evaluation data for fine-tuning) is not brand new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive, covering inside-domain/out-of-domain evaluation, Pointwise/Pairwise setups, comparison in preference learning, test-time search, and data/model scaling ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Clearly organized across two main scenarios (Judge/Preference), with detailed descriptions of the data construction process.
- Value: ⭐⭐⭐⭐⭐ An open-source, free GPT-4o alternative evaluator and preference signal source, offering extremely high practical value to LMM developers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] PhyCritic: Multimodal Critic Models for Physical AI](../../CVPR2026/multimodal_vlm/phycritic_multimodal_critic_models_for_physical_ai.md)
- [\[ICLR 2026\] LLaVA-FA: Learning Fourier Approximation for Compressing Large Multimodal Models](../../ICLR2026/multimodal_vlm/llava-fa_learning_fourier_approximation_for_compressing_large_multimodal_models.md)
- [\[ICCV 2025\] LLaVA-KD: A Framework of Distilling Multimodal Large Language Models](../../ICCV2025/multimodal_vlm/llava-kd_a_framework_of_distilling_multimodal_large_language_models.md)
- [\[ACL 2025\] Can Vision-Language Models Evaluate Handwritten Math?](../../ACL2025/multimodal_vlm/can_vision-language_models_evaluate_handwritten_math.md)
- [\[CVPR 2025\] Debiasing Multimodal Large Language Models via Noise-Aware Preference Optimization](debiasing_multimodal_large_language_models_via_noise-aware_preference_optimizati.md)

</div>

<!-- RELATED:END -->
