---
title: >-
  [Paper Note] Social Debiasing for Fair Multi-modal LLMs
description: >-
  [ICCV 2025][Causal Inference][Social bias] This paper constructs CMSC, a large-scale counterfactual dataset spanning 18 social concepts…
tags:
  - "ICCV 2025"
  - "Causal Inference"
  - "Social bias"
  - "multimodal large language models"
  - "anti-stereotyping"
  - "fairness"
  - "data debiasing"
date: 2026-05-08
content_hash: 7ba827bceabdf959
---

# Social Debiasing for Fair Multi-modal LLMs

**Conference**: ICCV 2025
**arXiv**: [2408.06569](https://arxiv.org/abs/2408.06569)  
**Code**: Available (project page)  
**Area**: Causal Inference / Fairness
**Keywords**: Social bias, multimodal large language models, anti-stereotyping, fairness, data debiasing

## TL;DR

This paper constructs CMSC, a large-scale counterfactual dataset spanning 18 social concepts, and proposes the Anti-Stereotype Debiasing (ASD) strategy—comprising bias-aware data resampling and a Social Fairness Loss—that effectively reduces social bias across four MLLM architectures with negligible degradation of general multimodal capability.

## Background & Motivation

**Background**: MLLMs (e.g., LLaVA, Qwen-VL, Bunny) have achieved remarkable progress in general vision-language understanding and are widely deployed across downstream tasks. However, these models inherit deep-seated social biases from their training data—for instance, strongly associating "nurse" with "female" or "scientist" with "white male."

**Limitations of Prior Work**: (1) Existing debiasing datasets are either small in scale (VisoGender contains only 690 images) or cover only a single social concept—SocialCounterfactuals provides 171K images but is restricted to the occupation concept—making comprehensive bias reduction infeasible. (2) Methodologically, naive fine-tuning (naive FT) on a balanced dataset has limited effectiveness because it treats all samples equally and cannot selectively correct the model's tendency to underrepresent certain demographic groups.

**Key Challenge**: Neutralizing biased data with a "neutral solution" (balanced dataset + uniform training weights) is insufficient; what is required is an "alkaline solution" (anti-stereotype data + differentiated training weights) to counteract the "acidic" social bias.

**Goal**: (1) Construct a large-scale debiasing dataset covering multiple social concepts; (2) design a training strategy that leverages the opposite of bias to correct bias.

**Key Insight**: Social bias can be quantified via a Skew metric that measures the deviation of each (social attribute, social concept) pair. If a combination (e.g., Female–Nurse) has $\text{Skew} > 0$ (over-predicted), its training weight should be reduced; conversely, if $\text{Skew} < 0$ (under-predicted, e.g., Male–Nurse), its weight should be increased.

**Core Idea**: Construct the CMSC dataset covering 18 social concepts × 3 social attribute types, paired with the ASD strategy—increasing the sampling frequency of under-represented samples at the data level, and rescaling each sample's loss contribution by a factor of $e^{-\text{Skew}}$ at the objective level.

## Method

### Overall Architecture

The overall pipeline consists of two major components: (1) **CMSC dataset construction**—defining 18 social concepts across three categories (personality, responsibility, education) and three social attributes (gender, race, age), and generating 60K high-quality counterfactual images using SDXL with Prompt-to-Prompt control; (2) **ASD training strategy**—during MLLM fine-tuning, first evaluating the model's current bias via Skew, then dynamically adjusting the training process through data resampling and loss scaling.

### Key Designs

1. **CMSC Dataset: Multi-Concept Counterfactual Image Generation**

    - **Function**: Addresses insufficient concept coverage in existing debiasing datasets.
    - **Mechanism**: Eighteen social concepts are defined across three groups—personality (compassionate / belligerent / authority / pleasant / unpleasant), responsibility (tool / weapon / career / family / chef / earning money), and education (middle school / high school / university / good student / bad student / science / arts). Carefully designed prompt templates are created for each concept. A cross-generation strategy is employed: a set of base images covering different gender × age combinations is first generated with a fixed race, and Prompt-to-Prompt is then used to produce race variants while preserving visual consistency. After manual filtering, 60K high-quality images are retained.
    - **Design Motivation**: A single occupation concept cannot capture the rich diversity of real-world stereotypes. CMSC's 18 concepts span multiple stereotype dimensions—from "tendency toward violence" to "educational attainment"—enabling the model to acquire more comprehensive debiasing knowledge.

2. **Dataset Resampling**

    - **Function**: Increases the exposure of under-represented demographic groups at the data level.
    - **Mechanism**: For each sample $\mathcal{P}_i$, its $\text{Skew}(\mathcal{P}_i)$ is computed as the maximum absolute Skew across all social attributes. If $\text{Skew} > 0$ (over-attended combination), the sample is discarded with probability $\text{Skew}/(\text{Skew}+\tau_1)$; if $\text{Skew} \leq 0$ (under-attended combination), it is retained and an over-sampling mechanism based on accumulated Skew is triggered—when $\text{AcmSkew}$ exceeds threshold $\tau_2$, additional copies of the sample are inserted. Training data is thus dynamically adjusted at every epoch.
    - **Design Motivation**: A balanced dataset assigns equal occurrence probability to all samples, yet the model's degree of bias varies across combinations. Intuitively, "male nurse" samples should appear more frequently (since the model tends to overlook them), while "female nurse" samples can appear less often.

3. **Social Fairness Loss (SFLoss)**

    - **Function**: Treats samples with different bias levels differently at the loss level.
    - **Mechanism**: The standard autoregressive loss $\mathcal{L}$ is multiplied by a weighting factor $e^{-\text{Skew}(\mathcal{P}_i)}$, yielding $\mathcal{E}_{fair} = \frac{1}{N}\sum_{i=1}^{N} e^{-\text{Skew}(\mathcal{P}_i)} \mathcal{L}(\mathbf{y}_i, \mathbf{x}_i^{\text{ins}}, \mathbf{x}_i^{\text{img}}; \theta)$. When $\text{Skew} > 0$, the weight is < 1 (down-weighting over-predicted combinations); when $\text{Skew} < 0$, the weight is > 1 (up-weighting under-predicted combinations).
    - **Design Motivation**: The design is inspired by re-weighting strategies in class-imbalance research (e.g., focal loss), but weights are assigned according to bias magnitude rather than class frequency. The exponential form of $e^{-\text{Skew}}$ causes the adjustment magnitude to increase non-linearly with the degree of bias.

### Loss & Training

The base loss is the standard autoregressive cross-entropy used in MLLMs: $\mathcal{L} = -\sum_t \log P(y_t | y_{<t}, x^{\text{ins}}, x^{\text{img}}; \theta)$. ASD augments this with: (1) a data resampling pass executed before each epoch; (2) SFLoss scaling during training. Hyperparameters are set to $\tau_1 = \tau_2 = 1.0$; learning rates vary by model (LLaVA: 5e-7, Qwen-VL: 1e-6, Bunny: 1e-7).

## Key Experimental Results

### Main Results

Models are fine-tuned on SocialCounterfactuals and evaluated in a cross-dataset fashion (3 bias benchmarks + 3 general benchmarks):

| Model | SCounter MinS@C / MaxS@C | FairFace MinS@C / MaxS@C | CMSC MinS@C / MaxS@C | VQAv2 | MMBench |
|---|---|---|---|---|---|
| LLaVA-7B | -2.06 / 0.40 | -2.88 / 0.65 | -1.62 / 1.48 | 78.50 | 58.21 |
| LLaVA-7B+FT | -0.27 / 0.40 | -1.78 / 0.54 | -1.60 / 0.81 | 78.12 | 58.12 |
| **LLaVA-7B+ASD** | **-0.17 / 0.37** | **-0.86 / 0.49** | **-1.50 / 0.73** | 78.18 | 58.36 |
| Qwen-VL-7B | -0.61 / 0.60 | -1.63 / 0.85 | -1.51 / 1.10 | 79.37 | 61.39 |
| **Qwen-VL-7B+ASD** | **-0.26 / 0.43** | **-0.92 / 0.42** | **-0.71 / 0.85** | 79.37 | 60.88 |
| Bunny-8B+ASD | -0.30 / 0.55 | -1.07 / 0.46 | -1.03 / 0.95 | 82.41 | 65.20 |

### Ablation Study

| Configuration | SCounter MinS@C | FairFace MinS@C | CMSC MinS@C |
|---|---|---|---|
| LLaVA-7B (baseline) | -2.06 | -2.88 | -1.62 |
| +FT (naive fine-tuning) | -0.27 | -1.78 | -1.60 |
| +Resample (resampling only) | -0.19 | -0.98 | -1.55 |
| +SFLoss (loss scaling only) | -0.18 | -0.95 | -1.53 |
| **+ASD (full method)** | **-0.17** | **-0.86** | **-1.50** |

### Key Findings

- **ASD consistently outperforms naive FT**: Across all models and datasets, ASD achieves MinSkew@C and MaxSkew@C values closer to 0 (i.e., fairer).
- **Contribution of the CMSC dataset itself**: Fine-tuning on CMSC alone yields better debiasing results than fine-tuning on SocialCounterfactuals (Figure 3), confirming that broader concept coverage is genuinely beneficial.
- **Model scale does not correlate negatively with bias**: LLaVA-13B exhibits stronger bias than LLaVA-7B, possibly because larger models more effectively absorb biases present in the training data.
- **General capability is largely preserved**: ASD induces performance changes of less than 0.5% on VQAv2, TextVQA, and MMBench.
- **Resampling and SFLoss contribute independently and are complementary**: Each component individually outperforms naive FT, and combining both yields the best results.
- **Concept-subset fine-tuning generalizes across concepts**: Fine-tuning on the personality subset partially reduces bias on the responsibility subset as well.

## Highlights & Insights

- **The "alkaline water neutralizes acid" analogy is highly intuitive**: Rather than simply providing balanced data (neutral solution), the method provides anti-stereotype data (alkaline solution). This intuition is elegantly formalized through the Skew-based weighting mechanism.
- **Prompt-to-Prompt ensures visual consistency**: When generating counterfactual images for different racial groups, only the cross-attention maps corresponding to race-related tokens are modified while all others remain unchanged, ensuring that images differ solely in racial features and avoiding the introduction of confounding variables.
- **Practical and architecture-agnostic**: The ASD method is independent of MLLM architecture and can be directly applied to any autoregressive MLLM.

## Limitations & Future Work

- CMSC relies on SDXL-generated images, introducing a distribution gap between synthetic and real images (though FID comparisons indicate CMSC is closer to real data than alternatives).
- Although 18 social concepts represent a substantial improvement over a single occupation concept, they still do not cover all forms of social stereotyping.
- Debiasing effectiveness depends on accurate Skew estimation, which requires re-evaluating model bias before each epoch.
- Learning rates must be carefully tuned per model; excessively high rates improve fairness at the cost of general capability.
- Future directions include: real-time adaptive Skew updates, expanded social attributes (e.g., disability, religion), and integration with RLHF-based safety alignment.

## Related Work & Insights

- **vs. SocialCounterfactuals (CVPR 2024)**: Both adopt counterfactual data for debiasing, but SocialCounterfactuals covers only the occupation concept and does not propose a differentiated training strategy. CMSC covers 18 concepts, and ASD provides a more effective training methodology.
- **vs. POPE (EMNLP 2023)**: POPE is a training-free debiasing method that reduces hallucination/bias by modifying the inference process, but its debiasing effect is experimentally weaker than FT/ASD.
- **vs. contrastive learning debiasing (FairerCLIP, etc.)**: These methods are designed for CLIP-style models and are not applicable to autoregressive MLLMs. ASD is the first debiasing strategy specifically tailored for autoregressive MLLMs.
- **vs. focal loss**: SFLoss is conceptually inspired by focal loss but weights samples by bias magnitude rather than class frequency, making it more appropriate for fairness-oriented scenarios.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The anti-stereotype weighting idea is clear and effective, though the core techniques (resampling + re-weighting) are not entirely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Four MLLM architectures, three bias benchmarks + three general benchmarks, and a comprehensive ablation study.
- **Writing Quality**: ⭐⭐⭐⭐ Problem motivation is clearly articulated, method description is complete, and experimental analysis is thorough.
- **Value**: ⭐⭐⭐⭐ Provides a meaningful contribution to MLLM fairness research; both the dataset and the method are reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A Principle of Targeted Intervention for Multi-Agent Reinforcement Learning](../../NeurIPS2025/causal_inference/a_principle_of_targeted_intervention_for_multi-agent_reinforcement_learning.md)
- [\[ICLR 2026\] On the Eligibility of LLMs for Counterfactual Reasoning: A Decompositional Study](../../ICLR2026/causal_inference/on_the_eligibility_of_llms_for_counterfactual_reasoning_a_decompositional_study.md)
- [\[ICLR 2026\] SelfReflect: Can LLMs Communicate Their Internal Answer Distribution?](../../ICLR2026/causal_inference/selfreflect_can_llms_communicate_their_internal_answer_distribution.md)
- [\[ICCV 2025\] A Visual Leap in CLIP Compositionality Reasoning through Generation of Counterfactual Sets](a_visual_leap_in_clip_compositionality_reasoning_through_gen.md)
- [\[NeurIPS 2025\] Performative Validity of Recourse Explanations](../../NeurIPS2025/causal_inference/performative_validity_of_recourse_explanations.md)

</div>

<!-- RELATED:END -->
