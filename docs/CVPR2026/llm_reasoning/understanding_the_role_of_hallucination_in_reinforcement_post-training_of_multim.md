---
title: >-
  [Paper Note] Understanding the Role of Hallucination in Reinforcement Post-Training of Multimodal Reasoning Models
description: >-
  [CVPR 2026][LLM Reasoning][Multimodal Reasoning] This paper proposes the Hallucination-as-Cue analytical framework, systematically investigating the true mechanisms underlying RL post-training of multimodal reasoning mod…
tags:
  - "CVPR 2026"
  - "LLM Reasoning"
  - "Multimodal Reasoning"
  - "Reinforcement Learning Post-Training"
  - "Hallucination Analysis"
  - "GRPO"
  - "Modality Corruption"
date: 2026-05-08
content_hash: 3576da0353d869a4
---

# Understanding the Role of Hallucination in Reinforcement Post-Training of Multimodal Reasoning Models

**Conference**: CVPR 2026
**arXiv**: [2604.03179](https://arxiv.org/abs/2604.03179)
**Code**: None
**Area**: LLM Reasoning
**Keywords**: Multimodal Reasoning, Reinforcement Learning Post-Training, Hallucination Analysis, GRPO, Modality Corruption

## TL;DR

This paper proposes the Hallucination-as-Cue analytical framework, systematically investigating the true mechanisms underlying RL post-training of multimodal reasoning models via three modality-specific corruption strategies (blank image, random image, text removal). The study finds that GRPO training with 100% corrupted visual inputs still yields significant improvements in reasoning performance, challenging the prevailing assumption that RL training effectively leverages visual information.

## Background & Motivation

1. **Background**: Inspired by the success of text-only reasoning LLMs such as DeepSeek-R1, a large body of work has applied RL post-training methods such as GRPO to multimodal LLMs (e.g., Qwen2.5-VL), achieving notable gains on visual mathematical reasoning benchmarks.
2. **Limitations of Prior Work**: Although RL post-training improves benchmark scores, no prior work has systematically investigated whether these gains stem from genuine visual understanding or merely from reinforced text-based reasoning. Current RL rewards are based solely on the correctness of final answers, independent of whether the model correctly utilizes visual information.
3. **Key Challenge**: If RL training primarily reinforces textual reasoning patterns rather than visual perception, resources invested in the current direction may yield diminishing returns—models may be learning to "guess answers" rather than to "reason from images."
4. **Goal**: To design a systematic diagnostic framework that quantitatively answers whether RL post-training genuinely leverages visual information.
5. **Key Insight**: Treating hallucination as a "diagnostic cue" rather than a defect to be eliminated, and deliberately inducing hallucination to expose the true mechanisms of training.
6. **Core Idea**: Three modality-specific corruptions (blank image / random image / text removal) are applied at both training and inference stages, enabling comprehensive analysis of RL training dynamics across eight experimental configurations.

## Method

### Overall Architecture

The Hallucination-as-Cue framework consists of three components: (1) design of modality-specific corruption strategies → (2) hallucination-induced training (GRPO training with corrupted data) → (3) hallucination-induced inference and analysis (cross-evaluation across eight configurations). The framework's purpose is not to train better models, but to diagnose the intrinsic mechanisms of existing RL training methods.

### Key Designs

1. **Blank Image (BI)**

   - **Function**: Completely removes visual information, forcing the model to rely solely on textual reasoning.
   - **Mechanism**: All training/test images are replaced with blank images. During GRPO training, the model must generate reasoning chains conditioned purely on text; correct answers are rewarded.
   - **Design Motivation**: If a model trained with BI still improves, this indicates that RL training can enhance reasoning ability without relying on visual information.

2. **Random Image (RI)**

   - **Function**: Provides incorrect visual information to test whether the model is misled.
   - **Mechanism**: Each training/test image is replaced with a randomly selected different image from the dataset, constructing mismatched image–text training pairs.
   - **Design Motivation**: More challenging than BI—the model not only lacks correct visual information but also faces distracting inputs. If RI training remains effective, it suggests the model learns to ignore visual distractors and rely on textual reasoning.

3. **Textual Removal (TR)**

   - **Function**: Removes textual conditions, forcing the model to rely on visual inputs.
   - **Mechanism**: Rule-based matching removes variable conditions and problem descriptions from questions, retaining only templated instructions and image inputs.
   - **Design Motivation**: Serves as a control—if RL can genuinely exploit visual information, TR training should perform best (since images still contain problem conditions and annotations). However, experiments show that TR training does not significantly outperform BI/RI.

### Loss & Training

Standard GRPO is employed with group-normalized advantages $A_i = \frac{R_i - \mu_{\text{group}}}{\sigma_{\text{group}} + \epsilon}$, a PPO-style clipped surrogate objective, and a KL penalty. Training runs for 15 episodes with rollout size 5, temperature 0.7, KL weight 0.01, and learning rate $1 \times 10^{-6}$. The only difference across conditions lies in whether the input data undergoes modality corruption.

## Key Experimental Results

### Main Results

| Model | Training | MathVision | MathVerse | MathVista | WeMath | AVG |
|---|---|---|---|---|---|---|
| Qwen2.5-VL-3B | Base | 18.19 | 34.82 | 51.40 | 54.48 | 39.72 |
| Qwen2.5-VL-3B | +GRPO | 22.73 | 37.72 | 58.40 | 60.11 | 44.74 |
| Qwen2.5-VL-3B | +GRPO-BI | 20.95 | 35.10 | 56.40 | 56.55 | 42.25 |
| Qwen2.5-VL-3B | +GRPO-RI | 20.86 | 35.76 | 58.00 | 55.17 | 42.45 |
| Qwen2.5-VL-7B | Base | 27.70 | 45.20 | 67.00 | 63.68 | 50.89 |
| Qwen2.5-VL-7B | +GRPO | 28.13 | 47.56 | 70.00 | 68.39 | 53.52 |
| Qwen2.5-VL-7B | +GRPO-BI | 28.39 | 48.86 | 68.50 | 66.84 | 53.15 |
| Qwen2.5-VL-7B | **+GRPO-RI** | **27.27** | **49.90** | **71.40** | **68.33** | **54.23** |

### Ablation Study

| Setting | Training Data | MathVision | MathVerse | MathVista | WeMath | AVG |
|---|---|---|---|---|---|---|
| GRPO | Geometry3K | 22.73 | 37.72 | 58.40 | 60.11 | 44.74 |
| GRPO | MMR1-V0 | 26.18 | 39.26 | 65.00 | 62.47 | 48.23 |
| GRPO | CLEVR | 23.06 | 35.96 | 58.20 | 55.75 | 43.24 |
| GRPO-BI | Geometry3K | 20.95 | 35.10 | 56.40 | 56.55 | 42.25 |
| GRPO-BI | MMR1-V0 | 24.28 | 40.03 | 61.20 | 61.61 | 46.78 |
| GRPO-BI | CLEVR | 21.51 | 35.05 | 58.20 | 54.20 | 42.24 |

### Key Findings

- **Most striking finding**: The 7B model trained with random images (GRPO-RI) achieves an AVG of 54.23%, **surpassing standard GRPO training at 53.52%**. This implies that training with entirely incorrect images can yield superior results.
- **Anomalous BI behavior on MathVision**: The 3B base model's accuracy on MathVision increases from 18.19% to 18.91% (+0.72%) under BI inference, suggesting that visual information may actually impede reasoning in smaller models.
- **Model scale effect**: Larger models benefit more from hallucination-induced trajectories—the gap between GRPO-BI/RI and standard GRPO is substantially smaller for 7B than for 3B.
- **TR does not outperform BI/RI**: Even though TR preserves visual cues within images, its training effectiveness is not meaningfully better than BI, further confirming that current RL training does not effectively leverage visual information.
- **Vision-intensive questions suffer most**: Accuracy on Vision Intensive questions drops 20–26% under BI inference, whereas Text Dominant questions decline by only 4–7%.

## Highlights & Insights

- **Counter-intuitive core finding with strong impact**: The observation that training on incorrect images outperforms training on correct ones is not merely an interesting empirical curiosity—it represents a profound challenge to the entire multimodal RL training paradigm.
- **Hallucination-as-Cue diagnostic methodology is broadly reusable**: The approach of repurposing a "defect" as a "diagnostic signal" is transferable to other settings, e.g., using noisy audio during training to diagnose text dependence in speech models.
- **Systematic 8-configuration cross-evaluation design**: The training × inference × corruption combinatorial matrix provides comprehensive coverage and ensures the reliability of the conclusions drawn.

## Limitations & Future Work

- Only GRPO is studied; other RL algorithms such as PPO and DPO may exhibit different behaviors.
- Experiments are limited to the 3B and 7B scales of Qwen2.5-VL; it remains unclear whether larger models (e.g., 72B) still rely predominantly on textual priors.
- Training data primarily covers visual mathematical reasoning (Geometry3K, MMR1-V0); generalizability to natural image VQA, video reasoning, and other settings has not been established.
- The paper focuses on diagnosis and analysis without proposing concrete improvements to enable RL training to genuinely exploit visual information.
- Future work may explore modality-aware reward function design (e.g., supplementary rewards based on visual grounding quality) to compensate for the limitations of current answer-correctness-only rewards.

## Related Work & Insights

- **vs. DeepSeek-R1 / OpenAI-o1**: The success of these text-only reasoning models already hints that reasoning ability may derive primarily from the language module; this paper validates that hypothesis in the multimodal setting.
- **vs. Ma et al. (2603.27201)**: This concurrent work focuses on hallucination mitigation during inference in MCoT models, whereas the present paper focuses on the role of hallucination during training. The two are complementary—one addresses "how to reduce hallucination at inference time," the other addresses "why hallucination can be beneficial at training time."
- **vs. concurrent work (models retaining accuracy without images)**: That work evaluates models with images removed at inference time; this paper additionally removes images during training, revealing that training-stage effects are more far-reaching.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Reframing hallucination from "a problem to be eliminated" to "a diagnostic instrument" is highly innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ The experimental matrix—2 model scales × 3 corruption types × 3 datasets × 8 configurations × 5 benchmarks—is extremely comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Logic is clear, though some figures and tables are excessively information-dense.
- **Value**: ⭐⭐⭐⭐⭐ The "emperor's new clothes" revelation regarding multimodal RL training carries significant cautionary implications and may substantially influence future research directions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Understanding the Role of Training Data in Test-Time Scaling](../../ICLR2026/llm_reasoning/understanding_the_role_of_training_data_in_test-time_scaling.md)
- [\[CVPR 2026\] Understanding and Mitigating Hallucinations in Multimodal Chain-of-Thought Models](understanding_and_mitigating_hallucinations_in_multimodal_chain-of-thought_model.md)
- [\[CVPR 2026\] Reinforcing Structured Chain-of-Thought for Video Understanding](reinforcing_structured_chain-of-thought_for_video_understanding.md)
- [\[CVPR 2026\] Harnessing Chain-of-Thought Reasoning in Multimodal Large Language Models for Face Anti-Spoofing](harnessing_chain-of-thought_reasoning_in_multimodal_large_language_models_for_fa.md)
- [\[ICLR 2026\] Native Reasoning Models: Training Language Models to Reason on Unverifiable Data](../../ICLR2026/llm_reasoning/native_reasoning_models_training_language_models_to_reason_on_unverifiable_data.md)

</div>

<!-- RELATED:END -->
