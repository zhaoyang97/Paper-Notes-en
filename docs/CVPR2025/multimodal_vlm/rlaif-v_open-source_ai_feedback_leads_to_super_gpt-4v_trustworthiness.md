---
title: >-
  [Paper Note] RLAIF-V: Open-Source AI Feedback Leads to Super GPT-4V Trustworthiness
description: >-
  [CVPR 2025][Multimodal VLM][AI Feedback] RLAIF-V proposes an alignment framework entirely based on open-source MLLMs. It generates high-quality preference data using a deconfounded candidate response generation strategy and a divide-and-conquer feedback annotation method. When integrated with iterative DPO training and self-feedback inference-time scaling, the framework slashes the hallucination rate of a 7B model by 80.7% and enables a 12B model to surpass the trustworthines…
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "AI Feedback"
  - "Hallucination Suppression"
  - "Preference Learning"
  - "Open-Source Alignment"
  - "Inference-Time Scaling"
date: 2026-05-08
content_hash: de5a9bd4952364c3
---

# RLAIF-V: Open-Source AI Feedback Leads to Super GPT-4V Trustworthiness

**Conference**: CVPR 2025  
**arXiv**: [2405.17220](https://arxiv.org/abs/2405.17220)  
**Code**: [https://github.com/RLHF-V/RLAIF-V](https://github.com/RLHF-V/RLAIF-V)  
**Area**: Multimodal VLM  
**Keywords**: AI Feedback, Hallucination Suppression, Preference Learning, Open-Source Alignment, Inference-Time Scaling

## TL;DR

RLAIF-V proposes an alignment framework entirely based on open-source MLLMs. It generates high-quality preference data using a deconfounded candidate response generation strategy and a divide-and-conquer feedback annotation method. When integrated with iterative DPO training and self-feedback inference-time scaling, the framework slashes the hallucination rate of a 7B model by 80.7% and enables a 12B model to surpass the trustworthiness of GPT-4V utilizing only its own feedback.

## Background & Motivation

MLLMs suffer from a severe hallucination issue where they "confidently generate incorrect content." Human feedback-based RLHF relies on expensive manual annotation and suffers from limited coverage. Meanwhile, existing RLAIF methods (such as Silkie) rely on closed-source models like GPT-4V to provide feedback. This effectively distills closed-source capabilities, rendering the approach unsustainable and bounded by the ceiling of closed-source systems.

Two key challenges persist: (1) **Infeasibility of Feedback Sources**—The community lacks foundational knowledge on how to construct high-quality feedback using open-source models of comparable capability. Simply replacing closed-source annotators with weaker open-source models leads to a steep decline in feedback quality. (2) **Absence of Inference-Time Scaling**—Existing efforts focus on the preference learning phase while ignoring feedback utilization during inference, where blindly increasing the inference compute budget fails to improve performance.

Core Idea of RLAIF-V: Accurate exposure of trustworthiness differences between responses is achieved through "deconfounded sampling," and the complex evaluation of responses is decomposed into simple atomic claim verification tasks via a "divide-and-conquer" approach, enabling open-source models to generate human-grade feedback.

## Method

### Overall Architecture

RLAIF-V consists of four phases: (1) Deconfounded candidate response generation—generating candidate responses by sampling multiple times with different random seeds for the same input; (2) Divide-and-conquer feedback annotation—splitting each response into atomic claims and verifying their trustworthiness one by one using an open-source MLLM; (3) Iterative preference learning—periodically updating the feedback data to perform DPO training; (4) Self-feedback inference-time scaling—using the aligned model itself as a reward function combined with length normalization for Best-of-N selection.

### Key Designs

1. **Deconfounded Response Generation**:
    - Function: Eliminates confounding factors such as text style in preference pairs to accurately expose trustworthiness differences.
    - Mechanism: For the same input $x$ (image + prompt), the model is used to decode $n$ times with the same decoding parameters and different random seeds to generate candidate responses $\{y_1, y_2, \cdots, y_n\}$. Since the responses are sampled from the same distribution, they share similar text styles and linguistic patterns.
    - Design Motivation: In conventional methods (e.g., RLHF-V), $y_w$ and $y_l$ originate from different sources (human annotation vs. model generation), containing many non-robust shallow pattern differences (e.g., writing style, lexical habits), which may cause the model to learn these shortcuts instead of genuine trustworthiness judgments. Deconfounded sampling forces the training to focus on trustworthiness differences at the content level.

2. **Divide-and-Conquer Annotation**:
    - Function: Decomposes the difficult "holistic response quality assessment" into simpler "atomic claim verification" tasks.
    - Mechanism: **Divide**—an LLM is used to split response $y$ into atomic claims $\{c_1, c_2, \cdots, c_m\}$ (excluding opinions and subjective content). **Conquer**—each claim is transformed into a polar question (e.g., "Does the clock in the picture show 11:20?"), enabling the open-source MLLM to output the probabilities $p_{yes}$ and $p_{no}$. **Combine**—the number of rejected claims $n_{rej}$ where $p_{no} > p_{yes}$ is counted, and $-n_{rej}$ is used as the overall score of the response. A higher score indicates fewer errors.
    - Design Motivation: Open-source MLLMs are far more capable of answering simple yes/no questions than evaluating complex holistic responses. By decomposing the task, the performance requirements for the annotator model are significantly reduced. Experiments show that this method achieves a 96.7% human agreement rate, surpassing GPT-4V's VL-Feedback (92.3%).

3. **Self-Feedback for Inference-time Scaling**:
    - Function: Utilizes the DPO-aligned model itself as a reward function to further enhance trustworthiness during inference.
    - Mechanism: The DPO training objective implicitly defines a reward function $r(y) = \beta \log \frac{\pi_\theta(y)}{\pi_{ref}(y)}$. However, direct utilization biases the model towards short responses, so length normalization is introduced: $r(y) = \frac{\beta}{T} \log \frac{\pi_\theta(y)}{\pi_{ref}(y)}$, where $T$ is the response length. Best-of-N selection is performed based on this reward.
    - Design Motivation: The formulation of the DPO objective inherently leads to a preference for short responses, because the total accumulated score of short responses is more likely to be positive. Length normalization eliminates this bias by averaging token-level scores. Experiments show that after normalization, the average response length difference changes from -7.7 words to +3.9 words.

### Loss & Training

- Standard **DPO loss** is used for preference learning with $\beta=0.1$ and a learning rate of $5\times10^{-7}$.
- **Iterative Training**: A total of 4 iterations are performed, with 4 epochs per iteration. In each round, the latest model $M_i$ is used to regenerate candidate responses and annotate preference data $D_i$, addressing the distribution shift challenge in DPO.
- 4k instructions are used to collect feedback in each iteration, covering diverse datasets such as MSCOCO, ShareGPT-4V, MovieNet, VQA v2, etc.
- Total training budget: Data collection for 48h + training for 6h for the 7B model, and data collection for 50h + training for 8h for the 12B model (using 8×A100 80GB).

## Key Experimental Results

### Main Results

| Model | Object HalBench Rsp.↓ | Object HalBench Men.↓ | MHumanEval Rsp.↓ | AMBER Acc. | AMBER F1 |
|------|----------------------|----------------------|------------------|-----------|---------|
| LLaVA 1.5 7B (Baseline) | 54.5 | 27.8 | 67.1 | 73.5 | 77.7 |
| + RLAIF-V 7B | **10.5** (↓80.7%) | **5.2** (↓81.3%) | **44.5** | 76.8 | 84.5 |
| + RLAIF-V 7B BoN | **6.8** | **3.8** | **39.7** | - | - |
| OmniLMM 12B (Baseline) | 19.4 | 10.9 | 52.7 | 86.5 | 89.5 |
| + RLAIF-V 12B (self) | **4.5** (↓76.8%) | **2.9** (↓73.4%) | **35.6** | 88.0 | 90.9 |
| GPT-4V | 13.6 | 7.3 | 45.9 | 83.4 | 87.4 |

### Ablation Study

| Configuration | ObjHal Rsp.↓ | ObjHal Men.↓ | AMBER Acc. | Description |
|------|-------------|-------------|-----------|------|
| RLHF-V (Human annotated feedback) | 28.5 | 12.3 | 76.4 | Human annotations but limited to specific model distributions |
| RLAIF-V | **10.1** | **4.7** | **80.1** | Open-source AI feedback significantly outperforms human annotation |
| RLAIF-V w/o Deconfounding | 25.7 | 11.8 | 73.3 | Deconfounding strategy is crucial |
| RLAIF-V w/o Divide-and-Conquer | - | - | 73.5 | Human agreement rate of only 66.7% vs 96.7% |
| VL-Feedback (GPT-4V feedback) | 37.9 | 21.0 | 72.8 | RLAIF-V open-source feedback outperforms GPT-4V feedback |

### Key Findings

- **Open-source feedback can transcend human annotation and GPT-4V feedback**: Training with feedback data generated by open-source MLLMs in RLAIF-V yields better outcomes than human-annotated feedback in RLHF-V and GPT-4V feedback in Silkie, with the deconfounding strategy being key to boosting learning efficiency.
- **Self-alignment potential**: The 12B model, utilizing only itself as the annotator, significantly outperforms GPT-4V on Object HalBench and MHumanEval after alignment, proving that open-source MLLMs possess self-improvement capability.
- **Generalizability of feedback**: RLAIF-V feedback data collected using OmniLMM can effectively train completely different models like LLaVA 1.5 7B/13B and MiniCPM-V, demonstrating high generalizability.
- **Inference-time scaling effectiveness**: RLAIF-V rewards consistently improve trustworthiness on LLaVA 1.5 and Qwen-VL-Chat, with length normalization effectively resolving the preference for short responses.

## Highlights & Insights

- **Paradigm Shift**: Transitioning from "strong models teaching weak models" to "peer-level models mutually improving each other" or even "self-improvement of models," exerting a profound impact on the field of MLLM alignment.
- **Generality of Divide-and-Conquer**: Decomposing complex evaluations into atomic verifications is not limited to hallucination detection and can be generalized to scenarios such as fact-checking and reasoning verification.
- **Deep Exploitation of Implicit Rewards in DPO**: Most preference learning works only utilize DPO for training, neglecting the fact that the DPO-trained model itself serves as a reward function. RLAIF-V fully taps into this capability.
- **Ingenuity of Deconfounded Sampling**: Sampling from the same model with the same parameters but using different seeds eliminates style discrepancy—the largest confounding factor—which is the key to accelerating feedback learning efficiency.

## Limitations & Future Work

- Iterative training costs remain high (approximately 50 hours of data collection), making rapid iteration challenging.
- The divide-and-conquer strategy relies on the LLM to split atomic claims accurately; split accuracy may decrease in complex reasoning scenarios.
- Best-of-N inference scaling requires sampling multiple candidate responses (16–32), leading to a linear increase in inference cost.
- The paper primarily focuses on hallucination suppression, leaving deep analysis of the impact on logical reasoning, complex question answering, etc., relatively unexplored.

## Related Work & Insights

- **vs RLHF-V**: RLHF-V relies on human-annotated corrective feedback, which is limited in source and expensive. RLAIF-V with AI feedback achieves better results with equivalent data volume (ObjHal 10.5% vs. 12.2%), largely because data can be generated on demand.
- **vs Silkie/VL-Feedback**: Silkie employs GPT-4V to gather feedback, whereas RLAIF-V achieves a higher human consistency rate (96.7% vs 92.3%) and superior training efficacy using open-source models, demonstrating that "closed-source distillation" is not the only path.
- **vs LLaVA-Critic**: LLaVA-Critic trains specialized evaluator models, whereas RLAIF-V directly leverages the implicit reward of DPO training; the two approaches represent complementary solutions for preference signal generation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Deconfounded sampling and divide-and-conquer annotation are highly original designs; self-feedback inference scaling opens up a new direction.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely thorough evaluation across 6 benchmarks, multidimensional ablations (deconfounded/divide-and-conquer/iterative/generalization/inference scaling), and validation across multiple models.
- Writing Quality: ⭐⭐⭐⭐ The methodology description is clear, but tables are dense, and some symbol definitions require careful re-reading.
- Value: ⭐⭐⭐⭐⭐ A completely open-source self-alignment scheme for MLLMs, providing a viable path for the community to break free from dependency on closed-source models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] MetaCaptioner: Towards Generalist Visual Captioning with Open-source Suites](../../ICLR2026/multimodal_vlm/metacaptioner_towards_generalist_visual_captioning_with_open-source_suites.md)
- [\[CVPR 2025\] Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models](molmo_and_pixmo_open_weights_and_open_data_for_state-of-the-art_vision-language_.md)
- [\[CVPR 2026\] Enhancing Part-Level Point Grounding for Any Open-Source MLLMs](../../CVPR2026/multimodal_vlm/enhancing_part-level_point_grounding_for_any_open-source_mllms.md)
- [\[ICML 2026\] VLA-Arena: An Open-Source Framework for Evaluating Vision-Language-Action Models](../../ICML2026/multimodal_vlm/vla-arena_an_open-source_framework_for_benchmarking_vision-language-action_model.md)
- [\[CVPR 2025\] Compositional Caching for Training-free Open-vocabulary Attribute Detection](compositional_caching_for_training-free_open-vocabulary_attribute_detection.md)

</div>

<!-- RELATED:END -->
