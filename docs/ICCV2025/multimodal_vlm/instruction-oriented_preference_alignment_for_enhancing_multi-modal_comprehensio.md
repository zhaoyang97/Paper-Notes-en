---
title: >-
  [Paper Note] Instruction-Oriented Preference Alignment for Enhancing Multi-Modal Comprehension Capability of MLLMs
description: >-
  [ICCV 2025][Multimodal VLM][Multimodal large language models] This paper proposes the **Instruction-oriented Preference Alignment (IPA)** framework, which anchors alignment signals to **instruction completion efficacy** rather than hallucination factors alone, via an automated preference construction mechanism and a progressive preference data collection pipeline. IPA achieves consistent improvements on Qwen2VL-7B across 9 benchmarks spanning hallucination evaluation, general VQA, and text comprehension.
tags:
  - ICCV 2025
  - Multimodal VLM
  - Multimodal large language models
  - preference alignment
  - instruction following
  - DPO
  - hallucination mitigation
date: 2026-05-08
content_hash: 953bd787eafa06c6
---

# Instruction-Oriented Preference Alignment for Enhancing Multi-Modal Comprehension Capability of MLLMs

**Conference**: ICCV 2025
**arXiv**: 2503.20309
**Code**: [Dataset](https://huggingface.co/datasets/wangzt-kghl/IPA)
**Area**: Multimodal VLM
**Keywords**: Multimodal large language models, preference alignment, instruction following, DPO, hallucination mitigation
**Authors**: Zitian Wang, Yue Liao, Kang Rong, Fengyun Rao, Yibo Yang, Si Liu (Beihang University, NUS, KAUST)

## TL;DR

This paper proposes the **Instruction-oriented Preference Alignment (IPA)** framework, which anchors alignment signals to **instruction completion efficacy** rather than hallucination factors alone, via an automated preference construction mechanism and a progressive preference data collection pipeline. IPA achieves consistent improvements on Qwen2VL-7B across 9 benchmarks spanning hallucination evaluation, general VQA, and text comprehension.

## Background & Motivation

Multimodal large language models (MLLMs) have achieved remarkable progress in vision-language understanding; however, noisy or ambiguous annotations in SFT data can lead models to learn incorrect information. **Preference alignment** has thus emerged as a post-training strategy.

Existing preference alignment methods, however, suffer from fundamental limitations:

**Over-focus on hallucination factors**: Methods such as RLHF-V, RLAIF-V, and Topic-Overwrite primarily construct preference pairs by injecting or detecting hallucinations, restricting improvements to hallucination mitigation.

**Neglect of intrinsic quality dimensions**: When multiple responses show no significant difference in hallucination, these methods lack discriminative capability.

**Dependence on human or commercial models**: Manual annotation is costly, and reliance on GPT-4V incurs substantial financial overhead.

The core question raised by the authors is: **What factors in preference pairs truly determine the direction of alignment?**

The answer is **instruction completion efficacy**—whether a response adequately fulfills the core requirements of the instruction. For example, given the question "Is there a cat in the image?", both "Yes." and "The image's lower-left corner displays a gray cat." are equally free of hallucination, yet the latter better satisfies the instruction by demonstrating observational reasoning and completeness of detail.

## Method

### Overall Architecture

IPA consists of two components: an **automated preference construction mechanism** and a **progressive preference data collection pipeline**.

### Stage 1: Response Sampling

For each sample $s = (V, I, r^{ref})$, two sampling strategies are employed:

**Normal sampling**:
$$r^{norm} \sim \pi_G(\cdot | V, I; \theta_{\pi_G})$$

**Contrastive sampling** (inspired by VCD):
$$r^{cont} \sim \pi_G(\cdot | t(V), I; \theta_{\pi_G}), \quad t \sim \mathcal{T}$$

where $t$ is a perturbation operator randomly selected from the perturbation set $\mathcal{T}$ (noise, blur, scaling, etc.). Contrastive sampling aims to enhance the diversity of deficient response patterns related to the robustness of $\pi_G$.

### Stage 2: Reflective Revision

Responses are refined through two sub-stages:

1. **Diagnostic feedback generation**: The revision model $\pi_R$ identifies critical deficiencies in the initial response:
$$fb \sim \pi_R(\cdot | V, I, r, I^{fb}; \theta_{\pi_R})$$

2. **Feedback-driven refinement**: The response is refined based on the generated feedback:
$$r^{rev} \sim \pi_R(\cdot | V, I, r, fb, I^{rf}; \theta_{\pi_R})$$

Beyond correcting visual hallucinations, the revision stage more importantly **expands the depth of task-relevant information**.

### Stage 3: Instruction-Oriented Verification

A key design component—the verification process **decouples** alignment signal extraction from format constraints:

$$v \sim \pi_V(\cdot | V, I, r, r^{ref}, I^{ver}; \theta_{\pi_V})$$

The verification model evaluates four dimensions:
- **Logical entailment**: Whether the reference answer can be logically derived from the response
- **Detail completeness**: Whether critical details are missing
- **Contradiction detection**: Whether the response contradicts the reference
- **Instruction compliance**: Whether the response follows the instruction

$v = 1$ indicates all conditions are satisfied (preferred); $v = 0$ indicates certain failure patterns are present (dispreferred). Preference pairs $\mathcal{P}^{resp} = \{(r^w, r^l)\}$ are constructed accordingly.

### Progressive Preference Collection Pipeline

**Round 1**: Preference construction is executed on the seed dataset to collect initial preference data. Hard samples for which $\mathcal{R}^{win}$ cannot be generated are retained for $\mathcal{D}_2$.

**Round 2 (Self-Evolution)**: The generator and reviser are optimized via DPO using Round 1 preference data:
$$\pi_G^+ = \text{DPO}(\pi_G, \mathcal{P}_1^{resp}), \quad \pi_R^+ = \text{DPO}(\pi_R, \mathcal{P}_1^{fb})$$
The enhanced models then reprocess the hard samples.

**Round 3 (Reference-Guided)**: For remaining hard samples, the reference response $r^{ref}$ is introduced during the revision stage:
$$fb \sim \pi_R(\cdot | V, I, r, r^{ref}, I^{fb}; \theta_{\pi_R})$$

A total of **89K preference pairs** are collected.

### Training Details

- Backbone model: Qwen2VL-7B
- Alignment method: DPO + LoRA
- Training duration: 1 epoch
- Dataset: Multi-source multimodal samples covering VQA, text comprehension, and open-ended instructions

## Key Experimental Results

### Main Results: Consistent Improvements Across Benchmarks

| Model | HallBench | POPE | MMHal | MMMU | MMStar | MMVet | MME | LLaVA | OCR |
|------|-----------|------|-------|------|--------|-------|-----|-------|-----|
| Qwen2VL-7B | 50.0 | 86.2 | 3.6 | 53.8 | 60.7 | 63.1 | 1676 | 76.6 | 86.2 |
| **+ IPA** | **54.3** | **87.2** | **3.7** | **54.6** | **61.7** | **64.2** | **1687** | **84.0** | **87.3** |
| Gain | +4.3 | +1.0 | +0.1 | +0.8 | +1.0 | +1.1 | +11 | +7.4 | +1.1 |
| Qwen2.5VL-7B | 54.7 | 86.2 | 3.7 | 57.6 | 64.7 | 65.6 | 1694 | 75.2 | 87.5 |
| **+ IPA** | **55.7** | **86.6** | **3.9** | **59.8** | **66.5** | **68.3** | **1707** | **87.3** | **88.2** |

Consistent gains on Qwen2.5VL-7B further demonstrate the **cross-model transferability** of the preference data.

### Comparison with Existing Methods

| Method | HallBench | POPE | MMMU | MMStar | MMVet | MME | LLaVA | OCR |
|------|-----------|------|------|--------|-------|-----|-------|-----|
| RLHF-V | 50.2 | 86.4 | 52.9 | 61.0 | 60.4 | 1682 | 76.6 | 86.3 |
| RLAIF-V | 50.2 | 86.5 | 53.8 | 61.1 | 58.2 | 1674 | 78.8 | 86.4 |
| VLFeedback | 52.2 | 84.7 | 53.7 | 60.7 | 60.6 | 1682 | 81.4 | 86.6 |
| MMPR | 53.4 | 86.4 | 54.3 | 58.5 | 61.7 | 1681 | 83.5 | 86.3 |
| **IPA (Ours)** | **54.3** | **87.2** | **54.6** | **61.7** | **64.2** | **1687** | **84.0** | **87.3** |

- Hallucination-oriented methods (RLHF-V, RLAIF-V, Topic-Overwrite) tend to exhibit performance trade-offs across multi-dimensional evaluations.
- IPA achieves consistent improvements across all dimensions, with particularly notable margins on MMVet (+1.1) and LLaVABench (+7.4).

### Ablation Study

| Configuration | Avg. | HallBench | MMMU | MMStar | MMVet | OCR |
|------|--------|-----------|------|--------|-------|-----|
| Baseline | 68.1 | 50.0 | 53.8 | 60.7 | 63.1 | 86.2 |
| Round 1 w/o CS | 69.1 | 52.3 | 53.4 | 61.2 | 61.7 | 87.0 |
| Round 1 | 70.1 | 52.9 | 54.2 | 61.9 | 62.7 | 87.0 |
| + Round 2 | 70.4 | 53.7 | 54.6 | 61.7 | 63.8 | 86.9 |
| + Round 3 w/ RGS | 69.8 | 53.4 | 54.7 | 61.1 | 62.7 | 87.2 |
| **+ Round 3** | **70.5** | **54.3** | **54.6** | **61.7** | **64.2** | **87.3** |

Key findings:
- **Contrastive sampling** contributes an average gain of 1.0 point, with +2.9 on HallBench.
- **Self-evolution** (Round 2) yields an additional 0.3-point improvement.
- **Reference-guided sampling (RGS)** induces simple imitation behavior, causing a 0.7-point performance drop.
- **Reference-guided feedback** (the full Round 3) effectively recovers hard samples.

## Highlights & Insights

1. **Paradigm shift**: Moving from hallucination-oriented to instruction-oriented preference alignment, anchored to instruction completion capability, achieves comprehensive performance improvement rather than single-dimensional optimization.
2. **Elegant verification design**: Preference judgment is reformulated as a binary verification task, establishing clear decision boundaries by decoupling format variation.
3. **Progressive collection pipeline**: The three-round progressive strategy maximizes the utilization of hard samples, with 89K preference pairs covering diverse multi-source multimodal scenarios.
4. **Cross-model generalization**: Consistent improvements on Qwen2.5VL-7B demonstrate the generality of the preference signals.
5. **Effectiveness of contrastive sampling**: Visual perturbations not only increase negative sample diversity but also expose weaknesses in the model's robustness to multimodal understanding.

## Limitations & Future Work

1. Validation on larger-scale models (e.g., tens of billions of parameters) is absent due to computational resource constraints.
2. The verification model and the aligned model both use Qwen2VL-7B, which may introduce bias.
3. Preference data is primarily constructed based on Qwen2VL, requiring additional filtering when applied to weaker models (e.g., LLaVA-1.5-7B).
4. Online preference optimization approaches (e.g., online DPO or RLHF with PPO) are not explored.

## Related Work & Insights

- **Difference from MMPR**: MMPR focuses on CoT reasoning and formatted answer matching, whereas IPA targets instruction completion capability.
- **Comparison with VLFeedback**: VLFeedback relies on GPT-4V annotation and optimizes for helpfulness, faithfulness, and ethics; IPA is fully automated and centers on instruction compliance.
- **Implications for future work**: The instruction-oriented preference construction paradigm is generalizable to arbitrary multimodal tasks and MLLM architectures.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The instruction-oriented perspective on preference alignment is novel, and the verification mechanism is elegantly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive coverage of 9 benchmarks, thorough ablations, and cross-model validation strengthen the claims.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with well-articulated motivation; Figure 1 provides an intuitive and compelling contrast.
- **Value**: ⭐⭐⭐⭐ — Provides a scalable paradigm for preference data construction with practical applicability to the MLLM community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Guiding Cross-Modal Representations with MLLM Priors via Preference Alignment](../../NeurIPS2025/multimodal_vlm/guiding_cross-modal_representations_with_mllm_priors_via_preference_alignment.md)
- [\[ICCV 2025\] Spatial Preference Rewarding for MLLMs Spatial Understanding](spatial_preference_rewarding_for_mllms_spatial_understanding.md)
- [\[ICCV 2025\] Bidirectional Likelihood Estimation with Multi-Modal Large Language Models for Text-Video Retrieval](bidirectional_likelihood_estimation_with_multi-modal_large_language_models_for_t.md)
- [\[ICCV 2025\] Scaling Inference-Time Search with Vision Value Model for Improved Visual Comprehension](scaling_inference-time_search_with_vision_value_model_for_improved_visual_compre.md)
- [\[ICCV 2025\] Mastering Collaborative Multi-modal Data Selection: A Focus on Informativeness, Uniqueness, and Representativeness](mastering_collaborative_multi-modal_data_selection_a_focus_on_informativeness_un.md)

</div>

<!-- RELATED:END -->
