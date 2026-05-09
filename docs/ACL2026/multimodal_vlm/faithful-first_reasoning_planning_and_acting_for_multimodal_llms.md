---
title: >-
  [Paper Note] Faithful-First Reasoning, Planning, and Acting for Multimodal LLMs
description: >-
  [ACL 2026][Multimodal VLM][Perceptual faithfulness] This paper proposes the Faithful-First RPA framework, which employs the FaithEvi pipeline to evaluate perceptual faithfulness at each reasoning step (i.e., whether claimed objects genuinely exist in the image), and the FaithAct mechanism to enforce evidence-grounded planning and acting during reasoning generation. The framework improves perceptual faithfulness by up to 24% without degrading task accuracy.
tags:
  - ACL 2026
  - Multimodal VLM
  - Perceptual faithfulness
  - reasoning planning and acting
  - multimodal hallucination
  - visual evidence verification
  - step-by-step reasoning
date: 2026-05-08
content_hash: b41410b95bf21005
---

# Faithful-First Reasoning, Planning, and Acting for Multimodal LLMs

**Conference**: ACL 2026
**arXiv**: [2511.08409](https://arxiv.org/abs/2511.08409)
**Code**: [GitHub](https://github.com/lijunxian111/Faithful-First-RPA)
**Area**: Multimodal VLM / Reasoning Faithfulness
**Keywords**: Perceptual faithfulness, reasoning planning and acting, multimodal hallucination, visual evidence verification, step-by-step reasoning

## TL;DR

This paper proposes the Faithful-First RPA framework, which employs the FaithEvi pipeline to evaluate perceptual faithfulness at each reasoning step (i.e., whether claimed objects genuinely exist in the image), and the FaithAct mechanism to enforce evidence-grounded planning and acting during reasoning generation. The framework improves perceptual faithfulness by up to 24% without degrading task accuracy.

## Background & Motivation

**Background**: Multimodal large language models (MLLMs) have achieved remarkable progress on tasks such as VQA and visual reasoning; however, their reasoning traces frequently exhibit "unfaithfulness"—generated explanations are inconsistent with visual evidence or post-hoc rationalizations of predicted answers.

**Limitations of Prior Work**: (1) Existing work primarily focuses on behavioral faithfulness (whether the reasoning chain reflects the model's decision process), neglecting perceptual faithfulness (whether reasoning steps are grounded in verifiable visual input). (2) Reasoning frameworks such as CoT and ReAct do not validate the perceptual grounding of intermediate steps. (3) Models may produce correct answers while relying on erroneous visual descriptions (e.g., describing a black bicycle as yellow).

**Key Challenge**: Existing reasoning frameworks adopt a "generate-then-verify" paradigm, in which perceptual errors are only discovered after the reasoning chain has been fully generated, making correction costly and of limited effectiveness. Faithfulness should be a design principle rather than a post-hoc evaluation metric.

**Goal**: To establish a unified framework that both quantitatively assesses the perceptual faithfulness of reasoning chains and actively enforces evidence verification during the reasoning process.

**Key Insight**: Grounded in the principle that "a perceptually faithful model only reasons about visually observable content," the reasoning process is formalized as a faithfulness-constrained planning problem.

**Core Idea**: At each reasoning step, claimed objects are first extracted, their existence is verified via preference voting and visual grounding, a faithfulness score is computed, and steps that fail to meet the threshold must be revised before being admitted into the reasoning chain.

## Method

### Overall Architecture

Faithful-First RPA consists of two core components: (1) the **FaithEvi** evaluation pipeline, which assesses perceptual faithfulness at both the step level and chain level; and (2) the **FaithAct** planning and acting mechanism, which dynamically verifies and corrects each step during reasoning generation using signals from FaithEvi. The overall pipeline is: image + question → MLLM generates a reasoning step → FaithEvi evaluates its faithfulness → if the threshold is not met, FaithAct triggers revision → upon passing, the step is admitted into the reasoning chain → the next step proceeds.

### Key Designs

1. **FaithEvi: Perceptual Faithfulness Evaluation Pipeline**

    - **Function**: Quantifies the degree of visual evidence support for each step in the reasoning chain.
    - **Mechanism**: Three stages. **Stage 1: Claimed Object Extraction**—Qwen2.5-7B-Instruct is used to extract the set of claimed objects $O_t = \{O_t^1, \dots, O_t^{m_t}\}$ from each reasoning step. **Stage 2: Preference Voting + Visual Grounding**—(a) A frozen CLIP-ViT-Large encodes the image and object text; a two-layer MLP (trained on the POPE dataset) predicts object existence probability $c_p$; (b) a frozen GroundingDINO localizes objects and yields detection confidence $c_g$. **Stage 3: Faithfulness Scoring**—the two confidences are fused as $c_t^i = 0.7 \cdot c_p + 0.3 \cdot c_g$, mapped to a three-level discrete score (<0.4→0, 0.4–0.6→$c_t^i$, >0.6→1); the step-level score is $F_{\text{step},t} = \frac{1}{m_t}\sum f_t^i$ and the chain-level score is $F_{\text{chain}} = \frac{1}{n}\sum F_{\text{step},t}$.
    - **Design Motivation**: Preference voting provides global existence verification (detector confidence is unreliable under weak visual cues), while grounding provides region-level spatial evidence; the two modalities are complementary.

2. **FaithAct: Faithfulness-First Planning and Acting**

    - **Function**: Reformulates the reasoning process as a faithfulness-constrained planning problem.
    - **Mechanism**: The planning objective is $S^* = \arg\max F_{\text{step}}(s_t)$ s.t. $\forall t, F_{\text{step}}(s_t) \geq c$. Each step is immediately verified by FaithEvi upon generation; steps that fail the threshold are returned to the MLLM for regeneration with updated evidence (object existence labels, bounding boxes, counts). The framework exposes an extensible function interface: `Poll()` (existence probability), `Ground()` (bounding box detection), `Select()` (confirmed present), `Abstain()` (confirmed absent), and `Count()` (counting reasoning).
    - **Design Motivation**: Unlike the "generate-then-verify" paradigm, FaithAct adopts a "verify-while-generate" principle, correcting perceptual errors early in the reasoning chain to prevent error propagation.

3. **Action-Guided Reasoning Correction**

    - **Function**: Repairs reasoning steps that fail the faithfulness threshold.
    - **Mechanism**: Steps that fail verification are not discarded outright; instead, they are regenerated with updated evidence. The correction prompt guides the model to revise perceptual descriptions while maintaining logical continuity.
    - **Design Motivation**: Faithfulness gains from FaithAct are especially pronounced in later reasoning steps—consistent with prior findings that longer CoT chains are more susceptible to noise in later stages.

### Loss & Training

This paper presents an inference-time framework and involves no model training. The preference voting head (a two-layer MLP) is trained on the POPE dataset; GroundingDINO and CLIP are both used in a frozen state. GroundingDINO is configured with a box threshold of 0.35 and a text threshold of 0.25.

## Key Experimental Results

### Main Results

**Perceptual Faithfulness Evaluation ($F_{\text{chain}}$, %)**

| Model + Method | LLaVA-bench | RealWorldQA | POPE | MMHal | Avg. |
|---|---|---|---|---|---|
| Qwen + CoT | 46.05 | 48.11 | 45.21 | 53.34 | 48.18 |
| Qwen + ReAct | 54.82 | 56.82 | 45.02 | 33.76 | 47.61 |
| **Qwen + FaithAct** | **55.10** | **57.22** | **56.87** | **66.45** | **58.91** |
| InternVL + CoT | 45.63 | 44.23 | 43.25 | 53.17 | 46.57 |
| **InternVL + FaithAct** | **52.64** | **57.35** | **56.01** | **61.71** | **56.93** |
| LLaVA + CoT | 47.56 | 52.31 | 52.28 | 30.63 | 45.70 |
| **LLaVA + FaithAct** | **52.82** | **58.11** | **56.09** | **39.91** | **51.73** |

**Task Accuracy Preservation**

| Model | Method | RealWorldQA (%) | MMHal (rating) |
|---|---|---|---|
| Qwen | CoT | 70.1 | 3.40 |
| Qwen | FaithAct | **74.5** | **3.48** |
| InternVL | CoT | 70.8 | 3.61 |
| InternVL | FaithAct | 71.2 | 3.58 |

### Ablation Study

**Core Component Ablation (Qwen, RealWorldQA / MMHal)**

| Configuration | RealWorldQA (%) | MMHal (%) |
|---|---|---|
| FaithAct (full) | 57.22 | 66.45 |
| w/o Poll() | 54.24 (−3.0) | 63.25 (−3.2) |
| w/o Ground() | 53.16 (−4.1) | 62.47 (−4.0) |

### Key Findings

- FaithAct achieves an average perceptual faithfulness of 55.86%, outperforming the strongest baseline ReAct (48.10%) by 7.76 percentage points.
- The largest gains are observed on MMHal, a hallucination-sensitive benchmark: +21.99% over CoT on average and +9.81% over tool-augmented methods on average.
- Faithfulness improvements do not compromise task accuracy—Qwen even improves from 70.1% to 74.5% on RealWorldQA.
- The contribution of `Ground()` slightly exceeds that of `Poll()`, suggesting that spatial localization provides more critical visual evidence.
- Replacing GroundingDINO with SAM3 leads to an approximately 5% performance drop, indicating that the framework requires a localization-specialized model.
- FaithAct's gains are more pronounced in later reasoning steps, corroborating the hypothesis that hallucinations are more likely to occur in later stages.
- Manual verification of LLM object extraction yields 99.42% accuracy (7,550 object-level labels) with a segment validity of 0.968.
- Inference time increases by approximately 2–3× (FaithAct: 14–19 s vs. CoT: 3–11 s).

## Highlights & Insights

- The principle that "faithfulness should be a design principle rather than a post-hoc metric" is compelling—embedding faithfulness constraints into the reasoning loop ensures that every step is evidence-grounded.
- The distinction between perceptual faithfulness and behavioral faithfulness has theoretical value: a model can "answer correctly for wrong reasons" (behaviorally faithful but perceptually unfaithful), or "have correct reasoning but a wrong answer" (perceptually faithful but behaviorally unfaithful).
- The extensible function interface (Poll/Ground/Select/Abstain/Count) makes the framework readily generalizable to attribute and relation verification.

## Limitations & Future Work

- Faithfulness verification is currently limited to object existence; attribute-level (color, size) and relation-level (spatial relationships, actions) verification are not addressed.
- Inference time increases by approximately 2–3×.
- Behavioral faithfulness is not directly evaluated; it is only assumed that perceptual faithfulness promotes behavioral consistency.
- Gains are less pronounced on benchmarks with weaker perceptual demands (e.g., MathVista).

## Related Work & Insights

- **vs. Grounded-CoT (Wu et al., 2025)**: The latter appends grounding information after reasoning, whereas the present work performs real-time verification during reasoning. FaithAct outperforms Grounded-CoT in 11 out of 12 settings.
- **vs. ReAct (Yao et al., 2022)**: ReAct permits tool invocation but does not enforce faithfulness constraints; this work demonstrates that $F_{\text{chain}}$ of ReAct is theoretically upper-bounded by FaithAct.
- **vs. VAT (Liu et al., 2025)**: Visual Abstract Thinking degrades severely on POPE (21.46%), suggesting that abstraction may exacerbate perceptual disconnection.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The formal definition of perceptual faithfulness and the "verify-while-generate" paradigm are original contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three models, four benchmarks, comprehensive ablation studies, and manual verification.
- **Writing Quality**: ⭐⭐⭐⭐ The distinction between perceptual and behavioral faithfulness is clearly articulated; the framework design logic is rigorous.
- **Value**: ⭐⭐⭐⭐ Provides a practical framework for trustworthy multimodal reasoning with an extensible function interface.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Tree-of-Evidence: Efficient "System 2" Search for Faithful Multimodal Grounding](tree-of-evidence_efficient_34system_234_search_for_faithful_multimodal_grounding.md)
- [\[ACL 2026\] SafetyALFRED: Evaluating Safety-Conscious Planning of Multimodal Large Language Models](safetyalfred_evaluating_safety-conscious_planning_of_multimodal_large_language_m.md)
- [\[ICLR 2026\] Evaluating VLMs' Spatial Reasoning Over Robot Motion: A Step Towards Robot Planning with Motion Preferences](../../ICLR2026/multimodal_vlm/evaluating_vlms_spatial_reasoning_over_robot_motion_a_step_towards_robot_plannin.md)
- [\[ACL 2026\] Omni-Embed-Audio: Leveraging Multimodal LLMs for Robust Audio-Text Retrieval](omni-embed-audio_leveraging_multimodal_llms_for_robust_audio-text_retrieval.md)
- [\[NeurIPS 2025\] To See or To Read: User Behavior Reasoning in Multimodal LLMs](../../NeurIPS2025/multimodal_vlm/to_see_or_to_read_user_behavior_reasoning_in_multimodal_llms.md)

<!-- RELATED:END -->
