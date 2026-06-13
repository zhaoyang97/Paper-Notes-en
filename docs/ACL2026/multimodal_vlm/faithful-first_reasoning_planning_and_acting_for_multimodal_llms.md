---
title: >-
  [Paper Note] Faithful-First Reasoning, Planning, and Acting for Multimodal LLMs
description: >-
  [ACL 2026][Multimodal VLM][Perceptual Faithfulness] Ours proposes the Faithful-First RPA framework, which evaluates perceptual faithfulness through the FaithEvi pipeline in each reasoning step (verifying if claimed objec…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Perceptual Faithfulness"
  - "Reasoning Planning and Execution"
  - "Multimodal Hallucination"
  - "Visual Evidence Verification"
  - "Step-by-step Reasoning"
date: 2026-05-08
content_hash: 63e2864cb6501123
---

# Faithful-First Reasoning, Planning, and Acting for Multimodal LLMs

**Conference**: ACL 2026 Findings  
**arXiv**: [2511.08409](https://arxiv.org/abs/2511.08409)  
**Code**: [GitHub](https://github.com/lijunxian111/Faithful-First-RPA)  
**Area**: Multimodal VLM / Reasoning Faithfulness  
**Keywords**: Perceptual Faithfulness, Reasoning Planning and Execution, Multimodal Hallucination, Visual Evidence Verification, Step-by-step Reasoning

## TL;DR

Ours proposes the Faithful-First RPA framework, which evaluates perceptual faithfulness through the FaithEvi pipeline in each reasoning step (verifying if claimed objects truly exist in the image) and enforces evidence-based planning and action via the FaithAct mechanism. This improves perceptual faithfulness by up to 24% without compromising task accuracy.

## Background & Motivation

**Background**: Multimodal Large Language Models (MLLMs) have made significant progress in tasks like VQA and visual reasoning. However, their reasoning trajectories often exhibit "unfaithfulness"—generated explanations do not align with visual evidence, or they provide post-hoc rationalizations for their predictions.

**Limitations of Prior Work**: (1) Existing works primarily focus on behavioral faithfulness (whether the reasoning chain reflects the model's decision process) while ignoring perceptual faithfulness (whether reasoning steps are grounded in verifiable visual input); (2) Reasoning frameworks like CoT and ReAct do not verify the perceptual foundations of intermediate steps; (3) Models may describe a black bicycle as yellow while still providing the correct final answer.

**Key Challenge**: Current reasoning frameworks follow a "generate-then-verify" paradigm, where perceptual errors are only discovered after the reasoning chain is complete, making correction costly and limited in effectiveness. Faithfulness should be a design principle rather than a post-hoc evaluation metric.

**Goal**: To establish a unified framework that can both quantitatively evaluate the perceptual faithfulness of reasoning chains and actively enforce evidence verification during the reasoning process.

**Key Insight**: Based on the principle that "perceptually faithful models only reason about what is visually observable," the reasoning process is formalized as a faithfulness-constrained planning problem.

**Core Idea**: At each step of reasoning, claimed objects are first extracted, their existence is verified through preference voting and visual grounding to compute a faithfulness score. Steps that fail to meet a threshold must be corrected before proceeding in the reasoning chain.

## Method

### Overall Architecture

Faithful-First RPA consists of two core components: (1) FaithEvi evaluation pipeline—performs step-level and chain-level perceptual faithfulness assessment of the reasoning chain; (2) FaithAct planning and acting mechanism—utilizes signals from FaithEvi to dynamically verify and correct each step during reasoning generation. The overall workflow is: Image + Question input → MLLM generates a reasoning step → FaithEvi evaluates step faithfulness → If sub-standard, FaithAct triggers correction → Once passed, it enters the reasoning chain → Continue to next step.

### Key Designs

1. **FaithEvi: Perceptual Faithfulness Evaluation Pipeline**:

    - **Function**: Quantifies the degree of visual evidence support for each step in the reasoning chain.
    - **Mechanism**: Three stages. **Stage 1: Claimed Object Extraction**—Uses Qwen2.5-7B-Instruct to extract the set of claimed objects $O_t = \{O_t^1, \dots, O_t^{m_t}\}$ from each reasoning step. **Stage 2: Preference Voting + Visual Grounding**—(a) Uses frozen CLIP-ViT-Large to encode image and object text, predicting existence probability $c_p$ via a two-layer MLP (trained on POPE); (b) Uses frozen GroundingDINO to localize objects and obtain detection confidence $c_g$. **Stage 3: Faithfulness Scoring**—Fuses two confidences $c_t^i = 0.7 \cdot c_p + 0.3 \cdot c_g$, mapped to a three-level discrete score (<0.4→0, 0.4-0.6→$c_t^i$, >0.6→1). Step-level score $F_{\text{step},t} = \frac{1}{m_t}\sum f_t^i$, chain-level score $F_{\text{chain}} = \frac{1}{n}\sum F_{\text{step},t}$.
    - **Design Motivation**: Preference voting provides global existence verification (as detector confidence is unreliable under weak visual cues), while grounding provides regional spatial evidence; the two are complementary.

2. **FaithAct: Faithfulness-First Planning and Acting**:

    - **Function**: Transforms the reasoning process into a faithfulness-constrained planning problem.
    - **Mechanism**: The planning objective is $S^* = \arg\max F_{\text{step}}(s_t)$ s.t. $\forall t, F_{\text{step}}(s_t) \geq c$. Immediately after each step is generated, it is verified via FaithEvi. Steps failing the threshold are sent back to the MLLM for re-generation with updated evidence (existence labels, bounding boxes, counts). It provides an extensible function interface: `Poll()` (existence probability), `Ground()` (bounding box detection), `Select()` (confirm existence), `Abstain()` (confirm non-existence), `Count()` (counting reasoning).
    - **Design Motivation**: Unlike the "generate-then-verify" paradigm, FaithAct adopts a "verify-while-generating" principle to correct perceptual errors as early as possible in the reasoning chain, preventing error propagation.

3. **Action-Guided Reasoning Correction**:

    - **Function**: Repairs reasoning steps that fail the faithfulness threshold.
    - **Mechanism**: Steps that fail verification are not simply discarded but are re-generated using updated evidence. The correction prompt guides the model to maintain logical continuity while correcting perceptual descriptions.
    - **Design Motivation**: The improvement in faithfulness for later steps in FaithAct is particularly significant—consistent with previous findings that longer CoT chains are more susceptible to noise in later stages.

### Loss & Training

Ours is an inference-time framework and does not involve model training. The preference voting head is trained on the POPE dataset (two-layer MLP), while GroundingDINO and CLIP are used frozen. GroundingDINO uses box threshold=0.35 and text threshold=0.25.

## Key Experimental Results

### Main Results

**Perceptual Faithfulness Evaluation ($F_{\text{chain}}$, %)**

| Model + Method | LLaVA-bench | RealWorldQA | POPE | MMHal | Average |
|------------|-------------|-------------|------|-------|------|
| Qwen + CoT | 46.05 | 48.11 | 45.21 | 53.34 | 48.18 |
| Qwen + ReAct | 54.82 | 56.82 | 45.02 | 33.76 | 47.61 |
| **Qwen + FaithAct** | **55.10** | **57.22** | **56.87** | **66.45** | **58.91** |
| InternVL + CoT | 45.63 | 44.23 | 43.25 | 53.17 | 46.57 |
| **InternVL + FaithAct** | **52.64** | **57.35** | **56.01** | **61.71** | **56.93** |
| LLaVA + CoT | 47.56 | 52.31 | 52.28 | 30.63 | 45.70 |
| **LLaVA + FaithAct** | **52.82** | **58.11** | **56.09** | **39.91** | **51.73** |

**Task Accuracy Retention**

| Model | Method | RealWorldQA(%) | MMHal(rating) |
|------|------|---------------|---------------|
| Qwen | CoT | 70.1 | 3.40 |
| Qwen | FaithAct | **74.5** | **3.48** |
| InternVL | CoT | 70.8 | 3.61 |
| InternVL | FaithAct | 71.2 | 3.58 |

### Ablation Study

**Core Component Ablation (Qwen, RealWorldQA / MMHal)**

| Configuration | RealWorldQA(%) | MMHal(%) |
|------|---------------|----------|
| FaithAct (Full) | 57.22 | 66.45 |
| w/o Poll() | 54.24 (-3.0) | 63.25 (-3.2) |
| w/o Ground() | 53.16 (-4.1) | 62.47 (-4.0) |

### Key Findings

- FaithAct achieves an average perceptual faithfulness of 55.86%, a 7.76 percentage point improvement over the strongest baseline ReAct (48.10%).
- The largest improvement is seen on the hallucination-sensitive benchmark MMHal: an average gain of 21.99% over CoT and 9.81% over tool-augmented methods.
- Faithfulness improvements do not harm task accuracy—Qwen even improved from 70.1% to 74.5% on RealWorldQA.
- The contribution of Ground() is slightly greater than Poll(), indicating that spatial localization provides more critical visual evidence.
- Replacing GroundingDINO with SAM3 led to a performance drop of approximately 5%, suggesting the framework requires localization-specific models.
- The benefits of FaithAct are more pronounced in the later steps of the reasoning chain, validating the hypothesis that later steps are more prone to hallucination.
- Human verification shows LLM object extraction accuracy reached 99.42% (7550 object-level labels) with a fragment validity of 0.968.
- Inference time increases by approximately 2-3 times (FaithAct 14-19s vs CoT 3-11s).

## Highlights & Insights

- The concept that "faithfulness should be a design principle rather than a post-hoc metric" is compelling—embedding faithfulness constraints into the reasoning loop ensures every step is supported by evidence.
- The distinction between perceptual faithfulness vs. behavioral faithfulness has theoretical value—a model can be "right for the wrong reasons" (behaviorally faithful but perceptually unfaithful) or "wrong for the right reasons" (perceptually faithful but behaviorally unfaithful).
- The extensible function interface design (Poll/Ground/Select/Abstain/Count) makes the framework easy to scale to attribute and relationship verification.

## Limitations & Future Work

- Currently, faithfulness is only verified at the object existence level, excluding attributes (color, size) and relationships (spatial relations, actions).
- Inference time increases by about 2-3 times.
- Behavioral faithfulness is not directly evaluated; it is only assumed that perceptual faithfulness promotes behavioral consistency.
- The advantage is less obvious on benchmarks with weak perceptual requirements (e.g., MathVista).

## Related Work & Insights

- **vs Grounded-CoT (Wu et al., 2025)**: The latter appends localization info after reasoning, while Ours performs real-time verification during reasoning—FaithAct outperforms Grounded-CoT in 11/12 settings.
- **vs ReAct (Yao et al., 2022)**: ReAct allows tool calls but does not enforce faithfulness constraints; Ours demonstrates that the $F_{\text{chain}}$ of ReAct is theoretically upper-bounded by FaithAct.
- **vs VAT (Liu et al., 2025)**: Visual Abstraction Thought degrades severely on POPE (21.46%), indicating that abstraction may exacerbate perceptual disconnection.

## Rating

- Novelty: ⭐⭐⭐⭐ Formal definition of perceptual faithfulness and the "verify-while-generating" paradigm are innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ 3 models, 4 benchmarks, complete ablations, and human verification.
- Writing Quality: ⭐⭐⭐⭐ Clear distinction between perceptual/behavioral faithfulness, rigorous framework design logic.
- Value: ⭐⭐⭐⭐ Provides a practical framework for the trustworthiness of multimodal reasoning; function interface is extensible.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Tree-of-Evidence: Efficient "System 2" Search for Faithful Multimodal Grounding](tree-of-evidence_efficient_34system_234_search_for_faithful_multimodal_grounding.md)
- [\[ACL 2026\] ShredBench: Evaluating the Semantic Reasoning Capabilities of Multimodal LLMs in Document Reconstruction](shredbench_evaluating_the_semantic_reasoning_capabilities_of_multimodal_llms_in_.md)
- [\[ICLR 2026\] Evaluating VLMs' Spatial Reasoning Over Robot Motion: A Step Towards Robot Planning with Motion Preferences](../../ICLR2026/multimodal_vlm/evaluating_vlms_spatial_reasoning_over_robot_motion_a_step_towards_robot_plannin.md)
- [\[ACL 2026\] A Survey of Multimodal Mathematical Reasoning: From Perception, Alignment to Reasoning](a_survey_of_multimodal_mathematical_reasoning_from_perception_alignment_to_reaso.md)
- [\[NeurIPS 2025\] To See or To Read: User Behavior Reasoning in Multimodal LLMs](../../NeurIPS2025/multimodal_vlm/to_see_or_to_read_user_behavior_reasoning_in_multimodal_llms.md)

</div>

<!-- RELATED:END -->
