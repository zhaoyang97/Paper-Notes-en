---
title: >-
  [Paper Note] FlexAC: Towards Flexible Control of Associative Reasoning in Multimodal Large Language Models
description: >-
  [NeurIPS 2025][VLM Reasoning][hallucination control] FlexAC identifies that associative reasoning in MLLMs is primarily encoded in intermediate layers. By extracting steering vectors from hallucinated responses and injecting them into intermediate-layer representations at inference time, it enables flexible control over faithfulness and creativity—reducing hallucination rate by 29% (CHAIR) and improving creativity by 5.8× (Creation-MMBench), all without any training.
tags:
  - "NeurIPS 2025"
  - "VLM Reasoning"
  - "hallucination control"
  - "creativity enhancement"
  - "associative reasoning"
  - "intermediate-layer intervention"
  - "steering vectors"
date: 2026-05-08
content_hash: 932312525b09ddec
---

# FlexAC: Towards Flexible Control of Associative Reasoning in Multimodal Large Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2510.11190](https://arxiv.org/abs/2510.11190)  
**Code**: [github.com/ylhz/FlexAC](https://github.com/ylhz/FlexAC)  
**Area**: Multimodal VLM
**Keywords**: hallucination control, creativity enhancement, associative reasoning, intermediate-layer intervention, steering vectors

## TL;DR
FlexAC identifies that associative reasoning in MLLMs is primarily encoded in intermediate layers. By extracting steering vectors from hallucinated responses and injecting them into intermediate-layer representations at inference time, it enables flexible control over faithfulness and creativity—reducing hallucination rate by 29% (CHAIR) and improving creativity by 5.8× (Creation-MMBench), all without any training.

## Background & Motivation
**Background**: Multimodal LLMs face an inherent tension between faithfulness (low association) and creativity (high association)—factual tasks require suppressing associative reasoning, while creative tasks require amplifying it.

**Limitations of Prior Work**: (1) Hallucination mitigation methods (e.g., contrastive decoding in VCD, preference optimization in Ha-DPO) broadly suppress associative capacity while reducing hallucinations, leading to degraded creativity (VDAT drops by 1.78); (2) No tunable mechanism exists—methods either fully suppress association or leave it unaddressed.

**Key Challenge**: Hallucination and creativity likely share the same associative mechanism, manifesting as either "harmful" or "beneficial" depending on the task, yet existing methods cannot distinguish between the two.

**Goal**: (1) Localize the layer at which associative behavior emerges in MLLMs; (2) Design a controllable mechanism for adjusting associative strength.

**Key Insight**: Drawing from cognitive science concepts of convergent thinking (fact-grounded association) and divergent thinking (atypical association), the paper hypothesizes that hallucination and creativity stem from a shared associative mechanism that can be regulated via directional interventions in intermediate-layer representations.

**Core Idea**: Hallucinated responses encode directional information about association. Extracting the representation difference between hallucinated and faithful outputs as a steering vector allows positive injection to enhance creativity and negative injection to suppress hallucination.

## Method

### Overall Architecture
The method proceeds in two phases: **Phase I (offline)** constructs general and task-specific associative steering vectors; **Phase II (inference-time)** injects steering vectors into intermediate layers with adaptive strength calibration. No retraining is required throughout.

### Key Designs

1. **Intermediate-Layer Associative Behavior Analysis and Localization**:

    - Function: Identify at which layers associative behavior emerges in the model.
    - Mechanism: Collect faithful responses $f^{(n)}$ and hallucination-induced responses $f^{(a)}$ for 1,000 COCO images, and compute cosine distance $\mathcal{D}_\text{cos}$ and Euclidean distance $\mathcal{D}_\text{Euc}$ layer by layer. A layer intervention experiment is further conducted—at layer $m$, the associative feature is replaced with the non-associative counterpart $f_m^{\text{modified}} = f_m^{(n)}$, and the effect on downstream layers is observed.
    - Key Findings: (1) Early layers (0–9) exhibit low distance, indicating shared low-level perception; (2) Cosine distance peaks at intermediate layers (10–15), indicating that associative direction is formed there; (3) Replacing intermediate-layer features substantially reduces downstream divergence, confirming that intermediate layers are the origin—not merely propagators—of associative behavior.

2. **Hallucination-Guided Associative Steering Vector Construction (Phase I)**:

    - Function: Extract directional vectors from hallucinated responses for association control.
    - Mechanism: For each sample, compute the directional difference at layer $l$ as $v_l = f_l^{(a)} - f_l^{(n)}$, select the Top-K sample pairs with the largest cosine distance, and average them to obtain a general steering vector:
    $\mathcal{I} = \text{Top-K}(\mathcal{D}_\text{cos}(f_{l,i}^{(a)}, f_{l,i}^{(n)})); \quad v_l = \frac{1}{|\mathcal{I}|} \sum_{i \in \mathcal{I}} (f_{l,i}^{(a)} - f_{l,i}^{(n)})$
    - Design Motivation: Top-K selection reduces noise; experiments show that randomly sampling 50 images from 2,000 suffices to construct an effective vector.

3. **Task-Specific Associative Vectors (Directional Integration)**:

    - Function: Construct dedicated steering directions for different creative tasks such as story generation and metaphor production.
    - Mechanism: GPT-4o is used to generate high-association outputs for the target task; the difference between their intermediate-layer features and those of the base model output defines a task-specific vector $v_l^{\text{task}}$. At inference time, this is combined with the general vector: $f_l^{\text{control}} = f_l + \alpha_\text{gen} \cdot v_l^{\text{gen}} + \alpha_\text{task} \cdot v_l^{\text{task}}$
    - Design Motivation: Associative reasoning is multi-dimensional (event planning vs. literary creation require different associative directions), and a single vector is insufficient.

4. **Strength-adaptive Intervention Calibration (SIC)**:

    - Function: Prevent semantic drift caused by excessive steering.
    - Mechanism: The steering strength $\alpha$ is adaptively adjusted based on the alignment between the current representation and the steering direction:
    $\alpha = \text{sigmoid}\left(\max\left(-\frac{f_l \cdot v_l}{\|f_l\|\|v_l\|}, 0\right)\right)$
      When the current representation is already aligned with the associative direction, $\alpha$ is small (suppressing over-steering); when misaligned, $\alpha$ is large (amplifying steering). Post-intervention normalization preserves feature scale: $f_l^{\text{control}} \leftarrow f_l^{\text{control}} \cdot \frac{\|f_l\|}{\|f_l^{\text{control}}\|}$
    - Design Motivation: Uniformly applying steering vectors causes excessive deviation for inputs that already exhibit strong associative tendencies.

### Loss & Training
FlexAC is entirely training-free. It only requires offline construction of steering vectors (50 images + GPT-4o-generated samples) and injects them into intermediate layers at inference time.

## Key Experimental Results

### Main Results: Hallucination Benchmarks

| Model | Method | CHAIR_S↓ | CHAIR_I↓ | POPE F1↑ |
|-------|--------|----------|----------|----------|
| Qwen-VL | Regular | 40.6 | 12.5 | 85.6 |
| Qwen-VL | VCD | 42.0 | 11.2 | 86.3 |
| Qwen-VL | **FlexAC** | **19.2** | **5.4** | **87.1** |
| LLaVA-1.5 | Regular | 50.8 | 14.3 | 86.5 |
| LLaVA-1.5 | Ha-DPO | 36.8 | 10.4 | 83.9 |
| LLaVA-1.5 | **FlexAC** | **36.6** | **10.4** | **87.9** |
| DeepSeek-VL2 | Regular | 32.6 | 9.2 | 88.5 |
| DeepSeek-VL2 | **FlexAC** | **28.6** | **8.1** | **88.6** |

### Creativity Benchmarks

| Method | VDAT (Qwen) | VDAT (LLaVA) | Creation-MMBench Reward |
|--------|-------------|--------------|------------------------|
| Regular | 84.85 | 86.89 | 0.00 |
| Ha-DPO | — | 85.11↓ | — |
| VCD | 83.69↓ | 86.83 | -3.86↓ |
| **FlexAC** | **86.58**↑ | **88.49**↑ | **10.92**↑ |

FlexAC is the only method that simultaneously improves both faithfulness and creativity; all other methods either degrade creativity when reducing hallucinations, or fail to substantially improve either.

### Ablation Study

| Configuration | CHAIR_S↓ | VDAT↑ |
|---------------|----------|-------|
| FlexAC-P (full, α=−1) | **19.2** | — |
| FlexAC-C (full, α=1) | — | **86.58** |
| FlexAC − IS − SIC | 30.4 | 85.05 |
| FlexAC − DI | ~20 | 85.8 |
| Regular | 40.6 | 84.85 |

### Key Findings
- Intermediate layers (Qwen: 15–17, LLaVA: 11–13, DeepSeek: 4–6) are the optimal intervention points; interventions at early or deep layers are marginally effective.
- Instance Selection and SIC contribute most to hallucination mitigation (removing both raises CHAIR_S from 19.2 to 30.4).
- Directional Integration is critical for creativity enhancement, reflecting the multi-dimensional nature of associative reasoning.
- FlexAC does not degrade general benchmarks (MME/MMMU/MMStar) and in fact yields gains, particularly on OCR tasks due to enhanced text–visual association.

## Highlights & Insights
- **Unified View of Hallucination as Association**: Framing "harmful hallucination" and "beneficial creativity" as a continuous spectrum of associative strength, and regulating both bidirectionally with the same mechanism, is a novel and elegant conceptual contribution.
- **Layer Intervention Methodology**: The feature-replacement experiment precisely localizes the layers at which association originates (rather than propagates), offering a generalizable analytical paradigm for dissecting other model behaviors.
- **Adaptive Design of SIC**: A simple cosine-angle threshold achieves sample-level dynamic strength control, preventing over-steering; its empirical necessity is confirmed by ablation results.

## Limitations & Future Work
- Requires white-box access to model intermediate layers, making it inapplicable to black-box API models such as ChatGPT.
- Steering vectors are constructed from COCO images; whether reconstruction is necessary for domain-shifted scenarios (e.g., medical imaging) remains to be validated.
- The VDAT metric is based on CLIP embedding semantic distance and may not fully capture human perception of creativity.
- Validation is limited to 7B-scale models; the intermediate-layer dynamics of larger models may differ.

## Related Work & Insights
- **vs. VCD (contrastive decoding)**: VCD contrasts clear and distorted inputs at the decoding level; FlexAC directly manipulates associative direction at the representation level, enabling more precise and bidirectional control.
- **vs. Ha-DPO (preference optimization)**: Ha-DPO requires additional training and irreversibly suppresses association; FlexAC is training-free and allows flexible switching.
- **vs. CAA (contrastive activation addition)**: FlexAC can be viewed as an extension of CAA to multimodal settings, augmented with SIC adaptive calibration and task-specific directional integration.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The unified hallucination–creativity perspective is a genuinely novel insight; the training-free bidirectional control framework offers strong practical utility.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 3 categories of benchmarks (hallucination / creativity / general), 7 benchmarks, and 3 models, with comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ The logical chain from analysis to findings to method is clear and well-illustrated.
- Value: ⭐⭐⭐⭐⭐ Highly practical—training-free, plug-and-play, and effective, with direct implications for MLLM deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] AffordBot: 3D Fine-grained Embodied Reasoning via Multimodal Large Language Models](affordbot_3d_fine-grained_embodied_reasoning_via_multimodal_large_language_model.md)
- [\[NeurIPS 2025\] PhysVLM-AVR: Active Visual Reasoning for Multimodal Large Language Models in Physical Environments](physvlm-avr_active_visual_reasoning_for_multimodal_large_language_models_in_phys.md)
- [\[NeurIPS 2025\] Recognition through Reasoning: Reinforcing Image Geo-localization with Large Vision-Language Models](recognition_through_reasoning_reinforcing_image_geo-localization_with_large_visi.md)
- [\[NeurIPS 2025\] MM-OPERA: Benchmarking Open-ended Association Reasoning for Large Vision-Language Models](mm-opera_benchmarking_open-ended_association_reasoning_for_large_vision-language.md)
- [\[NeurIPS 2025\] ACT as Human: Multimodal Large Language Model Data Annotation with Critical Thinking](act_as_human_multimodal_large_language_model_data_annotation.md)

</div>

<!-- RELATED:END -->
