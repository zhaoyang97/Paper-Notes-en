---
title: >-
  [Paper Note] Contamination Detection for VLMs using Multi-Modal Semantic Perturbation
description: >-
  [ICLR2026][Multimodal VLM][data contamination] This paper proposes a multi-modal semantic perturbation framework for detecting data contamination in VLMs. It uses an LLM to generate dense captions and Flux ControlNet to alter answer-relevant semantic elements while preserving image composition. Contaminated models suffer sharp performance drops on perturbed samples due to memorization of original image-text pairs, whereas clean models are unaffected thanks to genuine reasoning ability. The paper also provides the first systematic validation that most existing LLM-based contamination detection methods are unreliable in VLM settings.
tags:
  - ICLR2026
  - Multimodal VLM
  - data contamination
  - VLM
  - benchmark leakage
  - semantic perturbation
  - ControlNet
date: 2026-05-08
content_hash: b85adb75432afbcf
---

# Contamination Detection for VLMs using Multi-Modal Semantic Perturbation

**Conference**: ICLR2026
**arXiv**: [2511.03774](https://arxiv.org/abs/2511.03774)
**Code**: [https://github.com/jadenpark0/mm-perturb](https://github.com/jadenpark0/mm-perturb)
**Area**: Multimodal VLM
**Keywords**: data contamination, VLM, benchmark leakage, semantic perturbation, ControlNet

## TL;DR

This paper proposes a multi-modal semantic perturbation framework for detecting data contamination in VLMs. It uses an LLM to generate dense captions and Flux ControlNet to alter answer-relevant semantic elements while preserving image composition. Contaminated models suffer sharp performance drops on perturbed samples due to memorization of original image-text pairs, whereas clean models are unaffected thanks to genuine reasoning ability. The paper also provides the first systematic validation that most existing LLM-based contamination detection methods are unreliable in VLM settings.

## Background & Motivation

**Background**: VLMs (e.g., LLaVA, Qwen2-VL) achieve impressive results on benchmarks such as MMStar and RealWorldQA, but their training data are typically proprietary, internet-scale corpora. This raises a critical concern: performance inflation caused by test-set leakage—models may have memorized evaluation questions rather than acquiring genuine visual reasoning capabilities.

**Limitations of Prior Work**:
1. **LLM detection methods are unreliable for VLMs**: Text-based perturbations (e.g., option shuffling, n-gram detection) do not alter visual features, so VLMs can still answer correctly via image memorization.
2. **Lack of systematic study**: The effectiveness of detection methods across different contamination strategies (standard fine-tuning vs. LoRA, varying epoch counts) has never been comprehensively evaluated.
3. **No existing method satisfies all three requirements simultaneously**: practicality (no clean reference model needed), reliability (consistent detection across training strategies), and consistency (detection signal correlates with contamination degree).

**Key Challenge**: VLMs are multimodal—perturbing only the text is insufficient (the model can answer from visual memory), and perturbing only the answer choices is insufficient (the model may rely on option position)—both image and text semantics must be perturbed simultaneously to break memorization.

**Goal**: Generate semantically perturbed image-question pairs that preserve overall composition while modifying answer-relevant semantic elements. Contaminated models fail on these pairs due to memorized image-text associations; clean models succeed through genuine reasoning. Contamination is detected by comparing model performance on original versus perturbed benchmarks.

## Method

### Overall Architecture

A five-step detection pipeline:
1. **Answer randomization**: Replace the correct answer in the original question with a randomly selected alternative option.
2. **Dense caption generation**: GPT-4o generates a conditioned dense textual description based on the original image and the new answer.
3. **Image perturbation generation**: Flux ControlNet with a Canny edge map preserves global composition while generating a new image conforming to the dense caption.
4. **Quality filtering**: Manual (or automatic) filtering ensures that each perturbed question is unambiguously answerable.
5. **Contamination detection**: Compare model accuracy on original versus perturbed benchmarks—a large gap indicates contamination.

Core principle: contaminated models have memorized an "image → answer" mapping; when image semantics change and the correct answer changes accordingly, the model still outputs the old answer, causing a sharp drop. Clean models rely on genuine reasoning and should perform comparably or better on perturbed questions of equivalent or lower difficulty.

### Key Design 1: Multi-Modal Semantic Perturbation

**Why image perturbation is necessary**: Merely shuffling choice order (Choice Confusion) or rotating option positions (CircularEval) still allows VLMs to answer from visual memory. Experiments in the paper demonstrate that these methods fail under multiple contamination settings.

**Key constraints of the perturbation approach**:
- Preserve overall image composition (via Canny edge map + ControlNet).
- Modify only answer-relevant semantic elements (e.g., changing a speed-limit sign from 25 to 35).
- Ensure perturbed questions are no harder than the originals (so that clean model performance does not degrade due to increased difficulty).

**Importance of conditioned captioning**: GPT-4o receives both the original image and the new answer choices as input when generating descriptions, ensuring the caption precisely highlights the visual elements that need to be changed.

### Key Design 2: Formalization of Three Detection Requirements

The paper introduces a formal definition of contamination degree $\text{deg}_\mathcal{D}(x) = (\sum_{d \in \mathcal{D}} \mathbf{1}_{\{x=d\}}) \times n$ and proposes three requirements accordingly:

| Requirement | Definition | Ours | Prior Methods |
|:---|:---|:---:|:---:|
| Practicality | No clean model or training corpus required | ✓ | Mostly ✗ |
| Reliability | Consistent across training strategies (standard FT / LoRA) | ✓ | Mostly ✗ |
| Consistency | Detection signal ∝ contamination degree | ✓ | Partial ▲ |

### Key Design 3: Framework Agnosticism

The paper verifies that the approach is agnostic to:
- **Generative model**: other diffusion models can substitute Flux + ControlNet.
- **LLM**: other language models can generate the dense captions in place of GPT-4o.
- **Filtering strategy**: automatic filtering with a strong reasoning model can replace manual filtering.

## Key Experimental Results

### Main Results: Contamination Detection Comparison on MMStar

| Detection Method | Clean Model Required? | LLaVA LoRA 1ep | LLaVA LoRA 3ep | Qwen LLM 1ep | Qwen LLM 3ep |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Ours** (Δ) | No ✓ | -8.29 ✓ | -16.16 ✓ | -29.50 ✓ | -43.03 ✓ |
| CircularEval (Δ) | Yes ✗ | -23.44 ✓ | +1.22 ✗ | -15.96 ✓ | -28.69 ✓ |
| Choice Confusion (Δ) | No ✓ | +1.01 ✗ | +14.75 ✗ | +21.01 ✗ | +12.12 ✗ |
| Multi-modal Leakage (Δ) | Yes ✗ | +10.31 ✓ | +11.12 ✓ | +0.41 ✓ | -10.70 ✗ |

**Key Findings**:
- The proposed method successfully detects contamination across **all** 12 settings (2 models × 3 training strategies × 2 epoch ranges), making it the only method that satisfies all three requirements.
- Clean models perform **better** on perturbed data (LLaVA: +31.51, Qwen: +16.16), confirming that perturbed questions are no harder than the originals.
- Choice Confusion fails completely in 10 out of 12 settings, confirming that VLMs can indeed bypass text-level perturbations via visual memorization.

### Ablation Study: Detection Signal vs. Contamination Degree

| Model | Epoch 1 Δ | Epoch 2 Δ | Epoch 3 Δ |
|:---|:---:|:---:|:---:|
| LLaVA LoRA | -8.29 | -13.13 | -16.16 |
| LLaVA LLM+MLP | -8.49 | -11.52 | -13.74 |
| Qwen LoRA | -7.07 | -28.89 | -32.32 |
| Qwen LLM only | -29.50 | -43.03 | -43.03 |

The magnitude of performance drop increases monotonically (or saturates) with epoch count, perfectly satisfying the consistency requirement—stronger contamination produces a stronger detection signal.

### Representativeness of the Filtered Subset

| Dataset | Full Set | Filtered Subset | Difference |
|:---|:---:|:---:|:---:|
| RealWorldQA (LLaVA) | 49.01% | 52.05% | +3.04% |
| RealWorldQA (Qwen) | 70.33% | 70.45% | +0.12% |
| MMStar (LLaVA) | 32.87% | 37.78% | +4.91% |
| MMStar (Qwen) | 62.02% | — | — |

Model performance on the filtered subset closely mirrors that on the full set, indicating that filtering introduces no systematic bias.

## Highlights & Insights

### Strengths

1. **Clear problem formalization**: The three detection requirements (practicality / reliability / consistency) provide a unified evaluation framework for the field.
2. **Elegant core intuition**: Leveraging ControlNet to preserve composition while altering semantics directly attacks the memorization mechanism at the heart of contamination.
3. **Highly thorough experiments**: A complete cross-factorial evaluation covering 2 models × 3 training strategies × 3 epochs × 4 detection methods yields strong empirical evidence.

### Limitations & Future Work

1. The approach depends on the quality of generative models—current diffusion models still struggle with text rendering and complex geometry, leading to substantial sample attrition (57% retained for RealWorldQA, 32% for MMStar).
2. Manual filtering is costly; while automatic filtering is shown to be feasible, it introduces an additional dependency on a strong reasoning model.
3. Only fine-tuning-stage contamination is examined; detecting leakage at the pre-training stage is left unaddressed due to computational cost.

### Rating

⭐⭐⭐⭐

**Rationale**: This paper delivers the first reliable, practical, and consistent solution for VLM contamination detection. The core insight—that image semantics must be perturbed rather than text alone—is simple yet profound. The systematic and comprehensive experimental design is the strongest among comparable works in this area, laying a solid foundation for future research on the trustworthiness of VLM evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Multi-modal Data Spectrum: Multi-modal Datasets are Multi-dimensional](multi-modal_data_spectrum_multi-modal_datasets_are_multi-dimensional.md)
- [\[ICLR 2026\] Steering and Rectifying Latent Representation Manifolds in Frozen Multi-Modal LLMs for Video Anomaly Detection](steering_and_rectifying_latent_representation_manifolds_in_frozen_multi-modal_ll.md)
- [\[CVPR 2026\] UniMMAD: Unified Multi-Modal and Multi-Class Anomaly Detection via MoE-Driven Feature Decompression](../../CVPR2026/multimodal_vlm/unimmad_unified_multi-modal_and_multi-class_anomaly_detection_via_moe-driven_fea.md)
- [\[CVPR 2026\] Rethinking VLMs for Image Forgery Detection and Localization](../../CVPR2026/multimodal_vlm/rethinking_vlms_for_image_forgery_detection_and_localization.md)
- [\[AAAI 2026\] Multi-Agent VLMs Guided Self-Training with PNU Loss for Low-Resource Offensive Content Detection](../../AAAI2026/multimodal_vlm/multi-agent_vlms_guided_self-training_with_pnu_loss_for_low-resource_offensive_c.md)

</div>

<!-- RELATED:END -->
