---
title: >-
  [Paper Note] FORCE: Transferable Visual Jailbreaking Attacks via Feature Over-Reliance CorrEction
description: >-
  [CVPR 2026][Robotics][Visual jailbreaking attacks] By analyzing the over-reliance on layer-wise features and spectral-domain components in visual jailbreaking attacks, FORCE corrects non-generalizable feature dependencies and guides the attack toward flatter loss landscapes, thereby substantially improving cross-model transferability.
tags:
  - CVPR 2026
  - Robotics
  - Visual jailbreaking attacks
  - multimodal large language models
  - adversarial transferability
  - spectral analysis
  - loss landscape
date: 2026-05-08
content_hash: 9d5c62d58c6e6ad3
---

# FORCE: Transferable Visual Jailbreaking Attacks via Feature Over-Reliance CorrEction

**Conference**: CVPR 2026
**arXiv**: [2509.21029](https://arxiv.org/abs/2509.21029)
**Code**: [GitHub](https://github.com/tmllab/2026_CVPR_FORCE)
**Area**: Robotics / AI Safety
**Keywords**: Visual jailbreaking attacks, multimodal large language models, adversarial transferability, spectral analysis, loss landscape

## TL;DR

By analyzing the over-reliance on layer-wise features and spectral-domain components in visual jailbreaking attacks, FORCE corrects non-generalizable feature dependencies and guides the attack toward flatter loss landscapes, thereby substantially improving cross-model transferability.

## Background & Motivation

Although multimodal large language models (MLLMs) exhibit strong textual safety alignment, the newly introduced visual modality introduces additional vulnerabilities. Optimization-based visual jailbreaking attacks (e.g., PGD) can reliably bypass the safety guardrails of open-source MLLMs, yet their cross-model transferability remains extremely limited—attacks that achieve near-100% success on the source model often fail dramatically when transferred to target models.

The root causes of this limitation are:
- **Sharp loss landscapes**: Generated adversarial examples reside in high-sharpness regions, where even minor parameter perturbations cause a steep rise in loss and render the attack ineffective.
- **Model-specific feature over-reliance**: Attacks depend excessively on features unique to the source model and fail to generalize to target models with different parameters.
- As a result, visual jailbreaking attacks cannot effectively assess the safety of closed-source commercial MLLMs.

## Method

### Overall Architecture

FORCE (Feature Over-Reliance CorrEction) augments the standard PGD attack framework with two regularization components—**layer-aware regularization** and **spectral feature rescaling**—which respectively correct non-generalizable feature dependencies in intermediate layers and in the spectral domain, guiding the attack to explore a flatter feasible region.

### Key Designs

1. **Loss Landscape Analysis**: The authors first visualize the input-space and weight-space loss landscapes of visual jailbreaking attacks. They find that attacks become trapped in source-model local optima: a perturbation of merely 0.03 pixels or a weight shift of 0.0002 suffices to neutralize the attack. This sharp landscape indicates over-reliance on model-specific features.

2. **Layer-aware Regularization**: Analysis reveals that the feasible region of shallow-layer features is extremely narrow (layer 11 requires retaining >90% of adversarial features for success), whereas deeper-layer features are more robust. Stronger regularization is therefore applied to shallower layers by sampling reference points within a neighborhood $\eta$ and maximizing the $L_2$ distance between their layer-wise features and those of the adversarial example:

$$d_l = \|f_{\theta,l}(\mathbf{x}_{\text{img}}+\delta, \mathbf{x}_{\text{txt}}) - f_{\theta,l}(\mathbf{x}_{\text{img}}+\delta+\eta, \mathbf{x}_{\text{txt}})\|_2^2$$

Regularization strength decreases with layer depth: $\lambda_l = \lambda \cdot \max(1-(2l/L)^2, 0)$, ensuring stronger constraints on shallower layers.

3. **Spectral Feature Rescaling**: Analysis shows that as optimization proceeds, attacks become increasingly dependent on high-frequency components (which are semantically impoverished), deviating from the low-frequency-dominant distribution of natural images. Frequency bands are adaptively rescaled as:

$$w_m = \min\left(\beta, \frac{\ell_{m-1}}{\ell_m} \cdot \beta\right)$$

When the influence of a high-frequency band exceeds $\beta$ times that of its adjacent lower-frequency band, it is suppressed. The perturbation is then reconstructed via FFT/IFFT: $\delta_{\text{rescaled}} = \text{IFFT}((A \odot S) \odot e^{i\Phi})$.

### Loss & Training

The overall optimization objective combines the original jailbreaking loss with the layer-wise regularization loss:

$$\ell_{\text{reg}} = \frac{1}{N}\sum_{n=1}^{N}\sum_{l=1}^{L} \lambda_l \cdot \frac{\ell_{\text{ref}}}{d_l}$$

Here $\ell_{\text{ref}}$ ensures that reference samples remain within the feasible region, while $d_l$ encourages expansion of the layer-wise representation region. Both components are integrated into the standard PGD pipeline: spectral rescaling is applied first, followed by layer-feature exploration.

## Key Experimental Results

### Main Results

The source model is LLaVA-v1.5-7B; attacks are transferred to multiple black-box target models (MaliciousInstruct dataset):

| Target Model Type | Target Model | Metric (ASR↑) | PGD | FORCE | Gain |
|---|---|---|---|---|---|
| Adapter-Based | LLaVA-v1.6-mistral-7b | ASR | 61.00% | 69.00% | +12.3% |
| Adapter-Based | InstructBLIP-Vicuna-7B | ASR | 84.00% | 92.00% | +9.5% |
| Adapter-Based | Idefics3-8B-Llama3 | ASR | 53.00% | 64.00% | +20.8% |
| Early-Fusion | Qwen2.5-VL-7B | ASR | 5.00% | 11.00% | +120% |
| Early-Fusion | LLaMA-3.2-11B | ASR | 1.00% | 2.00% | +100% |
| Commercial | GPT-5 | ASR | 1.00% | 2.00% | +100% |
| Commercial | Gemini-2.5-Pro | ASR | 10.00% | 10.00% | +0% |

### Ablation Study

| Configuration | ASR↑ | Query↓ | Note |
|---|---|---|---|
| Baseline (PGD) | 53.00% | 50.73 | No regularization |
| + Layer Feature | 55.00% | 48.46 | Layer regularization contributes +3.8% |
| + Frequency Feature | 59.00% | 44.03 | Spectral rescaling contributes +11.3% |
| FORCE (Both) | 64.00% | 39.59 | Synergistic effect; total gain +20.6% |

### Key Findings

- Spectral rescaling (+11.3%) contributes more than layer regularization (+3.8%), indicating that high-frequency over-reliance is the primary bottleneck for poor transferability.
- Against early-fusion architectures (e.g., Qwen2.5-VL, LLaMA-3.2), the baseline attack nearly completely fails (93% failure rate); FORCE achieves approximately 100% relative improvement.
- FORCE remains effective in zero-shot settings, improving ASR on Gemini-2.5-Pro from 1% to 3%.
- Visualizations confirm that FORCE genuinely expands the shallow-layer feasible region and restores the natural spectral distribution.

## Highlights & Insights

- **Rigorous analytical framework**: The poor transferability of visual attacks is systematically examined from three dimensions—loss landscape, layer-wise features, and spectral domain.
- **Discovery of "shallow-layer vulnerability"**: The closer to the shallow layers, the narrower the attack's feasible region, providing a new perspective on the security mechanisms of MLLM visual encoders.
- **Innovation from a spectral perspective**: The abnormal growth of high-frequency dependence during attack optimization is revealed and effectively corrected through a concise spectral rescaling scheme.
- **Practical utility**: No access to target model parameters is required, making the method directly applicable to red-teaming evaluations of closed-source MLLMs.

## Limitations & Future Work

- Absolute ASR against closed-source commercial models (GPT-5, Claude-Sonnet-4) remains very low (1–2%), posing limited practical threat.
- Evaluation is conducted with only LLaVA-v1.5-7B as the source model; larger models and multi-source combinations warrant further exploration.
- Early-fusion architectures exhibit noticeably stronger safety guardrails; more specialized transfer strategies tailored to their unique structure may be needed.
- The method introduces additional computational overhead from spectral analysis and multi-reference-point sampling.

## Related Work & Insights

- This work draws a parallel with adversarial transferability research in classification tasks (e.g., SAM, spectral methods) and extends these ideas to the LLM safety domain.
- The findings have direct implications for MLLM safety evaluation: current visual safety alignment is far weaker than textual alignment.
- The layer-feature analysis methodology can inspire defensive approaches: strengthening adversarial robustness in shallow layers may constitute an effective defense strategy.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First systematic analysis of visual jailbreaking attack transferability from both layer-feature and spectral perspectives.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 3 architecture types, 8 models, and 3 datasets with complete ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Analysis is clear, visualizations are rich, and the logical chain is complete.
- **Value**: ⭐⭐⭐⭐ Provides important reference value for MLLM red-teaming and safety research.

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)
- [\[CVPR 2026\] GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer](geco-srt_geometry-aware_continual_adaptation_for_robotic_cross-task_sim-to-real_.md)
- [\[CVPR 2026\] Language-Grounded Decoupled Action Representation for Robotic Manipulation](language-grounded_decoupled_action_representation_for_robotic_manipulation.md)
- [\[CVPR 2026\] ForceVLA2: Unleashing Hybrid Force-Position Control with Force Awareness for Contact-Rich Manipulation](forcevla2_unleashing_hybrid_force-position_control_with_force_awareness_for_cont.md)
- [\[CVPR 2026\] DeepSketcher: Internalizing Visual Manipulation for Multimodal Reasoning](deepsketcher_internalizing_visual_manipulation_for_multimodal_reasoning.md)

<!-- RELATED:END -->
