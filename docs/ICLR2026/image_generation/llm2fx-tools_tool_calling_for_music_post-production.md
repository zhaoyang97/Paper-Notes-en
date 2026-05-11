---
title: >-
  [Paper Note] LLM2Fx-Tools: Tool Calling for Music Post-Production
description: >-
  [ICLR 2026][Image Generation][Fx-chain estimation] This paper proposes LLM2Fx-Tools, the first framework that applies LLM tool calling to audio effect modules. It leverages a multimodal LLM to understand audio inputs…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "Fx-chain estimation"
  - "tool calling"
  - "chain-of-thought reasoning"
  - "music post-production"
  - "multimodal LLM"
date: 2026-05-08
content_hash: dc7a202d975d791a
---

# LLM2Fx-Tools: Tool Calling for Music Post-Production

**Conference**: ICLR 2026
**arXiv**: [2512.01559](https://arxiv.org/abs/2512.01559)
**Code**: [Demo](https://seungheondoh.github.io/llm2fx-tools-demo/)
**Area**: Image Generation
**Keywords**: Fx-chain estimation, tool calling, chain-of-thought reasoning, music post-production, multimodal LLM

## TL;DR
This paper proposes LLM2Fx-Tools, the first framework that applies LLM tool calling to audio effect modules. It leverages a multimodal LLM to understand audio inputs, employs CoT reasoning to select effect types, determine processing order, and estimate parameters, enabling interpretable and controllable music post-production.

## Background & Motivation
- **Background**: Audio effects (Fx) processing is central to music post-production but demands substantial domain expertise.
- **Limitations of Prior Work**: Existing Fx-chain estimation methods suffer from three key limitations: gradient-based methods require differentiable modules; regression-based methods operate under fixed configurations without dynamic effect selection; and none provide user-interpretable explanations.
- **Key Challenge**: LLM capabilities in instruction following, CoT reasoning, and tool calling offer new opportunities to address flexibility and interpretability challenges. The prior LLM2Fx work supported only single effects (EQ and reverb) without explicit tool calling or CoT.
- **Goal**: Develop a flexible, interpretable Fx-chain estimation framework based on LLM tool calling.

## Method

### Overall Architecture
LLM2Fx-Tools is built on Qwen3-4B and accepts instructions, dry audio, and reference audio as inputs, producing CoT reasoning, an executable Fx-chain (tool call sequence), and a natural language response.

### Key Designs

1. **Audio Understanding Architecture**: Audio features are extracted using Fx-Encoder++ (pretrained via contrastive learning) and mapped to the LLM embedding space through a Transformer-based adapter (32 learnable query embeddings + cross-attention). The unified multimodal input sequence is formulated as: $[x_{\text{instruction}}, x_{\text{dry}}, x_{\text{ref}}, x_{\text{cot}}, \mathcal{C}, x_{\text{response}}]$

2. **CoT Fx-Chain Planning**: Reasoning is decomposed into four sub-tasks: ① user input analysis → ② effect module selection → ③ processing order determination → ④ parameter planning. The CoT output serves as the conditional context for tool calling.

3. **Number Token Loss (NTL)**: Standard cross-entropy is poorly suited for numerical prediction as it penalizes all errors equally. The Wasserstein-1 distance is introduced instead:
$$\mathcal{L}_{\text{NTL-WAS}} = \frac{1}{|\mathcal{I}_{\text{num}}|} \sum_{i \in \mathcal{I}_{\text{num}}} \sum_{v \in \mathcal{V}_{\text{num}}} \hat{P}_i(v) |y_i - \text{val}(v)|$$
   The final loss is: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \mathcal{L}_{\text{NTL}}$

4. **Robust Training**: Fx-Removal and Fx-Normalization preprocessing are applied to mitigate recording environment discrepancies. During training, dry audio is randomly masked (with probability $p_{\text{masking}}$) so the model can handle both reverse engineering and blind estimation.

### Loss & Training
Two-stage training:
- Stage 1 (Modality Alignment): Adapter only, LLM frozen, LR=1e-4, 100K steps.
- Stage 2 (LLM Fine-tuning): LoRA (rank=128, alpha=256), LR=5e-5, 400K steps, full dialogue data.

## Key Experimental Results

### Main Results (Reverse Engineering Fx-Chain Estimation)

| Method | Acc↑ | Order Corr.↑ | MAE↓ | MRS L/R↓ | AFx-Rep↑ | FxEnc↑ |
|--------|------|--------------|------|-----------|---------|--------|
| Regression | 55% | -0.03 | 0.20 | 3.81 | 0.62 | 0.64 |
| MultiTask | 61% | 0.00 | 0.23 | 3.17 | 0.63 | 0.66 |
| DeepAFx-ST | - | - | - | 1.75* | 0.62 | 0.66 |
| Gemini 2.5 Flash | 78% | 0.54 | 0.32 | 3.42 | 0.56 | 0.50 |
| **LLM2Fx-Tools** | **80%** | **0.56** | **0.23** | **3.13** | **0.68** | **0.67** |

### Ablation Study

| Configuration | Acc↑ | Order Corr.↑ | MAE↓ | MRS L/R↓ |
|---------------|------|--------------|------|-----------|
| LLM2Fx-Tools (Full) | 80% | 0.56 | 0.23 | 3.13 |
| w/o CoT | 67% | 0.49 | 0.24 | 3.34 |
| w/o NTL | 73% | 0.51 | 0.32 | 3.69 |
| w/o MST | 76% | 0.55 | 0.25 | 3.21 |

### Key Findings
- CoT substantially improves effect selection (67→80%) and ordering (0.49→0.56), making it the most critical component.
- NTL reduces parameter MAE from 0.32 to 0.23, contributing the most to numerical precision.
- Regression achieves the lowest MAE (0.20) but cannot select effects, resulting in worse perceptual distance overall.
- MUSHRA listening test: LLM2Fx-Tools scores 62.8, significantly outperforming Gemini (56.5) and DeepAFx-ST (54.8).
- LLM2Fx-Tools generalizes best on style transfer (AF=7.41 vs. second-best 7.62).

## Highlights & Insights
- This work is the first to introduce the LLM tool calling paradigm to audio effect processing, incorporating non-differentiable modules into a generative framework.
- CoT not only improves performance but also provides interpretable reasoning (i.e., explaining why a particular effect is chosen).
- Fx-chain ordering is critical to final audio quality — incorrect ordering degrades quality even when parameter estimates are accurate.
- The LP-Fx dataset (101K dialogues) provides foundational infrastructure for audio effect LLM research.

## Limitations & Future Work
- The tool set is limited to 9 modules and 26 parameters, whereas real-world post-production is considerably more complex.
- Data generation relies on LLM-synthesized dialogues, which may introduce distributional bias.
- The 4B parameter model may be insufficient for complex production scenarios.
- Iterative optimization (multi-turn dialogue for progressive effect refinement) remains unexplored.

## Related Work & Insights
- LLM2Fx's single-effect prediction motivated the extension to full Fx-chain estimation in this work.
- The tool calling paradigm (Toolformer, Gorilla) has been widely applied in vision and NLP; this paper is the first to transfer it to audio.
- This work establishes a new paradigm for interpretable, automated AI-assisted music production.

## Technical Details
- **Audio Source**: MedleyDB dataset (2,119 crosstalk-free audio files selected from 196 multi-track recordings).
- **Tool Environment**: 6 Pedalboard modules + 3 custom modules (9 modules total, 26 parameters), including compressor, distortion, reverb, delay, limiter, gain, three-band EQ, stereo widener, and panner.
- **LP-Fx Dataset**: 99,900 training + 900 test samples, stratified by Fx-chain length 1–9.
- Data generation uses LLM-synthesized dialogues with LLM-as-a-judge filtering to remove low-quality samples.
- NLG evaluation uses GPT-5 as judge, assessing tool call success rate, instruction following, and CoT quality.
- LLM2Fx-Tools achieves a tool call success rate of 99.8% (vs. 100% for Gemini 2.5 Flash).
- Qwen 2.5 Omni's zero-shot tool calling ability is extremely poor (only 0.2%), validating the necessity of fine-tuning.
- Regression models that fail to apply effects correctly score lower in MUSHRA than the no-effect baseline, indicating that incorrect application is worse than no application.
- The model supports both reverse engineering (with dry audio) and blind estimation (without dry audio).
- Style transfer experiments use MoisesDB and MedleyDB for cross-dataset generalization evaluation.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First audio effect tool calling framework; the combination of CoT and tool calling is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ MUSHRA listening tests + multi-dimensional metrics + ablation + style transfer + NLG evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ Task definitions are clear and method descriptions are complete.
- **Value**: ⭐⭐⭐⭐ Opens a new direction for LLM applications in audio post-production.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Melodia: Training-Free Music Editing Guided by Attention Probing in Diffusion Models](../../AAAI2026/image_generation/melodia_training-free_music_editing_guided_by_attention_probing_in_diffusion_mod.md)
- [\[AAAI 2026\] QuantVSR: Low-Bit Post-Training Quantization for Real-World Video Super-Resolution](../../AAAI2026/image_generation/quantvsr_low-bit_post-training_quantization_for_real-world_video_super-resolutio.md)
- [\[ICCV 2025\] DMQ: Dissecting Outliers of Diffusion Models for Post-Training Quantization](../../ICCV2025/image_generation/dmq_dissecting_outliers_of_diffusion_models_for_post-training_quantization.md)
- [\[NeurIPS 2025\] FairImagen: Post-Processing for Bias Mitigation in Text-to-Image Models](../../NeurIPS2025/image_generation/fairimagen_post-processing_for_bias_mitigation_in_text-to-image_models.md)
- [\[AAAI 2026\] Diff-V2M: A Hierarchical Conditional Diffusion Model with Explicit Rhythmic Modeling for Video-to-Music Generation](../../AAAI2026/image_generation/diff-v2m_a_hierarchical_conditional_diffusion_model_with_explicit_rhythmic_model.md)

</div>

<!-- RELATED:END -->
