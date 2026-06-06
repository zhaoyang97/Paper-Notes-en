---
title: >-
  [Paper Note] Probing Cross-modal Information Hubs in Audio-Visual LLMs
description: >-
  [ICML 2026][Audio & Speech][AVLLM] By combining causal tracing and a unimodal-dominant framework, the authors reveal the existence of hidden "cross-modal sink tokens" in audio-visual LLMs…
tags:
  - "ICML 2026"
  - "Audio & Speech"
  - "AVLLM"
  - "attention sink"
  - "cross-modal information"
  - "causal tracing"
  - "hallucination mitigation"
date: 2026-05-08
content_hash: e0d5aec246b56680
---

# Probing Cross-modal Information Hubs in Audio-Visual LLMs

**Conference**: ICML 2026  
**arXiv**: [2605.10815](https://arxiv.org/abs/2605.10815)  
**Code**: https://github.com/kaistmm/crossmodal-hub  
**Area**: Multimodal VLM / Mechanistic Interpretability / Audio-Visual LLM  
**Keywords**: AVLLM, attention sink, cross-modal information, causal tracing, hallucination mitigation

## TL;DR
By combining causal tracing and a unimodal-dominant framework, the authors reveal the existence of hidden "cross-modal sink tokens" in audio-visual LLMs, where the vast majority of cross-modal information is concentrated. Based on this, they propose a training-free attention amplification strategy that significantly alleviates object hallucinations.

## Background & Motivation

**Background**: Audio-visual large language models (AVLLMs) have adopted a unified architecture where audio encodings, video encodings, and text tokens are temporally interleaved and fed into the LLM backbone. This design underpins models such as Qwen2.5/3-Omni and the video-SALMONN series, and is considered key to achieving "all-scenario multimodal reasoning."

**Limitations of Prior Work**: While there is extensive mechanistic interpretability research (causal tracing, sparse autoencoders, circuit discovery) on text and vision LLMs, the internal mechanism by which AVLLMs fuse two non-text modalities remains a black box. This makes it difficult to pinpoint the source of hallucinations or conduct safety audits.

**Key Challenge**: Audio and video interact bidirectionally, with either modality potentially injecting semantics into the other via self-attention. However, the authors find that the location of sink tokens in AVLLMs is not layer-stable as in LVLMs, rendering traditional layer-wise localization methods ineffective.

**Goal**: To answer two concrete sub-questions: (1) Where is cross-modal information stored? Is it in object-aligned tokens or sink tokens? (2) Is there functional differentiation within sink tokens?

**Key Insight**: The authors devise a "unimodal-dominant" filtering strategy, retaining only those samples where the joint prediction equals the unimodal prediction but differs from the other modality (e.g., vision-dominant). Such samples naturally indicate the direction of information flow: the dominated modality must have transferred information to the other modality's tokens. Thus, causal tracing on non-dominant tokens clearly reveals which positions carry "external signals."

**Core Idea**: Sink tokens are subdivided into unimodal sinks and cross-modal sinks based on which modality attends to them. Only the latter serve as true cross-modal information hubs. Amplifying the attention weights of cross-modal sinks during decoding, without any retraining, substantially reduces hallucinations.

## Method

### Overall Architecture
The analysis pipeline consists of three stages: (i) On VGGSound, use 20-way multiple-choice questions to select audio-dominant and video-dominant samples, retaining only those where $\hat y_{av}=\hat y_a \neq \hat y_v$ or its dual; (ii) Measure the indirect effect (IE) across three forward passes: clean, corrupted (dominant modality zeroed out), and corrupted-with-restoration (patching the clean hidden states onto non-dominant tokens); (iii) Partition candidate tokens into object, sink, random, and all non-dominant categories, and compare which has the highest IE. In downstream applications, sink tokens are further split by whether they are attended by the other modality (cross-modal) or the same modality (unimodal), and only cross-modal sinks have their attention amplified during decoding, yielding a training-free hallucination suppressor.

### Key Designs

1. **Unimodal-Dominant Causal Tracing Framework**:

    - **Function**: Reliably localizes "where information is transferred" in bidirectional AVLLMs.
    - **Mechanism**: Use the condition $\hat y_{av}=\hat y_a \neq \hat y_v$ to select audio-dominant samples, zero out audio to create a corrupted run, then patch a subset $S$ of video token hidden states $h_S^{\text{clean}}$ from the clean run. If prediction is restored, $S$ must have absorbed audio information in the clean run. Define $\text{IE}_{\text{clean}}(S)=P_{h_S^{\text{clean}}}[o_{\text{clean}}]-P[o_{\text{clean}}]$ and its dual $\text{IE}_{\text{corrupt}}(S)$ as complementary metrics.
    - **Design Motivation**: Traditional causal tracing in LLM/LVLMs only tracks unidirectional text-to-other-modality flow. In AVLLMs, both audio and video can be sources, so tracing all samples is noisy. The "dominant sample" approach naturally specifies source and target, equivalent to an observed causal intervention experiment.

2. **Redefinition of Global Sink Tokens**:

    - **Function**: Obtains a stable set of sink tokens in AVLLMs where sink positions drift across layers.
    - **Mechanism**: Unlike LVLMs, AVLLMs have sink token positions that change at each layer. Instead of layer-wise selection, the frequency with which each token is identified as a sink across all layers is tallied, and the top $|\mathcal T|/N$ (with $N\in\{2,3,4\}$ controlling sparsity) are chosen as global sinks. Sink status is still determined by abnormally high activation in predefined sink dimensions.
    - **Design Motivation**: Layer-wise sinks mixed with dense non-sinks distort IE attribution. Aggregating by frequency preserves sparsity and yields a stable subset for causal tracing; in experiments, patching the sink subset yields significantly higher IE than object/random baselines.

3. **Cross-modal Sink Tokens and Training-Free Hallucination Mitigation**:

    - **Function**: Further splits sinks into "strongly attended by own modality (unimodal)" and "strongly attended by the other modality (cross-modal)", and modulates attention accordingly.
    - **Mechanism**: For each sink token, compute the average attention weight from the same vs. cross modality; assign to the category with higher proportion. During generation, multiply the attention matrix entries for cross-modal sink tokens in the LLM, making the model rely more on already fused cross-modal summaries rather than local tokens from each modality.
    - **Design Motivation**: Hallucinations often arise from LLMs over-relying on local noise from a single modality. Cross-modal sinks serve as "trustworthy fused summaries"; amplifying them steers inference toward facts jointly supported by both modalities. This intervention is entirely at the attention weight level, requiring no parameter changes or extra training.

### Loss & Training
No training is involved in the analysis phase—pure forward passes with hooks; hallucination mitigation is also inference-only, introducing just a scalar modulation coefficient. All experiments are conducted directly on five open-source checkpoints: Qwen2.5-Omni (7B/3B), video-SALMONN-o1 (7B), and video-SALMONN2+ (7B/3B).

## Key Experimental Results

### Main Results

Patching effects of different token subsets (higher IE indicates more cross-modal information carried; audio-dominant setting, values from Table 1):

| Model | All Non-dominant (Upper Bound) | Object | Sink (N=2) | Random (N=2) |
|---|---|---|---|---|
| Qwen2.5-Omni 7B | 9.61 / 5.28 | 5.04 / 2.44 | 6.24 / 2.94 | 4.24 / 2.37 |
| Qwen2.5-Omni 3B | 7.83 / 3.48 | 3.53 / 1.12 | 6.99 / 2.70 | 4.05 / 1.20 |
| video-SALMONN-o1 7B | 35.55 / 33.18 | 16.22 / 15.06 | 25.33 / 22.73 | 20.43 / 18.11 |
| video-SALMONN2+ 7B | 6.45 / 5.27 | 3.78 / 3.93 | 4.79 / 4.20 | 4.21 / 4.01 |

(Values are $\text{IE}_{\text{clean}}$ / $\text{IE}_{\text{corrupt}}$; under comparable token counts, sinks consistently outperform object and random.)

### Ablation Study

| Configuration | Key Findings | Meaning |
|---|---|---|
| Sink N=2/3/4 | Halving token count only slightly reduces IE | Sink information is highly concentrated; sparsity does not hurt performance |
| Object token | Only slightly better than random | Object-aligned tokens are not the main storage location, refuting the object-centric hypothesis from LVLMs |
| Cross-modal sink vs. unimodal sink | The former has significantly higher IE | Functional differentiation exists within sinks; true hubs are cross-modal sinks |

### Key Findings
- All five models consistently indicate that cross-modal information storage follows a "sink-centric" rather than "object-centric" hypothesis, in sharp contrast to LVLMs where object information is stored in object tokens.
- Sink positions in AVLLMs drift across layers, implying that interpretability conclusions from LVLMs cannot be directly transferred; frequency-aggregated global sinks provide a more portable definition.
- Amplifying cross-modal sink attention yields a clear reduction in object hallucinations without retraining, validating the "mechanism understanding → engineering intervention" loop.

## Highlights & Insights
- Using "unimodal dominance" as a natural tool for causal intervention cleverly sidesteps the challenge of undirected tracing in bidirectional interactions; this approach is also applicable to future LLMs with three or more modalities.
- The unimodal vs. cross-modal sink dichotomy advances the study of "what is an attention sink" from a positional to a functional perspective; cross-modal sinks can be seen as model-learned "multimodal summary registers."
- The training-free hallucination mitigation method requires only a few lines of hooks at the attention layer, with near-zero overhead, making it suitable for industrial deployment and highly interpretable—it can explain "why the model is now more trustworthy."

## Limitations & Future Work
- Experiments only cover VGGSound-like samples and multiple-choice protocols; whether sink functional differentiation holds in real open-ended QA remains to be verified.
- Only "audio-video" modalities are analyzed; models like Qwen3-Omni already include speech and image generation, and whether the cross-modal sink concept extends to these output ports is unknown.
- The attention amplification coefficient is currently a uniform scalar; future work could explore per-head/per-layer adaptation, or even learn a lightweight gating mechanism for dynamic amplification.

## Related Work & Insights
- **vs Neo et al. (LVLM object-centric)**: They found that LVLMs store object information in object tokens; this work shows that AVLLMs instead have sink tokens, suggesting that internal structures differ more across modality combinations than previously thought.
- **vs Kang/Luo (LVLM sink)**: Prior work only found that sinks aggregate global information; this work further splits sinks into unimodal and cross-modal types, pushing research granularity to the sub-class level.
- **vs retraining-based hallucination mitigation (RLHF / DPO)**: This work leaves parameters untouched and requires no preference data, offering a zero-data-cost alternative, especially suitable for urgent post-deployment fixes.

## Rating
- Novelty: ⭐⭐⭐⭐ First to extend causal tracing to bidirectional multimodal scenarios and propose the cross-modal sink concept.
- Experimental Thoroughness: ⭐⭐⭐⭐ Consistent validation on five open-source AVLLMs, with rigorous token count-matched controls.
- Writing Quality: ⭐⭐⭐⭐ Clear framework and illustrations, with a hypothesis-verification structure.
- Value: ⭐⭐⭐⭐ Provides both interpretability insights and a practical, training-free hallucination mitigation method, closing the loop from mechanism study to engineering intervention.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] CCFQA: A Benchmark for Cross-Lingual and Cross-Modal Speech and Text Factuality Evaluation](../../AAAI2026/audio_speech/ccfqa_a_benchmark_for_cross-lingual_and_cross-modal_speech_and_text_factuality_e.md)
- [\[CVPR 2026\] Semantic Audio-Visual Navigation in Continuous Environments](../../CVPR2026/audio_speech/semantic_audio-visual_navigation_in_continuous_environments.md)
- [\[ACL 2026\] Protecting Bystander Privacy via Selective Hearing in Audio LLMs](../../ACL2026/audio_speech/protecting_bystander_privacy_via_selective_hearing_in_audio_llms.md)
- [\[ACL 2026\] Omni-Embed-Audio: Leveraging Multimodal LLMs for Robust Audio-Text Retrieval](../../ACL2026/audio_speech/omni-embed-audio_leveraging_multimodal_llms_for_robust_audio-text_retrieval.md)
- [\[ICCV 2025\] Zero-AVSR: Zero-Shot Audio-Visual Speech Recognition with LLMs by Learning Language-Agnostic Speech Representations](../../ICCV2025/audio_speech/zero-avsr_zero-shot_audio-visual_speech_recognition_with_llms_by_learning_langua.md)

</div>

<!-- RELATED:END -->
