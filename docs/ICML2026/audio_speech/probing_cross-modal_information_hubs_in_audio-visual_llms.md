---
title: >-
  [Paper Note] Probing Cross-modal Information Hubs in Audio-Visual LLMs
description: >-
  [ICML 2026][Audio & Speech][AVLLM] The authors reveal hidden hubs termed "cross-modal sink tokens" in Audio-Visual LLMs using a causal tracing and unimodal dominance framework. Most cross-modal information is condensed o…
tags:
  - "ICML 2026"
  - "Audio & Speech"
  - "AVLLM"
  - "attention sink"
  - "cross-modal information"
  - "causal tracing"
  - "hallucination mitigation"
date: 2026-05-08
content_hash: 2e3b19d1c7d5ee12
---

# Probing Cross-modal Information Hubs in Audio-Visual LLMs

**Conference**: ICML 2026  
**arXiv**: [2605.10815](https://arxiv.org/abs/2605.10815)  
**Code**: https://github.com/kaistmm/crossmodal-hub  
**Area**: Multimodal VLM / Mechanistic Interpretability / Audio-Visual LLM  
**Keywords**: AVLLM, attention sink, cross-modal information, causal tracing, hallucination mitigation

## TL;DR
The authors reveal hidden hubs termed "cross-modal sink tokens" in Audio-Visual LLMs using a causal tracing and unimodal dominance framework. Most cross-modal information is condensed on these tokens. Based on this, they propose a training-free attention amplification strategy that significantly alleviates object hallucinations.

## Background & Motivation

**Background**: Audio-Visual Large Language Models (AVLLM) interweave audio tokens, video tokens, and text tokens temporally before feeding them into an LLM backbone. This has become the unified architecture for series like Qwen2.5/3-Omni and video-SALMONN, regarded as key to achieving "all-scenario multimodal reasoning."

**Limitations of Prior Work**: While extensive mechanistic interpretability research (causal tracing, sparse autoencoders, circuit discovery) exists for text LLMs and vision LLMs, the internal fusion mechanism of the two non-text modalities in AVLLMs remains a black box. This makes it difficult to locate the roots of hallucinations or perform safety audits.

**Key Challenge**: Audio and video interact bi-directionally, where either side can inject semantics into the other via self-attention. The authors found that the positions of sink tokens in AVLLMs are not as layer-stable as those in LVLMs, rendering traditional layer-wise localization methods ineffective.

**Goal**: To answer two specific sub-questions: (1) In which tokens is cross-modal information actually stored? Is it in object-aligned tokens or sink tokens? (2) Is there functional differentiation within sink tokens?

**Key Insight**: The authors developed a "unimodal dominance" filtering strategy, retaining only samples where the joint prediction equals the unimodal prediction but differs from the other modality's prediction (e.g., video-dominant). Such samples naturally indicate information flow: the non-dominant modality must move its information into the tokens of the dominant modality. Thus, causal tracing on non-dominant tokens clearly reveals which positions carry "external signals."

**Core Idea**: Sink tokens are categorized into unimodal sinks and cross-modal sinks based on "which modality they attend to." The latter are the true cross-modal hubs. Consequently, magnifying the attention weights of cross-modal sinks during decoding substantially reduces hallucinations at zero training cost.

## Method

### Overall Architecture
The analysis pipeline consists of three stages: (i) Filtering audio-dominant and video-dominant samples on VGGSound using a 20-choice question format, keeping only $\hat y_{av}=\hat y_a \neq \hat y_v$ or its dual; (ii) Measuring the indirect effect across three forward passes: clean / corrupted (dominant modality zeroed) / corrupted-with-restoration (patching clean hidden states to non-dominant tokens); (iii) Partitioning candidate tokens into object / sink / random / all non-dominant categories to compare Indirect Effect (IE). In the downstream application, sink tokens are subdivided into cross-modal and unimodal based on cross-modal attention, and attention is amplified only for cross-modal sinks to create a training-free hallucination suppressor.

### Key Designs

1.  **Unimodal Dominance Causal Tracing Framework**:
    - **Function**: Reliably locates "where information is moved" in bi-directional AVLLMs.
    - **Mechanism**: Filters audio-dominant samples using $\hat y_{av}=\hat y_a \neq \hat y_v$, zeros out the audio to form a corrupted run, and then patches back the hidden states $h_S^{\text{clean}}$ of a video token subset $S$ from the clean run. If the prediction is restored, it indicates $S$ absorbed audio information in the clean run. Two complementary metrics, $\text{IE}_{\text{clean}}(S)=P_{h_S^{\text{clean}}}[o_{\text{clean}}]-P[o_{\text{clean}}]$ and the dual $\text{IE}_{\text{corrupt}}(S)$, are defined.
    - **Design Motivation**: Traditional causal tracing in LLM/LVLM only tracks one-way flow (text $\rightarrow$ other modalities). In AVLLMs, both audio and video act as signal sources; direct tracing on all samples would be drowned in noise. "Dominant samples" naturally specify the source and target, equivalent to an observational causal intervention experiment.

2.  **Global Sink Token Redefinition**:
    - **Function**: Obtains a stable set of sinks in AVLLMs where sink positions drift across layers.
    - **Mechanism**: Unlike LVLMs, AVLLM sink positions change every layer. Instead of per-layer sinks, the authors count the frequency of each token being identified as a sink across all layers. The top $|\mathcal T|/N$ tokens by frequency are taken as global sinks ($N\in\{2,3,4\}$ controls sparsity). Sinks themselves are identified by "abnormally large activation magnitudes in predefined sink dimensions."
    - **Design Motivation**: Mixing per-layer sinks with dense non-sinks distorts IE attribution. Aggregating by frequency maintains sink sparsity while stabilizing causal tracing toward a consistent subset. Experiments show the IE of global sink subsets is significantly higher than object/random baselines after patching.

3.  **Cross-modal Sink Token and Training-free Hallucination Mitigation**:
    - **Function**: Splits sinks into "strongly attending to own modality (unimodal)" vs. "strongly attending to the other modality (cross-modal)" and modulates attention accordingly.
    - **Mechanism**: Calculates average attention weights from same-modality vs. cross-modality for each sink token; the higher ratio determines the category. During generation, a multiplier is applied to cross-modal sink tokens in the LLM's attention matrix. This forces the model to rely more on fused cross-modal summaries rather than local tokens of individual modalities.
    - **Design Motivation**: Hallucinations often stem from the LLM biasing toward local noise of a single modality. Cross-modal sinks act as "trusted fused summaries." Amplifying them pulls reasoning back to factual regions supported by both modalities. This intervention occurs entirely on attention weights without parameter updates or additional training.

### Loss & Training
The analysis phase involves no training (pure forward pass + hooks). The hallucination mitigation phase is an inference-only intervention, introducing only a scalar adjustment coefficient. All experiments were performed directly on five open-source checkpoints: Qwen2.5-Omni (7B/3B), video-SALMONN-o1 (7B), and video-SALMONN2+ (7B/3B).

## Key Experimental Results

### Main Results

Patching performance of different token subsets (higher IE indicates more cross-modal information; audio-dominant setting, values from Table 1):

| Model | All Non-dominant (Upper) | Object | Sink (N=2) | Random (N=2) |
|---|---|---|---|---|
| Qwen2.5-Omni 7B | 9.61 / 5.28 | 5.04 / 2.44 | 6.24 / 2.94 | 4.24 / 2.37 |
| Qwen2.5-Omni 3B | 7.83 / 3.48 | 3.53 / 1.12 | 6.99 / 2.70 | 4.05 / 1.20 |
| video-SALMONN-o1 7B | 35.55 / 33.18 | 16.22 / 15.06 | 25.33 / 22.73 | 20.43 / 18.11 |
| video-SALMONN2+ 7B | 6.45 / 5.27 | 3.78 / 3.93 | 4.79 / 4.20 | 4.21 / 4.01 |

(Values are $\text{IE}_{\text{clean}}$ / $\text{IE}_{\text{corrupt}}$; sinks consistently outperform object and random tokens under equivalent counts).

### Ablation Study

| Configuration | Key Finding | Meaning |
|---|---|---|
| Sink N=2/3/4 | Halving tokens leads to only slight IE drop | Sink information is highly concentrated and robust to sparsity |
| Object token | Only slightly better than random | Object-aligned tokens are not primary storage locations; refutes LVLM object-centric hypothesis |
| Cross-modal vs. unimodal sink | Former has significantly higher IE | Functional differentiation exists; cross-modal sinks are the true hubs |

### Key Findings
- Horizontal consistency across five models shows cross-modal information storage follows a "sink-centric" rather than "object-centric" hypothesis, contrary to findings in LVLMs.
- AVLLM sink positions drift across layers, implying that interpretability conclusions from LVLMs cannot be directly transferred; frequency-aggregated global sinks offer a more portable definition.
- Amplifying cross-modal sink attention results in a distinct drop in object hallucinations without retraining, validating the "mechanistic understanding $\rightarrow$ engineering modification" closed loop.

## Highlights & Insights
- Using "unimodal dominance" as a natural causal intervention tool cleverly bypasses the difficulty of directional tracking in bi-directional interactions. This approach is applicable to future LLMs with more than three modalities.
- Proposing the unimodal vs. cross-modal sink dichotomy moves the study of "what attention sinks are" from "position" to "function." Cross-modal sinks can be viewed as model-learned "multimodal summary registers."
- The training-free hallucination mitigation requires only a few lines of hooks in the attention layer, entailing near-zero overhead. It is suitable for industrial deployment and highly interpretable, explaining "why the model is now more reliable."

## Limitations & Future Work
- Experiments only cover VGGSound-type samples and multiple-choice protocols; functional differentiation of sinks in open-ended Q&A needs further validation.
- Analysis is limited to "audio-visual" modalities; it is unknown if the concept of cross-modal sinks extends to output ports like speech or image generation in models like Qwen3-Omni.
- The attention amplification coefficient is currently a uniform scalar; future work could explore per-head/per-layer adaptation or learning a lightweight gating mechanism for dynamic scaling.

## Related Work & Insights
- **vs. Neo et al. (LVLM object-centric)**: They found LVLMs store object information in object tokens; this paper proves AVLLMs use sink tokens instead, suggesting internal structure differences between modality combinations are larger than expected.
- **vs. Kang/Luo (LVLM sink)**: Existing work only found sinks aggregate global information; this paper further splits sinks into unimodal/cross-modal categories, pushing research to a sub-class level.
- **vs. Retraining-based mitigation (RLHF / DPO)**: This paper provides a zero-data alternative without parameter updates or preference data collection, particularly suitable for emergency patches post-deployment.

## Rating
- Novelty: ⭐⭐⭐⭐ First to extend causal tracing to bi-directional multimodal scenarios and propose cross-modal sinks.
- Experimental Thoroughness: ⭐⭐⭐⭐ Consistent verification across five open-source AVLLMs with rigorous token-count alignment.
- Writing Quality: ⭐⭐⭐⭐ Intuitive framework and diagrams; clear hypothesis-verification structure.
- Value: ⭐⭐⭐⭐ Provides both interpretability insights and a practical training-free mitigation method, closing the loop between mechanism and engineering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] CCFQA: A Benchmark for Cross-Lingual and Cross-Modal Speech and Text Factuality Evaluation](../../AAAI2026/audio_speech/ccfqa_a_benchmark_for_cross-lingual_and_cross-modal_speech_and_text_factuality_e.md)
- [\[ICML 2026\] Do Audio LLMs Listen or Read? Analyzing and Mitigating Paralinguistic Failures with VoxParadox](do_audio_llms_listen_or_read_analyzing_and_mitigating_paralinguistic_failures_wi.md)
- [\[ICML 2026\] JAEGER: Joint 3D Audio-Visual Grounding and Reasoning in Simulated Physical Environments](jaeger_joint_3d_audio-visual_grounding_and_reasoning_in_simulated_physical_envir.md)
- [\[ICML 2026\] Towards Understanding Modality Interaction in Multimodal Language Models via Partial Information Decomposition](towards_understanding_modality_interaction_in_multimodal_language_models_via_par.md)
- [\[CVPR 2026\] Semantic Audio-Visual Navigation in Continuous Environments](../../CVPR2026/audio_speech/semantic_audio-visual_navigation_in_continuous_environments.md)

</div>

<!-- RELATED:END -->
