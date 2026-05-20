---
title: >-
  [Paper Note] When One Modality Sabotages the Others: A Diagnostic Lens on Multimodal Reasoning
description: >-
  [NeurIPS 2025][Multimodal VLM][modality sabotage] This paper introduces the concept of *modality sabotage* as a diagnostic failure mode…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "modality sabotage"
  - "multimodal fusion"
  - "sentiment recognition"
  - "interpretability"
  - "diagnostic framework"
date: 2026-05-08
content_hash: 133c19e2cad7a39a
---

# When One Modality Sabotages the Others: A Diagnostic Lens on Multimodal Reasoning

**Conference**: NeurIPS 2025
**arXiv**: [2511.02794](https://arxiv.org/abs/2511.02794)  
**Code**: None  
**Area**: Multimodal VLM
**Keywords**: modality sabotage, multimodal fusion, sentiment recognition, interpretability, diagnostic framework

## TL;DR
This paper introduces the concept of *modality sabotage* as a diagnostic failure mode, proposes a lightweight and model-agnostic evaluation layer that treats each modality as an independent agent, and exposes "contributors" versus "saboteurs" through simple fusion. Applied to multimodal sentiment recognition benchmarks, the framework reveals systematic differences in per-modality reliability.

## Background & Motivation

Multimodal large language models have advanced rapidly on tasks combining vision, language, and audio, yet their decision-making processes remain black boxes: users cannot determine which data stream the system relies on, how conflicting evidence is resolved, or whether a single sensor dominates the final prediction.

Prior work has addressed related phenomena such as *modality collapse* (VLMs over-relying on text) and *unimodal bias* (one modality dominating fusion across an entire dataset), but these describe system-level trends. This paper proposes a distinct, diagnostic failure mode—**modality sabotage**: at the instance level, a high-confidence unimodal error not only fails on its own but actively overrides other evidence and biases the fused prediction.

The root cause lies in the fact that existing multimodal systems emphasize cross-modal feature interaction and modality complementation while offering almost no insight into how cues map to constructs or how conflicts are resolved. The core idea is to treat each modality as an independent agent and construct a diagnostic layer from its outputs to audit fusion dynamics.

## Method

### Overall Architecture

The framework adopts a *modality-as-agent* design. For each video clip, descriptive inputs are extracted independently for three modalities—text (T), audio (A), and vision (V)—as well as a joint view (TAV). Each modality agent independently produces candidate labels, confidence scores (1–100), and a data quality report; a simple aggregation mechanism then produces the fused decision.

### Key Designs

1. **Modality-Specific Input Extraction**: Each modality receives purely descriptive inputs, avoiding direct emotion inference. Text (T) uses Whisper ASR transcripts; audio (A) uses Qwen-Audio to extract non-lexical descriptors such as prosody, voice quality, and articulation; vision (V) uses OpenFace to compute facial action units (AUs), selects AU-peak frames, and generates objective descriptions with GPT-4 Vision (covering facial expressions, posture, and gestures, with psychological-state attribution explicitly prohibited). This design ensures that each modality supplies objective descriptions rather than direct labels.

2. **Fusion and Attribution Mechanism**: For each sample, the four agents T, A, V, and TAV each return a ranked set of candidate labels with confidence scores and a data quality report. The fusion formula is $\tilde{s}(y) = \sum_m w_m S_m(y)$, where $w_m = 1$ by default (a quality-weighted variant $w_m = q_m$ is also evaluated). Normalized scores are obtained as $p(y) = \tilde{s}(y) / \sum_{y'} \tilde{s}(y')$, and the ranking induced by $p(y)$ is assessed.

3. **Modality Sabotage Detection**: Two sabotage types are defined. **Potential sabotage**: modality $m$ satisfies (i) $c_m \geq \tau$ (high confidence, $\tau = 0.70$) and (ii) $y_m \neq y^*$ (incorrect prediction). **Successful sabotage**: additionally requires (iii) $\hat{y} = y_m$ (the fused model ultimately adopts $m$'s erroneous prediction). This framework makes attribution explicit at the instance level—identifying which modality contributed correctly and which sabotaged the outcome.

### Loss & Training

This paper involves no model training; the framework is purely diagnostic. GPT-5-nano and GPT-4o-mini serve as backbones, and experiments are conducted on three sentiment recognition benchmarks: MER, MELD, and IEMOCAP.

## Key Experimental Results

### Main Results

| Dataset / Model | Base T1 | Fus T1 | Fus T2 | Fus T3 | Fus T4 | Fus T5 |
|----------------|---------|--------|--------|--------|--------|--------|
| MER / GPT-5-nano | 0.38 | 0.33 | 0.62 | 0.85 | 0.92 | 0.97 |
| MER / GPT-4o-mini | 0.35 | 0.23 | 0.52 | 0.75 | 0.83 | 0.85 |
| MELD / GPT-5-nano | 0.27 | 0.36 | 0.58 | 0.73 | 0.86 | 0.92 |
| MELD / GPT-4o-mini | 0.30 | 0.45 | 0.64 | 0.76 | 0.85 | 0.90 |
| IEMOCAP / GPT-5-nano | 0.28 | 0.29 | 0.47 | 0.62 | 0.73 | 0.76 |
| IEMOCAP / GPT-4o-mini | 0.28 | 0.24 | 0.43 | 0.60 | 0.70 | 0.72 |

### Ablation Study (Quality-Weighted vs. Confidence-Only Weighting)

| Dataset / Model | ΔT1 | ΔT2 | ΔT3 | ΔT4 | ΔT5 | Note |
|----------------|-----|-----|-----|-----|-----|------|
| MER / GPT-5-nano | +0.00 | +0.01 | +0.00 | −0.02 | +0.01 | Negligible difference |
| MELD / GPT-5-nano | −0.08 | −0.06 | −0.03 | −0.03 | −0.04 | Quality weighting hurts |
| IEMOCAP / GPT-5-nano | −0.05 | −0.07 | −0.07 | −0.02 | +0.03 | Quality weighting negative |
| MER / GPT-4o-mini | −0.03 | +0.00 | +0.00 | +0.02 | +0.03 | Slight gain at high Top-k |

### Key Findings

- **Audio is the primary saboteur; text is the primary contributor.** This pattern appears consistently across datasets.
- Top-k coverage far exceeds Top-1 accuracy (e.g., on MER, from 0.33 to 0.97), indicating that fusion preserves recoverable uncertainty—the correct label typically remains among the model's top hypotheses.
- Self-reported data quality signals capture certain aspects of model self-awareness but correlate only weakly with correctness, introducing noise that degrades Top-1 accuracy.
- Cross-dataset differences in modality reliability align with dataset characteristics: MER suffers from noisy ASR but has rich visual cues; MELD's sitcom style produces visually misleading signals; IEMOCAP's seated dyadic setting limits visual reliability.
- The most striking improvement is observed on MELD with GPT-4o-mini, where fused Top-1 accuracy rises from 0.30 to 0.45 (+0.15).

## Highlights & Insights

- The concept of modality sabotage fills a semantic gap between modality collapse and unimodal bias, providing an instance-level rather than system-level diagnostic perspective.
- The framework is extremely lightweight—requiring no retraining or architectural modifications—and can be applied to any MLLM through prompting and simple aggregation alone.
- The use of Top-k inference is particularly apt: it is employed not to inflate accuracy through guessing, but to diagnose whether the model still preserves a correct internal ranking even when sabotaged.
- The negative results from the quality-weighting ablation are themselves a significant finding, demonstrating that models' self-calibration capabilities remain insufficient.

## Limitations & Future Work

- Validation is limited to sentiment recognition; generalization to other multimodal tasks such as VQA and video understanding remains unexplored.
- Self-reported confidence scores are used rather than internal model probabilities or logits, potentially introducing additional noise.
- The fusion mechanism is overly simple (linear aggregation); how modality sabotage manifests under more sophisticated fusion strategies is unknown.
- "Successful sabotage" does not establish strict causality—multiple agents may jointly support the same erroneous label.
- The relatively small number of label categories across datasets limits the interpretive power of the Top-k analysis.

## Related Work & Insights

- This work is related to modality collapse (VLMs over-relying on text in VQA) and debiasing methods such as RUBi, but focuses on instance-level diagnosis rather than system-level mitigation.
- Research in psychology and affective computing has shown that audio and visual cues carry complementary emotional information (facial expressions correlate with valence; speech acoustics track arousal), yet such work typically analyzes unimodal contributions in isolation.
- Implication: multimodal system deployment should incorporate modality-level interpretability and conflict detection mechanisms, rather than optimizing solely for aggregate post-fusion accuracy.

## Rating

- Novelty: ⭐⭐⭐⭐ (The modality sabotage concept is original; the diagnostic perspective is distinctive)
- Experimental Thoroughness: ⭐⭐⭐ (Three datasets and two backbones, but confined to sentiment recognition)
- Writing Quality: ⭐⭐⭐⭐ (Concepts are clearly defined; formalization is rigorous)
- Value: ⭐⭐⭐⭐ (Provides a practical tool for trustworthiness auditing of multimodal systems)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] When Semantics Mislead Vision: Mitigating Large Multimodal Models Hallucinations](when_semantics_mislead_vision_mitigating_large_multimodal_models_hallucinations_.md)
- [\[ICLR 2026\] SpinBench: Perspective and Rotation as a Lens on Spatial Reasoning in VLMs](../../ICLR2026/multimodal_vlm/spinbench_perspective_and_rotation_as_a_lens_on_spatial_reasoning_in_vlms.md)
- [\[ICLR 2026\] Through the Lens of Contrast: Self-Improving Visual Reasoning in VLMs](../../ICLR2026/multimodal_vlm/through_the_lens_of_contrast_self-improving_visual_reasoning_in_vlms.md)
- [\[ACL 2026\] When Slower Isn't Truer: Inverse Scaling Law of Truthfulness in Multimodal Reasoning](../../ACL2026/multimodal_vlm/when_slower_isn39t_truer_inverse_scaling_law_of_truthfulness_in_multimodal_reaso.md)
- [\[NeurIPS 2025\] MDReID: Modality-Decoupled Learning for Any-to-Any Multi-Modal Object Re-Identification](mdreid_modality-decoupled_learning_for_any-to-any_multi-modal_object_re-identifi.md)

</div>

<!-- RELATED:END -->
