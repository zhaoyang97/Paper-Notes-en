---
title: >-
  [Paper Note] A Unified Reasoning Framework for Holistic Zero-Shot Video Anomaly Analysis
description: >-
  [NeurIPS 2025][Interpretability][Video Anomaly Detection] A fully zero-shot, training-free video anomaly analysis framework that employs Intra-Task Reasoning (confidence-gated self-refinement) and Inter-Task Chaining (ca…
tags:
  - "NeurIPS 2025"
  - "Interpretability"
  - "Video Anomaly Detection"
  - "Zero-Shot"
  - "Chain-of-Thought Reasoning"
  - "VLM"
  - "Anomaly Localization & Understanding"
date: 2026-05-08
content_hash: e31f95f420f571fa
---

# A Unified Reasoning Framework for Holistic Zero-Shot Video Anomaly Analysis

**Conference**: NeurIPS 2025
**arXiv**: [2511.00962](https://arxiv.org/abs/2511.00962)
**Code**: [https://rathgrith.github.io/Unified_Frame_VAA/](https://rathgrith.github.io/Unified_Frame_VAA/)
**Area**: Interpretability
**Keywords**: Video Anomaly Detection, Zero-Shot, Chain-of-Thought Reasoning, VLM, Anomaly Localization & Understanding

## TL;DR
A fully zero-shot, training-free video anomaly analysis framework that employs Intra-Task Reasoning (confidence-gated self-refinement) and Inter-Task Chaining (cascaded prompt passing from temporal detection to spatial localization to semantic understanding), achieving comprehensive improvements of 4–6% AUC over prior zero-shot methods across 4 benchmarks.

## Background & Motivation
Video anomaly analysis has traditionally focused solely on temporal detection (producing frame-level anomaly scores), lacking spatial localization (where the anomaly occurs) and semantic explanation (why it is anomalous). Existing VLM-based methods either address only a single task (LAVAD handles temporal detection only), or require training data (STPrompt requires weak labels; Hawk/HolmesVAU require instruction fine-tuning). No prior method simultaneously supports temporal detection (VAD), spatial localization (VAL), and semantic understanding (VAU) in a fully zero-shot setting. Furthermore, anomaly definitions vary across datasets (crime/violence/synthetic), causing models trained in one domain to fail in another.

## Core Problem
Can frozen VLMs and LLMs alone—through carefully designed test-time reasoning strategies (prompt engineering and cascaded inference)—simultaneously achieve temporal detection, spatial localization, and semantic understanding of video anomalies? The key challenge is to avoid "overthinking" (hallucinations induced by excessive reasoning steps) while fully exploiting cross-task information.

## Method

### Overall Architecture
The framework consists of two major components: (1) **Intra-Task Reasoning (IntraTR)** for temporal VAD—performing an initial scoring pass, extracting anomaly labels $t_V$ from high-score segments, then applying a confidence gate to decide whether to trigger a second-round refinement for uncertain samples; (2) **Inter-Task Chaining (InterTC)** that propagates VAD outputs ($t_V$, $\tilde{s}_V$, $W_{max}$) to VAL and VAU—augmenting localization prompts with extracted labels, and augmenting understanding prompts with bounding-box overlays.

### Key Designs
1. **Score-Guided Anomaly Extraction**: A sliding window is applied over initial frame-level scores $S_V$ to identify the most suspicious segment $W_{max}$, from which a video-level proxy anomaly probability $\tilde{s}_V = \mu(W_{max})$ is computed. A VLM then extracts a concise phrase list $t_V$ (e.g., "physical altercation, assault, fighting") from the frames within $W_{max}$, serving as a sample-level anomaly prior. Experiments demonstrate that automatically extracted $t_V$ even outperforms manually annotated class names—because the former is more specific (e.g., "placing a phone into one's pocket" is more informative than "theft").

2. **Score-Based Reasoning Gate**: Inspired by research on LLM "overthinking," the second-round refinement is triggered only when $\tilde{s}_V \in [0.5 - m, 0.5 + m]$ (near the decision boundary, indicating model uncertainty)—injecting $t_V$ into the prompt for refined scoring. $m$ can be a fixed constant (default 0.05) or an adaptive value $\tilde{m}_V = \text{Var}(S_V)$. Ablation studies confirm that applying refinement unconditionally (no gate) degrades performance by 1% due to overthinking-induced hallucinations, whereas gated refinement yields a 6.61% improvement.

3. **InterTC Cascading**: VAD→VAL: $t_V$ is injected into localization prompts to focus VLM detection; VAD→VAU: when $\tilde{s}_V > 0.5$, spatial localization is first performed on frames from $W_{max}$ to obtain bounding boxes, which are overlaid onto the original frames as "visual prompts," then fed together with augmented text prompts into the VLM to generate anomaly descriptions. This explicit visual guidance proves more effective than text-only prompting.

### Loss & Training
Fully training-free. Default configuration: VideoLLaMA3-7B as VLM, Llama-3.1-8B-Instruct as LLM, Qwen2.5-VL-7B as localization VLM. Frame sampling at stride 16, window $\ell = \max(300, T/10)$, Gaussian smoothing $\sigma=10$. Runs on 2× RTX 3090. Achieves 4× speedup over LAVAD (0.029 vs. 0.117 sec/frame).

## Key Experimental Results

| Dataset | Method | AUC (%) | Training Required |
|---------|--------|---------|------------------|
| UCF-Crime | LAVAD | 80.28 | Zero-shot |
| UCF-Crime | **Ours** | **84.28** | Zero-shot |
| XD-Violence | LAVAD | 85.36 | Zero-shot |
| XD-Violence | **Ours** | **91.34** | Zero-shot |
| UBnormal | Ours | **86.0** | Zero-shot |
| MSAD | Ours | **76.4** AP | Zero-shot |

VAL: TIoU improves from 24.09% (baseline) to 25.21% (+InterTC).
VAU: GPT-C on UCF-Crime improves from 0.384 to 0.444 (+InterTC).

### Ablation Study
- **Reasoning step ablation**: LLM scoring + gated reasoning = 84.28%; LLM scoring + ungated reasoning = 77.40% (overthinking hurts!); VLM-only = 77.67%.
- **Automatic $t_V$ vs. manual labels**: $t_V$ (84.28%) > $t_{oracle}$ (83.91%) > empty (81.86%).
- **Sensitivity to $m$**: Results are stable for $m \in [0.05, 0.2]$; performance drops substantially at $m = 0.4$.
- **VLM/LLM generalization**: Replacing VLMs (2B–7B) or LLMs causes <4% performance variation, confirming the plug-and-play nature of the framework.
- **Comparison with modernized baseline**: Running LAVAD with VideoLLaMA3 + Llama3.1 yields only 72.99%, demonstrating that the gains stem from the proposed framework rather than stronger backbone models.

## Highlights & Insights
- The only fully zero-shot method that simultaneously supports VAD+VAL+VAU (Table 1 clearly illustrates this distinction).
- The confidence-gated design to prevent "overthinking" is elegant—drawing on LLM reasoning efficiency research, it controls inference depth via selective prediction.
- The finding that automatically extracted $t_V$ surpasses manually annotated labels is surprising, suggesting that contextualized, specific descriptions are more informative than abstract class names.
- 4× faster than LAVAD by reducing unnecessary VLM queries.

## Limitations & Future Work
- Performance is bounded by the prior knowledge of frozen VLMs/LLMs—rare anomaly types not covered by pretraining data may be missed.
- Sensitivity to brief anomalies (<10 seconds) is limited, as uniformly sampled clips $c_i$ may lack sufficient temporal granularity.
- Validation is confined to surveillance/violence domains; anomaly detection in medical or industrial settings remains unexplored.
- VAU outputs are occasionally overly verbose, a common issue with LLM-based reasoning.

## Related Work & Insights
- **vs. LAVAD**: LAVAD addresses temporal detection only, without localization or understanding; using the same VLM+LLM configuration, the proposed method outperforms LAVAD by 11% on UCF-Crime.
- **vs. HolmesVAU**: HolmesVAU requires LoRA fine-tuning; its zero-shot performance is only 58.54% AUC, compared to 84.28% achieved here.
- **vs. VERA**: VERA requires prompt-tuning and does not support VAL; its 86.55% AUC is slightly higher than 84.28%, but VERA is not training-free.
- The inter-task reasoning chain paradigm is transferable to other multi-level video analysis pipelines (e.g., cascading video QA → summarization → editing).
- The confidence-gated selective reasoning strategy offers broadly applicable insights for any LLM-based sequential decision-making system.
- The finding that "automatic labels outperform manual labels" suggests that contextually grounded, specific descriptions are more valuable than abstract category names in prompt engineering.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of unified three-task support and gated reasoning is novel, though each individual component is straightforward.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Four VAD datasets, VAL experiments, VAU experiments, 10+ ablations, VLM/LLM generalization tests, and runtime analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure; the scope comparison in Table 1 and the framework diagram in Figure 1 are effective; all prompt designs are publicly disclosed.
- **Value**: ⭐⭐⭐⭐ First fully zero-shot method covering all three video anomaly analysis tasks; strong practical utility; open-source code.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] VADTree: Explainable Training-Free Video Anomaly Detection via Hierarchical Granularity](vadtree_explainable_training-free_video_anomaly_detection_via_hierarchical_granu.md)
- [\[ICCV 2025\] SVIP: Semantically Contextualized Visual Patches for Zero-Shot Learning](../../ICCV2025/interpretability/svip_semantically_contextualized_visual_patches_for_zero-shot_learning.md)
- [\[AAAI 2026\] Induce, Align, Predict: Zero-Shot Stance Detection via Cognitive Inductive Reasoning](../../AAAI2026/interpretability/induce_align_predict_zero-shot_stance_detection_via_cognitive_inductive_reasonin.md)
- [\[CVPR 2026\] Text-guided Fine-Grained Video Anomaly Understanding](../../CVPR2026/interpretability/text-guided_fine-grained_video_anomaly_understanding.md)
- [\[NeurIPS 2025\] scPilot: Large Language Model Reasoning Toward Automated Single-Cell Analysis and Discovery](scpilot_large_language_model_reasoning_toward_automated_single-cell_analysis_and.md)

</div>

<!-- RELATED:END -->
