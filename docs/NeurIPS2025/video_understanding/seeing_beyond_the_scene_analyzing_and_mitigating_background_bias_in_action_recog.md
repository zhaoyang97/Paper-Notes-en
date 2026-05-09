---
title: >-
  [Paper Note] Seeing Beyond the Scene: Analyzing and Mitigating Background Bias in Action Recognition
description: >-
  [Video Understanding] This paper systematically analyzes background bias in action recognition across three model paradigms — classification models, contrastive pre-trained models (CLIP/SigLIP2), and video large language models (VLLMs) — and proposes two mitigation strategies: a dual-branch architecture that fuses segmented human inputs to reduce SBErr by 3.78% for classification models, and automated prompt tuning to reduce SBErr by 9.85% for VLLMs.
tags:
  - Video Understanding
date: 2026-05-08
content_hash: 24142ae9c23d0114
---

# Seeing Beyond the Scene: Analyzing and Mitigating Background Bias in Action Recognition

**Conference/Journal**: NeurIPS 2025 Workshop
**Authors**: Ellie Zhou, Jihoon Chung, Olga Russakovsky (Princeton University)
**arXiv**: [2512.17953](https://arxiv.org/abs/2512.17953)
**Area**: Video Understanding / Action Recognition / Bias Analysis

## TL;DR

This paper systematically analyzes background bias in action recognition across three model paradigms — classification models, contrastive pre-trained models (CLIP/SigLIP2), and video large language models (VLLMs) — and proposes two mitigation strategies: a dual-branch architecture that fuses segmented human inputs to reduce SBErr by 3.78% for classification models, and automated prompt tuning to reduce SBErr by 9.85% for VLLMs.

## Background & Motivation

**Background**: Action recognition models have long relied on background scene cues (e.g., snow → skiing) as prediction shortcuts rather than focusing on human body movements. Prior research has primarily studied background bias in classification models, leaving contrastive pre-trained models such as CLIP and VLLMs largely underexplored.

**Limitations of Prior Work**: (1) Background bias in CLIP/SigLIP2 and VLLMs has not been sufficiently investigated; (2) simply removing background information reduces bias but severely degrades accuracy on standard datasets (a 26.47% drop on Kinetics-50); (3) architectural modifications to VLLMs are costly, and lightweight mitigation approaches are lacking.

**Key Challenge**: Background information simultaneously serves as useful context (aiding prediction in normal scenes) and a harmful source of bias (misleading predictions in counterfactual scenes). The central challenge lies in suppressing bias while preserving beneficial contextual information.

**Goal**: (1) Quantify the severity of background bias across the three model paradigms; (2) design a bias-mitigation architecture for classification models that retains background context; (3) explore lightweight prompt-based mitigation for VLLMs.

**Key Insight**: The paper addresses two orthogonal dimensions — human body segmentation and prompt design — to develop paradigm-specific mitigation strategies for classification models and VLLMs, respectively.

**Core Idea**: A dual-branch architecture balances contextual utilization and bias suppression by fusing original video with segmented human video, while automated prompt tuning guides VLLMs to focus on human actions.

## Method

### Overall Architecture

The work is organized into two modules: **bias analysis** and **bias mitigation**.

- **Bias Analysis**: On two benchmarks — HAT Action Swap (transplanting the human body of class A onto the background of class B) and Mimetics (mimicked actions without matching scenes) — the paper evaluates classification models (Slow-Only), contrastive pre-trained models (CLIP ViT-B/32, SigLIP2), and VLLMs (InternVL3-8B/78B), quantifying bias via SHAcc (proportion of correct human action predictions) and SBErr (proportion of erroneous predictions toward background actions).
- **Bias Mitigation**: On the classification model side, a dual-branch fusion architecture with an adaptive weighting mechanism is designed; on the VLLM side, both manual prompts and an automated prompt tuning loop are explored.

### Key Designs

**1. Dual-Branch Fusion Architecture (Dual-Branch Sum / Stack)**

- **Function**: Introduces segmented human information while retaining background context, balancing accuracy and bias.
- **Mechanism**: Two parallel Slow-Only branches process the original video and the segmented human video, respectively. After Stage 2, features are fused via element-wise addition (Sum) or channel concatenation (Stack), followed by Stages 3–4 and the prediction head. Early layers learn low-level features independently; post-fusion layers learn a joint representation of human and context.
- **Design Motivation**: Although pure segmented input reduces SBErr from 23.42% to 2.09%, Kinetics accuracy drops by 26.47%. The dual-branch architecture improves Kinetics accuracy by 2.22% while reducing SBErr by 3.06%–3.62%.

**2. Adaptive Weighted Focus Mechanism**

- **Function**: Enables the model to adaptively control the weight assigned to human and background regions.
- **Mechanism**: An auxiliary 3D CNN learns a scalar parameter $\alpha$ (constrained to $[-1, 1]$) from early Slow-Only feature maps, constructing a weighted mask $M_{weighted} = (1+\alpha) \cdot M + (1-\alpha) \cdot (1-M)$, which is multiplied with feature maps before further propagation. When $\alpha=1$, the human region receives 2× weight and background 0×; when $\alpha=-1$, the relationship is reversed.
- **Design Motivation**: More flexible than fixed fusion, allowing the model to adaptively adjust the importance of human vs. background on a per-sample basis. Achieves the best balance: Kinetics +2.10%, SBErr −3.78%.

**3. Automated Prompt Engineering**

- **Function**: Systematically optimizes VLLM input prompts to reduce background bias.
- **Mechanism**: GPT-4.1 acts as the prompt engineer in a 20-round iterative loop: (1) GPT proposes a prompt → (2) SHAcc/SBErr are evaluated on GPT-4o-mini with the HAT dataset → (3) results are fed back to GPT → (4) GPT refines the prompt for the next round. 75% of the data is used for evaluation and 25% for prompt tuning.
- **Design Motivation**: Manually crafted human-focused prompts reduce SBErr by only 4.75%, whereas automated tuning achieves a 9.85% reduction, indicating that superior solutions exist in the prompt space that are difficult for humans to discover manually.

### Loss & Training

- All classification models are trained from scratch on Slow-Only (R50) for 300 epochs using the Adam optimizer, lr = 0.001, ReduceLROnPlateau (patience = 40), and batch size = 20.
- Human segmentation pipeline: YOLOv5 detects human bounding boxes → SAM2 propagates segmentation masks across frames → non-human regions are zeroed out.
- Data augmentation: segmented human bodies from Kinetics-50 videos are composited onto random Places365 backgrounds to break scene–action correlations. Training data doubles to 49,336 samples at the cost of reduced Kinetics accuracy but improved HAT/Mimetics performance.

## Key Experimental Results

### Main Results

Bias mitigation results for classification models (changes relative to Slow-Only baseline in parentheses):

| Model | Kinetics-50 ↑ | HAT SHAcc ↑ | HAT SBErr ↓ | Mimetics ↑ |
|---|---|---|---|---|
| Slow-Only | 49.93 | 9.62 | 23.42 | 6.87 |
| Segmented | 23.46 (−26.47) | 23.34 (+13.72) | 2.09 (−21.33) | 9.54 (+2.67) |
| Dual-Branch Sum | 52.15 (+2.22) | 12.76 (+3.14) | 20.36 (−3.06) | 7.85 (+0.98) |
| Dual-Branch Stack | 51.51 (+1.58) | 12.80 (+3.18) | 19.80 (−3.62) | 8.28 (+1.41) |
| Weighted-Focus | 52.03 (+2.10) | 12.80 (+3.18) | **19.64 (−3.78)** | 7.85 (+0.98) |

Background bias comparison across model paradigms:

| Model | HAT SHAcc ↑ | HAT SBErr ↓ | Mimetics ↑ |
|---|---|---|---|
| Slow-Only | 35.81 | 55.41 | 57.64 |
| CLIP ViT-B/32 | 29.25 | 53.66 | 46.84 |
| SigLIP2 | 25.46 | 58.91 | 48.95 |
| InternVL3-8B | 40.29 | 48.84 | 62.83 |
| InternVL3-78B | 45.73 | 48.39 | 66.61 |

### Ablation Study

VLLM prompt strategy comparison (GPT-4o-mini):

| Prompt Strategy | SHAcc ↑ | SBErr ↓ | SBErr Change |
|---|---|---|---|
| Neutral baseline | 39.14 | 51.40 | — |
| Prefixed-choices | 33.93 | 46.65 | −4.75 |
| Human-focused (manual) | 40.92 | 46.99 | −4.41 |
| Background-focused (manual) | 35.82 | 52.95 | +1.55 |
| Best automated prompt | 46.70 | 41.55 | **−9.85** |

### Key Findings

1. **Background bias is universal**: All three paradigms — classification models, CLIP/SigLIP2, and VLLMs — exhibit significant background bias, with VLLMs being relatively least affected.
2. **Scaling model capacity does not resolve bias**: Scaling InternVL3 from 8B to 78B improves SHAcc but leaves SBErr nearly unchanged — larger models learn richer features but do not learn to suppress background shortcuts.
3. **Temporal information is effective**: Increasing the number of input frames simultaneously improves SHAcc and reduces SBErr, indicating that temporal motion information is key to countering background bias.
4. **Class-level bias strongly correlates with scene distinctiveness**: Classes with visually distinctive backgrounds highly correlated with the action (e.g., "weather forecasting," SBErr = 89.09%) exhibit extreme bias, while classes without distinctive scenes (e.g., "playing violin," SBErr = 0%) show almost none.
5. **Data augmentation is a double-edged sword**: Places365 background replacement augmentation dramatically reduces HAT SBErr (Weighted-Focus: 19.64% → 1.85%) but at the cost of reduced Kinetics accuracy.

## Highlights & Insights

1. **Systematic cross-paradigm analysis**: For the first time, background bias is compared across classification models, contrastive pre-trained models, and VLLMs within a unified framework, revealing the universality of bias and its varying severity.
2. **Quantification of the accuracy–bias trade-off**: The paper clearly demonstrates the tension between background removal (bias reduction) and background retention (accuracy preservation), with the dual-branch architecture achieving positive gains on both dimensions simultaneously.
3. **Effectiveness of automated prompt tuning**: VLLMs are shown to be highly sensitive to prompt wording; automated search finds significantly better prompts than manual design (5.1% gap in SBErr), providing a low-cost pathway for VLLM bias mitigation.
4. **Counterintuitive finding that larger models ≠ less bias**: Increased model capacity improves overall capability but does not reduce reliance on background cues, suggesting that bias is data-driven rather than capacity-limited.

## Limitations & Future Work

1. **Limited model coverage**: Only a small set of representative models (Slow-Only, CLIP, InternVL3) are evaluated; whether findings generalize to broader architectures (e.g., TimeSFormer, VideoMAE) remains to be verified.
2. **Limitations of bias benchmarks**: HAT Action Swap is a synthetic dataset; human–background compositing may introduce out-of-distribution artifacts that differ from real-world background bias patterns.
3. **Limited mitigation effectiveness for classification models**: The best dual-branch variant reduces SBErr by only 3.78% (from 23.42% to 19.64%), leaving substantial bias.
4. **Instability of automated prompt tuning**: Performance does not improve monotonically across 20 iterations, and later prompts are not guaranteed to outperform earlier ones; convergence is not ensured.
5. **Fine-tuning strategies for VLLMs warrant exploration**: Instruction tuning on counterfactual data, for instance, may be more effective than pure prompt engineering.

## Related Work & Insights

- **HAT / Chung et al.**: Provides the HAT Action Swap benchmark and early analysis of background bias; this paper extends that analysis to additional model paradigms.
- **MASH-VLM (Bae et al.)**: Analyzes scene bias in VLLMs, but is limited to VLLMs and does not quantify the relative dependence on human versus background information.
- **SlowFast / Slow-Only**: The classification model baseline; the dual-branch architecture is built upon its backbone design.
- **LLM-as-optimizer (Yang et al., Gavrikov et al.)**: Methodological source for automated prompt engineering; this paper applies it to the bias mitigation setting.
- **Insights**: The dual-branch idea could be extended to downstream tasks such as video grounding and captioning; the automated prompt tuning framework is generalizable to other types of VLLM bias (e.g., gender bias, racial bias).

## Rating

| Dimension | Score | Rationale |
|---|---|---|
| Novelty | ⭐⭐⭐ | Cross-paradigm analysis perspective is novel, but the mitigation methods (dual-branch, prompt tuning) are relatively standard |
| Technical Depth | ⭐⭐⭐ | Analysis is comprehensive and systematic, but method design is relatively simple and lacks theoretical depth |
| Experimental Thoroughness | ⭐⭐⭐ | Ablation studies and cross-model comparisons are thorough, but model coverage and benchmark diversity are limited |
| Practical Value | ⭐⭐⭐⭐ | Automated prompt tuning is zero-cost and ready to use; dual-branch architecture is plug-and-play, offering practical reference value for real-world deployment |

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Enhancing Temporal Understanding in Video-LLMs through Stacked Temporal Attention in Vision Encoders](enhancing_temporal_understanding_in_videollms_through_stacke.md)
- [\[NeurIPS 2025\] FastVID: Dynamic Density Pruning for Fast Video Large Language Models](fastvid_dynamic_density_pruning_for_fast_video_large_languag.md)
- [\[ICCV 2025\] Learning to Generalize Without Bias for Open-Vocabulary Action Recognition](../../ICCV2025/video_understanding/learning_to_generalize_without_bias_for_open-vocabulary_action_recognition.md)
- [\[ICCV 2025\] Beyond Label Semantics: Language-Guided Action Anatomy for Few-shot Action Recognition](../../ICCV2025/video_understanding/beyond_label_semantics_language-guided_action_anatomy_for_few-shot_action_recogn.md)
- [\[NeurIPS 2025\] Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models](structured_sparse_transition_matrices_to_enable_state_tracking_in_state-space_mo.md)

<!-- RELATED:END -->
