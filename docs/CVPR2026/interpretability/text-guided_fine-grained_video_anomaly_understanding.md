---
title: >-
  [Paper Note] Text-guided Fine-Grained Video Anomaly Understanding
description: >-
  [CVPR2026][Interpretability][Video Anomaly Detection] This paper proposes the T-VAU framework, which achieves pixel-level spatiotemporal anomaly localization via an Anomaly Heatmap Decoder (AHD)…
tags:
  - "CVPR2026"
  - "Interpretability"
  - "Video Anomaly Detection"
  - "Anomaly Heatmap"
  - "Region-Aware Encoder"
  - "Large Vision-Language Model"
  - "Multi-turn Dialogue"
date: 2026-05-08
content_hash: 31ee3cc6e0fab0ef
---

# Text-guided Fine-Grained Video Anomaly Understanding

**Conference**: CVPR2026
**arXiv**: [2511.00524](https://arxiv.org/abs/2511.00524)
**Code**: [github.com/momiji-bit/T-VAU](https://github.com/momiji-bit/T-VAU)
**Area**: Interpretability
**Keywords**: Video Anomaly Detection, Anomaly Heatmap, Region-Aware Encoder, Large Vision-Language Model, Multi-turn Dialogue

## TL;DR
This paper proposes the T-VAU framework, which achieves pixel-level spatiotemporal anomaly localization via an Anomaly Heatmap Decoder (AHD), and introduces a Region-Aware Anomaly Encoder (RAE) that injects heatmap evidence into an LVLM for unified reasoning over anomaly detection, localization, and semantic explanation.

## Background & Motivation
Video Anomaly Detection (VAD) is critical for security surveillance. Existing approaches suffer from fundamental limitations:
- **Traditional VAD**: Outputs video- or frame-level anomaly scores, providing coarse-grained binary decisions without interpretable evidence. Fine-grained cues may be diluted during feature aggregation.
- **Direct LVLM application**: Can produce textual judgments but lacks pixel-level localization capability; unreliable capture of weak anomaly signals leads to unfaithful textual descriptions.
- **LVLM–diffusion hybrids**: Combine visualization and text but may be unstable or inconsistent.

Core requirement: Anomaly understanding demands not only "whether anomalous" but also "where," "which object is responsible," and "how it evolves over time"—requiring a closed loop from pixel-level evidence to language-level reasoning.

**Key Insight**: (i) Extract spatiotemporal anomaly evidence via visual–textual alignment; (ii) inject this evidence as structured prompts into an LVLM for multi-task, multi-turn reasoning.

## Method

### Overall Architecture
T-VAU adds two lightweight trainable modules—AHD (Anomaly Heatmap Decoder) and RAE (Region-Aware Anomaly Encoder)—on top of a frozen LVLM backbone. The input consists of a video, natural language queries, and normal/anomalous text prompts; the output is a pixel-level anomaly heatmap and multi-turn dialogue responses.

### Key Designs

1. **Anomaly Heatmap Decoder (AHD)**:

    - Extracts multi-scale features $V_i$ from the visual encoder (layers 1/8/16/32).
    - Projects visual features to the text space via MLP and computes cosine similarity: $h_c^i[t,h,w] = \text{CosineSimilarity}(V'_i[t,:,h,w], T_c)$
    - Fuses across layers with learnable weights $w_i$: $H_c = \sum_i w_i \cdot h_c^i$
    - Applies Softmax and selects the anomaly channel to obtain the final heatmap.
    - **Design Motivation**: Leverages visual–textual alignment to extract anomaly signals directly from intermediate representations, eliminating the need for threshold setting.

2. **Region-Aware Anomaly Encoder (RAE)**:

    - Computes temporal differences between adjacent frame heatmaps: $X[t] = H_c[t+1] - H_c[t]$, capturing motion-aware information.
    - Extracts region-aware features via a convolutional backbone.
    - Divides each frame into a 3×3 grid and applies adaptive pooling to obtain region prompts $P_{region}$.
    - Generates a global prompt $p_{global}$ via spatial average pooling.
    - Final prompt sequence: $P_{An} = [P_{base}, P_{region}, p_{global}]$
    - Concatenated with visual prompts and dialogue context before being fed into the LLM decoder.

3. **Fine-Grained Anomaly Understanding Dataset Construction**:

    - Based on ShanghaiTech and UBnormal, using a three-stage pipeline:
    - Frame-level structured prompting → extraction of object attributes and spatial information → aggregation into object timelines.
    - Anomaly-focused refinement: background suppression via anomaly masks and Gaussian blur.
    - Cross-modal consistency verification: bidirectional appearance↔motion validation.

### Loss & Training
- AHD stage: Only the AHD is optimized; all other components are frozen.
- RAE stage: Curriculum-based SFT training (appearance–motion narration → anomaly-focused refinement).
- Overall: The LVLM backbone is frozen; only the two lightweight modules, AHD and RAE, are trained.

## Key Experimental Results

### Main Results

| Dataset | Metric | T-VAU | Prev. SOTA | Gain |
|--------|------|-------|----------|------|
| UBnormal | Micro-AUC | 94.8 | 68.2 (Georgescu FT) | +26.6 |
| UBnormal | RBDC | 67.8 | 28.7 (Georgescu FT) | +39.1 |
| UBnormal | TBDC | 76.7 | 58.1 (Georgescu FT) | +18.6 |
| ShanghaiTech | BLEU-4 (Target) | 62.67 | 55.73 (InternVL 8B) | +6.94 |
| ShanghaiTech | BLEU-4 (Trajectory) | 88.84 | 82.65 (InternVL 8B) | +6.19 |
| ShanghaiTech | Yes/No Acc | 97.67% | 94.28% (InternVL 8B) | +3.39% |

### Ablation Study

| Configuration | RBDC/TBDC | BLEU-4 (Target) | Yes/No Acc |
|------|-----------|----------------|-----------|
| T-VAU (full) | 67.8/76.7 | 62.67 | 97.67% |
| w/o AHD | N/A | 61.82 | 95.38% |
| w/o RAE | 67.8/76.7 | - | - |
| w/o AHD & RAE | N/A | 61.82 | 95.38% |

### Key Findings
- AHD and RAE are strongly complementary: AHD provides pixel-level evidence while RAE translates this evidence into interpretable language.
- Under a one-shot setting, AHD already achieves 94.5% micro-AUC and 64.3% RBDC, demonstrating exceptional data efficiency.
- Fine-tuning yields further gains, but the one-shot baseline is already competitive.
- The parameter overhead is only approximately 50M (8,274→8,325M), making the framework lightweight and efficient.

## Highlights & Insights
- The closed-loop "evidence→reasoning" design: anomaly heatmaps serve as visual evidence, and RAE structurally injects this evidence into the language model.
- The fine-grained dataset construction pipeline is systematic and complete: frame-level extraction → temporal aggregation → anomaly-focused refinement → cross-modal verification.
- Trajectory visualization (cumulative heatmaps across frames) provides intuitive temporal consistency verification.
- The threshold-free anomaly localization design avoids the threshold sensitivity inherent in conventional methods.

## Limitations & Future Work
- Performance remains challenging for micro-actions (minimal displacement) and highly non-rigid motion scenarios.
- Scene-dependent appearance variations (specular reflections, fog, etc.) degrade localization accuracy.
- The dataset is constructed from ShanghaiTech and UBnormal, limiting scene diversity.
- Freezing the LVLM backbone may constrain deeper anomaly understanding capacity.

## Related Work & Insights
- Compared to VAU methods such as HAWK and Holmes-VAU, T-VAU provides explicit pixel-level evidence through AHD.
- Training-free methods such as LAVAD are conceptually appealing but lack precise localization.
- Examining anomaly detection through the lens of Subtle Visual Computing (SVC) represents a promising research direction.

## Rating
- Novelty: ⭐⭐⭐⭐ The evidence–reasoning closed-loop design via AHD+RAE is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dimensional evaluation with complete ablation and qualitative analysis.
- Writing Quality: ⭐⭐⭐⭐ Architecture diagrams are clear, and inter-component relationships are well articulated.
- Value: ⭐⭐⭐⭐ Advances anomaly detection from score prediction to interpretable reasoning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] FineVAU: A Novel Human-Aligned Benchmark for Fine-Grained Video Anomaly Understanding](../../AAAI2026/interpretability/finevau_a_novel_human-aligned_benchmark_for_fine-grained_video_anomaly_understan.md)
- [\[CVPR 2026\] Geometry-Guided Camera Motion Understanding in VideoLLMs](geometry-guided_camera_motion_understanding_in_videollms.md)
- [\[CVPR 2026\] SafeDrive: Fine-Grained Safety Reasoning for End-to-End Driving in a Sparse World](safedrive_fine-grained_safety_reasoning_for_end-to-end_driving_in_a_sparse_world.md)
- [\[ICLR 2026\] Dynamic Reflections: Probing Video Representations with Text Alignment](../../ICLR2026/interpretability/dynamic_reflections_probing_video_representations_with_text_alignment.md)
- [\[ACL 2026\] LLM-Guided Semantic Bootstrapping for Interpretable Text Classification with Tsetlin Machines](../../ACL2026/interpretability/llm-guided_semantic_bootstrapping_for_interpretable_text_classification_with_tse.md)

</div>

<!-- RELATED:END -->
