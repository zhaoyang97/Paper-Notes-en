---
title: >-
  [Paper Note] QA-TIGER: Question-Aware Gaussian Experts for Audio-Visual Question Answering
description: >-
  [CVPR 2025][Video Understanding][Audio-Visual Question Answering] QA-TIGER is proposed, a framework that models video temporal sequence using continuous adaptive weighting with a Mixture of Gaussian Experts (MoE), and injects question information early in the encoding process to achieve progressive semantic refinement, reaching SOTA on multiple AVQA benchmarks.
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Audio-Visual Question Answering"
  - "Mixture of Gaussian Experts"
  - "Temporal Modeling"
  - "Question-Aware Attention"
  - "Temporal Grounding"
date: 2026-05-08
content_hash: 75bdf6b67ca3ee16
---

# QA-TIGER: Question-Aware Gaussian Experts for Audio-Visual Question Answering

**Conference**: CVPR 2025  
**arXiv**: [2503.04459](https://arxiv.org/abs/2503.04459)  
**Code**: [Project Page](https://aim-skku.github.io/QA-TIGER/)  
**Area**: Video Understanding/Multimodal QA  
**Keywords**: Audio-Visual Question Answering, Mixture of Gaussian Experts, Temporal Modeling, Question-Aware Attention, Temporal Grounding

## TL;DR

QA-TIGER is proposed, a framework that models video temporal sequence using continuous adaptive weighting with a Mixture of Gaussian Experts (MoE), and injects question information early in the encoding process to achieve progressive semantic refinement, reaching SOTA on multiple AVQA benchmarks.

## Background & Motivation

Audio-Visual Question Answering (AVQA) requires multimodal reasoning and precise temporal localization based on the given question. However, existing methods suffer from two key limitations:

- **Underutilization of Question Information**: Most methods only integrate question information through simple multiplication at the final reasoning stage. The intermediate encoding process lacks question guidance, preventing progressive focus on relevant features.
- **Restricted Temporal Sampling Strategies**: Uniform frame sampling ignores question specificity. Although Top-K frame selection considers question relevance, discrete sampling loses inter-frame continuity and solely relies on visual cues, ignoring audio information.

Key Challenge: How to flexibly capture question-relevant continuous and non-continuous temporal segments while maintaining the question context throughout the entire encoding pipeline.

## Method

### Overall Architecture

QA-TIGER consists of three modules: (1) Question-Aware Fusion—injecting question context into visual and audio features early in the encoding stage; (2) Temporal Integration of Gaussian Experts—weighting the timeline using multiple Gaussian distributions and adaptively selecting experts via MoE routing; (3) Question-Guided Reasoning—fusing temporal features for answer prediction.

### Key Designs

#### Key Design 1: Question-Aware Fusion

- **Function**: Injects question context into visual and audio features early in the encoding pipeline.
- **Mechanism**: Self-attention is first applied to visual/audio features to enhance internal relations, followed by two rounds of cross-attention—first interacting with the other modality, and then interacting with word-level question features $q_w$. $\mathbf{v}_q = \mathbf{v} + SA(\mathbf{v}) + CA(\mathbf{v}, \mathbf{a}) + CA(\mathbf{v}, q_w)$, with audio features processed analogously. Subsequently, the question-aware frame-level features $\mathbf{v}_q, \mathbf{a}_q$ are used as queries to perform cross-attention on patch-level features to refine spatial details.
- **Design Motivation**: Early injection of question context allows the model to continuously align with question-relevant cues throughout the processing, avoiding the "too late" issue of only introducing question information at the final stage.

#### Key Design 2: Temporal Integration of Gaussian Experts

- **Function**: Performs adaptive continuous weighting of the timeline to focus on question-relevant temporal segments.
- **Mechanism**: For visual and audio modalities, $E$ Gaussian distributions $g_m^i = \mathcal{N}(\mu_m^i, (\sigma_m^i)^2)$ are generated, with their center locations distributed along the timeline and augmented with offsets. Weights $r_m^i$ are assigned to each expert via an MoE router, and the final temporal feature is computed as a weighted sum of all experts: $\mathcal{G}_m(x) = \sum_{i=1}^{E} g_m^i \cdot r_m^i \cdot \mathcal{E}_m^i(x)$, where $\mathcal{E}_m^i$ is the MLP of each expert.
- **Design Motivation**: Both uniform sampling and Top-K selection are discrete, which loses temporal continuity. Multiple Gaussian distributions provide soft masks that can simultaneously cover continuous segments and scattered key frames, while visual and audio modalities maintain independent temporal localization.

#### Key Design 3: Question-Guided Reasoning

- **Function**: Fuses temporally integrated visual and audio features for answer prediction.
- **Mechanism**: Hierarchical cross-attention fusion is performed first using the sentence-level question feature $q_s$ with the two visual temporal features $\tilde{v}_{p_v}, \tilde{v}_{p_a}$ to obtain $F_v$. Then, $F_v$ is fused with the audio temporal feature $\tilde{a}$ to yield the final representation $F_{va}$. A linear layer with softmax is used to predict the answer.
- **Design Motivation**: Hierarchical fusion (intra-visual fusion followed by audio-visual fusion) combined with residual connections prevents over-reliance on a single modality.

### Loss & Training

Standard cross-entropy loss: $\mathcal{L}_{qa} = -\sum_{c=1}^{C} y_c \log \mathcal{P}_c$.

## Key Experimental Results

### Main Results

#### MUSIC-AVQA Dataset

| Method | Audio | Visual | A-V | Overall Acc. |
|------|-------|--------|-----|-------------|
| AVST | 65.4 | 63.7 | 60.5 | 62.3 |
| APL | 72.3 | 71.9 | 70.1 | 70.9 |
| TSPM | 71.8 | 74.2 | 71.4 | 72.2 |
| **QA-TIGER** | **73.5** | **76.1** | **73.8** | **74.3** |

#### MUSIC-AVQA-v2.0 (Debiased)

| Method | Balanced Acc. |
|------|-------------|
| AVST | 48.3 |
| COCA | 52.1 |
| **QA-TIGER** | **56.2** |

### Ablation Study

| Component | Overall Acc. |
|------|-------------|
| Baseline (Uniform Sampling + Late Question Fusion) | ~70 |
| + Question-Aware Fusion | +2.1 |
| + Gaussian Experts | +1.8 |
| + Combination of Both | **74.3** |

### Key Findings

- State-of-the-art (SOTA) results are achieved across three datasets: MUSIC-AVQA, MUSIC-AVQA-R, and MUSIC-AVQA-v2.0.
- Early question injection and Gaussian experts contribute ~2% gain individually, and show stronger synergistic effects when combined.
- Independent temporal localization for visual and audio modalities yields better performance than forced alignment.
- The performance advantage is more pronounced on the debiased dataset (v2.0 balanced), indicating that the proposed method does not rely on dataset bias.

## Highlights & Insights

1. **Continuous Temporal Modeling with Gaussian Distributions**: Compared to discrete Top-K frame selection, Gaussian soft-masking is naturally suited for modeling temporal continuity.
2. **Value of Early Question Injection**: Guiding the entire encoding pipeline with the question allows for more thorough information utilization.
3. **Independent Audio-Visual Temporal Localization**: Key moments in visual and audio modalities may not be fully aligned (e.g., sound preceding action), making independent modeling more flexible.

## Limitations & Future Work

- The number of Gaussian experts $E$ is a hyperparameter, and different video complexities may require different numbers of experts.
- It relies on pre-trained feature extractors (CLIP, VGGish), which are limited by the capacity of the underlying models.
- Long video scenarios (>1 minute) are not considered, wherein the Gaussian coverage range might be insufficient.
- Integration with LLM-based reasoning methods can be explored in future work.

## Related Work & Insights

- **PSTP, TSPM**: Top-K frame selection methods based solely on visual cues.
- **COCA**: Causal graph modeling for multimodal collaboration.
- **APL**: Adaptive positive learning for aligning question-object semantics.
- The Gaussian experts concept can be extended to other multimodal reasoning tasks requiring temporal localization.

## Rating

⭐⭐⭐⭐ — The temporal modeling design with Gaussian experts is novel and effective, and the early question injection strategy is simple yet contributes significantly. Consistent SOTA across three datasets validates the robustness of the method.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] EgoTextVQA: Towards Egocentric Scene-Text Aware Video Question Answering](egotextvqa_towards_egocentric_scene-text_aware_video_question_answering.md)
- [\[CVPR 2025\] BIMBA: Selective-Scan Compression for Long-Range Video Question Answering](bimba_selective-scan_compression_for_long-range_video_question_answering.md)
- [\[CVPR 2025\] Cross-modal Causal Relation Alignment for Video Question Grounding](cross-modal_causal_relation_alignment_for_video_question_grounding.md)
- [\[NeurIPS 2025\] EgoGazeVQA: Egocentric Gaze-Guided Video Question Answering Benchmark](../../NeurIPS2025/video_understanding/egogazevqa_egocentric_gaze_guided_video_question_answering.md)
- [\[NeurIPS 2025\] Tool-Augmented Spatiotemporal Reasoning for Streamlining Video Question Answering Task](../../NeurIPS2025/video_understanding/toolaugmented_spatiotemporal_reasoning_for_streamlining_vide.md)

</div>

<!-- RELATED:END -->
