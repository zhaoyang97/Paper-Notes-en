---
title: >-
  [Paper Note] Retrieving to Recover: Towards Incomplete Audio-Visual Question Answering via Semantic-consistent Purification
description: >-
  [ACL 2026][Audio & Speech][Audio-visual question answering] This paper proposes the R2ScP framework, which shifts the missing-modality paradigm in AVQA from conventional generative completion to retrieval-based recovery.…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "Audio-visual question answering"
  - "missing modality"
  - "retrieval-based recovery"
  - "semantic purification"
  - "mixture of experts"
date: 2026-05-08
content_hash: bd5d31f3e2aa49b3
---

# Retrieving to Recover: Towards Incomplete Audio-Visual Question Answering via Semantic-consistent Purification

**Conference**: ACL 2026
**arXiv**: [2604.10695](https://arxiv.org/abs/2604.10695)
**Code**: N/A
**Area**: Audio & Speech / Multimodal Learning
**Keywords**: Audio-visual question answering, missing modality, retrieval-based recovery, semantic purification, mixture of experts

## TL;DR

This paper proposes the R2ScP framework, which shifts the missing-modality paradigm in AVQA from conventional generative completion to retrieval-based recovery. By combining cross-modal retrieval with a context-aware adaptive purification mechanism to eliminate retrieval noise, R2ScP achieves substantial performance gains in modality-incomplete settings.

## Background & Motivation

**Background**: Audio-visual question answering (AVQA) requires models to reason across visual, audio, and textual modalities to understand dynamic scenes. Existing methods typically assume complete availability of all modalities, causing severe performance degradation under practical conditions such as device failures, sensor occlusion, or data transmission interruptions.

**Limitations of Prior Work**: The dominant approach to handling missing modalities relies on generative completion—synthesizing pseudo-features for the missing modality from available ones. However, generative models tend to produce representations of "common knowledge," i.e., generalized embeddings that lack fine-grained modality-specific information. For instance, when inferring missing audio from a concert's visual scene, a generative model may synthesize a generic "music" embedding while failing to capture the timbre of the specific instruments visible in the frame, thereby introducing semantic hallucinations and noise.

**Key Challenge**: Generative approaches inherently "imagine" missing information from available modalities, and their outputs are constrained by cross-modal shared knowledge, making it impossible to recover modality-specific unique information. This information loss directly impairs question answering tasks that demand precise reasoning.

**Goal**: To shift the missing-modality paradigm from generation to retrieval—recalling genuine, high-quality feature segments from a semantic database rather than synthesizing imperfect hallucinations.

**Key Insight**: The authors observe that real-world feature repositories contain abundant reusable modality-specific knowledge; the key challenge lies in accurate retrieval and denoising.

**Core Idea**: Replace generative completion with cross-modal retrieval, and filter retrieval noise via a context-aware purification mechanism to preserve fine-grained, modality-specific knowledge.

## Method

### Overall Architecture

R2ScP takes audio/visual/text data—potentially with a missing modality—as input and produces a question answering response. The framework consists of three core modules: (1) a Cross-Modal Retrieval module (CMR) that retrieves candidate features for the missing modality from an external memory bank via a unified semantic space; (2) a Context-Aware Adaptive Purification mechanism (CAP) that filters retrieval noise and injects high-quality semantics; and (3) a two-stage expert training strategy that explicitly models the reliability of different information sources through a mixture-of-experts architecture.

### Key Designs

1. **Cross-Modal Retrieval Module (CMR)**:

    - **Function**: Retrieves candidate features for the missing modality from an external memory bank.
    - **Mechanism**: Constructs an external memory bank $\mathcal{B} = \{(\mathbf{k}_i, \mathbf{v}_i)\}_{i=1}^{M}$, where key-value pairs are generated as unified semantic embeddings by a pretrained multimodal model (e.g., ImageBind). Given a missing modality (e.g., audio), the available modality (e.g., visual) is used as a query, and top-$n$ candidates are retrieved via cosine similarity $S_i = \frac{\mathbf{Q}_{avl} \cdot \mathbf{k}_i}{\|\mathbf{Q}_{avl}\| \|\mathbf{k}_i\| + \epsilon}$.
    - **Design Motivation**: The unified semantic space enables cross-modal alignment so that visual queries can locate semantically related audio features, thereby preserving modality-specific knowledge present in real-world data.

2. **Context-Aware Adaptive Purification Mechanism (CAP)**:

    - **Function**: Eliminates semantic noise from retrieved candidates and injects high-quality features.
    - **Mechanism**: Executes three stages: (a) *Consistency noise identification*—computes an inconsistency score $\delta_i = 1 - \text{sim}(H_{miss} \cdot \mathbf{W}_{proj}, \mathbf{g}_{avl})$ between retrieved features and the global context anchor of the available modality, selecting the top-$k$ inconsistent tokens to form the noise index set $\Omega_{noise}$; (b) *Text-guided semantic acquisition*—uses multi-head cross-attention and self-attention to identify salient semantic indices $\Omega_{salient}$ most relevant to the question from common knowledge; (c) *Selective feature injection*—replaces noise positions with high-quality semantics: $H_{miss}^{pur} = (\mathbf{1} - \mathcal{M}_{noise}) \odot H_{miss} + \mathcal{M}_{noise} \odot \text{Gather}(H_{guided}, \Omega_{salient})$.
    - **Design Motivation**: Raw retrieval inevitably introduces irrelevant information (e.g., cello or applause features retrieved for a violin performance). CAP applies dual filtering through available-modality contextual constraints and question-guided attention to ensure semantic consistency.

3. **Two-Stage Mixture-of-Experts Training**:

    - **Function**: Explicitly models the reliability of different information sources (original vs. recovered).
    - **Mechanism**: In the first stage, three modality experts ($\mathcal{E}_v$, $\mathcal{E}_a$, $\mathcal{E}_t$) are pretrained independently to extract discriminative representations without relying on cross-modal shortcuts. In the second stage, the experts are frozen and a gating network (Router) is trained to dynamically assign importance weights $\alpha_{m'} = \frac{\exp(g_{m'})}{\sum_{m} \exp(g_m)}$, yielding the joint representation $\mathbf{Z}_{joint} = \alpha_a H_a + \alpha_t H_t + \alpha_v H_v$.
    - **Design Motivation**: Decoupled training prevents mutual dependence among experts. The gating network dynamically adjusts modality weights based on input context, allowing recovered modalities to receive lower weights that reflect their inherent uncertainty.

### Loss & Training

In addition to the standard cross-entropy loss $\mathcal{L}_{task}$, a semantic ranking loss $\mathcal{L}_{rank}$ is introduced to enforce that positively retrieved recovered features outrank negative samples but remain below ground-truth features in quality: $\mathcal{L}_{total} = \mathcal{L}_{task} + \lambda(\mathcal{L}_{rank}^+ + \mathcal{L}_{rank}^-)$. This ensures that retrieval-purified features lie within a valid semantic manifold.

## Key Experimental Results

### Main Results

| Dataset | Modality Setting | Ours (R2ScP) | Prev. SOTA (IMOL) | Gain |
|---------|-----------------|-------------|-------------------|------|
| Music-AVQA | Missing Audio | 69.37 | 67.11 | +2.26 |
| Music-AVQA | Missing Visual | 72.06 | 69.21 | +2.85 |
| Music-AVQA | Complete | 73.19 | 71.86 | +1.33 |
| AVQA | Missing Audio | 63.25 | 61.32 | +1.93 |
| AVQA | Missing Visual | 75.12 | 72.38 | +2.74 |
| AVQA | Complete | 90.64 | 90.28 | +0.36 |

### Ablation Study

| Configuration | Music-AVQA | AVQA | Note |
|--------------|-----------|------|------|
| w/o CMR, w/o CAP | 62.43 | 57.43 | Baseline (missing audio) |
| +CMR only | 67.21 | 61.78 | Retrieval contributes +4.78/+4.35 |
| +CAP only | 64.11 | 59.64 | Purification alone is also effective |
| +CMR+CAP (full) | 69.37 | 63.25 | Combination yields best performance |

### Key Findings

- Retrieval-based recovery outperforms generative completion, with the advantage particularly pronounced for missing visual modality (+2.85 vs. IMOL).
- CMR and CAP are each independently effective, but their combination achieves the best results, indicating that retrieval and purification are complementary.
- R2ScP surpasses competing methods even in the complete-modality setting, demonstrating that the retrieval-recovery framework benefits multimodal fusion beyond missing-modality scenarios.
- The two-stage training strategy avoids cross-modal feature collapse by decoupling expert pretraining from gated mixing.

## Highlights & Insights

- **Paradigm innovation**: The conceptual shift from "generative completion" to "retrieval-based recovery" is concise and compelling, circumventing the hallucination problem inherent in generative approaches.
- The three-stage CAP design (noise identification → semantic acquisition → selective injection) is logically rigorous and effectively leverages guidance signals from both the available modality and the question.
- The semantic ranking loss elegantly establishes a quality gradient of "ground-truth > positive retrieval > negative retrieval."
- Performance improvements in the complete-modality setting suggest that the retrieval mechanism can serve as a general-purpose modality augmentation strategy.

## Limitations & Future Work

- Constructing and storing the external memory bank incurs significant overhead, which may be a bottleneck for large-scale deployment.
- The current formulation assumes complete absence of the missing modality and does not address partial missingness or noise-degraded scenarios.
- Retrieval quality is highly dependent on the alignment quality of the unified semantic space (ImageBind).
- Validation is limited to the AVQA task; generalization to other multimodal reasoning tasks (e.g., visual question answering, dialogue) remains unexplored.

## Related Work & Insights

- **vs. Missing-AVQA (ECCV 2024)**: Missing-AVQA employs a relation-aware generator to synthesize missing features, but the output represents common knowledge; R2ScP preserves modality-specific information through retrieval.
- **vs. IMOL (ACL 2025)**: Although IMOL also employs retrieval, it is used primarily for contrastive alignment rather than direct feature recovery; R2ScP directly substitutes retrieved features for missing ones.
- **vs. MoMKE (MM 2024)**: MoMKE preserves modality-specific knowledge via mixture of experts but does not address feature recovery; R2ScP combines retrieval with a MoE architecture for a more complete solution.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The paradigm shift from generation to retrieval is a clear contribution, and the CAP purification mechanism is elegantly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Two datasets, multiple missing-modality configurations, and detailed ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Problem motivation is clearly articulated and the method is described in thorough detail.
- **Value**: ⭐⭐⭐⭐ Opens a new research direction for handling missing modalities in multimodal systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Music Audio-Visual Question Answering Requires Specialized Multimodal Designs](music_audio-visual_question_answering_requires_specialized_multimodal_designs.md)
- [\[CVPR 2026\] Semantic Audio-Visual Navigation in Continuous Environments](../../CVPR2026/audio_speech/semantic_audio-visual_navigation_in_continuous_environments.md)
- [\[ICLR 2026\] Query-Guided Spatial-Temporal-Frequency Interaction for Music Audio-Visual Question Answering](../../ICLR2026/audio_speech/query-guided_spatial-temporal-frequency_interaction_for_music_audio-visual_quest.md)
- [\[ACL 2026\] Jamendo-MT-QA: A Benchmark for Multi-Track Comparative Music Question Answering](jamendo-mt-qa_a_benchmark_for_multi-track_comparative_music_question_answering.md)
- [\[CVPR 2026\] ViDscribe: Multimodal AI for Customizing Audio Description and Question Answering in Online Videos](../../CVPR2026/audio_speech/vidscribe_multimodal_ai_for_customizing_audio_description_and_question_answering.md)

</div>

<!-- RELATED:END -->
