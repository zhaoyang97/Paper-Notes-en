---
title: >-
  [Paper Note] Retrieving to Recover: Towards Incomplete Audio-Visual Question Answering via Semantic-consistent Purification
description: >-
  [ACL 2026][Audio & Speech][Audio-Visual Question Answering] This paper proposes the R2ScP framework, which transforms the missing modality handling paradigm in AVQA from traditional generative completion to retrieval-bas…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "Audio-Visual Question Answering"
  - "Missing Modality"
  - "Retrieval-based Recovery"
  - "Semantic Purification"
  - "Mixture-of-Experts"
date: 2026-05-08
content_hash: 5ecc74ac8e7dc141
---

# Retrieving to Recover: Towards Incomplete Audio-Visual Question Answering via Semantic-consistent Purification

**Conference**: ACL 2026  
**arXiv**: [2604.10695](https://arxiv.org/abs/2604.10695)  
**Code**: None  
**Area**: Audio and Speech / Multimodal Learning  
**Keywords**: Audio-Visual Question Answering, Missing Modality, Retrieval-based Recovery, Semantic Purification, Mixture-of-Experts

## TL;DR

This paper proposes the R2ScP framework, which transforms the missing modality handling paradigm in AVQA from traditional generative completion to retrieval-based recovery. By employing cross-modal retrieval and a context-aware adaptive purification mechanism to eliminate retrieval noise, it significantly enhances question-answering performance in incomplete modality scenarios.

## Background & Motivation

**Background**: Audio-Visual Question Answering (AVQA) requires models to perform reasoning across visual, audio, and textual modalities to understand dynamic scenes. Current methods typically assume that all modality data is fully available, but performance degrades severely in practical scenarios such as equipment failure, sensor occlusion, or data transmission interruption.

**Limitations of Prior Work**: Mainstream approaches for handling missing modalities rely on generative completion—synthesizing pseudo-features of the missing modality using available modalities. However, generative models tend to produce "common knowledge," i.e., generalized representations that lack fine-grained modality-specific information. For example, when inferring missing audio from a visual scene of a concert, a generative model might synthesize a generic "music" embedding but fail to capture the specific instrument timbre visible in the frame, thereby introducing semantic hallucinations and noise.

**Key Challenge**: Generative methods essentially "imagine" missing information from existing modalities, and their output is limited by cross-modal shared knowledge, failing to recover unique modality-specific information. This information loss directly impacts question-answering tasks that require precise reasoning.

**Goal**: To shift the paradigm of handling missing modalities from generation to retrieval—recalling real, high-quality feature segments from a semantic database rather than synthesizing imperfect hallucinations.

**Key Insight**: The authors observe that real-world feature libraries contain a vast amount of reusable modality-specific knowledge; the key lies in how to accurately retrieve and denoise it.

**Core Idea**: Replace generative completion with cross-modal retrieval and filter retrieval noise through a context-aware purification mechanism to preserve fine-grained modality-specific knowledge.

## Method

### Overall Architecture

The input to R2ScP consists of audio/visual/textual data where a modality might be missing, and the output is the answer to the question. The framework comprises three core modules: (1) The Cross-Modal Retrieval (CMR) module retrieves candidate features of the missing modality from an external memory bank via a unified semantic space; (2) The Context-Aware adaptive Purification (CAP) mechanism filters retrieval noise and injects high-quality semantics; (3) A two-stage expert training strategy explicitly models the reliability of different information sources through a Mixture-of-Experts (MoE) architecture.

### Key Designs

1.  **Cross-Modal Retrieval Module (CMR)**:

    - **Function**: Retrieves candidate features of missing modalities from an external memory bank.
    - **Mechanism**: An external memory bank $\mathcal{B} = \{(\mathbf{k}_i, \mathbf{v}_i)\}_{i=1}^{M}$ is constructed, where key-value pairs are generated as unified semantic embeddings by a pre-trained multimodal model (e.g., ImageBind). Given a missing modality (e.g., missing audio), the available modality (e.g., visual) is used as a query to retrieve the top-n candidate set via cosine similarity $S_i = \frac{\mathbf{Q}_{avl} \cdot \mathbf{k}_i}{\|\mathbf{Q}_{avl}\| \|\mathbf{k}_i\| + \epsilon}$.
    - **Design Motivation**: Leveraging a unified semantic space facilitates cross-modal alignment, allowing visual queries to find semantically relevant audio features and preserving modality-specific knowledge found in real-world data.

2.  **Context-Aware adaptive Purification (CAP)**:

    - **Function**: Eliminates semantic noise in retrieval candidates and injects high-quality features.
    - **Mechanism**: Executed in three stages—(a) Consistency noise identification: Calculate the inconsistency score $\delta_i = 1 - \text{sim}(H_{miss} \cdot \mathbf{W}_{proj}, \mathbf{g}_{avl})$ between retrieved features and the global context anchor of available modalities, selecting the top-k discordant tokens to form a noise index set $\Omega_{noise}$; (b) Text-guided semantic acquisition: Use the textual question to guide the identification of high-quality semantic indices $\Omega_{salient}$ most relevant to question answering from common knowledge via multi-head cross-attention and self-attention; (c) Selective feature injection: Replace noise positions with high-quality semantics $H_{miss}^{pur} = (\mathbf{1} - \mathcal{M}_{noise}) \odot H_{miss} + \mathcal{M}_{noise} \odot \text{Gather}(H_{guided}, \Omega_{salient})$.
    - **Design Motivation**: Raw retrieval inevitably introduces irrelevant information (e.g., retrieving cello or applause features for a violin performance). CAP ensures semantic consistency through double filtering using both context constraints from available modalities and question guidance.

3.  **Two-Stage Mixture-of-Experts Training**:

    - **Function**: Explicitly models the reliability of different information sources (original vs. recovered).
    - **Mechanism**: In the first stage, three modal experts (Visual Expert $\mathcal{E}_v$, Audio Expert $\mathcal{E}_a$, Text Expert $\mathcal{E}_t$) are independently pre-trained to extract discriminative representations without relying on cross-modal shortcuts. In the second stage, the experts are frozen, and a gating network (Router) is trained to dynamically allocate importance weights $\alpha_{m'} = \frac{\exp(g_{m'})}{\sum_{m} \exp(g_m)}$. The final joint representation is the weighted sum $\mathbf{Z}_{joint} = \alpha_a H_a + \alpha_t H_t + \alpha_v H_v$.
    - **Design Motivation**: Decoupled training prevents interdependence between experts. The gating network dynamically adjusts modal weights based on input context; recovered modalities can be assigned lower weights to reflect uncertainty.

### Loss & Training

In addition to the standard cross-entropy loss $\mathcal{L}_{task}$, a semantic ranking loss $\mathcal{L}_{rank}$ is introduced to force features recovered from positive samples to be superior to negative samples but inferior to ground-truth features: $\mathcal{L}_{total} = \mathcal{L}_{task} + \lambda(\mathcal{L}_{rank}^+ + \mathcal{L}_{rank}^-)$. This ensures that retrieved and purified features remain within an effective semantic manifold.

## Key Experimental Results

### Main Results

| Dataset | Modality Setting | Ours | Prev. SOTA (IMOL) | Gain |
|--------|---------|-----------|---------------|------|
| Music-AVQA | Missing Audio | 69.37 | 67.11 | +2.26 |
| Music-AVQA | Missing Visual | 72.06 | 69.21 | +2.85 |
| Music-AVQA | Complete | 73.19 | 71.86 | +1.33 |
| AVQA | Missing Audio | 63.25 | 61.32 | +1.93 |
| AVQA | Missing Visual | 75.12 | 72.38 | +2.74 |
| AVQA | Complete | 90.64 | 90.28 | +0.36 |

### Ablation Study

| Configuration | Music-AVQA | AVQA | Description |
|------|-----------|------|------|
| w/o CMR w/o CAP | 62.43 | 57.43 | Baseline (Missing Audio) |
| +CMR only | 67.21 | 61.78 | Retrieval provides +4.78/+4.35 |
| +CAP only | 64.11 | 59.64 | Purification alone is also effective |
| +CMR+CAP (Full) | 69.37 | 63.25 | Combination yields best results |

### Key Findings

- Retrieval-based recovery is more effective than generative completion, with the advantage being more pronounced when the visual modality is missing (+2.85 vs IMOL).
- CMR and CAP are independently effective, but their combined use is optimal, suggesting that retrieval and purification are complementary.
- In complete modality settings, R2ScP still outperforms comparison methods, indicating that the retrieval-recovery framework serves as a beneficial tool for modality fusion as well.
- The two-stage training strategy avoids cross-modal feature collapse by decoupling expert pre-training and gated mixing.

## Highlights & Insights

- Paradigm Innovation: The conceptual shift from "generative completion" to "retrieval-based recovery" is simple yet powerful, effectively avoiding the hallucination problems inherent in generative methods.
- The three-stage purification design of CAP (noise identification → semantic acquisition → selective injection) is logically rigorous, making full use of guidance signals from available modalities and questions.
- The semantic ranking loss cleverly establishes a quality gradient of "Ground Truth > Positive Retrieval > Negative Retrieval."
- The ability to improve performance even in complete modality scenarios suggests that the retrieval mechanism can serve as a general modality enhancement technique.

## Limitations & Future Work

- The construction and storage overhead of the external memory bank are significant, which may be a bottleneck for large-scale deployment.
- It is currently assumed that modalities are completely missing; scenarios with partial loss or noise degradation have not been addressed.
- Retrieval quality is highly dependent on the alignment quality of the unified semantic space (ImageBind).
- The method has only been validated on AVQA tasks; its generalization to other multimodal reasoning tasks (e.g., VQA, multimodal dialogue) remains to be explored.

## Related Work & Insights

- **vs Missing-AVQA (ECCV 2024)**: Missing-AVQA uses relation-aware generators to synthesize missing features, which results in common knowledge; R2ScP preserves modality-specific information through retrieval.
- **vs IMOL (ACL 2025)**: While IMOL also utilizes retrieval, it is primarily for contrastive alignment rather than direct feature recovery; R2ScP directly replaces missing information with retrieved features.
- **vs MoMKE (MM 2024)**: MoMKE preserves modality-specific knowledge through Mixture-of-Experts but does not handle feature recovery; R2ScP provides a more comprehensive solution by combining retrieval with an MoE architecture.

## Rating

- Novelty: ⭐⭐⭐⭐ The paradigm shift from generation to retrieval is a clear innovation, and the CAP purification mechanism is exquisitely designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covering two datasets, multiple modality missing settings, and detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear problem motivation and detailed method description.
- Value: ⭐⭐⭐⭐ Provides a new research direction for handling missing modalities in multimodal learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Music Audio-Visual Question Answering Requires Specialized Multimodal Designs](music_audio-visual_question_answering_requires_specialized_multimodal_designs.md)
- [\[ICLR 2026\] Query-Guided Spatial-Temporal-Frequency Interaction for Music Audio-Visual Question Answering](../../ICLR2026/audio_speech/query-guided_spatial-temporal-frequency_interaction_for_music_audio-visual_quest.md)
- [\[CVPR 2026\] Semantic Audio-Visual Navigation in Continuous Environments](../../CVPR2026/audio_speech/semantic_audio-visual_navigation_in_continuous_environments.md)
- [\[ACL 2026\] Jamendo-MT-QA: A Benchmark for Multi-Track Comparative Music Question Answering](jamendo-mt-qa_a_benchmark_for_multi-track_comparative_music_question_answering.md)
- [\[CVPR 2026\] ViDscribe: Multimodal AI for Customizing Audio Description and Question Answering in Online Videos](../../CVPR2026/audio_speech/vidscribe_multimodal_ai_for_customizing_audio_description_and_question_answering.md)

</div>

<!-- RELATED:END -->
