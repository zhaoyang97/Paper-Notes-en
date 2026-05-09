---
title: >-
  [Paper Note] Towards Authentic Movie Dubbing with Retrieve-Augmented Director-Actor Interaction Learning
description: >-
  [AAAI 2026][Audio & Speech][Movie Dubbing] Authentic-Dubber simulates the director-actor interaction workflow in real-world dubbing by constructing a multimodal reference footage library, employing an emotion-similarity-based retrieval-augmented strategy, and adopting a progressive graph-based speech generation approach. The method significantly improves the emotional expressiveness of automatic movie dubbing, achieving state-of-the-art emotion accuracy and MOS scores on the V2C-Animation dataset.
tags:
  - AAAI 2026
  - "Audio & Speech"
  - Movie Dubbing
  - Emotional Expression
  - Retrieval-Augmented Generation
  - Graph Neural Networks
  - Multimodal Emotion Modeling
date: 2026-05-08
content_hash: 3ab79b9062d47694
---

# Towards Authentic Movie Dubbing with Retrieve-Augmented Director-Actor Interaction Learning

**Conference**: AAAI 2026
**arXiv**: [2511.14249](https://arxiv.org/abs/2511.14249)
**Code**: [https://github.com/AI-S2-Lab/Authentic-Dubber](https://github.com/AI-S2-Lab/Authentic-Dubber)
**Area**: Audio & Speech
**Keywords**: Movie Dubbing, Emotional Expression, Retrieval-Augmented Generation, Graph Neural Networks, Multimodal Emotion Modeling

## TL;DR
Authentic-Dubber simulates the director-actor interaction workflow in real-world dubbing by constructing a multimodal reference footage library, employing an emotion-similarity-based retrieval-augmented strategy, and adopting a progressive graph-based speech generation approach. The method significantly improves the emotional expressiveness of automatic movie dubbing, achieving state-of-the-art emotion accuracy and MOS scores on the V2C-Animation dataset.

## Background & Motivation

### State of the Field
Automatic movie dubbing (Visual Voice Cloning, V2C) aims to generate vivid speech from a given script while imitating the speaker's timbre and ensuring lip synchronization. Prior work has made progress in pronunciation quality (Speaker2Dubber), audio-visual synchronization (FlowDubber), and expressiveness (ProDubber).

### Limitations of Prior Work

**Existing methods simulate an oversimplified dubbing pipeline** in which actors dub directly from the target clip without any preparation or reference, ignoring the **critical director-actor interaction** central to real dubbing workflows.

In a real movie dubbing workflow:

1. The **director** provides rich reference materials (emotional reference clips) to the dubbing actor.
2. The **actor** must thoroughly study and internalize the emotional cues in these materials, especially emotional expressions.
3. Only after fully understanding the emotional context can the actor deliver an emotionally expressive performance.

Existing models rely solely on cross-modal modeling of the target clip itself for speech generation, resulting in **limited emotional expressiveness**—since a single clip contains limited emotional information, models struggle to capture rich emotional nuances.

### Starting Point
Drawing on real dubbing workflows, the paper proposes a three-stage architecture of "director provides materials → actor studies materials → actor dubs," introducing external emotional knowledge via retrieval-augmented generation (RAG) and accumulating emotional information through a progressive graph structure.

## Method

### Overall Architecture
Authentic-Dubber consists of three core modules: (1) Multimodal Reference Footage Library construction, simulating the director's provision of references; (2) Emotion-Similarity-based Retrieval-Augmentation, simulating the actor's efficient study of materials; and (3) Progressive Graph-based Speech Generation, simulating the actor's final dubbing. The inputs are the script text, silent video, and a timbre reference audio clip; the output is emotionally expressive dubbed speech.

### Key Designs

#### 1. **Multimodal Reference Footage Library (MRFL)**
- **Function**: Based on the V2C dataset, extracts emotion vectors across four modalities for each sample to construct an emotional reference footage library.
- **Mechanism**: Four dedicated emotion extractors are designed:
    - **Scene Emotion Extractor**: Uses VideoLLaMA 2 to generate scene emotion descriptions (incorporating low-level visual features such as hue, brightness, and saturation), then applies a RoBERTa emotion model to extract scene emotion vectors $S_i$.
    - **Facial Emotion Extractor**: Uses VideoLLaMA 2 to generate descriptions of facial expression changes, then applies RoBERTa to extract facial emotion vectors $F_i$.
    - **Text Emotion Extractor**: Dual-path design—direct text emotion $T_i^{self}$ combined with COMET-based commonsense reaction emotion $T_i^{react}$, concatenated to form the full text emotion vector $T_i$.
    - **Audio Emotion Extractor**: Uses Emotion2Vec to extract audio emotion vectors $A_i$.
- **Design Motivation**: Indirect emotions (scene/face/text) and direct emotions (audio) correspond to different dimensions of emotional cues. The deep semantic understanding capability of LLMs can unify multimodal signals into a shared semantic space, proving more effective than directly extracting embeddings with I3D or EmoFan (validated by ablation studies).

#### 2. **Emotion-Similarity-based Retrieval-Augmentation (ESRG)**
- **Function**: Uses the basic emotion of the target clip as a query to retrieve the most relevant multimodal emotional information from the MRFL.
- **Mechanism**:
    - **Speaker-independent strategy**: In animated dubbing, characters are virtually created and speaker-specific reference materials are limited; cross-speaker retrieval is therefore adopted to obtain richer emotional diversity.
    - **Three-path parallel retrieval**:
        - Scene query $S$ → retrieve Top-K scene information $S_{r1 \to rk}$ + matched audio $A_{r1 \to rk}^s$
        - Face query $F$ → retrieve Top-K facial information $F_{r1 \to rk}$ + matched audio $A_{r1 \to rk}^f$
        - Text query $T$ → retrieve Top-K text information $T_{r1 \to rk}$ + matched audio $A_{r1 \to rk}^t$
    - Special design for text retrieval: similarity scores of $T^{self}$ and $T^{react}$ are computed separately and averaged as the retrieval criterion.
    - Similarity metric: cosine similarity (experimentally shown to outperform dot product and Euclidean distance).
- **Design Motivation**: In real dubbing, actors cannot access the target speech (since it does not yet exist), so indirect emotional information is used for retrieval, with matching direct emotional audio obtained via index lookup.

#### 3. **Progressive Graph-based Speech Generation (PGSG)**
- **Function**: Progressively accumulates emotional knowledge through a three-layer graph structure in a "build-then-encode" paradigm.
- **Mechanism**: Three-stage progressive graph structure:

  **Stage 1 — Basic Emotion Graph $\mathcal{G}_{beg}$**:
  - Nodes: scene emotion $S$, facial emotion $F$, and text emotion $T$ of the target clip.
  - Edges: pairwise connections among the three nodes.
  - Encoded by a Graph Attention Encoder (GAE) to learn basic emotional knowledge.

  **Stage 2 — Indirect Emotion Expansion Graph $\mathcal{G}_{ieg}$**:
  - Retrieved indirect emotion nodes are appended to the encoded $\tilde{\mathcal{G}}_{beg}$.
  - Retrieved nodes are connected to the basic emotion node of the same modality.
  - Encoded to accumulate indirect emotional information.

  **Stage 3 — Direct Emotion Expansion Graph $\mathcal{G}_{deg}$**:
  - Matched direct emotional audio nodes are appended to the encoded $\tilde{\mathcal{G}}_{ieg}$.
  - Encoded by GAE to learn direct emotional knowledge.

  **Emotional Knowledge Speech Synthesizer**:
  - Node representations $H_{beg}$, $H_{ieg}$, $H_{deg}$ from the three graph layers are aggregated via hierarchical cross-attention:
  $$E_{t,v,r}^{beg} = \text{Conv1D}([H_{t,v,r}; \text{CA}(H_{t,v,r}, H_{beg}, H_{beg})])$$
  - Representations are stacked layer by layer: basic → indirect → direct, simulating the actor's progressive internalization of emotion.
  - The final representations are fed into a Mel decoder to generate Mel spectrograms, which are then converted to speech via the BigVGAN vocoder.

- **Design Motivation**: Real dubbing is a progressive process: first understanding basic emotions, then deepening comprehension through similar reference materials, and finally performing with real audio as guidance. The progressive graph structure precisely mirrors this workflow.

### Cross-Modal Alignment
The cross-modal aligner from StyleDubber is inherited to achieve audio-visual synchronization based on the input script and visual frames, and to learn voice characteristics from the timbre reference clip.

## Key Experimental Results

### Main Results (V2C-Animation Dataset)

| Method | EMO-ACC(↑) | WER(↓) | SECS(↑) | MCD-DTW-SL(↓) | MOS-DE(↑) | MOS-SE(↑) |
|--------|-----------|--------|---------|---------------|-----------|-----------|
| Ground-Truth | 99.96 | 22.03 | 100.00 | 0.00 | 4.416 | 4.497 |
| FastSpeech2 | 42.39 | 33.30 | 25.47 | 14.72 | 3.058 | 3.063 |
| V2C-Net | 43.07 | 67.98 | 40.65 | 19.16 | 3.146 | 3.149 |
| HPMDubbing | 43.94 | 135.72 | 34.11 | 12.64 | 3.362 | 3.320 |
| StyleDubber | 45.73 | 24.70 | 83.46 | 9.40 | 3.676 | 3.738 |
| Speaker2Dubber | 44.55 | **18.27** | 81.26 | 9.82 | 3.432 | 3.461 |
| **Authentic-Dubber** | **47.21** | 25.95 | **84.40** | **9.68** | **3.792** | **3.889** |

### Ablation Study

| # | Configuration | EMO-ACC(↑) | MOS-DE(↑) | MOS-SE(↑) |
|---|--------------|-----------|-----------|-----------|
| - | Full Model | **47.21** | **3.792** | **3.889** |
| 1 | w/o Scene Caption (replaced by I3D) | 46.34 | 3.582 | 3.612 |
| 2 | w/o Face Caption (replaced by EmoFan) | 46.52 | 3.653 | 3.684 |
| 3 | w/o Both Captions | 46.02 | 3.520 | 3.608 |
| 4 | w/o Scene Retrieval | 46.27 | 3.591 | 3.666 |
| 5 | w/o Face Retrieval | 46.64 | 3.657 | 3.690 |
| 6 | w/o Text Retrieval | 45.99 | 3.540 | 3.614 |
| 7 | w/o All Retrieval | 45.23 | 3.511 | 3.527 |
| 8 | w/o Indirect Information | 45.95 | 3.542 | 3.581 |
| 9 | w/o Direct Audio | 45.30 | 3.492 | 3.571 |
| 10 | w/o Graph Modeling | 45.92 | 3.518 | 3.549 |
| 11 | w/o Build-Encode Paradigm | 46.85 | 3.705 | 3.749 |
| 12 | w/o Hierarchical Aggregation | 46.71 | 3.661 | 3.710 |

### Key Findings
1. **Significant improvement in emotion accuracy (EMO-ACC)**: 47.21% vs. the previous SOTA of 45.73% (StyleDubber), a relative improvement of 3.2%.
2. **LLM-generated emotion descriptions outperform direct visual features**: Removing LLM captions reduces EMO-ACC by 0.7–1.2%, demonstrating the substantial contribution of LLM deep semantic understanding.
3. **Each retrieval modality contributes independently**: Removing all retrieval causes a 2.0% drop in EMO-ACC; text retrieval is the most important (1.2% drop upon removal).
4. **Every component of the progressive graph structure is indispensable**: Removing direct audio or graph modeling results in the largest performance degradation.
5. **Speaker-independent retrieval outperforms speaker-specific retrieval**: $K=3$ achieves the optimal 47.21%; excessive retrieval introduces noise.
6. **Cosine similarity is the optimal similarity metric**: More stable than dot product and Euclidean distance.

## Highlights & Insights
1. **Unique workflow-modeling perspective**: Rather than simply increasing model capacity or data volume, the paper distills the core "director-actor interaction" mechanism from real-world practice, translating domain knowledge into model design.
2. **Natural and well-motivated integration of RAG with dubbing**: Treating reference materials as a retrieval knowledge base and emotional understanding as a knowledge-intensive task is a highly apt analogy.
3. **Elegant progressive graph structure**: The three-level progression from basic emotion → indirect emotion → direct emotion corresponds to a process of emotional understanding from shallow to deep.
4. **Extremely thorough ablation study**: Twelve ablation groups cover all design choices, including LLM semantic understanding, retrieval strategies, and graph structure.
5. **Practical value of speaker-independent retrieval findings**: In virtual character scenarios such as animated dubbing, cross-speaker retrieval yields better results.

## Limitations & Future Work
1. Evaluation is limited to a single dataset (V2C-Animation), which consists of animated films; performance on live-action movie dubbing remains unknown.
2. The absolute value of emotion accuracy (EMO-ACC) remains low (47.21% vs. 99.96% for ground truth), indicating a substantial gap from human-level performance.
3. WER (25.95) is not optimal (Speaker2Dubber achieves 18.27), suggesting that emotion enhancement may slightly compromise pronunciation accuracy.
4. Library construction and retrieval processes introduce additional computational overhead during inference, potentially affecting real-time applicability.
5. The Top-K value is fixed at $K=3$, with no mechanism for dynamic adjustment.
6. Explicit modeling of controllable attributes (e.g., speech rate, pitch) is not explored.

## Related Work & Insights
- **RAG (Retrieval-Augmented Generation)**: Authentic-Dubber differs from standard RAG in that (1) it computes similarity across multiple emotional modalities and (2) it employs a progressive graph structure rather than directly concatenating retrieved results.
- **StyleDubber**: Serves as the base architecture for cross-modal alignment.
- **Emotion2Vec**: A general emotion representation model used to extract direct emotional audio features.
- Insight: **Translating domain workflows into model architectures is an underutilized design methodology**, particularly well-suited for tasks with well-defined human processes.

## Rating
- Novelty: ⭐⭐⭐⭐ (Workflow-modeling perspective is novel, though individual components—RAG, GNN, LLM-based emotion extraction—are combinations of existing techniques.)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Main results + 12 ablation groups + retrieval analysis + similarity metric analysis + spectrogram visualization; very comprehensive.)
- Writing Quality: ⭐⭐⭐⭐ (The "director-actor" metaphor runs throughout the paper, with a coherent narrative.)
- Value: ⭐⭐⭐⭐ (Advances the fields of movie dubbing and expressive speech synthesis; the combination of RAG with multimodal emotion modeling is inspiring.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Hearing More with Less: Multi-Modal Retrieval-and-Selection Augmented Conversational LLM-Based ASR](hearing_more_with_less_multi-modal_retrieval-and-selection_augmented_conversatio.md)
- [\[AAAI 2026\] Gene Incremental Learning for Single-Cell Transcriptomics](gene_incremental_learning_for_single-cell_transcriptomics.md)
- [\[CVPR 2026\] Omni-MMSI: Toward Identity-Attributed Social Interaction Understanding](../../CVPR2026/audio_speech/omni-mmsi_toward_identity-attributed_social_interaction_understanding.md)
- [\[AAAI 2026\] Do LLMs Feel? Teaching Emotion Recognition with Prompts, Retrieval, and Curriculum Learning](do_llms_feel_teaching_emotion_recognition_with_prompts_retrieval_and_curriculum_.md)
- [\[ACL 2026\] Learning Invariant Modality Representation for Robust Multimodal Learning from a Causal Inference Perspective](../../ACL2026/audio_speech/learning_invariant_modality_representation_for_robust_multimodal_learning_from_a.md)

</div>

<!-- RELATED:END -->
