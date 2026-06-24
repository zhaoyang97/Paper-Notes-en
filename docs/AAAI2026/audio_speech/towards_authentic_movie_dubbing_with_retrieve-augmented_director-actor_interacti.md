---
title: >-
  [Paper Note] Towards Authentic Movie Dubbing with Retrieve-Augmented Director-Actor Interaction Learning
description: >-
  [AAAI 2026][Audio & Speech][Movie Dubbing] Authentic-Dubber simulates the interaction between directors and actors in real dubbing workflows. By constructing a multimodal reference footage library, applying an emotion-similarity-based retrieval-augmented strategy, and designing a progressive graph-based speech generation method, it significantly enhances the emotional expressiveness of automatic movie dubbing, achieving SOTA emotional accuracy and MOS scores on the V2C-Animat…
tags:
  - "AAAI 2026"
  - "Audio & Speech"
  - "Movie Dubbing"
  - "Emotional Expression"
  - "Retrieval-Augmented Generation"
  - "Graph Neural Networks"
  - "Multimodal Emotion Modeling"
date: 2026-05-08
content_hash: 418337a165482191
---

# Towards Authentic Movie Dubbing with Retrieve-Augmented Director-Actor Interaction Learning

**Conference**: AAAI 2026  
**arXiv**: [2511.14249](https://arxiv.org/abs/2511.14249)  
**Code**: [https://github.com/AI-S2-Lab/Authentic-Dubber](https://github.com/AI-S2-Lab/Authentic-Dubber)  
**Area**: Audio and Speech  
**Keywords**: Movie Dubbing, Emotional Expression, Retrieval-Augmented Generation, Graph Neural Networks, Multimodal Emotion Modeling

## TL;DR
Authentic-Dubber simulates the interaction between directors and actors in real dubbing workflows. By constructing a multimodal reference footage library, applying an emotion-similarity-based retrieval-augmented strategy, and designing a progressive graph-based speech generation method, it significantly enhances the emotional expressiveness of automatic movie dubbing, achieving SOTA emotional accuracy and MOS scores on the V2C-Animation dataset.

## Background & Motivation

### Background
Automatic movie dubbing (Visual Voice Cloning, V2C) aims to generate vivid speech based on a given script, while mimicking the speaker's vocal characteristics and ensuring lip synchronization. Existing research has achieved progress in pronunciation quality (Speaker2Dubber), audio-to-video synchronization (FlowDubber), and expressiveness (ProDubber).

### Limitations of Prior Work

**Existing methods simulate an oversimplified dubbing workflow**: the actor directly dubs the target clip without any preparation or reference. This overlooks the **critical interaction process between the director and the actor** in real-world dubbing environments.

In a realistic movie dubbing workflow:

1. The **director** provides rich reference materials (emotional reference footages) to the voice actor.
2. The **actor** needs to fully study and internalize the emotional cues from these materials, especially the emotional expressions.
3. Only after fully understanding the emotional context can the actor perform highly expressive dubbing.

Existing models solely rely on cross-modal modeling of the target clip itself to generate speech, leading to **limited emotional expressiveness**—since a single clip contains restricted emotional information, making it difficult for the model to capture rich emotional details.

### Key Insight
Drawing inspiration from the actual dubbing workflow, this work designs a three-stage architecture: "director providing footage $\rightarrow$ actor learning footage $\rightarrow$ actor dubbing". It introduces external emotional knowledge through retrieval-augmented generation (RAG) and accumulates emotional information using a progressive graph structure.

## Method

### Overall Architecture
Authentic-Dubber consists of three core modules: (1) Multimodal Reference Footage Library Construction, simulating the director providing references; (2) Emotion-Similarity-based Retrieval-Augmentation, simulating the actor's efficient learning process; and (3) Progressive Graph-based Speech Generation, simulating the actor's final dubbing. The inputs are the script text, silent video, and a timbre prompt audio, while the output is realistic dubbed speech rich in emotional expression.

### Key Designs

#### 1. **Multimodal Reference Footage Library (MRFL) Construction**
- **Function**: Based on the V2C dataset, emotional vectors are extracted from four modalities for each sample to construct the emotional reference footage library.
- **Mechanism**: Four dedicated emotion extractors are designed:
    - **Scene Emotion Extractor**: Uses VideoLLaMA 2 to generate scene emotion descriptions (incorporating low-level visual features such as hue, brightness, and saturation), and then extracts the scene emotion vector $S_i$ via a RoBERTa emotion model.
    - **Face Emotion Extractor**: Uses VideoLLaMA 2 to generate facial expression change descriptions, and then extracts the facial emotion vector $F_i$ via RoBERTa.
    - **Text Emotion Extractor**: A dual-path design featuring direct text emotion $T_i^{self}$ and COMET-based commonsense reaction emotion $T_i^{react}$, which are concatenated to obtain the complete text emotion vector $T_i$.
    - **Audio Emotion Extractor**: Uses Emotion2Vec to extract the audio emotion vector $A_i$.
- **Design Motivation**: Indirect emotions (scene, face, text) and direct emotions (audio) correspond to different dimensions of emotional cues. The deep understanding capabilities of LLMs unify multimodal signals into a semantic space, which is more effective than directly using I3D or EmoFan to extract embeddings (validated by ablation studies).

#### 2. **Emotion-Similarity-based Retrieval-Augmentation (ESRG)**
- **Function**: Uses the visual/textual base emotion of the target clip as queries to retrieve the most relevant multimodal emotional information from the MRFL.
- **Mechanism**:
    - **Speaker-independent strategy**: In animation dubbing scenarios, characters are virtually created, meaning reference materials for specific speakers are limited. Therefore, cross-speaker retrieval is adopted to obtain richer emotional diversity.
    - **Three-way parallel retrieval**:
    - Scene query $S$ $\rightarrow$ retrieves Top-K scene information $S_{r1 \to rk}$ + matching audio $A_{r1 \to rk}^s$
    - Face query $F$ $\rightarrow$ retrieves Top-K facial information $F_{r1 \to rk}$ + matching audio $A_{r1 \to rk}^f$
    - Text query $T$ $\rightarrow$ retrieves Top-K textual information $T_{r1 \to rk}$ + matching audio $A_{r1 \to rk}^t$
    - **Special design for text retrieval**: Computes the similarities of $T^{self}$ and $T^{react}$ separately, using the average value as the retrieval criterion.
    - **Similarity metric**: Uses cosine similarity (experimentally proven to outperform dot product and Euclidean distance).
- **Design Motivation**: In real dubbing, actors cannot pre-listen to the target speech (since it has not been recorded yet). Thus, indirect emotional information is used for retrieval, and the corresponding direct emotional audio is located via index matching.

#### 3. **Progressive Graph-based Speech Generation (PGSG)**
- **Function**: Progressively accumulates emotional knowledge through a three-layer graph structure under a progressive "construction-encoding" paradigm.
- **Mechanism**: A three-stage progressive graph structure:

  **Stage 1 — Base Emotion Graph $\mathcal{G}_{beg}$**:
  - Nodes: Target clip's scene emotion $S$, facial emotion $F$, and textual emotion $T$.
  - Edges: Fully connected between the three nodes.
  - Uses a Graph Attention Encoder (GAE) to encode and learn base emotional knowledge.

  **Stage 2 — Indirect Emotion Extended Graph $\mathcal{G}_{ieg}$**:
  - Based on the encoded $\tilde{\mathcal{G}}_{beg}$, retrieved indirect emotion nodes are added to the graph.
  - The retrieved nodes are connected to the corresponding modal base emotion nodes.
  - Encourages cumulative learning of indirect emotional information after encoding.

  **Stage 3 — Direct Emotion Extended Graph $\mathcal{G}_{deg}$**:
  - Based on the encoded $\tilde{\mathcal{G}}_{ieg}$, matching direct emotional audio is added as a new node.
  - Encodes the graph via GAE to learn direct emotional knowledge.

  **Emotional Knowledge Speech Synthesizer**:
  - The node representations of the three graph layers, $H_{beg}$, $H_{ieg}$, and $H_{deg}$, are aggregated via hierarchical cross-attention:
  $$E_{t,v,r}^{beg} = \text{Conv1D}([H_{t,v,r}; \text{CA}(H_{t,v,r}, H_{beg}, H_{beg})]$$
  - Layer-by-layer stacking: Base $\rightarrow$ Indirect $\rightarrow$ Direct, simulating the actor's process of internalizing emotions from shallow to deep.
  - The final representation is fed into a Mel-decoder to generate Mel-spectrograms, which are converted to speech using a BigVGAN vocoder.

- **Design Motivation**: The physical dubbing workflow is developmental: first understanding basic emotions, then referencing similar materials to deepen understanding, and finally performing with real audio. The progressive graph structure mirrors this sequence perfectly.

### Cross-Modal Alignment
Inherits the Cross-Modal Aligner from StyleDubber to achieve audio-visual synchronization based on the input script and visual frames, and learns speech characteristics from the timbre prompt.

## Key Experimental Results

### Main Results (V2C-Animation Dataset)

| Method | EMO-ACC(↑) | WER(↓) | SECS(↑) | MCD-DTW-SL(↓) | MOS-DE(↑) | MOS-SE(↑) |
|------|-----------|--------|---------|---------------|-----------|-----------|
| Ground-Truth | 99.96 | 22.03 | 100.00 | 0.00 | 4.416 | 4.497 |
| FastSpeech2 | 42.39 | 33.30 | 25.47 | 14.72 | 3.058 | 3.063 |
| V2C-Net | 43.07 | 67.98 | 40.65 | 19.16 | 3.146 | 3.149 |
| HPMDubbing | 43.94 | 135.72 | 34.11 | 12.64 | 3.362 | 3.320 |
| StyleDubber | 45.73 | 24.70 | 83.46 | 9.40 | 3.676 | 3.738 |
| Speaker2Dubber | 44.55 | **18.27** | 81.26 | 9.82 | 3.432 | 3.461 |
| **Authentic-Dubber** | **47.21** | 25.95 | **84.40** | **9.68** | **3.792** | **3.889** |

### Ablation Study

| # | Configuration | EMO-ACC(↑) | MOS-DE(↑) | MOS-SE(↑) |
|---|------|-----------|-----------|-----------|
| - | Complete Model | **47.21** | **3.792** | **3.889** |
| 1 | w/o Scene Caption (replaced with I3D) | 46.34 | 3.582 | 3.612 |
| 2 | w/o Face Caption (replaced with EmoFan) | 46.52 | 3.653 | 3.684 |
| 3 | w/o both Captions | 46.02 | 3.520 | 3.608 |
| 4 | w/o Scene Retrieval | 46.27 | 3.591 | 3.666 |
| 5 | w/o Face Retrieval | 46.64 | 3.657 | 3.690 |
| 6 | w/o Text Retrieval | 45.99 | 3.540 | 3.614 |
| 7 | w/o All Retrieval | 45.23 | 3.511 | 3.527 |
| 8 | w/o Indirect Information | 45.95 | 3.542 | 3.581 |
| 9 | w/o Direct Audio | 45.30 | 3.492 | 3.571 |
| 10 | w/o Graph Modeling | 45.92 | 3.518 | 3.549 |
| 11 | w/o Construction-Encoding Paradigm | 46.85 | 3.705 | 3.749 |
| 12 | w/o Hierarchical Aggregation | 46.71 | 3.661 | 3.710 |

### Key Findings
1. **Significant improvement in emotional accuracy (EMO-ACC)**: 47.21% vs. Prev. SOTA of 45.73% (StyleDubber), yielding a relative improvement of 3.2%.
2. **LLM-generated emotional descriptions are more effective than direct visual features**: Removing LLM Captions causes a decrease in EMO-ACC of 0.7-1.2%, proving the significant contribution of LLM's deep semantic understanding.
3. **Each modality in the retrieval-augmented strategy contributes**: Removing all retrieval processes decreases EMO-ACC by 2.0%, with text retrieval being the most critical (dropping by 1.2% when removed).
4. **Every component of the progressive graph structure is indispensable**: Removing direct audio or graph modeling leads to the largest performance drops.
5. **Speaker-independent retrieval outperforms speaker-specific retrieval**: The optimal configuration is achieved at K=3 (47.21%), while excessive retrieval introduces noise.
6. **Cosine similarity is the optimal similarity metric**: It is more stable than dot product and Euclidean distance.

## Highlights & Insights
1. **Unique approach to workflow modeling**: Instead of simply increasing model capacity or dataset size, it distills the core mechanism of "director-actor interaction" from physical workflows, translating domain knowledge into model designs.
2. **Natural integration of RAG and dubbing**: Analogizing reference materials to a retrieval knowledge base and emotional understanding to a knowledge-intensive task is highly intuitive.
3. **Exquisitely designed progressive graph structure**: The three-layer progression (Base $\rightarrow$ Indirect $\rightarrow$ Direct) elegantly mirrors the evolutionary emotional learning process from surface to deep comprehension.
4. **Extremely thorough ablation studies**: Twelve groups of ablations cover all major design choices, including LLM semantic understanding, retrieval strategies, and graph architectures.
5. **Practical value of speaker-independent retrieval**: In virtual character scenarios such as animation dubbing, cross-speaker retrieval yields superior effectiveness.

## Limitations & Future Work
1. Evaluated solely on the V2C-Animation dataset. Since this dataset is restricted to animated movies, the effectiveness on real-life movie dubbing remains unverified.
2. The absolute value of emotional accuracy (EMO-ACC) remains relatively low (47.21% vs. 99.96% of GT), indicating a substantial gap compared to human levels.
3. The Word Error Rate (WER) of 25.95 is not the lowest (Speaker2Dubber reaches 18.27), indicating that emotional enhancement might slightly compromise pronunciation accuracy.
4. The construction of the reference library and the retrieval process increase computational overhead during inference, which may impact real-time capability.
5. The current implementation uses a fixed K=3, lacking a mechanism for dynamic K Adjustment.
6. Explicit modeling of controllable attributes (e.g., speed, pitch) has not been explored.

## Related Work & Insights
- **RAG (Retrieval-Augmented Generation)**: The difference between Authentic-Dubber and standard RAG lies in: (1) calculating similarities across multiple emotional modalities, and (2) utilizing a progressive graph structure instead of directly concatenating retrieval results.
- **StyleDubber**: Serves as the backbone architecture for cross-modal alignment.
- **Emotion2Vec**: A general emotion representation model used to extract direct emotional audio features.
- Insight: **Formulating domain workflows into model architectures is an underestimated design methodology**, which is especially suitable for tasks with well-defined human processes.

## Rating
- **Novelty**: ⭐⭐⭐⭐ (The workflow modeling concept is novel, but individual components (RAG, GNN, LLM emotional extraction) are combinations of existing techniques.)
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ (The evaluation is exceptionally comprehensive, including main results, 12 sets of ablation studies, retrieval analysis, similarity metric analysis, and spectrogram visualization.)
- **Writing Quality**: ⭐⭐⭐⭐ (The "director-actor" metaphor flows smoothly throughout the text, maintaining a coherent narrative.)
- **Value**: ⭐⭐⭐⭐ (Promotes progress in movie dubbing and emotional speech synthesis; the combination of RAG with multimodal emotion modeling is highly inspiring.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Hearing More with Less: Multi-Modal Retrieval-and-Selection Augmented Conversational LLM-Based ASR](hearing_more_with_less_multi-modal_retrieval-and-selection_augmented_conversatio.md)
- [\[CVPR 2026\] Omni-MMSI: Toward Identity-Attributed Social Interaction Understanding](../../CVPR2026/audio_speech/omni-mmsi_toward_identity-attributed_social_interaction_understanding.md)
- [\[ACL 2026\] MARQUIS: A Three-Stage Pipeline for Video Retrieval-Augmented Generation](../../ACL2026/audio_speech/marquis_a_three-stage_pipeline_for_video_retrieval-augmented_generation.md)
- [\[ACL 2025\] WavRAG: Audio-Integrated Retrieval Augmented Generation for Spoken Dialogue Models](../../ACL2025/audio_speech/wavrag_audio-integrated_retrieval_augmented_generation_for_spoken_dialogue_model.md)
- [\[AAAI 2026\] Do LLMs Feel? Teaching Emotion Recognition with Prompts, Retrieval, and Curriculum Learning](do_llms_feel_teaching_emotion_recognition_with_prompts_retrieval_and_curriculum_.md)

</div>

<!-- RELATED:END -->
