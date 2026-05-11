---
title: >-
  [Paper Note] Discovering and Steering Interpretable Concepts in Large Generative Music Models
description: >-
  [ICLR 2026][Audio & Speech][Sparse Autoencoder] This work presents the first application of Sparse Autoencoders (SAEs) to the audio/music domain…
tags:
  - "ICLR 2026"
  - "Audio & Speech"
  - "Sparse Autoencoder"
  - "Music Generation"
  - "Interpretability"
  - "MusicGen"
  - "Feature Steering"
date: 2026-05-08
content_hash: f12835d20db56574
---

# Discovering and Steering Interpretable Concepts in Large Generative Music Models

**Conference**: ICLR 2026
**arXiv**: [2505.18186](https://arxiv.org/abs/2505.18186)
**Code**: [musicdiscovery.media.mit.edu](https://musicdiscovery.media.mit.edu)
**Area**: Audio & Speech
**Keywords**: Sparse Autoencoder, Music Generation, Interpretability, MusicGen, Feature Steering

## TL;DR

This work presents the first application of Sparse Autoencoders (SAEs) to the audio/music domain, extracting interpretable musical concept features from the residual stream of the autoregressive music generation model MusicGen, and leveraging these features for controllable generation (steering).

## Background & Motivation

- Deep generative models produce high-quality music, implying that they have learned an implicit theory of musical structure internally; however, these internal representations remain a black box to humans.
- Existing probing methods can only verify whether a model encodes **already-known** concepts (e.g., chords, beats), and cannot discover **unknown** structures that the model has learned on its own.
- The music domain lacks large-scale paired music–text data, making concept discovery particularly challenging.
- In NLP and vision, SAEs have been shown to extract interpretable sparse features from Transformer activations (Templeton et al., 2024), yet they have not been applied to the audio modality.

Core motivation: **Shifting from "did the model learn X?" to "what did the model actually learn?"**—unsupervisedly discovering the full set of musical concepts encoded within the model.

## Core Problem

1. How to **unsupervisedly** discover interpretable musical concepts from the intermediate representations of a music generation model?
2. How to **automatically and at scale** evaluate and annotate thousands of latent features?
3. Can the discovered features **causally** steer the generation output?

## Method

### Overall Architecture

The pipeline consists of three stages: activation extraction → SAE training and feature filtering → automatic annotation and human validation.

### 1. Dataset and Activation Extraction

- Uses the MusicSet dataset (~160k clips of ~10s each, sourced from MTG-Jamendo / MusicCaps / MusicBench).
- Audio is fed into pretrained MusicGen-Large (MGL, $d=2048$) and MusicGen-Small (MGS, $d=1024$).
- Activations are extracted from 5 residual stream layers: layer 2 (early), layers at 25%/50%/75% depth, and the second-to-last layer (late).
- MGL layer indices: $\{2, 12, 24, 36, 46\}$; MGS layer indices: $\{2, 6, 12, 18, 22\}$.

### 2. Sparse Autoencoder (SAE) Training

- Uses a $k$-sparse autoencoder architecture: encoder $\mathbf{h} = \text{ReLU}(\mathbf{W}_e \mathbf{x} + \mathbf{b}_e)$, followed by a top-$k$ projection retaining the $k$ largest activations.
- Decoder $\hat{\mathbf{x}} = \mathbf{W}_d \mathbf{h} + \mathbf{b}_d$, minimizing MSE reconstruction loss.
- Hyperparameter combinations: expansion factor $\epsilon \in \{4, 32\}$, sparsity $k \in \{32, 100\}$.

### 3. Feature Filtering

For each feature, its activation rate $r_i$ across all tracks in the validation set is computed, and three categories of invalid features are removed:

- **Never activating**: $r_i = 0$
- **Overly common**: $r_i > 0.25$ (activates in more than 25% of tracks; semantically ambiguous)
- **Overly rare**: $r_i < 0.01$ (insufficient coverage for reliable interpretation)

### 4. Automatic Annotation Pipeline

Three complementary strategies applied in parallel:

- **Generative annotation**: The concatenated audio of the top-10 highest-activating samples per feature is passed to Gemini Flash 1.5, prompting the multimodal LLM to generate concept labels, confidence scores, and descriptions.
- **Classifier-based annotation**: Pretrained Essentia audio classifiers are used to extract labels (genre, instrument, mood, etc.).
- **CLAP alignment scoring**: The CLAP embedding cosine similarity between label text and activating audio samples is computed to quantify label quality.

### 5. Generation Steering

During forward inference, a scaled decoder weight vector is injected into the residual stream at the SAE layer:

$$\mathbf{x}' = \mathbf{x} + \alpha \cdot \beta \cdot \mathbf{W}_{d,j}$$

where $\alpha \in (0,1)$ is the steering intensity and $\beta$ is the maximum activation magnitude of feature $j$. A neutral prompt "Simple melody" is used for testing, comparing outputs at $\alpha=0$ (baseline) and $\alpha=1$ (maximum steering).

## Key Experimental Results

### Feature Discovery Statistics

- A total of **4,697 valid features** are retained after filtering.
- MGL substantially outperforms MGS: MGL at $\epsilon=32, k=100$ on layer L2 yields up to 2,344 features; MGS rarely exceeds 100 features across all configurations.
- Expansion factor 32 combined with $k=100$ performs best.

### Automatic Annotation Quality

- CLAP alignment scores for Essentia classifier labels are consistently higher than for Gemini-generated labels.
- Human evaluation (400 features/method, 80 participants): Essentia confidence 3.96/5 (71% > 4), Gemini confidence 3.19/5 (47% > 4).

### Layer-wise Patterns

- Deeper MGL features exhibit higher CLAP scores, indicating that **deeper layers encode more interpretable concepts**.
- Layer-prediction MLP accuracy: MGL 50.29%, MGS 40.51%—features in larger models exhibit **more pronounced cross-layer differentiation**.

### Steering Effectiveness

- Across different SAE configurations, **15%–35% of features** demonstrate positive steering improvement.
- Best configuration: MGL L36, $\epsilon=32, k=100$ achieves 35.1% positive improvement.
- Human listening test (10 participants × 10 groups): 66/100 trials correctly identified SAE-steered audio (vs. 17 for baseline and 17 for chance), $\chi^2=48.02, p<.0001$.

## Highlights & Insights

- **First SAE application to audio**: Successfully transfers the SAE interpretability methodology from NLP/vision to music generation models, opening a new research direction.
- **Unsupervised concept discovery**: Recovers classical music concepts (taiko drums, Hardstyle Techno, Baroque harpsichord, rock guitar solos) while also uncovering **novel patterns not captured by existing theory** (e.g., "electronic beeps and boops," "single instrument single note," "oscillating bell timbre").
- **Comprehensive evaluation framework**: A multi-level evaluation pipeline combining multimodal LLMs, pretrained classifiers, CLAP alignment, and human validation.
- **Causal validation via steering**: Steering experiments provide causal evidence that the discovered features correspond to **actionable directions** within the model's internal representations.

## Limitations & Future Work

- Steering success rate is only 15%–35%; most features, though interpretable, are not necessarily steerable.
- Validation is limited to MusicGen; diffusion-based music generation models and other architectures are not tested.
- Automatic annotation remains limited: Gemini label quality is less consistent than classifier labels, and the accuracy of open-ended labels warrants further improvement.
- Feature filtering thresholds (1%–25%) are heuristically set and may exclude borderline cases.
- MGS yields very few valid features (< 10 in most configurations); the lower-bound effect of model scale is insufficiently discussed.
- Activations are extracted using unconditional audio only; feature differences under conditional generation scenarios are not explored.

## Related Work & Insights

| Method | Strategy | Concept Source | Limitations |
|--------|----------|---------------|-------------|
| Probing (Wei et al., 2024a; Ma & Xia, 2024) | Supervised probing | Predefined known concepts | Can only verify known concepts |
| DecoderLens (Vásquez et al., 2024) | Intermediate activation visualization | Layer-wise "auditory" evolution | Primarily qualitative |
| Concept Bottleneck Models | Bottleneck layer constraints | Manually specified concept sets | Requires prior knowledge |
| Protein language model SAE (Simon & Zou, 2024) | SAE feature discovery | Unsupervised | Different domain |
| **Ours** | **SAE + automatic annotation + steering** | **Unsupervised discovery** | **First audio application with causal validation** |

The expansion of the SAE interpretability paradigm from text → vision → proteins → audio suggests that this approach has cross-modal generality and may extend further to video generation, 3D generation, and beyond.

The finding that "concepts learned by the model may transcend existing human theoretical frameworks" is of significant value to music theory research—positioning AI as a discovery tool.

The steering mechanism provides a novel paradigm for controllable generation that operates directly on internal representations, without relying on text prompts or conditioning signals.

Layer-wise differentiation patterns (deeper layers more interpretable; larger models exhibiting greater cross-layer feature differentiation) are consistent with findings in NLP, further supporting the Transformer hypothesis that shallow layers encode low-level features while deeper layers encode high-level semantics.

## Rating

- Novelty: 9/10 (first SAE application to audio; dual contribution of concept discovery and steering)
- Experimental Thoroughness: 8/10 (multiple models, layers, and hyperparameter combinations with human evaluation, but limited to MusicGen)
- Writing Quality: 9/10 (clear exposition, rich figures, complete description of each pipeline stage)
- Value: 8/10 (opens a new direction for music model interpretability; steering has practical value but success rate remains to be improved)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Aligning Generative Music AI with Human Preferences: Methods and Challenges](../../AAAI2026/audio_speech/aligning_generative_music_ai_with_human_preferences_methods_and_challenges.md)
- [\[ICLR 2026\] Improving Black-Box Generative Attacks via Generator Semantic Consistency](improving_black-box_generative_attacks_via_generator_semantic_consistency.md)
- [\[ACL 2026\] HalluAudio: A Comprehensive Benchmark for Hallucination Detection in Large Audio-Language Models](../../ACL2026/audio_speech/halluaudio_a_comprehensive_benchmark_for_hallucination_detection_in_large_audio-.md)
- [\[ACL 2026\] Temporal Contrastive Decoding: A Training-Free Method for Large Audio-Language Models](../../ACL2026/audio_speech/temporal_contrastive_decoding_a_training-free_method_for_large_audio-language_mo.md)
- [\[AAAI 2026\] AHAMask: Reliable Task Specification for Large Audio Language Models without Instructions](../../AAAI2026/audio_speech/ahamask_reliable_task_specification_for_large_audio_language.md)

</div>

<!-- RELATED:END -->
