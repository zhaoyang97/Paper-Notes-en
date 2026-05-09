---
title: >-
  [Paper Note] Let the Model Learn to Feel: Mode-Guided Tonality Injection for Symbolic Music Emotion Recognition
description: >-
  [AAAI 2026][Audio & Speech][symbolic music emotion recognition] The MoGE diagnostic strategy systematically identifies that MIDIBERT fails to encode mode–emotion associations. The proposed MoFi injection framework leverages FiLM conditioning to inject major/minor priors into Layer 1 of MIDIBERT (identified as the weakest layer in terms of emotional information). This achieves 75.2% accuracy (+11.8%) on EMOPIA and 59.1% (+11.8%) on VGMIDI, with F1 improvements of 12.3% and 15.5%, respectively.
tags:
  - AAAI 2026
  - "Audio & Speech"
  - symbolic music emotion recognition
  - MIDIBERT
  - mode injection
  - FiLM
  - music psychology
date: 2026-05-08
content_hash: 46fd50f46209aa7c
---

# Let the Model Learn to Feel: Mode-Guided Tonality Injection for Symbolic Music Emotion Recognition

**Conference**: AAAI 2026
**arXiv**: [2512.17946](https://arxiv.org/abs/2512.17946)
**Code**: [https://github.com/ZoeyHuang-paper/MoFi](https://github.com/ZoeyHuang-paper/MoFi)
**Area**: Music Emotion Recognition / Symbolic Music Understanding
**Keywords**: symbolic music emotion recognition, MIDIBERT, mode injection, FiLM, music psychology

## TL;DR

The MoGE diagnostic strategy systematically identifies that MIDIBERT fails to encode mode–emotion associations. The proposed MoFi injection framework leverages FiLM conditioning to inject major/minor priors into Layer 1 of MIDIBERT (identified as the weakest layer in terms of emotional information). This achieves 75.2% accuracy (+11.8%) on EMOPIA and 59.1% (+11.8%) on VGMIDI, with F1 improvements of 12.3% and 15.5%, respectively.

## Background & Motivation

Symbolic Music Emotion Recognition (SMER) is a core task in symbolic music understanding. Transformer-based pretrained models such as MIDIBERT excel at semantic comprehension but exhibit fundamental deficiencies in emotion recognition:

| Problem | Cause | Impact |
|---------|-------|--------|
| Missing mode–emotion association | MLM pretraining focuses on token reconstruction without explicit incentive to learn mode | The core music-psychological principle of major (happy) / minor (sad) is not encoded |
| Extremely small SMER datasets | EMOPIA has only 1,087 clips; VGMIDI has only 200 tracks | Fine-tuning cannot automatically learn emotional features from limited data |
| Black-box fine-tuning | Fine-tuning proceeds without diagnosing which layer lacks which knowledge | Injection strategies lack specificity and may be applied at the wrong layer |

**Music psychology basis**: Empirical studies consistently demonstrate that major mode → high-valence positive emotions (happiness, brightness) and minor mode → low-valence negative emotions (sadness, melancholy). This association is relatively invariant to key: C major and G major convey similar emotions, as the determining factor is interval structure rather than absolute pitch. This regularity has been repeatedly validated by Kastner & Crowder, Gerardi & Gerken, Dalla Bella, and others.

## Method

### Overall Architecture

A two-stage approach: **(1) MoGE Diagnosis**—systematically identifies MIDIBERT's knowledge gaps and optimal injection location via data augmentation experiments and layer-wise probing; **(2) MoFi Injection**—injects mode priors into the identified target layer via a FiLM conditioning module.

### Key Designs

1. **MoGE Diagnosis — Data Augmentation Experiment**

    - Mode-preserving pitch transposition is applied to EMOPIA (all notes shifted uniformly within a single octave, preserving interval structure → mode unchanged).
    - Result: accuracy improves from 67.5% (original) to 72.3% (+4.8%) with augmentation.
    - Conclusion: MIDIBERT **does not encode mode–emotion associations**; otherwise, augmentation should yield no significant difference.

2. **MoGE Diagnosis — Layer-wise Probing**

    - All 12 layers of MIDIBERT are frozen; a trainable self-attention head and classifier are inserted before each layer for fine-tuning.
    - Finding: middle layers yield the best performance; the bottom layer (Layer 1) contains the weakest emotional information; upper layers are specialized for MLM.
    - Conclusion: Layer 1 is the optimal target for knowledge injection.

3. **Mode Extraction**

    - The Krumhansl-Kessler (K-K) algorithm is used to automatically extract mode from MIDI (cognitively motivated, fitting human tonal perception).
    - Only binary major/minor classification is adopted (simplified to reduce noise; rare modes such as Dorian/Lydian have insufficient samples and inconsistent emotional characteristics).
    - Output is a one-hot vector.

4. **MoFi — FiLM Injection**

    - A FiLM conditioning module is inserted between the Compound Word embedding layer and the first Transformer layer.
    - Mode one-hot vector $c$ → parameter generation network $f_\text{cond}$ → scale factor $\gamma$ and shift factor $\beta$.
    - Affine transformation: $\text{FiLM}(x, c) = \gamma \odot x + \beta$
    - Initialization with $\gamma=1, \beta=0$ ensures training stability, allowing the model to gradually incorporate mode information starting from pretrained representations.

### Loss & Training

Cross-entropy loss (Russell 4Q four-class classification: HVHA/LVHA/LVLA/HVLA). MIDIBERT: 12 layers / 12 heads / 768 dimensions / 111M parameters. Batch size 16 (EMOPIA) / 8 (VGMIDI), fine-tuned on a single RTX 3090 GPU for ≤20 epochs with early stopping (patience=3); total training time <30 minutes.

## Key Experimental Results

### Main Results: Comparison with Existing Symbolic Music Models

| Method | Type | EMOPIA Acc↑ | EMOPIA F1↑ | VGMIDI Acc↑ | VGMIDI F1↑ |
|--------|------|------------|-----------|------------|-----------|
| SVM | Traditional ML | 0.477 | 0.476 | 0.451 | 0.377 |
| LSTM-Attn | RNN | 0.647 | 0.563 | 0.417 | 0.260 |
| MIDIGPT | GPT | 0.587 | 0.572 | 0.538 | 0.505 |
| MT-MIDIBERT | Multi-task | 0.676 | 0.664 | 0.498 | 0.453 |
| BiLMA | Transformer | 0.708 | 0.631 | 0.572 | 0.478 |
| MIDIBERT (baseline) | Pretrained | 0.634 | 0.628 | 0.473 | 0.432 |
| **MoFi (Ours)** | **Prior Injection** | **0.752** | **0.751** | **0.591** | **0.587** |

### Ablation Study

| Configuration | EMOPIA Acc↑ | VGMIDI Acc↑ | VGMIDI F1↑ | Notes |
|---------------|------------|------------|-----------|-------|
| Full MoFi | **0.752** | **0.591** | **0.587** | Mode injection at Layer 1 |
| w/o mode injection | 0.716 | 0.500 | 0.365 | No FiLM module |
| Inject at Layer 6 (middle) | 0.734 | 0.552 | 0.513 | Suboptimal position |
| Inject at last layer | 0.721 | 0.528 | 0.489 | Further degraded |
| Data augmentation only, no injection | 0.723 | — | — | Augmentation helps but is insufficient |

### Key Findings

- Mode injection yields a more pronounced effect on VGMIDI (F1: 0.365→0.587, +60.8%), as smaller datasets rely more heavily on prior knowledge to compensate for data scarcity.
- Layer 1 injection significantly outperforms middle and final layer injection, validating the effectiveness of layer-wise probing diagnosis.
- F1 and accuracy are closely aligned (0.752 vs. 0.751), indicating balanced four-quadrant classification without severe class bias.
- Data augmentation alone improves accuracy to 72.3%; MoFi further raises it to 75.2%, demonstrating complementarity.

## Highlights & Insights

- **Diagnose-then-inject** as a general paradigm: systematically identifying knowledge gaps in pretrained models before targeted domain prior injection, transferable to other domains.
- FiLM injection introduces minimal parameters (only two additional linear layers), making it parameter-efficient.
- Using only binary mode (major/minor) effectively reduces noise and aligns with the requirements of four-class classification.
- An innovative integration of music psychology theory with deep learning, yielding strong interpretability.
- The $\gamma=1, \beta=0$ initialization ensures a smooth transition from pretrained representations.

## Limitations & Future Work

- Only binary major/minor mode is considered, ignoring emotional distinctions of intermediate modes such as Dorian, Lydian, and Mixolydian.
- Validation is limited to MIDIBERT; other symbolic music pretrained models such as MusicBERT and PopMAG are not evaluated.
- VGMIDI contains only 200 tracks, potentially leading to high variance in results.
- The Russell 4Q emotion taxonomy has coarse granularity; continuous valence-arousal regression is not explored.
- Only single-instrument (piano) settings are addressed; mode extraction and injection for multi-instrument arrangements remain unverified.

## Related Work & Insights

| Direction | Representative Methods | Difference from This Work |
|-----------|----------------------|--------------------------|
| Traditional SMER | SVM + handcrafted features | Cannot capture long-range temporal dependencies; heavy feature engineering |
| Pretrained models | MIDIBERT, MusicBERT | MLM objective limited to token-level reconstruction; lacks emotional priors |
| Multi-task learning | MT-MIDIBERT | Implicitly enhances via auxiliary tasks; does not introduce music-theoretic priors |
| Conditioning techniques | FiLM (computer vision) | This work is the first to apply FiLM to symbolic music emotion recognition |

## Rating

- **Novelty**: ⭐⭐⭐⭐ Innovative integration of music psychology and deep learning; the diagnose-then-inject paradigm is original.
- **Experimental Thoroughness**: ⭐⭐⭐ Datasets are small (EMOPIA 1,087; VGMIDI 200), but ablation and diagnostic experiments are systematic.
- **Writing Quality**: ⭐⭐⭐⭐ Logical progression from diagnosis to solution; reasoning is clear and well-structured.
- **Value**: ⭐⭐⭐⭐ Provides an interpretable, theory-driven approach for symbolic music understanding with generalizable paradigm.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Do LLMs Feel? Teaching Emotion Recognition with Prompts, Retrieval, and Curriculum Learning](do_llms_feel_teaching_emotion_recognition_with_prompts_retrieval_and_curriculum_.md)
- [\[AAAI 2026\] Cross-Space Synergy: A Unified Framework for Multimodal Emotion Recognition in Conversation](cross-space_synergy_a_unified_framework_for_multimodal_emotion_recognition_in_co.md)
- [\[ACL 2026\] Anchored Cyclic Generation: A Novel Paradigm for Long-Sequence Symbolic Music Generation](../../ACL2026/audio_speech/anchored_cyclic_generation_a_novel_paradigm_for_long-sequence_symbolic_music_gen.md)
- [\[ICLR 2026\] Query-Guided Spatial-Temporal-Frequency Interaction for Music Audio-Visual Question Answering](../../ICLR2026/audio_speech/query-guided_spatial-temporal-frequency_interaction_for_music_audio-visual_quest.md)
- [\[ICLR 2026\] Dynamic Parameter Memory: Temporary LoRA-Enhanced LLM for Long-Sequence Emotion Recognition in Conversation](../../ICLR2026/audio_speech/dynamic_parameter_memory_temporary_lora-enhanced_llm_for_long-sequence_emotion_r.md)

</div>

<!-- RELATED:END -->
