---
title: >-
  [Paper Note] Let the Model Learn to Feel: Mode-Guided Tonality Injection for Symbolic Music Emotion Recognition
description: >-
  [AAAI 2026][Audio & Speech][symbolic music emotion recognition] Through a MoGE diagnosis strategy, it is systematically discovered that MIDIBERT fails to effectively encode mode-emotion associations. A MoFi injection framework is proposed to inject major/minor mode priors into the first layer of MIDIBERT (the layer with the weakest emotional information identified by diagnosis) via the FiLM mechanism, achieving an accuracy of 75.2% (+11.8%) on EMOPIA and 59.1% (+11.8%) on VGM…
tags:
  - "AAAI 2026"
  - "Audio & Speech"
  - "symbolic music emotion recognition"
  - "MIDIBERT"
  - "mode injection"
  - "FiLM"
  - "music psychology"
date: 2026-05-08
content_hash: 5828fcc2ec1d5b94
---

# Let the Model Learn to Feel: Mode-Guided Tonality Injection for Symbolic Music Emotion Recognition

**Conference**: AAAI 2026  
**arXiv**: [2512.17946](https://arxiv.org/abs/2512.17946)  
**Code**: [https://github.com/ZoeyHuang-paper/MoFi](https://github.com/ZoeyHuang-paper/MoFi)  
**Area**: Music Emotion Recognition / Symbolic Music Understanding  
**Keywords**: symbolic music emotion recognition, MIDIBERT, mode injection, FiLM, music psychology

## TL;DR

Through a MoGE diagnosis strategy, it is systematically discovered that MIDIBERT fails to effectively encode mode-emotion associations. A MoFi injection framework is proposed to inject major/minor mode priors into the first layer of MIDIBERT (the layer with the weakest emotional information identified by diagnosis) via the FiLM mechanism, achieving an accuracy of 75.2% (+11.8%) on EMOPIA and 59.1% (+11.8%) on VGMIDI, with F1 score improvements of 12.3%/15.5%.

## Background & Motivation

Symbolic Music Emotion Recognition (SMER) is a core task in symbolic music understanding. Pre-trained Transformer models like MIDIBERT excel in semantic understanding but exhibit fundamental flaws in emotion recognition:

| Problem | Cause | Impact |
|------|------|------|
| Lack of mode-emotion association | MLM pre-training objective focuses on token reconstruction, with no explicit motivation to learn mode | Essential music psychology rules of major (happy) / minor (sad) are not encoded |
| Extremely small SMER datasets | EMOPIA has only 1087 clips, VGMIDI has only 200 tracks | Difficult to automatically learn emotional features from small-scale data during the fine-tuning stage |
| Black-box fine-tuning | Directly fine-tuning without diagnosing which layers lack what knowledge | The injection strategy lacks targeted design and may inject at the wrong position |

*Music Psychological Foundation*: Empirical studies consistently show that major modes $\rightarrow$ high-valence positive emotions (happy, bright), and minor modes $\rightarrow$ low-valence negative emotions (sad, melancholy). This association is relatively invariant to the key: C major and G major convey similar emotions, where the determining factor is the interval structure rather than the absolute pitch. This rule has been repeatedly validated in the literature by multiple studies, including Kastner & Crowder, Gerardi & Gerken, and Dalla Bella.

## Method

### Overall Architecture

Two-stage method: **(1) MoGE Diagnosis**—systematically identifies the knowledge gaps and the optimal injection layer of MIDIBERT through data augmentation experiments and layer-wise probing; **(2) MoFi Injection**—injects mode priors at the identified target layer using a FiLM conditioning module.

### Key Designs

1. **MoGE Diagnosis — Data Augmentation Experiments**

    - Performs key-preserving pitch transposition on EMOPIA (uniformly shifting all notes within a single octave, keeping the interval structure completely intact $\rightarrow$ mode remains unchanged).
    - Result: Accuracy on original data 67.5% $\rightarrow$ 72.3% post-augmentation (+4.8%).
    - Conclusion: MIDIBERT **does not encode mode-emotion associations**; otherwise, the augmentation should not yield significant differences.

2. **MoGE Diagnosis — Layer-wise Probing**

    - Freezes all parameters of the 12 MIDIBERT layers, adding a trainable self-attention layer and a classification head in front of each layer for fine-tuning.
    - Finding: Intermediate layers perform best, the bottom layer (Layer 1) has the weakest emotional information, and the top layers are specialized for the MLM task.
    - Conclusion: Layer 1 is the optimal target position for knowledge injection.

3. **Mode Extraction**

    - Automatically extracts modes from MIDI using the Krumhansl-Kessler (K-K) algorithm (cognitive-psychology-driven, fitting human pitch class perception).
    - Employs only binary classification of major/minor (simplified for noise reduction, as rare modes like Dorian/Lydian have few samples and inconsistent emotional characteristics).
    - Output is a one-hot vector.

4. **MoFi — FiLM Injection**

    - Inserts a FiLM conditioning module between the Compound Word embedding layer and the first Transformer layer.
    - Mode one-hot vector $c$ $\rightarrow$ parameter generation network $f_\text{cond}$ $\rightarrow$ scaling factor $\gamma$ and shift factor $\beta$.
    - Affine transformation: $\text{FiLM}(x, c) = \gamma \odot x + \beta$.
    - Initializes $\gamma=1, \beta=0$ to maintain training stability, gradually integrating mode information starting from pre-trained representations.

### Loss & Training

Cross-entropy loss (Russell 4Q four-class classification: HVHA/LVHA/LVLA/HVLA). MIDIBERT has 12 layers, 12 heads, 768 dimensions, and 111M parameters. Fine-tuned with a batch size of 16 (EMOPIA) / 8 (VGMIDI) on a single 3090 GPU for $\le20$ epochs, using early stopping (patience=3) with a total time of <30 minutes.

## Key Experimental Results

### Main Results: Comparison with existing symbolic music models

| Method | Type | EMOPIA Acc↑ | EMOPIA F1↑ | VGMIDI Acc↑ | VGMIDI F1↑ |
|------|------|------------|-----------|------------|-----------|
| SVM | Traditional ML | 0.477 | 0.476 | 0.451 | 0.377 |
| LSTM-Attn | RNN | 0.647 | 0.563 | 0.417 | 0.260 |
| MIDIGPT | GPT | 0.587 | 0.572 | 0.538 | 0.505 |
| MT-MIDIBERT | Multi-task | 0.676 | 0.664 | 0.498 | 0.453 |
| BiLMA | Transformer | 0.708 | 0.631 | 0.572 | 0.478 |
| MIDIBERT (Baseline) | Pre-trained | 0.634 | 0.628 | 0.473 | 0.432 |
| **MoFi (Ours)** | **Prior Injection** | **0.752** | **0.751** | **0.591** | **0.587** |

### Ablation Study

| Configuration | EMOPIA Acc↑ | VGMIDI Acc↑ | VGMIDI F1↑ | Explanation |
|------|------------|------------|-----------|------|
| Full MoFi | **0.752** | **0.591** | **0.587** | Mode injected at Layer 1 |
| Without mode injection | 0.716 | 0.500 | 0.365 | No FiLM module |
| Injected at Layer 6 (middle layer) | 0.734 | 0.552 | 0.513 | Non-optimal position |
| Injected at last layer | 0.721 | 0.528 | 0.489 | Worse performance |
| Data augmentation only, no injection | 0.723 | - | - | Augmentation is helpful but insufficient |

### Key Findings

- The effect of mode injection is more significant on VGMIDI (F1 from 0.365 $\rightarrow$ 0.587, +60.8%), because small datasets rely more on prior knowledge to compensate for data scarcity.
- Layer 1 injection is significantly superior to intermediate and top layers, perfectly validating the effectiveness of the layer-wise probing diagnosis.
- F1 is very close to Accuracy (0.752 vs 0.751), indicating balanced four-quadrant classification without severe class bias.
- Data augmentation alone only improves the performance to 72.3%, whereas MoFi further boosts it to 75.2%, showing that the two are complementary.

## Highlights & Insights

- A general paradigm of **diagnose-then-inject**: systematically identifying knowledge gaps in the pre-trained model and then target-injecting domain priors, which can be generalized to other domains.
- FiLM injection involves extremely few parameters (adding only two linear layers), making it parameter-efficient.
- Using only binary modes (major/minor) effectively reduces noise and matches the precision requirements of four-class classification.
- An innovative integration of music psychology theories and deep learning, rendering the method highly interpretable.
- The design of initializing $\gamma=1, \beta=0$ ensures a smooth transition from the pre-trained representations.

## Limitations & Future Work

- Only considers major/minor binary classification, ignoring emotional differences in intermediate modes such as Dorian, Lydian, and Mixolydian.
- Only validated on MIDIBERT, without testing other symbolic music pre-trained models like MusicBERT or PopMAG.
- VGMIDI has only 200 tracks, which may lead to high variance in the results.
- Russell 4Q emotion classification is coarse-grained, without exploring continuous valence-arousal regression.
- Only handles single-instrument (piano) music; the effect of mode extraction and injection in multi-instrument arrangements remains unverified.

## Related Work & Insights

| Direction | Representative Method | Differences from Ours |
|------|---------|----------|
| Traditional SMER | SVM + Handcrafted Features | Unable to capture long-range temporal dependencies; heavy feature engineering. |
| Pre-trained Models | MIDIBERT, MusicBERT | MLM objective only performs token-level reconstruction, lacking emotional priors. |
| Multi-task Learning | MT-MIDIBERT | Implicitly enhanced through auxiliary tasks; does not introduce music theory priors. |
| Conditioning Techniques | FiLM (CV fields) | This work is the first to apply FiLM to symbolic music emotion recognition. |

## Rating

- Novelty: ⭐⭐⭐⭐ Innovative integration of music psychology and deep learning, with a novel diagnosis-injection paradigm.
- Experimental Thoroughness: ⭐⭐⭐ The datasets are relatively small (EMOPIA 1087, VGMIDI 200), but the ablation and diagnostic experiments are systematic.
- Writing Quality: ⭐⭐⭐⭐ Clear progression from diagnosis to solution, with a highly logical structure.
- Value: ⭐⭐⭐⭐ Provides an interpretable, theory-driven solution for symbolic music understanding, with a generalizable paradigm.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Do LLMs Feel? Teaching Emotion Recognition with Prompts, Retrieval, and Curriculum Learning](do_llms_feel_teaching_emotion_recognition_with_prompts_retrieval_and_curriculum_.md)
- [\[AAAI 2026\] Cross-Space Synergy: A Unified Framework for Multimodal Emotion Recognition in Conversation](cross-space_synergy_a_unified_framework_for_multimodal_emotion_recognition_in_co.md)
- [\[ICLR 2026\] Learnable Fractional Superlets with a Spectro-Temporal Emotion Encoder for Speech Emotion Recognition](../../ICLR2026/audio_speech/learnable_fractional_superlets_with_a_spectro-temporal_emotion_encoder_for_speec.md)
- [\[AAAI 2026\] Diff-V2M: A Hierarchical Conditional Diffusion Model with Explicit Rhythmic Modeling for Video-to-Music Generation](diff-v2m_a_hierarchical_conditional_diffusion_model_with_explicit_rhythmic_model.md)
- [\[ACL 2026\] Anchored Cyclic Generation: A Novel Paradigm for Long-Sequence Symbolic Music Generation](../../ACL2026/audio_speech/anchored_cyclic_generation_a_novel_paradigm_for_long-sequence_symbolic_music_gen.md)

</div>

<!-- RELATED:END -->
