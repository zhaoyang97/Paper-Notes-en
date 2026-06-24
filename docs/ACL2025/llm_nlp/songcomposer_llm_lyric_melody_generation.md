---
title: >-
  [Paper Note] SongComposer: A Large Language Model for Lyric and Melody Generation in Song Composition
description: >-
  [ACL 2025][LLM (Other)][Song composition] SongComposer is the first music-specific Large Language Model capable of simultaneously generating lyrics and melodies. Utilizing a word-level aligned tuple format, a music-knowledge-based scalar pitch initialization, and progressive structure-aware training (motif -> independent song -> phrase-level pairing), it comprehensively outperforms GPT-4 on tasks including lyric-to-melody, melody-to-lyric, song continuation…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Song composition"
  - "Large Language Model"
  - "Melody generation"
  - "Lyric generation"
  - "Symbolic music representation"
date: 2026-05-08
content_hash: 20e2c21282bd2286
---

# SongComposer: A Large Language Model for Lyric and Melody Generation in Song Composition

**Conference**: ACL 2025  
**arXiv**: [2402.17645](https://arxiv.org/abs/2402.17645)  
**Code**: [Project Page](https://pjlab-songcomposer.github.io/)  
**Area**: LLM/NLP  
**Keywords**: Song composition, Large Language Model, Melody generation, Lyric generation, Symbolic music representation  

## TL;DR

SongComposer is the first music-specific Large Language Model capable of simultaneously generating lyrics and melodies. Utilizing a word-level aligned tuple format, a music-knowledge-based scalar pitch initialization, and progressive structure-aware training (motif -> independent song -> phrase-level pairing), it comprehensively outperforms GPT-4 on tasks including lyric-to-melody, melody-to-lyric, song continuation, and text-to-song generation.

---

## Background & Motivation

**Background:** Symbolic song composition aims to generate vocal tracks consisting of lyrics and melodies as symbolic sequences, serving as a core task of song generation. While subtasks such as lyric generation, melody generation, lyric-to-melody, and melody-to-lyric have recently progressed individually, there is a lack of a unified framework capable of handling both lyrics and melodies simultaneously.

**Limitations of Prior Work:** (1) Traditional models like SongMASS and TeleMelody can only handle a single subtask, failing to cover all composition needs within a single model; (2) Directly using LLMs for song composition faces three major challenges: unexplored alignment methods for lyrics and melodies, difficulty in modeling hierarchical structures (motifs and phrases) of songs, and scarcity of high-quality paired datasets.

**Design Motivation:** The symbolic representation of songs shares structural similarities with natural language, and the instruction-following capabilities of LLMs allow the integration of multiple subtasks into a single model. However, three key technical challenges must be addressed: symbolic representation design, pitch understanding, and structural modeling.

---

## Method

### Overall Architecture

SongComposer is built upon InternLM2-7B and introduces three core innovations: a word-level tuple representation format, scalar pitch initialization, and three-stage progressive training. The training data is derived from the custom-built SongCompose dataset (280K lyric-only + 20K melody-only + 8K paired data).

### Key Designs

**1. Word-level Tuple Format:** The melody is decomposed into three attributes: pitch $p$, note duration $d$, and rest duration $r$. In the paired data, each note is aligned with its corresponding word in the `<p>,d,w` format. Durations are quantized using a unit of 1/16 beat:
$$d_k = \phi\left(\frac{\text{bpm}}{60}(\text{note-end}_k - \text{note-start}_k) \times 16\right)$$

**2. Scalar Pitch Initialization:** The central pitch `<66>` is first initialized with a Gaussian distribution, while the remaining pitches are defined as scalar multiples of this central embedding:
$$\text{emb}(\langle p \rangle) = c_p \cdot \text{emb}(\langle 66 \rangle)$$
where the scaling factor range is $[-\ln(e+17), \cdots, -\ln(e), \ln(e), \cdots, \ln(e+17)]$, explicitly encoding the numerical relationships among pitches.

**3. Progressive Structure-Aware Training:**
- **Stage 1 - Motif-level Melody Training:** Highly repetitive short note sequences are extracted as motif data to allow the model to learn basic repetition patterns.
- **Stage 2 - Independent Full-song Training:** The model is trained on lyric-only and melody-only datasets respectively to develop full-song comprehension.
- **Stage 3 - Phrase-level Paired Training:** Five phrase-specific tokens (intro/verse/chorus/bridge/outro) are introduced into the paired data, allowing the model to learn the structural sections of songs.

### Loss & Training

The standard autoregressive next-token prediction loss is employed to maximize the log-likelihood of tokens given their context.

---

## Experiments

### Main Results: Lyric-to-Melody and Melody-to-Lyric

| Method | PD(%)↑ | DD(%)↑ | MD↓ | Cosine Dist.↑ | ROUGE-2↑ | BS↑ |
|------|--------|--------|-----|---------------|----------|-----|
| SongMASS | 30.34 | 48.98 | 2.95 | 0.568 | 0.204 | 0.532 |
| TeleMelody | 46.81 | 51.77 | 2.60 | - | - | - |
| GPT-3.5 | 31.24 | 38.52 | 3.01 | 0.641 | 0.142 | 0.603 |
| GPT-4 | 36.43 | 42.94 | 2.87 | 0.654 | 0.158 | 0.610 |
| **SongComposer** | **50.75** | **57.71** | **2.20** | **0.697** | **0.234** | **0.657** |

### Ablation Study

| Ablation Dimension | Detailed Results |
|----------|----------|
| Pitch Initialization | Scalar > Interpolation > Average > Gaussian (MD: 2.33 vs 3.07/3.41/2.90) |
| Motif Repetition Threshold | Optimal RR-MD balance is achieved at a threshold of 10 (RR=2.03, MD=2.33) |
| Alignment Granularity | Word-level > Line-level > Song-level (MD: 2.12 vs 2.42 vs 3.71) |
| Phrase-level Tokens | Continuation performs better with phrase tokens (MD: 2.12 vs 2.58, BS: 0.662 vs 0.612) |

### Subjective Evaluation

| Method | Lyric-to-Melody HMY. | MLC. | Text-to-Song OVL. | REL. |
|------|----------------|------|-------------------|------|
| GPT-3.5 | 1.68 | 1.88 | 2.53 | 2.95 |
| GPT-4 | 2.82 | 2.79 | 2.43 | 3.27 |
| **SongComposer** | **3.82** | **3.76** | **3.41** | **3.88** |

### Key Findings

- **Effectiveness of a Unified Framework:** A single model comprehensively outperforms GPT-4 and specialized models across all four subtasks.
- **Critical Role of Scalar Initialization:** Explicitly encoding numerical relations among pitches achieves significantly better performance than random initialization, improving the Recall Rate from 1.44 to 2.03.
- **Necessity of Word-level Alignment:** Compared to line-level and song-level alignments, word-level alignment reduces Melody Distance from 3.71 to 2.12.
- **Effectiveness of Structure Awareness:** Motif training enhances the repetitiveness and structural sense of melodies, while phrase tokens improve subsection organization capability.

---

## Highlights & Insights

- The first to utilize an LLM to simultaneously generate both lyrics and melodies, achieving a unified song composition framework.
- Scalar pitch initialization delicately leverages numerical relations to encode pitch semantics, essentially serving as a form of music knowledge injection.
- The progressive training strategy introduces structural information incrementally from motif to phrase levels, aligning with human compositional cognition.
- The custom-built SongCompose dataset (containing 8K highly precise word-level aligned paired data) fills a major data vacancy in this domain.

## Limitations & Future Work

- The pitch range is restricted to C3-B5 (MIDI 48-83), which fails to cover composition requirements for wider vocal ranges.
- Being based on InternLM2-7B, the model scale limits the complexity and diversity of generated compositions.
- The paired dataset only contains 8,000 songs, which remains relatively limited in scale.
- Evaluation relies heavily on melody similarity metrics, lacking sufficient assessment of musical creativity and musicality.
- Only single-voice (vocal track) generation is supported, lacking the ability to generate polyphonic content such as accompaniment.

## Related Work

- **Song Subtasks:** SongMASS (Sheng et al., 2021) bidirectional Lyric↔Melody generation, TeleMelody (Ju et al., 2022) template-based melody generation.
- **LLM-based Music Generation:** ChatMusician (Yuan et al., 2024) symbolic music-only generation, whereas ours extends to joint lyric and melody generation.
- **Symbolic Representations:** REMI (Huang & Yang, 2020) beat-based musical representation.
- **Paired Datasets:** M4Singer (Zhang et al., 2022b) provides approximately 700 Chinese songs.

---

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |
| Overall Rating | 8/10 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] JoPA: Explaining Large Language Model's Generation via Joint Prompt Attribution](jopa_explaining_large_language_models_generation_via_joint_prompt_attribution.md)
- [\[ACL 2025\] Personalized Generation In Large Model Era: A Survey](personalized_generation_in_large_model_era_a_survey.md)
- [\[ACL 2025\] Representation Bending for Large Language Model Safety](repbend_representation_bending_safety.md)
- [\[ACL 2025\] An Empirical Study of Large Language Models for Automated Review Generation](an_empirical_study_of_large_language_models_for_automated_review_generation.md)
- [\[ACL 2025\] Binary Classifier Optimization for Large Language Model Alignment](bco_binary_classifier_alignment.md)

</div>

<!-- RELATED:END -->
