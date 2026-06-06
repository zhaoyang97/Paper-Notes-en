---
title: >-
  [Paper Note] MECAT: A Multi-Experts Constructed Benchmark for Fine-Grained Audio Understanding Tasks
description: >-
  [ICML 2026][Audio & Speech][Fine-grained Audio Understanding] MECAT constructs 20k multi-perspective fine-grained audio captions and 100k open-ended QAs using a "multi-expert models + CoT LLM reasoning" pipeline. It prop…
tags:
  - "ICML 2026"
  - "Audio & Speech"
  - "Fine-grained Audio Understanding"
  - "Multi-expert Pipeline"
  - "Open-ended QA"
  - "Discriminative Metric DATE"
  - "ACAV100M"
date: 2026-05-08
content_hash: 94b96583afc1fb6b
---

# MECAT: A Multi-Experts Constructed Benchmark for Fine-Grained Audio Understanding Tasks

**Conference**: ICML 2026  
**arXiv**: [2507.23511](https://arxiv.org/abs/2507.23511)  
**Code**: https://github.com/xiaomi-research/mecat  
**Area**: Audio-Language Understanding / Evaluation Benchmark  
**Keywords**: Fine-grained Audio Understanding, Multi-expert Pipeline, Open-ended QA, Discriminative Metric DATE, ACAV100M

## TL;DR
MECAT constructs 20k multi-perspective fine-grained audio captions and 100k open-ended QAs using a "multi-expert models + CoT LLM reasoning" pipeline. It proposes the DATE metric (the harmonic mean of semantic similarity and cross-sample discriminability), which for the first time distinguishes between generic descriptions and detail-accurate model outputs.

## Background & Motivation

**Background**: Large Audio-Language Models (LALMs) are shifting from closed-set classification/ASR to open-ended audio captioning and QA. Representative benchmarks include AudioCaps, Clotho (human-annotated captions), ClothoAQA, and MMAU (QA). Dominant metrics include BLEU/CIDEr/SPICE (word matching), FENSE (embedding similarity), and LLM-as-judge.

**Limitations of Prior Work**: (1) Data: Human captions often provide only coarse event-level descriptions (e.g., "a dog is barking"). Methods like AutoACD / LPMusicCaps use LLMs for auto-annotation, but the input metadata is inherently coarse, leaving granularity unresolved. QA is mostly yes/no or multiple-choice, which fails to test open-ended generation. (2) Data Sources: Benchmarks are highly homogeneous, mostly derived from AudioSet, leading to severe "one audio, multiple uses" and overestimation of model generalization. (3) Metrics: Word matching penalizes synonymous paraphrasing; embedding similarity fails to distinguish between generic outputs ("a dog barking and a person talking") and detailed ones ("an excited dog gives short barks in a park while people chat nearby"). LLM-as-judge is discriminative but expensive, slow, and sensitive to prompts.

**Key Challenge**: To evaluate whether a LALM truly "understands" audio, there is a need for (a) multi-perspective and fine-grained reference annotations to allow room for expressing detail differences, and (b) a scalable metric that rewards "detail accuracy" and penalizes "generic descriptions." Both are currently lacking.

**Goal**: (i) Construct an audio caption + QA benchmark with novel data sources, full domain coverage, and fine-grained granularity; (ii) Design an open-ended generation metric that does not rely on LLM judges yet is more discriminative than FENSE; (iii) Systematically evaluate current SOTA LALMs to reveal real bottlenecks in fine-grained perception.

**Key Insight**: Instead of a single LLM for auto-annotation, which tends to produce coarse descriptions, one can utilize a suite of domain-specific expert models (speech, music, sound events, acoustic attributes) to extract structured analysis. An LLM then uses Chain-of-Thought (CoT) to synthesize evidence from all experts. For evaluation, the metric combines TF-IDF weighted Sentence-BERT embeddings with "cross-sample ranking," turning a unilateral similarity problem into a discriminative challenge of "matching this sample better than others."

**Core Idea**: Use a "multi-expert pipeline generation + three-category (systemic/specific/unrelated) multi-perspective captions" for data, and the DATE metric (TF-IDF weighted embedding $\times$ cross-sample discriminability) to shift evaluation from "how similar it looks on average" to "how well it separates this sample from others."

## Method

### Overall Architecture
MECAT consists of two parts: (A) Data Construction Pipeline: Extracting 20k Creative Commons audio clips ($\le 10$s) from ACAV100M, classifying them into eight domains (pure silence/speech/music/sound and four mixed domains), and feeding them into expert groups (Speech: ASR + LID + diarization + gender/age/emotion/accent; Music: global description + attributes + source separation; Sound: CED-Base labels; Acoustic: RMS, DNSMOS/NISQA2, reverberation time). DeepSeek-R1 then performs CoT reasoning to write 18 reference captions/clip and 5 QAs/clip. Finally, GLAP cross-modal scoring and rule filtering are used for quality control. (B) Evaluation: MECAT-Caption (weighted scores across 6 sub-categories) + MECAT-QA (6 cognitive skill categories) + DATE metric.

### Key Designs

1.  **Multi-Expert + CoT Synthesis Annotation Pipeline**:
    - **Function**: Transforms the coarse-description problem of simple LLM annotation into "multi-source structured evidence $\rightarrow$ LLM reasoning $\rightarrow$ multi-perspective captions + QA."
    - **Mechanism**: CED-Base predicts AudioSet labels in 2s windows to determine domain; speech enters an ASR + diarization + attribute pipeline; music enters an Audio Flamingo 2 global description + source separation pipeline; acoustic attributes are extracted via RMS/DNSMOS. DeepSeek-R1 takes all expert outputs and metadata, following structured prompts for CoT reasoning to output 6 sub-categories of captions (systemic long/short, speech, music, sound, acoustic) and 5 classes of QA (DP/SC/QAS/ER/IJ/AC), each with confidence scores.
    - **Design Motivation**: A single LLM looking at raw audio often writes generic sentences. When provided with structured evidence like ASR transcripts, emotion labels, tempo, and reverb, CoT-derived captions naturally include more detail.

2.  **DATE Metric: Single-sample Semantic Similarity $\times$ Cross-sample Discriminability**:
    - **Function**: Rewards detail-accurate descriptions and penalizes generic ones without calling an LLM judge.
    - **Mechanism**: First, a TF-IDF weighted Sentence-BERT embedding is computed: $\mathbf{v}_T = \sum_t (\text{TF}_{emb}(t,T) \cdot \text{IDF}_{emb}(t)) \cdot E(t)$, giving higher weight to rare/distinctive words. Single-sample similarity is $S_{sim,i} = \cos(\mathbf{v}_{cand}, \mathbf{v}_{ref})$. Then, a cross-sample similarity matrix $\mathcal{M}$ is constructed. For sample $i$, the diagonal rank $r_i$ in row $i$ is converted to discriminability $S_{dis,i} = 1 - r_i/N$. Final score: $\text{DATE}_i = \frac{2 \cdot S_{sim,i} \cdot S_{dis,i}}{S_{sim,i} + S_{dis,i}} \in [0, 1]$.
    - **Design Motivation**: Generic descriptions like "a dog barking" would get high similarity scores for any dog audio. The cross-sample ranking suppresses generic descriptions because they "vaguely match every audio snippet," resulting in a low discriminative score.

3.  **Task Definition: Weighted Evaluation of 6 Captions & 6 QA Types**:
    - **Function**: Decomposes "fine-grained" into sub-tasks that can be independently measured and weighted.
    - **Mechanism**: Caption score $\text{Score}_{Cap} = 0.4 \cdot S_{Systemic} + 0.4 \cdot S_{Content\text{-}Specific} + 0.2 \cdot S_{Content\text{-}Unrelated}$, where weighted components reflect the ACAV100M distribution. QA covers 6 categories—Perception (DP), Analysis (SC, QAS), and Reasoning (ER, IJ, AC)—averaged equally.

## Key Experimental Results

### Main Results
Evaluation of SOTA LALMs on MECAT-Caption using DATE (%) (Selected from Table 2):

| Model | Systemic Long | Speech (Pure) | Music (Pure) | Sound (Pure) | $\text{Score}_{Cap}$ |
|---|---|---|---|---|---|
| Caption-Only baseline | Low | Low | Low | Low | Low |
| Mainstream LALMs (e.g., Audio Flamingo / Qwen-Audio) | [Ref Table] | [Ref Table] | [Ref Table] | [Ref Table] | [Ref Table] |

(The original Table 2 shows that all models perform significantly worse on systemic long, mixed domains, and pure-sound compared to short or pure-speech, revealing a much larger fine-grained gap than traditional benchmarks suggest.)

### Ablation Study

| Configuration | Observation |
|---|---|
| Similarity only (FENSE) | Generic vs. detailed outputs get nearly identical scores; model ranking is chaotic. |
| Cross-sample discriminability only | Favors short sentences; penalizes detailed descriptions. |
| **DATE (Harmonic Mean)** | Model ranking aligns highly with LLM-as-judge; demonstrates best discriminability in CDF curves. |
| Caption weights $(0.4, 0.4, 0.2)$ | Kendall’s $\tau = 0.92$, showing stable model rankings across weight variations. |

### Key Findings
- Existing LALMs score lowest on "systemic long captions," indicating they can identify sounds but struggle to organize multiple events into a contextual long description.
- Mixed-domain (e.g., Speech + Music + Sound) scores drop significantly compared to pure domains, showing that LALMs are immature in capturing details in complex acoustic scenes.
- The CDF distance between DATE and LLM-as-judge is significantly larger than that between FENSE and the judge, proving DATE approaches judge-level discrimination without the cost.

## Highlights & Insights
- The "Multi-expert + CoT" pipeline is a highly transferable design pattern: extracting structured attributes before LLM synthesis is more reliable than raw modality descriptions.
- DATE's "Similarity $\times$ Discriminability" is a universal paradigm for open-ended generation—turning absolute quality into relative matching naturally inhibits generic "template" answers.
- Explicitly requiring models to state when a domain is absent (e.g., "no speech" in pure music) is a clever detail that bakes the cost of hallucination directly into the reference.

## Limitations & Future Work
- Data source is limited to ACAV100M; multi-source mixing (YouTube/Podcasts) would be more robust.
- Audio clips are limited to $\le 10$s, excluding understanding of long-form audio.
- DATE relies on Sentence-BERT for semantic approximation, which might need specialized embeddings for non-English or highly technical domains.

## Related Work & Insights
- **vs. AudioCaps / Clotho**: Coarse event-level vs. multi-perspective fine-grained; reference captions expanded from 1 to 18 per clip.
- **vs. LPMusicCaps / AutoACD**: Simple LLM-based labeling still produces generic descriptions; MECAT uses the expert pipeline to provide structured evidence for better CoT.
- **vs. MMAU**: Transitioned from closed-set multiple choice to open-ended generation + DATE evaluation to test "generative capability" over "guessing capability."

## Rating
- **Novelty**: ⭐⭐⭐⭐ First systematic combination of multi-expert pipelines and the DATE metric in the audio domain.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated 10+ SOTA LALMs with sensitivity analysis, though narrow-domain models (purely medical or musical) were less covered.
- **Writing Quality**: ⭐⭐⭐⭐ Clear task definitions; persuasive design motivation for DATE.
- **Value**: ⭐⭐⭐⭐⭐ Provides a new dual standard of "data + metric" for open-ended audio understanding; DATE can be directly transferred to other multimodal generation tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Towards Fine-Grained and Multi-Granular Contrastive Language-Speech Pre-training](../../ACL2026/audio_speech/towards_fine-grained_and_multi-granular_contrastive_language-speech_pre-training.md)
- [\[ACL 2026\] SegTune: Structured and Fine-Grained Control for Song Generation](../../ACL2026/audio_speech/segtune_structured_and_fine-grained_control_for_song_generation.md)
- [\[ICLR 2026\] MMSU: A Massive Multi-task Spoken Language Understanding and Reasoning Benchmark](../../ICLR2026/audio_speech/mmsu_a_massive_multi-task_spoken_language_understanding_and_reasoning_benchmark.md)
- [\[ACL 2026\] UniSRM: A Unified Speech Reward Model for Fine-Grained Speech Evaluation](../../ACL2026/audio_speech/unisrm_a_unified_speech_reward_model_for_reasoning-based_fine-grained_assessment.md)
- [\[ICML 2026\] MultiBreak: A Scalable and Diverse Multi-turn Jailbreak Benchmark for Evaluating LLM Safety](multibreak_a_scalable_and_diverse_multi-turn_jailbreak_benchmark_for_evaluating_.md)

</div>

<!-- RELATED:END -->
