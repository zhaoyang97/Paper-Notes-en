---
title: >-
  [Paper Note] MECAT: A Multi-Experts Constructed Benchmark for Fine-Grained Audio Understanding Tasks
description: >-
  [ICML 2026][Audio & Speech][Fine-grained audio understanding] MECAT constructs 20k multi-perspective fine-grained audio captions and 100k open-ended QA using a "multi-expert model + CoT large model reasoning" pipeline…
tags:
  - "ICML 2026"
  - "Audio & Speech"
  - "Fine-grained audio understanding"
  - "multi-expert pipeline"
  - "open-ended QA"
  - "discriminative evaluation metric DATE"
  - "ACAV100M"
date: 2026-05-08
content_hash: ed63f51288b32096
---

# MECAT: A Multi-Experts Constructed Benchmark for Fine-Grained Audio Understanding Tasks

**Conference**: ICML 2026  
**arXiv**: [2507.23511](https://arxiv.org/abs/2507.23511)  
**Code**: https://github.com/xiaomi-research/mecat  
**Area**: Audio-Language Understanding / Benchmark Evaluation  
**Keywords**: Fine-grained audio understanding, multi-expert pipeline, open-ended QA, discriminative evaluation metric DATE, ACAV100M

## TL;DR
MECAT constructs 20k multi-perspective fine-grained audio captions and 100k open-ended QA using a "multi-expert model + CoT large model reasoning" pipeline, and proposes the DATE metric (harmonic mean of semantic similarity × cross-sample discriminability), enabling, for the first time, stable distinction between generic and detail-accurate audio model outputs.

## Background & Motivation

**Background**: Large Audio-Language Models (LALM) are shifting from closed-set classification/ASR to open-ended audio captioning and QA. Representative benchmarks include AudioCaps, Clotho (human-annotated captions), ClothoAQA, MMAU (QA); mainstream metrics are BLEU/CIDEr/SPICE (surface matching), FENSE (embedding similarity), and LLM-as-judge.

**Limitations of Prior Work**: (1) Data—Human captions only provide event-level coarse descriptions ("dog barking"); AutoACD / LPMusicCaps use LLMs for automatic annotation, but the input metadata is itself coarse, so granularity is not addressed; QA is mostly yes/no or multiple choice, unable to evaluate open-ended generation. (2) Data sources are highly homogeneous—many benchmarks are derived from AudioSet, leading to severe "one audio, multiple uses" and overestimated model generalization. (3) Metrics—surface matching penalizes paraphrasing; embedding similarity still cannot distinguish between "dog barking, people talking" (generic) and "an excited dog barks briefly in the park while people chat nearby" (detailed); LLM-as-judge is discriminative but costly, slow, and prompt-sensitive.

**Key Challenge**: To evaluate whether LALMs truly understand audio, (a) multi-perspective and fine-grained reference annotations are needed to allow models to express detail differences; (b) a scalable metric that rewards "detail accuracy" and penalizes "generic" outputs is required. Both are currently lacking.

**Goal**: (i) Construct a benchmark with novel data sources, comprehensive domain coverage, and fine-grained captions for audio captioning + QA; (ii) Design an open-ended generation evaluation metric that does not rely on LLM judges but is more discriminative than FENSE; (iii) Systematically evaluate current SOTA LALMs to reveal the true bottlenecks in fine-grained perception.

**Key Insight**: The authors observe that since single LLM annotation tends to produce coarse descriptions, it is preferable to first use a suite of domain expert models (speech/music/sound events/acoustic attributes) to extract structured analyses, then let an LLM use CoT to synthesize all expert evidence into multi-perspective descriptions; for evaluation, build on Sentence-BERT embeddings with "TF-IDF weighting" and "cross-sample ranking scores," turning one-sided similarity into a "relative matching" discrimination problem.

**Core Idea**: Use "multi-expert pipeline generation + three major multi-perspective caption types (systemic/content/unrelated)" for data, and "TF-IDF weighted embedding × cross-sample discriminability" to construct the DATE metric, shifting evaluation from "average similarity" to "can this sample be distinguished from others."

## Method

### Overall Architecture
MECAT consists of two parts: (A) Data construction pipeline—sample 20k Creative Commons audio clips (≤10s) from ACAV100M, classify into eight domains (silence/speech/music/sound four pure + four mixed), feed into corresponding expert groups (speech: ASR + LID + diarization + gender/age/emotion/accent; music: global description + attributes + source separation; sound: CED-Base labels; acoustics: RMS, DNSMOS/NISQA2, reverberation time), then use DeepSeek-R1 with CoT to synthesize all structured outputs into 18 reference captions and 5 QA per clip; finally, GLAP cross-modal scoring + rule-based filtering for quality control. (B) Evaluation—MECAT-Caption (6 subclass weighted caption scores) + MECAT-QA (6 cognitive skill QA types) + DATE metric.

### Key Designs

1. **Multi-Expert + CoT Synthesis Annotation Pipeline**:

    - **Function**: Converts the coarse annotation problem of single LLMs into "multi-source structured evidence → LLM reasoning → multi-perspective captions + QA."
    - **Mechanism**: First, use CED-Base to predict AudioSet labels in 2-second windows for domain classification; speech domain uses ASR + speaker separation + attribute recognition; music domain uses Audio Flamingo 2 for global description + attributes + vocal/instrument separation (vocal is routed back to speech pipeline); sound domain uses AudioSet labels directly; acoustic attribute pipeline extracts RMS / DNSMOS / NISQA2 / reverberation. DeepSeek-R1 receives all expert outputs + metadata, uses a rule-based prompt for CoT reasoning, and outputs 6 subclass captions (systemic long/short, speech, music, sound, acoustic) + 5 QA types (DP/SC/QAS/ER/IJ/AC), each with confidence scores. Quality control uses GLAP to compute audio-caption cosine similarity, requiring the correct match to exceed the average of 6 random captions by a threshold of 6; further filters by confidence threshold, domain consistency, and hallucination removal.
    - **Design Motivation**: A single LLM viewing raw audio tends to produce generic sentences like "a dog is barking, someone is talking"; but when the LLM sees ASR transcripts, emotion tags, tempo, reverberation time, etc., CoT reasoning naturally incorporates details. The 6-subclass multi-perspective design directly corresponds to human auditory perception's "overall scene / content-specific / physical attributes" layers.

2. **DATE Metric: Single-Sample Semantic Similarity × Cross-Sample Discriminability**:

    - **Function**: Rewards detail-accurate descriptions and penalizes generic ones without invoking LLM judges.
    - **Mechanism**: First, compute TF-IDF weighted Sentence-BERT embeddings—sentence vector $\mathbf{v}_T=\sum_t (\text{TF}_{emb}(t,T)\cdot\text{IDF}_{emb}(t))\cdot E(t)$, giving higher weight to rare/discriminative words. Single-sample similarity $S_{sim,i}=\cos(\mathbf{v}_{cand},\mathbf{v}_{ref})$. Then, construct a cross-sample similarity matrix $\mathcal{M}$; for sample $i$, convert the diagonal $M_{i,i}$'s rank $r_i$ among all candidates in row $i$ to discriminability $S_{dis,i}=1-r_i/N$—whether this caption matches its audio better than others. Finally, $\text{DATE}_i=\frac{2\cdot S_{sim,i}\cdot S_{dis,i}}{S_{sim,i}+S_{dis,i}}\in[0,1]$ (harmonic mean).
    - **Design Motivation**: With only similarity, "a dog is barking" scores high for all dog audios; introducing cross-sample ranking, generic descriptions, which "vaguely fit" many audios, receive low discriminative scores and are suppressed. The harmonic mean enforces that both must be high for a high score.

3. **Task Definition: Weighted Evaluation of 6 Caption Subclasses + 6 QA Types**:

    - **Function**: Decomposes "fine-grained" into independently measurable, aggregatable sub-tasks.
    - **Mechanism**: Caption side $\text{Score}_{Cap}=0.4\cdot S_{Systemic}+0.4\cdot S_{Content\text{-}Specific}+0.2\cdot S_{Content\text{-}Unrelated}$, where $S_{Systemic}=0.8\cdot S_{Long}+0.2\cdot S_{Short}$, $S_{Content\text{-}Specific}=0.6\cdot S_{Speech}+0.3\cdot S_{Music}+0.1\cdot S_{Sound}$ (weights roughly reflect ACAV100M content distribution; sensitivity analysis shows model ranking is stable, Kendall's $\tau=0.92$). QA side: 6 types—Perception (DP), Analysis (SC, QAS), Reasoning (ER, IJ, AC)—averaged equally $\text{Score}_{QA}=(S_{DP}+S_{SC}+S_{QAS}+S_{ER}+S_{IJ}+S_{AC})/6$. Each content subclass is also evaluated separately for "pure" and "mixed" domains (e.g., speech in S00 vs SM0/SMA) to assess robustness in complex acoustic scenes.
    - **Design Motivation**: A single overall score can be dominated by a strong aspect; subclass decomposition directly reveals model differences in long vs short descriptions, pure vs mixed audio, perception vs reasoning, providing precise diagnostic signals for model improvement.

### Loss & Training
This is not a training paper, so no loss is defined. For evaluation, all LALMs generate captions/QA answers via Huggingface APIs or official inference scripts, and scores are computed using DATE.

## Key Experimental Results

### Main Results
Evaluation of multiple SOTA LALMs on MECAT-Caption using DATE (%) (excerpt from Table 2):

| Model | Systemic Long | Speech (Pure) | Music (Pure) | Sound (Pure) | $\text{Score}_{Cap}$ |
|---|---|---|---|---|---|
| Caption-Only baseline | Low | Low | Low | Low | Low |
| Mainstream LALMs (e.g., Audio Flamingo / Qwen-Audio, etc.) | See original table | See original table | See original table | See original table | See original table |

(The original Table 2 lists scores for over ten models, including caption-only, general LALMs, MiMo-Audio, etc., across 12 fine-grained dimensions. Overall conclusion: all models perform significantly worse on systemic long, mixed domains, and sound-pure compared to short and speech-pure, revealing much larger fine-grained gaps than traditional benchmarks.)

### Ablation Study (Metrics / Weights)

| Configuration | Observation |
|---|---|
| Similarity only (FENSE) | Generic vs detailed outputs score almost the same; model ranking is chaotic |
| Cross-sample discrimination only | Short sentences are favored, detailed descriptions are penalized |
| **DATE (harmonic mean)** | Model ranking highly consistent with LLM-as-judge; CDF curves show optimal discriminability for both caption and QA |
| Caption weights $(0.4,0.4,0.2)$ changed to $(0.5,0.3,0.2)$ etc. | Kendall's $\tau=0.92$, model ranking stable |
| Content-Specific internal weights 0.6/0.3/0.1 | Adjusted according to ACAV100M content distribution, ranking remains stable |

### Key Findings
- Existing LALMs score lowest on systemic long captions, indicating they can recognize sounds but cannot organize multi-event context-rich long descriptions; this is the most easily exposed weakness in fine-grained evaluation.
- Mixed domains (e.g., SMA: speech + music + sound) score much lower than pure domains, indicating LALMs' detail capture in "multi-source mixed acoustic scenes" is far from mature.
- The CDF distance between DATE and LLM-as-judge is significantly greater than that between FENSE and LLM-as-judge, demonstrating that DATE approaches judge-level discriminability without LLM cost.

## Highlights & Insights
- The "multi-expert + CoT synthesis" annotation pipeline is a highly transferable design pattern: in any domain, using a set of small, specialized models to extract structured attributes before LLM synthesis is more reliable than letting LLMs describe from raw modality; this approach is also applicable to video, medical imaging, and other evaluation constructions.
- DATE's "single-sample similarity × cross-sample discriminability" is a general new paradigm for evaluating open-ended generation—turning "is this good" from an absolute score into "is this more matching than others," naturally suppressing generic template answers.
- The explicit design in the 6-subclass captions requiring models to state "absence" when a domain is not present (e.g., speech caption for pure music should answer "no one is speaking") is a clever detail, directly encoding the cost of hallucination into the reference annotation.

## Limitations & Future Work
- Data source is still single ACAV100M; although the source is changed, ecological diversity remains limited. Multi-source mixed datasets (YouTube/Podcast/Movie) would be more robust.
- Audio clips are limited to under 10s, unable to evaluate understanding of real long-form audio (podcasts, lectures).
- DATE relies on Sentence-BERT embeddings for semantic similarity; for Chinese/low-resource languages and specialized terminology, embedding replacement may be needed.
- Evaluation involves potential conflicts of interest (some authors from Xiaomi, with MiMo-Audio included in evaluation); external independent replication is important.

## Related Work & Insights
- **vs AudioCaps / Clotho**: Event-level coarse vs multi-perspective fine-grained; reference captions expanded from 1 to 18 per clip, greatly increasing lexical richness.
- **vs LPMusicCaps / AutoACD (LLM auto-annotation)**: LLMs writing captions from coarse metadata still produce generic descriptions; MECAT uses expert pipelines to feed structured evidence, enabling CoT to write real details.
- **vs MMAU (multiple-choice QA)**: MMAU is closed multiple-choice; this work uses open-ended generation + DATE evaluation, measuring "generation ability" rather than "guessing ability."
- **vs FENSE**: FENSE is an embedding similarity metric for audio captions, but experiments show it lacks discriminability for generic vs detailed outputs; DATE compensates with a cross-sample discrimination term.

## Rating
- Novelty: ⭐⭐⭐⭐ Both the multi-expert pipeline and DATE metric are first systematically combined solutions for open-ended audio evaluation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluates over ten SOTA LALMs, conducts weight sensitivity analysis, provides CDF discriminability visualization, but model selection is mostly "general LALMs," with narrow-domain models (music or medical audio) not covered.
- Writing Quality: ⭐⭐⭐⭐ Task definition is clear, formulas and flowcharts (Fig 1) are easy to understand; DATE's design motivation is convincingly explained.
- Value: ⭐⭐⭐⭐⭐ Provides a new "data + metric" dual standard for open-ended audio understanding evaluation; the DATE concept can be directly transferred to other multimodal open-generation evaluations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Towards Fine-Grained and Multi-Granular Contrastive Language-Speech Pre-training](../../ACL2026/audio_speech/towards_fine-grained_and_multi-granular_contrastive_language-speech_pre-training.md)
- [\[ICLR 2026\] MMSU: A Massive Multi-task Spoken Language Understanding and Reasoning Benchmark](../../ICLR2026/audio_speech/mmsu_a_massive_multi-task_spoken_language_understanding_and_reasoning_benchmark.md)
- [\[ICML 2026\] MedMosaic: A Challenging Large Scale Benchmark of Diverse Medical Audio](medmosaic_a_challenging_large_scale_benchmark_of_diverse_medical_audio.md)
- [\[ACL 2026\] MSU-Bench: Musical Score Understanding Benchmark](../../ACL2026/audio_speech/musical_score_understanding_benchmark_evaluating_large_language_models39_compreh.md)
- [\[AAAI 2026\] GOMPSNR: Reflourish the Signal-to-Noise Ratio Metric for Audio Generation Tasks](../../AAAI2026/audio_speech/gompsnr_reflourish_the_signal-to-noise_ratio_metric_for_audio_generation_tasks.md)

</div>

<!-- RELATED:END -->
