---
title: >-
  [Paper Note] Comprehensive Benchmarking of Long-Form Speech Generation in Diverse Scenarios
description: >-
  [ACL2026 Findings][Audio & Speech][Long-form speech generation] This paper proposes SwanBench-Speech, which systematically evaluates long-form speech generation using 1,101 samples across 17 real-world downstream scenarios and 7 automatic evaluation dimensions. The study concludes that while current models approach usability in content accuracy, they still significantly lag behind real recordings in reverb consistency, long-range prosody, and expressive hierarchy.
tags:
  - "ACL2026 Findings"
  - "Audio & Speech"
  - "Long-form speech generation"
  - "Speech evaluation benchmark"
  - "Expressiveness evaluation"
  - "Prosodic consistency"
  - "Multi-scenario TTS"
date: 2026-05-08
content_hash: 2da7fc3099ff8dad
---

# Comprehensive Benchmarking of Long-Form Speech Generation in Diverse Scenarios

**Conference**: ACL2026 Findings  
**arXiv**: [2605.28618](https://arxiv.org/abs/2605.28618)  
**Code**: https://swanaigc.github.io/#bench  
**Area**: Audio & Speech / Long-form Speech Generation Evaluation  
**Keywords**: Long-form speech generation, Speech evaluation benchmark, Expressiveness evaluation, Prosodic consistency, Multi-scenario TTS  

## TL;DR
This paper proposes SwanBench-Speech, which systematically evaluates long-form speech generation using 1,101 samples across 17 real-world downstream scenarios and 7 automatic evaluation dimensions. The study concludes that while current models approach usability in content accuracy, they still significantly lag behind real recordings in reverb consistency, long-range prosody, and expressive hierarchy.

## Background & Motivation
**Background**: Speech generation is evolving from sentence-level TTS to paragraph- and minute-level generation. Typical applications include podcasts, audiobooks, course explanations, news broadcasts, interviews, and multi-person dialogues. Existing models can generate high-fidelity short speech, but long-form scenarios require models to maintain timbre, acoustic environment, semantics, rhythm, and emotional dynamics simultaneously.

**Limitations of Prior Work**: Existing test sets often cover only a few domains or single-speaker scenarios, and metrics remain biased towards short-sentence quality, such as WER/CER, MOS, or single-sentence similarity. This leads to two problems: first, evaluation scenarios significantly diverge from real applications; second, model failures common in long-form text—such as timbre drift, reverb drift, prosodic collapse, and flat expression—are difficult to quantify.

**Key Challenge**: Long-form speech quality is not a single score but is jointly determined by acoustic stability, semantic intelligibility, and expressive dynamics. Traditional metrics measure clarity but fail to address whether a segment sounds like a continuous natural narrative, whether the acoustic field of a multi-person dialogue is unified, or whether emotions evolve with the progression of paragraphs.

**Goal**: The authors aim to construct a standardized benchmark for long-form text: covering various speaking styles in real applications while decomposing "long-form quality" into interpretable metrics that are automatically evaluable and correlated with human perception.

**Key Insight**: Instead of proposing a new TTS model, the paper contributes an evaluation system. It organizes data and metrics based on three types of challenges: Acoustics (timbre, reverb, fidelity); Semantics (content accuracy, prosodic naturalness); and Expressiveness (emotional richness, paragraph-level expressive hierarchy).

**Core Idea**: Replacing single MOS-style evaluations with "multi-scenario data + decoupled metrics + human correlation validation" to expose the genuine weaknesses of long-form speech generation models.

## Method
The methodology of SwanBench-Speech is a comprehensive evaluation protocol rather than a model training algorithm. It constructs a long-form speech test set covering broad scenarios, designs a set of automatic metrics to evaluate different models, and validates the consistency between key automatic metrics and human auditory perception through preference experiments.

### Overall Architecture
The input consists of long-form text or dialogue scripts for specific scenarios and the TTS/dialogue speech generation models to be evaluated. After the model generates long-form speech, SwanBench-Speech calculates metrics along three axes: the Acoustic axis measures timbre consistency, reverb consistency, and no-reference audio quality; the Semantic axis measures content restoration and prosodic coherence; the Expressive axis measures local emotional richness and overall expressive hierarchy. The output is a set of diagnostic multi-dimensional results rather than a black-box total score.

Data construction involves three sources. The first is online text corpora, such as audiobooks, dramas, and news scripts; the second is online audio media, which is manually verified after noise reduction, DNS-MOS quality filtering, speaker diarization, and SenseVoice transcription; the third comprises supplementary test samples generated by GPT-5 to expand topic and scenario diversity. The final data undergoes semantic deduplication, content quality filtering, privacy/ethical risk detection, and manual review, resulting in 1,101 samples.

### Key Designs

**1. Three-Axis, Seventeen-Scenario Long-Form Test Set: Decomposing "Long-Form Quality" into 17 Scenarios across 3 Challenges**

Short-sentence TTS test sets only reflect clarity but fail to expose accumulated timbre drift, acoustic field instability, and expression collapse in minute-level generation. SwanBench-Speech decomposes long-form speech quality into three challenge axes—Acoustics (timbre, reverb, fidelity), Semantics (content accuracy, prosody naturalness), and Expressiveness (emotional richness, expressive hierarchy)—mapping them to 17 real downstream scenarios: acoustic-related customer service, podcasts, chit-chat, debates, audiobooks, and interviews; semantic-dense courses, popular science, demos, seminars, and news; and expressive-rich drama, talk shows, hosting, speeches, live streaming, and sports commentary. By decomposing by scenario, model performance drops change from obscure total scores to locatable diagnostic results.

**2. Seven Diagnostic Automatic Metrics: Decomposing MOS into Seven Trackable Dimensions over Time**

Long-form failures are often not due to "unintelligibility" but due to identity, acoustic field, and expressive states drifting over minutes. The paper decomposes quality into seven algorithmically defined metrics: Timbre consistency is measured by pairwise cosine similarity of sliding-window speaker embeddings; Reverb consistency is measured by the standard deviation of SRMR sequences (lower is more stable); Content accuracy uses WER/CER from ASR transcriptions; Prosodic coherence is scored by SpeechJudge; and Expressive richness and hierarchy are evaluated by LALM/Gemini3-Pro using specific prompts. Once metrics are decoupled, researchers can determine whether to modify data, architecture, or expression modeling.

**3. Human Perception Alignment Verification: Proving Automatic Scores Predict Human Preferences via SRCC**

Expressiveness metrics risk becoming isolated "model-judging-model" scores, so the authors performed manual calibration. For the prosody experiment, 50 pairs of text-identical audio generated by different models were sampled, and 10 evaluators provided relative preferences from -2 to 2; for the expressiveness experiment, 200 audio segments were sampled and scored by 10 evaluators using the same prompts, comparing correlations between multiple MOS networks, LALM evaluators, and human MOS. By aligning automatic scores with human judgment using SRCC, the expressiveness metrics become credible tools.

### Loss & Training
This paper does not train a new speech generation model and thus lacks a traditional loss function. Its "training strategy" is reflected in the evaluation protocol: using sliding-window analysis, ASR restoration, no-reference quality models, SpeechJudge, and Gemini3-Pro evaluators for generated audio; and using manual SRCC to calibrate evaluator reliability. For evaluated models, the paper covers single-speaker long-form and dialogue generation tasks, comparing open-source and closed-source systems.

## Key Experimental Results

### Main Results

| Evaluated Object | Timbre↑ | Reverb↓ | Fidelity↑ | CER/WER↓ | Prosody↑ | Richness↑ | Hierarchy↑ |
|---------|---------|---------|-----------|----------|----------|-----------|------------|
| Avg. Open-source Single-Speaker | 0.93 | 1.95 | 3.63 | 0.073 / 0.164 | 3.43 | 3.03 | 2.67 |
| Avg. Closed-source Single-Speaker | 0.93 | 3.55 | 3.55 | 0.065 / 0.138 | 3.79 | 3.42 | 3.01 |
| Real Single-Speaker Recording | 0.96 | 1.91 | 3.62 | 0.070 / 0.074 | 4.04 | 4.35 | 3.94 |
| Avg. Open-source Dialogue | 0.92 | 3.45 | 3.02 | 0.129 / 0.137 | 3.41 | 3.07 | 3.06 |
| Avg. Closed-source Dialogue | 0.92 | 3.36 | 3.17 | 0.095 / 0.103 | 3.83 | 3.51 | 3.76 |
| Real Dialogue Recording | 0.95 | 2.73 | 2.94 | 0.050 / 0.137 | 3.95 | 4.42 | 4.17 |

Closed-source systems are generally stronger than open-source systems in prosody and expression but still significantly lag behind real recordings. In single-speaker scenarios, real recordings have a Richness of 4.35 and Hierarchy of 3.94, while closed-source averages are only 3.42 and 3.01. In dialogue scenarios, the closed-source average expressive hierarchy reaches 3.76, still lower than the 4.17 of real dialogues; reverb consistency also shows a clear gap (3.36 for closed-source vs. 2.73 for real).

### Ablation Study

| Validation Item | Setting | Key Result | Description |
|--------|------|----------|------|
| Prosody Metric Alignment | 50 audio pairs, 10 human evaluators | SRCC = 0.82 | SpeechJudge prosody scores correlate highly with human preferences |
| Expressive Richness Alignment | 200 audio segments, 10 human evaluators | SRCC = 0.71 | Gemini3-Pro has the highest correlation with human MOS in richness |
| Expressive Hierarchy Alignment | 200 audio segments, 10 human evaluators | SRCC = 0.62 | Paragraph-level dynamics are harder to evaluate automatically than local emotions |
| Generation Length Analysis | MegaTTS3, F5TTS, CosyVoice2, etc. | Metrics degrade after 100 words | Long-range dependencies affect accuracy, prosody, and expression simultaneously |

### Key Findings
- Content accuracy is no longer the sole bottleneck. Many models' CER/WER approach real speech, but Prosody, Richness, and Hierarchy remain significantly lower.
- Expressive scenarios are the most prone to performance drops. Theoretically, drama and sports commentary should have higher expressive ceilings, but current models degrade in these scenarios, indicating insufficient training data and modeling.
- AR vs. NAR architectures present a clear trade-off. NAR models are more stable and efficient but prone to over-smoothing; AR models are more expressive but susceptible to error propagation in long sequences.
- Data quality is more critical than mere scaling. Short-fragment training data introduces short-text bias, while unstable acoustic fields in in-the-wild data induce acoustic drift. Large-scale averaging may also weaken dynamic expression.

## Highlights & Insights
- A highlight of this paper is the granular decomposition of long-form TTS failure modes. It asks where a model loses—timbre, reverb, prosody, or expressive hierarchy—rather than just "which model is best."
- SwanBench-Speech is practical for real-world applications. The 17 scenarios correspond to speech patterns users actually hear, providing better guidance for model iteration than single reading tests.
- Human correlation validation makes the evaluation protocol credible. Specifically, expressiveness metrics would otherwise remain uninterpretable model scores.
- A key insight is that "long-form capability" cannot be solved by a larger context window alone. Speech models require training data that is temporally continuous, acoustically stable, and structurally expressive at the paragraph level.

## Limitations & Future Work
- Language coverage is limited. SwanBench-Speech primarily covers Chinese and English; low-resource languages and accents are not yet fully included.
- Semantic understanding metrics remain preliminary. The paper admits current metrics favor acoustic consistency over emotional/stylistic transitions driven by deep semantics.
- Reference timbres lack diversity. Prompt speech was mainly from ~20 open-source speakers, potentially introducing timbre bias.
- Expressive evaluation relies partly on the closed-source Gemini3-Pro, affecting reproducibility due to API updates; future work could involve distilling open-source evaluators.
- Future directions include broader language coverage, stronger open expressive evaluators, more real long-context recordings, and from-sentence-to-paragraph curriculum training.

## Related Work & Insights
- **vs. SeedTTS-Eval / EmergentTTS-Eval**: These cover some short speech or limited scenarios. SwanBench-Speech's advantage lies in its more complete 17 scenarios and long-form dimensions, though at the cost of a more complex protocol.
- **vs. MultiDialog / LibriSpeech-long**: The latter provide materials, but their metrics may not fully characterize expressive hierarchy. This paper integrates data coverage with multi-dimensional automatic metrics for better diagnostics.
- **vs. MOS / WER Single-Metric Eval**: MOS and WER are easy to use but conflate failure modes. SwanBench-Speech's insight is that long-form generation should decompose quality into locatable sub-problems.
- **Inspiration for Subsequent Research**: To improve long-form TTS, researchers should not only pursue larger models but also design training objectives and data recipes specifically for reverb stability, paragraph prosody, and emotional hierarchy.

## Rating
- Novelty: ⭐⭐⭐⭐ The technical form of the benchmark is not radical, but the organization of three axes, seven metrics, and 17 scenarios is highly diagnostic.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 20+ models across single-speaker and dialogue tasks with human correlation validation; the scale is solid.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and metric explanations; some tables have high information density requiring close attention.
- Value: ⭐⭐⭐⭐⭐ Very practical for the long-form TTS field, directly showing that models lack long-range consistency and expressive structure rather than just audio quality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] PlanRAG-Audio: Planning and Retrieval Augmented Generation for Long-form Audio Understanding](planrag-audio_planning_and_retrieval_augmented_generation_for_long-form_audio_un.md)
- [\[ICML 2025\] Long-Form Speech Generation with Spoken Language Models](../../ICML2025/audio_speech/long-form_speech_generation_with_spoken_language_models.md)
- [\[ICLR 2026\] YuE: Scaling Open Foundation Models for Long-Form Music Generation](../../ICLR2026/audio_speech/yue_scaling_open_foundation_models_for_long-form_music_generation.md)
- [\[ICCV 2025\] Latent Swap Joint Diffusion for 2D Long-Form Latent Generation](../../ICCV2025/audio_speech/latent_swap_joint_diffusion_for_2d_long-form_latent_generation.md)
- [\[ACL 2026\] Anchored Cyclic Generation: A Novel Paradigm for Long-Sequence Symbolic Music Generation](anchored_cyclic_generation_a_novel_paradigm_for_long-sequence_symbolic_music_gen.md)

</div>

<!-- RELATED:END -->
