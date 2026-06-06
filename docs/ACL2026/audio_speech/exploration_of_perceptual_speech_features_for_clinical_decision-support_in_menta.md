---
title: >-
  [Paper Note] Exploration of Perceptual Speech Features for Clinical Decision-Support in Mental Health Care
description: >-
  [ACL2026][Audio & Speech][Speech Mental Health] This paper proposes an explainable speech analysis framework for clinical mental health assistance. It combines perceptually understandable acoustic and linguistic features…
tags:
  - "ACL2026"
  - "Audio & Speech"
  - "Speech Mental Health"
  - "Explainable Machine Learning"
  - "Acoustic Features"
  - "Linguistic Features"
  - "SHAP/LIME"
date: 2026-05-08
content_hash: 1d4d21165d87b71c
---

# Exploration of Perceptual Speech Features for Clinical Decision-Support in Mental Health Care

**Conference**: ACL2026  
**arXiv**: [2605.24678](https://arxiv.org/abs/2605.24678)  
**Code**: No public code (source repository not found in cache)  
**Area**: Clinical Speech Analysis / Mental Health  
**Keywords**: Speech Mental Health, Explainable Machine Learning, Acoustic Features, Linguistic Features, SHAP/LIME  

## TL;DR
This paper proposes an explainable speech analysis framework for clinical mental health assistance. It combines perceptually understandable acoustic and linguistic features with XGBoost, statistical testing, SHAP, and LIME to identify stable vocal behavioral cues across multiple datasets—including stress, depression, anxiety, and ADHD—rather than pursuing black-box end-to-end diagnosis.

## Background & Motivation
**Background**: Speech and language have been widely used for mental health assessment because speaking patterns reflect emotion, cognitive load, neurological state, and social expression. Recent systems often use large-scale speech models, end-to-end acoustic representations, or multimodal deep models to achieve good classification performance on tasks such as depression, anxiety, stress, insomnia, and fatigue.

**Limitations of Prior Work**: In clinical scenarios, providing only a classification result is insufficient. Even if black-box models achieve high accuracy, they struggle to inform clinicians "where the decision comes from." It is even harder to distinguish whether the model captures symptomatic cues or confounding factors such as recording equipment, linguistic background, noise, or fatigue. Since mental health assessment is a high-risk scenario, incorrect labels can lead to stigmatization and inappropriate intervention, making explainability and clinical readability more important than pure leaderboard metrics.

**Key Challenge**: End-to-end representations are typically stronger, but the learned dimensions may not map to clinical phenomena; manual features are more explainable, but single feature groups are insufficient to cover the complexity of mental health expression. This paper attempts to compromise between "explainable features" and "non-linear modeling": features must be understandable to clinicians, while the model must be capable of capturing interactions across features.

**Goal**: The authors aim to build a reusable analysis pipeline across datasets to systematically compare the relationships between acoustic, linguistic, emotional, semantic coherence, and pragmatic cues with mental health scales or diagnostic labels. The goal is not to replace clinical diagnosis but to provides auditable, explainable candidate indicators for clinical decision support.

**Key Insight**: The paper starts from "perceptual features": speech rate, pauses, jitter, shimmer, pitch variation, syntactic complexity, lexical richness, semantic coherence, emotional polarity, and sarcasm probability can all be interpreted as specific speaking behaviors. XGBoost, which is suitable for tabular features, is used to handle non-linear relationships, with SHAP/LIME employed to track which features drive the judgment.

**Core Idea**: Replace pure black-box embeddings with perceptible and explainable acoustic and linguistic features, and jointly verify whether these features are stably associated with mental health states through statistical testing and explainable machine learning.

## Method
The method is structured as a clinical analysis pipeline rather than a single neural network. It converts each speech segment and transcript into 82 scalar features, constructs binary classification labels based on clinical tasks, and then uses statistical tests, XGBoost, SHAP, LIME, and feature group ablation to determine which cues have explanatory value.

### Overall Architecture
The input consists of speech samples with audio, transcripts, and mental health labels or scale scores. The system first performs audio preprocessing: converting to mono, resampling to 16 kHz, amplitude normalization, and extracting acoustic metrics from voiced segments. On the text side, spaCy and Stanza are used for tokenization, POS tagging, dependency parsing, and constituent parsing. Sentence-BERT is used to estimate semantic coherence, and VADER is used to extract emotional polarity.

Subsequently, all features are organized into several explainable groups: prosodic/fluency, voice quality, lexical, syntactic, semantic, psycholinguistic, etc. Each dataset forms a binary classification task based on existing labels or scale thresholds, such as stress / non-stress in STRESSID, PHQ-8 depression thresholds in DAIC-WOZ, and PHQ-9/GAD-7/ASRS clinical cutoffs in the REAL dataset.

The analysis phase proceeds via three paths: the first uses independent sample t-tests with FDR control for multiple comparisons to observe feature differences between groups; the second uses an XGBoost classifier to capture non-linear combinations; the third uses SHAP, LIME, and feature importance to explain the speech and linguistic cues relied upon by the model. Finally, the authors use feature group ablation to check the independent contribution of single feature groups.

### Key Designs
1. **Perceptual Explainable Multi-group Feature System**:
	- **Function**: Decomposes speech mental health analysis into behavioral dimensions understandable to clinicians and researchers, rather than directly relying on uninterpretable embeddings.
	- **Mechanism**: Acoustic side extracts pitch, intensity, jitter, shimmer, HNR, ZCR, pause, phonation/articulation rate, rhythm, and entropy, using a HuBERT emotion model for emotion-related features. Text side extracts TTR, MATTR, Brunet, Honore, POS/morphological diversity, syntactic depth, dependency graph metrics, Sentence-BERT coherence, VADER sentiment, and sarcasm probability.
	- **Design Motivation**: Mental health-related expressions may be reflected in both "how" and "what" is said. Text alone misses pauses, voice quality, and prosody; audio alone misses semantic coherence, self-reference, negative words, and syntactic complexity.

2. **Dual-layer Analysis of Statistical Testing and XGBoost**:
	- **Function**: Simultaneously answers "which features differ significantly between groups" and "which features have predictive power in non-linear classification."
	- **Mechanism**: First, t-tests compare two groups defined by clinical thresholds, corrected by Benjamini-Hochberg FDR; then, XGBoost is trained as a feature-level analysis model rather than an uninterpretable diagnoser.
	- **Design Motivation**: Pure significance tests easily miss interaction relationships, while pure classification metrics might mask the model's reasoning. Combining them retains statistical readability while capturing complex combinatorial patterns.

3. **SHAP/LIME and Feature Group Ablation**:
	- **Function**: Validates whether model judgments align with clinically reasonable speech behaviors and estimates the contribution of different feature groups.
	- **Mechanism**: XGBoost gain, SHAP summary, and aggregated LIME explanations are used to rank features. Ablation experiments retain only one feature group at a time to compare AUC-ROC trends across datasets.
	- **Design Motivation**: In medical scenarios, "why the model judged so" is as important as "how accurately the model judged." Reliability is higher if different explanation methods point to similar features like jitter, shimmer, pause, negative affect, or graph repetitions.

### Loss & Training
This paper is not an end-to-end deep training paper; the core training object is the XGBoost classifier. Training strategies include: constructing binary classification tasks by dataset; performing subject-level feature aggregation (using the median for audio files per participant in the REAL dataset); using 4-fold subject-independent cross-validation in REAL to avoid speaker leakage; and performing 10 random runs for STRESSID based on the original paper's settings. Sarcasm detection is trained as an auxiliary model on MUStARD, with a reported accuracy of approximately 70% in the cache, and thereafter only the sarcasm probability is used as an additional feature.

## Key Experimental Results

### Main Results
| Dataset / Task | Evaluation Setting | Ours | Prev. SOTA | Observation |
|--------|------|------|----------|------|
| STRESSID (Stress) | 10 Random Runs | Accuracy 0.70, F1 0.81 | Wav2Vec+LR: Acc 0.66, F1 0.70 | Perceptual features + XGBoost performed better here |
| DAIC-WOZ (Depression) | Participant speech + XGBoost | Accuracy 0.66, F1 0.56, AUC 0.63 | LSTM F1 0.64 | Moderate performance, but explanations show rational cues like pauses |
| ANDROIDS (Depression) | Participant-level aggregation | Accuracy 75.6%, F1 77.1%, AUC 87.6% | Original LSTM F1 0.83 | Strong AUC, but F1 is lower than LSTM |
| EATD (Chinese Depression) | Acoustic + Linguistic | Accuracy 82.1%, F1 53.9%, AUC 73.4% | GRU F1 0.71 | Performance unstable under category or language conditions |
| REAL (ASRS / PHQ-9 / GAD-7) | 4-fold speaker-disjoint CV | AUC 0.67 / 0.63 / 0.59 | N/A (External SOTA not in cache) | Real clinical intake scenarios are significantly harder |

### Significant Features & Analysis
| Scenario | Significant or Important Features | Value / Trend | Interpretation |
|------|---------|------|------|
| STRESSID | Shimmer_local | Non-stressed 0.343, Stressed -0.142, $p = 1.27 \times 10^{-5}$ | Vocal cord amplitude perturbation correlates with stress |
| STRESSID | Jitter_local | Non-stressed 0.368, Stressed -0.152, $p = 4.14 \times 10^{-4}$ | Pitch period perturbation is a primary distinguishing cue |
| REAL-PHQ-9 | vader_negative | Non-Dep 0.030, Dep 0.050, $p = 1.22 \times 10^{-4}$ | Higher negative text sentiment in the depression group, remains significant after FDR |
| REAL-ASRS | Tense=Pres | Non-ADHD 6.22, ADHD 7.81, $p = 2.28 \times 10^{-4}$ | Verb tense and repetitive graph structures are important in ADHD tasks |
| REAL-GAD-7 | vader_negative, Shimmer_local | $p = 6.81 \times 10^{-3} / 2.17 \times 10^{-2}$, but not significant after FDR | Anxiety signals show trends but lack statistical robustness |

### Key Findings
- No single feature group covers all tasks; in Figure 1, prosodic features have the highest single-group average AUC-ROC, followed by psycholinguistic language and acoustic features.
- Shimmer and voice quality repeatedly appear in stress and anxiety analyses, suggesting amplitude perturbation as a reusable candidate indicator, though the direction may vary across studies.
- In ADHD-related ASRS tasks, linguistic organizational features like repetitive graph structures, verb tense switches, and filler_count are more explanatory than pure sentiment words.
- AUC on real-world datasets is lower than on controlled datasets, indicating that the difficulty of clinical implementation comes from recording conditions, cultural-linguistic backgrounds, questionnaire cutoffs, and sample dynamics.

## Highlights & Insights
- **Placing "Explainability" as Priority 1**: Instead of chasing the largest models, the paper chooses acoustic and linguistic behaviors that clinicians can understand as features. This makes the results suitable for clinical assistance rather than becoming another opaque screener.
- **Testing the same pipeline across 5 datasets**: STRESSID, DAIC-WOZ, ANDROIDS, EATD, and REAL cover stress, depression, anxiety, ADHD, multi-language, and real-world intake scenarios. While results are not always perfect, this heterogeneity is closer to real deployment pressures.
- **Cross-validation of statistical significance and model explanation**: t-tests, FDR, XGBoost, SHAP, and LIME are not just piled up; they confirm from different angles which cues are worth considering as candidate clinical indicators.
- **Value of Negative Results**: Performance on EATD and REAL exposes vulnerability across languages, devices, and clinical scales, serving as a reminder to subsequent research not to over-promise.

## Limitations & Future Work
- The authors explicitly point out that speech is affected by fatigue, background noise, equipment, language/cultural background, and labeling protocols. Many "mental health signals" may be weakened or distorted by confounding factors.
- Some tasks use questionnaire cutoffs (PHQ-8/9, GAD-7, ASRS) as supervision signals. These scales are common but do not equal clinical ground truth; indicator errors may stem from the cutoffs rather than the model.
- Currently, mainly short speech segments and static aggregated features are used, potentially ignoring symptom trajectories over time. Future work could introduce longitudinal modeling, domain-invariant preprocessing, and more natural long-term speech data.
- The sarcasm detection model has an accuracy of ~70% and contains its own errors and training data bias; thus, sarcasm probability should be interpreted cautiously and not as strong clinical evidence.
- Differences in labels, languages, tasks, and recording settings between datasets are large. Future cross-site, cross-device, and cross-cultural validation is needed rather than just hyperparameter tuning on public benchmarks.

## Related Work & Insights
- **Vs. End-to-end Wav2Vec / HuBERT models**: These models usually learn strong representations but have high clinical interpretation costs; this paper uses HuBERT only for auxiliary emotional features while keeping the main framework feature-level explainable.
- **Vs. LSTM Depression Detection on DAIC-WOZ**: LSTM has higher F1, but this paper clearly identifies how factors like pauses, pitch variation, intensity, and jitter/shimmer contribute to judgments, aiding hypothesis generation.
- **Vs. Text-only Mental Health Analysis**: Looking only at text misses prosody, pause, and voice quality; this paper suggests that mental health speech analysis should simultaneously model content, expression, and pragmatic cues.
- **Inspiration**: For clinical NLP/speech screening, the most stable direction may not be "larger models for direct diagnosis" but "explainable candidate indicators + explicit uncertainty + clinician review."

## Rating
- Novelty: ⭐⭐⭐⭐☆ The multi-feature and XAI components themselves are not new, but organizing them into an explainable clinical analysis framework across multiple mental health tasks has practical value.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Broad dataset coverage with sufficient tables and explanations; the weakness lies in the limited consistency across datasets and lack of external clinical validation.
- Writing Quality: ⭐⭐⭐⭐☆ Clear motivation, readable methods and results; some appendix feature tables are long, and reporting of ablation values in the main text is slightly incomplete.
- Value: ⭐⭐⭐⭐☆ Highly relevant for clinical explainable speech analysis, especially for developing auditable mental health support systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] An Exploration of Mamba for Speech Self-Supervised Models](an_exploration_of_mamba_for_speech_self-supervised_models.md)
- [\[ICLR 2026\] MAPSS: Manifold-Based Assessment of Perceptual Source Separation](../../ICLR2026/audio_speech/mapss_manifold-based_assessment_of_perceptual_source_separation.md)
- [\[ACL 2026\] S2S-Arena: Evaluating Paralinguistic Instruction Following in Speech-to-Speech Models](s2s-arena_evaluating_paralinguistic_instruction_following_in_speech-to-speech_mo.md)
- [\[ACL 2026\] FC-TTS: Style and Timbre Control in Zero-Shot Text-to-Speech with Disentangled Speech Representations](fc-tts_style_and_timbre_control_in_zero-shot_text-to-speech_with_disentangled_sp.md)
- [\[ACL 2026\] From Flat Language Labels to Typological Priors: Structured Language Conditioning for Multilingual Speech-to-Speech Translation](from_flat_language_labels_to_typological_priors_structured_language_conditioning.md)

</div>

<!-- RELATED:END -->
