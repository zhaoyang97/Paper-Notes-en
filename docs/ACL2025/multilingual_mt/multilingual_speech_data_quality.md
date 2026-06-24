---
title: >-
  [Paper Note] Data Quality Issues in Multilingual Speech Datasets: The Need for Sociolinguistic Awareness and Proactive Language Planning
description: >-
  [ACL 2025][Multilingual & Machine Translation][Multilingual Speech Datasets] A systematic quality audit of three major public multilingual speech datasets (Common Voice 17.0, FLEURS, VoxPopuli) covering 40+ languages is conducted. Issues are categorized into programmable "micro-level issues" and linguistically involved "macro-level issues." It is found that macro-level issues are particularly severe for low-institutionalization languages. A 5-step dataset creation guide incor…
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "Multilingual Speech Datasets"
  - "Data Quality Audit"
  - "Sociolinguistics"
  - "Language Planning"
  - "Low-Resource Languages"
date: 2026-05-08
content_hash: 42e18541fde5233c
---

# Data Quality Issues in Multilingual Speech Datasets: The Need for Sociolinguistic Awareness and Proactive Language Planning

**Conference**: ACL 2025  
**arXiv**: [2506.17525](https://arxiv.org/abs/2506.17525)  
**Code**: None  
**Area**: Multilingual Translation  
**Keywords**: Multilingual Speech Datasets, Data Quality Audit, Sociolinguistics, Language Planning, Low-Resource Languages

## TL;DR

A systematic quality audit of three major public multilingual speech datasets (Common Voice 17.0, FLEURS, VoxPopuli) covering 40+ languages is conducted. Issues are categorized into programmable "micro-level issues" and linguistically involved "macro-level issues." It is found that macro-level issues are particularly severe for low-institutionalization languages. A 5-step dataset creation guide incorporating sociolinguistic awareness is proposed.

## Background & Motivation

**Background**: State-of-the-art ASR models such as Whisper, Google USM, SeamlessM4T, and MMS highly rely on large-scale multilingual speech datasets for training and evaluation. Mozilla Common Voice, FLEURS, and VoxPopuli are the three most widely used public datasets, supporting tasks such as speech recognition, cross-lingual representation learning, and multilingual speech generation.

**Limitations of Prior Work**: Although these datasets are cited and used by countless works, their intrinsic quality—especially the quality of low-resource language subsets—has barely been systematically studied. Existing data quality audit works (e.g., Kreutzer et al. 2022) mainly target text datasets, and the speech field lacks a corresponding audit methodology. More dangerously, problematic test sets may "create an illusion of success": models achieve seemingly good WER on erroneous data but perform poorly in real-world scenarios.

**Key Challenge**: Dataset quality issues are divided into two levels. One class consists of language-agnostic "micro-level issues" (e.g., excessively short sentences, excessive silence), which can be detected via automatic metrics and fixed programmatically. The other class consists of "macro-level issues" rooted in sociolinguistic contexts (e.g., mixed writing systems in digraphic languages, register confusion in diglossic languages), which cannot be detected automatically and require linguistic experts. The latter are particularly severe in low-institutionalization languages but have been almost completely ignored.

**Goal**: (1) Systematically audit the three major datasets using both quantitative and qualitative methods; (2) Establish a "micro-macro" quality issue categorization framework; (3) Quantify the actual impact of data quality on downstream ASR evaluation through case experiments; (4) Propose an actionable dataset creation guide.

**Key Insight**: Starting from a sociolinguistic perspective, the authors observe that many low-resource languages exhibit complex phenomena such as digraphia, diglossia, and dialect continuums. If these phenomena are neglected during dataset creation, they fundamentally damage the usability of the data. The author team has access to native speaker resources covering about 40 languages, making large-scale manual auditing feasible.

**Core Idea**: Systematically audit multilingual speech datasets from a sociolinguistic perspective to reveal overlooked "macro-level quality issues" in low-institutionalization languages, and redefine dataset creation as a community-led language planning practice.

## Method

### Overall Architecture

A dual-dimensional audit is conducted on three datasets: Mozilla Common Voice 17.0 (MCV17, 124 locales), FLEURS (101 languages), and VoxPopuli (16 European languages). Statistically, automatic metrics such as SNR, VAD speech ratio, median sentence duration, and median word count are calculated. Qualitatively, native speaker volunteers of approximately 40 languages are invited to manually audit 100 randomly sampled instances per language across five dimensions: coherence, audio-text alignment, dialect, thematic domain, and language classification. The identified quality issues are summarized into "micro-level" and "macro-level" categories, and the impact of macro-level issues on downstream evaluation is quantitatively verified through Norwegian ASR experiments.

### Key Designs

1. **Micro-Level Issue Detection Framework**:

    - **Function**: Detect and quantify language-agnostic, typically programmatically repairable data quality issues.
    - **Mechanism**: Systematically scan datasets across four dimensions: (a) Sentence duration distribution (35 languages in MCV17 have median duration <4 seconds; nan_tw/sr/br <3 seconds); (b) Voice activity ratio (using neural VAD models to classify speech/non-speech segments; Basaa/Zaza/Serbian/Danish speech ratio <50%); (c) Domain balance (FLEURS is biased toward encyclcemic genres due to its Wikipedia origin; MCV17 has templated repetitive sentences); (d) Speaker diversity (Macedonian has only 19 speakers; Zulu etc. have only 1).
    - **Design Motivation**: Micro-level issues form the foundational baseline of data quality. Although relatively easy to fix, if left undetected (for instance, nan_tw nominally has 21 hours of audio but actually only 10 hours of usable speech), they lead to a severe overestimation of downstream training data volume.

2. **Macro-Level Issue Analysis Framework**:

    - **Function**: Identify deep quality issues caused by sociolinguistic complexities that require linguistic expertise to diagnose.
    - **Mechanism**: Focus on three types of phenomena: (a) Digraphia, where writing systems are unspecified or mixed, e.g., Norwegian Bokmål/Nynorsk has 8.1%/8.8% of "incorrect" orthography mixed into MCV17 and FLEURS respectively, which the authors quantify using automated classification scripts (Algorithm 1); (b) Diglossia, where registers are confused, e.g., 98.6% of the FLEURS subset labeled as ar_eg (Egyptian Arabic) is actually MSA, and 89.8% of the yue_hk (Cantonese) subset is Standard Written Chinese without any Cantonese content; (c) Unspecified dialect continuums, e.g., Fula only contains the Senegalese variant, omitting the most widely spoken Guinean variant.
    - **Design Motivation**: Macro-level issues are more hidden and far-reaching than micro-level ones. They do not manifest in automatic metrics but cause downstream models to output incorrect registers (e.g., Whisper-v3 exhibits unpredictable "automatic translation" in Cantonese), and model distillation amplifies these issues (WER deteriorates from 10.8% to 46.1%).

3. **Proactive Language Planning Checklist**:

    - **Function**: Provide an actionable quality assurance checklist for the future creation of multilingual speech datasets.
    - **Mechanism**: (a) Sociolinguistic assessment—conduct a comprehensive survey on the target language regarding demographics, literacy rates, writing systems, and diglossia/digraphia before creation; (b) Language planning in dataset design—collaborate with linguists and the community to determine specific register, script, and dialect choices; (c) Proactive standardization—provide contributors with detailed orthographic, script, and register guidelines, particularly for languages with low literacy rates or lacking standard orthography; (d) Multi-layered quality assurance—combine automatic metrics (reject silence/extremely short audio, incorrect script) and manual evaluation (reject incorrect register/out-of-scope dialect); (e) Transparent metadata—publish detailed documentation of language planning decisions.
    - **Design Motivation**: The authors observe that community-driven projects like Common Voice, when expanding to languages with complex sociolinguistic backgrounds, generate implicit consensus-lacking language planning decisions (such as proposals to merge Norwegian Nynorsk and Bokmål). Proactive and conscious planning must replace passive chaos.

### Loss & Training

Since this is an auditing work, no model training is proposed. The baseline ASR experiments used for validation employ a 120M-parameter Conformer HAT model trained on Norwegian Bokmål data, evaluated for WER on MCV17 nn_no (Nynorsk) and FLEURS nb_no (Bokmål) respectively.

## Key Experimental Results

### Main Results

| Test Set | Total WER↓ | Deletion Rate | Insertion Rate | Substitution Rate |
|--------|---------|--------|--------|--------|
| MCV17 nn_no (Nynorsk) | **49.1%** | 11.8% | 1.6% | **35.0%** |
| FLEURS nb_no (Bokmål) | **23.8%** | 11.1% | 2.2% | **10.0%** |

While deletion and insertion error rates are nearly identical across both datasets, the substitution error rate on Nynorsk is **25% absolute percentage points higher** than on Bokmål. Manual inspection confirms that most substitution errors stem from orthographic variations (e.g., jeg↔eg), validating the destructive impact of mixed writing systems on WER evaluation.

### Ablation Study

| Dataset | Quality Dimension | Typical Problem Found | Severity |
|--------|----------|-------------|----------|
| MCV17 | Extremely short duration | Median <4s in 35 languages; nan_tw/sr/br <3s | High |
| MCV17 | Low voice ratio | nan_tw has only 48.3% speech (21h→10h available) | High |
| MCV17 | Lack of speakers | zu/nso/ht have only 1 speaker | High |
| MCV17 | Mixed orthography | nn_no contains 8.1% Bokmål | Medium |
| FLEURS | Register confusion | ar_eg is 98.6% MSA instead of Egyptian Arabic | High |
| FLEURS | Mislabeled language | yue_hk is 89.8% SWC, 0% Cantonese | Extremely High |
| FLEURS | Omitted dialects | ff_sn contains only Peul dialect, lacking Guinean variant | Medium |
| VoxPopuli | — | No macro-level issues found (contains only highly institutionalized European languages) | Low |

### Key Findings

- **Strong positive correlation between language institutionalization and data quality**: VoxPopuli, which contains only highly institutionalized European languages, exhibits no macro-level issues. In contrast, problems explode in low-institutionalization languages in MCV17 and FLEURS.
- **Micro-level issues can be fixed, whereas macro-level issues are fundamental**: Short durations and low speech ratios can be processed programmatically, but issues like the entire FLEURS yue_hk subset being labeled with the wrong language can only be resolved by rebuilding from scratch.
- **Downstream impacts are empirically validated**: The unpredictable output of Whisper-v3 in Cantonese is a direct consequence of the register mislabeling in FLEURS. Similarly, the inability of the LangID system of Costa-jussà et al. 2022 to distinguish between zh_hk and yue also stems from this.
- **Community-driven effort is a double-edged sword**: Open contribution in Common Voice improves coverage but introduces implicit language planning decisions lacking consensus. The dictionary-like structure of nan_tw is a compromise made by contributors to maximize participation.

## Highlights & Insights

- **First Systematic Audit**: This represents the first systematic quality audit covering 40+ languages across three mainstream speech datasets, filling a void in speech data quality research. Previously, only Kreutzer et al. 2022 had conducted similar work on text datasets.
- **Precise abstraction of the "micro-macro" classification framework**: The former corresponds to engineering problems (automatable), while the latter corresponds to sociolinguistic problems (requiring expert participation). This dichotomy can be transferred to the quality evaluation of any multilingual dataset.
- **Redefining dataset creation as language planning**: This is the most insightful perspective. For languages lacking a written tradition, the creation of an ASR dataset inherently forces the community to make orthographic, register, and dialect choices. Instead of letting these decisions happen implicitly and chaotically, they should be proactively leveraged as community-driven tools for language planning and revitalization.

## Limitations & Future Work

- Although covering 40+ languages, a large number of languages across the three datasets remain unchecked, meaning the audit coverage is limited.
- The 5-step guide assumes the availability of linguistic experts and native speaker resources, which may be impractical for small teams and communities that inherently lack these resources.
- There is a lack of tools to automatically detect macro-level issues; the paper identifies issues but does not propose scalable, automated solutions.
- The evaluation of domain balance is highly subjective (the definition of "everyday conversation" varies by application scenario) and lacks formal metrics.
- The in-depth case study of nan_tw is outstanding, but the analysis of other languages (e.g., Arabic, Cantonese) is relatively shallow and could be expanded.

## Related Work & Insights

- **vs. Kreutzer et al. 2022 (Text Data Auditing)**: While they audit text datasets, this paper extends the audit methodology to the speech domain, adding audio-specific dimensions (SNR, VAD, duration distribution) and sociolinguistic dimensions (digraphia/diglossia) to provide a more comprehensive perspective.
- **vs. Ardila et al. 2020 (Common Voice)**: While Common Voice emphasizes community involvement and multilingual coverage, this paper exposes quality hazards under such decentralized models in low-institutionalization languages, providing a complementary viewpoint.
- **vs. Bender & Friedman 2018 (Data Statements)**: The Data Statements framework proposes general transparency criteria. Building upon this, this paper puts forward more specific language planning metadata requirements targeted at multilingual speech scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ The first to introduce a sociolinguistic perspective to multilingual speech dataset audits. The "micro-macro" framework has high original value.
- Experimental Thoroughness: ⭐⭐⭐⭐ The manual audit of 40+ languages combined with the Norwegian ASR validation is persuasive, though automated tools are missing.
- Writing Quality: ⭐⭐⭐⭐⭐ The case analyses are vivid and in-depth (nan_tw, Norwegian, Cantonese), the arguments are well-structured, and the guidelines are clear and actionable.
- Value: ⭐⭐⭐⭐⭐ Provides direct guidance for both creators and users of multilingual datasets, diagnosing the data-level root causes of issues in models like Whisper.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Alleviating Distribution Shift in Synthetic Data for Machine Translation Quality Estimation](alleviating_distribution_shift_in_synthetic_data_for_machine_translation_quality.md)
- [\[ACL 2025\] Building Better: Avoiding Pitfalls in Developing Language Resources when Data is Scarce](building_better_avoiding_pitfalls_in_developing_language_resources_when_data_is_.md)
- [\[ACL 2025\] Comparative Analysis of Multilingual Hate Speech Detection](comparative_analysis_of_multilingual_hate_speech_detection.md)
- [\[ACL 2025\] SIFT-50M: A Large-Scale Multilingual Dataset for Speech Instruction Fine-Tuning](sift-50m_a_large-scale_multilingual_dataset_for_speech_instruction_fine-tuning.md)
- [\[ACL 2025\] CulFiT: A Fine-grained Cultural-aware LLM Training Paradigm via Multilingual Critique Data Synthesis](culfit_a_fine-grained_cultural-aware_llm_training_paradigm_via_multilingual_crit.md)

</div>

<!-- RELATED:END -->
