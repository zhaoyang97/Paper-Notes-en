---
title: >-
  [Paper Note] IndicMedDialog: A Parallel Multi-Turn Medical Dialogue Dataset for Accessible Healthcare in Indic Languages
description: >-
  [ACL 2026][Medical NLP][Indic Medical Dialogue] This paper introduces IndicMedDialog, the first **parallel multi-turn** medical diagnostic dialogue dataset covering English and 9 Indic languages (Assamese, Bengali…
tags:
  - "ACL 2026"
  - "Medical NLP"
  - "Indic Medical Dialogue"
  - "Parallel Multilingual Dataset"
  - "LoRA Fine-tuning"
  - "Clinical Diagnosis"
  - "Asha Translation Quality Assurance"
date: 2026-05-08
content_hash: 5bc5bcccdb952207
---

# IndicMedDialog: A Parallel Multi-Turn Medical Dialogue Dataset for Accessible Healthcare in Indic Languages

**Conference**: ACL 2026  
**arXiv**: [2605.13292](https://arxiv.org/abs/2605.13292)  
**Code**: https://github.com/ShubhamKumarNigam/IndicMedDialog (Available)  
**Area**: Medical NLP  
**Keywords**: Indic Medical Dialogue, Parallel Multilingual Dataset, LoRA Fine-tuning, Clinical Diagnosis, Asha Translation Quality Assurance

## TL;DR
This paper introduces IndicMedDialog, the first **parallel multi-turn** medical diagnostic dialogue dataset covering English and 9 Indic languages (Assamese, Bengali, Gujarati, Hindi, Marathi, Punjabi, Tamil, Telugu, and Urdu), comprising 2,980 dialogues × 10 languages (29,800 instances). The data was generated via LLaMA-3.3-70B synthesis, TranslateGemma translation, native speaker validation, and script-aware post-processing for phonetic/orthographic correction. Based on 4-bit quantized LLaMA-3.2-3B and LoRA, IndicMedLM was trained, achieving the highest post-processed accuracy in 7/10 languages (including English, Hindi, and Marathi) and a 95.3% medical safety pass rate. The study also identifies 5 types of systemic failure modes (ID, LC, CDC, TTF, PLG).

## Background & Motivation

**Background**: Medical dialogue AI holds significant potential for symptom assessment and preliminary diagnostic advice. However, current systems are mostly single-turn QA and English-centric. Authentic clinical diagnosis requires multi-turn follow-ups to narrow down differential diagnoses, but multi-turn medical dialogue data is virtually non-existent for the 1.5 billion speakers of Indic languages.

**Limitations of Prior Work**: (1) **Single-turn Dominance**: Systems like ChatDoctor assume single-turn interactions, failing to simulate the iterative inquiry-to-diagnosis process; (2) **Templated Datasets**: While MDDial provides multi-turn English diagnostic corpora, its template-based generation lacks linguistic diversity; (3) **Multilingual Gap**: BiMediX addresses English-Arabic, but a parallel corpus for the nine major Indic languages is missing; (4) **Translation Failures**: Off-the-shelf LLM translations for Indic languages suffer from systemic errors in transliteration, vocabulary, and character spacing.

**Key Challenge**: Deploying usable medical dialogue AI in low-resource languages requires solving the "triple constraint" of high-quality multi-turn clinical corpora × parallel multilingualism × affordable compute—factors where data is expensive or private, and technical barriers are high.

**Goal**: (a) Construct the first 10-language parallel multi-turn medical dialogue corpus using a hybrid pipeline of synthesis, translation, and human verification; (b) Train IndicMedLM using 4-bit quantized small models + LoRA for commodity hardware; (c) Introduce optional patient pre-context (age, gender, allergies, etc.) to simulate realistic clinical contexts; (d) Reveal real failure modes in Indic medical dialogues through physician evaluation and a structured error taxonomy.

**Key Insight**: Using MDDial as seed data → LLM-based augmentation for dialogue diversity → TranslateGemma combined with native speaker rating and script-aware post-processing for translation reliability → LoRA + quantization for deployable small models.

**Core Idea**: A tripartite approach of "corpus construction + small-model engineering + systematic error diagnosis" to address the challenges of low-resource Indic medical NLP.

## Method

### Overall Architecture
The framework is divided into data construction, model training, and error analysis:

1.  **Data Construction**: (i) Llama-3.3-70B-Versatile (via Groq) synthesized 1,101 multi-turn diagnostic dialogues covering 12 diseases, 118 symptoms, and 4-8 turns, incorporating non-deterministic patient responses and vague descriptions; merged with 1,879 turns from MDDial to reach 2,980. (ii) TranslateGemma translated the English version into 9 Indic languages with structured prompting to preserve clinical semantics. (iii) Script-aware post-processing mapped phonetic/spelling/spacing errors to the nearest correct forms. (iv) Two native speakers per language independently rated translation quality (T) and clinical safety (S) on 10-point scales (means: $\bar T = 9.50$, $\bar S = 9.56$).
2.  **Model Training (IndicMedLM)**: LLaMA-3.2-3B-Instruct base + 4-bit NF4 quantization + LoRA (rank 16, $\alpha=16$). LoRA was applied to all attention and MLP projections. Training used AdamW-8bit ($lr=2\times 10^{-4}$, $wd=0.001$), batch size 8 (2×4 grad accum), 300 steps with 5 warmups using BF16/FP16. Each language was trained separately using ShareGPT-style formatting; optional patient pre-context was prefixed to dialogues to allow context-aware questioning.
3.  **Two-stage Post-processing Evaluation**: Model outputs often wrap the correct diagnosis in explanatory text, causing raw accuracy to underestimate performance. ChatGPT was utilized as an LLM judge for "constrained semantic equivalence classification"—mapping free-form text to one of 12 standard labels or "NULL," thereby mitigating hallucinations while recovering formatted errors.

### Key Designs

1.  **Multilingual Pipeline with Synthesis + Translation + Script-aware Post-processing**:
    *   **Function**: Generates semantically consistent, clinically sound, and linguistically accurate 10-language parallel corpora without access to real clinical multi-turn transcripts.
    *   **Mechanism**: Llama-3.3-70B synthesis → TranslateGemma translation → Script-aware post-processing for phonetic/orthographic correction → Native speaker double-blind arbitration. Post-processing corrects script-specific issues (e.g., Bengali character spacing) via nearest-form mapping.
    *   **Design Motivation**: Direct translation often produces "Bengali-looking but incorrectly spelled" strings. Post-processing based on Unicode rules for target scripts, validated by native speakers ($\bar T = 9.50$), avoids self-scoring traps.

2.  **Deployable Small Model with LoRA + 4-bit Quantization + Patient Pre-context**:
    *   **Function**: Enables 3B models to perform multi-turn personalized history taking on consumer hardware.
    *   **Mechanism**: (i) 4-bit NF4 quantization reduces VRAM requirements. (ii) LoRA covers both attention and MLP layers ($r=16$) to adapt both linguistic representation and task knowledge. (iii) Optional patient pre-context (age, gender, allergies, etc.) allows the model to skip known info and focus on differential questioning.
    *   **Design Motivation**: The bottleneck for medical AI deployment in rural clinics is the lack of GPU clusters. The architecture is driven by low-compute constraints. Pre-context mirrors real clinical workflows where doctors do not repeat known information.

3.  **Two-stage Post-processing Evaluation + 5-Category Failure Mode Taxonomy**:
    *   **Function**: (a) Recovers "correct but poorly formatted" predictions; (b) Explains the divergence in failure modes across Indic languages.
    *   **Mechanism**: Post-processing uses ChatGPT as a closed-set judge. The study identifies 5 Failure Modes: **FM1 Instruction Drift** (prose without labels), **FM2 Label Collapse** (multiple diseases mapped to one label, e.g., mapping all Bengali cases to "Lung Infection"), **FM3 Cross-Domain Confusion** (e.g., CAD → Thyroiditis), **FM4 Tokenization/Truncation Failure** (truncation in Punjabi/Telugu), and **FM5 Paraphrase-over-Label Generation** (descriptive output instead of standard labels).
    *   **Design Motivation**: Raw accuracy severely underestimates Hindi/Marathi performance (19% vs. 73% post-processed). The taxonomy categorizes "model failure" into formatting, label collision, domain confusion, truncation, or paraphrasing.

### Loss & Training
Standard causal LM SFT loss (no special reward/KD). Each language was trained with identical hyperparameters. Inference parameters: $temp=0.1, top\_p=0.95, max\_new=128$. Evaluation included (i) automated diagnostic accuracy (raw vs. post) and (ii) expert Likert scoring (1-5) and safety checks by three medical students (MBBS), achieving Krippendorff's $\alpha=0.81$.

## Key Experimental Results

### Main Results

Diagnostic accuracy (%) across 10 languages (Raw: original string match; Post: LLM judge recovery):

| Language | GEMMA Post | Tiny-AYA Post | LLaMA Base Post | **IndicMedLM Raw** | **IndicMedLM Post** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| English | 45.11 | 13.19 | 15.74 | 80.85 | **80.85** |
| Hindi | 25.10 | 13.19 | 11.06 | 19.15 | **72.76 (+53.6pp)** |
| Marathi | 9.36 | 5.11 | 11.50 | 13.19 | **68.51 (+55.3pp)** |
| Bengali | 19.57 | 5.96 | 11.50 | 25.11 | **58.72** |
| Urdu | 2.12 | 13.61 | 2.55 | 4.26 | **28.51** |
| Gujarati | 18.72 | **37.02** | 18.30 | 18.30 | 19.57 |
| Punjabi | 7.66 | 8.12 | 8.51 | 5.96 | **20.42** |
| Assamese | 7.66 | 8.08 | 3.83 | 5.96 | 5.96 |
| Tamil | 11.91 | 3.83 | 6.80 | 6.38 | 6.80 |
| Telugu | 6.38 | 0.00 | 4.68 | 1.28 | 5.96 |

**Ours** achieves the highest post-processed accuracy in 7/10 languages. The massive Raw-to-Post jumps in Hindi (+53.6pp) and Marathi (+55.3pp) indicate that true diagnostic capability was hidden by formatting artifacts.

### Ablation Study / Expert Evaluation

| Dimension | IndicMedLM |
| :--- | :--- |
| Medical Safety Pass Rate | **95.3%** |
| Symptom Extraction (1-5) | 4.20 |
| Context Memory (1-5) | 4.40 |
| Diagnostic Correctness (1-5) | 4.10 |
| Conversational Flow (1-5) | 4.30 |
| Efficiency (1-5) | 4.00 |
| Inter-annotator Krippendorff $\alpha$ | 0.81 (Strong) |
| Translation Quality (10-pt) | $\bar T = 9.50$ |
| Clinical Safety (10-pt) | $\bar S = 9.56$ |

### Key Findings
-   **Pseudo-low scores in Hindi/Marathi**: Raw scores of 19%/13% jumped to 73%/69% post-processing, showing that the model tends to wrap answers in idiomatic hedging, a metric artifact rather than a lack of capability.
-   **Extreme variance across diseases**: Traumatic Brain Injury achieved 94.7% in English/Hindi but 0% in Assamese/Tamil/Telugu; conversely, Conjunctivitis achieved 100% in Punjabi despite low overall language scores.
-   **Tokenizer Gap**: FM4 Truncation appeared only in Punjabi/Telugu (Gurmukhi/Telugu scripts) and not in Hindi/Marathi (Devanagari), proving that LLaMA's Unicode coverage is the bottleneck rather than data quantity.
-   **Bengali Label Collapse**: Mapping all cases to "Lung Infection" reflects a majority-class bias in semantic hypernyms, suggesting the need for balanced SFT labels.
-   **95.3% Safety Pass**: 1483/1556 dialogues were physician-verified as safe, indicating controlled medical risk in the synthesis + LoRA route.

## Highlights & Insights
-   The creation of the first 10-language parallel multi-turn medical dialogue corpus is a significant community contribution for 1.5 billion people.
-   The **5-Category Failure Mode Taxonomy** serves as a "diagnostic framework for diagnostic errors," providing a roadmap for improvement (e.g., FM1 requires formatting rewards, while FM4 requires better tokenizers).
-   **Post-processing recovery** highlights a universal lesson: idiomatic hedging in low-resource languages can lead hard-matching metrics to drastically underestimate model intelligence.
-   **Patient Pre-context** is a simple but effective clinical design that prevents AI from asking redundant questions.
-   **LoRA + 4-bit Quantization** explicitly addresses the accessibility mission by enabling deployment in resource-constrained regions.

## Limitations & Future Work
-   The dataset covers only 12 diseases and 118 symptoms; it uses synthetic data which lacks the ground-truth nuance of real doctor-patient interactions.
-   Accuracy remains below 10% for languages like Assamese, Tamil, and Telugu, largely due to base model tokenizer limitations. A potential fix is using Indic-optimized base models (e.g., Sarvam-1).
-   Evaluation relies on ChatGPT as a judge (evaluator dependency). Expert evaluation was conducted on a limited sample size.

## Related Work & Insights
-   **vs. MDDial (2023)**: MDDial uses English templates; **Ours** upgrades it with synthesis-based diversity and 10-language parallel scaling.
-   **vs. BiMediX (2024)**: BiMediX covers English-Arabic; **Ours** extends this to the high-demand Indic language group.
-   **vs. NoteChat (2024)**: NoteChat generates dialogues from clinical notes; **Ours** uses disease schemas. Future work could combine both for higher realism.

## Rating
-   Novelty: ⭐⭐⭐⭐ (First parallel corpus; engineering composition is strong).
-   Experimental Thoroughness: ⭐⭐⭐⭐ (Comprehensive 10-language analysis, 12-disease breakdown, and physician evaluation).
-   Writing Quality: ⭐⭐⭐⭐ (Clear structure and well-designed tables).
-   Value: ⭐⭐⭐⭐⭐ (Opens source data and models for underserved regions).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ATPO: Adaptive Tree Policy Optimization for Multi-Turn Medical Dialogue](../../ICLR2026/medical_nlp/atpo_adaptive_tree_policy_optimization_for_multi-turn_medical_dialogue.md)
- [\[ACL 2026\] SEMA-RAG: A Self-Evolving Multi-Agent Retrieval-Augmented Generation Framework for Medical Reasoning](sema-rag_a_self-evolving_multi-agent_retrieval-augmented_generation_framework_fo.md)
- [\[ACL 2026\] Efficient and Effective Internal Memory Retrieval for LLM-Based Healthcare Prediction](efficient_and_effective_internal_memory_retrieval_for_llm-based_healthcare_predi.md)
- [\[ACL 2026\] Beyond the Individual: Virtualizing Multi-Disciplinary Reasoning for Clinical Intake via Collaborative Agents](beyond_the_individual_virtualizing_multi-disciplinary_reasoning_for_clinical_int.md)
- [\[ACL 2026\] BioHiCL: Hierarchical Multi-Label Contrastive Learning for Biomedical Retrieval with MeSH Labels](biohicl_hierarchical_multi-label_contrastive_learning_for_biomedical_retrieval_w.md)

</div>

<!-- RELATED:END -->
