---
title: >-
  [Paper Note] Natural Language Processing in Support of Evidence-based Medicine: A Scoping Review
description: >-
  [ACL2025][LLM (Other)][evidence-based medicine] This scoping review of 129 studies (2019-2024) follows the PRISMA guidelines, using the five-step EBM process (Ask-Acquire-Appraise-Apply-Assess) as an organizational framework to comprehensively survey the current application status, technological evolution pathways, and future directions of NLP technologies in evidence-based medicine.
tags:
  - "ACL2025"
  - "LLM (Other)"
  - "evidence-based medicine"
  - "NLP survey"
  - "clinical NLP"
  - "PICO extraction"
  - "clinical trial matching"
  - "systematic review"
  - "LLM"
date: 2026-05-08
content_hash: 341c2885c2886665
---

# Natural Language Processing in Support of Evidence-based Medicine: A Scoping Review

**Conference**: ACL2025  
**arXiv**: [2505.22280](https://arxiv.org/abs/2505.22280)  
**Code**: [GitHub](https://github.com/bionlplab/awesome-nlp-in-ebm)  
**Institution**: Weill Cornell Medicine, Columbia University, University of Sydney  
**Area**: LLM/NLP  
**Keywords**: evidence-based medicine, NLP survey, clinical NLP, PICO extraction, clinical trial matching, systematic review, LLM

## TL;DR

This scoping review of 129 studies (2019-2024) follows the PRISMA guidelines, using the five-step EBM process (Ask-Acquire-Appraise-Apply-Assess) as an organizational framework to comprehensively survey the current application status, technological evolution pathways, and future directions of NLP technologies in evidence-based medicine.

## Background & Motivation

**Core Position of Evidence-Based Medicine**: EBM is the cornerstone of modern healthcare, guiding clinical decisions by integrating the best scientific evidence, clinical expertise, and patient values to serve clinicians, patients, guideline developers, and policymakers.

**Literature Explosion and Manual Bottlenecks**: The volume of medical literature is massive and rapidly increasing. The cost of manually performing literature screening, data extraction, quality appraisal, and evidence synthesis is extremely high; a systematic review often takes months or even years, urgently requiring assistance from automated NLP tools.

**Driven by NLP Technology Evolution**: From early statistical machine learning and RNNs/LSTMs to pre-trained Transformer models (BioBERT, PubMedBERT), and recently to large language models like GPT-4, the capacity of NLP in medical text processing has continuously risen. However, existing reviews fail to comprehensively reflect the systematic impact of this technological evolution on EBM.

**The Double-Edged Sword of the LLM Era**: Large language models exhibit powerful capabilities in complex tasks (such as evidence appraisal and synthesis, clinical QA, and patient-trial matching). However, issues like hallucination generation, lack of source attribution, and sensitivity to prompts limit their reliable application in high-risk clinical settings.

**Lack of a Unified Framework**: Existing reviews are typically organized by NLP technology types or disease areas, lacking a systematic framework that uses the EBM practice workflow as the main line to map cross-tasks uniformly. This makes it difficult for clinical practitioners to find the best tools for specific EBM steps.

**Benchmark Dataset Gaps**: Critical areas such as evidence synthesis, appraisal, and medical QA lack dedicated benchmark datasets. Existing summarization datasets (e.g., CNN-DailyMail) target open domains rather than the medical domain, severely constraining the comparability and progress of methods.

## Method

### Literature Search and Screening Process

Following the PRISMA guidelines, literature published from 2019 to 2024 was retrieved from four databases: PubMed, IEEE Xplore, ACM Digital Library, and ACL Anthology. The search strategy used a Boolean AND combination of NLP domain keywords (natural language processing, language model, LLM, information extraction, etc.) and EBM domain keywords (Evidence-Based Medicine, Clinical Trial, etc.). The screening process was as follows:

| Stage | Operation | Number of Papers |
|------|------|----------|
| Initial Search | Database Search + Additional Sources | 601 + 9 = 610 papers |
| Deduplication | Duplicate papers removed | -8 papers |
| Title/Abstract Screening | Screening based on exclusion criteria | -386 papers |
| Full-Text Screening | Mismatched targets or unrelated to EBM | -88 papers |
| Final Inclusion | Met inclusion criteria | **129 papers** |

Inclusion criteria: Published in English, clear application of NLP techniques to EBM, and focused on human applications. Exclusion criteria: Unrelated to NLP+EBM, non-English, secondary literature such as systematic reviews or case reports, or descriptive papers without experimental results. Two annotators performed cross-validation, with discrepancies resolved by a third person.

### EBM Five-Step Process and NLP Task Mapping

The core contribution of this paper is the systematic mapping of NLP tasks to the five-step practice workflow of EBM (the 5A framework). The corresponding NLP tasks, typical methods, and representative systems for each step are as follows:

| EBM Step | Core Function | Corresponding NLP Task | Representative Methods / Systems |
|----------|----------|--------------|----------------|
| **Ask** | Searching and screening studies | Information Retrieval | BioBERT embedding retrieval, Bing AI/ChatGPT-assisted search |
| **Acquire** | Collecting and structuring data | Entity Extraction (PICO), Relation Extraction | SciBERT/PubMedBERT entity recognition, Knowledge graph relation encoding |
| **Appraise** | Appraising evidence quality | Quality appraisal, evidence ranking and screening, evidence synthesis, evidence summarization | BERT fine-tuning for quality classification, BM25+BERT re-ranking, RAG+GPT-4 |
| **Apply** | Clinical application | Clinical trial design and matching, specialty applications | AutoTrial, TrialGPT, Watson Oncology |
| **Assess** | Evaluation and improvement | Question Answering, drug repurposing | BioBERT/BioGPT QA, knowledge graph drug prediction |

### Ask Stage: Evolution of Information Retrieval Technology

The goal of this stage is to retrieve studies relevant to clinical questions from massive literature databases. The technical route has evolved through three generations:

- **First Generation (Keyword Matching)**: Based on structured keyword queries and rule-based filters (e.g., SR[pt], CQrs) on MEDLINE/PubMed; precise but costly to maintain and domain-sensitive.
- **Second Generation (Contextual Representation)**: Statistical ML and context-aware models improved text representation but lacked scalability. Lokker et al. (2023) used BioBERT embeddings and attention mechanisms to enhance the quality of biomedical literature retrieval.
- **Third Generation (Generative AI)**: Gwon et al. (2024) compared the acceleration effects of Bing AI and ChatGPT in systematic literature searches, finding that both can speed up the search workflow but suffer from hallucination issues.

### Acquire Stage: Entity Extraction and Relation Extraction

**PICO Entity Extraction** is the core task of EBM data acquisition:

- **Rule-Based Methods**: Utilizing predefined lexicons, syntax, and contextual rules to extract entities in clinical trial data; suitable for high-precision structured scenarios, but struggle with complex or ambiguous data.
- **RNN/LSTM**: Used for sequential sentence classification, improving context utilization in unstructured abstracts, but lacking long-range dependency capabilities.
- **Transformer-Dominated**: SciBERT and PubMedBERT are specifically pre-trained for Intervention extraction; SrBERT is used for literature inclusion/exclusion classification based on predefined criteria.

**Relation Extraction** associates identified PICO elements: evolving from rules and traditional ML $\rightarrow$ deep learning (BERT + Argument Mining to identify support/contradiction relations) $\rightarrow$ schema-based relation extraction (guided by C-TrO ontology) $\rightarrow$ knowledge graphs (nodes represent entities, edges represent relations, such as Pan et al. 2021 for organizing COVID-19 clinical entities).

### Appraise Stage: Quality Appraisal, Ranking, and Synthesis

**Quality Appraisal**: Brassey et al. (2021) developed a fully automated tool combining ML and rule-based methods to appraise evidence quality for RCTs and systematic reviews through sentiment analysis, risk-of-bias indicators, and sample size calculations. BioBERT, BlueBERT, and $\text{BERT}_{\text{BASE}}$ were fine-tuned for literature classification based on methodological quality standards.

**Evidence Ranking and Screening**: Norman et al. (2019b) developed a prioritization approach, enabling technology-assisted screening to significantly reduce the quantity of literature needing screening. The A2A platform employs BM25 (scoring based on term frequency and document length) and DFR (divergence from randomness-based information quantification) for document ranking. SciBERT/BioBERT/BlueBERT re-ranking further enhances robustness.

**Evidence Synthesis**: Mutinda et al. (2022b) automatically extracted and normalized PICO elements to replicate meta-analyses of breast cancer RCTs (calculating risk ratios and generating forest plots), though only binary outcomes were supported. EvidenceMap enhances the explainability and retrievability of evidence through a three-layer entity-proposition-map structure.

**Evidence Summarization** is categorized into extractive (TextRank outperforms LexRank and LSA; KeBioSum integrates PICO knowledge into PLMs to reduce computational costs) and generative methods (T5, BART, PEGASUS; TrialsSummarizer supports interactive user corrections; RAG + GPT-4 answers clinical trial eligibility questions; TriSum distills structured rationales into smaller models).

### Apply & Assess Stage: Clinical Application and Continuous Evaluation

**Specialty Applications**: Watson Oncology (WOLI) automatically identifies and extracts relevant oncology research; the CTM system screens lung cancer patients at an Australian cancer center with an overall accuracy of 92%. In cardiology, a hybrid model achieved an eligibility assessment accuracy of 87.3% on 40,000 patients.

**Clinical Trial Matching** technological path: Hybrid methods (pattern matching + knowledge-driven + feature weighting) $\rightarrow$ Deep Learning (CNN/RNN multi-label classification) $\rightarrow$ C2Q converting free-text criteria to structured queries $\rightarrow$ LLM enhancement (AutoTrial utilizes multi-step reasoning to generate trial eligibility criteria; TrialGPT implements large-scale patient-trial matching).

**Drug Repurposing**: The CovidX network algorithm utilizes NLP to analyze COVID-19 literature for drug candidate ranking; knowledge graph embeddings are used to predict drug candidates for Alzheimer's disease.

## Key Findings

### Performance Comparison of Representative Systems

| System/Method | Task | Disease/Scenario | Key Performance Metrics |
|-----------|------|-----------|-------------|
| CTM (Alexander 2020) | Clinical trial matching | Lung cancer | Overall accuracy 92% |
| Hybrid NLP+Rules (Tun 2023) | Patient eligibility assessment | Heart failure/Atrial fibrillation | Accuracy 87.3% (40,000 patients) |
| BioBERT (Lokker 2023) | High-quality literature identification | General | Deep learning significantly outperforms traditional methods |
| BM25+BERT re-ranking (Rybinski 2020b) | Clinical trial search | General | Robustness significantly improved |
| TrialGPT (Jin 2024) | Large-scale patient-trial matching | General | Real-world deployment, significant time savings |
| AutoTrial (Wang 2023b) | Trial eligibility criteria generation | General | Multi-step reasoning + hybrid prompting |
| TextRank (Gulden 2019) | Extractive summarization | General | Outperforms LexRank and LSA |
| TriSum (Jiang 2024) | Generative summarization | General | Structured rationale distillation + curriculum learning |
| RAG+GPT-4 (Unlu 2024) | Clinical trial eligibility QA | Heart failure | Based on Retrieval-Augmented Generation |

### Current State of Benchmark Datasets

| Dataset | Scale | Annotation Type | Main Task |
|--------|------|----------|----------|
| EBM-NLP (Nye 2018) | 4,993 abstracts | Manual annotation of P, I, O | PICO extraction |
| Chia (Kury 2020) | 1,000 trials / 12,409 criteria | Manual annotation of 12 classes | Eligibility criteria analysis |
| MS^2 (DeYoung 2021) | 470K documents / 20K summaries | Manual annotation of multiple classes | Multi-document summarization |
| PICO-Corpus (Mutinda 2022a) | 1,011 breast cancer RCTs | Manual annotation of P, I, C, O | PICO extraction |
| NICTA-PIBOSO (Kim 2024) | 1,000 abstracts | P, I, O + background/design | Sentence classification |
| Trialstreamer (Marshall 2020) | 191 RCTs | P, I, O + RCT classifier | Evidence streaming |
| LCT (Dobbins 2022) | 1,000+ eligibility statements | Manual annotation of 6 classes | Clinical trial parsing |

**Key Gaps**: There is a severe lack of dedicated benchmarks in the fields of evidence synthesis & appraisal and medical QA. Existing summarization datasets (e.g., CNN-DailyMail) target the general open domain rather than medical content.

## Highlights & Insights

1. **EBM 5A Framework Mapping**: For the first time, an NLP review is organized following the five-step EBM workflow as its main structure, providing a complete technical landscape from posing clinical questions to evaluating practice outcomes. This allows clinical practitioners to rapidly locate the optimal tools for each stage.
2. **Four-Generation Technology Evolution Pathway**: Clear presentation of the iterative technology pathway from Rule-Based Methods $\rightarrow$ Statistical ML $\rightarrow$ RNN/LSTM $\rightarrow$ Transformers $\rightarrow$ LLMs, rendering the advantages, limitations, and representative works of each generation clear at a glance.
3. **Open-Source Repository**: Broad dissemination of the awesome-nlp-in-ebm repository, including paper lists, dataset indexes, and detailed metadata, which supplies a convenient entry point for subsequent research.
4. **Systematic Metadata of 129 Papers**: The appendix provides complete annotations of model types, disease areas, and task classifications for each paper, constructing a comprehensive domain knowledge base.

## Limitations & Future Work

1. **Language Bias**: Only English papers are included, which may omit important studies in other languages (especially Chinese and Japanese).
2. **Time Window Limitation**: Covers only 2019–2024, overlooking seminal earlier works.
3. **Incomplete Database Coverage**: Only four databases were searched, potentially missing relevant studies from other platforms.
4. **Nature of Literature Reviews**: Direct empirical comparison of different methods under unified conditions is unavailable.
5. **Uneven Specialty Coverage**: NLP applications in specialties such as urology and hepatology remain insufficiently covered.
6. **Time Lag in LLM Research**: Given the publication cycles, the review may not fully capture the latest developments despite the explosion of LLM literature in 2023–2024.

## Related Work & Insights

- **Differentiation from Other Reviews**: Most existing reviews organize their findings around NLP technology types (e.g., NER, QA) or clinical disease areas (e.g., oncology). This paper innovatively uses the EBM 5A practice workflow as its spine, aligning closer with the realistic scenarios of clinical workers.
- **Future Research Directions**: (1) Integrating real-world data (mobile devices, social media, genomics) to capture health indicators beyond traditional clinical notes; (2) Few-shot learning to address data scarcity in rare diseases; (3) Constructing dedicated benchmark datasets for evidence synthesis and appraisal; (4) Enhancing the explainability and cross-population equity of NLP models; (5) Developing a comprehensive, adaptive framework capable of reasoning across diverse clinical questions and heterogeneous data sources.
- **Insights for NLP Research**: EBM serves as a natural integrative application scenario for NLP technologies, covering the full stack of NLP abilities from information extraction, information retrieval, and QA to text generation and reasoning, functioning as a comprehensive testbed for evaluating general NLP systems.

## Rating

- Novelty: ⭐⭐⭐ (Survey paper with organizational innovation via the 5A framework map but no methodological innovation)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Systematic inclusion of 129 papers, strict PRISMA workflow, search across 4 databases, and dual-annotator verification)
- Writing Quality: ⭐⭐⭐⭐⭐ (The 5A framework is clearly organized and comprehensive, with rich appendix metadata and high-quality figures/tables)
- Value: ⭐⭐⭐⭐ (Provides the most comprehensive landscape index and resource gateway for the interdisciplinary field of NLP and EBM)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Internal and External Impacts of Natural Language Processing Papers](internal_and_external_impacts_of_natural_language_processing_papers.md)
- [\[ACL 2025\] QualiSpeech: A Speech Quality Assessment Dataset with Natural Language Reasoning](qualispeech_a_speech_quality_assessment_dataset_with_natural_language_reasoning_.md)
- [\[ACL 2025\] Cooperating and Competing Through Natural Language](cooperating_and_competing_through_natural_language.md)
- [\[ACL 2025\] A Survey of LLM-based Agents in Medicine: How Far Are We from Baymax?](a_survey_of_llm-based_agents_in_medicine_how_far_are_we_from_baymax.md)
- [\[ACL 2025\] LLM×MapReduce: Simplified Long-Sequence Processing using Large Language Models](llm_mapreduce_simplified_long_sequence_processing.md)

</div>

<!-- RELATED:END -->
