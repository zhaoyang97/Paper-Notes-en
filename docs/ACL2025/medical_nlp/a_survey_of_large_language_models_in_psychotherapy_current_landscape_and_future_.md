---
title: >-
  [Paper Note] A Survey of Large Language Models in Psychotherapy: Current Landscape and Future Directions
description: >-
  [ACL 2025][Medical LLM][LLM Psychotherapy] The first survey to systematically organize and review LLM research in psychotherapy using the APA three-stage (Assessment $\to$ Diagnosis $\to$ Treatment) conceptual taxonomy. Covering over 60 works, it comprehensively analyzes four levels from symptom detection to virtual therapists, revealing a four-fold imbalance across disorder coverage, language bias, methodology fragmentation, and theoretical integration.
tags:
  - "ACL 2025"
  - "Medical LLM"
  - "LLM Psychotherapy"
  - "Assessment-Diagnosis-Treatment Taxonomy"
  - "Mental Health"
  - "Dynamic Interaction"
  - "Multi-Stage Modeling"
date: 2026-05-08
content_hash: b431a9d71ec23deb
---

# A Survey of Large Language Models in Psychotherapy: Current Landscape and Future Directions

**Conference**: ACL 2025  
**arXiv**: [2502.11095](https://arxiv.org/abs/2502.11095)  
**Code**: None  
**Area**: LLM / Medical NLP / Psychotherapy  
**Keywords**: LLM Psychotherapy, Assessment-Diagnosis-Treatment Taxonomy, Mental Health, Dynamic Interaction, Multi-Stage Modeling

## TL;DR

The first survey to systematically organize and review LLM research in psychotherapy using the APA three-stage (Assessment $\to$ Diagnosis $\to$ Treatment) conceptual taxonomy. Covering over 60 works, it comprehensively analyzes four levels from symptom detection to virtual therapists, revealing a four-fold imbalance across disorder coverage, language bias, methodology fragmentation, and theoretical integration.

## Background & Motivation

**Background**: Mental health is increasingly crucial in modern healthcare systems. The high prevalence of common mental disorders, such as depression and anxiety, has generated a massive demand for accessible and effective psychological treatments. The core of psychotherapy lies in dynamic, context-sensitive interpersonal interactions, where therapists must continuously adapt intervention strategies based on patients' emotional fluctuations, verbal expressions, and social contexts to build a therapeutic alliance for symptom relief.

**Limitations of Prior Work**: Traditional NLP approaches are generally confined to static or single-task settings, struggling to capture the deep and flexible interactive nature of psychotherapy. Existing LLM research is highly fragmented: some studies employ LLMs as isolated feature extraction tools for single tasks like depression detection or diagnosis, while others develop mental health conversational agents that are limited to auxiliary support, lacking integration with complete clinical workflows.

**Key Challenge**: LLMs possess long-context modeling and multi-turn reasoning abilities, granting them the potential to transcend the "discrete label recognition" paradigm and perform continuous clinical reasoning from assessment to intervention. However, constrained by traditional fragmented paradigms, their clinical potential across the entire continuum remains unleased.

**Goal**: A systematic framework is lacking to organize and analyze fragmented research on LLMs applied to psychotherapy.

**Key Insight**: Grounded in the American Psychological Association (APA) tripartite model of litigation, this work represents the first attempt to construct a conceptual taxonomy that partitions psychotherapy into three interconnected dimensions: assessment, diagnosis, and treatment.

**Core Idea**: To utilize the APA three-stage taxonomy to systematically organize fragmented LLM psychotherapy research, highlighting multi-dimensional research imbalances and pointing toward future research directions in continuous multi-stage modeling.

## Method

### Overall Architecture

This paper proposes a Conceptual Taxonomy based on the APA three-stage model, which organizes psychotherapy into three core components—Assessment, Diagnosis, and Treatment—and defines their dynamic interactive relationships:

- **Synthesizing**: Assessment $\to$ Diagnosis, dialectically integrating observational data with nosological frameworks to synthesize symptoms and behavioral patterns into diagnostic formulations.
- **Framing**: Diagnosis $\to$ Treatment, where diagnosis integrates diverse symptoms into coherent categories to establish a treatment blueprint.
- **Customization**: Assessment $\to$ Treatment, refining treatment plans based on ongoing assessments to account for individual differences without being rigidly bound by diagnostic labels.

The scope covers LLM studies from the emergence of ChatGPT (late 2022) to October 2024, focusing primarily on models with 7B+ parameters.

### Key Designs

1. **Four-Dimensional Classification System in the Assessment Stage**:

    - **Function**: Systematically organize LLM applications within psychological assessment: symptom detection, severity measurement, cognitive analysis, and behavioral analysis.
    - **Mechanism**: Literature is categorized by intersecting text granularity (single post / multi-turn conversation / user post collection) with assessment focus. Symptom detection is the most densely studied area (~30 papers). Technical approaches include Fine-Tuning (such as Bao 2024 for interpretable depression detection), Few-Shot (e.g., Tu 2024 using GPT-4 to automate PTSD assessment on 411 interviews, achieving 70.8% zero-shot accuracy), and Multi-Agent Debate (Lim 2024, which synthesizes opinions from multiple LLMs to improve accuracy and specificity in cognitive distortion classification).
    - **Design Motivation**: Assessment is the cornerstone of psychotherapy, but existing research heavily skews toward symptom detection, leaving cognitive and behavioral dimensions significantly under-researched.

2. **Four LLM Roles in the Treatment Stage**:

    - **Function**: Classify treatment research into four categories based on the role of the LLM: virtual therapists, auxiliary tools, simulated patients (for clinical education), and assessment quality analysis.
    - **Mechanism**: **Virtual therapists** represent the most ambitious yet high-risk category, where LLMs directly act as therapeutic conversational agents interacting with patients (Lee 2024b/c/d). **Auxiliary tools** are the most mature, assisting therapists in reframing cognitive distortions (Maddela 2023) and generating empathetic responses (Sharma 2023). **Simulated patients** are utilized to train students in clinical education (Chaszczewicz 2024). **Evaluation tools** automate therapeutic quality scoring (Lee 2024e, Na 2024).
    - **Design Motivation**: The safety requirements and technological maturity vary greatly across roles, necessitating separate discussions.

### Loss & Training

As a survey paper, this work does not involve model training. The summarized technical pathways include: predominant use of Fine-Tuning and Few-Shot Prompting in the assessment stage; static diagnosis (making one-time judgments based on complete conversations, such as the DORIS system integrating text embeddings with LLMs) versus dynamic diagnosis (real-time interactive assessment, such as WundtGPT integrating empathy and active guidance, and AMC self-improving conversational agents) in the diagnosis stage; and a combination of role prompting and psychology theory-driven dialogue strategies in the treatment stage.

## Key Experimental Results

### Main Results

| Dimension | Key Findings | Data |
|------|---------|------|
| Stage Distribution | Assessment is the most densely researched stage | ~30 papers (symptom detection dominates) |
| Disorder Coverage | Heavily skewed towards common disorders | Depression and anxiety account for the vast majority |
| Language Resources | Dominated by English | Only a few studies in Chinese, Korean, or Russian |
| Theory Integration | Insufficient integration of psychological theories | Most works lack an explicit theoretical foundation |

### Ablation Study

| Task | Technology | Representative Work | Performance |
|------|------|---------|------|
| Symptom detection | Zero-shot (GPT-4 Turbo) | So 2024 | 70.8% accuracy in Korean interviews |
| Symptom detection | Fine-tuning (GPT-3.5) | So 2024 | Multilabel classification 0.817 accuracy |
| Severity assessment | Zero-shot | Med-PaLM 2 | Clinician-level performance in depression rating |
| Cognitive analysis | Multi-Agent Debate | Lim 2024 | Significant gains in accuracy and specificity |
| Diagnosis | Fine-tuned small model | RoBERTa (Schirmer 2024) | Superiority over GPT-4 in cross-domain PTSD analysis |
| Suicidal risk | Fine-tuning (CHATGLM2) | PsyGUARD (Qiu 2024) | SOTA in Chinese suicide risk assessment |

### Key Findings

- **Stage Imbalance**: Papers on assessment vastly outnumber those on diagnosis and treatment. Within assessment, symptom detection is dominant, leaving cognitive/behavioral assessment highly under-researched.
- **Language Bias**: The vast majority of studies are conducted in English, with limited cross-lingual coverage. Translation-based methods introduce cultural biases.
- **Methodological Fragmentation**: Each stage is researched independently, lacking continuous multi-stage modeling from Assessment $\to$ Diagnosis $\to$ Treatment.
- **Specialized Small Models vs. Generalist LLMs**: Fine-tuned RoBERTa models still outperform zero-shot GPT-4 in cross-domain PTSD analysis, highlighting the balance between specialization and generalizability.
- **Potential of Dynamic Diagnosis**: LLMs' multi-turn reasoning capabilities are far more suited for real-time interactive dynamic diagnosis (e.g., WundtGPT) than traditional, one-off static diagnosis.

## Highlights & Insights

- **First Unified Framework**: The APA three-stage conceptual taxonomy represents the first systematic categorization in the field of LLM-based psychotherapy, map-guiding fragmented research.
- **Insightful Dynamic Relationships**: The identification of three dynamic interactive relationships—Synthesizing, Framing, and Customization—reveals that psychotherapy is a continuous, adaptive system rather than three isolated tasks.
- **Discovered "Theory-Technology Decoupling"**: Most LLM works lack psychological foundations (e.g., CBT, MI) while clinical efficacy is heavily dependent on theory-driven intervention design.
- **Distinction between Dynamic and Static Diagnosis**: Distinguishing dynamic from static diagnosis helps locate the true advantage of LLMs in multi-turn reasoning rather than label classification.

## Limitations & Future Work

- Temporal coverage only extends to October 2024, missing rapid advancements from late 2024 and 2025.
- Research is mostly drawn from computational linguistics venues, potentially overlooking LLM studies in clinical psychology and psychiatry journals.
- Lack of systematic quantitative benchmarks comparing different methodologies on unified datasets.
- Safety risks associated with virtual therapists (e.g., inappropriate dependence, incorrect interventions leading to deterioration) are not discussed in sufficient depth.
- Excludes AI systems in psychotherapy that integrate multimodal signals like vocal affect and facial expressions.

## Related Work & Insights

- Prior surveys (e.g., Malgaroli 2023) cover general NLP interventions but do not focus on LLMs. Scoping reviews like Hua 2024/2025 target general mental health rather than clinician-level psychotherapy. Lawrence 2024 explores risks and opportunities without offering a systematic taxonomy.
- The tripartite taxonomy from this work can inspire medical AI surveys in other domains requiring an entire Assessment $\to$ Diagnosis $\to$ Treatment pipeline (e.g., oncology, pain management).
- **Insights**: (1) The next frontier should lie in continuous, multi-stage modeling rather than pipeline-isolated optimization; (2) Future designs for LLM interventions must embed mature clinical theories like CBT and MI; (3) Cross-lingual and cross-cultural adaptation remain critical bottlenecks for real-world deployment.

## Rating

⭐⭐⭐ Serves as an invaluable and organizing reference by introducing the first systematic taxonomy for LLMs in psychotherapy. However, it lacks unified quantitative comparisons, features restricted temporal coverage, and lacks indepth discussions on ethical concerns and safety issues with virtual therapists.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Towards Omni-RAG: Comprehensive Retrieval-Augmented Generation for Large Language Models in Medical Applications](omni_rag_medical.md)
- [\[ACL 2025\] MedBioRAG: Semantic Search and Retrieval-Augmented Generation with Large Language Models for Medical and Biological QA](medbiorag_semantic_search_and_retrieval-augmented_generation_with_large_language.md)
- [\[NeurIPS 2025\] Position: Thematic Analysis of Unstructured Clinical Transcripts with Large Language Models](../../NeurIPS2025/medical_nlp/position_thematic_analysis_of_unstructured_clinical_transcripts_with_large_langu.md)
- [\[NeurIPS 2025\] Large Language Models as Medical Codes Selectors: A Benchmark Using the International Classification of Primary Care](../../NeurIPS2025/medical_nlp/large_language_models_as_medical_codes_selectors_a_benchmark_using_the_internati.md)
- [\[ACL 2026\] MHGraphBench: Knowledge Graph-Grounded Benchmarking of Mental Health Knowledge in Large Language Models](../../ACL2026/medical_nlp/mhgraphbench_knowledge_graph-grounded_benchmarking_of_mental_health_knowledge_in.md)

</div>

<!-- RELATED:END -->
