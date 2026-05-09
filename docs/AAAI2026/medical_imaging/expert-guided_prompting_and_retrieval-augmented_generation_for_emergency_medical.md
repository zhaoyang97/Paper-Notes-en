---
title: >-
  [Paper Note] Expert-Guided Prompting and Retrieval-Augmented Generation for Emergency Medical Service Question Answering
description: >-
  [AAAI2026][Medical Imaging][RAG] This paper constructs EMSQA, the first multiple-choice QA dataset for the emergency medical services domain (24.3K questions, 10 clinical topics, 4 certification levels), and proposes the Expert-CoT and ExpertRAG frameworks to inject domain expertise into LLM reasoning and retrieval, achieving up to 4.59% accuracy improvement over standard RAG.
tags:
  - AAAI2026
  - Medical Imaging
  - RAG
  - chain-of-thought
  - EMS
  - domain expertise
  - MCQA
date: 2026-05-08
content_hash: 6e4a8a6c03787cf1
---

# Expert-Guided Prompting and Retrieval-Augmented Generation for Emergency Medical Service Question Answering

**Conference**: AAAI2026  
**arXiv**: [2511.10900](https://arxiv.org/abs/2511.10900)  
**Code**: [EMSQA](https://uva-dsa.github.io/EMSQA)  
**Area**: Medical Imaging  
**Keywords**: RAG, chain-of-thought, EMS, domain expertise, MCQA

## TL;DR

This paper constructs EMSQA, the first multiple-choice QA dataset for the emergency medical services domain (24.3K questions, 10 clinical topics, 4 certification levels), and proposes the Expert-CoT and ExpertRAG frameworks to inject domain expertise into LLM reasoning and retrieval, achieving up to 4.59% accuracy improvement over standard RAG.

## Background & Motivation

**Background**: LLMs have demonstrated strong performance on general medical QA benchmarks (MedQA, MedMCQA), with CoT reasoning and RAG-based retrieval being the two dominant approaches for further performance gains.

**Limitations of Prior Work**: Existing methods treat reasoning and retrieval as undifferentiated, general-purpose processes — models directly reason or retrieve upon seeing a question, without distinguishing the specific clinical domain (e.g., trauma, airway management, pharmacology) or certification level (EMR/EMT/AEMT/Paramedic) involved. In practice, however, medical professionals always first identify the subject area of a question before reasoning from the corresponding knowledge framework.

**Key Challenge**: The EMS domain lacks publicly available QA datasets and structured knowledge bases, and existing CoT/RAG methods provide no mechanism to leverage question-level expertise attributes (subject area + certification level) to guide reasoning and retrieval.

**Goal**: The paper constructs the EMSQA dataset and an accompanying knowledge base, trains a lightweight Filter classifier to infer expertise attributes of questions, and then injects these attributes into CoT prompts via Expert-CoT and uses them to filter the knowledge base for targeted retrieval via ExpertRAG. The core idea is to formalize the cognitive process of domain experts as explicit signals that can be injected into LLMs.

## Method

### Overall Architecture

A three-stage pipeline: (1) **Data and knowledge construction** — EMSQA dataset (24.3K MCQA) + subject-partitioned knowledge base (40K documents, 2M tokens) + 4M NEMSIS patient records; (2) **Filter classifier** — LoRA fine-tuned LLM to infer subject area and certification level for each question; (3) **Inference augmentation** — Expert-CoT injects attributes into prompt templates, ExpertRAG filters retrieval scope by attribute, and the two can be combined.

### Key Designs

1. **Filter Classifier (Expertise Attribute Inference)**:
    - Function: Automatically infers subject area (10 classes, multi-label) and certification level (4 classes, single-label) from input questions and answer options.
    - Mechanism: Appends a `<classify>` token to the LLM input, extracts the final-layer hidden state $h_i$, and passes it through two classification heads $W_{sub}$ and $W_{lvl}$ for respective predictions. The joint loss is $\mathcal{L} = w_{sub} \cdot \text{BCE}(p_i^{sub}, y_i^{sub}) + w_{lvl} \cdot \text{CE}(p_i^{lvl}, y_i^{lvl})$, with DWA dynamically adjusting the weights of the two tasks. At inference time, subject area predictions are binarized with a threshold of 0.5, and certification level is determined by argmax.
    - Design Motivation: A lightweight module is needed to quickly determine expertise attributes prior to inference. LoRA fine-tuning requires only a small number of additional parameters ($r=8$), and multi-task training allows the two attributes to mutually reinforce each other.

2. **Expert-CoT (Expertise-Guided Prompting)**:
    - Function: Embeds the Filter-predicted subject area $\hat{s}_i$ and certification level $\hat{l}_i$ into a CoT prompt template to guide LLM reasoning from a domain-specific perspective.
    - Mechanism: While standard CoT encourages "step-by-step" reasoning without specifying a starting point, Expert-CoT explicitly provides a domain-grounded starting point. The final answer is $\hat{A}_i = f^{\text{CoT-Expert}}(q_i, \mathcal{O}_i, \hat{l}_i, \hat{s}_i)$, with the template containing instructions such as "You are an expert in {subject area} at {certification level}."
    - Design Motivation: This approach simulates the cognitive process of real medical professionals — first locating the relevant subject area, then reasoning from the corresponding knowledge framework, rather than reasoning in an undifferentiated, general manner.

3. **ExpertRAG (Expertise-Guided Retrieval)**:
    - Function: Leverages Filter-predicted subject areas to filter the knowledge base and patient records, enabling domain-aligned retrieval-augmented generation.
    - Mechanism: Three strategies are proposed — Global (full-corpus retrieval, baseline), FTR (Filter-then-Retrieve: first filter KB/PR by $\hat{s}_i$, then retrieve top-$M$/$N$), and RTF (Retrieve-then-Filter: first retrieve $10\times$ candidates, then filter). The final answer is $\hat{A}_i = f^{\text{RAG}}(q_i, \mathcal{O}_i, \mathcal{R}(q_i, \hat{s}_i), \hat{l}_i, \hat{s}_i)$, using MedCPT as the retriever with top-32 from KB and top-8 from patient records.
    - Design Motivation: Global RAG retrieval over the full corpus introduces large quantities of irrelevant documents that dilute relevance. Partitioning retrieval by subject area effectively provides the retriever with domain priors, significantly increasing the proportion of relevant documents.

### Loss & Training

The Filter classifier uses AdamW optimizer (weight decay 0.01) with LoRA parameters $r=8, \alpha=16$, dropout 0.05, and sequence length 128. DWA temperature $T=2$ is used for dynamic multi-task weight adjustment. No additional training is performed at the LLM inference stage; performance improvements are achieved solely through prompt engineering and retrieval augmentation. Experiments are conducted on NVIDIA H200 GPUs.

## Key Experimental Results

### Main Results

| Model | Method | Public Acc | Public F1 | Private Acc | Private F1 |
|-------|--------|-----------|-----------|------------|------------|
| Qwen3-32B | 0-shot | 83.55 | 83.55 | 85.11 | 85.89 |
| Qwen3-32B | CoT | 84.96 | 84.97 | 88.78 | 90.13 |
| Qwen3-32B | Expert-CoT (Filter) | 85.57 | 85.60 | 89.50 | 91.20 |
| OpenAI-o3 | 0-shot | 92.39 | 92.39 | — | — |
| Qwen3-4B | Global RAG + CoT | 78.12 | 79.17 | 75.46 | 76.87 |
| Qwen3-4B | ExpertRAG-GT RTF + Expert-CoT | **82.24** | **82.26** | **80.51** | **81.16** |
| Qwen3-4B | ExpertRAG-Filter RTF + Expert-CoT | 80.95 | 80.96 | 79.47 | 80.22 |

### Ablation Study

| Configuration | Public Acc | Private Acc | Notes |
|---------------|-----------|------------|-------|
| Qwen3-4B 0-shot (no RAG) | 70.99 | 69.88 | Pure LLM baseline |
| + CoT | 72.35 | 70.58 | +1.36 |
| + Global RAG + CoT | 78.12 | 75.46 | Significant RAG gain |
| + Global RAG + Expert-CoT | 79.59 | 76.75 | Expert-CoT adds +1.47 |
| + ExpertRAG-GT FTR + Expert-CoT | 81.62 | 80.40 | Partitioned retrieval +3.50 vs. Global |
| + ExpertRAG-GT RTF + Expert-CoT | 82.24 | 80.51 | RTF marginally outperforms FTR |

### Key Findings

- Expert-CoT consistently outperforms vanilla CoT by 1–2%, with larger gains on weaker models (OpenBioLLM: +2.05%) and smaller but consistent gains on stronger models (Qwen3-32B: ~+0.6%).
- The combination of ExpertRAG + Expert-CoT achieves up to 4.59% accuracy improvement over standard RAG, with the two enhancements exhibiting additive effects.
- The Filter classifier achieves subject area miF ~80% and certification level miF ~66%; significant improvements are observed even in the presence of classification errors.
- Using ground-truth expertise outperforms Filter predictions by ~1.3%, indicating room for further improvement in the classifier.
- Qwen3-32B + Expert-CoT passes all four levels of the NREMT certification simulation exams.

## Highlights & Insights

- The paper formalizes the cognitive process of domain experts as a computationally injectable attribute signal — a conceptually straightforward yet systematically well-designed approach with strong potential for cross-domain transfer (e.g., law, finance, and other specialized fields can reuse the same framework).
- EMSQA is the first EMS QA dataset covering multiple certification levels with an accompanying structured knowledge base and real patient records, providing enduring value as a long-term benchmark.
- The RTF strategy (broad retrieval followed by attribute-based filtering) marginally outperforms FTR (filtering before retrieval), suggesting that retrieval-filter ordering has a subtle effect — retrieving first preserves the possibility of retaining cross-domain relevant documents.

## Limitations & Future Work

- The Filter classifier's accuracy remains limited (subject area miF ≈ 80%); misclassifications propagate to downstream modules, and uncertainty-aware soft filtering warrants exploration.
- RAG experiments are conducted only with Qwen3-4B; whether ExpertRAG yields similarly significant gains on larger models remains unverified.
- Portions of the dataset are sourced from subscription-based websites, with only a public subset released, limiting reproducibility.
- The knowledge base lacks certification level annotations, precluding more fine-grained certification-level-aligned retrieval.
- Evaluation is limited to the multiple-choice QA setting and has not been extended to open-ended question answering or clinical decision support.

## Related Work & Insights

- **vs. MedRAG**: MedRAG employs general medical corpora with hybrid sparse-dense retrieval (74.31%); ExpertRAG exceeds this by ~8% through domain-aligned KB retrieval, demonstrating the advantage of partitioned retrieval.
- **vs. i-MedRAG**: i-MedRAG improves performance through iterative query rewriting (77.96%), a strategy orthogonal to ExpertRAG's subject area filtering; the two approaches are potentially complementary.
- **vs. Self-BioRAG**: Self-BioRAG achieves only 55.71% on EMSQA, demonstrating that general biomedical RAG is ill-suited for specialized subdomains such as EMS.

## Rating

- Novelty: ⭐⭐⭐ The core idea of injecting expertise into CoT/RAG is intuitive, but the overall system construction is thorough.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model, multi-RAG-baseline comparisons, including real certification exam evaluations.
- Writing Quality: ⭐⭐⭐⭐ Figures and tables are clear; dataset construction process is described in detail.
- Value: ⭐⭐⭐⭐ The dataset offers sustained benchmark value; the expertise injection framework is generalizable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Q-FSRU: Quantum-Augmented Frequency-Spectral Fusion for Medical Visual Question Answering](q-fsru_quantum-augmented_frequency-spectral_fusion_for_medical_visual_question_a.md)
- [\[ACL 2026\] Query Pipeline Optimization for Cancer Patient Question Answering Systems](../../ACL2026/medical_imaging/query_pipeline_optimization_for_cancer_patient_question_answering_systems.md)
- [\[ACL 2026\] HypEHR: Hyperbolic Modeling of Electronic Health Records for Efficient Question Answering](../../ACL2026/medical_imaging/hypehr_hyperbolic_modeling_of_electronic_health_records_for_efficient_question_a.md)
- [\[ACL 2026\] RA-RRG: Multimodal Retrieval-Augmented Radiology Report Generation with Key Phrase Extraction](../../ACL2026/medical_imaging/ra-rrg_multimodal_retrieval-augmented_radiology_report_generation_with_key_phras.md)
- [\[AAAI 2026\] NutriScreener: Retrieval-Augmented Multi-Pose Graph Attention Network for Malnourishment Screening](nutriscreener_retrieval-augmented_multi-pose_graph_attention_network_for_malnour.md)

</div>

<!-- RELATED:END -->
