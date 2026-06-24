---
title: >-
  [Paper Note] MTBBench: A Multimodal Sequential Clinical Decision-Making Benchmark in Oncology
description: >-
  [NeurIPS 2025][Medical Imaging][Multimodal Benchmark] This paper introduces MTBBench—the first clinical benchmark simultaneously covering three dimensions: multimodality, longitudinal temporal sequencing, and interactive agent workflows. It simulates the decision-making process of Molecular Tumor Boards (MTBs) to evaluate and enhance the multimodal longitudinal reasoning capabilities of AI agents in precision oncology.
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "Multimodal Benchmark"
  - "Oncology"
  - "Molecular Tumor Board"
  - "Longitudinal Reasoning"
  - "Clinical Decision-Making Agent"
date: 2026-05-08
content_hash: 0edc234e8d6e169d
---

# MTBBench: A Multimodal Sequential Clinical Decision-Making Benchmark in Oncology

**Conference**: NeurIPS 2025
**arXiv**: [2511.20490](https://arxiv.org/abs/2511.20490)  
**Code**: [GitHub](https://github.com/bunnelab/MTBBench) / [HuggingFace](https://huggingface.co/datasets/EeshaanJain/MTBBench)  
**Area**: Medical Imaging
**Keywords**: Multimodal Benchmark, Oncology, Molecular Tumor Board, Longitudinal Reasoning, Clinical Decision-Making Agent

## TL;DR

This paper introduces MTBBench—the first clinical benchmark simultaneously covering three dimensions: multimodality, longitudinal temporal sequencing, and interactive agent workflows. It simulates the decision-making process of Molecular Tumor Boards (MTBs) to evaluate and enhance the multimodal longitudinal reasoning capabilities of AI agents in precision oncology.

## Background & Motivation

Multimodal large language models have demonstrated promising performance in biomedical reasoning; however, existing evaluation benchmarks are severely misaligned with real clinical workflows. Current evaluations predominantly focus on single-modality, decontextualized static question answering, neglecting multi-expert, multi-round decision-making environments such as the Molecular Tumor Board (MTB). The MTB represents a canonical clinical decision scenario in oncology, wherein oncologists, radiologists, pathologists, and geneticists collaboratively analyze evolving patient cases by integrating multimodal data—including H&E staining, immunohistochemistry (IHC), hematology, and genomics—and render decisions at multiple time points.

Existing benchmarks such as MedAgentBench and MediQ address only one aspect of interactivity or longitudinality, and are typically restricted to a single modality (e.g., text-based EHRs). The primary gaps are: (1) the absence of a benchmark that simultaneously evaluates multimodal understanding, longitudinal reasoning, and agent interaction; and (2) untested agent cognition when facing partial data, sequential updates, conflicting information, and high-stakes outcomes.

## Method

### Overall Architecture

MTBBench comprises two tracks—MTBBench-Multimodal and MTBBench-Longitudinal—alongside an agent framework that treats foundation models as callable tools. The agent processes temporally evolving patient data across multi-turn dialogues, selectively requesting and reasoning over documents from different modalities to answer clinical questions.

### Key Designs

1. **MTBBench-Multimodal**: Twenty-six head-and-neck cancer patients are curated from the HANCOCK dataset, with an average of 40 modality documents per case (~1.2 H&E slides, ~26.2 IHC images, and 1 hematology report), yielding 390 multimodal question-answer pairs (15 per patient). Tasks are organized into three stages: pathological image interpretation (H&E/IHC tissue subtyping and spatial distribution of immune infiltration) → hematological reasoning (preoperative biochemical analysis of infection risk, bleeding tendency, etc.) → postoperative prognostic integration (5-year survival and 2-year recurrence prediction).

2. **MTBBench-Longitudinal**: Forty patients are curated from the MSK-CHORD dataset, with approximately 5 associated documents per case (copy number variations, somatic mutations, pathology reports, clinical timelines), from which 183 longitudinal question-answer pairs are manually constructed. The agent must answer questions on diagnostic trajectories, survival prediction, recurrence prediction, and treatment progression mapping at temporally segmented decision nodes; genomic data are introduced at key stages to support drug resistance pattern analysis.

3. **Agent Workflow and Foundation Model Tools**: At each turn $t$, the agent receives a clinical query $q_t$ and a set of modality documents $\mathcal{F}_t$, and may request a subset $\mathcal{R}_t \subseteq \mathcal{F}_t$ to retrieve information. Documents do not persist automatically across turns and must be actively re-requested. The tool suite includes:

    - **CONCH** (H&E vision-language model): returns the best-matching description based on image-text embedding similarity
    - **UNI2 + ABMIL** (IHC quantification tool): foundation model embeddings combined with attention-based multiple instance learning to regress positive staining proportions
    - **PubMed retrieval**: natural language query → BAAI-bge reranking → top-3 abstracts returned
    - **DrugBank**: automatically links drug names to therapeutic indications, mechanisms of action, and drug interaction information

### Loss & Training

- Expert validation pipeline: a web-based companion application was developed to allow clinicians to review clinical context, browse H&E/IHC images, and annotate feedback for each Q&A pair.
- IHC quantification tool training: 256×256 patches → UNI2 encoding into 1536-dimensional features → ABMIL regression of positive staining percentage; training data annotated manually via QuPath.
- Evaluation metric: 95% confidence intervals estimated via 1,000 bootstrap resamples.

## Key Experimental Results

### Main Results: Multimodal Track (Without Tools)

| Model | Digital Pathology | Hematology | Prognosis & Recurrence | Overall |
|-------|------------------|------------|------------------------|---------|
| gpt4o | 63.2±6.0 | 76.9±7.7 | 59.9±13.5 | 66.7±8.1 |
| internvl3-78b | 62.0±6.4 | 79.7±7.7 | **65.6±11.5** | **69.1±8.4** |
| qwen25-7b | 42.3±6.2 | 61.1±9.1 | 53.9±12.5 | 52.4±9.0 |
| llama90b | 54.6±6.2 | **82.8±7.2** | 51.7±13.5 | 63.0±14.8 |
| o4-mini | 59.5±6.4 | 77.8±8.2 | 55.7±14.4 | 64.3±10.5 |

### Ablation Study: Effect of Tool Augmentation

| Configuration | Key Metric Gain | Notes |
|---------------|----------------|-------|
| Multimodal + vision tools (CONCH/UNI) | Digital pathology accuracy ↑ up to 9% | FM tools significantly improve pathological image understanding |
| Longitudinal + knowledge tools (PubMed/DrugBank) | Progression/recurrence prediction ↑ >5% | External knowledge enhances temporal reasoning |
| Document access count vs. accuracy | Positive correlation (both tracks) | Information retrieval capacity more critical than model scale |
| Model scale vs. accuracy | No consistent positive correlation | gemma-3-12b outperforms 27b on certain tasks |

### Key Results: Longitudinal Track

| Model | Prognosis | Progression | Recurrence | Overall |
|-------|-----------|-------------|------------|---------|
| qwen3-32b | **83.0±9.2** | 63.3±12.3 | 54.6±13.6 | **67.0±13.5** |
| llama33-70b | 73.2±9.9 | **68.2±13.2** | **56.7±13.6** | 66.0±7.8 |
| gpt4o | 72.9±10.6 | 64.8±13.2 | 54.8±13.6 | 64.2±8.6 |

### Key Findings

- internvl3-78b achieves the best overall performance (69.1%), surpassing the closed-source gpt4o (66.7%) by 2.5 percentage points.
- Model scale is not the determining factor—the positive correlation between document access count and accuracy is substantially stronger.
- Prognosis and recurrence prediction remain highly challenging for all models, with accuracy approaching random chance (~50%).
- Tool augmentation yields improvements across all tasks, with pathology tasks benefiting most.
- In the longitudinal track, coarse-grained survival signals can be detected, but fine-grained temporal reasoning (progression/recurrence) remains difficult.

## Highlights & Insights

- MTBBench is the first clinical AI benchmark to jointly evaluate multimodality, longitudinality, and agent interaction within a single framework.
- The design of treating domain-specific foundation models (CONCH, UNI2) as agent-callable tools is novel, faithfully simulating the clinical process of expert consultation with specialized systems.
- The benchmark reveals an important finding: information retrieval capability outweighs model scale—effective retrieval and integration of information matters more than parameter count.
- The expert validation pipeline (web application) substantially enhances the clinical credibility of the benchmark.

## Limitations & Future Work

- MTBBench remains an offline controlled benchmark; agents have not been tested in real interactive clinical workflows.
- Prognosis and recurrence prediction performance is low, indicating that all current models are insufficiently capable for such high-level reasoning tasks.
- The longitudinal track lacks dedicated longitudinal reasoning foundation models and primarily relies on general-purpose knowledge tools.
- Data scale is limited (26 + 40 patients) and could be extended to additional cancer types.
- The IHC quantification tool relies on manual QuPath annotations for training, limiting scalability.

## Related Work & Insights

- Compared to MedAgentBench, MediQ, and MedJourney, MTBBench is more comprehensive across the three dimensions of multimodality, longitudinality, and interactivity.
- The design of foundation model tools within the agent framework is generalizable to other multidisciplinary clinical scenarios (e.g., cardiology MDTs, neurology imaging conferences).
- Implication: clinical AI evaluation should shift from static QA toward dynamic, process-oriented decision-making assessment.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The first clinical agent benchmark jointly covering three dimensions, with precise positioning.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Broad multi-model evaluation with tool/no-tool comparisons, though data scale is relatively small.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, professional description of clinical workflows, information-rich tables and figures.
- **Value**: ⭐⭐⭐⭐⭐ Fills an important gap in clinical AI evaluation with direct implications for advancing AI in precision oncology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Sequential Attention-based Sampling for Histopathological Analysis](sequential_attention-based_sampling_for_histopathological_analysis.md)
- [\[NeurIPS 2025\] SMMILE: An Expert-Driven Benchmark for Multimodal Medical In-Context Learning](smmile_an_expert-driven_benchmark_for_multimodal_medical_in-context_learning.md)
- [\[ICLR 2026\] MedLesionVQA: A Multimodal Benchmark Emulating Clinical Visual Diagnosis for Body Surface Health](../../ICLR2026/medical_imaging/medlesionvqa_a_multimodal_benchmark_emulating_clinical_visual_diagnosis_for_body.md)
- [\[CVPR 2026\] OmniBrainBench: A Comprehensive Multimodal Benchmark for Brain Imaging Analysis Across Multi-stage Clinical Tasks](../../CVPR2026/medical_imaging/omnibrainbench_a_comprehensive_multimodal_benchmark_for_brain_imaging_analysis_a.md)
- [\[NeurIPS 2025\] QoQ-Med: Building Multimodal Clinical Foundation Models with Domain-Aware GRPO Training](qoq-med_building_multimodal_clinical_foundation_models_with_domain-aware_grpo_tr.md)

</div>

<!-- RELATED:END -->
