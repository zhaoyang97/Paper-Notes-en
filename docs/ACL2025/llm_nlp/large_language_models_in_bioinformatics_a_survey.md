---
title: >-
  [Paper Note] Large Language Models in Bioinformatics: A Survey
description: >-
  [ACL 2025][LLM (Other)][survey] This paper systematically reviews the progress of large language models in four major areas of bioinformatics (DNA/genomics, RNA, protein, and single-cell analysis), covering the architectures, tasks, and datasets of over 30 representative models, and discusses core challenges and future directions such as data scarcity, computational complexity, and cross-omic integration.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "survey"
  - "bioinformatics"
  - "DNA"
  - "RNA"
  - "protein"
  - "single-cell"
date: 2026-05-08
content_hash: 6b62f876c5e73473
---

# Large Language Models in Bioinformatics: A Survey

**Conference**: ACL 2025  
**arXiv**: [2503.04490](https://arxiv.org/abs/2503.04490)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: survey, bioinformatics, DNA, RNA, protein, single-cell

## TL;DR

This paper systematically reviews the progress of large language models in four major areas of bioinformatics (DNA/genomics, RNA, protein, and single-cell analysis), covering the architectures, tasks, and datasets of over 30 representative models, and discusses core challenges and future directions such as data scarcity, computational complexity, and cross-omic integration.

## Background & Motivation

**Background**: Large language models have made breakthrough progress in NLP, and researchers have begun applying LLMs to various tasks in bioinformatics, including DNA sequence function prediction, RNA structure prediction, protein function inference, and single-cell transcriptomic analysis. In recent years, the number of LLMs in bioinformatics has grown exponentially.

**Limitations of Prior Work**: Biological data fundamentally differs from natural language data (in terms of sequence types, data scale, annotation costs, etc.), posing unique challenges for effectively adapting LLMs to bioinformatics tasks. Existing methods are scattered across different subfields, lacking systematic review and comparative analysis.

**Key Challenge**: On one hand, LLMs have demonstrated great potential in biological sequence modeling; on the other hand, issues such as data scarcity, high computational resource demands, and difficulties in multi-modal integration constrain further development. Prior surveys failed to comprehensively cover the latest progress across the four major domains of DNA, RNA, proteins, and single-cells.

**Goal**: To provide a comprehensive and systematic survey that covers representative methods of LLMs in various subfields of bioinformatics, summarizes architectural paradigms, analyzes common challenges, and points out future directions.

**Key Insight**: Designing a clear two-dimensional survey framework organized by biological sequence types (DNA → RNA → protein → single-cell) as chapters, and categorized by model architecture paradigms (Encoder-only / Decoder-only / Encoder-Decoder).

**Core Idea**: The first systematic survey comprehensively covering LLMs in the four major bioinformatic domains of DNA, RNA, proteins, and single-cells, providing model taxonomies, computational cost quantification, and perspectives on future directions.

## Method

### Overall Architecture

As a survey paper, this work does not propose a new method. The organizational structure is: Preliminaries (three architectural paradigms) → DNA and Genomics → RNA (structure and function) → Proteins (prediction and design) → Single-cell Analysis → Challenges and Future Directions. The core contribution is a systematic classification and comparison of over 30 representative models.

### Key Designs

1. **Systematic Comparison of Three Architectural Paradigms**:

    - Encoder-only (e.g., DNABERT, ProteinBERT, scBERT): Bidirectional self-attention captures sequence context, excelling in representation learning and downstream classification/functional prediction tasks. The average training resource demand is moderate (~43GB VRAM, ~14 days).
    - Decoder-only (e.g., ProGen2, Evo, DNAGPT): Autoregressive generation approach, suitable for sequence generation and de novo design tasks. Training is the fastest (~46GB, ~5 days), but unidirectional attention makes it difficult to capture long-range bidirectional dependencies.
    - Encoder-Decoder (e.g., RoseTTAFold, ESM-3, scGPT): Sequence-to-sequence transformation, suitable for cross-modal mapping and structured outputs requiring bidirectional understanding. It is the most powerful but also has the highest computational requirements (~81GB, ~40 days).

2. **Analysis of Four Major Application Areas**:

    - DNA/Genomics: From DNABERT (functional prediction) to DNABERT2 (cross-species) and to Evo (unifying DNA/RNA/protein), the development trajectory progresses from single-species single-task models toward a grand unified model across species and biological molecules.
    - RNA: Secondary structure prediction (RiNALMo, ERNIE-RNA achieve optimal performance) → tertiary structure prediction (RhoFold+ end-to-end) → functional prediction → sequence generation (RNA-GPT, RNA-DCGen).
    - Proteins: Structure prediction (AlphaFold2/3 reaching atomic-level accuracy) → functional inference (ESM-1b, ProtTrans) → design engineering (ProtGPT2, ESM-3 multi-modal prediction and design), forming a complete chain of prediction-understanding-design.
    - Single-cell: scBERT, Geneformer (pretrained on 29.9 million transcriptomes), scFoundation (100 million parameters), scGPT (pretrained on 33 million transcriptomes + multi-omics), achieving transfer learning tasks such as cell-type annotation, perturbation prediction, and batch integration.

3. **Summary of Challenges and Future Directions**:

    - Three Major Challenges: Data scarcity and bias (skewed towards model organisms and common diseases), computational complexity (long biological sequences are unfriendly to standard Transformers), and insufficient cross-omic integration (most models are still trained on single modalities).
    - Three Major Directions: Hybrid AI models (LLMs + GNNs + knowledge graphs + symbolic AI) → multimodal cross-omic integration (simultaneously processing DNA + RNA + protein + epigenetic data) → clinical translation (model validation, compliance, ethics).

### Loss & Training

The survey paper does not involve specific training strategies. The common summary is that self-supervised pre-training (MLM or autoregressive) + downstream task fine-tuning is the mainstream paradigm.

## Key Experimental Results

### Main Results

This is a survey paper and does not contain original experiments. The following table summarizes representative model comparisons from the survey:

| Model | Architecture | Domain | Key Achievement |
|------|------|------|----------|
| AlphaFold2 | Custom Architecture | Protein | CASP14 atomic-resolution protein structure prediction |
| ESM-3 | Enc-Dec | Protein | Multi-modal protein prediction and design |
| DNABERT2 | Enc-only | DNA | Efficient analysis of multi-species genomic functions |
| Evo | Dec-only | DNA/RNA/Protein | First unified foundation model across DNA/RNA/protein |
| scGPT | Enc-Dec | scRNA | Pre-trained on 33 million single cells, multi-omics analysis |
| RhoFold+ | Enc-only | RNA | End-to-end RNA 3D structure prediction |

### Model Scale and Computational Cost Statistics

| Architecture Type | Average VRAM/Device | Average Training Duration |
|----------|-------------|-------------|
| Encoder-only | ~43 GB | ~14 days |
| Decoder-only | ~46 GB | ~5 days |
| Encoder-Decoder | ~81 GB | ~40 days |

### Key Findings

- Encoder-only models perform robustly in classification and function prediction tasks with moderate training efficiency, making them the most commonly used architecture currently.
- Decoder-only models train the fastest but are weak at capturing long-range bidirectional dependencies, mostly used for sequence generation and de novo design.
- Encoder-Decoder models are the most powerful but consume the most resources, making them the preferred choice for protein structure prediction and single-cell foundation models.
- Single-modality training is the current mainstream limitation, and cross-omic integration (DNA + RNA + protein + epigenetics) is a key breakthrough direction.
- Data scarcity and annotation bias (skewed towards model organisms and common diseases) constrain the generalization capabilities of the models.

## Highlights & Insights

- **Comprehensive Model Matrix Table (Table 1)**: Summarizes the architectures, datasets, tasks, and capabilities of 30+ models. It serves as an efficient reference lookup handbook, saving significant literature research time.
- **Quantification of Computational Costs**: It rarely provides statistics on average GPU VRAM and training duration for different architectures, which offers practical reference value for researchers choosing models under resource limitations.
- **Clear Orientation for Future Directions**: Clear guidance is provided on three directions: hybrid AI models (LLMs + GNNs + knowledge graphs), multi-modal cross-omic integration, and clinical translation.

## Limitations & Future Work

- **Scope Limitations**: Crucial areas such as epigenomics and metagenomics are not covered.
- **Lack of Unified Benchmarking**: The survey only compiles self-reported results from various papers without testing models under unified conditions, making strict and fair performance ranking difficult.
- **Risk of Fast Obsolescence**: The field evolves extremely fast, and a survey up to early 2025 might require updates soon.
- The study does not deeply discuss the impact of differences in biological sequence tokenization strategies (such as k-mer, BPE, single nucleotide) on performance, which is a key technical aspect.

## Related Work & Insights

- **vs. Prior Surveys**: Previous surveys mostly focused on a single domain (such as proteins or genomics). This paper is the first to span four major fields—DNA, RNA, proteins, and single-cells, offering much higher comprehensiveness.
- **vs. General NLP Surveys**: LLMs in bioinformatics face unique challenges that do not exist in general NLP, such as biological sequence tokenization, processing of extremely long sequences (genomes can reach billions of base pairs), and cross-modal alignment.
- The attempt by the Evo model to unify DNA, RNA, and proteins deserves close attention, possibly representing the future direction of biological foundation models.

## Rating

- Novelty: ⭐⭐⭐ As a survey paper, methodological novelty is not emphasized, but its comprehensiveness across four major fields is the core contribution.
- Experimental Thoroughness: ⭐⭐⭐ No original experiments were conducted in the survey, but it covers a wide range of models and includes computational cost quantification.
- Writing Quality: ⭐⭐⭐⭐ The structure is clear, the classification system is rational, and Table 1 is highly valuable.
- Value: ⭐⭐⭐⭐ It serves as an excellent introductory guide for readers seeking to understand the full landscape of LLM applications in bioinformatics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Knowledge Boundary of Large Language Models: A Survey](knowledge_boundary_survey.md)
- [\[ACL 2025\] Recent Advances in Speech Language Models: A Survey](recent_advances_in_speech_language_models_a_survey.md)
- [\[ACL 2025\] When Large Language Models Meet Speech: A Survey on Integration Approaches](when_large_language_models_meet_speech_a_survey_on_integration_approaches.md)
- [\[ACL 2025\] Pragmatics in the Era of Large Language Models: A Survey on Datasets, Evaluation, Opportunities and Challenges](pragmatics_survey.md)
- [\[ACL 2025\] Personalized Generation In Large Model Era: A Survey](personalized_generation_in_large_model_era_a_survey.md)

</div>

<!-- RELATED:END -->
