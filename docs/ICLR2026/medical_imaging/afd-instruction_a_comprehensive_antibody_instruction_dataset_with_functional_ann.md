---
title: >-
  [Paper Note] AFD-INSTRUCTION: A Comprehensive Antibody Instruction Dataset with Functional Annotations for LLM-Based Understanding and Design
description: >-
  [ICLR 2026][Medical Imaging][Antibody language model] This work constructs AFD-Instruction, the first large-scale antibody functional annotation instruction dataset (430K+ entries)…
tags:
  - "ICLR 2026"
  - "Medical Imaging"
  - "Antibody language model"
  - "instruction tuning dataset"
  - "sequence-function alignment"
  - "antibody design"
  - "multi-agent data construction"
date: 2026-05-08
content_hash: 5ced80ffe7152db5
---

# AFD-INSTRUCTION: A Comprehensive Antibody Instruction Dataset with Functional Annotations for LLM-Based Understanding and Design

**Conference**: ICLR 2026
**arXiv**: [2602.04916](https://arxiv.org/abs/2602.04916)
**Code**: To be released
**Area**: Bioinformatics / LLM Instruction Tuning
**Keywords**: Antibody language model, instruction tuning dataset, sequence-function alignment, antibody design, multi-agent data construction

## TL;DR
This work constructs AFD-Instruction, the first large-scale antibody functional annotation instruction dataset (430K+ entries), aligning antibody sequences with natural-language functional descriptions via a multi-agent literature extraction pipeline. The dataset is used to instruction-tune general-purpose LLMs for antibody understanding and function-guided design, achieving an average accuracy improvement of 20+ points across five classification tasks.

## Background & Motivation

**Background**: LLMs have been widely applied to protein understanding (e.g., Mol-Instructions, InstructProtein); however, antibodies—a specialized class of proteins with significant therapeutic value—lack dedicated sequence-function alignment datasets.

**Limitations of Prior Work**: (a) Existing protein language models (PLMs) are trained on raw sequences in an unsupervised manner, lacking functional supervision signals; (b) although the OAS database contains millions of antibody sequences, the vast majority carry no functional annotations; (c) general-purpose LLMs cannot interpret antibody sequences, while PLMs cannot process natural language—creating a modality gap between the two.

**Key Challenge**: The core value of antibodies lies in their functionality (target binding, neutralization activity, etc.), yet sequence-function paired data remains extremely scarce.

**Goal**: To construct the first large-scale instruction dataset that systematically aligns antibody sequences with natural-language functional descriptions, enabling LLMs to infer function from sequence and to generate sequences under functional constraints.

**Key Insight**: A multi-agent extraction pipeline is applied to ~4,000 publications to derive antibody-function pairs, which are subsequently expanded into instruction-response pairs via a self-questioning strategy.

**Core Idea**: A literature-level multi-agent system combined with a self-questioning strategy is used to mine sequence-function pairs at scale from published antibody research, yielding an instruction dataset covering both understanding and design tasks.

## Method

### Overall Architecture
The construction of AFD-Instruction proceeds in three stages: (1) antibodies are collected from SAbDab/PDB and a balanced set of 4,305 entries is sampled using MMseqs2 sequence-distance-based sampling; (2) a multi-agent system extracts functional annotations from the corresponding literature; (3) a self-questioning strategy expands the annotations into 430K+ instruction pairs. The dataset supports two major applications: antibody understanding (classification QA and open-ended captioning) and antibody design (CDR3 design and full-sequence generation).

### Key Designs

1. **Multi-Agent Literature Extraction System**

   - Function: Automatically extracts antibody functional annotations from scientific publications.
   - Mechanism: Three specialized roles divide the labor—*Mr. Extractor* scans text and extracts basic information (category, target, source, function); *Dr. Mechanism* analyzes structural and mechanistic details (binding sites, molecular effects); *Prof. Function* synthesizes high-level interpretations (mode of action, therapeutic relevance, distinctive characteristics).
   - Design Motivation: A single agent is prone to information omission and hallucination; the division of labor ensures completeness and hierarchical integrity across factual extraction, mechanistic analysis, and functional synthesis.

2. **Self-Questioning Strategy**

   - Function: Automatically generates diverse instruction-response pairs from antibody-description pairs.
   - Mechanism: For **understanding**, five types of classification questions are generated (antibody category, disease association, binding site, mechanism of action, function) along with captioning tasks (free-text description). For **design**, the input consists of a functional description and antigen sequence (tagged with `<Anti></Anti>`), and the output is either a full antibody sequence or a CDR3 sequence (tagged with `<CDR3></CDR3>`). Seed prompts, LLM generation, automatic consistency checking, and deduplication are applied throughout.
   - Design Motivation: The raw antibody-description pairs are limited in scale (~4,305 entries); multi-perspective question generation expands the dataset to 430K+ entries.

3. **Sequence Format Specification**

   - Function: Marks antibody sequences with explicit chain-level tags.
   - Mechanism: Heavy chains are enclosed in `<H></H>`, light chains in `<L></L>`, antigens in `<Anti></Anti>`, and CDR3 regions in `<CDR3></CDR3>`.
   - Design Motivation: Enables text-based LLMs to comprehend the structural organization of protein sequences.

### Quality Control
- Automatic completeness checks and manual verification on a 10% random sample.
- A 5% subset of instruction pairs is reviewed by two independent domain experts; Cohen's $\kappa = 0.82$.
- Ambiguous cases are resolved through discussion, and extraction rules are updated accordingly.

## Key Experimental Results

### Main Results — Classification Tasks
Performance of instruction-tuned LLaMA-8B and Qwen2-7B on five antibody understanding classification tasks:

| Model | Class ACC | Disease ACC | Binding ACC | Mechanism ACC | Function ACC |
|-------|----------|------------|------------|--------------|-------------|
| GPT-4o | 82.02 | 72.15 | 50.31 | 63.99 | 56.17 |
| Claude-3 | 95.40 | 70.89 | 42.65 | 43.81 | 47.84 |
| DeepSeek-V3 (671B) | 93.99 | 74.45 | 47.88 | 59.20 | 49.39 |
| InstructProtein | 52.44 | 74.66 | 47.91 | 58.36 | 48.51 |
| **QwenAB (7B, Ours)** | **98.86** | **87.83** | **87.81** | **93.60** | **85.01** |
| **LLaMAB (8B, Ours)** | **98.48** | 85.11 | 87.01 | 92.91 | 83.81 |

QwenAB achieves an average accuracy 20.21 points higher than the strongest baseline. Notably, the 7B fine-tuned models comprehensively outperform both the 671B general-purpose model and closed-source commercial models.

### Antibody Design Experiments
Instruction-tuned models generate sequences with reasonable structural diversity and functional relevance in both CDR3 design and full antibody generation tasks. Significant improvements are also observed in captioning metrics (e.g., QwenAB achieves BLEU-4 = 17.25 on Binding captioning vs. GPT-4o's 6.74).

### Key Findings
- Even the strongest closed-source models (GPT-4o, Claude-3) perform substantially worse than the 7B fine-tuned models on antibody-specific tasks (e.g., Binding, Mechanism), demonstrating that antibody knowledge cannot be adequately acquired through general-purpose pretraining.
- Existing protein-domain models (Galactica, Mol-Instructions, etc.) also perform poorly on antibody tasks, indicating that general protein knowledge does not transfer to antibody-specific knowledge.
- GemmaAB-9B and DeepSeekAB-MoE-16B also perform strongly, demonstrating that the benefits of AFD-Instruction transfer across model architectures.

## Highlights & Insights
- **First antibody sequence-function instruction dataset**: Fills a critical gap and provides infrastructure-level resources for future research at 430K+ entries.
- The **multi-agent literature mining pipeline** is transferable to other biological domains—any field with paired literature and database records can adopt a similar approach to construct domain-specific instruction datasets.
- The result of **7B models outperforming 671B models** reaffirms that domain-specific data is far more valuable than model scale—an important practical implication for biomedical AI deployment, where the best data matters more than the largest model.

## Limitations & Future Work
- Data sources are limited to antibodies in SAbDab/PDB with corresponding literature, constraining coverage to published research.
- The extraction accuracy of the multi-agent system depends on LLM capability and may introduce factual errors.
- Evaluation is limited to instruction tuning of text-based LLMs; integration with protein structure models (e.g., ESMFold, AlphaFold) remains unexplored.
- Antibody design tasks lack wet-lab validation—whether the generated sequences exhibit the intended functions is unknown.
- CDR3 design covers only CDR-H3; the design of other CDR regions and framework regions is not addressed.

## Related Work & Insights
- **vs. Mol-Instructions**: A general protein instruction dataset lacking antibody-specific annotations; AFD substantially outperforms it on antibody tasks.
- **vs. InstructProtein**: Aligns protein-text via knowledge graphs, but similarly lacks antibody functional descriptions.
- **vs. ProtLLM**: Employs interleaved protein-text pretraining, yet still underperforms AFD-fine-tuned general LLMs on antibody classification tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ The dataset construction methodology (multi-agent + self-questioning) is creative, though the primary contribution skews toward data resources.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comparisons against 17+ baselines (including 5 closed-source commercial models) validated across 5 model architectures—highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with detailed descriptions of the data construction process.
- Value: ⭐⭐⭐⭐ Provides infrastructure-level contributions to the antibody AI field, though the absence of wet-lab validation limits practical impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Instruction-Guided Lesion Segmentation for Chest X-rays with Automatically Generated Large-Scale Dataset](../../CVPR2026/medical_imaging/instruction-guided_lesion_segmentation_for_chest_x-rays_with_automatically_gener.md)
- [\[ICLR 2026\] Omni-iEEG: A Large-Scale, Comprehensive iEEG Dataset and Benchmark for Epilepsy Research](omni-ieeg_a_large-scale_comprehensive_ieeg_dataset_and_benchmark_for_epilepsy_re.md)
- [\[ICLR 2026\] ConfHit: Conformal Generative Design with Oracle Free Guarantees](confhit_conformal_generative_design_with_oracle_free_guarantees.md)
- [\[ICLR 2026\] mCLM: A Modular Chemical Language Model that Generates Functional and Makeable Molecules](mclm_a_modular_chemical_language_model_that_generates_functional_and_makeable_mo.md)
- [\[ICLR 2026\] Boosting Medical Visual Understanding From Multi-Granular Language Learning](boosting_medical_visual_understanding_from_multi-granular_language_learning.md)

</div>

<!-- RELATED:END -->
