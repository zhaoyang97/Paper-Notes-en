---
title: >-
  [Paper Note] A Survey on Foundation Language Models for Single-cell Biology
description: >-
  [ACL 2025 (Long Paper)][Computational Biology][single-cell biology] This is the first systematic survey of foundation language models for single-cell biology from a language modeling perspective. It categorizes existing works into two major groups: PLMs (pre-trained from scratch) and LLMs (leveraging existing large models). The paper comprehensively analyzes tokenization strategies, pre-training/fine-tuning paradigms, and downstream task systems…
tags:
  - "ACL 2025 (Long Paper)"
  - "Computational Biology"
  - "single-cell biology"
  - "foundation language model"
  - "pre-trained language model"
  - "gene expression"
  - "tokenization"
date: 2026-05-08
content_hash: b54ec6cc9f07a28b
---

# A Survey on Foundation Language Models for Single-cell Biology

**Conference**: ACL 2025 (Long Paper)  
**Code**: None  
**Area**: Computational Biology / NLP Cross-disciplinary  
**Keywords**: single-cell biology, foundation language model, pre-trained language model, gene expression, tokenization  

## TL;DR
This is the first systematic survey of foundation language models for single-cell biology from a language modeling perspective. It categorizes existing works into two major groups: PLMs (pre-trained from scratch) and LLMs (leveraging existing large models). The paper comprehensively analyzes tokenization strategies, pre-training/fine-tuning paradigms, and downstream task systems, while highlighting key challenges in data quality, unified evaluation, and scaling laws.

## Background & Motivation

**Cross-domain Transfer Trend**: The success of language models (such as BERT, GPT, etc.) has penetrated the field of computational biology. Researchers have found that **cells can be analogized to "sentences" and genes to "words/tokens"**, enabling the construction of unified single-cell foundation models using language models.

**Value of Unified Representation**: Such models can obtain universal cell representations across datasets and tasks, outperforming traditional task-specific models in downstream tasks like cell-type annotation, gene perturbation prediction, and drug response modeling, thereby avoiding the high cost of designing separate models for each task.

**Motivation for Survey**: Most existing surveys analyze single-cell models from the perspective of Transformer architectures (e.g., Lan et al. 2024, Szalata et al. 2024), lacking a systematic analysis starting from "language modeling," which is the core paradigm of NLP. This paper fills this gap by reorganizing the knowledge in this field using the PLM vs. LLM dichotomy, which is more familiar to the NLP community.

## Method

### Overall Architecture
The single-cell foundation language models are categorized into two major camps:

1. **Single-cell PLMs** (Pre-trained Language Models): Treating genes as tokens and cells as sentences, these models are pre-trained from scratch on large-scale single-cell data. Typical representatives include scBERT, scGPT, GeneFormer, scFoundation, etc.

2. **Single-cell LLMs** (Large Language Models): Instead of pre-training from scratch, these models leverage existing general-purpose LLMs (such as GPT-2/3.5/4, LLaMA, T5) to perform fine-tuning or direct inference by converting single-cell data into text. Typical representatives include Cell2Sentence, GenePT, scELMo, etc.

### Key Designs

1. **Tokenization Strategies (PLM-side)**

    Convert the cell-gene expression matrix (N x G) into a format understandable by language models, focusing on three major directions:
    - **Discretization**: Binning discretizes continuous expression values into integer intervals (scBERT, CellLM); Rank Value Encoding ranks genes by expression levels and encodes them using a gene vocabulary (GeneFormer series).
    - **Continuous Embedding**: Leveraging protein language models to obtain gene embeddings (UCE, scPRINT); learnable projection layers (CellPLM); hierarchical Bayesian downsampling to handle sparsity (scFoundation).
    - **Auxiliary Information Integration**: Integrating metadata (cell states, organ of origin, donor information, sequencing technology) or prior knowledge from protein foundation models.

2. **Pre-training Paradigms (PLM-side)**

    - **Masked Language Modeling (MLM)**: The most mainstream paradigm, where 15-30% of genes are randomly masked and then reconstructed. Adopted by: scBERT, UCE, GeneFormer, CellPLM, scFoundation, Nicheformer.
    - **Next Token Prediction (NTP)**: Autoregressive pre-training, adopted only by tGPT and scGPT. It is not popular in the single-cell domain due to: (1) data scale still being much smaller than text; (2) high sparsity in gene expression causing many ground-truth values to be zero, which drives the model to learn trivial zero solutions.
    - **Multi-task Pre-training**: Superimposing supervised signals such as contrastive learning, classification, cell generation, metadata prediction, and denoising on top of MLM (CellLM, LangCell, scCello, scPRINT, scMulan, GeneCompass, CellFM).

3. **Cell-to-Text Conversion and Fine-tuning Paradigms (LLM-side)**

    **Conversion Methods**:
    - Cell-to-Sentence: Concatenating the names of the top-100 genes ranked by expression levels into natural language sentences (Cell2Sentence, CHATCELL, CELLama).
    - Text-level Gene Embeddings: Obtaining functional description embeddings for each gene using an LLM, and then performing weighted combinations with expression values (GenePT, scELMo, scInterpreter).

    **Fine-tuning Paradigms**:
    - Instruction Fine-tuning: Converting tasks into QA formats (Cell2Sentence, CHATCELL).
    - Embedding Fine-tuning: Directly utilizing cell/gene embeddings for supervised fine-tuning (currently the mainstream).
    - Tuning-free: LLMs serve as agents to directly generate Python code to perform analysis (scChat).

## Key Experimental Results

### Model Comparison Overview

| Model | Type | Tokenization | Pre-training Paradigm | Data Scale |
|------|------|-------------|-----------|---------|
| scBERT | PLM | Binning | MLM | 1M cells |
| GeneFormer | PLM | Rank Value | MLM | 27.4M cells |
| scGPT | PLM | Binning+Meta | NTP | 33M cells |
| scFoundation | PLM | Downsampling | MLM | 50M cells |
| GeneCompass | PLM | Ranking+Meta | Multi-task | 126M cells |
| CellFM | PLM | Padding+MLP | Multi-task | 100M cells |
| Nicheformer | PLM | Ranking+Meta | MLM | 57M cells |
| Cell2Sentence | LLM | Cell-to-Text | Instruction Fine-tuning | GPT-2 base |
| GenePT | LLM | Text Embedding | Embedding Fine-tuning | GPT-3.5 base |
| CELLama | LLM | Cell-to-Text | Instruction Fine-tuning | LLaMA-13B base |

### Downstream Task System

| Task Level | Specific Tasks |
|----------|---------|
| **Cell-level** | Cell type annotation, novel cell type discovery, batch effect correction, cell clustering, multi-omics integration, cell generation |
| **Gene-level** | Gene network analysis, gene perturbation prediction, gene function/expression prediction |
| **Drug-related** | Drug sensitivity prediction, drug response modeling |
| **Spatial-related** | Spatial transcriptomics imputation, spatial label prediction, spatial composition analysis |

### Key Findings
- MLM significantly outperforms NTP in the single-cell domain: data scale and sparsity are the primary reasons for NTP's poor performance.
- Multi-task pre-training integrates self-supervised and supervised signals, and generally achieves the best results.
- Scaling up data size from 1M to 126M cells yields consistent performance improvements, but the scaling laws remain unclear.
- Only scGPT and scELMo have demonstrated multi-omics integration capabilities, indicating vast potential in this direction.
- The Cell-to-Sentence approach is simple and intuitive but suffers from heavy information loss (retaining only the top-100 genes), whereas Text-level Embedding is more faithful but carries higher computational overhead.

## Highlights & Insights
- **Clear Classification System**: The PLM vs. LLM dichotomy, combined with detailed sub-classifications (three tokenization strategies, three pre-training paradigms, and three LLM fine-tuning modes), allows readers to rapidly build a panoramic view.
- **New Perspective of Language Modeling**: This study is the first to explicitly examine single-cell foundation models through the lens of NLP language modeling rather than a traditional bioinformatics perspective, making it more accessible to the NLP community.
- **Systematization of the Cell=Sentence Analogy**: The unified framework where genes are treated as tokens and cells as sentences is concise and elegant, serving as a prime example of cross-domain knowledge transfer.
- **Systematic Analysis of Challenges**: A deep analysis is provided across three dimensions: data quality (sparsity, batch effects, lack of multi-omics), model design (need for a unified tokenizer, unrealized scaling laws), and evaluation protocols (lack of standard benchmarks).

## Limitations & Future Work
- The survey itself contains no original experiments and lacks horizontal, quantitative comparisons of different models' empirical performance (as most models were evaluated on private datasets).
- The authors acknowledge their focus on technical analysis, with insufficient discussion on the **biological motivations** behind design choices—such as why a certain tokenization method is more biologically plausible.
- Limited timeliness: A significant number of subsequent new models (e.g., CellVerse) have emerged but are not covered.
- The lack of unified benchmarks is a critical pain point for the entire field; the paper highlights this but does not propose concrete solutions.
- The largest existing single-cell PLM has fewer than 1B parameters, and it remains unclear whether its scaling behavior mirrors that of the NLP domain.

## Related Work & Insights
- **vs. Lan et al. (2024), Szalata et al. (2024)**: While prior works analyze from the perspective of Transformer architectures, this paper is the first to organize knowledge from a language modeling viewpoint (the PLM vs. LLM dichotomy).
- **vs. LLM4Cell (Dip et al., 2025)**: The latter is more recent and covers agentic models (such as scChat).
- **vs. Traditional Surveys**: Traditional bioinformatics surveys focus on experimental methodologies and biological insights, whereas this paper highlights modeling paradigms and NLP technology transfer.

## Rating
- Novelty: ⭐⭐⭐⭐ Although the survey introduces no new methods, analyzing the field from a language modeling perspective provides a highly valuable new angle.
- Experimental Thoroughness: ⭐⭐⭐⭐ No original experiments are performed; the model comparison tables are relatively comprehensive but lack quantitative comparisons.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure, complete classification system, and intuitive diagrams make it highly suitable as a quick start guide.
- Value: ⭐⭐⭐⭐ Highly valuable for understanding the intersection of NLP and computational biology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] scPilot: Large Language Model Reasoning Toward Automated Single-Cell Analysis and Discovery](../../NeurIPS2025/computational_biology/scpilot_large_language_model_reasoning_toward_automated_single-cell_analysis_and.md)
- [\[ICML 2026\] Towards Universal Gene Regulatory Network Inference: Unlocking Generalizable Regulatory Knowledge in Single-cell Foundation Models](../../ICML2026/computational_biology/towards_universal_gene_regulatory_network_inference_unlocking_generalizable_regu.md)
- [\[ICML 2025\] DeepSeq: High-Throughput Single-Cell RNA Sequencing Data Labeling via Web Search-Augmented Agentic Generative AI Foundation Models](../../ICML2025/computational_biology/deepseq_high-throughput_single-cell_rna_sequencing_data_labeling_via_web_search-.md)
- [\[ACL 2025\] Concept Bottleneck Language Models For Protein Design](concept_bottleneck_language_models_for_protein_design.md)
- [\[NeurIPS 2025\] PRESCRIBE: Predicting Single-Cell Responses with Bayesian Estimation](../../NeurIPS2025/computational_biology/prescribe_predicting_single-cell_responses_with_bayesian_estimation.md)

</div>

<!-- RELATED:END -->
