---
title: >-
  [Paper Note] DeepSeq: High-Throughput Single-Cell RNA Sequencing Data Labeling via Web Search-Augmented Agentic Generative AI Foundation Models
description: >-
  [ICML 2025][Computational Biology][Single-cell RNA sequencing] Proposed the DeepSeq pipeline, which utilizes large language models (especially Agentic GPT-4o with real-time web search capabilities) to automatically annotate cell types in single-cell RNA sequencing data. It achieves a maximum accuracy of 82.5%, resolving the throughput bottleneck of large-scale omics data annotation.
tags:
  - "ICML 2025"
  - "Computational Biology"
  - "Single-cell RNA sequencing"
  - "Large Language Models"
  - "Cell type annotation"
  - "Agentic AI"
  - "Foundation models"
date: 2026-05-08
content_hash: c30cb26ed2fc1263
---

# DeepSeq: High-Throughput Single-Cell RNA Sequencing Data Labeling via Web Search-Augmented Agentic Generative AI Foundation Models

**Conference**: ICML 2025  
**arXiv**: [2506.13817](https://arxiv.org/abs/2506.13817)  
**Code**: [Available](https://github.com/saleemaldajani/deepseq)  
**Area**: Medical Imaging/Bioinformatics  
**Keywords**: Single-cell RNA sequencing, Large Language Models, Cell type annotation, Agentic AI, Foundation models

## TL;DR

Proposed the DeepSeq pipeline, which utilizes large language models (especially Agentic GPT-4o with real-time web search capabilities) to automatically annotate cell types in single-cell RNA sequencing data. It achieves a maximum accuracy of 82.5%, resolving the throughput bottleneck of large-scale omics data annotation.

## Background & Motivation

### Scaling Challenges of Single-Cell RNA Sequencing

Single-cell RNA sequencing (scRNA-seq) has revolutionized the ability to understand biological systems at cellular resolution. Unlike traditional bulk sequencing, scRNA-seq preserves cellular heterogeneity, supporting downstream analyses such as lineage tracing, perturbation inference, and cell type identification.

However, with improvements in barcoding technologies and experimental protocols, scRNA-seq datasets have scaled from thousands of cells to millions of cells per experiment. According to statistics from Svensson et al. (2018), the volume of single-cell sequencing has grown exponentially since 2009, with single studies expected to exceed $10^9$ cells by 2030.

### Annotation Bottlenecks

The current core bottleneck lies in **cell type annotation**:

- Manual annotation is far too slow to keep pace with data growth.
- As the number of clusters increases with data volume, the complexity of manual annotation scales dramatically.
- Downstream tasks such as supervised learning, pseudotime trajectory inference, and perturbation modeling heavily rely on accurate cell type labels.
- Manual annotation inevitably introduces human biases and errors.

### Limitations of Prior Work

Traditional automated annotation methods (such as reference atlas mapping) are limited by reference data coverage and cross-tissue generalization capabilities. Recently, Hou & Ji (2024) demonstrated the preliminary capability of GPT-4 in single-cell annotation in Nature Methods, but lacked a systematic pipeline design and multi-model benchmark comparisons.

### Goal

The authors propose: Can a modular, scalable LLM annotation system be constructed to support both local lightweight inference and online agentic inference, thereby addressing high-throughput annotation needs?

## Method

### Overall Architecture

DeepSeq is an end-to-end modular pipeline, with the overall workflow as:

$$\text{原始数据} \xrightarrow{\text{过滤}} \text{清洗后数据} \xrightarrow{\text{降维+聚类}} \text{细胞簇} \xrightarrow{\text{提取标志基因}} \text{结构化提示} \xrightarrow{\text{LLM推理}} \text{细胞类型标签}$$

The system supports two inference paths:
1. **Local Inference**: Deploying lightweight models (e.g., LLaMA3) via the Ollama client for on-device inference.
2. **Agentic Inference**: Performing online inference via GPT-4o + Web Search, where the agent autonomously retrieves and summarizes external biological knowledge.

### Key Designs

#### 1. **Data Preprocessing and Filtering Module**: Removing low-quality cells and genes → three complementary filtering strategies → ensuring input data quality

The raw data is processed into a gene $\times$ cell matrix and converted into the AnnData format. Filtering utilizes three strategies:

- **Standard Threshold Filtering**: Each cell must express at least $\geq 200$ genes.
- **Automatic Knee-point Detection**: Utilizing the KneeLocator algorithm to automatically determine filtering thresholds.
- **Smoothed Knee-point Filtering**: Detecting knee-points based on smoothed distribution curves.

These three strategies can generate quality control diagnostic plots, allowing users to select the most suitable filtering scheme.

#### 2. **Clustering and Marker Gene Extraction Module**: Grouping cells into clusters and extracting feature genes for each cluster → Leiden clustering + Scanpy gene ranking → providing structured inputs for the LLM

Specific steps:
- Perform dimensionality reduction using PCA.
- Cluster cells using the Leiden algorithm based on the neighborhood graph.
- Embed into 2D space using UMAP for visualization.
- For each cluster $C_i$, extract the top marker genes $G_i = \text{rank\_genes}(C_i)$ using the ranking function in Scanpy.

#### 3. **LLM Annotation Module**: Generating cell type predictions based on marker genes → structured prompts + dual inference paths → balancing efficiency and accuracy

Core algorithmic workflow:

For each cluster $C_i$:
1. Extract the top marker genes $G_i$.
2. Construct a structured prompt $P_i = \text{format}(G_i)$.
3. Depending on the inference mode:
    - **Ollama Path**: $\hat{y}_i = \text{local\_LLM}(P_i)$
    - **GPT-4o Path**: First execute web search to obtain context, followed by $\hat{y}_i = \text{gpt4o}(P_i, \text{web results})$

The prompt engineering refers to the format design by Hou & Ji (2024) and is adapted to structured transcriptomic data. LangChain orchestrates prompts and post-processing.

#### 4. **Evaluation Module**: Quantifying annotation accuracy → two-stage validation protocol → ensuring reproducibility

- **Stage 1: Marker Gene Validation** — Confirming that the top marker genes of each cluster adequately match known canonical marker genes to ensure biological relevance.
- **Stage 2: Label Accuracy Evaluation** — Comparing LLM-generated labels with human-annotated ground truth, employing fuzzy string matching and synonym resolution to robustly evaluate cluster-level consistency.

### Loss & Training

The proposed method does not involve model training; it is an inference-time pipeline. No loss function or backpropagation is required, and the LLM is directly used as a "zero-shot" annotator. The core of the system lies in:

- **Prompt Engineering**: Carefully designed structured prompts that convert marker gene information into a format comprehensible to LLMs.
- **Agentic Enhancement**: Introducing external biological knowledge via real-time web search to augment the domain-specific reasoning capabilities of the LLM.
- **Post-processing**: Fuzzy matching and synonym resolution to address inconsistencies in label formatting.

## Key Experimental Results

### Main Results

Experiments were conducted on a standard scRNA-seq dataset, using the top marker genes as prompt inputs to compare the annotation accuracy of different LLMs.

| Model | Parameters | Inference Mode | Annotation Accuracy | Features |
|---|---|---|---|---|
| LLaMA3-2-1B | ~1B | Local (Ollama) | Low | Lightweight, offline deployment |
| GPT-3.5-turbo | ~175B | Agentic (Web Search) | Moderate | Equipped with web search capability |
| GPT-4o | ~1.8T | Agentic (Web Search) | **82.5%** | Highest accuracy |

Key observations:
- GPT-4o achieved the highest accuracy of **82.5%**.
- The performance improvement from LLaMA3-2-1B to GPT-3.5 is larger than that from GPT-3.5 to GPT-4o.
- This indicates that agentic capabilities (web search) provide a baseline boost, but the gains from architectural optimization and parameter scaling yield diminishing returns.

### Ablation Study

| Configuration | Key Metrics | Description |
|---|---|---|
| Without Web Search (LLaMA3-1B) | Significantly lower than models with search | Web search is a key source of performance gain |
| With Web Search (GPT-3.5) | Substantially improved compared to no search | Agentic capabilities provide baseline gains |
| With Web Search + Larger Model (GPT-4o) | 82.5% | Doubled parameters but limited gains |
| Three Filtering Strategies | Respective advantages and disadvantages | Providing multiple quality control options |

### Key Findings

1. **Agentic Capability > Model Scale**: The gains from web search outweigh those from merely increasing parameter size. In the absence of domain-specific data, the returns of model scaling diminish.

2. **Scaling Law Extends to Biological Data**: Similar to the positive correlation between data volume and performance in language models, the accuracy of cell type annotation also depends on the scale and diversity of experimental data.

3. **Lightweight Models remain Competitive**: LLaMA3-1B performs respectably considering its size, confirming the feasibility of lightweight deployment in resource-constrained environments.

4. **Marker Gene Quality is the Bottleneck**: The information content of prompts depends on the discriminative capacity of the marker genes in each cluster. The model remains fragile in biologically ambiguous scenarios.

## Highlights & Insights

1. **Excellent System Design Philosophy**: Fully integrating LLMs with single-cell analysis to form an end-to-end reproducible pipeline, rather than a simple proof-of-concept of "throwing a gene list into ChatGPT."

2. **Dual-path Inference Architecture**: Simultaneously supporting offline local inference and online agentic inference, adapting to different deployment scenarios (privacy-sensitive environments vs. pursuit of maximum accuracy).

3. **Analogizing Scaling Laws to Biological Annotation**: Proposing the insight that LLM annotation may transcend human-level performance as single-cell datasets scale to the $10^9$ magnitude.

4. **High Practical Utility**: The code is open-source, and each step of the pipeline outputs interpretable logs, supporting the expansion of different LLM configurations and evaluation strategies.

5. **Quantified Validation of Web Search as Knowledge Augmentation** — This provides empirical data supporting the application of Agentic AI in scientific domains.

## Limitations & Future Work

1. **Single Dataset Limitation**: Only validated on a single scRNA-seq dataset, lacking cross-tissue and cross-species generalization testing. Whether the 82.5% accuracy remains stable across different datasets requires more experiments.

2. **Limited Evaluation Granularity**: Only cluster-level label accuracy was evaluated, lacking cell-level accuracy analysis and confusion matrices.

3. **Lack of Comparison with Domain-Specific Tools**: No comparison was made with specialized cell annotation tools like CellTypist, scType, or SingleR, making it difficult to judge whether the LLM approach genuinely outperforms existing specialized tools.

4. **Absence of Cost Analysis**: The cost of GPT-4o API calls could be high on large-scale datasets, but the economic viability is not discussed in the paper.

5. **Unexplored Prompt Sensitivity**: The effects of different prompt formats and choices of marker gene counts on performance was not systematically investigated.

6. **Practical Implications of 82.5% Accuracy**: In clinical and research scenarios demanding high precision, a 17.5% error rate may still be too high, and the paper lacks sufficient discussion on this.

7. **No Experiments on Multimodal Integration**: Although potential integration with scATAC-seq and spatial transcriptomics was discussed, empirical experiments are lacking.

## Related Work & Insights

- **Hou & Ji (2024), Nature Methods**: First evaluated the cell annotation capabilities of GPT-4 in single cells, serving as the foundation for the prompt design in this work.
- **Svensson et al. (2018)**: Documented the exponential growth trend of single-cell sequencing, supporting the scaling law discussions in this paper.
- **Human Cell Atlas / Human Tumor Atlas Network**: Large-scale cell atlas initiatives, providing application scenarios for DeepSeq.
- **Wang et al. (2025)**: SpatialAgent, an autonomous AI agent for spatial biology, echoing the agentic paradigm of this paper.

**Insights**:
- The paradigm of Agentic AI + domain knowledge retrieval for scientific data annotation is worth generalizing to other omics data (e.g., proteomics, metabolomics).
- Applying the Scaling Law framework to analyze the performance bounds of bioinformatics tools is an interesting perspective.
- The design concept of a hybrid inference architecture (local + cloud) can be applied to privacy-sensitive clinical scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ The method itself is an engineering combination of existing technologies (LLM + Scanpy + LangChain); the core innovation lies in the system design rather than the algorithm.
- Experimental Thoroughness: ⭐⭐⭐ Validated on a single dataset, lacking comparisons with domain-specific tools, and containing limited ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ Well-structured with detailed descriptions of the system architecture and rich illustrations/tables.
- Value: ⭐⭐⭐⭐ Demonstrates the feasibility of LLMs in biological data annotation, but the real-world impact is limited by the upper bound of accuracy and the lack of comprehensive validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] A Survey on Foundation Language Models for Single-cell Biology](../../ACL2025/computational_biology/foundation_lm_single_cell_survey.md)
- [\[ICML 2025\] scSSL-Bench: Benchmarking Self-Supervised Learning for Single-Cell Data](scssl-bench_benchmarking_self-supervised_learning_for_single-cell_data.md)
- [\[NeurIPS 2025\] Steering Generative Models with Experimental Data for Protein Fitness Optimization](../../NeurIPS2025/computational_biology/steering_generative_models_with_experimental_data_for_protein_fitness_optimizati.md)
- [\[ICML 2025\] PolyConf: Unlocking Polymer Conformation Generation through Hierarchical Generative Models](polyconf_unlocking_polymer_conformation_generation_through_hierarchical_generati.md)
- [\[ICLR 2026\] Riemannian High-Order Pooling for Brain Foundation Models](../../ICLR2026/computational_biology/riemannian_high-order_pooling_for_brain_foundation_models.md)

</div>

<!-- RELATED:END -->
