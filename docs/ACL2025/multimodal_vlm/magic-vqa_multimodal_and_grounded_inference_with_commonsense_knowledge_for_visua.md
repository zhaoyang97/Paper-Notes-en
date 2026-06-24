---
title: >-
  [Paper Note] MAGIC-VQA: Multimodal and Grounded Inference with Commonsense Knowledge for Visual Question Answering
description: >-
  [ACL 2025][Multimodal VLM][Visual Question Answering] The paper proposes the MAGIC-VQA framework, which systematically injects external commonsense knowledge into LVLMs through a three-stage pipeline (explicit knowledge retrieval $\rightarrow$ type-based post-processing $\rightarrow$ GNN-based implicit enhancement). This achieves plug-and-play commonsense reasoning enhancement on benchmarks like ScienceQA, TextVQA, and MMMU, requiring only 0.33M trainable parameters.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Visual Question Answering"
  - "Commonsense Knowledge"
  - "Knowledge Graph"
  - "Graph Neural Network"
  - "Large Vision-Language Models"
date: 2026-05-08
content_hash: eca60d09ea39aa07
---

# MAGIC-VQA: Multimodal and Grounded Inference with Commonsense Knowledge for Visual Question Answering

**Conference**: ACL 2025  
**arXiv**: [2503.18491](https://arxiv.org/abs/2503.18491)  
**Authors**: Shuo Yang, Soyeon Caren Han (University of Melbourne), Siwen Luo (UWA), Eduard Hovy  
**Code**: [adlnlp/magic_vqa](https://github.com/adlnlp/magic_vqa)  
**Area**: Multimodal VLMs  
**Keywords**: Visual Question Answering, Commonsense Knowledge, Knowledge Graph, Graph Neural Network, Large Vision-Language Models  

## TL;DR

The paper proposes the MAGIC-VQA framework, which systematically injects external commonsense knowledge into LVLMs through a three-stage pipeline (explicit knowledge retrieval $\rightarrow$ type-based post-processing $\rightarrow$ GNN-based implicit enhancement). This achieves plug-and-play commonsense reasoning enhancement on benchmarks like ScienceQA, TextVQA, and MMMU, requiring only 0.33M trainable parameters.

## Background & Motivation

### Background
Visual Question Answering (VQA) requires models to simultaneously understand visual and textual information. Recently, Large Vision-Language Models (LVLMs) such as LLaVA, Qwen2VL, and GPT-4o have made significant progress in VQA. However, they still underperform on tasks requiring commonsense reasoning, such as implicit contextual clues or everyday world knowledge.

### Limitations of Prior Work
- **Multimodal RAG Methods**: These inject external information through dense retrieval, but retrieval is static and input-agnostic, which easily introduces noise.
- **Prompt Tuning Methods**: These rely on carefully designed prompts to elicit the model's intrinsic commonsense, but static prompts lack dynamic adaptability to new scenarios.
- **Graph-based Methods**: These utilize GNNs to integrate structured commonsense knowledge, but ignore the dynamic interaction between external knowledge and the model's internal knowledge.
- **Key Challenge**: There is no unified framework that combines dynamic, context-aligned commonsense integration with structured graph reasoning.

### Design Motivation
To design a lightweight, plug-and-play framework that systematically injects explicit and implicit commonsense knowledge into any LVLM without large-scale pre-training or complex prompt tuning, thereby enhancing commonsense reasoning capabilities.

## Method

### Overall Architecture
MAGIC-VQA adopts a three-stage pipeline architecture:
1. **Explicit Commonsense Knowledge Retrieval**: Extracts knowledge triplets relevant to the input from an external knowledge graph.
2. **Type-based Commonsense Post-processing**: Filters and assigns relevance levels to the triplets based on dataset characteristics.
3. **Implicit Commonsense Knowledge Enhancement**: Builds a heterogeneous multimodal graph via a GCN to generate confidence scores.

Ultimately, the original inputs (image, question, caption), knowledge triplets with relevance levels, and GNN confidence scores are jointly fed into the LVLM for inference.

### Key Designs

#### Key Design 1: Explicit Commonsense Knowledge Retrieval

ATOMIC2020 is selected as the external knowledge source, as it covers 1.33 million triplets across 23 relation types, encompassing three main categories of commonsense:
- **Physical Entities (PE)**: Object properties and functions, such as "paper is made from cellulose."
- **Event-centric (EC)**: Sequences of situations, such as "X eats breakfast" usually happens before "X goes to work."
- **Social Interactions (SI)**: Interpersonal interactions and emotions, such as "X gives a gift" causes "Y to feel appreciative."

Retrieval process: Given an image $I$ and a question $Q$, BLIP2 is first used to generate a caption $C$. Then, $\{I, Q, C\}$ are encoded into a shared embedding space, and cosine similarity is calculated against the head and tail entities of all triplets in ATOMIC2020. The Top-K most relevant triplets are selected for each input source.

#### Key Design 2: Type-based Commonsense Post-processing

This stage consists of two steps:

**Type-based filtering**: Tailors the proportion of commonsense types according to the requirements of each dataset. Analysis reveals that ScienceQA requires more PE knowledge (ratio of 0.7:0.15:0.15), TextVQA relies more on EC knowledge (0.2:0.6:0.2), and MMMU requires a balanced mix (0.33:0.33:0.33). Triplets falling below a threshold $\tau$ are first discarded, and then the highest-scoring $k_t = \lfloor p_t \times k \rfloor$ triplets are selected from each type according to the target proportions.

**Relevance level assignment**: Employs a dynamic threshold mechanism based on the mean $\mu_f$ and standard deviation $\sigma_f$ of cosine similarity for each dataset. Triplets are classified into three levels: High ($\geq \mu_f + \sigma_f/2$), Medium, and Low. This helps the LVLM prioritize the most meaningful knowledge during inference.

#### Key Design 3: Implicit Commonsense Enhancement with GNN

A heterogeneous multimodal graph $G_n = \{V, E\}$ is constructed:
- **Nodes**: One node each for the image, question, and caption, plus $k$ commonsense nodes (obtained by flattening the filtered triplets into natural language sentences).
- **Edges**: Constructed based on the cosine similarity between node embeddings.
- **Inference**: A two-layer GCN is used to iteratively update node embeddings ($H^{(l+1)} = \rho(\widetilde{A}H^{(l)}W_l)$). After pooling, an MLP generates confidence scores for candidate answers.

These confidence scores are injected into the LVLM as additional signals, enabling it to prioritize answers backed by commonsense.

## Key Experimental Results

### Table 1: Ablation Study on Explicit Commonsense Knowledge (Accuracy % for each configuration)

| Model | Without CS | CS-Q | CS-I | CS-C | CS-PE | CS-EC | CS-SI | All CS |
|------|--------|------|------|------|-------|-------|-------|--------|
| **ScienceQA** | | | | | | | | |
| LLaVA1.6 | 67.50 | 68.83 | 71.56 | 70.35 | 71.12 | 69.01 | 70.83 | 72.30 |
| Qwen2VL | 71.39 | 72.21 | 74.83 | 71.86 | 74.22 | 72.03 | 72.57 | 75.95 |
| GPT4o-mini | 76.45 | 77.34 | 79.83 | 77.17 | 79.63 | 77.52 | 78.87 | 81.22 |
| **TextVQA** | | | | | | | | |
| Qwen2VL | 75.30 | 76.07 | 77.63 | 77.05 | 76.57 | 78.02 | 76.85 | 78.90 |
| GPT4o-mini | 78.98 | 79.34 | 81.25 | 80.63 | 80.93 | 81.51 | 81.22 | 82.13 |
| **MMMU** | | | | | | | | |
| Qwen2VL | 51.10 | 52.69 | 55.89 | 54.83 | 53.60 | 54.57 | 54.10 | 57.42 |
| GPT4o-mini | 55.89 | 56.53 | 58.79 | 56.21 | 58.12 | 57.57 | 57.89 | 60.87 |

### Table 2: Component Ablation (Qwen2VL / GPT4o-mini)

| Explicit CS | Relevance Level | GNN Confidence | SQA | MMMU | TextVQA |
|--------|-----------|----------|------|------|---------|
| ✗ | ✗ | ✗ | 71.39 / 76.45 | 51.10 / 55.89 | 75.30 / 78.98 |
| ✓ | ✗ | ✗ | 75.11 / 80.07 | 56.00 / 59.30 | 78.50 / 81.73 |
| ✗ | ✗ | ✓ | 72.88 / 77.02 | 53.41 / 57.64 | 76.42 / 79.55 |
| ✓ | ✓ | ✗ | 75.95 / 81.22 | 57.42 / 60.87 | 78.90 / 82.13 |
| ✓ | ✓ | ✓ | **77.12 / 82.50** | **58.72 / 61.03** | **79.80 / 83.37** |

### Table 3: Comparison with Knowledge-Enhanced Baselines

| Model | Knowledge Source | A-OKVQA | VCR |
|------|--------|---------|-----|
| VLC-BERT | COMET | 38.05 | 79.24 |
| KAT | Wikidata+GPT3 | 49.74 | 83.18 |
| KRISP | Wikidata+CNet | 27.10 | 65.12 |
| MAGIC-VQA (GPT4o-mini) | ATOMIC2020 | **76.55** | **93.42** |

## Key Findings

- **Image-driven commonsense contributes the most**: CS-I (image-related commonsense) yields the most significant improvement across all models and datasets. For instance, LLaVA1.6 on MMMU jumps from 48.38% to 53.52%, indicating that image-aligned commonsense provides more grounded reasoning clues.
- **Commonsense types need dataset-specific customization**: ScienceQA benefits the most from PE knowledge (scientific concepts), TextVQA benefits the most from EC knowledge (contextual understanding), and MMMU requires a balanced distribution.
- **Explicit and implicit knowledge are complementary**: Using GNN confidence alone also brings improvements (MMMU increases from 51.10% to 53.41%), but combining it with explicit knowledge yields the best results (58.72%), showing that they capture different dimensions of commonsense information.
- **Concrete objects benefit more than abstract concepts**: In TextVQA, performance gains in concrete categories like "uniform" and "books" are larger than those in abstract categories like "persons".
- **Easy questions benefit more than hard ones**: In MMMU, easy-level questions improve significantly, whereas hard-level questions require complex reasoning that goes beyond commonsense.

## Highlights & Insights

- **Extreme Lightweightness**: The entire framework has only 0.33M trainable parameters (the GNN part). Compared to LLaVA's 7B or GPT-4's 175B+, the parameter count is reduced by tens of thousands of times, enabling rapid adaptation to new LVLMs.
- **Plug-and-play Design**: It does not modify the LVLM itself and requires no fine-tuning or pre-training. It simply assembles the input prompt using external knowledge retrieval and confidence scores generated by the GNN, structurally decoupling knowledge acquisition from model capacity.
- **Insights on Type-based Filtering**: Different tasks vary greatly in their requirement for types of commonsense. Brute-force injection of all knowledge can easily introduce noise; thus, customizing the filtering ratios according to the intrinsic distribution of the datasets is crucial.
- **Soft Signals for Relevance Levels**: Labeling with High/Medium/Low instead of direct truncation allows the LVLM to judge the reliability of knowledge itself, serving as an elegant uncertainty propagation mechanism.

## Limitations & Future Work

- **Reliance on a Fixed Knowledge Graph**: Using ATOMIC2020 as the sole external knowledge source limits coverage; it may fail when encountering specialized domain knowledge not indexed in the graph.
- **Limitations of Predefined Commonsense Categories**: The tripartite division of PE/EC/SI is relatively coarse and may not accurately match the knowledge requirements of all VQA scenarios.
- **Value of GNN is Relatively Small Compared to Explicit Knowledge**: Ablation studies show that using GNN confidence alone brings limited improvements (1-2 percentage points), making its cost-effectiveness debatable.
- **Manual Setting of Commonsense Type Ratios**: Although type-based filtering is proposed, the optimal ratios must be determined through experimental search or GPT-4 analysis, limiting the level of automation.
- **Experiments Mostly Focused on Multiple-Choice Benchmarks**: ScienceQA and MMMU are in multiple-choice formats, while TextVQA uses the validation set, lacking validation in open-ended VQA or more complex scenarios.

## Related Work & Insights

- **VLC-BERT (Ravi et al. 2023)**: Encodes commonsense knowledge as additional word features to fine-tune VL-BERT, but requires modifying the model architecture. MAGIC-VQA's plug-and-play design is more flexible.
- **MM-CoT / KAM-CoT**: Fine-tunes models on CoT data to inject commonsense, but requires substantial training data and computational resources. MAGIC-VQA's zero-shot manner is more efficient.
- **VQA-GNN (Wang et al. 2022)**: Uses GNNs for multimodal semantic graph reasoning but is not integrated with LVLMs. MAGIC-VQA is more scalable by using the GNN as an auxiliary signal for the LVLM.
- **Insights**: Injecting structured knowledge into LVLMs via "soft prompts" is a paradigm worth exploring—it preserves the generalization capability of LVLMs while supplementing their missing concrete knowledge.

## Rating

- Novelty: ⭐⭐⭐⭐ — The three-stage framework design is highly systematic; the type-based filtering and relevance level assignment mechanisms are novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers 5 LVLMs, 5 datasets, and multidimensional ablation studies, with thorough quantitative and qualitative analyses.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured, with detailed tables and intuitive framework diagrams.
- Value: ⭐⭐⭐⭐ — The plug-and-play solution requiring only 0.33M parameters is highly practical, offering great reference value for knowledge-enhanced VQA.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MTVQA: Benchmarking Multilingual Text-Centric Visual Question Answering](mtvqa_benchmarking_multilingual_text-centric_visual_question_answering.md)
- [\[CVPR 2026\] CC-VQA: Conflict- and Correlation-Aware Method for Mitigating Knowledge Conflict in Knowledge-Based Visual Question Answering](../../CVPR2026/multimodal_vlm/cc-vqa_conflict-_and_correlation-aware_method_for_mitigating_knowledge_conflict_.md)
- [\[ACL 2025\] WikiMixQA: A Multimodal Benchmark for Question Answering over Tables and Charts](wikimixqa_a_multimodal_benchmark_for_question_answering_over_tables_and_charts.md)
- [\[CVPR 2026\] SEA: Evaluating Sketch Abstraction Efficiency via Element-level Commonsense Visual Question Answering](../../CVPR2026/multimodal_vlm/sea_evaluating_sketch_abstraction_efficiency_via_element-level_commonsense_visua.md)
- [\[CVPR 2026\] VQ-VA World: Towards High-Quality Visual Question-Visual Answering](../../CVPR2026/multimodal_vlm/vq-va_world_towards_high-quality_visual_question-visual_answering.md)

</div>

<!-- RELATED:END -->
