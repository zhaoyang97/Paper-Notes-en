---
title: >-
  [Paper Note] A Standardized Benchmark for Multilabel Antimicrobial Peptide Classification
description: >-
  [NeurIPS 2025][LLM Evaluation][Antimicrobial Peptide] This paper presents **ESCAPE**—the first standardized multilabel antimicrobial peptide classification benchmark, integrating 80,000+ peptides from 27 public databases…
tags:
  - "NeurIPS 2025"
  - "LLM Evaluation"
  - "Antimicrobial Peptide"
  - "Multilabel Classification"
  - "benchmark"
  - "Transformer"
  - "Cross-Attention"
  - "Drug Discovery"
date: 2026-05-08
content_hash: e2dc4e18044ed76c
---

# A Standardized Benchmark for Multilabel Antimicrobial Peptide Classification

**Conference**: NeurIPS 2025
**arXiv**: [2511.04814](https://arxiv.org/abs/2511.04814)  
**Authors**: Sebastian Ojeda, Rafael Velasquez, Nicolás Aparicio, Juanita Puentes, Paula Cárdenas, Nicolás Andrade, Gabriel González, Sergio Rincón, Carolina Muñoz-Camargo, Pablo Arbeláez (Universidad de los Andes, Colombia)
**Area**: LLM Evaluation
**Keywords**: Antimicrobial Peptide, Multilabel Classification, benchmark, Transformer, Cross-Attention, Drug Discovery

## TL;DR

This paper presents **ESCAPE**—the first standardized multilabel antimicrobial peptide classification benchmark, integrating 80,000+ peptides from 27 public databases, along with a dual-branch Transformer + bidirectional cross-attention baseline model that achieves a 2.56% relative improvement in mAP over the second-best method.

## Background & Motivation

**Antimicrobial Resistance (AMR) Crisis**: AMR infections are estimated to cause over 39 million deaths between 2025 and 2050, making the discovery of alternative molecules such as antimicrobial peptides (AMPs) an urgent priority.

**Potential of AMPs**: AMPs act through mechanisms that are difficult for pathogens to evade—such as membrane disruption and cell wall synthesis inhibition—conferring a lower resistance risk compared to conventional antibiotics.

**Bottleneck for AI-Accelerated AMP Discovery**: Most existing AI approaches frame the task as binary classification (AMP vs. non-AMP), overlooking the inherently multilabel nature of AMPs, which can simultaneously exhibit activity against multiple microorganisms (bacteria, fungi, viruses, and parasites).

**Data Fragmentation**: Existing databases differ substantially in format, annotation standards, and functional category granularity (e.g., dbAMP has 58 categories vs. only 8 in DRAMP), precluding fair cross-dataset model comparisons.

**Absence of a Standard Benchmark**: Most studies employ custom datasets and splits, hindering reproducibility and fair methodological comparison.

**Multilabel Gap**: Although a small number of multilabel approaches exist (e.g., AMPs-Net, TransImbAMP), no unified multilabel AMP benchmark is available to the community.

## Method

### Overall Architecture: ESCAPE (Dataset + Benchmark + Baseline Model)

ESCAPE contributes at three levels:

- **ESCAPE Dataset**: Compiles, cleans, and standardizes 80,000+ peptides from 27 public AMP databases into a unified 5-class multilabel scheme (antibacterial / antifungal / antiviral / antiparasitic / antimicrobial), supplemented with Non-AMP negative samples.
- **ESCAPE Benchmark**: Provides a fair multilabel evaluation of 7 representative methods on the unified dataset, using 2-fold cross-validation with a held-out test set and averaging results over 3 random seeds.
- **ESCAPE Baseline**: A dual-branch Transformer architecture that integrates sequence and structural information for multilabel classification.

### Key Design 1: Data Compilation and Cleaning Pipeline

- Experimentally validated AMP sequences covering four antimicrobial activities are collected from 27 databases.
- Non-AMP negative samples are obtained by applying UniProt keyword exclusion (removing entries associated with terms such as membrane, toxic, and antibiotic) and incorporating sequences from known non-antimicrobial datasets.
- Cleaning rules: sequences containing synthetic residues (O, U, Bal, etc.) or undefined amino acids (X) are removed; peptides are retained if their length falls within 5–250 residues; duplicate sequences across databases are merged with their multilabel annotations consolidated.
- Final dataset: 60,950 Non-AMP and 21,409 AMP sequences, stratified by label into a 2-fold cross-validation split plus a test set.

### Key Design 2: Dual-Branch Transformer Encoder

**Sequence Branch**:
- Amino acid sequences are tokenized (vocabulary size 27), padded or truncated to a fixed length of 200.
- Embedding dimension is 256; a [CLS] token and positional encodings are added.
- A 4-layer Transformer encoder with 8-head attention is applied.

**Structure Branch**:
- 3D structures (experimental structures from UniProt/PDB or structures predicted by RosettaFold/AlphaFold3) are used to compute a Cα inter-atomic distance matrix $\mathcal{M} \in \mathbb{R}^{N \times N}$.
- The matrix is resized to 224×224 and partitioned into non-overlapping patches via 2D convolution (kernel=16, stride=16), with each patch projected to 192 dimensions.
- A [CLS] token and positional encodings are similarly added, followed by a 4-layer 8-head Transformer encoder.

### Key Design 3: Bidirectional Cross-Attention Fusion

- The sequence-side [CLS] token serves as Query to attend over all structure-side tokens (Key/Value), modeling how sequence representations are enriched by structural context.
- The reverse direction applies symmetrically, with the structure-side [CLS] attending over sequence-side Key/Value pairs.
- Residual connections and feed-forward networks refine both updated [CLS] vectors, which are then concatenated and passed through a linear classification head to produce 5-dimensional multilabel predictions.

### Loss & Training

- Loss function: multilabel classification (not explicitly specified in the paper; Binary Cross-Entropy is presumed).
- Optimizer: AdamW, learning rate $1 \times 10^{-4}$, batch size 64, trained for 100 epochs.
- Evaluation: two models are trained via 2-fold cross-validation and their predicted probabilities are averaged at inference; results are reported as mean ± standard deviation over 3 random seeds (42, 1665, 8914).
- Metrics: mAP and F1-score.

## Key Experimental Results

### Tables 1 & 2: Main ESCAPE Benchmark Results

| Method | mAP (%) | F1 (%) | Antiparasitic AP (%) |
|--------|---------|--------|----------------------|
| AMPs-Net | 54.6±0.86 | 57.7±0.70 | 5.3±0.67 |
| TransImbAMP | 64.9±1.11 | 62.0±0.70 | 16.7±0.86 |
| AMP-BERT | 66.9±1.17 | 64.7±0.64 | 21.4±2.61 |
| amPEPpy | 68.5±0.48 | 66.5±0.37 | 23.8±1.61 |
| PEP-Net | 68.4±0.53 | 65.5±0.61 | 16.2±0.84 |
| AVP-IFT | 68.8±0.50 | 66.5±0.59 | 20.0±4.25 |
| AMPlify | 70.3±0.87 | 68.5±0.77 | 27.7±1.33 |
| **ESCAPE Baseline** | **72.1±0.60** | **69.8±0.43** | **37.6±2.87** |

### Ablation Study

| Structure Module | Sequence Module | Cross-Attention | mAP (%) | F1 (%) |
|-----------------|-----------------|-----------------|---------|--------|
| ✓ | - | - | 47.7 | 46.9 |
| - | ✓ | - | 69.4 | 67.6 |
| ✓ | ✓ | ✓ | **72.7** | **69.5** |

### Key Findings

1. **ESCAPE Baseline achieves comprehensive superiority**: mAP of 72.1%, a 2.56% relative improvement over the second-best method AMPlify (70.3%); F1 of 69.8%, a 1.90% relative improvement.
2. **Substantial gains on rare categories**: AP for the antiparasitic class jumps from 27.7% (AMPlify) to 37.6%, a relative improvement of **35.7%**.
3. **Sequence >> Structure**: The sequence-only branch achieves 69.4% mAP, far exceeding the structure-only branch at 47.7% (a gap of 21.7%), indicating that amino acid identity is the dominant factor for classification.
4. **Structure as complementary modality**: Incorporating structural information via bidirectional cross-attention fusion raises mAP from 69.4% to 72.7%, an additional gain of 3.3 percentage points.
5. **Model size ≠ performance**: The second-ranked method amPEPpy is based on random forests and is the least computationally demanding; BERT-based models (TransImbAMP, AMP-BERT) rank lower, suggesting limitations in transferring large language models to non-natural-language domains.
6. **Predicted vs. experimental structures**: Using predicted structures alone results in a 1.5% drop in mAP and a 1.9% drop in F1, demonstrating that noise introduced by structure prediction degrades model performance.

## Highlights & Insights

1. **Filling a critical gap**: ESCAPE is the first standardized multilabel AMP benchmark integrating 27 databases and 80,000+ peptides, genuinely addressing data fragmentation and annotation inconsistency.
2. **Well-designed label taxonomy**: The diverse functional categories from various databases are unified into a biologically meaningful 4+1 hierarchical scheme that balances discriminability and interpretability.
3. **Bidirectional cross-attention outperforms simple concatenation**: Allowing sequence and structure representations to mutually attend to each other extracts complementary information more effectively than naive feature concatenation or unimodal encoding.
4. **Fair evaluation reveals important insights**: "Larger models are not necessarily better"—random forests (amPEPpy) can match or surpass BERT-based methods, a finding of practical value in the AMP-specific domain.
5. **Marked improvement on rare categories**: Antiparasitic AP rises from 5.3% (AMPs-Net) to 37.6%, demonstrating the combined effect of a well-constructed benchmark and a strong method.

## Limitations & Future Work

1. **Data distribution representativeness**: The diversity of peptides in nature far exceeds the dataset's coverage; 80K samples may not reflect the true underlying distribution.
2. **Sequence length bias**: AMPs are naturally short (~30 aa) while Non-AMPs are longer (~90 aa), creating a potential classification shortcut based on length alone.
3. **Dependence on structure prediction**: Some peptides rely on AlphaFold3/RosettaFold-predicted structures, introducing additional error (mAP decreases by 1.5%).
4. **Severe class imbalance**: The antiparasitic class contains only 417 entries (130 unique), and all methods still have substantial room for improvement on this label.
5. **Lack of wet-lab validation**: Model predictions require experimental validation before being translated into practical drug discovery applications.
6. **Relatively simple baseline**: More advanced pretrained protein language models (e.g., ESM-2 as a backbone) or contrastive learning strategies have not been explored.

## Related Work & Insights

- **AMP Databases**: dbAMP (33K peptides/58 classes), DRAMP (30K/8 classes), LAMP2 (23K/38 classes), SATPdb (19K/10 classes), among others, each with differing scope and annotation granularity.
- **Sequence-based Methods**: AMPlify (BiLSTM + attention), AMP-BERT / TransImbAMP (pretrained BERT), dsAMPGAN (CNN + Attention + BiLSTM), AMPpred-DLFF (ESM-2 + GAT + CNN).
- **Feature-augmented Methods**: amPEPpy (CTD features + Random Forest), AMPs-Net (graph + physicochemical features + GNN), PEP-Net (one-hot + physicochemical + PLM embeddings + Transformer), AVP-IFT (contrastive learning + physicochemical features).
- **Distinction from prior work**: ESCAPE is the first unified multilabel benchmark; the baseline's novelty lies in bidirectional cross-attention fusion of amino acid sequences and 3D distance matrices as two complementary modalities.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The first large-scale standardized multilabel AMP benchmark filling an important gap; the bidirectional cross-attention mechanism introduces a new multimodal fusion paradigm for this domain.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Fair comparison of 7 methods, 3 random seeds, ablation studies, and sensitivity analysis on predicted structures constitute a highly comprehensive evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, well-motivated problem statement, and rich figures and tables; certain technical details (e.g., loss function selection) could be stated more explicitly.
- **Value**: ⭐⭐⭐⭐⭐ — Provides foundational infrastructure for AI-driven AMP research; the dataset and benchmark will meaningfully advance community development.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] PARROT: A Benchmark for Evaluating LLMs in Cross-System SQL Translation](parrot_a_benchmark_for_evaluating_llms_in_cross-system_sql_translation.md)
- [\[NeurIPS 2025\] RGB-to-Polarization Estimation: A New Task and Benchmark Study](rgb-to-polarization_estimation_a_new_task_and_benchmark_study.md)
- [\[NeurIPS 2025\] CLIMB: Class-Imbalanced Learning Benchmark on Tabular Data](climb_class-imbalanced_learning_benchmark_on_tabular_data.md)
- [\[NeurIPS 2025\] HouseLayout3D: A Benchmark and Training-Free Baseline for 3D Layout Estimation in the Wild](houselayout3d_a_benchmark_and_training-free_baseline_for_3d_layout_estimation_in.md)
- [\[NeurIPS 2025\] RDB2G-Bench: A Comprehensive Benchmark for Automatic Graph Modeling of Relational Databases](rdb2g-bench_a_comprehensive_benchmark_for_automatic_graph_modeling_of_relational.md)

</div>

<!-- RELATED:END -->
