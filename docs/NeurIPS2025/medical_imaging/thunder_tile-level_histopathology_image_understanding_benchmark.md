---
title: >-
  [Paper Note] THUNDER: Tile-level Histopathology image UNDERstanding benchmark
description: >-
  [NeurIPS 2025 Datasets and Benchmarks Track (Spotlight)][Medical Imaging][digital pathology] This paper presents THUNDER, a comprehensive tile-level benchmark for digital pathology foundation models…
tags:
  - "NeurIPS 2025 Datasets and Benchmarks Track (Spotlight)"
  - "Medical Imaging"
  - "digital pathology"
  - "benchmark"
  - "foundation models"
  - "robustness"
  - "uncertainty"
  - "tile-level analysis"
date: 2026-05-08
content_hash: 74b5e143a85fff31
---

# THUNDER: Tile-level Histopathology image UNDERstanding benchmark

**Conference**: NeurIPS 2025 Datasets and Benchmarks Track (Spotlight)
**arXiv**: [2507.07860](https://arxiv.org/abs/2507.07860)  
**Code**: Available ([https://github.com/MICS-Lab/thunder](https://github.com/MICS-Lab/thunder))  
**Area**: Medical Imaging / Digital Pathology
**Keywords**: digital pathology, benchmark, foundation models, robustness, uncertainty, tile-level analysis

## TL;DR

This paper presents THUNDER, a comprehensive tile-level benchmark for digital pathology foundation models, enabling efficient comparison of 23 foundation models across 16 datasets, covering downstream task performance, feature space analysis, robustness, and uncertainty estimation.

## Background & Motivation

### The Proliferation of Digital Pathology Foundation Models
In recent years, numerous digital pathology foundation models have been released (e.g., UNI, Virchow, CONCH, Phikon, CTransPath), serving as tile-level image feature extractors for various downstream tile-level and slide-level tasks. However, the pace of model release has far outstripped the community's understanding of their performance and differences.

### Limitations of Prior Work
- **Focus solely on downstream performance**: Overlooking fundamental differences between models (e.g., feature space structure)
- **Lack of robustness evaluation**: In safety-critical domains such as healthcare, accuracy alone is insufficient
- **Lack of uncertainty analysis**: Models must be capable of expressing uncertainty when appropriate
- **Poor reproducibility and extensibility**: Many model evaluations employ inconsistent protocols and data splits

### Paper Goals
To construct a **fast, accessible, and dynamic** benchmark that not only evaluates performance but also provides in-depth analysis of feature spaces, robustness, and uncertainty, offering the community a comprehensive understanding of model behavior.

## Method

### Overall Architecture

The THUNDER benchmark encompasses four evaluation dimensions:

```
Input: Foundation Models (extracting tile embeddings)
  ├── Dimension 1: Downstream Task Performance (classification, retrieval)
  ├── Dimension 2: Feature Space Analysis (structure, separability)
  ├── Dimension 3: Robustness Evaluation (distribution shift, adversarial perturbation)
  └── Dimension 4: Uncertainty Estimation (calibration, OOD detection)
```

### Key Designs

#### 1. Dataset Composition
THUNDER comprises **16 datasets** covering diverse tissue types and tasks:
- **Cancer classification**: Breast, lung, colorectal, gastric cancer, etc.
- **Tissue classification**: Normal tissue type recognition
- **Subtype classification**: Fine-grained tumor subtype classification
- **Cross-site evaluation**: Same task with data from different hospitals

#### 2. Evaluation Protocol
- **Linear probing**: Frozen foundation model with only a linear classification head trained
- **KNN classification**: K-nearest neighbor classification directly in the embedding space
- **Retrieval tasks**: Similar tile retrieval via embedding similarity
- **Unified data splits**: Identical train/validation/test sets used for all models

#### 3. Feature Space Analysis
- **t-SNE visualization**: Examining clustering quality of different classes in the embedding space
- **Inter-class/intra-class distance ratio**: Quantifying feature space separability
- **Embedding dimension utilization**: Analyzing the number of effectively used embedding dimensions

#### 4. Robustness & Uncertainty
- **Distribution shift**: Performance degradation evaluated on data from different sites/staining protocols
- **Uncertainty calibration**: Expected Calibration Error (ECE) for assessing reliability of predictive confidence
- **OOD detection**: Ability to distinguish in-distribution from out-of-distribution samples

### Loss & Training

THUNDER does not train models itself; evaluations employ:
- Linear probing: Cross-entropy loss + SGD optimizer
- KNN: No training required
- All foundation model parameters are frozen

## Key Experimental Results

### Main Results

Average tile-level linear probing performance of 23 foundation models across 16 datasets:

| Model | Pretraining Data Scale | Architecture | Embedding Dim | Mean Acc ↑ | Mean AUC ↑ |
|-------|----------------------|--------------|--------------|-----------|-----------|
| UNI (v1) | 100k slides | ViT-L | 1024 | 82.4 | 0.924 |
| Virchow | 1.5M slides | ViT-H | 1280 | 83.1 | 0.931 |
| CONCH | 1.17M slides | ViT-B | 512 | 80.7 | 0.912 |
| Phikon | 40k slides | ViT-B | 768 | 78.3 | 0.896 |
| CTransPath | 15k slides | Swin-T | 768 | 76.8 | 0.882 |
| Lunit-DINO | 33k slides | ViT-S | 384 | 77.5 | 0.889 |
| Prov-GigaPath | 171k slides | ViT-G | 1536 | **84.2** | **0.938** |
| UNI v2 | 350k slides | ViT-L | 1024 | 83.8 | 0.935 |
| ResNet-50 (ImageNet) | 1.2M imgs | ResNet-50 | 2048 | 68.2 | 0.812 |

Robustness evaluation (cross-site performance degradation):

| Model | Source Site Acc | Target Site Acc | Performance Drop ↓ | ECE ↓ |
|-------|----------------|----------------|-------------------|-------|
| UNI | 82.4 | 76.8 | -5.6 | 0.082 |
| Virchow | 83.1 | 78.2 | -4.9 | 0.071 |
| CONCH | 80.7 | 73.4 | -7.3 | 0.095 |
| Prov-GigaPath | 84.2 | **79.5** | **-4.7** | **0.068** |
| CTransPath | 76.8 | 68.1 | -8.7 | 0.112 |
| ResNet-50 | 68.2 | 58.4 | -9.8 | 0.145 |

### Ablation Study

Comparison of evaluation protocols:

| Evaluation Method | Accuracy Range | Computation Cost | Correlation with Full Fine-tuning |
|------------------|---------------|-----------------|----------------------------------|
| Linear probing | 68–84% | Fast (minutes) | r=0.92 |
| KNN (k=5) | 65–82% | Very fast (seconds) | r=0.88 |
| KNN (k=20) | 66–81% | Very fast (seconds) | r=0.86 |
| Few-shot (10-shot) | 55–72% | Fast | r=0.84 |

### Key Findings

1. **Pretraining data scale remains the primary factor**: Models pretrained on the largest datasets (Prov-GigaPath, Virchow) consistently achieve the best overall performance.
2. **Larger models are not necessarily more robust**: Certain large models exhibit greater performance degradation in cross-site settings.
3. **Uncertainty estimation is broadly inadequate**: Most models exhibit high calibration error, necessitating additional calibration steps for clinical deployment.
4. **Significant variation in feature space structure**: Different models show substantial differences in clustering quality and embedding dimension utilization.
5. **ImageNet pretraining is clearly insufficient**: General-purpose vision models lag considerably behind domain-specific models on pathology tasks.

## Highlights & Insights

1. **Comprehensiveness**: The first benchmark to simultaneously cover performance, feature space, robustness, and uncertainty for pathology models.
2. **Scale**: 23 models × 16 datasets = 368 evaluation combinations.
3. **Practicality**: Rapid execution, support for user-defined models, and dynamic extensibility.
4. **Spotlight acceptance**: Recognized as having significant reference value for the community.
5. **Open source**: Fully open-sourced to facilitate community reproduction and extension.

## Limitations & Future Work

1. **Tile-level only**: Slide-level tasks (e.g., WSI classification via MIL aggregation) are not covered.
2. **Limited evaluation protocols**: Primarily linear probing and KNN; methods such as prompt tuning are not included.
3. **Dataset bias**: Coverage is skewed toward common H&E-stained cancer types.
4. **Absence of multimodal evaluation**: The text-based capabilities of vision-language models (e.g., CONCH) are not assessed.
5. **Timeliness challenge**: Foundation models are evolving rapidly, requiring ongoing benchmark maintenance.

## Related Work & Insights

- **Pathology foundation models**: UNI, Virchow, CONCH, Phikon, etc.
- **General vision benchmarks**: ImageNet, VTAB, etc., inform benchmark design principles.
- **Directions for inspiration**: Developing comprehensive slide-level benchmarks, incorporating more rare disease data, and integrating clinical metrics for evaluation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First comprehensive tile-level pathology benchmark
- **Theoretical Depth**: ⭐⭐⭐ — Primarily an experimentally driven benchmark contribution
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 23 models × 16 datasets; exceptionally comprehensive
- **Practical Impact**: ⭐⭐⭐⭐⭐ — Significant reference value for the pathology AI community
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, easy to navigate

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] FGBench: A Dataset and Benchmark for Molecular Property Reasoning at Functional Group-Level in Large Language Models](fgbench_a_dataset_and_benchmark_for_molecular_property_reasoning_at_functional_g.md)
- [\[NeurIPS 2025\] CXReasonBench: A Benchmark for Evaluating Structured Diagnostic Reasoning in Chest X-rays](cxreasonbench_a_benchmark_for_evaluating_structured_diagnostic_reasoning_in_ches.md)
- [\[NeurIPS 2025\] Revisiting End-to-End Learning with Slide-level Supervision in Computational Pathology](revisiting_end-to-end_learning_with_slide-level_supervision_in_computational_pat.md)
- [\[AAAI 2026\] PanFoMa: A Lightweight Foundation Model and Benchmark for Pan-Cancer Pathology Image Analysis](../../AAAI2026/medical_imaging/panfoma_a_lightweight_foundation_model_and_benchmark_for_pan-cancer.md)
- [\[NeurIPS 2025\] Mol-LLaMA: Towards General Understanding of Molecules in Large Molecular Language Models](mol-llama_towards_general_understanding_of_molecules_in_large_molecular_language.md)

</div>

<!-- RELATED:END -->
