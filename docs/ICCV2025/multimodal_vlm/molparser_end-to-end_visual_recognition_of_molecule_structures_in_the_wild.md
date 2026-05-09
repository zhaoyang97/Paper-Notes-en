---
title: >-
  [Paper Note] MolParser: End-to-end Visual Recognition of Molecule Structures in the Wild
description: >-
  [Multimodal VLM] This paper proposes MolParser, an end-to-end Optical Chemical Structure Recognition (OCSR) method that handles Markush structures via an extended SMILES representation (E-SMILES), constructs a large-scale training set MolParser-7M with 7 million samples, and incorporates real-world literature data through active learning. MolParser achieves 76.9% accuracy on the WildMol benchmark, significantly outperforming existing methods.
tags:
  - Multimodal VLM
date: 2026-05-08
content_hash: 7ac9bd45c9e133f6
---

# MolParser: End-to-end Visual Recognition of Molecule Structures in the Wild

## Paper Information

- **Conference**: ICCV 2025
- **arXiv**: 2411.11098
- **Code/Data**: Publicly available on HuggingFace (MolParser-7M dataset)
- **Area**: Multimodal VLM
- **Keywords**: OCSR, molecular recognition, SMILES, end-to-end, active learning, Markush structure

## TL;DR

This paper proposes MolParser, an end-to-end Optical Chemical Structure Recognition (OCSR) method that handles Markush structures via an extended SMILES representation (E-SMILES), constructs a large-scale training set MolParser-7M with 7 million samples, and incorporates real-world literature data through active learning. MolParser achieves 76.9% accuracy on the WildMol benchmark, significantly outperforming existing methods.

## Background & Motivation

A large amount of critical information in chemical literature and patents is presented as molecular structure diagrams. Automatically extracting machine-readable molecular structures (the OCSR task) is of substantial practical value. Existing methods face three major challenges:

**Representational limitations**: Standard SMILES cannot represent Markush structures (molecular families containing R-group variables), attachment points, abstract rings, or polymers — all of which are common in patent literature.

**Data scarcity**: The largest publicly available dataset contains only 300K synthetic samples (MolGrapher-300k), and the distributional gap between synthetic data and real-world literature images is significant.

**Poor in-the-wild robustness**: Molecular images in real patents and papers exhibit abbreviations, noise, blurriness, and diverse drawing styles, causing existing methods to perform poorly.

## Method

### Overall Architecture

MolParser frames OCSR as an image captioning task: given a molecular structure image, the model outputs an E-SMILES string. The model consists of three components:

- **Image Encoder**: ImageNet-pretrained Swin-Transformer (Tiny/Small/Base variants)
- **Feature Compressor**: A two-layer MLP serving as a visual-language connector, similar to LLaVA
- **SMILES Decoder**: A BART-Decoder that autoregressively generates E-SMILES sequences

### Key Design 1: Extended SMILES (E-SMILES)

The format is `SMILES<sep>EXTENSION`, where:
- **SMILES part**: Standard RDKit-compatible SMILES
- **EXTENSION part**: XML-like special tokens describing special functional groups
    - `<a>...</a>`: Markush R-groups and abbreviated groups
    - `<r>...</r>`: Ring connections at unspecified positions
    - `<c>...</c>`: Abstract rings
    - `<dum>`: Attachment points
- Functional group description format: `[INDEX]:[GROUP_NAME]`

E-SMILES is both RDKit-compatible and LLM-friendly, facilitating downstream analysis.

### Key Design 2: MolParser-7M Dataset

**Pre-training data (~7.7M)**:

| Subset | Proportion | Source |
|--------|-----------|--------|
| Markush-3M | 40% | PubChem with random group substitution |
| ChEMBL-2M | 27% | ChEMBL database |
| Polymer-1M | 14% | Randomly generated polymers |
| PAH-600k | 8% | Random polycyclic aromatic molecules |
| BMS-360k | 5% | Long carbon-chain molecules |
| MolGrapher-300K | 4% | MolGrapher paper data |
| Pauling-100k | 2% | Pauling-style images |

**Fine-tuning data (~600K)**: 66% manually annotated real data + 32% filtered synthetic data + 1% handwritten molecules.

### Key Design 3: Active Learning Data Engine

1. Train a YOLO11 detection model (MolDet) to localize molecules in PDFs; extract 20M molecular images from 1.22M real PDFs.
2. After deduplication, retain 4M images; train 5 models via 5-fold cross-training.
3. Generate 5 predictions per image and compute Tanimoto similarity scores as confidence measures.
4. Select samples with confidence in the range 0.6–0.9 (challenging yet informative) for manual annotation.
5. Use model predictions as pre-annotations, reducing annotation time from 3 minutes to 30 seconds per sample (90% labor saving).
6. Update the model every 80K annotations and repeat the cycle, ultimately yielding 400K high-quality annotations.

### Loss & Training: Curriculum Learning

The pre-training phase progressively increases difficulty: starting with simple molecules (token < 60) without data augmentation → gradually increasing augmentation strength and molecular complexity → fine-tuning with real-world data.

## Key Experimental Results

### Main Results: Cross-Benchmark Comparison

| Method | USPTO | UoB | CLEF | JPO | ColoredBG | USPTO-10K | WildMol-10K |
|--------|-------|-----|------|-----|-----------|-----------|-------------|
| OSRA 2.1 | 89.3 | 86.3 | 93.4 | 56.3 | 5.5 | 89.7 | 26.3 |
| MolGrapher | 91.5 | 94.9 | 90.5 | 67.5 | 7.5 | 93.3 | 45.5 |
| DECIMER 2.7 | 59.9 | 88.3 | 72.0 | 64.0 | 14.5 | 82.4 | 56.0 |
| MolScribe | 93.1 | 87.4 | 88.9 | 76.2 | 21.0 | 96.0 | 66.4 |
| **MolParser-Base** | **93.0** | **91.8** | **90.7** | **78.9** | **57.0** | **94.5** | **76.9** |

- On the most challenging WildMol-10K (real patent molecules), MolParser (76.9%) substantially outperforms MolScribe (66.4%) and MolGrapher (45.5%).
- The gain on the ColoredBG dataset is particularly striking (57.0% vs. 21.0%).

### Ablation Study

| Training Data | Fine-tuning | WildMol-10K ↑ |
|--------------|-------------|---------------|
| MolGrapher-300k | — | 22.4 |
| MolParser-7M (pt) | — | 51.9 |
| MolParser-7M (pt+ft) | — | 75.9 |
| MolParser-7M (pt) | MolParser-7M (ft) | **76.9** |

| Data Augmentation | Curriculum Learning | WildMol-10K ↑ |
|-------------------|---------------------|---------------|
| ✗ | ✗ | 40.1 |
| ✓ | ✗ | 69.5 |
| ✓ | ✓ | **76.9** |

**Key Findings**:
- Training data scale is critical: expanding from 300K to 7M raises accuracy from 22.4% to 51.9%.
- Fine-tuning on real data contributes substantially: +25% gain (51.9 → 76.9).
- Curriculum learning yields an additional 7.4% improvement.

### Speed–Accuracy Pareto Front

| Model | Throughput (FPS) | WildMol-10K | WildMol-10K-M |
|-------|-----------------|-------------|---------------|
| MolParser-Tiny | 131.6 | 73.1 | 15.3 |
| MolParser-Small | 116.3 | 76.3 | 34.8 |
| MolParser-Base | 39.8 | 76.9 | 38.1 |
| MolGrapher | 2.2 | 45.5 | — |

MolParser-Tiny is 60× faster than MolGrapher while achieving 27.6% higher accuracy.

### Additional Finding: Molecular Property Prediction

The Swin-T visual encoder trained by MolParser can serve as a molecular fingerprint extractor. On the MoleculeNet benchmark, it achieves performance comparable to 2D/3D graph neural network methods (mean ROC-AUC 73.7 vs. best 74.5), demonstrating that OCSR training captures chemically meaningful semantic features.

## Highlights & Insights

1. **E-SMILES is practical and elegant**: It supports complex Markush structures while maintaining RDKit compatibility — a significant engineering extension of the SMILES standard.
2. **Active learning data engine**: The confidence selection strategy (0.6–0.9) is particularly well-motivated — samples below the threshold are of low quality, while those above are already well-handled by the model.
3. **Data scale vs. model scale**: Experiments demonstrate that data scale and real-world data matter far more than model parameter count.
4. **Unexpected finding**: The OCSR-pretrained visual encoder retains rich chemical semantic information, enabling direct use for molecular property prediction.

## Limitations & Future Work

- Chirality recognition remains underexploited.
- Markush structure recognition accuracy is still relatively low (38.1%), requiring more annotated data.
- Larger end-to-end models (e.g., Mini-InternVL, 2.2B parameters) are harder to train and underperform smaller counterparts.
- Decoding to E-SMILES strings may lack robustness for very long molecules.

## Related Work & Insights

- **End-to-end vs. graph reconstruction methods**: End-to-end methods are faster but require large amounts of training data; MolParser addresses this limitation through the data engine and the 7M dataset.
- **Relationship to LLaVA architecture**: MolParser adopts a similar visual-language connector design.
- **Extension to chemical reaction parsing**: Combined with GPT-4o for reaction equation recognition, this work demonstrates MolParser's broader value as a foundational component.

## Rating

⭐⭐⭐⭐ — A highly systematic piece of work that forms a complete loop from representation (E-SMILES) to data (7M) to model (end-to-end) to application (reaction parsing + molecular fingerprints). The active learning data engine is the core innovation. Markush structure recognition still has considerable room for improvement.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] WebDS: An End-to-End Benchmark for Web-based Data Science](../../ICLR2026/multimodal_vlm/webds_an_end-to-end_benchmark_for_web-based_data_science.md)
- [\[ICCV 2025\] Hints of Prompt: Enhancing Visual Representation for Multimodal LLMs in Autonomous Driving](hints_of_prompt_enhancing_visual_representation_for_multimodal_llms_in_autonomou.md)
- [\[ICCV 2025\] Large Multi-modal Models Can Interpret Features in Large Multi-modal Models](large_multi-modal_models_can_interpret_features_in_large_multi-modal_models.md)
- [\[ICCV 2025\] SCAN: Bootstrapping Contrastive Pre-training for Data Efficiency](scan_bootstrapping_contrastive_pre-training_for_data_efficiency.md)
- [\[ICCV 2025\] MM-IFEngine: Towards Multimodal Instruction Following](mm-ifengine_towards_multimodal_instruction_following.md)

<!-- RELATED:END -->
