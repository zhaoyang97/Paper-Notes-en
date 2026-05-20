---
title: >-
  [Paper Note] Mastering Collaborative Multi-modal Data Selection: A Focus on Informativeness, Uniqueness, and Representativeness
description: >-
  [ICCV 2025][Multimodal VLM][Data selection] This paper proposes DataTailor — a collaborative multimodal data selection framework grounded in three principles: informativeness, uniqueness…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "Data selection"
  - "instruction tuning"
  - "multimodal large language models"
  - "informativeness"
  - "uniqueness"
  - "representativeness"
date: 2026-05-08
content_hash: 8bcddf41acf705d3
---

# Mastering Collaborative Multi-modal Data Selection: A Focus on Informativeness, Uniqueness, and Representativeness

**Conference**: ICCV 2025
**arXiv**: [2412.06293](https://arxiv.org/abs/2412.06293)  
**Code**: Available (URL provided in the paper)  
**Area**: Multimodal VLM
**Keywords**: Data selection, instruction tuning, multimodal large language models, informativeness, uniqueness, representativeness

## TL;DR

This paper proposes DataTailor — a collaborative multimodal data selection framework grounded in three principles: informativeness, uniqueness, and representativeness. Using only 15% of the data, DataTailor achieves 101.3% of the performance obtained with full-data fine-tuning, embodying the "Less is More" philosophy.

## Background & Motivation

Instruction tuning is critical for enhancing the instruction-following capabilities of MLLMs. However, the rapid expansion of visual instruction datasets has introduced severe **data redundancy**:

**Growing computational costs**: Large-scale but low-quality instruction data makes fine-tuning extremely time-consuming.

**Limitations of existing data selection methods**:
   - **Data-specific methods** (e.g., IFD): Rely heavily on manually designed rules, lacking flexibility and robustness.
   - **Human feedback methods** (e.g., InsTag): Time-consuming and expensive.
   - **Gradient-based methods** (e.g., LESS, TIVE): Require additional training on downstream tasks, resulting in high overall computational cost.

**Neglect of inter-sample relationships**: Most existing methods evaluate individual sample quality in isolation, overlooking inter-sample similarity and noise relationships.

The authors advocate for a systematic perspective on data selection: a valuable sample should simultaneously satisfy three conditions — containing rich task information (informativeness), being distinct from other samples (uniqueness), and representing the overall data distribution rather than being an outlier (representativeness).

## Method

### Overall Architecture

DataTailor consists of four steps: (1) computing the informative value of each sample; (2) computing the unique value within the intra-cluster space; (3) computing the representative value in the inter-cluster space; and (4) adaptively integrating the three values for collaborative data selection.

### Key Designs

1. **Informative Value Estimation**:
   Based on information theory and spectral analysis, SVD decomposition is used to measure the information density of each sample. For sample $s_i$, a unified feature matrix $\mathbf{M_i} \in \mathbb{R}^{L_i \times d}$ is extracted from the penultimate layer and decomposed via SVD to obtain singular values $\{\sigma_j\}_{j=1}^{L_i}$. The informative value is defined as the entropy of the normalized singular values:
    $V_i^{Inf} = -\sum_{j=1}^{L_i} \frac{\sigma_j}{\sum_k \sigma_k} \log \frac{\sigma_j}{\sum_k \sigma_k}$
   Intuitively, simple samples exhibit high linear correlation among feature matrix columns, where a few singular values dominate and entropy is low; informative samples have a more uniform singular value distribution and thus higher entropy.

2. **Unique Value Estimation**:
   Similar samples are first grouped via **cross-modal domain clustering** using Ward's criterion for hierarchical clustering, with the merging criterion:
    $\Delta \text{SSE} = \frac{n_A \cdot n_B}{n_A + n_B} \cdot \|\boldsymbol{\mu}_A - \boldsymbol{\mu}_B\|_2$
   Clustering terminates when $\Delta\text{SSE}$ exceeds threshold $\lambda \cdot \Delta\text{SSE}_{\max}$ ($\lambda=0.1$). Unique value is then computed within the intra-cluster space — samples farther from others within the cluster receive higher uniqueness scores:
    $V_i^{Uni} = \sum_{s_j \in \mathbf{C}, j \neq i} \|\mathbf{p_j} - \mathbf{p_i}\|_2 \cdot \frac{V_j^{Inf}}{\sum_{k \in \mathbf{C}} V_k^{Inf}}$
   Samples with higher informative value receive greater weight in the distance computation.

3. **Representative Value Estimation**:
   Representative value is assessed in the inter-cluster space by measuring the connectivity between a sample's cluster and other clusters, thereby preventing the selection of outliers from isolated noise clusters:
    $\tau_i^c = \frac{1}{K-1} \sum_{k \neq c}^{K} \exp(\text{sim}(\overline{\mathbf{p_k}}, \overline{\mathbf{p_c}}))$
    $V_i^{Rep} = \tau_i^c \cdot \frac{V_i^{Inf}}{\sum_{k \in \mathbf{C}} V_k^{Inf}}$
   The last token's feature is used as the sample representation, as it aggregates all visual and textual features via cross-attention.

4. **Adaptive Collaborative Selection**:
   The weights of the three values are adaptively adjusted based on the number of dialogue turns in an instruction:
    $V_i = \frac{r_i}{r_i + 2} \cdot V_i^{Inf} + \frac{1}{r_i + 2} \cdot (V_i^{Uni} + V_i^{Rep})$
   Multi-turn instructions (larger $r_i$) emphasize informative value, while single-turn instructions place greater weight on uniqueness and representativeness.

   Additionally, the per-task selection ratio $k_p$ is adaptively determined based on the proportion of the maximum singular value per task:
    $k_p = \frac{x_p^2 \cdot |S_p|}{\sum_q x_q^2 \cdot |S_q|} \cdot k, \quad x_p = \text{avg}\left(\frac{\sigma_{\max}}{\sum_j \sigma_j}\right)$

### Loss & Training

DataTailor is a data selection method and introduces no additional training loss. The selected data is directly used for LoRA fine-tuning of MLLMs.

## Key Experimental Results

### Main Results (LLaVA-v1.5-7B on LLaVA-mix-665k)

| Method | Data Size | MME-P | SEED-I | POPE | MM-Vet | SciQA | VQA-v2 | TextVQA | Relative Perf. |
|--------|-----------|-------|--------|------|--------|-------|--------|---------|----------------|
| LLaVA-v1.5 (Full) | 665k | 1476.9 | 67.4 | 86.4 | 30.9 | 70.0 | 79.1 | 58.2 | 100.0% |
| Random | 50k | 1387.5 | 59.7 | 85.7 | 29.5 | 70.0 | 73.7 | 53.1 | 95.3% |
| IFD | 50k | 1113.4 | 55.1 | 76.7 | 27.6 | 48.2 | 64.2 | 43.6 | 87.3% |
| TIVE | 50k | 1334.8 | 62.2 | 85.9 | 30.2 | 71.4 | 73.8 | 51.1 | 94.6% |
| COINCIDE | 133k | 1495.6 | - | 86.1 | - | 69.2 | 76.5 | 55.6 | 98.0% |
| ICONS | 133k | 1485.7 | - | 87.5 | 29.7 | 70.8 | 76.3 | 55.6 | 98.8% |
| **DataTailor** | **50k** | **1461.2** | **61.7** | **82.1** | **30.4** | **70.9** | **75.0** | **53.1** | **100.1%** |
| **DataTailor** | **100k** | **1476.2** | **63.6** | **85.3** | **31.8** | **71.0** | **76.7** | **55.7** | **101.3%** |

### Ablation Study

| Configuration | MME | MMMU(val) | SciQA | Relative Perf. |
|---------------|-----|-----------|-------|----------------|
| Full Data (100%) | 1744.8 | 32.8 | 70.0 | 100.0% |
| Random (7.5%) | 1675.0 | 32.2 | 70.0 | 95.3% |
| $V_i^{Inf}$ only | 1759.3 | 34.9 | 70.2 | 98.0% |
| $V_i^{Uni}$ only | 1716.2 | 33.5 | 69.8 | 97.3% |
| $V_i^{Rep}$ only | 1771.4 | 33.8 | 68.5 | 97.5% |
| DataTailor (all three) | 1823.7 | 33.9 | 70.9 | 100.1% |
| w/o adaptive collaboration | 1770.2 | 34.0 | 70.2 | 98.8% |

### Key Findings

- **Severe data redundancy**: Randomly selecting a small subset (7.5%) already achieves 95%+ performance, and in some cases the subset even outperforms full-data training.
- **Complementarity of the three principles**: Using any single principle alone is inferior to their combination, validating the design rationale.
- **Effectiveness of adaptive mechanisms**: Removing adaptive collaboration or adaptive ratio allocation consistently degrades performance.
- **Strong transferability**: Data selected using LLaVA-7B transfers effectively to mPLUG-Owl-7B and Bunny-3B.
- **"Less is More"**: 15% of the data achieves 101.3% performance, demonstrating that high-quality data genuinely surpasses large volumes of low-quality data.

## Highlights & Insights

- **Systematic three-principle framework**: For the first time, multimodal data is evaluated along three dimensions — informativeness, uniqueness, and representativeness — offering a more comprehensive perspective than prior single-criterion methods.
- **SVD entropy as a difficulty measure**: Using the entropy of singular value distributions to quantify information density is theoretically elegant and practically effective.
- **Adaptive weighting design**: Adjusting weights based on dialogue turns and task difficulty avoids laborious hyperparameter search.
- **Cross-model transferability**: Data selected via a proxy model generalizes effectively to target models of different architectures.

## Limitations & Future Work

- The clustering threshold $\lambda$ is claimed to be adaptive, but still requires setting according to the overall selection ratio.
- Hierarchical clustering may face computational efficiency challenges on very large-scale datasets.
- Only LoRA fine-tuning is evaluated; the effectiveness under full-parameter fine-tuning remains unknown.
- Selection thresholds (e.g., the specific layer chosen for SVD) may require adjustment for different model architectures.

## Related Work & Insights

- Compared to MLLM-specific methods such as TIVE and ICONS, DataTailor requires no additional downstream task training.
- The SVD-based analysis may inspire data quality assessment in other settings, such as pre-training data filtering.
- The three-principle framework is potentially generalizable to text data selection and other modalities.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The collaborative three-principle framework is novel and well-grounded; the use of SVD entropy has a solid theoretical basis.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers multiple datasets, models, ablations, and transferability analyses — very comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Logically clear with rich illustrations, though the high density of mathematical formulations requires careful reading.
- **Value**: ⭐⭐⭐⭐⭐ Offers significant practical value for MLLM training efficiency; the "Less is More" conclusion is convincing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] CoIDO: Efficient Data Selection for Visual Instruction Tuning via Coupled Importance-Diversity Optimization](../../NeurIPS2025/multimodal_vlm/coido_efficient_data_selection_for_visual_instruction_tuning_via_coupled_importa.md)
- [\[ICCV 2025\] METEOR: Multi-Encoder Collaborative Token Pruning for Efficient Vision Language Models](meteor_multi-encoder_collaborative_token_pruning_for_efficient_vision_language_m.md)
- [\[ICCV 2025\] Instruction-Oriented Preference Alignment for Enhancing Multi-Modal Comprehension Capability of MLLMs](instruction-oriented_preference_alignment_for_enhancing_multi-modal_comprehensio.md)
- [\[ICLR 2026\] Multi-modal Data Spectrum: Multi-modal Datasets are Multi-dimensional](../../ICLR2026/multimodal_vlm/multi-modal_data_spectrum_multi-modal_datasets_are_multi-dimensional.md)
- [\[ICCV 2025\] Bidirectional Likelihood Estimation with Multi-Modal Large Language Models for Text-Video Retrieval](bidirectional_likelihood_estimation_with_multi-modal_large_language_models_for_t.md)

</div>

<!-- RELATED:END -->
