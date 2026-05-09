---
title: >-
  [Paper Note] PASER: Post-Training Data Selection for Efficient Pruned Large Language Model Recovery
description: >-
  [ICLR 2026][Model Compression][LLM pruning] This paper proposes PASER, a post-training data selection method for recovering pruned LLMs. It identifies capability-relevant instruction subsets via manifold learning and spectral clustering, and adaptively allocates data budgets according to the degree of capability degradation. Using only 4%–20% of the original data, PASER significantly outperforms full-data recovery.
tags:
  - ICLR 2026
  - Model Compression
  - LLM pruning
  - data selection
  - post-training recovery
  - manifold learning
  - capability degradation awareness
date: 2026-05-08
content_hash: b42e80fc1c123c1c
---

# PASER: Post-Training Data Selection for Efficient Pruned Large Language Model Recovery

**Conference**: ICLR 2026
**arXiv**: [2502.12594](https://arxiv.org/abs/2502.12594)
**Code**: Available
**Area**: Model Compression / LLM Efficiency
**Keywords**: LLM pruning, data selection, post-training recovery, manifold learning, capability degradation awareness

## TL;DR
This paper proposes PASER, a post-training data selection method for recovering pruned LLMs. It identifies capability-relevant instruction subsets via manifold learning and spectral clustering, and adaptively allocates data budgets according to the degree of capability degradation. Using only 4%–20% of the original data, PASER significantly outperforms full-data recovery.

## Background & Motivation

**Background**: Model pruning is an effective compression technique for LLMs, but inevitably causes capability degradation. The dominant recovery approach applies instruction fine-tuning data (e.g., Alpaca) in post-training. Conventional methods train on the full dataset, which is computationally expensive and not necessarily optimal.

**Limitations of Prior Work**:
   - Pruning affects different capabilities **unevenly** (e.g., mathematical reasoning may degrade severely while language modeling remains largely intact), yet existing methods ignore this heterogeneity.
   - Full-data recovery incurs high computational cost (e.g., LaMini contains 2.58M samples) and may introduce irrelevant or conflicting instructions, causing **negative fine-tuning effects**.
   - Random subset selection is unstable and highly sensitive to data composition.
   - Existing data selection methods (e.g., IFD, Nuggets) target general instruction quality and are not designed for the pruning recovery scenario.

**Key Challenge**: Efficiently recovering multiple capabilities with limited data is challenging because different capabilities require different amounts and types of data, and some data may produce negative effects.

**Goal**:
   - Identify instruction data groups corresponding to distinct LLM capabilities.
   - Adaptively allocate data budgets proportional to the degree of degradation.
   - Prioritize samples with the highest benefit-to-cost ratio within each group.
   - Filter conflicting or irrelevant data that may introduce negative effects.

**Key Insight**: The paper assumes that geometric structures in semantic space correspond to different LLM capabilities. Manifold learning is used to discover these structures, and the divergence between the output distributions of the original and pruned models (JSD) quantifies the degree of degradation.

**Core Idea**: Replace blind full-data training with *capability-aware data selection* to make the recovery of pruned LLMs more precise, efficient, and safe.

## Method

### Overall Architecture
Input: pruned model $M_p$ + original model $M_o$ + instruction dataset $D$
Step 1: Semantic Structure-aware Recovery Instruction Clustering (S2RIC) → $K$ capability clusters
Step 2: Capability Degradation-Aware Instruction Selection (CDAIS) → adaptive budget allocation + efficiency-driven sampling
Step 3: Negative Training Effect Mitigation (NTEM) → concept consistency graph filtering
Output: recovery training subset $S \subset D$, $|S| \leq B$

### Key Designs

1. **Semantic Structure-aware Recovery Instruction Clustering (S2RIC)**

    - Function: Group instruction data along capability dimensions.
    - Mechanism: Obtain instruction embeddings via SentenceBERT, apply a Diffusion Kernel for manifold learning to reduce dimensionality while preserving nonlinear structure, then apply NMF-based spectral clustering to discover natural groupings. The number of clusters $K$ is determined automatically by minimizing the NMF approximation error.
    - Design Motivation: Instructions requiring similar capabilities form identifiable topological structures in semantic space; the Diffusion Kernel preserves manifold geometry better than PCA or t-SNE.

2. **Capability Degradation-Aware Instruction Selection (CDAIS)**

    - Function: Allocate budgets according to degradation severity and sample by efficiency.
    - Capability Degradation Score (CDS): For each cluster $c_k$, the mean JSD between the output distributions of the original and pruned models is used to measure degradation.
    - Budget allocation: $n_k = \lfloor B \cdot \frac{\text{CDS}(c_k)}{\sum_j \text{CDS}(c_j)} \rfloor$; more data is allocated to more severely degraded capabilities.
    - Individual sample Efficiency Score (IES): $\text{IES}(x,y) = \frac{\text{JSD}_{avg}}{\log \text{ComputationalCost}(x,y)}$; samples with large degradation and low computational cost are prioritized.
    - Design Motivation: JSD captures full output distribution differences more comprehensively than loss values or accuracy; the logarithmic cost term prevents excessive penalization of high-potential samples.

3. **Negative Training Effect Mitigation (NTEM)**

    - Function: Detect and filter conceptually conflicting instructions.
    - Mechanism: Construct a Concept Consistency Graph (CCG) where vertices represent concepts and edges represent non-conflicting concept co-occurrences. A new sample is accepted only if its concepts are consistent with the CCG.
    - Enhancements: Semantic normalization (merging synonymous expressions), soft down-weighting of low-confidence samples, and optional NLI-based reranking.
    - Design Motivation: Introducing conflicting concepts during recovery training (e.g., contradictory answers to the same question) further damages the model; the filtering mechanism ensures data consistency.

### Loss & Training
- Standard instruction fine-tuning is applied after data selection.
- Time complexity: $O(N\log N + NC^2)$; since $C \ll N$ in practice, this simplifies to $O(N\log N)$.

## Key Experimental Results

### Main Results (LLaMA2-7B + LLM-Pruner 25% pruning + Alpaca data)

| Recovery Method | WikiText2↓ | BoolQ | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | OBQA | Avg. |
|-----------------|-----------|-------|------|-----------|------------|-------|-------|------|------|
| No Recovery | 20.34 | 61.87 | 76.61 | 65.86 | 60.22 | 63.13 | 37.37 | 39.40 | 57.78 |
| Full Data | 736.42 | 37.83 | 53.21 | 26.42 | 49.57 | 25.29 | 28.16 | 29.00 | 35.64 |
| Random | 93.77 | 57.61 | 64.37 | 45.39 | 55.87 | 43.78 | 31.94 | 34.90 | 47.69 |
| Nuggets | 20.02 | 63.62 | 77.43 | 67.36 | 61.08 | 63.77 | 37.64 | 39.90 | 58.69 |
| **PASER** | **16.40** | **67.25** | 77.29 | **68.98** | **66.97** | **67.84** | **39.54** | 39.80 | **61.10** |

Note: **Full-data recovery causes model collapse** (PPL: 20 → 736), whereas PASER with 20% of the data reduces PPL to 16.40 (even better than the unpruned model at 12.62), recovering average accuracy to 61.10 (close to the unpruned model's 62.91).

### Ablation Study

| Configuration | Key Findings |
|---------------|-------------|
| SliceGPT 25% + PASER | PPL: 44.53 → 12.24; avg. accuracy: 54.27 → 64.31, **surpassing the unpruned model** |
| Wanda 2:4 semi-structured + PASER | Avg. accuracy: 54.39 → 62.02, approaching full-parameter performance |
| SparseGPT 50% + PASER | Avg. accuracy: 59.93 → 61.62, consistent improvement |
| LaMini 2.58M data + PASER | Only 4% of data matches or exceeds full-data training |
| w/o S2RIC (uniform budget) | Performance drops 2–4%, validating the necessity of capability-aware allocation |
| w/o NTEM (no conflict filtering) | Performance drops 1–3%, validating the value of negative effect mitigation |

### Key Findings
- **Full-data recovery can be harmful**: Under the LLM-Pruner + Alpaca setting, full-data recovery causes complete model collapse (PPL > 700), demonstrating the risk of indiscriminate training.
- **4%–20% curated data > 100% full data**: PASER outperforms full-data training across all pruning configurations with a small data fraction.
- **SliceGPT + PASER surpasses the unpruned model**: The most striking result—pruning followed by precise recovery can exceed the original model's performance.
- **Generalization across scales and architectures**: Effective on LLaMA2-7B/13B/70B, LLaMA3-8B, and Baichuan2-7B/13B.

## Highlights & Insights
- **Core insight of degradation-aware data selection**: Not all data contributes equally to recovery; the key is identifying data matched to the degraded capabilities. JSD is a more robust degradation metric than simple loss differences because it captures the full output distribution shift.
- **Engineering insight of "less is more"**: The finding that full Alpaca recovery causes PPL explosion is highly valuable—it reveals that general instruction data contains many samples harmful to pruned models.
- **Elegant design of the Concept Consistency Graph (CCG)**: By modeling conflicts at the concept level rather than the sample level, CCG incurs low computational cost, supports incremental updates, and scales well to large datasets.

## Limitations & Future Work
- Computing JSD requires simultaneous access to both the original and pruned models; when the original model is very large (e.g., LLaMA-70B), JSD computation itself becomes costly.
- Concept extraction in the CCG relies on simple rules and may not detect complex semantic conflicts.
- Experiments are conducted primarily on English LLMs; applicability to multilingual settings is unknown.
- **Directions for improvement**: Can a lightweight proxy model replace the original model for JSD estimation? Can this framework be extended to post-quantization recovery?

## Related Work & Insights
- **vs. Nuggets (Li et al., 2024)**: A general data selection method and the strongest baseline in PASER's experiments (avg. 58.69), yet significantly outperformed by PASER (61.10) because it does not account for the distribution of capability degradation.
- **vs. IFD (Li et al., 2024)**: A score-based selection method relying on a trainable LLM; underperforms both Nuggets and PASER in structured pruning scenarios.
- **vs. LLM-Pruner (Ma et al., 2023)**: A pioneering structured pruning method that recommends full Alpaca recovery—PASER demonstrates this recommendation can be counterproductive.

## Rating
- Novelty: ⭐⭐⭐⭐ The capability degradation-aware data selection perspective is novel, and the technical pipeline combining manifold learning with JSD is complete.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four pruning configurations, seven LLMs, two dataset scales, and multiple data selection baselines—extremely comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Method description is detailed and code is publicly available.
- Value: ⭐⭐⭐⭐⭐ Directly applicable to practical LLM pruning and recovery workflows; the "less is more" finding carries important practical implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Pedagogically-Inspired Data Synthesis for Language Model Knowledge Distillation](pedagogically-inspired_data_synthesis_for_language_model_knowledge_distillation.md)
- [\[ICLR 2026\] MobileLLM-R1: Exploring the Limits of Sub-Billion Language Model Reasoners with Open Training Recipes](mobilellm-r1_exploring_the_limits_of_sub-billion_language_model_reasoners_with_o.md)
- [\[AAAI 2026\] Post Training Quantization for Efficient Dataset Condensation](../../AAAI2026/model_compression/post_training_quantization_for_efficient_dataset_condensation.md)
- [\[ICLR 2026\] PTQ4ARVG: Post-Training Quantization for AutoRegressive Visual Generation Models](ptq4arvg_post-training_quantization_for_autoregressive_visual_generation_models.md)
- [\[NeurIPS 2025\] Restoring Pruned Large Language Models via Lost Component Compensation](../../NeurIPS2025/model_compression/restoring_pruned_large_language_models_via_lost_component_compensation.md)

</div>

<!-- RELATED:END -->
