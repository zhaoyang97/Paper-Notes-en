---
title: >-
  [Paper Note] Reverse Distillation: Consistently Scaling Protein Language Model Representations
description: >-
  [ICLR 2026][Medical Imaging][Reverse Distillation] To address the anomalous scaling phenomenon in protein language models (PLMs) where larger models do not necessarily yield better performance…
tags:
  - "ICLR 2026"
  - "Medical Imaging"
  - "Reverse Distillation"
  - "Protein Language Model"
  - "Scaling Behavior"
  - "Matryoshka Nested Representations"
  - "ESM-2"
date: 2026-05-08
content_hash: e47c580a9501a55a
---

# Reverse Distillation: Consistently Scaling Protein Language Model Representations

**Conference**: ICLR 2026
**arXiv**: [2603.07710](https://arxiv.org/abs/2603.07710)
**Code**: [GitHub](https://github.com/rohitsinghlab/plm_reverse_distillation)
**Area**: Protein AI / Representation Learning
**Keywords**: Reverse Distillation, Protein Language Model, Scaling Behavior, Matryoshka Nested Representations, ESM-2

## TL;DR

To address the anomalous scaling phenomenon in protein language models (PLMs) where larger models do not necessarily yield better performance, this paper proposes a reverse distillation framework. It uses the representations of a smaller model as a base, extracts orthogonal residual information from a larger model via SVD, and constructs Matryoshka nested embeddings—ensuring that larger reverse-distilled models consistently outperform smaller ones. ESM-2 15B, after reverse distillation, becomes for the first time the strongest model in its family.

## Background & Motivation

**Background**: PLMs learn rich protein representations through self-supervised training on massive sequence corpora, achieving breakthrough performance in structure prediction, functional annotation, and protein design. In NLP and CV, scaling laws are stable—larger models perform better—yet PLM families exhibit counterintuitive scaling behavior.

**Limitations of Prior Work**: Taking the ESM-2 family as an example, performance peaks at 650M–3B parameters, while the largest 15B model suffers from performance degradation. This raises two critical issues: (1) **Non-monotonic scaling**—it is impossible to predict which downstream tasks will see large models underperform smaller ones, making model selection difficult; (2) **Non-truncatable embeddings**—embedding dimensions are incompatible across model scales, precluding the "embed once, truncate as needed" paradigm of Matryoshka embeddings in NLP.

**Key Challenge**: Although large models have sufficient capacity to encode richer higher-order features (enzyme catalytic sites, allosteric coupling, etc.), these higher-order features are entangled with basic features (secondary structure propensities, hydrophobicity patterns, etc.) in the same representation space. When linear probes are used downstream, task-irrelevant higher-order features act as noise, obscuring the basic patterns that drive performance.

**Key Insight**: The authors adopt a bias–variance tradeoff perspective: small models, constrained by capacity, are forced to prioritize the most frequent and broadly shared protein features (high bias, low variance); large models additionally encode rare higher-order phenomena but introduce variance. If the small model's representation serves as a "base" and the large model's representation is decomposed into "shared foundation + orthogonal residual," the destructive interference between the two feature types can be avoided.

**Core Idea**: Use the representations of a smaller model in the same family as the decomposition base; extract the orthogonal residual information of the larger model via linear regression + SVD; construct nested embeddings to restore monotonic scaling behavior.

## Method

### Overall Architecture

Given a smaller model $M_r$ and a larger model $M_p$ from the same family ($|M_r| < |M_p|$), reverse distillation decomposes the $k_p$-dimensional representation space of the larger model into two orthogonal subspaces: $\mathcal{S}_r$ (retaining the $k_r$-dimensional representations of the smaller model) and $\mathcal{S}_{res}$ (capturing the $(k_p - k_r)$-dimensional residual information unique to the larger model). The final output is $H_{rd} = [H_r, H_{res}]$, where the first $k_r$ dimensions are exactly the complete embeddings of the smaller model, naturally endowing the construction with Matryoshka nested properties.

The entire pipeline involves only linear transformations (regression + SVD) with no retraining of any model; training requires only 1,000 UniRef50 sequences. For the ESM-2 family, chain distillation proceeds along 8M → 35M → 150M → 650M → 3B → 15B, yielding reverse-distilled embeddings at each scale.

### Key Designs

1. **Reverse Distillation Decomposition (Algorithm 1)**:

    - **Function**: Decompose the large model's representations into two orthogonal subspaces—the part explainable by the small model and the unique contribution of the large model.
    - **Mechanism**: Three phases—Phase 1 runs the small and large models on the same sequence set to obtain $H_r \in \mathbb{R}^{L \times k_r}$ and $H_p \in \mathbb{R}^{L \times k_p}$; Phase 2 learns a linear mapping $W^* = \arg\min_W \|H_p - H_r W\|_F^2$ via principal component regression (PCR), applying PCA to $H_r$ and discarding noise components using the Johnstone threshold before regression to avoid overfitting; Phase 3 computes the residual $R = H_p - H_r W^*$, applies SVD to extract the top $k_p - k_r$ right singular vectors $V_{res}$, and projects to obtain $H_{res} = R V_{res}$.
    - **Design Motivation**: Linear decomposition preserves interpretability—$H_r$ is the complete feature space of the small model, and $H_{res}$ can be directly interpreted as features unique to the large model. The Johnstone threshold, derived from random matrix theory, effectively distinguishes signal principal components from noise.

2. **Chain Reverse Distillation (Algorithm 3)**:

    - **Function**: Extend reverse distillation between two models to a hierarchical decomposition across the entire model family.
    - **Mechanism**: Starting from the smallest model $M_1$, the accumulated embedding $H_{acc}^{(i-1)}$ serves as the base for reverse distillation against the next larger model $M_i$. At each step, a linear mapping is learned, residuals are computed, orthogonal components are extracted via SVD, and the result is concatenated to the accumulated embedding. For ESM-2, this proceeds stepwise along 8M → 35M → 150M → 650M → 3B → 15B.
    - **Design Motivation**: Experiments show that longer progressive chains (e.g., 8M → 35M → 150M → 650M) consistently outperform direct jump chains (e.g., 8M → 650M), because each incremental decomposition is more fine-grained and better separates biological features at different levels.

3. **Matryoshka Nested Structure and Optimality Guarantee**:

    - **Function**: Reverse distillation embeddings inherently possess the Matryoshka (Russian nesting doll) property—truncating to any prefix dimension yields a valid reverse distillation embedding at that scale.
    - **Mechanism**: By construction of $H_{rd} = [H_r, H_{res}]$, the first $k_1$ dimensions equal the 8M embedding, the first $k_1 + k_2$ dimensions equal the rd.35M embedding, and so on. Theorem 1 proves that among all $k_p$-dimensional representations $[H_r, X]$ that keep $H_r$ as a prefix, the reverse distillation $H_{res}$ minimizes reconstruction error with respect to the original large model representation (a direct consequence of the Eckart–Young theorem).
    - **Design Motivation**: The Matryoshka property enables "embed once, use at different dimensions as needed." Performance degrades smoothly with dimension truncation, eliminating the need to re-embed for different downstream tasks.

### Loss & Training

The entire framework requires no backpropagation training; all steps have closed-form solutions: regression coefficients are obtained via matrix inversion, and the residual subspace is obtained via SVD. Training data consists of only $N=1{,}000$ randomly sampled UniRef50 sequences (with <30% sequence identity to the evaluation datasets); the total number of amino acid positions $L = \sum n_i$ serves as the effective sample size. Since only linear transformations are involved, computational cost is minimal.

## Key Experimental Results

### Main Results: ProteinGym DMS Scaling Consistency (28 datasets, single mutants)

| Model Comparison | Fraction of Datasets Where Left Wins | Interpretation |
|---------|-------------------|------|
| rd.650M > original 650M | 71.4% | Reverse distillation improves same-scale performance |
| rd.3B > original 3B | 85.7% | Reverse distillation improves same-scale performance |
| rd.15B > original 15B | 67.9% | Reverse distillation repairs 15B degradation |
| original 3B > original 650M | 53.6% | Scaling barely works in original models |
| rd.3B > rd.650M | **92.9%** | Monotonic scaling restored after reverse distillation |
| rd.15B > rd.3B | **85.7%** | Monotonic scaling restored after reverse distillation |

### Downstream Protein Property Prediction (Table 4)

| Task | 650M | rd.650M | 3B | rd.3B | 15B | rd.15B |
|------|------|---------|-----|-------|-----|--------|
| SSP Q3 (aupr↑) | 0.831 | 0.833 | 0.791 | 0.816 | 0.845 | **0.861** |
| SSP Q8 (aupr↑) | 0.365 | 0.369 | 0.379 | 0.395 | 0.418 | **0.431** |
| MIB (aupr↑) | 0.881 | 0.855 | 0.893 | 0.891 | 0.900 | **0.901** |
| R2/R1 (aupr↑) | 0.343 | 0.405 | 0.369 | 0.425 | 0.368 | **0.468** |

### Ablation Study: Chain Configuration Comparison (Table 1, 3 DMS datasets)

| Chain Configuration | ARGR_ECOLI | DN7A_SACS2 | ILF3_HUMAN |
|---------|-----------|-----------|-----------|
| Original 650M (baseline) | 0.834 | 0.868 | 0.712 |
| Direct 8M → 650M | 0.849 | 0.878 | 0.765 |
| Direct 150M → 650M | 0.845 | 0.866 | 0.751 |
| Full chain 8M → 35M → 150M → 650M | **0.858** | 0.867 | **0.786** |
| Original 3B (baseline) | 0.845 | 0.880 | 0.749 |
| Full chain → 3B (rd.3B) | **0.873** | **0.890** | **0.801** |

### Key Findings

- **Substantial repair of scaling consistency**: In original ESM-2, the fraction of datasets where 3B outperforms 650M is only 53.6%; after reverse distillation, rd.3B outperforms rd.650M on 92.9% of datasets, largely restoring monotonic scaling behavior.
- **Same-scale performance improvement**: rd.650M outperforms original 650M on 71.4% of datasets, indicating that reverse distillation not only repairs scaling relationships but directly improves embedding quality.
- **15B degradation reversed**: rd.15B becomes the strongest model in the family, outperforming original 15B on both single-mutant (67.9%) and double-mutant (57.1%) tasks.
- **Longer chains consistently superior**: The full progressive chain (8M → 35M → 150M → 650M) outperforms direct jump chains (e.g., 8M → 650M) on all three exploratory datasets, validating the advantage of stepwise decomposition.
- **Largest gain on R2/R1 prediction**: rd.15B (0.468) improves over original 15B (0.368) by 27% on RNA secondary structure tasks, suggesting reverse distillation is particularly effective for tasks requiring fine-grained feature separation.
- **SAE analysis validates feature disentanglement**: Sparse autoencoders trained on rd.35M embeddings capture on average 25% more GO terms than those trained on original 35M embeddings (40 vs. 32), with significantly higher specificity (fewer general terms), supporting the claim that reverse distillation disentangles biological feature representations.

## Highlights & Insights

- **The elegance of reverse thinking**: Traditional distillation compresses knowledge from large to small models; reverse distillation inverts this direction, using the small model to "navigate" the large model—the small model serves as the decomposition base, from which the large model's incremental information is extracted. This perspective shift is highly elegant, reframing "why large models underperform" as "how to systematically combine contributions across scales."
- **Understanding PLMs through the bias–variance lens**: Small models, capacity-constrained, are biased toward encoding high-frequency shared features (high bias); large models additionally encode rare higher-order features but introduce variance. This framework not only explains why small models often outperform large ones, but directly guides the decomposition strategy—orthogonal decomposition along the bias–variance axis.
- **Engineering elegance of a minimal implementation**: The entire method requires only linear regression + SVD, can be trained on 1,000 sequences, and incurs only 1.5–1.7× inference overhead. No neural network training, no gradient backpropagation, no hyperparameter search—yet it systematically repairs scaling behavior. This style of "solving fundamental problems with the simplest mathematical tools" is instructive.
- **Transferability of the Matryoshka nesting idea**: The approach of constructing nested embeddings via reverse distillation is transferable to any setting where a model family exhibits scaling problems. In foundation models for genomics, drug discovery, and related domains, if similar non-monotonic scaling exists, the reverse distillation framework can be directly applied.

## Limitations & Future Work

- **Linear decomposition only**: The authors themselves note that nonlinear mappings significantly improve reconstruction R² (from 0.422 to 0.528 for 8M → 35M), but at the cost of interpretability and the Matryoshka guarantee. Future work could explore nonlinear residual extraction (e.g., UMAP) or LoRA fine-tuning to capture nonlinear interactions while preserving nested structure.
- **Multi-model inference overhead**: rd.15B requires forward passes through six ESM-2 models; though smaller models are fast, total inference time is still 1.7× the baseline. The authors suggest using LoRA to directly produce reverse distillation embeddings in the last layer of the large model, enabling single forward-pass inference.
- **Validation limited to ESM-2 family**: The method has not been validated on other PLM families (ProtTrans, Ankh, ProGen, etc.) or foundation models in non-protein domains. The claimed generality of the framework requires broader empirical support.
- **Very small training set (1,000 sequences)**: Although linear methods have low sample requirements, systematic analysis is lacking on whether larger training sets could further improve performance and where diminishing returns occur across different tasks.
- **Limited downstream task evaluation**: Evaluation is primarily on DMS mutational effect prediction and a small number of protein property prediction tasks; applicability to generative/complex tasks such as protein design and protein–protein interaction prediction remains unknown.

## Related Work & Insights

- **vs. Traditional knowledge distillation (Hinton et al.)**: Traditional distillation compresses a large model into a small one, aiming to "simulate the large model with the small model"; reverse distillation instead uses the small model to "understand the large model," with the goal of decomposition rather than compression. The two are opposite yet complementary.
- **vs. Matryoshka Representation Learning (Kusupati et al., NeurIPS 2022)**: MRL learns nested embeddings via multi-scale losses during training, requiring model retraining; reverse distillation is a post-hoc method that directly constructs nested structures on an already-trained model family without any training modification.
- **vs. o-LoRA / Adaptive SVD (continual learning)**: These methods maintain orthogonal subspaces across tasks; reverse distillation decomposes orthogonal subspaces across model scales. The perspectives differ, but the mathematical tools are similar.
- **vs. PLM embedding compression (Lu et al., Devkota et al.)**: Compression methods demonstrate that PLM embeddings contain substantial redundancy; reverse distillation leverages this finding—the redundancy in large models arises precisely from the entanglement of higher-order and basic features, which is naturally removed upon decomposition.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The concept of reverse distillation is concise and profound, inverting the direction of knowledge distillation; using small models to guide large model representation decomposition is a genuinely new perspective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — ProteinGym 28 datasets + 5 downstream tasks + SAE analysis constitute a fairly complete validation, but verification on other PLM families is absent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Motivation is clearly articulated (the bias–variance perspective is highly intuitive), algorithm pseudocode is complete, and theoretical proofs are concise and compelling.
- **Value**: ⭐⭐⭐⭐ — Directly relevant to addressing scaling challenges in protein AI, with a strong generalizability; actual impact depends on extensibility to more model families.
- Models with non-monotonically increasing embedding dimensions require additional dimensionality reduction preprocessing.

## Related Work & Insights

- Matryoshka representations (Kusupati 2022) → usable embedding prefixes in NLP → first realization in PLMs
- Li et al. (2024) found PLM downstream tasks rely on early low-level features → validates the rationale for using small models as the base
- Kaplan scaling laws (2020) → strong predictive power in NLP → fail in PLMs → repaired by this work
- PCA dimensionality reduction baseline → appendix validates that rd outperforms simple PCA concatenation
- Traditional knowledge distillation (large → small) → reverse distillation (small guides large) → complementary directions
- Insight: Scaling laws for biological foundation models require special treatment → reverse distillation may be a general solution
- Insight: The same approach could be applied to scaling problems in genomic/chemical language models

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Reverse distillation concept is original
- **Technical Depth**: ⭐⭐⭐⭐ — Theory (MSE optimality) + linear algebra are clearly developed
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive ProteinGym coverage + chain ablations
- **Practicality**: ⭐⭐⭐⭐⭐ — Minimal implementation + immediately usable
- **Overall**: ⭐⭐⭐⭐⭐ — Solves PLM scaling problem with concise elegance

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] How to Make the Most of Your Masked Language Model for Protein Engineering](how_to_make_the_most_of_your_masked_language_model_for_protein_engineering.md)
- [\[ICLR 2026\] Protein as a Second Language for LLMs](protein_as_a_second_language_for_llms.md)
- [\[ICLR 2026\] Controlling Repetition in Protein Language Models](controlling_repetition_in_protein_language_models.md)
- [\[ICLR 2026\] mCLM: A Modular Chemical Language Model that Generates Functional and Makeable Molecules](mclm_a_modular_chemical_language_model_that_generates_functional_and_makeable_mo.md)
- [\[ICLR 2026\] Dual Distillation for Few-Shot Anomaly Detection](dual_distillation_for_few-shot_anomaly_detection.md)

</div>

<!-- RELATED:END -->
