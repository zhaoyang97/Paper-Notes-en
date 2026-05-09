---
title: >-
  [Paper Note] KINDLE: Knowledge-Guided Distillation for Prior-Free Gene Regulatory Network Inference
description: >-
  [NeurIPS 2025][Model Compression][Gene Regulatory Network] This paper proposes KINDLE, a three-stage framework that transfers gene regulatory knowledge learned by a prior-guided teacher model to a prior-free student model via knowledge distillation, achieving state-of-the-art performance in gene regulatory network (GRN) inference without relying on any external prior knowledge.
tags:
  - NeurIPS 2025
  - Model Compression
  - Gene Regulatory Network
  - knowledge distillation
  - Prior-Free Inference
  - Temporal Attention
  - Single-Cell RNA-seq
date: 2026-05-08
content_hash: 06176f5d2059590f
---

# KINDLE: Knowledge-Guided Distillation for Prior-Free Gene Regulatory Network Inference

**Conference**: NeurIPS 2025
**arXiv**: [2505.09664](https://arxiv.org/abs/2505.09664)
**Code**: To be confirmed
**Area**: Model Compression
**Keywords**: Gene Regulatory Network, knowledge distillation, Prior-Free Inference, Temporal Attention, Single-Cell RNA-seq

## TL;DR

This paper proposes KINDLE, a three-stage framework that transfers gene regulatory knowledge learned by a prior-guided teacher model to a prior-free student model via knowledge distillation, achieving state-of-the-art performance in gene regulatory network (GRN) inference without relying on any external prior knowledge.

## Background & Motivation

Gene regulatory networks (GRNs) describe regulatory relationships between transcription factors (TFs) and target genes (TGs), forming the foundation for understanding cell fate determination. GRN inference faces the following challenges:

1. **Enormous search space**: For a genome with approximately 30,000 genes, there are roughly one billion potential TF–TG interaction pairs, severely limiting the performance of expression-data-only approaches.
2. **Limitations of prior-dependent methods**: Mainstream methods rely on external priors such as scATAC-seq and Hi-C data to narrow the search space, but suffer from two critical issues:
   - Algorithmic performance is highly sensitive to the overlap between the prior and the true network; inaccurate priors cause performance collapse.
   - Restricting the search to known interactions precludes the discovery of novel regulatory relationships, severely constraining the potential for scientific discovery.
3. **Practical necessity**: In settings such as non-model organisms or emerging pathological states, reliable prior networks are often unavailable.

Inspired by the **privileged information learning** paradigm, the authors propose a novel strategy: leveraging prior-guided teacher models during training, then transferring regulatory knowledge to a student model that requires no prior at inference time via knowledge distillation.

## Core Problem

How can prior knowledge be fully exploited during training to improve GRN inference quality, while eliminating all dependence on priors at inference time, so that the model maintains high accuracy and retains the potential to discover novel regulatory relationships?

## Method

KINDLE (**K**nowledge-gu**I**ded **N**etwork **D**ist**IL**lation for prior-free GRN inf**E**rence) consists of three stages:

### Theoretical Foundation

The core assumption is that GRNs intrinsically drive transcriptional dynamics through temporally evolving interactions. An accurate GRN adjacency matrix $\mathbf{A}$ should enable prediction of future expression states from historical ones:

$$\mathbf{A}^* = \arg\min_{\mathbf{A}} \|\mathcal{F}(\mathbf{G}_{1:T}, \mathbf{A}) - \mathbf{G}_{T+1:T+W}\|_2^2$$

This reformulates GRN inference as a temporal prediction problem of learning a minimally sufficient interaction matrix.

### Stage 1: Teacher Model Training

The teacher model employs a hierarchical attention mechanism comprising:

- **Temporal attention layer**: A lower-triangular mask ensures that each gene at time step $t$ attends only to its historical states $\{1,\ldots,t-1\}$, simulating the irreversibility of cell differentiation.
- **Spatial attention layer**: A binary mask $\mathbf{M}^{spatial} \in \{0,1\}^{M \times M}$ generated from prior knowledge restricts attention computation to TF–TG pairs recorded in the prior.

Architecturally, the model first processes the temporal dimension $\mathbb{R}^{B \times T \times M}$, then transposes to the gene dimension $\mathbb{R}^{B \times M \times T}$ for spatial attention, and finally projects to $\mathbb{R}^{B \times W \times M}$ for $W$-step prediction. The loss function is mean squared error.

### Stage 2: Knowledge Distillation

The student model introduces two key modifications:
1. **Removal of spatial prior mask**: All gene pairs are permitted to model interactions freely.
2. **Removal of temporal attention layer**: Temporal causality is preserved through distillation while making the model more lightweight.

The student model is optimized with:

$$\alpha \cdot \mathcal{L}_{\text{pred}} + (1-\alpha) \cdot \mathcal{L}_{\text{distill}}$$

where $\alpha$ balances expression prediction accuracy against regulatory knowledge transfer. Four distillation strategies are explored:

| Distillation Strategy | Core Idea |
|----------------------|-----------|
| **Hard Distillation** | Directly aligns final predictions of teacher and student via L2 norm |
| **Soft Distillation** | Softens logits with a temperature parameter and matches probability distributions via KL divergence |
| **Bilinear Pool** | Captures inter-instance correlation structure via outer product operations |
| **Gaussian RBF** | Captures nonlinear manifold structure via exponentially decayed similarity (best performing) |

### Stage 3: Prior-Free Inference

The input gene expression time series is partitioned into non-overlapping segments of length $T$. Each segment is passed through the student model to produce an attention matrix $\mathbf{A}^{(g)}$, and the final GRN is obtained by temporal aggregation:

$$\hat{\mathbf{A}} = \frac{1}{H}\sum_{g=1}^{H}\mathbf{A}^{(g)}$$

The top-$k$ edges ranked by weight are then selected to construct the predicted network.

## Key Experimental Results

### Benchmark Evaluation (BEELINE, Four Datasets)

Across four mouse differentiation datasets (mESC, mHSC-E, mHSC-L, mHSC-GM), KINDLE-Gaussian achieves the best performance on 11 of 12 metric–dataset combinations:

| Dataset | Metric | Best Baseline | KINDLE-Gaussian | Gain |
|---------|--------|--------------|----------------|------|
| mESC | AUROC | 0.545 (GENIE3) | **0.757** | +39% |
| mESC | AUPRC | 0.253 (CEFCON) | **0.646** | +155% |
| mHSC-E | AUPRC | 0.405 (CEFCON) | **0.601** | +48% |
| mHSC-GM | AUPRC | 0.444 (CEFCON) | **0.799** | +80% |
| mHSC-GM | F1 | 0.647 (CEFCON/Prior_Random) | **0.875** | +35% |

A key observation is that KINDLE's advantage is especially pronounced on AUPRC and F1, as true regulatory edges account for only 0.65%–1.15% of all candidate edges (severe class imbalance), making AUPRC/F1 more informative than AUROC in this setting.

### Biological Validation

1. **Key TF identification (mESC)**: KINDLE successfully identifies 25 key TFs, of which 18 (72%) are confirmed by existing literature to participate in mESC differentiation. Hierarchical clustering reveals two groups with antiphasic activation patterns:
   - Early regulators (Nanog, Sox2, Nr0b1, etc.): highly active during early differentiation and subsequently declining, consistent with their known roles in maintaining stem cell pluripotency.
   - Late regulators (Gata4, Sox17, Kdm5b, etc.): silenced initially and significantly activated from the third stage onward, consistent with lineage specification mechanisms.

2. **In vitro perturbation prediction (mHSC)**:
   - **Gata1** knockout: perturbation vectors for erythroid cells (CMP, MEP) point opposite to the differentiation direction, indicating differentiation suppression → consistent with experimental results.
   - **Spi1** knockout: CMP differentiation toward the erythroid lineage is promoted while LMPP/GMP differentiation is suppressed → consistent with experimental results.

## Highlights & Insights

1. **Elegant problem reformulation**: The prior-dependency problem in GRN inference is recast as a privileged information distillation problem, providing a general paradigm for prior-free inference.
2. **Unification of theory and practice**: A mathematical framework grounded in causal assumptions tightly couples GRN inference with temporal prediction.
3. **Systematic distillation strategies**: Four distillation schemes are explored; the Gaussian RBF kernel captures nonlinear manifold structure most effectively.
4. **Multi-level biological validation**: KINDLE outperforms baselines not only on benchmark metrics but also receives biological confirmation through TF function identification and gene knockout effect prediction.
5. **High practical value**: No additional omics data are required at inference time, substantially lowering the barrier to adoption.

## Limitations & Future Work

1. **Data type restriction**: The method depends on time-series gene expression data and cannot be directly applied to static scRNA-seq data from a single time point.
2. **Bias propagation risk**: The student model may inherit systematic biases introduced during teacher training due to incomplete or noisy priors.
3. **Single regulatory layer**: Only transcriptional regulation is modeled; post-transcriptional and epigenetic regulatory layers are not considered.
4. **Ground-truth-dependent edge count**: The number of predicted edges $k$ must match the ground truth, requiring additional edge selection strategies in practical settings where the ground truth is unknown.
5. **Scalability**: Validation is performed only on relatively small benchmark datasets; performance and computational efficiency at the whole-genome scale remain to be assessed.

## Related Work & Insights

| Method | Requires Prior | Core Technique | Limitation |
|--------|---------------|---------------|-----------|
| GENIE3/GRNBoost2 | No | Tree-based regression | Large search space, limited performance |
| CEFCON | Yes (scATAC-seq) | Graph attention network + network control theory | Prior-dependent, cannot discover new interactions |
| CellOracle | Yes (scATAC-seq) | Motif analysis + ridge regression | Prior-dependent |
| KINDLE | Training only | Temporal causal attention + knowledge distillation | Requires time-series data |

The key distinction of KINDLE is that it is the only framework requiring no prior at inference time while outperforming all prior-guided methods.

**Broader implications**:
1. **Generality of privileged information distillation**: The paradigm of "using auxiliary information at training time but not at inference time" generalizes to other bioinformatics tasks that require expensive annotations or auxiliary data.
2. **Attention matrix as network**: Directly using the Transformer's spatial attention matrix as the network inference output is an elegant design for interpretability.
3. **Causal constraints in temporal modeling**: Temporally causal attention enforced via lower-triangular masking is applicable to other sequence modeling problems with irreversible dynamics.

## Rating

- Novelty: ⭐⭐⭐⭐ — First application of privileged information distillation to GRN inference; the problem reformulation is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Four benchmarks + two biological validations + four distillation ablations; large-scale validation is lacking.
- Writing Quality: ⭐⭐⭐⭐ — Clear logic, complete mathematical derivations, and well-grounded biological interpretations.
- Value: ⭐⭐⭐⭐ — Provides a practically viable solution for GRN inference in prior-limited settings.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Less is More but Where: Dynamic Token Compression via LLM-Guided Keyframe Prior](less_is_more_but_where_dynamic_token_compression_via_llm-guided_keyframe_prior.md)
- [\[NeurIPS 2025\] Single-Teacher View Augmentation: Boosting Knowledge Distillation via Angular Diversity](single-teacher_view_augmentation_boosting_knowledge_distillation_via_angular_div.md)
- [\[NeurIPS 2025\] A Token is Worth over 1,000 Tokens: Efficient Knowledge Distillation through Low-Rank Clone](a_token_is_worth_over_1000_tokens_efficient_knowledge_distillation_through_low-r.md)
- [\[ICCV 2025\] Knowledge Distillation with Refined Logits](../../ICCV2025/model_compression/knowledge_distillation_with_refined_logits.md)
- [\[NeurIPS 2025\] On the Creation of Narrow AI: Hierarchy and Nonlocality of Neural Network Skills](on_the_creation_of_narrow_ai_hierarchy_and_nonlocality_of_neural_network_skills.md)

<!-- RELATED:END -->
