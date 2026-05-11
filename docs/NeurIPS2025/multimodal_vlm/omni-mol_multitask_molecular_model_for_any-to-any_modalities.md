---
title: >-
  [Paper Note] Omni-Mol: Multitask Molecular Model for Any-to-Any Modalities
description: >-
  [NeurIPS2025][Multimodal VLM][Molecular large language model] This paper proposes Omni-Mol, a unified molecular understanding and generation framework built upon a multimodal LLM. Through a 1.42M-sample instruction tunin…
tags:
  - "NeurIPS2025"
  - "Multimodal VLM"
  - "Molecular large language model"
  - "multitask learning"
  - "mixture of experts"
  - "adaptive LoRA"
  - "unified instruction tuning"
date: 2026-05-08
content_hash: bf012bb35c1d9322
---

# Omni-Mol: Multitask Molecular Model for Any-to-Any Modalities

**Conference**: NeurIPS2025  
**arXiv**: [2502.01074](https://arxiv.org/abs/2502.01074)  
**Authors**: Chengxin Hu, Hao Li, Yihe Yuan, Zezheng Song, Chenyang Zhao, Haixin Wang (NUS, UMD, UCLA)
**Code**: [Omni-Mol-Code](https://github.com/)  
**Area**: Multimodal VLM  
**Keywords**: Molecular large language model, multitask learning, mixture of experts, adaptive LoRA, unified instruction tuning

## TL;DR

This paper proposes Omni-Mol, a unified molecular understanding and generation framework built upon a multimodal LLM. Through a 1.42M-sample instruction tuning dataset, Gradient Adaptive LoRA (GAL), and a Mixture-of-GAL-Experts (MoGE) architecture, Omni-Mol is the first single model to jointly learn 16 molecular tasks (Mol2Mol / Mol2Text / Mol2Num / Text2Mol), achieving SOTA on 13 tasks with only 2.2B parameters.

## Background & Motivation

Building a general-purpose molecular AI (AI Chemist) is a central goal in drug discovery and chemical research. However, existing molecular multimodal LLMs fall short of a true "one-model-fits-all" solution in three key respects:

**Insufficient data scale and coverage**: Existing molecular instruction datasets are small and task-limited. For example, PRESTO supports Mol2Num and Mol2Mol but not Mol2Text or Text2Mol; InstructMol does not support Text2Mol.

**Difficulty in joint multitask learning**: Tasks across different molecular sub-domains exhibit significant distributional shifts and task competition, making it hard for LLMs to stably learn all tasks simultaneously.

**Intrinsic dimensionality mismatch**: Different tasks and modalities require different intrinsic dimensions in the language space; standard LoRA with a fixed rank cannot balance redundancy and insufficiency across tasks.

The core goal is to build a truly universal molecular model supporting arbitrary modality combinations (any input → any output), while addressing dimensionality adaptation and task conflict in multitask learning.

## Method

### Task Taxonomy and Dataset Construction

Small-molecule tasks are innovatively categorized into four classes based on input/output modalities:

- **Mol2Mol** (689K samples): Forward reaction prediction, retrosynthesis, reagent prediction, solvent prediction, catalyst prediction, molecular editing
- **Mol2Num** (412K samples): HOMO-LUMO quantum mechanical property prediction, molecular weight, TPSA, LogP, yield prediction
- **Mol2Text** (248K samples): Experimental procedure description, description Q&A, molecule captioning (Molcap)
- **Text2Mol** (73K samples): IUPAC→SELFIES conversion, text-guided molecule generation

The total of **1.42M samples** constitutes the largest molecular instruction tuning dataset to date. SELFIES rather than SMILES is adopted as the molecular representation, as SELFIES guarantees the validity of decoded molecules.

### Overall Architecture

Omni-Mol consists of three components:
1. **LLM backbone**: LLaMA 3.2-1B
2. **Graph encoder** $f_{\mathcal{G}}$: MoleculeSTM, encoding molecular 2D/3D graph structures
3. **Projector** $f_p$: A single linear layer aligning graph representations to the LLM hidden space

The model is formulated as autoregressive generation:
$$P(\mathbf{Y}|\mathbf{X}_I, \mathbf{X}_S, \mathbf{H}_G) = \prod_i P_\theta(\mathbf{Y}_i | \mathbf{X}_I, \mathbf{X}_S, \mathbf{H}_G, \mathbf{Y}_{<i})$$

### Gradient Adaptive LoRA (GAL)

**Design Motivation**: Empirical analysis reveals that different tasks have different optimal LoRA ranks (e.g., forward reaction prediction peaks at rank=128, Molcap at rank=32), and a fixed-rank standard LoRA cannot accommodate the varying intrinsic dimensionality across tasks.

**Core Idea**: A learnable dynamic scaling factor replaces LoRA's fixed scaling:
$$\gamma_\theta = \alpha / r^p + \beta$$
where $\theta = \{\alpha, p, \beta\}$ are learnable parameters. The exponent $p$ models the rank effect, and $\beta$ provides a direct adjustment offset. During training, gradient magnitudes are dynamically adjusted so that the adapter self-adapts to the intrinsic dimension of the data.

### Mixture-of-GAL-Experts (MoGE)

**Design Motivation**: The model must simultaneously handle graph features, text, and SELFIES, where SELFIES, though tokenized as text, is semantically distant from natural language.

**Key Designs**:
- The FFN layers in the last 3/4 of the LLM are replaced by MoGE layers
- Each MoGE layer contains $\mathcal{N}$ **routed experts** (learning specialized knowledge) + 1 **shared expert** (learning cross-task general knowledge)
- All experts are initialized from pretrained FFN weights; routers are initialized with Kaiming uniform initialization
- Practical configuration: 5 experts total — 2 routed + 1 shared (active per token)
- MHA layers are wrapped with GAL adapters; FFN layers in the first 1/4 of the network are also wrapped with GAL

### Two-Stage Training

- **Stage 1 (Multimodal Alignment)**: The model learns to describe molecules from graph features using PubChem data; only the projector is trained.
- **Stage 2 (Unified Instruction Tuning)**: Pretrained parameters are frozen; GAL adapters, expert routers, and the projector are trained. Total loss = language modeling loss + $\lambda$ × load-balancing loss.

## Key Experimental Results

### Table 3: Core Mol2Mol Tasks (vs. Expert and Generalist Models)

| Task | Model | Params | Type | Exact Match | Morgan ↑ | Lev ↓ |
|------|-------|--------|------|-------------|----------|-------|
| **Forward Reaction** | InstructMol | 6.7B | Expert | 0.54 | 0.74 | 10.85 |
| | PRESTO | 3.2B | Generalist | 0.69 | 0.84 | 6.53 |
| | **Omni-Mol** | **2.2B** | **Generalist** | **0.73** | **0.87** | **5.55** |
| **Retrosynthesis** | InstructMol | 6.7B | Expert | 0.41 | 0.71 | 13.97 |
| | PRESTO | 3.2B | Generalist | 0.53 | 0.79 | 10.30 |
| | **Omni-Mol** | **2.2B** | **Generalist** | **0.57** | **0.83** | **8.97** |
| **Reagent Prediction** | PRESTO | 3.2B | Generalist | 0.21 | 0.48 | 16.31 |
| | **Omni-Mol** | **2.2B** | **Generalist** | **0.23** | **0.52** | **14.59** |
| **Solvent Prediction** | PRESTO | 6.7B | Generalist | 0.42 | 0.51 | 2.76 |
| | **Omni-Mol** | **2.2B** | **Generalist** | **0.52** | **0.64** | **2.71** |

Omni-Mol surpasses nearly all expert baselines with only 33% of their parameter count, improving over PRESTO by approximately 5%, 7%, and 9% on forward reaction prediction, retrosynthesis, and reagent prediction, respectively.

### Table 3 (cont.): Mol2Num and Mol2Text Tasks

| Task | Model | Metric | Result |
|------|-------|--------|--------|
| **HOMO-LUMO** | InstructMol | Avg MAE | 0.0050 |
| | **Omni-Mol** | **Avg MAE** | **0.0044** (↓12%) |
| **MW / LogP / TPSA** | 3D-MoLM | MAE | 14.79 / 0.66 / 9.71 |
| | **Omni-Mol** | **MAE** | **11.07 / 0.49 / 5.89** (↓25–39%) |
| **Molcap** | HIGHT | BLEU-4 | 0.397 |
| | **Omni-Mol** | **BLEU-4** | **0.440** (↑11%) |
| **Description Q&A** | 3D-MoLM | BLEU-4 | 0.26 |
| | **Omni-Mol** | **BLEU-4** | **0.44** (↑69%) |

### Scaling Experiments

- **Data scaling**: Performance improves consistently as the data fraction increases from 20% → 100%, with no saturation observed, indicating that further data expansion can yield additional gains.
- **Parameter scaling**: Scaling LLaMA from 1B → 3B → 8B yields consistent performance improvements across all tasks.
- Data scaling yields greater gains than parameter scaling, suggesting that data expansion remains a high-potential direction.

### Ablation Study

- **Joint vs. separate training**: Joint multitask training consistently outperforms single-task training on the Omni-Mol dataset.
- **Removing GAL**: Replacing GAL with standard LoRA leads to consistent performance degradation.
- **Removing MoGE**: Using GAL alone without MoGE expansion degrades performance on multiple tasks including reagent prediction, Molcap, and yield regression, with the largest drop observed on yield regression.
- **Representation convergence analysis**: As the number of tasks increases from 1 to 8, the mutual similarity of representations learned by Omni-Mol increases continuously (converging toward a universal representation), whereas InstructMol's representations become progressively less similar (diverging).

## Highlights & Insights

- **Most comprehensive molecular generalist model**: The first unified framework simultaneously supporting all four modality combinations — Mol2Mol / Mol2Text / Mol2Num / Text2Mol — covering 16 tasks and 1.42M samples.
- **GAL adaptive mechanism**: Learnable scaling factors elegantly resolve the intrinsic dimensionality mismatch in multitask settings, directly addressing the fundamental limitation of fixed-rank LoRA in multi-task scenarios.
- **MoGE architecture**: The combination of shared and routed experts simultaneously preserves general knowledge and enables task specialization, leveraging the complementary strengths of MoE and adaptive LoRA.
- **Strong scaling evidence**: Clear scaling trends are observed along both data and parameter dimensions; representation convergence analysis provides empirical support for the hypothesis of universal molecular representations.
- **Exceptional parameter efficiency**: 2.2B parameters outperform 6.7B expert models and even surpass 685B DeepSeekV3 in few-shot settings.

## Limitations & Future Work

- **Computational constraints**: The performance ceiling of larger-scale models remains unexplored; the scaling trends observed at 8B suggest further gains are achievable with larger models.
- **Small molecules only**: The current dataset and tasks are limited to small molecules, excluding important biological scenarios such as proteins and protein–small molecule interactions.
- **Fixed MoGE configuration**: The number of experts (5) and the starting layer for MoGE (1/4L) are hyperparameters; optimal configurations may differ depending on task scale.
- **Dependence on SELFIES**: Although SELFIES guarantees molecular validity, the broader community predominantly uses SMILES, which may limit compatibility with existing toolchains.
- **Lack of downstream application validation**: End-to-end utility has not been evaluated in real drug discovery pipelines (e.g., molecular docking, ADMET prediction).

## Related Work & Insights

- **Molecular foundation models**: Mol-Instruction (first molecular instruction tuning) → InstructMol (2D graph + multi-LoRA) → HIGHT (multi-level 2D graph features) → 3D-MoLM (3D molecular representations) → PRESTO (domain pretraining + multitask) → Omni-Mol, achieving the most comprehensive "one-model-fits-all" solution.
- **Unified generative modeling**: Inspired by GPT/Flamingo/LLaVA's unification of multimodal understanding and generation, Omni-Mol introduces this paradigm into the molecular domain.
- **LoRA and variants**: Building on standard LoRA, learnable dynamic scaling is introduced and combined with MoE to enable adaptive multitask fine-tuning.
- **Representation convergence hypothesis**: Motivated by the Platonic Representation Hypothesis, the paper empirically validates the trend of molecular representations converging toward a universal space under multitask training.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of GAL and MoGE is novel and the task taxonomy is well-structured; however, the core components (LoRA + MoE) are built upon existing techniques.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across 16 tasks, with complete ablation studies, scaling experiments, and representation convergence analysis; baselines include both expert and generalist models.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure and rigorous motivation; however, the heavy use of notation and formulas, along with some extremely dense tables, may impede readability.
- **Value**: ⭐⭐⭐⭐ — Provides a solid baseline framework and large-scale dataset for building a general-purpose AI chemist; both data and model are open-sourced.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MDReID: Modality-Decoupled Learning for Any-to-Any Multi-Modal Object Re-Identification](mdreid_modality-decoupled_learning_for_any-to-any_multi-modal_object_re-identifi.md)
- [\[ICCV 2025\] MAVias: Mitigate Any Visual Bias](../../ICCV2025/multimodal_vlm/mavias_mitigate_any_visual_bias.md)
- [\[ICLR 2026\] Grasp Any Region: Towards Precise, Contextual Pixel Understanding for Multimodal LLMs](../../ICLR2026/multimodal_vlm/grasp_any_region_towards_precise_contextual_pixel_understanding_for_multimodal_l.md)
- [\[NeurIPS 2025\] Structure-Aware Fusion with Progressive Injection for Multimodal Molecular Representation Learning](structure-aware_fusion_with_progressive_injection_for_multimodal_molecular_repre.md)
- [\[NeurIPS 2025\] Can LLMs Reason Over Non-Text Modalities in a Training-Free Manner? A Case Study with In-Context Representation Learning](can_llms_reason_over_non-text_modalities_in_a_training-free_manner_a_case_study_.md)

</div>

<!-- RELATED:END -->
