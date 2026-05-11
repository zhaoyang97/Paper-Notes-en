---
title: >-
  [Paper Note] FALCON: Fine-grained Activation Manipulation by Contrastive Orthogonal Unalignment for Large Language Model
description: >-
  [NeurIPS 2025][Robotics][machine unlearning] This paper proposes FALCON, a representation-guided LLM unlearning framework that employs mutual information for parameter selection…
tags:
  - "NeurIPS 2025"
  - "Robotics"
  - "machine unlearning"
  - "contrastive learning"
  - "gradient orthogonal projection"
  - "mutual information"
  - "knowledge disentanglement"
date: 2026-05-08
content_hash: 1809dc331d276fba
---

# FALCON: Fine-grained Activation Manipulation by Contrastive Orthogonal Unalignment for Large Language Model

**Conference**: NeurIPS 2025
**arXiv**: [2502.01472](https://arxiv.org/abs/2502.01472)
**Code**: [FALCON](https://github.com/CharlesJW222/FALCON)
**Area**: AI Safety / Machine Unlearning / LLM Alignment
**Keywords**: machine unlearning, contrastive learning, gradient orthogonal projection, mutual information, knowledge disentanglement

## TL;DR
This paper proposes FALCON, a representation-guided LLM unlearning framework that employs mutual information for parameter selection, a contrastive mechanism for fine-grained knowledge separation, and gradient orthogonal projection to resolve forgetting–retention conflicts. FALCON consistently outperforms existing methods on harmful knowledge, copyright, and entity unlearning benchmarks.

## Background & Motivation

1. **LLM Safety Risks**: Large language models may encode harmful, biased, or sensitive information, leading to ethical violations and compliance issues.
2. **Limitations of Prior Work**:
   - Guardrail-based approaches are computationally expensive and vulnerable to adversarial attacks.
   - Full retraining is impractical.
   - Existing unlearning methods rely on coarse-grained loss combinations, making it difficult to precisely disentangle knowledge while balancing forgetting efficacy and model utility.
3. **Three Core Challenges**: (I1) Lack of efficient and interpretable guidance for parameter selection; (I2) Coarse-grained manipulation causes representations to scatter randomly with uncontrollable gradient dynamics; (I3) Forgotten knowledge can be recovered via jailbreak attacks.

## Core Problem
How to precisely unlearn specific domain knowledge in LLMs while preserving general model capability and resisting knowledge-recovery attacks?

## Method

### Overall Architecture (Three-Step Pipeline)
**Step 1: Mutual Information-Guided Parameter Selection** → **Step 2: Contrastive Orthogonal Unalignment** → **Step 3: Model Update**

### Step 1: Mutual Information-Based Layer Selection

Given activations from the forget set $\mathcal{F}$ and retain set $\mathcal{R}$ at each layer $l$, the mutual information is computed as:

$$I(\mathcal{F};\mathcal{R}) = H(\mathcal{F}) + H(\mathcal{R}) - H(\mathcal{F},\mathcal{R})$$

The layer with the lowest mutual information (i.e., least knowledge entanglement) is selected for intervention:

$$l^* = \arg\min_l I(\mathcal{F}^{(l)};\mathcal{R}^{(l)})$$

For multi-domain unlearning, an aggregated mutual information is defined as:

$$I^{(l)} = \sum_{i=1}^m I(\mathcal{F}_i^{(l)};\mathcal{R}^{(l)}) + \eta\sum_{i=1}^m\sum_{j=i+1}^m I(\mathcal{F}_i^{(l)};\mathcal{F}_j^{(l)})$$

KDE with PCA dimensionality reduction is applied to approximate entropy estimation for high-dimensional continuous activations.

### Step 2.1: Contrastive Representation Unlearning

**Principal Offset Vectors (POVs)** are constructed to guide model activations away from the principal directions of the knowledge to be forgotten. SVD is applied to the frozen model's activation matrix to obtain principal directions $v_1, \ldots, v_K$, and the POVs are defined as:

$$\mathcal{H}^+ = \frac{f(r \cdot (I - w\sum_{i=1}^K v_iv_i^\top), \epsilon)}{|f(r \cdot (I - w\sum_{i=1}^K v_iv_i^\top), \epsilon)|}$$

The forgetting loss is formulated using InfoNCE:

$$\mathcal{L}_\mathcal{F} = -\frac{1}{|B|}\sum_{b=1}^{|B|}\log\frac{\exp(S_b^+/\tau)}{\exp(S_b^+/\tau)+\sum \exp(S_b^-/\tau)}$$

The retention loss aligns the activations of the updated model with those of the frozen model on the retain set via cosine similarity:

$$\mathcal{L}_\mathcal{R} = 1 - \frac{1}{|B|}\sum \text{cos}(\mathcal{H}_b^u, \mathcal{H}_b^f)$$

### Step 2.2: Gradient Orthogonal Projection

When the forgetting gradient conflicts with the retention gradient (i.e., $\cos(\nabla\mathcal{L}_\mathcal{F}, \nabla\mathcal{L}_\mathcal{R}) < 0$), the forgetting gradient is projected onto the orthogonal complement of the retention gradient:

$$\nabla\mathcal{L}_\mathcal{F}^{\text{proj}} = \nabla\mathcal{L}_\mathcal{F} - \frac{\nabla\mathcal{L}_\mathcal{F} \cdot \nabla\mathcal{L}_\mathcal{R}}{\|\nabla\mathcal{L}_\mathcal{R}\|^2}\nabla\mathcal{L}_\mathcal{R}$$

The final update direction is: $\nabla\mathcal{L}_{FALCON} = \alpha\nabla\mathcal{L}_\mathcal{F}^{\text{proj}} + \beta\nabla\mathcal{L}_\mathcal{R}$

## Key Experimental Results

### Harmful Knowledge Unlearning (WMDP Benchmark)

| Method | WMDP-Bio ↓ | WMDP-Cyber ↓ | MMLU ↑ | PPL ↓ |
|--------|-----------|--------------|--------|-------|
| Zephyr-7B Baseline | 63.7 | 43.8 | 58.1 | 1.5 |
| + RMU | 34.5 | 28.9 | 57.4 | 1.5 |
| + SCRUB | 38.7 | 35.4 | 50.0 | 16.5 |
| **+ FALCON** | **26.7** | **25.3** | **57.4** | **1.5** |
| Yi-6B-Chat Baseline | 65.4 | 42.6 | 61.8 | 1.5 |
| + RMU | 50.8 | 33.5 | 59.6 | 1.6 |
| **+ FALCON** | **27.7** | **25.3** | **60.3** | **1.5** |

### Copyright Content Unlearning (MUSE Benchmark)

| Method | forget_know ↓ | forget_verb ↓ | retain_know ↑ |
|--------|--------------|--------------|---------------|
| GradAscent | 0.00 | 0.00 | 0.00 |
| NPO | 0.56 | 0.35 | 0.51 |
| RMU | 0.48 | 0.05 | 0.51 |
| **FALCON** | **0.02** | **0.03** | **0.54** |

## Highlights & Insights
- **Elegant Design**: The three-stage pipeline—mutual information → contrastive learning → gradient orthogonal projection—is tightly coupled with clear theoretical motivation.
- **Dual Role of MI**: Mutual information simultaneously provides interpretable parameter selection and substantially reduces the parameter search space.
- **Cross-Task Generalization**: FALCON achieves state-of-the-art performance across three unlearning benchmarks covering harmful knowledge, copyright, and entity forgetting.
- **Resistance to Recovery Attacks**: The method demonstrates strong robustness against jailbreak-based knowledge recovery attempts.

## Limitations & Future Work
- MI estimation relies on KDE with PCA dimensionality reduction; the threshold of 95% explained variance may not generalize across different model architectures.
- The single-layer intervention assumption may be limiting for models where knowledge is highly distributed across layers.
- The weighting hyperparameter $\eta$ in multi-domain unlearning lacks theoretical grounding for its selection.
- Experiments are conducted primarily on 7B-scale models; scalability to larger models remains unexplored.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The combination of contrastive learning and orthogonal projection is highly original; MI-guided parameter selection is a novel contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Three benchmarks, three model architectures, extensive ablation studies and comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with complete mathematical derivations.
- **Overall Value**: ⭐⭐⭐⭐ — A solid advancement in the field of LLM machine unlearning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] VITRIX-CLIPIN: Enhancing Fine-Grained Visual Understanding in CLIP via Instruction Editing Data and Long Captions](vitrix-clipin_enhancing_fine-grained_visual_understanding_in_clip_via_instructio.md)
- [\[NeurIPS 2025\] Uncovering Strategic Egoism Behaviors in Large Language Models](uncovering_strategic_egoism_behaviors_in_large_language_models.md)
- [\[NeurIPS 2025\] RoboCerebra: A Large-scale Benchmark for Long-horizon Robotic Manipulation Evaluation](robocerebra_a_large-scale_benchmark_for_long-horizon_robotic_manipulation_evalua.md)
- [\[ICCV 2025\] Selective Contrastive Learning for Weakly Supervised Affordance Grounding](../../ICCV2025/robotics/selective_contrastive_learning_for_weakly_supervised_affordance_grounding.md)
- [\[NeurIPS 2025\] CogVLA: Cognition-Aligned Vision-Language-Action Model via Instruction-Driven Routing & Sparsification](cogvla_cognition-aligned_vision-language-action_model_via_instruction-driven_rou.md)

</div>

<!-- RELATED:END -->
