---
title: >-
  [Paper Note] Single-Teacher View Augmentation: Boosting Knowledge Distillation via Angular Diversity
description: >-
  [NeurIPS 2025][Model Compression][Knowledge Distillation] This paper proposes Angular-KD, which attaches multiple lightweight linear branches to a single teacher model and introduces two angular diversity losses — a cons…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "Knowledge Distillation"
  - "Knowledge Augmentation"
  - "Angular Diversity"
  - "Ensemble Learning"
date: 2026-05-08
content_hash: ccceea672a0bd3d7
---

# Single-Teacher View Augmentation: Boosting Knowledge Distillation via Angular Diversity

**Conference**: NeurIPS 2025  
**arXiv**: [2510.22480](https://arxiv.org/abs/2510.22480)  
**Authors**: Seonghoon Yu (GIST), Dongjun Nam (POSTECH), Dina Katabi (MIT CSAIL), Jeany Son (POSTECH)  
**Code**: [github.com/june6423/Angular-KD](https://github.com/june6423/Angular-KD)  
**Area**: Model Compression  
**Keywords**: Knowledge Distillation, Knowledge Augmentation, Angular Diversity, Ensemble Learning, Model Compression  

## TL;DR

This paper proposes Angular-KD, which attaches multiple lightweight linear branches to a single teacher model and introduces two angular diversity losses — a constrained inter-angle diversity loss and an intra-angle diversity loss — to generate diverse supervisory signals from a single teacher. This approach serves as a low-cost alternative to multi-teacher distillation and achieves state-of-the-art performance across multiple KD benchmarks.

## Background & Motivation

### State of the Field
Knowledge distillation (KD) transfers knowledge from a large teacher model to a compact student model and is a core technique in model compression. Recent multi-teacher distillation methods aggregate complementary knowledge from multiple teachers to significantly improve student generalization; however, they require training and maintaining multiple large models, incurring substantial computational and memory overhead.

### Limitations of Prior Work
- **Multi-teacher distillation**: Requires training multiple independent teacher models, multiplying resource costs; diversity obtained via different random seeds under the same architecture is inherently limited by shared structural biases.
- **TeKAP (prior knowledge augmentation method)**: Injects random noise perturbations into single-teacher features to simulate multi-view supervision, reducing computational cost, but diversity is entirely noise-driven with no control over semantic structure or informativeness.
- **Key Challenge**: Can a more structured and controllable knowledge augmentation approach be designed to generate semantically rich and diverse supervisory signals within a single-teacher framework?

## Method

### Overall Architecture: Angular-KD
Angular-KD attaches $N$ lightweight linear branches (View Augmentation Heads) to a pretrained teacher model, generates $N$ augmented views, and then ensembles them together with the original teacher output to produce a stronger supervisory signal for distillation into the student.

### View Augmentation Heads
1. **Feature extraction**: The teacher network extracts a feature vector $\mathbf{F}^T \in \mathbb{R}^{d_T}$ and logit probabilities $\mathbf{Z}^T \in \mathbb{R}^C$.
2. **Feature-level augmentation**: $N$ independent linear heads $\{\phi_i\}_{i=1}^N$ with orthogonal initialization; different Dropout masks with varying probabilities (0.2–0.4) are applied to each head's input, yielding augmented features $\mathbf{F}_i^A = \text{BN}(\mathbf{W}^{\phi_i}(M_i \odot \mathbf{F}^T))$.
3. **Logit-level augmentation**: Each augmented feature is transformed into class probabilities via an independent logit head $\psi_i$: $\mathbf{Z}_i^A = \sigma(\mathbf{W}^{\psi_i}\mathbf{F}_i^A / \tau^Z)$.

### Angular Diversity Losses

#### Constrained Inter-angle Diversity Loss
This loss maximizes angular separation among augmented views while constraining each augmented view to remain within a learnable angular margin $\gamma$ of the teacher output. It consists of two components:

- **Constraint term**: Ensures augmented views do not deviate too far from the teacher representation, preventing drift toward non-target class boundaries.
- **Diversity term**: Activated once all views satisfy the constraint, further maximizing inter-view angular separation.
- Cosine similarity is used instead of arccosine to ensure numerical stability.

#### Intra-angle Diversity Loss
This loss ensures that augmented views are uniformly distributed around the teacher output. It computes the offset vector of each augmented view relative to the teacher, $\mathbf{\Delta}_i^{T-A} = \mathbf{R}^T - \mathbf{R}_i^A$, and minimizes the cosine similarity among these offset vectors to promote structured, uniform dispersion.

#### Overall Augmentation Loss
$$\mathcal{L}^{\text{aug}} = \mathcal{L}_{\text{inter}}^{\text{aug}} + \mathcal{L}_{\text{intra}}^{\text{aug}} + \mathcal{L}_{\text{gt}}^{\text{aug}}$$
where $\mathcal{L}_{\text{gt}}^{\text{aug}}$ is cross-entropy label supervision on the augmented logits, preventing the augmented predictions from drifting away from true semantics.

### Distillation Process
An $(N+1)$-way ensemble is constructed by uniformly averaging the original teacher and the $N$ augmented views as the supervisory signal. The distillation loss includes:
- Feature-level CRD contrastive distillation loss
- Logit-level KL divergence loss
- Student cross-entropy label loss

Training proceeds with a 30-epoch warmup phase in which only the augmentation heads are trained, followed by 240 epochs of joint training of the augmentation heads and the student.

### Theoretical Analysis
The paper proves that the ensemble diversity metric $\mathbb{D}$ increases as the inter-view cosine similarity $s_{ij}^A$ and the offset-vector cosine similarity $s_{ij}^{\Delta}$ decrease. The inter-angle loss explicitly minimizes $s_{ij}^A$, while the intra-angle loss minimizes $s_{ij}^{\Delta}$. Increased diversity directly tightens the upper bound on the ensemble expected loss, providing theoretical justification for the design.

## Key Experimental Results

### Main Results on CIFAR-100 (ResNet32x4 → ResNet8x4)

| Distillation Type | No Aug. | TeKAP | **Angular-KD** |
|---------|-------|-------|---------------|
| Logit-level (KD) | 73.33 | 74.79 | **76.08** |
| Feature-level (CRD) | 75.51 | 75.65 | **75.82** |
| Combined (KD+CRD) | 75.46 | 75.98 | **76.46** |

### ImageNet (ResNet34 → ResNet18)

| Metric | No Aug. | TeKAP | **Angular-KD** |
|-----|-------|-------|---------------|
| Top-1 | 70.41 | 70.67 | **71.07** |
| Top-5 | 89.88 | 89.92 | **90.39** |

### Comparison with Multi-Teacher Methods (WideRN-40-2 → WideRN-16-2)

| Method | Acc. | Params (M) | FLOPs (M) |
|-----|------|----------|---------|
| Ensemble Distil. (5 teachers) | 76.31 | 11.28 | 1645 |
| TAKD (4 teachers) | 75.04 | 6.69 | 797 |
| DGKD (4 teachers) | 76.24 | 6.69 | 797 |
| TeKAP (single teacher) | 76.20 | 2.26 | 329 |
| **Angular-KD (single teacher)** | **76.33** | 2.40 | 329 |

Angular-KD surpasses all multi-teacher methods at single-teacher computational cost.

### Cross-Dataset Transfer (CIFAR-100-distilled Student → New Datasets)

| Dataset | No Aug. | TeKAP | **Angular-KD** |
|-------|-------|-------|---------------|
| STL-10 | 68.01 | 68.71 | **70.23** |
| TinyImageNet | 31.17 | 31.54 | **32.97** |

### Plug-and-Play Integration with Existing KD Methods (RN32x4 → RN8x4)

| KD Method | Original | +TeKAP | +Angular-KD |
|-------|------|--------|------------|
| DKD | 76.32 | 76.65 | **76.51** |
| MLKD | 77.08 | 77.04 | **77.28** |

### Ablation Study

| Inter-angle | Intra-angle | Acc. | Ensemble Diversity |
|------------|------------|------|----------|
| ✗ | ✗ | 75.46 | - |
| ✓ | ✗ | 76.16 | 11.522 |
| ✗ | ✓ | 76.28 | 11.617 |
| ✓ | ✓ | **76.46** | **11.633** |

The optimal number of augmentation heads is $N=5$; each head adds only approximately 0.092M parameters and 0.092M FLOPs, representing negligible overhead.

## Highlights & Insights

- **Efficient alternative to multi-teacher distillation**: A single teacher with lightweight linear branches surpasses 5-teacher ensemble distillation, reducing parameters and FLOPs by approximately 5×.
- **Structured angular diversity**: Unlike TeKAP's random noise, Angular-KD explicitly controls diversity through complementary constrained inter-angle and intra-angle losses, balancing semantic consistency with view dispersion.
- **Theoretical guarantees**: Angular diversity is proven to directly improve the ensemble diversity metric, which in turn tightens the upper bound on ensemble expected loss.
- **Plug-and-play**: Compatible with existing KD frameworks such as DKD, ReviewKD, and MLKD, consistently yielding performance gains.
- **Strong generalizability**: Effective across image classification, binary segmentation, imbalanced data, few-shot learning, and cross-dataset transfer scenarios.

## Limitations & Future Work

- **Bounded by teacher knowledge**: Augmented views are fundamentally constrained by the original teacher's knowledge and cannot introduce genuinely new semantic information.
- **Training overhead**: Despite being more efficient than multi-teacher approaches, generating multiple views still incurs non-negligible additional training cost.
- **Bias propagation risk**: Since all augmentations originate from a single teacher, biases present in the teacher may be transferred to or even amplified in the student.
- **Lack of semantic grounding in angular perturbations**: Augmentation is not semantically driven, which may limit interpretability.
- **Validation limited to classification and segmentation**: The method has not been verified on detection, NLP, or other task types.

## Related Work & Insights

- **TeKAP**: Also a single-teacher augmentation approach, but relies on random noise perturbations with uncontrolled diversity; Angular-KD consistently outperforms it across all experiments by explicitly optimizing diversity through angular losses.
- **Ensemble Distillation**: Uses 5 teachers at approximately 5× the computational cost, yet achieves lower accuracy.
- **TAKD/DGKD**: Employ multiple teachers of varying scales in a teacher chain, with approximately 3× the parameters and FLOPs of Angular-KD.
- **ArcFace/CosFace and related angular learning methods**: Impose angular margins between different classes to enhance classification boundaries; Angular-KD instead maximizes angular separation among multiple augmented views of the same teacher — a fundamentally different objective.

## Rating

- Novelty: ⭐⭐⭐⭐ — Angular diversity-driven knowledge augmentation is a novel idea; the inter/intra dual-loss design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive evaluation covering multiple datasets, architectures, tasks, ablations, visualizations, transfer, and few-shot settings.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with well-integrated theoretical analysis and experiments; motivation is thoroughly articulated.
- Value: ⭐⭐⭐⭐ — Highly practical; the plug-and-play nature lowers the barrier to adoption, and surpassing multi-teacher methods with a single teacher offers real deployment value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] A Good Teacher Adapts Their Knowledge for Distillation](../../ICCV2025/model_compression/a_good_teacher_adapts_their_knowledge_for_distillation.md)
- [\[NeurIPS 2025\] ATLAS: Autoformalizing Theorems through Lifting, Augmentation, and Synthesis of Data](atlas_autoformalizing_theorems_through_lifting_augmentation_and_synthesis_of_dat.md)
- [\[NeurIPS 2025\] KINDLE: Knowledge-Guided Distillation for Prior-Free Gene Regulatory Network Inference](kindle_knowledge-guided_distillation_for_prior-free_gene_regulatory_network_infe.md)
- [\[NeurIPS 2025\] A Token is Worth over 1,000 Tokens: Efficient Knowledge Distillation through Low-Rank Clone](a_token_is_worth_over_1000_tokens_efficient_knowledge_distillation_through_low-r.md)
- [\[CVPR 2026\] Distilling Balanced Knowledge from a Biased Teacher](../../CVPR2026/model_compression/distilling_balanced_knowledge_from_a_biased_teacher.md)

</div>

<!-- RELATED:END -->
