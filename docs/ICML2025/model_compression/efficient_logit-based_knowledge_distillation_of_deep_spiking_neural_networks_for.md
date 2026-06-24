---
title: >-
  [Paper Note] Efficient Logit-based Knowledge Distillation of Deep Spiking Neural Networks for Full-Range Timestep Deployment
description: >-
  [ICML 2025][Model Compression][Spiking Neural Networks] A temporally decoupled logit distillation framework is proposed, which exploits the inherent spatio-temporal dynamics of SNNs to decompose the training objective to each timestep. This achieves high-performance deployment of a single model across a full range of inference timesteps without needing to retrain for different timesteps.
tags:
  - "ICML 2025"
  - "Model Compression"
  - "Spiking Neural Networks"
  - "Knowledge Distillation"
  - "Temporal Decoupling"
  - "Full-Range Timestep Deployment"
  - "Self-Distillation"
date: 2026-05-08
content_hash: 8b32094835a52bf6
---

# Efficient Logit-based Knowledge Distillation of Deep Spiking Neural Networks for Full-Range Timestep Deployment

**Conference**: ICML 2025  
**arXiv**: [2501.15925](https://arxiv.org/abs/2501.15925)  
**Code**: [GitHub](https://github.com/Intelli-Chip-Lab/snn_temporal_decoupling_distillation)  
**Area**: Human Understanding  
**Keywords**: Spiking Neural Networks, Knowledge Distillation, Temporal Decoupling, Full-Range Timestep Deployment, Self-Distillation  

## TL;DR

A temporally decoupled logit distillation framework is proposed, which exploits the inherent spatio-temporal dynamics of SNNs to decompose the training objective to each timestep. This achieves high-performance deployment of a single model across a full range of inference timesteps without needing to retrain for different timesteps.

## Background & Motivation

Spiking Neural Networks (SNNs), as biologically-inspired computing paradigms that transmit information using discrete binary spike sequences, offer significant energy-efficiency advantages on neuromorphic hardware. However, SNNs face two core challenges:

**Accuracy Gap**: Due to the non-differentiability of spiking activities and the limited representation capacity of binary feature maps, SNNs suffer from a noticeable performance degradation compared to full-precision Artificial Neural Networks (ANNs).

**Rigid Deployment**: The inference timesteps of SNNs are fixed at the time of deployment and must match the training timesteps. If there is a need to modify the inference timesteps, the model must be retrained for the new timesteps, which severely limits deployment flexibility.

Existing knowledge distillation methods directly transfer ANN distillation strategies to SNNs, treating SNNs as purely spatial end-to-end models while neglecting the inherent spatio-temporal characteristics of SNNs. These approaches use the SNN's ensemble voting output or average feature maps as distillation targets, failing to fully leverage the SNN's unique advantage of generating multiple sets of logits across multiple timesteps.

The core motivation of this paper is: **to treat the outputs of SNNs at different timesteps as ensemble learning along the temporal dimension**. By decoupling the training objective to each individual timestep, this approach not only enhances overall performance but also enables a single trained model to be flexibly deployed across a full range of timesteps.

## Method

### Overall Architecture

The temporally decoupled distillation framework proposed in this paper incorporates three types of label signals:

- **Truth Target**: Ground-truth one-hot label $\mathbf{y}$
- **Teacher Label**: Output logits of the pretrained ANN teacher model $\mathbf{z}^A$
- **Ensemble Label**: Voting outputs of the SNN itself across all timesteps $\mathbf{z}_{\text{ens}}^S$

While conventional methods define the loss on the final ensemble output, this paper decouples the loss target to the individual output of each timestep, ensuring that the implicit sub-model of each timestep is effectively trained.

### Key Designs

#### 1. From Standard Distillation to Temporal-Dimension Distillation

**Standard logit distillation** defines the loss on the SNN's ensemble voting output $\mathbf{z}_{\text{ens}}^S = \frac{1}{T}\sum_t \mathbf{z}^S(t)$:

$$\mathcal{L}_{\text{SKD}} = \mathcal{L}_{\text{SCE}} + \alpha \mathcal{L}_{\text{SKL}}$$

where $\mathcal{L}_{\text{SCE}}$ is the cross-entropy loss of the ensemble output, and $\mathcal{L}_{\text{SKL}}$ is the KL divergence with the teacher model.

**Temporal-dimension distillation** decouples the targets to each individual timestep:

- **Temporal-Wise Cross-Entropy (TWCE)**: Computes the hard label loss independently for each timestep

$$\mathcal{L}_{\text{TWCE}} = \frac{1}{T}\sum_t \mathcal{L}_{\text{CE}}(\mathbf{S}(\mathbf{z}^S(t)), \mathbf{y})$$

- **Temporal-Wise KL Divergence (TWKL)**: Computes the soft label loss with the teacher independently for each timestep

$$\mathcal{L}_{\text{TWKL}} = \frac{1}{T}\sum_t \mathcal{L}_{\text{KL}}(\mathbf{S}(\mathbf{z}^S(t)/\tau), \mathbf{S}(\mathbf{z}^A/\tau))$$

#### 2. Ensemble Logit Self-Distillation

A key innovation lies in utilizing the final voting logits as an additional soft label for self-distillation. Since the ensemble output typically outperforms individual timestep outputs, it serves as an "internal teacher" to guide the convergence of sub-models at each timestep:

$$\mathcal{L}_{\text{TWSD}} = \frac{1}{T}\sum_t \mathcal{L}_{\text{KL}}(\mathbf{S}(\mathbf{z}^S(t)/\tau), \mathbf{S}(\mathbf{z}_{\text{ens}}^S/\tau))$$

By utilizing information from existing backbone pathways, this self-distillation loss **does not introduce any additional forward computation branches**, yielding minimal computational overhead.

#### 3. Theoretical Convergence Guarantees

The paper presents three key theoretical results:

- **Lemma 1**: $\mathcal{L}_{\text{TWCE}}$ forms an upper bound of $\mathcal{L}_{\text{SCE}}$; optimizing the temporally decoupled objective is equivalent to optimizing the upper bound of the original objective.
- **Proposition 2**: $\mathcal{L}_{\text{TWKD}}$ forms an upper bound of $\mathcal{L}_{\text{SKD}}$, i.e., $\mathcal{L}_{\text{SKD}} \leq \mathcal{L}_{\text{TWKD}}$.
- **Proposition 3**: $\mathcal{L}_{\text{TWKD}}^{(T)}$ trained on timestep $T$ forms a scaled upper bound for any implicit sub-model $T_k \leq T$: $\mathcal{L}_{\text{SKD}}^{(T_k)} \leq \frac{T}{T_k}\mathcal{L}_{\text{TWKD}}^{(T)}$.

Proposition 3 serves as the theoretical foundation for full-range timestep deployment: when training a model with $T=6$, the implicit sub-models of $T=2, 4$ are also guaranteed to converge.

### Loss & Training

The final training objective is a weighted combination of the three loss components:

$$\mathcal{L}_{\text{final}} = \mathcal{L}_{\text{TWCE}} + \alpha \mathcal{L}_{\text{TWKL}} + \beta \mathcal{L}_{\text{TWSD}}$$

- $\alpha = 0.2$: Balancing factor for teacher distillation loss
- $\beta = 0.5$: Balancing factor for self-distillation loss
- $\tau$: Temperature scaling factor

Training pipeline:
1. Input samples pass through the SNN to obtain outputs at each timestep $\{\mathbf{z}^S(t)\}_{t \leq T}$
2. Input samples pass through the pretrained ANN to obtain teacher logits $\mathbf{z}^A$
3. Compute the ensemble voting output $\mathbf{z}_{\text{ens}}^S$
4. Compute the three loss components separately and perform weighted summation
5. Update SNN parameters based on the final loss

Key Advantage: Compared with standard logit distillation, the training overhead is identical, as it only alters where the loss is defined without introducing additional computational paths.

## Key Experimental Results

### Main Results

| Dataset | Model | Timesteps | Ours | Prev. SOTA | Gain |
|--------|------|--------|------|----------|------|
| CIFAR-10 | ResNet-19 | T=6 | **97.00%** | 96.82% (SM) | +0.18% |
| CIFAR-10 | ResNet-19 | T=4 | **96.97%** | 96.82% (SM) | +0.15% |
| CIFAR-10 | ResNet-19 | T=2 | **96.65%** | 96.19% (EnOF) | +0.46% |
| CIFAR-100 | ResNet-19 | T=6 | **82.56%** | 82.43% (EnOF) | +0.13% |
| CIFAR-100 | ResNet-19 | T=4 | **82.47%** | 81.70% (SM) | +0.77% |
| CIFAR-100 | ResNet-19 | T=2 | **81.47%** | 82.43% (EnOF) | Comparable |
| ImageNet | ResNet-34 | T=4 | **71.04%** | 70.04% (SAKD) | +1.00% |
| CIFAR10-DVS | ResNet-18 | T=10 | **86.40%** | 83.19% (SM) | +3.21% |
| CIFAR10-DVS | ResNet-18 | T=4 | **83.50%** | 81.50% (SAKD) | +2.00% |

### Ablation Study

| Configuration | CIFAR-100 (T=6) | Description |
|------|-----------------|------|
| $\mathcal{L}_{\text{TWCE}}$ only | 79.26% | Temporally decoupled cross-entropy only |
| + $\mathcal{L}_{\text{TWSD}}$ | 79.63% | With self-distillation, +0.37% |
| + $\mathcal{L}_{\text{TWKL}}$ | 79.56% | With teacher distillation, +0.30% |
| + $\mathcal{L}_{\text{TWKL}}$ & $\mathcal{L}_{\text{TWSD}}$ | **79.80%** | Full framework, +0.54% |
| Standard $\mathcal{L}_{\text{SCE}} + \mathcal{L}_{\text{SKL}}$ | 79.07% | Standard distillation baseline |
| Fully decoupled $\mathcal{L}_{\text{TWCE}} + \mathcal{L}_{\text{TWKL}}$ | 79.56% | Temporally decoupled distillation |

**Full-Range Timestep Deployment Analysis** (ResNet-19, CIFAR-100, Training T=6):

| Inference Timesteps | T=1 | T=2 | T=3 | T=4 | T=5 | T=6 |
|-----------|-----|-----|-----|-----|-----|-----|
| Ours (Training T=6) | 79.87% | 81.72% | 82.29% | 82.50% | 82.55% | 82.56% |
| Ours (Training T=4) | 79.40% | 81.58% | 82.14% | 82.47% | 82.39% | 82.49% |
| Standard KD (Training T=6) | 71.08% | 76.25% | 77.52% | 78.25% | 78.63% | 79.07% |

### Key Findings

1. **Full-Range Robustness**: The model trained with T=6 consistently outperforms standard distillation across all inference timesteps from T=1 to T=6. Moreover, the model trained with T=6 even surpasses the model trained specifically with T=2 when evaluated at T=2 inference.
2. **Zero Extra Training Overhead**: Compared to standard logit distillation, it only alters the loss definition location without adding extra computational branches.
3. **Mutual Complementarity of Three Loss Components**: TWCE, TWKL, and TWSD each contribute performance gains and are fully compatible with each other.
4. **t-SNE Visualization**: Sub-models at individual timesteps in temporally decoupled distillation exhibit highly consistent clustering patterns, validating the unified convergence of the implicit sub-models.

## Highlights & Insights

1. **Perspective Shift**: Treating the temporal step outputs of SNNs as sub-models in ensemble learning is a clever perspective. This analogy not only provides intuition for method design but also supports the theoretical analysis.
2. **Minimalist Design**: The entire method only alters the location of the loss function definition (from the ensemble output to individual timesteps), supplemented by a zero-cost self-distillation term. It requires no new modules, no extra forward paths, and no architectural modifications.
3. **Theory-Practice Consistency**: Proposition 3 predicts that training with larger timesteps guarantees the convergence of smaller-timestep sub-models, which is perfectly validated by the experiments in Table 7.
4. **Deployment Value**: The capability of a single model to support a full range of timesteps is highly significant for practical neuromorphic hardware deployment, as it allows dynamic scaling of inference timesteps based on real-time computational constraints.

## Limitations & Future Work

1. **Limited to Logit Distillation**: The paper explicitly focuses only on logit distillation without incorporating feature-level distillation. Extending the idea of temporal decoupling to feature-level distillation could potentially yield larger gains.
2. **Limited Architectural Scope**: The experiments are primary based on the ResNet family and have not been validated on Transformer-based SNNs or larger-scale models.
3. **Hyperparameter Sensitivity**: Although $\alpha=0.2$ and $\beta=0.5$ perform stably across multiple datasets, their generalizability to more diverse tasks requires further validation.
4. **Bound Tightness at Small Timesteps**: Theoretically, the scaled upper bound $\frac{T}{T_k}$ for $T_k \leq T$ becomes looser as $T_k$ decreases, which may provide a less tight guarantee for extremely small timesteps (e.g., T=1).
5. **Task Generalization**: The method has only been validated on classification tasks; downstream tasks such as detection and segmentation have not been explored.

## Related Work & Insights

- **TET (Deng et al., 2022)**: First proposed a temporally decoupled training target, serving as a major inspiration for this study. This work extends it to a distillation framework.
- **Self-Distillation (Zhang et al., 2019; Allen-Zhu & Li, 2020)**: Utilizes the model's own outputs as soft labels. This paper innovatively leverages the SNN's ensemble output for zero-cost self-distillation.
- **EnOF (Guo et al.)**: Views SNN timestep sub-models from an ensemble perspective and proposes KL divergence between adjacent timesteps. This paper further explores the relation between the ensemble output and each individual sub-model.
- **Future Directions**: The idea of temporal decoupling can be generalized to any models with sequential structures (such as RNNs and temporal Transformers), making its application in broader scenarios highly worth exploring.

## Rating

| Criterion | Score (1-5) | Description |
|------|-----------|------|
| Novelty | 4 | Clever interpretation of SNN timesteps from an ensemble learning perspective + zero-cost self-distillation design |
| Theoretical Depth | 4 | Provides complete upper-bound proofs and full-range timestep convergence guarantees |
| Experimental Thoroughness | 4 | Four datasets, multiple architectures, extensive ablation, and visualization analyses |
| Value | 5 | Full-range deployment of a single model is highly meaningful for practical SNN deployment |
| Writing Quality | 4 | Clear logic, progressing step-by-step from problem definition to methodology, theory, and experiments |
| **Total Score** | **4.2** | The method is simple and effective, showing strong consistency between theory and experiments |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Synergy between the Strong and the Weak: Spiking Neural Networks Are Inherently Superior in Temporal Processing](../../NeurIPS2025/model_compression/synergy_between_the_strong_and_the_weak_spiking_neural_networks_are_inherently_s.md)
- [\[AAAI 2026\] A Closer Look at Knowledge Distillation in Spiking Neural Network Training](../../AAAI2026/model_compression/a_closer_look_at_knowledge_distillation_in_spiking_neural_ne.md)
- [\[ICML 2025\] FGFP: A Fractional Gaussian Filter and Pruning for Deep Neural Networks Compression](fgfp_a_fractional_gaussian_filter_and_pruning_for_deep_neural_networks_compressi.md)
- [\[NeurIPS 2025\] Spiking Brain Compression: Post-Training Second-Order Compression for Spiking Neural Networks](../../NeurIPS2025/model_compression/spiking_brain_compression_post-training_second-order_compression_for_spiking_neu.md)
- [\[ICCV 2025\] Local Dense Logit Relations for Enhanced Knowledge Distillation](../../ICCV2025/model_compression/local_dense_logit_relations_for_enhanced_knowledge_distillation.md)

</div>

<!-- RELATED:END -->
