---
title: >-
  [Paper Note] Causal-JEPA: Learning World Models through Object-Level Latent Masking
description: >-
  [ICML2026][Causal Inference][World models] Ours proposes C-JEPA, which extends the mask prediction of JEPA from the image patch level to the level of object-centric latent representations. By utilizing object-level maski…
tags:
  - "ICML2026"
  - "Causal Inference"
  - "World models"
  - "object-level masking"
  - "JEPA"
  - "causal inductive bias"
  - "object-centric representations"
date: 2026-05-08
content_hash: 93a2ebef99dd56a4
---

# Causal-JEPA: Learning World Models through Object-Level Latent Masking

**Conference**: ICML2026  
**arXiv**: [2602.11389](https://arxiv.org/abs/2602.11389)  
**Code**: https://github.com/galilai-group/cjepa  
**Area**: Causal Inference/World Models  
**Keywords**: World models, object-level masking, JEPA, causal inductive bias, object-centric representations  

## TL;DR

Ours proposes C-JEPA, which extends the mask prediction of JEPA from the image patch level to the level of object-centric latent representations. By utilizing object-level masking as a latent intervention, the model is forced to learn interaction-dependent dynamics. This results in an approximately 20% improvement in counterfactual reasoning over unmasked baselines and achieves comparable performance in control tasks using only 1% of the tokens, while accelerating planning by more than 8x.

## Background & Motivation

**Background**: World models provide a unified framework for scalable planning and control by learning, predicting, and reasoning about environment dynamics within a latent space. Object-centric representations (e.g., Slot Attention) serve as useful abstractions and have been widely adopted for learning visual dynamics and constructing world models.

**Limitations of Prior Work**: Purely using object-centric representations is insufficient to capture interaction-dependent dynamics. Existing research indicates that without an explicit mechanism to guide interaction learning, models tend to degenerate by relying on an object's own self-dynamics or exploiting spurious correlations. Current methods enforce interactions by separating temporal dynamics from object interactions, regularizing attention sparsity, utilizing graph structures, or relying on downstream task-specific methods, but these approaches either introduce additional architectural constraints or depend on reconstruction loss.

**Key Challenge**: Existing patch-level mask prediction methods (such as I-JEPA and V-JEPA) optimize for local patch correlations, failing to mandate object-level interaction reasoning. How interaction structures can become functionally necessary through the learning objective itself remains an open problem.

**Goal**: Design a simple and flexible object-centric world model where interaction reasoning is a necessary condition for minimizing the prediction objective, rather than being enforced through architectural constraints or reconstruction loss.

**Key Insight**: If the historical latent trajectory of a specific object is masked during training, the model must infer the state of the masked object from the state evolution of other objects. This essentially constitutes a counterfactual prediction query, preventing shortcuts like trivial temporal interpolation.

**Core Idea**: Elevate JEPA's mask prediction from the patch level to the object level. By using object-level latent masks as observational interventions, the predictor is forced to depend on interaction-related variables, thereby introducing a causal inductive bias.

## Method

### Overall Architecture

The C-JEPA workflow consists of three stages: (1) utilizing a frozen object-centric encoder (e.g., VideoSAUR) to encode video frames into object-level slot representations $S_t = \{s_t^1, \dots, s_t^N\}$; (2) masking the slots of selected objects within a history window, retaining only the earliest time step as an identity anchor; (3) employing a predictor (a ViT-style bidirectional attention Transformer) to simultaneously recover masked historical slots and predict future slots. During inference, no masking is performed, and forward prediction is conducted directly from the full observed history.

### Key Designs

1.  **Object-Level Latent Masking**:
    *   **Function**: Masks the latent representations of selected objects across the entire history window during training to construct structured partial observability.
    *   **Mechanism**: Given a set of mask indices $\mathcal{M} \subset \{1,\dots,N\}$, the slots of masked objects are replaced with mask tokens $\tilde{z}_\tau^i = \phi(z_{t_0}^i) + e_\tau$, where $\phi$ is a linear projection, $z_{t_0}^i$ is the identity anchor from the earliest time step, and $e_\tau$ is a learnable embedding with temporal position encoding. Since slot representations are permutation-equivariant, identity anchors are necessary to allow the Transformer to identify which entity is being masked.
    *   **Design Motivation**: Unlike patch-level masking which only optimizes local correlations, object-level masking ensures that the state of a masked object must be inferred from the interactions of other objects. This essentially forms a counterfactual query and prevents self-dynamic shortcuts.

2.  **Joint Masked-History and Forward Prediction Objective**:
    *   **Function**: Simultaneously optimizes history mask reconstruction and future state prediction.
    *   **Mechanism**: The total loss is defined as $\mathcal{L}_{\text{mask}} = \mathcal{L}_{\text{history}} + \mathcal{L}_{\text{future}}$, where $\mathcal{L}_{\text{history}}$ computes the L2 reconstruction error for masked object tokens in the history window, and $\mathcal{L}_{\text{future}}$ computes the L2 prediction error for all future tokens. The predictor takes the masked sequence $\bar{Z}_\mathcal{T}$ as input and outputs $\hat{Z}_\mathcal{T} = f(\bar{Z}_\mathcal{T})$.
    *   **Design Motivation**: The history term suppresses the model's reliance on self-dynamics under partial observability, while the future term aligns with forward world modeling. The combination makes interaction reasoning a necessity for minimizing the objective.

3.  **Auxiliary Variables as Independent Entity Nodes**:
    *   **Function**: Inputs action and proprioception signals as independent tokens to the predictor, rather than concatenating them into object representations.
    *   **Mechanism**: Defines the set of entity tokens as $Z_t = \{S_t, U_t\}$, where $U_t = \{a_t, p_t\}$ contains actions and proprioception signals. Auxiliary variables participate in attention calculations as additional conditioning tokens without being mixed into the object slots.
    *   **Design Motivation**: Experiments demonstrate that treating auxiliary variables as independent entities is significantly superior to concatenation, as it maintains the purity of object representations and allows the model to explicitly model action-object interactions.

## Key Experimental Results

### Main Results — CLEVRER Visual Question Answering

| Model | Encoder | Mask Count $\|\mathcal{M}\|$ | Overall Acc (%) | Counterfactual per-opt (%) | Counterfactual per-que (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| OC-JEPA | VideoSAUR | 0 | 82.79 | 79.53 | 47.68 |
| C-JEPA | VideoSAUR | 4 | **89.40** | **88.67** | **68.81** |
| SlotFormer | SAVi | — | 79.44 | 79.28 | 47.29 |
| SlotFormer (w/o Recon) | SAVi | — | 44.94 | 55.62 | 11.10 |
| OCVP-Seq | SAVi | — | 83.11 | 83.21 | 56.06 |
| C-JEPA | SAVi | 2 | **83.88** | **85.16** | **60.19** |

### Push-T Robotic Manipulation Task

| Model | Token Count × Dim | Success Rate (%) | Planning Time |
| :--- | :--- | :--- | :--- |
| DINO-WM | 196 × 384 | 91.33 | 5763 s |
| DINO-WM-Reg. | 196 × 384 | 88.00 | — |
| OC-DINO-WM | 6 × 128 | 60.67 | — |
| OC-JEPA | 6 × 128 | 76.00 | — |
| C-JEPA | 6 × 128 | **88.67** | **673 s (8× speedup)** |

### Key Findings
- The gains from object-level masking are most significant in counterfactual reasoning: counterfactual per-question accuracy improved from 47.68% to 68.81% (+21.13%), which is much larger than the overall accuracy improvement (+6.61%). This indicates that masking indeed enhances counterfactual reasoning rather than just improving prediction precision.
- Excessive masking can remove meaningful dependencies: when using the SAVi encoder, masking 4 objects resulted in a 4% drop, suggesting that the optimal masking ratio depends on the representation quality of the encoder.
- C-JEPA achieves control performance comparable to patch-level world models using only 1.02% of the token space (6×128 vs 196×384), resulting in a planning speedup of over 8x.
- SlotFormer's performance plummeted by 34.5% after removing the reconstruction loss, indicating its heavy reliance on pixel-level supervision; C-JEPA requires no reconstruction loss at all.

## Highlights & Insights
- **Object-Level Masking as Latent Intervention**: Reinterprets the masking operation as an intervention on the predictor’s observability, essentially creating counterfactual queries during training. This perspective cleverly links self-supervised masked learning with causal reasoning without requiring actual causal graphs or multi-environment data.
- **Efficiency-Performance Synergy**: Object-centric representations reduce the token count from 196 to 6. Combining this with object-level masking recovers performance lost due to representation compression, achieving an 8x planning speedup. This paradigm has direct value for real-time robotic control.
- **Neighborhood of Influence Theory**: Formalizes the concept of the "minimal sufficient set of context variables," proving that object-level masking makes interaction reasoning a necessary condition for optimal prediction, providing a theoretical foundation for the masking strategy.

## Limitations & Future Work
- Performance is limited by the quality of the object-centric encoder: excessive masking on the SAVi encoder leads to performance degradation, indicating that the encoder’s representation capability is a system bottleneck.
- Influence neighborhood correctness has not been verified on datasets with explicit temporal causal graphs.
- Experimental scenarios are relatively simple (CLEVRER synthetic videos, Push-T 2D manipulation); more complex 3D scenes and multi-agent interactions remain to be validated.
- Future Directions: Jointly fine-tune the object-centric encoder to avoid representation collapse; extend to more complex interaction environments.

## Related Work & Insights
- **JEPA Series**: I-JEPA → V-JEPA → V-JEPA2; ours is the first to combine JEPA with object-centric world models.
- **DINO-WM**: A patch-level world model baseline; performs well but has high token overhead. C-JEPA achieves equivalent performance using object-level representations.
- **SlotFormer / OCVP-Seq**: Previous object-centric world models that rely on reconstruction loss or architectural separation to guide interaction learning.
- **Insight**: The idea of using object-level masking as an inductive bias can be transferred to other fields requiring interaction reasoning, such as multi-agent reinforcement learning, social behavior prediction, and molecular dynamics simulation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Distributional Equivalence in Linear Non-Gaussian Latent-Variable Cyclic Causal Models](../../ICLR2026/causal_inference/distributional_equivalence_in_linear_non-gaussian_latent-variable_cyclic_causal_.md)
- [\[ICML 2026\] Causal Fine-Tuning under Latent Confounded Shift](causal_fine-tuning_under_latent_confounded_shift.md)
- [\[NeurIPS 2025\] Bi-Level Decision-Focused Causal Learning for Large-Scale Marketing Optimization](../../NeurIPS2025/causal_inference/bi-level_decision-focused_causal_learning_for_large-scale_marketing_optimization.md)
- [\[AAAI 2026\] Causally-Grounded Dual-Path Attention Intervention for Object Hallucination Mitigation in LVLMs](../../AAAI2026/causal_inference/causally-grounded_dual-path_attention_intervention_for_objec.md)
- [\[ACL 2026\] Learning Invariant Modality Representation for Robust Multimodal Learning from a Causal Inference Perspective](../../ACL2026/causal_inference/learning_invariant_modality_representation_for_robust_multimodal_learning_from_a.md)

</div>

<!-- RELATED:END -->
