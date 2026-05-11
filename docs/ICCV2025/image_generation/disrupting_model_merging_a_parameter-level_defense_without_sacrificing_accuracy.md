---
title: >-
  [Paper Note] Disrupting Model Merging: A Parameter-Level Defense Without Sacrificing Accuracy
description: >-
  [ICCV 2025][Image Generation][model merging defense] This paper proposes PaRaMS (Parameter Rearrangement & Random Multi-head Scaling), a parameter-level proactive defense method that displaces a protected model away from…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "model merging defense"
  - "intellectual property protection"
  - "parameter rearrangement"
  - "attention head scaling"
  - "functionally equivalent transformation"
date: 2026-05-08
content_hash: 2155d130d7effddd
---

# Disrupting Model Merging: A Parameter-Level Defense Without Sacrificing Accuracy

**Conference**: ICCV 2025
**arXiv**: [2503.07661](https://arxiv.org/abs/2503.07661)
**Code**: None
**Area**: Diffusion Models / Image Generation
**Keywords**: model merging defense, intellectual property protection, parameter rearrangement, attention head scaling, functionally equivalent transformation

## TL;DR

This paper proposes PaRaMS (Parameter Rearrangement & Random Multi-head Scaling), a parameter-level proactive defense method that displaces a protected model away from the shared loss basin via functionally equivalent parameter transformations, causing severe performance degradation upon merging while preserving original performance when the model is used standalone.

## Background & Motivation

The pretrain-then-finetune paradigm has made model merging a low-cost approach to acquiring specialized capabilities from others. Through simple linear combinations of parameters (e.g., Task Arithmetic), a free-rider can merge an openly released fine-tuned model into their own, inheriting its specialized capabilities at virtually zero cost. This raises serious intellectual property (IP) concerns:

**Limitations of passive defenses**: Existing watermarking and fingerprinting techniques can only detect infringement after merging has occurred and cannot prevent the merging itself.

**Concealment of merging operations**: Unlike code copying, model merging blends parameter spaces, making it difficult to trace the source.

**Extremely low cost of IP infringement**: Nearly 30,000 merged models already exist on Hugging Face, and model merging has become a widespread practice.

The central research question is therefore: **How can a model be made to degrade in performance when merged, without altering its original functionality?** This requires simultaneously satisfying two conditions: (1) functional equivalence before merging; and (2) significant performance degradation after merging.

The key insight of this paper stems from loss landscape analysis: models fine-tuned from the same pretrained checkpoint typically reside in a shared low-loss basin, which is the fundamental reason model merging succeeds. If a model can be displaced into a different basin via functionally equivalent transformations, merging will fail.

## Method

### Overall Architecture

PaRaMS consists of two complementary parameter transformation modules targeting the two core components of the Transformer architecture: MLP layers and Attention layers. Both transformations guarantee functional equivalence (model outputs remain completely unchanged) while substantially displacing parameters in the parameter space, thereby disrupting merging. The final defense is the composition of the two: $\eta = \eta_{\text{perm}} \circ \eta_{\text{scaling}}$.

### Key Designs

1. **Parameter Rearrangement (for MLP layers)**:

    - Function: Reorders hidden-layer neurons in MLP layers via permutation matrices.
    - Mechanism: For a two-layer MLP $\text{MLP}(X) = W_2 \sigma(W_1 X + b_1) + b_2$, a permutation matrix $P$ is introduced such that:
    $W_1' = PW_1, \quad b_1' = Pb_1, \quad W_2' = W_2 P^T$
      Since $\sigma$ is an element-wise activation function, the effects of $P$ and $P^T$ cancel each other out, leaving the output unchanged.
    - Design Motivation: The permutation matrix is chosen to maximize the parameter distance from the pretrained model:
    $\arg\max_{\eta_{\text{perm}}} \|\theta_{\text{pre}}^{\text{MLP}} - \eta_{\text{perm}}(\theta_{\text{def}}^{\text{MLP}})\|^2$
      This optimization is reformulated as a Linear Assignment Problem and solved efficiently.

2. **Random Multi-head Scaling (for Attention layers)**:

    - Function: Applies diagonal scaling transformations to the Q/K/V matrices of Attention modules.
    - Mechanism: For each attention head $i$, diagonal matrices $A_i$ and $B_i$ are sampled with diagonal entries drawn from $\mathcal{U}(s_{\min}, s_{\max})$, and the following transformations are applied:
    $Q_i \leftarrow Q_i A_i, \quad K_i \leftarrow K_i A_i^{-1}$
    $V_i \leftarrow V_i B_i, \quad W_O[:,i] \leftarrow B_i^{-1} W_O[:,i]$
      Functional invariance is guaranteed by the identity $QK^T = QA(KA^{-1})^T$.
    - Design Motivation: Random scaling substantially shifts attention layer parameters, complementing MLP rearrangement to jointly cover all key modules in the Transformer.

3. **Dropout-based Pruning Defense (against adaptive attacks)**:

    - Function: Targets potential adaptive bypass attacks (e.g., searching for inverse permutations) by augmenting robustness with dropout-based pruning.
    - Mechanism: Dropout is applied to a subset of parameters before rearrangement, so that even if an attacker recovers the permutation order, the parameters cannot be fully restored.
    - Design Motivation: Various potential countermeasures by adversaries are considered, ensuring the defense remains effective in adversarial scenarios.

### Loss & Training

PaRaMS is a **post-processing** method that requires no retraining. The defense procedure only requires:
- Solving for the optimal permutation matrix for each MLP layer (linear programming)
- Sampling random scaling factors for each attention head
- Applying a one-time parameter transformation with minimal computational overhead

## Key Experimental Results

### Main Results

Image classification (ViT-B-32, Task Arithmetic merging, $\lambda=0.8$):

| Setting | MMP− Acc. (%) | MMP+ Acc. (%) | Drop |
|---------|--------------|--------------|------|
| Cars/RESISC45 | 70.29/94.24 | 0.51/2.13 | >65% |
| EuroSAT/SVHN | 98.06/95.90 | 9.67/19.41 | >75% |
| GTSRB/DTD | 98.20/67.82 | 1.93/2.66 | >65% |
| MNIST/RESISC45 | 99.60/90.62 | 2.10/10.10 | >80% |

Image generation (Stable Diffusion 1.5, text-image alignment score):

| Model | UMP− | UMP+ | MMP− | MMP+ |
|-------|------|------|------|------|
| Prompt1 | 0.3286 | 0.3306 | 0.3416 | **0.1277** |
| Prompt2 | 0.3335 | 0.3428 | 0.3386 | **0.0820** |

### Ablation Study

Defense effectiveness under different merging methods (ViT-B-32 average accuracy):

| Merging Method | MMP− | MMP+ | Drop |
|----------------|------|------|------|
| Task Arithmetic | ~75% | <10% | >65% |
| TIES-Merging | ~70% | <10% | >60% |
| Weight Average | ~65% | <15% | >50% |
| AdaMerging | ~75% | <10% | >65% |
| TA + DARE | ~70% | <10% | >60% |

Text classification (Llama2):

| Dataset | UMP−/UMP+ | MMP− (TA) | MMP+ (TA) |
|---------|-----------|-----------|-----------|
| Emotion | 99.8 | 97.6 | **21.4** |
| Twitter | 99.7 | 95.3 | **35.8** |

### Key Findings

- PaRaMS effectively disrupts merging across **all** evaluated merging methods (TA, TIES, WA, ADA, and their DARE variants), typically reducing MMP+ accuracy to below 10%.
- UMP− and UMP+ performance are **identical**, validating functional equivalence.
- The defense is effective when merging between 2 and 7 models.
- The method generalizes across tasks (classification, generation, NLP) and architectures (ViT, SD, Llama2).

## Highlights & Insights

- **First proactive defense**: Shifts the paradigm from passive detection to active prevention, addressing a critical gap in model merging security.
- **Guaranteed functional equivalence**: Mathematically rigorous proofs confirm that the transformations do not alter model outputs, with no approximations required.
- **Loss landscape-grounded analysis**: The "shared basin" perspective elegantly explains both why model merging succeeds and how to disrupt it.
- **Multi-modal validation**: The method's broad applicability is verified across vision, language, and generative tasks.

## Limitations & Future Work

- If an attacker possesses substantial computational resources (e.g., performing knowledge distillation rather than parameter merging), the defense may be circumvented.
- For PEM-TA scenarios that merge only LoRA parameters, the defense is relatively weaker since only scaling transformations can be applied.
- The choice of the random scaling range $[s_{\min}, s_{\max}]$ influences defense effectiveness; identifying the optimal range remains an open problem.
- The scenario in which multiple defended models are merged with each other has not been explored.

## Related Work & Insights

- Complementary to model watermarking (IPR protection) techniques: watermarking detects infringement, while PaRaMS prevents it.
- The permutation symmetry of neural networks is a long-studied topic; this paper repurposes it cleverly for security defense.
- Key insight: the success of model merging relies on shared structure in parameter space, and disrupting this structure is sufficient to defeat merging.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to propose the concept of proactively defending against model merging; both the problem formulation and solution are highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive validation across tasks, architectures, and merging methods, with analysis of adaptive attacks.
- Writing Quality: ⭐⭐⭐⭐ Problem formalization is clear and the threat model is well-defined.
- Value: ⭐⭐⭐⭐ Significant practical relevance in the area of open-source model security.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Efficient Input-Level Backdoor Defense on Text-to-Image Synthesis via Neuron Activation Variation](efficient_input-level_backdoor_defense_on_text-to-image_synthesis_via_neuron_act.md)
- [\[ICCV 2025\] Omegance: A Single Parameter for Various Granularities in Diffusion-Based Synthesis](omegance_a_single_parameter_for_various_granularities_in_diffusion-based_synthes.md)
- [\[ICCV 2025\] Towards Robust Defense against Customization via Protective Perturbation Resistant to Diffusion-based Purification](towards_robust_defense_against_customization_via_protective_perturbation_resista.md)
- [\[ICLR 2026\] Latent Diffusion Model without Variational Autoencoder](../../ICLR2026/image_generation/latent_diffusion_model_without_variational_autoencoder.md)
- [\[ICCV 2025\] DCT-Shield: A Robust Frequency Domain Defense against Malicious Image Editing](dct-shield_a_robust_frequency_domain_defense_against_malicious_image_editing.md)

</div>

<!-- RELATED:END -->
