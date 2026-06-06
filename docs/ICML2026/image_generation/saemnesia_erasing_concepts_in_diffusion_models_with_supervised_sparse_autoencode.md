---
title: >-
  [Paper Note] SAEmnesia: Erasing Concepts in Diffusion Models with Supervised Sparse Autoencoders
description: >-
  [ICML 2026][Image Generation][Concept Erasing] By adding a supervised "concept-latent" assignment loss during Sparse Autoencoder (SAE) training…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Concept Erasing"
  - "Sparse Autoencoders"
  - "Supervised Training"
  - "Feature Centralization"
  - "Diffusion Models"
date: 2026-05-08
content_hash: 97ae71c7c278103c
---

# SAEmnesia: Erasing Concepts in Diffusion Models with Supervised Sparse Autoencoders

**Conference**: ICML 2026  
**arXiv**: [2509.21379](https://arxiv.org/abs/2509.21379)  
**Code**: https://github.com/EIDOSLAB/SAEmnesia  
**Area**: AI Safety / Concept Unlearning / Diffusion Model Interpretability  
**Keywords**: Concept Erasing, Sparse Autoencoders, Supervised Training, Feature Centralization, Diffusion Models

## TL;DR
By adding a supervised "concept-latent" assignment loss during Sparse Autoencoder (SAE) training, each target concept is forced to concentrate on a single neuron (feature centralization). This reduces concept erasing in diffusion models from a 2D hyperparameter search ("find neurons + adjust strength") to a single multiplier adjustment. On UnlearnCanvas, it achieves a 9.22-point average improvement over the SOTA SAeUron, reduces hyperparameter search costs by 96.67%, and is more robust to adversarial attacks.

## Background & Motivation

**Background**: The safe deployment of text-to-image diffusion models (SD series) requires "concept unlearning"—selectively erasing unwanted concepts like nudity, copyrighted characters, or specific objects while maintaining other generative capabilities. Current approaches fall into two categories: (i) fine-tuning model weights (ESD, UCE, SalUn, etc.); (ii) frozen weights with mechanistic intervention on cross-attention activations using sparse autoencoders (SAE) (Concept Steerers, SAeUron). The latter is **reversible** (removing the SAE restores the model fully) and interpretable.

**Limitations of Prior Work**: SAeUron, the strongest representative of the SAE route, trains SAEs in an **unsupervised** manner, leading to *feature splitting*—one concept (e.g., "Bears") is scattered across multiple latents. Erasing it requires: (1) searching thousands of latents for the combination representing "Bears" (evaluating 30 latent subsets × 7 strengths = 210 trials in experimental reports); (2) managing overlap between latents of related concepts, which risk damaging nearby concepts like "Cats."

**Key Challenge**: Monosemanticity (one neuron sensitive to one concept) and one-to-one mapping (one concept corresponding to one neuron) in unsupervised SAEs are not naturally guaranteed; the latter has been consistently missing. Without one-to-one mapping, mechanistic intervention requires combinatorial searching, and interpretability remains post-hoc.

**Goal**: Bind each target concept to a unique latent **during training** while maintaining SAE reconstruction quality, simplifying inference-time erasure to a single scalar negation.

**Key Insight**: Diffusion training already provide supervision signals—anchor prompts (e.g., `An image of Bears`) naturally carry concept labels. Reintroducing these signals into SAE training is more direct than post-hoc alignment using score functions.

**Core Idea**: Add two supervised losses to the standard TopK SAE loss—a Concept Assignment loss to push each concept's activation to a designated latent, and a Decorrelation loss to suppress activation correlation between distinct macro-classes (objects vs. styles).

## Method

### Overall Architecture

The SAEmnesia pipeline is attached after the frozen Stable Diffusion v1.5 cross-attention block `up.1.1`:

1. **Activation Collection**: For each target concept $c$, 80 anchor prompts (e.g., `An image of Bears`) are used to generate 50-step denoising trajectories. Cross-attention feature maps $\mathbf{F}_t \in \mathbb{R}^{h\times w\times d}$ are extracted from `up.1.1`. The $d$-dimensional vector at each spatial position serves as a training sample, labeled with the corresponding object/style.
2. **Two-stage Training**: (i) Standard unsupervised TopK SAE training (reconstruction loss + auxiliary loss for dead latents); (ii) Finetuning with the composite SAEmnesia loss to reinforce "concept-latent" binding.
3. **Concept-Latent Assignment $\Phi$**: Before training, a score function is used to assign the latent index $i_c = \Phi(c)$ with the highest score for each concept $c$.
4. **Inference-time Erasure**: To delete concept $c$, the activation of latent $i_c$ is multiplied by a negative scalar $\gamma_c < 0$, then decoded back to the activation space and fed back to the diffusion backbone.

The process leaves the diffusion backbone unchanged. The SAE is hot-swappable, allowing switching between "forget" and "recover" with one line of code.

### Key Designs

1. **Supervised Concept-Latent Assignment and Concept Assignment Loss**:
    - **Function**: Transitions the mapping of "which latent encodes which concept" from post-hoc attribution into a hard training constraint.
    - **Mechanism**: Uses the score function $\text{score}(i,t,c,D) = \frac{\mu(i,t,D_c)}{\sum_j \mu(j,t,D_c)+\delta} - \frac{\mu(i,t,D_{\neg c})}{\sum_j \mu(j,t,D_{\neg c})+\delta}$ to measure the exclusivity of latent $i$ for concept $c$. The latent with the maximum score is chosen as the designated slot $i_c$. During training, $\mathcal{L}_{\text{CA}} = -\frac{1}{B}\sum_b \frac{1}{|\mathcal{T}^{(b)}|}\sum_{c \in \mathcal{T}^{(b)}} \log \sigma(v^{(b)}_{i_c})$ is applied to push the pre-activation $v_{i_c}$ to activate strongly when the concept is present.
    - **Design Motivation**: CA loss only acts on specific concept-latent pairs, providing local sparse supervision that doesn't disrupt unsupervised feature learning.

2. **Cross-Macro-Group Decorrelation Constraint**:
    - **Function**: Prevents designated latents from different macro-classes (e.g., objects vs. styles) from co-activating, which would cause interference during intervention.
    - **Mechanism**: Groups concepts into disjoint macro-classes $\mathcal{C} = \bigcup_m \mathcal{C}_m$. Penalizes Pearson correlation $\rho$ between activation vectors $\mathbf{a}_c = [v_{i_c}^{(1)}, \dots, v_{i_c}^{(B)}]^\top$ of latents from **different** groups using $\mathcal{L}_{\text{DC}} = \frac{\sum_{m<m'} \sum_{i\in\mathcal{I}_m, j\in\mathcal{I}_{m'}} \rho(\mathbf{a}_i, \mathbf{a}_j)}{\sum_{m<m'}|\mathcal{I}_m||\mathcal{I}_{m'}|}$.
    - **Design Motivation**: Full decorrelation might destroy natural semantic relationships (e.g., "Cats" vs "Dogs"), so it is applied only at the macro-group level to ensure erasing an object does not affect a style.

3. **Single Latent Thresholding (Inference-time steering)**:
    - **Function**: Condenses erasure to a scalar multiplication, active only when the latent is truly triggered.
    - **Mechanism**: $z_{i_c}$ is modified as $z_{i_c} = \gamma_c \mu(i_c, t, D_c) z_{i_c}$ if and only if $z_{i_c} > \mu(i_c, t, D)$ (exceeding the mean activation). $\gamma_c < 0$ is the only tunable hyperparameter.
    - **Design Motivation**: Thresholding acts as a guardrail, ensuring intervention only occurs when the model is actually generating the target concept, minimizing collateral damage.

### Loss & Training

The complete objective is $\mathcal{L}_{\text{SAEmnesia}} = \mathcal{L}_{\text{unsupSAE}} + \beta(\mathcal{L}_{\text{CA}} + \eta \mathcal{L}_{\text{DC}}) + \lambda \mathcal{L}_{L_1}$. Training proceeds in two stages: pure unsupervised pre-training, followed by supervised finetuning. Activations are sourced from SD v1.5 `up.1.1` across all timesteps.

## Key Experimental Results

### Main Results

Object erasure on UnlearnCanvas (percentage; UA = Unlearning Accuracy, IRA/CRA = In-/Cross-domain Retain Accuracy):

| Method Group | Method | UA ↑ | IRA ↑ | CRA ↑ | Avg ↑ |
|--------------|--------|------|-------|-------|-------|
| Fine-tune | ESD | 92.15 | 55.78 | 44.23 | 64.05 |
| Fine-tune | SalUn | 86.91 | 96.35 | 99.59 | 94.28 |
| Adapter | SPM | 71.25 | 90.79 | 81.65 | 81.23 |
| Unsupervised SAE | SAeUron | 87.16 | 85.57 | 74.14 | 82.29 |
| **Supervised SAE** | **Ours** | **94.65** | 91.39 | 88.48 | **91.51** |

Compared to SAeUron: UA +7.49, IRA +5.82, CRA +14.34, Avg **+9.22**.

### Ablation Study

| Configuration / Scenario | Key Metric | Description |
|--------------------------|------------|-------------|
| Hyperparam Search Cost | 7 vs. 210 evals | SAEmnesia only tunes $m$; cost reduced by 96.67% |
| Sequential Erasure (9 objs) | UA 92.4% vs. 64.0% | +28.4 points over baseline; RA 60.9% vs. 48.4% |
| White-box Attack (UnlearnDiffAtk)| UA 57.50% vs. 34.20% | Drop of 40.1 pts vs. SAeUron drop of 49.5 pts |
| Black-box Attack (Ring-A-Bell) | UA 97.0% vs. 79.5% | Robustness holds across threat models |
| NSFW Mitigation (I2P) | 9 detected vs. 18 | Using only 2 latents ("naked man"/"naked woman") |
| Feature score distribution | 0.0404 vs. 0.0166 | 2.43× increase in score peak after supervised training |

### Key Findings

- **CA loss is the primary contributor**: The 2.43× score peak is the root of downstream gains in sequential erasure, adversarial robustness, and efficiency.
- **Decorrelation works at the macro-group level**: The authors state that interference within the same macro-group (e.g., Dogs vs. Cats) remains an open issue.
- **Emergent adversarial robustness**: One-to-one mapping narrows the attack surface; adversarial prompts must hit a specific latent precisely.
- **Uniform multiplier stability**: SAEmnesia with a global $\gamma_c$ outperforms SAeUron even with optimal per-concept search.

## Highlights & Insights

- **Post-hoc attribution to training-time constraint**: Monosemanticity is treated as a training objective rather than just an evaluation metric, transforming mechanistic interpretability from an observation tool to a control tool.
- **Search space reduction**: Shifting from $m \times l$ to $m$ search avoids combinatorial explosion, which is vital for sequential concept erasure.
- **Activation thresholding guardrail**: Decoupling "high static score" from "active in current forward pass" is the hidden key to precision.
- **Macro-group decorrelation**: Penalizing correlations based on user-centric semantic groups rather than raw data distributions is a transferable strategy for multi-task SAEs.

## Limitations & Future Work

- **Structural Limitations**: Only validated on U-Net; transformer-based models like FLUX require adaptation.
- **Closed-vocabulary**: New concepts require recalculating scores (post-hoc binding is used if not retraining).
- **Within-group interference**: Interference between similar concepts (e.g., Dogs vs. Cats) is not fully resolved.
- **Storage Cost**: Maintaining multiple SAE checkpoints (unsupervised vs. supervised) involves higher engineering overhead.

## Related Work & Insights

- **vs. SAeUron**: Both use SAEs on cross-attention activations; the difference is SAEmnesia introduces supervised assignment to solve feature splitting.
- **vs. Concept Steerers**: Concept Steerers intervene on text embeddings; SAEmnesia intervenes on the visual path, making it more robust to attacks bypassing the text encoder (e.g., Ring-A-Bell).
- **vs. Fine-tuning (SalUn/ESD)**: Fine-tuning is irreversible and risks damaging downstream capabilities; SAEs are modular and auditable.
- **vs. ScaPre**: ScaPre uses spectral trace regularization for sequential unlearning; SAEmnesia provides an alternative by ensuring representation additivity through one-to-one mapping.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Introducing supervision to SAEs for feature centralization is a clear leap, though the innovation is primarily in the loss design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers UnlearnCanvas, I2P, adversarial attacks, and sequential erasure, but lacks validation on latest-generation architectures.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation-method-evidence chain with honest discussion of limitations.
- **Value**: ⭐⭐⭐⭐⭐ Advances mechanistic erasure toward engineering viability (single latent, scalar multiplication, modularity).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Interpretable and Steerable Concept Bottleneck Sparse Autoencoders](../../CVPR2026/image_generation/interpretable_and_steerable_concept_bottleneck_sparse_autoencoders.md)
- [\[ICML 2026\] Skipping the Zeros in Diffusion Models for Sparse Data Generation](skipping_the_zeros_in_diffusion_models_for_sparse_data_generation.md)
- [\[NeurIPS 2025\] Learning Interpretable Features in Audio Latent Spaces via Sparse Autoencoders](../../NeurIPS2025/image_generation/learning_interpretable_features_in_audio_latent_spaces_via_sparse_autoencoders.md)
- [\[NeurIPS 2025\] When Are Concepts Erased From Diffusion Models?](../../NeurIPS2025/image_generation/when_are_concepts_erased_from_diffusion_models.md)
- [\[ICML 2026\] RAIGen: Rare Attribute Identification in Text-to-Image Generative Models](raigen_rare_attribute_identification_in_text-to-image_generative_models.md)

</div>

<!-- RELATED:END -->
