---
title: >-
  [Paper Note] Consistent Text-to-Image Generation via Scene De-Contextualization
description: >-
  [ICLR 2026][Image Generation][consistent T2I] This paper identifies the root cause of identity (ID) shift in T2I models as *scene contextualization* — the injection of contextual information from scene tokens into ID tokens — and proposes a training-free method, Scene De-Contextualization (SDeC), that leverages SVD singular value directional stability analysis to identify and suppress latent scene–ID associations in prompt embeddings, enabling per-scene identity-consistent generation.
tags:
  - ICLR 2026
  - Image Generation
  - consistent T2I
  - identity preservation
  - scene contextualization
  - SVD
  - training-free
  - prompt embedding
date: 2026-05-08
content_hash: c66c2ff061a8683f
---

# Consistent Text-to-Image Generation via Scene De-Contextualization

**Conference**: ICLR 2026
**arXiv**: [2510.14553](https://arxiv.org/abs/2510.14553)
**Code**: [https://github.com/tntek/SDeC](https://github.com/tntek/SDeC)
**Area**: Diffusion Models / Consistent Generation
**Keywords**: consistent T2I, identity preservation, scene contextualization, SVD, training-free, prompt embedding

## TL;DR
This paper identifies the root cause of identity (ID) shift in T2I models as *scene contextualization* — the injection of contextual information from scene tokens into ID tokens — and proposes a training-free method, Scene De-Contextualization (SDeC), that leverages SVD singular value directional stability analysis to identify and suppress latent scene–ID associations in prompt embeddings, enabling per-scene identity-consistent generation.

## Background & Motivation

**Background**: Consistent T2I generation requires that a given subject maintains a consistent identity across different scenes. Existing methods (e.g., ConsiStory, 1P1S) typically require all target scenes to be known in advance, or require model training/fine-tuning.

**Limitations of Prior Work**: (a) The assumption that all target scenes are available upfront is unrealistic in practice (e.g., scenes in film/game production are determined iteratively); (b) training-based methods require model retraining, leading to low efficiency; (c) the root cause of ID shift has not been systematically investigated.

**Key Challenge**: T2I models trained on large-scale natural images naturally learn co-occurrence priors between subjects and scenes (e.g., cows are typically found in grasslands rather than oceans), causing the model to alter subject appearance under different scene prompts. The attention mechanism allows information from scene tokens to inevitably leak into ID tokens.

**Goal**: (a) Provide a theoretical explanation for the source of ID shift; (b) propose a training-free, per-scene solution that does not require knowledge of all scenes in advance.

**Key Insight**: Starting from the attention mechanism, the paper proves that scene contextualization (scene-to-ID information leakage) is nearly unavoidable (it can only be prevented if $W_V$ is exactly block-diagonal — a measure-zero event), and then identifies and suppresses this association in the prompt embedding space via SVD analysis.

**Core Idea**: Scene contextualization is the root cause of ID shift and is nearly unavoidable; however, it can be inversely decoupled at the prompt embedding level through SVD directional stability analysis.

## Method

### Overall Architecture

SDeC is a training-free prompt embedding editing method:
- **Input**: Text prompt $\mathcal{P}^k = \mathcal{P}_{\text{id}} \oplus \mathcal{P}_{\text{sc}}^k$ (ID description + scene description)
- **Encoding**: Prompt embedding $[\mathcal{Z}_{\text{id}}^o; \mathcal{Z}_{\text{sc}}^k]$ obtained via the text encoder
- **SDeC Processing**: Identifies and suppresses scene-associated components in $\mathcal{Z}_{\text{id}}^o$
- **Output**: Corrected embeddings fed into the T2I model for image generation

Crucially, only a single scene prompt is processed at a time; no prior knowledge of other scenes is required.

### Key Designs

1. **Scene Contextualization Theory (Theorem 1 + Corollary 1)**:

    - **Function**: Proves that scene-to-ID information injection via the attention mechanism is nearly unavoidable.
    - **Mechanism**: The attention output is decomposed into an ID term $T_{\text{id}}$ and a scene term $T_{\text{sc}}$. For $T_{\text{sc}} \neq 0$, two conditions must hold simultaneously: (A) $\alpha_{\text{sc}} \neq 0$ (non-zero scene attention weights) and (B) $\Pi_{\text{id}} \circ W_V|_{\mathcal{H}_{\text{sc}}} \neq 0$ ($W_V$ is not block-diagonal with respect to the ID/scene subspaces). Both conditions are nearly always satisfied in practice.
    - **Design Motivation**: Provides the theoretical foundation for SDeC — since scene contextualization is unavoidable, post-hoc processing is needed to remove it.

2. **Quantification of Directional Variability (QDV)**:

    - **Function**: Quantifies the degree to which each SVD direction is affected by scene information via a forward–backward singular value optimization procedure.
    - **Mechanism**: SVD is applied to the original ID embedding $\mathcal{Z}_{\text{id}}^o$ to obtain singular values $\sigma_j$. The stability of each singular direction is then analyzed by measuring the change in singular values when scene information is added or removed — directions exhibiting large absolute excursions are considered "contaminated" by scene information.
    - **Design Motivation**: Directly constructing the shared subspace projection matrix $P_\cap$ between ID and scene is numerically unstable in high dimensions; a learned soft estimation is more robust.

3. **Adaptive Singular Value Re-weighting**:

    - **Function**: Based on QDV results, reduces the weight of scene-affected directions and reinforces stable directions.
    - **Mechanism**: The absolute excursion of singular values is used as a re-weighting coefficient; the re-weighted singular values are then used to reconstruct the ID embedding.
    - **Design Motivation**: Rather than hard removal of certain directions, this soft adaptive re-weighting preserves directions that carry important ID information even if they are mildly scene-correlated.

### Loss & Training

- **Training-Free**: SDeC operates entirely on prompt embeddings at inference time without modifying model parameters.
- Compatible with multiple T2I backbones: SDXL, SD3, Flux, PlayGround-v2.5, etc.
- Complementary to attention-adapter-based methods such as ConsiStory.

## Key Experimental Results

### Main Results (SDXL-based)

| Method | DreamSim-F ↓ | CLIP-I ↑ | DreamSim-B ↑ | CLIP-T ↑ | Type |
|--------|-------------|---------|-------------|---------|------|
| SDXL Baseline | 0.2778 | 0.8558 | 0.3861 | 0.8865 | — |
| ConsiStory | 0.2729 | 0.8604 | **0.4207** | **0.8942** | Training-free |
| 1P1S | **0.2238** | **0.8798** | 0.2955 | 0.8883 | Training-free |
| **SDeC** | 0.2589 | 0.8655 | 0.3675 | 0.8946 | Training-free |
| **SDeC+ConsiStory** | 0.2542 | 0.8744 | 0.4155 | 0.8967 | Training-free |

User study win rate: SDeC **42.67%** vs. 1P1S 15% vs. ConsiStory 20.83%.

### Ablation Study

| Variant | DreamSim-F ↓ | CLIP-I ↑ | CLIP-T ↑ |
|---------|-------------|---------|---------|
| **SDeC (full)** | **0.2589** | **0.8655** | **0.8946** |
| w/o soft-estimation | 0.2646 | 0.8603 | 0.8912 |
| w/o abs-excursion | 0.2631 | 0.8627 | 0.8893 |

### Key Findings
- 1P1S achieves the best ID metrics but exhibits the worst scene diversity (DreamSim-B of only 0.2955), indicating severe inter-scene interference. SDeC achieves the best balance between ID consistency and scene diversity.
- SDeC and ConsiStory are highly complementary — the former operates on prompt embeddings while the latter operates on attention, and their combination yields significant improvements.
- Training-based methods (BLIP-Diffusion, PhotoMaker) underperform training-free methods on ID consistency.
- SDeC incurs negligible computational overhead (0.61s for QDV), with minimal additional inference time or memory cost.
- Both the soft estimation of $P_\cap$ and the absolute excursion design contribute positively.

## Highlights & Insights
- **Theoretical rigor**: Beyond qualitatively explaining the cause of ID shift, the paper derives the "near-inevitability" of scene contextualization from the attention mechanism (via a measure-zero argument) and provides an upper bound on its magnitude. The logic of "first proving the problem is unavoidable, then proposing a solution" is highly compelling.
- **Training-free + per-scene**: No training is required and no prior knowledge of all scenes is needed — satisfying both constraints simultaneously makes the method highly practical for real-world engineering. The approach is transferable to any setting where conditioning signals need to be decoupled in conditional generation.
- **SVD directional stability analysis** is a novel contribution — identifying "contaminated" directions by observing singular value changes before and after applying perturbations is a general technique transferable to other signal decoupling domains.

## Limitations & Future Work
- 1P1S still achieves higher ID purity (CLIP-I 0.8798 vs. 0.8655), suggesting that SDeC's de-contextualization is not yet complete.
- Validation is limited to text-only prompt settings; image-conditioned scenarios (e.g., IP-Adapter) are not evaluated.
- Theoretical analysis focuses on the first attention layer; the cumulative effect across multiple layers is not quantified.
- The forward–backward optimization in QDV introduces an additional 0.61s latency.
- The method relies on SVD decomposition, which may become less efficient for long prompts with a large number of tokens.

## Related Work & Insights
- **vs. 1P1S**: 1P1S requires all scenes for prompt restructuring and an IPCA adapter. SDeC removes these dependencies while achieving superior performance on composite metrics. The two methods operate at different levels: 1P1S restructures the prompt, while SDeC edits the prompt embedding.
- **vs. ConsiStory**: ConsiStory enforces self-attention consistency at the attention layer, which is complementary to SDeC's embedding-level operation.
- **vs. DreamBooth/PhotoMaker**: Training-based methods learn identity from reference images, whereas SDeC operates from text prompts alone without requiring reference images.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to theorize scene contextualization and prove its near-inevitability; SVD stability analysis is a novel and principled approach.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-backbone evaluation (SDXL/SD3/Flux), user study, and ablations are comprehensive; image-conditioned experiments are absent.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain from theory to insight to method to experiments is clear and coherent.
- Value: ⭐⭐⭐⭐ The training-free per-scene solution has strong practical value; the theoretical contributions are also highly inspiring.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Infinite-Story: A Training-Free Consistent Text-to-Image Generation](../../AAAI2026/image_generation/infinite-story_a_training-free_consistent_text-to-image_gene.md)
- [\[ICLR 2026\] Generate Any Scene: Scene Graph Driven Data Synthesis for Visual Generation Training](generate_any_scene_scene_graph_driven_data_synthesis_for_visual_generation_train.md)
- [\[NeurIPS 2025\] SceneDecorator: Towards Scene-Oriented Story Generation with Scene Planning and Scene Consistency](../../NeurIPS2025/image_generation/scenedecorator_towards_scene-oriented_story_generation_with_scene_planning_and_s.md)
- [\[CVPR 2026\] Match-and-Fuse: Consistent Generation from Unstructured Image Sets](../../CVPR2026/image_generation/match-and-fuse_consistent_generation_from_unstructured_image_sets.md)
- [\[ICLR 2026\] Directional Textual Inversion for Personalized Text-to-Image Generation](directional_textual_inversion_for_personalized_text-to-image_generation.md)

<!-- RELATED:END -->
