---
title: >-
  [Paper Note] InterEdit: Navigating Text-Guided Multi-Human 3D Motion Editing
description: >-
  [CVPR 2026][Image Generation][multi-human motion editing] This paper proposes InterEdit, the first text-guided multi-human 3D motion editing framework. Through two alignment mechanisms—Semantic-Aware Plan Token Alignment and Interaction-Aware Frequency Token Alignment—InterEdit achieves precise editing of two-person interactive motions within a conditional diffusion model, while preserving source motion consistency and interaction coherence.
tags:
  - CVPR 2026
  - Image Generation
  - multi-human motion editing
  - text-guided
  - interaction-aware
  - frequency-domain alignment
  - conditional diffusion models
date: 2026-05-08
content_hash: de04b601594f3584
---

# InterEdit: Navigating Text-Guided Multi-Human 3D Motion Editing

**Conference**: CVPR 2026
**arXiv**: [2603.13082](https://arxiv.org/abs/2603.13082)
**Code**: [GitHub](https://github.com/YNG916/InterEdit)
**Area**: 3D Motion Generation / Diffusion Models
**Keywords**: multi-human motion editing, text-guided, interaction-aware, frequency-domain alignment, conditional diffusion models

## TL;DR

This paper proposes InterEdit, the first text-guided multi-human 3D motion editing framework. Through two alignment mechanisms—Semantic-Aware Plan Token Alignment and Interaction-Aware Frequency Token Alignment—InterEdit achieves precise editing of two-person interactive motions within a conditional diffusion model, while preserving source motion consistency and interaction coherence.

## Background & Motivation

Text-guided single-person motion editing has achieved notable progress, yet extending it to multi-human scenarios introduces unique challenges:

**Scarce paired data**: No multi-human motion editing datasets exist in the form of (source motion, target motion, editing instruction) triplets.

**Complex interaction semantics**: Motion meaning arises not only from individual actions but also from spatiotemporal coupling—synchronization, phase alignment, role switching, and contact timing.

**Stricter editing constraints**: Edits must "change the requested part and preserve the rest," yet in interactive settings even minor temporal offsets can alter semantics.

**Core gap**: Existing single-person editing methods (MotionFix, MotionLab) ignore interaction coupling, and naively concatenating dual-person features disrupts coordination; multi-human generation methods (InterGen, TIMotion) lack mechanisms for "what to change vs. what to preserve." No dedicated multi-human motion editing benchmark currently exists.

## Method

### Overall Architecture

InterEdit is a conditional diffusion framework parameterized with Start_X, directly predicting clean motion $\hat{\mathbf{x}}_0 = \mathcal{D}_\theta(\mathbf{x}_t, t; \mathbf{c}_{\text{text}}, \mathbf{c}_{\text{src}})$. The backbone is a Transformer-based denoiser with conditions injected via AdaLN:

$$\mathbf{e}_t = \mathrm{EmbedTime}(t) + W_{\text{text}}\mathbf{c}_{\text{text}} + W_{\text{src}}\mathbf{c}_{\text{src}}$$

The core contributions lie in two auxiliary alignment mechanisms: Semantic-Aware Plan Token Alignment and Interaction-Aware Frequency Token Alignment.

### Key Designs

1. **Symmetric Interleaved Token Aggregation (base architecture)**

   A causal interleaved sequence is constructed to model bidirectional temporal influence and role switching between persons A and B. Given motion tokens $\mathbf{x}_c^A, \mathbf{x}_c^B \in \mathbb{R}^{L \times C}$, an interleaved sequence $\mathbf{x}_{\mathrm{cii}}$ and a role-swapped symmetric sequence $\mathbf{x}_{\mathrm{sym}}$ are built:

   $\mathbf{x}_{\mathrm{cii}}(2\ell-1) = \mathbf{x}_c^A(\ell), \quad \mathbf{x}_{\mathrm{cii}}(2\ell) = \mathbf{x}_c^B(\ell)$

   After concatenation, a Transformer processes the sequence; de-interleaving and role-perspective fusion yield global features, complemented by an LPA (Localized Pattern Amplification) branch for short-range temporal patterns.

2. **Semantic-Aware Plan Token Alignment (semantic guidance)**

   $N_M=16$ learnable Plan Tokens $\mathbf{P} \in \mathbb{R}^{N_M \times 2C}$ are appended to the denoiser sequence. At Transformer block $L_p$, they are projected into semantic space and aligned with target motion embeddings $\mathbf{z}_{\text{tgt}} = f_T(\mathbf{x}_0)$ extracted by a frozen motion teacher encoder:

   $\mathcal{L}_{\text{plan}} = \frac{1}{N_M}\sum_{k=1}^{N_M}\left[-\log\frac{\exp((\tilde{\mathbf{z}}^{(k)})^\top \tilde{\mathbf{z}}_{\text{tgt}} / \tau)}{\sum_n \exp((\tilde{\mathbf{z}}^{(k)})^\top \tilde{\mathbf{z}}_{\text{tgt}}^{(n)} / \tau)}\right]$

   Aligned via InfoNCE loss, the Plan Tokens provide high-level editing semantic guidance to motion tokens through self-attention.

3. **Interaction-Aware Frequency Token Alignment (interaction dynamics)**

   Two interaction signals are constructed: mean $\mathbf{z}_S = (\mathbf{x}^A + \mathbf{x}^B)/2$ (synchronization component) and difference $\mathbf{z}_D = \mathbf{x}^A - \mathbf{x}^B$ (opposition component). DCT is applied to each, and frequency-band energy descriptors are obtained by pooling over low/mid/high bands, yielding six band-energy descriptors:

   $\mathbf{E}(\mathbf{C};b) = \sqrt{\frac{1}{|b|}\sum_{k \in b} \mathbf{C}[k]^2 + \epsilon}$

   The band energies are projected into six Frequency Tokens and injected into the sequence; at block $L_f$ they are decoded and aligned to the target motion's band energies via a weighted regression loss: $\mathcal{L}_{\text{freq}} = \frac{1}{N_f}\sum_i w_i \|\hat{\mathbf{g}}_i - \mathbf{g}_i(\mathbf{x}_0)\|_2^2$. High-frequency terms are down-weighted by 0.25 during training, and frequency tokens are randomly dropped with probability $p_f=0.04$ to prevent overfitting.

### Loss & Training

Total objective: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{motion}} + \lambda_p \mathcal{L}_{\text{plan}} + \lambda_f \mathcal{L}_{\text{freq}}$

The motion loss comprises: diffusion reconstruction $\mathcal{L}_{\text{diff}}$, velocity $\mathcal{L}_{\text{vel}}$, foot contact $\mathcal{L}_{\text{foot}}$, bone length $\mathcal{L}_{\text{BL}}$, plus interaction losses (distance map $\mathcal{L}_{\text{DM}}$, relative orientation $\mathcal{L}_{\text{RO}}$). $\lambda_p = 0.03$, $\lambda_f = 0.01$. Inference uses Synchronized CFG ($\gamma=3.5$), jointly dropping text and source motion conditions. Sampling is performed with DDIM in 50 steps. The model has 358.8M parameters (85.0M trainable), trained for 1500 epochs on 8× RTX Pro 6000 GPUs.

## Key Experimental Results

### Main Results

Evaluation on the InterEdit3D test set (5,161 triplets, 80/10/10 split):

| Method | FID↓ | g2s R@1↑ | g2s R@3↑ | g2t R@1↑ | g2t R@3↑ |
|--------|------|---------|---------|---------|---------|
| MotionFix | 2.547 | 2.51 | 6.76 | 3.86 | 7.73 |
| MotionLab | 0.550 | 7.90 | 16.43 | 13.26 | 20.69 |
| InterGen | 0.624 | 9.52 | 18.91 | 18.93 | 31.64 |
| TIMotion | 0.445 | 12.54 | 22.33 | 24.97 | 40.68 |
| **InterEdit** | **0.371** | **17.08** | **29.32** | **30.82** | **47.65** |

Compared to the strongest baseline TIMotion: g2t R@1/2/3 improve by +5.85/+7.07/+6.97 respectively, and FID decreases by 16.7%.

### Ablation Study

| Configuration | FID↓ | g2t R@1↑ | g2t R@3↑ |
|--------------|------|---------|---------|
| w/o plan + freq | 0.445 | 24.97 | 40.68 |
| only plan token | 0.367 | 28.72 | 43.50 |
| only freq token | 0.380 | 28.75 | 44.05 |
| **plan + freq (full)** | **0.371** | **30.82** | **47.65** |

Ablation over frequency token dropout rate shows $p_f=0.04$ to be optimal, balancing overfitting and signal strength.

### Key Findings

- Multi-human generation baselines (InterGen/TIMotion) substantially outperform single-person editing baselines, confirming the necessity of interaction modeling.
- Plan Tokens and Frequency Tokens serve complementary roles: the former guides "what to change," the latter stabilizes "how to change it."
- Combining both yields larger gains than either alone (g2t R@3: 40.68→43.50/44.05→47.65).

## Highlights & Insights

- **First multi-human motion editing task and benchmark**: fills a gap in the field; InterEdit3D contains 5,161 high-quality triplets.
- **Frequency-domain interaction modeling**: DCT decomposition with band-energy descriptors elegantly captures the rhythmic and synchronization properties of interactions.
- **InfoNCE alignment for Plan Tokens**: editing intent is acquired automatically via contrastive learning without explicit annotation of which joints to modify.

## Limitations & Future Work

- Only dual-person interactions are supported; extending to three or more persons requires redesigning the interleaving strategy.
- The model depends on the action-type coverage of the InterHuman dataset (daily activities, martial arts, dance); broader scenarios require additional data.
- The motion representation is based on joint coordinates, lacking appearance and body-shape information.

## Related Work & Insights

- **MotionFix**: pioneered single-person motion editing; this work extends it to multi-human settings.
- **TIMotion**: the strongest multi-human generation baseline; InterEdit reuses its symmetric interleaved token design and LPA module.
- **TMR**: a contrastively trained motion encoder serving as the frozen teacher for Plan Token alignment.

## Rating

- **Novelty**: ★★★★☆ — Task definition and frequency-domain interaction alignment are novel contributions.
- **Technical Depth**: ★★★★☆ — The dual-axis Plan/Frequency Token design is complete, with a rich set of loss terms.
- **Experimental Thoroughness**: ★★★★☆ — Comprehensive quantitative, qualitative, and ablation results, though all baselines are adapted rather than native methods.
- **Practicality**: ★★★☆☆ — Research-driven work; dataset and code are forthcoming.

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Vinedresser3D: Agentic Text-guided 3D Editing](vinedresser3d_agentic_text-guided_3d_editing.md)
- [\[CVPR 2026\] BiMotion: B-spline Motion for Text-guided Dynamic 3D Character Generation](bimotion_b-spline_motion_for_text-guided_dynamic_3d_character_generation.md)
- [\[CVPR 2026\] Ar2Can: An Architect and an Artist Leveraging a Canvas for Multi-Human Generation](ar2can_an_architect_and_an_artist_leveraging_a_canvas_for_multi-human_generation.md)
- [\[CVPR 2026\] FG-Portrait: 3D Flow Guided Editable Portrait Animation](fg-portrait_3d_flow_guided_editable_portrait_animation.md)
- [\[CVPR 2026\] EgoFlow: Gradient-Guided Flow Matching for Egocentric 6DoF Object Motion Generation](egoflow_gradient-guided_flow_matching_for_egocentric_6dof_object_motion_generati.md)

<!-- RELATED:END -->
