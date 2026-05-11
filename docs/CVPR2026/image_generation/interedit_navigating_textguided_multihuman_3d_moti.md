---
title: >-
  [Paper Note] InterEdit: Navigating Text-Guided Multi-Human 3D Motion Editing
description: >-
  [CVPR 2026][Image Generation][Multi-human motion editing] This paper is the first to formally define the Text-guided Multi-human Motion Editing (TMME) task. It constructs the InterEdit3D dataset containing 5…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Multi-human motion editing"
  - "text-guided diffusion model"
  - "interaction-aware frequency-domain alignment"
  - "semantic planning token"
  - "TMME"
date: 2026-05-08
content_hash: 33e7a275f3dcf1e1
---

# InterEdit: Navigating Text-Guided Multi-Human 3D Motion Editing

**Conference**: CVPR 2026
**arXiv**: [2603.13082](https://arxiv.org/abs/2603.13082)
**Code**: [github.com/YNG916/InterEdit](https://github.com/YNG916/InterEdit)
**Area**: 3D Human Motion Editing
**Keywords**: Multi-human motion editing, text-guided diffusion model, interaction-aware frequency-domain alignment, semantic planning token, TMME

## TL;DR
This paper is the first to formally define the Text-guided Multi-human Motion Editing (TMME) task. It constructs the InterEdit3D dataset containing 5,161 source–target–instruction triplets and proposes the InterEdit conditional diffusion model. The model captures high-level editing intent via semantic-aware planning token alignment and models periodic interaction dynamics via interaction-aware frequency-domain token alignment, achieving state-of-the-art performance on instruction following (g2t R@1 30.82%) and source preservation (g2s R@1 17.08%), outperforming all four baselines across the board.

## Background & Motivation
**Background**: Text-guided 3D motion editing has achieved notable progress in single-person scenarios (MotionFix, MotionLab), but multi-human interaction motion editing remains almost entirely unexplored. Many real-world activities inherently involve multi-person interaction—collaboration, competition, physical contact—requiring the coordinated participation of multiple agents.

**Limitations of Prior Work**: (1) Paired data for multi-human motion editing (source motion–target motion–editing instruction triplets) are lacking; (2) naively concatenating dual-person features in single-human editing methods disrupts interaction consistency (MotionFix g2t R@1 only 3.86%); (3) multi-human generation methods lack an explicit mechanism to disentangle "what to change" from "what to preserve," leading to global drift.

**Key Challenge**: Multi-human motion editing must simultaneously satisfy "precisely executing editing instructions" and "preserving unedited parts along with spatiotemporal coupling consistency"—even a minor modification to one person can break synchronization, spatial consistency, or contact timing.

**Goal**: Given a two-person source motion and a text editing instruction, generate a target multi-human motion that modifies only the relevant parts according to the instruction while preserving unedited content and interpersonal interaction consistency.

**Key Insight**: The editing process is constrained from two complementary dimensions—semantic (planning tokens + motion teacher contrastive learning) and frequency (DCT band energy descriptors).

**Core Idea**: Learnable semantic planning tokens guide "what to change," while DCT frequency-domain tokens constrain "how interaction rhythm is maintained." The two mechanisms work synergistically to ensure editing precision and interaction consistency.

## Method

### Overall Architecture
A conditional diffusion model parameterized with Start_X (directly predicting clean motion rather than noise). Inputs consist of two-person source motions (non-canonicalized representation; each person has $d_m$ dimensions including global joint positions, velocities, 6D rotations, and foot-ground contact) and CLIP-encoded text editing instructions. Source motions are encoded by a Transformer encoder to obtain source embeddings, which are injected into the denoiser together with text embeddings via AdaLN. The denoiser adopts symmetric interleaved token aggregation—arranging dual-person motions in interleaved order as $(x^A_1, x^B_1, x^A_2, x^B_2, \ldots)$ along with their role-swapped counterparts, merging them after the Transformer to obtain global features, and then refining short-range temporal patterns through an LPA branch. An additional 16 planning tokens and 6 frequency control tokens participate in self-attention. Inference uses DDIM with 50 steps and SCFG ($\gamma=3.5$).

### Key Designs
1. **Semantic-Aware Plan Token Alignment**:
    - **Function**: Provides high-level semantic editing guidance to the denoiser, indicating "what the output should look like."
    - **Mechanism**: $N_M=16$ learnable planning tokens are appended to the denoiser sequence and interact with motion tokens via self-attention. After extraction at the 3rd Transformer layer, they are projected into a semantic space. A frozen TMR motion teacher encoder extracts the target motion semantic embedding as the positive sample, and an InfoNCE contrastive loss performs alignment: $\mathcal{L}_{plan} = -\frac{1}{N_M}\sum_k \log \frac{\exp(\tilde{z}^{(k)\top}\tilde{z}_{tgt}/\tau)}{\sum_n \exp(\tilde{z}^{(k)\top}\tilde{z}_{tgt}^{(n)}/\tau)}$
    - **Design Motivation**: InfoNCE preserves the discriminative structure of the latent space better than MSE or cosine similarity. Planning tokens guide motion tokens indirectly rather than constraining them directly, affording the model greater flexibility.

2. **Interaction-Aware Frequency Token Alignment**:
    - **Function**: Captures and preserves the rhythm, synchronization, and periodic dynamics of two-person interactions.
    - **Mechanism**: The dual-person motion is decomposed into a mean signal $z_S=(x^A+x^B)/2$ (synchronization component) and a difference signal $z_D=x^A-x^B$ (opposition component). DCT is applied along the time axis, and band energy descriptors are computed over low/mid/high frequency bands (cutoffs $r_l$=0.08, $r_m$=0.25, $r_h$=0.35): $E(C;b) = \sqrt{\frac{1}{|b|}\sum_{k \in b}C[k]^2 + \epsilon}$. The resulting 6 band energies are projected into 6 frequency control tokens that participate in self-attention. The target motion's band energies are regressed at the 5th layer.
    - **Design Motivation**: The frequency domain naturally captures the periodic characteristics of interactions (beat, synchrony vs. alternation, phase alignment). High-frequency weights are reduced to 0.25 to mitigate noise sensitivity. Frequency tokens are randomly dropped with 4% probability during training as regularization to prevent over-reliance.

3. **Synchronized Classifier-Free Guidance (SCFG)**:
    - **Function**: Balances generation quality and diversity under conditioning.
    - **Mechanism**: During training, both text and source conditions are dropped simultaneously with 10% probability (synchronized dropping prevents one-sided leakage); at inference, conditional and unconditional predictions are combined with $\gamma=3.5$.
    - **Design Motivation**: The two-branch approach achieves performance comparable to three-branch CFG at lower inference cost.

### Loss & Training
Total loss: $\mathcal{L}_{total} = \mathcal{L}_{motion} + 0.03 \cdot \mathcal{L}_{plan} + 0.01 \cdot \mathcal{L}_{freq}$. The motion loss comprises MSE reconstruction + 30× velocity + 30× foot-ground contact + 10× bone length + 3× masked distance map + 0.01× relative orientation. A 1000-step diffusion cosine schedule with DDIM 50-step sampling is used. Optimizer: AdamW (lr=1e-4 with cosine decay, 10-epoch warmup). Architecture: 5-layer Transformer (16 heads, dim=512). Model size: 358.8M parameters (85.0M trainable). Training: 1500 epochs on 8× RTX Pro 6000 Blackwell GPUs.

## Key Experimental Results

### Main Results

| Method | FID↓ | g2s R@1↑ | g2s R@3↑ | g2t R@1↑ | g2t R@3↑ |
|--------|------|----------|----------|----------|----------|
| MotionFix (single-human editing) | 2.547 | 2.51 | 6.76 | 3.86 | 7.73 |
| MotionLab (single-human editing) | 0.550 | 7.90 | 16.43 | 13.26 | 20.69 |
| InterGen (multi-human generation) | 0.624 | 9.52 | 18.91 | 18.93 | 31.64 |
| TIMotion (multi-human generation) | 0.445 | 12.54 | 22.33 | 24.97 | 40.68 |
| **InterEdit** | **0.371** | **17.08** | **29.32** | **30.82** | **47.65** |

### Ablation Study

| Configuration | g2t R@1 | FID | Notes |
|---------------|---------|-----|-------|
| w/o plan + freq tokens | 24.97 | 0.445 | Base diffusion model |
| Plan token only | 28.72 | 0.367 | Semantic guidance is effective |
| Freq token only | 28.75 | 0.380 | Frequency constraint is effective |
| **Plan + freq (joint)** | **30.82** | **0.371** | Two modules are complementary; best overall |
| Freq dropout p=0.04 | Best | — | Too low or too high both degrade performance |

### Key Findings
- Multi-human generation baselines (InterGen/TIMotion) substantially outperform single-human editing baselines (MotionFix/MotionLab), confirming that interaction modeling is central to multi-human editing.
- Plan and Freq tokens are individually effective and yield further gains when combined (g2t R@1: 28.7→30.8), demonstrating the complementarity of semantic and frequency-domain signals.
- Human evaluation confirms the advantage: overall win rate 75.5%, interaction realism win rate 81.0%.
- Frequency token dropout at 4% is the optimal balance between regularization and signal preservation.

## Highlights & Insights
- The paper pioneers the definition of the TMME task and constructs the first large-scale multi-human motion editing dataset (5,161 triplets, annotated by 8 annotators with cross-validation), laying a foundation for the field.
- Frequency-domain token alignment elegantly captures interaction dynamics through a mean/difference decomposition → DCT → band energy → learnable token pipeline that gracefully models rhythmic synchronization.
- Planning tokens as learnable semantic control signals participating in self-attention constitute a reusable conditional diffusion design paradigm.
- The dataset construction pipeline is generalizable: motion retrieval → sliding window → TMR encoding → top-2 nearest neighbors → human annotation.

## Limitations & Future Work
- The authors acknowledge gesture ambiguity issues—fine-grained gestures such as self-applause vs. mutual clapping can be confused.
- Spatial drift in long sequences makes it difficult to maintain strict interpersonal spatial relationships over long, complex motions.
- Only dyadic interactions are covered; group motion editing for 3+ persons is not addressed.
- The dataset is constructed via retrieval from InterHuman, limiting motion diversity to that of the source data.
- Only text-based control is supported; spatial constraints such as trajectory sketches or target positions are not incorporated.

## Related Work & Insights
- **vs. MotionFix/MotionLab (single-human editing)**: Treating concatenated dual-person sequences as a single stream lacks interaction modeling, yielding g2t R@1 of only 3.86%/13.26%, far below InterEdit's 30.82%.
- **vs. TIMotion (strongest baseline)**: A multi-human generation model that lacks an explicit "what to change / what to preserve" mechanism. InterEdit surpasses it on all metrics (g2t +5.85, g2s +4.54, FID −16.7%).
- **vs. InterGen**: A joint denoising diffusion model without editing capability. Even after adaptation, it remains inferior to the interaction-aware InterEdit.
- Frequency-domain token regularization is transferable to temporal consistency constraints in video generation/editing or audio-visual synchronization tasks.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — First to define the TMME problem and dataset; frequency-domain token design is novel; overall framework builds on a mature diffusion paradigm.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Four baselines, multi-dimensional ablation, human evaluation, and failure case analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, well-motivated, complete formulations.
- **Value**: ⭐⭐⭐⭐ — Establishes dataset and methodological foundations for the multi-human motion editing field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Vinedresser3D: Agentic Text-guided 3D Editing](vinedresser3d_agentic_text-guided_3d_editing.md)
- [\[CVPR 2026\] BiMotion: B-spline Motion for Text-guided Dynamic 3D Character Generation](bimotion_b-spline_motion_for_text-guided_dynamic_3d_character_generation.md)
- [\[CVPR 2026\] Ar2Can: An Architect and an Artist Leveraging a Canvas for Multi-Human Generation](ar2can_an_architect_and_an_artist_leveraging_a_canvas_for_multi-human_generation.md)
- [\[CVPR 2026\] FG-Portrait: 3D Flow Guided Editable Portrait Animation](fg-portrait_3d_flow_guided_editable_portrait_animation.md)
- [\[CVPR 2026\] EgoFlow: Gradient-Guided Flow Matching for Egocentric 6DoF Object Motion Generation](egoflow_gradient-guided_flow_matching_for_egocentric_6dof_object_motion_generati.md)

</div>

<!-- RELATED:END -->
