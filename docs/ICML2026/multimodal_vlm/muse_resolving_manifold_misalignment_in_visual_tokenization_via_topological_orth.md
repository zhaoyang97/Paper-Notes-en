---
title: >-
  [Paper Note] MUSE: Resolving Manifold Misalignment in Visual Tokenization via Topological Orthogonality
description: >-
  [ICML 2026][Multimodal VLM][Unified visual tokenizer] MUSE attributes the "understanding-generation" zero-sum dilemma of unified visual tokenizers to manifold misalignment…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Unified visual tokenizer"
  - "manifold alignment"
  - "gradient orthogonality"
  - "topological alignment"
  - "multimodal understanding-generation"
date: 2026-05-08
content_hash: 739b078b3297dc56
---

# MUSE: Resolving Manifold Misalignment in Visual Tokenization via Topological Orthogonality

**Conference**: ICML 2026  
**arXiv**: [2605.05646](https://arxiv.org/abs/2605.05646)  
**Code**: Available (GitHub link noted in the paper, see main text for repository)  
**Area**: Interpretability / Multimodal / Visual Tokenizer  
**Keywords**: Unified visual tokenizer, manifold alignment, gradient orthogonality, topological alignment, multimodal understanding-generation

## TL;DR
MUSE attributes the "understanding-generation" zero-sum dilemma of unified visual tokenizers to manifold misalignment, proposing the gradient orthogonality hypothesis—injecting semantics into $W_V$ while structural gradients flow through $W_{Q,K}$. Through Synergistic Block + DINOv3 topological alignment + NCE semantic anchoring, the two are fully decoupled. As a result, gFID 3.08 and linear probing 85.2% (even surpassing the InternViT-300M teacher at 82.5%) coexist, achieving genuine "mutual reinforcement" rather than trade-off for the first time.

## Background & Motivation

**Background**: As large multimodal models move toward unification, the community seeks a unified visual tokenizer to serve both understanding (CLIP-style semantic encoding) and generation (VQ-VAE/diffusion latent). Approaches like UniTok, TokenFlow, UniLIP, and VTP attempt to fit both objectives into a single codebook or shared latent space.

**Limitations of Prior Work**: Despite architectural unification, the objectives remain in conflict—pixel reconstruction prefers a "spread-out" manifold (preserving high-frequency details), while semantic alignment prefers a "compressed" manifold (filtering out irrelevant textures). This leads to "perceptual polarization" in representations: attention becomes either fragmented (as in VA-VAE) or overly blurred (as in UniLIP), with loss of mid-frequency structural information.

**Key Challenge**: The two objectives directly compete within shared parameters (especially self-attention's $W_Q, W_K, W_V$), with gradient directions often negatively correlated ($\cos\theta_g \ll 0$, see Fig. 2a), resulting in "destructive interference"—one pulls while the other pushes, so neither learns well. The authors term this Manifold Misalignment.

**Goal**: (1) Eliminate the generation-understanding trade-off without increasing architectural overhead; (2) Make "structural information" a bridge serving both objectives; (3) Empirically validate that the gradient orthogonality hypothesis can turn "parameter sharing = gradient conflict" into "subspace separation = gradient synergy".

**Key Insight**: From a manifold geometry perspective, understanding requires a "compressed" manifold $\mathcal M_S$ (semantic invariance), generation requires an "expanded" manifold $\mathcal M_T$ (structural equivariance), and a missing $S$ (Structural State) is needed as the geometric foundation. In Transformer blocks, $W_{Q,K}$ control routing topology, $W_V$ controls content values—naturally forming two orthogonal subspaces.

**Core Idea**: Route semantic gradients to $W_V$ and structural gradients to $W_{Q,K}$; use DINOv3 attention distillation for topological alignment and NCE to anchor content to the vision-language manifold, enabling the two objectives to be physically isolated and optimized within the Transformer.

## Method

### Overall Architecture
$f_\theta: \mathcal X \to \mathcal Z$ learns to map images to latents with both semantic invariance $\mathcal M_S$ and structural equivariance $\mathcal M_T$. MUSE uses six Synergistic Blocks as a connector, with InternVL3's InternViT as the visual backbone and DC-AE as the pixel decoder. Training proceeds in three stages: (1) **Topology warmup**: freeze the encoder, use only $\mathcal L_{topo}$ to align student attention topology with the DINOv3 teacher; (2) **Semantic injection**: while maintaining topology, use $\mathcal L_{ITC}$ to anchor token values to the vision-language manifold; (3) **Synergistic tuning**: unfreeze the backbone for end-to-end joint training of reconstruction, semantics, and topology, using stop-gradient to isolate the semantic branch from reconstruction gradients.

### Key Designs

1. **Synergistic Block: Physical Decoupling of $W_V$ and $W_{Q,K}$**:

    - **Function**: Ensures structural gradients update only routing parameters, and semantic gradients update only value parameters, eliminating "parameter sharing → gradient conflict" at the architectural level.
    - **Mechanism**: For input $H_l\in\mathbb R^{N\times D}$, the **Topology Stream** uses $W_Q, W_K$ to compute the adjacency matrix $A = \text{Softmax}(Q_{topo}K_{topo}^T/\sqrt{d_k})$ ("where to look"); the **Semantic Stream** uses an independent $W_V$ to project $V_{sem}=H_l W_V$, then aggregates via $A$ as $H_{attn}=A\cdot V_{sem}$ ("what is seen"). Structural loss backpropagates only to $W_{Q,K}$, semantic loss only to $W_V$. A stop-gradient is applied to the semantic branch (/// in Fig. 3 lower right), preventing reconstruction gradients from contaminating the topology routing.
    - **Design Motivation**: Violin plots (Fig. 2c-d) empirically show that, under natural training, semantic gradients concentrate on $W_V$ and structural gradients on $W_{Q,K}$; standard optimizers forcibly mix them, causing negative cosine conflicts. The Synergistic Block leverages this intrinsic functional specialization for physical isolation, adding minimal parameter overhead but shifting gradient cosine from negative to ≈ 0.

2. **Structural Topology Alignment**:

    - **Function**: Maximizes $I(Z;S)$ by distilling the object geometry emerging in DINOv3 teacher attention maps into the student routing.
    - **Mechanism**: DINOv3 and similar self-supervised models' attention maps naturally reveal object-level segmentation. MUSE introduces a 4D interpolation function $\Psi(\cdot)$ for resolution alignment, then uses KL divergence to align student and teacher attention for each layer and head: $\mathcal L_{topo} = \frac{1}{LH}\sum_l\sum_h D_{KL}(\Psi(A_T^{(l,h)})\,\|\,A_S^{(l,h)})$. This loss is architecturally guaranteed to backpropagate only to $W_{Q,K}$.
    - **Design Motivation**: The authors argue that in the mutual information chain decomposition $I(Z;X,Y)\approx I(Z;S)+I(Z;Y|S)+I(Z;X|S,Y)$, $S$ is the geometric foundation; learning "where to look" first, then "what it is", is more information-theoretically sound than optimizing all terms simultaneously (curriculum justification). DINOv3's attention map provides free, high-quality topological supervision.

3. **Active Semantic Anchoring**:

    - **Function**: Physically anchors token values to the vision-language manifold, preventing reconstruction gradients from "squeezing out" semantics.
    - **Mechanism**: Introduces a projector $g_\phi(\cdot)$ to map pooled token $\bar z$ into the vision-language joint space, using an NCE upper bound $\mathcal L_{anchor} = \mathcal L_{NCE}(g_\phi(\bar z), t) \approx -I_{LB}(Z;Y|S)$, where $t$ is the paired text embedding. This loss is architecturally guaranteed to update only $W_V$ and the projector.
    - **Design Motivation**: Previous distillation-based semantic alignment (e.g., UniLIP) is "passive distillation" and easily overridden by reconstruction gradients; using NCE as an information-theoretic lower bound plus stop-gradient isolates the semantic branch from reconstruction gradients, equivalent to a Lagrangian constraint on $W_V$, forcing value parameters to remain close to $\mathcal M_S$.

### Loss & Training
Three-stage curriculum: Stage 1 (topology warmup, 50k steps, 224×224, lr 4e-4, frozen backbone) → Stage 2 (semantic injection, 50k steps, lr 2e-4, add NCE) → Stage 3 (synergistic fine-tuning, 50k steps, lr 1e-5, enable adversarial training). MUSE-1B/3B variants are based on InternVL3-1B + SANA-0.6B and InternVL3-2B + SANA-1.6B, respectively. The connector uses six Synergistic Blocks and $N=256$ learnable queries. Pretraining corpus: 36M image-text pairs (27M Qwen2.5-VL-7B recaption + 5M CC12M + 4M JourneyDB).

## Key Experimental Results

### Main Results
Table 1 (ImageNet-1K + ADE-20K; all unified methods retrained on the same BLIP3-o corpus for fairness):

| Method | rFID↓ | gFID↓ | PSNR↑ | Zero-Shot↑ | Linear Probe↑ | mIoU↑ |
|--------|-------|-------|-------|------------|---------------|-------|
| InternViT-300M (teacher, understanding only) | – | – | – | 77.4 | 82.5 | 40.2 |
| VA-VAE-d32 (generation only) | 0.52 | 4.56 | 26.2 | – | – | 19.6 |
| TokenFlow | 1.37 | 7.66 | 21.6 | 65.4 | 72.4 | 17.4 |
| UniTok | 0.76 | 6.45 | 24.1 | 68.6 | 74.3 | 19.5 |
| UniLIP | 0.79 | 5.73 | 23.0 | 73.5 | 76.2 | 15.4 |
| VTP-L-d64 | 0.75 | 3.01 | 24.7 | 71.2 | 80.5 | 36.8 |
| **MUSE (Ours)** | 0.62 | 3.08 | 24.9 | **76.1** | **85.2** | **46.5** |

Key numbers: linear probing 85.2% > teacher 82.5%, with gFID on par with VTP and much higher mIoU (46.5 vs 36.8).

### Ablation Study

| Configuration | Key Phenomenon | Description |
|---------------|---------------|-------------|
| Full MUSE | best | Three-stage + Synergistic Block |
| naive shared $W_{Q,K,V}$ + multi-objective sum | $\cos\theta_g \ll 0$ | Classic destructive interference, both gFID/Zero-Shot drop |
| remove stop-gradient | semantic drift | Reconstruction gradients pollute $W_V$, Zero-Shot drops significantly |
| remove $\mathcal L_{topo}$ | mIoU drops sharply | Attention degrades to fragmentation |
| remove NCE / switch to passive distillation | Zero-Shot degrades | Semantics squeezed out by reconstruction gradients |
| reverse curriculum order (semantic before topology) | no convergence/degradation | Without geometric foundation, $I(Z;Y\|S)$ is hard to maximize |

### Key Findings
- Gradient cosine shifts from negative to ≈ 0 (Fig. 2a-b), and split violin plots show semantic/structural gradients naturally specialize to different parameters (Fig. 2c-d), empirically supporting the Gradient Orthogonality Hypothesis.
- "Student surpasses teacher": MUSE linear probing 85.2% > InternViT-300M 82.5%; authors attribute this to structural topology constraints preventing attention degradation (mIoU rises from 15.4–36.8 to 46.5), indirectly enhancing semantic interpretability.
- Reconstruction and understanding are no longer zero-sum: with gFID close to generation specialists (VTP 3.01), the understanding side (MMVP 74.8) improves significantly over UniLIP.

## Highlights & Insights
- **Causal attribution from "manifold misalignment → gradient orthogonality"**: The work connects visualization (gradient cosines and violin plots in Fig. 2), theory (mutual information chain decomposition), and architecture (Synergistic Block), turning what appears to be an "engineering trick" into a theoretical inevitability—a template for any multi-objective shared-parameter scenario.
- **Precise use of stop-gradient in multi-objective settings**: While many multi-task works use stop-gradient heuristically, this paper clearly specifies which gradient paths should be cut, with architectural $W_V$/$W_{Q,K}$ separation justified both theoretically and practically.
- **Structure as a bridge**: Topological information is often overlooked; here, DINOv3 attention distillation serves as free geometric supervision, suggesting that self-supervised models' implicit geometric priors are underutilized resources in unified systems.

## Limitations & Future Work
- The topology teacher must be a model like DINOv3/iBOT with "attention spontaneously exhibiting segmentation ability"; if the teacher's attention is degraded, $\mathcal L_{topo}$ may mislead.
- The three-stage curriculum is sensitive to hyperparameters (lr decay, stage steps); while the paper provides details, reproduction cost is nontrivial.
- Multimodal extension to video and audio is not addressed; currently only image tokens are validated, and whether "mutual reinforcement" holds in the temporal dimension remains to be seen.
- The physical separation of $W_V$ and $W_{Q,K}$ is a property of vanilla self-attention; applicability to attention variants with RoPE, grouped-query, or shared-projection requires separate evaluation.

## Related Work & Insights
- **vs UniLIP / Tang 2025**: UniLIP uses passive distillation to inject CLIP semantics into the tokenizer, but is continually eroded by reconstruction gradients; MUSE uses stop-gradient + NCE for active anchoring, fundamentally avoiding erosion.
- **vs VTP-L-d64**: VTP uses more aggressive pixel supervision to push gFID to 3.01, but Zero-Shot drops to 71.2; MUSE achieves similar gFID while raising Zero-Shot to 76.1, truly breaking the trade-off.
- **vs UniTok / TokenFlow**: Early unified methods rely on codebook/Q-Former for coarse alignment, lacking architectural-level gradient routing; MUSE's fine-grained routing within the Transformer is a new paradigm.
- **vs DINOv3 / DINOv2**: This work elevates their attention maps to topological supervision for unified tokenizers, highlighting self-supervised attention as a free source of geometric priors.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Gradient orthogonality hypothesis + structural bridge, the first theoretically consistent and empirically supported solution in this line of work.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-task coverage (ImageNet/ADE/MMVP/WISE/Editing) + strong baseline retraining + gradient visualization, but lacks video/audio.
- Writing Quality: ⭐⭐⭐⭐⭐ Figures 1–3 clearly explain motivation, validation, and method, with theory and architecture tightly aligned.
- Value: ⭐⭐⭐⭐⭐ Provides a feasible "mutual reinforcement" path for unified multimodal systems, with direct guidance for future UMM design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] BLOCK-EM: Preventing Emergent Misalignment via Latent Blocking](block-em_preventing_emergent_misalignment_via_latent_blocking.md)
- [\[ICML 2026\] Manifold-Aligned Guided Integrated Gradients for Reliable Feature Attribution](manifold-aligned_guided_integrated_gradients_for_reliable_feature_attribution.md)
- [\[ICLR 2026\] When Thinking Backfires: Mechanistic Insights Into Reasoning-Induced Misalignment](../../ICLR2026/interpretability/when_thinking_backfires_mechanistic_insights_into_reasoning-induced_misalignment.md)
- [\[CVPR 2026\] Draft and Refine with Visual Experts](../../CVPR2026/interpretability/draft_and_refine_with_visual_experts.md)
- [\[CVPR 2026\] Pixel2Phys: Distilling Governing Laws from Visual Dynamics](../../CVPR2026/interpretability/pixel2phys_distilling_governing_laws_from_visual_dynamics.md)

</div>

<!-- RELATED:END -->
