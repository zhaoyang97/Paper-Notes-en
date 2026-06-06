---
title: >-
  [Paper Note] MUSE: Resolving Manifold Misalignment in Visual Tokenization via Topological Orthogonality
description: >-
  [ICML 2026][Interpretability][Unified visual tokenizer] MUSE attributes the "understanding-generation" zero-sum dilemma of unified visual tokenizers to manifold misalignment. It proposes the Gradient Orthogonality Hypoth…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Unified visual tokenizer"
  - "Manifold alignment"
  - "Gradient orthogonality"
  - "Topological alignment"
  - "Multimodal understanding-generation"
date: 2026-05-08
content_hash: d8cdec7b08c7d969
---

# MUSE: Resolving Manifold Misalignment in Visual Tokenization via Topological Orthogonality

**Conference**: ICML 2026  
**arXiv**: [2605.05646](https://arxiv.org/abs/2605.05646)  
**Code**: Yes (GitHub mentioned in paper; refer to the main text for the repository)  
**Area**: Interpretability / Multimodal / Visual Tokenizer  
**Keywords**: Unified visual tokenizer, Manifold alignment, Gradient orthogonality, Topological alignment, Multimodal understanding-generation

## TL;DR
MUSE attributes the "understanding-generation" zero-sum dilemma of unified visual tokenizers to manifold misalignment. It proposes the Gradient Orthogonality Hypothesis—injecting semantics into $W_V$ while structural gradients follow $W_{Q,K}$. This is achieved through the Synergistic Block, DINOv3 topological alignment, and NCE semantic anchoring for complete decoupling. Ultimately, it achieves the coexistence of gFID 3.08 and 85.2% linear probing (surpassing the InternViT-300M teacher's 82.5%), realizing true "mutual reinforcement" rather than a compromise for the first time.

## Background & Motivation

**Background**: As multimodal large models move toward unification, the industry attempts to use a single unified visual tokenizer to serve both understanding (CLIP-style semantic encoding) and generation (VQ-VAE/diffusion latent). UniTok, TokenFlow, UniLIP, and VTP all try to fit both objectives into the same codebook or shared latent.

**Limitations of Prior Work**: Although the architecture is unified, the objectives remain in conflict. Pixel reconstruction favors "expanded" manifolds (preserving high-frequency details), while semantic alignment favors "compressed" manifolds (filtering out irrelevant textures). This leads to "perceptual polarization" in representations: attention is either fragmented (like VQ-VAE) or excessively blurred (like UniLIP), with mid-frequency structural information missing.

**Key Challenge**: The two objectives compete directly within shared parameters (specifically $W_Q, W_K, W_V$ of self-attention). Gradient directions even exhibit negative cosine similarity ($\cos\theta_g \ll 0$, Fig. 2a), resulting in "destructive interference"—where one side pulls while the other pushes. This leads to poor learning for both, a phenomenon the authors call Manifold Misalignment.

**Goal**: (1) Eliminate the zero-sum trade-off between generation and understanding without increasing architectural overhead; (2) Use "structural information" as a bridge to serve both objectives; (3) Empirically verify that the gradient orthogonality hypothesis can break "shared parameters = gradient conflict" into "subspace decomposition = gradient synergy."

**Key Insight**: From a manifold geometry perspective, understanding requires "compressing" the manifold $\mathcal M_S$ (semantic invariance), while generation requires "expanding" the manifold $\mathcal M_T$ (structural equivariance). A Structural State ($S$) is missing as a geometric foundation. In a Transformer block, $W_{Q,K}$ controls routing topology and $W_V$ controls content values, naturally forming two orthogonal subspaces.

**Core Idea**: Route semantic gradients to $W_V$ and structural gradients to $W_{Q,K}$. Use DINOv3 attention distillation to align topology and NCE to anchor content to the vision-language manifold, allowing both objectives to be optimized in physical isolation within the Transformer.

## Method

### Overall Architecture
$f_\theta: \mathcal X \to \mathcal Z$ learns to map images to a latent that possesses both semantic invariance $\mathcal M_S$ and structural equivariance $\mathcal M_T$. MUSE uses a connector composed of 6 Synergistic Blocks, utilizing InternVL3's InternViT as the visual backbone and DC-AE as the pixel decoder. Training consists of three stages: (1) **Topology warmup**: Freeze the encoder and use $\mathcal L_{topo}$ to align the student's attention topology with the DINOv3 teacher; (2) **Semantic injection**: Anchor token values to the vision-language manifold using $\mathcal L_{ITC}$ while maintaining topology; (3) **Synergistic tuning**: Unfreeze the backbone for end-to-end joint training of reconstruction, semantics, and topology, using stop-gradients to isolate the semantic branch from reconstruction gradients.

### Key Designs

1.  **Synergistic Block: Physical Decoupling of $W_V$ and $W_{Q,K}$**:
    - **Function**: Allows structural gradients to update only routing parameters and semantic gradients to update only value parameters, eliminating "parameter sharing → gradient conflict" via architecture.
    - **Mechanism**: For input $H_l\in\mathbb R^{N\times D}$, the **Topology Stream** calculates the adjacency matrix $A = \text{Softmax}(Q_{topo}K_{topo}^T/\sqrt{d_k})$ via $W_Q, W_K$, handling "how to look." The **Semantic Stream** projects $V_{sem}=H_l W_V$ through an independent $W_V$, then aggregates $H_{attn}=A\cdot V_{sem}$ according to $A$, handling "what is seen." Structural loss backpropagates only to $W_{Q,K}$, while semantic loss backpropagates only to $W_V$. Simultaneously, a stop-gradient (the /// mark in Fig. 3 bottom-right) is added to the semantic branch to prevent reconstruction gradients from contaminating topological routing.
    - **Design Motivation**: Violin plots (Fig. 2c-d) show that under natural training, semantic gradients concentrate on $W_V$ and structural gradients on $W_{Q,K}$. Standard optimizers force them together, causing negative cosine conflicts. The Synergistic Block follows this intrinsic functional specialization through physical isolation, with almost no parameter overhead, reducing the gradient cosine from negative to ≈ 0.

2.  **Structural Topology Alignment**:
    - **Function**: Maximize $I(Z;S)$ by distilling object geometry emerging in DINOv3's attention maps to the student's routing.
    - **Mechanism**: DINOv3's attention maps naturally display object-level segmentation structures. MUSE introduces a 4D interpolation function $\Psi(\cdot)$ to align resolutions, then uses KL divergence to align student and teacher attention for each layer and head: $\mathcal L_{topo} = \frac{1}{LH}\sum_l\sum_h D_{KL}(\Psi(A_T^{(l,h)})\,\|\,A_S^{(l,h)})$. This loss is architecturally guaranteed to backpropagate only to $W_{Q,K}$.
    - **Design Motivation**: The authors argue that in the chain decomposition of mutual information $I(Z;X,Y)\approx I(Z;S)+I(Z;Y|S)+I(Z;X|S,Y)$, $S$ is the geometric foundation. Learning "where to look" before "what it is" is more theoretically sound than optimizing all terms simultaneously (curriculum justification). DINOv3 attention maps serve as high-quality, free topological supervision.

3.  **Active Semantic Anchoring**:
    - **Function**: Physically nails token values to the vision-language manifold, preventing reconstruction gradients from "squeezing out" semantics.
    - **Mechanism**: A projector $g_\phi(\cdot)$ maps pooled tokens $\bar z$ to the joint vision-language space using an NCE upper bound $\mathcal L_{anchor} = \mathcal L_{NCE}(g_\phi(\bar z), t) \approx -I_{LB}(Z;Y|S)$, where $t$ is the paired text embedding. This loss only updates $W_V$ and the projector.
    - **Design Motivation**: Previous distillation-based semantic alignment (like UniLIP) is "passive" and easily overwritten by reconstruction gradients. Using NCE as an information-theoretic lower bound combined with stop-gradients isolates the semantic branch, equivalent to adding a Lagrangian constraint on $W_V$ that prevents value parameters from drifting away from $\mathcal M_S$.

### Loss & Training
Three-stage curriculum: Stage 1 (Topology warmup, 50k steps, 224×224, lr 4e-4, frozen backbone) → Stage 2 (Semantic injection, 50k steps, lr 2e-4, with NCE) → Stage 3 (Synergistic tuning, 50k steps, lr 1e-5, with adversarial training). MUSE-1B/3B variants are based on InternVL3-1B + SANA-0.6B and InternVL3-2B + SANA-1.1B respectively. The connector uses 6 Synergistic Blocks with $N=256$ learnable queries. Pre-training uses 36M image-text pairs (27M Qwen2.5-VL-7B recaption + 5M CC12M + 4M JourneyDB).

## Key Experimental Results

### Main Results
Table 1 (ImageNet-1K + ADE-20K, all unified methods retrained on the same BLIP3-o corpus for fairness):

| Method | rFID↓ | gFID↓ | PSNR↑ | Zero-Shot↑ | Linear Probe↑ | mIoU↑ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| InternViT-300M (Teacher, Understanding only) | – | – | – | 77.4 | 82.5 | 40.2 |
| VA-VAE-d32 (Generation only) | 0.52 | 4.56 | 26.2 | – | – | 19.6 |
| TokenFlow | 1.37 | 7.66 | 21.6 | 65.4 | 72.4 | 17.4 |
| UniTok | 0.76 | 6.45 | 24.1 | 68.6 | 74.3 | 19.5 |
| UniLIP | 0.79 | 5.73 | 23.0 | 73.5 | 76.2 | 15.4 |
| VTP-L-d64 | 0.75 | 3.01 | 24.7 | 71.2 | 80.5 | 36.8 |
| **MUSE (Ours)** | 0.62 | 3.08 | 24.9 | **76.1** | **85.2** | **46.5** |

Most critical figures: Linear probing 85.2% > Teacher 82.5%, with gFID comparable to VTP and significantly higher mIoU (46.5 vs 36.8).

### Ablation Study

| Configuration | Key Phenomenon | Explanation |
| :--- | :--- | :--- |
| Full MUSE | Best | Three-stage + Synergistic Block |
| Naive shared $W_{Q,K,V}$ + multi-objective sum | $\cos\theta_g \ll 0$ | Classic destructive interference; gFID/Zero-Shot both drop |
| w/o stop-gradient | Semantic shift | Reconstruction gradients pollute $W_V$, Zero-Shot drops significantly |
| w/o $\mathcal L_{topo}$ | mIoU sharp drop | Attention degrades to fragmentation |
| w/o NCE / Passive distillation | Zero-Shot degradation | Semantics squeezed out by reconstruction gradients |
| Reversed curriculum (Semantics then Topology) | No convergence/degradation | Difficulty maximizing $I(Z;Y\|S)$ without geometric foundation |

### Key Findings
- Gradient cosine increases from negative to ≈ 0 (Fig. 2a-b), and split violins show semantic/structural gradients naturally specialize to different parameters (Fig. 2c-d), empirically supporting the Gradient Orthogonality Hypothesis.
- "Student surpassing teacher" phenomenon: MUSE linear probing 85.2% > InternViT-300M 82.5%. The authors attribute this to structural topology constraints preventing attention degradation (mIoU rising from 15.4-36.8 to 46.5), indirectly strengthening semantic readability.
- Reconstruction and understanding are no longer zero-sum: While maintaining gFID close to the generation specialist (VTP 3.01), understanding metrics (MMVP 74.8) show a significant gain over UniLIP.

## Highlights & Insights
- **Causal attribution of "Manifold Misalignment → Gradient Orthogonality"**: Seamlessly connects visualization (Fig. 2 gradient cosine and violin) → theory (mutual information decomposition) → architecture (Synergistic Block). It serves as a blueprint for transforming "ad-hoc engineering tricks" into "theoretical necessity," applicable to any multi-objective shared parameter scenario.
- **Precise use of stop-gradient in multi-objective learning**: Unlike works that use stop-gradients randomly, this paper explicitly defines which gradient path should be cut and leverages the $W_V$ / $W_{Q,K}$ architectural separation, making sense from both theoretical and engineering perspectives.
- **Structure as a bridge**: Topological information is often overlooked. By using DINOv3 attention distillation as free geometric supervision, it suggests that geometric priors implicit in self-supervised models are undervalued resources in unified systems.

## Limitations & Future Work
- The topological teacher must be a model like DINOv3 / iBOT where "attention has spontaneously gained segmentation capability"; if the teacher's attention is degraded, $\mathcal L_{topo}$ will mislead the student.
- The three-stage curriculum is sensitive to hyperparameters (lr decay, stage duration). While details are provided, reproduction costs are not low.
- Multimodal expansion to video and audio is not yet explored; verification of "mutual reinforcement" over the temporal dimension is required.
- The physical isolation of $W_V$ and $W_{Q,K}$ is a characteristic of vanilla self-attention; applicability to variants with RoPE / grouped-query / shared-projection needs separate evaluation.

## Related Work & Insights
- **vs UniLIP / Tang 2025**: UniLIP uses passive distillation to inject CLIP semantics, which are eroded by reconstruction gradients. MUSE uses stop-gradient + NCE for active anchoring, fundamentally avoiding erosion.
- **vs VTP-L-d64**: VTP uses more aggressive pixel supervision to push gFID to 3.01, but Zero-Shot drops to 71.2. MUSE achieves nearly the same gFID while pulling Zero-Shot to 76.1, effectively breaking the trade-off.
- **vs UniTok / TokenFlow**: Early unified methods relied on codebooks / Q-Formers for coarse-grained alignment, lacking architecture-level gradient routing. MUSE's fine-grained routing within the Transformer is a new paradigm.
- **vs DINOv3 / DINOv2**: This work elevates their attention maps to topological supervision for unified tokenizers, indicating self-supervised attention is a source of free geometric priors.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Gradient orthogonality hypothesis + structural bridge; the first solution in this line with theoretical consistency and empirical support.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers ImageNet/ADE/MMVP/WISE/Editing multi-tasking + strong baseline retraining + gradient visualization, though video/audio are missing.
- Writing Quality: ⭐⭐⭐⭐⭐ Figures 1-3 explain motivation, validation, and method with extreme clarity; theoretical decomposition corresponds perfectly with architecture.
- Value: ⭐⭐⭐⭐⭐ Provides a viable "mutual reinforcement" path for unified multimodal systems, with direct guidance for future UMM designs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Learning Coherent Representations: A Topological Approach to Interpretability](learning_coherent_representations_a_topological_approach_to_interpretability.md)
- [\[ICML 2026\] BLOCK-EM: Preventing Emergent Misalignment via Latent Blocking](block-em_preventing_emergent_misalignment_via_latent_blocking.md)
- [\[ICML 2026\] Manifold-Aligned Guided Integrated Gradients for Reliable Feature Attribution](manifold-aligned_guided_integrated_gradients_for_reliable_feature_attribution.md)
- [\[ICML 2026\] Finding the Correct Visual Evidence Without Forgetting: Mitigating Hallucination in LVLMs via Inter-Layer Visual Attention Discrepancy](finding_the_correct_visual_evidence_without_forgetting_mitigating_hallucination_.md)
- [\[ICLR 2026\] When Thinking Backfires: Mechanistic Insights Into Reasoning-Induced Misalignment](../../ICLR2026/interpretability/when_thinking_backfires_mechanistic_insights_into_reasoning-induced_misalignment.md)

</div>

<!-- RELATED:END -->
