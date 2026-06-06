---
title: >-
  [Paper Note] DIVA: Harnessing the Representation Divergence in Unified Multimodal Models for Mutual Reinforcement
description: >-
  [ICML 2026][Multimodal VLM][Unified Multimodal Models] DIVA observes that Unified Multimodal Models (UMMs) spontaneously decouple "understanding" and "generation" information flows in middle layers. Consequently…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Unified Multimodal Models"
  - "Representation Divergence"
  - "Mutual Information"
  - "Shared/Unique Decomposition"
  - "Post-training"
date: 2026-05-08
content_hash: 82050dba6d03ca6a
---

# DIVA: Harnessing the Representation Divergence in Unified Multimodal Models for Mutual Reinforcement

**Conference**: ICML 2026  
**arXiv**: [2605.25328](https://arxiv.org/abs/2605.25328)  
**Code**: https://github.com/Jayyy-H/DIVA  
**Area**: Multimodal VLM / Unified Multimodal Models / Representation Learning  
**Keywords**: Unified Multimodal Models, Representation Divergence, Mutual Information, Shared/Unique Decomposition, Post-training

## TL;DR
DIVA observes that Unified Multimodal Models (UMMs) spontaneously decouple "understanding" and "generation" information flows in middle layers. Consequently, it explicitly factorizes representations into shared and unique components, utilizing contrastive/CLUB mutual information constraints to achieve "shared alignment + unique decoupling." This approach simultaneously improves understanding by +7.82% and generation by +8.46% on Show-o/Liquid/Nexus-Gen without architectural modifications.

## Background & Motivation

**Background**: Unified Multimodal Models (UMM, e.g., Janus-Pro, Show-o, Liquid, Nexus-Gen) employ a single transformer to handle both "image understanding" and "image generation" tasks. While conceptually efficient, existing reports acknowledge that the two objectives often conflict. To mitigate this, mainstream solutions often "separate" components—distinct visual encoders (e.g., BLIP3-o), separate transformer backbones (Bagel using MoT), or hybrid AR + Diffusion architectures.

**Limitations of Prior Work**: All "separation" strategies betray the core premise of UMMs—that a shared backbone should enable beneficial transfer between understanding and generation. Once encoders or backbones are decoupled, the models become serial components, closing the channel for mutual assistance. Conversely, maintaining a shared backbone creates tension between the inductive biases of the two tasks: understanding requires semantic invariance, low-frequency focus, and detail discarding; whereas generation requires high fidelity, high-frequency focus, and detail preservation.

**Key Challenge**: The inductive biases of the two objectives are mathematically non-equivalent. Forcing them into the same parameter set results in "compromised representations" that are neither semantically abstract nor pixel-precise. Through triple analysis (gradient/geometry/spectral), the authors discover an overlooked fact: UMM middle layers **spontaneously** push the two information flows into different subspaces (gradient conflicts peak at shallow and deep layers but are weakest in middle layers; effective rank surges in middle layers; understanding flows exhibit low-frequency bias while generation flows preserve high frequencies), while deep layers realign due to shared physical anchors.

**Goal**: Transform this spontaneous "middle-layer divergence, deep-layer convergence" phenomenon from an unconscious byproduct into an explicitly controllable factorized structure, thereby converting conflicts into mutual gains.

**Key Insight**: Since middle layers naturally diverge and deep layers share semantic anchors, it is beneficial to explicitly define two components—a shared component for cross-task transfer and a unique component for task-specific biases—and use mutual information tools to align the shared and decouple the unique.

**Core Idea**: Factorize visual representations into "shared + unique" parts, maximizing the mutual information lower bound of shared components and minimizing the upper bound (CLUB) of unique components to turn conflicts into controllable bidirectional gains.

## Method

### Overall Architecture
DIVA is a post-training framework that does not alter the backbone architecture. The process involves: (1) Constructing two information flows for the same image-text pair $(I, T)$—the understanding flow uses a caption instruction, and the generation flow uses random masking + text conditioning for inpainting; (2) Extracting image-token hidden states from middle layers $\mathcal{I}_{mid}=\{l \mid l_{start}\leq l \leq l_{end}\}$, passing them through a shared encoder $E_{sh}^i$ and a unique encoder $E_{uni}^i$ to obtain shared/unique components; (3) Two-stage post-training: Stage 1 freezes the backbone to train encoders (cross-task conditioning + logit injection), and Stage 2 unfreezes the backbone for refinement using asymmetric contrastive + CLUB losses.

### Key Designs

1. **Dual-flow Construction and Middle Layer Positioning**:
    - **Function**: Generate two information flows with explicit inductive biases from the same image-text anchor to serve as inputs for factorization.
    - **Mechanism**: The understanding flow takes the original image + a template prompt $t_{prompt}$ (e.g., "Please describe this image in detail"), driven by a captioning loss $\mathcal{L}_{\text{Und}} = \mathcal{L}(f_\theta(\text{concat}(t_{question}, h_v)), t_{answer})$. The generation flow takes a corrupted image $I_{mask} = I \odot (1-M)$ with mask ratio $r \in [0.2, 0.6]$ plus the original text $T$ as condition, driven by an inpainting loss $\mathcal{L}_{\text{Gen}}$. The middle layer range is positioned based on observed effective-rank peaks, set to layers 8–18 in the paper. For each layer $\ell$, $h_i^{(\ell)} = \text{Pool}(H_i^{img,\ell}) \in \mathbb{R}^d$ is pooled.
    - **Design Motivation**: Both flows must share a physical anchor (the same $(I,T)$ pair), otherwise shared factors are meaningless. Simultaneously, inductive biases must be significantly different—captioning elicits a low-frequency semantic flow, while inpainting elicits a high-frequency detail flow, allowing for a decomposable "shared vs. unique" structure.

2. **Shared/Unique Factorization and Gated Encoders**:
    - **Function**: Explicitly decompose middle-layer representations into shared components $z_{sh}^{\ell,i}$ and unique components $z_{uni}^{\ell,i}$, corresponding to the mutual information decomposition $I(X_1, X_2; Y) = \Pi_{sh} + \Pi_{uni}^i + \Pi_{uni}^j + \epsilon_{noise}$.
    - **Mechanism**: For each flow $i \in \{U, G\}$, two 3-layer Gated MLPs are introduced: $z_{sh}^{\ell,i} = g_{sh}^{(i)}(\ell) \odot \phi_{sh}(h_i^{(\ell)})$ and $z_{uni}^{\ell,i} = g_{uni}^{(i)}(\ell) \odot \phi_{uni}(h_i^{(\ell)})$, where $g_{(\cdot)}^{(i)}(\ell) = \sigma(W_{(\cdot)}^i h_i^{(\ell)})$ is an element-wise soft gate. During Stage 1, the backbone is frozen, and factorization outputs are injected as logit biases: $\tilde{s}_U = s_U + A_U z_{sh}^{\ell,G} + B_U z_{uni}^{\ell,U}$ and $\tilde{s}_G = s_G + A_G z_{sh}^{\ell,U} + B_G z_{uni}^{\ell,G}$ (notably using cross-task injection—the understanding flow uses the shared component from the generation flow). Encoders are trained with native task losses. An additional orthogonality constraint $\mathcal{L}_\perp = \sum_i \|(\mathbf{z}_{sh}^i)^\top \mathbf{z}_{uni}^i\|_F^2$ prevents the unique encoder from encoding shared semantics.
    - **Design Motivation**: Cross-task conditioning forces the shared encoder to learn components that are "useful for the other flow," otherwise logit injection wouldn't reduce loss. Orthogonality prevents the two encoders from collapsing into the same subspace. This setup stabilizes the decomposition in Stage 1, preventing catastrophic drift when the backbone is unfrozen in Stage 2.

3. **Mutual Information-driven Asymmetric Alignment (Stage 2)**:
    - **Function**: After unfreezing the backbone, aligns shared components in the representation space and decouples unique components to transform conflict into synergy.
    - **Mechanism**: Shared components are maximized via InfoNCE lower bound $I_{sha}(X_i^s; X_j^s) = \mathbb{E}_{x_i, x_j^+}[\log \frac{\exp f(x_i, x_j^+)}{\sum_k \exp f(x_i, x_j^-)}]$. Unique components are minimized via the CLUB upper bound $I_{uni}(X_i^u; X_j^u)$ to prevent the shared space from leaking task-specific information or the unique space from redundantly encoding shared semantics. To prevent loss dominance due to scale differences (understanding vs. generation), asymmetric alignment with stop-gradients is used: $\mathcal{L}_{U \to G} = -\log \frac{\exp(\text{sim}(z_{sh}^U, \text{sg}[z_{sh}^G])/\tau)}{\sum_j \exp(\text{sim}(z_{sh}^U, \text{sg}[z_{sh}^{G,j}])/\tau)}$ and symmetrically for $\mathcal{L}_{G \to U}$. Final total loss: $\mathcal{L}_{total} = \mathcal{L}_{U \to G} + \mathcal{L}_{G \to U} + \mathcal{L}_{uni} + \mathcal{L}_{Und} + \mathcal{L}_{Gen}$.
    - **Design Motivation**: InfoNCE + CLUB is a standard combination for mutual information bounds; the former encourages proximity for shared elements, while the latter enforces distance for unique ones. Stop-gradients stabilize optimization under heterogeneous task scales, ensuring a large loss on one side does not distort the representation of the other.

### Loss & Training
Two-stage post-training: Stage 1 uses native task losses + $\mathcal{L}_\perp$ to train encoders (backbone frozen, shared-only warmup followed by unique residuals). Stage 2 unfreezes the backbone and optimizes via five combined terms in $\mathcal{L}_{total}$. Training data consists of 200K image-text pairs (60K each from CapsFusion-120M and Infinity-MM, 70K each from JourneyDB and MidjourneyV6, with captions refined by Qwen2.5-VL-32B). Backbones include Show-o (1.5B), Nexus-Gen (7B), and Liquid (7B).

## Key Experimental Results

### Main Results
Post-training evaluation on three UMMs across 8 benchmarks:

| Backbone | Setting | MMMU | POPE | MMVet | GenEval | DPG-Bench | WISE |
|------|------|------|------|-------|---------|-----------|------|
| Show-o (1.5B) | Base | 26.3 | 73.1 | 32.5 | 0.57 | 69.81 | 0.29 |
| Show-o (1.5B) | +DIVA | **32.4 (+6.1)** | **79.1 (+6.0)** | **33.8** | **0.64 (+0.07)** | **76.03 (+6.22)** | **0.34** |
| Nexus-Gen (7B) | Base | 43.5 | 83.6 | 45.2 | 0.77 | 81.30 | 0.39 |
| Nexus-Gen (7B) | +DIVA | **49.4 (+5.9)** | **87.4 (+3.8)** | **46.6** | **0.83 (+0.06)** | **87.87 (+6.57)** | **0.45** |
| Liquid (7B) | Base | 30.2 | 77.4 | 36.9 | 0.70 | 80.63 | 0.41 |
| Liquid (7B) | +DIVA | **34.0 (+3.8)** | **84.5 (+7.1)** | **37.8** | **0.81 (+0.11)** | **83.47 (+2.84)** | **0.44** |

Understanding improved by an average of +7.82%, and generation by +8.46%. All three backbones showed **unidirectional gains**, providing credible evidence for "mutual synergy."

### Ablation Study
| Configuration | MMMU | POPE | GenEval | DPG-Bench |
|------|------|------|---------|-----------|
| Base | 26.3 | 73.1 | 0.69 | 69.81 |
| Base + Standard SFT | 26.8 | 74.5 | 0.67 | 70.75 |
| Base + DIVA | **32.4** | **79.1** | **0.75** | **76.03** |
| DIVA w/o $I_{uni}$ (No CLUB) | 28.3 | 75.8 | 0.70 | 71.58 |
| DIVA w/o sg[·] | 31.7 | 78.2 | 0.73 | 74.92 |
| Mid-Layer (9–17) | 31.5 | 78.4 | 0.72 | 73.36 |
| Mid-Layer (8–18) | **32.4** | **79.1** | **0.75** | **76.03** |
| Linear+LN encoder | 29.4 | 75.9 | 0.71 | 72.37 |

### Key Findings
- Simple SFT on the same 200K data yields minimal gains or even degradation (GenEval 0.69 → 0.67), proving DIVA's success is due to structured factorization rather than data volume.
- Removing the CLUB term ($I_{uni}$) results in the largest drop (MMMU -4.1), indicating that "strict decoupling of unique components" is more critical than "aligning shared components"—the former prevents interference while the latter accelerates transfer.
- Performance is sensitive but not fragile to the middle layer range: 8–18 is optimal, 7–19 is nearly identical, and 9–17 drops by 1 point, confirming that the effective-rank peak accurately identifies a stable working zone.
- Gated MLP encoders outperform Linear+LN by 3 points (MMMU 32.4 vs 29.4), highlighting the importance of soft gates for hierarchical feature filtering.

## Highlights & Insights
- Reinterpreting "UMM mutual interference" as "spontaneous decoupling in middle layers that isn't being explicitly utilized" is the paper's most significant insight. The empirical evidence from inverted-parabolic gradient conflicts and middle-layer effective rank peaks is robust.
- The combination of Shared/Unique factorization + InfoNCE/CLUB is a "textbook" application of mutual information decomposition, but applying it to multimodal post-training while using asymmetric stop-gradients to handle task scale heterogeneity is an impressive engineering feat.
- Cross-task conditioning (using the shared factor of the opposing flow to bias one's own logit) is the key to Stage 1, as it enforces a "cross-task usability" constraint on the shared factor, preventing it from devolving into a copy of task-specific features.
- The paradigm is extensible to other scenarios where tasks compete for the same backbone, such as video understanding/generation, 3D reconstruction/rendering, or ASR/TTS.

## Limitations & Future Work
- The 200K post-training data size is relatively small for UMM standards; the stability of the shared/unique structure when scaled to 1M+ samples is unverified.
- The middle layer range (8-18) was set manually for an 18-layer backbone; the paper lacks a systematic method for deeper models (e.g., 32-layer Llama-style).
- All experimented backbones are Autoregressive (AR); effectiveness on hybrid AR + Diffusion (BLIP3-o) or MoT (Bagel) architectures remains untested.
- CLUB upper bound estimates can exhibit high variance and instability in high dimensions; although stop-gradients help, a formal convergence analysis is missing.

## Related Work & Insights
- **vs. Bagel / BLIP3-o (Architectural Separation)**: These models use MoT or independent visual encoders to isolate tasks. DIVA proves that "representation decomposition within a shared backbone" is sufficient to match or exceed separate schemes (e.g., Show-o + DIVA 1.5B achieving 80% of Janus-Pro 7B's GenEval score) with superior parameter efficiency.
- **vs. Show-o / Liquid / Nexus-Gen Baselines**: DIVA is a plug-in post-training method that improves all three backbones without changing architecture, suggesting it addresses a **universal** structural issue in UMMs.
- **vs. General MoE / LoRA Branches**: While MoE uses routers to divert flows at the token level, DIVA uses encoders to divert at the representation level. The former modifies architecture and increases inference costs, whereas the latter adds only lightweight MLPs in middle layers, which can potentially be discarded after fine-tuning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The empirical validation of "spontaneous middle-layer decoupling in UMMs" and the prescribed "explicit factorization + alignment/decoupling" is a first in the field.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid horizontal comparisons across three backbones and 8 benchmarks with 5-dimensional ablations; however, scaling laws and hybrid architecture validations are missing.
- Writing Quality: ⭐⭐⭐⭐ Clear logical progression from observation to method; some redundancies in formulas and symbols exist (e.g., undefined dimensions for $W_{sh}^i, W_{uni}^i$).
- Value: ⭐⭐⭐⭐⭐ Provides the first provably effective lightweight solution for "how to post-train UMMs," which is highly valuable for industrial UMM deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Seeing is Understanding: Unlocking Causal Attention into Modality-Mutual Attention for Multimodal LLMs](seeing_is_understanding_unlocking_causal_attention_into_modality-mutual_attentio.md)
- [\[ICML 2026\] Breaking Dual Bottlenecks: Evolving Unified Multimodal Models into Self-Adaptive Interleaved Visual Reasoners](breaking_dual_bottlenecks_evolving_unified_multimodal_models_into_self-adaptive_.md)
- [\[ICML 2026\] Calibrated Multimodal Representation Learning with Missing Modalities](calibrated_multimodal_representation_learning_with_missing_modalities.md)
- [\[NeurIPS 2025\] Unified Reinforcement and Imitation Learning for Vision-Language Models](../../NeurIPS2025/multimodal_vlm/unified_reinforcement_and_imitation_learning_for_vision-language_models.md)
- [\[ICLR 2026\] Modal Aphasia: Can Unified Multimodal Models Describe Images From Memory?](../../ICLR2026/multimodal_vlm/modal_aphasia_can_unified_multimodal_models_describe_images_from_memory.md)

</div>

<!-- RELATED:END -->
