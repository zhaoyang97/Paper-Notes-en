---
title: >-
  [Paper Note] Toward Structural Multimodal Representations: Specialization, Selection, and Sparsification via Mixture-of-Experts
description: >-
  [ICML 2026][Multimodal VLM][Multimodal Representation] This paper proposes the S3 framework, which uses MoE to decompose multimodal representations into concept-level experts (Specialization)…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Multimodal Representation"
  - "MoE"
  - "Task Sufficiency"
  - "Information Minimality"
  - "Inference-time Pruning"
date: 2026-05-08
content_hash: a5cdbb6aae8dd622
---

# Toward Structural Multimodal Representations: Specialization, Selection, and Sparsification via Mixture-of-Experts

**Conference**: ICML 2026  
**arXiv**: [2605.03348](https://arxiv.org/abs/2605.03348)  
**Code**: None  
**Area**: Multimodal VLM / MoE / Representation Learning  
**Keywords**: Multimodal Representation, MoE, Task Sufficiency, Information Minimality, Inference-time Pruning

## TL;DR
This paper proposes the S3 framework, which uses MoE to decompose multimodal representations into concept-level experts (Specialization), activates relevant experts per task via routing (Selection), and prunes low-contribution paths at inference based on routing scores (Sparsification). On four MultiBench benchmarks, it reveals an "inverted-U" curve where performance peaks at intermediate sparsity, presenting a third paradigm for multimodal representation beyond contrastive learning/InfoMax.

## Background & Motivation

**Background**: Two mainstream paradigms in multimodal representation learning—contrastive learning (CLIP/AudioCLIP, etc.) maps paired modalities to a shared space to maximize cross-modal mutual information; InfoMax-style methods (FOCAL, DisentangledSSL, JointOpt) aim to retain both shared and modality-unique information. Both target "learning a fixed embedding."

**Limitations of Prior Work**: Contrastive learning has a theoretical ceiling—its optimal solution’s mutual information depends only on the entropy $H(X_S)$ of the shared factor $X_S$ (Proposition 2.3). If a task depends on modality-unique factors $X_U^m$, contrastive representations cannot achieve Bayes optimality. InfoMax can be task-sufficient but also maximizes $I(Z^m;X^m|Y)$, retaining much task-irrelevant information, violating the InfoMin principle and hindering downstream classification.

**Key Challenge**: A single monolithic embedding is tasked with "alignment + preserving differences + adapting to task changes"—three conflicting demands. The combination of sample- and task-relevant factors is variable, but the representation is fixed and cannot be selectively adjusted.

**Goal**: Construct multimodal representations that are both **Task-Sufficient** ($I(Z_Y^{1*},Z_Y^{2*};Y)=I(X^1,X^2;Y)$) and **Information-Minimal** ($I(Z_Y^{1*},Z_Y^{2*};X^1,X^2|Y)=0$), with controllability at the sample/task level.

**Key Insight**: Shift focus from "optimizing the objective function" to "adding structural inductive bias"—explicitly decompose the representation space into a set of concept subspaces $\mathcal{Z}=\bigoplus_{c\in\mathcal{C}}\mathcal{Z}_c$, each realized by a MoE expert. The same latent concept should activate the same expert across modalities (proposing Distributional Semantic Coherence), enabling "concept-level" rather than "instance-level" cross-modal alignment.

**Core Idea**: Reinterpret MoE as a tool for semantic specialization (not just parameter expansion), using a three-stage pipeline—Specialization → Selection → Sparsification—to address "how to construct a semantic expert space," "how to activate relevant experts per task," and "how to prune redundant paths at inference," achieving structurally controllable Task-Sufficient + Information-Minimal multimodal representations.

## Method

### Overall Architecture
For two modalities, use MoE encoders $f^1, f^2$, each MoE layer having $N_{\mathrm{expert}}=\chi\cdot\rho$ experts (granularity $\chi$ + expansion ratio $\rho$). The router $g$ uses top-$k$ softmax to determine which experts each token traverses: $g(\mathbf{x})=\mathrm{TOP}_k(\mathrm{softmax}(\mathbf{W}_g\mathbf{x}))$, outputting $\mathrm{MoE}(\mathbf{x})=\sum_i g(\mathbf{x})_i e_i(\mathbf{x})$. The three stages are: Stage 1 SSL pretrain encoder + router; Stage 2 fine-tune only the router; Stage 3 prune at inference.

### Key Designs

1. **Specialization: Concept-level Expert Space Pretraining**:

    - **Function**: Each expert anchors a semantic concept, ensuring the same concept is activated to aligned subspaces in both modalities (DSC constraint).
    - **Mechanism**: Objective is $\max_{f^1,f^2}[I(Z^1;X^1)+I(Z^2;X^2)]$ s.t. DSC (Proposition 3.4: for all shareable concepts $c$, $p(\pi_c(Z^1)|c\in C^1)=p(\pi_c(Z^2)|c\in C^2)$). Mutual information is lower-bounded by InfoNCE; loss has three parts: intra-modal $\mathcal{L}_{\mathrm{rep}}=\tfrac12(\mathcal{L}_{\mathrm{InfoNCE}}^{[1\to1]}+\mathcal{L}_{\mathrm{InfoNCE}}^{[2\to2]})$ for diversity; cross-modal $\mathcal{L}_{\mathrm{dsc}}=\tfrac12(\mathcal{L}_{\mathrm{InfoNCE}}^{[1\to2]}+\mathcal{L}_{\mathrm{InfoNCE}}^{[2\to1]})$ for implicit alignment of concept activation patterns; auxiliary routing loss $\mathcal{L}_{\mathrm{aux}}$ prevents expert collapse and encourages balanced, confident activation.
    - **Design Motivation**: Pure InfoNCE is instance-level, but its contrastive signal implicitly shapes expert activation distributions, clustering synonymous concepts to the same expert. Explicit concept decomposition + DSC constraint ensures cross-modal alignment occurs at the "expert level" rather than "feature vector level," naturally accommodating modality-unique parts (unique concepts routed to modality-specific experts).

2. **Selection: Task-Adaptive Router Fine-Tuning Only**:

    - **Function**: Freeze all experts and attention, fine-tune only the router $g$ (a tiny fraction of total parameters), activating relevant experts per task and suppressing irrelevant ones.
    - **Mechanism**: Objective $\max_g[I(Z_Y^1,Z_Y^2;Y)-\alpha\cdot I(Z_Y^1,Z_Y^2;X^1,X^2|Y)]$ for Task-Sufficiency and Information-Minimality. The first term is approximated by SupCon loss—pulling together samples with the same label (Proposition E.2 proves it is a valid lower bound for task-conditioned MI): $\mathcal{L}_{\mathrm{SupCon}}^{[m\to\bar m]}=-\mathbb{E}_{i,s\in\mathcal{S}_{y_i}}\log\frac{\exp(\langle z_i^m,z_s^{\bar m}\rangle/\tau)}{\sum_j\exp(\langle z_i^m,z_j^{\bar m}\rangle/\tau)}$. The second term $I(Z;X|Y)=\mathbb{E}_{p(x,y)}[D_{KL}(p(z|x)\|p(z|y))]$ is approximated by vMF distribution (features are on the sphere after InfoNCE), simplified to an inner-product compactness loss $\mathcal{L}_{\mathrm{Comp}}^{[m\to\bar m]}=-\mathbb{E}[\langle\mu_x^m,\hat\mu_y^{\bar m}\rangle]$, pulling samples toward their class mean direction.
    - **Design Motivation**: Conventional fine-tuning also updates the encoder, destroying the semantic expert structure learned in Stage 1. Tuning only the router strictly decouples "what is learned" from "what is used for the task"—the former is a fixed semantic basis, the latter is a task-dependent selector, similar to prompt tuning but with a more structured goal.

3. **Sparsification: Inference-time Routing Score Pruning**:

    - **Function**: Without further training, for each batch, sort top-$k$ routing pairs by score and retain only the top-$p$ proportion, pruning the rest.
    - **Mechanism**: After Stage 2, router scores estimate the "input-expert contribution to the task." Standard MoE uses fixed top-$k$ regardless of utility, activating unnecessary experts. The pruning process is expected to show an inverted-U curve: as $p$ decreases from 1, irrelevant paths are pruned first (performance rises or holds), reaching a sweet spot with minimal sufficient representation (performance peaks), then key paths are pruned (performance drops). Residual connections remain, so pruning a single routing path does not sever information flow.
    - **Design Motivation**: Extends "information minimization" from training to inference, making representation compression an inference-time knob to adjust efficiency-accuracy tradeoff in real time without extra training. Also provides a natural diagnostic for "how many task-relevant routes exist."

### Loss & Training

- Stage 1: $\mathcal{L}_{\mathrm{special}}=\lambda_{\mathrm{rep}}\mathcal{L}_{\mathrm{rep}}+\lambda_{\mathrm{dsc}}\mathcal{L}_{\mathrm{dsc}}+\lambda_{\mathrm{aux}}\mathcal{L}_{\mathrm{aux}}$ (includes expert balancing regularization).
- Stage 2: $\mathcal{L}_{\mathrm{select}}=\lambda_{\mathrm{suff}}\mathcal{L}_{\mathrm{suff}}+\lambda_{\mathrm{min}}\mathcal{L}_{\mathrm{min}}$ (no balancing regularization, as the goal is to activate relevant experts unevenly).
- Fair comparison: set $k=\chi$ to ensure each token activates the same number of expert parameters as a dense FFN (not winning by total parameter count).

## Key Experimental Results

### Main Results
Four MultiBench benchmarks (MOSEI / MOSI / UR-FUNNY / MUStARD), linear probing accuracy. The table below shows S3’s best results on MOSEI at different granularity $\chi$ compared to baselines with equivalent active parameters (data extracted from the original MOSEI detailed table, best $p$):

| Dataset | Method | Best Accuracy (%) | Note |
|---------|--------|------------------|------|
| MOSEI | CLIP (Contrastive) | ~74.5 | shared-only |
| MOSEI | FactorCL / DisentSSL (InfoMax) | 74-76 | retains all info |
| MOSEI | **S3 (χ=8, sweet spot)** | **77.95** | $p\approx 0.3$ |
| MOSI | InfoMax baseline | ~63 | |
| MOSI | **S3 (χ=8)** | **66.13** | $p\approx 0.6$ |
| UR-FUNNY | InfoMax baseline | ~63 | |
| UR-FUNNY | **S3 (χ=4)** | **64.74** | $p\approx 0.4$ |

S3 consistently outperforms contrastive + InfoMax baselines on all four benchmarks, with the peak at intermediate sparsity rather than $p=1$.

### Ablation Study

| Configuration | MOSEI Accuracy (%) | Curve Shape |
|---------------|-------------------|-------------|
| χ=2 (coarse) | 77.25 (peak at $p=0.1$) | Delayed U—routing ambiguity drops then rises |
| χ=4 (medium) | 77.18 (peak at $p=0.1$) | Smooth transition |
| χ=8 (fine) | **77.95** (peak at $p=0.3$) | Classic inverted-U |
| χ=8, p=1.0 (no pruning) | 75.78 | 2 points lower than pruning peak |
| Specialization only (skip Selection) | < 75 | router not task-adapted |

### Key Findings
- **Granularity determines pruning curve shape**: At low granularity ($\chi=2$), each expert holds multiple concepts, routing is ambiguous, and early pruning can hurt—only aggressive pruning to $p=0.1$ helps (delayed U). At high granularity ($\chi=8$), each expert specializes in one concept, routing confidence is high, pruning from $p=0.9$ immediately helps, with the sweet spot at $p\approx 0.3$. This pattern holds across all four benchmarks.
- **Inverted-U = Empirical InfoMin**: Performance peaks at intermediate sparsity, strong evidence that "task-irrelevant info indeed hinders downstream"—a direct experimental validation of the InfoMin principle in multimodal settings.
- **Router is <5% of total parameters** (Appendix H.3), yet enables task adaptation—showing that once a structured latent space is built, "what to use" matters more than "what to learn."
- **Robust across batch sizes**: From 64 to 512, the curve shape is unchanged, indicating pruning behavior is determined by structural properties, not training details.

## Highlights & Insights
- **Paradigm shift from "tuning loss" to "adding structure"**: The authors point out that the failure of contrastive learning and InfoMax is not just about the wrong loss, but the lack of structural inductive bias. This perspective, moving beyond "objective function centralism," is inspiring for all representation learning, and applies beyond SSL (e.g., few-shot, transfer learning).
- **Semantic interpretation of MoE**: Reinterpreting MoE from "parameter expansion tool" to "concept expert" is a theoretically grounded new perspective. With the DSC concept, it provides a new mathematical language for cross-modal alignment ("expert activation distribution alignment" replaces traditional "feature alignment").
- **Inference-time pruning knob**: Information-Minimality becomes an inference-time hyperparameter, with the inverted-U curve directly revealing the sweet spot—practically useful, as the same model can dynamically trade off efficiency/accuracy for downstream needs.
- **Theory + empirical closure**: First, it is proven that contrastive learning is strictly suboptimal when the task depends on unique factors (Proposition 2.5); then, InfoMax’s task-irrelevant information decomposition (Eq. 12) characterizes its limitations; finally, S3’s inverted-U curve in experiments echoes the theoretical prediction—a rare, complete syllogism.

## Limitations & Future Work
- All experiments are on relatively small-scale, bimodal (mostly text-audio/visual feature vector) tasks like MultiBench; stability, training cost, and convergence when scaling to large-scale image-text (COCO/LAION) or tri-modal (e.g., video-audio-text) scenarios remain unknown.
- DSC assumes "shareable concepts activate the same expert across modalities," but in practice, such natural alignment may not exist—e.g., can "timbre" in audio and "texture" in vision really map to the same expert? Lacks interpretability analysis or visualization evidence.
- The vMF approximation in Stage 2 requires features on the unit sphere (as after InfoNCE); if the backbone outputs are not normalized, the KL derivation fails.
- The sweet spot $p$ for Sparsification currently requires validation set search, with no theoretical guidance; there is no method for automatic $p$ selection across tasks/data scales.
- Although fair comparison is made with FFN for active parameters, MoE’s extra total parameters still mean higher storage and loading costs, limiting deployment on edge devices.

## Related Work & Insights
- **vs CLIP/ImageBind (Contrastive Learning)**: CLIP compresses all information into a single embedding, with a theoretical ceiling set by $H(X_S)$; S3 uses expert subspaces to separately carry shared and unique information, theoretically achieving task-sufficiency.
- **vs FOCAL/JointOpt/DisentangledSSL (InfoMax)**: These methods explicitly split $Z_S+Z_U^m$ but still use fixed vectors, unable to select per task; S3 explicitly delegates "selection" to a trainable router, with further inference-time pruning.
- **vs FactorCL (Enhanced Contrastive Learning)**: FactorCL indirectly expands shared factors via augmentation, still limited by the contrastive objective; S3 structurally escapes the contrastive vs InfoMax dichotomy.
- **vs prompt tuning / LoRA**: Both are "lightweight downstream adaptation" approaches, but prompt/LoRA tunes backbone input or low-rank increments, while S3 tunes the router—the router’s expert selection is more like a "controller," requiring no extra parameters beyond the MoE router itself.
- **vs Switch Transformer/MoE**: Traditional MoE focuses on computational scaling and load balancing; S3 instead uses MoE to express "concept decomposition" as a semantic structure, fundamentally different purposes—pointing MoE research toward "semantic specialization."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ "Structural inductive bias + MoE concept experts + inference-time pruning knob" is a highly integrated trio, forming a clear contrast with contrastive/InfoMax paradigms; formalization of DSC is also a new contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four benchmarks + three granularities + three batch sizes + complete pruning curves, with clear structural conclusions; but scenario scale is small, lacking large-scale image-text validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical derivations (Propositions 2.3/2.5, Def 3.1/3.2, Def 3.3/3.4) are clean and self-consistent, each theoretical claim is echoed by experiments, and the overall logic is very well-structured.
- Value: ⭐⭐⭐⭐ Points to a third path for multimodal representation learning; inference-time pruning knob is of high engineering value; but model deployment cost and large-scale scalability need further validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Quant Experts: Token-aware Adaptive Error Reconstruction with Mixture of Experts for Large Vision-Language Models Quantization](../../CVPR2026/multimodal_vlm/quant_experts_token_aware_vlm_quantization.md)
- [\[ICLR 2026\] Capacity-Aware Inference: Mitigating the Straggler Effect in Mixture of Experts](../../ICLR2026/multimodal_vlm/capacity-aware_inference_mitigating_the_straggler_effect_in_mixture_of_experts.md)
- [\[CVPR 2026\] MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping](../../CVPR2026/multimodal_vlm/modes_accelerating_mixture-of-experts_multimodal_large_language_models_via_dynam.md)
- [\[AAAI 2026\] MCMoE: Completing Missing Modalities with Mixture of Experts for Incomplete Multimodal Action Quality Assessment](../../AAAI2026/multimodal_vlm/mcmoe_completing_missing_modalities_with_mixture_of_experts_for_incomplete_multi.md)
- [\[ICML 2025\] Dynamic Mixture of Curriculum LoRA Experts for Continual Multimodal Instruction Tuning](../../ICML2025/multimodal_vlm/dynamic_mixture_of_curriculum_lora_experts_for_continual_multimodal_instruction_.md)

</div>

<!-- RELATED:END -->
