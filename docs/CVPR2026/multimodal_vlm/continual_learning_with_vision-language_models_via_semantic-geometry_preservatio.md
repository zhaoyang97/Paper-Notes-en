---
title: >-
  [Paper Note] Continual Learning with Vision-Language Models via Semantic-Geometry Preservation
description: >-
  [CVPR2026][Multimodal VLM][Continual Learning] This paper proposes SeGP-CL, which constructs adversarial anchors via dual-objective projected gradient descent to probe fragile regions at old–new semantic boundaries. Comb…
tags:
  - "CVPR2026"
  - "Multimodal VLM"
  - "Continual Learning"
  - "Vision-Language Models"
  - "Semantic-Geometry Preservation"
  - "Adversarial Anchors"
  - "Cross-Modal Distillation"
  - "CLIP"
  - "Exemplar-Free Replay"
date: 2026-05-08
content_hash: 6078e416bf7f6e03
---

# Continual Learning with Vision-Language Models via Semantic-Geometry Preservation

**Conference**: CVPR2026
**arXiv**: [2603.12055](https://arxiv.org/abs/2603.12055)  
**Code**: To be confirmed  
**Area**: Multimodal VLM
**Keywords**: Continual Learning, Vision-Language Models, Semantic-Geometry Preservation, Adversarial Anchors, Cross-Modal Distillation, CLIP, Exemplar-Free Replay

## TL;DR

This paper proposes SeGP-CL, which constructs adversarial anchors via dual-objective projected gradient descent to probe fragile regions at old–new semantic boundaries. Combined with Anchor-guided Cross-modal Geometry Distillation (ACGD) and Text Semantic Geometry Regularization (TSGR), SeGP-CL effectively preserves the cross-modal semantic-geometric structure of VLMs under exemplar-free conditions, substantially alleviating catastrophic forgetting.

## Background & Motivation

**Background**: Pre-trained vision-language models (e.g., CLIP) suffer from catastrophic forgetting in continual learning. Existing methods lack explicit mechanisms to preserve cross-modal semantic-geometric structure during adaptation to new tasks, causing geometric distortion induced by new-task supervision signals.

**Limitations of Prior Work**: A key observation of the authors is that harmful representational drift is not uniformly distributed across the embedding space but is concentrated at old–new semantic boundaries. In these regions, new samples share visual patterns with old classes and are prone to being "reinterpreted" by new textual semantics, thereby disrupting established visual-textual alignment.

**Key Challenge**: Conservative strategies that freeze backbones and employ task-specific components (e.g., L2P, DualPrompt, PROOF) over-isolate knowledge and limit forward transfer; parameter-efficient adaptation methods (LoRA/Adapter) lack targeted modeling of cross-modal stability; approaches leveraging textual priors (DesCLIP, CLG-CBM) still do not adequately address cross-modal geometry preservation under exemplar-free conditions. Methods using reference datasets (ZSCL, DualTeacher) introduce non-trivial data overhead and apply insufficiently precise constraints that fail to focus on the boundary regions most susceptible to distortion. Furthermore, the modality gap in VLMs means that textual semantics alone cannot fully represent the visual space, necessitating complementary reasoning with raw visual cues. Finally, VLMs' sensitivity to small perturbations can be exploited constructively—adversarial perturbations can expose and cover the most fragile neighborhoods in the old geometric structure, providing an efficient probing mechanism for geometry preservation without exemplar replay.

## Method

### Overall Architecture (SeGP-CL)

SeGP-CL is a three-phase exemplar-free continual learning framework:

- **Pre-training**: Freeze teacher snapshots $(F^T, G^T)$; construct a set of adversarial anchors $\mathcal{A}_t$ from new-task data via Dual-objective Projected Gradient Descent (DPGD) to probe fragile regions at old–new semantic boundaries.
- **During training**: Optimize the cross-entropy loss on new-task data while performing ACGD distillation on anchors to preserve cross-modal structure, and applying TSGR to stabilize the textual semantic reference frame.
- **Post-training**: Use anchors to estimate drift in the raw visual space, transfer old-class visual prototypes, and fuse cross-modal and visual cues via dual-path inference.

### Key Design 1: Dual-objective Projected Gradient Descent (DPGD) for Adversarial Anchor Construction

**Core Idea**: Seed samples from new-task data with the highest semantic affinity to old classes are selected and pushed toward old-class semantic regions via adversarial perturbation.

1. **Seed selection**: For each old class $c$, rank new-task samples by the teacher model's cross-modal similarity $Q(x, c) = \bar{v}^T(x)^\top u_c^T$ and select the top-$K_{\text{seed}}$ samples as seeds.
2. **Dual-objective optimization**: The textual objective pushes perturbed samples toward old-class text embeddings ($\mathcal{L}_{\text{adv}}$); the visual objective pulls them toward old-class raw visual prototypes ($\mathcal{L}_{\text{v-adv}}$), correcting instabilities caused by the modality gap.
3. **PGD iterations**: Run $K_{\text{adv}}=10$ signed-gradient iterations under $\ell_\infty$ constraints with step size $\gamma = 1.5 \times 10^{-3}$:

$$\delta^{(k+1)} = \Pi_{\|\delta\|_\infty \leq \epsilon}\big(\delta^{(k)} - \gamma \cdot \text{sign}(\nabla_\delta \mathcal{L}'_{\text{adv}})\big)$$

### Key Design 2: Anchor-guided Cross-modal Geometry Distillation (ACGD)

Align the teacher and student old-class probability distributions on adversarial anchors to constrain cross-modal structure at fragile boundary regions:

$$\mathcal{L}_{\text{ACGD}} = \tau_A^2 \cdot \mathbb{E}_{x^{adv} \sim \mathcal{A}_t}\left[\text{KL}(\pi_T^{\tau_A}(\cdot | x^{adv}) \| \pi_S^{\tau_A}(\cdot | x^{adv}))\right]$$

where $\tau_A = 20$ is the distillation temperature and both teacher/student distributions are computed over the old-class set $\mathcal{C}_{<t}$.

### Key Design 3: Text Semantic Geometry Regularization (TSGR)

Cross-task drift in the relative geometric structure among text concepts implicitly re-parameterizes old-class semantics. TSGR constrains the textual neighborhood structure of new classes via $k$-NN subgraph matching:

- A reference subgraph is constructed using the LoRA-reset pre-trained text encoder $G^0$.
- For each new class $c \in \mathcal{C}_t$, its $k=10$ nearest neighbors are identified and teacher–student subgraph neighborhoods are matched.
- Only subgraphs rooted at new classes are constrained, yielding complexity $\mathcal{O}(|\mathcal{C}_t| \cdot k)$, far below global constraints.

### Key Design 4: Anchor-driven Prototype Transfer and Dual-path Inference

- **Prototype transfer**: The raw visual feature displacement $d_t(x^{adv})$ of anchors before and after training is used to estimate a weighted drift direction $\Delta_{t,c}$ for each old class, with magnitude modulated by anchor-to-prototype proximity, enabling old-class prototype transfer.
- **Dual-path inference**: CLIP cross-modal scores and visual prototype scores are fused as $\ell_t(x, c) = s_t^{\text{clip}}(x, c) + \beta \cdot s_t^v(x, c)$, with $\beta=0.5$.

### Loss & Training

$$\mathcal{L}_{\text{CL}}^t = \mathcal{L}_{\text{cls}} + \lambda_{\text{ACGD}} \cdot \mathcal{L}_{\text{ACGD}} + \lambda_{\text{GR}} \cdot \mathcal{L}_{\text{GR}}$$

where $\lambda_{\text{ACGD}}=5$ and $\lambda_{\text{GR}}=1$; only the LoRA up-projection matrix B is updated.

## Key Experimental Results

### Main Results: Comparison with SOTA on Five Benchmarks (CLIP ViT-B/16)

| Method | CIFAR100 Avg/Last | ImageNet-R Avg/Last | ImageNet-Sub Avg/Last | CUB-200 Avg/Last | UCF Avg/Last |
|---|---|---|---|---|---|
| MG-CLIP (ICCV'25) | 87.0/80.6 | 87.6/82.7 | 87.3/78.4 | 80.6/72.0 | – |
| RAPF (ECCV'24) | 86.2/79.0 | 85.6/80.3 | 87.5/80.2 | 82.7/76.2 | 92.5/87.5 |
| ENGINE (ICCV'25) | 82.1/73.1 | 84.4/77.0 | – | 83.9/76.2 | 95.0/90.1 |
| **SeGP-CL (Ours)** | **89.8/84.6** | **88.9/84.8** | **89.9/80.5** | **85.4/80.1** | **95.9/92.8** |

SeGP-CL achieves state-of-the-art results on all five benchmarks. CIFAR100 Last improves by +4.0 over MG-CLIP; CUB-200 Last improves by +3.9 over RAPF.

**Transfer and Forgetting Metrics (CLIP branch only, CIFAR100)**:

| Method | FWT ↑ | BWT ↑ | Forgetting ↓ |
|---|---|---|---|
| MG-CLIP | 70.2 | -3.9 | 4.9 |
| DesCLIP | 68.7 | -2.1 | 6.5 |
| **SeGP-CL** | **72.3** | **-0.43** | **0.9** |

SeGP-CL achieves a Forgetting of only 0.9, far below MG-CLIP's 4.9, with BWT near zero (−0.43), indicating virtually no backward forgetting.

### Ablation Study

| ACGD | TSGR | Prototype Transfer | Visual Branch | CIFAR100 Last | Forgetting ↓ |
|---|---|---|---|---|---|
| ✗ | ✗ | ✗ | ✗ | 77.0 | 10.9 |
| ✓ | ✗ | ✗ | ✗ | 81.7 | 5.8 |
| ✓ | ✓ | ✗ | ✗ | 82.8 | 4.7 |
| ✓ | ✓ | ✓ | ✗ | 83.2 | 4.3 |
| ✓ | ✓ | ✓ | ✓ | 84.6 | 4.5 |

ACGD contributes the most (Last +4.7, Forgetting −5.1); TSGR, prototype transfer, and the visual branch provide incremental improvements.

### Key Findings

- **Adversarial anchors vs. other distillation sources**: Anchor-based distillation (+5.8 Last) substantially outperforms reference data (ZSCL +1.9), synthetic data (GIFT +2.7), and new-task data (−0.5).
- **Cross-scenario generalization**: After training on CIFAR100, SeGP-CL maintains near-zero-shot generalization on Food101/Oxford-Pets/ImageNet-1K, largely attributable to TSGR.
- **Parameter efficiency**: With LoRA rank=32, only 3.44M trainable parameters are used (vs. MoE-Adapter's 13.35M), with an additional overhead of only ~79 ms per iteration.
- **DPGD iterations**: 10 iterations suffice for stable convergence; the textual objective converges more slowly than the visual objective, corroborating the modality gap.

## Highlights & Insights

- **Precise problem identification**: The paper is the first to systematically demonstrate that cross-modal geometric distortion in VLM continual learning concentrates at old–new semantic boundaries, supported by empirical evidence using JSD measurements.
- **Constructive use of adversarial attacks**: The VLM's adversarial vulnerability is cleverly repurposed as a tool for locating fragile regions, enabling boundary neighborhood probing without storing old data.
- **Dual-objective design addresses modality gap**: The visual anchoring term in DPGD compensates for the modality gap, preventing pure textual objectives from producing unstable anchors.
- **Lightweight and efficient**: TSGR constrains only the $k$-NN subgraphs of new classes, keeping parameter overhead small and per-iteration time cost manageable.
- **Unified theory and experiments**: The argument is logically complete, from first-order optimality of adversarial optimization to comprehensive state-of-the-art results across five benchmarks.

## Limitations & Future Work

- Adversarial anchor quality depends on hyperparameters such as the $\ell_\infty$ budget and iteration count, which may require tuning across datasets.
- TSGR only constrains the textual neighborhood subgraph of new classes and cannot detect drift in inter-old-class textual relationships.
- Prototype transfer assumes that feature drift on anchors is a reliable proxy for old-class drift; this assumption may break down when semantic distances between new and old classes are large.
- Evaluation is limited to CLIP ViT-B/16; larger backbones (e.g., ViT-L) and other VLMs (e.g., SigLIP, EVA-CLIP) remain untested.
- The fusion coefficient $\beta$ in dual-path inference is fixed; adaptive fusion strategies are not explored.

## Related Work & Insights

- **VLM continual learning**: Contrasted with MG-CLIP (modality gap preservation), ZSCL/DualTeacher (reference data distillation), and ENGINE/RAPF (task-specific components); SeGP-CL requires no extra data and precisely constrains fragile regions.
- **Cross-modal distillation**: SGCL distills semantic pseudo-label reference distributions on new-task data, but is less precise than adversarial anchors.
- **Synthetic data**: GIFT synthesizes old-class images via Stable Diffusion for distillation, but domain gaps limit effectiveness.
- **Adversarial robustness**: The PGD attack framework is adopted, but the objective shifts from "attacking" to "probing fragile neighborhoods."

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The idea of using adversarial anchors to probe semantic boundaries is highly novel, repurposing an attack mechanism as a defense tool.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive state-of-the-art results across five benchmarks, with detailed comparisons of distillation strategies, ablations, and generalization analyses.
- Writing Quality: ⭐⭐⭐⭐ — Mathematical derivations are rigorous, though the dense notation raises the reading barrier.
- Value: ⭐⭐⭐⭐⭐ — Establishes a new geometry-preservation paradigm for VLM continual learning, achieving substantial gains over prior work under exemplar-free conditions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Enhanced Continual Learning of Vision-Language Models with Model Fusion](../../ICLR2026/multimodal_vlm/enhanced_continual_learning_of_vision-language_models_with_model_fusion.md)
- [\[AAAI 2026\] Branch, or Layer? Zeroth-Order Optimization for Continual Learning of Vision-Language Models](../../AAAI2026/multimodal_vlm/branch_or_layer_zeroth-order_optimization_for_continual_lear.md)
- [\[AAAI 2026\] Harnessing Textual Semantic Priors for Knowledge Transfer and Refinement in CLIP-Driven Continual Learning](../../AAAI2026/multimodal_vlm/harnessing_textual_semantic_priors_for_knowledge_transfer_and_refinement_in_clip.md)
- [\[CVPR 2026\] On Token's Dilemma: Dynamic MoE with Drift-Aware Token Assignment for Continual Learning of Large Vision Language Models](on_tokens_dilemma_dynamic_moe_with_drift-aware_token_assignment_for_continual_le.md)
- [\[ICCV 2025\] Instruction-Grounded Visual Projectors for Continual Learning of Generative Vision-Language Models](../../ICCV2025/multimodal_vlm/instruction-grounded_visual_projectors_for_continual_learning_of_generative_visi.md)

</div>

<!-- RELATED:END -->
