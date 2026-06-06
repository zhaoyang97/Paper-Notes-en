---
title: >-
  [Paper Note] Learning Context-Conditioned Predicate Semantics via Prototype Feedback
description: >-
  [ICML 2026][Optimization][Scene Graph Generation] AlignG transforms the static predicate prototypes of PE-Net into "image-conditioned" dynamic prototypes: first…
tags:
  - "ICML 2026"
  - "Optimization"
  - "Scene Graph Generation"
  - "Relationship Reasoning"
  - "Prototype Learning"
  - "Context-Conditioning"
  - "Predicate Disambiguation"
date: 2026-05-08
content_hash: 15d2d51653f5b99d
---

# Learning Context-Conditioned Predicate Semantics via Prototype Feedback

**Conference**: ICML 2026  
**arXiv**: [2605.29610](https://arxiv.org/abs/2605.29610)  
**Code**: https://github.com/Namgyu97/AlignG-SGG.pytorch  
**Area**: Multimodal VLM / Scene Graph Generation  
**Keywords**: Scene Graph Generation, Relationship Reasoning, Prototype Learning, Context-Conditioning, Predicate Disambiguation  

## TL;DR
AlignG transforms the static predicate prototypes of PE-Net into "image-conditioned" dynamic prototypes: first, relationship candidates are used to incrementally update prototypes via a GRU to obtain image-specific prototypes; then, these are used inversely to recalibrate relationship features. By anchoring the alignment loss to static global prototypes to prevent drift, the model achieves F@100 gains of 1.4 and 2.7 on the SGDet setting of VG-150 and GQA-200, respectively.

## Background & Motivation

**Background**: Scene Graph Generation (SGG) aims to represent an image as a graph of "objects + pairwise predicates," serving as a core task for structured scene understanding. One mainstream approach is prototype learning: PE-Net assigns a static prototype $\bar{\mathbf{p}}_r = \mathbf{W}_p \mathbf{t}_r$ (projected from word embeddings) to each predicate category, aligning relationship embeddings $\mathbf{e}_j$ with their corresponding prototypes. Subsequent works like C-SGG, UP-Net, and MCL further split predicates into multiple sub-prototypes to cover semantic diversity, while RA-SGG introduces retrieval-augmented external examples.

**Limitations of Prior Work**: Predicates are inherently ambiguous. "On" can denote spatial contact or functional use; "riding" and "standing on" often share nearly identical visual features in static images, differing only in intentionality. Whether using single, multiple, or retrieval-based prototypes, **as long as the prototypes remain static after training**, the model cannot utilize image-specific evidence—such as the set of relationship candidates existing in a specific image—to reorganize predicate semantics. This leads to systematic confusion in ambiguous scenes and the suppression of long-tail predicates by frequent ones.

**Key Challenge**: Prototypes must maintain **dataset-level semantic stability** (not drifting due to a single image) while possessing **image-level semantic flexibility** (distinguishing between skiing versus standing on a snowboard). These two requirements are mutually exclusive within a "static prototype" framework.

**Goal**: To redefine predicate learning from "image-agnostic static matching" to "image-conditioned adaptation," while providing an update mechanism that does not destroy the global topology.

**Key Insight**: The authors observe that the set of $N$ relationship candidates $\{\mathbf{e}_j\}_{j=1}^N$ in an image provides natural contextual evidence that can be "fed back" to the prototypes. Furthermore, gated incremental updates like GRU are inherently suitable for "absorbing new information without losing steady-state stability."

**Core Idea**: Establish a bidirectional interaction between "prototypes ↔ relationship candidates." Relationship candidates are first aggregated into image-conditioned prototypes, which then provide feedback to recalibrate relationship features. Crucially, the alignment loss is calculated against the **static prototypes** rather than the adapted ones, forcing the model to use image evidence to "adjust representations" rather than simply shifting both prototypes and relations toward each other.

## Method

### Overall Architecture
Input: Object features $\mathbf{x}_s, \mathbf{x}_o$ and category embeddings $\mathbf{t}_s, \mathbf{t}_o$ extracted by Faster R-CNN are fused to obtain relationship embeddings $\mathbf{e}_j = F(\mathbf{v}_s, \mathbf{v}_o) \in \mathbb{R}^d$, alongside PE-Net-style global static prototypes $\bar{\mathbf{p}}_r$.

AlignG adds two sequential modules:

1.  **Stage 1: Prototype Contextualization**: Uses cross-attention (query=prototype, key/value=relationship candidates) + GRUCell to incrementally update $\bar{\mathbf{p}}_r$ into image-specific prototypes $\mathbf{p}_r^{(I)}$.
2.  **Stage 2: Relationship Recalibration**: Uses inverse cross-attention (query=relationship, key/value=adapted prototypes) to obtain prototype-informed feedback $\mathbf{u}_j$. This is concatenated with $\mathbf{e}_j$ and passed through a projection network to obtain $\tilde{\mathbf{e}}_j$.

Finally, $\tilde{\mathbf{e}}_j$ is used for predicate classification, but the **alignment loss is anchored to the static $\bar{\mathbf{p}}_r$**, serving as the stability anchor for the framework.

### Key Designs

1.  **Context-Conditioned Prototype Update (Stage 1)**:
    - **Function**: Injects image-specific signals regarding "which relationship candidates are present" into the globally shared predicate prototypes.
    - **Mechanism**: For each prototype $r$, compatibility-weighted cross-attention generates a context vector $\mathbf{u}_r = \sum_j \alpha_{rj} \mathbf{W}_v \mathbf{e}_j$, where $\alpha_{rj} \propto \exp((\mathbf{W}_q \bar{\mathbf{p}}_r)^\top (\mathbf{W}_k \mathbf{e}_j) / \sqrt{d})$. A GRUCell then updates the prototype: $\mathbf{p}_r^{(I)} = \mathrm{GRUCell}(\mathbf{u}_r, \mathrm{LayerNorm}(\bar{\mathbf{p}}_r))$.
    - **Design Motivation**: Directly overwriting $\bar{\mathbf{p}}_r$ with $\mathbf{u}_r$ would destroy the global semantic topology. The GRU's reset/update gates provide a "selective absorption" mechanism, restricting the drift magnitude. This allows significant adjustments when scene evidence is strong and maintains the status quo when evidence is weak. This step converts "prototypes" from constants frozen after training into per-image variables.

2.  **Relationship Recalibration via Inverse Cross-Attention (Stage 2)**:
    - **Function**: Allows adapted prototypes to reshape relationship features, enabling embeddings to absorb the global semantic structure of the current image.
    - **Mechanism**: For each relationship $j$, inverse cross-attention computes prototype-informed feedback $\mathbf{u}_j = \sum_r \beta_{jr} \mathbf{W}_v' \mathbf{p}_r^{(I)}$, where $\beta_{jr} \propto \exp((\mathbf{W}_q' \mathbf{e}_j)^\top (\mathbf{W}_k' \mathbf{p}_r^{(I)}) / \sqrt{d})$. The final embedding is $\tilde{\mathbf{e}}_j = f_{\mathrm{proj}}([\mathrm{LayerNorm}(\mathbf{e}_j); \mathbf{u}_j])$ via a single-step concat-projection.
    - **Design Motivation**: Relationship embeddings are transient—they exist only for the current image and do not persist across samples. Therefore, they are suited for "one-shot" strong calibration rather than gated incremental updates. Computationally, by performing cross-attention between a fixed $R$ prototypes and $P$ candidates, the complexity is reduced from the $\mathcal{O}(P^2)$ of traditional self-attention to $\mathcal{O}(RP)$, which is particularly beneficial in dense scenes.

3.  **Static Prototype-Anchored Alignment Loss**:
    - **Function**: Imposes a steady-state constraint on the prototype adaptation to prevent representations and prototypes from collapsing into degenerate local solutions within a single image.
    - **Mechanism**: The alignment loss uses a triplet-margin form $\mathcal{L}_{\mathrm{align}} = \max\{0, \|\tilde{\mathbf{e}}_j - \bar{\mathbf{p}}^+\|_2^2 - \|\tilde{\mathbf{e}}_j - \bar{\mathbf{p}}^-\|_2^2 + \gamma\}$. **Crucially, $\bar{\mathbf{p}}^+$ and $\bar{\mathbf{p}}^-$ are taken from the static global prototypes**, not the adapted ones. This is combined with prototype regularization $\mathcal{L}_{\mathrm{reg}}$ and classification loss $\mathcal{L}_{\mathrm{cls}}$ for a total objective $\mathcal{L} = \mathcal{L}_{\mathrm{cls}} + \mathcal{L}_{\mathrm{reg}} + \mathcal{L}_{\mathrm{align}}$.
    - **Design Motivation**: If $\mathbf{p}_r^{(I)}$ were used as the alignment target, the relationship and prototype might "collude"—both being pushed toward a local solution within the image, leading the classifier to overfit to image-specific biases. Anchoring with $\bar{\mathbf{p}}_r$ forces the image-conditioned adaptation to stay within the "neighborhood" of the global prototype, ensuring stability while allowing local shifts.

### Loss & Training
The loss structure inherits from PE-Net without introducing additional frequency or co-occurrence priors; optional long-tail weighting is denoted by †. The optimizer is SGD (lr $1\times 10^{-3}$, momentum 0.9, weight decay $1\times 10^{-4}$) with a batch size of 8 for 60k iterations on an RTX 4090. Prototype dimensions $d$ are derived from GloVe 300-d projections, with $\gamma_{\mathrm{div}}=3.0$ and $\gamma=20.0$.

## Key Experimental Results

### Main Results (VG-150 + GQA-200)

| Dataset / Setting | Metric | PE-Net (backbone) | MCL† (Prev. SOTA) | RA-SGG† (Prev. SOTA) | AlignG† (Ours) |
|---|---|---|---|---|---|
| VG-150 / SGDet | mR@100 | 14.5 | 17.3 | 17.1 | **19.7** |
| VG-150 / SGDet | F@100 | 20.4 | 22.4 | 21.9 | **23.8 (+1.4)** |
| VG-150 / SGCls | F@100 | 25.8 | 29.9 | 28.6 | **30.2** |
| GQA-200 / SGDet | mR@100 | 11.9 | – | 15.0 | **15.5** |
| GQA-200 / SGDet | F@100 | 15.7 | – | 16.8 | **19.5 (+2.7)** |
| GQA-200 / PredCls | F@100 | 36.5 | – | 42.4 | **43.4 (+1.0)** |

Compared to PE-Net, AlignG† improves mR@100 by +8.8 / +7.2 / +5.2 across the three VG-150 settings, indicating that gains stem from the prototype feedback mechanism rather than conceptual expansion or external retrieval.

### Ablation Study (VG-150)

| Configuration | PredCls F@100 | SGCls F@100 | SGDet F@100 | Notes |
|---|---|---|---|---|
| PE-Net baseline | 45.0 | 25.8 | 20.4 | Static prototypes |
| + Edge update | 46.5 | 25.4 | 21.0 | Relationship-level modeling only |
| + Edge + Proto (concat) | 46.7 | 27.0 | 20.8 | Prototype update via concatenation |
| + Edge + Proto (GRU) | **47.5** | **27.2** | **21.3** | Gated GRU update |
| + † Freq Weighting | 50.3 | 30.2 | 23.8 | Long-tail weighting |

The GRU consistently outperforms concatenation across all three settings (+0.8/+0.2/+0.5 F@100), confirming that "gated incremental updates" are critical for prototypes that require steady-state stability.

### Key Findings
- **GRU > Concat**: Replacing GRU with concatenation leads to consistent performance drops, proving that "gated increments" are core to preventing prototype adaptation from degrading.
- **Static Anchoring > Adapted Anchoring**: Calculating alignment loss against $\bar{\mathbf{p}}_r$ instead of $\mathbf{p}_r^{(I)}$ is a key design choice to prevent collusion between relations and prototypes.
- **Greater gains on GQA-200**: The improvement in F@100 is more significant on the fine-grained, highly compositional GQA-200 (+2.7 vs +1.4), suggesting that context-conditioning yields higher returns in scenes with richer semantic structures.

## Highlights & Insights
- **Bidirectional Interaction**: Instead of unidirectional "prototype → relationship" influence, AlignG models "prototype ↔ relationship candidates," elevating context from an implicit internal variable to an explicit update signal.
- **Transferable Design Principles**: The distinction of "gated incremental updates for state variables and single-step strong calibration for transient variables" is a valuable lesson for other tasks requiring a balance between global and local information (e.g., prompt learning, retrieval-augmented embeddings).
- **Static Anchor + Dynamic Offset Paradigm**: Anchoring alignment loss to static prototypes is a clever "anti-collusion" mechanism, similar to the role of an EMA teacher in self-distillation, but applied to prototypes.

## Limitations & Future Work
- **Intent Inference Difficulty**: Confusion analysis shows that semantic distinctions like "riding ↔ standing on" require temporal or motion cues. Single-frame SGG frameworks are inherently limited here, and AlignG can only perform relative optimization on existing visual evidence.
- **Dependence on Pre-trained Detectors**: Frozen Faster R-CNN sets the ceiling via object proposal quality. If the detector misses key objects, the prototype feedback has no "raw material" to work with.
- **Fixed Prototype Count**: $R$ is pre-defined by the dataset. Extensions to open-vocabulary scenarios would require dynamic prototype generation mechanisms.

## Related Work & Insights
- **vs PE-Net (CVPR'23)**: PE-Net serves as the direct backbone. AlignG upgrades its "single static prototype" to a "global static prototype + per-image GRU update" while maintaining compatibility via static anchoring.
- **vs MCL (TIP'25)**: MCL uses multi-concepts to split predicates into multiple fixed sub-prototypes. AlignG follows a "lean but flexible" route—keeping the number of prototypes constant but allowing them to reorganize per image, proving "adaptive > diverse static."
- **vs RA-SGG (AAAI'25)**: RA-SGG uses externally retrieved examples, which are image-agnostic. AlignG's augmentation signal comes entirely from the relationship candidates within the current image, saving retrieval costs and avoiding external noise.

## Rating
- Novelty: ⭐⭐⭐⭐ — The transition from static to dynamic prototypes is a clear paradigm shift in SGG, though components (cross-attn + GRU) are classic.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive coverage across standard benchmarks and settings, including ablation, overhead, and confusion analysis.
- Writing Quality: ⭐⭐⭐⭐ — Clear derivation of motivation and design choices.
- Value: ⭐⭐⭐⭐ — Provides interpretable gains in a mature task; the "bidirectional + static anchor + gated" combination is transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Enhancing Visual Representation with Textual Semantics: Textual Semantics-Powered Prototypes for Heterogeneous Federated Learning](../../CVPR2026/optimization/enhancing_visual_representation_with_textual_semantics_textual_semantics_powered_p.md)
- [\[ICML 2026\] Test time training enhances in-context learning of nonlinear functions](test_time_training_enhances_in-context_learning_of_nonlinear_functions.md)
- [\[NeurIPS 2025\] Deep Taxonomic Networks for Unsupervised Hierarchical Prototype Discovery](../../NeurIPS2025/optimization/deep_taxonomic_networks_for_unsupervised_hierarchical_prototype_discovery.md)
- [\[ICLR 2026\] COLD-Steer: Steering Large Language Models via In-Context One-step Learning Dynamics](../../ICLR2026/optimization/cold-steer_steering_large_language_models_via_in-context_one-step_learning_dynam.md)
- [\[ICLR 2026\] Provable and Practical In-Context Policy Optimization for Self-Improvement](../../ICLR2026/optimization/provable_and_practical_in-context_policy_optimization_for_self-improvement.md)

</div>

<!-- RELATED:END -->
