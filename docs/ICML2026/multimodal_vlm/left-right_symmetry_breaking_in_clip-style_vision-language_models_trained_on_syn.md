---
title: >-
  [Paper Note] Left-Right Symmetry Breaking in CLIP-style Vision-Language Models Trained on Synthetic Spatial-Relation Data
description: >-
  [ICML 2026][Multimodal VLM][CLIP] The authors train a CLIP-style Transformer end-to-end using a 1D synthetic image-text testbed and find that such models can learn "left/right" relations and generalize to unseen object p…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "CLIP"
  - "Spatial Reasoning"
  - "Mechanistic Interpretability"
  - "Attention Decomposition"
  - "Position Embeddings"
date: 2026-05-08
content_hash: 6ddd8d25d14f5380
---

# Left-Right Symmetry Breaking in CLIP-style Vision-Language Models Trained on Synthetic Spatial-Relation Data

**Conference**: ICML 2026  
**arXiv**: [2601.12809](https://arxiv.org/abs/2601.12809)  
**Code**: None (Modified from OpenAI CLIP and Sea-Snell/grokking public code)  
**Area**: Multimodal VLM  
**Keywords**: CLIP, Spatial Reasoning, Mechanistic Interpretability, Attention Decomposition, Position Embeddings

## TL;DR
The authors train a CLIP-style Transformer end-to-end using a 1D synthetic image-text testbed and find that such models can learn "left/right" relations and generalize to unseen object pairs. The mechanism is a **horizontal gradient induced by the cross-term $EW_{QK}P^T$ of position and token embeddings in the vision encoder attention logits**, which breaks left-right symmetry; ablating this term drops left-right discrimination accuracy to random levels.

## Background & Motivation

**Background**: CLIP-style VLMs are powerful in zero-shot retrieval and classification but consistently fail in relationship understanding ("who is to the left of whom"), spatial reasoning, and compositional generalization. Benchmarks like ARO, CLEVR, Winoground, and NLVR2 consistently show that large VLMs often degenerate into a "bag-of-words"—identifying what is present but not how they are arranged.

**Limitations of Prior Work**: While evaluative studies are numerous, **mechanistic explanations are scarce**: it remains unclear which specific path VLMs use to perceive "left vs right," and no study has proven causality by directly disabling this ability through the ablation of specific components. Recent work has pointed to visual tokens suppressing positional information in LLMs (Qi 2025) or attributed spatial failure to training data (Chen 2024), but a unified picture is missing.

**Key Challenge**: The CLIP training objective does not explicitly require the model to distinguish between "left of X" and "right of X"; the contrastive loss can be fully satisfied without utilizing compositional structure. Why do some models learn it while others do not? Which part of the architecture makes the difference?

**Goal**: To answer in a fully controllable minimal setting: (a) Can CLIP-style Transformers learn faithful encodings of relative spatial relations? (b) What mechanism implements this? (c) Which training factors are critical?

**Key Insight**: Following the tradition of mechanistic interpretability (Elhage 2021/2022, Olsson 2022, Okawa 2023), the authors reverse-engineer attention circuits using a minimalist toy task and small models. Specifically, they reduce images to a 1D sequence of 10 pixels, where objects occupy 1 pixel, and use simple text templates like "X is on the left of Y" with a 1-layer / 4-head Transformer.

**Core Idea**: First, demonstrate that this minimalist version can replicate the "label diversity drives generalization" phenomenon. Then, perform a four-term decomposition of the attention logits into token and position embeddings to identify the unique term that breaks left-right symmetry, confirming it as a necessary condition through ablation experiments.

## Method

The methodology is not a new model but a four-part suite: **controllable synthetic dataset + simplified Transformer + attention decomposition + ablation**. The synthetic data allows for precise control of variables; the simplified Transformer (no LayerNorm/MLP, 1 layer, 4 heads, Elhage 2021 style) makes analysis tractable; term-by-term decomposition reveals which part causes left-right asymmetry; and ablation upgrades correlation to causality.

### Overall Architecture
(1) Synthetic 1D Image-Text Data: Images are 1D sequences of length $D^{\rm image}=10$ with background 0 and object IDs $\geq 1$ (single or dual objects); captions use the template "[label] is on the left/right of [label]". Training uses all ordered pairs from $N_{\rm pair}=15$ labels ($N_{\rm val}=5$ held out), with positions randomly sampled.  
(2) Dual-Encoder CLIP: The vision encoder uses bidirectional self-attention, while the text encoder uses causal masking. Both share $d_{\rm model}=128$, $d_{\rm head}=32$. CLS / EOT tokens are used for final representations with cosine similarity and standard CLIP contrastive loss.  
(3) Evaluation of Three Generalizations: Single-object positional, seen-pair configuration, and unseen-pair generalization.  
(4) **Attention Decomposition**: For a generalized 1-layer 4-head model, decompose the pre-softmax logit $QK^T$ into weight-bias terms, then expand the main term $XW_{QK}X^T$ (where $X=E+P$) into four terms ($EE, EP, PE, PP$), visualizing and ablating them individually.

### Key Designs

1.  **Controllable 1D Synthetic Dataset + Dual-Axis Scanning**:
    *   **Function**: Enables the study of drivers for spatial relation generalization as intervenable variables while keeping the CLIP pipeline intact.
    *   **Mechanism**: Reduces images to 10-pixel 1D sequences where "left/right" is the only spatial degree of freedom. Two axes are scanned during training: label diversity $N_{\rm pair} \in \{5, \dots, 15\}$ and layout diversity $n_2$ (number of position combinations per pair). Key observation: Increasing $N_{\rm pair}$ significantly improves all three types of generalization, while increasing $n_2$ has almost no effect—**label diversity, not layout diversity**, is the primary driver of generalization. This aligns with Uselis 2025 regarding "diversity driving compositionality" but specifies that diversity must occur along the "label" axis for relational tasks.
    *   **Design Motivation**: Mechanistic research requires both "generalizable" and "non-generalizable" samples for comparison. The 1D design limits the vision encoder to 10 key positions, making the resulting logit decomposition heatmaps human-interpretable.

2.  **Simplified Transformer + Weight-Bias-Token-Position Logit Decomposition**:
    *   **Function**: Splits vision encoder attention logits into interpretable components to identify which term encodes the "left/right" signal.
    *   **Mechanism**: Following Elhage 2021, LayerNorm and MLP are removed, leaving a 1-block 4-head Transformer. Defining $Q=XW_Q^T+B_Q^T$ and $K=XW_K^T+B_K^T$, then $QK^T = XW_{QK}X^T + XW_Q^TB_K + B_Q^TW_KX^T + B_Q^TB_K$, where $W_{QK}=W_Q^TW_K$. Since Softmax is invariant to row-wise constant subtractions, only columns that vary—$XW_{QK}X^T$ and $B_Q^TW_KX^T$—affect the CLS row attention. Substituting $X=E+P$ results in: $XW_{QK}X^T = EW_{QK}E^T + EW_{QK}P^T + PW_{QK}E^T + PW_{QK}P^T$ (denoted as EE, EP, PE, PP). Visualization reveals that **only the EP term $EW_{QK}P^T$ produces a clear left-to-right monotonic gradient on the CLS row**, adding logit bias to right-side objects. In non-generalizing models, this EP horizontal gradient is entirely absent (App. G).
    *   **Design Motivation**: Additive decomposition treats attention as four channels: Content-Content (EE), Content-Position (EP), Position-Content (PE), and Position-Position (PP). This directly locates the channel through which left-right asymmetry flows.

3.  **EP Term Ablation as Causal Evidence**:
    *   **Function**: Manually zeros out the EP term during inference to see if unseen-pair accuracy collapses.
    *   **Mechanism**: The EP term in pre-softmax logits is forced to 0 for all 4 heads (using pre-trained weights without retraining). Other terms like PP/PE/BP ($B_Q^TW_KP^T$) are ablated as negative controls. Result: **EP ablation drops accuracy from $\approx 0.9$ to $\approx 0.5$ (random)**, while PP/PE ablation has minimal impact. App. H shows ablated models still recognize "X and Y are in the image" but cannot judge their relative positions, cleanly decoupling "recognition" from "spatial encoding."
    *   **Design Motivation**: While seeing a gradient in the EP term provides correlational evidence, "hard ablation" provides causal evidence: without this term, the capability disappears. This adheres to the ablation-as-causation principle in mechanistic interpretability.

### Text Side and Alignment
The text encoder's causal mask inherently encodes sequence order. At least one of the four heads strongly biases EOT→word attention toward the first mentioned entity, independent of the label, forming a "linguistic left-right break" symmetrical to the vision side. The authors also find that while image/text token embeddings for the same label have low cosine similarity in raw space, they can be aligned on unseen labels (16-20) by fitting a rotation matrix on seen labels (1-15)—suggesting CLIP alignment exists in a rotation quotient space.

## Key Experimental Results

### Generalization Improvement with Label Diversity

| $N_{\rm pair}$ (Training Labels) | Single-object positional | Seen-pair configuration | Unseen-pair |
| :--- | :--- | :--- | :--- |
| 5 (Low) | Medium | Medium | Near Random |
| 15 (High) | High | High | High (Near Perfect) |
| Layout Diversity $n_2$ variation | Almost No Impact | Almost No Impact | Almost No Impact |

(Values summarized from Fig. 3 trend; shows "label diversity" is the driver.)

### Effect of Logit Term Ablation on Unseen-pair Accuracy

| Ablation Condition (Zeroed at Inference) | Unseen-pair Accuracy | Note |
| :--- | :--- | :--- |
| Baseline (No Ablation) | $\approx 0.9$ | Model generalizes |
| Ablate EP term $EW_{QK}P^T$ | $\approx 0.5$ | **Drops to random**; loses spatial judgment |
| Ablate PE term $PW_{QK}E^T$ | Near Baseline | Not responsible for spatial encoding |
| Ablate PP term $PW_{QK}P^T$ | Near Baseline | Not responsible for spatial encoding |
| Ablate BP term $B_Q^TW_KP^T$ | Moderate Drop | Bias-position coupling carries some signal |
| Ablate EP + Value-channel VP term $PW_V^T$ | $\approx 0.5$ | Attention and values cooperate |

(Values summarized from Fig. 5(e) and App. I.)

### Key Findings
- **Label Diversity >> Layout Diversity**: Increasing labels used in pairs from 5 to 15 lifts all generalizations to near-perfect levels; increasing position combinations $n_2$ does nothing.
- **EP Term is Necessary for Unseen-pair Generalization**: Ablating it drops accuracy to $\approx 0.5$ while retaining label-set recognition, precisely excising spatial ability.
- In non-generalizing models, the horizontal gradient in the EP term is totally missing (App. G)—the presence of the gradient correlates exactly with the capability.
- Scaling captions from "only left" to "left + right" requires 2 text layers to maintain generalization, while the vision side mechanism remains focused on the EP term.
- The mechanism replicates in 2D settings ($4\times 4$ grid) and 3-object settings, and is observed in autoregressive VLMs (App. O).

## Highlights & Insights
- **Transformation of a Vague Capability into a Causal Hypothesis**: The study moves from "CLIP can do left/right" to "which logit term drives it," a clean demonstration of mechanistic interpretability.
- **EP Term as a Functional Unit**: The content-position cross-term is not just a mathematical artifact but a functional circuit; position embeddings are the carrier of compositional generalization, not just a syntactic skeleton.
- **Label vs. Layout Diversity Asymmetry**: This provides direct guidance for CLIP data curation—budgets should favor label combinations over positional variations. This also explains why large-scale VLMs struggle with spatial relations: web-scale entity-pair coverage remains sparse despite massive label counts.
- **Rotation Quotient Space**: The discovery that token embeddings align only after a rotation hints that CLIP's geometric space is far richer than what simple cosine similarity reveals.

## Limitations & Future Work
- Experiments use 1D/2D toy tasks and tiny Transformers (1-2 layers); verification on web-scale CLIP is required. This is a "first-order mechanistic understanding."
- Only covers left/right relations; "front/back," "inside/outside," or "overlap" may rely on different mechanism carriers.
- Text-side mechanism becomes less "clean" with more layers; the authors focused on the vision side.
- Removing LayerNorm/MLP simplifies analysis, but non-linear components likely modify the mechanism (App. R).

## Related Work & Insights
- **vs. Yuksekgonul 2023 (ARO bag-of-words)**: They empirically noted CLIP's degeneration; this paper explains "why it doesn't degenerate under certain conditions"—label diversity allows the EP term to learn the necessary gradient.
- **vs. Qi 2025 (Beyond Semantics)**: They observed positional suppression in LLMs; this paper shows position is crucial *inside* the vision encoder for spatial generalization.
- **vs. Uselis 2025 (Diversity → Compositionality)**: They focused on additive factorization for attributes; this paper focuses on positional dependencies for relations.
- **vs. Elhage 2021/2022 (Circuits)**: Extension of mechanistic interpretability to multimodal contrastive learning.

## Rating
- Novelty: ⭐⭐⭐⭐ Toy tasks are common, but identifying the EP term as the causal path for CLIP is a significant step in mechanistic understanding.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers various generalizations, decompositions, and ablations across 1D/2D and different paradigms.
- Writing Quality: ⭐⭐⭐⭐ Clear concepts, but some crucial data is relegated to the appendix.
- Value: ⭐⭐⭐⭐ Offers the first mechanistic answer to a problem previously only analyzed via benchmarks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] CLIPSym: Delving into Symmetry Detection with CLIP](../../ICCV2025/multimodal_vlm/clipsym_delving_into_symmetry_detection_with_clip.md)
- [\[ICML 2026\] 3ViewSense: Spatial and Mental Perspective Reasoning from Orthographic Views in Vision-Language Models](3viewsense_spatial_and_mental_perspective_reasoning_from_orthographic_views_in_v.md)
- [\[ICML 2026\] Active Exploring like a Pigeon: Reinforcing Spatial Reasoning via Agentic Vision-Language Models](active_exploring_like_a_pigeon_reinforcing_spatial_reasoning_via_agentic_vision-.md)
- [\[ICML 2026\] Hierarchical Synthetic Tabular Data Generation: A Hybrid Top-Down and Bottom-Up Framework](hierarchical_synthetic_tabular_data_generation_a_hybrid_top-down_and_bottom-up_f.md)
- [\[ICLR 2026\] Breaking the Limits of Open-Weight CLIP: An Optimization Framework for Self-supervised Fine-tuning of CLIP](../../ICLR2026/multimodal_vlm/breaking_the_limits_of_open-weight_clip_an_optimization_framework_for_self-super.md)

</div>

<!-- RELATED:END -->
