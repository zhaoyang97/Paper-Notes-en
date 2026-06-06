---
title: >-
  [Paper Note] Text-Conditional JEPA for Learning Semantically Rich Visual Representations
description: >-
  [ICML 2026][Self-Supervised Learning][JEPA] This paper proposes TC-JEPA, which conditions the I-JEPA masked feature predictor additionally on image captions. By applying multi-layer sparse cross-attention…
tags:
  - "ICML 2026"
  - "Self-Supervised Learning"
  - "JEPA"
  - "text conditioning"
  - "feature prediction"
  - "fine-grained vision-language"
  - "cross-attention"
date: 2026-05-08
content_hash: 39cc8a3938e6cdf2
---

# Text-Conditional JEPA for Learning Semantically Rich Visual Representations

**Conference**: ICML 2026  
**arXiv**: [2605.03245](https://arxiv.org/abs/2605.03245)  
**Code**: None  
**Area**: Multimodal VLM / Self-supervised Representation Learning  
**Keywords**: JEPA, text conditioning, feature prediction, fine-grained vision-language, cross-attention

## TL;DR
This paper proposes TC-JEPA, which conditions the I-JEPA masked feature predictor additionally on image captions. By applying multi-layer sparse cross-attention, patch representations become predictable under textual "prompts," enabling the learning of semantically richer and dense prediction-friendly visual representations without contrastive loss.

## Background & Motivation

**Background**: Current visual self-supervised learning is dominated by two approaches. One is invariance-based methods (DINO, MoCo v3, iBOT, etc.), which learn high-level semantics by enforcing consistency between representations of different augmented views of the same image. The other is masked image modeling (MIM), with I-JEPA as a representative, which predicts masked patch features in feature space—easier to balance local structure and high-level semantics compared to pixel reconstruction methods like MAE.

**Limitations of Prior Work**: The core pretext task of I-JEPA is inherently uncertain—given context patches, there are many plausible answers for the masked patch (e.g., in a dog image, the masked area could be a bookshelf or a clean wall). This ambiguity makes training highly sensitive to masking strategy; when context and target have low mutual information, feature prediction degrades or even collapses. Existing fixes like positional encoders or random position encoding do not introduce new information sources.

**Key Challenge**: JEPA aims to "replace alignment with prediction," but image signals alone cannot resolve the multimodal ambiguity of masked regions. Without addressing this, the prediction target cannot converge to semantically meaningful representations.

**Goal**: (i) Inject additional information sources into the JEPA predictor to reduce prediction uncertainty; (ii) Achieve finer-grained vision-language alignment than CLIP/SigLIP without introducing contrastive loss or relying on grounding annotations.

**Key Insight**: Human or synthetic captions for images almost always describe scene composition ("dog + bookshelf"), directly informing the model what the masked region "should be." Feeding this supervision to the predictor (not the encoder) can greatly compress the prediction distribution while preserving the JEPA representation structure.

**Core Idea**: Replace the original JEPA predictor with a fine-grained "text-conditioned predictor"—patch features become predictable latent variables modulated by the caption word sequence; captions are used only during pretraining and discarded at downstream inference.

## Method

### Overall Architecture
TC-JEPA structurally follows I-JEPA: the image is split into context patches $x$ and target patches $y$, with context encoder $f_\theta$ and EMA target encoder $f_{\bar\theta}$ producing $z_x, z_y$. The narrow ViT predictor $g_\phi$ predicts $\hat z_y$ at masked token positions, and the training loss is $\mathcal{L}_{\text{predict}}=\frac{1}{|B_y|}\sum_j\|\hat z_{y_j}-z_{y_j}\|_2$. The key change: $g_\phi$ also receives up to $N=8$ captions, each mapped to a word sequence $t\in\mathbb{R}^{d_t\times S}$ via a pretrained T5. At every predictor layer, patch representations are modulated by cross-attention over $t$. The entire pipeline is trained with feature prediction loss only—no contrastive loss, no grounding boxes.

### Key Designs

1. **Multi-layer Fine-grained Text Conditioner (cross-attention over word sequence)**:

    - **Function**: At each predictor layer, patch features $q\in\{\hat z_x^{(l)}, \hat z_y^{(l)}\}$ attend to the caption word sequence $t$ via lightweight cross-attention, allowing each patch to "select" the most relevant words to aid its feature prediction.
    - **Mechanism**: For each layer, define $q^{(l)}=W_Q^{(l)}q$, $K^{(l)}=W_K^{(l)}t$, $V^{(l)}=W_V^{(l)}t$, then update $q\leftarrow q+\sum_s\text{softmax}(q^{(l)\top}K_{:,s}^{(l)})V_{:,s}^{(l)}$, followed by a residual MLP+LayerNorm. Compared to "sequence conditioning" (concatenating captions as extra tokens), this approach does not lengthen the ViT sequence, injects text signals at all layers, and is not limited to shallow layers.
    - **Design Motivation**: The core argument is to make patch representations "predictable under textual prompts," so conditioning must penetrate every layer and enable sparse patch-word correspondence (akin to self-supervised visual grounding), thereby enforcing patch-language alignment.

2. **Sparse + Cross-layer Consistency Regularization**:

    - **Function**: Constrains patch-word cosine similarity $O_i^{(l)}=\max(\cos(q^{(l)},K^{(l)}),0)$ to be sparse and consistent across layers, preventing text conditioning from degenerating into meaningless uniform attention.
    - **Mechanism**: For each patch at each layer, compute $O_i^{(l)}$, add $\ell_1$ sparsity penalty $\mathcal{L}_{\text{sparse}}=\frac{1}{|B_x|+|B_y|}\sum_i\frac{1}{L}\sum_l\|O_i^{(l)}\|_1$ to encourage each patch to select only a few keywords; add $\mathcal{L}_{\text{consistency}}=\frac{1}{|B_x|+|B_y|}\sum_i\frac{1}{L}\sum_l\|O_i^{(l)}-\bar O_i\|_1$ to enforce consistent word selection across layers, where $\bar O_i=\frac{1}{L}\sum_l O_i^{(l)}$.
    - **Design Motivation**: Without explicit grounding supervision, cross-attention may form meaningless alignments; the combination of sparsity and consistency regularization encourages training to converge to "each patch corresponds to a few stable relevant words," effectively constructing unsupervised visual grounding so that text conditioning truly aids prediction.

3. **Multiple Caption Independent Conditioning + Feature-level Max-pool Fusion**:

    - **Function**: When an image has $N$ captions, instead of concatenating them, each caption independently conditions the predictor, and max-pooling is performed across the feature dimension, preserving each caption's perspective and amplifying the most useful signal.
    - **Mechanism**: At layer $l$, use caption $t^n$ to obtain $\hat z_{y_{j,n}}^{(l)}$ and $\hat z_{x_{i,n}}^{(l)}$, then max-pool along $n$ to get $\hat z_{y_j}^{(l)}$, $\hat z_{x_i}^{(l)}$ for the next layer; final loss is $\mathcal{L}=\mathcal{L}_{\text{predict}}+\frac{\lambda}{N}\sum_n\mathcal{L}_{\text{sparse}}^n+\frac{\beta}{N}\sum_n\mathcal{L}_{\text{consistency}}^n$, with $\lambda=0.1$, $\beta=0.5$.
    - **Design Motivation**: Concatenating multiple captions into a long sentence causes each patch to attend to all captions simultaneously, leading to interference; conditioning on each caption separately preserves their differences, and max-pooling naturally selects the "most useful" caption for each patch, acting as a caption-level sparse selection.

### Loss & Training
The total loss includes feature prediction, sparsity, and consistency terms. The target encoder uses EMA + stop-gradient to prevent collapse. Pretraining datasets include IN-1k / IN-21k (with 8.3–8.7 synthetic captions per image via ShareGPT4V) and CC12M+YFCC15M image-text pairs (also supplemented with synthetic captions). Backbones include ViT-B/16, ViT-L/16, ViT-H/14; IN-21k is trained for 600–300 epochs; hyperparameters $\lambda,\beta$ are insensitive.

## Key Experimental Results

### Main Results

| Task | Model / Data | I-JEPA / StoP | TC-JEPA | Gain |
|------|--------------|---------------|---------|------|
| IN-1k linear (ViT-H/14, IN-1k) | Top-1 | 79.3 / 79.6 | 80.4 | +1.1 |
| IN-1k linear (ViT-L/16, IN-21k) | Top-1 | 77.2 (I-JEPA) | 82.1 | +4.9 |
| ADE20k mIoU (linear, ViT-H/14) | mIoU | 36.9 / 36.6 | 39.5 | +2.6 |
| COCO det (ViT-H/14) | AP$^b$ | 53.7 / 53.5 | 55.2 | +1.5 |
| ADE20k mIoU (ViT-L/16, CC27M) | mIoU | – | 42.1 | New SOTA |
| vs SigLIP2 (ViT-L/16, ADE20k mIoU) | mIoU | 24.6 | 41.2 | +16.6 |

The second table compares pretraining on image-text pairs: TC-JEPA on IN-21k achieves ADE20k mIoU surpassing DINOv2 (41.8, distilled from 5× data) and Web-DINO (40.3, 75× data); training on CC27M yields 42.1, clearly outperforming CLIP/SigLIP with comparable data for dense tasks.

### Ablation Study

| Configuration | IN-1k Top-1 / ADE20k mIoU | Notes |
|---------------|---------------------------|-------|
| Full TC-JEPA (ViT-L/16, IN-21k) | 82.1 / 41.2 | Complete method |
| Remove sparse + consistency regularization | Significant drop | Patch-word attention degenerates to uniform, text modulation fails |
| Sequence conditioning (concatenate captions as input) | Worse than cross-attn | Conditioning only in shallow layers, longer sequence increases cost |
| Single caption ($N=1$) | Worse than $N=8$ max-pool | Single caption cannot cover all visual details, multi-caption max-pool brings clear gains |
| I-JEPA baseline | 77.2 / 38.2 | No text conditioning |

### Key Findings
- Text conditioning yields much greater gains for dense tasks (segmentation, detection) than for classification, indicating that reducing prediction uncertainty mainly improves patch-level feature quality, directly addressing the weaknesses of contrastive methods like SigLIP.
- On IN-21k, TC-JEPA's ADE20k mIoU matches Franca (which combines invariance + MIM), demonstrating that fine-grained text conditioning can replace handcrafted invariance constraints.
- As data scales up, TC-JEPA's scaling curve consistently outperforms I-JEPA, while I-JEPA shows no clear scaling on IN-1k classification, suggesting that text signals are key to stable scaling.

## Highlights & Insights
- Placing "text" in the predictor rather than the encoder is a key shift: the encoder is no longer compressed by language into CLIP-like global abstractions; patch features retain visual detail but become "predictable under textual prompts" as latent variables. At inference, text is discarded and pure visual representations are used, fully compatible with existing visual backbones.
- The combination of sparse + consistent regularization pushes cross-attention to form implicit visual grounding, bypassing the need for grounding data. This approach of "using auxiliary loss to drive attention for semantic alignment" can be generalized to any pretraining requiring cross-modal alignment.
- Multi-caption max-pool fusion is a practical trick: it avoids the "multiple captions interfering with the same patch" problem caused by concatenation, performs fusion in feature space rather than token space, is lightweight, and naturally induces sparse selection.

## Limitations & Future Work
- TC-JEPA requires 5–10 synthetic captions per image and is sensitive to caption quality and quantity; the LMM cost of generating captions is non-negligible for industrial deployment.
- Text conditioning is only used during pretraining; downstream inference cannot explicitly leverage text prompts for zero-shot retrieval/classification, so TC-JEPA still lags behind contrastive methods on zero-shot tasks (the paper does not compare zero-shot retrieval).
- Multi-layer cross-attention and multi-caption conditioning add computational overhead to the predictor; scaling to ViT-G level requires further validation of training cost and stability.
- The paper does not deeply discuss how synthetic caption bias and hallucination may contaminate representations, which could be exacerbated when changing caption generators.

## Related Work & Insights
- **vs I-JEPA / StoP / CAPI**: All are latent MIM methods, but TC-JEPA addresses "uncertainty" by introducing a new information source (text) rather than architectural tricks (positional conditioning, random position encoding, cluster prediction), yielding more direct and comprehensive improvements.
- **vs CLIP / SigLIP series**: Also use image-caption pairs, but TC-JEPA does not use contrastive loss, so the feature space is not globally compressed, leading to much better performance on dense tasks; the downside is the inability to perform zero-shot image-text retrieval directly.
- **vs DINOv2 / iBOT / Franca**: These methods rely on "invariance+MIM" combinations for strong representations, requiring carefully designed image augmentations; TC-JEPA replaces augmentation with text conditioning, achieving comparable or better mIoU on IN-21k, suggesting "language can be another form of data augmentation."
- **vs SPARC / DreamLIP (fine-grained contrastive methods with synthetic captions)**: Also use synthetic captions, but TC-JEPA treats captions as predictor conditions rather than contrastive targets, making it more suitable for dense tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ Injecting captions into the JEPA predictor is a relatively natural but previously unexplored direction; the combination of cross-attention, sparse-consistency regularization, and multi-caption max-pool is a clear engineering contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 3 model scales, 3 data scales, multiple tasks (classification/detection/segmentation), and systematically compares MIM, invariance, and contrastive methods.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and derivation, good combination of method diagrams and formulas, though some sections are dense to fit 8 pages.
- Value: ⭐⭐⭐⭐ Opens a "weak text supervision" scaling path for JEPA methods, with high downstream value for dense prediction and visual foundation models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MINE-JEPA: In-Domain Self-Supervised Learning for Mineral Exploration](../../CVPR2026/self_supervised/mine-jepa_in-domain_self-supervised_learning_for_mine-like_object_classification.md)
- [\[ICML 2026\] Provable Accuracy Collapse in Embedding-Based Representations under Dimensionality Mismatch](provable_accuracy_collapse_in_embedding-based_representations_under_dimensionali.md)
- [\[ICCV 2025\] Scaling Language-Free Visual Representation Learning](../../ICCV2025/self_supervised/scaling_languagefree_visual_representation_learning.md)
- [\[NeurIPS 2025\] Towards Reliable and Holistic Visual In-Context Learning Prompt Selection](../../NeurIPS2025/self_supervised/towards_reliable_and_holistic_visual_in-context_learning_prompt_selection.md)
- [\[NeurIPS 2025\] Contrastive Representations for Temporal Reasoning](../../NeurIPS2025/self_supervised/contrastive_representations_for_temporal_reasoning.md)

</div>

<!-- RELATED:END -->
