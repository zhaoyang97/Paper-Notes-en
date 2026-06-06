---
title: >-
  [Paper Note] Text-Conditional JEPA for Learning Semantically Rich Visual Representations
description: >-
  [ICML 2026][Multimodal VLM][JEPA] This paper proposes TC-JEPA, which additionally conditions the I-JEPA mask feature predictor on image captions. Through multi-layer sparse cross-attention…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "JEPA"
  - "text-conditional"
  - "feature prediction"
  - "fine-grained vision-language"
  - "cross-attention"
date: 2026-05-08
content_hash: 4cb5e1364fbcfcbe
---

# Text-Conditional JEPA for Learning Semantically Rich Visual Representations

**Conference**: ICML 2026  
**arXiv**: [2605.03245](https://arxiv.org/abs/2605.03245)  
**Code**: None  
**Area**: Multimodal VLM / Self-supervised representation learning  
**Keywords**: JEPA, text-conditional, feature prediction, fine-grained vision-language, cross-attention

## TL;DR
This paper proposes TC-JEPA, which additionally conditions the I-JEPA mask feature predictor on image captions. Through multi-layer sparse cross-attention, patch representations become predictable under text "prompts." This enables the learning of semantically richer visual representations that are particularly friendly to dense prediction tasks without using contrastive loss.

## Background & Motivation

**Background**: Visual self-supervised learning is currently dominated by two categories of methods. One is invariance-based methods (DINO, MoCo v3, iBOT, etc.), which learn high-level semantics by making representations of different augmented views of the same image consistent. The other is Masked Image Modeling (MIM), represented by I-JEPA, which predicts masked patch features in the feature space, better balancing local structure and high-level semantics compared to pixel-reconstruction methods like MAE.

**Limitations of Prior Work**: The core pretext task of I-JEPA suffers from inherent uncertainty—given context patches to predict features at a specific mask location, there are many plausible answers (e.g., a masked area in an image of a dog could be a bookshelf or a clean wall). This ambiguity makes training highly sensitive to masking strategies; when the mutual information between context and target is low, feature prediction degrades or representation collapse occurs. Existing patches like positional conditional encoders or stochastic positional encoding do not introduce new information sources.

**Key Challenge**: JEPA aims to "replace alignment with prediction," but image signals alone cannot eliminate the multimodal ambiguity of masked regions. If ambiguity is not resolved, the prediction target will not converge to semantically meaningful representations.

**Goal**: (i) Inject additional information sources into the JEPA predictor to reduce prediction uncertainty; (ii) learn finer-grained vision-language alignment than CLIP/SigLIP without introducing contrastive loss or relying on grounding annotations.

**Key Insight**: Human or synthetic captions for images almost always describe scene composition ("dog + bookshelf"), which tells the model what the masked area "should be." Feeding this supervision to the predictor rather than the encoder allows for significant compression of the prediction distribution while preserving the JEPA representation structure.

**Core Idea**: Replace the original JEPA predictor with a fine-grained "text-conditional predictor"—patch features are no longer unconditional feature vectors but predictable latent variables "modulated" by caption word sequences. Captions are used only during the pre-training phase and discarded during downstream inference.

## Method

### Overall Architecture
TC-JEPA follows the I-JEPA structure: an image is divided into context patches $x$ and target patches $y$. The context encoder $f_\theta$ and EMA target encoder $f_{\bar\theta}$ produce $z_x$ and $z_y$, respectively. A narrow ViT predictor $g_\phi$ predicts $\hat z_y$ at mask token positions, using the training loss $\mathcal{L}_{\text{predict}}=\frac{1}{|B_y|}\sum_j\|\hat z_{y_j}-z_{y_j}\|_2$. The key change is that $g_\phi$ simultaneously takes a set of (up to $N=8$) captions as input. Each caption is mapped to a word sequence $t\in\mathbb{R}^{d_t\times S}$ using a pre-trained T5, and cross-attention modulation on $t$ is overlaid on the patch representations at every layer of the predictor. The entire pipeline is trained only with feature prediction loss, without contrastive loss or grounding boxes.

### Key Designs

1.  **Multi-layer fine-grained text conditioner (cross-attention over word sequence)**:
    - **Function**: Performs lightweight cross-attention between patch features $q\in\{\hat z_x^{(l)}, \hat z_y^{(l)}\}$ and caption word sequences $t$ at each predictor layer, allowing each patch to "on-demand" select the most relevant words to assist its feature prediction.
    - **Mechanism**: Each layer defines $q^{(l)}=W_Q^{(l)}q$, $K^{(l)}=W_K^{(l)}t$, $V^{(l)}=W_V^{(l)}t$, then $q\leftarrow q+\sum_s\text{softmax}(q^{(l)\top}K_{:,s}^{(l)})V_{:,s}^{(l)}$, followed by an MLP and LayerNorm after the residual update. Compared to "sequence conditioning" which appends captions as tokens to the ViT input, this approach does not extend sequence length, acts beyond just the bottom layers, and continuously injects text signals across all layers.
    - **Design Motivation**: The core argument is to make patch representations "predictable under text prompts," so conditioning must go deep into every layer and form sparse correspondences between patches and words (similar to self-supervised visual grounding) to constrain patch representations toward linguistic alignment.

2.  **Sparse + cross-layer consistency regularization**:
    - **Function**: Constraints the patch-word cosine similarity $O_i^{(l)}=\max(\cos(q^{(l)},K^{(l)}),0)$ to be a sparse and cross-layer consistent distribution, preventing the text condition from degrading into meaningless averaging over all words.
    - **Mechanism**: $O_i^{(l)}$ is calculated for each patch at each layer. An $\ell_1$ sparsity penalty $\mathcal{L}_{\text{sparse}}=\frac{1}{|B_x|+|B_y|}\sum_i\frac{1}{L}\sum_l\|O_i^{(l)}\|_1$ forces each patch to select only a few keywords. $\mathcal{L}_{\text{consistency}}=\frac{1}{|B_x|+|B_y|}\sum_i\frac{1}{L}\sum_l\|O_i^{(l)}-\bar O_i\|_1$ ensures consistency in word selection across layers, where $\bar O_i=\frac{1}{L}\sum_l O_i^{(l)}$.
    - **Design Motivation**: In the absence of explicit grounding supervision, cross-attention might form meaningless alignments. The sparsity and consistency constraints encourage training to converge to "each patch corresponding to stable relevant words," implicitly constructing unsupervised visual grounding.

3.  **Multi-caption independent conditioning + feature-level max-pool fusion**:
    - **Function**: When multiple captions ($N$) are available for an image, they are not concatenated. Instead, each caption independently conditions the predictor, followed by max-pooling fusion in the feature dimension to preserve individual perspectives and amplify the most useful signals.
    - **Mechanism**: At layer $l$, the $n$-th caption $t^n$ produces $\hat z_{y_{j,n}}^{(l)}$ and $\hat z_{x_{i,n}}^{(l)}$. Max-pooling is then applied along the $n$ dimension to obtain $\hat z_{y_j}^{(l)}$ and $\hat z_{x_i}^{(l)}$ for the next layer. The final loss is $\mathcal{L}=\mathcal{L}_{\text{predict}}+\frac{\lambda}{N}\sum_n\mathcal{L}_{\text{sparse}}^n+\frac{\beta}{N}\sum_n\mathcal{L}_{\text{consistency}}^n$, with $\lambda=0.1, \beta=0.5$.
    - **Design Motivation**: Concatenating multiple captions into a long sentence causes interference. Conditioning separately preserves differentiated information, while max-pooling naturally selects the "most useful" caption for each patch, acting as a caption-level sparse selection.

### Loss & Training
The total loss includes feature prediction, sparsity, and consistency terms. The target encoder follows EMA + stop-gradient to prevent collapse. Pre-training datasets include IN-1k / IN-21k (with 8.3–8.7 synthetic captions per image using ShareGPT4V) and CC12M+YFCC15M pairs. Backbones include ViT-B/16, ViT-L/16, and ViT-H/14. IN-21k training lasts for 600–300 epochs. Hyperparameters $\lambda, \beta$ are non-sensitive.

## Key Experimental Results

### Main Results

| Task | Model / Data | I-JEPA / StoP | TC-JEPA (Ours) | Gain |
| :--- | :--- | :--- | :--- | :--- |
| IN-1k linear (ViT-H/14, IN-1k) | Top-1 | 79.3 / 79.6 | 80.4 | +1.1 |
| IN-1k linear (ViT-L/16, IN-21k) | Top-1 | 77.2 (I-JEPA) | 82.1 | +4.9 |
| ADE20k mIoU (linear, ViT-H/14) | mIoU | 36.9 / 36.6 | 39.5 | +2.6 |
| COCO det (ViT-H/14) | AP$^b$ | 53.7 / 53.5 | 55.2 | +1.5 |
| ADE20k mIoU (ViT-L/16, CC27M) | mIoU | – | 42.1 | New SOTA |
| vs SigLIP2 (ViT-L/16, ADE20k mIoU) | mIoU | 24.6 (Prev. SOTA) | 41.2 | +16.6 |

TC-JEPA's ADE20k mIoU on IN-21k surpasses DINOv2 (41.8) trained with 5× distilled data and Web-DINO (40.3) with 75× data. Using CC27M, it reaches 42.1, significantly better suited for dense tasks than CLIP/SigLIP given equivalent data.

### Ablation Study

| Configuration | IN-1k Top-1 / ADE20k mIoU | Description |
| :--- | :--- | :--- |
| Full TC-JEPA (ViT-L/16, IN-21k) | 82.1 / 41.2 | Full method |
| W/o sparse + consistency | Significant drop | Patch-word attention degrades; text modulation fails |
| Sequence conditioning | Weaker than cross-attn | Conditioning only at shallow layers; high overhead |
| Single caption ($N=1$) | Weaker than $N=8$ max-pool | Hard to cover all details; max-pool provides clear gains |
| I-JEPA baseline | 77.2 / 38.2 | No text conditioning |

### Key Findings
- Text conditioning provides significantly higher gains for dense tasks (segmentation, detection) than classification, indicating that reduced prediction uncertainty primarily improves local patch feature quality, addressing a weakness of contrastive methods like SigLIP.
- On IN-21k, TC-JEPA's ADE20k mIoU matches Franca (which combines invariance + MIM), proving that fine-grained text conditioning can replace invariance constraints from manual augmentations.
- TC-JEPA's scaling curve remains consistently above I-JEPA; while I-JEPA shows unclear scaling on IN-1k classification, TC-JEPA demonstrates that text signals are key to stable scaling.

## Highlights & Insights
- Placing text in the predictor instead of the encoder is a crucial pivot: the encoder is no longer compressed into global CLIP-style abstractions. Patch features retain visual details but become predictable latent variables under text prompts. Discarding text during inference allows for purely visual representations compatible with existing backbones.
- Using sparsity and consistency regularizations to drive cross-attention toward implicit visual grounding avoids strong reliance on grounding data. The idea of using auxiliary losses to drive attention for semantic alignment is generalizable.
- Multi-caption max-pool fusion is a practical trick: it avoids interference between multi-source captions at the same patch. Performing fusion in feature space rather than token space is low-cost and provides inherent sparse selection.

## Limitations & Future Work
- TC-JEPA requires 5–10 synthetic captions per image, making it sensitive to caption quality and quantity. The LMM cost for generating these captions in industrial deployments is non-negligible.
- Text conditioning is limited to the pre-training phase. Downstream inference cannot explicitly utilize text prompts for zero-shot retrieval/classification, thus Top-1 accuracy on IN-1k still lags behind specialized zero-shot contrastive methods.
- The additional computation for multi-layer cross-attention and multiple captions in the predictor is significant; training stability and costs when scaling to ViT-G levels need verification.
- The impact of biases or hallucinations in synthetic captions on representations was not explored in depth.

## Related Work & Insights
- **vs I-JEPA / StoP / CAPI**: All belong to latent MIM, but TC-JEPA shifts the "uncertainty" problem from architectural tricks (positional conditioning, random encoding) to "introducing new information sources (text)," leading to comprehensive leads.
- **vs CLIP / SigLIP series**: Also uses image-caption pairs, but TC-JEPA avoids contrastive loss, ensuring the feature space isn't flattened by global alignment, hence the large lead in dense tasks. The downside is the lack of direct zero-shot retrieval.
- **vs DINOv2 / iBOT / Franca**: These methods rely on "invariance + MIM" and require carefully designed augmentations. TC-JEPA suggests that language can serve as an alternative form of data augmentation.
- **vs SPARC / DreamLIP**: These use synthetic captions for fine-grained contrastive methods, but TC-JEPA's use of captions as predictor conditions is friendlier to dense tasks.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Injecting captions into the JEPA predictor is a natural yet previously underdeveloped direction. The combination of cross-attention, sparse consistency, and max-pool fusion is a clear contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 3 model scales, 3 data scales, and multiple tasks (classification/detection/segmentation) with systematic comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ Logical motivation and good integration of diagrams and formulas.
- **Value**: ⭐⭐⭐⭐ Opens a "weak text supervision" scaling path for JEPA methods, with high value for dense prediction and visual foundation models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Conditional Diffusion Sampling](conditional_diffusion_sampling.md)
- [\[ICML 2026\] CHARM: Using Multimodal JEPA + Channel Descriptions for Time Series Foundation Embedding](giving_sensors_a_voice_multimodal_jepa_for_semantic_time-series_embeddings.md)
- [\[AAAI 2026\] Conditional Information Bottleneck for Multimodal Fusion: Overcoming Shortcut Learning in Sarcasm Detection](../../AAAI2026/multimodal_vlm/conditional_information_bottleneck_for_multimodal_fusion_overcoming_shortcut_lea.md)
- [\[NeurIPS 2025\] Learning Shared Representations from Unpaired Data](../../NeurIPS2025/multimodal_vlm/learning_shared_representations_from_unpaired_data.md)
- [\[ICML 2026\] Learning GUI Grounding with Spatial Reasoning from Visual Feedback](learning_gui_grounding_with_spatial_reasoning_from_visual_feedback.md)

</div>

<!-- RELATED:END -->
