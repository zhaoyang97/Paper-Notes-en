---
title: >-
  [Paper Note] Parameter-efficient Prompt Tuning and Hierarchical Textual Guidance for Few-shot Whole Slide Image Classification
description: >-
  [CVPR 2026][Medical Imaging][Whole slide image classification] HIPSS introduces two key innovations for few-shot WSI classification: (1) parameter-efficient prompt tuning via Scaling and Shifting Features (SSF) as a repl…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "Whole slide image classification"
  - "few-shot learning"
  - "vision-language models"
  - "parameter-efficient fine-tuning"
  - "hierarchical textual guidance"
date: 2026-05-08
content_hash: 61e0f4691f3a9175
---

# Parameter-efficient Prompt Tuning and Hierarchical Textual Guidance for Few-shot Whole Slide Image Classification

**Conference**: CVPR 2026
**arXiv**: [2603.21504](https://arxiv.org/abs/2603.21504)  
**Code**: Unavailable  
**Area**: Medical Imaging / Few-shot Learning
**Keywords**: Whole slide image classification, few-shot learning, vision-language models, parameter-efficient fine-tuning, hierarchical textual guidance

## TL;DR

HIPSS introduces two key innovations for few-shot WSI classification: (1) parameter-efficient prompt tuning via Scaling and Shifting Features (SSF) as a replacement for CoOp, substantially reducing the number of trainable parameters; and (2) a soft hierarchical textual guidance strategy that exploits the pretrained knowledge of VLMs and the inherent hierarchical structure of WSIs without hard patch filtering. The method achieves up to 13.8% improvement across three cancer datasets.

## Background & Motivation

1. **Background**: WSI classification in computational pathology typically relies on multiple instance learning (MIL), which segments gigapixel-level WSIs into patches for weakly supervised training. However, MIL methods require large numbers of annotated WSIs and are thus limited in data-scarce medical settings.

2. **Limitations of Prior Work**:
    - Existing few-shot WSI classification (FSWC) methods employ CoOp for prompt tuning, which introduces a large number of trainable parameters and inference overhead, making them prone to overfitting under extreme few-shot settings.
    - Methods that fuse visual and textual features via cross-attention (e.g., FOCUS, ViLa-MIL) incur even greater parameter costs.
    - Existing approaches typically apply hard filtering to discard patches based on alignment scores with text embeddings—sometimes removing >60% of patches—potentially discarding diagnostically relevant information.

3. **Key Challenge**: VLMs are pretrained on instance-level image–text pairs, whereas downstream tasks require slide-level classification, creating a semantic gap between local instance features and global slide labels. Furthermore, WSIs possess an inherent spatial hierarchy (cells → regions → slides) that existing aggregation methods largely ignore.

4. **Goal**: (a) How can pretrained VLMs be adapted more effectively with fewer parameters? (b) How can the hierarchical spatial structure of WSIs and VLM-based textual guidance be jointly exploited for representation learning without discarding potentially diagnostic patches?

5. **Key Insight**: Drawing on the success of SSF (Scaling and Shifting Features) in visual encoders, this work transfers the paradigm to prompt tuning in text encoders. Simultaneously, a hierarchical attention aggregation mechanism is designed that uses cosine similarity with text embeddings as a soft weighting signal rather than a hard filter.

6. **Core Idea**: Replace CoOp's prompt vectors with a minimal set of scaling/shifting parameters in the text encoder, and substitute hard patch filtering with soft hierarchical textual guidance, achieving parameter-efficient and robust few-shot WSI classification.

## Method

### Overall Architecture

HIPSS comprises two core modules: SSF-based prompt tuning and hierarchical textual guidance for WSI representation learning. First, an LLM generates region-level and slide-level textual descriptions, which are concatenated after tokenization and fed into a text encoder equipped with SSF parameters to produce text embeddings $T$. The WSI is then partitioned hierarchically into regions and instances, and two-level attention pooling (a Region Encoder followed by a WSI Encoder) aggregates these features into a slide-level representation $F(X_i)$; attention weights at each level are softly guided by the text embeddings. The model is trained via contrastive learning.

### Key Designs

1. **SSF-based Prompt Tuning (Scaling and Shifting Features)**:

    - **Function**: Adapts the pretrained text encoder to downstream WSI classification with minimal parameters.
    - **Mechanism**: For selected layers of the text encoder, a pair of learnable scaling and shifting parameters $\gamma, \beta \in \mathbb{R}^{D_t}$ is attached to each layer's features $x \in \mathbb{R}^{D_t}$, applying the transformation $y = \gamma \cdot x + \beta$. Only $\gamma$ and $\beta$ are updated during training; pretrained weights remain frozen. A depth parameter $d_s$ controls the number of adapted layers, with SSF applied only to the last $d_s$ blocks. At inference, the transformation can be reparameterized and merged into the model weights, incurring zero additional inference cost.
    - **Design Motivation**: CoOp learns context vectors in prompt embedding space, resulting in high parameter counts and additional inference overhead. SSF requires only $2 \times D_t$ parameters per layer for distribution alignment, achieving greater efficiency without affecting inference speed.

2. **Hierarchical Textual Guidance for WSI Representation Learning**:

    - **Function**: Incorporates the spatial hierarchical structure of WSIs into representation learning, using text embeddings to softly guide attention weights.
    - **Mechanism**: WSIs are first divided into $4096 \times 4096$ regions, each of which is further partitioned into $256 \times 256$ instances. Two-level attention pooling is applied: the Region Encoder aggregates instance features within each region via attention pooling to obtain a region embedding $F(R_{i,m}) = \sum_j a_{i,m}^j h_{i,m}^j$; the WSI Encoder then aggregates all region embeddings into a slide-level representation. Crucially, a text-guided correction score $s_{i,m}^j$ is incorporated into the attention weights.
    - **Design Motivation**: The natural spatial hierarchy of WSIs is exploited—local aggregation within regions first captures cellular interactions, followed by global aggregation to capture tissue-level patterns—which is more effective than flattening all patches into a single pool.

3. **Soft Textual Guidance for Attention Correction**:

    - **Function**: Guides attention weights using text embeddings without discarding any patches.
    - **Mechanism**: The cosine similarity between each instance feature and the text embedding $T$ is computed and processed in three tiers: negative similarity is set to 0 (semantically inconsistent; suppressed); low positive similarity retains its linear weight (uncertain but potentially informative); similarity above a threshold $\alpha$ is amplified by a factor $\lambda$ (high-confidence semantic match). Formally: $s_{i,m}^j = \lambda \cdot \cos(h_{i,m}^j, T)$ when $\cos > \alpha$; $s_{i,m}^j = \cos(h_{i,m}^j, T)$ when $0 < \cos < \alpha$; and $s_{i,m}^j = 0$ otherwise.
    - **Design Motivation**: Hard filtering (e.g., FOCUS discards >60% of patches) risks losing diagnostic information not covered by text embeddings. Soft guidance retains all patches through reliability-aware gated weighting, yielding greater robustness in few-shot settings.

### Loss & Training

- A contrastive learning objective is used: $P(Y_i = c \mid X_i) = \frac{\exp(\cos(F(X_i), T_c) / \tau)}{\sum_j \exp(\cos(F(X_i), T_j) / \tau)}$
- Cross-entropy loss is computed between the predicted probabilities and ground-truth slide labels.
- The VLM image encoder is fully frozen; only SSF parameters in the text encoder are trained.
- LLM-generated class-specific descriptions serve as text prompts, with two query types: region-level and slide-level.

## Key Experimental Results

### Main Results

AUC on Camelyon16 (breast cancer):

| Method | 16-shot | 8-shot | 4-shot | 2-shot | 1-shot |
|--------|---------|--------|--------|--------|--------|
| ABMIL | 0.782 | 0.720 | 0.609 | 0.597 | 0.571 |
| TOP (NeurIPS'23) | 0.835 | 0.730 | 0.728 | 0.705 | 0.685 |
| FOCUS (CVPR'25) | 0.840 | 0.797 | 0.637 | 0.568 | 0.527 |
| **HIPSS (Ours)** | **0.911 (+8.4)** | **0.859 (+7.7)** | **0.816 (+10.9)** | **0.807 (+6.2)** | **0.743 (+7.2)** |

AUC on UBC-OCEAN (ovarian cancer) under the 1-shot setting: HIPSS 0.854 vs. FOCUS 0.751, a **gain of 13.8%**.

### Ablation Study

Component contributions on Camelyon16:

| H (Hierarchical) | T (Text Guidance) | S (SSF Tuning) | 16-shot | 4-shot | 1-shot |
|:-:|:-:|:-:|---------|--------|--------|
| ✗ | ✗ | ✗ | 0.782 | 0.609 | 0.571 |
| ✓ | ✗ | ✗ | 0.867 | 0.690 | 0.643 |
| ✓ | ✓ | ✗ | 0.872 | 0.755 | 0.651 |
| ✗ | ✗ | ✓ | 0.885 | 0.780 | 0.707 |
| ✓ | ✓ | ✓ | **0.911** | **0.816** | **0.743** |

### Key Findings

- **SSF contributes most**: Adding SSF alone (without hierarchical structure or text guidance) improves performance from 0.782 to 0.885 (16-shot), indicating that text encoder adaptation is the primary bottleneck.
- **Parameter efficiency**: HIPSS reduces trainable parameters by 18.1% on breast and lung cancer datasets, and by 5.8% on the ovarian cancer dataset.
- **Advantages grow under more extreme few-shot settings**: The largest gains occur at 1-shot (ovarian cancer, +13.8%), as methods relying on hard filtering are more likely to discard critical information when training data is extremely scarce.
- **Soft textual guidance is more stable than hard filtering**: FOCUS exhibits sharp performance drops at 4-shot and 2-shot (e.g., Camelyon16 from 0.840 to 0.568), whereas HIPSS degrades gracefully.

## Highlights & Insights

- **First application of SSF to text encoder prompt tuning**: Two vectors per layer ($\gamma, \beta$) suffice to effectively align pretrained feature distributions to the target task, and reparameterization eliminates inference overhead. This paradigm is broadly transferable to other VLM adaptation scenarios.
- **Philosophy of soft guidance over hard filtering**: In few-shot settings where information is extremely scarce, retaining all information with tiered weighting is more robust than aggressive discarding. The three-tier design (suppress / retain / amplify) balances noise filtering with information preservation.
- **Hierarchical structure exploits WSI's intrinsic properties**: The two-level partitioning from $4096 \times 4096$ regions to $256 \times 256$ instances naturally corresponds to the pathological hierarchy from tissue to cell, constituting an effective domain-specific inductive bias.

## Limitations & Future Work

- The hierarchical partition sizes ($4096 \times 4096 \rightarrow 256 \times 256$) are fixed; different magnification levels or tissue types may require alternative partitioning strategies.
- The SSF tuning depth $d_s$ requires hyperparameter search, and the optimal value may vary across datasets.
- Evaluation is limited to binary classification (breast and lung cancer) and multi-class classification (ovarian cancer); more complex multi-label or grading tasks remain untested.
- Directions for future work include: adaptive learning of partition sizes; incorporating graph neural networks to model spatial relationships among regions; and extending the framework to fully supervised WSI classification to validate generalizability.

## Related Work & Insights

- **vs. TOP (NeurIPS'23)**: TOP employs CoOp-based instance- and bag-level prompt learning, requiring domain experts to design prompts; HIPSS's SSF is simpler and uses fewer parameters.
- **vs. FOCUS (CVPR'25)**: FOCUS applies adaptive visual token compression to reduce computational overhead, but discards potentially valuable regions; HIPSS's soft guidance retains all information.
- **vs. ViLa-MIL (CVPR'24)**: ViLa-MIL fuses multi-scale features via cross-attention, incurring high parameter counts and severe overfitting under extreme few-shot conditions (1-shot AUC: 0.457 vs. HIPSS 0.743).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Applying SSF to text encoder prompt tuning is a novel transfer; the soft hierarchical guidance design is elegant, though innovations are primarily at the component level.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across three datasets and five shot settings, thorough ablation, and additional validation via tumor localization.
- **Writing Quality**: ⭐⭐⭐⭐ — Technical descriptions are detailed and clear, mathematical derivations are complete, and motivations are well-argued.
- **Value**: ⭐⭐⭐⭐ — Strong practical utility for few-shot medical imaging; the combination of parameter reduction and performance gain is meaningful for resource-constrained environments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MUSE: Harnessing Precise and Diverse Semantics for Few-Shot Whole Slide Image Classification](muse_harnessing_precise_and_diverse_semantics_for_few-shot_whole_slide_image_cla.md)
- [\[CVPR 2026\] Act Like a Pathologist: Tissue-Aware Whole Slide Image Reasoning](act_like_a_pathologist_tissue-aware_whole_slide_image_reasoning.md)
- [\[CVPR 2026\] SD-FSMIS: Adapting Stable Diffusion for Few-Shot Medical Image Segmentation](sd_fsmis_adapting_stable_diffusion_for_few_shot_medical_image_segmentation.md)
- [\[CVPR 2026\] Unlocking Positive Transfer in Incrementally Learning Surgical Instruments: A Self-reflection Hierarchical Prompt Framework](unlocking_positive_transfer_in_incrementally_learning_surgical_instruments_a_sel.md)
- [\[CVPR 2026\] Interpretable Cross-Domain Few-Shot Learning with Rectified Target-Domain Local Alignment](interpretable_cross-domain_few-shot_learning_with_rectified_target-domain_local_.md)

</div>

<!-- RELATED:END -->
