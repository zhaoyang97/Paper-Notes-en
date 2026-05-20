---
title: >-
  [Paper Note] Index-Preserving Lightweight Token Pruning for Efficient Document Understanding
description: >-
  [ICLR 2026 (Workshop on MM Intelligence)][Multimodal VLM][token pruning] A binary patch classifier with only 203K parameters is inserted before the VLM visual encoder to remove background tokens from document images. A $…
tags:
  - "ICLR 2026 (Workshop on MM Intelligence)"
  - "Multimodal VLM"
  - "token pruning"
  - "document understanding"
  - "VLM efficiency"
  - "patch classifier"
  - "index preservation"
date: 2026-05-08
content_hash: 05b9357e15fd4518
---

# Index-Preserving Lightweight Token Pruning for Efficient Document Understanding

**Conference**: ICLR 2026 (Workshop on MM Intelligence)
**arXiv**: [2509.06415](https://arxiv.org/abs/2509.06415)  
**Code**: [GitHub](https://github.com/jaeminSon/index-preserving-lightweight-token-pruning)  
**Area**: Multimodal VLM / Document Understanding
**Keywords**: token pruning, document understanding, VLM efficiency, patch classifier, index preservation

## TL;DR

A binary patch classifier with only 203K parameters is inserted before the VLM visual encoder to remove background tokens from document images. A $3 \times 3$ max-pooling operation is then applied to recover fragmented text regions while preserving original spatial indices, achieving 40–60% FLOPs reduction on Qwen2.5-VL with accuracy degradation of no more than ~5 percentage points.

## Background & Motivation

**Background**: VLMs such as Qwen2.5-VL, Gemini, and LLaMA-3 have achieved strong performance on tasks including document parsing, key information extraction, and document VQA. Nevertheless, inference over high-resolution document images remains computationally expensive. A single A4 scanned document at 300 DPI measures 2481×3507 pixels, producing a large number of visual tokens after ViT patch splitting, all of which are fed into the LLM decoder and cause severe computational redundancy.

**Limitations of Prior Work**: Existing token compression methods fall into two categories: (1) attention-driven pruning methods (DynamicViT, FocusDETR, SPViT) that dynamically select tokens to retain within intermediate transformer layers, requiring architectural modifications or retraining; and (2) token merging methods (ToMe, Token Fusion) that merge neighboring tokens via cosine similarity at each layer, but in doing so disrupt the token index ordering. Both categories have been validated on classification and detection tasks but remain largely unexplored for document understanding.

**Key Challenge**: Document understanding tasks are highly sensitive to spatial position information—the row-column alignment of tables and the contextual positions of text paragraphs are all encoded in token position indices. When generic token merging or pruning methods destroy these indices, text recognition performance degrades catastrophically. However, document images possess a unique advantage: large proportions of blank margins and background regions (experiments show that on average 41.6%–65.7% of patches are background), which naturally facilitates aggressive early-stage pruning.

**Key Insight**: The authors observe that distinguishing text from background in documents is straightforward—a shallow classifier achieves an AP of 0.99—enabling pruning to be performed before the visual encoder and thus reducing computation across all subsequent modules from the source. The key insight is that the position indices of retained patches in the original grid must be preserved; otherwise, the positional encodings of the VLM become misaligned.

**Core Idea**: A lightweight binary patch classifier combined with max-pooling spatial refinement and an index-preservation strategy is applied as a preprocessing module, achieving efficient token pruning for document scenarios without modifying any VLM parameters.

## Method

### Overall Architecture

The overall pipeline consists of three modules connected in series, inserted between the input image and the frozen off-the-shelf VLM:

1. **Binary Text-Region Classifier**: Predicts a binary "text/background" label for each image patch.
2. **Max-Pooling Foreground Refinement**: Applies $3 \times 3$ max-pooling to the binary mask output by the classifier, recovering fragmented text regions that were incorrectly classified as background.
3. **Index-Preserving Token Pruning**: Feeds only foreground-labeled patches into the frozen VLM (visual encoder + language decoder), together with their original position indices.

Note: The entire framework requires no modification or retraining of the VLM; the classifier serves as a plug-and-play preprocessing module.

### Key Designs

1. **Lightweight Binary Patch Classifier**:

    - **Function**: Performs binary text/background classification for each square patch of the input image.
    - **Mechanism**: The classifier consists of Layer Norm → MLP → GELU → Classification Head, totaling only 203K parameters. Each patch is classified independently; patches with logit > 0 are retained as text regions. Training data consists of 800 OCR document images from AI-Hub, with text bounding boxes extracted via PSENet for automatic annotation—patches overlapping with text boxes are treated as positive samples and the remainder as negative, yielding approximately 99,600 training samples and 400 validation samples with balanced class distribution.
    - **Design Motivation**: Unlike DynamicViT and similar methods that prune within intermediate transformer layers, this approach completes filtering before the visual encoder, maximizing computational savings. The visual distinction between text and background in documents is highly pronounced: at patch size 28, the classifier achieves an AP of 0.99, demonstrating that a minimal classifier suffices.

2. **Max-Pooling Foreground Refinement**:

    - **Function**: Repairs fragmented foreground masks produced by the classifier, recovering missed text regions.
    - **Mechanism**: A $3 \times 3$ max-pooling operation is applied to the binary classification mask. If any patch within the $3 \times 3$ neighborhood of a background-labeled patch is labeled as foreground, that patch is restored to foreground. This is equivalent to a morphological dilation of the foreground regions.
    - **Design Motivation**: The patch-level classifier does not exploit spatial context and tends to misclassify certain patches within text lines as background. Experiments show that pruning alone causes ANLS to drop by 10–17 percentage points on document parsing and F1 to drop by 14–31 percentage points on key information extraction, whereas adding max-pooling reduces this loss to 0–5 percentage points. The trade-off is a reduction in pruning rate from 65.7% to 41.6%, which represents a worthwhile accuracy-efficiency balance.

3. **Index Preservation Strategy**:

    - **Function**: Ensures that pruned tokens retain correct spatial position information.
    - **Mechanism**: Each patch has a unique index $i \in \{0, 1, \ldots, n-1\}$ in the original image grid. After pruning, only the retained subset of patches is passed forward, but each token remains bound to its original index. The VLM's positional encoding (e.g., position embeddings in Qwen2.5-VL) is computed from these original indices rather than from renumbered sequential indices.
    - **Design Motivation**: The ablation study (Table 3) provides decisive evidence—replacing indices with a constant (all zeros) causes ANLS on the Scan task to plummet from 61.8 to 9.1; even using ordered sequential indices recovers only to 36.2. This demonstrates that positional semantics in document understanding are entirely dependent on original indices.

### Loss & Training

- The classifier is trained with standard binary cross-entropy (BCE) loss.
- Patch size is set to 28×28 pixels (based on ablation: size 14 is too small to capture complete character structure, while sizes 28/56/112 yield similar performance but 28 requires fewest parameters).
- Qwen2.5-VL is kept fully frozen; inference uses BFloat16 precision with FlashAttention-2 acceleration.
- Maximum generation length is 2048 tokens.

## Key Experimental Results

### Main Results: Effect of Pruning + Max-Pooling

Evaluation on the CC-OCR dataset using Qwen2.5-VL (3B/7B/32B/72B), covering document parsing (Scan/Photo) and key information extraction (CORD/SROIE):

| Model | Method | Scan (ANLS) | Photo (ANLS) | CORD (F1) | CORD (Acc) | SROIE (F1) | SROIE (Acc) |
|-------|--------|-------------|--------------|-----------|------------|------------|-------------|
| 3B | Original | 62.4 | 73.7 | 87.2 | 94.7 | 88.7 | 97.5 |
| 3B | Pruned (∆) | -13.6 | -13.0 | -30.8 | -17.1 | -15.4 | -2.7 |
| 3B | Pruned+MaxPool (∆) | -0.6 | -2.7 | -4.2 | -4.4 | -0.8 | +0.3 |
| 7B | Original | 64.7 | 69.9 | 89.5 | 96.3 | 90.7 | 98.1 |
| 7B | Pruned (∆) | -11.0 | -10.2 | -24.7 | -15.9 | -13.7 | -1.9 |
| 7B | Pruned+MaxPool (∆) | -2.7 | -1.3 | -5.3 | -4.8 | -0.4 | +0.1 |
| 32B | Original | 60.9 | 67.5 | 87.0 | 95.2 | 88.5 | 97.5 |
| 32B | Pruned+MaxPool (∆) | +1.7 | -1.9 | -3.3 | -4.0 | -0.0 | +0.1 |
| 72B | Original | 67.7 | 70.0 | 92.8 | 97.6 | 91.6 | 98.6 |
| 72B | Pruned+MaxPool (∆) | -0.4 | +1.5 | -3.8 | -4.0 | -0.5 | -0.1 |

Key observations: Pruning alone causes CORD F1 to drop by 24–31 percentage points; adding max-pooling reduces the loss to only 3–5 percentage points. SROIE benefits more due to its larger proportion of background regions. The 32B model even shows a +1.7 percentage point gain on Scan, suggesting that removing background noise can sometimes be beneficial.

### Comparison with Prior Methods (Qwen2.5-VL-3B)

| Method | Scan (ANLS) | Photo (ANLS) | CORD (F1 / Acc) | SROIE (F1 / Acc) |
|--------|-------------|--------------|-----------------|------------------|
| Original | 62.4 | 73.7 | 87.2 / 94.7 | 88.7 / 97.5 |
| ToMe | 8.8 | 11.1 | 6.0 / 13.5 | 0.0 / 9.9 |
| DocKylin (DTS) | 34.3 | 47.7 | 73.1 / 84.1 | 69.6 / 83.5 |
| **Ours** | **61.8** | **71.0** | **83.0 / 90.3** | **87.9 / 97.8** |

ToMe disrupts the token index structure through per-layer merging, reducing ANLS and F1 to near zero. DocKylin's DTS module assumes that tokens highly correlated with others correspond to background, but this assumption does not hold for documents where repetitive text patterns can also exhibit high correlation. The proposed method substantially outperforms both baselines.

### Ablation Study: Effect of Index Strategy (Qwen2.5-VL-3B)

| Index Strategy | Scan (ANLS) | Photo (ANLS) | CORD (F1 / Acc) | SROIE (F1 / Acc) |
|----------------|-------------|--------------|-----------------|------------------|
| Constant (all zeros) | 9.1 | 5.8 | 3.5 / 19.7 | 0.0 / 10.9 |
| Random | 16.0 | 13.7 | 8.5 / 27.9 | 0.2 / 11.7 |
| Ordered (sequential) | 36.2 | 49.2 | 38.8 / 58.0 | 40.7 / 65.1 |
| **Preserved (original)** | **61.8** | **71.0** | **83.0 / 90.3** | **87.9 / 97.8** |

This is the most critical ablation in the paper: performance increases monotonically from constant → random → ordered → preserved. The preserved strategy outperforms ordered by 44.2 percentage points on CORD F1 and 47.2 percentage points on SROIE F1, conclusively demonstrating the indispensability of position indices in document understanding.

### Computational Efficiency

- Pruning alone reduces visual tokens by an average of 65.7%; with max-pooling, the reduction is 41.6%.
- End-to-end TFLOPs reduction with pruning+maxpool is 40–60% across all datasets, reaching ~80% on SROIE due to the high background proportion in receipt images.
- The classifier contributes only 203K parameters (at patch size 28) and introduces negligible overhead to total inference time.

## Highlights & Insights

1. **Value of Pre-Encoder Pruning**: Unlike token compression performed within intermediate ViT layers, this work completes pruning before the visual encoder, ensuring that the entire subsequent pipeline—ViT encoding and LLM decoding—benefits from the reduced token count, maximizing efficiency gains.
2. **Index Correctness Outweighs Token Count**: The ablation study provides empirical evidence that in document understanding, the correctness of position indices matters more than the number of tokens. The failure of ToMe (ANLS: 62.4→8.8) serves as a compelling negative example. This finding carries important implications for all token compression work targeting document scenarios.
3. **Max-Pooling as a Morphological Repair Tool**: A simple $3 \times 3$ max-pooling operation is sufficient to "dilate" fragmented text regions back into coherent foreground, recovering CORD F1 loss from -30.8 to -4.2 percentage points. This is considerably more elegant than designing complex spatial consistency constraints.

## Limitations & Future Work

1. **Validation limited to document scenarios**: The method relies heavily on the assumption that documents contain abundant blank background; densely typeset materials (newspapers, magazines) or natural image scenes offer limited pruning headroom.
2. **Classifier considers only local patches**: Independent classification of 28×28 patches cannot leverage global context, potentially causing missed detections for small-font or low-contrast text.
3. **Validated only on Qwen2.5-VL**: This model supports variable-length token inputs, but VLMs with fixed-length token representations (e.g., PaliGemma) may not be directly compatible.
4. **Limited experimental scope as a workshop paper**: Evaluation is restricted to a single benchmark (CC-OCR with 4 subsets); broader document understanding benchmarks such as DocVQA and InfographicVQA are not covered.
5. **Fixed max-pooling kernel size of 3×3**: Documents of varying text density may require different kernel sizes; adaptive strategies warrant further exploration.
6. **Evaluation limited to English documents**: The generalizability of the classifier to documents with complex character structures such as CJK scripts has not been verified.

## Related Work & Insights

- **vs. ToMe**: ToMe merges tokens via cosine similarity at each ViT layer, which is suitable for classification tasks but completely destroys the spatial index structure of documents, rendering it nearly unusable for document understanding (ANLS ≈ 9). The key lesson from this paper is that document scenarios require strictly index-preserving operations.
- **vs. DocKylin**: DocKylin performs token merging in the language embedding space and uses Sobel filtering to remove white backgrounds. However, its DTS module's assumption that "high correlation implies background" does not hold for documents. The proposed direct binary classifier with pre-encoder pruning substantially outperforms DocKylin.
- **vs. DynamicViT / SPViT**: These methods insert trainable pruning modules at intermediate transformer layers and require end-to-end training. The proposed classifier is trained independently with the VLM kept fully frozen, making deployment considerably simpler.
- The proposed method can be combined in series with internal token compression methods: an external classifier first performs coarse pruning (removing 40–60% of background), followed by fine-grained token merging within the ViT, potentially yielding further compression.

## Rating

- **Novelty**: ⭐⭐⭐ The individual components are straightforward, but the combination of pre-encoder pruning and index preservation represents the first systematic validation in document scenarios.
- **Experimental Thoroughness**: ⭐⭐⭐ Covers 4 model scales and 4 datasets with well-designed ablations, though benchmark coverage is relatively narrow.
- **Writing Quality**: ⭐⭐⭐⭐ Concise and clear within workshop paper constraints; tables and figures convey sufficient information.
- **Value**: ⭐⭐⭐⭐ The ablation findings on index preservation provide strong guidance for the broader field of efficient document VLM optimization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] When Token Pruning is Worse than Random: Understanding Visual Token Information in VLLMs](../../CVPR2026/multimodal_vlm/when_token_pruning_is_worse_than_random_understanding_visual_token_information_i.md)
- [\[CVPR 2026\] Efficient Document Parsing via Parallel Token Prediction](../../CVPR2026/multimodal_vlm/efficient_document_parsing_via_parallel_token_prediction.md)
- [\[CVPR 2026\] VLM-Pruner: Buffering for Spatial Sparsity in an Efficient VLM Centrifugal Token Pruning Paradigm](../../CVPR2026/multimodal_vlm/vlm-pruner_buffering_for_spatial_sparsity_in_an_efficient_vlm_centrifugal_token_.md)
- [\[ACL 2026\] HiPrune: Hierarchical Attention for Efficient Token Pruning in Vision-Language Models](../../ACL2026/multimodal_vlm/hiprune_hierarchical_attention_for_efficient_token_pruning_in_vision-language_mo.md)
- [\[AAAI 2026\] URaG: Unified Retrieval and Generation in Multimodal LLMs for Efficient Long Document Understanding](../../AAAI2026/multimodal_vlm/urag_unified_retrieval_and_generation_in_multimodal_llms_for.md)

</div>

<!-- RELATED:END -->
