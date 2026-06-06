---
title: >-
  [Paper Note] DenseMLLM: Standard Multimodal LLMs for Dense Prediction
description: >-
  [ICML 2026][Multimodal VLM][Dense prediction] The authors integrate dense prediction tasks such as semantic segmentation, depth estimation…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Dense prediction"
  - "Multimodal LLM"
  - "vision token supervision"
  - "multi-label NTP"
  - "unified architecture"
date: 2026-05-08
content_hash: c260ffadd5779521
---

# DenseMLLM: Standard Multimodal LLMs for Dense Prediction

**Conference**: ICML 2026  
**arXiv**: [2602.14134](https://arxiv.org/abs/2602.14134)  
**Code**: https://github.com/Eli-YiLi/DenseMLLM (Available)  
**Area**: Multimodal VLM  
**Keywords**: Dense prediction, Multimodal LLM, vision token supervision, multi-label NTP, unified architecture

## TL;DR
The authors integrate dense prediction tasks such as semantic segmentation, depth estimation, and referring expression segmentation directly into a standard 4B MLLM (ViT + Projector + LLM). By eliminating task-specific decoders and introducing "Multi-label Next-Token Prediction" (NTP-M) supervision for vision tokens, the model achieves 54.2 mIoU on ADE20K, 87.6 $\delta_1$ on DDAD, and 80.7 cIoU on RefCOCO val, while maintaining general VL performance on par with Qwen3-VL-4B.

## Background & Motivation

**Background**: Current mainstream MLLMs utilize a "ViT + Projector + LLM" trinity to unifiedly handle tasks like VQA, OCR, and grounding. However, for dense prediction tasks requiring pixel-level outputs (e.g., semantic segmentation, depth estimation, referring expression segmentation), almost all existing solutions append task-specific decoders—GLaMM/UniPixel use SAM mask decoders, UFO introduces mask retrieval embeddings, and VisionLLM employs a Deformable-DETR head.

**Limitations of Prior Work**: These add-on designs fragment the architecture, requiring new modules for every new task, which contradicts the MLLM philosophy of "unifying all tasks with a next-token interface." A few attempts at pure text output (e.g., DepthLM's point-wise sampling or VisionLLM's polygon coordinates) suffer from either extreme inference costs or unsatisfactory accuracy.

**Key Challenge**: The standard MLLM training objective only calculates NTP loss for text tokens. Vision tokens rely on indirect global image-text alignment, resulting in a lack of fine-grained pixel semantics in the final layer. To perform dense prediction without additional decoders, direct supervision of vision token output probabilities is essential.

**Goal**: To enable a completely standard MLLM (no architectural changes, no extra heads, no retrieval) to directly derive pixel-level segmentation and depth maps from the argmax of vision token logits.

**Key Insight**: The authors observe a fundamental difference between vision and text tokens: a vision token corresponds to an image patch that may simultaneously contain multiple semantic labels (e.g., "dog," "chair," "background," "depth bin 20"); in contrast, a text token always corresponds to a single vocabulary ID. Therefore, single-label softmax NTP is inherently unsuitable for vision tokens.

**Core Idea**: Extend NTP from "single-label softmax" to "multi-label sigmoid (independent Bernoulli distribution) + relevant negative sampling." This allows vision token logits of a standard LLM to handle both classification and localization. During inference, results are obtained by performing argmax over the target category vocabulary.

## Method

### Overall Architecture
DenseMLLM consists of three standard components: SigLIP-2 (siglip2-so400m-patch16-naflex) as the vision encoder, a $2\times 2$ spatial-merge + two-layer MLP as the projector, and a standard transformer LLM with 4B parameters. The input image is patched and encoded into a vision token sequence. The prompt (e.g., "Segment: dog, chair.") is tokenized and fed into the LLM alongside the vision tokens. The final layer of the LLM outputs a vocabulary-dimension logit vector $Z_i \in \mathbb{R}^{|V|}$ for each vision token. Dense predictions are derived directly by applying argmax to these logits, bypassing any mask decoders.

### Key Designs

1.  **Decoding Dense Predictions from Vision Token Logits**:
    - **Function**: Treats the vision token logits from the LLM's final layer as a classification map and uses unified vocabulary IDs to index target category scores for pixel-level output via argmax.
    - **Mechanism**: For semantic segmentation, the model first identifies the set of categories $\{k\}$ present in the image via text NTP. It then extracts the subset of token IDs $S_k$ corresponding to these categories from the vision token logits $Z$, averaging the multi-token scores as $\hat Z_k = \frac{1}{|S_k|}\sum_{v\in S_k} Z_v$. $\hat Z$ is reshaped into an $H\times W$ patch grid and bilinearly upsampled to the original resolution, resulting in the predicted map $M = A(I(R(\hat Z)))$. For depth estimation, depth ranges are linearly or logarithmically discretized into 1–1000 bins, each mapped to a `<custom k>` vocabulary ID, following the same argmax workflow.
    - **Design Motivation**: Vision tokens processed through multiple LLM transformer layers already incorporate global context and instruction information. The logits are already within the semantic space; they merely lack an interface to be "exposed." This approach allows a 4B model to achieve 87.6 $\delta_1$ on DDAD depth estimation in a single inference pass, whereas DepthLM requires separate inference for each sampling point.

2.  **Multi-label Next-Token Prediction (NTP-M)**:
    - **Function**: Replaces the single-label softmax of traditional NTP with a multi-label sigmoid, allowing a single vision token to simultaneously contribute to multiple supervision signals (e.g., "dog," "chair," and "depth bin 20").
    - **Mechanism**: A multi-hot vector $y_{i,v}\in\{0,1\}$ is constructed, where all object categories, depth bins, and foreground/background labels spatially related to the $i$-th vision token are set to 1. The joint probability is modeled using independent Bernoulli distributions: $p(Y|X_v, X_{\text{instruct}}) = \prod_{i,v} \sigma(Z_{i,v})^{y_{i,v}}(1-\sigma(Z_{i,v}))^{1-y_{i,v}}$. Prompts for different tasks (e.g., "Estimate depth using NYUv2 ranges") control which vocabulary segment is activated via $X_{\text{instruct}}$.
    - **Design Motivation**: The multi-label nature of vision tokens means the mutual exclusivity assumption of softmax causes conflict; sigmoid allows multiple semantics to coexist. It remains fully compatible with the existing NTP framework on text tokens without requiring new loss branches.

3.  **Relevant Negative Sampling**:
    - **Function**: Addresses the extreme imbalance between positive and negative samples caused by the massive MLLM vocabulary (hundreds of thousands of entries).
    - **Mechanism**: For each vision token, let the valid index set be $M_i$, the positive sample set be $P_i = \{v\in M_i \mid y_{i,v}=1\}$, and the negative candidate set be $C_i = \{v\in M_i \mid y_{i,v}=0\}$. The top-$k$ high-scoring negative samples based on predicted probabilities $p_{i,v}=\sigma(Z_{i,v})$ are selected as the "relevant negative set" $N_i^{\text{relev}}$. The final loss is defined as $L_{\text{NTP-M}} = \sum_i \left[ -\frac{1}{|P_i|}\sum_{v\in P_i} \log p_{i,v} -\frac{1}{k}\sum_{v\in N_i^{\text{relev}}} \log(1-p_{i,v}) \right]$. Independent averaging of positives and negatives prevents gradients from being diluted by the vast number of irrelevant negatives.
    - **Design Motivation**: While standard OHEM selects hard examples in the spatial dimension, the vocabulary dimension exhibits a sparse distribution. Sampling relevant negatives along the vocabulary dimension is crucial; ablation studies show that moving from BCE to independent averaging and finally to relevant negative sampling improves ADE20K mIoU from 16.7 to 32.7 to 51.2, with this step alone accounting for 70% of the total gain.

### Loss & Training
The model uses a four-stage training strategy:
1.  **Stage I**: Multimodal foundation pre-training (mixed language and vision training, with vision codebook supervision on the vocabulary).
2.  **Stage II**: Annealing stage, involving high-quality fine-tuning for dense prediction tasks while mixing in VQA/OCR to maintain general capabilities.
3.  **Stage III**: SFT with the context window expanded from 16K to 32K.
4.  **Stage IV**: RL in the DAPO style, introducing class-label IoU rewards for segmentation. This stage removes the KL penalty and utilizes FP16 to ensure convergence.

## Key Experimental Results

### Main Results
DenseMLLM-4B handles three major dense prediction tasks using a single standard architecture and remains competitive against methods with task-specific decoders:

| Dataset / Task | Metric | DenseMLLM-4B | Prev. SOTA | Remarks |
|---|---|---|---|---|
| ADE20K Segmentation | mIoU | 54.2 | VisionLLM-v2 52.3 (with Deform-DETR) | No decoder |
| Cityscapes Segmentation | mIoU | 70.4 | X-Decoder 81.7 (Expert model) | General model |
| NYUv2 Depth | $\delta_1$ | 90.4 | DepthLM 86.8 (Point-wise inference) | Single inference |
| DDAD Depth | $\delta_1$ | 87.6 | DepthLM 74.7 | Single inference |
| RefCOCO val | cIoU | 80.7 | UniPixel 80.5 (with SAM) / UFO 80.0 (retrieval) | No decoder |
| RefCOCO+ val | cIoU | 76.2 | UniPixel 74.3 | No decoder |

Regarding general VL capabilities, DenseMLLM-4B performs comparably or slightly better than Qwen3-VL-4B and InternVL-3.5-4B across 15 benchmarks (e.g., MMStar 71.1 vs 69.8, MME 2384 vs 2309), demonstrating that dense prediction does not sacrifice general performance.

### Ablation Study
Incremental component additions on ADE20K mIoU highlight NTP-M and relevant negative sampling as the primary contributors:

| Configuration | BCE | Indiv. Mean | Rel. Sampling | Data Scale | RL | ADE20K mIoU |
|---|---|---|---|---|---|---|
| Base (BCE) | ✓ |  |  |  |  | 16.7 |
| + Indiv. Mean | ✓ | ✓ |  |  |  | 32.7 |
| + Rel. Sampling | ✓ | ✓ | ✓ |  |  | 51.2 |
| + Data Scaling | ✓ | ✓ | ✓ | ✓ |  | 52.3 |
| + RL (Stage IV) | ✓ | ✓ | ✓ | ✓ | ✓ | 54.2 |
| + Single-dataset FT| ✓ | ✓ | ✓ | ✓ | ✓ | 55.2 |

### Key Findings
- The combination of "independent positive/negative averaging + relevant negative sampling" is what makes dense prediction work: it raises mIoU from 16.7 (pure BCE) to 51.2, a massive 34.5-point jump. Subsequent data scaling and RL provide marginal gains of 1.1–1.9 points.
- A standard MLLM without task-specific decoders, powered by vision token logits, can outperform UniPixel (which uses a SAM decoder) on RefCOCO (80.7 vs 80.5). This suggests vision tokens already contain sufficient fine-grained information; the key is providing the correct supervision signal.
- Depth estimation requires only a single forward pass to obtain a full dense depth map. This is both more efficient and accurate than DepthLM's point-wise querying, outperforming it by 12.9 $\delta_1$ points on DDAD.

## Highlights & Insights
- **Extending NTP from 1D text to 2D vision**: The authors identify a neglected fact—vision tokens differ semantically from text tokens as they are naturally multi-label. This insight led to NTP-M, allowing standard MLLMs to perform dense prediction without task-specific decoders for the first time. This logic is applicable to any token-grid output task, such as spatio-temporal video tokens or 3D voxel tokens.
- **Vocabulary-dimension relevant negative sampling**: Traditional OHEM selects hard examples in the spatial dimension, but LLM vocabularies are sparse and multi-hot. Sampling relevant negatives along the vocabulary dimension significantly boosts training efficiency (16.7 → 51.2 mIoU). This trick is directly applicable to any "large vocabulary + multi-label" scenario.
- **Training interface equivalent to NTP**: NTP-M simply replaces softmax with sigmoid plus selective averaging. All LLM training frameworks can adopt this with zero modifications. The RL stage also integrates seamlessly with DAPO using class-label IoU as a reward.

## Limitations & Future Work
- Instance and panoptic segmentation are not fully discussed. While RefCOCO covers single-instance referring, distinguishing between different instances of the same class (e.g., "segment all people") using multi-label NTP-M remains an open question.
- Depth estimation relies on discrete bins (`<custom 0>`–`<custom 999>`), limiting resolution by vocabulary size; continuous regression for metric depth would require new interfaces.
- The four-stage training and the use of FP16 with KL removal during RL imply high reproduction costs. Comprehensive comparisons between pure SFT and SOTA without RL are limited.
- The selection of $k$ for relevant negative sampling and whether it should vary by task (segmentation vs. depth) warrants further ablation.

## Related Work & Insights
- **vs. GLaMM / UniPixel**: These models append SAM-based mask decoders for referring segmentation. Ours matches or exceeds their performance on RefCOCO without any decoder, proving "add-ons" are not strictly necessary.
- **vs. UFO**: UFO uses mask retrieval embeddings, which still requires an additional retrieval process. Ours takes a more thorough "vision tokens as pixel maps" approach with a simpler architecture.
- **vs. DepthLM**: DepthLM queries depth point-wise through prompts, requiring $N$ inferences. Ours generates full dense maps in one pass and is 12.9 $\delta_1$ points higher on DDAD, showing token-grid supervision is superior to point-wise prompt regression.
- **vs. VisionLLM**: VisionLLM outputs polygon coordinates, which are limited by coordinate precision (74.5 cIoU on RefCOCO). Ours uses vision token logits for pixel-level output, offering a higher precision ceiling.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Realizes the first extension of NTP to multi-label vision token scenarios with relevant negative sampling; a clean, unified interface for dense prediction + MLLM.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 3 major dense tasks + 15 general VL benchmarks with clear ablations; however, instance/panoptic segmentation is missing, and the scale is limited to 4B.
- Writing Quality: ⭐⭐⭐⭐ Strong logic from observation to method; visual aids like PCA and multi-label histograms are intuitive.
- Value: ⭐⭐⭐⭐⭐ Successfully proves that standard MLLMs can perform dense prediction without decoders, significantly advancing unified multimodal architectures.

## Rating
- Novelty: To be evaluated
- Experimental Thoroughness: To be evaluated
- Writing Quality: To be evaluated
- Value: To be evaluated

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Efficient Document Parsing via Parallel Token Prediction](../../CVPR2026/multimodal_vlm/efficient_document_parsing_via_parallel_token_prediction.md)
- [\[ICML 2026\] Seeing is Understanding: Unlocking Causal Attention into Modality-Mutual Attention for Multimodal LLMs](seeing_is_understanding_unlocking_causal_attention_into_modality-mutual_attentio.md)
- [\[CVPR 2026\] Beyond the Mean: Modelling Annotation Distributions in Continuous Affect Prediction](../../CVPR2026/multimodal_vlm/beyond_the_mean_modelling_annotation_distributions_in_continuous_affect_predicti.md)
- [\[ICML 2026\] V-LynX: Token Interface Alignment for VideoX LLMs](v-lynx_token_interface_alignment_for_videox_llms.md)
- [\[CVPR 2026\] PersonaVLM: Long-Term Personalized Multimodal LLMs](../../CVPR2026/multimodal_vlm/personavlm_long_term_personalized_multimodal_llms.md)

</div>

<!-- RELATED:END -->
