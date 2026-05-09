---
title: >-
  [Paper Note] MEMO: Human-like Crisp Edge Detection Using Masked Edge Prediction
description: >-
  [CVPR 2026][Model Compression][edge detection] This paper proposes MEMO, a framework that generates crisp single-pixel edge maps using only cross-entropy loss, achieved through masked edge training and a confidence-ordered progressive inference strategy. MEMO substantially outperforms prior methods on crispness-aware evaluation (CEval ODS on BSDS improves from 0.749 to 0.836).
tags:
  - CVPR 2026
  - Model Compression
  - edge detection
  - masked prediction
  - confidence-ordered inference
  - multi-granularity prediction
  - synthetic data pre-training
date: 2026-05-08
content_hash: 047e6ef0e0ee32af
---

# MEMO: Human-like Crisp Edge Detection Using Masked Edge Prediction

**Conference**: CVPR 2026
**arXiv**: [2603.20782](https://arxiv.org/abs/2603.20782)
**Code**: [https://github.com/cplusx/MEMO_Edge_Detection](https://github.com/cplusx/MEMO_Edge_Detection)
**Area**: Model Compression / Edge Detection
**Keywords**: edge detection, masked prediction, confidence-ordered inference, multi-granularity prediction, synthetic data pre-training

## TL;DR

This paper proposes MEMO, a framework that generates crisp single-pixel edge maps using only cross-entropy loss, achieved through masked edge training and a confidence-ordered progressive inference strategy. MEMO substantially outperforms prior methods on crispness-aware evaluation (CEval ODS on BSDS improves from 0.749 to 0.836).

## Background & Motivation

1. **State of the Field**: Deep learning-based edge detection is typically formulated as a pixel-wise binary classification task optimized with cross-entropy loss. Dominant methods such as HED, RCF, and BDCN have achieved competitive detection accuracy.

2. **Limitations of Prior Work**: Models trained with cross-entropy consistently produce "thick edge" predictions—predicted edge widths far exceed the single-pixel width of human annotations. Existing remedies either design specialized sparse losses (e.g., CATS, CED) or resort to diffusion models (e.g., DiffEdge), yet crispness on benchmarks such as BSDS remains below 50%.

3. **Root Cause**: Label ambiguity introduced by multiple annotators—where slightly shifted edges are provided for the same location—softens the training signal, causing models to assign high probability to multiple pixels in the vicinity of each edge.

4. **Paper Goals**: (a) Produce crisp edges without modifying the loss function or network architecture; (b) avoid overfitting on small datasets; (c) support multi-granularity edge prediction at inference time.

5. **Starting Point**: The authors observe that **thick edge predictions exhibit a confidence gradient**—the central edge pixel carries the highest confidence, which decays toward both sides. This property suggests first committing to high-confidence predictions and then progressively resolving uncertain regions.

6. **Core Idea**: Masked training teaches the model to predict remaining edges given a partial observation. At inference, the edge map is progressively "revealed" in descending order of confidence, naturally yielding single-pixel-wide edges.

## Method

### Overall Architecture

MEMO comprises three components: a frozen image encoder $F_I$ (DINOv2-b), a masked edge encoder $F_E$, and a shared edge decoder $D$. Training proceeds in two stages: (1) pre-training $F_E$ and $D$ on 400K synthetic edge images; (2) fine-tuning on downstream datasets via LoRA adapters, adding only 1.2% additional parameters. At inference, the process starts from a fully masked edge map and iteratively reveals predictions in order of confidence.

### Key Designs

1. **Masked Edge Training**:

    - **Function**: Teaches the model to predict masked edge pixels given a partially visible edge map.
    - **Mechanism**: For each training sample, a masking ratio $r \in (0, 1]$ is sampled uniformly, and each pixel is independently masked via a Bernoulli draw. The masking ratio is embedded via sinusoidal positional encoding and injected into every layer of $F_E$ and $D$. The loss is computed only over masked pixels: $\mathcal{L} = -\mathbb{E}\!\left[\frac{1}{r}\sum_i \mathbf{1}[E_r[i]=\text{mask}] \cdot \text{BCE}\right]$
    - **Design Motivation**: This training scheme enables the model to process "partially completed" edge maps at inference time, learning to suppress redundant activations near already-confirmed edges and thereby producing thinner outputs.

2. **Confidence-Ordered Inference with LocMax**:

    - **Function**: Progressively reveals the edge map at inference to avoid the thick-edge artifacts of one-shot prediction.
    - **Mechanism**: At each step, edge probabilities are predicted for all masked pixels, but only the pixel whose confidence $c_i = \max(p_i, 1-p_i)$ is the local maximum within its $3\times 3$ neighborhood is committed. Uncommitted pixels are re-masked for the next iteration.
    - **Design Motivation**: A naïve Top-K strategy simultaneously confirms spatially adjacent high-confidence pixels, producing thick edge clusters. LocMax enforces at most one confirmed pixel per local region, naturally yielding single-pixel-wide edges. The strategy is guaranteed to converge since the number of masked pixels decreases monotonically.

3. **Multi-Granularity Prediction via Classifier-Free Guidance**:

    - **Function**: Controls edge density at inference through a single scalar, without additional training or annotations.
    - **Mechanism**: During training, the image input is replaced with a zero tensor with probability 10% (unconditional training). At inference, conditional and unconditional predictions are extrapolated to achieve granularity control: $p(E|I,E_r) = \text{Sigmoid}(s \cdot D_{\text{cond}} + (1-s) \cdot D_{\text{uncond}})$. The granularity scale $s \geq 1$; $s=1$ corresponds to standard inference, and increasing $s$ produces denser edges.
    - **Design Motivation**: This adapts classifier-free guidance from diffusion models, redefining it as a granularity controller for edge detection. Unlike methods such as MuGE that require multi-granularity supervision, MEMO achieves unsupervised multi-granularity control purely at inference time.

### Loss & Training

- **Training loss**: Standard binary cross-entropy applied only over masked pixels.
- **Pre-training**: 400K synthetic edge images are constructed by extracting segments from LAION via SAM and applying morphological erosion to obtain single-pixel boundaries.
- **Fine-tuning**: LoRA adapters are injected into the edge encoder and decoder while pre-trained weights are frozen. AdamW optimizer with learning rate $2 \times 10^{-5}$.
- **Data augmentation**: Horizontal/vertical flips and 90° rotations only, to preserve edge structure.

## Key Experimental Results

### Main Results

**BSDS results (single-scale prediction):**

| Method | SEval ODS | SEval OIS | CEval ODS | CEval OIS | AC |
|--------|-----------|-----------|-----------|-----------|-----|
| HED | 0.788 | 0.808 | 0.588 | 0.608 | 0.215 |
| RCF | 0.798 | 0.815 | 0.585 | 0.604 | 0.189 |
| EDTER | 0.824 | 0.841 | 0.698 | 0.706 | 0.288 |
| UAED | 0.829 | 0.847 | 0.722 | 0.731 | 0.227 |
| MuGE | 0.831 | 0.847 | 0.721 | 0.729 | 0.296 |
| DiffEdge | 0.834 | 0.848 | 0.749 | 0.754 | 0.476 |
| **MEMO (C*)** | **0.854** | **0.861** | **0.836** | **0.841** | **0.663** |

**Visual similarity to human annotations:**

| Method | AC | FID↓ | LPIPS↓ |
|--------|----|------|--------|
| DiffEdge | 0.476 | 89.96 | 0.300 |
| MuGE | 0.296 | 115.89 | 0.456 |
| **MEMO (C*)** | **0.663** | **83.95** | **0.282** |
| **MEMO (AC*)** | **0.705** | **75.55** | **0.291** |

### Ablation Study

| Configuration | SEval ODS | CEval ODS | AC | Note |
|---------------|-----------|-----------|----|------|
| LocMax, 10 steps | 0.854 | 0.836 | 0.663 | Full model |
| Random reveal | 0.819 | 0.794 | 0.671 | Fragmented edges, poor detection |
| Top-K reveal | 0.825 | 0.715 | 0.510 | Edges cluster and thicken |
| 5-step inference | 0.855 | 0.835 | 0.594 | Faster but lower crispness |
| Full-step inference | 0.846 | 0.842 | 0.840 | Crispest but 10.46s latency |
| Synthetic data only | — | Lower | Highest | Crisp but insufficient detection accuracy |
| Real data only | — | Higher | Lower | Duplicate edge artifacts |

### Key Findings

- **LocMax is the core contribution**: Compared to Top-K and Random strategies, LocMax improves CEval ODS by 17% and 5%, respectively, and is the only strategy that performs well across all metrics.
- **10-step inference offers the best trade-off**: Visually sufficiently crisp at 1.33s vs. 10.46s for full-step inference.
- **Synthetic data pre-training is critical**: It prevents duplicate edge artifacts and provides a single-edge inductive bias.
- **AC on BSDS improves substantially from 0.476 (DiffEdge) to 0.663/0.705**, representing a ~50% gain in crispness.
- **Multi-granularity prediction**: Smooth transitions are observed for $s \in [1.0, 2.0]$; with $M=11$ granularity levels, multi-granularity CEval ODS reaches 0.846.

## Highlights & Insights

- **"No special loss function required" philosophy**: Achieving crisp edges with only cross-entropy challenges the field's prevailing assumption that sparse losses are necessary. The key insight is shifting the problem from the training stage to the inference stage.
- **Elegance of the LocMax strategy**: By exploiting the natural confidence gradient of thick edge predictions, local maximum selection achieves precise per-pixel localization in a simple and effective manner.
- **Cross-domain transfer of classifier-free guidance**: Redefining a generative control technique from diffusion models as an edge density controller enables multi-granularity prediction without multi-granularity annotations, a principle transferable to other pixel-level prediction tasks such as multi-granularity semantic segmentation.

## Limitations & Future Work

- **Inference speed**: Ten-step iterative inference is approximately 10× slower than a single forward pass (1.33s vs. ~0.1s), limiting applicability in real-time scenarios.
- **SEval slightly below DiffEdge on BIPED**: Under the SEval (with NMS post-processing) protocol, ODS is 0.888 vs. DiffEdge's 0.899, indicating room for improvement in texture-rich scenes.
- **Synthetic data quality depends on SAM**: The quality of synthetic edges is bounded by SAM's segmentation accuracy, which may provide insufficient coverage for fine-grained edges.
- **Future directions**: (a) Distill inference to 1–2 steps for acceleration; (b) leverage stronger segmentation models such as SAM2 to construct higher-quality synthetic data; (c) explore adaptive dynamic step counts rather than a fixed 10 steps.

## Related Work & Insights

- **vs. DiffEdge**: DiffEdge uses a diffusion backbone to achieve crisp edges but is slower and exhibits fragmentation/blurring in detailed regions. MEMO attains superior crispness through a more lightweight combination of masked training and iterative inference.
- **vs. MuGE/SAUGE**: These methods require multi-granularity annotations for supervised training; MEMO achieves unsupervised multi-granularity control via classifier-free guidance.
- **vs. CATS/Refined Label**: These methods improve crispness through sparse losses or label refinement, yet AC remains below 0.5. MEMO demonstrates that training and inference strategy design can outweigh loss function engineering.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of masked training and confidence-ordered inference is novel, though the masked training concept bears resemblance to MAE.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three datasets, multiple evaluation protocols, comprehensive ablation studies, and visual similarity analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear motivation, coherent logical flow, and well-designed figures and tables.
- **Value**: ⭐⭐⭐⭐ A significant contribution to edge detection; the LocMax strategy is generalizable to other pixel-level prediction tasks.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Critical Patch-Aware Sparse Prompting with Decoupled Training for Continual Learning on the Edge](critical_patch-aware_sparse_prompting_with_decoupled_training_for_continual_lear.md)
- [\[AAAI 2026\] Lightweight Optimal-Transport Harmonization on Edge Devices](../../AAAI2026/model_compression/lightweight_optimal-transport_harmonization_on_edge_devices.md)
- [\[CVPR 2026\] Memory-Efficient Transfer Learning with Fading Side Networks via Masked Dual Path Distillation](memory_efficient_transfer_learning_with_fading_side_networks.md)
- [\[CVPR 2026\] Markovian Scale Prediction: A New Era of Visual Autoregressive Generation](markovian_scale_prediction_a_new_era_of_visual_autoregressive_generation.md)
- [\[CVPR 2026\] Towards Generalizable AI-Generated Image Detection via Image-Adaptive Prompt Learning](iapl_aigenerated_image_detection_adaptive_prompt.md)

<!-- RELATED:END -->
