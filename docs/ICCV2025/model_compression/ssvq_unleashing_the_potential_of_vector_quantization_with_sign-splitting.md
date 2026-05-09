---
title: >-
  [Paper Note] SSVQ: Unleashing the Potential of Vector Quantization with Sign-Splitting
description: >-
  [ICCV 2025][Model Compression][vector quantization] This paper proposes Sign-Splitting Vector Quantization (SSVQ), which decouples the sign bits of weights from the codebook, introduces learnable sign parameters and an enhanced iterative freezing strategy, enabling each quantized weight to update independently along its own gradient direction during VQ fine-tuning. SSVQ significantly outperforms conventional VQ and scalar quantization under extreme compression ratios.
tags:
  - ICCV 2025
  - Model Compression
  - vector quantization
  - sign splitting
  - learnable sign bits
  - iterative freezing
  - hardware acceleration
date: 2026-05-08
content_hash: d7ac914cae5c5785
---

# SSVQ: Unleashing the Potential of Vector Quantization with Sign-Splitting

**Conference**: ICCV 2025
**arXiv**: [2503.08668](https://arxiv.org/abs/2503.08668)
**Code**: [https://github.com/list0830/SSVQ](https://github.com/list0830/SSVQ)
**Area**: Model Compression / Vector Quantization
**Keywords**: vector quantization, sign splitting, learnable sign bits, iterative freezing, hardware acceleration

## TL;DR

This paper proposes Sign-Splitting Vector Quantization (SSVQ), which decouples the sign bits of weights from the codebook, introduces learnable sign parameters and an enhanced iterative freezing strategy, enabling each quantized weight to update independently along its own gradient direction during VQ fine-tuning. SSVQ significantly outperforms conventional VQ and scalar quantization under extreme compression ratios.

## Background & Motivation

Vector quantization (VQ) demonstrates lower quantization error than uniform quantization (UQ) in extreme compression scenarios and is a promising weight compression technique. However, VQ suffers from a fundamental limitation during fine-tuning: all weight vectors assigned to the same codeword can only be updated in the same direction (since only the codebook is tunable). This leads to a "gradient dominance" phenomenon, where a small number of high-magnitude gradients dominate the update direction for an entire cluster, forcing the majority of weights away from their local optima.

The authors empirically validate this issue on MobileNet-V2: the cosine similarity between the top 5% gradients and the codeword gradient reaches 0.66, while the bottom 60% of gradients achieve only 0.26, confirming that most weights are "hijacked" by a minority of gradients. Moreover, even with all parameters trainable (BN+FC+Codebook), VQ fine-tuning yields only 55.27% accuracy, far below the desired level.

## Method

### Overall Architecture

The core idea of SSVQ is to decompose the weight matrix $W$ into a sign matrix $S$ and absolute values $|W|$, apply k-means clustering only to the absolute values, and then introduce learnable sign parameters $L_s$ so that each quantized weight can independently determine its update direction. The quantized weight is reconstructed as:

$$W_q = \mathcal{C}[A] \circ \text{sign}(L_s)$$

### Key Designs

1. **Sign-Splitting**: The sign bits of weights are extracted as a 1-bit mask, and clustering is performed on the resulting all-positive values. This significantly reduces the number of required codewords (clustering over non-negative values is simpler), allowing higher-dimensional, smaller codebooks to offset the 1-bit storage overhead. Under the same compression ratio, SSVQ achieves clustering error comparable to conventional VQ.

2. **Learnable Sign Bits**: A continuous latent variable $L_s$ is introduced and initialized as $L_s \leftarrow \alpha \cdot W$. The forward pass uses $\text{sign}(L_s)$ as the sign bit, and gradients are approximated via the Straight-Through Estimator (STE) during backpropagation. The gradient of the sign bit is $g_{L_s} \approx \frac{\nabla L}{\nabla W_q} \circ c$, where the codeword magnitude $c$ is incorporated, which contributes to training stability.

3. **Enhanced Iterative Freezing**: Since a sign flip induces a change of $2|c|$ in the quantized value—a large perturbation—learnable signs are prone to oscillation. The authors identify two key phenomena: (a) the frequency EMA fluctuates drastically in early training, so premature freezing captures unstable states; (b) sign-EMA-based freezing decisions themselves oscillate frequently, making simple majority voting more stable. The proposed strategy checks the oscillation frequency EMA every $F_i$ iterations and permanently freezes signs whose frequency exceeds a cosine-decaying threshold, using the majority vote over positive/negative counts as the frozen value.

### Loss & Training

- AdamW optimizer with cosine annealing learning rate schedule
- All classification and detection tasks trained for 10 epochs without knowledge distillation
- Sign change rate is controlled via the learning rate or initialization parameter $\alpha$
- SSVQ requires storing three components: the codebook, assignment indices, and sign masks

## Key Experimental Results

### Main Results

| Model | Method | Compression Ratio | Metric |
|-------|--------|-------------------|--------|
| DeiT-Tiny (ImageNet) | VQ | 21× | ~47% Top-1 |
| DeiT-Tiny (ImageNet) | SSVQ | 21× | ~59% Top-1 (+12%) |
| MobileNet-V2 | MVQ (16×) | 16× | 65.1% |
| MobileNet-V2 | SSVQ (16×) | 16× | 65.9% |
| EfficientNet-lite | MVQ (16×) | 16× | 68.2% |
| EfficientNet-lite | SSVQ (18×) | 18× | 69.5% |
| YOLOV9-S (COCO) | VQ (20×) | 20× | 30.0 AP |
| YOLOV9-S (COCO) | SSVQ (20×) | 20× | 35.2 AP (+5.2) |
| GELAN-C (COCO) | VQ (21×) | 21× | 43.5 bbox AP |
| GELAN-C (COCO) | SSVQ (21×) | 21× | 47.6 bbox AP (+4.1) |
| SD v1-4 (COCO) | VQ 2-bit | 2-bit | 24.5 FID-to-FP |
| SD v1-4 (COCO) | SSVQ 2-bit | 2-bit | 13.6 FID-to-FP |
| Llama3.2-1B (Wiki2) | VQ 2-bit | 2-bit | 63.1 PPL |
| Llama3.2-1B (Wiki2) | SSVQ 2-bit | 2-bit | 13.9 PPL |

### Ablation Study

| Configuration (DeiT-T, k=16, d=8) | Top-1 Acc |
|------------------------------------|-----------|
| Fixed sign (baseline) | ~47% (21×) |
| Learnable sign (no freezing) | ~54% |
| SSVQ (learnable sign + iterative freezing) | ~62% (+14.8%) |
| Freeze interval 100, MSV | 60.9% |
| Freeze interval 200, MSV | 61.47% |
| Freeze interval 500, MSV | **61.90%** |
| Freeze interval 1000, MSV | 61.55% |
| Freeze interval 500, EMA | 60.8% |

### Key Findings

- Learnable sign bits consistently improve accuracy by 3%–9% across all compression ratios, with larger gains at higher compression
- Majority voting outperforms EMA as the freezing criterion (61.90% vs. 60.8%)
- Too-small freezing intervals (100) capture unstable states prematurely, while too-large intervals (1000) harm training stability; 500 is optimal
- SSVQ significantly outperforms VQ on NLP tasks (Llama3.2-1B): 2-bit PPL drops from 63.1 to 13.9
- Hardware simulation demonstrates a 3× inference speedup on DeiT-Tiny through reduced main memory access

## Highlights & Insights

1. **Insightful Analysis**: The diagnosis of VQ fine-tuning limitations via the gradient dominance phenomenon is rigorous; cosine similarity experiments clearly expose the root cause.
2. **Elegant Design**: Sign splitting restores per-weight update freedom from codebook-level granularity, and the 1-bit overhead is offset by the simplified clustering over non-negative values.
3. **Extensive Validation**: Experiments span five task categories—classification, detection, segmentation, generation, and NLP—across CNN, ViT, UNet, DiT, and LLM architectures.
4. **Hardware Deployment**: A hardware simulator supporting SSVQ is constructed, validating the 3× speedup beyond a purely theoretical claim.

## Limitations & Future Work

- The storage overhead of 1-bit sign masks is non-trivial at extremely low bit-widths; more efficient sign encoding warrants exploration
- Hyperparameters for iterative freezing (interval, threshold decay schedule) still require manual tuning
- Evaluation is limited to vision and language models; multimodal large models (e.g., VLMs) and audio models are not covered
- Hardware simulation is conducted on a single accelerator; practical ASIC/FPGA deployment remains to be validated

## Related Work & Insights

- SSVQ is compatible with methods such as MVQ (codebook with pruning) and DKM (differentiable k-means) and can be combined with them
- The sign-splitting paradigm is generalizable to other compression approaches (e.g., product quantization), warranting further exploration
- The iterative freezing strategy is conceptually related to Oscillation-free QAT (OsQAT) but incorporates key adaptations for the VQ setting

## Rating

- **Novelty**: ⭐⭐⭐⭐ Sign splitting with learnable sign bits constitutes a concise and effective new paradigm
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Five task categories, diverse architectures, thorough ablations, and hardware validation
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clearly articulated with rich figures and tables
- **Value**: ⭐⭐⭐⭐ Practically significant for edge deployment; code is open-sourced

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Task Vector Quantization for Memory-Efficient Model Merging](task_vector_quantization_for_memory-efficient_model_merging.md)
- [\[ICCV 2025\] VQ-SGen: A Vector Quantized Stroke Representation for Creative Sketch Generation](vq-sgen_a_vector_quantized_stroke_representation_for_creative_sketch_generation.md)
- [\[CVPR 2026\] RDVQ: Differentiable Vector Quantization for Rate-Distortion Optimization of Generative Image Compression](../../CVPR2026/model_compression/rdvq_differentiable_vq_image_compression.md)
- [\[AAAI 2026\] SIGN: Schema-Induced Games for Naming](../../AAAI2026/model_compression/sign_schema-induced_games_for_naming.md)
- [\[ICLR 2026\] KBVQ-MoE: KLT-guided SVD with Bias-Corrected Vector Quantization for MoE Large Language Models](../../ICLR2026/model_compression/kbvq-moe_klt-guided_svd_with_bias-corrected_vector_quantization_for_moe_large_la.md)

</div>

<!-- RELATED:END -->
