---
title: >-
  [Paper Note] MPM: Mutual Pair Merging for Efficient Vision Transformers
description: >-
  [CVPR 2026][Segmentation][Token Merging] This paper proposes Mutual Pair Merging (MPM), a parameter-free, training-free token merging module for ViTs that reduces sequence length via mutual nearest-neighbor pairing and mean fusion. On ADE20K, MPM achieves a 60% latency reduction on Raspberry Pi 5 for ViT-Tiny and a 20% throughput improvement on H100 with FlashAttention-2, while keeping mIoU degradation within 3%.
tags:
  - CVPR 2026
  - Segmentation
  - Token Merging
  - Semantic Segmentation
  - Vision Transformer
  - Inference Acceleration
  - Training-Free Method
date: 2026-05-08
content_hash: eda93c276a22c78c
---

# MPM: Mutual Pair Merging for Efficient Vision Transformers

**Conference**: CVPR 2026
**arXiv**: [2604.05718](https://arxiv.org/abs/2604.05718)
**Code**: None
**Area**: Segmentation
**Keywords**: Token Merging, Semantic Segmentation, Vision Transformer, Inference Acceleration, Training-Free Method

## TL;DR

This paper proposes Mutual Pair Merging (MPM), a parameter-free, training-free token merging module for ViTs that reduces sequence length via mutual nearest-neighbor pairing and mean fusion. On ADE20K, MPM achieves a 60% latency reduction on Raspberry Pi 5 for ViT-Tiny and a 20% throughput improvement on H100 with FlashAttention-2, while keeping mIoU degradation within 3%.

## Background & Motivation

1. **Background**: Vision Transformers deliver strong performance in semantic segmentation, but the $O(N^2)$ complexity of self-attention causes inference costs to scale rapidly with resolution. Reducing sequence length (token reduction) is a natural acceleration strategy, with existing approaches including token pruning/selection (DynamicViT, EViT) and token aggregation/merging (ToMe, ALGM).

2. **Limitations of Prior Work**: (a) Most token reduction methods target classification, whereas segmentation requires reconstructing dense, pixel-aligned features, imposing stricter constraints on token reduction; (b) existing methods commonly report FLOPs or theoretical speedups, but on modern accelerators (e.g., GPUs with FlashAttention), the overhead of merging operations may offset or even reverse expected gains; (c) many methods require fine-tuning or additional trainable parameters, hindering plug-and-play deployment.

3. **Key Challenge**: Token reduction theoretically reduces computation, but (a) segmentation requires reconstructing the full token sequence for the decoder; (b) on highly optimized GPU kernels, the additional overhead of merging operations can erase speedup benefits; (c) variable-length sequences require padding, which negatively impacts batched throughput.

4. **Goal**: Design a training-free, segmentation-specific token merging method that achieves genuine wall-clock speedups end-to-end, while honestly quantifying actual latency inclusive of merging overhead.

5. **Key Insight**: Adopt the simplest possible design—mutual nearest-neighbor pairing and mean fusion—to minimize overhead; control the speed–accuracy trade-off via discrete insertion positions (rather than continuous thresholds or retention rates); and store merge maps for exact reconstruction.

6. **Core Idea**: Identify mutually nearest-neighbor token pairs in feature space via cosine similarity and merge them by averaging; use an integer merge map for gather-based exact reconstruction, allowing segmentation decoders to operate without any modification.

## Method

### Overall Architecture

An input image is processed by ViT patch embedding to produce $N$ image tokens. MPM modules are inserted before specific encoder layers (default: layers 3 and 6, i.e., 0-based indices 2 and 5). Each insertion reduces the token count (by at most half, though the actual reduction is data-dependent), and subsequent layers operate on the reduced sequence. After encoding, the stored merge maps are used in a gather operation to restore the original $N$-token sequence, which is then passed to a standard Mask Transformer decoder for segmentation prediction.

### Key Designs

1. **Mutual Nearest-Neighbor Pairing Merging**:

    - **Function**: Deterministically identifies token pairs most suitable for merging.
    - **Mechanism**: All image tokens are L2-normalized and a dense cosine similarity matrix $S = \tilde{X}\tilde{X}^\top$ is computed. For each token $i$, the most similar neighbor is identified as $b(i) = \arg\max_{j \neq i} S_{ij}$. Only mutually nearest-neighbor pairs $(i,j)$ (i.e., $b(i)=j$ and $b(j)=i$) are merged, with the token of the smaller index serving as the representative; the merged token is the mean of both. Tokens without a mutual nearest neighbor remain as singletons. The entire process involves no learnable parameters and is fully deterministic.
    - **Design Motivation**: The mutual nearest-neighbor condition ensures symmetry and determinism—avoiding the complexity of bipartite matching in ToMe and eliminating conflicts where multiple tokens compete to merge with a single popular token. The reduction ratio is adaptive in practice, as not all tokens find a mutual nearest neighbor, yielding at most 50% reduction.

2. **Multi-Stage Merge Map Composition and Reconstruction**:

    - **Function**: Supports multiple MPM insertions and enables exact restoration of the original sequence length prior to decoding.
    - **Mechanism**: Each MPM call returns an integer mapping vector $r$ recording the correspondence from original tokens to their merged representatives. Mappings from two stages are composed via simple index operations: $r^{(*)}(i) = r^{(2)}(r^{(1)}(i))$. After encoding, the full sequence is recovered in a single gather step: $Z_{\text{img}}^{\uparrow}[i] = Z_{\text{img}}[r^{(*)}(i)]$. This is a pure copy operation that preserves the original raster-scan order, requiring no modifications to the decoder.
    - **Design Motivation**: Segmentation decoders (e.g., Mask Transformer) expect a full $\frac{H}{P} \times \frac{W}{P}$ grid of features. By storing and composing merge maps, arbitrary-depth token merging can be performed without modifying the decoder.

3. **Discrete Insertion Schedule**:

    - **Function**: The sole knob for controlling the speed–accuracy trade-off.
    - **Mechanism**: MPM has no continuous compression parameters (no retention rate, no threshold); the trade-off is determined entirely by which layers receive an MPM insertion. The default configuration inserts MPM at layers 3 and 6. Earlier insertions yield greater compression and acceleration but larger accuracy degradation; later insertions have smaller accuracy impact but also smaller speedup. This design eliminates the need to retune parameters when deployment conditions change.
    - **Design Motivation**: In real-world deployments (e.g., fixed-scene edge surveillance), scene statistics such as lighting and weather vary over time, making manually selected thresholds or retention rates potentially obsolete. The knob-free design eliminates this concern. The paper demonstrates behavioral differences in merging between daytime and nighttime images of the same scene—approximately 6% fewer tokens are merged at night.

### Loss & Training

MPM is a completely training-free module that is inserted directly into a pretrained ViT encoder without any fine-tuning.

## Key Experimental Results

### Main Results (ADE20K, H100 without FlashAttention)

| Model | Method | mIoU | GFLOPs | FPS (B=32) |
|-------|--------|------|--------|------------|
| Seg-T/16 | No merging | 38.1 | 25 | 660 |
| Seg-T/16 | ToMe | 38.1 | ~19 | 751 |
| Seg-T/16 | ALGM* | 38.9 | ~16.7 | 665 |
| Seg-T/16 | **MPM(2,5)** | 37.6 | ~17.6 | **831** |
| Seg-B/16 | No merging | 48.5 | 258 | 133 |
| Seg-B/16 | **MPM(2,5)** | 48.0 | ~184 | **177** |
| Seg-L/16 | No merging | 51.7 | 800 | 47 |
| Seg-L/16 | **MPM(2,5)** | 50.4 | ~496 | **74** |

### Cross-Platform Latency Comparison

| Platform | ViT-T Baseline | MPM | Speedup |
|----------|---------------|-----|---------|
| Raspberry Pi 5 (B=1) | 1.06 FPS | 1.71 FPS | 1.61× |
| Raspberry Pi 5 (B=2) | 1.05 FPS | 1.75 FPS | 1.67× |
| H100 FA2 (B=32, ViT-L) | 375 FPS | 456 FPS | 1.22× |

### Ablation Study (Effect of Insertion Position)

Earlier insertion leads to greater compression, higher speedup, and larger accuracy degradation. The default configuration (2,5) consistently provides Pareto-optimal trade-offs across multiple datasets and model scales.

### Key Findings

- **Actual wall-clock gains are not strictly proportional to FLOPs reduction**: On H100 with FlashAttention-2, a 38% FLOPs reduction yields only 22% FPS improvement (ViT-L), as FA2 itself highly optimizes attention computation.
- **Largest gains on Raspberry Pi 5**: Edge devices lack parallelization optimizations, so reductions in token count translate more directly to linear latency decreases.
- **Spatial locality of merging**: Although MPM performs global pairing without local constraints, in practice most mutual nearest-neighbor pairs occur between spatially adjacent patches—the method naturally discovers spatial locality.
- **Well-controlled mIoU degradation**: Seg-L/16 drops from 51.7 to 50.4 (−1.3); Seg-T/16 drops from 38.1 to 37.6 (−0.5).
- **Cross-dataset consistency**: Reasonable speed–accuracy trade-offs are maintained on ADE20K, Pascal Context, and Cityscapes.

## Highlights & Insights

- **Honest efficiency evaluation** is the paper's most notable contribution: while many token reduction works report only FLOPs, this paper measures end-to-end latency inclusive of merging overhead on Raspberry Pi 5 and H100 (with and without FlashAttention-2), separately reporting merge and reconstruction time. This sets a higher standard for evaluation in this research direction.
- **The "knob-free" design philosophy** is worth emulating: adaptive compression is achieved through the natural sparsity of mutual nearest-neighbor matching (not every token finds a mutual neighbor), eliminating hyperparameters that would otherwise require cross-dataset tuning—particularly valuable for online deployment.
- **Simplicity is effective**: the entire method reduces to cosine similarity + mutual nearest neighbors + averaging + gather, with no learnable parameters, yet achieves speedups on par with or better than more complex methods (e.g., CTS and ALGM, which require training) across multiple hardware platforms.

## Limitations & Future Work

- mIoU degradation, while modest, is consistently present and may be unacceptable in precision-critical applications such as medical image segmentation.
- Compared to fine-tuning-based methods such as ALGM, MPM generally achieves slightly lower mIoU (ALGM sometimes even improves mIoU), indicating that training-free methods have an inherent accuracy ceiling.
- The $O(N^2)$ similarity computation for mutual nearest-neighbor pairing introduces overhead that, while currently negligible, may become a bottleneck at very high resolutions.
- Combinations with other acceleration techniques (e.g., knowledge distillation, quantization) are not explored.
- The analysis of variable-length sequences on batched throughput is insufficiently deep—padding strategies may significantly affect actual throughput.

## Related Work & Insights

- **vs. ToMe**: ToMe employs bipartite graph matching with a fixed merging rate; MPM uses mutual nearest neighbors with an adaptive rate. MPM achieves higher FPS on segmentation tasks (831 vs. 751 on Seg-T/16) due to lower computational overhead of the mutual nearest-neighbor computation.
- **vs. ALGM**: ALGM is the strongest training-based baseline specifically designed for segmentation, employing a two-stage local-to-global merging strategy. MPM achieves slightly lower mIoU but is completely training-free and more readily plug-and-play.
- **vs. CTS**: CTS requires training a policy network to determine token sharing; while inference-time behavior is fixed, it is less robust to distribution shift. MPM computes merging from actual content at each frame.

## Rating

- **Novelty**: ⭐⭐⭐ The core idea (mutual nearest-neighbor merging) is straightforward with limited technical novelty, though the "knob-free + segmentation reconstruction" design positioning offers distinctive value.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three segmentation datasets, four model scales, three hardware platforms (Pi5 / H100 / H100+FA2), and end-to-end latency across multiple batch sizes—a benchmark standard for efficiency evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ Method descriptions are precise, design motivations are clearly articulated, and limitations are discussed candidly.
- **Value**: ⭐⭐⭐⭐ Provides clear quantitative evidence of the practical benefits of token reduction in segmentation and offers practical utility for edge deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Revisiting \[CLS\] and Patch Token Interaction in Vision Transformers](../../ICLR2026/segmentation/revisiting_cls_and_patch_token_interaction_in_vision_transformers.md)
- [\[NeurIPS 2025\] Vision Transformers with Self-Distilled Registers](../../NeurIPS2025/segmentation/vision_transformers_with_self-distilled_registers.md)
- [\[ICLR 2026\] Thicker and Quicker: A Jumbo Token for Fast Plain Vision Transformers](../../ICLR2026/segmentation/thicker_and_quicker_a_jumbo_token_for_fast_plain_vision_transformers.md)
- [\[ICCV 2025\] LeGrad: An Explainability Method for Vision Transformers via Feature Formation Sensitivity](../../ICCV2025/segmentation/legrad_an_explainability_method_for_vision_transformers_via_feature_formation_se.md)
- [\[CVPR 2026\] Generalizable Knowledge Distillation from Vision Foundation Models for Semantic Segmentation](gkd_generalizable_knowledge_distillation_vfm.md)

</div>

<!-- RELATED:END -->
