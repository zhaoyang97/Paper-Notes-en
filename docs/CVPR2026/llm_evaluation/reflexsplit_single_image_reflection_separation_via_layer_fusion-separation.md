---
title: >-
  [Paper Note] ReflexSplit: Single Image Reflection Separation via Layer Fusion-Separation
description: >-
  [CVPR 2026][LLM Evaluation][Single image reflection separation] ReflexSplit proposes an explicit layer fusion-separation framework that addresses the transmission-reflection confusion problem in single image reflection s…
tags:
  - "CVPR 2026"
  - "LLM Evaluation"
  - "Single image reflection separation"
  - "differential attention"
  - "cross-scale fusion"
  - "curriculum learning"
  - "dual-stream architecture"
date: 2026-05-08
content_hash: 8a5fd1686e682ab6
---

# ReflexSplit: Single Image Reflection Separation via Layer Fusion-Separation

**Conference**: CVPR 2026
**arXiv**: [2601.17468](https://arxiv.org/abs/2601.17468)  
**Code**: [https://github.com/wuw2135/ReflexSplit](https://github.com/wuw2135/ReflexSplit)  
**Area**: Image Restoration
**Keywords**: Single image reflection separation, differential attention, cross-scale fusion, curriculum learning, dual-stream architecture

## TL;DR
ReflexSplit proposes an explicit layer fusion-separation framework that addresses the transmission-reflection confusion problem in single image reflection separation (SIRS). It employs Cross-scale Gated Fusion (CrGF) for adaptive multi-scale feature aggregation, a differential dual-dimensional attention mechanism $\mathbf{A}^t - \lambda_\ell \mathbf{A}^r$ within the Layer Fusion-Separation Block (LFSB) for cross-stream interference suppression, and a curriculum training strategy with depth-dependent initialization and epoch-wise warmup to progressively strengthen separation intensity, achieving state-of-the-art performance on both synthetic and real-world benchmarks.

## Background & Motivation

1. **Background**: SIRS decomposes a mixed image $\mathbf{I}$ into a transmission layer $\mathbf{T}$ and a reflection layer $\mathbf{R}$. The field has evolved from simple linear superposition models $\mathbf{I}=\mathbf{T}+\mathbf{R}$ to nonlinear residual formulations $\mathbf{I}=\mathbf{T}+\mathbf{R}+\Phi(\mathbf{T},\mathbf{R})$, with methods such as YTMT, DSRNet, and DSIT progressively enhancing inter-layer interaction.

2. **Limitations of Prior Work**: Under strong reflections (e.g., specular highlights on water surfaces) or semantically ambiguous scenes (e.g., a painting of the moon on a wall misidentified as a reflection), networks erroneously confuse the transmission and reflection layers. As network depth increases, progressive information loss causes intra-layer and inter-layer features to become inseparable, which is particularly severe in deep decoder stages.

3. **Key Challenge**: Existing methods suffer from two deficiencies: (a) insufficient multi-scale feature aggregation leading to gradient instability—DSIT lacks gradient stability, RDNet lacks explicit scale coordination, and MuGI operates at only a single scale; (b) implicit fusion mechanisms causing progressive layer confusion—DSIT directly aggregates dual-dimensional attention outputs without separation constraints.

4. **Goal**: (a) How to adaptively aggregate features from multiple sources (semantic priors, texture details, decoder context) across scales? (b) How to enforce layer-specific separation while simultaneously fusing shared structural information? (c) How to avoid training instability caused by excessively strong separation constraints in early training?

5. **Key Insight**: The paper models reflection separation explicitly as an alternating fusion-separation process—first fusing to obtain shared structural information, then applying differential attention for layer-specific disentanglement. This extends the attention cancellation idea from Differential Transformer from single-stream noise suppression to dual-stream layer separation.

6. **Core Idea**: By alternating between fusion (shared structure extraction) and differential attention separation (cross-stream subtraction $\mathbf{A}^t - \lambda_\ell \mathbf{A}^r$) within a dual-stream architecture, combined with curriculum training that progressively increases separation intensity, the framework achieves robust transmission-reflection disentanglement throughout all network levels.

## Method

### Overall Architecture

ReflexSplit adopts a dual-stream encoder-decoder architecture. The encoder side includes two parallel feature extraction branches: a pretrained Swin Transformer (GFEB) for extracting global semantic priors $\{\mathbf{P}_\ell\}$ and a MuGI-based CNN (LFEB) for capturing local texture details $\{\mathbf{E}_\ell\}$. On the decoder side, CrGF adaptively aggregates multi-scale features at each level, and LFSB alternates between fusion and differential separation at each decoder stage. The network outputs the transmission layer $\mathbf{T}$, the reflection layer $\mathbf{R}$, and a residual term $\mathbf{RR}$ capturing nonlinear interactions.

### Key Designs

1. **Cross-scale Gated Fusion (CrGF)**:

    - **Function**: Adaptively aggregates semantic priors, texture details, and decoder context at each decoder level to stabilize gradient flow and prevent feature degradation.
    - **Mechanism**: At decoder levels $\{4, 3, 2\}$, raw features are formed as $\mathbf{F}_\ell^{\text{raw}} = \mathbf{F}_{\ell+1} + \mathbf{P}_\ell + \mathbf{E}_\ell$ (decoder context + semantics + texture), then fused with decoder context via bidirectional gating paths: $\mathbf{F}_\ell^{\text{main}} = \mathcal{G}_1(\mathbf{F}_\ell^{\text{raw}}) \odot \mathcal{G}_2(\mathbf{F}_{\ell+1})$ and $\mathbf{F}_\ell^{\text{aux}} = \mathcal{G}_1(\mathbf{F}_{\ell+1}) \odot \mathcal{G}_2(\mathbf{F}_\ell^{\text{raw}})$, where $\mathcal{G}$ selects complementary channels via channel splitting. A softmax-weighted combination yields the final output.
    - **Design Motivation**: MuGI operates at only a single scale; RobustSIRR uses direct concatenation without adaptive gating; RDNet employs fixed invertible paths that lack explicit scale coordination. CrGF addresses cross-scale, cross-source feature coordination through bidirectional adaptive gating.

2. **Layer Fusion-Separation Block (LFSB)**:

    - **Function**: Alternates between fusion (shared structure extraction) and separation (layer-specific disentanglement) at each decoder stage to prevent progressive transmission-reflection confusion.
    - **Mechanism**: Executed in three steps—
      (a) **Early fusion**: Aligns semantic spaces via bidirectional cross-stream projection $\mathbf{F}^{t'}_\ell = \mathbf{W}^t[\mathbf{F}^t_\ell \| \mathbf{F}^r_\ell]$, providing each stream with complementary information;
      (b) **Differential dual-dimensional attention**: Self-attention (SA) is computed by concatenating along the batch dimension for spatial refinement, and cross-attention (CA) is computed by concatenating along the sequence dimension to capture inter-layer dependencies. A differential operator is then applied: $\mathbf{A}^t_{\text{diff}} = (\mathbf{A}^t_{\text{SA}} + \mathbf{A}^t_{\text{CA}}) - \sigma(\lambda_\ell)(\mathbf{A}^r_{\text{SA}} + \mathbf{A}^r_{\text{CA}})$, suppressing cross-stream interference via subtraction;
      (c) **Late aggregation**: An FFN with residual connections integrates the separated features.
    - **Design Motivation**: Unlike Differential Transformer, which applies intra-head subtraction to suppress noise within a single stream, LFSB extends subtraction across streams—using the reflection stream's attention patterns to suppress residual reflection artifacts in the transmission stream, and vice versa. DSIT directly aggregates SA and CA outputs without separation constraints, causing progressive confusion.

3. **Curriculum Training Strategy**:

    - **Function**: Progressively strengthens differential separation intensity, allowing the network to first learn holistic reconstruction before focusing on layer-specific separation.
    - **Mechanism**: Two complementary mechanisms control $\lambda_\ell$:
      (a) **Depth-dependent initialization** $\lambda_\ell^{\text{init}} = 0.8 - 0.6 e^{-0.3\ell}$: deeper layers receive stronger separation weights ($\lambda \to 0.8$), while shallow layers maintain weak weights ($\lambda \to 0.2$) to preserve fine-grained details;
      (b) **Epoch-wise warmup** $\lambda_{\text{diff}}(e)$: linearly increases from 0.1 to 1.0 over the first 30 epochs. The effective coefficient is $\lambda_\ell(e) = \lambda_\ell^{\text{init}} \cdot \lambda_{\text{diff}}(e)$.
    - **Design Motivation**: Excessively strong differential separation in early training destabilizes optimization since features are not yet well-structured; excessively weak separation fails to achieve effective disentanglement. Curriculum training adaptively controls separation intensity across both spatial (depth-dependent) and temporal (epoch-wise) dimensions.

### Loss & Training

The total loss consists of six terms: a Charbonnier reconstruction loss $\mathcal{L}_{\text{rec}}$ (for the transmission layer), an $\ell_1$ reflection loss $\mathcal{L}_{\text{refl}}$, a VGG perceptual loss $\mathcal{L}_{\text{vgg}}$ (layers $\{2, 7, 12, 21, 30\}$), a color consistency loss $\mathcal{L}_{\text{color}}$, an exclusion loss $\mathcal{L}_{\text{exclu}}$, and a reconstruction consistency loss $\mathcal{L}_{\text{recons}}$.

## Key Experimental Results

### Main Results

| Dataset | Metric | ReflexSplit | Prev. SOTA (RDNet) | Gain |
|--------|------|------|----------|------|
| Real20 | PSNR / SSIM | 25.22 / 0.846 | 25.17 / 0.841 | +0.05 / +0.005 |
| Objects | PSNR / SSIM | 27.08 / 0.929 | 27.11 / 0.925 | −0.03 / +0.004 |
| Postcard | PSNR / SSIM | 25.38 / 0.927 | 25.04 / 0.910 | +0.34 / +0.017 |
| Wild | PSNR / SSIM | 27.30 / 0.933 | 27.86 / 0.931 | −0.56 / +0.002 |
| Nature | PSNR / SSIM | 27.03 / 0.854 | 26.75 / 0.846 | +0.28 / +0.008 |
| Average (540 images) | PSNR / SSIM | 26.40 / 0.898 | 26.38 / 0.890 | +0.02 / +0.008 |

Note: ReflexSplit uses 174M parameters vs. RDNet's 266.4M, demonstrating higher parameter efficiency.

### Ablation Study

| Configuration | Key Effect | Notes |
|------|---------|------|
| DSIT (baseline) | Transmission-reflection confusion in deep layers | No separation constraint, progressive degradation |
| + CrGF | Stabilized gradient flow | Adaptive cross-scale aggregation |
| + LFSB (w/o diff) | Fusion without separation | Shared structure extracted but layer confusion unresolved |
| + LFSB (w/ diff) | Effective layer-specific feature separation | Differential operator suppresses cross-stream interference |
| + Curriculum training | Improved training stability | Progressive strengthening of separation intensity |

### Key Findings
- The largest gains are observed on the Postcard subset (+0.34 PSNR / +0.017 SSIM), which contains stronger reflections and more pronounced nonlinear mixing.
- Differential attention visualizations (Figure 6) clearly demonstrate how cross-stream subtraction suppresses overlapping attention patterns, transforming ambiguous mixed attention into layer-specific, more balanced distributions.
- Compared to RDNet (266.4M parameters, two-stage training), ReflexSplit achieves comparable or superior performance with fewer parameters (174M) and a simpler training pipeline.

## Highlights & Insights
- **Transferring Differential Transformer to dual-stream separation**: The original Differential Transformer performs intra-head subtraction to suppress noise within a single stream. This paper extends the idea cross-stream—using the attention patterns of one stream to cancel inter-layer interference in the other. This cross-modal/cross-stream subtraction paradigm can be broadly transferred to any multi-stream architecture requiring separation of entangled signals.
- **Spatiotemporal co-design of curriculum training**: Depth-dependent initialization combined with epoch-wise warmup forms a 2D control surface over separation intensity, enabling optimal fusion-separation balance at different training stages and network depths. This fine-grained training intensity control strategy is generalizable to other multi-scale decomposition tasks.

## Limitations & Future Work
- Performance on certain subsets (Objects, Wild) falls slightly below RDNet in PSNR, indicating room for improvement in adapting to specific scene categories.
- The reliance on a pretrained Swin Transformer for global semantic extraction may limit out-of-domain generalization.
- The initialization formula for the differential coefficient $\lambda_\ell$ is manually designed and may not generalize well across different data distributions.
- The paper does not provide detailed computational efficiency comparisons (FLOPs, inference latency); the 174M parameter count is larger than DSIT (136M) but smaller than RDNet (266M).

## Related Work & Insights
- **vs. DSIT**: DSIT employs dual-dimensional attention but directly aggregates outputs without separation constraints, resulting in progressive confusion in deep layers. ReflexSplit uses the differential operator for explicit disentanglement.
- **vs. RDNet**: RDNet employs an invertible encoder for lossless gradient flow but requires 266M parameters and two-stage training. ReflexSplit achieves adaptive cross-scale coordination via CrGF with fewer parameters.
- **vs. DSRNet**: DSRNet introduces MuGI for inter-layer interaction but operates at only a single scale without attention separation constraints. CrGF extends its gating idea to cross-scale aggregation, while LFSB provides a more systematic fusion-separation solution.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The application of differential attention to dual-stream separation is creative, though the overall framework represents incremental improvement.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-dataset evaluation with visualization analysis; detailed ablation numbers could be more comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clear, method description is detailed, and figures are informative.
- **Value**: ⭐⭐⭐ Contributes meaningfully within the SIRS subfield; broader impact on the general vision community is limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ASIDE: Architectural Separation of Instructions and Data in Language Models](../../ICLR2026/llm_evaluation/aside_architectural_separation_of_instructions_and_data_in_language_models.md)
- [\[CVPR 2026\] AdaBet: Gradient-free Layer Selection for Efficient Training of Deep Neural Networks](adabet_gradient-free_layer_selection_for_efficient_training_of_deep_neural_netwo.md)
- [\[ICML 2026\] Automatic Layer Selection for Hallucination Detection](../../ICML2026/llm_evaluation/automatic_layer_selection_for_hallucination_detection.md)
- [\[AAAI 2026\] MicroEvoEval: A Systematic Evaluation Framework for Image-Based Microstructure Evolution Prediction](../../AAAI2026/llm_evaluation/microevoeval_a_systematic_evaluation_framework_for_image-based_microstructure_ev.md)
- [\[ACL 2026\] Minos: A Multimodal Evaluation Model for Bidirectional Generation Between Image and Text](../../ACL2026/llm_evaluation/minos_a_multimodal_evaluation_model_for_bidirectional_generation_between_image_a.md)

</div>

<!-- RELATED:END -->
