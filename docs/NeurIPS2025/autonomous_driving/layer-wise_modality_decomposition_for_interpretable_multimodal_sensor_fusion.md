---
title: >-
  [Paper Note] Layer-wise Modality Decomposition for Interpretable Multimodal Sensor Fusion
description: >-
  [NeurIPS 2025][Autonomous Driving][interpretability] This paper proposes LMD (Layer-Wise Modality Decomposition), a post-hoc, model-agnostic interpretability method that linearizes neural network operations layer by laye…
tags:
  - "NeurIPS 2025"
  - "Autonomous Driving"
  - "interpretability"
  - "sensor fusion"
  - "modality decomposition"
  - "LRP"
date: 2026-05-08
content_hash: 795e62dd85f723ab
---

# Layer-wise Modality Decomposition for Interpretable Multimodal Sensor Fusion

**Conference**: NeurIPS 2025
**arXiv**: [2511.00859](https://arxiv.org/abs/2511.00859)  
**Code**: N/A  
**Area**: Autonomous Driving
**Keywords**: interpretability, sensor fusion, autonomous driving, modality decomposition, LRP

## TL;DR

This paper proposes LMD (Layer-Wise Modality Decomposition), a post-hoc, model-agnostic interpretability method that linearizes neural network operations layer by layer to exactly decompose the predictions of multimodal fusion models into per-sensor modality contributions. LMD is the first method to achieve prediction attribution to individual input modalities in autonomous driving perception models, and its effectiveness is validated across camera-radar, camera-LiDAR, and camera-radar-LiDAR fusion settings.

## Background & Motivation

1. **Lack of interpretability in multi-sensor fusion**: Camera, radar, and LiDAR fusion improves perception performance in autonomous driving, but the intertwined modality information after fusion makes it difficult to determine each sensor's contribution to predictions, hindering fault diagnosis and system validation.

2. **Safety-critical scenarios demand transparent decision-making**: Even a single perception error in autonomous driving can lead to catastrophic consequences, necessitating a clear understanding of the model's decision basis—particularly which sensor dominates critical predictions.

3. **Existing interpretability methods struggle with multimodal fusion**: Intrinsically interpretable models (GAM/NAM) cannot capture cross-modal interactions due to structural constraints; local surrogate models (LIME/LORE) cannot handle high-dimensional dependencies; fANOVA scales poorly to high-dimensional multimodal settings.

4. **Attribution methods such as LRP do not account for modality separation**: Existing LRP/DTD methods attribute predictions to input pixel level but do not perform modality-level decomposition for multimodal fusion scenarios, and cannot answer questions such as "does this detection primarily rely on camera or radar?"

5. **Post-hoc methods that require no architectural modification are needed**: High-performance fusion model architectures should not be modified for interpretability; a post-hoc analysis tool that does not affect the original model's performance is required.

6. **Exponential computational cost of Shapley-based methods**: Shapley-value-based attribution requires $O(2^M)$ forward passes to enumerate all modality subsets, which is infeasible as the number of modalities grows, and is further limited by the modality independence assumption.

## Method

### Overall Architecture: Modality Decomposition via Layer-wise Linearization

- **Function**: Each nonlinear operation in a pretrained multimodal fusion model is linearized layer by layer, transforming the entire network into a linear system and enabling the application of the superposition principle to exactly decompose the output into independent contributions from each modality.
- **Why**: For a linear system, $F(x_c, x_r) = F(x_c, 0) + F(0, x_r) + F(0, 0)$ (superposition principle), meaning each modality's contribution can be precisely extracted by zeroing out the others, with the decomposition guaranteed to be complete (the three terms sum to the original output).
- **How**:
  1. **First forward pass**: Run the original model with the full multimodal input and record the behavior of each nonlinear operation (activation ratio of input to output, variance of LayerNorm).
  2. **Second forward pass**: Replace the original nonlinear layers with their linearized counterparts and propagate each modality's input separately to obtain modality-specific features.
  3. Final output = camera contribution + radar contribution + bias contribution, strictly equal to the original model output.

### Key Design 1: Activation Layer Linearization

- **Function**: Replace nonlinear activation functions such as ReLU with element-wise multiplication operations, where the ratio coefficients are the output/input ratios recorded during the first forward pass.
- **Why**: The nonlinearity of ReLU violates the superposition principle (i.e., $f(a+b) \neq f(a) + f(b)$); linearization is necessary for modality decomposition. Using the runtime ratio rather than gradients precisely preserves the functional behavior of the original model.
- **How**: For ReLU, the ratio degenerates to a binary mask $\{0, 1\}$ indicating neuron activation; for general activation functions, $c_j^l = F_j^l(x_c, x_r) / (F_j^{l-1}(x_c, x_r) + \varepsilon)$, i.e., the slope of the secant connecting two operating points.

### Key Design 2: Normalization Layer Linearization and Bias Allocation Strategy

- **Function**: Decompose BatchNorm and LayerNorm into modality-specific terms and bias terms, and achieve modality separation via different bias allocation rules (identity/uniform/ratio).
- **Why**: The bias terms in normalization layers (mean subtraction and affine $\beta$) do not belong to any specific modality and must be allocated appropriately to maintain separation; LayerNorm's variance depends on the current input and requires special handling.
- **How**:
    - **BatchNorm**: Statistics are fixed after training; the identity rule is applied to uniformly assign the bias to the bias term ($\delta_c = \delta_r = 0,\ \delta_b = 1$).
    - **LayerNorm**: Variance is cached as a constant from the first forward pass; the ratio rule is applied—each modality is independently centered by computing its mean over the spatial dimension.
    - Experiments confirm that the identity (BN) + ratio (LN) combination achieves the best modality separation.

### Key Design 3: Perturbation-based Evaluation Metrics

- **Function**: Quantitative evaluation metrics based on modality replacement are proposed—one modality's input is replaced with an unrelated sample, and it is checked whether the perturbed modality's prediction changes and whether the unperturbed modality's prediction remains stable.
- **Why**: Standard methods for evaluating modality decomposition quality in fusion networks are lacking; new quantitative metrics are needed to measure whether the separation property is satisfied.
- **How**: Pearson correlation and MSE are computed for each modality's predictions before and after perturbation. Ideally, the perturbed modality should have low correlation (large prediction change), while the unperturbed modality should have correlation of 1 (prediction unchanged).

## Key Experimental Results

### Experiment 1: Linearization Effectiveness and Modality Separation Evaluation

| Method | Activation | Normalization | Rp/R Corr↓ | Rp/C Corr↑ | Cp/R Corr↑ | Cp/C Corr↓ |
|--------|-----------|--------------|-----------|-----------|-----------|-----------|
| Baseline | ✗ | ✗ | 0.22 | 0.76 | 0.22 | 0.09 |
| Activation only | ✓ | ✗ | 0.56 | 0.80 | 0.22 | 0.12 |
| Normalization only | ✗ | ✓ | 0.50 | 0.99 | 0.93 | 0.38 |
| **LMD (Ratio)** | **✓** | **✓** | **0.05** | **1.00** | **1.00** | **0.15** |

**Key Findings**:
- LMD achieves near-perfect modality separation after jointly linearizing both activation and normalization layers: after perturbing radar, the camera prediction remains completely unchanged (correlation = 1.00), while the radar prediction changes drastically (correlation = 0.05).
- Linearizing either class of layers alone is insufficient for effective separation; both are indispensable.
- LMD is equally effective in the LiDAR+Camera setting (Rp/R = 0.09, Cp/C = 0.44).

### Experiment 2: LMD Variant Comparison

| BN-LN Rule | Rp/R↓ | Rp/C↑ | Cp/C↓ |
|-----------|-------|-------|-------|
| Uniform-Identity | 0.50 | 1.00 | 0.42 |
| Identity-Identity | 0.15 | 1.00 | 0.38 |
| Identity-Uniform | 0.18 | 1.00 | 0.38 |
| **Identity-Ratio** | **0.05** | **1.00** | **0.15** |

**Key Findings**:
- The Identity (BN) + Ratio (LN) combination achieves the best performance across all separation metrics.
- The ratio rule is critical for LayerNorm; applying the identity or uniform rule to LN leads to insufficient modality separation.
- Different BN rules have a smaller impact on results; the key bottleneck lies in the strategy for handling LN.

### Experiment 3: LMD + SHAP Combination

| Setting | Metric | SHAP | LMD + SHAP |
|---------|--------|------|-----------|
| Radar+Camera | Rp/C↑ | 0.69 | **0.94** |
| Radar+Camera | Cp/R↑ | 0.67 | **0.89** |
| LiDAR+Camera | Lp/C↑ | 0.71 | **0.92** |
| LiDAR+Camera | Cp/L↑ | 0.72 | **0.91** |

**Key Findings**:
- LMD+SHAP achieves significantly better modality separation than SHAP alone, with Pearson correlation improving from 0.67–0.72 to 0.89–0.94.
- Improvements are equally notable in the three-modality setting, demonstrating that LMD is complementary to other attribution methods.
- The two-stage strategy of decomposition followed by attribution outperforms direct attribution.

## Highlights & Insights

- **Novelty**: To the best of the authors' knowledge, this is the first method to achieve modality-level prediction attribution in autonomous driving sensor fusion, filling an important gap.
- **Theoretical rigor**: Layer-wise decomposition formulas are derived based on first-order Taylor expansion, with proofs that the linearized system satisfies the functional preservation constraint and the separation property; the mathematical derivation is complete.
- **Fully post-hoc and model-agnostic**: Only two forward passes are required; the original model architecture and weights are not modified, and model performance is unaffected.
- **Computationally efficient**: $O(1)$ additional memory and $O(1)$ additional computation (2 forward passes), far superior to Shapley-based methods at $O(2^M)$.

## Limitations & Future Work

- **First-order approximation error**: Taylor expansion neglects higher-order terms, and the decomposition may be insufficiently accurate for highly nonlinear network regions.
- **Bilinear terms in attention mechanisms**: Cross-modal bilinear interaction terms arising from $QK^T$ in attention modules cannot be fully attributed to a single modality, which is an inherent limitation of linear decomposition.
- **Bias terms contain interaction information**: The bias term contains not only constant offsets but also higher-order interaction approximations generated by linearization, which may obscure part of the modality interaction information.
- **Validation limited to BEV perception models**: Experiments are primarily conducted on BEV fusion architectures such as SimpleBEV; applicability to Transformer-heavy fusion architectures (e.g., pure attention-based fusion) requires further validation.

## Related Work & Insights

| Dimension | Ours (LMD) | LRP (Bach et al., 2015) |
|-----------|-----------|------------------------|
| Attribution granularity | Modality-level (camera/radar/LiDAR) | Pixel/feature level |
| Decomposition target | Modality contributions in multimodal fusion models | Input variable contributions in unimodal models |
| Forward passes | 2 forward passes | 2 forward + 1 backward |
| Separation guarantee | Yes (theoretically proven) | No (not designed for multimodal settings) |
| Applicable scenario | Multi-sensor fusion | General deep networks |

| Dimension | Ours (LMD) | SHAP (Shapley-based) |
|-----------|-----------|---------------------|
| Computational complexity | $O(1)$ (2 forward passes) | $O(2^M)$ (enumerate all modality subsets) |
| Memory consumption | $O(1)$ | $O(1)$ |
| Independence assumption | None | Feature independence assumption |
| Composability | Can be combined with SHAP for improved performance | Used independently |
| Modality separation quality | Strong (perturbation correlation 0.94–1.00) | Moderate (perturbation correlation 0.67–0.71) |

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First application of layer-wise decomposition to modality attribution in multimodal fusion, introducing a new analytical perspective and evaluation metrics.
- **Technical Depth**: ⭐⭐⭐⭐ — Linearization schemes for activation layers, BatchNorm, and LayerNorm are each derived and proven to preserve original functionality, forming a complete theoretical framework; however, the core idea (linearization + superposition) is relatively straightforward.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers radar-camera, LiDAR-camera, and three-modality fusion configurations with newly proposed evaluation metrics; however, validation is primarily conducted on a single baseline architecture (SimpleBEV).
- **Value**: ⭐⭐⭐⭐ — Directly applicable to autonomous driving safety auditing and fault diagnosis with minimal computational overhead; code is open-sourced.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Availability-aware Sensor Fusion via Unified Canonical Space](availability-aware_sensor_fusion_via_unified_canonical_space.md)
- [\[ICCV 2025\] Self-Supervised Sparse Sensor Fusion for Long Range Perception](../../ICCV2025/autonomous_driving/self-supervised_sparse_sensor_fusion_for_long_range_perception.md)
- [\[NeurIPS 2025\] DBLoss: Decomposition-based Loss Function for Time Series Forecasting](dbloss_decomposition-based_loss_function_for_time_series_forecasting.md)
- [\[ICLR 2026\] SiMO: Single-Modality-Operable Multimodal Collaborative Perception](../../ICLR2026/autonomous_driving/simo_single-modality-operable_multimodal_collaborative_perceptio.md)
- [\[NeurIPS 2025\] Extremely Simple Multimodal Outlier Synthesis for Out-of-Distribution Detection and Segmentation](extremely_simple_multimodal_outlier_synthesis_for_out-of-distribution_detection_.md)

</div>

<!-- RELATED:END -->
