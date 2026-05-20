---
title: >-
  [Paper Note] DP-LLM: Runtime Model Adaptation with Dynamic Layer-wise Precision Assignment
description: >-
  [NeurIPS 2025][Model Compression][Dynamic mixed precision] DP-LLM identifies that per-layer quantization sensitivity varies dynamically across decoding steps…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "Dynamic mixed precision"
  - "runtime adaptation"
  - "layer-wise quantization sensitivity"
  - "on-device LLM inference"
  - "relative error"
date: 2026-05-08
content_hash: 325978072bac2d41
---

# DP-LLM: Runtime Model Adaptation with Dynamic Layer-wise Precision Assignment

**Conference**: NeurIPS 2025
**arXiv**: [2508.06041](https://arxiv.org/abs/2508.06041)  
**Code**: [github.com/SNU-ARC/DP-LLM](https://github.com/SNU-ARC/DP-LLM)  
**Area**: Model Compression / Quantization
**Keywords**: Dynamic mixed precision, runtime adaptation, layer-wise quantization sensitivity, on-device LLM inference, relative error

## TL;DR
DP-LLM identifies that per-layer quantization sensitivity varies dynamically across decoding steps, and proposes a dynamic layer-wise precision selection mechanism based on relative error. At runtime, each layer is assigned a precision (h-bit or l-bit) conditioned on the current input, achieving a better performance–latency trade-off than static mixed-precision methods.

## Background & Motivation

**Background**: On-device LLM inference operates under strict computation, latency, and memory constraints. Multi-scale quantization approaches (e.g., Any-Precision LLM) enable memory-efficient runtime model adaptation by overlapping stored model variants of different bit-widths.

**Limitations of Prior Work**: (a) Uniform precision assignment (same bit-width across all layers) cannot support non-integer average precision (e.g., 3.5-bit), missing efficiency optimization opportunities; (b) Existing layer-wise mixed-precision methods (LLM-MQ, HAWQ-V2) adopt static assignment—once per-layer bit-widths are determined, they remain fixed throughout the entire decoding process.

**Key Challenge**: Per-layer quantization sensitivity is not a static property; it changes dynamically across decoding steps (token-by-token). Layers that require high precision at certain steps may need only low precision at others—static assignment cannot capture this dynamic behavior.

**Goal**: Dynamically assign precision to each layer at runtime (with an independent decision per decoding step) while incurring minimal inference latency overhead.

**Key Insight**: Use the difference between GEMV outputs computed with h-bit and l-bit weights (relative error $\|\Delta W x\|$) as a runtime-estimable proxy for quantization sensitivity.

**Core Idea**: At each decoding step, dynamically estimate the relative error of each layer conditioned on the current input; use high precision if the estimate exceeds a threshold, and low precision otherwise.

## Method

### Overall Architecture
DP-LLM consists of an offline preparation phase and a runtime selection phase. The offline phase determines the candidate precision set {h-bit, l-bit} and threshold $T$ for each layer. The runtime phase employs a lightweight precision selector to estimate the relative error and select the appropriate precision. The overall pipeline is: input token → per-layer precision selector estimates $\|\Delta W x\|$ → compare against threshold $T$ → use $W_h$ or $W_l$ → standard GEMV computation.

### Key Designs

1. **Three-Phase Offline Precision Configuration**:

    - **Phase 1 — Maximum Precision Selection**: Based on second-order Taylor expansion static sensitivity, integer programming is used under memory budget constraints to determine the maximum available precision $B[i]$ for each layer.
    - **Phase 2 — Average Precision Tuning**: The average precision $p_i$ of each layer is parameterized, and linear layers are replaced with $y = r W_l x + (1-r) W_h x$ (where $r = 1-(p_i - l)$). A regularization loss is introduced: $\mathcal{L}' = \mathcal{L} + \alpha(\sum_i p_i M_i / \sum_i M_i - b_{\text{targ}})^2$ to prevent $p$ values from collapsing to the highest precision. Only $\{p_i\}$ are updated, making this phase computationally inexpensive.
    - **Phase 3 — Threshold Derivation**: The distribution of $\|\Delta W_i x\|$ for each layer is characterized using a calibration set, and the $r_i$-quantile ($r_i = 1-(p_i-l)$) is taken as the threshold $T[i]$.

2. **Hybrid Relative Error Estimation**:

    - **Function**: Efficiently approximate $\|\Delta W x\|$ at runtime.
    - **Mechanism**: One of two estimators is selected per layer:
        - **Linear Regression** ($R^2 > R_{\text{th}}^2 = 0.9$): $\|\Delta W x\| \approx \|x\| \times \alpha + \beta$, with near-zero overhead. Approximately half of all layers satisfy this condition.
        - **Random Projection** (Johnson–Lindenstrauss Lemma): Precompute $G = A \Delta W$ (where $A$ is a $k \times n$ random matrix, $k=64$); at runtime, perform a low-dimensional GEMV $\|Gx\|$ to estimate the error. This guarantees less than 15% estimation error with 91% confidence.
    - **Design Motivation**: Exact computation of $\|\Delta W x\|$ requires an additional full GEMV operation, making it infeasible. The hybrid approach achieves the best balance between accuracy and efficiency.

3. **Asynchronous Estimation**:

    - **Function**: Move relative error estimation off the critical inference path.
    - **Mechanism**: Exploiting the property that activations change slowly across adjacent decoding steps due to Transformer residual connections, layers directly connected to the residual stream (query/key/value/up-projection, etc.) use the residual output from the previous step for estimation.
    - **Design Motivation**: Estimation latency can be overlapped with the computation of other layers, further reducing overhead.

### Loss & Training
Offline tuning updates only the $\{p_i\}$ parameters (not model weights), using 1,000 C4 samples of 512 tokens each, with minimal computational cost. The regularization coefficient $\alpha$ controls the deviation between the realized and target average precision.

## Key Experimental Results

### Main Results (Llama-3-8B, 5-bit memory budget)

| Target Precision | Method | WikiText2 PPL↓ | C4 PPL↓ | GSM8K↑ | BBH↑ | MATH↑ |
|---------|------|---------------|---------|--------|------|-------|
| 3.25-bit | LLM-MQ | 7.62 | 12.01 | 33.1 | 43.9 | 11.2 |
| 3.25-bit | HAWQ-V2 | 7.47 | 11.77 | 37.7 | 44.5 | 10.0 |
| 3.25-bit | **DP-LLM** | **7.35** | **11.57** | 36.7 | **46.3** | 10.6 |
| 4.00-bit | LLM-MQ | 7.07 | 11.21 | 38.8 | 47.5 | 11.2 |
| 4.00-bit | HAWQ-V2 | 6.83 | 10.76 | 43.7 | 49.1 | 13.4 |
| 4.00-bit | **DP-LLM** | **6.59** | **10.25** | 42.8 | 48.5 | **12.8** |
| 4.75-bit | LLM-MQ | 6.73 | 10.66 | 41.3 | 48.3 | 11.8 |
| 4.75-bit | HAWQ-V2 | 6.44 | 10.06 | 45.2 | 50.0 | 14.4 |
| 4.75-bit | **DP-LLM** | **6.37** | **9.89** | **46.9** | **50.6** | **14.8** |

### Latency Overhead Analysis

| Hardware Platform | Model | Mean Latency Overhead |
|---------|------|------------|
| Jetson Orin AGX | Llama-3-8B | 3.12% |
| RTX 4060 Ti | Llama-3-8B | 0.68% |
| Jetson Orin AGX | Phi-3-Medium | 1.32% |
| RTX 4060 Ti | Phi-3-Medium | 0.50% |

The PPL difference between the approximate and exact estimators is less than 0.03 (Llama-3-8B, WikiText2), validating the effectiveness of the approximation.

### Key Findings
- DP-LLM outperforms static mixed-precision methods (LLM-MQ, HAWQ-V2) across nearly all datasets, models, and precision configurations.
- The largest gains occur in the low-precision regime (3.25–4.0 bit), where dynamic selection is most valuable.
- Latency overhead is minimal (geometric mean < 1% on RTX 4060 Ti), making the precision selector virtually free.
- Approximately half of all layers exhibit a strong linear relationship between $\|x\|$ and $\|\Delta W x\|$, enabling linear regression estimation.

## Highlights & Insights
- **Observation of Dynamic Layer-wise Sensitivity**: This is an important correction to the conventional static mixed-precision assumption—quantization sensitivity is not an intrinsic property of a layer, but changes dynamically with the input. This observation naturally motivates the necessity of dynamic precision assignment.
- **Relative Error as a Runtime-Estimable Proxy**: The high-dimensional quantization sensitivity problem is reduced to a scalar comparison ($\|\Delta W x\|$ vs. threshold $T$), making runtime decision-making extremely lightweight.
- **Engineering Elegance of the Hybrid Estimation Strategy**: Approximately half of the layers use near-free linear regression, while the other half use random projection based on the Johnson–Lindenstrauss lemma, balancing accuracy and efficiency.

## Limitations & Future Work
- The candidate precision set is restricted to two choices (h-bit and l-bit); finer-grained multi-precision selection could yield further improvements.
- Asynchronous estimation assumes that activations change slowly between adjacent decoding steps, which may not hold in long sequences or scenarios with abrupt activation shifts.
- Validation is limited to weight-only quantization; dynamic precision assignment for weight-activation co-quantization remains unexplored.
- The calibration set relies on C4; generalization to other domain distributions has not been verified.
- Experiments cover only Llama-3-8B and Phi-3-Medium; performance on larger-scale models (70B+) is unknown.

## Related Work & Insights
- **vs. Any-Precision LLM**: The foundational multi-scale quantization framework employs uniform precision assignment. DP-LLM builds upon it by introducing a dynamic layer-wise selection mechanism.
- **vs. LLM-MQ**: Uses weight gradients to estimate static sensitivity $\Delta\mathcal{L} \approx g^T \Delta W$; DP-LLM instead makes dynamic decisions based on the runtime relative error with respect to the current input.
- **vs. HAWQ-V2**: Uses second-order Hessian information for static precision assignment; DP-LLM demonstrates that static analysis fails to capture sensitivity variations during the decoding phase.

## Rating
- Novelty: ⭐⭐⭐⭐ The observation of dynamic layer-wise precision and the proposed mechanism are novel and practically motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple models, datasets, and precision levels, with latency analysis and approximation validation.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and a well-structured three-phase methodology presented in a logical progression.
- Value: ⭐⭐⭐⭐ Directly applicable to on-device LLM deployment; the dynamic precision idea is broadly generalizable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Quantization Error Propagation: Revisiting Layer-Wise Post-Training Quantization](quantization_error_propagation_revisiting_layer-wise_post-training_quantization.md)
- [\[ACL 2026\] Adaptive Layer Selection for Layer-Wise Token Pruning in LLM Inference](../../ACL2026/model_compression/adaptive_layer_selection_for_layer-wise_token_pruning_in_llm_inference.md)
- [\[NeurIPS 2025\] Beyond Higher Rank: Token-wise Input-Output Projections for Efficient Low-Rank Adaptation](beyond_higher_rank_token-wise_input-output_projections_for_efficient_low-rank_ad.md)
- [\[NeurIPS 2025\] Less is More but Where: Dynamic Token Compression via LLM-Guided Keyframe Prior](less_is_more_but_where_dynamic_token_compression_via_llm-guided_keyframe_prior.md)
- [\[ACL 2026\] A Layer-wise Analysis of Supervised Fine-Tuning](../../ACL2026/model_compression/a_layer-wise_analysis_of_supervised_fine-tuning.md)

</div>

<!-- RELATED:END -->
