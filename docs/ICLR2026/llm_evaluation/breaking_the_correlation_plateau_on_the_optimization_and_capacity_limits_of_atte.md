---
title: >-
  [Paper Note] Breaking the Correlation Plateau: On the Optimization and Capacity Limits of Attention-Based Regressors
description: >-
  [ICLR 2026][LLM Evaluation][Pearson correlation coefficient] This paper provides the first theoretical analysis of the "PCC plateau" phenomenon observed when training attention-based regression models with a joint MSE+PCC objective. The root causes are identified as the conflict between MSE optimization and PCC gradients, together with an expressivity upper bound imposed by the convex aggregation of softmax. The authors propose the ECA (Extrapolative Correlation Attention) framework, which breaks through this limitation via three components: scaled residual aggregation, dispersion-aware temperature softmax, and dispersion-normalized PCC loss.
tags:
  - ICLR 2026
  - LLM Evaluation
  - Pearson correlation coefficient
  - attention regression
  - PCC plateau
  - convex aggregation
  - optimization dynamics
date: 2026-05-08
content_hash: b7485bc0562c7104
---

# Breaking the Correlation Plateau: On the Optimization and Capacity Limits of Attention-Based Regressors

**Conference**: ICLR 2026  
**arXiv**: [2602.17898](https://arxiv.org/abs/2602.17898)  
**Code**: Unavailable  
**Area**: LLM Evaluation  
**Keywords**: Pearson correlation coefficient, attention regression, PCC plateau, convex aggregation, optimization dynamics

## TL;DR
This paper provides the first theoretical analysis of the "PCC plateau" phenomenon observed when training attention-based regression models with a joint MSE+PCC objective. The root causes are identified as the conflict between MSE optimization and PCC gradients, together with an expressivity upper bound imposed by the convex aggregation of softmax. The authors propose the ECA (Extrapolative Correlation Attention) framework, which breaks through this limitation via three components: scaled residual aggregation, dispersion-aware temperature softmax, and dispersion-normalized PCC loss.

## Background & Motivation

**Background**: Attention mechanisms are widely applied to set-level regression tasks (e.g., computational pathology, video sentiment analysis, spatial transcriptomics), where each sample consists of multiple elements and element embeddings are aggregated via attention to predict a continuous target. The training objective typically adopts a joint MSE+PCC loss, simultaneously targeting the absolute magnitude and the rank/shape of predictions.

**Limitations of Prior Work**: A **PCC plateau** frequently occurs during training—PCC stops improving and flattens at an early stage, even as MSE continues to decrease. Increasing the PCC loss weight $\lambda_{\text{PCC}}$ fails to resolve this issue. The phenomenon is particularly severe when intra-sample data are highly homogeneous.

**Key Challenge**:
   - **Optimization perspective**: MSE optimization drives the predicted standard deviation $\sigma_{\hat{y}}$ toward the target standard deviation $\sigma_y$. Since the global scaling factor of the PCC gradient is proportional to $1/\sigma_{\hat{y}}$, the PCC gradient signal is suppressed as $\sigma_{\hat{y}}$ grows.
   - **Capacity perspective**: Softmax attention is a convex combination, constraining aggregated outputs to lie within the convex hull of intra-sample embeddings. The maximum achievable PCC improvement is therefore bounded by the convex hull radius.

**Key Insight**: The authors begin from the widely observed but never theoretically explained phenomenon of PCC stagnation. They provide rigorous analyses from both the optimization dynamics and model capacity perspectives, and design targeted solutions grounded in these analyses.

**Core Idea**: By theoretically revealing the dual bottleneck of PCC gradient suppression by MSE optimization and the convex hull constraint in attention regression, the paper proposes a three-pronged remedy—convex-hull extrapolation, dispersion-adaptive temperature, and gradient normalization—to break the plateau.

## Method

### Overall Architecture

ECA is a plug-and-play attention module that replaces standard softmax attention and is trained end-to-end. The input remains a set of element embeddings $\{\mathbf{h}_{si}\}$ per sample, and the output is the aggregated sample-level embedding $\mathbf{v}_s$, from which a scalar prediction $\hat{y}_s$ is obtained via a linear regression head.

The overall loss function is:
$$\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{MSE}} + \lambda_{\text{PCC}} \cdot \tilde{\mathcal{L}}_{\text{PCC}} + \mathcal{L}_{\gamma}$$

### Theoretical Analysis (Theoretical Foundation of the Method)

**Proposition 2.1 (MSE Decomposition)**: MSE can be decomposed into three terms: a mean-matching term, a standard-deviation-matching term, and a weighted correlation term. Since PCC is invariant to affine transformations while MSE is not, optimization primarily reduces MSE by adjusting the mean and scale, leaving limited room for correlation improvement.

**Theorem 2.1 (PCC Gradient)**: The gradient of PCC with respect to the attention logit $z_{si}$ contains a factor $1/\sigma_{\hat{y}}$ and a local structural factor $\alpha_{si} \mathbf{w}^\top (\mathbf{h}_{si} - \mathbf{v}_s)$.

**Corollary 2.1 (Gradient Ratio Decay)**: The RMS ratio of the PCC gradient to the MSE gradient decays at a rate of $O(1/\sigma_{\hat{y}}^{3/2})$.

**Theorem 2.2 (PCC Gain Upper Bound for Convex Aggregation)**: The improvement in PCC achievable by any convex aggregator (including softmax) relative to mean pooling is bounded by $2\tilde{R} / (\sigma_0/\|\mathbf{w}\|_2 - \tilde{R})$.

### Key Designs

1. **Scaled Residual Aggregation (SRA)**:

    - Function: Allows the aggregated embedding to exceed the convex hull, breaking the convex combination constraint.
    - Mechanism: A learnable scaling factor $\gamma_s \geq 1$ amplifies the residual beyond standard attention aggregation: $\mathbf{v}_s^{ECA} = \boldsymbol{\mu}_s + \gamma_s \sum_i \alpha_{si}(\mathbf{h}_{si} - \boldsymbol{\mu}_s)$, where $\gamma_s = 1 + \text{Softplus}(\text{MLP}(\boldsymbol{\mu}_s))$.
    - Design Motivation: Theorem 2.2 proves that the PCC improvement of convex aggregators is bounded by the convex hull radius. With $\gamma_s > 1$, the model can extrapolate along the residual direction, fundamentally bypassing the convex constraint.
    - Regularization: $\mathcal{L}_\gamma = \frac{\lambda_\gamma}{S} \sum_s (\gamma_s - 1)^2$ prevents excessive scaling.

2. **Dispersion-Aware Temperature Softmax (DATS)**:

    - Function: Adaptively adjusts the softmax temperature based on intra-sample dispersion.
    - Mechanism: $\tau_s = T_{\min} + \beta \sqrt{\frac{1}{n_s} \sum_i \|\mathbf{h}_{si} - \boldsymbol{\mu}_s\|^2}$. Homogeneous samples have low dispersion → lower temperature → small differences are amplified → more selective attention → larger residual $\Delta \mathbf{v}_s$, providing meaningful directions for SRA to amplify.
    - Design Motivation: When intra-sample embeddings are highly similar, standard softmax produces near-uniform weights $\alpha_{si} \approx 1/n_s$, causing residuals to approach zero and rendering SRA ineffective. DATS restores the discriminability of attention.

3. **Dispersion-Normalized PCC Loss (DNPL)**:

    - Function: Compensates for the $1/\sigma_{\hat{y}}$ decay in the PCC gradient.
    - Mechanism: $\tilde{\mathcal{L}}_{\text{PCC}} = \text{StopGrad}(\sigma_{\hat{y}}) \cdot (1 - \rho)$. Multiplying by $\sigma_{\hat{y}}$ cancels the $1/\sigma_{\hat{y}}$ factor in the gradient; StopGrad ensures the stationary points of the loss remain unchanged.
    - Design Motivation: Directly addresses the $O(1/\sigma_{\hat{y}}^{3/2})$ decay of the PCC/MSE gradient ratio identified in Corollary 2.1.

## Key Experimental Results

### Main Results

| Dataset / Model | Metric | Baseline | +ECA | Gain |
|----------------|--------|----------|------|------|
| Appliance (UCI) | PCC↑ | 0.556 | **0.598** | +0.042 |
| Appliance (UCI) | MSE↓ | 6.108 | **5.790** | −5.2% |
| Online News (UCI) | PCC↑ | 0.408 | **0.420** | +0.012 |
| Superconductivity (UCI) | PCC↑ | 0.920 | **0.930** | +0.010 |
| 10xProteomic (Pathology) | PCC@F↑ | 0.602 | **0.690** | +14.6% |
| 10xProteomic (Pathology) | PCC@M↑ | 0.629 | **0.716** | +13.8% |
| 10xProteomic (Pathology) | MSE↓ | 0.056 | **0.051** | −9.8% |
| MOSI (Sentiment Analysis) | PCC↑ | 0.783 | **0.806** | +2.3% |
| MOSI (Sentiment Analysis) | F1↑ | 0.851 | **0.859** | +0.8% |

### Ablation Study

| Configuration (Appliance) | MAE↓ | MSE↓ | PCC↑ |
|--------------------------|------|------|------|
| FT-Transformer (baseline) | 39.333 | 6.108 | 0.556 |
| +ECA (full) | **38.665** | **5.790** | **0.598** |
| +ECA w/o SRA | 39.208 | 5.994 | 0.575 |
| +ECA w/o DATS | 38.906 | 6.037 | 0.561 |
| +ECA w/o DNPL | 39.742 | 5.910 | 0.583 |

### Key Findings
- **All three components are indispensable**: Removing DATS causes the largest performance drop (PCC decreases from 0.598 to 0.561), indicating that temperature adaptation is critical for addressing homogeneity.
- **Synthetic data validation**: Across different homogeneity levels ($\tilde{\sigma} \in [0.10, 0.73]$), ECA achieves PCC gains of 4.80%, 5.76%, 4.68%, and 3.05% respectively, with simultaneous MSE improvements of 20.3%–66.7%.
- **Plateau is broken**: On pathology dataset fold 2, the PCC of the EGN baseline plateaus around epoch 4, whereas EGN+ECA continues to improve, ultimately achieving approximately 16.5% higher validation PCC.
- **Stronger homogeneity yields larger ECA gains**: Improvements are particularly prominent in highly homogeneous settings, such as spatial transcriptomics ($\tilde{\sigma}=0.068$) and video sentiment ($\tilde{\sigma}=0.098$).

## Highlights & Insights
- **Theory-driven method design**: Each component has a clear theoretical motivation—SRA addresses the convex hull constraint of Theorem 2.2, DATS corresponds to the dispersion term in Corollary 2.2, and DNPL targets the gradient decay in Corollary 2.1. This paradigm of "diagnose the root cause before designing the solution" is highly instructive.
- **Insight from MSE–PCC decomposition (Proposition 2.1)**: Decomposing MSE into mean-matching, standard-deviation-matching, and weighted correlation terms concisely reveals why a decrease in MSE does not imply an improvement in PCC.
- **Generality of the convex-hull extrapolation idea**: The transition from convex combination to scaled residual extrapolation in SRA is not limited to regression tasks; it can be transferred to any scenario employing softmax attention pooling (e.g., MIL, document classification), offering a new pathway to overcome expressivity bottlenecks.
- **Dispersion-adaptive temperature**: DATS dynamically adjusts temperature based on the intra-sample dispersion of each instance, providing greater flexibility than global temperature scheduling—particularly suited to heterogeneous datasets where different samples exhibit varying degrees of homogeneity.

## Limitations & Future Work
- The current theoretical analysis assumes a single-layer attention aggregator with a linear regression head; analysis of attention interactions in deep Transformers remains incomplete (though discussed in the appendix).
- The scaling factor $\gamma_s$ in SRA requires an additional MLP, introducing extra parameters and computational overhead.
- Experimental datasets are relatively small-scale (MOSI contains only approximately 2,200 video segments), and performance in large-scale settings remains to be validated.
- The choice of $\gamma_{\max}$ (e.g., set to 2) appears empirical, lacking an adaptive determination method.
- Future work could decouple the contributions of the three components into a sequential training strategy (e.g., warming up MSE first and then introducing DNPL) to examine their effects at different training stages.

## Rating
- Novelty: ⭐⭐⭐⭐ (First theoretical explanation of the PCC plateau with rigorous analysis)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Covers synthetic, UCI, pathology, and sentiment dimensions with complete ablations)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear theoretical derivations, polished figures, coherent narrative)
- Value: ⭐⭐⭐⭐ (Plug-and-play module with transferable theoretical insights)

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Spectral Attention Steering for Prompt Highlighting](spectral_attention_steering_for_prompt_highlighting.md)
- [\[ICLR 2026\] Mitigating Spurious Correlation via Distributionally Robust Learning with Hierarchical Ambiguity Sets](mitigating_spurious_correlation_via_distributionally_robust_learning_with_hierar.md)
- [\[ICLR 2026\] Soft Quality-Diversity Optimization](soft_quality-diversity_optimization.md)
- [\[ICLR 2026\] Prompt and Parameter Co-Optimization for Large Language Models](prompt_and_parameter_co-optimization_for_large_language_models.md)
- [\[CVPR 2026\] Anchoring and Rescaling Attention for Semantically Coherent Inbetweening](../../CVPR2026/llm_evaluation/anchoring_and_rescaling_attention_for_semantically_coherent_inbetweening.md)

<!-- RELATED:END -->
