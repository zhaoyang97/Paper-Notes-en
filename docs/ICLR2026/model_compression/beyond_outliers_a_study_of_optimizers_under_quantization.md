---
title: >-
  [Paper Note] Beyond Outliers: A Study of Optimizers Under Quantization
description: >-
  [ICLR 2026][Model Compression][Quantization] The authors provide the first systematic study of the relationship between "optimizer selection" and "quantization robustness." Training 50M–1.5B LLMs with six different optimizers, they find that traditional outlier metrics (MMR, Kurtosis) fail to predict post-quantization accuracy. Instead, they propose an analytical
tags:
  - ICLR 2026
  - Model Compression
  - Quantization
date: 2026-05-08
content_hash: c5bcc53225419474
---
# Beyond Outliers: A Study of Optimizers Under Quantization

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=mVldAuDAn5](https://openreview.net/forum?id=mVldAuDAn5)  
**Code**: TBD  
**Area**: Model Compression  
**Keywords**: Quantization, Optimizer, Error Propagation, Outliers, Scaling Laws

## TL;DR
The authors provide the first systematic study of the relationship between "optimizer selection" and "quantization robustness." Training 50M–1.5B LLMs with six different optimizers, they find that traditional outlier metrics (MMR, Kurtosis) fail to predict post-quantization accuracy. Instead, they propose an analytical ABC error propagation decomposition and a new metric $R_L$, revealing the counter-intuitive conclusion that Shampoo, despite having the most severe outliers, exhibits the least accuracy drop under PTQ/QAT and the highest parameter efficiency.

## Background & Motivation
**Background**: New optimizers (Muon, Shampoo, SOAP, PSGD, Scion) are emerging rapidly, and quantization (PTQ and QAT) has become standard for LLM deployment. However, these two research lines have remained largely decoupled: optimizer benchmarks usually compare models in full precision, while quantization studies typically default to training with AdamW. No systematic investigation has questioned whether changing the optimizer alters a model's quantization robustness.

**Limitations of Prior Work**: The difficulty of joint quantization (compressing both weights and activations to low bits) is widely attributed to **outlier features** in the input matrices. The community has developed various metrics to quantify this outlier degree—such as the Max-to-Median Ratio (MMR) and Kurtosis—and corresponding "de-outliering" methods like rotations or architectural modifications. However, these metrics are "static" measures of single-layer outliers; no one has verified how training optimizers affect these outliers or if more outliers necessarily lead to greater accuracy drops.

**Key Challenge**: Folklore suggests "larger outliers $\to$ worse quantization collapse," but this empirical rule has never been tested across the dimension of different optimizers. Upon investigation, the authors find this rule fails: the optimizer with the most severe outliers is actually the most quantization-robust. This indicates that using single-layer outlier metrics to predict network-wide quantization error fundamentally ignores **how errors accumulate, amplify, and propagate across layers**.

**Goal**: The study addresses three progressive questions: (1) Do models trained to the same validation loss with different optimizers perform identically after PTQ? (2) How sensitive is QAT to optimizer choice, and does the best full-precision optimizer remain the best under quantization? (3) Can these findings be extrapolated to larger models?

**Key Insight**: Rather than focusing on single-layer outliers, one should treat quantization error as a signal that **evolves layer-by-layer during the forward pass**, analytically tracking its propagation from layer 1 to layer $L$. This perspective naturally explains why single-layer metrics fail.

**Core Idea**: A provable ABC error decomposition (breaking per-layer quantization error into "accumulated error + local error + interaction") is used to replace static outlier metrics. This yields a new metric $R_L$ that accurately predicts quantization drops and reveals the intrinsic differences in error propagation across optimizers.

## Method

### Overall Architecture
This is an analysis-oriented work. The "method" is not a new model but rather a **unified experimental testbed + a theoretical toolset for error propagation** used to explain counter-intuitive experimental phenomena. The research pipeline involves: training six optimizers to their respective optima in full precision $\to$ performing PTQ on models trained to the same loss level $\to$ introducing the ABC decomposition and the $R_L$ metric to explain why traditional metrics fail $\to$ performing QAT and fitting scaling laws to confirm results extrapolate to larger scales.

The experimental foundation is the OLMo2 architecture (RoPE, RMSNorm, QKNorm, re-ordered pre-norm, input-output weight tying, and ReLU² activations), covering six sizes (50M / 125M / 350M / 500M / 760M / 1.5B), all trained on ClimbMix (400B high-quality tokens) following Chinchilla-optimal ratios. PTQ uses 4-bit symmetric AbsMax per-row round-to-nearest (W4A4 for linear layers); QAT uses the SOTA QuEST method (Hadamard transforms for inputs/weights + optimal clipping ratios).

### Key Designs

**1. Fair Full-Precision Baseline Protocol: Optimizing each optimizer before quantization**

To compare the impact of optimizers on quantization, no optimizer should be disadvantaged by poor tuning. The authors designed a two-stage tuning process: first, a **one-dimensional sweep** for each optimizer's hyper-parameters on a 50M model; second, sweeping 8 learning rates for every "model $\times$ optimizer" combination and training to completion. For the 1.5B scale, only AdamW (baseline), Muon (best full-precision), and Shampoo (most quantization-robust) were analyzed.

To make PTQ comparisons meaningful, the authors introduced **Common Loss (CL)**: the lowest validation loss achievable by all optimizers within a 20$\times$ token/parameter ratio budget. All models were trained to this CL before PTQ—ensuring that pre-quantization performance was equal and post-quantization differences reflected purely "quantization robustness." Full-precision results showed Muon as the strongest and having the lowest MMR, seemingly supporting the "low outliers = good" intuition, setting the stage for the later reversal.

**2. ABC Error Decomposition: Decomposing relative quantization error into three terms**

This is the theoretical core of the paper. A network is viewed as a series of $L$ modules $f_\ell(\cdot)$, where activations are $h_\ell = f_\ell(h_{\ell-1})$. After quantization, activations become $h^q_\ell = f^q_\ell(h^q_{\ell-1})$. Quantization introduces **two changes**: the input changes from $h_{\ell-1}$ to $h^q_{\ell-1}$ (propagated error), and the function changes from $f_\ell$ to $f^q_\ell$ (local perturbation). The authors prove the activation difference $\Delta h_\ell := h^q_\ell - h_\ell$ can be written as $\Delta h_\ell = a_\ell + b_\ell$, where $a_\ell$ is the average input change (previous error) and $b_\ell$ is the average function change (local error).

Taking the relative L2 norm squared and using the law of cosines yields the exact decomposition:

$$R_\ell := \left(\frac{\|\Delta h_\ell\|}{\|h_\ell\|}\right)^2 = A_\ell + B_\ell + C_\ell,\quad A_\ell = \left(\frac_{\|a_\ell\|}{\|h_\ell\|}\right)^2,\ B_\ell = \left(\frac{\|b_\ell\|}{\|h_\ell\|}\right)^2,\ C_\ell = \frac{2\langle a_\ell, b_\ell\rangle}{\|h_\ell\|^2}$$

$A_\ell$ is the **accumulated and amplified** error from previous layers, $B_\ell$ is the **newly introduced** error (what traditional outlier metrics attempt to characterize), and $C_\ell$ is the interaction term. Measurements show that $A_\ell$ is much larger than $B_\ell$ and $C_\ell$ in most cases. This explains the counter-intuitive phenomenon: **even if MMR predicts the local error $B_\ell$ well, the total network error $R_\ell$ is dominated by the accumulated error $A_\ell$**. Thus, single-layer outliers ($\approx B_\ell$) cannot predict final degradation. Regular "dips" in $R_\ell$ correspond to RMSNorm layers, which attenuate rather than amplify errors.

**3. New Metric $R_L$ and Gain Decomposition: Predicting degradation at the network end**

Since $R_\ell$ tracks the layer-wise deviation of the quantized network, the **terminal relative error $R_L$** should correlate strongly with loss degradation. The correlation between $R_L$ and post-PTQ average zero-shot accuracy is $\rho = -0.89$, far exceeding MMR ($\rho=0.62$) and Kurtosis ($\rho=0.70$). $R_L$ requires four forward passes per sample, which is more expensive than MMR but significantly more accurate.

Furthermore, to understand how a module transforms previous error $R_{\ell-1}$ into accumulated error $A_\ell$, the **Gain** is defined as $G_\ell := A_\ell / R_{\ell-1}$. For linear layers, gain decomposes into $G_\ell = G_{1,\ell}G_{2,\ell}$: the **Spectral Ratio** $G_{1,\ell}$ measures how quantization changes the weight spectral norm, and the **Alignment Ratio** $G_{2,\ell}$ measures the relative alignment of activation changes with quantized weights versus original activations with original weights. Analysis shows Spectral Ratios are near 1 for all optimizers, meaning Gain is dominated by the Alignment Ratio. This attributes Muon's poor quantization robustness to a mechanism: **Muon has the highest linear layer gain** (strongest error amplification). In contrast, AdamW and Shampoo have the lowest gains.

### Loss & Training
QAT uses QuEST for 4-bit forward quantization and Straight-Through Estimators (STE) for gradients. Each "model $\times$ optimizer" pair is trained with its optimal full-precision hyper-parameters on a 20$\times$ token/parameter budget. Scaling laws follow the Hoffmann et al. (2022) form, incorporating **per-optimizer parameter efficiency** $\rho$ (where full-precision $\rho=1$ and $\rho_{4bit}$ denotes 4-bit QAT efficiency):

$$L = \frac{A'}{(N\cdot\rho)^\alpha} + E$$

This implies a $N$-parameter model trained with 4-bit QAT is equivalent to a $\rho_{4bit}\cdot N$ parameter model trained in full precision.

## Key Experimental Results

### Main Results

Average zero-shot accuracy (PIQA + HellaSwag + ARC-Easy) after PTQ (W4A4). All models are trained to the same CL prior to quantization:

| Model | AdamW | Muon | Shampoo | Observation |
|------|-------|------|---------|------|
| 350M | 49.23 | 47.42 | **53.93** | Shampoo leads Muon by 6.5 pts |
| 500M | 55.17 | 50.60 | **55.65** | Muon lags significantly |
| 760M | 59.22 | 50.00 | **59.26** | Muon drops ~14.6 pts from FP |
| 1.5B | 62.51 | 47.75 | **63.88** | Muon collapses to 47.75 |

Counter-intuitive point: Shampoo has the highest MMR (outliers) but is the most quantization-robust; Muon has the lowest MMR but collapses most severely.

### Ablation Study

Correlation between outlier/error metrics and post-PTQ accuracy (760M):

| Metric | Correlation $\rho$ | Predictive Power |
|------|----------------------|--------------------|
| MMR (Max-to-Median Ratio) | 0.62 | Weak |
| Kurtosis | 0.70 | Weak |
| **$R_L$ (Ours)** | **−0.89** | Strong |

Accuracy degradation for QAT (4-bit QuEST) relative to full-precision baseline (drop in brackets, smaller is better):

| Model | AdamW | Muon | Shampoo |
|------|-------|------|---------|
| 760M | 62.22 (−2.63) | 62.32 (−3.57) | 62.76 (**−0.46**) |
| 1.5B | 66.82 (−1.63) | 67.08 (−2.11) | 67.34 (**−1.20**) |

Scaling law parameter efficiency $\rho_{4bit}$: Shampoo is highest (0.879), followed by AdamW (0.863), Scion (0.856), Muon (0.852), and PSGD (0.739). Optimizers that perform best in PTQ are also the most parameter-efficient in QAT.

### Key Findings
- **Failure of the Outlier Paradigm**: Single-layer metrics like MMR/Kurtosis cannot predict PTQ performance across different optimizers because they only capture local error $B_\ell$, while total error is dominated by accumulated error $A_\ell$.
- **Optimizer-Specific Error Propagation**: $R_\ell$ in AdamW and Scion networks shows sharp peaks at the end, while PSGD and Muon are "flatter." Muon’s high linear layer gain explains its poor quantization despite low outliers.
- **Full-Precision Ranking $\neq$ Quantization Ranking**: Muon is strongest in full precision but worst under quantization; Shampoo is mediocre in full precision but superior in PTQ, QAT, and parameter efficiency.

## Highlights & Insights
- **Turning "Heuristics" into Falsifiable Science**: The authors provide a rigorous decomposition $\Delta h_\ell = a_\ell + b_\ell$, using the law of cosines to prove $R_\ell = A_\ell+B_\ell+C_\ell$. This explains why outlier metrics fail—the ABC framework serves as a reusable diagnostic tool for error attribution in any module or PTQ scheme.
- **$R_L$ as a Practical Quantization Probe**: With only four forward passes, the correlation jumps from ~0.6 to 0.89. This allows practitioners to predict which checkpoint will be more robust before actually performing quantization.
- **The Shampoo Revelation**: The discovery that the most outlier-heavy optimizer is the most quantization-robust directly overturns the community's "de-outliering = accuracy preservation" assumption, shifting focus from "suppressing outliers" to "controlling error propagation gain $G_\ell$."
- **Gain Perspective Points to New Optimizers**: Since PSGD/Muon are already solving constrained optimization problems, the authors suggest adding a "$G_\ell$ must be small" constraint to potentially train "quantization-native" optimizers.

## Limitations & Future Work
- **Limited Bit-width Coverage**: Only 4-bit quantization was tested. Conclusions for 8-bit, 6-bit, or other formats like micro-scaling are unknown.
- **Scope of Gain Decomposition**: The $G_\ell = G_{1,ell}G_{2,ell}$ decomposition is only derived for linear layers. MHSA was treated as a single unit, limiting understanding of attention module error behavior.
- **Limited PTQ Schemes**: While RTN was the primary method, the behavior of the ABC decomposition under more SOTA PTQ schemes remains to be verified.
- **Optimizer Subset**: PSGD, Scion, and SOAP were not scaled to 1.5B due to costs, meaning the largest scale conclusions rely primarily on AdamW, Muon, and Shampoo.
- **Future Directions**: Directly using $R_L$ as a training-time regularizer or embedding $G_\ell$ constraints into optimizer update rules to actively train quantization-robust models.

## Related Work & Insights
- **vs. Outlier-based Methods (QuaRot, SmoothQuant)**: These focus on suppressing outliers (MMR/Kurtosis). This paper proves single-layer outliers correspond to a negligible $B_\ell$, whereas accumulated error $A_\ell$ dominates, suggesting "de-outliering" may not be the most direct path to accuracy.
- **vs. Optimizer Benchmarks (Wen et al. 2025)**: Previous work compared optimizers purely in full precision, finding Muon-like methods superior. This paper adds the quantization dimension, revealing that full-precision rankings can invert.
- **vs. QAT Scaling Laws (Kumar et al. 2024)**: This work adopts the parameter efficiency $\rho$ concept but makes it **per-optimizer**, revealing that Shampoo achieves the highest efficiency in 4-bit regimes.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic study of optimizer $\times$ quantization interaction; overturns outlier paradigm with provable ABC decomposition.
- Experimental Thoroughness: ⭐⭐⭐⭐ Broad coverage of sizes and optimizers, though 1.5B is limited to 3 optimizers and only 4-bit is tested.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear mapping between theoretical derivations and experimental phenomena with logical closure.
- Value: ⭐⭐⭐⭐⭐ Provides a practical predictive metric ($R_L$) and identifies a new direction for quantization-friendly optimizers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A Granular Study of Safety Pretraining under Model Abliteration](../../NeurIPS2025/model_compression/a_granular_study_of_safety_pretraining_under_model_abliteration.md)
- [\[CVPR 2026\] TWEO: Transformers Without Extreme Outliers Enables FP8 Training And Quantization For Dummies](../../CVPR2026/model_compression/tweo_transformers_without_extreme_outliers_enables_fp8_training_and_quantization.md)
- [\[ICLR 2026\] Study of Training Dynamics for Memory-Constrained Fine-Tuning (TraDy)](study_of_training_dynamics_for_memory-constrained_fine-tuning.md)
- [\[ICLR 2026\] AgilePruner: An Empirical Study of Attention and Diversity for Adaptive Visual Token Pruning in LVLMs](agilepruner_an_empirical_study_of_attention_and_diversity_for_adaptive_visual_to.md)
- [\[ICLR 2026\] Beyond Student: An Asymmetric Network for Neural Network Inheritance](beyond_student_an_asymmetric_network_for_neural_network_inheritance.md)

</div>

<!-- RELATED:END -->
