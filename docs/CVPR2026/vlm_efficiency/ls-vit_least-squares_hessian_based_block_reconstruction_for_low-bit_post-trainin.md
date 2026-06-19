---
title: >-
  [Paper Note] LS-ViT: Least-Squares Hessian Based Block Reconstruction for Low-Bit Post-Training Quantization of Vision Transformers
description: >-
  [CVPR 2026][Model Compression][Vision Transformer] LS-ViT reformulates the estimation of the "representative Hessian" in ViT block reconstruction as a least-squares problem—fitting a shared Hessian using $(g, \Delta z)$ pairs across the entire calibration set. This explicitly recovers the covariance terms lost by previous methods due to the "sample independence assumpt
tags:
  - CVPR 2026
  - Model Compression
  - Vision Transformer
date: 2026-05-08
content_hash: 89b81f6d28a6d448
---
# LS-ViT: Least-Squares Hessian Based Block Reconstruction for Low-Bit Post-Training Quantization of Vision Transformers

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Hwang_LS-ViT_Least-Squares_Hessian_Based_Block_Reconstruction_for_Low-Bit_Post-Training_Quantization_CVPR_2026_paper.html)  
**Code**: https://github.com/hhh7748/LS-ViT  
**Area**: Model Compression / Quantization  
**Keywords**: Post-Training Quantization, Vision Transformer, Block Reconstruction, Hessian Approximation, Least-Squares

## TL;DR
LS-ViT reformulates the estimation of the "representative Hessian" in ViT block reconstruction as a least-squares problem—fitting a shared Hessian using $(g, \Delta z)$ pairs across the entire calibration set. This explicitly recovers the covariance terms lost by previous methods due to the "sample independence assumption," achieving new SOTA in ultra-low bits like W2/A3 and W2/A4. Each block requires only one backpropagation, making training 1.8–2.7x faster than FIMA-Q.

## Background & Motivation
**Background**: Deploying ViTs to edge devices requires quantization. Post-Training Quantization (PTQ) is more practical as it uses a small amount of unlabeled calibration data and costs significantly less than Quantization-Aware Training (QAT). Among PTQ techniques, block reconstruction methods perform best at 4-bit or lower. These methods retain second-order terms from the Taylor expansion of the task loss, approximating "quantization-induced loss increment" as $\frac{1}{2}\Delta z^\top H^{(z)} \Delta z$. The core challenge lies in estimating the Hessian $H^{(z)}$.

**Limitations of Prior Work**: A full Hessian is $d\times d$ and computationally prohibitive, necessitating approximation. Recent methods like APHQ-ViT and FIMA-Q average gradient information across multiple samples to obtain a stable representative Hessian. However, they implicitly assume **independence between samples**. This paper points out that Figure 2 shows wide distributions of Hessian diagonal/rank-one components for the same channel across different samples—indicating significant inter-sample variance where the independence assumption fails.

**Key Challenge**: Regarding $\mathbb{E}[H^{(z)}\Delta z] = \mathbb{E}[H^{(z)}]\mathbb{E}[\Delta z] + \mathrm{Cov}(H^{(z)}, \Delta z)$, the independence assumption essentially discards the covariance term $\mathrm{Cov}(H^{(z)}, \Delta z)$. As bit-widths decrease, the perturbation $\Delta z$ increases, making the neglected covariance-induced error $H^{(z)}\Delta z - H\Delta z$ non-negligible (Figure 4), leading to "unrepresentative" Hessian estimates. Furthermore, these methods rely on multiple backpropagations, incurring high computational overhead.

**Goal**: Find a single Hessian that is "most representative" across the entire calibration set, recovering the covariance term while reducing estimation costs to a single gradient calculation per block.

**Key Insight**: Finding a matrix shared by all samples that best explains all $(g, \Delta z)$ observations is naturally a **least-squares regression** problem—fitting, rather than averaging.

**Core Idea**: Replace "averaging" with the least-squares solution $\widehat{H} := \arg\min_H \mathbb{E}\big[\|H^{(z)}\Delta z - H\Delta z\|^2\big]$. This results in a Hessian estimate that explicitly minimizes approximation residuals and requires only a single backpropagation, implemented as LS-ViT.

## Method

### Overall Architecture
LS-ViT processes each Transformer block sequentially in two stages: **First stage**, prior to reconstruction, a representative Hessian is derived via least squares using gradients $g$ and block output perturbations $\Delta z$ from all calibration samples. **Second stage**, this fixed Hessian serves as the reconstruction metric, following the QDrop baseline to optimize weight rounding (AdaRound) and activation scaling parameters. The pipeline utilizes standard uniform affine quantizers (channel-wise for weights, layer-wise for activations) and does not rely on ViT-specific quantizers, making inference as hardware-friendly as standard PTQ.

The starting point is the standard block reconstruction objective: approximating the loss increment due to quantization as $\mathcal{L}_\mathrm{recon}(\Delta z) = \frac{1}{2}\Delta z^\top H^{(z)} \Delta z$. Differentiating with respect to $\Delta z$ yields $g = H^{(z)}\Delta z$ (assuming $H^{(z)}$ is symmetric and locally constant with respect to $\Delta z$; symmetry is guaranteed by Clairaut’s Theorem for continuously differentiable functions, and local constancy is assumed by treating the Hessian as an inherent property of the pretrained model, following APHQ-ViT/FIMA-Q). To find a shared $H$, this equation should hold for **every** $(g, \Delta z)$ pair in the calibration set—turning Hessian estimation into an overdetermined system of equations to "find a matrix that best fits these $(g, \Delta z)$ pairs," solved via closed-form least squares. This is a pure improvement to the loss metric without multi-module pipelines; thus, it is best explained via formulas rather than a framework diagram.

### Key Designs

**1. Least-Squares Hessian Estimation: Shifting from Averaging to Fitting to Recover Covariance**

This is the core of the paper, addressing the "discarded covariance term" in the independence assumption. While FIMA-Q also uses $g = H^{(z)}\Delta z$, it assumes samples are independent and takes expectations of both sides, effectively losing $\mathrm{Cov}(H^{(z)}, \Delta z)$. LS-ViT avoids averaging and frames estimation as minimizing expected squared residuals:

$$\widehat{H} := \arg\min_H \mathbb{E}\big[\|H^{(z)}\Delta z - H\Delta z\|^2\big]$$

The least-squares solution automatically incorporates the inter-sample covariance structure. The paper provides a clear comparison for the diagonal case: $\widehat{H}_{i,i} = \frac{\mathbb{E}[g_i\Delta z_i]}{\mathbb{E}[\Delta z_i^2]} = \frac{\mathrm{Cov}(g_i,\Delta z_i) + \mathbb{E}[g_i]\mathbb{E}[\Delta z_i]}{\mathrm{Var}(\Delta z_i) + (\mathbb{E}[\Delta z_i])^2}$. Setting $\mathrm{Cov}(g_i,\Delta z_i)=0$ and $\mathrm{Var}(\Delta z_i)=0$ reduces this to FIMA-Q-D, explicitly demonstrating that FIMA-Q is a special case of the proposed method under the independence assumption. As bit-widths decrease and $\Delta z$ grows, these terms become indispensable, explaining LS-ViT's advantage in ultra-low bit scenarios.

**2. Diagonal Least-Squares Hessian (LSH-D): Stable Per-Channel Sensitivity via Closed-Form Ratios**

This addresses numerical instability in sample-wise Hessian estimation. A naive per-sample diagonal estimate is $H^{(z)}_{i,i} = g_i/\Delta z_i$, but since $\Delta z_i$ is often small, division amplifies noise, resulting in extreme variance (only 23.59% for ViT-S in ablations). Under diagonal approximation, $g_i = H^{(z)}_{i,i}\Delta z_i$ forms an overdetermined system for each channel $i$ ($N$ sample equations, 1 unknown). The closed-form least-squares solution is:

$$\widehat{H}_{i,i} = \frac{\sum_{n=1}^{N} g_i^{(n)} \Delta z_i^{(n)}}{\sum_{n=1}^{N} (\Delta z_i^{(n)})^2} = \overline{g_i \Delta z_i}\big/ \overline{\Delta z_i^2}$$

The corresponding reconstruction loss is $\mathcal{L}_\mathrm{LSH,D}(\Delta z) = \frac{1}{2}\sum_i \Delta z_i^2 \cdot (\overline{g_i\Delta z_i}/\overline{\Delta z_i^2})$. By aggregating across samples, it preserves inter-sample relationships while suppressing variance, accurately characterizing individual parameter sensitivity.

**3. Rank-one Low-rank Least-Squares Hessian (LSH-L): Capturing Dominant Non-diagonal Interactions**

Diagonal approximation ignores cross-channel terms, so a low-rank component is added. While $N$ samples can support a rank-$N$ Hessian, the reconstruction cost would be $O(Nd)$. This work utilizes a cost-effective rank-one approximation $H^{(z)} = uu^\top$. Applying least squares directly to $g^{(n)} = uu^\top \Delta z^{(n)}$ introduces high-order terms of $u$. Multiplying both sides by $\Delta z^{(n)\top}$ yields $\Delta z^{(n)\top} g^{(n)} = (u^\top \Delta z^{(n)})^2$. Rewriting as $g^{(n)} = u\sqrt{\Delta z^{(n)\top} g^{(n)}}$, the least-squares solution is:

$$\widehat{u} = \frac{\sum_{n} g^{(n)}\sqrt{\Delta z^{(n)\top} g^{(n)}}}{\sum_{n} \Delta z^{(n)\top} g^{(n)}} = \overline{g\sqrt{\Delta z^\top g}}\big/\overline{\Delta z^\top g}$$

LS-ViT combines both: $\mathcal{L}_\mathrm{LSH}(\Delta z) = \mathcal{L}_\mathrm{LSH,D}(\Delta z) + \mathcal{L}_\mathrm{LSH,L}(\Delta z)$. The diagonal term handles per-parameter sensitivity, while the rank-one term captures dominant non-diagonal interactions. Crucially, the entire estimation requires only **one backpropagation** per block, avoiding the multiple full-model passes required by FIMA-Q and high-rank operations during reconstruction.

### Loss & Training
During reconstruction, the QDrop mechanism is used to obtain $\Delta z'$ via a forward pass. $\mathcal{L}_\mathrm{LSH}(\Delta z')$ is minimized to update weight rounding via AdaRound and optimize activation scaling parameters. Default settings: 1024 calibration images, batch size 32, 20k iterations, weight learning rate 1e-3, and activation learning rate 4e-5.

## Key Experimental Results

### Main Results
ImageNet classification, seven ViT/DeiT/Swin models, Top-1 accuracy (%). In comparisons, "\*" denotes identical settings except for the reconstruction metric:

| Setting (W/A) | Model | FIMA-Q* | APHQ-ViT | LS-ViT (Ours) |
|------|------|---------|----------|---------------|
| 2/4 | ViT-S | 56.76 | 56.04 | **58.48** |
| 2/4 | Swin-S | 70.64 | 71.75 | **73.89** |
| 2/4 | Swin-B | 72.36 | 72.64 | **74.95** |
| 2/3 | ViT-B | 61.15 | 58.06 | **62.89** |
| 2/3 | Swin-S | 59.57 | 62.90 | **65.77** |
| 3/3 | ViT-S | 64.09 | 63.17 | **64.10** |

At W2/A4, average Gain vs FIMA-Q is +1.46%p (Swin-S +3.25, Swin-B +2.59); at W2/A3, average Gain is +2.89%p (Swin-S +6.20, Swin-B +4.58); at W3/3, average Gain is +0.17%p. The advantage increases as bit-widths decrease, confirming the criticality of the covariance term at low bits. Swin models benefit significantly due to high inter-sample variance in important features. Consistent minor improvements (+0.1 to +0.2 AP) were also observed on COCO detection/segmentation (W4/A4).

### Ablation Study
Comparison of diagonal Hessian estimation methods (W3/A3, Top-1 %):

| Estimation Method | Formula | ViT-S | Swin-S |
|------|------|-------|--------|
| Per-sample $g_i/\Delta z_i$ | Naive Division | 23.59 | 69.42 |
| Per-sample $g_i$ | Remove Division | 58.16 | 76.02 |
| FIMA-Q-D | $\mathbb{E}[g_i]/\mathbb{E}[\Delta z_i]$ | 60.02 | 75.08 |
| LS-ViT-D (Ours) | $\mathbb{E}[g_i\Delta z_i]/\mathbb{E}[\Delta z_i^2]$ | **63.25** | **77.29** |

Training cost (Single GPU, minutes): LS-ViT outperforms FIMA-Q with only 1 full-model calculation, whereas FIMA-Q requires 15. Reducing FIMA-Q's budget from 15 to 1 results in a drop of 0.15–1.41%p. LS-ViT is 1.8× (DeiT-T) to 2.7× (Swin-B) faster than FIMA-Q, comparable to QDrop.

### Key Findings
- The covariance term contributes most in ultra-low bit settings: the average gain over FIMA-Q widens from +0.17%p (3/3) to +2.89%p (2/3).
- Naive per-sample $g_i/\Delta z_i$ is highly unstable due to small $\Delta z_i$ (23.59% for ViT-S). Simply removing the division (using $g_i$) improves accuracy by 16.81%p on average; least-squares aggregation further improves on FIMA-Q-D by 1.22%p.
- For architectures like Swin, where important features have high inter-sample variance, LS-ViT’s robust capturing of these variances provides the largest gains.

## Highlights & Insights
- Elevates "representative Hessian estimation" from heuristic averaging to closed-form least-squares regression. The $\mathbb{E}[g\Delta z]/\mathbb{E}[\Delta z^2]$ expansion elegantly frames FIMA-Q as a special case of independence—a strong proving paradigm.
- Key diagnostic insight: The authors identified a shared implicit error (discarding covariance) across block reconstruction methods and proved its amplification in critical ultra-low bit scenarios.
- "Single backprop per block + standard uniform quantizer" makes the method accurate, fast, and plug-and-play. The rank-one reformulation (multiplying by $\Delta z^\top$ to avoid high-order terms) is a clever utility for $uu^\top$ least-squares problems.

## Limitations & Future Work
- Low-rank approximation is restricted to rank-one due to $O(Nd)$ cost; whether higher ranks are worth it for high-variance blocks remains open.
- The "Hessian local constant" assumption under aggressive perturbations (e.g., W2/A2) is not deeply explored; applicability at extreme bit boundaries needs verification.
- Gains on detection/segmentation (+0.1 to +0.2 AP) are modest compared to classification, suggesting thinner margins for dense prediction tasks.
- Several derivations rely on supplementary materials (closed-form solutions, component analyses); ⚠️ refer to the original paper and supplement for full formula details.

## Related Work & Insights
- **vs FIMA-Q**: Both use $g = H^{(z)}\Delta z$. FIMA-Q assumes independence and discards covariance, requiring multiple model passes for low-rank estimation. LS-ViT uses least-squares to resolve all sample pairs, explicitly retaining covariance with only a single backprop—making it more accurate and faster.
- **vs APHQ-ViT**: APHQ-ViT uses averaged perturbation Hessians with MLP reconstruction. LS-ViT is consistently superior in fair comparisons (only replacing the metric), proving gains stem from the metric itself.
- **vs BRECQ / QDrop / AdaRound**: These use sample-independent constants or per-sample squared gradients, which are coarser approximations. LS-ViT serves as a direct upgrade by refining the Hessian estimation within the same framework.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Unified representative Hessian estimation into least-squares, recovering covariance cleanly.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers various models, bit-widths, tasks, and costs; excellent ablations. Detection gains are small.
- Writing Quality: ⭐⭐⭐⭐ Clear chain from motivation to diagnosis to derivation, though heavy reliance on supplementary material.
- Value: ⭐⭐⭐⭐⭐ Plug-and-play, achieves SOTA in ultra-low bits while being 2x faster; high practical utility for edge deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] BinaryAttention: One-Bit QK-Attention for Vision and Diffusion Transformers](binaryattention_one-bit_qk-attention_for_vision_and_diffusion_transformers.md)
- [\[CVPR 2026\] CAR-SAM: Cross-Attention Reconstruction for Post-Training Quantization of the Segment Anything Model](car-sam_cross-attention_reconstruction_for_post-training_quantization_of_the_seg.md)
- [\[CVPR 2026\] VLM-PTQ: Efficient Post-Training Quantization for Large Vision-Language Models](vlm-ptq_efficient_post-training_quantization_for_large_vision-language_models.md)
- [\[CVPR 2026\] TWEO: Transformers Without Extreme Outliers Enables FP8 Training And Quantization For Dummies](tweo_transformers_without_extreme_outliers_enables_fp8_training_and_quantization.md)
- [\[CVPR 2025\] FIMA-Q: Post-Training Quantization for Vision Transformers by Fisher Information Matrix Approximation](../../CVPR2025/model_compression/fima-q_post-training_quantization_for_vision_transformers_by_fisher_information_.md)

</div>

<!-- RELATED:END -->
