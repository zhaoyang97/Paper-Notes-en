---
title: >-
  [Paper Note] UGround: Towards Unified Visual Grounding with Unrolled Transformers
description: >-
  [ICML 2026][Segmentation][visual grounding] UGround reverses the LMM-based visual grounding paradigm from "using the final layer $\langle\text{SEG}\rangle$ token as prompt" to "using dynamically selected intermediate lay…
tags:
  - "ICML 2026"
  - "Segmentation"
  - "visual grounding"
  - "reasoning segmentation"
  - "similarity map"
  - "RL-based layer selection"
  - "SAM"
date: 2026-05-08
content_hash: 3d64b1356a867f4f
---

# UGround: Towards Unified Visual Grounding with Unrolled Transformers

**Conference**: ICML 2026  
**arXiv**: [2510.03853](https://arxiv.org/abs/2510.03853)  
**Code**: https://github.com/rui-qian/UGround (available)  
**Area**: Segmentation / Multimodal VLM / Visual Grounding  
**Keywords**: visual grounding, reasoning segmentation, similarity map, RL-based layer selection, SAM

## TL;DR
UGround reverses the LMM-based visual grounding paradigm from "using the final layer $\langle\text{SEG}\rangle$ token as prompt" to "using dynamically selected intermediate layer similarity maps as prompt." Through the RL-based SSC strategy, the $\langle\text{SEG}\rangle$ token slides across all transformer layers, treating the similarity map as both a soft logit mask for SAM and a backward supervision signal. For the first time, it unifies five visual grounding tasks—RES, RS, FP-RES, gRES, Multi-RS—within a single framework, achieving cIoU +9.0% on ReasonSeg test and N-acc +12.1% on gRefCOCO val.

## Background & Motivation

**Background**: Visual grounding is evolving from explicit referring expression segmentation (RES) to implicit reasoning segmentation (RS), from single-object to multi-object (gRES, Multi-RS), and from purely affirmative queries to handling false premises (FP-RES). Existing SOTA methods such as LISA, SESAME, GLaMM, GSVA, and PixelLM each cover only 2-3 attributes; no method supports all five tasks simultaneously.

**Limitations of Prior Work**: (1) **Fixed final layer**—LMMs have 32-40 transformer layers, but all methods only use the final layer's $\langle\text{SEG}\rangle$ embedding as input to SAM, accumulating errors at the last layer like a "telephone game"; (2) **$\langle\text{SEG}\rangle$ as prompt lacks spatial cues**—$\langle\text{SEG}\rangle$ is a text placeholder, essentially mapping text embeddings to visual space via an MLP, lacking coordinates and mask shapes, leaving SAM to "guess."

**Key Challenge**: Intermediate layers of LMMs actually contain more discriminative semantics (experiments show cIoU from layers 10-40 is higher than the last layer), but the traditional paradigm gives SAM no access to these representations. Moreover, the similarity map between the $\langle\text{SEG}\rangle$ token and image tokens is itself an H×W "soft mask," carrying more explicit spatial information than the $\langle\text{SEG}\rangle$ embedding.

**Goal**: (i) Handle RES + RS + FP-RES + gRES + Multi-RS within a unified architecture; (ii) Address the two major flaws of "fixed final layer" and "$\langle\text{SEG}\rangle$ lacks spatial cues"; (iii) Allow SAM to "cheat" by accessing intermediate semantic cues early.

**Key Insight**: Treat the layered structure as unrolled transformers, making each layer a potential input port for SAM; use similarity maps as "bidirectional masks" that both prompt SAM and provide backward supervision.

**Core Idea**: Replace "fixed final layer + $\langle\text{SEG}\rangle$ as prompt" with "policy-prompted masking = RL-based layer selection + similarity map as prompt," reconstructing visual grounding as a differentiable segmentation pipeline with skip connections.

## Method

### Overall Architecture
The input image $\mathbf{x}_{img}$ is processed by an LMM (LLaVA) with $L=32$ or $40$ transformer layers, yielding hidden states $\mathcal{H}^{(\ell)}$ at each layer, where position $t^*$ is the $\langle\text{SEG}\rangle$ token. The core module, PPM (Policy-Prompted Masking), performs two tasks at each forward pass $\mathcal{T}_t$: (1) **SSC** samples a layer $\ell^*$ from the policy distribution $\pi_\theta(\ell|\mathcal{H}_{t^*})$, allowing $\langle\text{SEG}\rangle$ to skip directly to SAM at layer $\ell^*$; (2) **MasP** computes the similarity map $\mathcal{M}\in[0,1]^{H\times W}$ between $\langle\text{SEG}\rangle$ and all image tokens at layer $\ell^*$, using $\mathcal{M}$ as a soft logit mask for the SAM decoder $\mathcal{G}_\mathcal{V}^{dec}(\mathbf{f}, \bm{h}_{seg}, \mathcal{M})$ to generate the final mask $\hat{\mathbf{M}}$. Throughout, $\mathcal{M}$ serves three roles: prompt (input to SAM), constraint (supervised by BCE+Dice), and signal (reward for REINFORCE).

### Key Designs

1. **Stochastic Skip Connection (SSC)**:

    - **Function**: Allows each $\langle\text{SEG}\rangle$ token to adaptively select "which layer to skip out to SAM."
    - **Mechanism**: Defines a policy distribution $\pi_\theta(\ell|\mathcal{H}_{t^*})=\frac{\exp(s_\ell)}{\sum_j\exp(s_j)}$, where $s_\ell=\bm{h}_{t^*}^{(\ell)}\cdot\mathbf{w}_\ell$ and each layer has its own learnable weight $\mathbf{w}_\ell$. During training, sampling $\ell^*\sim\pi_\theta$ enables exploration; reward is $r=-(\mathcal{L}_{bce}(\mathcal{M}, M_\sigma)+\mathcal{L}_{dice}(\mathcal{M}, M_\sigma))$, with $M_\sigma$ as the ground-truth mask smoothed by Gaussian. An EMA baseline $b_t=\alpha b_{t-1}+(1-\alpha)r$ reduces variance, and the REINFORCE loss is $\mathcal{L}_{policy}=-(r-b_t)\log\pi_\theta(\ell^*|\mathcal{H}_{t^*})$.
    - **Design Motivation**: A single forward resembles a skip connection (skipping $L-\ell^*$ layers to SAM); multiple forwards resemble dropout (activating different paths each time), equivalent to Monte Carlo uncertainty estimation. This structure mitigates "telephone game" error accumulation and improves robustness via ensembling.

2. **Mask as Prompt (MasP)**:

    - **Function**: Directly uses the similarity map between $\langle\text{SEG}\rangle$ and image tokens as SAM's soft logit mask prompt.
    - **Mechanism**: At the selected layer $\ell^*$, for each image token $z_i$, compute $\mathcal{S}_i^{(\ell^*)}=(\bm{h}_{z_i}^{(\ell^*)})^\top\bm{h}_{t^*}^{(\ell^*)}$; arrange the $k$ scores into a 2D grid and interpolate to $H\times W$ to obtain $\mathcal{M}$, then call the modified SAM: $\hat{\mathbf{M}}=\mathcal{G}_\mathcal{V}^{dec}(\mathbf{f}, \bm{h}_{seg}, \mathcal{M})$. $\mathcal{M}$ is continuously differentiable, allowing gradients to flow through SAM and be further constrained by explicit supervision $\mathcal{L}_\mathcal{M}=\lambda_{bce}\mathcal{L}_{bce}(\mathcal{M}, M_\sigma)+\lambda_{dice}\mathcal{L}_{dice}(\mathcal{M}, M_\sigma)$.
    - **Design Motivation**: As shown in Table 2, even without training, directly feeding the similarity map as prompt to the original SAM achieves 17% cIoU, indicating that LMMs have implicitly learned spatial distributions; explicit prompting and supervision amplify this implicit capability.

3. **Unified Attribute Architecture (5-attribute coverage)**:

    - **Function**: Supports RES, RS, FP-RES, gRES, and Multi-RS visual grounding tasks within a single model.
    - **Mechanism**: Leverages PPM's flexibility—each target in multi-object scenarios corresponds to a $\langle\text{SEG}\rangle$ token, each independently sampling a layer $\ell^*$; in false premise scenarios, if all similarity maps have low responses, the model can reject; for reasoning, intermediate layers provide stronger semantics than the final layer, suiting implicit descriptions.
    - **Design Motivation**: Previous methods like LISA only cover RES+RS, GSVA covers RES+RS+FP-RES+gRES but not Multi-RS, PixelLM supports Multi-RS but not empty targets; UGround is the first to achieve full 5/5 coverage.

### Loss & Training
The total loss is a weighted sum of four terms: $\mathcal{L}=\lambda_{txt}\mathcal{L}_{txt}+\lambda_{mask}\mathcal{L}_{mask}+\lambda_\mathcal{M}\mathcal{L}_\mathcal{M}+\lambda_{policy}\mathcal{L}_{policy}$. Here, $\mathcal{L}_{txt}$ is the standard LMM text generation loss, $\mathcal{L}_{mask}$ supervises the SAM output mask (BCE+Dice), $\mathcal{L}_\mathcal{M}$ supervises the similarity map against soft GT (BCE+Dice), and $\mathcal{L}_{policy}$ is the REINFORCE policy gradient. The base model is LLaVA1.5-7B/13B, segmentation decoding uses SAM, and fine-tuning is performed on 239 ReasonSeg train samples.

## Key Experimental Results

### Main Results

ReasonSeg test set (reasoning segmentation):

| Method | val gIoU | val cIoU | test gIoU | test cIoU |
|--------|----------|----------|-----------|-----------|
| LISA-7B-LLaVA1.5 (ft) | 61.3 | 62.9 | 55.6 | 56.9 |
| READ-7B-LLaVA1.5 (ft) | 59.8 | 67.6 | 58.5 | 58.6 |
| LISA++-7B-LLaVA1.5 (ft) | 64.2 | 68.1 | 57.0 | 59.5 |
| RSVP-GPT | 64.7 | 63.1 | 60.3 | 60.0 |
| **UGround-7B-LLaVA1.5 (ft)** | **66.1** | **72.1** | **63.6** | **65.4** |
| LISA-13B-LLaVA1.5 (ft) | 65.0 | 72.9 | 61.3 | 62.2 |
| **UGround-13B-LLaVA1.5 (ft)** | **67.9** | **74.9** | **65.0** | **65.5** |

Compared to LISA-7B (48.4 cIoU on test), UGround achieves **+17 cIoU**; compared to READ-7B (58.6), **+6.8 cIoU**; the "+9% cIoU" in the paper refers to the stronger RSVP-GPT baseline.

### Ablation Study

| Configuration | ReasonSeg test cIoU | Notes |
|---------------|---------------------|-------|
| Fixed final layer + $\langle\text{SEG}\rangle$ prompt (LISA paradigm) | ~48.4 | baseline |
| Dynamic layer selection + $\langle\text{SEG}\rangle$ prompt | Intermediate layers cIoU improved (layers 10-40 all surpass last layer) | SSC contribution |
| Fixed final layer + similarity map prompt | 35.0 (`SESAME`) → 30.7, +4.3% improvement | MasP effect |
| Full UGround (PPM = SSC + MasP) | 65.4 | full model |

From Table 2: the untrained SAM with similarity map prompt achieves 17% cIoU; directly binarizing the similarity map yields 35.0% (even surpassing SESAME's trained 30.7%).

### Key Findings
- **Intermediate layers outperform the final layer**: All layers 10-40 yield higher cIoU than the fixed final layer (Fig 2a); intermediate layers converge from layer 19, final layer only by 28, indicating dynamic selection both raises the upper bound and accelerates convergence.
- **Similarity maps inherently encode spatial semantics**: Even untrained SAM can output reasonable results with similarity map prompts, proving LMMs' internal similarity structures encode spatial cues, which prior methods failed to exploit.
- **FP-RES task N-acc +12.1%**: On gRefCOCO, the ability to reject false premises (empty targets) far exceeds baselines, thanks to layer ensemble from policy sampling providing uncertainty estimation.

## Highlights & Insights
- **The "unrolled transformer" framing is elegant**: Treating stacked transformers as a sequence of selectable paths exposes all 39 intermediate representations as candidate prompt sources for SAM, a "black box opening" perspective transferable to any downstream task needing intermediate layer information.
- **Triple reuse of similarity maps**: The same $\mathcal{M}$ serves as SAM prompt, loss supervision target, and RL reward, sharing computation for all three uses—extremely efficient.
- **REINFORCE + LMM intermediate layer selection**: Modeling "which layer to connect" as a discrete action for policy gradient provides a clean implementation for "compositional differentiable + discrete layer selection."
- **Unified engineering value**: Full 5-attribute coverage means no more task-specific models at deployment, enabling use as a general-purpose grounding backend.

## Limitations & Future Work
- Training compute: The policy samples from $L=32$ or $40$ layers per step, and REINFORCE's high variance theoretically requires multiple forwards for stability; the paper does not fully disclose training time costs.
- The similarity computation between $\langle\text{SEG}\rangle$ and image tokens is still limited by SAM's input resolution; $H\times W$ grid interpolation may distort small targets.
- The REINFORCE baseline uses EMA; lacks comparison with critic networks, so variance control could be improved.
- Only validated on LLaVA1.5; compatibility with newer LMMs like Qwen-VL, InternVL is unknown.
- Layer ensemble during training provides uncertainty estimation, but if only one path is sampled at inference, this benefit is lost—the paper does not clarify whether MC averaging is used at inference.

## Related Work & Insights
- **vs LISA / SESAME / READ**: All use fixed final layer + $\langle\text{SEG}\rangle$ prompt; UGround uses dynamic layer + similarity map prompt, representing a paradigm shift.
- **vs GSVA / PixelLM**: The former covers 4 attributes, the latter 4; UGround achieves full 5/5 coverage.
- **vs HyperSeg / OMG-LLaVA**: HyperSeg is versatility-oriented (unifying different modality tasks), UGround is attribute-oriented (unifying different attributes of the same task); the two are orthogonal and can be combined.
- **vs Mask2Former**: Mask2Former unifies segmentation on the vision side; UGround unifies grounding at the language-vision interface, serving as the LMM-era counterpart.
- **Insights**: (a) "Intermediate layers outperform the final layer" may hold for many LMM downstream tasks and merits systematic study; (b) Using attention/similarity maps as prompts rather than hidden states can be generalized to detection, tracking, and open-vocab segmentation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Unrolled transformers + policy-prompted masking is a clear new perspective; full 5-attribute coverage is also a first.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers ReasonSeg / RefCOCO / gRefCOCO benchmarks, detailed ablations, but lacks comparison between single-path and MC averaging at inference.
- Writing Quality: ⭐⭐⭐⭐ The "telephone game" analogy in Sec 3 and visualizations in Fig 1/2/5 are effective, though the policy gradient formulas are dense.
- Value: ⭐⭐⭐⭐⭐ Achieves SOTA results and releases code; the "intermediate layer + similarity map" paradigm is likely to have long-term impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] ReferDINO: Referring Video Object Segmentation with Visual Grounding Foundations](../../ICCV2025/segmentation/referdino_referring_video_object_segmentation_with_visual_grounding_foundations.md)
- [\[CVPR 2026\] RealVLG-R1: A Large-Scale Real-World Visual-Language Grounding Benchmark for Robotic Perception and Manipulation](../../CVPR2026/segmentation/realvlg-r1_a_large-scale_real-world_visual-language_grounding_benchmark_for_robo.md)
- [\[AAAI 2026\] EAGLE: Episodic Appearance- and Geometry-Aware Memory for Unified 2D-3D Visual Query Localization](../../AAAI2026/segmentation/eagle_episodic_appearance-_and_geometry-aware_memory_for_unified_2d-3d_visual_qu.md)
- [\[CVPR 2025\] DA-VPT: Semantic-Guided Visual Prompt Tuning for Vision Transformers](../../CVPR2025/segmentation/da-vpt_semantic-guided_visual_prompt_tuning_for_vision_transformers.md)
- [\[AAAI 2026\] RSVG-ZeroOV: Exploring a Training-Free Framework for Zero-Shot Open-Vocabulary Visual Grounding in Remote Sensing Images](../../AAAI2026/segmentation/rsvg-zeroov_exploring_a_training-free_framework_for_zero-shot_open-vocabulary_vi.md)

</div>

<!-- RELATED:END -->
