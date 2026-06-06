---
title: >-
  [Paper Note] UGround: Towards Unified Visual Grounding with Unrolled Transformers
description: >-
  [ICML 2026][Segmentation][Visual Grounding] UGround shifts the LMM-based visual grounding paradigm from "using the last-layer $\langle\text{SEG}\rangle$ token as a prompt" to "using dynamically selected intermediate-laye…
tags:
  - "ICML 2026"
  - "Segmentation"
  - "Visual Grounding"
  - "Reasoning Segmentation"
  - "Similarity Map"
  - "Reinforcement Learning Layer Selection"
  - "SAM"
date: 2026-05-08
content_hash: 0a0c1eb774579445
---

# UGround: Towards Unified Visual Grounding with Unrolled Transformers

**Conference**: ICML 2026  
**arXiv**: [2510.03853](https://arxiv.org/abs/2510.03853)  
**Code**: https://github.com/rui-qian/UGround (Available)  
**Area**: Segmentation / Multimodal VLM / Visual Grounding  
**Keywords**: Visual Grounding, Reasoning Segmentation, Similarity Map, Reinforcement Learning Layer Selection, SAM

## TL;DR
UGround shifts the LMM-based visual grounding paradigm from "using the last-layer $\langle\text{SEG}\rangle$ token as a prompt" to "using dynamically selected intermediate-layer similarity maps as prompts." By employing a Reinforcement Learning policy (SSC), $\langle\text{SEG}\rangle$ traverses all transformer layers, treating similarity maps as both soft logit masks for SAM and backward supervision signals. It is the first framework to unify five visual grounding tasks (RES / RS / FP-RES / gRES / Multi-RS) within a single architecture, achieving a $+9.0\%$ gain in cIoU on ReasonSeg test and $+12.1\%$ N-acc on gRefCOCO val.

## Background & Motivation

**Background**: Visual grounding is evolving from explicit Referring Expression Segmentation (RES) to implicit Reasoning Segmentation (RS), from single-target to multi-target (gRES, Multi-RS), and from affirmative queries to rejecting premises (FP-RES). Existing SOTAs like LISA, SESAME, GLaMM, GSVA, and PixelLM only cover 2-3 of these attributes; no single method satisfies all five simultaneously.

**Limitations of Prior Work**: (1) **Fixed Last Layer**: LMMs have 32–40 transformer layers, yet all methods feed only the last-layer $\langle\text{SEG}\rangle$ embedding to SAM. This resembles a "telephone game" where cumulative errors are dumped onto the final layer. (2) **$\langle\text{SEG}\rangle$ Prompt Lacks Spatial Cues**: $\langle\text{SEG}\rangle$ is a text placeholder; essentially, an MLP implicitly maps text embeddings to visual space. It lacks coordinates or mask shapes, forcing SAM to "guess."

**Key Challenge**: Intermediate layers of LMMs actually contain more discriminative semantics (experiments show cIoU for layers 10–40 is higher than the last layer), but traditional paradigms deprive SAM of any opportunity to see these intermediate representations. Meanwhile, the similarity map between $\langle\text{SEG}\rangle$ and image tokens is natively an $H \times W$ "soft mask," carrying more explicit spatial information than the $\langle\text{SEG}\rangle$ embedding itself.

**Goal**: (i) Handle RES + RS + FP-RES + gRES + Multi-RS tasks simultaneously within a unified architecture; (ii) Address the dual defects of "fixed last layer" and "spatial cue-deficient $\langle\text{SEG}\rangle$"; (iii) Allow SAM to "cheat" by accessing intermediate semantic cues early.

**Key Insight**: Treat the hierarchical structure as unrolled transformers, making every layer a potential input port for SAM. Use similarity maps as "bidirectional masks" that can both prompt SAM and provide backward supervision.

**Core Idea**: Replace "fixed last layer + $\langle\text{SEG}\rangle$ prompt" with "policy-prompted masking = RL layer selection + similarity map prompt," restructuring visual grounding as a differentiable segmentation pipeline with skip connections.

## Method

### Overall Architecture
The input image $\mathbf{x}_{img}$ is processed by $L=32$ or $40$ transformer layers of an LMM (LLaVA), yielding hidden states $\mathcal{H}^{(\ell)}$ for each layer, where the $t^*$ position corresponds to the $\langle\text{SEG}\rangle$ token. The core module, Policy-Prompted Masking (PPM), performs two actions during each forward pass $\mathcal{T}_t$: (1) **SSC** samples a layer $\ell^*$ from a policy distribution $\pi_\theta(\ell|\mathcal{H}_{t^*})$, allowing $\langle\text{SEG}\rangle$ at layer $\ell^*$ to skip-connect directly to SAM; (2) **MasP** calculates the similarity map $\mathcal{M}\in[0,1]^{H\times W}$ between $\langle\text{SEG}\rangle$ and all image tokens at layer $\ell^*$. $\mathcal{M}$ is fed as a soft logit mask to the SAM decoder $\mathcal{G}_\mathcal{V}^{dec}(\mathbf{f}, \bm{h}_{seg}, \mathcal{M})$ to generate the final mask $\hat{\mathbf{M}}$. Throughout this process, $\mathcal{M}$ serves three roles: prompt (to SAM), constraint (supervised by BCE+Dice), and signal (as a reward for REINFORCE).

### Key Designs

1. **Stochastic Skip Connection (SSC)**:
    - **Function**: Allows each $\langle\text{SEG}\rangle$ token to adaptively select "which layer to skip out and connect to SAM" across transformer layers.
    - **Mechanism**: Defines a policy distribution $\pi_\theta(\ell|\mathcal{H}_{t^*})=\frac{\exp(s_\ell)}{\sum_j\exp(s_j)}$, where $s_\ell=\bm{h}_{t^*}^{(\ell)}\cdot\mathbf{w}_\ell$ and each layer has its own learnable weight $\mathbf{w}_\ell$. Sampling $\ell^*\sim\pi_\theta$ during training allows for exploration. The reward is defined as $r=-(\mathcal{L}_{bce}(\mathcal{M}, M_\sigma)+\mathcal{L}_{dice}(\mathcal{M}, M_\sigma))$, where $M_\sigma$ is the ground-truth mask after Gaussian smoothing. An EMA baseline $b_t=\alpha b_{t-1}+(1-\alpha)r$ is used to reduce variance, with the REINFORCE loss being $\mathcal{L}_{policy}=-(r-b_t)\log\pi_\theta(\ell^*|\mathcal{H}_{t^*})$.
    - **Design Motivation**: A single forward pass acts like a skip connection (skipping $L-\ell^*$ layers to connect to SAM), while multiple passes act like dropout (activating a different path each time), equivalent to Monte Carlo uncertainty estimation. This structure mitigates error accumulation from the "telephone game" and improves robustness through ensemble effects.

2. **Mask as Prompt (MasP)**:
    - **Function**: Uses the similarity map between $\langle\text{SEG}\rangle$ and image tokens directly as a soft logit mask prompt for SAM.
    - **Mechanism**: At the selected layer $\ell^*$, it computes $\mathcal{S}_i^{(\ell^*)}=(\bm{h}_{z_i}^{(\ell^*)})^\top\bm{h}_{t^*}^{(\ell^*)}$ for each image token $z_i$. The scores are arranged in a 2D grid and interpolated to $H\times W$ to obtain $\mathcal{M}$. The modified SAM is then called: $\hat{\mathbf{M}}=\mathcal{G}_\mathcal{V}^{dec}(\mathbf{f}, \bm{h}_{seg}, \mathcal{M})$. $\mathcal{M}$ is continuously differentiable, allowing gradients to propagate both through SAM and via explicit supervision $\mathcal{L}_\mathcal{M}=\lambda_{bce}\mathcal{L}_{bce}(\mathcal{M}, M_\sigma)+\lambda_{dice}\mathcal{L}_{dice}(\mathcal{M}, M_\sigma)$.
    - **Design Motivation**: Empirical results in Table 2 show that even without training, feeding the similarity map directly as a prompt to a vanilla SAM achieves 17% cIoU, suggesting that LMMs implicitly learn spatial distributions. Explicit prompting and supervision amplify this capability.

3. **Unified Architecture (5-attribute coverage)**:
    - **Function**: Supports RES, RS, FP-RES, gRES, and Multi-RS tasks in a single model.
    - **Mechanism**: Relies on the flexibility of PPM. In multi-target scenarios, each target corresponds to a $\langle\text{SEG}\rangle$ token, each independently sampling a layer $\ell^*$. In false-premise scenarios, the model can reject the target if similarity maps across all layers show low response. In reasoning scenarios, the stronger semantics of intermediate layers facilitate handling implicit descriptions.
    - **Design Motivation**: Previous methods like LISA only cover RES+RS; GSVA covers RES+RS+FP-RES+gRES but lacks Multi-RS support; PixelLM supports Multi-RS but cannot handle null targets. UGround is the first to achieve 5/5 coverage.

### Loss & Training
The total loss is a weighted sum of four terms: $\mathcal{L}=\lambda_{txt}\mathcal{L}_{txt}+\lambda_{mask}\mathcal{L}_{mask}+\lambda_\mathcal{M}\mathcal{L}_\mathcal{M}+\lambda_{policy}\mathcal{L}_{policy}$. $\mathcal{L}_{txt}$ is the standard LMM text generation loss, $\mathcal{L}_{mask}$ is the SAM output mask supervision (BCE+Dice), $\mathcal{L}_\mathcal{M}$ is the BCE+Dice for the similarity map against soft GT, and $\mathcal{L}_{policy}$ is the REINFORCE policy gradient. The base models are LLaVA1.5-7B/13B with SAM for decoding, fine-tuned on 239 samples from the ReasonSeg training set.

## Key Experimental Results

### Main Results

ReasonSeg Test Set (Reasoning Segmentation):

| Method | val gIoU | val cIoU | test gIoU | test cIoU |
|------|----------|----------|-----------|-----------|
| LISA-7B-LLaVA1.5 (ft) | 61.3 | 62.9 | 55.6 | 56.9 |
| READ-7B-LLaVA1.5 (ft) | 59.8 | 67.6 | 58.5 | 58.6 |
| LISA++-7B-LLaVA1.5 (ft) | 64.2 | 68.1 | 57.0 | 59.5 |
| RSVP-GPT | 64.7 | 63.1 | 60.3 | 60.0 |
| **UGround-7B-LLaVA1.5 (ft)** | **66.1** | **72.1** | **63.6** | **65.4** |
| LISA-13B-LLaVA1.5 (ft) | 65.0 | 72.9 | 61.3 | 62.2 |
| **UGround-13B-LLaVA1.5 (ft)** | **67.9** | **74.9** | **65.0** | **65.5** |

**Gain**: Compared to LISA-7B (48.4 cIoU on test), UGround provides a **+17 cIoU** improvement. Compared to the fine-tuned READ-7B (58.6), it shows a **+6.8 cIoU** gain. The "$+9\%$ cIoU" advertised in the paper refers to the improvement over stronger baselines like RSVP-GPT.

### Ablation Study

| Configuration | ReasonSeg test cIoU | Description |
|------|---------------------|------|
| Fixed last layer + $\langle\text{SEG}\rangle$ prompt (LISA paradigm) | ~48.4 | Baseline |
| Dynamic layer selection + $\langle\text{SEG}\rangle$ prompt | Improved cIoU for intermediate layers (10-40 surpass last) | SSC contribution |
| Fixed last layer + Similarity map prompt | 30.7 → 35.0 (compared to `SESAME`) | MasP contribution (+4.3%) |
| Full UGround (PPM = SSC + MasP) | 65.4 | Full model |

Similarity map analysis from Table 2: A vanilla, untrained SAM using similarity map prompts achieves 17% cIoU. Converting the similarity map directly to a binary mask reaches 35.0% (surpassing the 30.7% of a trained SESAME).

### Key Findings
- **Intermediate Layers > Last Layer**: Predicted cIoU for all layers 10–40 is higher than the fixed last-layer strategy (Fig 2a). Intermediate layers begin to converge from layer 19, whereas the last layer requires layer 28, indicating that dynamic layer selection both raises the performance ceiling and accelerates convergence.
- **Similarity Maps Carry Spatial Semantics**: An untrained SAM can produce reasonable outputs using only similarity map prompts, proving that LMM internal similarity structures already encode spatial cues—traditional methods simply failed to utilize them.
- **N-acc +12.1% on FP-RES**: The ability to reject false premises (null targets) on gRefCOCO far exceeds baselines. This is attributed to the layer ensemble generated by policy sampling, which provides effective uncertainty estimation.

## Highlights & Insights
- **The "Unrolled Transformer" framing is elegant**: Treating fixed, stacked transformers as a sequence of unrolled optional paths turns the 39 intermediate representations previously invisible to SAM into candidate prompt sources. This "open the black box" perspective can be transferred to any downstream task requiring intermediate layer info.
- **Triple-role reuse of similarity maps**: $\mathcal{M}$ serves as a SAM prompt, a loss supervision target, and an RL reward. Sharing the same computation for three purposes is extremely efficient.
- **REINFORCE for LMM intermediate layer selection**: Modeling "where to connect" as a discrete-action policy gradient provides a clean implementation paradigm for "composable differentiable modules + discrete layer selection."
- **Engineering value of unification**: 5-attribute coverage means task-specific models are no longer needed for deployment; UGround can serve as a universal grounding backend.

## Limitations & Future Work
- Training computational cost: The policy requires sampling from $L=32$ or $40$ layers at every step. Combined with the high variance of REINFORCE, this likely requires multiple forward passes to stabilize; training time overhead is not fully disclosed.
- Similarity computation is still limited by SAM input resolution; interpolating the $H\times W$ grid may cause distortion for small targets.
- REINFORCE uses an EMA baseline; the lack of experimental comparison with a critic network suggests variance control could still be improved.
- Validation is limited to LLaVA1.5; compatibility with newer LMMs like Qwen-VL or InternVL is unknown.
- While layer ensembles during training provide uncertainty estimation, if only a single path is sampled during inference, this benefit may be lost (it is unclear if MC averaging is used during inference).

## Related Work & Insights
- **vs. LISA / SESAME / READ**: All use a fixed last layer + $\langle\text{SEG}\rangle$ prompt. UGround represents a paradigm shift using dynamic layers + similarity map prompts.
- **vs. GSVA / PixelLM**: The former covers 4 attributes and the latter 4; UGround provides full 5/5 coverage.
- **vs. HyperSeg / OMG-LLaVA**: HyperSeg is versatility-oriented (unifying different modality tasks), whereas UGround is attribute-oriented (unifying different attributes of the same task). The two are orthogonal and combinable.
- **vs. Mask2Former**: Mask2Former performs unified segmentation on the vision side; UGround performs unified grounding at the vision-language interface, serving as its LMM-era counterpart.
- **Insights**: (a) The "intermediate > last layer" phenomenon likely holds for many LMM downstream tasks and warrants systematic study. (b) Using attention/similarity maps as prompts instead of hidden states can be extended to detection, tracking, and open-vocab segmentation.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Unrolled transformers + policy-prompted masking is a clear new perspective; 5-attribute unification is a first.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers ReasonSeg / RefCOCO / gRefCOCO with detailed ablations, though comparisons between single-path inference and MC averaging are missing.
- **Writing Quality**: ⭐⭐⭐⭐ The "telephone game" analogy and visualizations are excellent, though policy gradient formulas are somewhat dense.
- **Value**: ⭐⭐⭐⭐⭐ Provides SOTA results and open-source code; the "intermediate layer + similarity map" paradigm will have a long-term impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] ReferDINO: Referring Video Object Segmentation with Visual Grounding Foundations](../../ICCV2025/segmentation/referdino_referring_video_object_segmentation_with_visual_grounding_foundations.md)
- [\[CVPR 2026\] RealVLG-R1: A Large-Scale Real-World Visual-Language Grounding Benchmark for Robotic Perception and Manipulation](../../CVPR2026/segmentation/realvlg-r1_a_large-scale_real-world_visual-language_grounding_benchmark_for_robo.md)
- [\[AAAI 2026\] EAGLE: Episodic Appearance- and Geometry-Aware Memory for Unified 2D-3D Visual Query Localization](../../AAAI2026/segmentation/eagle_episodic_appearance-_and_geometry-aware_memory_for_unified_2d-3d_visual_qu.md)
- [\[NeurIPS 2025\] UniPixel: Unified Object Referring and Segmentation for Pixel-Level Visual Reasoning](../../NeurIPS2025/segmentation/unipixel_unified_object_referring_and_segmentation_for_pixel-level_visual_reason.md)
- [\[ICCV 2025\] UniGlyph: Unified Segmentation-Conditioned Diffusion for Precise Visual Text Synthesis](../../ICCV2025/segmentation/uniglyph_unified_segmentation-conditioned_diffusion_for_precise_visual_text_synt.md)

</div>

<!-- RELATED:END -->
