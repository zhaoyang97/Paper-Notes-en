---
title: >-
  [Paper Note] Envisioning Beyond the Few: Disentangled Semantics and Primitives for Few-Shot Atypical Layout-to-Image Generation
description: >-
  [ICML 2026][Image Generation][layout-to-image] Addressing "representation fragmentation" in 5-shot atypical domain (aerial/underwater/extremely dark) layout-to-image generation…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "layout-to-image"
  - "few-shot adaptation"
  - "representation disentanglement"
  - "visual primitives"
  - "diffusion models"
date: 2026-05-08
content_hash: 21f80a3b287ee6d6
---

# Envisioning Beyond the Few: Disentangled Semantics and Primitives for Few-Shot Atypical Layout-to-Image Generation

**Conference**: ICML 2026  
**arXiv**: [2605.31266](https://arxiv.org/abs/2605.31266)  
**Code**: https://github.com/iCVTEAM/DSP  
**Area**: Diffusion Models / Image Generation / Few-Shot Learning  
**Keywords**: layout-to-image, few-shot adaptation, representation disentanglement, visual primitives, diffusion models

## TL;DR
Addressing "representation fragmentation" in 5-shot atypical domain (aerial/underwater/extremely dark) layout-to-image generation, this work explicitly disentangles the conditional representation of each category into global semantic anchors and local re-composable primitives. By employing a saliency-aware loss to enforce foreground consistency, it reduces Bootstrap FID from 82.5 to 74.3 and improves mAP to 26.1 on the DIOR dataset.

## Background & Motivation

**Background**: Layout-to-image (L2I) generation utilizes categories and bounding boxes (bboxes) to control diffusion models in generating complex scenes. Mainstream approaches (e.g., MIGC, CC-Diff) are based on Stable Diffusion and rely on COCO-scale paired data for training instance-level condition injection.

**Limitations of Prior Work**: When targeting atypical domains such as aerial, underwater, or low-light environments, annotations are scarce, typically with only 5-shot samples available. Directly fine-tuning L2I models in such few-shot settings leads to **representation fragmentation**—generated chimneys are severed, or turtle shells split into fragments, causing simultaneous failures in texture and geometry. This is not merely "overfitting" but a failure of the model to form coherent representations.

**Key Challenge**: The authors attribute the root cause to **granularity mismatch**. High-level semantic identity (e.g., a "turtle") should remain stable across instances, whereas low-level visual details (e.g., shell patterns, lighting) are inherently local and highly variable. Existing L2I conditional paths cram both into the same set of cross-attention embeddings. In few-shot scenarios, high-variance local details dilute stable global semantics, leading to failure in both. Furthermore, since foreground classes of base and novel sets are disjoint while background statistics are largely shared, the loss tends to prioritize fitting the background as a shortcut.

**Goal**: Without modifying the parameters of the base diffusion model, the goal is to (i) stabilize classification identity for novel categories; (ii) recover fine-grained local details despite extremely few samples; and (iii) prevent optimization from taking shortcuts by fitting the background.

**Key Insight**: Rather than competing for capacity in the parameter space (e.g., LoRA, DreamBooth styles), it is more effective to perform granularity disentanglement in the **representation space**. One conditional path focuses on "what category," while another handles "what it looks like," supplemented by a spatial loss to force the model to utilize foreground semantic signals.

**Core Idea**: Explicitly decompose the L2I conditional representation into global *Semantic Anchors* (governing identity) and local *Visual Primitives* (governing details), while using a GradCAM-driven saliency loss, *Conceptual Steering*, to pull gradients toward foreground regions.

## Method

### Overall Architecture
The framework is built upon SD v1.5. During the base stage, general L2I capabilities are learned on large datasets via standard cross-attention, resulting in $\Theta_{\text{base}}$. In the novel stage, the base weights are **completely frozen**, and only the parameters of three new modules, $\Theta_{\text{novel}}$, are updated.

Input conditions for the novel stage undergo *Prior-Grounded Encoding*: global captions and category labels are encoded via frozen CLIP text encoders to produce $\phi_g, \phi_c$; bboxes are Fourier-encoded to yield $\phi_p$ and a sigmoid spatial mask $\mathbf{S}$; visual features are provided by a Resampler pre-trained and **frozen** on the base set to yield background $\phi_b$ and foreground $\phi_f$. Here, the source for $\phi_f$ shifts from the base set to an *exemplar pool* $\Delta$ cropped from the 5-shot novel samples. These five sets of embeddings are injected into the U-Net's middle and upsampling blocks via masked cross-attention. Three new modules take over at different positions along this path: Semantic Anchoring is added to the U-Net middle and first upsampling blocks (modifying $\phi_f$), Primitive Imbuing is added to the last two upsampling blocks (modifying spatial features $\mathbf{h}$), and Conceptual Steering modifies the loss $\mathcal{L}$.

### Key Designs

1.  **Semantic Anchoring**:
    - **Function**: Distills a stable anchor matrix $\mathbf{A}\in\mathbb{R}^{n\times d}$ representing the "visual identity of category $c$" from 5 exemplars and injects it back into the foreground embedding $\phi_f$.
    - **Mechanism**: For category $c$, a subset of exemplars $\delta_c$ is processed by a frozen DINOv2 to extract dense features $\phi_{(\Delta,c)}\in\mathbb{R}^{n\times h\times w\times d}$. $r$ learnable tokens pass through a frozen Resampler via cross-attention to produce $\phi_r\in\mathbb{R}^{n\times r\times d}$, followed by intra-exemplar self-attention and averaging across tokens to obtain $\mathbf{A}$. Injection is performed via gated cross-attention: $\tilde{\phi}_f=\phi_f+\eta\cdot\text{softmax}((\phi_f\mathbf{W}_Q')(\mathbf{A}\mathbf{W}_K')^\top/\sqrt{d}+\mathcal{M})(\mathbf{A}\mathbf{W}_V')$, where the gate $\eta$ is initialized to 0 to preserve the base prior, and mask $\mathcal{M}$ handles padding.
    - **Design Motivation**: Novel categories are disjoint from the base set. A frozen Resampler provides only "coarse" features lacking fine-grained semantics for novel classes. Anchoring via "cross-exemplar consensus" rather than "per-sample fitting" mitigates semantic drift in few-shot settings.

2.  **Primitive Imbuing**:
    - **Function**: Explicitly models re-composable local details using a set of $s=128$ learnable primitives $\mathbf{P}\in\mathbb{R}^{s\times d}$, injected into the late upsampling stages of the U-Net to enhance fine-grained textures.
    - **Mechanism**: DINOv2 features of $\delta_c$ are flattened into $\mathbf{T}\in\mathbb{R}^{nhw\times d}$. $\mathbf{P}$ is initialized via K-Means. **Alternating minimization** is performed targeting $\mathbf{T}\approx\mathbf{W}\mathbf{P}$: with $\mathbf{P}$ fixed, coefficients have a Tikhonov closed-form solution $\hat{\mathbf{W}}=\mathbf{T}\mathbf{P}^\top(\mathbf{P}\mathbf{P}^\top+\lambda\mathbf{I})^{-1}$ ($\lambda=0.1$); with $\mathbf{W}$ fixed, $\mathbf{P}$ is updated by minimizing the Frobenius reconstruction error over $N_{\text{iter}}=50$ iterations. Injection uses spatial gated cross-attention $\tilde{\mathbf{h}}=\mathbf{h}+\gamma\cdot\mathcal{G}\odot\text{softmax}(\cdot)$, where $\mathcal{G}=\mathbf{S}\odot\mathbf{1}_{\text{top}}(\mathbf{S})$ is a sparse spatial gate—point-wise multiplication of the sigmoid mask and top-k selection—restricting primitives strictly to salient foreground regions.
    - **Design Motivation**: Fine-grained details are ill-posed in few-shot learning. The closed-form ridge regression is more numerically stable than iterative SGD. A library of re-composable primitives is more capable of "assembling" plausible textures for new bboxes than learning a complete holistic appearance.

3.  **Conceptual Steering**:
    - **Function**: Adds a spatial penalty mask $\boldsymbol{\Omega}$ to the standard LDM loss to pull gradients toward the foreground, preventing the model from achieving low loss by merely fitting the background.
    - **Mechanism**: Activation maps $\mathbf{M}(I,c)$ and $\mathbf{M}(\hat{I},c)$ for the target class $c$ are obtained via text-driven GradCAM on the ground truth image $I$ and one-step prediction $\hat{I}$. $\boldsymbol{\Omega}$ is defined as $\boldsymbol{\Omega}=\mathbf{1}+\min(|\mathbf{M}(I,c)-\mathbf{M}(\hat{I},c)|/\mu,\,1)$ ($\mu=0.95$), serving as an element-wise weight: $\mathcal{L}_{\text{final}}=\mathbb{E}[\|\boldsymbol{\Omega}\odot(\epsilon_\Theta(x_t,t,\tau(y),\Delta)-\epsilon)\|_2^2]$. Gradients are amplified where there is high activation discrepancy or where the foreground "fails to activate."
    - **Design Motivation**: Base/novel background statistics are shared while foregrounds differ. Standard MSE prioritizes background optimization. By encoding semantic alignment discrepancies into the loss weights, the model is forced to utilize anchors and primitives for the foreground rather than bypassing them.

### Loss & Training
The final training objective is $\mathcal{L}_{\text{final}}$. Optimization uses AdamW with a base learning rate of $1\times 10^{-4}$ and a $100\times$ multiplier for gate parameters $\eta, \gamma$. The base stage involves 100 epochs with a batch size of 320. The novel stage is fixed at 100 steps, using gradient accumulation across all novel samples for a full-batch update. The alternating minimization for primitives is performed as a **pre-processing offline step** and is not part of the SGD loop. Inference uses Euler Discrete Scheduler for 50 steps with CFG=7.5.

## Key Experimental Results

### Main Results
Under 5-shot settings across three atypical domains, the model is compared against MIGC, CC-Diff, and CC-Diff++. Performance is evaluated using a pre-trained Faster R-CNN for alignment (mAP) and Bootstrap FID (to address bias in small sample sizes).

| Dataset | Metric | Prev. SOTA (CC-Diff++) | Ours | Gain |
| :--- | :--- | :--- | :--- | :--- |
| DIOR (Aerial) | FID↓ | 82.62 | **74.34** | -8.28 |
| DIOR | mAP↑ | 24.63 | **26.06** | +1.43 |
| DIOR | AP50↑ | 54.60 | **57.22** | +2.62 |
| RUOD (Underwater) | FID↓ | 46.46 | **45.44** | -1.02 |
| RUOD | mAP↑ | 18.37 | **19.45** | +1.08 |
| ExDark (Very Dark)| FID↓ | 93.09 | **91.36** | -1.73 |
| ExDark | mAP↑ | 35.34 | **35.93** | +0.59 |

When re-evaluated on DIOR using a stronger YOLOv8 detector, the model achieved mAP 20.80 vs CC-Diff++ 19.50, and AP50 43.34 vs 41.20, maintaining the same conclusion.

### Ablation Study (DIOR)
SA = Semantic Anchoring, PI = Primitive Imbuing, CS = Conceptual Steering.

| Configuration | FID↓ | mAP↑ | AP50↑ | Description |
| :--- | :--- | :--- | :--- | :--- |
| Baseline (No modules) | 94.96 | 19.15 | 47.97 | Prior-Grounded Encoding only |
| + SA | 88.57 | 22.84 | 52.05 | Semantic Anchor: FID -6.4, mAP +3.7 |
| + PI | 87.34 | 21.09 | 51.12 | Primitives: FID -7.6, lower mAP gain |
| SA + PI | 85.00 | 25.28 | 56.15 | Synergy; mAP jumps to 25.3 |
| SA + PI + CS (Full) | **74.34** | **26.06** | **57.22** | CS further reduces FID by 10.7 |

In a variant ablation, *PI-SA Swapping* (placing anchors in late upsampling and primitives in the middle) caused mAP to crash to 11.89 and AP50 to 30.77, validating that the hierarchical binding—global semantics in early layers and local primitives in late layers—is essential.

### Key Findings
- All three modules are necessary and have distinct roles: SA primarily improves mAP (semantic alignment), PI focuses on FID (texture fidelity), and CS improves FID while stabilizing mAP—consistent with its design motivation of "pulling gradients toward the foreground."
- The positions of PI and SA injection are not interchangeable, suggesting that "high-level semantics—early layers / local details—late layers" is a structural inductive bias of the U-Net.
- Bootstrap FID is more suitable for few-shot evaluation (50 generated images per class) than vanilla FID; the authors proposed this as an independent metric contribution.

## Highlights & Insights
- The term "representation fragmentation" accurately describes the failure mode—moving beyond vague "overfitting" to "high-frequency details diluting low-frequency semantics," which informs the entire methodology.
- Using a **closed-form ridge regression + K-Means initialization** for learning primitives is a clean trick: it avoids SGD instability in few-shot settings and is computationally efficient as an offline step.
- *Conceptual Steering* utilizes GradCAM as a spatial weight for the loss rather than just a post-hoc visualization tool—a generalizable approach for tasks with small foregrounds, large backgrounds, and asymmetric base/novel distributions (e.g., medical lesions, remote sensing).
- Initializing gate parameters $\eta, \gamma$ to 0 with a $100\times$ learning rate multiplier preserves pre-trained capabilities while allowing rapid adaptation of new modules, mirroring effective strategies used in ControlNet and IP-Adapter.

## Limitations & Future Work
- Evaluation is limited to 5-shot settings; the potential degradation of "cross-exemplar consensus" in 1-shot or the suitability of $s=128$ primitives in 10-shot remains unexplored.
- The atypical domains are "object-level" (aerial, underwater, low-light); generalization to truly heterogeneous layouts, such as medical or industrial defects, is unknown, as foreground saliency assumptions may not hold.
- FID improvement on ExDark was only 1.7 points, significantly lower than on DIOR—suggesting that when the original foreground signal is extremely weak, the $\boldsymbol{\Omega}$ provided by GradCAM may be inaccurate.
- The methodology is specific to the U-Net cross-attention structure of SD v1.5; migrating to DiT, SD3, or Flux-style Transformer backbones would require re-locating the hierarchical binding of SA and PI.

## Related Work & Insights
- **vs MIGC / CC-Diff / CC-Diff++**: Similar instance-level L2I goals, but those assume large-scale training data and degrade in 5-shot atypical domains; this work leaves base parameters frozen and adds disentangled modules, reducing FID on DIOR by 8 points over CC-Diff++.
- **vs DreamBooth / DataDream**: Similar few-shot personalization, but those are *subject-driven* (one token per instance) without bbox control; this work explicitly uses layout inputs and disentangles identity from details, making it more suitable for multi-instance scenes.
- **vs IP-Adapter Style Methods**: Both use gated cross-attention for image conditions, but IP-Adapter uses a single global image embedding. This work splits the image-conditioning path into dual granularities—anchor and primitive—targeting semantics and details respectively.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Precise naming of representation fragmentation and a clear disentanglement framework. While individual components (DINOv2, Resamplers, Primitives, GradCAM) are existing tricks, their specific combination is the primary contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers three atypical domains, two detectors, comprehensive ablations including variant swapping, and introduces Bootstrap FID. Lacks K-shot scaling and migration to larger models.
- **Writing Quality**: ⭐⭐⭐⭐ The Introduction clearly builds the logic for "representation fragmentation." The Method section is organized logically (identity, then details, then shortcut prevention) with complete formulas and algorithms.
- **Value**: ⭐⭐⭐⭐ Provides a ready-to-use solution for few-shot atypical L2I with open-source code; the disentanglement strategy and GradCAM-as-loss are valuable for other few-shot generation tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Uni-DAD: Unified Distillation and Adaptation of Diffusion Models for Few-step Few-shot Image Generation](../../CVPR2026/image_generation/uni-dad_unified_distillation_and_adaptation_of_diffusion_models_for_few-step_few.md)
- [\[CVPR 2026\] Few-shot Acoustic Synthesis with Multimodal Flow Matching](../../CVPR2026/image_generation/few-shot_acoustic_synthesis_with_multimodal_flow_matching.md)
- [\[CVPR 2026\] V-Bridge: Bridging Video Generative Priors to Versatile Few-shot Image Restoration](../../CVPR2026/image_generation/v-bridge_bridging_video_generative_priors_to_versatile_few-shot_image_restoratio.md)
- [\[ICCV 2025\] HypDAE: Hyperbolic Diffusion Autoencoders for Hierarchical Few-shot Image Generation](../../ICCV2025/image_generation/hypdae_hyperbolic_diffusion_autoencoders_for_hierarchical_few-shot_image_generat.md)
- [\[ICML 2026\] OcclusionFormer: Arranging Z-Order for Layout-Grounded Image Generation](occlusionformer_arranging_z-order_for_layout-grounded_image_generation.md)

</div>

<!-- RELATED:END -->
