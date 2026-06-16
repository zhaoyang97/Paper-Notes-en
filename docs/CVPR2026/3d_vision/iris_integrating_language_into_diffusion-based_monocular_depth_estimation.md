---
title: >-
  [Paper Note] Iris: Integrating Language into Diffusion-based Monocular Depth Estimation
description: >-
  [CVPR 2026][3D Vision][Diffusion Model] Iris systematically validates a naive hypothesis: feeding additional text descriptions of scene objects into a diffusion-based monocular depth estimator leverages the "text $\leftrightarrow$ 3D scene" conditional distribution learned during text-to-image pre-training. This reduces the depth solution space, leading to o
tags:
  - CVPR 2026
  - 3D Vision
  - Diffusion Model
date: 2026-05-08
content_hash: dbc668b8e0971695
---
# Iris: Integrating Language into Diffusion-based Monocular Depth Estimation

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Zeng_Iris_Integrating_Language_into_Diffusion_based_Monocular_Depth_Estimation_CVPR_2026_paper.html)  
**Code**: [Project Page Iris-website](https://adonis-galaxy.github.io/Iris-website/)  
**Area**: 3D Vision / Monocular Depth Estimation  
**Keywords**: Monocular Depth Estimation, Diffusion Models, Language Conditioning, Vision-Language, Text-guidance

## TL;DR
Iris systematically validates a naive hypothesis: feeding additional text descriptions of scene objects into a diffusion-based monocular depth estimator leverages the "text $\leftrightarrow$ 3D scene" conditional distribution learned during text-to-image pre-training. This reduces the depth solution space, leading to overall zero-shot accuracy improvements across three diffusion MDEs (Marigold, Lotus, and E2E-FT), particularly for small objects and blurry regions, while also accelerating training and inference convergence.

## Background & Motivation

**Background**: Monocular Depth Estimation (MDE) aims to predict pixel-wise depth from a single RGB image. Diffusion-based MDEs (such as Marigold, Lotus, and E2E-FT), fine-tuned from pre-trained text-to-image (T2I) diffusion models, generate high-quality and detail-rich depth maps through iterative denoising, representing a mainstream research direction.

**Limitations of Prior Work**: Monocular depth estimation is inherently an ill-posed problem; recovering 3D geometry from a single 2D image is geometrically non-unique. This difficulty is compounded by visual ambiguities such as repetitive textures and homogeneous surfaces. Specifically, **small objects, fine structures, and distant objects with low pixel coverage** are easily smoothed out or omitted during denoising or filtering, leading to poor detail fidelity. While feed-forward methods (like Depth Anything v2) exhibit strong generalization, they rely on massive training datasets that are difficult to replicate, whereas diffusion methods still frequently fail in these ambiguous regions.

**Key Challenge**: Existing diffusion MDEs only take images as input, discarding their "ancestry"—the conditional distribution of "text descriptions $\leftrightarrow$ plausible 3D scene layouts" that the models originally learned during T2I pre-training. In other words, a readily available clue for narrowing the solution space—language—is being wasted.

**Goal**: To re-inject text descriptions as additional inputs into the training and inference of diffusion MDEs, systematically verifying whether "language can improve depth estimation fidelity" and quantifying its impact across four dimensions: overall accuracy, local small objects, iterative refinement, and convergence speed.

**Key Insight**: The authors hypothesize that knowing which objects are in a scene helps locate and more faithfully reconstruct them, even if the objects are visually inconspicuous. Knowledge of typical object shapes can also provide more regularized estimates in ambiguous areas, and object co-occurrence relationships can even help infer unmentioned objects. Text naturally carries these priors.

**Core Idea**: Instead of retraining the entire model, a text encoder is added to existing diffusion MDEs to use descriptions as denoising conditions. This leverages the "text $\leftrightarrow$ scene" distribution left over from T2I pre-training to narrow the depth solution space. This is the first work to systematically verify that text can improve diffusion MDE.

## Method

### Overall Architecture
The Iris method itself is lightweight: based on existing diffusion MDEs (Marigold/Lotus/E2E-FT), it introduces a **frozen CLIP text encoder** to use the image $x$ and text description $c$ together as denoising conditions. During training, the ground truth depth $y^*$ is encoded into a latent variable $z_y$ using a frozen VAE. Forward diffusion gradually adds noise to obtain $z_t=\sqrt{\bar\alpha_t}z_y+\sqrt{1-\bar\alpha_t}\epsilon$. The diffusion U-Net (initialized from Stable Diffusion v2) predicts the noise $\epsilon_\theta(z_t,t,x,c)$ at step $t$, with the loss being the standard noise regression $L(\theta)=\mathbb{E}_{y,\epsilon,t}[\|\epsilon-\epsilon_\theta(z_t,t,x,c)\|^2]$. During inference, starting from pure Gaussian noise $z_T$, reverse denoising is performed at each step according to $z_{t-1}=\frac{1}{\sqrt{\alpha_t}}(z_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(z_t,t,x,c))$ to obtain $z_0$, which is then decoded by the frozen VAE decoder to get the depth $\hat y=\mathcal D(z_0)$. Since the modification is simply adding a condition to standard diffusion MDE, the mechanism is explained via formulas and process text.

### Key Designs

**1. Injecting Text as a Denoising Condition into Diffusion MDE: Narrowing the Solution Space with Pre-trained T2I Distributions**

This is the essence of the method. Given text $c=\{c_1,c_2,\dots\}$, it is first encoded by the frozen CLIP text encoder and fed into the diffusion model. The image $x$ is encoded using the same VAE as the depth, $\mathcal E(x)$, and concatenated with the depth latent $z_t$ as input. Thus, the noise prediction is conditioned on $(z_t,t,x,c)$, gradually refining Gaussian noise into a depth map that honors both the image and the text description. Why it works: T2I pre-training teaches the model to generate images consistent with text under various perspectives and layouts, meaning object shapes, sizes, and spatial relationships are implicitly encoded in its latent representation. Re-introducing text as a condition invokes this "text $\leftrightarrow$ plausible 3D scene" distribution, narrowing the set of 3D scenes compatible with the image and thereby improving depth accuracy. It also allows users to input different descriptions to "emphasize" specific error-prone objects.

**2. Simulating Human Annotation with VLM-Generated Descriptions: Overcoming the Engineering Bottleneck of Missing Text Labels**

Standard depth benchmarks do not come with scene descriptions, and manual annotation per image is impractical. The authors use off-the-shelf vision-language models (VLMs) to generate text aimed at simulating human annotation: LLaVA v1.6 is used for Marigold, and InternVL3-8B for Lotus and E2E-FT, generating a description for each training and test image. This step enables the "text condition" to be evaluated on standard benchmarks without human labels, providing the prerequisite for the systemic experiments. However, it also means the performance upper bound is constrained by the quality of the generated descriptions.

**3. Iterative Text Refinement: Turning Descriptions into an Interactive Error Correction Tool**

Since text is a conditional input, the authors demonstrate a natural application: by appending a new description to the end of the original one and rerunning inference, the depth estimation can be refined step-by-step for the regions or objects named (e.g., adding "a metal rack" and then "a black football helmet hanging on the rack"). This transforms depth estimation from a "one-time prediction" into an "iterative correction" process. When the model makes an error, the user can provide a specific description to correct that region, echoing the finding that text can locally influence depth.

### An Example: Correcting Depth via Appended Descriptions 
Taking the cluttered indoor scene in Figure 6 of the paper as an example, one can observe how text gradually narrows the depth solution space: the initial description is just "a cluttered indoor space with an armchair in the foreground and a folding chair against the wall," leaving the depth of the wall area ambiguous. Next, "a metal rack with various items hanging on it" is appended to the description, and inference is rerun, correcting the depth of the rack area. Finally, appending "a black football helmet hanging on the rack" allows the even smaller, more easily ignored object to be correctly perceived. Throughout this process, model weights remain fixed; only the enriched condition $c$ narrows the "set of compatible 3D scenes"—this is the interactivity provided by treating language as a condition rather than a fixed parameter.

## Key Experimental Results

### Main Results
The models were trained on two synthetic datasets, HyperSim (indoor) and Virtual KITTI 2 (street view), and evaluated zero-shot on five real-world datasets: NYUv2, KITTI, ETH3D, ScanNet, and DIODE. Metrics: **$\delta1 \uparrow$** (percentage of pixels where the ratio of prediction to ground truth is within 1.25), and **AbsRel $\downarrow$** (average absolute relative error). All evaluations follow the affine-invariant protocol without ensemble. `+ Text` indicates injecting text into the base model (default: training and inference). `*` denotes results reproduced by the authors using open-source code.

| Model | NYUv2 $\delta1 \uparrow$ / AbsRel $\downarrow$ | KITTI $\delta1 \uparrow$ / AbsRel $\downarrow$ | ETH3D $\delta1 \uparrow$ / AbsRel $\downarrow$ | ScanNet $\delta1 \uparrow$ / AbsRel $\downarrow$ |
|------|------|------|------|------|
| Marigold | 95.9 / 6.0 | 90.4 / 10.5 | 95.1 / 7.1 | 94.5 / 6.9 |
| Marigold + Text | 95.9 / **5.9** | **90.6** / **10.4** | **95.7** / **6.5** | **94.9** / **6.7** |
| Lotus-D* | 96.6 / 5.6 | 92.2 / 8.7 | 96.8 / 6.1 | 96.0 / 6.0 |
| Lotus-D + Text | **96.8** / **5.4** | **93.0** / **8.4** | **97.0** / **6.0** | **96.6** / **5.6** |
| E2E-FT* | 95.4 / 6.9 | 90.1 / 10.5 | 94.1 / 8.1 | 94.6 / 7.7 |
| E2E-FT + Text | **96.3** / **6.2** | **91.7** / **9.7** | **94.7** / **7.8** | **95.0** / 7.5 |

Overall trend: Indices improved for most datasets after injecting text into the three diffusion MDEs. The authors also found that injecting text "training-only" or "inference-only" occasionally yields improvements, indicating that language conditions can both regularize the diffusion training process and guide depth during zero-shot inference.

### Local and Efficiency Analysis
MaskDINO was used for panoptic segmentation on the NYUv2 test set, evaluating depth for small objects based on mask area percentages of $<5\%/10\%/20\%$.

| NYUv2 Region | Marigold $\delta1 \uparrow$ / AbsRel $\downarrow$ | Marigold+Text $\delta1 \uparrow$ / AbsRel $\downarrow$ |
|------|------|------|
| Full Standard | 95.7 / 6.1 | 95.9 / 5.9 |
| Small (<5%) | 91.7 / 9.0 | **92.8 / 8.3** |
| Small (<10%) | 92.2 / 8.4 | **93.1 / 7.9** |
| Small (<20%) | 93.9 / 7.3 | **94.6 / 6.8** |

### Key Findings
- **Small objects gain the most**: The smaller the region, the harder it is for the baseline to estimate accurately, and the more significant the relative improvement from text injection ($<5\%$ area $\delta1$: $91.7 \to 92.8$). This confirms that "language provides existence and shape priors for inconspicuous objects."
- **Accelerated Convergence**: Injecting text not only results in faster training convergence but also significantly reduces the number of denoising steps during inference. `+ Text` converges in just 10 steps, whereas the baseline requires 25 steps to reach comparable results. The authors attribute this to the additional semantic/geometric constraints provided by language.
- **Text can correct or mislead**: Iteratively appending correct descriptions can improve specific regions. However, describing a "bookshelf with glass" as a "window with curtains" will mislead the model into misjudging the structure (Figure 9)—performance is heavily dependent on description quality.

## Highlights & Insights
- **"Reviving" discarded pre-training priors**: Diffusion MDEs were originally chosen for their implicit 3D scene knowledge but discarded text inputs. Iris points out that "reconnecting text" recovers this prior almost for free. Simple but effective.
- **Zero-shot controllability**: Text is a condition rather than a fixed parameter. Users can change descriptions, emphasize objects, or iteratively supplement info at inference time. This transforms an end-to-end black box into an interactive, error-correctable system with potential for safety-sensitive scenarios like autonomous driving.
- **Consistency across models**: Gains are consistent across Marigold, Lotus, and E2E-FT, suggesting that "text conditioning" is a universal, decoupled increment that can be plugged into various diffusion MDEs.
- **Practical efficiency bonus**: Text injection allows inference to drop from 25 steps to 10 while maintaining convergence, effectively yielding a $\sim 2.5\times$ speedup while improving accuracy—a significant longitudinal benefit for the slow inference problem in diffusion MDE.
- **Identifies an overlooked evaluation dimension**: Using panoptic segmentation to evaluate small object depth by area quantifies how "overall average metrics mask the degradation of small objects," which provides methodological value for future work.

## Limitations & Future Work
- **High dependence on description quality**: Ambiguous or incorrect text can hinder or mislead predictions. Manual descriptions can also be incomplete or subtly ambiguous. The authors suggest introducing language robustness modules (e.g., uncertainty estimation, consistency filtering) to correct or filter bad descriptions.
- **Reliance on VLM-generated labels**: The descriptions used for training and evaluation are generated by LLaVA/InternVL. Biases and hallucinations from these VLMs may propagate into depth estimation, and the gap between "simulated" and "real" human descriptions is not fully quantified.
- **Incremental methodological novelty**: The core contribution is systematic empirical evidence rather than a brand-new architecture—it is essentially "adding a text condition"—but its value lies in being the first to clarify this efficacy.

## Related Work & Insights
- **vs. Marigold / Lotus / E2E-FT (Base models)**: These models fine-tune from T2I diffusion but only accept images. Iris re-introduces text conditions as an addition rather than a replacement, universally improving accuracy and convergence.
- **vs. WorDepth / RSA (Text-aided depth)**: WorDepth learns variational priors of 3D scenes from text, and RSA uses text to predict scale to align relative depth to metric depth. These mostly use text in non-diffusion architectures. This paper focuses on the previously unstudied question of "the impact of text on **diffusion-based** MDE."
- **vs. DepthCLIP**: DepthCLIP uses CLIP's semantic depth responses for zero-shot ordinal depth. This paper does not use language as the depth signal itself but as a condition to narrow the solution space and guide diffusion denoising.

## Rating
- Novelty: ⭐⭐⭐ Simple idea (reconnecting text conditions), but wins on "first systematic verification" rather than architectural innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three models $\times$ five datasets $\times$ three injection modes + multi-dimensional analysis (small objects/convergence/iteration); solid empirical evidence.
- Writing Quality: ⭐⭐⭐⭐ Clear hypothesis-verification logic, four findings are well-organized, with sufficient visualization.
- Value: ⭐⭐⭐⭐ Provides a plug-and-play, interactively correctable increment that is practical for the diffusion MDE community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Iris: Bringing Real-World Priors into Diffusion Model for Monocular Depth Estimation](iris_bringing_realworld_priors_into_diffusion_model_for_monocular_depth_estimation.md)
- [\[CVPR 2026\] DuoMo: Dual Motion Diffusion for World-Space Human Reconstruction](duomo_dual_motion_diffusion_for_world-space_human_reconstruction.md)
- [\[CVPR 2026\] MD2E: Modeling Depth-to-Edge Cues for Monocular Metric Depth Estimation](md2e_modeling_depth-to-edge_cues_for_monocular_metric_depth_estimation.md)
- [\[CVPR 2026\] Depth Hypothesis Guided Iterative Refinement for Event-Image Monocular Depth Estimation](depth_hypothesis_guided_iterative_refinement_for_event-image_monocular_depth_est.md)
- [\[CVPR 2026\] AIMDepth: Asymmetric Image-Event Mamba for Monocular Depth Estimation](aimdepth_asymmetric_image-event_mamba_for_monocular_depth_estimation.md)

</div>

<!-- RELATED:END -->
