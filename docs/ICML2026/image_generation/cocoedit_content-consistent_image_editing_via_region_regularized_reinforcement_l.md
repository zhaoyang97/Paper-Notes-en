---
title: >-
  [Paper Note] CoCoEdit: Content-Consistent Image Editing via Region Regularized Reinforcement Learning
description: >-
  [ICML 2026][Image Generation][Content-consistent editing] This paper addresses the issue where "editing models often modify regions that should remain unchanged." It constructs the CoCoEdit-40K local editing dataset…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Content-consistent editing"
  - "pixel-level similarity reward"
  - "regional regularization"
  - "DiffusionNFT"
  - "FLUX/Qwen-Image-Edit"
date: 2026-05-08
content_hash: 25fdb4fe9a5b7067
---

# CoCoEdit: Content-Consistent Image Editing via Region Regularized Reinforcement Learning

**Conference**: ICML 2026  
**arXiv**: [2602.14068](https://arxiv.org/abs/2602.14068)  
**Code**: https://github.com/CoCoEdit (Available)  
**Area**: Image Generation / Instructive Image Editing / RL Post-training  
**Keywords**: Content-consistent editing, pixel-level similarity reward, regional regularization, DiffusionNFT, FLUX/Qwen-Image-Edit  

## TL;DR
This paper addresses the issue where "editing models often modify regions that should remain unchanged." It constructs the CoCoEdit-40K local editing dataset, proposes a pixel-level similarity reward to supplement the MLLM reward, and designs a region-regularized RL objective (where high-reward samples constrain non-editing area consistency, and low-reward samples force changes in editing areas). This approach improves both editing scores and PSNR/SSIM for FLUX.1 Kontext and Qwen-Image-Edit, breaking the existing trade-off where "improving editing capability inevitably hurts consistency."

## Background & Motivation
**Background**: Modern instructive editing models (FLUX.1 Kontext, Qwen-Image-Edit, Step1X-Edit, BAGEL, OmniGen2) can understand instructions well due to massive data and strong generation backbones. Recently, works like Edit-R1 and MotionNFT have used RL (DPO/PPO/GRPO/DiffusionNFT) with MLLM rewards for post-training to further boost editing scores.

**Limitations of Prior Work**: (i) While editing models perform well in target regions, **non-editing areas** are often modified unintentionally—for example, a background pillow might disappear while modifying a foreground person. (ii) Existing RL post-training relies solely on MLLM rewards, which are insensitive to fine-grained pixel differences in non-editing areas. This pushes the model toward drastic modifications for higher scores, causing significant PSNR drops (e.g., Edit-R1 reduces FLUX's PSNR by 5.15 dB).

**Key Challenge**: Editing ability (MLLM Score) and content consistency (PSNR) are in conflict under existing training objectives. The MLLM reward is a space-agnostic scalar and insensitive to small changes; using it for RL inherently sacrifices consistency.

**Goal**: Construct a post-training framework that simultaneously drives (i) accurate editing and (ii) strict preservation of non-editing areas without requiring masks during inference (maintaining fairness with baselines), and modify benchmarks to allow for quantitative consistency evaluation.

**Key Insight**: (a) Use MLLM + SAM2 for offline annotation of editing masks and rewritten instructions for each training sample; (b) Supplement the reward end with a pixel-level similarity reward (masked PSNR/SSIM) to quantify detail differences invisible to MLLMs; (c) At the loss level, use masks to decouple latents into editing and non-editing regions, applying region-level regularization to positive and negative samples respectively.

**Core Idea**: Inject "non-editing area consistency" into RL post-training **simultaneously** as a reward and a region-aware regularizer. High-reward samples (successful edits) preserve non-editing areas, while low-reward samples (under-edited) are forced to change target regions—forming a dual-directional correction loop.

## Method

### Overall Architecture
A three-step RL loop (per iteration). **Step 1: Data & Annotation**: Select local editing samples from OmniEdit/ImgEdit. Use Qwen2.5-VL-72B to generate bboxes $\rightarrow$ SAM2 for masks $\rightarrow$ dilate. Use MLLM with masks to rewrite instructions and filter via Qwen2.5-VL-72B based on instruction clarity, mask accuracy, and target prominence to obtain 40K triplets (image, mask, instruction). **Step 2: Online RL Training**: (i) Sample $N$ outputs $\hat x_0^{1:N}$ from the old policy $v^{old}$; (ii) Calculate MLLM reward $r_{mllm}$ plus normalized PSNR/SSIM in non-editing areas as $r_{sim}$, fused and converted to optimality via $\mathrm{op}(\cdot)$; (iii) Update the policy using DiffusionNFT's implicit positive/negative strategies $v_\theta^\pm$ combined with $L_{ner}^+$ and $L_{er}^-$ regional regularization terms. **Step 3: Inference**: Masks are used only during training. Inference uses pure text instructions. FLUX/Qwen are loaded via LoRA.

### Key Designs

1.  **Pixel-level similarity reward $r_{sim}$ to supplement MLLM blind spots**:
    - **Function**: Explicitly quantifies "whether non-editing area details were modified," which MLLM rewards fail to capture.
    - **Mechanism**: Given input $\hat c_I$, sampled output $\hat x_0$, and editing mask $m$, calculate $\mathrm{PSNR}_m$ and $\mathrm{SSIM}_m$ in non-editing areas. Normalize PSNR to $[0,1]$ to match SSIM scale and average them to get $r_{sim}^{1:N}$. Total reward $r=\mathrm{op}(\lambda_{mllm} r_{mllm}+\lambda_{sim} r_{sim})$ (default $\lambda_{mllm}=0.8, \lambda_{sim}=0.2$).
    - **Design Motivation**: A single MLLM reward might give identical scores to images with the same pose but slightly shifted background details. Adding a pixel-level reward makes "preserving non-editing areas" a differentiable optimization objective. However, if $\lambda_{sim}$ is too large, the model becomes overly conservative and fails to edit; thus, weight favors MLLM.

2.  **Region-decoupled regularization $L_{ner}^+$ + $L_{er}^-$ (Divide and conquer for positive/negative samples)**:
    - **Function**: Applies "non-editing area should be similar" and "editing area should change" constraints only to the most relevant samples, avoiding conflicting goals in a single loss.
    - **Mechanism**: Obtain $x_\theta^+(x_t\mid c)$ (positive policy output) and $x_\theta^-(x_t\mid c)$ (negative policy output) based on DiffusionNFT's $x$-prediction formula. Use downsampled mask $\tilde m$ to define projection operators $P_{ner}(z)=z\odot\tilde m$ and $P_{er}(z)=z\odot(1-\tilde m)$. For **high-reward (positive) samples**, use $L_{ner}^+=\max(0, d(x_\theta^+, c_I)_{\tilde m}-\tau^+)$ to enforce similarity with input latent in non-editing areas. For **low-reward (negative) samples**, use $L_{er}^-=\max(0,\tau^- - d(x_\theta^-, c_I)_{1-\tilde m})$ to force a larger difference in editing areas. Final loss: $\mathcal{L}=\mathbb{E}[r\cdot(\mathcal{L}^+ +\lambda_{ner}L_{ner}^+)+(1-r)\cdot(\mathcal{L}^- +\lambda_{er}L_{er}^-)]$.
    - **Design Motivation**: Scalar rewards lack spatial information. Pixel rewards provide global consistency but cannot constrain regions differently. Splitting the objectives by sample quality—positive samples ensure background preservation, while negative samples ensure foreground change—forms complementary optimization signals and fits the DiffusionNFT framework naturally.

3.  **CoCoEdit-40K: Mask + Rewritten instruction + RL-friendly data pipeline**:
    - **Function**: Upgrades standard image-instruction pairs into (image, mask, refined instruction) triplets and filters by "conditional signal quality" rather than GT result quality.
    - **Mechanism**: (a) Mask Annotation: Qwen2.5-VL-72B generates bbox $\rightarrow$ SAM2 generates mask; (b) Instruction & Mask Augmentation: MLLM expands short instructions into refined versions with spatial locations and attributes; masks are dilated for replace/motion types to cover new content; (c) Data Filtering: Scoring based on clarity, mask accuracy, and prominence. RL doesn't need GT pixels since it learns through exploration.
    - **Design Motivation**: Previous datasets filtered for "good editing results" for SFT. RL needs "clear instructions + accurate masks" so the policy receives precise regional signals during optimization.

### Loss & Training
$\mathcal{L}(\theta)=\mathbb{E}[r\cdot(\mathcal{L}^+ + \lambda_{ner}L_{ner}^+)+(1-r)\cdot(\mathcal{L}^- + \lambda_{er}L_{er}^-)]$. Base $\mathcal{L}^\pm=\|x_\theta^\pm-x_0\|_2^2$, where positive/negative policies are derived from NFT as $v_\theta^\pm = (1\mp\beta)v^{old}\pm\beta v_\theta$. LoRA rank=32. Fine-tuned on 8×A800, batch 3, group 12, 1K steps. VRAM ≈ 70 GB, training time ≈ 12 min per step.

## Key Experimental Results

### Main Results (GEdit-Bench-EN, including PSNR/SSIM/LPIPS/DINO + Rank)

| Method | Overall↑ | PSNR↑ | SSIM↑ | LPIPS↓ | DINO↑ | Human Rank↓ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| FLUX.1 Kontext | 6.286 | 24.168 | 0.825 | 0.150 | 0.871 | 2.1 |
| w/ Edit-R1 | 7.113 | 19.013 | 0.716 | 0.214 | 0.804 | 2.6 |
| **w/ CoCoEdit** | 6.939 | **25.331** | **0.874** | **0.139** | **0.882** | **1.6** |
| Qwen-Image-Edit | 7.560 | 19.488 | 0.662 | 0.185 | 0.831 | 2.7 |
| w/ Edit-R1 | 7.746 | 18.441 | 0.639 | 0.214 | 0.804 | 3.3 |
| w/ MotionNFT | 7.711 | 18.709 | 0.642 | 0.201 | 0.813 | 2.9 |
| **w/ CoCoEdit** | **7.754** | **22.283** | **0.774** | **0.162** | **0.852** | **1.4** |

On Qwen-Image-Edit, CoCoEdit achieves the highest overall editing score (7.754) and highest PSNR (22.283, +2.8 dB), whereas Edit-R1/MotionNFT show improved editing scores at the cost of PSNR.

### Ablation Study

| Setting | GEdit Overall↑ | GEdit PSNR↑ | ImgEdit Overall↑ | ImgEdit PSNR↑ |
| :--- | :--- | :--- | :--- | :--- |
| Qwen-Image-Edit (base) | 7.560 | 19.488 | 3.70 | 17.635 |
| w/ SFT on 40K (incl. consistency loss) | 7.219 | 20.293 | 3.61 | 18.048 |
| w/ RL on 120K | 7.723 | 22.204 | 3.79 | 19.201 |
| **w/ RL on 40K (CoCoEdit)** | **7.754** | 22.283 | 3.79 | 19.125 |

### Key Findings
- 40K high-quality data is sufficient for RL convergence; scaling to 120K yields minimal gains (7.754 vs 7.723), confirming RL prioritizes quality over scale.
- SFT with consistency loss only slightly improves PSNR while Overall scores drop. This indicates CoCoEdit-40K gains come from the RL algorithm and regional regularizers, not just the data.
- Edit-R1 cuts PSNR by 5.15 dB on FLUX; CoCoEdit reverses this trend, validating the motivation to fix the consistency sacrifice in existing RL post-training.

## Highlights & Insights
- **Dual Reward-Regularizer Philosophy**: MLLM reward handles the general direction, pixel reward handles details, and regional regularization handles spatial constraints. These signals complement each other across different dimensions.
- **Sample-Decoupled Regional Regularization**: Applying consistency to positive samples and divergence to negative samples cleverly utilizes the DiffusionNFT framework.
- **Aligning Data Strategy with Training Paradigm**: While SFT requires high-quality target images, RL requires high-quality conditional signals (instructions + masks). This shift is applicable to other RL tasks like video or 3D editing.
- **Training with Masks, Inference without**: Unlike methods that require masks at test time, CoCoEdit performs pure instruction-based inference, ensuring fair comparison with baselines.

## Limitations & Future Work
- Only validated on FLUX.1 Kontext and Qwen-Image-Edit; generalizability to Step1X-Edit or OmniGen2 remains to be tested.
- Primarily focused on local editing. While global style/tone remains competitive, there is no significant gain in those categories.
- Regional regularization thresholds $\tau^\pm$ require type-specific tuning. In "ultra-large" edits (e.g., 90% region replacement), the mask effectively covers everything, nullifying the regularizer.
- RL training cost remains high due to the MLLM-as-reward server requirements.

## Related Work & Insights
- **vs Edit-R1**: Both use DiffusionNFT, but Edit-R1 lacks pixel rewards and regional constraints, leading to lower PSNR (19.01 vs 25.33 for FLUX).
- **vs MotionEdit / MotionNFT**: CoCoEdit provides a more general consistency framework, matching MotionNFT in motion categories while leading in overall metrics.
- **vs DPO / PPO / GRPO**: Using DiffusionNFT avoids the instability of policy gradients in "noisy reward" scenarios like image editing.
- **vs SeedEdit / Step1X-Edit**: These are large-scale pre-trained models. CoCoEdit demonstrates that 40K samples + 1K RL iterations can significantly push the performance of existing top-tier models.

## Rating
- Novelty: ⭐⭐⭐⭐ Pixel-level reward + sample-decoupled regional regularization is a novel and effective combination.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive testing across multiple baselines, benchmarks, and human evaluation.
- Writing Quality: ⭐⭐⭐⭐ Clear derivation and logical flow.
- Value: ⭐⭐⭐⭐⭐ High industrial value for deploying editing models without "background drift" issues.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Path-Coupled Bellman Flows for Distributional Reinforcement Learning](path-coupled_bellman_flows_for_distributional_reinforcement_learning.md)
- [\[ICML 2026\] Offline Multi-agent Reinforcement Learning via Sequential Score Decomposition](offline_multi-agent_reinforcement_learning_via_sequential_score_decomposition.md)
- [\[ICCV 2025\] Multi-turn Consistent Image Editing](../../ICCV2025/image_generation/multi-turn_consistent_image_editing.md)
- [\[ICML 2026\] Content-Style Identification via Differential Independence](content-style_identification_via_differential_independence.md)
- [\[ICLR 2026\] DragFlow: Unleashing DiT Priors with Region Based Supervision for Drag Editing](../../ICLR2026/image_generation/dragflow_unleashing_dit_priors_with_region_based_supervision_for_drag_editing.md)

</div>

<!-- RELATED:END -->
