---
title: >-
  [Paper Note] CoCoEdit: Content-Consistent Image Editing via Region Regularized Reinforcement Learning
description: >-
  [ICML 2026][Image Generation][Content-consistent editing] This work addresses the issue of "editing models making unintended changes in non-editing regions" by constructing the CoCoEdit-40K local editing dataset…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Content-consistent editing"
  - "pixel-level similarity reward"
  - "region regularization"
  - "DiffusionNFT"
  - "FLUX/Qwen-Image-Edit"
date: 2026-05-08
content_hash: b8897230402b16c4
---

# CoCoEdit: Content-Consistent Image Editing via Region Regularized Reinforcement Learning

**Conference**: ICML 2026  
**arXiv**: [2602.14068](https://arxiv.org/abs/2602.14068)  
**Code**: https://github.com/CoCoEdit (available)  
**Area**: Image Generation / Instruction-based Image Editing / RL Post-training  
**Keywords**: Content-consistent editing, pixel-level similarity reward, region regularization, DiffusionNFT, FLUX/Qwen-Image-Edit

## TL;DR
This work addresses the issue of "editing models making unintended changes in non-editing regions" by constructing the CoCoEdit-40K local editing dataset, introducing a pixel-level similarity reward to complement the MLLM reward, and designing a region-regularized RL objective (constraining non-editing regions for high-reward samples, forcing changes in editing regions for low-reward samples). This approach improves both editing scores and PSNR/SSIM for FLUX.1 Kontext and Qwen-Image-Edit, breaking the existing trade-off between editing capability and content consistency.

## Background & Motivation
**Background**: Modern instruction-based editing models (FLUX.1 Kontext, Qwen-Image-Edit, Step1X-Edit, BAGEL, OmniGen2) leverage large-scale data and strong generative backbones to understand instructions well. Recently, Edit-R1, MotionNFT, etc., have used RL (DPO/PPO/GRPO/DiffusionNFT) with MLLM reward for post-training, further boosting editing scores.

**Limitations of Prior Work**: (i) Editing models perform well in the intended editing regions but often unintentionally modify non-editing regions—for example, changing the foreground person causes the background pillow to disappear; (ii) Existing RL post-training uses only MLLM reward, which is insensitive to fine-grained pixel differences in non-editing regions, pushing models to make drastic changes for higher scores, resulting in significant PSNR drops (Edit-R1 reduces FLUX's PSNR by 5.15 dB).

**Key Challenge**: Editing capability (MLLM Score) and content consistency (PSNR) are conflicting under current training objectives: MLLM reward is a spatially-agnostic scalar and insensitive to minor changes; using it for RL inevitably sacrifices consistency.

**Goal**: Construct a post-training framework that simultaneously drives (i) accurate editing and (ii) strict preservation of non-editing regions, without requiring additional masks at inference (ensuring fair comparison with baselines), and modify benchmarks to enable quantifiable evaluation of consistency.

**Key Insight**: (a) Use MLLM + SAM2 to offline annotate editing masks and rewrite instructions for each training sample; (b) Add a pixel-level similarity reward (masked PSNR/SSIM) to the reward to quantify details invisible to MLLM; (c) Use masks in the loss to separate latents into editing and non-editing regions, applying region-level regularization to positive and negative samples respectively.

**Core Idea**: Treat "non-editing region consistency" as both a reward and a region-aware regularizer in RL post-training, constraining non-editing regions for high-reward (successful edit) samples and forcing changes in editing regions for low-reward (under-edited) samples—forming a bidirectional correction loop.

## Method

### Overall Architecture
Three-step RL loop (per iteration). **Step 1 Data & Annotation**: Select local editing samples from OmniEdit/ImgEdit, use Qwen2.5-VL-72B to generate bbox → SAM2 for mask → dilate → MLLM rewrites instruction using mask, then filter with Qwen2.5-VL-72B based on instruction clarity / mask accuracy / target prominence, yielding 40K triplets (image, mask, instruction). **Step 2 Online RL Training**: (i) Sample $N$ samples $\hat x_0^{1:N}$ from old policy $v^{old}$; (ii) Compute MLLM reward $r_{mllm}$ + normalized PSNR/SSIM in non-editing regions to get $r_{sim}$, fuse with weights and apply $\mathrm{op}(\cdot)$ for optimality; (iii) Use DiffusionNFT's implicit positive/negative policies $v_\theta^\pm$ + add $L_{ner}^+$ and $L_{er}^-$ region regularization terms for policy update. **Step 3 Inference**: Mask is used only during training; inference uses pure text instructions, FLUX/Qwen loaded via LoRA.

### Key Designs

1. **Pixel-level similarity reward $r_{sim}$ complements MLLM blind spots**:

    - **Function**: Explicitly quantifies "whether non-editing region details are unintentionally changed" as a reward signal, which MLLM reward cannot see.
    - **Mechanism**: Given input $\hat c_I$, sampled output $\hat x_0$, and editing mask $m$, compute $\mathrm{PSNR}_m, \mathrm{SSIM}_m$ in non-editing regions; normalize PSNR to $[0,1]$ to match SSIM scale, average to get $r_{sim}^{1:N}$; total reward $r=\mathrm{op}(\lambda_{mllm} r_{mllm}+\lambda_{sim} r_{sim})$ ($\lambda_{mllm}=0.8,\lambda_{sim}=0.2$ by default, op is optimality conversion).
    - **Design Motivation**: MLLM reward alone gives nearly identical scores for "same pose but slight background changes" (see Fig.5), causing non-editing regions to drift after RL training; adding pixel-level reward makes "preserving non-editing regions" a differentiable optimization target. However, if pixel reward weight is too high ($\lambda_{sim}=0.5$), the model becomes overly conservative and does not edit at all—thus $\lambda$ must favor MLLM.

2. **Region-decoupled regularization $L_{ner}^+$ + $L_{er}^-$ (positive/negative sample separation)**:

    - **Function**: Applies "non-editing region similarity" and "editing region difference" constraints only to the most relevant samples, avoiding a single loss handling two opposing objectives.
    - **Mechanism**: Based on DiffusionNFT's $x$-prediction, obtain $x_\theta^+(x_t\mid c)$ (positive policy output) and $x_\theta^-(x_t\mid c)$ (negative policy output), use downsampled mask $\tilde m$ to define projection operators $P_{ner}(z)=z\odot\tilde m$, $P_{er}(z)=z\odot(1-\tilde m)$. For **high-reward (positive) samples**, use $L_{ner}^+=\max(0, d(x_\theta^+, c_I)_{\tilde m}-\tau^+)$ to encourage similarity to input latent in non-editing regions (hinge $\tau^+$ allows small deviations); for **low-reward (negative) samples**, use $L_{er}^-=\max(0,\tau^- - d(x_\theta^-, c_I)_{1-\tilde m})$ to force greater difference in editing regions (prevent under-editing). Final loss: $\mathcal{L}=\mathbb{E}[r\cdot(\mathcal{L}^+ +\lambda_{ner}L_{ner}^+)+(1-r)\cdot(\mathcal{L}^- +\lambda_{er}L_{er}^-)]$.
    - **Design Motivation**: Single reward is a scalar, lacking spatial information; pixel reward provides global consistency but cannot separately constrain editing/non-editing regions. Directly combining "non-editing region similarity + editing region difference" in one loss causes conflict; separating by positive/negative samples—positive samples focus on "don't damage other regions" after successful editing, negative samples focus on "make changes" if editing fails—provides complementary optimization signals, naturally fitting NFT's implicit positive/negative policy framework.

3. **CoCoEdit-40K: mask + rewritten instruction + RL-friendly data pipeline**:

    - **Function**: Upgrades OmniEdit/ImgEdit from "image-instruction pairs" to (image, mask, refined instruction) triplets, filtering by "conditional signal quality" rather than GT editing result quality.
    - **Mechanism**: (a) Mask Annotation: Qwen2.5-VL-72B for bbox → SAM2 for mask; (b) Instruction & Mask Augmentation: MLLM expands short instructions into refined instructions with spatial location and object attributes, for replace/motion types dilate mask to cover new content; (c) Data Filtering: Score by instruction clarity / mask accuracy / target prominence, retain high-scoring samples, no need for GT edited images (RL does not learn ground-truth pixels).
    - **Design Motivation**: Previous datasets filter by "editing result quality" for SFT; RL explores via reward, requiring "clear instructions + accurate masks" rather than pretty GTs—these two signals enable precise regional feedback per sample. This data strategy is tightly coupled with the RL objective.

### Loss & Training
$\mathcal{L}(\theta)=\mathbb{E}[r\cdot(\mathcal{L}^+ + \lambda_{ner}L_{ner}^+)+(1-r)\cdot(\mathcal{L}^- + \lambda_{er}L_{er}^-)]$. Base $\mathcal{L}^\pm=\|x_\theta^\pm-x_0\|_2^2$, positive/negative policies from NFT's $v_\theta^\pm = (1\mp\beta)v^{old}\pm\beta v_\theta$. LoRA rank=32, FLUX.1 Kontext / Qwen-Image-Edit each fine-tuned, 8×A800, batch 3, group 12, 1K steps, VRAM ≈ 70 GB (same as Edit-R1), each step ≈ 12 min (2 min more than Edit-R1).

## Key Experimental Results

### Main Results (GEdit-Bench-EN, with PSNR/SSIM/LPIPS/DINO + Rank)

| Method | Overall↑ | PSNR↑ | SSIM↑ | LPIPS↓ | DINO↑ | Human Rank↓ |
|--------|---------|-------|-------|--------|-------|-------------|
| FLUX.1 Kontext | 6.286 | 24.168 | 0.825 | 0.150 | 0.871 | 2.1 |
| w/ Edit-R1 | 7.113 | 19.013 | 0.716 | 0.214 | 0.804 | 2.6 |
| **w/ CoCoEdit** | 6.939 | **25.331** | **0.874** | **0.139** | **0.882** | **1.6** |
| Qwen-Image-Edit | 7.560 | 19.488 | 0.662 | 0.185 | 0.831 | 2.7 |
| w/ Edit-R1 | 7.746 | 18.441 | 0.639 | 0.214 | 0.804 | 3.3 |
| w/ MotionNFT | 7.711 | 18.709 | 0.642 | 0.201 | 0.813 | 2.9 |
| **w/ CoCoEdit** | **7.754** | **22.283** | **0.774** | **0.162** | **0.852** | **1.4** |

On Qwen-Image-Edit, CoCoEdit achieves both the highest editing score (7.754) and highest PSNR (22.283, +2.8 dB), while Edit-R1/MotionNFT improve editing scores but decrease PSNR. On ImgEdit-Bench, PSNR increases by +1.16 dB / +1.49 dB, with Overall also improving.

### Ablation Study

| Setting | GEdit Overall↑ | GEdit PSNR↑ | ImgEdit Overall↑ | ImgEdit PSNR↑ |
|---------|--------------|-------------|-----------------|----------------|
| Qwen-Image-Edit (base) | 7.560 | 19.488 | 3.70 | 17.635 |
| w/ SFT on 40K (with consistency loss) | 7.219 | 20.293 | 3.61 | 18.048 |
| w/ RL on 120K | 7.723 | 22.204 | 3.79 | 19.201 |
| **w/ RL on 40K (CoCoEdit)** | **7.754** | 22.283 | 3.79 | 19.125 |

| Reward Ratio | Phenomenon |
|--------------|------------|
| $\lambda_{mllm}=0.5,\lambda_{sim}=0.5$ | Editing score collapses, PSNR soars → model does not edit at all |
| $\lambda_{mllm}=0.8,\lambda_{sim}=0.2$ | Editing steadily improves, consistency has an upper bound |
| + Region regularization (default) | Editing score further improves + faster convergence |

### Key Findings
- 40K high-quality data is sufficient for RL convergence; scaling to 120K yields almost no gain (7.754 vs 7.723), confirming RL values quality over quantity.
- SFT with consistency loss only slightly improves PSNR, but Overall drops (7.219 < 7.560), indicating CoCoEdit-40K is not designed for SFT—the gain mainly comes from the RL algorithm + region regularizer, not the data.
- Edit-R1 reduces PSNR by 5.15 dB on FLUX and 1.04 dB on Qwen; MotionNFT is similar—validating the core motivation that "existing RL post-training favors editing capability at the expense of consistency."
- Global editing (style/tone/extract), though not trained, remains competitive, even slightly outperforming in Style (Qwen 6.992 vs 6.666); pixel consistency training benefits global style transformations that preserve structure.

## Highlights & Insights
- **Dual design philosophy of reward and regularizer**: MLLM reward captures overall direction, pixel reward captures details, region regularization applies spatial constraints to positive/negative samples separately—three signals complement each other in different dimensions, avoiding side effects from any single signal being too strong (pure pixel reward leads to no editing, pure MLLM reward leads to over-editing).
- **Region regularization with positive/negative sample separation**: Assigning "non-editing region consistency" to positive samples and "editing region must change" to negative samples leverages DiffusionNFT's implicit positive/negative policy—a single loss delivers opposite gradients on different samples, achieving automatic trade-off.
- **Transferable data strategy**: Traditional editing datasets filter by "GT image quality" for SFT; this work filters by "conditional signal (instruction + mask) quality" for RL—aligning data selection with training paradigm, a strategy applicable to other RL post-training tasks (video editing, 3D editing).
- **Mask used in training, not inference, for alignment**: Unlike methods such as FireEdit that require masks at inference, CoCoEdit uses pure instructions at inference, enabling fair comparison with baselines—an easily overlooked but important design choice for practitioners.

## Limitations & Future Work
- Only validated on FLUX.1 Kontext and Qwen-Image-Edit; whether other large editing models (Step1X-Edit, OmniGen2, BAGEL) require retraining LoRA is untested.
- Training data is all local editing; for global style/tone, there is no significant improvement though no severe degradation; motion editing only shows clear improvement on Qwen.
- Region regularization thresholds $\tau^\pm$ are adaptive but require type-specific tuning (Appendix B.3); optimal $\tau$ may differ by editing type; for large edits (e.g., 90% region replaced), mask degenerates to all ones, making region regularizer ineffective.
- Training requires MLLM-as-reward (Qwen2.5-VL-32B on a dedicated server), so training cost remains high; for negative sentiment/extreme instructions, reward noise increases risk of under-editing.

## Related Work & Insights
- **vs Edit-R1 (UniWorld-V2)**: Also uses DiffusionNFT for image editing RL, but Edit-R1 uses image generation datasets and only MLLM reward; CoCoEdit uses image editing data + pixel reward + region regularization, improving PSNR by 6.3 dB over Edit-R1 (FLUX 25.33 vs 19.01).
- **vs MotionEdit / MotionNFT**: MotionNFT adds motion-alignment reward to enhance motion editing; CoCoEdit provides a more general consistency framework, matching MotionNFT in motion category (Qwen 7.758 vs MotionNFT 7.760) and outperforming in others.
- **vs DPO / PPO / GRPO**: This work uses DiffusionNFT to avoid policy gradient (DanceGRPO, Flow-GRPO also use GRPO), demonstrating that in image editing—where reward signals are noisy and spatially sensitive—contrasting positive/negative policies + region regularization is more stable than pure PG.
- **vs SeedEdit / Step1X-Edit / LongCat-Image**: These are strong baselines trained on large-scale data; CoCoEdit does not require large-scale retraining, only 40K + 1K iterations of RL post-training to further boost top models, making it very accessible for resource-limited researchers.

## Rating
- Novelty: ⭐⭐⭐⭐ Pixel-level reward complements MLLM + region regularization with positive/negative sample separation is the first explicit combination in this subfield; DiffusionNFT application is also clever.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two major baseline models × two major benchmarks × multiple baselines + local/global editing + data scale / SFT ablation + human evaluation, very comprehensive.
- Writing Quality: ⭐⭐⭐⭐ "Three-step loop" process is clear, motivation for positive/negative sample separation is naturally derived, limitations and efficiency analysis are honest.
- Value: ⭐⭐⭐⭐⭐ Effectively solves the pain point of "unintended background changes" in large editing model deployment, can be plug-and-play for existing SOTA models to simultaneously improve editing quality and consistency, with high industrial application value.

## Related Papers

- [\[ICML 2026\] Consistent Diffusion Language Models](consistent_diffusion_language_models.md)
- [\[ECCV 2024\] RegionDrag: Fast Region-Based Image Editing with Diffusion Models](../../ECCV2024/image_generation/regiondrag_fast_region-based_image_editing_with_diffusion_models.md)
- [\[ICCV 2025\] Multi-turn Consistent Image Editing](../../ICCV2025/image_generation/multi-turn_consistent_image_editing.md)
- [\[ICLR 2026\] DragFlow: Unleashing DiT Priors with Region Based Supervision for Drag Editing](../../ICLR2026/image_generation/dragflow_unleashing_dit_priors_with_region_based_supervision_for_drag_editing.md)
- [\[CVPR 2026\] Refining Few-Step Text-to-Multiview Diffusion via Reinforcement Learning](../../CVPR2026/image_generation/refining_few-step_text-to-multiview_diffusion_via_reinforcement_learning.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Multi-turn Consistent Image Editing](../../ICCV2025/image_generation/multi-turn_consistent_image_editing.md)
- [\[ICLR 2026\] DragFlow: Unleashing DiT Priors with Region Based Supervision for Drag Editing](../../ICLR2026/image_generation/dragflow_unleashing_dit_priors_with_region_based_supervision_for_drag_editing.md)
- [\[ICLR 2026\] Follow-Your-Shape: Shape-Aware Image Editing via Trajectory-Guided Region Control](../../ICLR2026/image_generation/follow-your-shape_shape-aware_image_editing_via_trajectory-guided_region_control.md)
- [\[AAAI 2026\] EchoGen: Cycle-Consistent Learning for Unified Layout-Image Generation and Understanding](../../AAAI2026/image_generation/echogen_cycle-consistent_learning_for_unified_layout-image_generation_and_unders.md)
- [\[ICLR 2026\] Offline Reinforcement Learning with Generative Trajectory Policies](../../ICLR2026/image_generation/offline_reinforcement_learning_with_generative_trajectory_policies.md)

</div>

<!-- RELATED:END -->
