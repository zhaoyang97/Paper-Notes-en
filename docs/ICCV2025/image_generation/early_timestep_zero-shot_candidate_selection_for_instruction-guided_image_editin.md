---
title: >-
  [Paper Note] Early Timestep Zero-Shot Candidate Selection for Instruction-Guided Image Editing
description: >-
  [ICCV 2025][Image Generation][Image Editing] This paper proposes ELECT (Early-timestep Latent Evaluation for Candidate selecTion), a zero-shot framework that selects the optimal seed by estimating background inconsistenc…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Image Editing"
  - "Seed Selection"
  - "Diffusion Models"
  - "Background Consistency"
  - "Zero-Shot"
date: 2026-05-08
content_hash: abf6d03d214f1100
---

# Early Timestep Zero-Shot Candidate Selection for Instruction-Guided Image Editing

**Conference**: ICCV 2025
**arXiv**: [2504.13490](https://arxiv.org/abs/2504.13490)
**Code**: [https://github.com/Joow0n-Kim/ELECT](https://github.com/Joow0n-Kim/ELECT)
**Area**: Image Generation
**Keywords**: Image Editing, Seed Selection, Diffusion Models, Background Consistency, Zero-Shot

## TL;DR

This paper proposes ELECT (Early-timestep Latent Evaluation for Candidate selecTion), a zero-shot framework that selects the optimal seed by estimating background inconsistency at early denoising timesteps, reducing computational overhead by 41% (up to 61%) while improving background consistency and instruction-following quality, without requiring external supervision or additional training.

## Background & Motivation

### Reliability Issues in Instruction-Guided Image Editing

Diffusion model-based instruction-guided image editing (e.g., InstructPix2Pix) enables users to modify images via natural language instructions and has broad applications. However, the inherent stochasticity of diffusion models causes the same editing instruction to produce vastly different results across random seeds — some seeds precisely edit the foreground while others severely distort the background or fail to edit altogether.

### Limitations of Prior Work

Users typically resort to trial-and-error by manually switching seeds to find satisfactory results, which is highly inefficient. Existing candidate selection methods suffer from the following issues:

**Designed solely for T2I generation**: Existing seed selection methods (rejection sampling, seed optimization, etc.) are designed for text-to-image tasks, focusing on image quality and prompt alignment without considering background consistency with the source image.

**Reliance on external validators**: Metrics such as aesthetic scores and CLIPScore require complete inference before evaluation, precluding early-stage filtering.

**Linear scaling of computational cost**: Best-of-N strategies require N full inference passes, with computation proportional to N.

### Core Observations and Motivation

Three key observations motivate the proposed method:

**Background MSE selection is effective**: Even without additional models, selecting the seed with the lowest background MSE effectively reduces background distortion and improves editing quality.

**Aggregated relevance maps can substitute GT masks**: Averaging the editing relevance maps across multiple seeds yields masks that approach the quality of ground-truth masks.

**Early timesteps already localize the editing region**: Analysis shows that the early denoising steps (first 20–30 steps) already determine the primary editing regions, with subsequent steps only refining details. This indicates that candidates can be evaluated and filtered at an early stage.

## Method

### Overall Architecture

The ELECT pipeline proceeds as follows: (1) all candidate seeds are denoised in parallel up to a stopping timestep $t_{\text{stop}}$; (2) at $t_{\text{stop}}$, the Tweedie formula is applied to estimate the final editing result for each candidate; (3) the Background Inconsistency Score (BIS) is computed to select the optimal seed; (4) only the selected seed continues the remaining denoising steps. Since most seeds require only partial inference, computation is substantially reduced.

### Key Designs

1. **Background Inconsistency Score (BIS)**:

    - Function: Quantifies undesired background changes in the edited result.
    - Mechanism: BIS combines a softened editing mask with the estimated editing result to measure background deviation:
    $S^{\text{BIS}}(i, t) = (1 - (M_t^{\text{mean}})^2) \odot |\hat{z}_0^i - \mathcal{E}(I)|$
      where $M_t^{\text{mean}} = \frac{1}{|\mathbb{S}|}\sum_{i \in \mathbb{S}} M_t^i$ is the aggregated editing relevance map, and $\hat{z}_0^i$ is the denoised estimate for seed $i$ at timestep $t$ obtained via the Tweedie formula:
    $\hat{z}_0^i = \frac{z_t^i - \sqrt{1-\alpha_t}\epsilon_\theta(z_t^i, t, I, C_T)}{\sqrt{\alpha_t}}$
    - Design Motivation: Using $(M_t^{\text{mean}})^2$ rather than a hard threshold softens the mask, avoiding threshold sensitivity and preventing background regions that are poorly protected from being misclassified as foreground. The continuous weighting assigns high attention to over-edited background regions while down-weighting correctly edited foreground regions.

2. **Aggregated Relevance Maps as GT Mask Substitute**:

    - Function: Estimates the editing region at inference time without ground-truth foreground masks, by averaging relevance maps across multiple seeds.
    - Mechanism: Based on the Watch Your Steps (WYS) relevance map, the difference between conditional and unconditional noise predictions is computed as $M_t = |\epsilon_\theta(z_t, t, I, C_T) - \epsilon_\theta(z_t, t, I, \emptyset)|$. After averaging across all candidate seeds, regions consistently edited across seeds receive high values in $M_t^{\text{mean}}$.
    - Design Motivation: Relevance maps from individual seeds can be noisy; aggregating across seeds yields more stable and reliable estimates.

3. **Early-Stop Evaluation Strategy**:

    - Function: Evaluates all candidates at denoising step $t_{\text{stop}}$, retaining only the best seed for complete inference.
    - Mechanism: The Tweedie formula approximates the final output from an early timestep. Since the signal-to-noise ratio (SNR) reaches 1 after approximately 20 steps, reliable comparisons can be made thereafter. Empirically, most diffusion models stabilize at $t_{\text{stop}}=70$ (i.e., after 30 denoising steps).
    - Design Motivation: Conventional Best-of-N requires full inference for all N seeds ($N \times 100$ steps), whereas ELECT requires only $N \times t_{\text{stop}} + 1 \times (100 - t_{\text{stop}})$ steps.

4. **Temporal Mask Averaging**:

    - Function: Averages relevance maps over timesteps $t \in [80, 100]$ (the first 20 denoising steps).
    - Mechanism: Different samples capture the editing region most reliably at different timesteps; averaging eliminates the instability of any single timestep.
    - Design Motivation: Removes dependence on a fixed timestep and improves the robustness of mask extraction.

5. **Extension: Joint Seed and Prompt Selection**:

    - Function: When seed selection saturates or the instruction itself is problematic, an MLLM is introduced to generate alternative prompts.
    - Mechanism: An MLLM evaluates editing results on instruction-following and background consistency (scored 0/0.5/1 each); if either score is 0, the MLLM rewrites the instruction, and ELECT then selects the optimal prompt.
    - Design Motivation: Seed selection has an inherent upper bound; some failure cases stem from instructions that lie outside the model's distribution.

### Loss & Training

ELECT is a **zero-shot, training-free** inference-time framework. It involves no training process and relies solely on the diffusion model's own denoising procedure to evaluate candidates. It is compatible with both DDIM-based diffusion models and Rectified Flow models.

## Key Experimental Results

### Main Results

Comparison of methods on PIE-Bench ($N=11$ seeds, $t_{\text{stop}}=60$, NFE=500):

| Model | Selection Method | MSE×10⁴↓ | LPIPS×10³↓ | PSNR↑ | CLIP-T↑ | VIEScore↑ |
|------|---------|-----------|-----------|-------|---------|-----------|
| IP2P | Vanilla (1 seed) | 248.5 | 162.4 | 20.73 | 24.38 | 3.43 |
| IP2P | Best of 5 (BIS) | 146.2 | 113.8 | 22.95 | 24.68 | 3.57 |
| IP2P | **ELECT** | **127.5** | **103.3** | **23.33** | **24.97** | **3.67** |
| InsDiff | Vanilla | 372.5 | 154.0 | 20.25 | 24.09 | 3.53 |
| InsDiff | **ELECT** | **180.5** | **104.5** | **22.85** | **24.75** | **3.82** |
| MGIE | Vanilla | 341.4 | 145.5 | 21.16 | 24.44 | 3.68 |
| MGIE | **ELECT** | **185.1** | **102.5** | **23.61** | **24.73** | **3.95** |
| UltraEdit | Vanilla | 87.5 | 115.4 | 22.93 | 25.20 | 4.47 |
| UltraEdit | **ELECT** | **63.8** | **92.3** | **24.49** | **25.36** | **4.70** |

### Ablation Study

Effect of $t_{\text{stop}}$ on performance (IP2P, PIE-Bench):

| Configuration | Description | Performance Trend |
|------|------|---------|
| $t_{\text{stop}}=90$ | Select after only 10 denoising steps | Poor performance; excessive noise |
| $t_{\text{stop}}=80$ | 20 denoising steps, SNR≈1 | Begins to be reliable |
| $t_{\text{stop}}=70$ | 30 denoising steps | Convergence for most models |
| $t_{\text{stop}}=60$ | 40 denoising steps | Stable; required by UltraEdit |
| $t_{\text{stop}}=0$ | Full inference | Equivalent to Best of N by BIS |

Computational efficiency (NFE required for Best-of-N to match ELECT performance):

| Model | Best of N NFE | ELECT NFE | Savings |
|------|-------------|-----------|--------|
| IP2P | ~500 | ~250 | ~50% |
| MagicBrush | ~500 | ~300 | ~40% |
| InsDiff | ~500 | ~200 | ~60% |
| UltraEdit | ~500 | ~320 | ~36% |
| **Average** | - | - | **41%** |

### Key Findings

- ELECT consistently outperforms Best of N by BIS across all tested models, as more seeds can be evaluated within the same NFE budget.
- Improvements in background consistency simultaneously yield better editing quality (both CLIP-T and VIEScore increase), indicating that seeds preserving the background well also tend to edit more precisely.
- Aggregated relevance maps achieve comparable performance to GT masks across all metrics, validating that no additional annotations are required.
- Joint prompt selection provides an additional gain of +0.56 VIEScore, addressing out-of-distribution instruction failures that seed selection alone cannot resolve.
- Approximately 40% of previously failed cases can be successfully recovered by ELECT.

## Highlights & Insights

- The problem is precisely formulated: background sensitivity in image editing is quantified as a computable BIS metric, transforming "finding a good seed" from a vague manual process into a tractable optimization problem.
- Evaluation leverages the diffusion model's own intermediate states, making the framework entirely zero-shot without requiring external models or training.
- The softened mask design eliminates threshold tuning and is more robust than the hard-threshold approach used in WYS.
- The method is compatible with multiple editing models (IP2P, MagicBrush, InsDiff, MGIE, UltraEdit) and supports both diffusion and Rectified Flow architectures.
- MLLM-based evaluation metrics such as VIEScore align closely with human judgment, enhancing the credibility of the results.

## Limitations & Future Work

- BIS is a relative comparison metric and cannot guarantee absolute quality when all candidates are poor.
- There is a potential risk of over-optimizing background preservation at the expense of editing magnitude, though the authors note this is rare and does not significantly affect performance.
- Prompt selection requires an MLLM (GPT-4V), introducing external dependencies and additional computation.
- The default $t_{\text{stop}}$ must be chosen according to the model type (approximately 70 for diffusion models and 60 for Rectified Flow), and a fully adaptive mechanism is currently lacking.

## Related Work & Insights

- The Best-of-N strategy has been widely adopted in inference-time scaling for LLMs; this work extends it to diffusion-based image editing.
- The editing relevance map originates from Watch Your Steps (WYS), but its usage is improved here through aggregation across seeds (replacing single-seed maps) and softening (replacing hard thresholding).
- Unlike mask-guided methods such as Focus on Your Instruction and ZONE, ELECT does not fix the mask but instead filters for the optimal result through multi-seed evaluation.
- The early-stopping concept is generalizable to other generative tasks that require evaluating multiple candidates.

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] AnyPortal: Zero-Shot Consistent Video Background Replacement](anyportal_zero-shot_consistent_video_background_replacement.md)
- [\[ICCV 2025\] ADIEE: Automatic Dataset Creation and Scorer for Instruction-Guided Image Editing Evaluation](adiee_automatic_dataset_creation_and_scorer_for_instruction-guided_image_editing.md)
- [\[ICCV 2025\] AID: Adapting Image2Video Diffusion Models for Instruction-guided Video Prediction](aid_adapting_image2video_diffusion_models_for_instruction-guided_video_predictio.md)
- [\[ICCV 2025\] Addressing Text Embedding Leakage in Diffusion-Based Image Editing](addressing_text_embedding_leakage_in_diffusion-based_image_editing.md)
- [\[ICCV 2025\] SuperEdit: Rectifying and Facilitating Supervision for Instruction-Based Image Editing](superedit_rectifying_and_facilitating_supervision_for_instruction-based_image_ed.md)

</div>

<!-- RELATED:END -->
