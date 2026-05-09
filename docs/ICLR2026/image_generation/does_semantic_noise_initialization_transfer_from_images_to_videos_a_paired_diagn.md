---
title: >-
  [Paper Note] Does Semantic Noise Initialization Transfer from Images to Videos? A Paired Diagnostic Study
description: >-
  [ICLR 2026][Image Generation][semantic noise initialization] Through rigorous prompt-level paired statistical testing, this work finds that transferring semantic noise initialization (golden noise) from the image domain to video diffusion models yields a marginally positive but statistically insignificant trend on temporal metrics (p≈0.17). Noise-space diagnostics reveal that insufficient directional stability and spatiotemporal frequency structure discrepancies are the root causes.
tags:
  - ICLR 2026
  - Image Generation
  - semantic noise initialization
  - text-to-video diffusion
  - golden noise
  - paired evaluation
  - noise-space diagnostics
date: 2026-05-08
content_hash: 0bb33d861e9dcbf7
---

# Does Semantic Noise Initialization Transfer from Images to Videos? A Paired Diagnostic Study

**Conference**: ICLR 2026
**arXiv**: [2603.06672](https://arxiv.org/abs/2603.06672)
**Code**: [GitHub](https://github.com/klkds/golden-noise-transfer)
**Area**: Image/Video Generation
**Keywords**: semantic noise initialization, text-to-video diffusion, golden noise, paired evaluation, noise-space diagnostics

## TL;DR

Through rigorous prompt-level paired statistical testing, this work finds that transferring semantic noise initialization (golden noise) from the image domain to video diffusion models yields a marginally positive but statistically insignificant trend on temporal metrics (p≈0.17). Noise-space diagnostics reveal that insufficient directional stability and spatiotemporal frequency structure discrepancies are the root causes.

## Background & Motivation

- **Background**: Text-to-video (T2V) diffusion models are highly sensitive to random seeds: identical prompts with different initial Gaussian noise can produce videos that differ substantially in semantics and motion. Recent work in image generation has demonstrated that teacher-aligned semantic noise initialization (the "golden noise" strategy) improves generation robustness and controllability by shifting the initial noise distribution closer to regions that yield high-quality outputs.
- **Limitations of Prior Work**: A natural hypothesis is that video generation may benefit even more from this strategy, since temporal dynamics amplify seed-induced variance. However, the spatiotemporal coupling inherent to video introduces additional degrees of freedom and instability, making the success of such a transfer unclear.
- **Goal**: To conduct a systematic diagnostic study: can semantic noise initialization transfer successfully from images to video diffusion models, and if not, what are the underlying reasons?

## Method

### Overall Architecture

The study is framed as a diagnostic rather than a breakthrough contribution. The overall pipeline proceeds as follows:
1. Train a lightweight noise mapper NPNet on a frozen VideoCrafter-style T2V backbone.
2. Conduct standardized evaluation on 100 prompts using VBench.
3. Rigorously quantify the effect via prompt-level paired statistical testing (bootstrap CI + sign-flip permutation test).
4. Analyze root causes through noise-space diagnostics comparing Open-Sora2 and VideoCrafter.

### Key Designs

1. **Semantic (Golden) Noise Targets**: Semantically aligned target noise $z_T^*$ is constructed by searching in noise space for initializations that produce higher semantic and temporal quality. Concretely, DDIM inversion or optimization within a teacher diffusion model is used to extract target noise. For video, this is computationally expensive, as each candidate noise requires a full spatiotemporal denoising pass.

2. **NPNet Lightweight Noise Mapper**: A prompt-conditioned mapping network $f_\phi$ is trained to transform standard Gaussian noise into semantically initialized noise:
   - Input: standard Gaussian noise $z_T$ and text prompt $p$ (injected via text embedding)
   - Output: semantically aligned noise $\hat{z}_T = f_\phi(z_T, p)$
   - Training loss: regression loss $\mathcal{L}(\phi) = \mathbb{E}[\|f_\phi(z_T, p) - z_T^*\|^2]$
   - At inference, only the initial noise is replaced; the backbone remains fully frozen.

3. **Paired Statistical Testing Design**: This constitutes the paper's core methodological contribution. For each prompt, 5 random seeds are used; scores are averaged over seeds and then subjected to prompt-level paired difference analysis (statistical unit: prompt, $N=100$). Bootstrap 95% CIs and sign-flip permutation test $p$-values are reported.

4. **Cross-Model Noise-Space Diagnostics**: The geometry and frequency characteristics of golden noise are analyzed across two video diffusion models, Open-Sora2 and VideoCrafter:
   - Displacement vector $d = z_g - z$ (difference between golden and standard noise)
   - Directional Stability (DirStab): mean cosine similarity of unit displacement vectors across seeds
   - Explained Variance Ratio (EVR1): variance fraction captured by the first PCA component
   - Spatial/temporal high-frequency ratio: FFT analysis of the frequency structure of displacements

### Loss & Training

- NPNet is trained with a simple L2 regression loss.
- The backbone (VideoCrafter) is fully frozen throughout training.
- Golden noise targets are pre-extracted using the teacher model.

## Key Experimental Results

### Main Results

Evaluation on 100 VBench prompts with 5 seeds per prompt:

| Metric | Baseline | NPNet | Δ |
|--------|----------|-------|---|
| aesthetic_quality | 0.638 | 0.635 | -0.003 |
| imaging_quality | 0.715 | 0.708 | -0.007 |
| background_consistency | 0.977 | 0.977 | +0.000 |
| subject_consistency | 0.978 | 0.978 | +0.000 |
| temporal_style | 0.077 | 0.079 | +0.002 |

The 95% bootstrap CI for temporal_style includes zero; $p=0.1687$, not significant.

### Ablation Study

| Metric | Open-Sora2 | VideoCrafter |
|--------|-----------|-------------|
| DirStab (Directional Stability) | 0.631 | 0.200 |
| EVR1 (Explained Variance Ratio) | 0.464 | 0.343 |
| CV_\|d\| (Displacement Norm CV) | 0.064 | 0.110 |
| Spatial HF Δ | -0.0005 | -0.0149 |

### Key Findings

- The directional stability of golden noise displacements in Open-Sora2 is far higher than in VideoCrafter (0.631 vs. 0.200).
- DDIM sampling in VideoCrafter rotates and diffuses initial directional perturbations, causing signal degradation.
- Displacement $d$ is spatially smooth but temporally high-frequency; this spatiotemporal imbalance may amplify flicker/jitter through the denoising process.
- The overall regime is one of low signal-to-noise ratio: prompt-level variance substantially exceeds the effect size.

## Highlights & Insights

1. **Methodological Demonstration**: The paper exemplifies how rigorous paired statistical testing can be used to evaluate small effects, serving as a valuable corrective to the often imprecise comparisons prevalent in the diffusion model community.
2. **Noise-Space Diagnostic Toolkit**: The proposed metrics—directional stability, frequency analysis, etc.—provide a systematic framework for understanding noise-space transformations.
3. **Value of Negative Results**: The honest reporting of an unsuccessful transfer, accompanied by in-depth mechanistic analysis, carries greater academic value than reporting only positive outcomes.
4. **Cross-Model Comparison Reveals Mechanism**: The Open-Sora2 vs. VideoCrafter comparison highlights the critical role of the sampler (DDIM vs. others) in how noise perturbations propagate.
5. **"Signal Exists but Is Fragile" Diagnosis**: The paper precisely characterizes the core issue—not an absence of signal, but spatiotemporal frequency characteristics that render the signal insufficiently stable under standard evaluation.

## Limitations & Future Work

1. Validation is limited to a VideoCrafter-style backbone; different architectures (e.g., DiT-based models) may yield different outcomes.
2. VBench metrics may not fully capture human preferences, particularly prompt-specific motion artifacts.
3. Frequency analysis reveals correlations but does not constitute causal proof.
4. The computational cost of extracting golden noise targets for video is prohibitively large, and cost-effectiveness may remain poor even with improvements.
5. Whether dynamic sampling strategies (e.g., adaptive guidance scale) could mitigate signal instability remains unexplored.

## Related Work & Insights

- **Golden Noise (Zhou et al., 2025)**: The original image-domain semantic noise method, of which this work is a video extension.
- **Noise Hypernetworks (Eyring et al., 2025)**: An alternative approach that amortizes test-time computation into the noise space.
- **VBench (Huang et al., 2024)**: A standardized evaluation suite for video generation.
- This work provides important baselines and methodological references for research on noise initialization strategies in video generation.
- It suggests that future work should analyze the propagation characteristics of the sampler in noise space prior to applying noise-space manipulations.

## Rating

- Novelty: ⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Conjuring Semantic Similarity](conjuring_semantic_similarity.md)
- [\[AAAI 2026\] Rectified Noise: A Generative Model Using Positive-incentive Noise](../../AAAI2026/image_generation/rectified_noise_a_generative_model_using_positive-incentive_noise.md)
- [\[ICLR 2026\] Diverse Text-to-Image Generation via Contrastive Noise Optimization](diverse_text-to-image_generation_via_contrastive_noise_optimization.md)
- [\[ICLR 2026\] Does FLUX Already Know How to Perform Physically Plausible Image Composition?](does_flux_already_know_how_to_perform_physically_plausible_image_composition.md)
- [\[ICLR 2026\] Flow Matching with Injected Noise for Offline-to-Online Reinforcement Learning](flow_matching_with_injected_noise_for_offline-to-online_reinforcement_learning.md)

<!-- RELATED:END -->
