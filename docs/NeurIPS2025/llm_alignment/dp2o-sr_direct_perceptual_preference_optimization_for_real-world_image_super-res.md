---
title: >-
  [Paper Note] DP²O-SR: Direct Perceptual Preference Optimization for Real-World Image Super-Resolution
description: >-
  [NeurIPS 2025][LLM Alignment][Image Super-Resolution] This paper proposes DP²O-SR, a framework that exploits the inherent stochasticity of diffusion models to generate diverse super-resolution outputs…
tags:
  - "NeurIPS 2025"
  - "LLM Alignment"
  - "Image Super-Resolution"
  - "Preference Optimization"
  - "Diffusion Models"
  - "Perceptual Quality"
  - "DPO"
date: 2026-05-08
content_hash: f44dbcd39455129a
---

# DP²O-SR: Direct Perceptual Preference Optimization for Real-World Image Super-Resolution

**Conference**: NeurIPS 2025
**arXiv**: [2510.18851](https://arxiv.org/abs/2510.18851)
**Code**: [github.com/cswry/DP2O-SR](https://github.com/cswry/DP2O-SR)
**Area**: LLM Alignment
**Keywords**: Image Super-Resolution, Preference Optimization, Diffusion Models, Perceptual Quality, DPO

## TL;DR

This paper proposes DP²O-SR, a framework that exploits the inherent stochasticity of diffusion models to generate diverse super-resolution outputs, constructs preference pairs via a hybrid perceptual reward, and introduces a Hierarchical Preference Optimization (HPO) strategy to adaptively weight training pairs — significantly improving perceptual quality in real-world image super-resolution without any human annotations.

## Background & Motivation

Real-world Image Super-Resolution (Real-ISR) aims to reconstruct high-resolution images from low-resolution inputs. Traditional methods focus on pixel-level fidelity (PSNR/SSIM) but tend to produce over-smoothed results. Recent methods based on pretrained Text-to-Image (T2I) diffusion models (e.g., Stable Diffusion, FLUX) demonstrate strong capability in synthesizing rich details, yet they share a fundamental issue: **the inherent stochasticity of diffusion models** — different noise inputs lead to outputs with substantially varying perceptual quality.

Existing methods typically treat this stochasticity as a defect and attempt to eliminate it by stabilizing the generation process or training single-step models. This paper takes a fundamentally different perspective: **treating stochasticity as a source of high-quality supervision signals**. Outputs from different noise seeds form a diverse distribution over perceptual quality, which can be leveraged to construct preference pairs for optimization.

Furthermore, existing DPO methods (e.g., Diff-DPO) construct only a single "best vs. worst" preference pair from different model outputs, yielding limited supervision. There is also a lack of carefully designed perceptual rewards and preference data curation strategies tailored to Real-ISR.

## Method

### Overall Architecture

The DP²O-SR framework consists of three core components:

1. **Diverse ISR Sample Generation**: A frozen reference model $\pi_{\text{ref}}$ samples $M$ SR candidates from the same LR input using different noise seeds.
2. **Perceptual Reward Ranking and Preference Pair Construction**: A hybrid IQA reward ranks candidates; top-$N$ and bottom-$N$ outputs are selected to form $N^2$ preference pairs.
3. **Hierarchical Preference Optimization (HPO)**: Training pairs are adaptively weighted to focus on the most informative comparisons.

### Key Designs

#### 1. Hybrid Perceptual Reward Design

The reward signal combines **full-reference (FR)** and **no-reference (NR)** metrics:

- **FR set $\mathcal{FR}$**: LPIPS, TOPIQ-FR, AFINE-FR — promoting structural fidelity and suppressing hallucinated content.
- **NR set $\mathcal{NR}$**: MANIQA, MUSIQ, CLIPIQA+, TOPIQ-NR, AFINE-NR, Q-Align — encouraging realism and aesthetic consistency.

For each candidate $I_m$ and metric $\phi$, raw scores $s_m^\phi$ are direction-aligned and min-max normalized:

$$\bar{s}_m^\phi = \frac{s_m^\phi - s_{\min}^\phi}{s_{\max}^\phi - s_{\min}^\phi}$$

The final reward equally weights FR and NR contributions:

$$R_m = \frac{0.5}{|\mathcal{FR}|}\sum_{\phi \in \mathcal{FR}} \bar{s}_m^\phi + \frac{0.5}{|\mathcal{NR}|}\sum_{\phi \in \mathcal{NR}} \bar{s}_m^\phi$$

Key finding: using only FR rewards leads to over-smoothing; using only NR rewards introduces hallucinated details; the hybrid reward preserves structural consistency while enhancing realism.

#### 2. Preference Pair Curation Strategy

Unlike Diff-DPO, which constructs a single "best vs. worst" pair, this work samples $M$ outputs from the same model and selects top-$N$ and bottom-$N$ to form $N^2$ preference pairs. Two key control parameters are introduced:

- **Sample count $M$**: Larger $M$ improves perceptual diversity and training stability, with diminishing returns.
- **Selection ratio $N/M$**: Small $N/M$ yields stronger reward contrast; large $N/M$ increases coverage and diversity.

Key finding (architecture sensitivity):
- **Smaller model C-SD2 (0.8B)**: Optimal at $N/M = 1/4$, requiring more redundancy and smoother gradients.
- **Larger model C-FLUX (12B)**: Optimal at $N/M = 1/16$, capable of learning effectively from high-contrast signals.

#### 3. Hierarchical Preference Optimization (HPO)

Adaptive weighting is applied at two levels:

**Intra-group weight** (focusing on pairs with larger reward gaps):
$$w_{\text{intra}}(\mathbf{x}_0^w, \mathbf{x}_0^l) = |R_w - R_l| + (1 - \mu_{\text{gap}})$$

where $\mu_{\text{gap}}$ is the mean reward gap across all pairs within the group.

**Inter-group weight** (prioritizing LR input groups with higher perceptual diversity):
$$w_{\text{inter}}(g) = \sigma_g + (1 - \mu_\sigma)$$

where $\sigma_g$ is the reward standard deviation within group $g$, and $\mu_\sigma$ is the mean standard deviation across all groups.

### Loss & Training

The final loss extends Diff-DPO with hierarchical weights:

$$\mathcal{L}_{HPO} = \sum_{(\mathbf{x}_0^w, \mathbf{x}_0^l)} w \cdot \ell(\mathbf{x}_0^w, \mathbf{x}_0^l; \theta)$$

where $w = w_{\text{intra}} \cdot w_{\text{inter}}$ and $\ell(\cdot)$ denotes the Diff-DPO loss.

Training configuration: batch size 1024, learning rate $2 \times 10^{-5}$, $\beta = 5000$, 1000 training steps, 8×A800 GPUs. C-SD2 uses $N=8, M=32$; C-FLUX uses $N=4, M=64$.

## Key Experimental Results

### Main Results

Evaluation is conducted on Syn-Test and RealSR benchmarks across 14 IQA metrics in 4 categories.

**Key improvements on Syn-Test** (DP²O-SR vs. baseline):

| Metric | C-SD2 | DP²O-SR(SD2) | C-FLUX | DP²O-SR(FLUX) |
|--------|-------|-------------|--------|--------------|
| MANIQA↑ | 0.6684 | **0.7165** | 0.6857 | **0.7199** |
| CLIPIQA+↑ | 0.7595 | **0.8124** | 0.7473 | **0.7993** |
| QALIGN↑ | 4.2481 | **4.5526** | 4.4266 | **4.7060** |
| VQ-R1↑ | 4.43 | **4.57** | 4.53 | **4.65** |

**Results on RealSR** (out-of-domain generalization, with standard deviation):

| Metric | C-SD2 | DP²O-SR(SD2) | C-FLUX | DP²O-SR(FLUX) |
|--------|-------|-------------|--------|--------------|
| MANIQA↑ | 0.664±0.019 | **0.705±0.012** | 0.665±0.025 | **0.694±0.013** |
| MUSIQ↑ | 70.34±1.79 | **73.24±0.81** | 69.70±2.15 | **72.78±0.93** |
| QALIGN↑ | 3.630±0.187 | **4.017±0.117** | 3.654±0.231 | **4.143±0.113** |

DP²O-SR not only improves perceptual quality but also **substantially reduces output variance** (reflected in markedly smaller standard deviations), indicating improved generation stability.

### Ablation Study

**Effect of sample count $M$**: Increasing $M$ consistently improves performance with diminishing returns; saturation is approached at $M=64$.

**Effect of selection ratio $N/M$**:
- C-SD2 is optimal at $N/M = 1/4$; excessively low ratios cause reward collapse.
- C-FLUX is optimal at $N/M = 1/16$; performance degrades at $N/M = 1/2$.

**Effectiveness of HPO**: Hierarchical weighting substantially improves training efficiency and final perceptual quality over uniform-weight Diff-DPO.

### Key Findings

1. **Perception–distortion trade-off**: Improvements in perceptual quality accompany slight decreases in PSNR/SSIM, consistent with classical theory.
2. **Strong generalization**: Superior performance is also observed on metrics not used during training, such as VQ-R1 and NIMA.
3. **Architecture generality**: The method is effective on both diffusion models (SD2, UNet) and flow-based models (FLUX, DiT).
4. **Fast convergence**: 500 steps suffice to surpass strong baselines such as SeeSR and OSEDiff.
5. **Improved robustness**: DP²O-SR significantly improves Worst@M, ensuring even the worst-case outputs reach high quality.

## Highlights & Insights

1. **Turning stochasticity into an advantage**: Reframing diffusion model randomness as a source of preference learning signals is a clever and non-obvious insight.
2. **Balanced hybrid reward**: Equal weighting of FR and NR metrics avoids the biases introduced by relying on either alone.
3. **Two-level adaptive weighting in HPO**: Intra-group weighting targets hard pairs; inter-group weighting prioritizes informative inputs — a well-motivated design.
4. **Architecture-aware hyperparameter selection**: Reveals a systematic relationship between model capacity and optimal preference pair curation strategy.
5. **No human annotations required**: Fully automated preference data construction offers strong scalability.

## Limitations & Future Work

1. **High offline candidate generation cost**: Sampling 64 outputs for 30K images requires 168–432 GPU hours; IQA annotation adds another 72 hours.
2. **Degradation in PSNR/SSIM**: Although theoretically expected, this may be problematic in applications with strict fidelity requirements.
3. **Robustness of reward metric selection**: Whether equal weighting of FR and NR is universally optimal remains an open question; different applications may require adjustment.
4. **Validation limited to ControlNet architectures**: Generalizability to other Real-ISR architectures (e.g., end-to-end trained models) has not been verified.
5. **Potential reward hacking**: Whether prolonged training causes over-optimization toward specific IQA metrics warrants investigation.

## Related Work & Insights

- **Diff-DPO**: The direct baseline of this work; constructs only a single preference pair. This paper effectively extends it via multi-sampling and ranking.
- **RLHF → DPO**: Transfers preference alignment ideas from the LLM domain to visual generation — a productive cross-domain adaptation.
- **Perception–distortion trade-off theory**: Provides theoretical grounding for the observed trade-off between perceptual quality and pixel-level accuracy.
- **Inspiration**: The preference alignment paradigm is extendable to other visual generation tasks (video super-resolution, image restoration, image generation); the hybrid reward design is broadly applicable.

## Rating

- Novelty: ⭐⭐⭐⭐ — Converting diffusion stochasticity into preference supervision signals is innovative; HPO design is creative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 14 metrics, two architectures, extensive ablations; very comprehensive.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with natural logical progression.
- Value: ⭐⭐⭐⭐ — Provides a systematic framework for preference alignment in visual generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Rethinking Direct Preference Optimization in Diffusion Models](rethinking_direct_preference_optimization_in_diffusion_models.md)
- [\[NeurIPS 2025\] On Extending Direct Preference Optimization to Accommodate Ties](on_extending_direct_preference_optimization_to_accommodate_ties.md)
- [\[NeurIPS 2025\] KL Penalty Control via Perturbation for Direct Preference Optimization](kl_penalty_control_via_perturbation_for_direct_preference_optimization.md)
- [\[NeurIPS 2025\] PolyJuice Makes It Real: Black-Box, Universal Red Teaming for Synthetic Image Detectors](polyjuice_makes_it_real_black-box_universal_red_teaming_for_synthetic_image_dete.md)
- [\[NeurIPS 2025\] Robust LLM Alignment via Distributionally Robust Direct Preference Optimization](robust_llm_alignment_via_distributionally_robust_direct_preference_optimization.md)

</div>

<!-- RELATED:END -->
