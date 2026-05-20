---
title: >-
  [Paper Note] Implicit Preference Alignment for Human Image Animation
description: >-
  [ICML 2026][Image Generation][Preference Alignment] The authors propose Implicit Preference Alignment (IPA): a post-training method that requires only "good samples" and does not need to construct good/bad pairs. By maxi…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Preference Alignment"
  - "DPO"
  - "Flow Matching"
  - "Human Animation"
  - "Hand Generation"
date: 2026-05-08
content_hash: aa9b06429d875858
---

# Implicit Preference Alignment for Human Image Animation

**Conference**: ICML 2026  
**arXiv**: [2605.07545](https://arxiv.org/abs/2605.07545)  
**Code**: https://github.com/mdswyz/IPA (available)  
**Area**: Alignment RLHF / Video Generation / Human Animation  
**Keywords**: Preference Alignment, DPO, Flow Matching, Human Animation, Hand Generation

## TL;DR
The authors propose Implicit Preference Alignment (IPA): a post-training method that requires only "good samples" and does not need to construct good/bad pairs. By maximizing the KL gap with a pretrained reference model, IPA equivalently maximizes an implicit reward. Combined with a HALO module that weights hand masks in the loss, this enables large-scale video DiT models to significantly improve hand fidelity in human animation using only 93 selected samples.

## Background & Motivation

**Background**: Human image animation has recently transitioned from GAN paradigms to diffusion paradigms (Animate Anyone, MimicMotion), and further to large DiT models (VACE, Wan-Animate), achieving high levels of subject appearance and temporal consistency.

**Limitations of Prior Work**: Hands have the highest degrees of freedom and most complex motion, leading to common issues in generated videos such as blurring, missing fingers, and deformation—collectively, "hand collapse." Using RLHF/DPO for hand preference alignment is a natural idea, but DPO requires "strict winner-loser pairing." However, hand states are unstable between frames: in most sampled video pairs, both are acceptable (Case 1), both fail (Case 2), or are of mixed quality (Case 3). True DPO-eligible pairs (Case 4: one good, one bad, frame-aligned) are extremely rare.

**Key Challenge**: The annotation cost for strict good/bad pairing is nearly prohibitive for hand tasks, but abandoning RLHF makes it difficult for SFT to address fine-grained structure.

**Goal**: (i) Design an objective that achieves preference alignment using only "good samples"; (ii) Explicitly focus alignment on the hand ROI; (iii) Retain the large-scale prior knowledge of the pretrained DiT to avoid collapse.

**Key Insight**: The authors observe that "strict pairing is hard, but isolating good samples is relatively cheap"—of 6000 candidates, only 93 good samples were selected (about 7.5% could form DPO pairs). If one can jointly optimize for "closeness to the good sample distribution" and "not deviating from the pretrained prior," the bottleneck of loser samples can be bypassed.

**Core Idea**: Express "the model distribution is closer to the preference distribution $q(X)$ than the reference distribution" as a KL gap $\Delta(p_{\text{ref}}, p_\theta) = D_{\text{KL}}(q\|p_{\text{ref}}) - D_{\text{KL}}(q\|p_\theta) > 0$, and use $-\log\sigma(\beta\Delta)$ as the loss. Theoretically, this is equivalent to reward maximization with a KL constraint (i.e., implicit reward), thus achieving preference alignment using only good samples.

## Method

### Overall Architecture
Using the large DiT model VACE-14B as the pretrained reference $v_{\text{ref}}$, 1500 dance clips are collected from the internet, poses are extracted with DWPose, and a random frame is used as the reference image. VACE generates 4 candidates per pair, totaling 6000 videos. 93 "clear hand" samples are manually selected as preference data $q(X)$. LoRA (rank 128, applied only to QKV projections) is used to train $v_\theta$ with a hand mask-weighted Flow-IPA loss for 1000 steps, $\beta=600$, $\lambda=10$, 8×H20. During inference, $v_\theta$ is used for reverse flow matching sampling.

### Key Designs

1. **Implicit Preference Alignment Objective (IPA loss)**:

    - **Function**: Achieves $p_\theta$ closer to the preference distribution $q$ than $p_{\text{ref}}$, without needing loser samples.
    - **Mechanism**: Enforces $D_{\text{KL}}(q\|p_\theta) < D_{\text{KL}}(q\|p_{\text{ref}})$, rewritten as $\Delta(p_{\text{ref}}, p_\theta) > 0$, then applies log-sigmoid to obtain $\mathcal{L} = -\log\sigma(\beta\Delta(p_{\text{ref}}, p_\theta))$. The authors prove this is equivalent to reward maximization with KL regularization: $\max \mathbb{E}_q[r] - \beta D_{\text{KL}}(p_\theta\|p_{\text{ref}})$, whose optimum satisfies $p_\theta \propto p_{\text{ref}}\exp(r/\beta)$. Back-substituting gives $\mathbb{E}_q[r] = \beta\Delta + C$, so minimizing IPA loss implicitly maximizes the unspecified reward $r$.
    - **Design Motivation**: DPO's contrastive term "pulls good samples, pushes bad samples," but bad samples are scarce. IPA replaces "pushing bad samples" with "not deviating from the reference model (KL constraint)," using the prior as a soft negative signal, thus avoiding loser annotation and preventing mode collapse.

2. **Computable Flow-IPA**:

    - **Function**: Makes the abstract $\Delta(p_{\text{ref}}, p_\theta)$ directly backpropagatable in flow-matching DiT.
    - **Mechanism**: Uses the analytic form of KL increment in Rectified Flow at $t\in[0,1]$: $\frac{d}{dt}D_{\text{KL}} = \frac{1}{2}(1-t)^2 \mathbb{E}\|v - v_\phi(Z_t;t,I,\mathcal{P})\|^2$. Integrating over time yields $\Delta = \mathbb{E}_{t,v}[\frac{1}{2}(1-t)^2(\|v - v_{\text{ref}}\|^2 - \|v - v_\theta\|^2)]$, which is substituted into the log-sigmoid for the final training loss.
    - **Design Motivation**: Direct integration over the probability path is intractable; Flow Matching's "linear interpolation + constant velocity field" allows each sampled time to estimate the KL differential, turning trajectory alignment into a single-point mini-batch loss.

3. **Hand-Aware Local Optimization (HALO)**:

    - **Function**: Explicitly biases alignment toward hand pixels, preventing the "easy-to-learn" body/background from dominating.
    - **Mechanism**: Binary hand masks $\mathbf{M}$ are obtained from DWPose keypoints, constructing spatial weights $\mathbf{W} = \mathbf{1} + \lambda\mathbf{M}$. The loss term $\|v - v_\phi\|^2$ is replaced by $\|\sqrt{\mathbf{W}}\odot(v - v_\phi)\|^2$, i.e., learning velocity field deviations with hand-weighted emphasis. $\lambda=10$ is optimal.
    - **Design Motivation**: Hands occupy a small spatial region in good samples; global MSE weighting causes the model to "spend" loss on the large body area, neglecting hands. HALO's mask weighting pushes gradients back to the hands, amplifying the ROI signal from the limited 93 good samples.

### Loss & Training
The final loss (see Eq.(29)):  
$\mathcal{L} = \mathbb{E}_{t,v}[-\log\sigma(\frac{\beta}{2}(1-t)^2(\|\sqrt{\mathbf{W}}\odot(v - v_{\text{ref}})\|^2 - \|\sqrt{\mathbf{W}}\odot(v - v_\theta)\|^2))]$.  
LoRA finetuning (not full parameters), rank 128, 1000 steps, batch size 8; $\beta=600$ controls constraint strength (both KL penalty and sigmoid slope), $\lambda=10$ controls hand weighting.

## Key Experimental Results

### Main Results

| Dataset | Metric | IPA | Prev. SOTA (Wan-Animate) | Gain |
|---------|--------|-----|--------------------------|------|
| TikTok | FID-VID ↓ | 5.9 | 8.6 | −31% |
| TikTok | FVD ↓ | 255 | 316 | −19% |
| TikTok | SSIM ↑ | 0.841 | 0.799 | +5.3% |
| TikTok | PSNR ↑ | 23.8 | 20.5 | +3.3dB |
| Custom hand bench | FID-VID ↓ | 6.3 | 10.6 (UniAnimate-DiT) | −41% |
| Custom hand bench | SSIM-Hand ↑ | 0.606 | 0.544 | +0.06 |
| Custom hand bench | PSNR-Hand ↑ | 18.9 | 15.3 (VACE) | +3.6dB |

### Ablation Study

| Dataset | IPA | HALO | FID-VID ↓ | FVD ↓ | SSIM ↑ | PSNR ↑ |
|---------|-----|------|-----------|-------|--------|--------|
| TikTok | ✓ | ✓ | 5.9 | 255 | 0.841 | 23.8 |
| TikTok | ✓ | × | 7.9 | 288 | 0.819 | 22.7 |
| TikTok | × | × | 13.4 | 427 | 0.777 | 20.2 |
| Custom | ✓ | ✓ | 6.3 | 224 | 0.757 | 21.5 |
| Custom | × | × | 12.5 | 327 | 0.668 | 18.2 |

### Key Findings
- IPA alone reduces FID from 13.4 → 7.9 (−41%), being the main contributor; HALO further reduces it to 5.9, showing "global alignment + local weighting" are complementary.
- $\beta$ has a clear sweet spot: $\beta=200$ is too weak and overfits the 93 samples; $\beta=1000$ is too strong and learning stalls; $\beta=600$ is optimal.
- $\lambda$ also shows a unimodal trend: increases monotonically from 0.1→10, but at 100 global quality degrades.
- Data efficiency: Only 7 out of 93 good samples (7.5%) can be paired for DPO; under equal cost, DPO is neither fair nor feasible. IPA's main value is lowering the data construction threshold.

## Highlights & Insights
- **Theoretical Rigor**: Derives the log-sigmoid loss from the KL gap and proves its equivalence to implicit reward maximization—filling the theoretical gap of "why RLHF without losers is still valid." Structurally similar to Flow-DPO but with a fundamentally different motivation.
- **Data Paradigm Inspiration**: Relaxing "strict preference pairs (winner, loser)" to "winner only + soft prior constraint" is transferable to many ROI-focused, hard-to-define-loser tasks (medical imaging, handwriting, fine-grained textures).
- **High HALO Reusability**: Mask weighting can be extended from hands to face, eyes, text, or any "small, difficult ROI," making it a nearly free engineering upgrade.
- The dual interpretation of $\beta$—KL strength and sigmoid slope—is a valuable perspective for training dynamics.

## Limitations & Future Work
- Only 93 good samples, all from internet dance videos, so the distribution is narrow; generalization to more complex gestures (sports, sign language, 3D object grasping) is untested.
- Depends on DWPose for masks; mask quality directly limits HALO, and pose estimation failures in extreme occlusion may be detrimental.
- $\beta=600$ and $\lambda=10$ are empirically set; new models/resolutions require re-tuning.
- Only QKV is LoRA-tuned; whether LoRA on all attention/MLP or full finetuning is better is not compared.

## Related Work & Insights
- **vs Diffusion-DPO / Flow-DPO**: Structurally identical, but DPO is derived from the Bradley-Terry model for "winner-loser" contrast, while IPA is derived from the KL gap for "winner vs reference" contrast, thus eliminating the need for losers. The authors emphasize that the contribution lies in the derivation and applicable scenarios, not operator novelty.
- **vs MimicMotion's hand region enhancement**: MimicMotion uses loss reweighting during training; IPA applies mask weighting at the preference alignment (post-training) stage, which is much more cost-effective.
- **vs Animate Anyone / VACE / Wan-Animate**: This work does not redesign the architecture, directly reusing VACE-14B as $v_{\text{ref}}$, exemplifying a "small investment, big return" post-training approach.

## Rating
- Novelty: ⭐⭐⭐⭐ Structure is similar to Flow-DPO but independently derived, and the first to systematically analyze the infeasibility of good/bad pairing for hand tasks.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple baselines, dual benchmarks, dedicated hand metrics, $\beta$ and $\lambda$ grid search; the only regret is no direct DPO comparison on the 7-pair small dataset (though the authors explain why this is unfair).
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical derivation, the 4-case motivation is intuitive; formula numbering is slightly excessive but necessary.
- Value: ⭐⭐⭐⭐ Provides the RLHF post-training community with a "loser-free alignment" paradigm, plug-and-play in engineering, with high transfer potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Boost Your Human Image Generation Model via Direct Preference Optimization](../../CVPR2025/llm_alignment/boost_your_human_image_generation_model_via_direct_preference_optimization.md)
- [\[ACL 2025\] Aligned but Blind: Alignment Increases Implicit Bias by Reducing Awareness of Race](../../ACL2025/llm_alignment/aligned_but_blind_implicit_bias.md)
- [\[ICML 2026\] TUR-DPO: Topology- and Uncertainty-Aware Direct Preference Optimization](tur-dpo_topology-_and_uncertainty-aware_direct_preference_optimization.md)
- [\[ACL 2025\] PKU-SafeRLHF: Towards Multi-Level Safety Alignment for LLMs with Human Preference](../../ACL2025/llm_alignment/pku-saferlhf_towards_multi-level_safety_alignment_for_llms_with_human_preference.md)
- [\[NeurIPS 2025\] Human-assisted Robotic Policy Refinement via Action Preference Optimization](../../NeurIPS2025/llm_alignment/human-assisted_robotic_policy_refinement_via_action_preference_optimization.md)

</div>

<!-- RELATED:END -->
