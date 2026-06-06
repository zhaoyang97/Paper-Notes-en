---
title: >-
  [Paper Note] Implicit Preference Alignment for Human Image Animation
description: >-
  [ICML 2026][LLM Alignment][Preference Alignment] The authors propose Implicit Preference Alignment (IPA): a post-training method that requires only "good samples" without the need for constructing winner-loser pairs. By…
tags:
  - "ICML 2026"
  - "LLM Alignment"
  - "Preference Alignment"
  - "DPO"
  - "Flow Matching"
  - "Human Animation"
  - "Hand Generation"
date: 2026-05-08
content_hash: 858167cad4880296
---

# Implicit Preference Alignment for Human Image Animation

**Conference**: ICML 2026  
**arXiv**: [2605.07545](https://arxiv.org/abs/2605.07545)  
**Code**: https://github.com/mdswyz/IPA (Available)  
**Area**: Alignment RLHF / Video Generation / Human Animation  
**Keywords**: Preference Alignment, DPO, Flow Matching, Human Animation, Hand Generation

## TL;DR
The authors propose Implicit Preference Alignment (IPA): a post-training method that requires only "good samples" without the need for constructing winner-loser pairs. By maximizing the KL interval relative to a pre-trained reference model to equivalently maximize implicit rewards, combined with a Hand-Aware Local Optimization (HALO) module that weights hand masks into the loss, the method significantly improves hand fidelity in large-scale video DiTs using only 93 selected samples.

## Background & Motivation

**Background**: Human image animation has recently transitioned from GAN paradigms to diffusion paradigms (Animate Anyone, MimicMotion) and further toward large-scale DiT models (VACE, Wan-Animate), achieving high levels of subject appearance and temporal consistency.

**Limitations of Prior Work**: Fingers have the highest degrees of freedom and motion complexity, leading to common "hand collapse" issues such as blurring, missing fingers, or deformities in generated videos. While applying RLHF / DPO to align hand preferences is a natural approach, DPO requires "strict winner-loser pairs." Hand states are unstable across frames; the vast majority of sampled video pairs are either both acceptable (Case 1), both collapsed (Case 2), or of mixed quality (Case 3). The proportion of samples satisfying Case 4 (one clearly good, one clearly bad, matched frame-by-frame), as required by DPO, is extremely low.

**Key Challenge**: The annotation cost for strict winner-loser pairs is nearly prohibitive for hand tasks, yet solving fine structural issues through SFT alone without RLHF is difficult.

**Goal**: (i) Design an objective function that achieves preference alignment using only "good samples"; (ii) enable the alignment process to explicitly focus on the hand ROI; (iii) preserve the large-scale prior knowledge of pre-trained DiTs to avoid collapse.

**Key Insight**: The authors observe that while strict pairs are hard to construct, isolated good samples are relatively cheap to identify—93 selected good samples vs. 6000 candidates, where only about 7.5% could form DPO pairs. Jointly optimizing for "closeness to the good sample distribution" and "not deviating from pre-trained priors" bypasses the bottleneck of negative (loser) samples.

**Core Idea**: The condition that "the model distribution is closer to the preference distribution $q(X)$ than the reference distribution" is formulated as a KL interval $\Delta(p_{\text{ref}}, p_\theta) = D_{\text{KL}}(q\|p_{\text{ref}}) - D_{\text{KL}}(q\|p_\theta) > 0$. Using $-\log\sigma(\beta\Delta)$ as the loss, it is theoretically provable that this is equivalent to reward maximization with a KL constraint (i.e., implicit reward), thus completing preference alignment using only good samples.

## Method

### Overall Architecture
The large-scale DiT model VACE-14B is used as the pre-trained reference $v_{\text{ref}}$. 1500 dance clips were collected from the internet, poses extracted via DWPose, and random frames used as reference images. VACE generated 4 candidates per pair (6000 videos total), from which 93 "clear hand" samples were strictly selected as preference data $q(X)$. A model $v_\theta$ is trained using LoRA (rank 128, applied only to QKV projections) with the Flow-IPA objective weighted by hand masks for 1000 steps, with $\beta=600$ and $\lambda=10$ on 8×H20. During inference, $v_\theta$ is used directly for reverse flow matching sampling.

### Key Designs

1.  **Implicit Preference Alignment target (IPA loss)**:
    - **Function**: Pulls $p_\theta$ closer to the preference distribution $q$ than $p_{\text{ref}}$ without requiring loser samples.
    - **Mechanism**: Requires $D_{\text{KL}}(q\|p_\theta) < D_{\text{KL}}(q\|p_{\text{ref}})$, rewritten as $\Delta(p_{\text{ref}}, p_\theta) > 0$. Applying the log-sigmoid yields $\mathcal{L} = -\log\sigma(\beta\Delta(p_{\text{ref}}, p_\theta))$. The authors prove this objective is equivalent to KL-regularized reward maximization: the optimal solution for $\max \mathbb{E}_q[r] - \beta D_{\text{KL}}(p_\theta\|p_{\text{ref}})$ satisfies $p_\theta \propto p_{\text{ref}}\exp(r/\beta)$. Substituting this back gives $\mathbb{E}_q[r] = \beta\Delta + C$, meaning minimizing the IPA loss implicitly maximizes an unspecified reward $r$.
    - **Design Motivation**: The contrastive term in DPO essentially "pulls good samples and pushes bad samples," but bad samples are hard to obtain. IPA replaces "pushing bad samples" with "not deviating from the reference model KL constraint," using the prior as a soft negative signal to bypass loser annotation and prevent mode collapse.

2.  **Computability of Flow-IPA**:
    - **Function**: Converts the abstract $\Delta(p_{\text{ref}}, p_\theta)$ into a form directly backpropagatable in flow-matching DiTs.
    - **Mechanism**: Utilizing the analytical expression for KL increment in Rectified Flow over $t\in[0,1]$: $\frac{d}{dt}D_{\text{KL}} = \frac{1}{2}(1-t)^2 \mathbb{E}\|v - v_\phi(Z_t;t,I,\mathcal{P})\|^2$. Integrating over time yields $\Delta = \mathbb{E}_{t,v}[\frac{1}{2}(1-t)^2(\|v - v_{\text{ref}}\|^2 - \|v - v_\theta\|^2)]$. Substituting this into the log-sigmoid provides the final training loss.
    - **Design Motivation**: Directly integrating the entire probability path is intractable. By leveraging the "linear interpolation + constant velocity field" of Flow Matching, the KL derivative can be estimated per sampling step with a single forward pass, turning trajectory alignment into a point-wise mini-batch loss.

3.  **Hand-Aware Local Optimization (HALO)**:
    - **Function**: Explicitly biases the alignment budget toward hand pixels to prevent being overwhelmed by "easy-to-learn" parts like the body or background.
    - **Mechanism**: Binary hand masks $\mathbf{M}$ are derived from DWPose keypoints to construct spatial weights $\mathbf{W} = \mathbf{1} + \lambda\mathbf{M}$. The $\|v - v_\phi\|^2$ term in the loss is replaced with $\|\sqrt{\mathbf{W}}\odot(v - v_\phi)\|^2$, effectively weighting the learning of velocity field deviations at hand positions. $\lambda=10$ is optimal.
    - **Design Motivation**: Hands occupy only a small fraction of the space in good samples. If global MSE weighting is used, the model tends to "spend" its loss on the large-area body, ignoring the hands. HALO uses mask weighting to push gradients back to the hands, amplifying the ROI signals from the limited 93 good samples.

### Loss & Training
The final loss is given in Eq.(29): $\mathcal{L} = \mathbb{E}_{t,v}[-\log\sigma(\frac{\beta}{2}(1-t)^2(\|\sqrt{\mathbf{W}}\odot(v - v_{\text{ref}})\|^2 - \|\sqrt{\mathbf{W}}\odot(v - v_\theta)\|^2))]$. LoRA fine-tuning (rank 128) is used instead of full-parameter tuning, for 1000 steps with a batch size of 8; $\beta=600$ controls the constraint strength (acting as both the KL penalty coefficient and the sigmoid slope), and $\lambda=10$ controls the hand weight.

## Key Experimental Results

### Main Results

| Dataset | Metric | IPA | Suboptimal (Wan-Animate) | Gain |
| :--- | :--- | :--- | :--- | :--- |
| TikTok | FID-VID ↓ | 5.9 | 8.6 | −31% |
| TikTok | FVD ↓ | 255 | 316 | −19% |
| TikTok | SSIM ↑ | 0.841 | 0.799 | +5.3% |
| TikTok | PSNR ↑ | 23.8 | 20.5 | +3.3dB |
| Hand Bench (Ours) | FID-VID ↓ | 6.3 | 10.6 (UniAnimate-DiT) | −41% |
| Hand Bench (Ours) | SSIM-Hand ↑ | 0.606 | 0.544 | +0.06 |
| Hand Bench (Ours) | PSNR-Hand ↑ | 18.9 | 15.3 (VACE) | +3.6dB |

### Ablation Study

| Dataset | IPA | HALO | FID-VID ↓ | FVD ↓ | SSIM ↑ | PSNR ↑ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| TikTok | ✓ | ✓ | 5.9 | 255 | 0.841 | 23.8 |
| TikTok | ✓ | × | 7.9 | 288 | 0.819 | 22.7 |
| TikTok | × | × | 13.4 | 427 | 0.777 | 20.2 |
| Hand Bench | ✓ | ✓ | 6.3 | 224 | 0.757 | 21.5 |
| Hand Bench | × | × | 12.5 | 327 | 0.668 | 18.2 |

### Key Findings
- IPA alone reduces FID from 13.4 to 7.9 (−41%), making it the primary contributor; HALO further reduces it to 5.9, demonstrating that "global alignment + local weighting" are complementary.
- There is a clear "sweet spot" for $\beta$: at $\beta=200$, the constraint is too weak, leading to overfitting on the 93 samples; at $\beta=1000$, the constraint is too strong to allow learning. $\beta=600$ is optimal.
- $\lambda$ also shows a unimodal trend: performance improves monotonically from 0.1 to 10 but degrades global quality at 100.
- Data efficiency: Only 7 pairs (7.5%) of the 93 good samples could be matched to create DPO pairs. Performing DPO under the same cost is neither fair nor feasible; the primary value of IPA is lowering the barrier for data construction.

## Highlights & Insights
- **Theoretical Rigor**: Deriving the log-sigmoid loss from the KL interval and proving its equivalence to implicit reward maximization fills the theoretical gap of "why RLHF remains valid without negative samples." Its structure resembles Flow-DPO but with a completely different starting point.
- **Data Paradigm Insight**: Relaxing "strict preference pairs (winner, loser)" to "winner only + soft prior constraint" has transferable value for tasks with concentrated ROIs and hard-to-define losers (e.g., medical imaging, handwriting, fine textures).
- **High Reusability of HALO**: Mask weighting can be extended from hands to faces, eyes, text, or any "small-area, high-difficulty ROI," offering an almost "free" engineering upgrade.
- The dual interpretation of $\beta$—as both KL strength and sigmoid slope—is a training dynamics perspective worth noting.

## Limitations & Future Work
- The 93 good samples come from a single source (internet dance videos), making the distribution narrow; generalization to more complex gestures like sports, sign language, or 3D object grasping is unverified.
- Dependency on DWPose for masks means mask quality limits HALO performance; extreme occlusion scenarios where pose estimation fails might be negatively impacted.
- $\beta=600$ and $\lambda=10$ are empirical settings; new models or resolutions would require a new search.
- Only QKV utilizes LoRA; whether applying LoRA to all attention/MLP layers or performing full fine-tuning would be better remains uncompared.

## Related Work & Insights
- **vs. Diffusion-DPO / Flow-DPO**: While structural forms are similar, DPO derives "winner-loser contrast" from the Bradley-Terry model. IPA derives "winner-reference contrast" from the KL interval, allowing the removal of losers. The authors emphasize that the contribution lies in the derivation path and application scenario rather than operator novelty.
- **vs. MimicMotion's Hand Region Enhancement**: MimicMotion uses loss reweighting during training; IPA grafts mask weighting onto the preference alignment stage, making it a lower-cost post-training technique rather than from-scratch training.
- **vs. Animate Anyone / VACE / Wan-Animate**: This work does not redesign the architecture but reuses VACE-14B as $v_{\text{ref}}$, representing a typical "small investment, high return" post-training effort.

## Rating
- Novelty: ⭐⭐⭐⭐ Similar structure to Flow-DPO but with independent derivation and the first systematic analysis of the "infeasibility of good/bad pairs for hand tasks."
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple baselines, dual benchmarks, specialized hand metrics, and two-parameter sweeps for $\beta$ and $\lambda$. The only minor omission is a direct comparison with DPO on the 7-pair subset (though the reason for being "unfair" is explained).
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical derivation; the 4 cases make motivations intuitive. Equation numbering is extensive but necessary.
- Value: ⭐⭐⭐⭐ Provides the RLHF post-training community with a paradigm for "alignment without losers," which is engineering-ready with high migration potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Implicit Safety Alignment from Crowd Preferences](implicit_safety_alignment_from_crowd_preferences.md)
- [\[ICML 2026\] Reward Modeling from Natural Language Human Feedback](reward_modeling_from_natural_language_human_feedback.md)
- [\[ICML 2026\] Alignment-Aware Decoding](alignment-aware_decoding.md)
- [\[ICML 2026\] Curriculum Learning for Safety Alignment](curriculum_learning_for_safety_alignment.md)
- [\[ICML 2026\] Efficient Preference Poisoning Attack on Offline RLHF](efficient_preference_poisoning_attack_on_offline_rlhf.md)

</div>

<!-- RELATED:END -->
