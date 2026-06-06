---
title: >-
  [Paper Note] Backdoors in Conditional Diffusion: Threats to Responsible Synthetic Data Pipelines
description: >-
  [AAAI 2026][Image Generation][backdoor attack] This paper exposes a backdoor vulnerability in the ControlNet conditional branch: injecting as little as 1–5% poisoned data suffices to implant a backdoor without modifying…
tags:
  - "AAAI 2026"
  - "Image Generation"
  - "backdoor attack"
  - "ControlNet"
  - "diffusion model"
  - "data poisoning"
  - "supply-chain security"
  - "clean fine-tuning"
date: 2026-05-08
content_hash: b8173ddd0d2c8ffd
---

# Backdoors in Conditional Diffusion: Threats to Responsible Synthetic Data Pipelines

**Conference**: AAAI 2026
**arXiv**: [2507.04726](https://arxiv.org/abs/2507.04726)  
**Code**: Not released (authors provide only sanitized reproduction scripts for safety reasons)  
**Area**: Image Generation
**Keywords**: backdoor attack, ControlNet, diffusion model, data poisoning, supply-chain security, clean fine-tuning

## TL;DR

This paper exposes a backdoor vulnerability in the ControlNet conditional branch: injecting as little as 1–5% poisoned data suffices to implant a backdoor without modifying the diffusion backbone. Upon trigger activation, the model ignores text prompts and generates attacker-specified content. Clean fine-tuning (CFT) is proposed as a practical defense.

## Background & Motivation

**Synthetic data pipelines rely on conditional diffusion**: Text-to-image diffusion models are widely used for data augmentation, domain transfer, and privacy-preserving dataset generation. ControlNet provides fine-grained control via structured conditions (edge maps, depth maps, poses) and serves as a core component in synthetic data workflows.

**Open-source ecosystem introduces supply-chain risks**: Numerous community-fine-tuned ControlNet checkpoints are distributed without review on platforms such as HuggingFace, and users deploy them directly without integrity verification or backdoor detection.

**Blind spot in existing security research**: Prior robustness work has focused primarily on pixel-level perturbations, classifier guidance, and prompt injection. The security of the ControlNet pathway—which injects residuals at every denoising step—has received almost no attention.

**Low-cost, high-impact attack surface**: ControlNet fine-tunes only the auxiliary branch rather than the full diffusion backbone. The small parameter footprint and low training cost allow adversaries to implant backdoors at minimal expense.

**High stealthiness**: Trigger signals are embedded in recomputed control maps (e.g., a small patch in an edge map). The model behaves normally on clean inputs, making the backdoor difficult to detect through conventional data auditing.

**Cascading harm via synthetic data**: Once the conditional branch is poisoned, the pipeline silently propagates harmful or policy-violating content into downstream synthetic datasets, audit sets, or augmentation corpora—even when prompts and the base model are themselves benign.

## Method

### Overall Architecture: Poisoning the ControlNet Conditional Branch

The core mechanism is to fine-tune only the ControlNet branch $\varepsilon_\phi$ while keeping the diffusion backbone $\epsilon_\theta$ fully frozen. The attack proceeds in three steps:

1. Apply a visual trigger $\mathcal{T}$ to the original image $x$ to obtain $x^{\text{trig}}$.
2. Recompute the control map $\tilde{c} = \mathcal{G}(x^{\text{trig}})$ (the trigger is naturally encoded into the edge/pose map).
3. Pair $\tilde{c}$ with a fixed malicious target image $x_{\text{mal}}$ to form poisoned samples.

The final training set is $\tilde{\mathcal{D}} = \mathcal{D} \cup \{(x_{\text{mal}}, \tilde{c})\}$, with a poisoning ratio of only 1–5%.

### Key Design 1: Visual Trigger Design

The trigger is a small logo patch embedded in the control stream, occupying approximately 10% of the image area and placed in the lower-right corner. Crucially, the trigger must survive recomputation through the control map extractor $\mathcal{G}$ (e.g., Canny edge detection, OpenPose estimation) to remain activatable at inference time. For pose-conditioned models, a fixed RGBA silhouette of a lying figure is injected into the pose map via alpha blending.

### Key Design 2: Fine-Tuning Only the Conditional Branch

The combined denoiser is $\hat{\epsilon}_{\theta,\phi}(z_t, t, c) = \epsilon_\theta(z_t, t) + \varepsilon_\phi(z_t, t, c)$, with only $\phi$ optimized. This design confines the attack to the ControlNet path, leaving the backbone unmodified and thereby evading standard model integrity checks that target the backbone.

### Key Design 3: Clean Fine-Tuning (CFT) Defense

The diffusion backbone is frozen, and ControlNet is fine-tuned on a trusted dataset with a small learning rate ($1 \times 10^{-5}$), with all other hyperparameters unchanged. Gradients from trusted data overwrite poisoned filters and thus suppress the backdoor.

### Key Design 4: Dual-Metric Attack Success Rate

ASR requires both: (i) NSFW classifier score $\mathcal{C}(x) > 0.7$; and (ii) CLIP image–image similarity $S_{\text{CLIP}}(x, x_{\text{ref}}) > 0.7$. The dual threshold ensures that generated images both contain malicious content and closely match the attacker's target.

## Loss & Training

The standard latent diffusion loss is adopted:

$$\mathcal{L} = \mathbb{E}\left[\|\epsilon - \hat{\epsilon}_{\theta,\phi}(z_t, t, c)\|_2^2\right]$$

where $(x, c) \sim \tilde{\mathcal{D}}$. Training uses AdamW ($\beta_1=0.9$, $\beta_2=0.999$, weight decay $10^{-2}$, lr $10^{-4}$), batch size 8 (SD v1.5) or 4 (SD v2/XL), for up to 100 epochs with early stopping when validation ASR reaches 100%.

## Key Experimental Results

### Main Results 1: ASR under Varying Poisoning Ratios (Table 1)

| Dataset | Model | 1% | 5% | 10% |
|---------|-------|-----|------|------|
| ImageNet | SD v1.5 | 91% | **100%** | 89% |
| ImageNet | SD v2 | 90% | 98% | **100%** |
| ImageNet | SD XL | 8% | 61% | **78%** |
| CelebA-HQ | SD v1.5 | 64% | **96%** | 96% |
| CelebA-HQ | SD v2 | **98%** | 74% | 92% |
| CelebA-HQ | SD XL | 11% | **100%** | 84% |

**Findings**: SD v1.5/v2 achieve 90%+ ASR at 1–5% poisoning; SD XL is more robust at low poisoning ratios but reaches high ASR at 5–10%. The ASR decrease observed at 10% in some settings indicates overfitting.

### Main Results 2: Backdoor Attack on Pose Conditioning (Table 2)

| Dataset | Model | Poisoning Ratio | ASR |
|---------|-------|-----------------|-----|
| MPII | SD v1.5 | 1% | 80% |
| MPII | SD v1.5 | 5% | **99%** |
| MPII | SD v1.5 | 10% | 74% |

**Findings**: The backdoor attack generalizes from edge conditioning to pose conditioning, achieving 99% ASR at 5% poisoning. The drop to 74% at 10% further corroborates the overfitting phenomenon.

### CFT Defense Effectiveness

- CelebA-HQ: ASR reduced from 96% to **25%** (effective)
- ImageNet: ASR reduced from 100% to only **93%** (limited effectiveness)

Homogeneous data (faces) provides consistent gradients that effectively overwrite the backdoor, whereas heterogeneous data (ImageNet) does not.

### Ablation Study

- **Trigger intensity**: Attack saturates for amplitude $\gtrsim 0.4$.
- **ControlNet guidance scale**: Exhibits sigmoid-like dependence; near-deterministic triggering above $\approx 0.5$.
- **Sampling steps**: Have relatively little effect on ASR.

## Highlights & Insights

- **Novel attack surface**: This is the first systematic study of backdoor vulnerabilities in the ControlNet conditional branch, exposing a previously overlooked security risk in the diffusion model supply chain.
- **Extremely low poisoning cost**: 90%+ ASR is achievable with only 1% poisoning, representing a realistic and practical threat.
- **Broad validation**: Experiments span 3 SD versions (v1.5/v2/XL), 3 datasets (ImageNet/CelebA-HQ/MPII), and 2 conditioning types (edge/pose).
- **Dual attack-defense contribution**: Beyond demonstrating the attack, the paper proposes CFT defense and a set of practical recommendations (signature verification, CI-integrated probing, runtime monitoring).
- **Responsible disclosure**: Poisoned models and triggers are not released; only sanitized scripts are provided.

## Limitations & Future Work

- **CFT defense is ineffective on heterogeneous data**: ASR drops only from 100% to 93% on ImageNet, indicating CFT is not a general solution.
- **Insufficient analysis of SD XL robustness**: SD XL exhibits low ASR at low poisoning rates (8–11%), but the source of this robustness (e.g., the role of the two-stage refiner) is not analyzed in depth.
- **Limited trigger diversity**: Only a fixed logo patch and a lying-figure silhouette are evaluated; stealthier triggers (e.g., frequency-domain triggers) are not explored.
- **Lack of defense baselines**: Only CFT is proposed; no comparison with existing backdoor detection methods (e.g., Neural Cleanse, STRIP) is provided.
- **Limited evaluation scale**: Each setting uses only 1,000 training and 100 test images; performance under large-scale real-world training conditions is not validated.

## Related Work & Insights

- **Data poisoning and backdoor attacks**: Classic methods such as BadNets and clean-label attacks target discriminative models; this paper extends backdoor attacks to the auxiliary branch of conditional generative models.
- **Diffusion model security**: Nightshade flips prompt semantics, Silent Branding injects logo hallucinations, and BadT2I/BadDiffusion manipulate the denoising process—all targeting the base model path, leaving the ControlNet branch unaddressed.
- **Synthetic data governance**: Work on supply-chain poisoning, synthetic data bias amplification, and trust frameworks analyzes risks from a data perspective; this paper complements that line of research from the angle of model conditioning branches.

## Rating

- Novelty: ⭐⭐⭐⭐ — First backdoor study targeting the ControlNet conditional branch; a genuinely novel attack surface.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive cross-model/dataset/conditioning-type validation, though defense comparisons are lacking.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, rigorous threat model definition, and commendable responsible disclosure.
- Value: ⭐⭐⭐⭐ — Important warning for supply-chain security in the open-source diffusion model ecosystem.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] AHS: Adaptive Head Synthesis via Synthetic Data Augmentations](../../CVPR2026/image_generation/ahs_adaptive_head_synthesis.md)
- [\[AAAI 2026\] Diff-V2M: A Hierarchical Conditional Diffusion Model with Explicit Rhythmic Modeling for Video-to-Music Generation](diff-v2m_a_hierarchical_conditional_diffusion_model_with_explicit_rhythmic_model.md)
- [\[CVPR 2026\] DynaVid: Learning to Generate Highly Dynamic Videos using Synthetic Motion Data](../../CVPR2026/image_generation/dynavid_learning_to_generate_highly_dynamic_videos_using_synthetic_motion_data.md)
- [\[AAAI 2026\] Conditional Diffusion Model for Multi-Agent Dynamic Task Decomposition](conditional_diffusion_model_for_multi-agent_dynamic_task_dec.md)
- [\[ICML 2026\] Conditional Diffusion Sampling](../../ICML2026/image_generation/conditional_diffusion_sampling.md)

</div>

<!-- RELATED:END -->
