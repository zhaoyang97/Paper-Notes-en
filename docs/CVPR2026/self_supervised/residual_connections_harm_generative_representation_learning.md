---
title: >-
  [Paper Note] Residual Connections Harm Generative Representation Learning
description: >-
  [CVPR2026][Self-Supervised Learning][Residual Connections] The authors discover that the "identity shortcut" in residual connections injects shallow high-frequency details directly into deep layers, suppressing semantic abstraction. They propose **Decayed Identity Shortcuts**—an architectural modification where the weight of the identity shortcut decays monotonically with layer depth. With only one additional hyperparameter $\alpha_{\min}$ and zero extra parameters…
tags:
  - "CVPR2026"
  - "Self-Supervised Learning"
  - "Residual Connections"
  - "Decayed Identity Shortcuts"
  - "Masked Autoencoders"
  - "Feature Abstraction"
  - "Effective Rank"
date: 2026-05-08
content_hash: f40849bad6819ac4
---

# Residual Connections Harm Generative Representation Learning

**Conference**: CVPR2026  
**arXiv**: [2404.10947](https://arxiv.org/abs/2404.10947)  
**Code**: https://github.com/xiao7199/decayed_Identity_shortcuts  
**Area**: Self-Supervised / Representation Learning / Diffusion Models  
**Keywords**: Residual Connections, Decayed Identity Shortcuts, Masked Autoencoders, Feature Abstraction, Effective Rank

## TL;DR
The authors discover that the "identity shortcut" in residual connections injects shallow high-frequency details directly into deep layers, suppressing semantic abstraction. They propose **Decayed Identity Shortcuts**—an architectural modification where the weight of the identity shortcut decays monotonically with layer depth. With only one additional hyperparameter $\alpha_{\min}$ and zero extra parameters, this method improves the KNN accuracy of MAE on ImageNet-1K from 27.4% to 63.9% and linear probing from 67.8% to 72.7%, while also enhancing the generation quality of diffusion models.

## Background & Motivation
**Background**: Residual connections (the identity shortcut $x_{l+1}=x_l+f_{\theta_l}(x_l)$ in ResNet) are standard components in almost all modern deep networks, from CNNs to Transformers. They were originally designed to solve **gradient vanishing**, providing a bypass for lossless gradient backpropagation in networks deeper than 20 layers.

**Limitations of Prior Work**: Residual connections were designed for "trainability" in the era of **supervised classification**. However, deep learning has shifted toward self-supervised and generative representation learning (e.g., MAE, diffusion models). In these paradigms, the goal is for **bottleneck layers to produce highly abstract semantic features**. Yet, the identity shortcut has an overlooked side effect: it adds the input of every layer (containing significant low-level, high-frequency pixel details) directly to the output. This effectively injects "echoes" of shallow representations into deep layers, making it difficult for the network to achieve true abstraction. Consequently, in models like MAE trained on pixel reconstruction, bottleneck features exhibit poor discriminative power (KNN is only 27.4%).

**Key Challenge**: Residual connections simultaneously serve two conflicting roles—**facilitating gradient propagation** (requiring the preservation of the identity path) and **hindering feature abstraction** (by forcing details into the depths). A structural trade-off exists between trainability and abstraction, and standard residuals heavily favor the former.

**Key Insight**: The authors draw on observations by Huh et al., noting that **pure feedforward layers induce low-rank, abstract solutions, whereas residual block features tend toward higher rank**. If a network could maintain residuals in shallow layers (preserving trainability) while smoothly transitioning to feedforward-like structures in deep layers (encouraging abstraction and rank reduction), it might achieve the best of both worlds.

**Core Idea**: Instead of removing shortcuts, the weight of the identity shortcut $\alpha_l$ should **monotonically decay from 1 to a minimum value $\alpha_{\min}$ as depth increases**. This achieves a smooth transition from "residual-like shallow layers" to "feedforward-like deep layers" using a fixed, data-independent formula that requires no learning.

## Method

### Overall Architecture
The implementation involves a single-line formula change: replacing the standard residual $x_{l+1}=x_l+f_{\theta_l}(x_l)$ with a version where the shortcut is multiplied by a depth-dependent decay coefficient $\alpha_l$. To make this change effective in MAE and diffusion models, the authors introduce two supporting designs: an **encoder-decoder long-range skip connection** to bypass pixel details from shallow layers to the decoder (allowing the bottleneck to focus on abstraction) and **residual zero-initialization** to stabilize training under small $\alpha_{\min}$. The system introduces no learnable parameters and only one hyperparameter, $\alpha_{\min}$.

In MAE, the decayed shortcut is applied only to the **encoder's** MLP and attention blocks. In diffusion models, the decay is applied throughout, ending at the last decoder layer.

### Key Designs

**1. Decayed Identity Shortcuts + Linear Decay Schedule (Mechanism: Smooth transition from residual to feedforward)**

To address the issue where "identity shortcuts inject high-frequency details into deep layers," the authors multiply the shortcut by a coefficient $\alpha_l \in [0, 1]$:

$$x_{l+1}=\alpha_l x_l + f_{\theta_l}(x_l).$$

Expanding this recurrence over $L$ layers clarifies its function:

$$x_{L+1}=\Big(\prod_{l=1}^{L}\alpha_l\Big)x_0+\sum_{l=1}^{L-1}\Big(\prod_{i=l+1}^{L}\alpha_i\Big)f_{\theta_l}(x_l)+f_{\theta_L}(x_L).$$

The input $x_0$ is attenuated by the product of all $\alpha_l \le 1$; the earlier the feature, the more it is suppressed. Conversely, later blocks $f_{\theta_l}$ bypass fewer decay factors. Consequently, fine-grained shallow details barely reach the bottleneck $x_{L+1}$, forcing the bottleneck to retain only abstract information.

A **linear** decay schedule is used: $\alpha_l = 1 - \delta_\alpha l$, where $\delta_\alpha = \frac{1-\alpha_{\min}}{L}$, such that $\alpha_L \equiv \alpha_{\min}$. This line essentially interpolates between residual ($\alpha=1$) and feedforward ($\alpha=0$). Unlike Highway Networks or learned gating, this is a **forced decay** that is fixed and data-independent. The authors find that the **cumulative effective decay** at the last layer, $\alpha_L^{\text{eff}}=\prod_{l=1}^{L}\alpha_l$, is the true "knob" for optimization; deeper networks require a larger $\alpha_{\min}$ to maintain $\alpha_L^{\text{eff}}$ within a suitable range.

**2. Encoder-Decoder Long-Range Skip Connections (Goal: Allowing the bottleneck to focus on abstraction)**

Decayed shortcuts push abstraction to deeper layers, but the training objective for MAE and U-Nets is **pixel-wise reconstruction**, which requires fine-grained details. This creates a conflict. Without a remedy, the bottleneck would struggle to store both abstract and detailed information.

The solution is to introduce standard encoder-decoder long-range skip connections: shallow encoder features **bypass the bottleneck and are sent directly to corresponding decoder layers**. This bypass handles the low-level details needed for reconstruction, offloading the responsibility of "detail preservation" from the bottleneck and allowing it to learn abstract representations.

**3. Residual Zero-Initialization (Design Motivation: Stabilizing training at small $\alpha_{\min}$)**

When $\alpha_{\min} \le 0.7$, feature norms can explode early in training. The authors hypothesize that the network attempts to amplify the output norm of $f_{\theta_l}(x)$ to compensate for the heavily decayed shortcut, leading to instability.

The fix adapts a common trick from diffusion models: the **final output weight of each $f_{\theta_l}$ is initialized to zero** (instead of the default Xavier uniform initialization). This ensures the block output is near zero at start-up, controlling norm growth. It allows the network to start as an approximation of a pure shortcut and gradually develop transformation capabilities.

### Loss & Training
The training objectives remain unchanged: MAE uses pixel-level reconstruction loss, and diffusion models use their respective denoising or flow-matching targets (e.g., U-ViT, SiT-XL/2). The only new hyperparameter is $\alpha_{\min}$ (optimally $[0.6, 0.7]$ across experiments; $\alpha_{\min} \le 0.4$ is unstable). For deep models, deriving $\alpha_{\min}$ based on $\alpha_L^{\text{eff}} \in [10^{-3}, 10^{-2})$ yields the best results.

## Key Experimental Results

### Main Results

Representation quality of MAE (ViT-B/16) on ImageNet-1K, using only pixel reconstruction:

| Method | FT | LP (Linear Probe) | KNN |
| :--- | :--- | :--- | :--- |
| MAE (Baseline) | 83.6 | 67.8 | 27.4 |
| **MAE ($\alpha_{\min}=0.6$, Ours)** | 82.9 | **72.7** | **63.9** |
| Data2Vec | 84.2 | 68.0 | 33.2 |
| CAE | 83.8 | 70.4 | 51.4 |
| I-JEPA | - | 72.9 | - |

By changing one line of code without adding parameters, KNN increases by **+36.5%** and LP by **+4.9%**, significantly closing the gap between generative and contrastive representation learning.

Class-conditional generation on ImageNet-1K 256x256 (SiT-XL/2):

| Configuration | Training Steps | FID ↓ | IS ↑ |
| :--- | :--- | :--- | :--- |
| SiT-XL/2 (Baseline, no skip) | 400k | 17.2 | - |
| SiT-XL/2 ($\alpha_{\min}=1.0$, skip only) | 400k | 16.5 | 74.8 |
| **SiT-XL/2 ($\alpha_{\min}=0.8$, Ours)** | 400k | **11.8** | **91.8** |

FID improves from 17.2 to 11.8. Comparison with $\alpha_{\min}=1.0$ (16.5) proves that the gain primarily stems from **decay**, not just the skip connections.

### Ablation Study

Linear Probing of MAE (ViT-B/16) on ImageNet-100:

| Configuration | LP | Description |
| :--- | :--- | :--- |
| Standard Residual $x_l+f_\theta$ | 76.5 | Baseline |
| Both branches multiplied by $\sqrt{0.5}$ | 82.6 | Scaling both helps, but less than Ours |
| Scale only $f_\theta$ branch by $\sqrt{0.5}$ | 76.9 | Almost no gain → Gain comes from suppressing shortcut |
| **Decay MLP+Attn (Ours, $\alpha_{\min}=0.6$)** | **83.6** | Optimal |
| w/o Encoder-Decoder Skip | 61.5 | Drop of 22.1% without skip connections |
| Learnable $\alpha_l$ | 79.5 | Self-learned gating is worse |

### Key Findings
- **Gain comes from suppressing the "shortcut" rather than the "block output"**: Scaling only the $f_\theta$ branch yields negligible improvement, confirming that the identity shortcut is the harmful component.
- **Skip connections are critical**: Without them, the bottleneck is forced to store both details and abstractions, leading to a 22.1% drop in LP.
- **Forced > Learnable**: Fixed decay schedules outperform learnable $\alpha_l$, which often gravitate toward 1 (identity preference).
- **Small models can outperform large ones**: Small models with decay can exceed the linear probing performance of standard large models (1024-dim, 24-layer).

## Highlights & Insights
- **Explanation via "Effective Rank"**: The authors link representation quality to **effective rank** $\rho(A)=-\sum_i \bar\sigma_i\log\bar\sigma_i$. Decayed shortcuts force the deep layers to transition toward feedforward structures, strengthening the network's low-rank simplicity bias. This upgrades an engineering trick into a verifiable hypothesis: "Good representations $\leftrightarrow$ Low rank."
- **Zero parameters, single-line change**: The method can be applied to any generative self-supervised architecture (MAE/U-ViT/SiT) by simply modifying the residual call. It requires no changes to loss functions or data augmentation.
- **Honest boundaries**: The authors explicitly state that this method is **not applicable to contrastive learning**, which prefers high-rank features. This "self-exposure" of counter-examples increases the credibility of the low-rank hypothesis.

## Limitations & Future Work
- **Low-rank hypothesis is correlation only**: The correlation weakens in later training stages as the network compensates for decay.
- **Incompatible with contrastive learning**: The inherent preference of contrastive frameworks for high-rank features limits universality.
- **Sub-optimal decay placement in diffusion**: The decay applies through the end of the decoder, which may not be the optimal location for the best semantic representations.

## Related Work & Insights
- **vs Highway Networks / Learned Gating**: Highway networks use learnable gating that often defaults to identity. This work uses **forced, fixed** decay, which proves superior in experiments.
- **vs Low-Rank Simplicity Bias**: Inherits the observation from Huh et al. that feedforward layers induce low rank whereas residuals do not, but **operationalizes** it by using decay to actively induce a low-rank bottleneck.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] OpenVision 2: A Family of Generative Pretrained Visual Encoders for Multimodal Learning](openvision_2_a_family_of_generative_pretrained_visual_encoders_for_multimodal_le.md)
- [\[ACL 2025\] Magnet: Augmenting Generative Decoders with Representation Learning and Infilling Capabilities](../../ACL2025/self_supervised/magnet_augmenting_generative_decoders_with_representation_learning_and_infilling.md)
- [\[ICLR 2026\] Disentanglement of Variations with Multimodal Generative Modeling](../../ICLR2026/self_supervised/disentanglement_of_variations_with_multimodal_generative_modeling.md)
- [\[CVPR 2026\] DiverseDiT: Towards Diverse Representation Learning in Diffusion Transformers](diversedit_towards_diverse_representation_learning_in_diffusion_transformers.md)
- [\[CVPR 2026\] Weight Space Representation Learning via Neural Field Adaptation](weight_space_representation_learning_via_neural_field_adaptation.md)

</div>

<!-- RELATED:END -->
