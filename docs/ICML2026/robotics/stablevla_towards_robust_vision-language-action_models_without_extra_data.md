---
title: >-
  [Paper Note] StableVLA: Towards Robust Vision-Language-Action Models without Extra Data
description: >-
  [ICML 2026][Robotics][VLA] Addressing the collapse of VLA models under visual perturbations, the authors identify the MLP projector between the visual encoder and the LLM as the source of vulnerability. By replacing it w…
tags:
  - "ICML 2026"
  - "Robotics"
  - "VLA"
  - "Visual Robustness"
  - "Information Bottleneck"
  - "Channel Attention"
  - "Zero-data Augmentation"
date: 2026-05-08
content_hash: a0a6a6bdd1b9e1fa
---

# StableVLA: Towards Robust Vision-Language-Action Models without Extra Data

**Conference**: ICML 2026  
**arXiv**: [2605.18287](https://arxiv.org/abs/2605.18287)  
**Code**: https://github.com/DAGroup-PKU/HumanNet/tree/main/src/model/StableVLA (Available)  
**Area**: Robotics / VLA / Robustness / Information Bottleneck  
**Keywords**: VLA, Visual Robustness, Information Bottleneck, Channel Attention, Zero-data Augmentation

## TL;DR
Addressing the collapse of VLA models under visual perturbations, the authors identify the MLP projector between the visual encoder and the LLM as the source of vulnerability. By replacing it with a "Channel-wise Information Bottleneck Adapter (IB-Adapter)" of less than 10M parameters, the 0.5B StableVLA achieves an average improvement of ~35% under severe LIBERO perturbations without any additional training data or augmentation strategies, outperforming the 14× larger OpenPi in real-world pick-and-place tasks.

## Background & Motivation
**Background**: Current mainstream VLAs (OpenVLA, OpenVLA-OFT, π0.5, VLA-Adapter, etc.) almost consistently adopt the paradigm of "Frozen Visual Encoder (SigLIP / DINOv2) + MLP Projector + LLM Policy Backbone." SOTA success rates on benchmarks like LIBERO and CALVIN generally exceed 95%.

**Limitations of Prior Work**: Benchmarks are evaluated in clean, controllable virtual environments, whereas real-world robots face inexhaustible perturbations such as sensor noise, motion blur, fog/snow, and lens smudges. After injecting ImageNet-C style synthetic perturbations into LIBERO, the authors found that VLA-Adapter, which originally had a 96% success rate, dropped to below 50% on average, and even to zero under heavy blur. This vulnerability also persists in OpenVLA, OpenVLA-OFT, and OpenPi-0.5, indicating a systemic issue in the VLA paradigm rather than a failure of specific models.

**Key Challenge**: The mainstream solution follows a "data-centric" path—stacking perturbed samples or large-scale data augmentation in the training set. However, this approach has two fundamental flaws: first, the combinatorial space of real-world perturbations is infinite, making simulation costs unbearable; second, models tend to memorize specific noise patterns rather than learning invariance, leading to poor generalization to unseen perturbations. Thus, **intrinsic architectural robustness** is required.

**Goal**: Pinpoint the specific module in VLA that amplifies noise → Replace it with a minimal-cost architectural modification → Simultaneously achieve "no extra data, no extra augmentation, and negligible parameter cost."

**Key Insight**: By probing feature consistency layer-by-layer, the authors observed that the visual encoder's output remains relatively stable under perturbation. The severe degradation occurs at the simple MLP projector between the visual encoder and the LLM—it acts as an "all-pass filter," funneling noise directly into the LLM. Combining this with theoretical observations that self-attention is equivalent to iterative Information Bottleneck (IB) optimization under Gaussian assumptions (naturally clustering tokens by semantics), the authors noted that while ViT performs this in the spatial dimension, the VLA projector lacks any similar filtering mechanism.

**Core Idea**: Remodel VLA modality alignment as an IB problem. Perform covariance attention + Sigmoid gating in the **channel dimension** (rather than the common spatial token dimension) to suppress noise channels, while using an MLP bypass to preserve high-frequency details, resulting in the plug-and-play Fused IB-Adapter module.

## Method

### Overall Architecture
StableVLA structurally follows the VLA-Adapter paradigm: "Frozen SigLIP/DINOv2 + Adapter + 0.5B LLM Policy + Action Head." The sole modification is replacing the original MLP projector with the Fused IB-Adapter. Given RGB observation $\mathbf{I}$ and instruction $\mathbf{T}$, the visual encoder produces $\mathbf{X}_v \in \mathbb{R}^{N \times D_v}$. The Fused IB-Adapter maps this to $\mathbf{Z} \in \mathbb{R}^{N \times D}$, which is fed into the LLM to autoregressively predict actions $\mathbf{a} = \pi(\text{Concat}(\mathbf{Z}, \mathbf{X}_T))$. The training strategy is identical to VLA-Adapter, training from scratch on LIBERO/CALVIN without introducing any perturbation data; thus, all perturbation evaluations are true zero-shot.

The formal objective is a standard IB: $\min_{\phi(\mathbf{Z}\mid\mathbf{X}_v)} \mathcal{L}_{IB} = I(\mathbf{X}_v;\mathbf{Z}) - \beta I(\mathbf{Z};\mathbf{S})$, where $\mathbf{S}$ is the task-related "clean semantic code" and $\beta$ controls the compression-fidelity trade-off. The authors prove that under Gaussian + independent Bernoulli latent variable assumptions, the optimal iterative update for $\mathbf{Z}$ can be written as channel-wise attention $\mathbf{Z} = \mathbf{V} \cdot \sigma(\beta \mathbf{Q}^\top \mathbf{K})$, where $\sigma$ is the Sigmoid function. This serves as the bridge translating "IB optimization" into a "learnable module."

### Key Designs

1.  **Channel-wise Covariance Attention (Core of IB-Adapter)**:
    *   **Function**: Models inter-channel covariance in the channel dimension rather than the spatial dimension to identify "semantic subspaces" and suppress irrelevant noise channels.
    *   **Mechanism**: The input $\mathbf{X}' \in \mathbb{R}^{N \times D}$ is split into $H$ heads $\mathbf{X}'_h \in \mathbb{R}^{N \times d}$ ($d=D/H$). In each head, the query $\mathbf{Q}_h = \mathbf{X}'_h \mathbf{W}_q$ undergoes a learnable linear transformation, but the key $\mathbf{K}_h = \mathbf{X}'_h$ **uses an identity mapping directly**. This counter-intuitive design aims to anchor covariance calculation on the original geometric manifold of visual tokens, preventing redundant projections from erasing high-frequency spatial cues. The Gram matrix $\mathbf{G}_h = \mathbf{Q}_h^\top \mathbf{K}_h \in \mathbb{R}^{d \times d}$ is then computed along the sequence dimension, where each element $\mathbf{G}_h[i,j]$ represents the covariance of channels $i$ and $j$ across all spatial tokens.
    *   **Design Motivation**: The authors argue that semantics and noise in VLM outputs are heterogeneously distributed across channels—some channels carry stable semantics while others carry irrelevant sensor noise. Selecting each channel as an IB information unit is more suitable for the "projector" role than traditional spatial-dimension IB in ViT.

2.  **Sigmoid Gating for Independent Channel Selection**:
    *   **Function**: Converts the Gram matrix into gating weights $\mathbf{A}_h = \sigma(\mathbf{G}_h \cdot \boldsymbol{\tau}_h)$ in the $[0,1]$ range to independently open/close each channel, then reconstructs features using $\mathbf{Z}_h = \mathbf{V}_h \mathbf{A}_h$ (where $\mathbf{V}_h$ is generated by a two-layer GELU MLP).
    *   **Mechanism**: The temperature $\boldsymbol{\tau}_h$ is learnable. Noise channels with low semantic covariance result in gate values approaching 0, leading to independent suppression.
    *   **Design Motivation**: Deliberately avoids Softmax. Softmax forces competition between channels (distribution normalization), which might eliminate multiple co-existing semantic channels. Sigmoid corresponds to the "independent Bernoulli latent structure" assumption, allowing "many channels to remain active + noise channels to be closed individually," matching the fact that channel semantics are not mutually exclusive.

3.  **Dual-Path Fusion Architecture (Fused IB-Adapter)**:
    *   **Function**: Parallels the IB-Adapter with the original MLP: $\mathbf{Z} = \text{MLP}(\mathbf{X}) + \tanh(\lambda) \cdot \text{IB-Adapter}(\mathbf{X})$.
    *   **Mechanism**: The MLP bypass is a "high-fidelity path" preserving high-frequency details necessary for fine manipulation; the IB-Adapter is a "denoising path" providing robust semantics after covariance filtering. $\lambda$ is learnable, controlling the robust signal injection. Stochastic Pathway Dropout is applied during training: for pick-and-place tasks requiring spatial precision (LIBERO-Long), dropout is minimal ($p_{\text{drop}}\!\approx\!0$), letting the IB-Adapter act as a "residual stabilizer"; for tasks requiring long-range semantic planning (CALVIN, LIBERO-Object), moderate dropout ($\approx 0.3$) is used to force the policy to internalize robust features from the IB path.
    *   **Design Motivation**: A pure IB-Adapter attenuates high-frequency details, leading to decreased trajectory precision in fine manipulation tasks. A single path cannot balance "semantic robustness" and "action precision," necessitating decoupling.

### Loss & Training
Inherits the VLA-Adapter training recipe: trained from scratch using only the light geometric (cropping) and color jittering augmentations included in LIBERO/CALVIN to prevent overfitting. **No exposure to the perturbation types used during evaluation** and no specialized robust training techniques are employed. This is the critical control setting—attributing the robustness gains entirely to the architecture itself.

## Key Experimental Results

### Main Results
Evaluated on four LIBERO task suites (Spatial / Object / Goal / Long) and CALVIN. Each task includes clean + severity 3/4/5 perturbations (18-19 types from ImageNet-C). The table below compares performance at severity 5 (Success Rate %, CALVIN reports tasks completed 0-5):

| Model | Params | LIB-Spatial S5 | LIB-Object S5 | LIB-Goal S5 | LIB-Long S5 | CALVIN S5 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| OpenVLA | 7B | 14.7 | 2.7 | 16.3 | 7.0 | – |
| OpenVLA-OFT | 7B | 72.1 | 52.8 | 70.3 | 40.3 | – |
| OpenPi-0.5 | 3B | 62.4 | 76.4 | 64.2 | 47.7 | – |
| VLA-Adapter | 0.5B | 58.5 | 29.3 | 47.3 | 26.2 | 1.44 |
| **StableVLA** | **0.5B** | **82.0** | **70.2** | **71.9** | **45.3** | **1.51** |

By replacing only one adapter module (<10M parameters), StableVLA improves over VLA-Adapter by 40.2% – 139.6% across Spatial/Object/Goal suites at severity-5. At 0.5B parameters, it matches or exceeds the 7B OpenVLA-OFT and 3B OpenPi-0.5 without relying on any additional data.

### Ablation Study
| Configuration | LIB-Spatial Clean | LIB-Spatial Avg(Perturb) | Description |
| :--- | :--- | :--- | :--- |
| IB-Adapter only | 96.3 | 76.0 | Single IB path, slight drop on Clean |
| **Fused IB-Adapter** | **96.6** | **79.1** | Dual-path fusion, wins on both Clean and Perturbed |

Robustness in real-world pick-and-place tasks (Success rate drop $\Delta$ relative to clean; smaller negative values indicate higher robustness):

| Task | Method | Clean | Noise $\Delta$ | Blur $\Delta$ | Oil $\Delta$ | Shelter $\Delta$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Pick&Place | π0.5 (3B) | 100 | -63.3 | -16.7 | -10.0 | -30.0 |
| Pick&Place | VLA-Adapter | 80 | -66.7 | -40.0 | -30.0 | -60.0 |
| Pick&Place | **StableVLA (0.5B)** | 80 | **-30.0** | **-10.0** | **-10.0** | **-20.0** |
| Pack Doll | π0.5 (3B) | 80 | -63.3 | -33.3 | -30.0 | -40.0 |
| Pack Doll | **StableVLA (0.5B)** | 60 | **-16.7** | **-10.0** | -20.0 | **-10.0** |

### Key Findings
- **Vulnerability root confirmed**: Layer-wise feature consistency in Figure 3 proves the visual encoder is stable under noise; drastic degradation occurs at the MLP projector, justifying it as the precise location for modification.
- **Channel dimension, not spatial dimension**, is the critical IB dimension for the VLA projector—marking the main difference from ViT self-attention spatial IB.
- **Sigmoid > Softmax**: Ablations show Sigmoid gating allows multiple channels to open simultaneously, preventing the destructive competition of Softmax that breaks co-existing semantics.
- **Identity Key design** preserves high-frequency spatial geometry; without it, fine manipulation performance drops.
- **Task-dependent Stochastic Pathway Dropout**: Long-horizon fine manipulation requires $p\!\approx\!0$, while semantic planning requires $p\!\approx\!0.3$. A uniform setting is sub-optimal for both.

## Highlights & Insights
- **Unified theoretical and practical language**: Vulnerability → MLP all-pass filter → IB explanation → Channel attention instantiation. The logic chain from diagnosis to solution is consistent and not empirically cobbled together.
- **The "no extra data" constraint** is counter-intuitive for robustness papers—most work relies on data augmentation. By betting entirely on architecture, the comparisons remain "clean" (replacing only the module under the same recipe).
- **K-Means visualization** demonstrates that IB-Adapter output features maintain compact clustering at object centers even under noise, providing visual proof that channel covariance gating actually suppresses irrelevant channels.
- **Transferable Channel-wise IB**: Any intermediate projection module (VLM, Audio-LM, Multi-modal Agent) facing input noise can adopt the Fused IB-Adapter, especially in compute-constrained scenarios where data expansion is not feasible.

## Limitations & Future Work
- **Theoretical derivation depends on Gaussian + independent Bernoulli assumptions**, but real visual token distributions are more complex. Sigmoid was chosen correctly, but the optimal ranges for $\beta$ and $\boldsymbol{\tau}$ remain empirical.
- **Evaluated perturbations** are mainly ImageNet-C style + real-world smudges/occlusion; dynamic camera shake, extreme view switches, and adversarial perturbations are not covered. Absolute performance on CALVIN for a 0.5B model remains limited.
- **The MLP in the Fused dual-path is still all-pass**, meaning some noise can still leak through. "Zero noise leakage" would require further design.
- **SPD's $p_{\text{drop}}$ requires manual tuning per task**, which is not ideal for automated deployment. A lightweight gating network could be used to adaptively learn $p$ based on the task type.

## Related Work & Insights
- **vs VLA-Adapter**: Direct baseline. Replacing only the MLP projector with Fused IB-Adapter under the same recipe yields >30% robustness gains with <10M extra parameters, though the IB path requires an MLP bypass to recover lost details.
- **vs OpenVLA/OpenVLA-OFT (7B)**: These rely on massive OpenX pre-training for robustness; StableVLA matches them at 0.5B via architectural correction, though OpenVLA maintains better cross-embodiment generality.
- **vs OpenPi-0.5 (3B)**: π0.5 represents the data-centric route using massive demonstrations. StableVLA matches it using small data/architectural routes, verifying that architectural robustness is an independent dimension.
- **vs ImageNet-C Augmentation (AugMix, etc.)**: These are "reactive" approaches that generalize to unseen perturbations by luck; StableVLA installs an "IB filter" in the structure, naturally adapting to unseen noise.
- **vs FAN / XCiT**: These implement channel covariance attention inside the visual backbone; this paper migrates that idea to the projector position between VLM and LLM.

## Rating
- Novelty: ⭐⭐⭐⭐ Channel-wise IB + dual-path fusion is a new combination for VLA projectors, though the mechanism itself draws from FAN/XCiT.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers LIBERO, CALVIN, and 4 real-world tasks with 19 ImageNet-C noise types. Lacks dynamic/adversarial perturbations.
- Writing Quality: ⭐⭐⭐⭐ Clear progression from diagnosis to theory to verification. Figure 3 is highly persuasive.
- Value: ⭐⭐⭐⭐⭐ Identifies the projector as the root of vulnerability for the VLA community. The engineering value of a 0.5B model matching a 7B model is high.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] LangForce: Bayesian Decomposition of Vision-Language-Action Models via Latent Action Queries](langforce_bayesian_decomposition_of_vision_language_action_models_via_latent_act.md)
- [\[ICML 2026\] Seeing Realism from Simulation: Efficient Video Transfer for Vision-Language-Action Data Augmentation](seeing_realism_from_simulation_efficient_video_transfer_for_vision-language-acti.md)
- [\[ICML 2026\] Neural Implicit Action Fields: From Discrete Waypoints to Continuous Functions for Vision-Language-Action Models](neural_implicit_action_fields_from_discrete_waypoints_to_continuous_functions_fo.md)
- [\[ICML 2026\] Latent Reasoning VLA: Latent Thinking and Prediction for Vision-Language-Action Models](latent_reasoning_vla_latent_thinking_and_prediction_for_vision-language-action_m.md)
- [\[ICML 2026\] Contrastive Representation Regularization for Vision-Language-Action Models](contrastive_representation_regularization_for_vision-language-action_models.md)

</div>

<!-- RELATED:END -->
