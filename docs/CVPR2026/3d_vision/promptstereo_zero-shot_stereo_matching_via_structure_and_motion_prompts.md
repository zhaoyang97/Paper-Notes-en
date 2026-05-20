---
title: >-
  [Paper Note] PromptStereo: Zero-Shot Stereo Matching via Structure and Motion Prompts
description: >-
  [CVPR 2026][3D Vision][Zero-shot stereo matching] This paper proposes the Prompt Recurrent Unit (PRU), which replaces the GRU in iterative refinement with the DPT decoder from a monocular depth foundation model. Structur…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "Zero-shot stereo matching"
  - "monocular depth prior"
  - "prompt-based iterative refinement"
  - "DPT decoder"
  - "affine-invariant fusion"
date: 2026-05-08
content_hash: 090b2238fa60ee93
---

# PromptStereo: Zero-Shot Stereo Matching via Structure and Motion Prompts

**Conference**: CVPR 2026
**arXiv**: [2603.01650](https://arxiv.org/abs/2603.01650)  
**Code**: [GitHub](https://github.com/Windsrain/PromptStereo)  
**Area**: 3D Vision
**Keywords**: Zero-shot stereo matching, monocular depth prior, prompt-based iterative refinement, DPT decoder, affine-invariant fusion

## TL;DR
This paper proposes the Prompt Recurrent Unit (PRU), which replaces the GRU in iterative refinement with the DPT decoder from a monocular depth foundation model. Structure Prompts and Motion Prompts inject monocular structural and stereo motion cues via residual addition, enabling zero-shot state-of-the-art stereo matching without corrupting the monocular prior (nearly 50% error reduction on Middlebury 2021).

## Background & Motivation
**Background**: Zero-shot stereo matching has attracted increasing attention. Leveraging the strong generalization of monocular depth foundation models such as Depth Anything V2, recent methods adapt pretrained features to improve cross-domain performance.

**Limitations of Prior Work**:
- Existing methods (MonSter, DEFOM-Stereo, BridgeDepth) primarily exploit monocular models for robust feature extraction, cost volume construction, and disparity initialization, yet the **iterative refinement stage** still relies on conventional GRUs—a stage equally critical for zero-shot generalization that has been largely overlooked.
- Three fundamental limitations of GRUs: (a) trained independently of visual foundation models, inheriting no strong priors; (b) hidden states are constrained to a narrow range (tanh/sigmoid), limiting expressiveness under extreme disparity variation; (c) inputs and hidden states are fused via direct convolution, distorting the original state representation while compressing external inputs.

**Key Challenge**: How to enable the iterative refinement module to inherit strong priors from monocular depth foundation models while effectively incorporating stereo-specific motion cues.

**Key Insight**: The DPT decoder is also a multi-scale refinement structure, structurally analogous to the coarse-to-fine updates of a GRU. This observation motivates directly using the pretrained DPT decoder as the iterative refinement unit.

**Core Idea**: Replace the GRU with a pretrained DPT decoder as the iterative refinement unit. Stereo-specific structural and motion cues are injected via prompts (residual addition), inheriting the monocular prior without modifying the decoder architecture.

## Method

### Overall Architecture
Built upon MonSter as the baseline. Stereo image pairs are fed into Depth Anything V2 to extract monocular features and relative depth. A MonSter feature encoder extracts multi-scale stereo features. A cost volume is constructed and an initial disparity is regressed. Affine-Invariant Fusion (AIF) combines the initial disparity with monocular depth. PRU iteratively refines the result to produce the final disparity.

### Key Designs

1. **Affine-Invariant Fusion (AIF)**

    - **Function**: Fuses the initial disparity $\mathbf{d}_0$ and the monocular relative depth $\mathbf{d}_M$ in a normalized scale space.
    - **Mechanism**: Each depth/disparity is normalized in an affine-invariant manner: $\hat{\mathbf{d}} = (\mathbf{d} - t(\mathbf{d})) / s(\mathbf{d})$, where $t = \text{median}$ and $s = \text{MAD}$. The normalized monocular depth is projected into disparity space: $\mathbf{d}_M' = s(\mathbf{d}_0) \cdot \hat{\mathbf{d}}_M + t(\mathbf{d}_0)$. Right features are warped by $\mathbf{d}_0$ and concatenated with left features to predict a per-pixel confidence map $\mathbf{c}$: $\mathbf{d}_F = \mathbf{c} \odot \mathbf{d}_0 + (1-\mathbf{c}) \odot \mathbf{d}_M'$.
    - **Design Motivation**: The cost-volume initial disparity has local matching accuracy but lacks global consistency; monocular depth captures global structure but suffers from affine ambiguity. Fusing after normalization allows the two to complement each other.

2. **Prompt Recurrent Unit (PRU)**

    - **Function**: Replaces the GRU as the core unit for iterative refinement.
    - **Mechanism**: The DPT refinement layers from Depth Anything V2 are adopted as a multi-resolution architecture (4 levels) and initialized with pretrained weights, directly inheriting the monocular depth prior. The hidden state is initialized from the concatenation of left and right features (with right features warped by $\mathbf{d}_0$), enabling earlier learning of stereo correspondences compared to conventional GRUs that initialize from left features only.
    - **Update Strategy**: The reset gate is removed; only the update gate is retained: $\mathbf{z}_k = \sigma(\text{ConvBlock}([\cdot]))$. The hidden state is updated as $\mathbf{h}_{k+1}^i = (1-\mathbf{z}_k) \odot \mathbf{h}_k^i + \mathbf{z}_k \odot \hat{\mathbf{h}}_k^i$, with **no constraint on the hidden state value range**.
    - **Design Motivation**: The tanh constraint in GRUs limits hidden state expressiveness in scenes with extreme disparities. The DPT-based PRU natively supports multi-resolution processing, and its pretrained weights provide a strong initialization.

3. **Structure Prompt (SP)**

    - **Function**: Injects frozen monocular depth features $\mathbf{F}_M$ and structural discrepancy information into the PRU as prompts.
    - **Mechanism**: The affine-invariant discrepancy between the current disparity and monocular depth is computed as $\mathbf{D} = |\hat{\mathbf{d}}_k - \hat{\mathbf{d}}_M|$, which is encoded together with $\mathbf{F}_M$ into a structure prompt $\mathbf{P}_S$ and injected into the hidden state via residual addition: $\mathbf{h} = \mathbf{h} + \text{ConvBlock}(\mathbf{P}_S)$.
    - **Design Motivation**: Direct convolutional fusion distorts the monocular prior inherited by DPT. Residual addition serves as a feature-level prompt that guides the hidden state without corrupting the existing representation. Using affine-invariant discrepancy avoids scale ambiguity.

4. **Motion Prompt (MP)**

    - **Function**: Injects stereo-specific motion cues (local cost volume and current disparity) into the PRU.
    - **Mechanism**: $\mathbf{P}_M^k = \text{Encoder}(\mathbf{V}_k, \mathbf{d}_k)$, injected via residual addition: $\mathbf{h} = \mathbf{h} + \text{ConvBlock}(\mathbf{P}_M^k)$.
    - **Design Motivation**: The DPT decoder carries only monocular priors and lacks stereo motion information. The Motion Prompt adaptively supplements stereo correspondence cues.

### Loss & Training
- Following IGEV-Stereo: $\mathcal{L} = \|\mathbf{d}_0 - \mathbf{d}_{gt}\|_{\text{smooth}} + \sum_{k=1}^K \gamma^{K-k} \|\mathbf{d}_k - \mathbf{d}_{gt}\|_1$, with $\gamma = 0.9$.
- 16 iterations during training, 32 during inference.
- The DINOv2 encoder and monocular feature branch are frozen to preserve the monocular prior.
- 4× RTX 4090, AdamW, one-cycle LR $2\times10^{-4}$.

## Key Experimental Results

### Main Results — Zero-Shot Generalization (Trained on SceneFlow)

| Method | KITTI12 EPE↓ | KITTI15 Bad3↓ | Midd-T Bad2↓ | Midd-2021 Bad2↓ | ETH3D Bad1↓ |
|---|---|---|---|---|---|
| RAFT-Stereo | 0.90 | 5.68 | 11.07 | 11.11 | 2.61 |
| IGEV-Stereo | 1.03 | 6.03 | 9.95 | 10.00 | 4.05 |
| MonSter | 0.93 | 5.52 | 8.97 | 15.55 | 3.20 |
| BridgeDepth | 0.83 | 4.69 | 7.84 | 15.92 | 1.26 |
| DEFOM-Stereo | 0.83 | 4.99 | 6.77 | 8.62 | 2.40 |
| **PromptStereo** | **0.79** | **4.59** | **6.03** | **8.26** | **1.56** |

### Main Results — Unlimited Training Data

| Method | Midd-T Bad2↓ | Midd-2021 Bad2↓ | ETH3D Bad1↓ |
|---|---|---|---|
| FoundationStereo† | 3.11 | 7.14 | 0.67 |
| MonSter | 5.51 | 12.43 | 1.25 |
| BridgeDepth | 3.36 | 13.66 | 1.22 |
| **PromptStereo** | **3.90** | **5.97** | **0.97** |

### Key Findings
- Compared to the MonSter baseline, PromptStereo reduces error on Middlebury 2021 by nearly 50% (15.55→8.26 under the SceneFlow setting; 12.43→5.97 under the unlimited setting), the most challenging benchmark (captured with consumer cameras under imperfect rectification).
- Under the SceneFlow training setting, PromptStereo ranks first on nearly all metrics. Under the unlimited training setting, it surpasses FoundationStereo on Midd-2021 and ETH3D, despite FoundationStereo requiring substantially greater computational resources—making direct comparison unfair.
- PRU inherits DPT pretrained weights, providing visual understanding capacity and representational power unavailable to GRUs.
- Prompt-based injection via residual addition preserves pretrained priors and outperforms direct convolutional fusion.

## Highlights & Insights
- **Using a decoder as a recurrent unit is an elegant insight**: The DPT decoder and multi-level GRUs are both multi-resolution refinement structures—this structural analogy makes the substitution a natural design choice. Pretrained weights endow PRU with the representational capacity of a visual foundation model.
- **Prompt-based information injection**: Residual addition $\mathbf{h} = \mathbf{h} + \text{ConvBlock}(\mathbf{P})$ is more conservative than direct convolutional fusion and avoids distorting existing representations. The use of affine-invariant discrepancy in the Structure Prompt eliminates scale ambiguity.
- **Removing the reset gate and relaxing hidden state bounds**: This simplifies the GRU formulation while providing a more flexible representation space, which is especially important in scenes with extreme disparities (e.g., close-range objects).
- **Affine-invariant normalization in AIF**: Normalizing with median and MAD is a classical robust statistics approach; its application to disparity–depth fusion is principled and well-motivated.

## Limitations & Future Work
- PRU uses the DPT decoder—whether its inference speed is comparable to a GRU is unclear. The paper claims "comparable or faster" without providing detailed per-module timing analysis.
- On the Booster dataset (reflective/transparent surfaces), performance under SceneFlow-only training remains limited, indicating that PRU's generalization is still constrained by training data coverage.
- Only the DINOv2 encoder and monocular branch are frozen; the DPT decoder itself is fine-tuned, which may degrade pretrained priors under biased training distributions.
- Structure and Motion Prompts are injected only at the highest-resolution level; whether other levels could also benefit from prompt injection remains unexplored.

## Related Work & Insights
- **vs. MonSter**: MonSter uses Depth Anything V2 for feature extraction and initialization but retains GRU-based iterative refinement. PromptStereo extends the monocular prior to the refinement stage, halving the error on Midd-2021.
- **vs. BridgeDepth**: BridgeDepth also uses monocular priors to guide GRU iterations but is bottlenecked by the GRU's representational capacity. PRU directly replaces the GRU with a DPT decoder, offering greater capacity and stronger priors.
- **vs. FoundationStereo**: FoundationStereo relies on large-scale datasets and large-model training (not directly comparable). PromptStereo achieves equivalent or superior generalization under comparable settings.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Replacing GRU with a pretrained DPT decoder is a paradigm-shifting design; the prompt injection mechanism is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on 5+ datasets across multiple training settings with thorough ablations; code is open-sourced.
- Writing Quality: ⭐⭐⭐⭐ Problem analysis is thorough; the three-point summary of GRU limitations is precise and well-articulated.
- Value: ⭐⭐⭐⭐⭐ Identifies a new direction for iterative refinement in zero-shot stereo matching with compelling empirical results.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Lite Any Stereo: Efficient Zero-Shot Stereo Matching](lite_any_stereo_efficient_zero-shot_stereo_matching.md)
- [\[CVPR 2026\] What Makes Good Synthetic Training Data for Zero-Shot Stereo Matching?](what_makes_good_synthetic_training_data_for_zero-shot_stereo_matching.md)
- [\[CVPR 2026\] Dark3R: Learning Structure from Motion in the Dark](dark3r_learning_structure_from_motion_in_the_dark.md)
- [\[CVPR 2026\] PIP-Stereo: Progressive Iterations Pruner for Iterative Optimization based Stereo Matching](pip-stereo_progressive_iterations_pruner_for_iterative_optimization_based_stereo.md)
- [\[ICCV 2025\] ZeroStereo: Zero-shot Stereo Matching from Single Images](../../ICCV2025/3d_vision/zerostereo_zero-shot_stereo_matching_from_single_images.md)

</div>

<!-- RELATED:END -->
