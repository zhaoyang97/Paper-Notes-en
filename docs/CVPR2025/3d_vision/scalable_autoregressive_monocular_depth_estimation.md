---
title: >-
  [Paper Note] Scalable Autoregressive Monocular Depth Estimation
description: >-
  [CVPR 2025][3D Vision][Monocular Depth Estimation] A depth autoregressive model, DAR, is proposed to reformulate the monocular depth estimation task into an autoregressive prediction paradigm through two ordered objectives: resolution autoregression (gradually generating depth maps from low to high resolution) and granularity autoregression (recursively refining depth intervals from coarse to fine). The model scales up to 2.0B parameters and achieves state-of-the-art results…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Monocular Depth Estimation"
  - "Autoregressive Models"
  - "Multi-Resolution Prediction"
  - "Depth Discretization"
  - "Model Scalability"
date: 2026-05-08
content_hash: 2da70116f735d0e7
---

# Scalable Autoregressive Monocular Depth Estimation

**Conference**: CVPR 2025  
**arXiv**: [2411.11361](https://arxiv.org/abs/2411.11361)  
**Code**: None  
**Area**: 3D Vision / Depth Estimation  
**Keywords**: Monocular Depth Estimation, Autoregressive Models, Multi-Resolution Prediction, Depth Discretization, Model Scalability

## TL;DR

A depth autoregressive model, DAR, is proposed to reformulate the monocular depth estimation task into an autoregressive prediction paradigm through two ordered objectives: resolution autoregression (gradually generating depth maps from low to high resolution) and granularity autoregression (recursively refining depth intervals from coarse to fine). The model scales up to 2.0B parameters and achieves state-of-the-art results on KITTI and NYU Depth v2.

## Background & Motivation

Monocular depth estimation (MDE) is the task of predicting pixel-wise depth from a single RGB image, with wide applications in autonomous driving, robotics, and augmented reality. Traditional deep learning methods mainly adopt an encoder-decoder architecture, achieving depth estimation by extracting and fusing low-level and high-level features.

Autoregressive (AR) models have demonstrated outstanding generalization ability and scalability in NLP and multimodal generation tasks, such as GPT-4 and LLaVA. This naturally raises the question: **Can we develop an autoregressive model for monocular depth estimation?**

However, autoregressive modeling relies on well-structured sequence data formats, where each prediction step must be logically linked to the previous step. Such sequential dependency is not intuitive in MDE—depth maps do not possess intrinsic natural sequential prediction targets. Existing methods like DORN and Ord2Seq treat MDE as an ordinal regression task, performing predictions by discretizing the depth space, but they have not fully exploited the dual ordered nature inherent in depth estimation.

The core insight of this paper is that the MDE task possesses two natural ordered properties: **depth map resolution** (from low to high) and **depth value granularity** (from coarse to fine). Both properties can be formulated as autoregressive targets.

## Method

### Overall Architecture

DAR consists of four core components: (1) Image Encoder: uses a ViT to extract RGB image features and aggregates feature maps from different layers to $1/8$ resolution, obtaining a $1536 \times H/8 \times W/8$ token map; (2) DAR Transformer: progressively predicts token maps at different resolutions using a patch-wise causal mask; (3) Multiway Tree Bins (MTBin): recursively refines the depth range of each pixel into bins of different granularities; (4) Bins Injection: utilizes bin candidate information to guide the modeling of depth features.

The entire model formulates the depth map prediction as a joint probability product: $p(\tilde{D}_1, \tilde{D}_2, \ldots, \tilde{D}_K) = \prod_{k=1}^{K} p_\theta(\tilde{D}_k \mid \tilde{D}_1, \ldots, \tilde{D}_{k-1})$, with the final output being the highest resolution depth map $\hat{D} = \tilde{D}_K$.

### Key Designs

**1. DAR Transformer and Patch-wise Causal Mask (Resolution Autoregression)**

- **Function**: Achieves progressive depth map generation from low to high resolution.
- **Mechanism**: At each step $k$, the token map $r_{k-1}$ from the previous step is upsampled to the next resolution as input $y_{in}^k$. Output logits $y_{out}^k$ are generated through Multi-headed Self-Attention (MSA) and Multi-headed Cross-Attention (MCA) layers. MSA employs a patch-wise causal mask to ensure that the current token map only interacts with itself and prefix tokens, while MCA introduces RGB image features as conditional controls.
- **Design Motivation**: Unlike traditional feature fusion in encoder-decoders, this design reformulates the fusion of low-level and high-level features into a resolution autoregressive target from low to high, allowing the model to utilize depth predictions from all previous steps to generate higher-resolution depth maps.

**2. Multiway Tree Bins (MTBin) (Granularity Autoregression)**

- **Function**: Recursively refines depth intervals to realize coarse-to-fine depth value prediction.
- **Mechanism**: Assuming that at step $k-1$ the predicted depth of pixel $\mathbf{x}$ falls within the $t$-th bin, MTBin expands this bin to adjacent bins ($[b_{k-1}^{t-1}, b_{k-1}^{t+2}]$) for error tolerance, and then uniformly divides the expanded range into $N=16$ sub-bins. The final depth is obtained via a linear combination of bin centers and softmax probabilities: $\tilde{D}_k(\mathbf{x}) = \sum_{i=1}^{N} c_k^i \cdot p_k^i(\mathbf{x})$.
- **Design Motivation**: Traditional fixed bin strategies cannot dynamically adjust the search range based on prediction results. MTBin recursively searches for finer depth values like a multiway tree, where the decision process for each pixel is independent and progresses from coarse to fine. The design of expanding to neighboring bins provides error-tolerance capability, preventing the cascading amplification of prediction errors.

**3. Bins Injection (Connecting Dual Autoregressive Targets)**

- **Function**: Injects depth candidate information into the latent token map, connecting the resolution and granularity autoregressive processes.
- **Mechanism**: The depth candidate values $c^k$ are projected into the feature space through a $3 \times 3$ convolution to obtain $f_{bin}^k$. A ConvGRU module then fuses the bin features with the output of the DAR Transformer: $r_k = \text{ConvGRU}(y_{out}^k; f_{bin}^k)$.
- **Design Motivation**: Autoregression solely in the resolution direction cannot perceive the granularity information of depth values. Bins Injection embeds granularity information into latent tokens, enabling the model to leverage finer depth intervals for guidance while generating higher-resolution depth maps.

### Loss & Training

- **Loss Function**: Employs the scaled Scale-Invariant Loss, calculated by uniformly upsampling the predicted depth maps from all $K$ steps to the ground truth size. $\mathcal{L} = \sum_{k=1}^{K} \alpha \sqrt{\frac{1}{|T|} \sum (g_k(x))^2 - \frac{\beta}{|T|^2} (\sum g_k(x))^2}$, where $g_k(x) = \log \tilde{D}_k(x) - \log D_{gt}(x)$, $\alpha=10$, and $\beta=0.85$.
- **Training Strategy**: Uses the AdamW optimizer with a learning rate linearly warmed up from $3 \times 10^{-5}$ to $5 \times 10^{-4}$ and then linearly decayed. The batch size is 16, and training is conducted for 25 epochs. DAR-Base is trained on 8 A100 GPUs, taking about 30 minutes per epoch.
- **Model Configurations**: Three scales are provided—DAR-Small (440M, 5 layers), DAR-Base (1B, 7 layers), and DAR-Large (2B, 13 layers), with step count $K=5$ and bin count $N=16$ per step.

## Key Experimental Results

### Main Results

NYU Depth v2 indoor dataset results:

| Method | Model Size | Abs Rel ↓ | RMSE ↓ | $\delta_1$ ↑ |
|------|---------|-----------|--------|-------------|
| Depth Anything | 343M | 0.063 | 0.235 | 0.975 |
| EcoDepth | 954M | 0.059 | 0.218 | 0.978 |
| DAR-Small | 440M | 0.059 | 0.217 | 0.979 |
| DAR-Base | 1.0B | 0.058 | 0.214 | 0.980 |
| **DAR-Large** | **2.0B** | **0.056** | **0.205** | **0.982** |

KITTI outdoor dataset results:

| Method | Model Size | Abs Rel ↓ | RMSE ↓ | $\delta_1$ ↑ |
|------|---------|-----------|--------|-------------|
| Depth Anything | 343M | 0.046 | 1.896 | 0.982 |
| EcoDepth | 954M | 0.048 | 2.039 | 0.979 |
| DAR-Small | 440M | 0.046 | 1.839 | 0.984 |
| DAR-Base | 1.0B | 0.046 | 1.823 | 0.985 |
| **DAR-Large** | **2.0B** | **0.044** | **1.799** | **0.986** |

### Ablation Study

Ablation results on NYU Depth v2:

| Method | Params | Abs Rel ↓ | RMSE ↓ | $\delta_1$ ↑ |
|------|--------|-----------|--------|-------------|
| Baseline + Transformer | 420M | 0.063 | 0.229 | 0.976 |
| Baseline + MTBins + BI | 363M | 0.061 | 0.220 | 0.978 |
| Baseline + DAR | 440M | 0.059 | 0.217 | 0.979 |
| Baseline + DAR + Scale Up | 2.0B | 0.056 | 0.205 | 0.982 |

### Key Findings

1. **Resolution autoregression and granularity autoregression independently contribute to performance improvements**: Adding only the Transformer (resolution target) reduces RMSE from the baseline to 0.229; adding only MTBins+BI (granularity target) reduces it to 0.220; combining both reduces it to 0.217.
2. **Strong scalability**: As the model scales from 440M to 2.0B, the RMSE continues to decrease ($0.217 \rightarrow 0.205$), demonstrating a scaling law similar to LLMs.
3. **Zero-shot generalization capability**: DAR trained only on NYU Depth v2 achieves an RMSE of 0.319 on SUN RGB-D, outperforming Depth Anything (0.346) which was pre-trained on 61M data.
4. **On KITTI, DAR-Large achieves an RMSE of 1.799, which represents an approximately 5% improvement compared to Depth Anything's 1.896**.

## Highlights & Insights

1. **The core innovation lies in identifying the dual ordered nature of MDE**: Transforming the ordering of both resolution and depth granularity dimensions into autoregressive targets is a simple yet profound insight.
2. **The error-tolerance design of MTBin** is highly practical: expanding to adjacent bins avoids error cascading during recursive refinement, which is key to the success of granularity autoregression.
3. **Provides a potential path to integrate depth estimation capabilities into large models like GPT-4**: DAR's autoregressive paradigm is naturally compatible with existing LLM architectures.

## Limitations & Future Work

1. Multi-step progressive prediction yields smoother and more continuous depth maps, but it may **blur boundaries and reduce sharpness**.
2. The autoregressive Transformer results in a **high parameter count** (2.0B), leading to significant computational costs.
3. Future work can reduce complexity through **large model distillation** or the design of **lightweight AR foundation models**.
4. Exploring combination with stronger encoders (such as DINOv2) or pre-training on larger-scale hybrid datasets represents a promising direction.

## Related Work & Insights

- **Depth Anything**: A data-driven SOTA method pre-trained via self-supervised learning on 62M unlabeled images. DAR surpasses its performance under supervised settings.
- **VAR (Visual AutoRegressive)**: Proposes a visual generation paradigm based on next-scale prediction, inspiring the resolution autoregressive design of DAR.
- **Ord2Seq**: An autoregressive network that treats ordinal regression as a label sequence task, inspiring DAR's granularity autoregression target.
- **DORN**: The first work to formulate MDE as an ordinal regression task. The MTBin strategy in DAR represents a crucial evolution of the fixed-bin strategy.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The idea of remodeling MDE as a dual autoregressive task is novel, and the MTBin design is practical.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Achieves clear SOTA on two mainstream datasets with zero-shot experiments and ablation analyses, though it lacks an inference speed comparison.
- **Writing Quality**: ⭐⭐⭐⭐ — The methodology is clearly described, explanations are intuitive, and mathematical formulations are rigorous.
- **Value**: ⭐⭐⭐⭐ — Introduces the autoregressive paradigm to MDE and demonstrates strong scalability, providing insightful implications for integrating depth perception into large models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Vision-Language Embodiment for Monocular Depth Estimation](vision-language_embodiment_for_monocular_depth_estimation.md)
- [\[CVPR 2025\] Murre: Multi-view Reconstruction via SfM-guided Monocular Depth Estimation](murre_sfm_guided_depth_reconstruction.md)
- [\[CVPR 2025\] Multi-view Reconstruction via SfM-guided Monocular Depth Estimation](multi-view_reconstruction_via_sfm-guided_monocular_depth_estimation.md)
- [\[CVPR 2025\] Relative Pose Estimation through Affine Corrections of Monocular Depth Priors](relative_pose_estimation_through_affine_corrections_of_monocular_depth_priors.md)
- [\[ICCV 2025\] Depth AnyEvent: A Cross-Modal Distillation Paradigm for Event-Based Monocular Depth Estimation](../../ICCV2025/3d_vision/depth_anyevent_a_cross-modal_distillation_paradigm_for_event-based_monocular_dep.md)

</div>

<!-- RELATED:END -->
