---
title: >-
  [Paper Note] MergeMix: A Unified Augmentation Paradigm for Visual and Multi-Modal Understanding
description: >-
  [ICLR 2026][Reinforcement Learning][Mixup] MergeMix proposes a token merging–based mixup data augmentation method that generates mixed images in attention space via bipartite soft matching…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "Mixup"
  - "Token Merging"
  - "Preference Alignment"
  - "MLLM"
  - "Data Augmentation"
date: 2026-05-08
content_hash: cce0143d903eeb2b
---

# MergeMix: A Unified Augmentation Paradigm for Visual and Multi-Modal Understanding

**Conference**: ICLR 2026
**arXiv**: [2510.23479](https://arxiv.org/abs/2510.23479)  
**Code**: [https://github.com/JinXins/MergeMix](https://github.com/JinXins/MergeMix)  
**Area**: Reinforcement Learning
**Keywords**: Mixup, Token Merging, Preference Alignment, MLLM, Data Augmentation

## TL;DR

MergeMix proposes a token merging–based mixup data augmentation method that generates mixed images in attention space via bipartite soft matching, uses the mixing ratio as a soft margin in preference optimization, and unifies SFT and RL training paradigms across image classification and multimodal large language model settings.

## Background & Motivation

**Background**: Post-training of multimodal large language models (MLLMs) primarily relies on two paradigms: supervised fine-tuning (SFT), which is stable but requires high-quality annotations and lacks task generalization; and reinforcement learning (RL)-based preference optimization, which can search for better responses from reward signals but incurs high computational cost and training instability. Recent works such as SeVa and SIMA attempt to bridge the two by constructing preference pairs.

**Limitations of Prior Work**: The central challenge in constructing preference pairs lies in how to controllably generate high-quality "loser" samples. Methods such as SeVa rely on classical augmentations (e.g., RandomCrop) to construct losers, but the augmentation is highly stochastic and cannot control loser quality. Furthermore, the DPO loss is decoupled from the data itself, meaning one can only *select* useful training data rather than *generate* effective preference pairs. In image classification, adaptive mixup methods (e.g., PuzzleMix, AutoMix) achieve strong performance but rely on additional forward passes to compute saliency or gradient information, resulting in low efficiency.

**Key Challenge**: In mixup augmentation, a fundamental trade-off exists between efficiency and performance—static methods are fast but weak, while adaptive methods are effective but slow. In MLLM alignment, SFT is stable but lacks preference modeling, while RL incorporates preferences but is unstable. A method that achieves balance along both dimensions simultaneously is needed.

**Goal**: (1) How to design a mixup strategy that is both efficient and effective? (2) How to naturally extend mixup augmentation to MLLM preference alignment? (3) How to establish a direct connection between the loss function and the augmented data?

**Key Insight**: The authors observe that the merging ratio in token merging (ToMe) and the mixing ratio in mixup can be naturally linked—the merge operation is itself a form of information selection, and attention maps recovered after merging can directly guide mixing mask generation. Meanwhile, the mixing ratio can serve as a soft margin in the SimPO preference loss: the more similar the mixed image to the original (larger $\lambda$), the harder the discrimination, and thus the smaller the margin.

**Core Idea**: Bipartite soft matching from token merging is used to generate attention-guided mixed images, and the mixing ratio serves as a dynamic margin in the preference loss, thereby unifying classification augmentation and MLLM alignment training.

## Method

### Overall Architecture

MergeMix operates in two settings: (a) In image classification, input images are encoded through a ToMeAttention encoder to obtain merged token sequences and source mapping matrices; attention maps are then recovered and TopK sampling is applied to generate binary mixing masks, followed by standard mixup training with recalibrated mixing ratios. (b) In the MLLM setting, original images serve as winners and MergeMix-augmented images serve as losers, forming preference pairs that are trained with a mixed SimPO loss combined with an SFT loss.

### Key Designs

1. **Token Merge–Based Image Mixing Strategy**:

    - **Function**: Efficiently generate mixed images that preserve semantic structure.
    - **Mechanism**: Given an initial token sequence $Z_L = f_\theta(\hat{x})$, bipartite soft matching (BSM) is performed via ToMeAttention to merge $r$ similar tokens at $O(N)$ complexity, yielding a compressed sequence $Z_K$ and source mapping matrix $S$. The key innovation is the attention recovery function $\hat{A_L} = \mathcal{R}_{K \to L}(A_K, S)$, which leverages spatial similarity relations in $S$ to expand the merged attention map $A_K$ back to the original length $A_L$. Based on the recovered attention map, TopK selection of $p = \lfloor \lambda \times L \rfloor$ positions generates a binary mask $\mathcal{M}$.
    - **Design Motivation**: Unlike vanilla TopK, which discards low-attention tokens, BSM performs global pairwise matching that preserves spatial topology and contextual continuity. The attention recovery mechanism avoids information loss caused by hard selection.

2. **Mixing Ratio Re-scaling Policy**:

    - **Function**: Ensure that the mixing ratio $\hat{\lambda}$ simultaneously reflects spatial proportions and intrinsic model features.
    - **Mechanism**: Gaussian-based sampling is used to refine the ratio, with mean $\mu = K/L$ (proportion of tokens after merging) and standard deviation $\sigma = p / \sum_i^L \mathcal{M}$ (ratio of selected count to total mask values), followed by clipped normalization: $\hat{\lambda} \sim \mathcal{N}(\mu, \sigma)$, $\hat{\lambda} = \text{clip}(\frac{\hat{\lambda} - \min(\hat{\lambda})}{\max(\hat{\lambda}) - \min(\hat{\lambda}) + \tau}, 0, 1)$.
    - **Design Motivation**: Simple linear mapping fails to capture the degree of information aggregation after token merging. Gaussian smoothing avoids abrupt transitions and makes augmentation more robust.

3. **Mixed SimPO Preference Loss**:

    - **Function**: Organically integrate mixup augmentation with MLLM preference optimization.
    - **Mechanism**: In the MLLM setting, responses to original images serve as winners and responses to mixed images serve as losers. The mixing ratio $\hat{\lambda}$ is mapped to a preference margin $\gamma = 1 - \hat{\lambda}$: a larger $\lambda$ indicates the mixed image is similar to the original (hard to discriminate), so $\gamma$ is reduced to avoid over-optimization; a smaller $\lambda$ indicates a larger discrepancy (easy to discriminate), so $\gamma$ is increased to strengthen the constraint. The final loss is: $\mathcal{L}_{\text{SimPO}}^{\text{Mix}} = -\mathbb{E}[\log \sigma(\frac{\beta}{|y|}\log \pi_\theta(y|x) - \frac{\beta}{|y|}\log \pi_\theta(y|\hat{x}) - (1-\hat{\lambda}))]$.
    - **Design Motivation**: The margin in standard DPO is a fixed hyperparameter decoupled from the data. By directly linking the margin to the degree of augmentation, the model adaptively adjusts optimization intensity according to sample difficulty.

### Loss & Training

Image classification total loss: $\mathcal{L}_{\text{Total}} = \underbrace{\mathcal{L}_{\text{CE}}(f_\theta(\hat{x}), y_i) \cdot \hat{\lambda} + \mathcal{L}_{\text{CE}}(f_\theta(\hat{x}), y_j) \cdot (1-\hat{\lambda})}_{\text{mixup CE}} + \underbrace{\mathcal{L}_{\text{CE}}(f_\theta(x), y)}_{\text{one-hot CE}}$

MLLM total loss: $\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{SFT}} + \mathcal{L}_{\text{SimPO}}^{\text{Mix}}$

## Key Experimental Results

### Main Results (Image Classification)

| Method | DeiT-T | DeiT-S | ViT-S | ViT-B | ViT-L |
|--------|--------|--------|-------|-------|-------|
| Vanilla | 64.70 | 65.81 | 62.64 | 63.33 | 61.83 |
| CutMix | 75.98 | 74.21 | 69.67 | 72.18 | 68.97 |
| PuzzleMix | 73.40 | 73.60 | 70.92 | 71.13 | 69.77 |
| TransMix | 75.31 | 76.17 | 74.15 | 72.87 | 71.40 |
| MixPro | 74.78 | 75.26 | 73.49 | 73.18 | 72.28 |
| **MergeMix** | **77.46** | **78.68** | **77.02** | **75.75** | **76.19** |

CIFAR-100, 200 epochs. MergeMix substantially outperforms all baselines across model scales, exceeding the strongest baseline TransMix by 2.5% on DeiT-S.

### MLLM Benchmark Comparison

| Model | VQAv2 | GQA | SciVQA | TextVQA | MMBench | POPE | AVG | Gain |
|-------|-------|-----|--------|---------|---------|------|-----|------|
| LLaVA-7B | 78.5 | 62.0 | 66.8 | 58.2 | 64.3 | 85.87 | 65.57 | - |
| + CutMix | 79.18 | 62.40 | 70.60 | 57.06 | 66.32 | 86.47 | 65.84 | +0.27 |
| + **MergeMix** | **79.24** | **62.44** | **69.86** | **57.56** | **66.58** | **86.10** | **66.40** | **+0.83** |
| LLaVA-NeXT-7B | 81.8 | 64.2 | 70.1 | 64.9 | 67.4 | 86.5 | 69.3 | - |

### Key Findings

- **Significant efficiency advantage**: With ToMe, MergeMix reduces FLOPs by 16% (4.24G → 3.56G) and improves throughput by 16% (1375 → 1592 img/s) while achieving higher accuracy.
- **Efficiency–performance trade-off on ImageNet-1K**: MergeMix achieves 80.71% top-1 accuracy on DeiT-Small, surpassing all methods that do not employ dynamic forward passes, and is the only mixup method that achieves dynamic acceleration.
- **Consistent gains in the MLLM setting**: Compared to random augmentation methods (CutMix, ResizeMix), MergeMix yields more consistent improvements across all benchmarks; ResizeMix even causes negative gains (−2.24%), underscoring the importance of controllable augmentation.
- **Coupling of mixing ratio and preference margin**: The design $\gamma = 1 - \hat{\lambda}$ applies stronger constraints to easily discriminable samples and milder constraints to hard ones, improving calibration.

## Highlights & Insights

- **Elegant bridging of token merging and mixup**: Token merging is intrinsically an information compression operation, and the source mapping matrix $S$ produced during merging naturally encodes inter-token similarity relations, which can be directly used to generate semantically aware mixing masks without additional saliency computation.
- **Aesthetic unity of the design**: A single mixing ratio $\hat{\lambda}$ simultaneously serves three roles—controlling mask size, calibrating label proportions, and regulating the preference margin—organically connecting all three through a single variable.
- **Transferable idea for bridging SFT and RL**: The approach of constructing preference pairs via data augmentation is generalizable to other modalities (e.g., audio and video multimodal models).

## Limitations & Future Work

- **Limited MLLM experimental scale**: Validation is primarily conducted on LLaVA-7B/13B and Qwen2.5-VL-7B; results on larger-scale models (e.g., 70B) are absent.
- **Mixup restricted to visual tokens**: Text tokens in MLLMs do not participate in mixing; whether purely visual augmentation suffices to model multimodal preferences remains an open question.
- **Lack of theoretical guarantees on preference pair quality**: Although empirically effective, the paper lacks theoretical analysis of why mixed-image responses constitute valid losers.
- **Missing comparison with generative augmentation**: No comparison is made with diffusion model–based augmentation methods such as DiffuseMix.

## Related Work & Insights

- **vs. SeVa (Zhu et al., 2024)**: SeVa constructs losers via RandomCrop, resulting in uncontrollable augmentation with a DPO loss decoupled from the data. MergeMix generates controllable losers through attention-guided mixing and embeds the mixing ratio into the preference loss.
- **vs. PuzzleMix (Kim et al., 2020)**: PuzzleMix generates masks based on gradient information, requiring additional forward–backward passes. MergeMix obtains attention maps "for free" during encoding via ToMe's BSM.
- **vs. TransMix (Chen et al., 2022)**: TransMix uses raw attention maps to reweight label ratios but does not alter the mask generation process. MergeMix operates at the token merging level, jointly optimizing both the mask and the labels.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The idea of unifying token merging, mixup, and preference optimization is novel, though individual components are not entirely new.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 5 classification datasets and 16 MLLM benchmarks; comprehensive evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ Logic is clear, the two settings are described coherently, and mathematical derivations are detailed.
- **Value**: ⭐⭐⭐⭐ Practical significance in both image classification and MLLM settings; the unified paradigm is conceptually inspiring.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] cadrille: Multi-modal CAD Reconstruction with Reinforcement Learning](cadrille_multi-modal_cad_reconstruction_with_reinforcement_learning.md)
- [\[NeurIPS 2025\] NoisyRollout: Reinforcing Visual Reasoning with Data Augmentation](../../NeurIPS2025/reinforcement_learning/noisyrollout_reinforcing_visual_reasoning_with_data_augmenta.md)
- [\[ICLR 2026\] Controllable Exploration in Hybrid-Policy RLVR for Multi-Modal Reasoning](controllable_exploration_in_hybrid-policy_rlvr_for_multi-modal_reasoning.md)
- [\[ICLR 2026\] Chain-of-Context Learning: Dynamic Constraint Understanding for Multi-Task VRPs](chain-of-context_learning_dynamic_constraint_understanding_for_multi-task_vrps.md)
- [\[ICLR 2026\] Understanding and Improving Hyperbolic Deep Reinforcement Learning](understanding_and_improving_hyperbolic_deep_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
