---
title: >-
  [Paper Note] DPC: Dual-Prompt Collaboration for Tuning Vision-Language Models
description: >-
  [CVPR 2025][Multimodal VLM][prompt tuning] This work proposes the Dual-Prompt Collaboration (DPC) framework. By freezing the original tuned prompt to maintain new-class generalization and training a parallel prompt to strengthen base-class performance, along with a weighted decoupled inference mechanism, DPC serves as a plug-and-play module that consistently improves the base-new harmonic mean across four prompt tuning baselines.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "prompt tuning"
  - "base-new trade-off"
  - "dual-prompt collaboration"
  - "hard negative samples"
  - "CLIP fine-tuning"
date: 2026-05-08
content_hash: 2cf314a3ad575c63
---

# DPC: Dual-Prompt Collaboration for Tuning Vision-Language Models

**Conference**: CVPR 2025  
**arXiv**: [2503.13443](https://arxiv.org/abs/2503.13443)  
**Code**: [https://github.com/JREion/DPC](https://github.com/JREion/DPC)  
**Area**: Multimodal VLM  
**Keywords**: prompt tuning, base-new trade-off, dual-prompt collaboration, hard negative samples, CLIP fine-tuning

## TL;DR
This work proposes the Dual-Prompt Collaboration (DPC) framework. By freezing the original tuned prompt to maintain new-class generalization and training a parallel prompt to strengthen base-class performance, along with a weighted decoupled inference mechanism, DPC serves as a plug-and-play module that consistently improves the base-new harmonic mean across four prompt tuning baselines.

## Background & Motivation

**Background**: CLIP prompt tuning adapts to downstream tasks by learning continuous prompt vectors, achieving strong performance on base classes. Representative approaches include CoOp (text prompt), MaPLe (multimodal prompt), and PromptSRC (independent vision-language prompts with regularization).

**Limitations of Prior Work**: All existing methods suffer from the Base-New Trade-off (BNT) problem, where better fine-tuning on base classes leads to worse generalization on new (unseen) classes. This occurs because the prompt is optimized as a single entity, and the gradient directions for enhancing base-class discriminability and maintaining new-class generalization are often mutually exclusive.

**Key Challenge**: The prompt must simultaneously serve two contradictory objectives: precise adaptation to base classes (requiring prompt specialization) and generalization to new classes (requiring prompt generality). Existing methods apply regularization or decoupling at the feature level but do not fundamentally address this conflict at the prompt level.

**Goal**: Decouple base-class optimization and new-class generalization at the prompt level so that they no longer constrain each other.

**Key Insight**: Since a single prompt cannot satisfy both objectives simultaneously, two prompts are employed: a frozen, tuned prompt to preserve generalization, and a new parallel prompt optimized via stronger contrastive learning to enhance base-class performance. During inference, different weights are assigned based on whether the input belongs to the base or new class.

**Core Idea**: Completely separate base-class optimization and new-class generalization at the prompt level using "dual prompts + weighted decoupling", making both objectives independently controllable.

## Method

### Overall Architecture
A two-step training pipeline: Step 1 trains the target prompt learner (e.g., CoOp) normally to obtain the tuned prompt $P$; Step 2 freezes $P$, clones it to initialize a parallel prompt $P'$, and fine-tunes $P'$ using the Dynamic Hard Negative Optimizer (DHNO). During inference, the base class uses a weighted mixture $\tilde{P}_b = \omega_b P' + (1-\omega_b) P$, while the new class almost exclusively uses the frozen prompt $P$.

### Key Designs

1. **Dual-Prompt Initialization and Freezing Strategy**:

    - **Function**: Decouple the base/new class objectives at the prompt level.
    - **Mechanism**: Completely freeze the prompt $P$ obtained from the first step of training (to preserve new-class generalization), and then clone the parameters of $P$ to initialize the parallel prompt $P'$. $P'$ is further optimized to enhance base-class performance, while freezing $P$ ensures that the new-class performance is unaffected.
    - **Design Motivation**: After the first step of training, $P$ already possesses optimal generalization ability on new classes. Continuing training on it would inevitably decay this generalization. The freeze-and-clone strategy fundamentally breaks the causal chain of BNT.

2. **Dynamic Hard Negative Optimizer (DHNO)**:

    - **Function**: Efficiently enhance the base-class discriminative ability of the parallel prompt using a harder contrastive learning task.
    - **Mechanism**: Three sub-modules collaborate: (1) Negative Sampler: evaluates base-class data using the frozen prompt $P$ and selects the incorrect classes from the top-$K$ predictions as hard negative samples, constructing a mini-batch $C' = \{T_g, T_j^-\}_{j=1}^{K-1}$; (2) Feature Filter: performs L2 normalization on all base-class text features to maintain the global distribution and uses a selection matrix $Q$ to extract hard negative features; (3) Hard Negative Optimizer: replaces standard cross-entropy with a symmetric InfoNCE contrastive loss to optimize bidirectional matching (text→image and image→text).
    - **Design Motivation**: The model's own top-$K$ predictions yield the most confusing classes (the semantically closest negative samples), which are more effective than random sampling. The symmetric contrastive loss achieves deeper cross-modal alignment compared to cross-entropy (in ablation studies, cross-entropy only improves HM by 0.22, whereas the contrastive loss improves it by 1.29).

3. **Weighted Decoupling Module (WDM)**:

    - **Function**: Assign independent prompt weights for base and new classes during inference.
    - **Mechanism**: Base-class inference uses $\tilde{P}_b = \omega_b P' + (1-\omega_b) P$ (defaulting to $\omega_b = 0.2$, primarily utilizing the original prompt supplemented by the parallel prompt); new-class inference uses $\tilde{P}_n = \omega_n \cdot \mathcal{F}^{-1}(\tilde{P}_b) + (1-\omega_n) P$ (defaulting to $\omega_n \approx 0$, which essentially relies entirely on the frozen original prompt).
    - **Design Motivation**: It is theoretically proven that prompt optimization does not alter the feature channel distribution (feature channel invariance), justifying the linear weighted combination. Setting $\omega_n \approx 0$ means the parallel prompt is completely uninvolved during new-class inference, fundamentally avoiding BNT.

### Loss & Training
Step 2 utilizes a symmetric InfoNCE contrastive loss $\mathcal{L}_{CL}$ to compute the text→image and image→text matching scores on mini-batches constructed with hard negative samples. Trained for 20 epochs, it introduces an extra ~8K parameters (from 8K to 16K) with a training memory overhead of only 0.25GB.

## Key Experimental Results

### Main Results

| Method | Base | New | HM |
|------|------|------|------|
| CoOp | 81.98 | 68.84 | 74.84 |
| CoOp+DPC | **85.15** | 68.84 | **76.13** |
| MaPLe | 83.52 | 73.31 | 78.08 |
| MaPLe+DPC | **85.93** | 73.31 | **79.12** |
| PromptSRC | 83.45 | 74.78 | 78.87 |
| PromptSRC+DPC | **86.10** | 74.78 | **80.04** |
| PromptKD | 86.86 | 80.55 | 83.59 |
| PromptKD+DPC | **87.55** | 80.55 | **83.91** |

### Ablation Study

| Configuration | Base | New | HM | Explanation |
|------|------|------|------|------|
| CoOp baseline | 81.98 | 68.84 | 74.84 | Baseline |
| +Dual-Prompts | 82.69 | 68.39 | 74.86 | Minor improvement with cloning only |
| +Dual-Prompts+DHNO | 84.28 | 64.12 | 72.83 | Severe BNT without decoupling |
| +Dual-Prompts+DHNO+Weighting+Decoupling | **85.15** | **68.84** | **76.13** | Optimal full framework |

### Key Findings
- **DPC is consistently effective across all baselines**: On 4 different prompt-learning architectures, it improves base accuracy by 0.69 to 3.17 percentage points while perfectly maintaining new-class accuracy.
- **Decoupling is a necessary condition**: Using DHNO alone causes new-class accuracy to drop from 68.84 to 64.12 (exacerbating BNT); it must be coupled with the decoupling mechanism of WDM to prevent this.
- **Symmetric contrastive loss is far superior to cross-entropy**: Using cross-entropy with hard negative samples improves HM by only 0.22, whereas the contrastive loss improves it by 1.29, demonstrating that deep cross-modal alignment requires bidirectional optimization.
- **Extremely low computational overhead**: It adds only ~8K parameters and 0.25GB GPU memory, with the inference speed (FPS) remaining practically unaffected (767 -> 758).

## Highlights & Insights
- **"Prompt-level decoupling" is more fundamental than feature-level decoupling**: Prior methods like DePT decouple in the feature space, which is limited by feature expressiveness. In contrast, DPC decouples in the prompt space, providing a larger optimization space and allowing precise control of base/new class preferences during inference.
- **Clever design of self-hard-negative sampling**: Using the model's own top-$K$ predictions as hard negative samples eliminates the need for external knowledge bases or extra data, serving as an elegant form of self-supervised hard sample mining.
- **Plug-and-play engineering friendliness**: It does not modify any baseline architectures and adds only a single step after training, rendering it compatible with various prompt tuning methods.

## Limitations & Future Work
- It requires prior knowledge of whether a test sample belongs to the "base" or "new" class to select different $\omega$ values, which may necessitate an additional open/closed-set detection mechanism in practical applications.
- Using a fixed hyperparameter $\omega_b = 0.2$ may not suit all datasets, as evidenced by MaPLe achieving optimal performance at $\omega_b = 1.0$.
- The method is only verified on ViT-B/16; its effectiveness on larger backbones like ViT-L/14 remains unstudied.
- The total number of epochs for two-step training is doubled (20 + 20 = 40). Although the paper shows that 5 + 5 = 10 is also effective, it still increases training time.

## Related Work & Insights
- **vs. DePT (feature decoupling)**: DePT separates base and new class directions in the feature space, which is constrained by the representation capacity of the feature space. DPC's decoupling in the prompt space is more flexible, yielding a larger HM improvement on PromptSRC (+1.17) than DePT (+0.86).
- **vs. CoPrompt / TCP**: These methods attempt to mitigate BNT by improving training strategies, yet they still rely on single-prompt optimization. DPC structurally introduces dual prompts, which is orthogonal to and combinable with these methods.
- **vs. PromptKD**: Knowledge-distillation-based methods have already significantly mitigated BNT, yet DPC still further improves base-class performance by 0.69 percentage points.

## Rating
- Novelty: ⭐⭐⭐⭐ First to perform base-new decoupling at the prompt level; the approach is simple yet highly effective. The self-hard-negative sampling is also clever.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 baselines, 11 datasets, and extremely detailed ablation studies where the contribution of each component is clearly isolated.
- Writing Quality: ⭐⭐⭐⭐ The methodology motivation is clear, and the theoretical analysis (feature channel invariance) provides solid support for the linear weighting.
- Value: ⭐⭐⭐⭐ The plug-and-play nature gives it strong practical utility, though the scenario assumptions of the BNT problem itself are somewhat constrained.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Visual and Semantic Prompt Collaboration for Generalized Zero-Shot Learning](visual_and_semantic_prompt_collaboration_for_generalized_zero-shot_learning.md)
- [\[ICML 2025\] Understanding and Mitigating Miscalibration in Prompt Tuning for Vision-Language Models](../../ICML2025/multimodal_vlm/understanding_and_mitigating_miscalibration_in_prompt_tuning_for_vision-language.md)
- [\[CVPR 2026\] Towards Calibrating Prompt Tuning of Vision-Language Models](../../CVPR2026/multimodal_vlm/towards_calibrating_prompt_tuning_of_vision-language_models.md)
- [\[CVPR 2025\] NLPrompt: Noise-Label Prompt Learning for Vision-Language Models](nlprompt_noise-label_prompt_learning_for_vision-language_models.md)
- [\[ICCV 2025\] FedMVP: Federated Multimodal Visual Prompt Tuning for Vision-Language Models](../../ICCV2025/multimodal_vlm/fedmvp_federated_multimodal_visual_prompt_tuning_for_vision-language_models.md)

</div>

<!-- RELATED:END -->
