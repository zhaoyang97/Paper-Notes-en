---
title: >-
  [Paper Note] Complexity Experts are Task-Discriminative Learners for Any Image Restoration
description: >-
  [CVPR 2025][Image Restoration][All-in-One Image Restoration] MoCE-IR is proposed to replace the uniform architecture of traditional MoEs with "complexity experts" possessing varying computational complexities and receptive field sizes. Complemented by a spring-like routing mechanism biased towards low complexity, it unexpectedly achieves task-discriminative allocation—different degradation types are automatically routed to experts of appropriate complexity…
tags:
  - "CVPR 2025"
  - "Image Restoration"
  - "All-in-One Image Restoration"
  - "Mixture-of-Experts"
  - "Complexity-aware Routing"
  - "Task-Discriminative Learning"
  - "Sparse Computation"
date: 2026-05-08
content_hash: b1bca508a5aee504
---

# Complexity Experts are Task-Discriminative Learners for Any Image Restoration

**Conference**: CVPR 2025  
**arXiv**: [2411.18466](https://arxiv.org/abs/2411.18466)  
**Code**: [https://eduardzamfir.github.io/MoCE-IR/](https://eduardzamfir.github.io/MoCE-IR/)  
**Area**: Image Restoration / Model Compression  
**Keywords**: All-in-One Image Restoration, Mixture-of-Experts, Complexity-aware Routing, Task-Discriminative Learning, Sparse Computation

## TL;DR
MoCE-IR is proposed to replace the uniform architecture of traditional MoEs with "complexity experts" possessing varying computational complexities and receptive field sizes. Complemented by a spring-like routing mechanism biased towards low complexity, it unexpectedly achieves task-discriminative allocation—different degradation types are automatically routed to experts of appropriate complexity, allowing irrelevant experts to be bypassed during inference.

## Background & Motivation

**Background**: All-in-one image restoration handles multiple degradations, such as denoising, dehazing, and deraining, using a single model. Many parameters of dense models (e.g., PromptIR) remain idle during specific tasks. The MoE architecture is a natural extension, but existing MoE methods (AnyIR, InstructIR, etc.) suffer from routing inconsistency—some experts unexpectedly generalize across tasks, while others perform poorly on their assigned tasks.

**Limitations of Prior Work**: (1) Uniform architecture of traditional MoEs: all experts possess identical computational complexity and receptive fields, whereas different degradations require distinct processing—motion blur requires local spatial awareness, while dehazing demands global context understanding. (2) Routing difficulties: degradation complexity is unknown beforehand, and routing based on language or degradation priors leads to unbalanced optimization.

**Key Challenge**: How to automatically determine which expert is suitable for which task, allowing different experts to specialize in distinct tasks to enable bypassing irrelevant experts during inference (computational efficiency)?

**Goal**: To design an MoE architecture where experts automatically align with degradation tasks of varying complexities, supporting the bypassing of unnecessary experts during inference.

**Key Insight**: Different degradations have different inherent complexities. Designing experts with diverse computation costs and receptive fields, coupled with a routing scheme biased towards simple experts, surprisingly allows tasks to be automatically allocated to experts of appropriate complexity just through this simplicity preference.

**Core Idea**: Design expert blocks with progressively increasing computational complexity and receptive fields, and employ a spring-force routing mechanism biased toward low complexity, achieving automatic optimal matching between degradation tasks and experts.

## Method

### Overall Architecture
A U-shaped asymmetric encoder-decoder architecture. The encoder uses standard Transformer blocks, and the decoder integrates MoCE layers. Each MoCE layer contains $n$ complexity-increasing experts $\mathbf{E}$ and a shared expert $\mathbf{S}$. High-frequency Sobel feature vectors guide the gating function. Finally, high-frequency details are refined via global residual connections.

### Key Designs

1. **Complexity Experts**

    - **Function**: Provide a range of processing capacities from lightweight to heavyweight to match the demands of different degradations.
    - **Mechanism**: The channel dimension of the $i$-th expert is $r_i = C/2^i$ (narrower as $i$ increases), and the window size is $w_i$ (larger as $i$ increases). Thus, lightweight experts process local, simple degradations, while heavyweight experts handle complex degradations requiring global context. Each expert utilizes FFT-accelerated window self-attention. The outputs of the experts are modulated via element-wise multiplication with the shared expert (which uses transposed attention across channel dimensions) and are ultimately merged through cross-attention. The shared expert captures features shared across degradations.
    - **Design Motivation**: Uniform experts fail to reflect the inherent complexity differences of tasks, leading to chaotic routing. The heterogeneous design provides clear "destinations" for routing.

2. **Complexity-aware Routing**

    - **Function**: Automatically allocate input degraded images to experts with appropriate complexity.
    - **Mechanism**: Image-level top-1 routing (non-token-level), where the expert with the highest score after softmax is selected. The key innovation is a complexity bias $\mathbf{b} = [p_1/p_{max}, p_2/p_{max}, ..., p_n/p_{max}]$, which is multiplied by the importance scores to grant higher effective weights to low-complexity experts. Analogous to a spring system: parameter counts correspond to displacement, and the normalization factor corresponds to the spring constant, jointly determining the magnitude of the biasing force. The auxiliary loss $\mathcal{L}_{aux}$ utilizes complexity-weighted importance and coefficient of variation for load balancing.
    - **Design Motivation**: Without bias, routing becomes randomized, leading to underutilized model capacity. The intuition behind biasing towards simple experts is: if a simple expert can handle the task, there is no need to deploy a complex expert; only complex degradations will "overcome the spring force" to be assigned to larger experts.

3. **Inference-time Expert Bypassing**

    - **Function**: Skip experts unrelated to the current degradation during inference based on post-training routing statistics, saving computation.
    - **Mechanism**: Due to the task-discriminative nature of the routing, each degradation type is almost always assigned to the same expert after training. During inference, only the expert corresponding to the given task needs to be activated, while other experts can be bypassed.
    - **Design Motivation**: This represents the core practical value of MoCE—while dense models always activate all parameters, MoCE achieves true conditional computation.

### Loss & Training
$\mathcal{L} = \mathcal{L}_{pixel} + \lambda \mathcal{L}_{aux}$, pixel loss + complexity-aware routing auxiliary loss. Based on the Restormer architecture.

## Key Experimental Results

### Main Results

| Method | Params | Dehaze PSNR↑ | Derain PSNR↑ | Denoise σ25↑ | Average↑ |
|------|------|---------|---------|---------|------|
| AirNet | 9M | 27.94 | 34.90 | 31.26 | 31.20 |
| PromptIR | 36M | 30.58 | 36.37 | 31.31 | 32.06 |
| Art-PromptIR | 33M | 30.83 | 37.94 | 31.42 | 32.49 |
| **MoCE-IR-S** | **11M** | **30.94** | **38.22** | **31.42** | **32.57** |

The lightweight version with 11M parameters surpasses the state-of-the-art (SOTA) models with 33-36M parameters.

### Ablation Study

| Configuration | Average PSNR | Description |
|------|---------|------|
| Uniform Experts + Standard Routing | 32.19 | Traditional MoE |
| Complexity Experts + Standard Routing | 32.38 | Heterogeneous Design Only |
| **Complexity Experts + Complexity Routing** | **32.57** | Full Method |

### Key Findings
- Routing biased toward low complexity unexpectedly achieves task-discriminative allocation—denoising is assigned to the simplest expert, while dehazing is assigned to the most complex expert.
- The 11M lightweight version surpasses the 36M PromptIR, proving that efficient parameter utilization is more effective than blindly increasing parameters.
- Skipping experts during inference can further reduce FLOPs with almost no performance degradation.
- The complementarity between the channel attention of the shared expert and the spatial attention of the complexity experts is key to the performance.

## Highlights & Insights
- **"Preference for Simplicity" Enabling Task Discrimination** is the most core finding—simple degradations naturally remain with simple experts, while complex degradations "overcome the spring force" and are pushed to large experts. This emergent task allocation behavior is highly elegant.
- **Spring Force Analogy** is intuitive and theoretically grounded—the complexity bias perfectly corresponds to the restoring force in physical systems.
- **Heterogeneous Expert Design** opens up a new direction for the application of MoE in low-level vision tasks.

## Limitations & Future Work
- Image-level routing (non-token-level) cannot handle cases where multiple degradations are mixed within a single image.
- The number of experts and the complexity gradients require manual design.
- Currently verified only on the three-degradation benchmark; expansion to more degradation types (such as super-resolution, deblurring) has not been fully tested.

## Related Work & Insights
- **vs PromptIR**: PromptIR encodes degradation-specific information using tunable prompts with 36M parameters. MoCE-IR surpasses it with only 11M parameters, owing to more efficient conditional computation.
- **vs AnyIR**: AnyIR employs MoE but uses uniform experts + routing based on degradation priors, leading to routing inconsistency. This work resolves the routing issue using heterogeneous experts and complexity bias.
- **vs InstructIR**: InstructIR relies on LLM text prompts, which demand substantial resources. This work is fully vision-driven and lightweight.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The combination of complexity experts and spring-force routing is highly original, and the emergent task-discriminative property is surprising.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-scale models + ablations + efficiency analysis, but more degradation types could be included.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The logical chain from motivation to observation to method to results is flawless, and the spring force analogy is intuitive.
- **Value**: ⭐⭐⭐⭐⭐ It holds important reference value for both all-in-one restoration and efficient MoE design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Devil is in the Uniformity: Exploring Diverse Learners within Transformer for Image Restoration](../../ICCV2025/image_restoration/devil_is_in_the_uniformity_exploring_diverse_learners_within_transformer_for_ima.md)
- [\[ICCV 2025\] Exploiting Diffusion Prior for Task-driven Image Restoration](../../ICCV2025/image_restoration/exploiting_diffusion_prior_for_task-driven_image_restoration.md)
- [\[CVPR 2025\] Degradation-Aware Feature Perturbation for All-in-One Image Restoration](degradation-aware_feature_perturbation_for_all-in-one_image_restoration.md)
- [\[CVPR 2025\] Visual-Instructed Degradation Diffusion for All-in-One Image Restoration](visual-instructed_degradation_diffusion_for_all-in-one_image_restoration.md)
- [\[ICLR 2026\] Learning Domain-Aware Task Prompt Representations for Multi-Domain All-in-One Image Restoration](../../ICLR2026/image_restoration/learning_domain-aware_task_prompt_representations_for_multi-domain_all-in-one_im.md)

</div>

<!-- RELATED:END -->
