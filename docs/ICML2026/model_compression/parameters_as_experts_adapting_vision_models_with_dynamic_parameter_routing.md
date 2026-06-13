---
title: >-
  [Paper Note] Parameters as Experts: Adapting Vision Models with Dynamic Parameter Routing
description: >-
  [ICML 2026][Model Compression][PEFT] The authors treat "parameters themselves as experts"—maintaining a pool of trainable parameter matrices (shared expert center) shared across stages. The ParaX adapter in each layer dy…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "PEFT"
  - "MoE Adapter"
  - "Shared Expert Center"
  - "Dynamic Parameter Routing"
  - "Dense Prediction"
date: 2026-05-08
content_hash: 1a7b35568e29bede
---

# Parameters as Experts: Adapting Vision Models with Dynamic Parameter Routing

**Conference**: ICML 2026  
**arXiv**: [2602.06862](https://arxiv.org/abs/2602.06862)  
**Code**: https://github.com/LMMMEng/ParaX  
**Area**: Model Compression / Parameter-Efficient Fine-Tuning  
**Keywords**: PEFT, MoE Adapter, Shared Expert Center, Dynamic Parameter Routing, Dense Prediction  

## TL;DR
The authors treat "parameters themselves as experts"—maintaining a pool of trainable parameter matrices (shared expert center) shared across stages. The ParaX adapter in each layer dynamically **synthesizes** weights for low-rank projections and multi-scale depthwise convolutions via a lightweight router based on the current input. This simultaneously addresses two major flaws of traditional adapters: "input-independence" and "cross-layer redundancy," consistently outperforming full fine-tuning on dense prediction tasks with <5% trainable parameters.

## Background & Motivation

**Background**: For PEFT (Parameter-Efficient Fine-Tuning) on vision models, the field is currently divided into two main schools: the prompt school (VPT, etc.) which inserts learnable tokens into the input sequence, and the adapter school (LoRA, AdaptFormer, Mona, etc.) which embeds a pair of low-rank matrices $W_1\in\mathbb{R}^{C\times\hat C}, W_2\in\mathbb{R}^{\hat C\times C}$ into each layer. The adapter school currently holds a clear advantage in dense prediction tasks (segmentation, detection). Mona further strengthens spatial modeling by embedding multi-kernel depthwise convolutions within the adapter.

**Limitations of Prior Work**: The authors empirically identify two "fatal flaws" in existing adapters.

- **Representation Deficiency**: Once trained, adapter weights are **input-independent**. Visualizing tuned Swin-L models on COCO using ERF (effective receptive field) for Mona and AdaptFormer reveals that ERFs are significantly smaller than those of full fine-tuning—meaning low-rank, input-independent transforms cannot customize optimal spatial responses for different image contents.
- **Feature Redundancy**: Adapter parameters for each layer are isolated. CKA analysis shows learned patterns across different layers are extremely similar, indicating a lack of explicit information interaction across layers, where the same patterns are repeatedly learned.

**Key Challenge**: Under strict parameter budgets (LoRA-style, a few million trainable parameters), **static and isolated** low-rank adapters can neither be customized based on input nor allow cross-layer information flow, resulting in only "universal but mediocre" transformations. Fixing these issues cannot be achieved by simply increasing $\hat C$ (parameter budget explosion) or sharing the same set of weights across adapters (forcing identical transformations across layers, further collapsing expressivity).

**Goal**: (1) Make adapter weights change dynamically with input to restore ERF; (2) Enable implicit information flow between multi-layer adapters to reduce CKA redundancy; (3) Achieve both without significantly increasing the trainable parameter budget.

**Key Insight**: Move the MoE concept of "selecting different experts with a router" to the **parameter level**—experts are not sub-networks but **trainable parameter matrices** of the same size. Different layers share the same group of experts, but each layer has its own router providing different mixing coefficients. This achieves both "input-dependence" (the router sees the input) and "cross-layer coupling" (shared experts must serve multiple layers, forcing them to learn universal and diverse bases).

**Core Idea**: Place a **shared expert center** (a pool of trainable parameter matrices of the same size) in each stage. Let the ParaX module of each layer use an ultra-lightweight router to **linearly blend** them into low-rank projections and multi-scale depthwise convolution kernels specific to the current layer and input—parameters are experts, and routing is synthesis.

## Method

### Overall Architecture
Taking a four-stage hierarchical backbone (Swin / ConvNeXt) as an example. A shared expert center is set in each stage, with a ParaX adapter assigned to each building block: two in each Swin block (after token mixer and channel mixer) and one in each ConvNeXt block (after the entire residual module). During the forward pass, each ParaX module independently uses a router to read the current input features → output dynamic coefficients → linearly combine several parameter matrices from the expert center into $W_1, W_2$ and three-scale depthwise kernels specific to the current module → perform low-rank transformations ("dimensionality reduction → multi-scale spatial mixing → dimensionality expansion") with these dynamic weights → add residual back to the backbone. The entire process introduces no additional trainable parameters except for the expert center + router + task-specific heads; all backbone weights are frozen.

### Key Designs

1. **Shared Expert Center**:

    - **Function**: Maintains a pool of trainable parameter matrices at the stage level to serve as "bases" for subsequent dynamic synthesis.
    - **Mechanism**: Channel-wise low-rank projection experts are stored in pairs, $\mathbf{E}_A\in\mathbb{R}^{M\times C\times\hat C}, \mathbf{E}_B\in\mathbb{R}^{M\times\hat C\times C}$, where $M$ is the expert capacity and $\hat C$ is the adapter hidden dimension; these are the two hyperparameters controlling the total parameter count. Spatial depthwise convolution experts are added in three groups $\mathbf{S}_A, \mathbf{S}_B, \mathbf{S}_C\in\mathbb{R}^{M\times\hat C\times K_i^2}$ corresponding to different kernel sizes. All ParaX modules pull weights from the same (stage-level) expert center.
    - **Design Motivation**: Place "parameter sharing" at the **expert pool level** instead of the adapter level—achieving both cross-layer coupling (shared experts used across layers, forced to learn universal and diverse features) without forcing all layers to produce identical adapters (individual layer routers are independent). This precisely resolves the conflicting requirements of "feature redundancy" and "expressivity collapse."

2. **Dynamic Parameter Routing**:

    - **Function**: Outputs a set of coefficients given the current input features to linearly synthesize $W_1, W_2$ specific to the current layer and sample from the expert center.
    - **Mechanism**: Input $\mathbf{X}\in\mathbb{R}^{HW\times C}$ first undergoes GAP to obtain a channel descriptor, then passes through a linear layer that reduces the dimension to 16 for a hidden vector. Subsequently, two parallel linear layers + softmax produce two gating vectors $\mathbf{G}_1, \mathbf{G}_2\in\mathbb{R}^M$. Dynamic weights are synthesized via tensor contraction: $\mathbf{W}_1=\sum_{m=1}^M \mathbf{G}_1[m]\,\mathbf{E}_A[m]\in\mathbb{R}^{C\times\hat C}$, $\mathbf{W}_2=\sum_{m=1}^M \mathbf{G}_2[m]\,\mathbf{E}_B[m]\in\mathbb{R}^{\hat C\times C}$. Then, standard LoRA-style residual updates are performed: $\mathbf{Y}=\mathbf{X}+\sigma(\mathbf{X}\mathbf{W}_1)\mathbf{W}_2$ (with spatial mixing in between).
    - **Design Motivation**: Replace the classical MoE "expert selection" with "linear expert mixing"—preserving the dynamic nature of "deciding weights based on input" while avoiding the training instability of top-k sparse routing. The router dimension is intentionally compressed to 16 to make the router's own parameters and computation nearly negligible, ensuring the "minimal trainable parameters" constraint of PEFT holds.

3. **Dynamic Multi-scale Spatial Mixing (D²Conv)**:

    - **Function**: Inserts spatial mixing between low-rank projections, upgrading the adapter from a pure channel transformation to a spatio-temporal aware sub-network where spatial kernels are also dynamically synthesized.
    - **Mechanism**: The router outputs three additional gates $\mathbf{G}_A, \mathbf{G}_B, \mathbf{G}_C\in\mathbb{R}^M$ to synthesize three **dynamic depthwise convolution kernels** with $\mathbf{S}_A, \mathbf{S}_B, \mathbf{S}_C$ (kernel sizes increasing, typically $3\times3, 5\times5, 7\times7$ in the paper). Spatial mixing utilizes a **sequential stacking + residual** structure: input features pass through three D²Convs sequentially, each with a residual shortcut, gradually expanding the receptive field. Then, a Spatially-varying Aggregation (SA) module—a $1\times1$ conv + softmax—generates three spatial attention maps to perform point-wise multiplication and summation with the three-scale features. Note that ParaX uses **depthwise** dynamic convolutions (D²Conv), differing from works like KernelWarehouse that use standard conv; this is a hard requirement for the PEFT budget.
    - **Design Motivation**: Mona has proven multi-kernel depthwise conv is crucial for dense prediction, but their kernels are **static and shared across all samples**. ParaX allows kernels to participate in "synthesis based on input," extending dynamics from channels to space. In coordination with the expanded ERF, this directly boosts dense prediction. SA provides a final layer of dynamic weighting in the spatial dimension, with negligible parameter cost, to further refine which scale is used at each pixel location.

### Loss & Training
ParaX follows a pure PEFT setup: backbone frozen, training only the expert center + router + task heads (standard heads for segmentation/detection/classification). Standard training recipes are used for all tasks (UperNet/ADE20K 160K iter; Mask R-CNN/COCO; MAE-pretrained ViT-B/16 on classification) without introducing new losses. $M$ (expert capacity) and $\hat C$ (hidden dimension) are the two core hyperparameters controlling the trainable parameter budget; multi-kernel combinations $\{K_1, K_2, K_3\}$ are ablated in Section 4.5.

## Key Experimental Results

### Main Results

Comparison of ADE20K semantic segmentation (mIoU) and COCO2017 detection/instance segmentation (AP$^b$/AP$^m$) (extracted from Table 1, Table 2):

| Backbone | Method | Trainable Params (M) | ADE20K mIoU | COCO AP$^b$ | COCO AP$^m$ |
|---|---|---|---|---|---|
| Swin-B | Full fine-tuning | 86.8 | 50.2 | 47.5 | 42.8 |
| Swin-B | LoRA | 5.4 | 49.4 | 40.1 | 38.5 |
| Swin-B | Mona | 5.2 | 49.8 | 46.6 | 42.4 |
| Swin-B | **ParaX** | **5.2** | **50.3** | **47.3** | **42.7** |
| Swin-L | Full fine-tuning | 195.0 | 51.2 | 48.6 | 43.8 |
| Swin-L | Mona | 7.5 | 51.6 | 48.1 | 43.9 |
| Swin-L | **ParaX** | **7.3** | **52.0** | **48.6** | **44.0** |
| ConvNeXt-B | Full fine-tuning | 87.6 | 51.4 | 47.8 | 43.0 |
| ConvNeXt-B | Mona | 6.5 | 50.7 | 47.5 | 43.2 |
| ConvNeXt-B | **ParaX** | **6.5** | **51.1** | **48.0** | **43.5** |
| ConvNeXt-L | Full fine-tuning | 196.2 | 52.4 | 48.1 | 43.2 |
| ConvNeXt-L | Mona | 9.1 | 51.5 | 48.9 | 44.4 |
| ConvNeXt-L | **ParaX** | **9.2** | **52.0** | **49.5** | **44.8** |

Highlights: ParaX achieves SOTA in all 8 settings across 4 backbones and 2 tasks. For large models like Swin-L (segmentation) / ConvNeXt-L (detection), it outperforms full fine-tuning by 0.8% mIoU and 1.4% AP$^b$ with <5% trainable parameters. Visualizations of ERF/CKA in panel (c) empirically demonstrate that ParaX's ERF is close to full fine-tuning and cross-layer redundancy (CKA) is significantly reduced.

### Ablation Study: Cross-task transferability (panoptic segmentation, COCO2017)

| Backbone | Method | Params (M) | PQ | SQ | RQ |
|---|---|---|---|---|---|
| Swin-B | Full-tuning | 86.8 | 50.3 | 81.3 | 60.6 |
| Swin-B | AdaptFormer | 5.4 | 47.1 | 79.4 | 57.4 |
| Swin-B | Mona | 5.2 | 48.1 | 79.9 | 58.3 |
| Swin-B | **ParaX** | **5.2** | **48.8** | **80.8** | **59.0** |
| Swin-L | Full-tuning | 195.0 | 51.4 | 81.5 | 61.9 |
| Swin-L | Mona | 7.5 | 49.7 | 80.7 | 60.2 |
| Swin-L | **ParaX** | **7.3** | **50.2** | **81.3** | **60.5** |

Panoptic segmentation combines segmentation and detection, placing the highest demand on representation, and is the weakest link in prior PEFT work. ParaX outperforms AdaptFormer by 1.7 PQ and Mona by 0.7 PQ on Swin-B, narrowing the gap with full fine-tuning to 1.2–1.5 PQ. This is the most discriminative task compared to other baselines, validating the value of expressivity brought by "dynamic + cross-layer sharing" for unified dense tasks.

### Key Findings
- **Alignment of ERF/CKA with Prediction Failure Modes**: The authors use ERF/CKA in Figure 1(c) to diagnose "representation deficiency" and "feature redundancy." ParaX aligns both metrics closer to full fine-tuning, with main metrics subsequently proving this alignment yields higher accuracy. This empirical loop of "diagnosing lesions with metrics, then treating them with methodology" is rare in PEFT papers and reusable.
- **Expert Center Scale vs. Task Difficulty**: In the ablation study (Section 4.5), the optimal ratio of $M$ and $\hat C$ varies with task granularity; dense prediction benefits from larger $M$, while classification is more sensitive to $\hat C$. This suggests the "expert pool" in dynamic parameter routing should be configured based on task granularity rather than a one-size-fits-all approach.
- **Sequential Stacking of Multi-kernel D²Conv > Parallel**: Sequential + residual stacking is more conducive to smooth ERF expansion compared to parallel branches, consistent with works like SegMan and SegFormer on "progressive receptive fields."

## Highlights & Insights
- **Elegant "Parameters as Experts" Perspective**: Classic MoE uses routers to select sub-networks, at the cost of routing sparsity and large experts. This paper shrinks experts to parameter matrices of the same size and uses dense linear mixing, reducing complexity to $O(M)$ matrix weighted sums. This makes the MoE concept feasible within the "few million parameters" budget of PEFT.
- **Shared Expert Center Resolves Two Conflicting Goals**: Cross-layer sharing = information flow (lower CKA); layer-specific routers = diversity of expression (higher ERF). Two seemingly conflicting goals are naturally made compatible by the shared pool design.
- **Transferability of Dynamic Kernel Synthesis**: The D²Conv's "coefficients × basis" logic is essentially kindred to KernelWarehouse and CondConv, but its integration into the PEFT + depthwise + residual stacking combination is novel. This "router + parameter basis" synthesis primitive can be directly transferred to other adapter forms like LoRA or AdaLoRA.
- **Another Case for PEFT Outperforming Full Fine-tuning**: When the backbone is sufficiently pre-trained, too many trainable parameters can lead to overfitting on downstream distributions. ParaX outperforming full fine-tuning by 1.4% AP$^b$ on ConvNeXt-L reinforces this phenomenon—PEFT is not just about saving costs.

## Limitations & Future Work
- **Inference Computational Overhead**: Dynamically synthesizing $\mathbf{W}_1, \mathbf{W}_2$ and three D²Conv kernels requires running the router and tensor contraction for every token (or image). Although the authors claim efficiency is acceptable, it is a disadvantage compared to LoRA's "merge into backbone after training, zero inference overhead," requiring caution in latency-sensitive scenarios.
- **Lack of Automation for Expert Center Scale Selection**: $M$ and $\hat C$ need to be searched per task; the paper provides empirical values but no closed-form criteria. When $M$ is too large, experts may collapse (similar to expert collapse in MoE), a risk not discussed in depth.
- **Minimalist Router may be Limiting**: The router design (16-dim hidden + softmax) saves parameters, but dense prediction might require token-level (rather than image-level) routing. How token-level routing interacts with depthwise conv remains an open question.
- **Not Validated on LLM/VLM**: The method is architecture-neutral, but experiments are limited to vision backbones and dense prediction. Performance on LLM SFT or VLM fine-tuning has not yet been demonstrated.

## Related Work & Insights
- **vs LoRA (Hu et al. 2022)**: LoRA uses static low-rank $W_1 W_2$ updates; ParaX is its dynamic, cross-layer shared version. It degenerates into LoRA when $M=1$ and the router outputs a constant 1.
- **vs AdaptFormer / Mona**: AdaptFormer is a "static, no-spatial-kernel" special case of ParaX. Mona adds static multi-kernel depthwise conv; ParaX dynamizes these kernels and puts them in a shared expert pool, solving both ERF and CKA issues.
- **vs MoELoRA / HydraLoRA / MoLA**: These methods introduce MoE inside LoRA, but experts are still sub-modules and require sparse routing. ParaX shrinks experts into parameter matrices and uses dense mixing, leading to more stable routing training.
- **vs KernelWarehouse / OmniNet (Dynamic Convolution)**: The core idea (coefficients × basis) is similar, but ParaX uses depthwise convolutions to fit the PEFT budget and extends the "parameter pool" to the full adapter (including channel projection and spatial kernels) rather than just dynamizing the convolution itself.

## Rating
- Novelty: ⭐⭐⭐⭐ Stripping MoE down to the granularity of "parameters as experts" is a refreshing perspective in the adapter family.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers various tasks (semantic/detection/instance/panoptic/classification) and two backbone types across two scales, with both main metrics and ERF/CKA diagnostics, though it lacks cross-domain (LLM/VLM) evaluation.
- Writing Quality: ⭐⭐⭐⭐ The "lesion diagnosis" narrative in Figure 1(c) is very persuasive; formulas and diagrams are well-coordinated.
- Value: ⭐⭐⭐⭐ Consistently outperforms full fine-tuning in PEFT, and the "dynamically synthesized adapter weights" is a universal primitive transferable to LLM/VLM.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] LD-MoLE: Learnable Dynamic Routing for Mixture of LoRA Experts](../../ICLR2026/model_compression/ld-mole_learnable_dynamic_routing_for_mixture_of_lora_experts.md)
- [\[ICML 2026\] PRISM: Synergizing Vision Foundation Models via Self-Organized Expert Specialization](prism_synergizing_vision_foundation_models_via_self-organized_expert_specializat.md)
- [\[CVPR 2026\] Quant Experts: Token-aware Adaptive Error Reconstruction with Mixture of Experts for Large Vision-Language Models Quantization](../../CVPR2026/model_compression/quant_experts_token_aware_vlm_quantization.md)
- [\[ICML 2026\] Continual Model Routing in Evolving Model Hubs](continual_model_routing_in_evolving_model_hubs.md)
- [\[ICML 2026\] FRISM: Fine-Grained Reasoning Injection via Subspace-Level Model Merging for Vision–Language Models](frism_fine-grained_reasoning_injection_via_subspace-level_model_merging_for_visi.md)

</div>

<!-- RELATED:END -->
