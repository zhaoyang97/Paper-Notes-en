---
title: >-
  [Paper Note] VLA-Cache: Efficient Vision-Language-Action Manipulation via Adaptive Token Caching
description: >-
  [NeurIPS 2025][Multimodal VLM][VLA acceleration] This paper proposes VLA-Cache, a training-free inference acceleration method for VLA models that identifies and caches KV representations of static visual tokens across frames, filters out task-relevant tokens, and adaptively adjusts the reuse ratio per layer, achieving 1.7× speedup with negligible loss in task success rate.
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - VLA acceleration
  - token caching
  - inference speedup
  - training-free
  - robotic manipulation
date: 2026-05-08
content_hash: d90d38d6dbdc708f
---

# VLA-Cache: Efficient Vision-Language-Action Manipulation via Adaptive Token Caching

**Conference**: NeurIPS 2025
**arXiv**: [2502.02175](https://arxiv.org/abs/2502.02175)
**Code**: [Project Page](https://vla-cache.github.io)
**Area**: Multimodal VLM
**Keywords**: VLA acceleration, token caching, inference speedup, training-free, robotic manipulation

## TL;DR

This paper proposes VLA-Cache, a training-free inference acceleration method for VLA models that identifies and caches KV representations of static visual tokens across frames, filters out task-relevant tokens, and adaptively adjusts the reuse ratio per layer, achieving 1.7× speedup with negligible loss in task success rate.

## Background & Motivation

VLA models (e.g., OpenVLA) integrate vision and language for end-to-end action generation, but their substantial computational overhead is a primary bottleneck for real-time robot control. The specific issues are as follows:

**Severe temporal redundancy in visual tokens**: In closed-loop robotic manipulation, the majority of regions (background, stationary objects, etc.) in consecutive observation frames change very little, yet VLA models recompute all visual tokens from scratch at every timestep, resulting in massive redundant computation.

**Existing acceleration methods are ill-suited to VLA characteristics**: General-purpose approaches such as model compression, quantization, and early exit require architectural modifications or retraining and lack designs tailored to the temporal nature of VLA. VLM acceleration methods such as FastV and SparseVLM prune or merge tokens within a single frame, disrupting spatial fidelity and performing poorly on VLA tasks that demand high manipulation precision.

**Naïve reuse causes severe performance degradation**: Indiscriminately reusing all visually static tokens seriously harms performance (success rate drops from 84.4% to 74.2%), because certain regions that change little visually but are semantically critical (e.g., near the gripper, target objects) require fresh computation at every step.

The key insight of VLA-Cache is to exploit **temporal continuity** inherent in VLA tasks by caching and reusing KV representations of static tokens across frames, while using decoder attention scores to identify and protect **task-relevant tokens**, and adaptively adjusting the reuse ratio per layer to optimize the accuracy–efficiency trade-off.

## Method

### Overall Architecture

VLA-Cache consists of two core steps: (a) **Dynamic Token Selection**—identifying static tokens across frames and filtering out task-relevant tokens; and (b) **Adaptive Token Caching**—dynamically adjusting the reuse ratio per layer based on attention distributions. The entire method requires no architectural modification or retraining and is plugged directly into the VLA inference pipeline as a drop-in module.

### Key Designs

1. **Static Token Selection**: The input image is divided into $N \times N$ patches, and the cosine similarity between corresponding patches in the current and previous frames is computed: $\text{Sim}(\mathbf{P}_t^{i,j}, \mathbf{P}_{t-1}^{i,j}) = \frac{\mathbf{P}_t^{i,j} \cdot \mathbf{P}_{t-1}^{i,j}}{\|\mathbf{P}_t^{i,j}\|_2 \cdot \|\mathbf{P}_{t-1}^{i,j}\|_2}$. Patches whose similarity exceeds threshold $\tau$ (default 0.996) are marked as static, then further filtered via Top-$k$ (default 100) to retain the most stable tokens. This step operates in pixel space with negligible overhead ($\mathcal{O}(H^2)$).

2. **Task-Relevant Token Filtering**: The text-to-vision cross-attention matrix $\mathbf{A}_{\text{vis-text}}^l$ is extracted from the language decoder and averaged across multiple layers and heads to obtain a task-relevance score $\mathbf{S}_{\text{task-relevance}}$ for each visual token. Tokens whose scores exceed threshold $\tau_{\text{task}}$ (default 0.5) are marked as task-relevant and removed from the reusable set: $\mathcal{P}_{\text{reuse}} = \mathcal{P}_{\text{static}} \setminus \mathcal{P}_{\text{task-relevant}}$. This ensures that critical regions such as the gripper and target object always use up-to-date features. This mechanism recovers the success rate from 74.2% to 82.6%.

3. **Layer-Adaptive Token Reuse**: Attention distributions vary substantially across decoder layers—shallow layers exhibit diffuse attention while deep layers exhibit concentrated attention. The paper proposes an attention-entropy-based adaptive strategy: the entropy ratio between adjacent layers is computed as $R^l = (\mathcal{E}^{l-1} - \mathcal{E}^l) / \mathcal{E}^{l-1}$, and the per-layer reuse ratio is set as $\alpha^l = \min(k \sum_{j=1}^l R^j, 1)$. Layers with larger cumulative entropy drop (i.e., more concentrated attention) are allowed to reuse more tokens. This further improves the success rate to 83.8%.

### Loss & Training

VLA-Cache is a **fully training-free** method with no loss function design or additional training. It directly modifies the forward inference pass of the VLA decoder: for tokens marked as reusable, the corresponding Key and Value vectors are read directly from the previous frame's KV cache; for tokens requiring recomputation, the forward pass proceeds normally and the KV cache is updated. Due to the permutation invariance of the Transformer, this partial update does not affect the correctness of the attention computation.

## Key Experimental Results

### Main Results

**LIBERO Benchmark — OpenVLA**

| Method | Spatial | Object | Goal | Long | Mean | FLOPs(T)↓ | Latency(ms)↓ | Ctrl Freq(Hz)↑ |
|--------|---------|--------|------|------|------|-----------|--------------|----------------|
| OpenVLA | 84.4% | 86.6% | 75.6% | 53.2% | 75.0% | 1.864 | 51.91 | 4.23 |
| +SparseVLM | 79.8% | 67.0% | 72.6% | 39.4% | 64.7% | 1.407 | 83.39 | 3.72 |
| +FastV | 83.4% | 84.0% | 74.2% | 51.6% | 73.3% | 1.864 | 53.28 | 4.19 |
| **+VLA-Cache** | **83.8%** | **85.8%** | **76.4%** | **52.8%** | **74.7%** | **1.355** | **31.83** | **4.59** |

**OpenVLA-OFT (High-Frequency VLA Architecture)**

| Method | Mean SR | FLOPs(T) | Latency(ms) | Ctrl Freq(Hz) |
|--------|---------|----------|-------------|---------------|
| OpenVLA-OFT | 96.8% | 4.013 | 79.05 | 65.10 |
| +VLA-Cache | **97.4%** | **3.097** | **62.59** | **78.98** |

**Real Robot (Kinova Jaco2)**

| Method | PickPot | PlaceCube | PutSausage | WipeTable | Mean | Latency(ms) |
|--------|---------|-----------|------------|-----------|------|-------------|
| OpenVLA | 95.0% | 83.3% | 80.0% | 70.0% | 82.1% | 64.16 |
| +VLA-Cache | 90.0% | **90.0%** | **85.0%** | **73.3%** | **84.6%** | **51.85** |

### Ablation Study

| Token Selection Strategy | SR (%) | FLOPs↓ | Latency(ms)↓ | Note |
|--------------------------|--------|--------|--------------|------|
| Baseline OpenVLA | 84.4 | 1.888 | 52.37 | No caching |
| +Static token reuse | 74.2 | – | 31.03 | Naïve reuse, severe degradation |
| +Task-relevant filtering | 82.6 | – | 31.03 | Attention filtering recovers performance |
| +Layer-adaptive | **83.8** | – | 32.22 | Final version, best accuracy |

### Key Findings

- **SparseVLM and FastV fail on VLA**: Because VLA action outputs are very short (only 7 tokens), the advantages of VLM acceleration methods designed for long-sequence decoding do not apply; SparseVLM even increases latency.
- **VLA-Cache is compatible with high-frequency architectures**: On OpenVLA-OFT, it further improves control frequency by approximately 14 Hz (65→79 Hz), demonstrating that VLA-Cache directly accelerates the decoding bottleneck.
- **Robust under dynamic backgrounds**: When background motion perturbations are introduced, VLA-Cache maintains success rate while reducing FLOPs by 42% and latency by 35%.

## Highlights & Insights

- **Training-free, plug-and-play**: No retraining or architectural modification is required; the method directly accelerates inference of existing VLA models, resulting in an extremely low deployment barrier.
- **Exploits VLA-specific temporal redundancy**: Unlike general VLM acceleration, VLA-Cache specifically leverages the unique property that consecutive frames in robotic manipulation are highly similar.
- **Attention scores as a proxy for task relevance**: The method cleverly uses the decoder's own attention patterns to determine which tokens cannot be reused, without requiring any additional detector or segmentation model.

## Limitations & Future Work

- In highly dynamic environments (extensive motion, drastic background changes), the number of reusable tokens decreases, reducing the speedup benefit.
- Validation is currently limited to VLA architectures with a LLaMA2 backbone (OpenVLA, CogAct, OpenVLA-OFT); applicability to other backbones such as Gemma2 remains unverified.
- The method is not directly applicable to pure diffusion policy models that lack a VLM backbone.
- The hyperparameter $k$ in the layer-adaptive strategy requires tuning based on model architecture.

## Related Work & Insights

- The approach is complementary to methods requiring retraining, such as DeeR-VLA (dynamic depth control) and TinyVLA (model distillation).
- VLA-Cache is compatible with high-frequency VLA architectures such as π0-FAST and HiRT, accelerating the decoding bottleneck in these systems.
- Inspiration: Future work could extend the cross-frame token caching strategy to multi-view inputs or 3D voxel representations, or explore learning-based dynamic threshold adjustment.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The idea of cross-frame token caching exploiting VLA temporal redundancy is concise and effective; the attention-based filtering mechanism is elegantly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three VLA models × two simulation platforms + real robot experiments, comprehensive ablations, sensitivity analysis, and dynamic background testing.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The method is presented in a progressive, step-by-step manner with tabular validation at each stage; the logic is clear.
- **Value**: ⭐⭐⭐⭐ Addresses a practical pain point in VLA deployment; the training-free nature makes adoption by the community straightforward.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] T-Rex: Task-Adaptive Spatial Representation Extraction for Robotic Manipulation with VLMs](t-rex_task-adaptive_spatial_representation_extraction_for_robotic_manipulation_w.md)
- [\[NeurIPS 2025\] PrefixKV: Adaptive Prefix KV Cache is What Vision Instruction-Following Models Need for Efficient Generation](prefixkv_adaptive_prefix_kv_cache_is_what_vision_instruction.md)
- [\[NeurIPS 2025\] ForceVLA: Enhancing VLA Models with a Force-aware MoE for Contact-rich Manipulation](forcevla_enhancing_vla_models_with_a_force-aware_moe_for_contact-rich_manipulati.md)
- [\[NeurIPS 2025\] SCOPE: Saliency-Coverage Oriented Token Pruning for Efficient Multimodal LLMs](scope_saliency-coverage_oriented_token_pruning_for_efficient_multimodel_llms.md)
- [\[NeurIPS 2025\] BridgeVLA: Input-Output Alignment for Efficient 3D Manipulation Learning with Vision-Language Models](bridgevla_input-output_alignment_for_efficient_3d_manipulation_learning_with_vis.md)

<!-- RELATED:END -->
