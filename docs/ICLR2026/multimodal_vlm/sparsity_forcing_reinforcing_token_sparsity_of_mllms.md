---
title: >-
  [Paper Note] Sparsity Forcing: Reinforcing Token Sparsity of MLLMs
description: >-
  [ICLR 2026][Multimodal VLM][token sparsity] This paper proposes Sparsity Forcing — a GRPO-based RL post-training framework that treats a sparse-attention MLLM as the policy model and the original MLLM as the reference mo…
tags:
  - "ICLR 2026"
  - "Multimodal VLM"
  - "token sparsity"
  - "RL post-training"
  - "GRPO"
  - "joint efficiency-performance reward"
  - "multi-budget exploration"
date: 2026-05-08
content_hash: cca3d26a8f5ba914
---

# Sparsity Forcing: Reinforcing Token Sparsity of MLLMs

**Conference**: ICLR 2026
**arXiv**: [2504.18579](https://arxiv.org/abs/2504.18579)  
**Area**: Multimodal VLM
**Keywords**: token sparsity, RL post-training, GRPO, joint efficiency-performance reward, multi-budget exploration

## TL;DR
This paper proposes Sparsity Forcing — a GRPO-based RL post-training framework that treats a sparse-attention MLLM as the policy model and the original MLLM as the reference model. Through multi-budget rollouts exploring different token retention thresholds $p$, and using a joint reward combining efficiency (token reduction rate) and performance (answer correctness) for within-group contrastive optimization, the method improves the token reduction rate of Qwen2/2.5-VL from 20% to 75% with minimal accuracy loss, achieving 3× memory reduction and 3.3× decoding speedup.

## Background & Motivation

**MLLM Inference Bottleneck**: When processing high-resolution images or long videos, visual encoders generate massive numbers of visual tokens, severely limiting generation efficiency (e.g., video inputs with 16k+ tokens).

**Natural Sparsity Exploitation Has Reached Its Limit**: Methods such as FastV and ZipVL leverage inherent sparsity in attention maps to prune redundant tokens, but can only safely reduce tokens by approximately 50%; further compression (e.g., retaining 20% or 10%) causes sharp accuracy degradation.

**Limitations of Trainable Sparse Attention**: Methods such as MOBA and NSA predefine rigid sparsity patterns, ignoring the dynamic nature of inputs and layers. They also require training from scratch, making them impractical for MLLM post-training scenarios.

**Proxy Objective Problem in Attention Sharpening Regularization**: Regularizers such as $L_\infty$/entropy minimization optimize proxy objectives targeting attention distribution sharpness rather than directly controlling token budgets. The learned sharpness cannot reliably translate into end-to-end token savings.

**Train-Inference Mismatch in SFT**: Existing SFT methods apply sparsity under teacher forcing on ground-truth tokens rather than generated outputs, creating inconsistency with autoregressive decoding at inference time and resulting in limited practical efficiency gains.

**Core Motivation**: An inference-aligned post-training method is needed that directly optimizes end-to-end efficiency-performance objectives rather than proxies, enabling the model to actively learn which tokens can be safely discarded.

## Method

### Key Design 1: Dual Policy-Reference Model Architecture

- **Function**: The MLLM equipped with top-$p$ sparse attention (e.g., Qwen2-VL + ZipVL) serves as the policy model $\pi_\theta$, while the original standard-attention MLLM (with frozen parameters) serves as the reference model $\pi_{\text{ref}}$.
- **Mechanism**: The policy model performs sparse token selection and KV cache pruning during decoding, while the reference model anchors training via KL divergence $\mathbb{D}_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$ to prevent excessive deviation from original capabilities.
- **Design Motivation**: The reference model stabilizes learning and constrains accuracy loss, maintaining task fidelity even under high sparsity rates. Top-$p$ sparse attention independently determines the number of retained tokens per layer:
$$b = \min\{p \in \mathbb{Z} \mid \sum_{j=1}^{p} a_{\text{sorted}(j)} \geq p \times \ell\}$$
where $a_j = \sum_{c=1}^{\ell} \mathbf{A}_{c,j}$ is the cumulative attention score and $\ell$ is the sequence length.

### Key Design 2: Multi-Budget Rollout Exploration

- **Function**: For each vision-language query, $N$ independent rollouts are performed using $N$ different attention retention thresholds $p_n \sim \mathcal{U}(0,1)$, generating $N$ answers $\{\mathbf{o}_1, \dots, \mathbf{o}_N\}$ and corresponding token ratios $\{\tau_1, \dots, \tau_N\}$.
- **Mechanism**: Progressive budget sweep — different values of $p$ constitute a gradient test from sparse to dense: small $p$ retains few tokens to test whether correct answers are still achievable; large $p$ retains more tokens as a correctness safety net. The training range is set to $p \in [0.94, 0.975]$ with a step size of 0.005.
- **Design Motivation**: This avoids manually defining positive/negative sample pairs (a pain point of DPO), as multi-budget rollouts naturally produce contrastive signals — rollouts that are both correct and efficient receive positive advantage, while incorrect or inefficient ones receive negative advantage. As training progresses, the minimum correct budget changes dynamically, and rollouts adapt automatically.

### Key Design 3: Joint Efficiency-Performance Reward and GRPO Update

- **Function**: A joint reward is computed for each rollout, and the policy is updated via GRPO's within-group normalized advantage.
- **Mechanism**: Performance reward $r_{\text{per}} \in \{0, 1\}$ (whether the answer is correct) + efficiency reward $r_{\text{eff}} = 1 - \tau_i$ (token reduction rate). A group-level indicator $C$ is introduced:
$$C = \mathbb{1}\{\exists j: \text{Correct}(\mathbf{o}_j) = 1\}$$
The efficiency reward is only included when at least one rollout in the group is correct:
$$r_i = r_{\text{per},i} + C \cdot r_{\text{eff},i}$$
Advantages are normalized within the group: $A_i = (r_i - \text{mean}) / \text{std}$. The final policy update uses GRPO's clipped surrogate objective:
$$\mathcal{J}(\theta) = \mathbb{E}\left[\min\left(\frac{\pi_\theta(\mathbf{o}_n|\mathbf{x})}{\pi_{\theta_{\text{old}}}(\mathbf{o}_n|\mathbf{x})} A_i,\; \kappa(\cdot) A_i\right) - \beta \mathbb{D}_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})\right]$$
- **Design Motivation**: (1) The group-level indicator $C$ prevents the efficiency signal from driving extreme sparsity when all rollouts fail, which would otherwise degenerate into empty outputs; (2) the joint reward makes efficiency and performance end-to-end objectives rather than proxies; (3) GRPO's on-policy nature ensures positive/negative contrasts are dynamically updated during training, avoiding the stale preference pairs problem in DPO.

### Key Design 4: Inference Consistency

- **Function**: Training and inference use an identical sparse attention pipeline — the same token pruning strategy and KV cache management.
- **Mechanism**: At inference time, $p$ is fixed at $0.975$ (the upper bound of the training range) to ensure accuracy while leveraging the efficiency gains learned during training. The model has learned to produce sparser attention distributions even at $p=0.975$.
- **Design Motivation**: SFT methods use teacher forcing during training and autoregressive decoding at inference, creating a pipeline mismatch that leads to unreliable efficiency gains. The RL approach uses autoregressive rollouts during training, making it deployment-aligned.

## Key Experimental Results

### Table 1: Image Benchmark Comparison (7 Tasks)

| Model | Method | Token Ratio↓ | MME | MMBench | MMStar | ChartQA | TextVQA | OCRBench | MMMU-Pro | Avg. |
|-------|--------|-------------|-----|---------|--------|---------|---------|----------|----------|------|
| Qwen2.5-VL-7B | Full | 100% | 2303 | 83.9 | 62.2 | 84.0 | 82.9 | 845 | 36.7 | 73.8 |
| | FastV | 52.1% | 2115 | 81.9 | 61.2 | 80.2 | 79.6 | 760 | 34.5 | 69.9 |
| | ZipVL | 79.5% | 2290 | 83.9 | 60.4 | 82.0 | 82.6 | 837 | 36.2 | 72.9 |
| | **Sparsity Forcing** | **24.7%** | **2286** | **84.1** | **62.5** | **83.1** | **82.6** | **847** | **36.7** | **73.6** |

### Table 2: Comparison with Enhanced Sparsity Methods (Qwen2.5-VL-7B)

| Method | Type | Token Ratio↓ | MME | MMStar | ChartQA | VideoMME | Avg. |
|--------|------|-------------|-----|--------|---------|----------|------|
| Full | - | 100% | 2303 | 62.2 | 84.0 | 64.5 | 73.2 |
| MOBA | Trainable sparse attention | 25% | 1906 | 58.6 | 77.3 | 62.6 | 66.6 |
| Sharpness Loss | Attention sharpening regularization | 25% | 1965 | 59.6 | 77.0 | 63.7 | 67.6 |
| ZipVL (post-training) | Sparse attention fine-tuning | 61.7% | 2264 | 62.0 | 78.9 | 64.2 | 71.5 |
| **Sparsity Forcing** | **RL post-training** | **26.4%** | **2286** | **62.5** | **83.1** | **64.0** | **72.8** |

### Table 3: Ablation on Sparse Attention Type (Qwen2.5-VL-7B)

| Sparse Attention | Token Ratio | MME | VideoMME |
|-----------------|-------------|-----|----------|
| Top-$k$ | 25% | 2160 | 60.2 |
| Threshold | 37.8% | 2218 | 61.6 |
| **Top-$p$** | **24.1%** | **2286** | **64.0** |

## Key Findings

1. **Sparsity Can Be Reinforced 3.75× via RL**: ZipVL's inherent sparsity can only safely reduce tokens from 100% to ~80% (a 20% reduction), whereas Sparsity Forcing enables safe reduction to ~25% (a 75% reduction) after training, demonstrating that the sparsity potential of MLLMs is far from fully exploited.

2. **Group-Level Indicator $C$ Is Critical for Preventing Collapse**: Without $C$, efficiency signals from all-incorrect groups still drive extreme sparsity, causing the model to degenerate into outputting empty answers. $C$ ensures efficiency is rewarded only when "at least one rollout is correct."

3. **Top-$p$ Outperforms Top-$k$ and Threshold**: Top-$p$ is an online policy that adaptively adjusts the number of retained tokens based on each layer's attention distribution, whereas Top-$k$ and Threshold are offline policies that do not adapt to input variation, resulting in 3–4 points lower accuracy at the same token ratio.

4. **Per-Layer Sparsity Varies Substantially**: After training, token retention rates differ significantly across layers — shallow layers retain more tokens (requiring global context), while deeper layers retain fewer (having already focused on key tokens), validating the necessity of a dynamic per-layer strategy.

5. **Sparsity Adaptively Increases with Sequence Length**: As input length increases from 4k to 20k tokens, the retention ratio decreases from ~35% to ~20% with nearly unchanged accuracy, indicating that longer sequences contain more redundancy and that the method scales naturally.

## Highlights & Insights

- **Paradigm Shift from "Exploitation" to "Reinforcement"**: Prior methods passively exploit inherent sparsity; Sparsity Forcing actively trains for greater sparsity. The essential distinction lies in training the model to reorganize its attention distribution.
- **Inference-Consistent Design**: The training loop fully mirrors the inference pipeline (autoregressive + sparse KV cache), eliminating the teacher forcing gap present in SFT.
- **Practical Deployment Value**: 3× memory reduction + 3.3× decoding speedup transforms long-video processing from infeasible to feasible, directly impacting MLLM deployment.
- **Lightweight Post-Training**: Training Qwen2.5-VL-7B requires 883 GPU hours (8×A100), without the need for training from scratch, adding efficiency on top of already strong models.

## Limitations & Future Work

1. **Non-Trivial Training Cost**: While 883/164 GPU hours is less than training from scratch, multi-rollout post-training itself increases training cost (group size of 8 means 8 inference passes per sample).
2. **Validation Limited to QwenVL and LLaVA Families**: Generalizability to other architectures (e.g., InternVL, Gemini) has not been verified; attention sparsity characteristics may differ across MLLMs.
3. **Requires Sparse Attention Framework Support at Inference**: The method depends on sparse attention implementations such as ZipVL, which are not natively supported by all inference engines, requiring additional engineering for deployment.
4. **Primarily Targets Visual Tokens**: The approach mainly prunes visual tokens, providing limited gains for pure-text or text-heavy tasks; OCR-type tasks inherently have less sparsity headroom.

## Related Work & Insights

### vs. ZipVL (He et al., 2024)
ZipVL is a training-free sparse attention method that exploits inherent sparsity via top-$p$ threshold-based dynamic token selection, but can only safely reduce tokens by ~20%. Sparsity Forcing builds upon ZipVL by reinforcing sparsity through RL post-training, upgrading ZipVL from "exploitation" to "reinforcement" — within the same framework, token reduction improves from 20% to 75%. The essential distinction is that ZipVL does not modify model weights, whereas Sparsity Forcing alters the model's attention distribution to be inherently sparser.

### vs. MOBA (Lu et al., 2025)
MOBA is a trainable sparse attention method featuring block-wise attention probing and a MoE-inspired approach with predefined sparsity patterns, requiring training from scratch. In the post-training comparison (Table 3), MOBA achieves an MME score of only 1906 at 25% token ratio (vs. 2286 for Sparsity Forcing), a substantial performance gap of 6.2 average points. The reason is that MOBA's rigid patterns ignore input and layer dynamics, making it unsuitable for post-training fine-tuning of existing MLLMs.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First application of GRPO to MLLM sparsity reinforcement; the combination of joint reward, multi-budget exploration, and inference consistency is uniquely designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 13 benchmarks (7 image + 6 video) + 4 models + detailed ablations (sparsity mechanism / rollout range / group size / hallucination robustness).
- **Writing Quality**: ⭐⭐⭐⭐⭐ Problem analysis is thorough; the narrative arc from "exploitation" to "reinforcement" is clear; figures are effective.
- **Value**: ⭐⭐⭐⭐⭐ 3× memory + 3.3× speed directly applicable to MLLM deployment acceleration; the post-training approach lowers the barrier to adoption.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] VLM-Pruner: Buffering for Spatial Sparsity in an Efficient VLM Centrifugal Token Pruning Paradigm](../../CVPR2026/multimodal_vlm/vlm-pruner_buffering_for_spatial_sparsity_in_an_efficient_vlm_centrifugal_token_.md)
- [\[ICLR 2026\] SophiaVL-R1: Reinforcing MLLMs Reasoning with Thinking Reward](sophiavl-r1_reinforcing_mllms_reasoning_with_thinking_reward.md)
- [\[ICCV 2025\] SparseMM: Head Sparsity Emerges from Visual Concept Responses in MLLMs](../../ICCV2025/multimodal_vlm/sparsemm_head_sparsity_emerges_from_visual_concept_responses_in_mllms.md)
- [\[ICCV 2025\] SparseVILA: Decoupling Visual Sparsity for Efficient VLM Inference](../../ICCV2025/multimodal_vlm/sparsevila_decoupling_visual_sparsity_for_efficient_vlm_inference.md)
- [\[NeurIPS 2025\] Video-R1: Reinforcing Video Reasoning in MLLMs](../../NeurIPS2025/multimodal_vlm/video-r1_reinforcing_video_reasoning_in_mllms.md)

</div>

<!-- RELATED:END -->
