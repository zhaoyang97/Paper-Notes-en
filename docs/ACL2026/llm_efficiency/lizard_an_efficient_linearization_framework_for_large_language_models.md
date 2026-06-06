---
title: >-
  [Paper Note] Lizard: An Efficient Linearization Framework for Large Language Models
description: >-
  [ACL 2026][LLM Efficiency][Linearization] Lizard replaces the softmax attention of pretrained Transformers with a hybrid subquadratic attention consisting of "Gated Linear Attention (global compression) + Anchor Window A…
tags:
  - "ACL 2026"
  - "LLM Efficiency"
  - "Linearization"
  - "Gated Linear Attention"
  - "Long-context Extrapolation"
  - "teacher-student"
  - "tensor core"
date: 2026-05-08
content_hash: e450d4451010d44f
---

# Lizard: An Efficient Linearization Framework for Large Language Models

**Conference**: ACL 2026  
**arXiv**: [2507.09025](https://arxiv.org/abs/2507.09025)  
**Code**: TBD (Adobe Research × University of Oregon)  
**Area**: LLM Efficiency / Linear Attention / Model Compression  
**Keywords**: Linearization, Gated Linear Attention, Long-context Extrapolation, teacher-student, tensor core

## TL;DR
Lizard replaces the softmax attention of pretrained Transformers with a hybrid subquadratic attention consisting of "Gated Linear Attention (global compression) + Anchor Window Attention (local precision) + learnable gate instead of RoPE." With only 0.04B tokens of distillation, it outperforms existing linearization methods by 9.4–24.5 points on 5-shot MMLU and incorporates a tensor-core-friendly training algorithm that increases throughput by 32%.

## Background & Motivation
**Background**: Approaches to solving the $O(L^2)$ complexity of softmax attention generally follow two paths: (a) pretraining linear/State Space Models (SSMs) from scratch (e.g., Mamba, RWKV, Griffin), or (b) *linearization* of existing Transformers (replacing attention modules with subquadratic forms), represented by LoLCATs, Liger, SUPRA, and Mamba2-LLaMA.

**Limitations of Prior Work**: Pretraining from scratch requires a budget of trillions of tokens, and systematically suffers performance drops in in-context learning and retrieval tasks (Transformers outperform Mamba/Mamba-2 by 15 points on 5-shot MMLU in equivalent settings). While the linearization path is less expensive, it face serious issues: LoLCATs drops 13.8 points and Liger drops 21.9 points on MMLU compared to the teacher. Furthermore, by inheriting RoPE, these models' length extrapolation capabilities remain restricted.

**Key Challenge**: Existing linearization methods strictly adhere to the rule that "the teacher architecture cannot be modified." This results in two fundamental design flaws that cannot be bypassed: (1) lack of adaptive memory control (LoLCATs has no gate, while Liger fixes the gate to parameter-free pooling, creating an information bottleneck); (2) fixed RoPE positional embeddings sever the inherent extrapolation capability of recurrent forms.

**Goal**: To introduce a small number of learnable modules on the student side (feature map $\phi$, gate $W_\gamma$, meta-memory tokens $t$) in exchange for near-lossless teacher performance, genuine long-context extrapolation, and a training pipeline accelerated by modern tensor cores.

**Key Insight**: It is observed that the decay pattern of gated recurrent models like GLA can inherently encode relative positions, meaning RoPE can be discarded. Furthermore, while global GLA is inferior to sliding window attention for precise local attention, the two can be placed in parallel to manage different scopes.

**Core Idea**: A hybrid of "GLA (global compression + learnable gate replacing RoPE) + local Anchor Window Attention (including meta-memory tokens to patch attention sinks)," combined with numerical re-parameterization of GLA to enable execution on bf16 + tensor cores.

## Method

### Overall Architecture
Lizard replaces the standard softmax attention of each layer with a Lizard Attention module:
- **Global Branch**: Gated Linear Attention (GLA), providing $O(Ld^2)$ complexity, constant memory inference, and data-dependent decay (replacing RoPE).
- **Local Branch**: Anchor Window Attention, performing precise softmax attention within a fixed window and incorporating several learnable meta-memory tokens $t$ as "attention sinks."
- The outputs of the two branches are merged using specific weights.

The training is divided into two stages:
1. **Stage 1 — Attention Approximation**: All other weights are frozen, training only $\phi$, $W_\gamma$, and $t$. The objective is for Lizard Attention outputs to approximate the teacher's RoPE-softmax attention outputs (MSE distillation).
2. **Stage 2 — End-to-end Fine-tuning**: The entire model is unfrozen and fine-tuned on a small volume of tokens (0.04B) using a language modeling loss to ensure synergy between the new architecture and the rest of the model (MLP/embedding).

### Key Designs

1. **Gated Linear Attention replaces RoPE for Positional Encoding**:
    - **Function**: Uses a learnable, token-dependent gate $\gamma_i$ to control the decay rhythm of the hidden state $h_i = \gamma_i \odot h_{i-1} + \phi(k_i) v_i^\top$. This transforms "positional information" from hard-coded (RoPE rotation angles) into data-driven forgetting patterns.
    - **Mechanism**: The recurrence of traditional linear attention is modified to $\mathbf{h}_i = \gamma_i \odot \mathbf{h}_{i-1} + \phi(\mathbf{k}_i)\mathbf{v}_i^\top$, with output $\mathbf{y}_i = \phi(\mathbf{q}_i)^\top \mathbf{h}_i$; $\gamma_i$ is calculated by $W_\gamma$, enabling controllable relative position perception and extrapolation. The RoPE module from the teacher is discarded.
    - **Design Motivation**: (a) RoPE is the primary culprit for attention length extrapolation failures; (b) LoLCATs lacks a gate resulting in no memory control, while Liger uses parameter-free pooling causing an information bottleneck. Learnable gates solve both issues.

2. **Anchor Window Attention (Local Precision + Meta-Memory Tokens)**:
    - **Function**: Retains precise softmax attention within the most recent $W$ tokens and prepends several learnable meta-memory tokens $t_1, \dots, t_M$ to the key/value sequences, allowing queries to always attend to these "anchors."
    - **Mechanism**: $\text{Attn}_{\text{local}}(\mathbf{q}_i) = \text{softmax}([\mathbf{q}_i^\top T;\ \mathbf{q}_i^\top \mathbf{k}_{i-W:i}]) \cdot [V_T;\ \mathbf{v}_{i-W:i}]$, where $T = [t_1, \dots, t_M]$ are learnable global anchors.
    - **Design Motivation**: Pure linear attention inevitably struggles with fine-grained recall tasks like "needle-in-a-haystack." Meta-memory tokens act similarly to attention sinks (explicitly modeling the "first few tokens are always attended to" phenomenon observed in StreamingLLM), stabilizing long-sequence inference.

3. **Hardware-Aware Numerical Re-parameterization (Solving GLA Instability at Low Precision)**:
    - **Function**: Rewrites the GLA decay product $\prod \gamma$ as an accumulation in log-space $\log \mathbf{h}_i = \log \mathbf{h}_{i-1} + \log \gamma_i + \dots$ to avoid exponential underflow/overflow in bf16/fp16.
    - **Mechanism**: Re-parameterization maintains intermediate states within the precision range supported by tensor cores, allowing training to utilize bf16 + tensor cores, which increases throughput by approximately 32%.
    - **Design Motivation**: Existing GLA training must revert to fp32 for numerical stability, losing tensor core acceleration. This improvement ensures Lizard is not only accurate but also fast to train, determining its feasibility for large-scale deployment.

### Loss & Training
- **Stage 1**: MSE distillation $\mathcal{L}_1 = \sum_\ell \|\mathbf{y}_\ell^{\text{Lizard}} - \mathbf{y}_\ell^{\text{teacher}}\|_2^2$, training only the newly added modules ($\phi$, $W_\gamma$, $t$).
- **Stage 2**: Standard next-token cross-entropy for full-model fine-tuning.
- Total training budget is 0.04B tokens (500× less than the 20B tokens for Mamba2-LLaMA-3-8B), making it feasible on consumer-grade hardware.

## Key Experimental Results

### Main Results
Standard short-context benchmarks from LM-eval-harness (excerpt from Table 1, all figures are acc/acc_norm, tokens in B):

| Model | Training Tokens (B) | MMLU (5-shot) | ARC-c | Hella. | Avg |
| :--- | :--- | :--- | :--- | :--- | :--- |
| LLaMA-3-8B (teacher) | 15000 | **66.6** | 53.3 | 79.1 | 73.1 |
| Mamba-7B (from scratch) | 1200 | 33.3 | 46.7 | 77.9 | 71.0 |
| Mistral-7B-LoLCATs | 0.04 | 51.4 | 54.9 | 80.7 | 74.5 |
| LLaMA-3-8B-LoLCATs | 0.04 | 52.8 | 54.9 | 79.7 | 74.2 |
| Liger-GLA-Llama-3-8B | 0.02 | 43.4 | 52.5 | 76.3 | 72.4 |
| Mamba2-LLaMA-3-8B | 20 | 43.2 | 48.0 | 70.8 | 65.6 |
| **Mistral-7B-Lizard (ours)** | 0.04 | **60.8** | 55.8 | 79.8 | 74.5 |
| **LLaMA-3-8B-Lizard (ours)** | 0.04 | **61.2** | 56.7 | 79.3 | **74.6** |

Lizard improves 5-shot MMLU from LoLCATs' 52.8 to 61.2 (+8.4), only 5.4 points behind the teacher's 66.6; it outperforms Liger by +17.8 points. When a hybrid approach retaining 50% softmax attention layers is used, it reaches 65.1, nearly equal to the teacher's 66.6.

### Ablation Study (Key architectural comparisons from the authors, measured by 5-shot MMLU)

| Configuration | MMLU (5-shot) | Description |
| :--- | :--- | :--- |
| Full Lizard | 61.2 | Full GLA + Anchor Window + meta-memory + re-parameterization |
| w/o learnable gate (degrades to LoLCATs-style) | ~52.8 | Removing adaptive memory control drops performance to LoLCATs level |
| w/o meta-memory tokens | Significant Drop | Degradation in long-context needle-in-haystack tasks |
| w/o tensor-core re-parameterization | Around 61 | Precision remains, but training throughput drops ~32% |
| 50% softmax / 50% Lizard hybrid | 65.1 | Close to teacher's 66.6 |

### Key Findings
- The "learnable gate" is the fundamental reason for the 9–18 point lead Lizard holds over LoLCATs/Liger—memory control is more important than feature map selection.
- Data-driven gates replacing RoPE allow Lizard to extrapolate to unseen lengths in the *Unbounded* category, which Bounded methods like LoLCATs/Liger cannot achieve.
- A 0.04B token budget is sufficient to recover 92% of the teacher's MMLU, implying the marginal cost of converting existing Large Language Models to subquadratic is negligible.
- The 32% throughput increase from hardware-aware improvements does not affect precision; it is a pure engineering bonus, making reproduction very friendly for the open-source community.

## Highlights & Insights
- The observation that "GLA decay is inherently a positional encoding" allows for the direct removal of RoPE—a rare "less is more" design. This can be migrated to other recurrent architectures requiring extrapolation.
- Meta-memory tokens are effectively a learnable, explicit version of the "attention sink" phenomenon in StreamingLLM, elegantly turning an empirical observation into a module design.
- Distilling a subquadratic model close to teacher performance with 0.04B tokens essentially demonstrates that training Mamba/RWKV from scratch is not cost-effective for short-context QA; linearization with appropriate architectural patches is the more economical direction.
- Treating numerical stability as a first-class citizen (occupying a full section) is a scarce and valuable engineering contribution to GLA-related research.

## Limitations & Future Work
- Experiments were primarily conducted at the 7B–8B scale; whether 70B+ teachers can still yield similar performance with 0.04B tokens remains unverified.
- The main table focuses on standard short-context benchmarks; while "associative recall" advantages are claimed via diagrams, complete needle-in-haystack / long-context QA numerical comparisons are missing.
- Using a 50% hybrid almost matches the teacher, which conversely suggests that pure-Lizard has not yet fully recovered in the most difficult reasoning (high MMLU sub-items); the capacity upper bound for the "linear + local window" approach may reside there.
- Self-reflection: Linking the GLA gate with a MoE router might further improve memory efficiency, which is a direction worth following.

## Related Work & Insights
- **vs LoLCATs**: Both perform linearization, but LoLCATs lacks a gate and forcibly retains the teacher architecture; Lizard adds learnable gates, leading to +8.4 MMLU.
- **vs Liger**: Liger has a gate but parameterizes it as fixed pooling, creating an information bottleneck; Lizard uses true learnable $W_\gamma$, leading to +17.8 MMLU.
- **vs Mamba2-LLaMA-3-8B**: Uses 20B tokens for cross-architecture distillation, but because it inherits RoPE, it cannot extrapolate and MMLU only reaches 43.2; Lizard uses 1/500th of the tokens and scores 18 points higher on MMLU.
- **vs SUPRA**: An early linearization scheme without learnable gates and with attention equivalence issues; Lizard outperforms it in all aspects.

## Rating
- Novelty: ⭐⭐⭐⭐ Engineering the observation that "GLA decay ≈ positional encoding" while packaging a tensor-core-friendly training algorithm represents strong combinatorial innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ The main table covers 8 baselines and two teacher families; long-context and ablation sections are relatively concise, relying primarily on textual description.
- Writing Quality: ⭐⭐⭐⭐ Motivation and design trade-offs are clearly explained, and formula derivations are compact.
- Value: ⭐⭐⭐⭐⭐ Distilling near-teacher performance with 0.04B tokens plus a 32% training speedup is highly attractive for actual deployment of long-context LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Are Large Language Models Economically Viable for Industry Deployment?](are_large_language_models_economically_viable_for_industry_deployment.md)
- [\[ACL 2026\] Tandem: Riding Together with Large and Small Language Models for Efficient Reasoning](tandem_riding_together_with_large_and_small_language_models_for_efficient_reason.md)
- [\[ACL 2026\] CreditDecoding: Accelerating Parallel Decoding in Diffusion Large Language Models with Trace Credit](creditdecoding_accelerating_parallel_decoding_in_diffusion_large_language_models.md)
- [\[ACL 2026\] Breaking Block Boundaries: Anchor-based History-stable Decoding for Diffusion Large Language Models](breaking_block_boundaries_anchor-based_history-stable_decoding_for_diffusion_lar.md)
- [\[ICLR 2026\] DND: Boosting Large Language Models with Dynamic Nested Depth](../../ICLR2026/llm_efficiency/dnd_boosting_large_language_models_with_dynamic_nested_depth.md)

</div>

<!-- RELATED:END -->
