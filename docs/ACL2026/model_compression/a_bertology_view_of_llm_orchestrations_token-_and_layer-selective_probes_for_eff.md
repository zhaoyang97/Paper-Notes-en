---
title: >-
  [Paper Note] A BERTology View of LLM Orchestrations: Token- and Layer-Selective Probes for Efficient Single-Pass Classification
description: >-
  [ACL 2026][Model Compression][probing] The token $\times$ layer hidden state tensors of production LLMs are treated as mineable resources. A two-stage aggregation probe ("compress tokens first…
tags:
  - "ACL 2026"
  - "Model Compression"
  - "probing"
  - "hidden state reuse"
  - "safety classification"
  - "single-pass forward"
  - "BERTology"
date: 2026-05-08
content_hash: 8221aa58b5c2c65a
---

# A BERTology View of LLM Orchestrations: Token- and Layer-Selective Probes for Efficient Single-Pass Classification

**Conference**: ACL 2026  
**arXiv**: [2601.13288](https://arxiv.org/abs/2601.13288)  
**Code**: No public repository  
**Area**: Model Compression / LLM Efficiency  
**Keywords**: probing, hidden state reuse, safety classification, single-pass forward, BERTology

## TL;DR
The token $\times$ layer hidden state tensors of production LLMs are treated as mineable resources. A two-stage aggregation probe ("compress tokens first, then layers") achieves safety/sentiment classification within a single forward pass. With only 35M trainable parameters, the performance approaches that of independent guard models, eliminating the need for additional LLM calls.

## Background & Motivation
**Background**: LLM services in production environments typically utilize an orchestration architecture consisting of a "serving LLM + auxiliary classifiers (safety / policy / jailbreak / retrieval filter)". Each auxiliary model requires separate training, deployment, and an additional forward pass for every request, which multiplies memory usage and latency.

**Limitations of Prior Work**: Existing methods for reusing serving LLM computations (e.g., MULI using first-token logits, ShieldHead using the final layer hidden state, OmniGuard using a fixed layer) prescribe fixed "readout positions," selecting only one token or one layer. However, BERTology research consistently demonstrates that different Transformer layers encode different levels of abstraction (low-level syntax vs. high-level semantics). Fixing the readout position wastes signals distributed across the depth of the model.

**Key Challenge**: Information is distributed across the entire $L \times T \times d$ hidden state tensor. Methods like logit-only or single-layer probes reduce the tensor to 1D before classification, discarding critical discriminative signals during this initial compression step.

**Goal**: Reformulate moderation/NLU classification as a "representation selection problem across the entire token $\times$ layer tensor," allowing the probe to learn which tokens and layers are most discriminative.

**Key Insight**: Treat the LLM as a frozen feature extractor and "piggyback" classification during its forward pass. This allows classification and generation to share the same KV/attention computation, incurring negligible latency overhead.

**Core Idea**: "Two-stage aggregation = internal token-level condensation into layer summaries, followed by layer-level condensation into a single vector." The same family of aggregation operators (pooling / scoring gate / MHA) is inserted into both stages, with tunable parameters ranging from 3K to 35M.

## Method

### Overall Architecture
Given a frozen decoder-only LLM, an input prompt generates $L+1$ hidden state matrices $\mathbf{h}^{(l)} \in \mathbb{R}^{T \times d}$ (including the embedding layer) after $T$ tokens. The pipeline for the probe $C_\theta$ is:

1.  **Stage 1 (token agg)**: For each layer $l$, an operator $\mathcal{A}^{(l)}_\text{token}$ compresses $T \times d$ into $\mathbf{v}^{(l)} \in \mathbb{R}^d$, resulting in $L+1$ layer summaries.
2.  **Stage 2 (layer agg)**: An operator $\mathcal{A}_\text{layer}$ compresses the $L+1$ summaries into a single vector $\mathbf{v} \in \mathbb{R}^d$.
3.  **Classification Head**: $\text{logits} = \mathbf{W}_\text{out} \mathbf{v} + \mathbf{b}_\text{out}$, trained with cross-entropy while the LLM remains frozen.

During inference, classification shares the forward pass with generation. If categorized as unsafe, the orchestration layer can intercept the request before any tokens are streamed, and call the serving LLM to generate a contextual refusal (without an additional full model call).

### Key Designs

1.  **Two-stage aggregation architecture**:
    - **Function**: Decomposes the $L \times T \times d$ 3D tensor into two 1D reduction steps, preventing parameter explosion that would result from feeding a 3D tensor directly into a lightweight classifier.
    - **Mechanism**: The first step summarizes tokens within each layer; the second step summarizes across layers. Using the same operator family ensures architectural consistency. Layer-level aggregation allows the model to learn task-specific weights $\alpha_l$, serving as a learnable version of "layer selection."
    - **Design Motivation**: Separating tokens and layers preserves BERTology's intuition regarding varying abstractions while avoiding $T \cdot L \cdot d$ parameters. This design also scales to long prompts.

2.  **Scoring attention gate (lightweight weighted sum, ~100K parameters)**:
    - **Function**: Uses a learned scalar importance score to perform weighted summation of tokens/layers in both stages.
    - **Mechanism**: For each token/layer vector $\mathbf{X}[i, :]$, a gate score $s_i = \tanh(\mathbf{w}^\top \mathbf{X}[i, :])$ is calculated. Padding positions are set to $-\infty$. Weights $\alpha_i$ are obtained via softmax, and the output is $\mathbf{v} = \sum_i \alpha_i \mathbf{X}[i, :]$. Stage 1 uses independent gates per layer, while Stage 2 shares a single gate. Total parameters: $(L+2)d$.
    - **Design Motivation**: Replacing softmax-attention with linear projections results in parameters two to three orders of magnitude fewer than MHA, while maintaining task-aware selection capabilities superior to mean/max pooling.

3.  **Downcast MHA probe (35M expressivity ceiling)**:
    - **Function**: Utilizes Multi-head Attention for stronger representation capability, while keeping parameters low through aggressive QKV dimension downsampling.
    - **Mechanism**: QKV dimensions are reduced from $d$ to $d_\text{inner} \in \{d/16, d/32\}$. With $H$ heads, head computation is $\text{Attn}_h(\mathbf{Q}_h, \mathbf{K}_h, \mathbf{V}_h) = \text{softmax}(\mathbf{Q}_h \mathbf{K}_h^\top / \sqrt{d_\text{head}}) \mathbf{V}_h$. Results are concatenated and projected back to $d$ before mean/max pooling. $L+2$ MHA modules are used. Total parameters: $(L+2) \cdot 4 d \cdot d_\text{inner}$, approximately 35M. It utilizes PyTorch `scaled_dot_product_attention` for FlashAttention optimization.
    - **Design Motivation**: MHA captures relationships between tokens and layers (rather than independent scoring), representing the expressivity ceiling of the probe. "Downcasting" keeps parameters at 35M, which is still much smaller than standalone guard models (billions).

### Loss & Training
The LLM parameters are frozen throughout. Only the probe head $\theta$ is trained using a standard cross-entropy objective. All backbones (Llama-3.2-3B-Instruct, GPT-OSS-20B, Qwen3-30B-A3B) share the same architectural template while training independent probes.

## Key Experimental Results

### Main Results: ToxicChat Safety Classification (F1 / AUPRC, bracketed numbers denote added params in M)

| Backbone | Method | F1 (%) | AUPRC | Added Params (M) | Extra LM Call |
|----------|--------|--------|-------|------------------|---------------|
| - | ToxicChat-T5-large (Standalone) | 82.2 | 0.885 | 780 | Yes |
| - | MULI (logit-only reuse) | 77.8 | 0.829 | 0.13 | No |
| Llama-3.2-3B | Direct pooling | 73.53 ± 0.68 | 0.812 | 0.003 | No |
| Llama-3.2-3B | Scoring attention | 80.49 ± 1.17 | 0.854 | 0.10 | No |
| Llama-3.2-3B | **Multi-head self-attn** | **84.51 ± 0.43** | **0.898** | 35 | No |
| GPT-OSS-20B | Multi-head self-attn | 86.17 ± 0.51 | 0.915 | 27 | No |
| Qwen3-30B-A3B | Multi-head self-attn | 83.76 ± 0.9 | — | — | No |

The MHA probe outperforms MULI (+6 F1+) across all backbones and matches the F1 of the 780M standalone T5 model with only ~35M parameters, demonstrating 22x parameter efficiency.

### Ablation Study: Aggregation Operator Family

| Configuration | F1 (Llama-3.2-3B) | Params (M) | Description |
|---------------|-------------------|------------|-------------|
| Direct pooling (mean/max) | 73.53 | 0.003 | No learnable parameters, baseline only |
| Scoring attention gate | 80.49 | 0.10 | Added token/layer importance scoring, +7 F1 |
| Multi-head self-attn (downcast) | 84.51 | 35 | Full expressivity ceiling, further +4 F1 |
| w/o layer agg (last layer only) | Close to ShieldHead | — | Loses distributed signals |
| w/o token agg (first-token only)| Close to MULI | — | Loses token dimension information |

### Key Findings
- The ranking of the three operators is consistent across 3 backbones (pooling < scoring < MHA), suggesting that gains from "joint token+layer readout" are architecture-agnostic.
- The MHA probe achieves higher F1 on GPT-OSS-20B (86.17 vs. 84.51), supporting the hypothesis that larger models' hidden states contain more signals extractable by probes.
- The 35M MHA probe approaches the performance of an independent 780M T5 classifier (~0.01 AUPRC difference) while saving an extra LM call; end-to-end latency is nearly identical to pure generation.
- The Scoring attention gate is a "sweet spot" for the performance/parameter trade-off, outperforming MULI by ~3 F1 points with only 100K parameters.

## Highlights & Insights
- **"BERTology meets production" is an overlooked entry point**: Applying a decade of research on Transformer layer abstraction to deployment scenarios serves as a reminder not to arbitrarily pick a single layer for readout. This retrofitting method is particularly valuable for existing LLM services.
- **Two-stage decomposition avoids dimensional explosion**: A direct $T \cdot L \cdot d \to C$ classifier would have millions of parameters; the two-stage approach reduces parameters from $O(TLd)$ to $O(L \cdot d)$, making it transferable to other hidden-state probing tasks (e.g., fact detection, toxicity attribution).
- **"Same forward pass" is the key to end-to-end latency**: Many computation reuse works emphasize parameter efficiency but ignore the latency cost of redundant LLM calls. This work physically attaches the probe to the generation forward pass, achieving true zero-overhead inference.
- **Generalization to MoE**: Effectiveness is verified on GPT-OSS-20B (MoE) and Qwen3-30B-A3B, indicating that hidden state readout is independent of specific routing strategies.

## Limitations & Future Work
- Experiments primarily focus on English safety/sentiment tasks, with no coverage of multilingual or complex multi-label policy classification.
- Probe training requires labeled data (same annotation cost as standalone classifiers); the savings are in deployment-side LM calls, not annotation.
- While the 35M MHA probe is smaller than guard models, it still requires keeping $L+1$ layers of hidden states in memory for aggregation, creating $O(LTd)$ memory pressure beyond the KV cache, which is non-trivial for long contexts.
- Adversarial robustness is not discussed: Could an attacker perturb prompts to specifically target a token/layer that causes the probe to fail?
- Future work could explore joint training of the probe and router (teaching MoE to concentrate discriminative signals in probe-selected layers/tokens).

## Related Work & Insights
- **vs MULI (logit-only)**: MULI trains a sparse classifier on first-token logits. It uses only 0.13M parameters but reads only the last layer/position. This work uses 35M parameters to read the full tensor, gaining +7 F1 points at the cost of 250x more parameters (still an order of magnitude smaller than standalone guards).
- **vs ShieldHead / OmniGuard (single-layer probe)**: These methods fix readout to the final hidden state layer. This work learns "which layer" to read, with the layer-level scoring gate acting as a soft layer selector to avoid manual bias.
- **vs Standalone guard (Llama-Guard / ToxicChat-T5)**: Guard models are independent LLMs requiring separate forward passes (seconds of latency, GBs of VRAM). This probe is effectively "free," using almost no additional computational resources.
- **vs EAGLE-3 (multi-layer feature fusion for speculative decoding)**: EAGLE-3 also proved multi-layer fusion is superior to top-layer only, but for draft generation. This work extends the intuition to classification.

## Rating
- Novelty: ⭐⭐⭐⭐ Combines BERTology with production LLM orchestration; the two-stage aggregation architecture is a clean contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 3 backbones, dense/MoE architectures, and two task types. Lacks multilingual and robustness testing.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation (BERTology → representation selection → two-stage aggregation); well-organized formulas and tables.
- Value: ⭐⭐⭐⭐ Highly practical for LLM serving engineering; eliminates extra calls and has low deployment barriers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Adaptive Layer Selection for Layer-Wise Token Pruning in LLM Inference](adaptive_layer_selection_for_layer-wise_token_pruning_in_llm_inference.md)
- [\[ICLR 2026\] A Fano-Style Accuracy Upper Bound for LLM Single-Pass Reasoning in Multi-Hop QA](../../ICLR2026/model_compression/a_fano-style_accuracy_upper_bound_for_llm_single-pass_reasoning_in_multi-hop_qa.md)
- [\[ACL 2026\] GlimpRouter: Efficient Collaborative Inference by Glimpsing One Token of Thoughts](glimprouter_efficient_collaborative_inference_by_glimpsing_one_token_of_thoughts.md)
- [\[ACL 2026\] TELL-TALE: Task Efficient LLMs with Task Aware Layer Elimination](tell-tale_task_efficient_llms_with_task_aware_layer_elimination.md)
- [\[ACL 2026\] Cognitive-Uncertainty Guided Knowledge Distillation for Accurate Classification of Student Misconceptions](cognitive-uncertainty_guided_knowledge_distillation_for_accurate_classification_.md)

</div>

<!-- RELATED:END -->
