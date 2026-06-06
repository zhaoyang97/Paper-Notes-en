---
title: >-
  [Paper Note] DAPD: Dependency-Aware Parallel Decoding via Attention for Diffusion LLMs
description: >-
  [ICML 2026][Image Restoration][dLLM] DAPD transforms the single-step parallel unmasking problem of dLLMs into a dynamic graph coloring problem of "selecting independent sets on a self-attention-induced MRF." Without any…
tags:
  - "ICML 2026"
  - "Image Restoration"
  - "dLLM"
  - "Parallel Decoding"
  - "Self-Attention"
  - "Markov Random Field"
  - "Graph Coloring"
date: 2026-05-08
content_hash: 21867df02e086fa7
---

# DAPD: Dependency-Aware Parallel Decoding via Attention for Diffusion LLMs

**Conference**: ICML 2026  
**arXiv**: [2603.12996](https://arxiv.org/abs/2603.12996)  
**Code**: https://ai-isl.github.io/dapd (Project Page)  
**Area**: LLM Efficiency / Diffusion Language Models / Parallel Decoding  
**Keywords**: dLLM, Parallel Decoding, Self-Attention, Markov Random Field, Graph Coloring

## TL;DR
DAPD transforms the single-step parallel unmasking problem of dLLMs into a dynamic graph coloring problem of "selecting independent sets on a self-attention-induced MRF." Without any training, it simultaneously unmasks weakly dependent positions, reducing the decoding steps for multi-query mixed prompts to 1/3.87 of the original while maintaining near-zero accuracy loss on LLaDA / Dream.

## Background & Motivation
**Background**: Diffusion Language Models (dLLMs) such as LLaDA and Dream generate text by iteratively denoising mask tokens. Their primary claimed advantage over autoregressive models is the ability to unmask multiple tokens in a single step, significantly lowering the Number of Function Evaluations (NFE)—the primary determinant of inference latency.

**Limitations of Prior Work**: dLLM training objectives only model the conditional **marginal distribution** $p_\theta(x^i\mid\mathbf{x})$ of each mask position, without explicitly modeling the joint distribution. Sampling multiple tokens independently from these marginals leads to a "joint-marginal mismatch": for example, given the prompt "The capital of [M] is [M]", the models might provide high marginal probabilities for "France" and "London" at the two positions respectively. Individually they are reasonable, but together they are incorrect.

**Key Challenge**: Existing training-free parallel decoding methods (Fast-dLLM / EB-Sampler / KLASS) only use token-wise signals like "marginal confidence / entropy / KL stability" to filter positions, **entirely failing to model dependencies between mask positions**. Consequently, they are either too conservative (low speed) or too aggressive (collapsed quality by unmasking strongly coupled tokens). Introducing auxiliary planners or retraining (dParallel, Learn-to-Parallel) breaks the ELBO framework and incurs high overhead.

**Goal**: To explicitly estimate "which mask positions can be safely unmasked together" at each decoding step without additional training or auxiliary models.

**Key Insight**: dLLMs already compute a self-attention map. If position $i$ rarely attends to position $j$, then the prediction of $X_i$ is largely independent of $X_j$ given the other context. In other words, **self-attention itself is a free probe for conditional independence**.

**Core Idea**: Use symmetrized attention scores $s_{ij}=\tfrac{1}{2}(a_{ij}+a_{ji})$ to induce an MRF dependency graph over mask positions. Parallel decoding is then reduced to finding "independent sets" on this graph, using a Welsh–Powell degree-priority greedy coloring strategy to select a maximal independent set for simultaneous unmasking at each step.

## Method

### Overall Architecture
The input is the current masked sequence $\mathbf{x}_t$ of the dLLM. Each step performs one Transformer forward pass to obtain marginals $p_\theta(x^i\mid\mathbf{x}_t)$ and attention matrices from all layers and heads. DAPD inserts a lightweight "Graph Construction + Independent Set Selection" module after the forward pass:

1. Aggregate attention from the last 30% of layers and all heads to get $a_{ij}$, then symmetrize it as $s_{ij}$.
2. Construct a dependency graph $G_t=(V_t,E_t)$ for current mask nodes by using a conservative threshold $\tau_t$, where edges are $\{s_{ij}>\tau_t\}$.
3. Calculate a "confidence-weighted proxy degree" $\tilde d_i\cdot\mathrm{conf}_i$ for each mask node $i$. Traverse nodes in descending order, adding $i$ to a candidate set $S$ if and only if it is not adjacent to any node already in $S$.
4. Simultaneously unmask all tokens in $S$ using argmax from their respective marginals.
5. Switch to a fast-tail strategy—"unmask all nodes with confidence > 0.9"—when the remaining mask ratio is < 50%.

This entire process requires no additional models or retraining. The only new overhead is graph construction and greedy sorting, which is negligible compared to a Transformer forward pass.

### Key Designs

1. **Self-Attention → MRF Dependency Graph**:
    - **Function**: Reinterprets internal self-attention as a proxy for the "strength of conditional dependency between mask positions" to construct an undirected graph for scheduling parallel decoding.
    - **Mechanism**: Defines symmetric edge scores $s_{ij}=\tfrac{1}{2}(a_{ij}+a_{ji})$ on the mask index set $V_t$. An edge $(i,j)\in E_t$ is triggered if $s_{ij}>\tau_t$. The theoretical basis is the local Markov property of Transformers: if $i$ rarely attends to $j$, then $p_\theta(X_i\mid X_{V_t\setminus\{i\}})\approx p_\theta(X_i\mid X_{V_t\setminus\{i,j\}})$, implying $X_i\perp X_j \mid X_{V_t\setminus\{i,j\}}$. Validations on synthetic length-9 sequences show an edge detection AUC of 0.928 and a low Order Violation Ratio (OVR) of 0.04, proving attention is a reliable proxy for MRF structure.
    - **Design Motivation**: Prior methods treated mask positions as independent units, discarding the nearly free dependency information already computed by the model. Using attention as MRF edge weights directly addresses the root cause of joint-marginal mismatch without extra parameters.

2. **Parallel Decoding = Dynamic Graph Coloring + Welsh–Powell Degree-First Strategy**:
    - **Function**: Selects a subset of mask nodes $S$ in $G_t$ that is as large as possible but contains no internal edges, allowing these tokens to be unmasked in parallel.
    - **Mechanism**: Maps "covering all mask positions in minimum steps" to "finding a valid coloring of $G_t$ with minimum colors"—where the same color represents a single parallel step. Since $V_t$ shrinks and $E_t$ changes dynamically with new context, this is "dynamic graph coloring." Instead of seeking the Maximum Independent Set (which favors low-degree nodes), the authors use Welsh–Powell: nodes are scanned in descending order of degree $\tilde d_i:=\sum_{j\ne i}s_{ij}$. Nodes non-adjacent to the current set $S$ are added to form a **maximal** (not necessarily maximum) independent set. Prioritizing "hub" nodes sparsifies the remaining graph aggressively, leading to fewer total steps. The final implementation uses $\tilde d_i\cdot\mathrm{conf}_i$ as the sorting key.
    - **Design Motivation**: The goal is to minimize total NFE, not just single-step width. A naive "Maximum Independent Set" is a short-sighted greedy approach that might leave high-degree hubs for later, dragging out the tail. A degree-first strategy breaks strong dependency chains early, allowing massive parallelization of isolated nodes in subsequent steps.

3. **Late-Stage Sparsification Strategy**:
    - **Function**: Disables graph construction and switches to an aggressive "unmask all $\mathrm{conf}_i>0.9$" strategy when the dependency graph naturally becomes nearly edge-less.
    - **Mechanism**: Triggered when the mask ratio falls below 50%. At this stage, most nodes have degrees near 0 (approximate conditional independence), making confidence thresholds a low-cost approximation of an independent set. Any position with confidence exactly 1.0 can be safely unmasked at any time without risk of mismatch.
    - **Design Motivation**: While graph construction is cheap, it is redundant when dependencies are mostly resolved. This step pushes the NFE curve below pure confidence-based methods while maintaining accuracy.

### Loss & Training
Completely training-free: DAPD does not modify any dLLM weights or introduce additional trainable parameters. Evaluations are performed directly on public LLaDA-8B-Instruct and Dream-7B-Instruct models.

## Key Experimental Results

### Main Results
Evaluated on LLaDA and Dream across Coding (HumanEval/MBPP), Math (GSM8K/Math500), Instruction Following (IFEval), and ParallelBench. Results for "multi-question mixed prompts" (TriviaQA × 5) on LLaDA are shown below:

| Method | Accuracy (↑) | Steps | Relative Speedup |
|--------|--------------|-------|------------------|
| Original (Confidence) | 52.64 | 256.0 | 1.00× |
| Fast-dLLM | 52.12 | 124.4 | 2.06× |
| KLASS | 52.20 | 177.4 | 1.44× |
| EB-Sampler | 51.20 | 131.3 | 1.95× |
| **DAPD (Ours)** | **52.08** | **66.2** | **3.87×** |

On MBPP and IFEval, DAPD significantly outperforms baselines in single-block settings where baselines usually require block-splitting or EOS suppression to maintain accuracy. It also dominates the Score-Steps Pareto frontier on ParallelBench.

### Ablation Study
| Configuration | Key Observation | Description |
|---------------|-----------------|-------------|
| Attention Layer Selection | Last 30% layers are best | High layers integrate global information; low layers are local. |
| Sorting Key | $\tilde d_i\cdot\mathrm{conf}_i$ is superior | Considers both structural importance and prediction reliability. |
| Welsh–Powell vs. MIS | Degree-first uses fewer steps | Solving hubs early sparsifies the remaining graph faster. |
| Late-stage Threshold | Further reduces NFE | Confidence thresholds approximate independent sets when edges are sparse. |

### Key Findings
- **Decoding Trajectory Shift**: Visualizing prompts containing five independent sub-questions shows that baselines (Fast-dLLM/KLASS) follow a "pseudo-autoregressive" pattern moving from ends inward. DAPD unmasks tokens scattered across the entire sequence in the first 50% of steps, truly leveraging the bidirectional, any-order capabilities of dLLMs.
- **Source of Acceleration**: The 3.87× speedup (vs. 2.06× for Fast-dLLM) demonstrates that explicit dependency modeling uncovers significantly more parallelization opportunities than marginal confidence alone.
- **Cross-Model Generalization**: DAPD maintains its lead on Dream without needing tricks like EOS suppression, proving the improvements stem from the method itself.
- **Negligible Overhead**: Since DAPD reuses attention, tokens-per-second (TPS) shows an actual end-to-end gain over baselines, rather than being "step-efficient but slow-per-step."

## Highlights & Insights
- **"Attention = Free Conditional Independence Probe"**: This is a highly reusable perspective. While previous token-wise signals (confidence/entropy/KL) focus on marginals, DAPD is the first to systematically reinterpret attention as a dependency graph, logically targeting the root of joint-marginal mismatch with zero training.
- **Elegant Formalization via Graph Coloring**: By reducing parallel decoding to a combinatorial optimization problem, DAPD can leverage established heuristics like Welsh–Powell and leaves room for future stronger graph algorithms (e.g., branch-and-bound).
- **Degree-Priority vs. Short-term Greed**: Selecting hubs first is counter-intuitive for single-step width but optimal for long-term NFE. This insight is transferable to other "batch scheduling" scenarios like KV cache replacement or draft selection in speculative decoding.
- **Natural EOS Mitigation**: The authors found baselines often fail in single-block settings due to EOS overflow. DAPD's scattered unmasking pattern naturally delays the generation of structured endings, avoiding this common pitfall.

## Limitations & Future Work
- **Graph Construction Cost**: Though lightweight, $O(L^2)$ edge scoring may become a bottleneck for sequences with thousands of tokens; the paper lacks benchmarks for long sequences (>1k tokens).
- **Hyperparameter Robustness**: Core parameters (layer selection, threshold $\tau_t$, 0.9 confidence switch) are somewhat tuned for LLaDA/Dream. Automating these for different architectures remains an open problem.
- **First-Order Approximation**: The "Low attention $\implies$ Independence" assumption is an approximation. Path-based indirect dependencies or cross-head semantics are averaged out, which might fail on adversarial dependency structures.
- **Task Dependency**: On tasks with "single global consistency" (like GSM8K), the gap between DAPD and baselines is smaller. The maximum dividend of DAPD is realized on prompts that inherently contain parallelizable independent sub-structures.

## Related Work & Insights
- **vs. Fast-dLLM**: Fast-dLLM uses a fixed confidence threshold. DAPD reuses this for its late-stage tail but gains its 3.87× vs 2.06× advantage entirely through dependency modeling in the early stages.
- **vs. EB-Sampler**: EB-Sampler uses entropy bounds for risk control, which remains a marginal-only signal. DAPD's structural information-driven approach is more direct.
- **vs. KLASS**: KLASS uses KL divergence between adjacent steps, a token-position scalar. DAPD uses structural (graph) signals to distinguish between "confident but conflicting" token pairs.
- **vs. Training-based Methods (dParallel, APD)**: Training a planner generally performs better but is costly and can break ELBO. DAPD provides a nearly zero-cost engineering alternative by optimizing on top of existing internal signals.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ High. Reinterpreting attention as MRF weights and using graph coloring is a clean, unified perspective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Good coverage of models and tasks, including synthetic validation. Needs more long-sequence/larger-scale model results.
- **Writing Quality**: ⭐⭐⭐⭐ Clear reasoning and visual comparisons.
- **Value**: ⭐⭐⭐⭐⭐ High. Training-free, plug-and-play, and doubles the speedup of prior SOTA methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] DyLLM: Efficient Diffusion LLM Inference via Saliency-based Token Selection and Partial Attention](dyllm_efficient_diffusion_llm_inference_via_saliency-based_token_selection_and_p.md)
- [\[ICML 2026\] Triadic Dynamics Aware Diffusion Posterior Sampling for Inverse Problems: Optimizing Guidance and Stochasticity Schedules](triadic_dynamics_aware_diffusion_posterior_sampling_for_inverse_problems_optimiz.md)
- [\[CVPR 2026\] DRFusion: Degradation-Robust Fusion via Degradation-Aware Diffusion Framework](../../CVPR2026/image_restoration/drfusion_degradation_robust_fusion_via_degradation_aware_diffusion_framework.md)
- [\[ICLR 2026\] Skip to the Good Part: Representation Structure & Inference-Time Layer Skipping in Diffusion vs. Autoregressive LLMs](../../ICLR2026/image_restoration/skip_to_the_good_part_representation_structure_inference-time_layer_skipping_in_.md)
- [\[ICML 2026\] Degradation-Aware Metric Prompting for Hyperspectral Image Restoration](degradation-aware_metric_prompting_for_hyperspectral_image_restoration.md)

</div>

<!-- RELATED:END -->
</div>

## Related Papers

- [\[ICML 2026\] DyLLM: Efficient Diffusion LLM Inference via Saliency-based Token Selection and Partial Attention](dyllm_efficient_diffusion_llm_inference_via_saliency-based_token_selection_and_p.md)
- [\[ICLR 2026\] Skip to the Good Part: Representation Structure & Inference-Time Layer Skipping in Diffusion vs. Autoregressive LLMs](../../ICLR2026/image_restoration/skip_to_the_good_part_representation_structure_inference-time_layer_skipping_in_.md)
- [\[ICML 2025\] ε-VAE: Denoising as Visual Decoding](../../ICML2025/image_restoration/epsilon-vae_denoising_as_visual_decoding.md)
- [\[CVPR 2026\] DRFusion: Degradation-Robust Fusion via Degradation-Aware Diffusion Framework](../../CVPR2026/image_restoration/drfusion_degradation_robust_fusion_via_degradation_aware_diffusion_framework.md)
- [\[ICML 2026\] Triadic Dynamics Aware Diffusion Posterior Sampling for Inverse Problems: Optimizing Guidance and Stochasticity Schedules](triadic_dynamics_aware_diffusion_posterior_sampling_for_inverse_problems_optimiz.md)

</div>

<!-- RELATED:END -->
