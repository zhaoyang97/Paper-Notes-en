---
title: >-
  [Paper Note] Any-SSR: How Recursive Least Squares Works in Continual Learning of Large Language Models
description: >-
  [ICCV 2025][LLM/NLP][LLM continual learning] This paper proposes the Analytic Subspace Routing (Any-SSR) framework, which eliminates inter-task interference by assigning each task an independent LoRA subspace…
tags:
  - "ICCV 2025"
  - "LLM/NLP"
  - "LLM continual learning"
  - "catastrophic forgetting"
  - "recursive least squares"
  - "LoRA subspace routing"
  - "replay-free learning"
date: 2026-05-08
content_hash: 568320d958207821
---

# Any-SSR: How Recursive Least Squares Works in Continual Learning of Large Language Models

**Conference**: ICCV 2025
**arXiv**: N/A (CVF OpenAccess)  
**Code**: [https://github.com/ZHUANGHP/Any-SSR](https://github.com/ZHUANGHP/Any-SSR)  
**Area**: LLM/NLP
**Keywords**: LLM continual learning, catastrophic forgetting, recursive least squares, LoRA subspace routing, replay-free learning

## TL;DR

This paper proposes the Analytic Subspace Routing (Any-SSR) framework, which eliminates inter-task interference by assigning each task an independent LoRA subspace, and trains a zero-forgetting analytic router via a recursive least squares (RLS) closed-form solution, enabling replay-free continual learning for LLMs.

## Background & Motivation

### Challenges in LLM Continual Learning

LLMs must continuously absorb new domain knowledge in dynamic real-world environments. However, the fine-tuning process is inherently a continual learning procedure in which all pretraining data cannot be revisited. This leads to **catastrophic forgetting**—the model rapidly loses previously acquired knowledge upon receiving new tasks. This problem is particularly severe for LLMs, as the general capabilities encoded in the high-dimensional parameter space are easily disrupted by fine-tuning.

### Fundamental Limitations of Existing Methods

**Why are existing methods insufficient?**

**Replay-based methods** (e.g., Replay, SEEKR): require storing and replaying historical data, incurring high computational costs and privacy risks, making them impractical given the enormous pretraining data of LLMs.

**Parameter-efficient fine-tuning methods** (e.g., LoRAMoE, O-LoRA): absorb knowledge from all tasks using a fixed set of shared parameters; sequential fine-tuning of different tasks on shared modules inevitably causes catastrophic forgetting.

**Regularization methods** (e.g., EWC): impose constraints in parameter space, but identifying parameter importance remains difficult in the vast parameter space of LLMs.

### Core Insight

The authors observe that the fundamental issue with existing PET methods is **knowledge interference** across tasks—all tasks share the same parameter space. Inspired by the hierarchical processing of cortical systems, they hypothesize that the lower layers of a Transformer encode cross-task semantic features, while the upper layers handle task-specific semantic composition. Based on this, one can freeze the lower layers to preserve general capabilities and assign each task an independent LoRA subspace in the upper layers.

## Method

### Overall Architecture

The Any-SSR architecture comprises three core components:

1. **Frozen general feature extractor**: the first $L_f$ layers of the LLM are completely frozen.
2. **Task-specific LoRA Bank**: independent LoRA adapters are maintained for each task in the subsequent layers.
3. **Recursive Analytic Learning (AL) router**: a task router based on the RLS closed-form solution.

The forward pass is formulated as:
$$y_{t+1} = h_{\leq L_f}(x) \cdot f_{\theta_{k^*}}(h_{>L_f}(x))$$

where $k^* = \arg\max_k g_k(h_{\leq L_f}(x))$ is the task ID selected by the router.

### Key Design 1: Hierarchical Feature Decoupling

**Why layer-wise decomposition?** Based on the hypothesis that "lower layers encode general semantics while upper layers handle task-specific information," the pretrained LLM parameters are partitioned into:

- **Frozen lower layers** $h_{\leq L_f}$: preserve the general language understanding capabilities acquired during pretraining.
- **Adaptable upper layers** $h_{>L_f}$: accommodate independent LoRA adapters for each new task.

The LoRA for each task $D_k$ updates the weights via low-rank decomposition:
$$\Delta W_l^{(k)} = B_l^{(k)} A_l^{(k)}, \quad B_l^{(k)} \in \mathbb{R}^{d_{in} \times r}, A_l^{(k)} \in \mathbb{R}^{r \times d_{out}}$$

**Why do different tasks not interfere with each other?** The LoRA adapters for all tasks are trained independently on top of frozen base parameters, resulting in completely disjoint parameter spaces.

### Key Design 2: Analytic Routing Mechanism

The router's core objective is to process the lower-layer features $h_{\leq L_f}(x)$ to predict the task assignment distribution $p(k|x)$, and to update its weights when new tasks arrive without accessing historical data.

**Feature expansion**: features are first mapped to a higher-dimensional space to enhance linear separability (based on Cover's theorem):
$$\tilde{h} = \phi(\text{mean-pool}(h_{\leq L_f}(X))) \in \mathbb{R}^E$$

where $\phi: \mathbb{R}^d \rightarrow \mathbb{R}^E$ is a fixed Gaussian-initialized projection ($E > d$) implemented as a ReLU-activated linear transformation.

**Ridge regression closed-form solution**: router weights are obtained by solving a convex optimization problem:
$$\hat{W}_k^r = \left(\sum_{i=1}^{k} \tilde{h}_i^\top \tilde{h}_i + \lambda I\right)^{-1} \left(\sum_{i=1}^{k} \tilde{h}_i^\top Y_i\right)$$

where $\lambda > 0$ is the regularization coefficient and $Y_i$ denotes the task ID labels.

### Key Design 3: Recursive Incremental Update

**This is the most central contribution of the method.** Define the autocorrelation matrix $R_k$ and cross-correlation matrix $Q_k$:

$$R_k = \left(\sum_{i=1}^{k} \tilde{h}_i^\top \tilde{h}_i + \lambda I\right)^{-1}, \quad Q_k = \sum_{i=1}^{k} \tilde{h}_i^\top Y_i$$

When a new task $D_{k+1}$ arrives, the **Woodbury matrix identity** enables recursive updates:

$$R_{k+1} = R_k - R_k \tilde{h}_{k+1}^\top (I + \tilde{h}_{k+1} R_k \tilde{h}_{k+1}^\top)^{-1} \tilde{h}_{k+1} R_k$$

$$\hat{W}_{k+1}^r = (I - R_{k+1} X_{k+1}^\top X_{k+1}) \hat{W}_k^r + R_{k+1} X_{k+1}^\top Y_{k+1}$$

**Why does this guarantee zero forgetting?** The mathematical equivalence of recursive updates ensures that training the router sequentially on each task yields identical weights to joint training on all task data simultaneously. Consequently, the incorporation of a new task does not alter the routing decisions for previously learned tasks.

### Inference Procedure

1. The input passes through the frozen feature extractor to obtain $h_{\leq L_f}(X_t)$.
2. Features are expanded into the higher-dimensional space.
3. The router computes task probabilities: $p(k|X_t) = \text{softmax}(\tilde{h}(X_t) W_k^r)$.
4. The dominant task is selected: $k^* = \arg\max_k p(k|X_t)$.
5. The corresponding LoRA adapter is activated to generate the next token.

## Key Experimental Results

### Main Results

**TRACE Benchmark (LLaMA-2-7B-Chat)**:

| Method | Order1 OP(BWT) | Order2 OP(BWT) |
|--------|----------------|----------------|
| AdaLoRA | 22.60 (-30.11) | 23.34 (-27.54) |
| LoRAMoE | 48.54 (-4.27) | 47.48 (-4.28) |
| SeqFT | 47.63 (-11.45) | 45.12 (-12.27) |
| EWC | 48.20 (-9.48) | 44.54 (-12.00) |
| O-LoRA | 44.64 (-4.20) | 42.83 (-9.11) |
| Replay (1%) | 48.47 (-9.69) | 47.04 (-10.24) |
| SEEKR (1%) | 54.99 (-2.61) | 54.69 (-2.53) |
| **Any-SSR** | **55.69 (0.00)** | **55.69 (0.00)** |
| *Upper-bound (MTL)* | *59.38* | - |

Any-SSR is the only method achieving BWT = 0, and produces identical results under both task orders, demonstrating its invariance to task ordering. OP falls only 3.69% below the joint training upper bound.

### Ablation Study

**Component ablation (heatmap analysis)**:

| Configuration | Characteristic | Issue |
|---------------|---------------|-------|
| Single LoRA | All tasks share one LoRA | Performance on old tasks drops sharply after new task learning |
| Multi-LoRA + BP router | Independent LoRA with backprop-trained router | Routing accuracy degrades progressively (avg. −21.7% per stage) |
| **Multi-LoRA + AL router** | **Any-SSR** | **100% routing accuracy; zero forgetting on all historical tasks** |

**Hyperparameter analysis**:

| $L_f$ (frozen layers) | $E$ (expansion dim) | OP(BWT) |
|-----------------------|---------------------|---------|
| 2 | 10000 | 54.79 (-0.19) |
| **4** | **10000** | **55.69 (0.00)** |
| 6 | 10000 | 53.21 (0.00) |

$L_f = 4$ and $E = 10000$ constitute the optimal configuration. Too small an $L_f$ yields overly generic lower-layer features that degrade routing precision; too large an $L_f$ leaves insufficient learning capacity in the upper-layer LoRAs.

**General capability retention**:

| Method | MMLU | GSM | BBH | GA (DeltaGA) |
|--------|------|-----|-----|--------------|
| LLaMA-2-7B-Chat | 46.89 | 27.14 | 39.73 | 47.77 |
| SeqFT | 45.16 | 14.03 | 32.50 | 43.50 (-4.27) |
| SEEKR (1%) | 46.32 | 20.85 | 38.52 | 46.72 (-1.05) |
| **Any-SSR** | 45.77 | 25.43 | 37.01 | **46.51 (-1.26)** |

Without any replay, Any-SSR incurs only −1.26 degradation in general capability, approaching SEEKR (−1.05), which relies on 1% replay data.

### Key Findings

1. **Mathematical guarantee of BWT = 0**: Any-SSR is the only method that provably achieves zero backward transfer regardless of task order.
2. **Routing accuracy is critical**: backprop-trained routers suffer catastrophic forgetting (accuracy degrades progressively), while the analytic router maintains 100% accuracy throughout.
3. **Storage efficiency**: each task requires only $O(rL_{adapt})$ parameters (<1% of full fine-tuning), and the router stores only the autocorrelation matrix $R_k$.
4. **Task-order invariance**: two distinct task orderings produce completely identical results.

## Highlights & Insights

1. **Innovative application of classical mathematical tools**: the paper applies recursive least squares (RLS), a classical tool from signal processing, to LLM continual learning, providing rigorous mathematical guarantees.
2. **Elegant parameter decoupling design**: complete task parameter isolation is achieved through hierarchical partitioning and independent LoRAs, eliminating interference at the architectural level.
3. **Extremely low-cost router updates**: recursive updates require only the storage of the autocorrelation matrix and can be performed on a CPU without GPU resources.
4. **Unity of theory and practice**: the zero-forgetting property is not only theoretically proven but also perfectly validated by experiments.

## Limitations & Future Work

1. Minor routing failures occur between NumGLUE-sm and NumGLUE-ds, attributable to ambiguous routing caused by shared prompts (e.g., "Solve the math problem").
2. Storage requirements of the LoRA Bank grow linearly with the number of tasks.
3. The general capability evaluation requires using a 1% validation set to train the general task branch of the router, which does not fully satisfy the ideal of "zero data access."
4. Validation is limited to LLMs at the 7B–9B scale; effectiveness at larger scales (70B+) remains to be confirmed.
5. Routing errors in critical application domains (finance, healthcare) may introduce risks.

## Related Work & Insights

- **O-LoRA**: enables LoRA continual learning in orthogonal subspaces, but still employs a fixed parameter count that causes interference.
- **SEEKR**: mitigates forgetting through attention distillation, but requires storing replay samples.
- **LoRAMoE**: introduces a mixture of LoRA experts, but the router training does not provide zero-forgetting guarantees.
- **Cover's theorem**: theoretical basis for enhancing linear separability via feature dimensionality expansion.
- **Woodbury matrix identity**: mathematical tool enabling recursive updates.
- Broader implication: closed-form solution methods offer a mathematically rigorous pathway for continual learning that bypasses gradient-based optimization.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] VA-GPT: Aligning Effective Tokens with Video Anomaly in Large Language Models](va_gpt_aligning_effective_tokens_video_anomaly.md)
- [\[ICCV 2025\] VIM: Versatile Interactive Motion-Language Model](vim_versatile_interactive_motion_language_model.md)
- [\[ICCV 2025\] ShadowHack: Hacking Shadows via Luminance-Color Divide and Conquer](shadowhack_hacking_shadows_via_luminance-color_divide_and_conquer.md)
- [\[ICCV 2025\] FW-Merging: Scaling Model Merging with Frank-Wolfe Optimization](fw-merging_scaling_model_merging_with_frank-wolfe_optimization.md)
- [\[NeurIPS 2025\] C²Prompt: Class-aware Client Knowledge Interaction for Federated Continual Learning](../../NeurIPS2025/llm_nlp/c2prompt_class-aware_client_knowledge_interaction_for_federated_continual_learni.md)

</div>

<!-- RELATED:END -->
