---
title: >-
  [Paper Note] Scaling Depth Capacity via Zero/One-Layer Model Expansion
description: >-
  [ICML 2026][LLM Pretraining][progressive training] This paper proposes "Zero/One-Layer Progressive Training"—training an extremely shallow model with almost no Transformer layers first…
tags:
  - "ICML 2026"
  - "LLM Pretraining"
  - "progressive training"
  - "depth expansion"
  - "zero/one-layer"
  - "WSD schedule"
  - "muP"
date: 2026-05-08
content_hash: ff9fed2e73aa842c
---

# Scaling Depth Capacity via Zero/One-Layer Model Expansion

**Conference**: ICML 2026  
**arXiv**: [2511.04981](https://arxiv.org/abs/2511.04981)  
**Code**: None  
**Area**: LLM Pre-training / Efficient Training / Model Expansion  
**Keywords**: progressive training, depth expansion, zero/one-layer, WSD schedule, muP

## TL;DR
This paper proposes "Zero/One-Layer Progressive Training"—training an extremely shallow model with almost no Transformer layers first, then expanding the depth to the target number of layers at a late stage ($\approx 80\%$ iterations). Combined with WSD learning rate schedules and muP hyperparameter transfer, this approach saves approximately 80% of computation ($\approx 5\times$ speedup) on GPT2, LLAMA3, and DeepSeekV3 with almost no degradation in final loss.

## Background & Motivation
**Background**: Training large language models is prohibitively expensive (LLAMA-4 training exceeds 7M GPU hours). A primary acceleration strategy is **progressive training / model expansion**: training a small "teacher/source model" first, then expanding the model to a larger size at time $t=\tau$. The computation cost is approximately $6B(\tau N_{\text{small}} + (T-\tau) N_{\text{large}})$, which is significantly smaller than the $6BTN_{\text{large}}$ required for fixed-size training, provided that $\tau$ is close to $T$ and $N_{\text{small}} \ll N_{\text{large}}$.

**Limitations of Prior Work**: Existing methods restrict depth expansion to 2-4$\times$, and the source model still requires over a dozen layers. Consequently, the compute savings are only $\approx 30-45\%$ (compared to target model training). Furthermore, most works are validated only on classification models like BERT/ViT, yielding only 1.4-2$\times$ speedups on generative LLMs. Worse, multi-stage expansion (e.g., 0→2→12), while appearing more "progressive," fails to demonstrate *mixing* behavior (where loss catches up to the baseline) across expansion points.

**Key Challenge**: Prior methods have not pushed to the limits in two dimensions: first, none have dared to use extremely shallow 0/1-layer source models (due to uncertainty about whether they can learn anything transferable); second, *function-preserving* initialization (e.g., zero-initializing sub-layers) conflicts with *feature learning*: while zero-init prevents loss spikes, it results in dead gradients, preventing new layers from learning. Additionally, standard learning rate schedules (like cosine decay) lead to late-stage expansion failing to converge as the learning rate approaches zero.

**Goal**: (1) Push the source model to the extreme of 0 or 1 layer; (2) Push the expansion point $\tau$ to 0.8T; (3) Ensure hyperparameters remain unchanged before and after expansion; (4) Provide a unified recipe covering dense/MoE, MHA/GQA/MLA, and cosine/WSD, supported by convex optimization convergence proofs to explain why this approach works.

**Key Insight**: Re-model "depth expansion" as an **initialization problem** in large model training—decomposing the large model into $\mathbf{W}_t = [\mathbf{w}_t, \mathbf{x}_t]$ comprising "small model part + newly added layers." Progressive training is then equivalent to projected gradient descent on $\mathbf{x}$ (masking it to 0), followed by a "teleportation" to a good initialization, and then normal SGD. Under this unified perspective, initialization strategies and learning rate schedules can be derived from convergence bounds for convex+Lipschitz losses.

**Core Idea**: Zero/One-layer progressive training + WSD schedule + muP hyperparameter transfer shifts the "loss-compute Pareto front" significantly to the bottom-left compared to existing work.

## Method

### Overall Architecture
The pipeline is extremely simple: train a 0-layer model (consisting only of Embedding + LM_head + final LayerNorm, *entirely without* Transformer layers) or a 1-layer model. During the *stable phase* of a WSD schedule, select a point $\tau \approx 0.8T$ and expand the model to the target depth $L$ in one step. For zero-layer models, new layers must be randomly initialized; for one-layer models, layers can be initialized via random init or copying (e.g., $\mathbf{w}\to[\mathbf{w},\mathbf{w},\mathbf{w}]$). After expansion, **continue training with the same learning rate** until completion. This process is consistent across GPT2, LLAMA3, Qwen3, Mixtral, DeepSeekV3, and ResNet, covering variants such as weight-tying, dense/MoE, MHA/GQA/MLA, absolute/rotary positional embeddings, LayerNorm/RMSNorm, and GeLU/SwiGLU.

### Key Designs

1.  **Unified "Initialization-Theory" Perspective for Depth Expansion**:
    - **Function**: Explicitly characterizes the gap between the progressive training loss upper bound and the fixed-size training loss upper bound.
    - **Mechanism**: Decompose $\mathbf{W}_t=[\mathbf{w}_t, \mathbf{x}_t]$ into "reused part + new part" and assume an optimal $\mathbf{W}^*=[\mathbf{w}^*, \mathbf{x}^*]$. Under convex + $G$-Lipschitz loss, two-stage SGD yields a gap = $\frac{\sum_{t=1}^{\tau}\eta_t}{\sum_{t=1}^{T}\eta_t}(L(\mathbf{w}^*)-L(\mathbf{W}^*)) + \frac{\|\mathbf{x}_\tau-\mathbf{x}^*\|^2-\|\mathbf{x}_0-\mathbf{x}^*\|^2}{2\sum_{t=1}^{T}\eta_t}$. The first term requires $L(\mathbf{w}^*) \approx L(\mathbf{W}^*)$, and the second term requires $\mathbf{x}_\tau$ to be closer to the optimum than $\mathbf{x}_0$.
    - **Design Motivation**: The algebraic equivalent is viewing progressive training as **PGD (masking new layers) + teleportation + SGD**. Thus, choices for initialization (copying vs. random) and learning rate scheduling (cosine vs. WSD) can be derived from the same bound. Random init makes the second term 0, while copying potentially makes it < 0. For scheduling, $\frac{\sum_{t\le\tau}\eta_t}{\sum_t\eta_t}$ should be small, meaning the LR before expansion cannot be too high and must not decay too early, which matches the shape of WSD.

2.  **muP-scaled Initialization + Zero Hyperparameter Retuning**:
    - **Function**: Allows the 0/1-layer small model and the target deep model to share the **same set of hyperparameters** (LR, weight decay, etc.), eliminating the need for retuning at expansion.
    - **Mechanism**: Requires element-wise scale consistency for activations: $\|\mathbf{A}_l\|_2/\sqrt{n_l} \sim \|\mathbf{A}_{l+1}\|_2/\sqrt{n_{l+1}}$, which for linear layers leads to the spectral scaling condition $\|\mathbf{W}_l\|_* \sim \sqrt{n_{l+1}/n_l}$. On this basis, Muon-NSGD is used (Muon for 2D tensors, normalized SGD for others, shared LR, weight decay=0.01). New layers use random Gaussian to satisfy muP; copying also satisfies it. However, "zero" and "copying_zero" (zeroing specific sub-layers) are discarded as they break feature learning.
    - **Design Motivation**: A major engineering pain point in progressive training is the need to retune hyperparameters after expansion. muP makes the optimal hyperparameters constant across model sizes. Table 1 summarizes the trade-offs: copying/random satisfy feature learning and trainability but are *not* function-preserving (resulting in a loss spike); zero-init is function-preserving but blocks learning. Ours prioritizes trainability + feature learning > function-preserving.

3.  **WSD Schedule + 0.8T Single-Stage Expansion + Mixing-Time Inference**:
    - **Function**: Pushes expansion time $\tau$ to 80% of total training and explains why WSD is the "natural partner" for progressive training.
    - **Mechanism**: Define mixing time $t_{\text{mix}}$ such that $L(\mathbf{W}_{\tau+t_{\text{mix}}}^{\text{progressive}}) \approx L(\mathbf{W}_{\tau+t_{\text{mix}}}^{\text{fixed-size}})$. Experiments show $t_{\text{mix}}(\tau)$ is highly sensitive to $\tau$ under cosine schedules (GPT fails to "catch up" if $\tau \ge 0.5T$), whereas it is insensitive under WSD ($\tau \ge 0.8T$ still catches up). This aligns with the theory where $\eta_t$ remains constant during the stable phase. Protocol: 2% warmup + long stable phase + 10% decay. Subtract $t_{\text{mix}}$ (measured from small-scale pilot runs) from the total length to set $\tau \approx 0.8T$. The paper argues **single-stage expansion is sufficient**: based on mixing behavior, $0\to 2\to 12$ can be decomposed into $0\to 2$ and $2\to 12$, yielding FLOPs similar to $2\to 12$ but worse than $0\to 12$.
    - **Design Motivation**: Existing works fail to observe mixing behavior—and thus resort to multi-stage expansion—because they use cosine schedules and a "grown-vs-target" comparison perspective. Switching to the "full training process" + WSD reveals mixing, making single-stage late expansion optimal.

### Loss & Training
- **Data**: OpenWebText, sequence length 1024, based on nanoGPT codebase.
- **Optimizer**: Muon-NSGD (primary), AdamW and SGD (ancillary), weight decay=0.01, no gradient clipping.
- **LR Schedule**: Cosine and WSD (warmup-stable-decay), decaying to 0; 2% warmup.
- **Token-per-param**: 50 for LLAMA3, 100 for DeepSeekV3 (MoE).
- **Expansion Moment**: $\tau \approx 0.8T$ (e.g., 480k/528k iterations for GPT2 124M).

## Key Experimental Results

### Main Results
(Example: GPT2 on OpenWebText with WSD schedule. "FLOPs ratio" is relative to fixed-size training; lower is faster.)

| Setting | Source Model | Target Model | FLOPs ratio | val loss gap |
|--|--|--|--|--|
| Fixed-size | — | 12-layer 124M | 100% | Baseline |
| Zero-layer progressive | 0-layer 39M | 12-layer 124M | ≈20% | <0.5% |
| One-layer progressive | 1-layer 46M | 12-layer 124M | ≈20% | <0.5% |
| Fixed-size | — | 60-layer 7B | 100% | Baseline |
| Zero-layer progressive | 0-layer 0.15B | 60-layer 7B | ≈20% | <0.2% |
| One-layer progressive | 1-layer 0.27B | 60-layer 7B | ≈20% | <0.2% |

Scaling Law Perspective: On LLAMA3 (dense, 0.25B–2B) and DeepSeekV3 (MoE, 0.2B–0.5B active), the progressive scaling exponent consistently outperforms fixed-size, with 3–5$\times$ computational efficiency gains that **amplify as the model grows**.

### Ablation Study

| Ablation Dimension | Key Findings |
|--|--|
| Initialization (copying vs random vs zero / copying_zero) | Both random and copying work; copying is slightly better. Zero-type init breaks feature learning. |
| Multi-layer expansion strategy (copying_last / _stack / _inter) | copying_last is significantly worse; _stack and _inter are nearly indistinguishable—"copying all layers" is key. |
| Schedule (cosine vs WSD) | WSD allows $\tau$ to be pushed to 0.8T while still mixing; cosine fails GPT at $\tau \ge 0.5T$ and ResNet at $\tau \ge 0.7T$. |
| Multi-stage (0→2→12 vs 0→12) | Multi-stage provides no extra benefit; FLOPs are similar to 2→12, inferior to 0→12. |
| Source model depth (0/1/2/4/6/8) | 0/1-layer models uniquely occupy the Pareto front; layers $\ge 2$ sit towards the top-right. |

### Key Findings
- **"Mixing" is the soul of this method**: The loss peak at expansion looks severe, but as long as $\tau + t_{\text{mix}} \le T$, the final loss aligns with fixed-size training. This was systematically overlooked in previous "grown-vs-target" perspectives.
- **Mixing time is largely independent of source model size**: Whether expanding from 1-layer or 6-layers, the latest expansion moment $t_{\text{mix}}$ remains similar. Since 6-layer source models are expensive, "shallower is better."
- **Theory provides strong support for WSD**: The theoretical gap is minimized when $\frac{\sum_{t\le\tau}\eta_t}{\sum_t\eta_t}$ is small. WSD maintains a constant LR in the stable phase, keeping this ratio low and robust against $\tau$.
- **Consistent behavior on MoE**: DeepSeekV3 and Mixtral exhibit mixing, and this approach is orthogonal to upcycling (increasing experts without increasing depth).

## Highlights & Insights
- **Elegant Perspective Shift**: Re-evaluating depth expansion as an initialization problem + teleportation allows initialization strategy and LR scheduling to be derived from a single convergence bound.
- **Extreme Courage with "Layer 0"**: Daring to push the source model to 0 layers—which essentially has only embeddings—and finding it sufficient to reach a good "teleportation" point via WSD and muP is a foundational contribution.
- **Multi-stage = Cascaded Single-stage**: By analyzing mixing behavior, the authors debunk the necessity of complex multi-stage stacking strategies.
- **Practical Engineering Recipe**: Measuring $t_{\text{mix}}$ via small-scale pilot runs provides a reliable heuristic for setting $\tau \approx T - t_{\text{mix}}$.

## Limitations & Future Work
- Convergence theory is based on convex + Lipschitz assumptions, whereas deep learning is non-convex; theory serves as a guide for training dynamics rather than absolute proof.
- Experiments focus on OpenWebText and ImageNet. The largest dense LLM is 7B and MoE active is 0.5B; performance at the 100B+ scale remains to be verified.
- Only **depth** was studied. Does the same extreme "zero-extension" apply to width or expert count?
- Orthogonal to upcycling (dense→MoE), but combined experiments were not provided.
- Expansion time $\tau$ still requires calibration runs.
- Lack of downstream benchmark evaluation (SFT, RLHF); results rely on validation loss and scaling laws.

## Related Work & Insights
- **vs Function-preserving approaches (Net2Net, bert2BERT, etc.)**: These ensure no loss spike at expansion by sacrificing trainability; ours sacrifices function-preserving for trainability and feature learning, leading to lower final loss.
- **vs Gradual stacking / Multi-stage (gong2019, shen2022, etc.)**: They typically save 30-45% compute. Ours achieves 80% savings via single-stage 60$\times$ expansion by leveraging mixing behavior.
- **vs muP / WSD Relationship**: This work does not "invent" muP or WSD but demonstrates they are the optimal missing pieces for making extreme progressive training viable.
- **vs Upcycling MoE (he2024, etc.)**: Upcycling scales expert count; this work scales depth. They are complementary.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Pushing source models to 0/1 layers and $\tau$ to 0.8T while unifying it under a convergence bound is highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 5 LLM architectures, dense/MoE, 150+ scanning iterations on Pareto curves, and 7B scale validation.
- **Writing Quality**: ⭐⭐⭐⭐ Theoretical and empirical links are clear. "Perspective matters" section directly addresses literature misconceptions.
- **Value**: ⭐⭐⭐⭐⭐ Provides a practical recipe for 5$\times$ speedup with negligible loss, offering significant economic value for LLM pre-training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Inverse Depth Scaling From Most Layers Being Similar](inverse_depth_scaling_from_most_layers_being_similar.md)
- [\[ICML 2026\] Model Merging Scaling Laws in Large Language Models](model_merging_scaling_laws_in_large_language_models.md)
- [\[ICML 2026\] Dropout Universality: Scaling Laws and Optimal Scheduling at the Edge-of-Chaos](dropout_universality_scaling_laws_and_optimal_scheduling_at_the_edge-of-chaos.md)
- [\[NeurIPS 2025\] Gemstones: A Model Suite for Multi-Faceted Scaling Laws](../../NeurIPS2025/llm_pretraining/gemstones_a_model_suite_for_multi-faceted_scaling_laws.md)
- [\[ICML 2026\] InfoLaw: Information Scaling Laws for Large Language Models with Quality-Weighted Mixture Data and Repetition](infolaw_information_scaling_laws_for_large_language_models_with_quality-weighted.md)

</div>

<!-- RELATED:END -->
