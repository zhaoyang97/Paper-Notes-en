---
title: >-
  [Paper Note] Scaling Inference-Efficient Language Models
description: >-
  [ICML 2025][LLM Pretraining][Scaling Laws] This paper proposes an inference-aware Scaling Law that jointly optimizes parameter count, training token count, and model shape by introducing a model aspect ratio term into the Chinchilla loss function. Sixty-three models are trained to fit this law, which then guides the design of the Morph-1B model, achieving a 1.8× speedup in inference latency while maintaining downstream task accuracy.
tags:
  - "ICML 2025"
  - "LLM Pretraining"
  - "Scaling Laws"
  - "Inference Efficiency"
  - "Model Architecture"
  - "Wide and Shallow Models"
  - "Morph-1B"
date: 2026-05-08
content_hash: d7858421466e91f1
---

# Scaling Inference-Efficient Language Models

**Conference**: ICML 2025  
**arXiv**: [2501.18107](https://arxiv.org/abs/2501.18107)  
**Code**: [https://github.com/Waterpine/open-lm-morph](https://github.com/Waterpine/open-lm-morph)  
**Area**: LLM Pre-training  
**Keywords**: Scaling Laws, Inference Efficiency, Model Architecture, Wide and Shallow Models, Morph-1B

## TL;DR

This paper proposes an inference-aware Scaling Law that jointly optimizes parameter count, training token count, and model shape by introducing a model aspect ratio term into the Chinchilla loss function. Sixty-three models are trained to fit this law, which then guides the design of the Morph-1B model, achieving a 1.8× speedup in inference latency while maintaining downstream task accuracy.

## Background & Motivation

**Background**: Scaling Laws (particularly Chinchilla) have become the core tools for predicting LLM performance, guiding the balance between model parameter count $N$ and training token count $D$ under a fixed compute budget, yielding the classic conclusion of $D \approx 20N$.

**Limitations of Prior Work**:
   - Existing Scaling Laws focus solely on **training cost** and completely ignore **inference cost**. In practical deployment, however, inference overhead far exceeds training since models undergo repetitive inference over their lifetimes.
   - FLOP constraints are unrealistic: In practice, model size is bound by deployment hardware memory, and training token count is determined by available data size (e.g., LLaMA-3-8B used 15T tokens, far exceeding Chinchilla's recommended 160B). Neither is strictly bound by a FLOP budget.
   - While Sardana et al. (2023) attempted to incorporate inference FLOPs, their method requires estimating the total number of inference tokens over the model's lifetime, which is impractical in reality.

**Key Challenge**: Models with the same parameter count can exhibit inference latencies that differ by up to **3.5×**—for example, the latency of MiniCPM-1B is even higher than that of Qwen2.5-14B. This indicates that parameter count is not the sole factor determining inference efficiency; **model architecture (shape)** is the critical variable, yet existing Scaling Laws fail to model this factor entirely.

**Goal**
   - Sub-problem 1: How to introduce model architectural variables (width $d_{\text{model}}$ vs. depth $n_{\text{layers}}$) into the Scaling Law?
   - Sub-problem 2: How to incorporate the inference latency budget $T_C$ into the optimization constraints?
   - Sub-problem 3: There is a gap between the loss predicted by the Scaling Law and downstream task accuracy; how can the truly optimal model configuration be selected?

**Key Insight**: The authors observe through extensive experiments that inference latency grows linearly with layer depth (since layer-wise computations must be serialized), whereas the impact of increasing width is far smaller than that of depth. Consequently, under the same parameter budget, a **wider and shallower** model runs faster during inference.

**Core Idea**: Multiply the Chinchilla loss function by an architectural correction term $(1 + \varepsilon R^\gamma)$ involving the aspect ratio $R = d_{\text{model}} / n_{\text{layers}}$, thereby co-optimizing loss and inference efficiency.

## Method

### Overall Architecture

The entire method can be divided into three phases:

- **Phase A (Candidate Generation)**: Given a parameter budget (e.g., 1B), generate multiple model configuration candidates with different aspect ratios (e.g., 12 layers × 3072 hidden size, 16 layers × 2560 hidden size, 24 layers × 2048 hidden size, etc.).
- **Phase B (Predict + Measure)**: For each candidate, (1) predict its loss using the inference-aware Scaling Law; (2) measure its actual inference latency.
- **Phase C (Rank + Train + Evaluate)**: Select top-k candidates based on predicted loss rankings and latency constraints, train them fully, evaluate them on downstream tasks, and release the optimal model.

Core Inputs: Parameter upper bound $N_C$, training token upper bound $D_C$, and inference latency budget $T_C$.  
Core Outputs: The optimal model architectural configuration that satisfies the constraints.

### Key Designs

1. **Inference-Aware Scaling Law Formula**

    - Function: Predict model training loss given parameters $N$, training tokens $D$, and aspect ratio $R$.
    - Mechanism: On top of Chinchilla's $L(N, D) = E + AN^{-\alpha} + BD^{-\beta}$, a scaling correction factor is multiplied:
    $$L(N, D, R) = (E + AN^{-\alpha} + BD^{-\beta}) \cdot (1 + \varepsilon R^\gamma)$$
      where $R = d_{\text{model}} / n_{\text{layers}}$, and $A, B, E, \alpha, \beta, \gamma, \varepsilon$ are learnable parameters. In experiments, $\alpha = \beta = \gamma$ is enforced to simplify fitting.
    - Design Motivation: Experiments reveal that the impact of aspect ratio $R$ on loss exhibits a pattern captureable by $(1 + \varepsilon R^\gamma)$—specifically, overly wide models (with excessively large $R$) have a slightly increased loss but yield a substantial speedup in inference. This multiplicative form allows the architectural correction to be applied uniformly across different $(N, D)$ combinations.

2. **Constraint Rewriting: From FLOP Constraints to Tri-Constraints**

    - Function: Convert the optimization target of the Scaling Law from an impractical FLOP constraint to a tri-constraint of parameters, tokens, and inference latency.
    - Mechanism: The original Chinchilla goal is $\arg\min_{N,D} L(N,D) \text{ s.t. } \text{FLOPs}(N,D) = C$. This paper changes it to:
    $$\arg\min_{N,D} L(N,D) \text{ s.t. } N \leq N_C, \; D \leq D_C, \; T_{\text{inf}} \leq T_C$$
      This explicitly incorporates the inference latency budget into the optimization. The inference latency $T_{\text{inf}}$ is measured empirically on target hardware (taking only a few minutes).
    - Design Motivation: In practical deployments, model size is limited by hardware memory, and training data size is limited by corpora scale—both are hard constraints rather than trade-offs. The latency budget is determined by application scenarios (e.g., chatbots requiring low latency).

3. **Predict-Rank-Select Model Selection Method**

    - Function: Solve the issue where models have similar losses but vastly different downstream task performances, selecting the truly superior model from multiple candidates.
    - Mechanism: Instead of relying on the absolute loss value of the Scaling Law to directly predict downstream accuracy (which is too noisy), this method leverages its **ranking capability**—specifically, a lower predicted loss ranks higher. The top-k candidates are selected for full training, and downstream evaluation is used for the final decision.
    - Design Motivation: The authors observe on PIQA, BoolQ, and HellaSwag that the relationship between loss and accuracy displays varying patterns (some approximately linear, others "step-like"), making accurate prediction of absolute accuracy highly challenging. However, the ranking accuracy of the Scaling Law is significantly higher than its absolute prediction accuracy (achieving a Spearman correlation of 1.0).

### Loss & Training

- **Training Data**: A uniformly sampled subset of the DCLM-Baseline dataset is used, with single-epoch training (no repetitions).
- **Optimizer**: AdamW, bfloat16 precision.
- **Scaling Law Fitting**: 63 models (80M–339M parameters, 1.6B–12.8B tokens) are trained, and the formulation is fitted via non-linear least squares using the Levenberg-Marquardt algorithm.
- **Importance of Over-trained Data**: Ablation studies show that fitting solely with Chinchilla-optimal data ($D=20N$) yields poor results, while incorporating over-trained data ($D=160N$) significantly improves prediction accuracy (MSE drops from 0.1165 to 0.0006).
- **Morph-1B Final Configuration**: $d_{\text{model}}=3072$, $n_{\text{layers}}=12$, $n_{\text{heads}}=16$, intermediate size 8192, trained on 30B tokens.

## Key Experimental Results

### Main Results: Comparison of Morph-1B and Open-Source Models of the Same Scale

| Model | $d_{\text{model}}$ | $n_{\text{layers}}$ | Downstream Avg. Accuracy | Inference Latency (s) |
|------|----:|----:|:----:|:----:|
| Open-LM-1B | 2048 | 24 | 0.49 | 3.61 |
| OPT-1.3B | 2048 | 24 | 0.50 | 2.55 |
| Pythia-1.3B | 2048 | 22 | 0.49 | 3.28 |
| NeoX-1.3B | 2048 | 24 | 0.49 | 3.99 |
| OPT-IML-1.3B | 2048 | 24 | 0.54 | 2.54 |
| Morph-1B-v1 | 2048 | 24 | 0.52 | 3.61 |
| Morph-1B-v2 | 2560 | 16 | 0.52 | 2.57 |
| **Morph-1B** | **3072** | **12** | **0.52** | **1.96** |

Using an ultra-shallow architecture of only 12 layers, Morph-1B maintains a downstream average accuracy of 0.52 (matching v1/v2) while reducing inference latency to 1.96s, which is **1.8× faster** than the standard 24-layer architecture (3.61s). The slightly superior accuracy of OPT-IML-1.3B (0.54) stems from its 6× larger volume of training data (180B vs 30B) and instruction fine-tuning.

### Scaling Law Fitting Accuracy Comparison

| Metric | Chinchilla | Inference-Aware Scaling Law |
|------|:----:|:----:|
| MSE | 0.0033 | **0.0006** |
| $R^2$ | 0.9895 | **0.9982** |
| Relative prediction error | 2.7%–4.1% | **< 1.2%** |
| Spearman (1B Prediction) | -0.40 | **1.00** |

The inference-aware Scaling Law outperforms Chinchilla substantially across all metrics. Crucially, the Spearman correlation coefficient improves from -0.40 (completely incorrect ranking) to 1.00 (perfect ranking), demonstrating its exceptionally strong ability to predict ranking.

### Ablation Study

| Ablation Setting | MSE | $R^2$ | Spearman | Description |
|---------|:---:|:---:|:---:|------|
| Full (with over-trained data) | 0.0006 | 0.9982 | 1.00 | Complete model |
| Remove over-trained data (Chinchilla) | 0.9825 | -2.1259 | — | Completely fails |
| Remove over-trained data (Inference-aware) | 0.1165 | 0.6293 | — | Regresses significantly but remains superior to Chinchilla |
| Random model shape (Chinchilla) | 0.0198 | 0.9369 | — | Underperforms / Large error |
| Random model shape (Inference-aware) | 0.0008 | 0.9973 | — | Remains robust |

### Key Findings

- **Over-training data is crucial**: Fitting the Scaling Law solely with $D=20N$ Chinchilla-optimal data yields poor results (both methods degrade significantly). Recruiting over-training data points ($D=160N$) restores accuracy. This suggests that the over-training phenomenon active in deployment must be modeled in scaling laws.
- **Inference-aware Scaling Law is more robust**: Even when fitting with randomly chosen model shapes, the inference-aware version maintains $R^2 > 0.99$, whereas Chinchilla drops to 0.93 with a high prediction error of 11.8%–13.4%.
- **Consistency of wide and shallow models**: Across three scales (1B/3B/7B), different batch sizes, two frameworks (HuggingFace/vLLM), and two GPUs (A100/A30), wider and shallower models consistently exhibit lower latency and higher throughput, showing highly stable patterns.
- **Extremely low fitting cost**: Developing an adequately accurate inference-aware Scaling Law requires only 6 data points and 85 GPU hours.

## Highlights & Insights

- **Ingenious multiplicative correction factor design**: Multiplying the Chinchilla formulation by $(1 + \varepsilon R^\gamma)$ rather than adding it allows the architectural correction to naturally couple with base predictions of $(N, D)$. It is both simple and effective, introducing only 2 additional parameters to fit.
- **The "ranking is superior to absolute prediction" insight is highly practical**: Admitting that absolute loss predictions from Scaling Laws are insufficiently precise (making it difficult to reliably map them to downstream accuracy) and instead exploiting their ranking capability for screening candidates is a pragmatic and highly effective methodology.
- **Integrating "free" inference latency measurements into the selection pipeline**: Empirical measurements of inference latency take only a few minutes with virtually zero overhead, yet they directly filter out candidates that do not meet latency criteria. This design concept can be generalized to any hardware-aware model selection setup.
- **Physical intuition of wide and shallow models**: Computations between layers must be sequential (attention → FFN → next layer), meaning more layers result in more sequential steps, whereas matrix operations within a single layer can be parallelized. Thus, increasing width does not noticeably increase latency. This elegant physics-inspired intuition is a guiding thread throughout the work.

## Limitations & Future Work

- **Limited scale**: Constrained by resources, the largest model trained is 1.5B parameters. Whether the pattern of how the aspect ratio affects loss remains consistent at larger scales, such as 7B/13B/70B, has not been validated.
- **Single architecture focus**: Only the standard MHA + FFN Transformer is considered, without incorporating efficient attention variants like GQA, MQA, or MLA. Although the authors claim the framework can generalize to these architectures, no empirical validation is provided.
- **Inference system dependency**: Latency measurements are based on HuggingFace generate (unquantized, unoptimized). Whether the advantages of wide and shallow structures persist in actual deployment environments (such as TensorRT-LLM or quantized modes of vLLM) requires further validation, though vLLM experiments in the appendix indicate similar trends.
- **Limited downstream tasks**: Evaluation is restricted to knowledge/commonsense benchmarks (ARC, PIQA, HellaSwag, etc.), leaving out more practical tasks such as generative quality, instruction-following, and coding.
- **KV cache size overlooked**: Although wide and shallow models have lower latency, a larger $d_{\text{model}}$ implies a larger KV-cache footprint, which may become a memory bottleneck in scenarios with long sequences and large batch sizes.

## Related Work & Insights

- **vs. Chinchilla (Hoffmann et al., 2022)**: Chinchilla optimizes only the ratio of $N$ and $D$, neglecting architecture and inference costs. This paper adds architectural variables to its formulation to make it more practical. Chinchilla's Spearman rank correlation drops to a negative value when predicting 1B models, showing that its predictions fail entirely when dealing with different architectures.
- **vs. Beyond Chinchilla-Optimal (Sardana et al., 2023)**: Sardana integrates inference cost by estimating total inference FLOPs, which requires knowing the model's lifetime inference volume beforehand—something impractical in reality. In contrast, this paper directly measures latency, making it more practical.
- **vs. DCLM/Gadre et al. (2024)**: Gadre extends Scaling Laws to downstream task error predictions. However, this paper discovers that the loss-to-accuracy mapping is highly noisy when losses are close, and thus shifts to a ranking method, which is much more robust.
- **vs. MiniCPM (Hu et al., 2024)**: MiniCPM also trains small models with a focus on efficiency; however, its 1B model, owing to its deep and narrow architecture, has a higher latency than some 14B models, perfectly validating the core arguments of this paper.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of introducing architectural variables into Scaling Laws is not entirely brand new (already discussed in Kaplan et al., 2020), but formalizing it as a multiplicative correction term and integrating it with inference latency constraints is a valuable contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 63 models were trained, covering 5 parameter scales × multiple aspect ratios × 3 training lengths. The appendix includes a large number of supplementary experiments (A30 GPU, vLLM, TTFT, etc.), making it highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ The structure is clear, with abundant plots/tables and natural motivational derivations. There are minor LaTeX typesetting glitches in equations, but they do not hinder comprehension.
- Value: ⭐⭐⭐⭐ This study directly guides model architecture choices in LLM deployment. The conclusion that "wide and shallow models infer faster" is elegant and practical, ready to be immediately applied to model design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Scaling Embedding Layers in Language Models](../../NeurIPS2025/llm_pretraining/scaling_embedding_layers_in_language_models.md)
- [\[ICLR 2026\] Scaling Behavior of Discrete Diffusion Language Models](../../ICLR2026/llm_pretraining/scaling_behavior_of_discrete_diffusion_language_models.md)
- [\[ACL 2025\] AsyncLM: Efficient and Adaptive Async Pre-training of Language Models](../../ACL2025/llm_pretraining/asynclm_efficient_and_adaptive_async_pre-training_of_language_models.md)
- [\[ACL 2025\] Towards Effective and Efficient Continual Pre-training of Large Language Models](../../ACL2025/llm_pretraining/towards_effective_and_efficient_continual_pre-training_of_large_language_models.md)
- [\[ICLR 2026\] Pretraining Scaling Laws for Generative Evaluations of Language Models](../../ICLR2026/llm_pretraining/pretraining_scaling_laws_for_generative_evaluations_of_language_models.md)

</div>

<!-- RELATED:END -->
