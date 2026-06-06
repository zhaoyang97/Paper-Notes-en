---
title: >-
  [Paper Note] Beyond Sunk Costs: Boosting LLM Pre-training Efficiency via Orthogonal Growth of Mixture-of-Experts
description: >-
  [ICML 2026][LLM Efficiency][MoE Model Growth] This work proposes an "orthogonal growth" strategy for converged MoE models—utilizing interpositional layer replication for the depth dimension and noisy expert cloning for t…
tags:
  - "ICML 2026"
  - "LLM Efficiency"
  - "MoE Model Growth"
  - "Checkpoint Recycling"
  - "Efficient Pre-training"
  - "Orthogonal Expansion"
  - "Sunk Costs"
date: 2026-05-08
content_hash: 39b05c3b03cbd4dd
---

# Beyond Sunk Costs: Boosting LLM Pre-training Efficiency via Orthogonal Growth of Mixture-of-Experts

**Conference**: ICML 2026  
**arXiv**: [2510.08008](https://arxiv.org/abs/2510.08008)  
**Code**: None  
**Area**: LLM Pre-training  
**Keywords**: MoE Model Growth, Checkpoint Recycling, Efficient Pre-training, Orthogonal Expansion, Sunk Costs  

## TL;DR

This work proposes an "orthogonal growth" strategy for converged MoE models—utilizing interpositional layer replication for the depth dimension and noisy expert cloning for the width dimension. By expanding a 17B model to 70B, it achieves a 10.6% accuracy improvement over training from scratch under the same additional computational budget.

## Background & Motivation

**Background**: LLM pre-training follows scaling laws, necessitating continuous increases in model size and data volume to enhance performance. However, the computational cost of training from scratch is enormous. The industry generates numerous intermediate checkpoints and smaller models during training, which are typically discarded upon completion.

**Limitations of Prior Work**: Most existing model growth research only performs expansion after brief training in the early stages, failing to fully utilize the "sunk costs" accumulated by fully trained models. Furthermore, with the popularity of Mixture-of-Experts (MoE) architectures, there is a lack of systematic research into MoE model growth strategies.

**Key Challenge**: Pre-training checkpoints represent significant computational investment but cannot be directly reused for larger models due to architectural constraints. Existing stacking methods for layer cloning disrupt the learned inter-layer structures once a model has sufficiently converged, leading to performance degradation.

**Goal**: Design a parameter expansion framework suitable for fully converged MoE models that efficiently scales in both depth and width dimensions to maximize the recovery of sunk costs.

**Key Insight**: The authors observe that fully converged LLMs exhibit a characteristic layer-wise weight norm distribution—small and fluctuating in the initial layers, monotonically increasing in the middle layers, and slightly decreasing at the end. Stacking methods create a "norm cliff" at the splicing point, whereas interpositional methods maintain this smooth structure.

**Core Idea**: Use interpositional layer replication to increase depth (preserving weight norm continuity) and noisy expert cloning to increase width (breaking symmetry to promote expert specialization). These two dimensions are orthogonal and can be combined freely.

## Method

### Overall Architecture

Given a fully converged MoE model checkpoint, the model is expanded through two orthogonal dimensions: (1) Depth growth—cloning each layer in-place $k$ times to increase depth; (2) Width growth—doubling the number of experts and adding infinitesimal noise. These operations are order-independent and equivalently effective, resulting in a larger model for continued training.

### Key Designs

1.  **Interpositional Depth Growth**:
    - **Function**: Increases model depth through layer cloning (e.g., $n$ layers $\rightarrow$ $kn$ layers).
    - **Mechanism**: Unlike the stacking method that concatenates all layers end-to-end $k$ times ($l_1,...,l_n, l_1,...,l_n$), the interpositional method clones each layer in-place $k$ times ($l_1,l_1,..., l_2,l_2,..., l_n,l_n$). Fully converged LLMs possess a monotonically increasing layer-wise weight norm distribution. Stacking produces a norm cliff at the junction of the $n$-th and 1st layers, approximately $10\times$ the average inter-layer difference (e.g., 0.0356 $\rightarrow$ 0.0323), requiring extra compute to repair this discontinuity. Interpositional growth maintains a smooth transition.
    - **Design Motivation**: When training FLOPs exceed the Chinchilla optimal value $F_c \approx 6 \cdot N_a \cdot 20N_a$, the layer-wise weight norms form a stable increasing pattern, at which point interpositional growth significantly outperforms stacking.

2.  **Noisy Expert Cloning (Width Growth)**:
    - **Function**: Expands the number of experts in MoE layers from $E$ to $2E$ and active experts from $k$ to $2k$.
    - **Mechanism**: Original experts are cloned with the addition of slight Gaussian noise $\epsilon \sim \mathcal{N}(0, (\alpha \sigma_{\text{orig}})^2)$, where $\alpha = 0.01$. Router weights are similarly cloned and denoised. Direct cloning ($\alpha=0$) causes perfect symmetry where all copies receive identical gradients and fail to specialize; excessive noise (e.g., $\geq 50\%$ in upcycling) destroys learned knowledge. Infinitesimal noise breaks symmetry while preserving representations, allowing the router to gradually differentiate experts.
    - **Design Motivation**: Unlike dense-to-MoE upcycling, this MoE-to-MoE expansion reuses existing pre-trained routers, necessitating only minimal noise.

3.  **Orthogonal Combination and Growth Timing**:
    - **Function**: Verifies that depth and width growth can be combined freely and analyzes optimal growth timing.
    - **Mechanism**: Experiments show that depth-then-width and width-then-depth yield identical final performance (Adam first-moment cosine similarity $|\cos| < 0.04$), proving the two dimensions are nearly orthogonal in the optimization space. Regarding timing, larger sunk costs correlate positively with better final models, though marginal returns diminish after the learning rate annealing stage. Under fixed total FLOPs, growth methods perform equally to or slightly better than training from scratch.
    - **Design Motivation**: Orthogonality allows for staged expansion and flexible computational budget planning.

## Key Experimental Results

### Main Results (17B $\rightarrow$ 70B expansion, 1T tokens)

| Model | Parameters | Extra FLOPs | Avg Accuracy | Relative Gain |
| :--- | :--- | :--- | :--- | :--- |
| 17B Baseline (Fully Trained) | 17B | — | 58.55 | — |
| 70B From Scratch (Same extra compute) | 70B | Equivalent | 57.99 | -0.96% |
| 70B Orthogonal Growth (Ours) | 70B | Equivalent | 64.17 | **+10.6%** |
| 35B Intermediate (After depth growth) | 35B | — | 61.96 | +5.8% |

### Ablation Study: Depth Growth Strategy (3B $\rightarrow$ 6B)

| Growth Method | Post-convergence Training Loss | Downstream Avg Accuracy | Norm Continuity |
| :--- | :--- | :--- | :--- |
| Stacking | Higher | Lower | L19$\rightarrow$L20 Cliff (0.0356$\rightarrow$0.0323) |
| Interpositional (Ours) | **Lower** | **Higher** | Smooth Transition |
| From Scratch 6B | Reference | Reference | — |

### Noise Scale Ablation for Width Growth

| Noise Scale $\alpha$ | Training Loss | Downstream Accuracy | Note |
| :--- | :--- | :--- | :--- |
| 0 (Direct Clone) | Comparable | Baseline | Symmetry cannot be broken |
| 0.01 (Ours) | Comparable | **+1%** | Optimal balance point |
| Large Value | Higher | Decrease | Destruction of learned knowledge |
| Random Init New Experts| Significantly Higher | Significantly Decrease | Loss of knowledge inheritance |

### Relationship between Sunk Costs and Final Performance

| Growth Start Step | Starting Accuracy | Final Accuracy | Trend |
| :--- | :--- | :--- | :--- |
| 0k (Scratch) | 30.59 | 38.79 | Baseline |
| 16k | 37.66 | 44.65 | ↑ |
| 48k | 42.26 | 47.20 | ↑ |
| 88k | 46.43 | 48.99 | ↑ Optimal range |
| 96k (Annealing) | 46.19 | 48.52 | Diminishing marginal returns |

## Highlights & Insights

- First systematic study of growth strategies for fully converged MoE models, revealing the fundamental reason why stacking fails on converged models: the weight norm cliff.
- The proposed Chinchilla FLOPs boundary condition ($F_c$) serves as a practical criterion for selecting growth strategies: interpositional growth should be used when exceeding $1 \times F_c$.
- Orthogonality verification is not merely about order independence; it is evidenced from an optimization dynamics perspective through Adam momentum cosine similarity and cumulative weight update analysis.
- The discovery of positive correlation with sunk costs challenges the industry practice of "discarding checkpoints," providing a blueprint for sustainable LLM development.

## Limitations & Future Work

- The growth factor is fixed at $k=2$; more flexible non-uniform growth ratios remain unexplored.
- Width growth is less effective than depth growth in the short term, requiring more training steps to fully manifest its benefits.
- When growing from checkpoints in the learning rate annealing stage, the learning rate for continued training requires additional tuning.
- Validated only on language models; not yet extended to multimodal or other modalities.

## Related Work & Insights

- **Stacking Your Transformers** (Du et al., 2024): Proposed stacking but primarily for early-stage training; this work reveals its limitations on converged models.
- **LLaMA Pro** (Wu et al., 2024): Expands LLaMA by adding new layers but lacks systematic analysis.
- **MoE Upcycling** (Komatsuzaki et al., 2023): Dense $\rightarrow$ MoE conversion requires 50%+ noise; this work's MoE $\rightarrow$ MoE expansion requires only 1%.

## Rating

- Novelty: ⭐⭐⭐⭐ — Systematic study of converged MoE growth; weight norm analysis and orthogonality verification provide original value.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Multi-scale validation from 3B to 70B, including scaling law analysis and extensive ablations.
- Writing Quality: ⭐⭐⭐⭐ — Clear logic and rich visualizations.
- Value: ⭐⭐⭐⭐ — Provides a practical methodology for the sustainable development of LLM pre-training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Hyperparameter Transfer with Mixture-of-Experts Layers](hyperparameter_transfer_with_mixture-of-expert_layers.md)
- [\[ICML 2026\] ProbMoE: Differentiable Probabilistic Routing for Mixture-of-Experts](probmoe_differentiable_probabilistic_routing_for_mixture-of-experts.md)
- [\[ICML 2026\] ReMoE: Boosting Expert Reuse through Router Fine-Tuning in Memory-Constrained MoE LLM Inference](remoe_boosting_expert_reuse_through_router_fine-tuning_in_memory-constrained_moe.md)
- [\[ICML 2026\] SiameseNorm: Breaking the Barrier to Reconciling Pre/Post-Norm](siamesenorm_breaking_the_barrier_to_reconciling_prepost-norm.md)
- [\[ICML 2026\] RepetitionCurse: Measuring and Understanding Router Imbalance in Mixture-of-Experts LLMs under DoS Stress](repetitioncurse_measuring_and_understanding_router_imbalance_in_mixture-of-exper.md)

</div>

<!-- RELATED:END -->
