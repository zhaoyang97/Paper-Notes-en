---
title: >-
  [Paper Note] Full-Graph vs. Mini-Batch Training: Comprehensive Analysis from a Batch Size and Fan-Out Size Perspective
description: >-
  [ICLR 2026][Graph Learning][batch size] This paper treats GNN full-graph training as a special case of mini-batch training where both batch size and fan-out size are maximized. By analyzing convergence, generalization, and computational efficiency from the perspective of these two hyperparameters, it reaches a counter-intuitive conclusion: full-graph trainin
tags:
  - ICLR 2026
  - Graph Learning
  - batch size
  - fan-out size
date: 2026-05-08
content_hash: 919a438eed61d337
---
# Full-Graph vs. Mini-Batch Training: Comprehensive Analysis from a Batch Size and Fan-Out Size Perspective

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=ZSfgsh43vT](https://openreview.net/forum?id=ZSfgsh43vT)  
**Code**: https://github.com/LIUMENGFAN-gif/GNN_fullgraph_minibatch_training (Available)  
**Area**: Graph Learning / GNN Training Analysis  
**Keywords**: Full-graph training, mini-batch training, batch size, fan-out size, generalization bound

## TL;DR
This paper treats GNN full-graph training as a special case of mini-batch training where both batch size and fan-out size are maximized. By analyzing convergence, generalization, and computational efficiency from the perspective of these two hyperparameters, it reaches a counter-intuitive conclusion: full-graph training is not always superior to carefully tuned small-batch mini-batch training.

## Background & Motivation
**Background**: GNN training follows two mainstream paradigms. Full-graph training processes the entire graph at once, aggregating all neighbor information across multiple message-passing layers. Mini-batch training partitions the graph into subgraphs and iterates on sampled local neighborhoods. These two routes require distinct system designs—full-graph training needs efficient cross-device communication, while mini-batch training focuses on CPU-GPU data loading optimizations. Choosing between them determines the underlying system architecture.

**Limitations of Prior Work**: To systematically compare these routes, `batch size` (nodes sampled per batch) and `fan-out size` (neighbors sampled per skip) are natural analytical perspectives, as full-graph training is an extreme mini-batch case where batch size $b = n_{\text{train}}$ and fan-out $\beta = d_{\max}$. However, existing studies either examine these hyperparameters in isolation or focus on a single aspect—convergence, accuracy, or efficiency—failing to provide a holistic trade-off insight. Empirical comparisons (e.g., Bajaj et al.) are highly dependent on hardware and environment, making their conclusions hard to generalize.

**Key Challenge**: Current GNN theoretical analyses often rely on oversimplified assumptions—infinite width (averaging out neuron gradient noise) or linear models with convex losses (eliding local minima). These assumptions hide the real impact of batch size and fan-out size on training dynamics, failing to explain their actual roles.

**Goal**: In the context of transductive node classification, systematically characterize how batch size $b$ and fan-out size $\beta$ affect GNN optimization dynamics, generalization ability, and computational efficiency, and determine the true relative merits of full-graph versus mini-batch training.

**Key Insight**: The authors use a single-layer, finite-width GNN with ReLU as an analytical proxy. This model preserves phenomena emerging from finite width and non-linearity—which were neglected in prior work—yet remains simple enough to precisely characterize the roles of both hyperparameters. By unifying full-graph training into the mini-batch framework, both can be directly compared within the same theory.

**Core Idea**: By using batch size and fan-out size as a "unified scale," the paper proves that their effects on convergence and generalization are **non-isotropic**—batch size dominates convergence, while fan-out size dominates generalization. This provides an actionable guide for parameter tuning under resource constraints.

## Method

### Overall Architecture
The "method" is not a training system but a **unified analytical framework**. Full-graph training is rewritten as the limit of mini-batch training when $b=n_{\text{train}}$ and $\beta=d_{\max}$. Thus, the "full-graph vs. mini-batch" debate is translated into "how to set these two hyperparameters." Within this framework, the authors establish tools across three dimensions—optimization dynamics (convergence), generalization, and computational efficiency. Each dimension involves theoretical derivation followed by empirical validation, culminating in tuning recommendations.

The primary object of analysis is the normalized adjacency matrix $\tilde{A} = (D^{\text{in}}+I)^{-\frac12}(A+I)(D^{\text{out}}+I)^{-\frac12}$ with rows $\tilde{a}_i$. The single-layer GNN output is $z_i = \sigma(\tilde{a}_{\text{train},i} X W^\top)$, where $\tilde{a}_{\text{train},i}X$ is the aggregated embedding for node $i$. Batch size changes the set of training nodes included (rows of $\tilde{A}$), while fan-out size changes the number of non-zero neighbor entries in each row. Both hyperparameters approach the full-graph $\tilde{A}^{\text{full}}$ from different directions, forming the physical intuition for the subsequent analysis.

### Key Designs

**1. Decoupling Aggregation and Non-linear Activation**

To address the limitations of infinite-width or linear assumptions, the difficulty lies in the analytical intractability of ReLU when applied to aggregated features. The authors suggest "stripping" aggregation from the activation: either by rewriting the squared loss to extract aggregation from ReLU or by using a position-wise 0/1 indicator matrix to represent ReLU so it can multiply aggregated features directly. Consequently, the effect of the graph structure (determined by $b,\beta$) in the loss and gradient can be isolated and tracked without relying on infinite-width assumptions.

**2. Convergence Analysis & Obs.1: Batch size dominates optimization dynamics more than fan-out size**

The authors derive upper bounds for the number of iterations $T$ required to reach $\epsilon$ error under MSE (Theorem 1) and CE (Theorem 2). For example, under MSE, $T = O(n_{\text{train}} h^2 b^{5/2}\beta^{-1/2}\epsilon^{-1}\log(h^2\epsilon^{-1}))$. As $\beta\to d_{\max}$ and $b\to n_{\text{train}}$, the bound reduces to the full-graph result. Key observations include: Remark 3.1 notes that **increasing batch size requires more iterations under MSE but fewer under CE** (opposing directions, unlike DNNs), while **increasing fan-out size consistently reduces iterations under both losses**. Remark 3.2 uses the slope $|\partial T/\partial \beta|$ to show diminishing returns as $\beta$ increases. This leads to **Obs.1**: Optimization is more sensitive to batch size because $b$ induces opposite convergence trends across losses, whereas $\beta$ is monotonic. **Mechanism**: Concerns about convergence under memory constraints favor tuning fan-out size, where medium values (around 15) balance speed and cost.

**3. Generalization Analysis via Wasserstein Distance & Obs.2: Fan-out size dominates generalization more than batch size**

The authors provide a generalization bound for mini-batch + MSE in a PAC-Bayesian framework (Theorem 3): the difference between expected and empirical risk is $O\!\left(\frac{1}{n_{\text{train}}} + \Delta(\beta,b)\right)$. The core involves using **Wasserstein distance** $\Delta(\beta,b)$ to quantify the structural disparity between training and test graphs. The key term $\delta^{\text{full-mini}}_i = \lVert \tilde{a}^{\text{full}}_{\text{train},i} - \tilde{a}^{\text{mini}}_{\text{train},i}\rVert_F^2$ characterizes the aggregation difference for each node. Remark 4.1 shows that while increasing either parameter improves generalization, increasing $\beta$ re-incorporates missing but existing edges, making zero-entries non-zero and introducing complex fluctuations. Increasing $b$ simply sums more nodes. Thus, **Obs.2**: Generalization is more sensitive to fan-out size as $\beta$ directly controls the receptive field of each node. **Mechanism**: Concerning generalization under memory constraints, tuning batch size is more stable (fewer fluctuations).

**4. iteration-to-accuracy: A hardware-agnostic convergence metric**

Traditional time-to-accuracy depends heavily on hardware, entangling "iterative performance gain" with "computational throughput." The authors propose **iteration-to-accuracy** (number of iterations to reach a target validation accuracy) as a hardware-independent metric. Empirically, its variance across hardware is 41.28%, compared to 2787.05% for time-to-accuracy. This allows practitioners to narrow hyperparameter ranges based on trends before considering hardware-specific runtime.

### Loss & Training
Full-graph training uses Gradient Descent (GD): $W^{\text{full}}_{t+1} = W^{\text{full}}_t - \eta_t \nabla \hat{L}_{\text{train}}$. Mini-batch training uses SGD with stochastic gradients: $\hat{G}_t = \frac1b\sum_{i\in\text{sampled}}\nabla l(W^{\text{mini}}_t, \tilde{a}^{\text{mini}}_{\text{train},i})$. Both Cross-Entropy (CE) and Mean Squared Error (MSE) are covered, as the impact of batch size on convergence depends on the loss function.

## Key Experimental Results

### Main Results
Validated on Reddit, ogbn-arxiv, ogbn-products, and ogbn-papers100M datasets with GCN, GraphSAGE, and GAT. The table below shows the best test accuracy for multi-layer GraphSAGE (without dropout) after grid search for graph hyperparameters:

| Dataset | Full-graph | Mini-batch | Gain |
|--------|---------|----------------|------|
| Reddit | 96.13 | **96.32** | MB +0.19 |
| Ogbn-arxiv | 70.96 | **71.16** | MB +0.20 |
| Ogbn-products | 77.92 | **78.80** | MB +0.88 |
| Ogbn-papers100M | **59.54** | 58.52 | Full +1.02 |

The gap between mini-batch and full-graph is consistently within 1.74%, with mini-batch outperforming in three cases—supporting the conclusion that full-graph training is not inherently superior.

### Ablation Study
| Analysis Dimension | Key Phenomenon | Explanation |
|---------|---------|------|
| iteration-to-acc vs time-to-acc | Variance 41.28% vs 2787.05% | Iteration-to-acc is a reliable hardware-agnostic metric |
| Batch size impact | Opposite trends in MSE vs CE | Validates Obs.1: Convergence is batch-size sensitive |
| Fan-out size impact | Frequent/sharp accuracy fluctuations | Validates Obs.2: Generalization is fan-out sensitive |
| Large fan-out / batch (CE) | Degradation when fan-out > 15 or batch > 50% nodes | Large fan-out causes more severe degradation than batch |

### Key Findings
- **Non-isotropy is the central theme**: Batch size governs convergence while fan-out size governs generalization. They act differently across dimensions and must be tuned separately.
- **Performance degradation under CE**: Large batch sizes lead to sharp minima, while large fan-out sizes lead to overfitting due to excessive neighbor aggregation. Both harm generalization, but fan-out is more damaging. MSE is less prone to this as its gradients are weaker near the decision boundary, favoring flatter minima.
- **Computational Efficiency**: Throughput increases with batch size (fixed costs amortized) but decreases with fan-out size (increased message-passing calculation). Overall, mini-batch efficiency is superior.
- **Tuning Advice**: For datasets with an average degree < 50, it is recommended to keep batch size below half the training nodes and fan-out under 15 to balance performance and efficiency.

## Highlights & Insights
- **Reduction of Paradigms to Hyperparameters**: Viewing full-graph as the limit of mini-batch $b, \beta$ is a clean perspective that transforms empirical hardware-dependent debates into provable theoretical problems.
- **Utility of Non-isotropy**: The conclusion that batch size manages convergence and fan-out manages generalization allows for targeted tuning decisions under memory limits.
- **Portability of iteration-to-accuracy**: Decoupling performance gains from hardware throughput provides a reusable framework for hardware-agnostic efficiency evaluation.
- **Aggregation-ReLU Decoupling**: Using indicator matrices to isolate the graph structure from non-linearity is a key technique for future GNN training dynamics theory.

## Limitations & Future Work
- **Theoretical scope**: Analysis is primarily on single-layer GNNs. While multi-layer experiments align, a rigorous multi-layer theory is needed for complex optimization dynamics.
- **Task variety**: Focuses on transductive node classification. Transferability to tasks like link prediction is not yet verified.
- **Loss/Activation variety**: Analysis is centered on ReLU and MSE/CE; other functions remain future work.
- **Generalization bounds**: As upper bounds, they provide directional guidance, but small non-monotonic fluctuations in $\Delta(\beta,b)$ may complicate precision in tuning.

## Related Work & Insights
- **vs. Bajaj et al.**: They perform purely empirical performance comparisons. This work supplements them with theoretical insights into hyperparameter trade-offs.
- **vs. Yuan et al. / Hu et al.**: Yuan lacks theoretical support; Hu only uses gradient variance to explain batch size while ignoring fan-out. This paper models both and their interaction, showing why standard DNN gradient variance explanations do not directly apply to GNNs.
- **vs. DNN Batch Theory**: Due to message-passing, the DNN conclusion that "large batches speed up convergence" does not trivially hold (e.g., the MSE counter-example in Obs.1).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Out-of-Distribution Graph Models Merging](out-of-distribution_graph_models_merging.md)
- [\[ICLR 2026\] DHG-Bench: A Comprehensive Benchmark for Deep Hypergraph Learning](dhg-bench_a_comprehensive_benchmark_for_deep_hypergraph_learning.md)
- [\[ICML 2025\] Does Graph Prompt Work? A Data Operation Perspective with Theoretical Analysis](../../ICML2025/graph_learning/does_graph_prompt_work_a_data_operation_perspective_with_theoretical_analysis.md)
- [\[ICLR 2026\] Adaptive Mixture of Disentangled Experts for Dynamic Graph Out-of-Distribution Generalization](adaptive_mixture_of_disentangled_experts_for_dynamic_graph_out-of-distribution_g.md)
- [\[ICML 2025\] LLM Enhancers for GNNs: An Analysis from the Perspective of Causal Mechanism Identification](../../ICML2025/graph_learning/llm_enhancers_for_gnns_an_analysis_from_the_perspective_of_causal_mechanism_iden.md)

</div>

<!-- RELATED:END -->
