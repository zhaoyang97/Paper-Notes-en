---
title: >-
  [Paper Note] TreeLoRA: Efficient Continual Learning via Layer-Wise LoRAs Guided by a Hierarchical Gradient-Similarity Tree
description: >-
  [ICML 2025][Model Compression][Continual Learning] This paper proposes TreeLoRA, which organizes LoRA adapters of historical tasks by constructing a hierarchical K-D tree based on gradient similarity. It utilizes the Lower Confidence Bound (LCB) multi-armed bandit algorithm to search for the most relevant task branches efficiently for knowledge sharing. Working in tandem with sparse gradient updates, it achieves a 3.2× speedup on ViT and a 2.4× speedup on LLMs…
tags:
  - "ICML 2025"
  - "Model Compression"
  - "Continual Learning"
  - "LoRA"
  - "Hierarchical Tree Structure"
  - "Gradient Similarity"
  - "Multi-Armed Bandit"
date: 2026-05-08
content_hash: 3b32f6aa2767fab9
---

# TreeLoRA: Efficient Continual Learning via Layer-Wise LoRAs Guided by a Hierarchical Gradient-Similarity Tree

**Conference**: ICML 2025  
**arXiv**: [2506.10355](https://arxiv.org/abs/2506.10355)  
**Code**: [https://github.com/ZinYY/TreeLoRA](https://github.com/ZinYY/TreeLoRA)  
**Area**: Model Compression / Continual Learning / Parameter-Efficient Fine-Tuning  
**Keywords**: Continual Learning, LoRA, Hierarchical Tree Structure, Gradient Similarity, Multi-Armed Bandit

## TL;DR

This paper proposes TreeLoRA, which organizes LoRA adapters of historical tasks by constructing a hierarchical K-D tree based on gradient similarity. It utilizes the Lower Confidence Bound (LCB) multi-armed bandit algorithm to search for the most relevant task branches efficiently for knowledge sharing. Working in tandem with sparse gradient updates, it achieves a 3.2× speedup on ViT and a 2.4× speedup on LLMs, while maintaining or surpassing SOTA performance.

## Background & Motivation

**Background**: The three major paradigms in the Continual Learning (CL) field—regularization methods (EWC), replay methods (GEM), and architectural methods (HiDeLoRA)—all face efficiency issues in the era of Large Pre-trained Models (LPMs). The computational complexity of existing methods typically scales linearly with the number of tasks.

**Limitations of Prior Work**: The core challenge of CL in the era of large models is efficiency. Existing CL methods either have a computational complexity of $O(N)$ that grows with the number of tasks (requiring reviews of gradients from all historical tasks), or fail to fully utilize shared knowledge among tasks to accelerate adaptation. Although hierarchical methods like HiDeLoRA perform well, their training times remain long.

**Key Challenge**: Estimating task similarity requires computing gradients for all historical tasks ($O(N)$ complexity), yet efficient CL must avoid this linear complexity. On the other hand, directly training a LoRA independently for each new task (SeqLoRA) leads to severe catastrophic forgetting.

**Goal**: How to efficiently identify the most similar historical task to the current task and reuse its knowledge without traversing all historical tasks?

**Key Insight**: The authors observe a hierarchical similarity structure among tasks—shallow features are shared across tasks, whereas deep features are task-specific. Explicitly modeling this structure as a K-D tree allows replacing linear search with tree search, reducing complexity from $O(\sqrt{N})$ to $O(\sqrt{\log N})$.

**Core Idea**: Utilizing a hierarchical gradient-similarity tree in combination with a multi-armed bandit algorithm to reduce the task similarity search complexity in continual learning from linear to logarithmic.

## Method

### Overall Architecture

TreeLoRA maintains a dynamically growing K-D tree, where the root node represents all tasks and leaf nodes correspond to individual task LoRA adapters. When a new task arrives, the LCB bandit algorithm searches along the tree to find the branch with the most similar gradient direction, loads the corresponding LoRA adapter as initialization, and then adapts to the new task via sparse gradient updates. Once the task is completed, the newly learned adapter is inserted into the tree, and the structure is updated.

### Key Designs

1. **Gradient-Similarity Tree Structure**:

    - **Function**: Organizes historical tasks into a hierarchical tree based on the similarity of their gradient directions.
    - **Mechanism**: Uses the $L_1$ norm to measure the difference between gradient directions of two tasks; tasks within a threshold $\delta$ are grouped together. Adapters corresponding to shallow nodes in the tree capture low-level shared task features, while deep nodes capture high-level task-specific semantics. The threshold $\delta$ does not require manual tuning; rather, it is automatically split using the median during construction, similar to a K-D tree: $\mathcal{N}_j = \max\{\mathcal{N} \subseteq [N]: \|\mathbf{g}_i - \mathbf{g}_{i'}\|_1 \leq \delta, \forall i,i' \in \mathcal{N}\}$
    - **Design Motivation**: Transformer models inherently possess a hierarchical structure (shallow layers are general, deep layers are specific). The tree structure naturally aligns with the model's hierarchical characteristics, capturing multi-grained relationships between tasks better than a flat list of tasks.

2. **Lower Confidence Bound Bandit**:

    - **Function**: Efficiently finds the most relevant task branch without computing gradients for all historical tasks.
    - **Mechanism**: Models the choice of which historical task to use for similarity estimation as a multi-armed bandit problem. When each training sample arrives, only one historical task is selected to compute gradient discrepancy. The LCB for a leaf node is $\text{LCB}_k = \hat{\mu}_k - 2\sqrt{\frac{\log t}{n_k}}$, while non-leaf nodes take the optimal value of their children's LCBs minus $\delta$. The branch with the minimum LCB is selected for exploration.
    - **Design Motivation**: Reduces the search complexity from traversing all $N$ tasks to $O(1)$ along the tree search; theoretically, the regret is reduced from $O(\sqrt{N})$ to $O(\sqrt{\log N})$.

3. **Sparse Gradient Updates**:

    - **Function**: Only updates the subset of parameters most relevant to the new task.
    - **Mechanism**: After selecting the most relevant branch, an adaptive regularization term $\ell_{\text{reg}}^t = \|\hat{\mathbf{g}}_n^t - \hat{\mathbf{g}}_k^t\|_1$ is used to identify the parameters that need to be updated. The update formula is given by $\mathbf{w}_n^{t+1} = \mathbf{w}_n^t - \alpha \cdot \mathcal{S}[\nabla\ell] - \lambda \cdot \nabla\ell_{\text{reg}}^t$, where $\mathcal{S}$ is a low-rank sparsification function (implemented in practice using LoRA adapters).
    - **Design Motivation**: Sparse updates reduce both computational and storage overheads (requiring only low-rank adapters to be maintained), and mitigate catastrophic forgetting by modifying only the relevant parameters.

### Loss & Training

Total Loss = Task Loss + $\lambda \times$ Gradient Discrepancy Regularization. The tree depth is set to 5 for ViT and 64 for LLMs. After each task, the task-specific LoRA adapter is stored and inserted into the nearest branch via depth-first search. During inference, the most recently learned adapter is utilized.

## Key Experimental Results

### Main Results (ViT)

| Dataset | Method | Accuracy (%) ↑ | BWT (%) ↓ | Training Time (s) ↓ |
|--------|------|-------------|-----------------|--------------|
| Split CIFAR-100 | HiDeLoRA | 88.46±0.04 | 4.33±0.41 | 692±7 |
| Split CIFAR-100 | TreeLoRA | **88.54±0.05** | 4.37±0.15 | **214±4** |
| Split ImageNet-R | HiDeLoRA | **72.28±0.15** | 4.16±0.05 | 796±9 |
| Split ImageNet-R | TreeLoRA | 71.94±0.47 | 4.06±0.40 | **260±5** |
| Split CUB-200 | HiDeLoRA | 73.48±1.35 | 5.38±0.21 | 194±3 |
| Split CUB-200 | TreeLoRA | **73.66±0.22** | **4.87±0.30** | **86±3** |

### Main Results (LLM, TRACE Dataset)

| Model | Method | OP (%) ↑ | BWT (%) ↓ | Training Time (s) ↓ |
|------|------|---------|----------|--------------|
| Mistral-7B | HiDeLoRA | 51.81±0.9 | 6.25±0.3 | 1288±28 |
| Mistral-7B | TreeLoRA | **54.77±1.1** | **3.77±0.4** | **510±20** |
| LLaMA-2-7B | HiDeLoRA | 41.60±0.8 | 7.12±0.4 | 1286±27 |
| LLaMA-2-7B | TreeLoRA | **43.52±1.0** | **3.46±0.4** | **485±18** |
| Gemma-2B | O-LoRA | **33.73±0.8** | 12.36±0.4 | 1302±29 |
| Gemma-2B | TreeLoRA | 33.41±0.9 | **8.50±0.5** | **510±15** |

### Ablation Study

| Configuration | Accuracy (%) | BWT (%) | Training Time (s) | Description |
|------|-----------|---------|-------------|------|
| TreeLoRA (2 epochs) | 87.73±0.12 | 4.13±0.25 | 23 | Effective with minimal training |
| TreeLoRA (10 epochs) | 88.23±0.16 | 4.29±0.11 | 108 | Moderate training |
| TreeLoRA (20 epochs) | 88.54±0.05 | 4.37±0.15 | 214 | Complete training |

### Key Findings

- TreeLoRA achieves a 3.2× speedup on ViT (CUB-200: 194s to 86s) and a 2.4× speedup on LLMs (Mistral: 1288s to 510s), while maintaining comparable or superior performance.
- An accuracy of 87.73% is achieved with only 2 training epochs (vs 88.54% for 20 epochs), demonstrating the effective reuse of shared task knowledge.
- Forgetting is significantly reduced in the LLM setting (Mistral: 6.25% to 3.77%), which indicates that the tree structure effectively manages knowledge conflicts between tasks.
- SeqLoRA (training LoRA independently for each task) suffers from severe catastrophic forgetting (97.62% BWT on CIFAR-100), validating the necessity of knowledge sharing.

## Highlights & Insights

- **Tree Structure Aligned with Transformer Hierarchy**: Shallow-layer LoRAs in the tree capture task-shared features, while deep-layer LoRAs capture task-specific features, which naturally aligns with the general-to-specific hierarchical characteristic of Transformers. This design paradigm can be extended to other scenarios involving hierarchical knowledge management.
- **Bandit Theoretical Guarantees**: Modeling task search as a bandit problem and proving under the smooth tree assumption that regret improves from $O(\sqrt{N})$ to $O(\sqrt{\log N})$ makes this one of the few works in continual learning to offer solid theoretical guarantees.
- **Automatic Threshold Selection**: Borrowing the median-split idea of K-D trees to automatically determine thresholds avoids manual hyperparameter tuning for gradient similarity metrics.

## Limitations & Future Work

- The tree depth (ViT=5, LLM=64) must be manually set as a hyperparameter and cannot exceed the number of Transformer layers; extremely deep models might require a more flexible, adaptive depth mechanism.
- The OP on Gemma-2B is slightly lower than that of O-LoRA (33.41 vs 33.73), indicating that its applicability to smaller models may be somewhat limited.
- The paper assumes that tasks arrive sequentially with known task boundaries (task-incremental setting); validation under more challenging settings, such as class-incremental or domain-incremental setups, is needed.
- Currently, sparse updates are only realized via LoRA; exploring whether this can be generalized to other PEFT methods (e.g., AdaLoRA, Prefix-Tuning) remains an interesting future direction.

## Related Work & Insights

- **vs HiDeLoRA/HiDePrompt**: While both are hierarchical decomposition methods, the HiDe series relies on predefined, fixed hierarchies, whereas TreeLoRA constructs the tree dynamically, thereby adaptively fitting the task stream's structure.
- **vs O-LoRA**: O-LoRA employs orthogonal subspace allocation to prevent forgetting, whereas TreeLoRA achieves a similar objective via gradient similarity clustering, yielding an additional speedup advantage.
- **vs GEM**: GEM relies on storing historical data to constrain gradient directions, which incurs heavy computational overhead (CIFAR-100: 22456s vs 214s for TreeLoRA). Conversely, TreeLoRA does not require storing raw data, requiring only the maintenance of lightweight LoRA adapters.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of a tree structure and a bandit algorithm is novel in continual learning, backed by solid theoretical analysis.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers both ViT and LLMs, three vision datasets plus the NLP TRACE dataset, and three distinct LLMs.
- Writing Quality: ⭐⭐⭐⭐ Direct method motivation, with thorough theoretical and experimental logical progression.
- Value: ⭐⭐⭐⭐ Successfully addresses both efficiency and performance bottlenecks in continual fine-tuning of LLMs—a direction with immense practical demand.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Quantized Gradient Projection for Memory-Efficient Continual Learning](../../ICLR2026/model_compression/quantized_gradient_projection_for_memory-efficient_continual_learning.md)
- [\[ICML 2025\] BECAME: BayEsian Continual Learning with Adaptive Model MErging](became_bayesian_continual_learning_with_adaptive_model_merging.md)
- [\[ICCV 2025\] PLAN: Proactive Low-Rank Allocation for Continual Learning](../../ICCV2025/model_compression/plan_proactive_low-rank_allocation_for_continual_learning.md)
- [\[CVPR 2025\] LoRA Subtraction for Drift-Resistant Space in Exemplar-Free Continual Learning](../../CVPR2025/model_compression/lora_subtraction_for_drift-resistant_space_in_exemplar-free_continual_learning.md)
- [\[NeurIPS 2025\] REP: Resource-Efficient Prompting for Rehearsal-Free Continual Learning](../../NeurIPS2025/model_compression/rep_resource-efficient_prompting_for_rehearsal-free_continual_learning.md)

</div>

<!-- RELATED:END -->
