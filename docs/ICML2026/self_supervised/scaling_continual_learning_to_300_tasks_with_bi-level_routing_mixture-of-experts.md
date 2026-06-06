---
title: >-
  [Paper Note] Scaling Continual Learning to 300+ Tasks with Bi-Level Routing Mixture-of-Experts
description: >-
  [ICML 2026][Self-Supervised Learning][Class-Incremental Learning] The authors propose CaRE: a **Bi-Level Routing Mixture-of-Experts (BR-MoE)** module inserted into each block of a ViT. It first selects Top-M relevant tas…
tags:
  - "ICML 2026"
  - "Self-Supervised Learning"
  - "Class-Incremental Learning"
  - "Bi-Level Routing"
  - "Mixture-of-Experts"
  - "Long Task Sequences"
  - "OmniBenchmark-1K"
date: 2026-05-08
content_hash: cac34eb46f2d1f91
---

# Scaling Continual Learning to 300+ Tasks with Bi-Level Routing Mixture-of-Experts

**Conference**: ICML 2026  
**arXiv**: [2602.03473](https://arxiv.org/abs/2602.03473)  
**Code**: https://github.com/LMMMEng/CaRE (Available)  
**Area**: Continual Learning / Class-Incremental Learning / MoE / Parameter-Efficient Fine-Tuning  
**Keywords**: Class-Incremental Learning, Bi-Level Routing, Mixture-of-Experts, Long Task Sequences, OmniBenchmark-1K

## TL;DR
The authors propose CaRE: a **Bi-Level Routing Mixture-of-Experts (BR-MoE)** module inserted into each block of a ViT. It first selects Top-M relevant task routers using entropy via "class perceptors," then activates Top-K task experts through these routers, overlaid with a shared EMA expert. This allows the model to retain old knowledge while absorbing new classes even as the task sequence extends to 300+, filling the gap in "long-sequence CIL" research (and introducing the OmniBenchmark-1K dataset with 1000 classes).

## Background & Motivation

**Background**: Class-incremental learning (CIL) based on Pre-trained Models (PTMs) is a highly active research area, primarily focused on prompt-based methods (L2P, DualPrompt, CODA-Prompt) and adapter-based methods (EASE, APER, SEMA, MOS, TUNA, MIN). The latter typically train a task-specific adapter for each task and activate the appropriate one during inference.

**Limitations of Prior Work**: (1) Individual adapters are only discriminative for the specific classes they transition on; as task sequences grow, distinguishing similar classes across tasks (e.g., animal sub-classes in different tasks) degrades. (2) Existing methods either use coarse-grained "global aggregation of all historical adapters" or a single adapter, failing to extract fine-grained supplementary knowledge from related historical tasks. (3) Most CIL methods are evaluated on only 5–20 tasks. Many crash under ultra-long sequences (hundreds of tasks), and the community lacks a benchmark capable of supporting 100+ tasks—CIFAR-100 becomes trivial when split into 100 tasks, and ImageNet often overlaps with PTM training sets.

**Key Challenge**: To create feature representations that are both discriminative and comprehensive, one must (a) identify which tasks a sample potentially belongs to, (b) fuse adapter knowledge from these tasks fine-grainedly at every layer, and (c) maintain "cross-task universal" shared knowledge. While some methods address these traits individually, a unified architecture capable of **bi-level "routing-then-expert" decisions** at each layer is missing.

**Goal**: (i) Design a PEFT module for fine-grained cross-task knowledge retrieval at each layer; (ii) Ensure scalability to 300+ tasks; (iii) Provide a benchmark that truly tests long-sequence scalability.

**Key Insight**: The authors decompose the MoE router into two levels: coarse (by task) + fine (by adapter expert). They use the "entropy of task-specific classification heads" as a signal for "how certain the model is that the sample belongs to this task." This observation is critical: low entropy indicates high confidence and task relevance, which is more robust than direct "task ID prediction."

**Core Idea**: Inject a **triplet of (Class Perceptor $C_t$, Router $R_t$, Expert $E_t$)** into each ViT block, adding a new triplet for every new task. During inference, Top-M routers are selected by entropy; each router then selects Top-K experts via gating, complemented by a shared expert maintained via EMA. This bi-level routing replaces the dualistic designs of "global aggregation" vs "single adapter."

## Method

### Overall Architecture
The backbone is a frozen ViT-B/16 (pre-trained on ImageNet-21K). Each Transformer block is modified: $z_a = \text{MHSA}(\text{Norm}_1(z)) + z$, $z_f = \text{FFN}(\text{Norm}_2(z_a)) + z_a$, and finally $z' = \text{BR-MoE}(z_a) + z_f$. For each new task $t$ in incremental learning: (1) A new triplet $(C_t, R_t, E_t)$ is added to the BR-MoE in each block; (2) Only the current triplet and the shared expert $\bar{E}$ are updated (all other parameters remain frozen); (3) Inference uses the bi-level process to dynamically aggregate outputs from each layer. Final classification utilizes a concatenated angular margin head $W_t = [w^1, \dots, w^t]$, with class logits computed as cosine similarity $\cos(\theta_i^j) = \frac{w_j^t \cdot \phi^t(x_i^t)}{\|w_j^t\| \|\phi^t(x_i^t)\|}$ and a scaling factor $\tau = 20$.

### Key Designs

1.  **Bi-Level Routing: Dynamic Route Selection**:
    -   **Function**: Select the Top-M most relevant historical task routers for each layer's input.
    -   **Mechanism**: The [CLS] token of $z_a$ is fed to each task's class perceptor $C_t = \rho^t \in \mathbb{R}^{d \times |G^t|}$ to obtain a class distribution $s_t = \text{Softmax}(C_t(z_a^{[CLS]}))$. The entropy is calculated as $\mathcal{H}_t = -\sum_j s_t^{(j)} \log s_t^{(j)}$. The Top-M routers $R_t$ corresponding to the lowest entropy values are selected. Low entropy indicates the classifier is certain about the input, implying the input likely belongs to that task. This entropy-based selection is more robust to train-inference distribution shifts than task ID prediction. During training, the latest task router $R_T$ is always included, while it is dynamically selected during inference.
    -   **Design Motivation**: Hard-selection of a task via a single task classifier is fragile, while global aggregation dilutes relevance. Ranking by entropy + Top-M balances robustness and focus, allowing local decisions at each layer.

2.  **Bi-Level Routing: Dynamic Expert Routing + Shared EMA Expert**:
    -   **Function**: Within the selected M routes, fine-grained selection of Top-K adapter experts combined with a cross-task shared expert.
    -   **Mechanism**: Each selected router $R_t$ is a linear layer $\eta^t \in \mathbb{R}^{d \times t} + \text{Softmax}$ that produces $t$ gating scores for $z_a^{[CLS]}$. The Top-K are selected and re-normalized to get $\{a_i\}$, and corresponding adapter $E_i$ outputs are weighted. For instance, with M=2, K=2: $z_1 = a_2 E_2(z_a) + a_t E_t(z_a)$, $z_2 = b_{T-1} E_{T-1}(z_a) + b_T E_T(z_a)$, $z_r = z_1 + z_2$. A shared expert $\bar{E}$, trained on the first task and maintained via EMA $\delta_s \leftarrow \mu \delta_s + (1 - \mu)\delta_t$ ($\mu = 0.999$), provides the final output $z_o = z_r + \bar{E}(z_a)$. Defaults are M=2, K=3, with 16-dim bottleneck for task adapters and 64-dim for the shared adapter.
    -   **Design Motivation**: Selecting tasks is insufficient; specific adapters within those tasks must be activated. The shared expert provides a universal cross-task prior (inspired by DeepSeek-MoE).

3.  **Layer-wise Class Perceptor Supervision**:
    -   **Function**: Ensures intermediate class perceptors generate reliable entropy signals.
    -   **Mechanism**: An auxiliary loss $\mathcal{L}_{cp}^\ell = \mathcal{L}_{cls}^\ell + \mathcal{L}_{KL}^\ell$ is added to each layer's $C_t$. $\mathcal{L}_{cls}^\ell$ is the angular margin loss, and $\mathcal{L}_{KL}^\ell$ is the KL divergence between $s_t$ and the final layer's softmax output $p_t$, encouraging shallow perceptors to mimic high-semantic deep distributions. Total objective: $\mathcal{L} = \mathcal{L}_{cls} + \lambda \frac{1}{L}\sum_\ell \mathcal{L}_{cp}^\ell$ ($\lambda = 1$).
    -   **Design Motivation**: Since BR-MoE makes independent routing decisions per block, entropy must reflect task relevance even in shallow layers where features might be weak. KL distillation aligns each layer with the final decision.

### Loss & Training
When a new task $t$ arrives, all historical parameters are frozen. Only $(C_t, R_t, E_t)$ of the current layer and the shared expert $\bar{E}$ are trained. Optimizer: SGD (momentum=0.9, weight decay=5e-4), batch size=16, 20 epochs per task, and lr=0.01 with cosine annealing. $R_T$ is forced to activate during training to prevent cold starts.

## Key Experimental Results

### Main Results
Comparison on the new OmniBenchmark-1K (1000 classes / 190k images / 21 domains) long-sequence settings. Metrics: $\bar{\mathcal{A}}$ (Average Accuracy) / $\mathcal{A}_B$ (Last Accuracy):

| Method | 100 tasks (B0 Inc10) $\mathcal{A}_B$ | 200 tasks (B0 Inc5) $\mathcal{A}_B$ | 151 tasks (B100 Inc6) $\mathcal{A}_B$ | 301 tasks (B100 Inc3) $\mathcal{A}_B$ |
| :--- | :--- | :--- | :--- | :--- |
| L2P | 48.87 | 45.25 | 10.49 | 9.03 |
| DualPrompt | 49.45 | 45.62 | 12.90 | 9.30 |
| APER-Adapter | 62.24 | 61.53 | 62.99 | 62.99 |
| TUNA | 60.04 | 59.14 | 62.77 | 62.21 |
| MOS | 64.27 | 63.51 | 65.20 | 64.37 |
| MIN | 63.60 | 62.50 | 60.33 | 59.63 |
| **CaRE** | **68.27** | **67.46** | **69.01** | **68.51** |

On the longest sequence of 301 tasks, CaRE outperforms MOS by 4 points and maintains a gap of dozens of points over prompt-based methods (which collapsed to ~9%). On short-sequence CIL (CIFAR-100, ImageNet-R/-A, etc.), CaRE remains mostly SOTA.

### Ablation Study

| Configuration | Change in Key Metric (OmniBenchmark-1K) | Description |
| :--- | :--- | :--- |
| Full CaRE | 67.46 | Complete model |
| Single Router (M=1) | Significant Decrease | Validates the need for multiple active routes |
| No Shared Expert | Decrease | EMA shared expert handles cross-task knowledge |
| Hard Task Pred instead of Entropy | Decrease | Validates that entropy is more robust |
| No Intermediate KL Supervision | Decrease | Shallow entropy signals become unreliable |

### Key Findings
- **Bi-level routing > single routing**: Selecting Top-M tasks then Top-K experts per task is far superior to a single gating over all adapters.
- **Entropy > Task ID Prediction**: Entropy reflects the overall uncertainty of the head, making it more stable than hard argmax.
- **Shared expert as a safety net**: In long sequences, new samples might not match any task-specific adapters; the EMA shared expert provides foundational features.
- **Degradation of prior SOTA**: Methods like MIN and MOS are competitive in short sequences but degrade significantly at 100+ tasks.

## Highlights & Insights
- **Layered MoE Routing**: Coarse-fine hierarchical routers with clear semantic meaning (Task-level / Expert-level) are naturally suited for long-sequence CIL.
- **Entropy as a Router Signal**: Using the confidence of task-specific heads as a relevance measure avoids the fragility of a global task classifier.
- **OmniBenchmark-1K Contribution**: Provides a necessary standard for testing long-sequence CIL robustness without data leakage from PTM training sets.
- **Local Per-Layer Decisions**: BR-MoE allows each block to make context-aware decisions independently, providing more targeted aggregation than a single global step.

## Limitations & Future Work
- **Linear Parameter Growth**: Adding $(C_t, R_t, E_t)$ for each task results in linear growth. At 301 tasks, calculating entropy for all $C_t$ at every layer increases inference complexity.
- **Task Boundaries**: CaRE assumes clear boundaries to train independent triplets and may require a task detection mechanism for task-free CL.
- **Shared EMA Hyperparameter**: The fixed $\mu = 0.999$ might not adapt well to drastic distribution shifts.
- **Formal Analysis**: Lack of formal analysis on forgetting patterns or specifically how routing weights evolve for forgotten classes.

## Related Work & Insights
- **vs MOS / TUNA / MIN**: These are strong in short-to-medium sequences but fail in long sequences; CaRE's bi-level routing decouples task and expert selection for stability.
- **vs DeepSeek-MoE**: The shared expert design is directly inspired by DeepSeek-MoE, adapted for CIL with EMA maintenance.
- **vs Prompt-based methods**: Prompt pools lack sufficient capacity for hundreds of tasks; adapter-based MoE is a more scalable path.

## Rating
- Novelty: ⭐⭐⭐⭐ (Combination of bi-level routing, entropy signals, and EMA shared experts is fresh in CIL context).
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Extensive long-sequence settings and a new benchmark).
- Writing Quality: ⭐⭐⭐⭐ (Clear motivation and intuitive diagrams).
- Value: ⭐⭐⭐⭐⭐ (Scales PTM-based CIL to 300+ tasks with sustained performance gains).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Soft Task-Aware Routing of Experts for Equivariant Representation Learning](../../NeurIPS2025/self_supervised/soft_task-aware_routing_of_experts_for_equivariant_representation_learning.md)
- [\[ICML 2026\] Learning to Extrapolate to New Tasks: A Relational Approach to Task Extrapolation](learning_to_extrapolate_to_new_tasks_a_relational_approach_to_task_extrapolation.md)
- [\[ICML 2026\] PartCo: Part-Level Correspondence Priors Enhance Category Discovery](partco_part-level_correspondence_priors_enhance_category_discovery.md)
- [\[ICML 2026\] LEC: Linear Expectation Constraints for Selection-Conditioned Risk Control in Selective Prediction and Routing Systems](lec_linear_expectation_constraints_for_selection-conditioned_risk_control_in_sel.md)
- [\[ACL 2026\] LLMSurgeon: Diagnosing Data Mixture of Large Language Models](../../ACL2026/self_supervised/llmsurgeon_diagnosing_data_mixture_of_large_language_models.md)

</div>

<!-- RELATED:END -->
