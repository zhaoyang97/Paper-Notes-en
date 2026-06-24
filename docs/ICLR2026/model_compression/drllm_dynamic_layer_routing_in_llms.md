---
title: >-
  [Paper Note] Dr.LLM: Dynamic Layer Routing in LLMs
description: >-
  [ICLR 2026][Model Compression][Dynamic Layer Routing] Lightweight routers are attached to each layer of a frozen pretrained LLM to decide whether a layer should be "skipped, executed, or repeated." Using high-quality paths searched via offline MCTS as supervision, this approach improves accuracy and saves computation without modifying base weights or requiring inference-time search.
tags:
  - "ICLR 2026"
  - "Model Compression"
  - "Dynamic Layer Routing"
  - "Adaptive Depth"
  - "Layer Skip/Repeat"
  - "MCTS Supervision"
  - "Frozen LLM Retrofitting"
date: 2026-05-08
content_hash: 959ecbc3321b87df
---

# Dr.LLM: Dynamic Layer Routing in LLMs

**Conference**: ICLR 2026  
**Code**: [https://github.com/parameterlab/dr-llm](https://github.com/parameterlab/dr-llm)  
**Area**: Model Compression / Adaptive Depth Inference  
**Keywords**: Dynamic Layer Routing, Adaptive Depth, Layer Skip/Repeat, MCTS Supervision, Frozen LLM Retrofitting  

## TL;DR
Lightweight routers are attached to each layer of a frozen pretrained LLM to decide whether a layer should be "skipped, executed, or repeated." Using high-quality paths searched via offline MCTS as supervision, this approach improves accuracy and saves computation without modifying base weights or requiring inference-time search.

## Background & Motivation
**Background**: LLMs typically process all transformer layers for every token regardless of input difficulty. This leads to computational waste on simple queries and a lack of "thinking deeper" flexibility for difficult reasoning problems. Prior works on adaptive depth include early-exit, layer pruning, looped blocks, dynamic routing, Mixture-of-Depth, MoE, and search-based routing (e.g., CoLa).

**Limitations of Prior Work**: Existing methods usually suffer from at least one of three drawbacks: (i) trading accuracy for speed, where computation is saved at the cost of performance; (ii) requiring architectural changes and large-scale retraining (e.g., FlexiDepth and MindSkip require hundreds of thousands of samples); (iii) depending on expensive search during inference, making them difficult to deploy at scale (e.g., CoLa requires MCTS and gold labels during inference).

**Key Challenge**: Developing a retrofittable framework that can be applied to existing frozen models, is cheap to train, requires zero additional search during inference, keeps base weights untouched, and truly improves accuracy rather than degrading it. No existing work satisfies all five criteria simultaneously.

**Goal**: Create a retrofittable framework providing lightweight routers for frozen LLMs to achieve budget-aware, accuracy-driven inference without modifying base weights or performing inference-time search.

**Core Idea**: **Explicitly supervised layer routing**. The decision to "skip, execute, or repeat" a layer is treated as an offline problem. A length-aware MCTS is used to search for execution paths for each sample that preserve or improve accuracy under computational budgets. These paths are converted into per-layer labels to train tiny routers via supervised learning. During inference, routers make greedy decisions, eliminating the need for search.

## Method

### Overall Architecture
Given an $L$-layer frozen decoder-only LLM $M=[B_1,\dots,B_L]$, Dr.LLM equips each block $B_\ell$ with a lightweight MLP router $r_\ell:\mathbb{R}^d\to\mathbb{R}^3$, which outputs logits for three actions: $\{\text{skip},\text{execute},\text{repeat}\}$. These determine if the layer is bypassed, executed once, or executed twice consecutively. Training involves two offline phases: first, length-aware MCTS searches for optimal paths $\pi^\star$ on ARC/DART samples (4k supervision samples) and converts them into per-layer labels $y^\star_\ell\in\{0,1,2\}$. Second, these labels are distilled into routers using focal loss. Throughout the process, base weights are frozen. During inference, routers make greedy decisions based on window-pooled hidden states, which is zero-search and compatible with KV cache.

```mermaid
flowchart TB
    subgraph Offline["Offline Supervision Generation (MCTS)"]
        A[ARC/DART sample q,a] --> B[Length-aware MCTS<br/>Search skip/exec/repeat path within budget]
        B --> C[Retain accuracy-preserving/improving paths π*<br/>~4k samples]
        C --> D[Convert to per-layer labels y*∈{0,1,2}]
    end
    subgraph Training["Router Training (Base Frozen)"]
        D --> E[Focal loss + Class re-balancing<br/>Teacher forcing on execute only]
        E --> F[Per-layer routers r_ℓ]
    end
    subgraph Inference["Inference (No Search)"]
        G[Input Sequence] --> H[Window mean-pool hidden states]
        H --> F2[r_ℓ Greedy decision]
        F2 --> I[skip→Identity / exec→×1 / repeat→×2]
    end
    F --> F2
```

### Key Designs
**1. Per-layer Routers with Skip/Execute/Repeat: Discretizing Adaptive Depth.** Each layer action $y_\ell\in\{\text{skip},\text{execute},\text{repeat}\}$ is defined such that skip makes $H^{(\ell)}=H^{(\ell-1)}$, execute applies $B_\ell$ once, and repeat applies it twice. The vector $y=(y_1,\dots,y_L)$ induces a customized execution path while base weights remain frozen. Unlike LayerSkip, which only supports skipping and requires retraining, Dr.LLM supports both "saving computation (skip)" and "increasing depth on demand (repeat)." Skip actions offset the layer count increase from looping, achieving true computation reallocation. The router is a simple Linear-GELU-Linear structure (width $h=128$), adding only $O(Ldh)$ parameters (11M for a 3B model, 0.14% of base). Training takes only 4 hours on a single A100.

**2. Window Mean-pooling: Stabilizing Decisions on Long Contexts.** Routers do not process individual tokens. Instead, the sequence is divided into $W$ continuous windows $\{S_w\}$. Mid-window means $m_w=\frac{1}{|S_w|}\sum_{t\in S_w}H^{(\ell-1)}_t$ are used to generate logits, which are then averaged to vote: $z_\ell=\frac{1}{W}\sum_w r_\ell(m_w),\ p_\ell=\mathrm{softmax}(z_\ell),\ \hat y_\ell=\arg\max_c p_{\ell,c}$ (default $W=8$). This suppresses decision jitter in long sequences and makes router overhead independent of the number of generated tokens, as decision-making happens only once per sequence.

**3. Length-aware MCTS for Offline Supervision: Searching for Labels.** For each $(q,a)$, the search starts from the default path $\pi_0=[1,\dots,L]$. Nodes represent "action at a specific layer," selected using UCB with a length penalty: $\mathrm{UCB}(\pi)=\frac{Q(\pi)}{v(\pi)}+c\sqrt{\frac{\ln V}{v(\pi)}}-\lambda\frac{|\pi|}{L}$ ($c=1.8, \lambda=3.0$). Constraints like "max 2 consecutive skips" and "path length $\le 2L$" control computational expansion. Each simulation performs a constrained forward pass, backpropagating a binary reward (correct/incorrect answer). Accuracy-preserving or improving paths are retained, prioritizing the shortest correct path. 

**4. Focal Loss + Class Rebalancing + Execute-only Teacher Forcing.** Since most layers are labeled "execute," naive training leads to a majority-class bias. Dr.LLM uses focal loss with effective-number weights $\alpha_c=\frac{1-\beta}{1-\beta^{n_c}}$: $L=-\frac{1}{L}\sum_\ell \alpha_{y^\star_\ell}(1-p_{\ell,y^\star_\ell})^\gamma\log p_{\ell,y^\star_\ell}$. During training, teacher forcing is applied only to "execute" actions to decouple routers. This prevents the formation of a serial dependency chain where $\text{router}_i$ depends on $\text{router}_{i-1}$, which would slow down training and drop accuracy.

## Key Experimental Results
Evaluation across six backbones: LLaMA-3.2 (3B/8B Instruct/Base) and Qwen-2.5 (3B/7B Instruct). Routers were trained in 4 hours on a single A100 40GB.

### Main Results (In-domain: ARC Logic + DART Math)

| Metric | Results |
|------|------|
| Accuracy Gain (All models improved) | Avg +2.25%p, Max +4.0%p |
| Avg Layers Saved | ~5.0 layers, Max 11.0 layers |
| 1k-token Generation Speedup | 15.3% (Router overhead < 1%) |
| vs. Prior SoTA Routing | Up to +7.7%p Accuracy |

### Ablation Study

| Setting | Effect |
|------|------|
| Repeat block size 4→1 | Search significantly faster; accuracy/efficiency unchanged; simulations 200→50 |
| Length penalty $\lambda$ 5→3 | Search samples decreased by 14.8%p |
| Removing "Execute-only teacher forcing" | Accuracy dropped by 1.7%; training slowed down |
| MCTS Statistics | 4k supervision samples / 24,330 candidate paths (all offline) |

### Key Findings
- **Strong OOD Generalization**: Routers migrated to OOD benchmarks (MMLU, GSM8k, GPQA, etc.) show an average accuracy drop of only 0.85%p while maintaining efficiency, indicating the learned routing strategy generalizes well.
- **Skip and Repeat Complementarity**: Repeat increases depth for hard reasoning, while skip saves computation in simple sections, achieving global computation reallocation based on difficulty.
- **Unique Fulfillment of Criteria**: It is the only method to simultaneously achieve accuracy gain, retrofittability, low inference cost, low training cost, and frozen base weights.

## Highlights & Insights
- **Reducing Adaptive Depth to Supervised Classification**: The difficulty of determining which layer to skip is offloaded to offline MCTS. Inference is reduced to a cheap forward pass, avoiding the "inference-time search + gold label" requirements of methods like CoLa.
- **Thorough Retrofit Philosophy**: With frozen base weights and only 0.1~0.6% new router parameters, it is deployment-friendly and compatible with KV cache—a rarity among layer routing methods.
- **The Power of "Repeat"**: While most adaptive depth methods only perform subtraction (early-exit/pruning/skipping), Dr.LLM allows doubling computation on critical layers, using skips to offset the cost.

## Limitations & Future Work
- **Supervision Tasks**: Supervision is derived only from ARC and DART. Whether 4k samples across two domains cover broader distributions (code, multi-lingual) remains to be verified.
- **MCTS Cost**: Although offline and one-time, 961k forward passes are significant and may scale poorly with larger models or longer contexts.
- **Action Constraints**: Hard limits on skips and repeats might restrict the deeper recursion needed for extremely complex problems.
- **Independent Decisions**: Decoupling routers via teacher forcing improves efficiency but sacrifices explicit coordination between layers.

## Related Work & Insights
- **vs. CoLa (Li et al. 2025)**: CoLa treats layers as modules and uses MCTS to find "chains of layers" but requires search and gold labels during inference. Dr.LLM moves MCTS offline.
- **vs. Early-exit / LayerSkip**: Early-exit requires additional classifiers; LayerSkip requires finetuning the base and cannot repeat layers.
- **vs. Mixture-of-Depth**: MoD performs routing at the token level but requires modifying base weights. Dr.LLM's sequence-level routing is complementary and could be combined with MoD for local redundancy control.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The combination of offline MCTS supervision, three-action routers, and frozen base weights is a clean solution to deployment pain points.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers six backbones, multiple in-domain/OOD benchmarks, and comparison with SoTA routing methods.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear definitions of actions, MCTS algorithms, and tables. Reproducibility is high.
- **Value**: ⭐⭐⭐⭐ — Being able to retrofit frozen LLMs in 4 hours on a single GPU to improve accuracy while saving computation is highly valuable for resource-constrained deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] LD-MoLE: Learnable Dynamic Routing for Mixture of LoRA Experts](ld-mole_learnable_dynamic_routing_for_mixture_of_lora_experts.md)
- [\[ICLR 2026\] Reassessing Layer Pruning in LLMs: New Insights and Methods](reassessing_layer_pruning_in_llms_new_insights_and_methods.md)
- [\[ICLR 2026\] Alignment-Enhanced Integration of Connectivity and Spectral Sparsity in Dynamic Sparse Training of LLM](alignment-enhanced_integration_of_connectivity_and_spectral_sparsity_in_dynamic_.md)
- [\[ICLR 2026\] GradPruner: Gradient-guided Layer Pruning Enabling Efficient Fine-Tuning and Inference for LLMs](gradpruner_gradient-guided_layer_pruning_enabling_efficient_fine-tuning_and_infe.md)
- [\[NeurIPS 2025\] DP-LLM: Runtime Model Adaptation with Dynamic Layer-wise Precision Assignment](../../NeurIPS2025/model_compression/dp-llm_runtime_model_adaptation_with_dynamic_layer-wise_precision_assignment.md)

</div>

<!-- RELATED:END -->
