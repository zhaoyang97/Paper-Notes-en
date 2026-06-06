---
title: >-
  [Paper Note] Scaling Continual Learning to 300+ Tasks with Bi-Level Routing Mixture-of-Experts
description: >-
  [ICML 2026][Model Compression][Class-Incremental Learning] The authors propose CaRE: inserting a **bi-level routing MoE (BR-MoE)** into each ViT block—first…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Class-Incremental Learning"
  - "Bi-Level Routing"
  - "Mixture-of-Experts"
  - "Long Task Sequences"
  - "OmniBenchmark-1K"
date: 2026-05-08
content_hash: b28097c27ee428b0
---

# Scaling Continual Learning to 300+ Tasks with Bi-Level Routing Mixture-of-Experts

**Conference**: ICML 2026  
**arXiv**: [2602.03473](https://arxiv.org/abs/2602.03473)  
**Code**: https://github.com/LMMMEng/CaRE (available)  
**Area**: Continual Learning / Class-Incremental Learning / MoE / Parameter-Efficient Fine-Tuning  
**Keywords**: Class-Incremental Learning, Bi-Level Routing, Mixture-of-Experts, Long Task Sequences, OmniBenchmark-1K

## TL;DR
The authors propose CaRE: inserting a **bi-level routing MoE (BR-MoE)** into each ViT block—first, a "class recognizer" selects the Top-M relevant task routers based on entropy, then each router activates its Top-K task experts and adds a shared EMA expert. This enables retention of old knowledge and continual absorption of new classes even with 300+ tasks, filling the gap in "long-sequence CIL" (and releasing the 1000-class OmniBenchmark-1K benchmark).

## Background & Motivation

**Background**: Class-incremental learning (CIL) based on pre-trained models (PTMs) is a hot topic, with two main approaches—prompt-based (L2P, DualPrompt, CODA-Prompt) and adapter-based (EASE, APER, SEMA, MOS, TUNA, MIN). The latter typically trains a task-specific adapter for each task, activating the appropriate one during inference.

**Limitations of Prior Work**: (1) A single adapter is only discriminative for its trained classes—when the task sequence is long, distinguishing related classes across tasks (e.g., animal subclasses in different tasks) becomes poor; (2) Existing methods either use "global aggregation of all historical adapters" (coarse-grained) or a single adapter, lacking fine-grained retrieval of supplementary knowledge from related tasks; (3) Most CIL work is evaluated on 5–20 tasks, and many methods collapse on long sequences (100+ tasks)—yet there is no benchmark that supports 100+ tasks (CIFAR-100 split into 100 tasks leaves only 1 class per task, and ImageNet overlaps with PTM pre-training).

**Key Challenge**: To achieve "discriminative and integrative" feature representations, it is necessary to (a) know which tasks the current sample may belong to, (b) fuse the adapters of these tasks at each layer in a fine-grained manner, and (c) retain a "cross-task general" shared knowledge. While each property has been partially addressed, what is missing is a unified architecture that enables **per-layer "router selection then expert selection" bi-level decisions**.

**Goal**: (i) Design a PEFT module for per-layer fine-grained cross-task knowledge retrieval; (ii) Make it scalable to 300+ tasks; (iii) Provide a benchmark that truly tests long-sequence scalability.

**Key Insight**: The authors decompose the MoE router into two levels—coarse (by task) and fine (by adapter expert), using the "entropy of the task-specific classification head" as a signal for "how confident this sample belongs to the task." This is crucial: low entropy = high confidence = task relevance, and is more robust than direct "task ID prediction."

**Core Idea**: Inject a **(class recognizer $C_t$, router $R_t$, expert $E_t$) triplet** into each ViT block, adding a new triplet for each new task. During inference, select Top-M routers by entropy, then for each router select Top-K experts via gating, and add a shared EMA-maintained expert as a fallback—bi-level routing replaces the "global aggregation/single adapter" dichotomy.

## Method

### Overall Architecture
The backbone is a frozen ViT-B/16 (pre-trained on ImageNet-21K). Each Transformer block is replaced as follows: $z_a = \text{MHSA}(\text{Norm}_1(z)) + z$, $z_f = \text{FFN}(\text{Norm}_2(z_a)) + z_a$, $z' = \text{BR-MoE}(z_a) + z_f$. For each new task $t$ in incremental learning: (1) add a new triplet $(C_t, R_t, E_t)$ to the BR-MoE in each block; (2) during training, only update the new triplet and the shared expert $\bar{E}$ (all other parameters are frozen); (3) during inference, dynamically aggregate outputs per layer via the bi-level process. Final classification uses a concatenated angular margin head $W_t = [w^1, \dots, w^t]$, with class logits computed by cosine similarity $\cos(\theta_i^j) = \frac{w_j^t \cdot \phi^t(x_i^t)}{\|w_j^t\| \|\phi^t(x_i^t)\|}$ and a scaling factor $\tau = 20$.

### Key Designs

1. **Bi-Level Routing: Dynamic Router Selection**:

    - **Function**: At each layer, select the Top-M most relevant historical task routers for the input (not all or just the latest).
    - **Mechanism**: Feed the [CLS] token of $z_a$ to each task's class recognizer $C_t = \rho^t \in \mathbb{R}^{d \times |G^t|}$ to obtain the class distribution $s_t = \text{Softmax}(C_t(z_a^{[CLS]}))$ for task $t$; compute entropy $\mathcal{H}_t = -\sum_j s_t^{(j)} \log s_t^{(j)}$; select Top-M routers $R_t$ with lowest entropy. Low entropy means "this classifier is confident about the input," indicating likely task relevance—thus, using entropy instead of task ID prediction is more robust to train-test distribution shift. During training, always include the latest task router $R_T$ (to ensure new task learning); during inference, all routers are selected dynamically by entropy.
    - **Design Motivation**: Using a single "task classifier" to select the most relevant task is brittle (prone to errors), while aggregating all historical routers dilutes relevance; entropy ranking + Top-M balances robustness and focus, and allows independent local decisions per layer.

2. **Bi-Level Routing: Dynamic Expert Routing + Shared EMA Expert**:

    - **Function**: Within each of the selected M routers, further select Top-K adapter experts, and add a cross-task shared expert.
    - **Mechanism**: Each selected router $R_t$ is a linear layer $\eta^t \in \mathbb{R}^{d \times t}$ + softmax, producing $t$ gating scores for $z_a^{[CLS]}$, from which Top-K are selected and softmax-normalized to $\{a_i\}$. The corresponding adapters $E_i$ are weighted and summed, e.g., for M=2, K=2: $z_1 = a_2 E_2(z_a) + a_t E_t(z_a)$, $z_2 = b_{T-1} E_{T-1}(z_a) + b_T E_T(z_a)$, $z_r = z_1 + z_2$. A shared expert $\bar{E}$ is added—fully trained on the first task, and for subsequent tasks maintained by EMA $\delta_s \leftarrow \mu \delta_s + (1 - \mu)\delta_t$ ($\mu = 0.999$). The final BR-MoE output is $z_o = z_r + \bar{E}(z_a)$. Default M=2, K=3; regular adapters use 16-d bottleneck, shared adapter uses 64-d.
    - **Design Motivation**: Task-level routing alone is not fine enough; within each task, adapters must further select "the most relevant few." The shared expert provides a cross-task prior, preventing samples from being "missed" by all task-specific adapters (inspired by DeepSeek-MoE).

3. **Per-Layer Class Recognizer Supervision**:

    - **Function**: Ensures that class recognizers in intermediate layers produce reliable entropy signals, avoiding misrouting due to weak semantic features in shallow layers.
    - **Mechanism**: Add $\mathcal{L}_{cp}^\ell = \mathcal{L}_{cls}^\ell + \mathcal{L}_{KL}^\ell$ to each layer's $C_t$, where $\mathcal{L}_{cls}^\ell$ is the angular margin classification loss for $C_t$ at that layer, and $\mathcal{L}_{KL}^\ell$ is the KL divergence between $s_t$ and the final layer's softmax output $p_t$, encouraging shallow class recognizers to mimic the deep semantic distribution. The total objective is $\mathcal{L} = \mathcal{L}_{cls} + \lambda \frac{1}{L}\sum_\ell \mathcal{L}_{cp}^\ell$ ($\lambda = 1$).
    - **Design Motivation**: BR-MoE makes routing decisions independently at each block, which requires $C_t$ at that layer to produce entropy reflecting task relevance; if shallow semantics are insufficient, entropy is meaningless. KL distillation aligns each layer with the final decision, making bi-level routing effective even in shallow layers.

### Loss & Training
When a new task $t$ arrives, all historical parameters are frozen; only the current layer's $(C_t, R_t, E_t)$ and the shared expert $\bar{E}$ are trained. Optimizer: SGD (momentum=0.9, weight decay=5e-4), batch=16, 20 epochs per task, lr=0.01 with cosine annealing. The latest router $R_T$ is always activated at each layer during training (to prevent cold start).

## Key Experimental Results

### Main Results
On the newly proposed OmniBenchmark-1K (1000 classes / 190k images / 21 vision domains) long-sequence benchmark, metrics are $\bar{\mathcal{A}}$ (mean accuracy) / $\mathcal{A}_B$ (final accuracy):

| Method | 100 tasks (B0 Inc10) $\mathcal{A}_B$ | 200 tasks (B0 Inc5) $\mathcal{A}_B$ | 151 tasks (B100 Inc6) $\mathcal{A}_B$ | 301 tasks (B100 Inc3) $\mathcal{A}_B$ |
|---|---|---|---|---|
| L2P | 48.87 | 45.25 | 10.49 | 9.03 |
| DualPrompt | 49.45 | 45.62 | 12.90 | 9.30 |
| APER-Adapter | 62.24 | 61.53 | 62.99 | 62.99 |
| TUNA | 60.04 | 59.14 | 62.77 | 62.21 |
| MOS | 64.27 | 63.51 | 65.20 | 64.37 |
| MIN | 63.60 | 62.50 | 60.33 | 59.63 |
| **CaRE** | **68.27** | **67.46** | **69.01** | **68.51** |

On the longest sequence (301 tasks), CaRE outperforms MOS by 4 points and is dozens of points ahead of prompt-based methods (which collapse to 9% accuracy). On short-sequence CIL (CIFAR-100/ObjectNet/ImageNet-R/-A/VTAB, 5–20 tasks), CaRE remains SOTA in most cases; e.g., on ImageNet-A with 20 tasks, $\mathcal{A}_B$ = 59.91, 1.2 points higher than TUNA.

### Ablation Study

| Configuration | Key Metric Change (OmniBenchmark-1K) | Notes |
|---|---|---|
| Full CaRE | 67.46 | Complete model |
| Single router (M=1) | Significant drop | Validates necessity of "multiple routers" |
| No shared expert | Drop | EMA shared expert provides cross-task knowledge; without it, all samples fall to task-specific adapters only |
| Replace entropy-based router with single task classifier | Drop | Validates robustness of entropy over "hard task ID prediction" |
| Remove intermediate layer KL supervision | Drop | Shallow entropy signals become unreliable, bi-level routing fails |

### Key Findings
- **bi-level routing > single routing**: Selecting Top-M tasks per layer, then Top-K experts per task, is much better than "gating all adapters at once."
- **Entropy > task ID prediction**: Entropy reflects the classifier's overall uncertainty about the sample, more stable than hard argmax.
- **Shared expert as fallback**: Especially in late stages of long sequences, new task samples may not match any historical adapter; the EMA shared expert provides basic features and prevents collapse.
- **MIN/SEMA/MoAL and similar methods perform close to CaRE on short sequences but degrade severely on 100+ tasks**—long sequences are the real test for PTM-based CIL.

## Highlights & Insights
- **Layered MoE routing**: Coarse-fine two-level routers, each with clear semantics (task-level / expert-level); this "cluster by task, then select experts within" approach is naturally suited for long-sequence CIL and easily transferable to other scenarios requiring "coarse retrieval then fine combination" (e.g., retrieval-augmented generation: select document set, then select chunk).
- **Entropy as routing signal**: Using the entropy of each task-specific classifier as a "relevance measure" avoids the brittleness of training a "global task classifier"—this design, replacing "explicit task ID" with "model confidence," is clever and can inspire other modular systems.
- **OmniBenchmark-1K is a real contribution**: The CL community has long lacked a benchmark that can challenge all methods; this 1000-class, 21-domain, PTM-leak-free dataset fills the gap and will become the standard for long-sequence CIL evaluation.
- **Per-layer local decisions**: Different feature abstraction levels at different depths; BR-MoE allows each layer to independently select routers/experts based on its own features, more targeted than "global aggregation at the final output."

## Limitations & Future Work
- Each new task adds a $(C_t, R_t, E_t)$ triplet, so parameters grow linearly. With 301 tasks, each layer has 301 adapters—total parameters are not huge, but inference requires traversing all $C_t$ to compute entropy, so **computational complexity grows linearly with task count**.
- Does not address task-free CL (fuzzy task boundaries)—CaRE assumes clear task boundaries and independent triplets; streaming, boundary-free scenarios would need extra task detection mechanisms.
- Shared expert EMA $\mu = 0.999$ is fixed, not adaptive; with drastic task distribution shifts, EMA may lag.
- No formal analysis of task forgetting, e.g., which classes are most easily forgotten, how adapter weights evolve in routing, etc.
- Mainly validated on ViT; transferability to CNNs or LLM-style decoders is unknown.

## Related Work & Insights
- **vs MOS / TUNA / MIN**: These SOTA adapter-based methods perform well on short/medium sequences but degrade on long ones; CaRE's key difference is that bi-level routing decouples router and expert selection, enabling stable scaling to 300+ tasks.
- **vs MoE-Adapter / SEMA**: MoE-Adapter also uses router + expert, but only single-level routing; SEMA automatically decides whether to add a new adapter. CaRE's two-level routing + shared expert enables finer-grained knowledge retrieval.
- **vs DeepSeek-MoE**: The shared expert design is directly inspired by DeepSeek-MoE, but adapted to CIL, with EMA maintenance as a new engineering detail.
- **vs prompt-based (L2P/DualPrompt)**: Prompt-based methods collapse on long sequences (300 tasks retain only 9% accuracy), indicating prompt pool capacity is insufficient for 100+ tasks; adapter+MoE is a more suitable approach.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of bi-level routing + entropy-based router + shared EMA expert is new for CIL; each component has precedent, but the engineering integration and long-sequence scenario fill a gap.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 4 long-sequence settings (100/151/200/301 tasks) + 4 classic CIL datasets; the new OmniBenchmark-1K is also convincing.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clear, bi-level process diagrams are intuitive; formulas are dense but followable.
- Value: ⭐⭐⭐⭐⭐ The first work to push PTM-based CIL to 300+ tasks with continual improvement, plus the release of OmniBenchmark-1K, is a dual contribution to the long-sequence CL community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] DAG-MoE: From Simple Mixture to Structural Aggregation in Mixture-of-Experts](dag-moe_from_simple_mixture_to_structural_aggregation_in_mixture-of-experts.md)
- [\[ICLR 2026\] LD-MoLE: Learnable Dynamic Routing for Mixture of LoRA Experts](../../ICLR2026/model_compression/ld-mole_learnable_dynamic_routing_for_mixture_of_lora_experts.md)
- [\[ICML 2026\] Continual Model Routing in Evolving Model Hubs](continual_model_routing_in_evolving_model_hubs.md)
- [\[ICLR 2026\] FlyPrompt: Brain-Inspired Random-Expanded Routing with Temporal-Ensemble Experts for General Continual Learning](../../ICLR2026/model_compression/flyprompt_brain-inspired_random-expanded_routing.md)
- [\[ICML 2026\] Parameters as Experts: Adapting Vision Models with Dynamic Parameter Routing](parameters_as_experts_adapting_vision_models_with_dynamic_parameter_routing.md)

</div>

<!-- RELATED:END -->
