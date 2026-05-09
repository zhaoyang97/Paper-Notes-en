---
title: >-
  [Paper Note] Memory-Integrated Reconfigurable Adapters: A Unified Framework for Settings with Multiple Tasks
description: >-
  [NeurIPS 2025][Signal & Communication][Associative Memory] MIRA embeds Hopfield-style associative memory modules into each layer of a ViT, storing and retrieving LoRA adapter weights as key-value pairs. Through a two-stage training procedure (Adaptation + Consolidation), a single unified architecture simultaneously addresses Domain Generalization (DG), Class-Incremental Learning (CIL), and Domain-Incremental Learning (DIL), achieving substantial improvements over task-specific methods across multiple benchmarks.
tags:
  - NeurIPS 2025
  - "Signal & Communication"
  - Associative Memory
  - Hopfield Networks
  - Adapters
  - Continual Learning
  - Domain Generalization
date: 2026-05-08
content_hash: d6a54000a60bf9a6
---

# Memory-Integrated Reconfigurable Adapters: A Unified Framework for Settings with Multiple Tasks

**Conference**: NeurIPS 2025
**arXiv**: [2512.00940](https://arxiv.org/abs/2512.00940)
**Code**: [https://snimm.github.io/mira_web/](https://snimm.github.io/mira_web/)
**Area**: Continual Learning / Domain Generalization
**Keywords**: Associative Memory, Hopfield Networks, Adapters, Continual Learning, Domain Generalization

## TL;DR

MIRA embeds Hopfield-style associative memory modules into each layer of a ViT, storing and retrieving LoRA adapter weights as key-value pairs. Through a two-stage training procedure (Adaptation + Consolidation), a single unified architecture simultaneously addresses Domain Generalization (DG), Class-Incremental Learning (CIL), and Domain-Incremental Learning (DIL), achieving substantial improvements over task-specific methods across multiple benchmarks.

## Background & Motivation

**Background**: Domain Generalization (DG), Class-Incremental Learning (CIL), and Domain-Incremental Learning (DIL) are three important yet independently developed research directions in deep learning. DG requires models to generalize to unseen domains, while Continual Learning (CL) requires models to retain prior knowledge when new tasks arrive. Existing methods are typically designed for a single scenario with specialized architectures and strategies.

**Limitations of Prior Work**: Biological organisms can switch between multiple behavioral modes within milliseconds (e.g., bats adjusting echolocation from 20 Hz to 200 Hz, or jazz pianists improvising) while retaining previously acquired knowledge. This capability relies on the dynamic reuse of the same neural circuits via neuromodulatory signals (dopamine, acetylcholine, etc.). However, existing deep learning methods lack such a unified mechanism for "rapid multi-task switching with persistent memory."

**Key Challenge**: Although DG, CIL, and DIL appear distinct, they fundamentally all require efficient adaptation across multiple tasks or domains while preserving knowledge. Existing work treats them as entirely separate problems and fails to leverage biological associative memory (AM) to unify their solutions.

**Goal**: (1) How to construct a unified architecture that simultaneously handles DG, CIL, and DIL; (2) How to leverage associative memory to enable per-sample dynamic adapter composition and retrieval; (3) How to learn effective retrieval keys for indexing stored adapter weights.

**Key Insight**: Inspired by neuroscience, associative memory can store and retrieve task-specific weight modulation signals. If LoRA adapters are stored as "values" in associative memory and retrieved per sample via learnable "keys," the system can achieve rapid task switching analogous to neural modulation in the brain.

**Core Idea**: Embed Hopfield associative memory into each ViT layer, storing task-specific LoRA adapters as values, and retrieve affine combinations of adapters on a per-sample basis via post-hoc learned retrieval keys, thereby unifying DG, CIL, and DIL within a single framework.

## Method

### Overall Architecture

MIRA builds upon a frozen ViT-B/16 backbone (initialized with CLIP) and attaches a Universal Hopfield Network (UHN) memory unit to each layer. As an input image passes through the backbone, each layer's memory unit generates a query vector from the previous layer's output, retrieves a weighted combination of stored adapter weights, and loads this combination into the current layer. Training proceeds in two stages: **Adaptation** (training independent LoRA adapters per task/domain and writing them into memory) and **Consolidation** (optimizing only the retrieval keys and query modules to minimize task loss using the retrieved combinations). Inference requires only a forward pass.

### Key Designs

1. **Associative Memory for Adapter Storage and Retrieval**:

    - **Function**: Stores task/domain-trained LoRA adapters as key-value pairs and retrieves weighted combinations at inference time.
    - **Mechanism**: A memory unit $\mathcal{M}_\ell$ is attached to each layer $\ell$ of the ViT. Write operations store the trained adapter $\theta_\ell^{(t)}$; read operations compute similarity between a query vector $q$ and all keys $\mathbf{K}$, then form a weighted combination: $\hat{\theta}_\ell = \Theta_\ell \cdot \text{sep}(\text{sim}(K_\ell^\top, q))$. Critically, an **affine function** is used as the separation function (rather than Softmax), allowing negative weights to actively suppress interfering information rather than merely masking it.
    - **Design Motivation**: Storing adapter weights rather than raw data enables dynamic composition of multi-task parametric knowledge at inference time without gradient computation. Ablation studies confirm that the affine function outperforms Softmax for both CIL and DG.

2. **Two-Stage Training (Adaptation + Consolidation)**:

    - **Function**: Decouples adapter training from retrieval optimization.
    - **Mechanism**: In the Adaptation stage, LoRA adapters (rank=4) are trained for each task using cross-entropy loss, then written into memory with randomly initialized Gaussian keys. In the Consolidation stage, the adapter values are frozen, and only the query modules $g_\ell$ and keys $\mathbf{K}_\ell$ are trained to minimize cross-entropy on the retrieved combinations. For DG, all domain data are consolidated jointly; for CL, tasks are consolidated sequentially.
    - **Design Motivation**: Decoupling the two stages allows the retrieval space to be optimized independently. Consolidation is formally equivalent to solving for the optimal adapter combination via AM inner products, as established by Lemma 1 in the paper.

3. **Learnable Query Module**:

    - **Function**: Aligns the previous layer's output from representation space to key space.
    - **Mechanism**: Each layer is equipped with a lightweight module $g_\ell: \mathbb{R}^{d_h} \to \mathbb{R}^{d_k}$, which may be an identity map, a linear projection, or a small network. The query $q_\ell = g_\ell(h_{\ell-1})$ is compared against the keys via inner product, and the resulting separation function output yields the adapter combination weights. Keys and query modules are jointly optimized via backpropagation.
    - **Design Motivation**: The layer output and keys may reside in different representation spaces; the query module bridges this gap. Post-hoc key learning (rather than fixed keys) enables retrieval to be adaptively optimized.

### Loss & Training

Cross-entropy loss is used in both stages. Only LoRA parameters are updated during Adaptation; only keys and query modules are updated during Consolidation. In CL settings, forgetting mitigation techniques such as DualGPM can be integrated within the Consolidation stage. The additional parameters introduced by the Hopfield keys account for less than 0.4% of total parameters (~276K / 86M), and the inference latency overhead is approximately ~0.4%.

## Key Experimental Results

### Main Results

| Dataset | Setting | Metric | MIRA | Prev. SOTA | Gain |
|---------|---------|--------|------|------------|------|
| iDigits | CIL | Avg Acc↑ | **83.00%** | 71.53% (ICON) | +11.47% |
| CORe50 | CIL | Avg Acc↑ | **83.39%** | 80.85% (ICON) | +2.54% |
| DomainNet | CIL | Avg Acc↑ | **67.29%** | 65.43% (ICON) | +1.86% |
| CORe50 | DIL | Avg Acc↑ | **93.89%** | 89.01% (ICON) | +4.88% |
| DomainNet | DIL | Avg Acc↑ | **69.18%** | 54.44% (ICON) | +14.74% |
| PACS | DG | Acc | **97.01%** | 96.50% (PEGO) | +0.51% |
| OfficeHome | DG | Acc | **87.36%** | 84.20% (PEGO) | +3.16% |
| DomainNet | DG | Acc | **61.19%** | 59.80% (CoOp) | +1.39% |
| DN4IL | DIL | Last Acc | **78.40%** | 44.45% (DUCA) | +33.95% |
| ImageNet-R | CIL-5 | ACC5 | **78.06%** | 75.85% (C-LoRA) | +2.21% |

### Ablation Study

| Separation Function | CIL Acc | DIL Acc | DG Acc | Avg |
|--------------------|---------|---------|--------|-----|
| Affine (default) | **67.29** | 69.18 | **61.19** | **65.89** |
| Softmax | 66.87 | **69.21** | 60.82 | 65.63 |
| ReLU | 66.60 | 69.20 | 60.90 | 65.57 |
| Tanh | 66.73 | 68.96 | 60.94 | 65.54 |

| Adapters per Task | CIL Acc | DIL Acc | DG Acc |
|-------------------|---------|---------|--------|
| 1 | 63.75 | 69.08 | 61.21 |
| 5 | 67.21 | 69.10 | 61.01 |
| 10 | **67.29** | **69.18** | 61.19 |

### Key Findings

- The affine separation function performs best on CIL and DG because negative weights actively suppress interfering information, whereas Softmax/ReLU can only mask but not remove such interference.
- Increasing the number of adapters from 1 to 5 yields substantial gains (CIL: +3.46%), while marginal improvements are observed from 5 to 10, indicating that 5 adapters are sufficient to capture most task-specific variation.
- On DN4IL, MIRA achieves 78.40%, substantially outperforming DARE++ (44.11%) which uses a 200-sample replay buffer, demonstrating that associative memory can serve as an effective substitute for replay buffers.
- Inference latency increases by only ~0.4% (0.0241s vs. 0.0240s) and additional parameters account for less than 0.4% of total, making the practical deployment overhead negligible.

## Highlights & Insights

- **A novel use of associative memory: storing weights rather than data**: Conventional AM stores data or features for replay; MIRA stores adapter weights for direct network modulation. Storage scales with the number of tasks rather than data volume, making the approach highly efficient. This represents a paradigm shift in how AM is applied within deep learning.
- **The elegance of post-hoc key learning**: Training adapters first and then learning retrieval keys is analogous to first "writing an encyclopedia" and then "building the index system." This two-step decoupling simplifies the optimization of each component independently.
- **Unified architecture with minimal overhead**: The same architecture handles three distinct learning paradigms by simply varying the loss function and data provision strategy, with near-zero inference overhead. This "change the loss, not the architecture" design philosophy merits broader adoption in multi-paradigm learning.
- **Theoretical guarantee (Lemma 1)**: A formal proof demonstrates that AM retrieval can express the optimal adapter combination problem, providing rigorous theoretical justification for the proposed approach.

## Limitations & Future Work

- The current framework employs only affine (linear) combinations; nonlinear combinations (e.g., MoE-style gating) may yield further improvements, particularly for OOD extrapolation scenarios.
- Experiments are limited to classification tasks and ViT architectures; generalization to detection, segmentation, generation, and NLP tasks remains unverified.
- Each task requires training an independent set of adapters, leading to linear growth in storage as the number of tasks increases.
- The retrieval fidelity of Hopfield networks under high-dimensional adapter representations is insufficiently analyzed.
- All ablations are conducted on DomainNet; whether the same trends hold on other datasets is not validated.

## Related Work & Insights

- **vs. ICON**: ICON unifies CIL and DIL but does not support DG, relying on a dedicated prompt pool mechanism. MIRA provides a more natural unification via associative memory, and DG capability is absent from ICON.
- **vs. L2P/DualPrompt/CODA-P**: Prompt-learning-based CL methods append learnable prompts at each layer but lack explicit storage-retrieval semantics between prompts. MIRA's Hopfield memory provides a clear index-lookup mechanism.
- **vs. PEGO**: A DG-specific method that marginally outperforms MIRA on VLCS but cannot handle CL scenarios. MIRA substantially surpasses PEGO on OfficeHome and DomainNet.
- **vs. LoRA/VeRA and other PEFT methods**: Standard PEFT addresses only single-task adaptation without considering cross-task knowledge consolidation. MIRA augments PEFT with a memory indexing layer that enables multi-task knowledge management.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The idea of storing adapter weights in associative memory with post-hoc key learning is novel and theoretically well-grounded.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers three learning paradigms (CIL/DIL/DG) across 7+ datasets with thorough ablation analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — The biological motivation is vividly presented, and the chain from theory to experiments is complete.
- **Value**: ⭐⭐⭐⭐ — Provides an elegant framework for unified multi-task learning; the AM+adapter paradigm is likely to inspire subsequent work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] The Last Vote: A Multi-Stakeholder Framework for Language Model Governance](the_last_vote_a_multi-stakeholder_framework_for_language_model_governance.md)
- [\[AAAI 2026\] Text-Guided Channel Perturbation and Pretrained Knowledge Integration for Unified Multi-Modality Image Fusion](../../AAAI2026/signal_comm/text-guided_channel_perturbation_and_pretrained_knowledge_integration_for_unifie.md)
- [\[NeurIPS 2025\] Contrastive Consolidation of Top-Down Modulations Achieves Sparsely Supervised Continual Learning](contrastive_consolidation_of_top-down_modulations_achieves_sparsely_supervised_c.md)
- [\[NeurIPS 2025\] The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning](the_surprising_effectiveness_of_negative_reinforcement_in_llm_reasoning.md)
- [\[NeurIPS 2025\] Estimation of Stochastic Optimal Transport Maps](estimation_of_stochastic_optimal_transport_maps.md)

</div>

<!-- RELATED:END -->
