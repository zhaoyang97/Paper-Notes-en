---
title: >-
  [Paper Note] ViterbiPlanNet: Injecting Procedural Knowledge via Differentiable Viterbi for Planning
description: >-
  [CVPR 2026][Graph Learning][Procedural Planning] This work embeds a Procedural Knowledge Graph (PKG) into a planning model end-to-end via a differentiable Viterbi layer, enabling the neural network to learn only emission probabilities rather than memorizing complete procedural structures. With only 5–7M parameters—one to three orders of magnitude fewer than diffusion- or LLM-based methods—the approach achieves state-of-the-art success rates on CrossTask/COIN/NIV and establishes a unified evaluation benchmark.
tags:
  - CVPR 2026
  - Graph Learning
  - Procedural Planning
  - Differentiable Viterbi
  - Procedural Knowledge Graph
  - Instructional Videos
  - Structure-Aware Training
date: 2026-05-08
content_hash: 8243dc400f6e261b
---

# ViterbiPlanNet: Injecting Procedural Knowledge via Differentiable Viterbi for Planning

**Conference**: CVPR 2026
**arXiv**: [2603.04265](https://arxiv.org/abs/2603.04265)
**Code**: [Project Page](https://gigi-g.github.io/ViterbiPlanNet/)
**Area**: Graph Learning
**Keywords**: Procedural Planning, Differentiable Viterbi, Procedural Knowledge Graph, Instructional Videos, Structure-Aware Training

## TL;DR

This work embeds a Procedural Knowledge Graph (PKG) into a planning model end-to-end via a differentiable Viterbi layer, enabling the neural network to learn only emission probabilities rather than memorizing complete procedural structures. With only 5–7M parameters—one to three orders of magnitude fewer than diffusion- or LLM-based methods—the approach achieves state-of-the-art success rates on CrossTask/COIN/NIV and establishes a unified evaluation benchmark.

## Background & Motivation

**State of the Field**: Video procedural planning is the task of predicting intermediate action sequences given initial and goal visual states, and is a core capability for applications such as wearable AI assistants. Current SOTA methods include diffusion-based models (PDPP, KEPP, MTID, ~42M–1B parameters), LLM-based models (PlanLLM, ~385M parameters), and Transformer-based architectures (SCHEMA, ~6M parameters).

**Limitations of Prior Work**: All existing methods implicitly encode procedural knowledge in network parameters, requiring models to learn complex procedural structures (e.g., the canonical ordering "place bottom bun → add turkey → add lettuce → place top bun") from large amounts of data. This leads to: (1) low data efficiency—large numbers of training samples are needed to memorize procedural rules; (2) high computational cost—increasingly complex and large models are required; (3) limited generalization—performance degrades sharply at planning horizons shorter than those seen during training.

**Root Cause**: Human procedural planning does not reason from scratch; rather, intrinsic procedural knowledge (action priors, preconditions, canonical orderings, etc.) is naturally integrated into the planning process. Yet even existing methods that employ PKGs use them only as post-processing decoders at inference time, leaving the model unaware of procedural structure during training.

**Paper Goals**: To elevate the PKG from an inference-time post-processor to a training-time supervisory signal, so that structured priors participate directly in end-to-end optimization.

**Starting Point**: Procedural planning is formulated as optimal state-sequence decoding in a Hidden Markov Model (HMM), solved via the Viterbi algorithm. The key innovation is replacing the non-differentiable max and argmax operations in Viterbi with differentiable log-sum-exp and softmax relaxations, enabling gradients to flow from the planning loss back to the visual encoder.

**Core Idea**: Rather than having the network memorize procedures, it is trained only to learn "how well the current visual state matches each action" (emission probabilities), while the plausibility of action orderings is guaranteed by the knowledge graph. The differentiable Viterbi layer makes this division of labor trainable end-to-end.

## Method

### Overall Architecture

ViterbiPlanNet consists of four stages: (1) **Procedural Knowledge Encoding**—a PKG is constructed from training data, where nodes represent actions and edges carry transition probabilities estimated from co-occurrence statistics; (2) **Visual Encoding**—a frozen visual backbone with a learnable projection encodes start/goal video clips into vectors $v_s^{enc}, v_g^{enc} \in \mathbb{R}^E$; (3) **Emission Probability Estimation**—a Transformer encoder followed by an MLP and Sigmoid predicts the emission matrix $b \in \mathbb{R}^{T \times N}$ ($T$: planning horizon, $N$: number of actions) from the visual encodings; (4) **Structured Decoding**—a differentiable Viterbi layer takes the PKG as fixed parameters and the emission matrix as input, producing a soft plan $\tilde{\pi} \in [0,1]^{T \times N}$.

### Key Designs

1. **Differentiable Viterbi Layer (DVL)**

    - **Function**: Transforms the standard Viterbi decoding algorithm into a differentiable operation, enabling end-to-end training.
    - **Mechanism**: Replaces max with S-max (log-sum-exp relaxation) and argmax with S-argmax (softmax relaxation). At each time step $t$, predecessor scores are computed as $s_{i \to j}^{(t)} = \delta_{t-1}(i) \cdot \omega(i,j)$, where $\omega(i,j)$ is the fixed transition probability from the PKG. State scores are updated as $\delta_t(j) = b[t,j] \cdot \text{S-max}(\{s_{i \to j}^{(t)}\})$. Soft backpointers $\boldsymbol{\psi}_t(j, \cdot) = \text{S-argmax}(\{s_{i \to j}^{(t)}\})$ are computed simultaneously and recursively combined to yield the soft plan $\tilde{\pi}$.
    - **Design Motivation**: The max/argmax operations in standard Viterbi have zero gradients and cannot propagate training signals. Through smooth relaxation, gradients from the planning loss flow through the DVL to train the emission network $f_{emiss}$, encouraging it to learn emission probabilities consistent with the procedural structure. The DVL introduces no additional trainable parameters.

2. **Probabilistic Graphical Model Framework**

    - **Function**: Rigorously formulates procedural planning as optimal sequence decoding in an HMM.
    - **Mechanism**: Under the Markov assumption $P(a_t | a_{t-1})$, the joint probability is factored as $P(\pi = a_{1:T} | v_{0:T}) \propto \prod_{t=1}^T P(a_t | a_{t-1}) \cdot P(v_t | a_t)$, where transition probabilities are provided by the PKG and emission probabilities are predicted by the neural network. The optimal plan maximizes this probability.
    - **Design Motivation**: This formulation decomposes the planning problem into two independent components—procedural structure (transitions) and visual matching (emissions). The former is provided by the knowledge graph as a fixed prior, greatly simplifying the neural network's learning task.

3. **PKG Construction and Utilization**

    - **Function**: Constructs external procedural knowledge from training data to serve as fixed parameters of the DVL.
    - **Mechanism**: The PKG $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \omega)$ has actions as nodes and directed edges representing valid transitions; edge weights $\omega(i,j)$ are estimated from action co-occurrence statistics in the training set. A separate PKG is constructed for each dataset. At inference time, standard Viterbi decoding is applied to the soft plan to obtain a discrete action sequence.
    - **Design Motivation**: Compared to having the model implicitly memorize procedural rules, the explicit graph structure provides a reliable structural prior at zero additional parameter cost. Experiments show that the PKG yields a larger gain for the proposed method (+5.98% SR) than when used as a post-processor for SCHEMA (+3.31%) or as a conditioning signal for KEPP (+2.56%).

### Loss & Training

The total loss is the sum of three terms: $\mathcal{L} = \mathcal{L}_{plan} + \mathcal{L}_{align} + \mathcal{L}_{task}$. $\mathcal{L}_{plan}$ is the MSE loss between the soft plan output by the DVL and the ground-truth one-hot plan, serving as the primary supervision signal. $\mathcal{L}_{align}$ is a visual–semantic alignment loss that encourages visual encodings to align with textual descriptions of procedural states. $\mathcal{L}_{task}$ is a task classification loss that guides the encoder to preserve global task semantics. Each model is trained with 5 different random seeds; mean and 90% confidence intervals are reported.

## Key Experimental Results

### Main Results

| Dataset | Metric | ViterbiPlanNet | SCHEMA | PlanLLM | PDPP | Gain (vs. runner-up) |
|---------|--------|----------------|--------|---------|------|----------------------|
| CrossTask T=3 | SR↑ | **38.45±0.32** | 37.24±0.60 | 36.84±1.21 | 36.73±0.59 | +1.21±0.69 |
| CrossTask T=3 | mAcc↑ | **63.07±0.17** | 62.69±0.28 | 61.56±1.03 | 61.96±0.59 | +0.38 |
| COIN T=3 | SR↑ | **33.99±0.23** | 32.89±0.61 | 33.44±0.15 | 22.37±0.57 | +0.55±0.27 |
| NIV T=3 | SR↑ | **32.37±0.96** | 26.30±1.49 | 30.00±1.41 | 26.52±1.56 | +2.37±1.63 |
| NIV T=4 | SR↑ | **27.54±0.70** | 24.39±1.84 | 23.42±1.40 | 21.40±0.53 | +3.15±1.93 |

In terms of parameter count, ViterbiPlanNet uses only 5–7M parameters, compared to PDPP (42M), PlanLLM (385M), and MTID (1085M). Compared to MTID (1B parameters), SR and mAcc are comparable while mIoU is substantially higher (T=3: 76.92 vs. 69.17), with a 200× reduction in parameters.

### Ablation Study

| Configuration | Train DVL | Infer VD | SR↑ | mAcc↑ | mIoU↑ | Notes |
|---------------|-----------|----------|-----|-------|-------|-------|
| Base Model | ✗ | ✗ | 32.47 | 60.63 | 82.45 | No structural guidance |
| +Infer VD | ✗ | ✓ | 32.99 | 58.57 | 82.34 | Marginal post-processing gain |
| +Train DVL+Infer VD | ✓ | ✓ | **38.45** | **63.07** | **83.89** | Full method, +5.98 SR |
| Train DVL only | ✓ | ✗ | 20.05 | 54.61 | 76.99 | Emission probs ≠ direct predictions |

### Key Findings

- **Structure-aware training is the critical factor**: Incorporating the DVL during training yields +5.98% SR, far exceeding the gains from adding Viterbi decoding at inference time (+0.52%) or DVL at inference only (−0.38%).
- **PKG utilization efficiency**: ViterbiPlanNet benefits most from the PKG (+5.98%), far surpassing SCHEMA (post-processing: +3.31%), KEPP (conditioning: +2.56%), and PlanLLM (post-processing: +1.97%).
- **Sample efficiency**: With only 20% of training data, ViterbiPlanNet already outperforms SCHEMA trained on 100% of the data.
- **Cross-horizon generalization**: In cross-horizon experiments (trained on T=6, tested on T=3), ViterbiPlanNet achieves 27.77% SR, outperforming the runner-up SCHEMA (16.12%) by 11.65 percentage points.
- LLM/VLM baselines (including Gemini 2.5 Pro and Qwen3-30B) fall far short of trained methods; even simple PKG beam search surpasses most LLM/VLM results.

## Highlights & Insights

- The core idea is remarkably elegant: the complex planning problem is decomposed into "visual matching" and "procedural knowledge," with the network responsible only for learning simple emission probabilities while structural validity is guaranteed by the knowledge graph.
- The application of differentiable dynamic programming is highly principled, seamlessly integrating a classical algorithm into deep learning—the DVL introduces no additional parameters.
- The work establishes and open-sources a unified evaluation benchmark, resolving long-standing inconsistencies in data splits and metrics within the field. The experimental practice of using 5 seeds with bootstrap confidence intervals is commendable and worth broader adoption.
- Cross-horizon generalization consistency is a methodologically important contribution, demonstrating that structural priors confer genuine generalization rather than mere memorization.

## Limitations & Future Work

- PKG quality depends on action co-occurrence statistics in training data, which may be unreliable in data-scarce or distribution-shifted scenarios.
- The method handles only discrete action spaces and cannot be directly applied to continuous control tasks.
- The soft approximation may lose accuracy for very long sequences (T > 6); the paper primarily reports results for T = 3, 4.
- The Markov assumption of the HMM limits modeling of long-range dependencies, though the PKG partially mitigates this issue.
- The approach requires pre-defined action category names and PKG construction, and is thus not a fully end-to-end setting.

## Related Work & Insights

- **vs. SCHEMA**: Comparable parameter count (~6M), yet ViterbiPlanNet consistently leads on SR (CrossTask T=3: 38.45 vs. 37.24), because SCHEMA applies the PKG only as inference-time post-processing, whereas this work incorporates it as a training-time guide.
- **vs. PlanLLM**: PlanLLM has 70× more parameters than ViterbiPlanNet (385M vs. 5.5M) yet achieves lower performance, demonstrating the inefficiency of implicitly memorizing procedural knowledge.
- **vs. MTID**: With 200× fewer parameters (1085M vs. 5.5M), ViterbiPlanNet achieves comparable SR and mAcc but significantly higher mIoU, demonstrating that lightweight structured methods can match heavy generative approaches.
- **vs. LLM/VLM**: Even Gemini 2.5 Pro achieves only 29.18% SR (CrossTask T=3), far below ViterbiPlanNet's 38.45%, exposing a notable deficiency of current large models in structured procedural reasoning.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The design concept of the differentiable Viterbi layer is original and insightful, drawing a fundamental distinction between structure-guided training and inference-time post-processing.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Experiments span 3 datasets, a unified evaluation benchmark, 5-seed bootstrap confidence intervals, LLM/VLM baseline comparisons, cross-horizon generalization, and sample efficiency analysis—extremely comprehensive and rigorous.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The derivation of the probabilistic framework is clear and self-consistent; the logical chain from problem formulation to differentiable approximation to experimental validation is complete and coherent.
- **Value**: ⭐⭐⭐⭐ The methodological contribution is outstanding; the paradigm of differentiable dynamic programming is likely to generalize to other sequence prediction tasks that require structural priors.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Graph2Eval: Automatic Multimodal Task Generation for Agents via Knowledge Graphs](graph2eval_multimodal_task_generation_agents.md)
- [\[CVPR 2026\] M3KG-RAG: Multi-hop Multimodal Knowledge Graph-enhanced Retrieval-Augmented Generation](m3kg_rag_multi_hop_multimodal_knowledge_graph_enhanced_retrieval_augmented_genera.md)
- [\[CVPR 2026\] Graph-to-Frame RAG: Visual-Space Knowledge Fusion for Training-Free and Auditable Video Reasoning](graph-to-frame_rag_visual-space_knowledge_fusion_for_training-free_and_auditable.md)
- [\[ICLR 2026\] RAS: Retrieval-And-Structuring for Knowledge-Intensive LLM Generation](../../ICLR2026/graph_learning/ras_retrieval-and-structuring_for_knowledge-intensive_llm_generation.md)
- [\[ICLR 2026\] Explore-on-Graph: Incentivizing Autonomous Exploration of LLMs on Knowledge Graphs](../../ICLR2026/graph_learning/explore-on-graph_incentivizing_autonomous_exploration_of_large_language_models_o.md)

<!-- RELATED:END -->
