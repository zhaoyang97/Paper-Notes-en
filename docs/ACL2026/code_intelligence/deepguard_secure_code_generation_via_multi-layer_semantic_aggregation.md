---
title: >-
  [Paper Note] DeepGuard: Secure Code Generation via Multi-Layer Semantic Aggregation
description: >-
  [ACL 2026][Code Intelligence][Secure code generation] DeepGuard is proposed to overcome the "final-layer bottleneck" by aggregating representations from multiple upper Transformer layers via an attention mechanism. Combi…
tags:
  - "ACL 2026"
  - "Code Intelligence"
  - "Secure code generation"
  - "multi-layer aggregation"
  - "vulnerability detection"
  - "contrastive learning"
  - "inference-time guidance"
date: 2026-05-08
content_hash: 8754cbbed67f59c2
---

# DeepGuard: Secure Code Generation via Multi-Layer Semantic Aggregation

**Conference**: ACL 2026
**arXiv**: [2604.09089](https://arxiv.org/abs/2604.09089)
**Code**: [https://github.com/unknownhl/DeepGuard](https://github.com/unknownhl/DeepGuard)
**Area**: Code Intelligence / Security
**Keywords**: Secure code generation, multi-layer aggregation, vulnerability detection, contrastive learning, inference-time guidance

## TL;DR
DeepGuard is proposed to overcome the "final-layer bottleneck" by aggregating representations from multiple upper Transformer layers via an attention mechanism. Combined with multi-objective training and a lightweight inference-time safety guidance strategy, it achieves an average improvement of 11.9% in secure-and-correct generation rate across 5 code LLMs.

## Background & Motivation

**Background**: Code LLMs have demonstrated strong performance in code generation — GitHub Copilot reportedly assists in generating up to 46% of code on the platform. However, these models also replicate insecure coding patterns from training data: approximately 40% of Copilot-generated code contains vulnerabilities, and developers often fail to identify these AI-introduced defects.

**Limitations of Prior Work**: Existing security hardening approaches (e.g., SVEN's prefix tuning, SafeCoder's security-instruction fine-tuning) almost exclusively extract supervision signals from the final Transformer layer. However, final-layer representations are primarily optimized for next-token prediction rather than fine-grained vulnerability discrimination. The authors find that vulnerability-discriminative signals are strongest in the middle-to-upper layers and actually decay toward the final layer — a phenomenon termed the "final-layer bottleneck."

**Key Challenge**: Preventing insecure code requires integrating diverse syntactic and semantic evidence (e.g., recognizing syntactic patterns of string concatenation and reasoning about the semantic properties of untrusted data flows). This information is distributed across Transformer layers — shallow layers capture local syntax while deep layers encode abstract semantics — yet the final layer optimizes token prediction at the cost of vulnerability discriminability.

**Goal**: To leverage security-relevant cues distributed across the model's internal layers, rather than relying solely on the final layer, in order to improve secure code generation.

**Key Insight**: Layer-wise linear probing — training a linear classifier at each layer to detect vulnerability patterns — reveals that probe confidence peaks at the middle-to-upper layers and decays toward the final layer.

**Core Idea**: An attention mechanism aggregates hidden states from multiple upper layers to construct a stronger security analysis signal than any single final layer, supporting multi-objective training and inference-time guidance.

## Method

### Overall Architecture
DeepGuard consists of a training phase and an inference phase. During training, LoRA is applied to paired data (vulnerable/secure code pairs) under a multi-objective adaptation scheme: security contrastive loss + generation loss + KL regularization. During inference, a lightweight safety guidance strategy is applied via prompt-conditioned biasing.

### Key Designs

1. **Attention-Based Multi-Layer Aggregator**:

    - **Function**: Fuses hidden states from the top $N$ layers to construct a stronger security analysis signal.
    - **Mechanism**: For each token position $j$, the hidden states from the top $N$ layers are stacked into $h^{(j)} \in \mathbb{R}^{N \times D}$. Their mean serves as the query vector, and the aggregated representation is obtained via attention-weighted fusion: $h_{agg}^{(j)} = \text{Softmax}(\frac{QK^\top}{\sqrt{D}})V$. The mean provides a cross-layer "consensus" summary, while the attention mechanism allows the model to adaptively focus on the layers most informative for security analysis.
    - **Design Motivation**: Different layers exhibit different sensitivities to different vulnerability types; fixed-weight fusion is less expressive than adaptive attention-based selection.

2. **Security Analyzer with Contrastive Training**:

    - **Function**: Learns to distinguish between secure and vulnerable code.
    - **Mechanism**: The security analyzer $f_{sa}$ consumes the aggregated representation $H_{agg}$ and learned token-level security embeddings $E_{sec}$, producing a per-token security score $s_i(x) \in [0,1]$. For each (vulnerable, secure) code pair, sequence-level scores are computed and a margin-based contrastive loss is applied: $\mathcal{L}_{sec} = \max(0, \Delta - (s_{sec} - s_{vul}))$.
    - **Design Motivation**: Contrastive learning directly trains for separability between secure and insecure representations, yielding more robust discrimination than plain classification.

3. **Lightweight Inference-Time Safety Guidance**:

    - **Function**: Biases token selection toward safer choices during generation.
    - **Mechanism**: A token-level safety prior vector $T_{stats}$ is maintained, computed during training by tracking each token's tendency to appear in secure versus vulnerable samples. At inference time, a single forward pass over the input prompt yields a safety score $\bar{s}_{prompt}$. A bias is computed as $b = (1 - \bar{s}_{prompt}) \cdot T_{stats}$ — the less safe the prompt, the stronger the bias — and this bias is added to the logits at each decoding step.
    - **Design Motivation**: This design avoids the high overhead of re-running the security analyzer at every decoding step; a single forward pass produces a bias that is reused throughout generation.

### Loss & Training
$\mathcal{L}_{total} = \mathcal{L}_{gen} + w_{sec}\mathcal{L}_{sec} + w_{kl}\mathcal{L}_{kl}$, where $\mathcal{L}_{gen}$ is the standard generation loss on secure code, and $\mathcal{L}_{kl}$ is a KL divergence term against the frozen base model to prevent catastrophic forgetting.

## Key Experimental Results

### Main Results (Qwen2.5-Coder-3B)

| Method | pass@1 | sec@1_pass | sec-pass@1 | SVEN-SR |
|--------|--------|-----------|------------|---------|
| Base | 91.00 | 76.47 | 69.59 | 77.95 |
| SVEN | 83.00 | 84.90 | 70.47 | 82.60 |
| SafeCoder | 63.94 | 82.34 | 52.65 | 87.02 |
| **DeepGuard** | **86.65** | **93.21** | **80.76** | **94.11** |

### Ablation Study

| Configuration | Description |
|---------------|-------------|
| Final layer only (standard approach) | Weak security signal; limited improvement |
| Multi-layer mean fusion | Better than final layer alone, but inferior to attention fusion |
| Attention-based multi-layer aggregation | **Best**, adaptively selects the most relevant layers |
| w/o inference-time guidance | Training-phase gains are retained but no additional protection at inference |

### Key Findings
- DeepGuard improves sec-pass@1 by an average of 11.9% across 5 models while largely preserving functional correctness.
- Semantic analysis of the security embeddings $E_{sec}$ indicates that the model learns meaningful associations between tokens and security/insecurity.
- The approach generalizes to vulnerability types not seen during training (held-out CWEs).

## Highlights & Insights
- **Layer-wise linear probing** provides direct empirical evidence for the "final-layer bottleneck" hypothesis — this diagnostic methodology is broadly applicable to understanding information distribution across Transformer layers in other tasks.
- **Extremely lightweight inference-time guidance** — only one additional forward pass and a logit addition are required, making deployment overhead negligible.
- **Security–correctness trade-off** is well managed: many baselines sacrifice substantial functional correctness for security gains (e.g., SafeCoder achieves only 63.94% pass@1), whereas DeepGuard maintains 86.65% pass@1 while significantly improving security.

## Limitations & Future Work
- The token-level safety prior $T_{stats}$ is a coarse-grained statistical association and may produce incorrect biases in specific contexts.
- Training requires paired vulnerable/secure code samples, which are costly to obtain.
- Evaluation is currently limited to Python; cross-language generalization remains unexplored.

## Related Work & Insights
- **vs. SVEN**: Employs prefix tuning and extracts signals from the final layer, inheriting the final-layer bottleneck; DeepGuard aggregates across multiple layers.
- **vs. SafeCoder**: Uses security-instruction fine-tuning at the cost of significant functional correctness degradation; DeepGuard balances both objectives through multi-objective training.

## Rating
- Novelty: ⭐⭐⭐⭐ Clear multi-layer aggregation concept with a complete diagnosis-to-solution pipeline.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Five models, multiple baselines, generalization tests, and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ The logical chain from diagnosis to solution is clearly articulated.
- Value: ⭐⭐⭐⭐ Directly practical for secure code generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MARS2: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation](mars2_scaling_multi-agent_tree_search_via_reinforcement_learning_for_code_genera.md)
- [\[ACL 2026\] Sense and Sensitivity: Examining the Influence of Semantic Recall on Long Context Code Understanding](sense_and_sensitivity_examining_the_influence_of_semantic_recall_on_long_context.md)
- [\[AAAI 2026\] MoSE: Hierarchical Self-Distillation Enhances Early Layer Embeddings](../../AAAI2026/code_intelligence/mose_hierarchical_self-distillation_enhances_early_layer_embeddings.md)
- [\[ACL 2026\] CollabCoder: Plan-Code Co-Evolution via Collaborative Decision-Making for Efficient Code Generation](collabcoder_plan-code_co-evolution_via_collaborative_decision-making_for_efficie.md)
- [\[ACL 2026\] CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment](coderl_improving_code_generation_via_reinforcement_with_execution_semantics_alig.md)

</div>

<!-- RELATED:END -->
