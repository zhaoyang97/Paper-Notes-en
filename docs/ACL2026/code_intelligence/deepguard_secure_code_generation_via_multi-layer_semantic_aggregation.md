---
title: >-
  [Paper Note] DeepGuard: Secure Code Generation via Multi-Layer Semantic Aggregation
description: >-
  [ACL 2026][Code Intelligence][Secure Code Generation] DeepGuard is proposed to overcome the "final layer bottleneck" by aggregating Transformer upper multi-layer representations via an attention mechanism. Combined with…
tags:
  - "ACL 2026"
  - "Code Intelligence"
  - "Secure Code Generation"
  - "Multi-layer Aggregation"
  - "Vulnerability Detection"
  - "Contrastive Learning"
  - "Reasoning Guidance"
date: 2026-05-08
content_hash: 75068a1680dd4cff
---

# DeepGuard: Secure Code Generation via Multi-Layer Semantic Aggregation

**Conference**: ACL 2026  
**arXiv**: [2604.09089](https://arxiv.org/abs/2604.09089)  
**Code**: [https://github.com/unknownhl/DeepGuard](https://github.com/unknownhl/DeepGuard)  
**Area**: Code Intelligence / Security  
**Keywords**: Secure Code Generation, Multi-layer Aggregation, Vulnerability Detection, Contrastive Learning, Reasoning Guidance

## TL;DR
DeepGuard is proposed to overcome the "final layer bottleneck" by aggregating Transformer upper multi-layer representations via an attention mechanism. Combined with multi-objective training and a lightweight inference-time security guidance strategy, it improves the secure-correct generation rate by an average of 11.9% across five code LLMs.

## Background & Motivation

**Background**: Code LLMs perform excellently in code generation, with GitHub Copilot reportedly assisting in generating up to 46% of the code on the platform. However, these models also replicate insecure coding patterns from training data—approximately 40% of Copilot-generated code contains vulnerabilities, and developers often fail to identify these AI-generated defects.

**Limitations of Prior Work**: Existing security hardening methods (e.g., prefix tuning in SVEN, security instruction fine-tuning in SafeCoder) almost exclusively extract supervisory signals from the final Transformer layer. However, final layer representations are primarily optimized for next-token prediction rather than fine-grained vulnerability discrimination. The authors found that vulnerability discrimination signals are strongest in the middle-to-upper layers and actually decay in the final layer—a phenomenon termed the "final layer bottleneck."

**Key Challenge**: Preventing insecure code requires integrating diverse syntactic and semantic evidence (e.g., identifying syntactic patterns of string concatenation + reasoning about semantic properties of untrusted data flows). This information is distributed across Transformer layers—shallow layers capture local syntax while deep layers encode abstract semantics—yet the final layer optimizes for token prediction at the expense of vulnerability discriminative power.

**Goal**: To improve secure code generation by leveraging security-related cues distributed across internal model layers rather than relying solely on the final layer.

**Key Insight**: Through layer-wise linear probe diagnostics—training linear classifiers at each layer to detect vulnerability patterns—it was discovered that probe confidence peaks in the middle-to-upper layers and decays toward the final layer.

**Core Idea**: Use an attention mechanism to aggregate hidden states from multiple upper layers to construct a stronger security analysis signal than a single final layer, supporting multi-objective training and inference-time guidance.

## Method

### Overall Architecture
DeepGuard consists of a training phase and an inference phase. In the training phase, LoRA is used for multi-objective adaptation on paired data (vulnerable/secure code pairs) involving security contrastive loss, generation loss, and KL regularization. In the inference phase, lightweight security guidance is performed using prompt-conditioned biasing.

### Key Designs

1.  **Attention Multi-Layer Aggregator**:
    - **Function**: Fuses hidden states from the top $N$ layers to construct a stronger security analysis signal.
    - **Mechanism**: For each token position $j$, the hidden states of the top $N$ layers are stacked as $h^{(j)} \in \mathbb{R}^{N \times D}$. Using the mean as the query vector, they are fused via an attention mechanism: $h_{agg}^{(j)} = \text{Softmax}(\frac{QK^\top}{\sqrt{D}})V$. The mean provides a "consensus" summary across layers, while the attention allows the model to adaptively focus on layers most valuable for security analysis.
    - **Design Motivation**: Different layers have varying sensitivities to different types of vulnerabilities; adaptive attention-based fusion is superior to fixed-weight fusion.

2.  **Security Analyzer and Contrastive Training**:
    - **Function**: Learns to distinguish between secure and vulnerable code.
    - **Mechanism**: The security analyzer $f_{sa}$ consumes the aggregated representation $H_{agg}$ and learned token-level security embeddings $E_{sec}$, outputting a security score $s_i(x) \in [0,1]$ for each token. A sequence-level score is calculated for each (vulnerable, secure) pair, and a margin contrastive loss is applied: $\mathcal{L}_{sec} = \max(0, \Delta - (s_{sec} - s_{vul}))$.
    - **Design Motivation**: Directly training for the separability of secure/insecure code via contrastive learning is more robust than simple classification.

3.  **Lightweight Inference-Time Security Guidance**:
    - **Function**: Biases token selection toward security during generation.
    - **Mechanism**: Maintains a token-level security prior vector $T_{stats}$ (statistical tendency of tokens to appear in secure/vulnerable samples during training). At inference, a single forward pass on the input prompt yields a security score $\bar{s}_{prompt}$. A bias $b = (1 - \bar{s}_{prompt}) \cdot T_{stats}$ is calculated (stronger bias for less secure prompts), which is added to the logits at each decoding step.
    - **Design Motivation**: Avoids the high overhead of re-running the security analyzer at every step; a single forward pass obtains a bias reused throughout the generation.

### Loss & Training
$\mathcal{L}_{total} = \mathcal{L}_{gen} + w_{sec}\mathcal{L}_{sec} + w_{kl}\mathcal{L}_{kl}$, where $\mathcal{L}_{gen}$ is the standard generation loss on secure code, and $\mathcal{L}_{kl}$ is the KL divergence with the frozen base model to prevent catastrophic forgetting.

## Key Experimental Results

### Main Results (Qwen2.5-Coder-3B)

| Method | pass@1 | sec@1_pass | sec-pass@1 | SVEN-SR |
| :--- | :--- | :--- | :--- | :--- |
| Base | 91.00 | 76.47 | 69.59 | 77.95 |
| SVEN | 83.00 | 84.90 | 70.47 | 82.60 |
| SafeCoder | 63.94 | 82.34 | 52.65 | 87.02 |
| **DeepGuard** | **86.65** | **93.21** | **80.76** | **94.11** |

### Ablation Study

| Configuration | Description |
| :--- | :--- |
| Final layer only (standard) | Weak security signals, limited improvement |
| Multi-layer mean fusion | Better than final layer but inferior to attention fusion |
| Attention multi-layer aggregation | **Optimal**; adaptively selects the most relevant layers |
| w/o Inference guidance | Training improvements remain but lacks additional runtime protection |

### Key Findings
- DeepGuard improves sec-pass@1 by an average of 11.9% across five models while largely maintaining functional correctness.
- Semantic analysis of security embeddings $E_{sec}$ shows that the model learns meaningful associations between secure/insecure tokens.
- Demonstrates generalization capability to vulnerability types (held-out CWEs) not seen during training.

## Highlights & Insights
- **Layer-wise linear probe diagnostics** provide direct evidence supporting the "final layer bottleneck" hypothesis—this diagnostic methodology can be generalized to understand information distribution in other Transformer tasks.
- **Ultra-lightweight design for inference-time guidance**—requiring only one extra forward pass and logit addition, the actual deployment overhead is negligible.
- **Proper handling of the security-correctness trade-off**—many baselines sacrifice significant functional correctness for security (e.g., SafeCoder pass@1 is only 63.94%), whereas DeepGuard maintains a pass@1 of 86.65% while significantly enhancing security.

## Limitations & Future Work
- The token-level security prior $T_{stats}$ is a coarse-grained statistical association that might provide incorrect biases in specific contexts.
- Training requires paired vulnerable/secure code, which is costly to obtain.
- Currently only validated on Python code; cross-language generalization remains unknown.

## Related Work & Insights
- **vs SVEN**: SVEN uses prefix tuning to extract signals from the final layer, limited by the final layer bottleneck; DeepGuard aggregates multiple layers.
- **vs SafeCoder**: SafeCoder uses security instruction fine-tuning, sacrificing significant functional correctness; DeepGuard balances both through multi-objective training.

## Rating
- Novelty: ⭐⭐⭐⭐ Clear multi-layer aggregation approach with a complete diagnostic+solution chain.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive testing across five models, multiple baselines, generalization, and ablations.
- Writing Quality: ⭐⭐⭐⭐ Logical flow from diagnostics to methodology is clear.
- Value: ⭐⭐⭐⭐ Direct practical value for secure code generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Aligned Multi-View Scripts for Universal Chart-to-Code Generation](aligned_multi-view_scripts_for_universal_chart-to-code_generation.md)
- [\[ACL 2026\] MARS2: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation](mars2_scaling_multi-agent_tree_search_via_reinforcement_learning_for_code_genera.md)
- [\[ACL 2026\] SecureVibeBench: Evaluating Secure Coding Capabilities of Code Agents with Realistic Vulnerability Scenarios](securevibebench_evaluating_secure_coding_capabilities_of_code_agents_with_realis.md)
- [\[ACL 2026\] Sense and Sensitivity: Examining the Influence of Semantic Recall on Long Context Code Understanding](sense_and_sensitivity_examining_the_influence_of_semantic_recall_on_long_context.md)
- [\[ACL 2026\] QAQ: Bidirectional Semantic Coherence for Selecting High-Quality Synthetic Code Instructions](qaq_bidirectional_semantic_coherence_for_selecting_high-quality_synthetic_code_i.md)

</div>

<!-- RELATED:END -->
