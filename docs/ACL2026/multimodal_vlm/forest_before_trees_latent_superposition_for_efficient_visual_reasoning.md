---
title: >-
  [Paper Note] Forest Before Trees: Latent Superposition for Efficient Visual Reasoning
description: >-
  [ACL 2026][Multimodal VLM][Latent reasoning] This paper proposes Laser, which performs visual reasoning in latent space through Dynamic Window Alignment Learning (DWAL). It enables the model to maintain a "probabilistic…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Latent reasoning"
  - "dynamic window alignment"
  - "semantic superposition"
  - "visual reasoning"
  - "token efficiency"
date: 2026-05-08
content_hash: d71e491da0eb5717
---

# Forest Before Trees: Latent Superposition for Efficient Visual Reasoning

**Conference**: ACL 2026  
**arXiv**: [2601.06803](https://arxiv.org/abs/2601.06803)  
**Code**: [GitHub](https://github.com/Laser-VLM/Laser)  
**Area**: Interpretability  
**Keywords**: Latent reasoning, dynamic window alignment, semantic superposition, visual reasoning, token efficiency

## TL;DR

This paper proposes Laser, which performs visual reasoning in latent space through Dynamic Window Alignment Learning (DWAL). It enables the model to maintain a "probabilistic superposition state" of future semantics rather than step-by-step exact token prediction during reasoning, achieving a "global-to-local" cognitive hierarchy. Performance reaches SOTA among latent reasoning methods on 6 benchmarks with only 6 reasoning tokens (a 97%+ reduction), outperforming Monet by an average of 5.03%.

## Background & Motivation

**Background**: Vision-Language Models (VLMs) have achieved powerful visual understanding by integrating LLMs with vision encoders, with Chain-of-Thought (CoT) being introduced for multi-step reasoning. Simultaneously, latent space reasoning methods (such as Coconut, SoftCoT, and Monet) attempt to reason within high-dimensional hidden states to avoid information loss during explicit tokenization.

**Limitations of Prior Work**: (1) Explicit textual reasoning faces an information bandwidth bottleneck—continuous visual details are lost during discrete tokenization; (2) Existing latent reasoning methods still follow standard autoregressive objectives, forcing each hidden state to strictly minimize prediction error for the next token, leading to "premature semantic collapse"—being forced to focus on a single concrete token before grasping the global context; (3) This point-to-point mapping is fundamentally inconsistent with the hierarchical nature of visual perception, which proceeds from global semantics to local features.

**Key Challenge**: The disconnect between strict token-by-token prediction objectives and the hierarchical characteristics of visual reasoning—reasoning should maintain global semantic openness in early stages and gradually narrow down to specific answers later.

**Goal**: Design a latent reasoning paradigm that allows reasoning states to encode a "superposition" of global semantics early on and gradually converge to local precise information as reasoning progresses.

**Key Insight**: Inspired by the Global Precedence Hypothesis—human visual perception processes global structures before local details—the reasoning objective is redefined from point-to-point prediction to dynamic window alignment.

**Core Idea**: Replace token-by-token prediction objectives with dynamic semantic windows. Each latent state does not need to predict the next token but instead aligns with a dynamic window containing all remaining reasoning steps. The window naturally shrinks as reasoning progresses, achieving a gradual transition from global exploration to local precision.

## Method

### Overall Architecture

Laser consists of two stages: (1) Latent visual reasoning stage—the model generates a sequence of high-dimensional hidden states as an intermediate reasoning path, aligned with dynamic semantic windows via DWAL; (2) Explicit answer generation stage—the final answer is generated using standard cross-entropy based on the evolved visual understanding. Training data is synthesized via GPT-4o as cognitive ScanPaths (270K samples) following a global-to-local order.

### Key Designs

1.  **Dynamic Window Alignment Learning (DWAL)**:
    - **Function**: Replaces standard token-by-token prediction objectives, allowing hidden states to encode global semantic superpositions.
    - **Mechanism**: For reasoning step $t$, a dynamic semantic window is defined as $W_t = \{c_k | t \leq k \leq T\}$, containing all remaining reasoning tokens from the current step to the end. The hidden state $h_t$ is not required to predict $c_{t+1}$ but aligns with the entire $W_t$. As $t$ increases, the window naturally shrinks ($|W_t| \to 1$), realizing a progression from global superposition to local precision.
    - **Design Motivation**: Standard autoregressive targets force early hidden states to collapse prematurely into single semantic points, losing global context; dynamic windows allow early states to remain open.

2.  **Self-Refined Superposition**:
    - **Function**: Constructs stable supervision targets for dynamic windows without external soft labels.
    - **Mechanism**: Extracts logits corresponding to tokens in window $W_t$ and constructs a reference superposition distribution $Q_t$ via stop-gradient and temperature-scaled Softmax. It uses the model's own estimation of future semantics as soft targets to avoid unstable self-reinforcement loops.
    - **Design Motivation**: Purely soft targets may cause optimization to diverge into high-entropy uniform distributions, necessitating a stable self-supervision mechanism.

3.  **Entropy-Regularized Intervention**:
    - **Function**: Injects hard label guidance when model uncertainty is high to prevent semantic drift.
    - **Mechanism**: Calculates the normalized entropy $H(Q_t)$ of the reference distribution. When $H(Q_t) > \eta$ (high uncertainty), it blends hard labels and the soft distribution: $P^{target}_t = \alpha \cdot \mathbf{y}_{hard} + (1-\alpha) \cdot Q_t$; otherwise, $Q_t$ is used directly. This forms an implicit curriculum—enforcing precise alignment during high uncertainty and allowing superposition exploration during low uncertainty.
    - **Design Motivation**: Completely unconstrained latent spaces may diverge into meaningless high-entropy distributions, requiring hard corrections at critical moments.

### Loss & Training

The total loss is $\mathcal{L}_{Total} = \mathcal{L}_{DWAL} + \mathcal{L}_{CE}$, where the DWAL loss aligns hidden states with hybrid targets along the reasoning chain, and the CE loss is used during answer generation. The base model is Qwen2.5-VL-7B-Instruct, with the vision tower frozen and only LLM parameters optimized. Hyperparameters are set to $\eta=0.6, \alpha=0.8$.

## Key Experimental Results

### Main Results

| Method | Type | MMVP | BLINK | SEED2+ | MMStar | Hallusion | HRBench | Overall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Qwen2.5-VL-7B | Zero-shot | 65.67 | 53.60 | 65.31 | 59.70 | 56.57 | 68.25 | 61.52 |
| Vision-R1 | RL | 72.67 | 52.71 | 68.95 | 62.67 | 63.83 | 75.12 | 65.99 |
| VL-Rethinker | RL | 72.67 | 55.55 | 70.27 | 63.20 | 71.08 | 63.50 | 66.05 |
| Monet | Latent | 68.00 | 50.71 | 65.88 | 60.33 | 56.36 | 68.00 | 61.55 |
| LVR | Latent | 64.00 | 53.60 | 47.39 | 57.93 | 65.19 | 53.62 | 56.96 |
| **Laser** | Latent | **72.00** | **56.92** | **70.05** | **60.27** | **67.72** | **72.50** | **66.58** |

### Ablation Study

**Efficiency Comparison (Average Reasoning Tokens)**

| Method | BLINK Avg Tokens | HRBench Avg Tokens | Reduction |
| :--- | :--- | :--- | :--- |
| Qwen2.5-VL-7B | 223.5 | 55.9 | — |
| VL-Rethinker | 207.0 | 143.8 | +157.2% (HRBench) |
| Monet | 118.3 | 86.8 | — |
| LVR | 8.0 | 8.0 | -96.4% |
| **Laser** | **6.0** | **5.7** | **-97.3%** |

### Key Findings

- Laser outperforms all latent reasoning baselines by an average of 5.03% and even surpasses computation-intensive RL methods like Vision-R1 and VL-Rethinker.
- It requires only 6 reasoning tokens (a 97.3% reduction) while improving performance, proving that latent superposition states can encode rich semantics in extremely compact spaces.
- Ablations show that removing DWAL (falling back to token-by-token prediction) primarily hurts fine-grained perception, while removing dynamic windows (using fixed windows) primarily hurts complex reasoning.
- Significant gains were observed in out-of-distribution (OOD) tasks (Web +8.03%, Chart +5.18%) without catastrophic forgetting.
- Latent trajectories can be decoded via the LM head into interpretable top-k tokens, revealing a multi-hop reasoning process: "Entity Localization → Spatial Analysis → Semantic Inference."

## Highlights & Insights

- The concept of "Semantic Superposition" is elegant—introducing the intuition of quantum superposition into visual reasoning, allowing reasoning states to maintain multiple possibilities before collapsing into an answer.
- Achieving a 97%+ token reduction alongside performance gains fundamentally challenges the assumption that reasoning requires lengthy Chains-of-Thought.
- The implicit curriculum design is sophisticated—using an entropy threshold to automatically control when to enforce alignment and when to allow exploration.

## Limitations & Future Work

- Slight deficiency in absolute pixel-level localization tasks (e.g., Object Localization, Jigsaw)—the "global-to-local" strategy naturally favors semantic understanding over precise measurement.
- Dependence on GPT-4o for synthetic data may inherit its inherent biases.
- Validation was limited to 7B models; performance on larger models remains unknown.
- The shrinking strategy for dynamic windows (linear) may not be optimal; adaptive shrinking could be more effective.

## Related Work & Insights

- **vs Monet**: Monet performs latent space reasoning but still generates dense sequences (118 tokens); Laser compresses this to 6 tokens via superposition.
- **vs LVR**: LVR enforces strict autoregressive reconstruction leading to semantic degradation (-9.62%); Laser avoids collapse through flexible window alignment.
- **vs Vision-R1/VL-Rethinker**: These methods improve performance via RL and long-text reasoning but incur massive computational overhead; Laser's pure latent reasoning is far more efficient.
- **vs CoT**: Explicit CoT is limited by the information bottleneck of discrete tokenization; Laser bypasses this by reasoning in continuous space.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The approach of dynamic window alignment and semantic superposition is highly novel, redefining the optimization objective for latent reasoning.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Includes 6 benchmarks, efficiency analysis, fine-grained task analysis, OOD transfer, interpretability, and detailed ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Elegantly presented concepts, with the "Forest Before Trees" metaphor consistently applied.
- Value: ⭐⭐⭐⭐⭐ Significant token reduction (97%) and performance improvement, holding high importance for real-time VLM deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] DRIFT: Transferring Reasoning Priors for Efficient MLLM Fine-Tuning](drift_transferring_reasoning_priors_for_efficient_mllm_fine-tuning.md)
- [\[ICML 2026\] Vision-aligned Latent Reasoning for Multi-modal Large Language Model](../../ICML2026/multimodal_vlm/vision-aligned_latent_reasoning_for_multi-modal_large_language_model.md)
- [\[ICML 2026\] Efficient Reasoning with Hidden Thinking](../../ICML2026/multimodal_vlm/efficient_reasoning_with_hidden_thinking.md)
- [\[CVPR 2026\] DeepSketcher: Internalizing Visual Manipulation for Multimodal Reasoning](../../CVPR2026/multimodal_vlm/deepsketcher_internalizing_visual_manipulation_for_multimodal_reasoning.md)
- [\[CVPR 2026\] AdaptVision: Efficient Vision-Language Models via Adaptive Visual Acquisition](../../CVPR2026/multimodal_vlm/adaptvision_efficient_vision-language_models_via_adaptive_visual_acquisition.md)

</div>

<!-- RELATED:END -->
