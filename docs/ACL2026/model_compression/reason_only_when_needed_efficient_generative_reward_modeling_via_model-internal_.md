---
title: >-
  [Paper Note] Reason Only When Needed: Efficient Generative Reward Modeling via Model-Internal Uncertainty
description: >-
  [ACL 2026][Model Compression][Generative Reward Model] Proposes the E-GRM framework, which utilizes the convergence behavior of parallel decoding to estimate uncertainty…
tags:
  - "ACL 2026"
  - "Model Compression"
  - "Generative Reward Model"
  - "Dynamic CoT Triggering"
  - "Model-Internal Uncertainty"
  - "Discriminative Scoring"
  - "Inference Efficiency"
date: 2026-05-08
content_hash: 4f4c5ecfd014221c
---

# Reason Only When Needed: Efficient Generative Reward Modeling via Model-Internal Uncertainty

**Conference**: ACL 2026  
**arXiv**: [2604.10072](https://arxiv.org/abs/2604.10072)  
**Code**: None  
**Area**: Model Compression/LLM Efficiency  
**Keywords**: Generative Reward Model, Dynamic CoT Triggering, Model-Internal Uncertainty, Discriminative Scoring, Inference Efficiency

## TL;DR

Proposes the E-GRM framework, which utilizes the convergence behavior of parallel decoding to estimate uncertainty, triggering CoT reasoning only when necessary. It evaluates the quality of reasoning paths through a discriminative scorer trained with hybrid loss, achieving SOTA performance on multiple reward model benchmarks while reducing inference latency by 62%.

## Background & Motivation

**Background**: Generative Reward Models (GRM) enhance the reasoning evaluation capabilities of LLMs via CoT prompting, showing outstanding performance in complex tasks such as mathematical problem solving and multi-step decision making.

**Limitations of Prior Work**: Existing GRMs suffer from two core issues. First, CoT reasoning is applied indiscriminately to all inputs regardless of difficulty; simple questions still undergo the full CoT process, leading to significant unnecessary computational overhead. Second, existing methods primarily rely on voting mechanisms to aggregate CoT output answers, which is a coarse-grained evaluation that fails to distinguish fine-grained quality differences between reasoning paths.

**Key Challenge**: A dual bottleneck of efficiency and quality—there is a need to adaptively allocate reasoning resources based on task complexity while simultaneously requiring a more refined scoring mechanism to differentiate reasoning quality. Existing adaptive CoT methods (e.g., AdaCoT) rely on task-specific heuristics or handcrafted features, limiting their generalization.

**Goal**: (1) Identify a task-agnostic signal to determine whether CoT is necessary; (2) Design a more fine-grained reasoning path evaluation method than simple voting.

**Key Insight**: The authors observe that when performing multiple parallel decodings for the same prompt, outputs for simple questions converge quickly, whereas outputs for difficult questions exhibit high divergence—this convergence behavior serves as a natural indicator of problem complexity.

**Core Idea**: Use the consensus formed during the model's own parallel generation as an uncertainty estimation signal to dynamically decide whether to trigger CoT, while training a lightweight discriminative scorer using a hybrid regression-ranking loss for precise scoring.

## Method

### Overall Architecture

E-GRM consists of two core modules: (1) a dynamic CoT triggering mechanism based on model-internal uncertainty; (2) a discriminative scoring module based on hybrid loss. Training involves two stages: first, SFT to enable the model to learn both short-reasoning and long-reasoning modes, followed by preference optimization via extended GRPO. During inference, the model first quickly determines if CoT is needed; if so, it generates multiple reasoning paths and selects the optimal one using the scorer.

### Key Designs

1.  **Dynamic CoT Triggering**:

    - **Function**: Automatically decides whether to enable CoT reasoning based on problem complexity.
    - **Mechanism**: Performs $M$ parallel decodings for input $x$ (using different temperatures/sampling parameters) and calculates answer consistency: $\text{Consensus}(x) = \max_y \text{Count}(y) / M$. If $\text{Consensus} \geq \tau$ (default 0.8), the consensus answer is output directly; otherwise, full CoT generation is triggered. In experiments, approximately 58% of samples were identified as "short reasoning," allowing them to skip CoT.
    - **Design Motivation**: Leverages the model's own generation behavior as a complexity probe without requiring external features or task-specific heuristics, achieving true task-agnostic adaptive reasoning.

2.  **Discriminative Scoring Module**:

    - **Function**: Performs fine-grained quality scoring of generated reasoning paths.
    - **Mechanism**: A lightweight scoring model $\mathcal{S}_\phi$ is trained to output a quality score in $[0, 1]$. The loss function combines Huber Loss (for regression robustness, transitioning smoothly from L2 to L1 for outliers) and Hinge Loss (for ranking discriminability, enforcing a margin $m$ between high-quality and low-quality paths). Total loss is $\mathcal{L} = \alpha \cdot \ell_{\text{Huber}} + (1-\alpha) \cdot \ell_{\text{Hinge}}$.
    - **Design Motivation**: Pure voting mechanisms only check answer consistency and ignore the quality of the reasoning process; hybrid loss allows the scorer to both calibrate absolute quality and reliably distinguish subtle differences.

3.  **Extended GRPO Preference Optimization (Coupled-GRPO)**:

    - **Function**: Optimizes the policy during the RL stage using paired preference data.
    - **Mechanism**: Introduces a paired reward signal based on standard GRPO: $R_{\text{pair}} = \mathcal{S}_\phi(x, r^+) - \mathcal{S}_\phi(x, r^-) + \beta \cdot \mathbb{I}(\text{Ans}(r^+) = y)$. This directly compares the scorer output differences between positive and negative samples, providing a stronger learning signal.
    - **Design Motivation**: While standard GRPO calculates relative rewards within independently sampled groups, paired data naturally contains contrastive information; explicitly utilizing this structure provides more targeted gradients.

### Loss & Training

Training follows two stages: (1) An SFT stage that mixes short-reasoning samples (direct answer prediction) and long-reasoning samples (learning CoT sequences), with datasets partitioned automatically via uncertainty estimation; (2) A GRPO stage using paired preference data and the discriminative scorer for alignment optimization, including KL regularization to prevent excessive deviation from the reference policy.

## Key Experimental Results

### Main Results

| Benchmark | Metric | E-GRM (32B) | Prev. SOTA | Gain |
| :--- | :--- | :--- | :--- | :--- |
| RM-Bench | Avg | 79.2% | 76.4% (14B) | +2.8% |
| RMB | Overall | 0.743 | 0.738 (GPT-4o) | +0.005 |
| RewardBench | Overall | 91.5% | 90.0% (Self-taught-70B) | +1.5% |
| RewardBench | Reasoning | 95.4% | 88.4% (Self-taught-70B) | +7.0% |

### Ablation Study

| Configuration | Acc (%) | FLOPs (T) | Latency (s) |
| :--- | :--- | :--- | :--- |
| Full E-GRM | 78.4 | 15.7 | 2.2 |
| w/o Dynamic CoT | 75.2 | 23.4 | 3.4 |
| w/o Discrim. Scoring | 72.8 | 15.9 | 2.2 |
| Base CoT-GRM | 69.1 | 23.7 | 3.6 |

### Key Findings

- **Discriminative Scoring Module contributes the most**: Removing it leads to a 5.6% drop in accuracy, indicating that fine-grained scoring is crucial for reasoning quality.
- **Dynamic CoT Triggering yields a 49% reduction in FLOPs and a 55% reduction in latency**, while accuracy actually improves by 3.2%, proving that unnecessary CoT can introduce error propagation.
- **Comparison with heuristic methods like AdaCoT**: E-GRM achieves higher accuracy (78.4% vs 76.8%) and lower latency (2.2s vs 2.9s) without requiring task-specific priors.
- **Extended GRPO provides consistent but modest gains** over standard GRPO (MATH: 78.4% vs 76.9%).

## Highlights & Insights

- **Parallel decoding consistency as a complexity probe**: This is an elegant design—leveraging the model's own behavioral characteristics rather than external signals to judge reasoning needs. It naturally possesses task-agnosticism and zero additional parameter cost. This idea is transferable to any scenario requiring adaptive computation (e.g., early exit, dynamic depth).
- **Hybrid Regression-Ranking Loss**: The combination of Huber + Hinge cleverly resolves the conflict of requiring a scorer to achieve both "absolute calibration" and "relative ranking," making it more robust than pure MSE or pure ranking losses.
- **58% of samples identified as "simple questions" not requiring CoT**: This proportion itself is a significant finding—it indicates that current GRMs suffer from severe computational waste on a large number of simple tasks.

## Limitations & Future Work

- Uncertainty estimation via parallel decoding itself requires $M$ forward passes ($M=5$). Although the overhead is smaller than full CoT, it is not zero-cost; its acceptability in extreme low-latency scenarios needs verification.
- The selection of threshold $\tau$ and the number of parallels $M$ is currently manual; different task domains may require different settings.
- The discriminative scorer requires annotated quality data for training, and the cost of data acquisition may limit rapid deployment in new domains.
- **Future Explorations**: Combining uncertainty estimation with speculative decoding, or replacing multiple samplings with internal representations from a single forward pass (such as attention entropy).

## Related Work & Insights

- **vs DeepSeek-GRM**: DeepSeek-GRM is also a generative reward model but lacks an adaptive reasoning mechanism, using CoT uniformly for all inputs. E-GRM significantly reduces computational costs at equivalent or higher accuracy via dynamic triggering.
- **vs AdaCoT**: AdaCoT uses task-specific heuristics based on estimated solution length to decide if CoT is needed. E-GRM’s parallel consistency method is entirely task-agnostic and experimentally proven to be superior.

## Rating

- Novelty: ⭐⭐⭐⭐ Parallel decoding consistency as an uncertainty signal is a novel entry point, though the hybrid loss and GRPO extensions are somewhat incremental.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across three major benchmarks, thorough ablation studies, and well-designed comparison experiments with AdaCoT.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and detailed method description, though some formulas are quite lengthy.
- Value: ⭐⭐⭐⭐ Successfully addresses the efficiency pain point of GRMs; the 62% latency reduction holds significant value for practical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Latent-Condensed Transformer for Efficient Long Context Modeling](latent-condensed_transformer_for_efficient_long_context_modeling.md)
- [\[ICML 2026\] When Shared Knowledge Hurts: Spectral Over-Accumulation in Model Merging](../../ICML2026/model_compression/when_shared_knowledge_hurts_spectral_over-accumulation_in_model_merging.md)
- [\[ACL 2026\] Cognitive-Uncertainty Guided Knowledge Distillation for Accurate Classification of Student Misconceptions](cognitive-uncertainty_guided_knowledge_distillation_for_accurate_classification_.md)
- [\[AAAI 2026\] Credal Ensemble Distillation for Uncertainty Quantification](../../AAAI2026/model_compression/credal_ensemble_distillation_for_uncertainty_quantification.md)
- [\[NeurIPS 2025\] Linear Attention for Efficient Bidirectional Sequence Modeling](../../NeurIPS2025/model_compression/linear_attention_for_efficient_bidirectional_sequence_modeling.md)

</div>

<!-- RELATED:END -->
