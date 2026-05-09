---
title: >-
  [Paper Note] VisRef: Visual Refocusing while Thinking Improves Test-Time Scaling in Multi-Modal Large Reasoning Models
description: >-
  [CVPR 2026][LLM Reasoning][Visual refocusing] This paper proposes VisRef, a training-free visual refocusing framework that dynamically selects and re-injects semantically relevant and diverse visual tokens—chosen via a Determinantal Point Process (DPP)—into the reasoning context of Multimodal Large Reasoning Models (MLRMs) at each inference step, addressing the progressive decay of visual attention during long-chain reasoning. VisRef achieves improvements of up to 6.4% on benchmarks such as MathVista.
tags:
  - CVPR 2026
  - LLM Reasoning
  - Visual refocusing
  - test-time scaling
  - multimodal reasoning
  - determinantal point process
  - training-free
date: 2026-05-08
content_hash: 593bc54d09c6e2c2
---

# VisRef: Visual Refocusing while Thinking Improves Test-Time Scaling in Multi-Modal Large Reasoning Models

**Conference**: CVPR 2026
**arXiv**: [2603.00207](https://arxiv.org/abs/2603.00207)
**Code**: None
**Area**: LLM Reasoning / Multimodal VLM
**Keywords**: Visual refocusing, test-time scaling, multimodal reasoning, determinantal point process, training-free

## TL;DR

This paper proposes VisRef, a training-free visual refocusing framework that dynamically selects and re-injects semantically relevant and diverse visual tokens—chosen via a Determinantal Point Process (DPP)—into the reasoning context of Multimodal Large Reasoning Models (MLRMs) at each inference step, addressing the progressive decay of visual attention during long-chain reasoning. VisRef achieves improvements of up to 6.4% on benchmarks such as MathVista.

## Background & Motivation

1. **State of the Field**: MLRMs such as InternVL and Qwen-VL have achieved strong performance on visual reasoning tasks by generating chain-of-thought reasoning traces. Test-time scaling further improves performance by increasing inference-time computation.

2. **Limitations of Prior Work**: As reasoning chains grow longer, visual tokens are progressively diluted within the expanding context window, causing the model's attention to shift from image content toward textual priors, leading to visual information loss. Existing text-based self-reflection methods only extend textual reasoning without maintaining visual perception.

3. **Root Cause**: RL fine-tuning-based visual refocusing methods (e.g., Look-Back) are effective but require substantial training data and computational resources. Meanwhile, text-based test-time scaling methods may actually degrade performance on visually demanding tasks.

4. **Paper Goals**: Can visual grounding be restored purely at test time without any fine-tuning? The core questions are: which visual tokens should be re-injected, and when should reasoning terminate?

5. **Starting Point**: The approach mimics human problem-solving behavior of alternating between "glance at image → reason → glance at image again," adaptively injecting a carefully selected core subset of visual tokens at each reasoning step.

6. **Core Idea**: Use DPP to select a subset of visual tokens that are both relevant to the current reasoning state and diverse in visual coverage, ensuring that visual grounding is maintained throughout the entire reasoning process.

## Method

### Overall Architecture

VisRef operates during the inference process of an MLRM. Given an image-text input $x_{\text{input}} = [I, T]$, the model generates reasoning step $z_k$. At each step, VisRef selects a visual token subset $V_k \subseteq \mathcal{V}$ and re-injects it into the context, forming a visually augmented reasoning trajectory $\tau_{1:k} = \{(z_1, V_1), \ldots, (z_k, V_k)\}$. Once the entropy-based stopping criterion is satisfied, the final answer is generated.

### Key Designs

1. **DPP-based Visual Token Selection**

   - **Function**: Selects a visual token subset that is relevant to the reasoning context and mutually diverse at each reasoning step.
   - **Mechanism**: The second-moment matrix of the text token embeddings from the current reasoning step is computed as $M_k = \sum_{i} z_k^{(i)} (z_k^{(i)})^\top$ to capture the geometry of the reasoning subspace. A kernel function $L_k(v_i, v_j) = v_i^\top M_k v_j$ is then defined, projecting visual tokens into the textual reasoning subspace to measure similarity. The DPP determinant $\det(L_k^{V_k})$ naturally balances relevance (high diagonal entries = tokens aligned with the reasoning state) and diversity (low off-diagonal entries = broad visual coverage). Via the $\log\det$ decomposition, the objective factorizes into a relevance term $\sum \log(r_i^2)$ and a diversity term $\log\det(\bar{L}_k^{V_k})$.
   - **Design Motivation**: The naive approach of re-injecting all visual tokens is computationally expensive (2.3× inference latency increase) and introduces redundancy. DPP provides a theoretically principled and practically feasible framework for subset selection that simultaneously accounts for both relevance and diversity.

2. **Greedy Approximation**

   - **Function**: Efficiently solves the NP-hard DPP subset selection problem.
   - **Mechanism**: Starting from an empty set, the token yielding the maximum marginal gain is selected at each iteration: $v_{k,i} = \arg\max_{v} \log(\det(L_k^{V_k^{(i-1)} \cup \{v\}}) / \det(L_k^{V_k^{(i-1)}}))$. Iterating $m$ times yields a subset of size $m$. The token budget is set to $m = \lfloor 0.3|\mathcal{V}|\rfloor$ (i.e., 30% of visual tokens are selected).
   - **Design Motivation**: The greedy algorithm enjoys a classical $(1-1/e)$ approximation guarantee for DPP optimization, with empirical performance close to the exact solution.

3. **Entropy-based Adaptive Stopping Criterion**

   - **Function**: Determines when to terminate reasoning and output the final answer.
   - **Mechanism**: At each reasoning step, the entropy of the model's answer distribution is computed as $H_k = -\mathbb{E}[\log \pi_\theta(y | x_{\text{input}}, \tau_{1:k})]$. Reasoning stops when $H_k < \delta_{\text{entropy}}$ (default $\delta_{\text{entropy}} = 0.25$). A maximum step count $K_{\max} = 10$ is also imposed to prevent indefinite reasoning.
   - **Design Motivation**: Low entropy indicates that the model is sufficiently confident, and further reasoning risks "overthinking" without performance gains. High entropy signals remaining uncertainty and the need for additional reasoning, naturally adapting to problem difficulty.

### Loss & Training

VisRef is a completely training-free framework requiring no parameter updates or fine-tuning; it operates directly at inference time.

## Key Experimental Results

### Main Results

Accuracy (%) on three visual reasoning benchmarks:

| Model | Method | MathVision | MathVista | MM-Star |
|-------|--------|-----------|-----------|---------|
| InternVL3.5-8B | Standard Inference | 39.2 | 68.1 | 57.2 |
| | Text Self-Reflection | 40.1 | 73.9 | 58.3 |
| | **VisRef** | **44.6 (+5.4)** | **79.3 (+11.2)** | **63.1 (+5.9)** |
| Qwen3-VL-8B | Standard Inference | 53.8 | 74.1 | 66.5 |
| | **VisRef** | **56.6 (+2.8)** | **77.1 (+3.0)** | **69.1 (+2.6)** |
| SAIL-VL2-8B | Standard Inference | 29.8 | 73.1 | 47.7 |
| | **VisRef** | **37.3 (+7.5)** | **78.2 (+5.1)** | **55.3 (+7.6)** |

Comparison with training-based methods (InternVL3.5-8B):

| Method | MathVista | MathVision | MM-Star |
|--------|-----------|-----------|---------|
| Look-Back (RL fine-tuning) | 80.8 | 44.2 | 63.7 |
| VisRef (training-free) | 79.3 | 44.6 | 63.1 |
| Look-Back + VisRef | **83.1** | **48.2** | **66.0** |

### Ablation Study

Relevance vs. diversity ablation (InternVL3.5-8B):

| Relevance | Diversity | MathVista | MathVision | MM-Star |
|-----------|-----------|-----------|-----------|---------|
| ✓ | ✗ | 75.6 | 43.3 | 61.0 |
| ✗ | ✓ | 77.4 | 42.9 | 62.8 |
| **✓** | **✓** | **79.3** | **44.6** | **63.1** |

### Key Findings

- Text self-reflection (TSR) yields unstable gains on visual tasks (−0.6% to +2.1%), whereas VisRef consistently delivers large improvements.
- Selecting tokens by relevance alone causes a substantial performance drop; the diversity term is critical for avoiding redundancy.
- A token budget of 30% is optimal: 20% is insufficient (76.1%), while 40% yields no additional gain.
- VisRef is orthogonal to training-based methods (Look-Back); combining the two yields an additional 2–4% improvement.
- Under a fixed token budget, VisRef's multi-chain parallel inference outperforms pure text parallel inference at all budget levels.

## Highlights & Insights

- **Completely training-free plug-and-play design**: No fine-tuning, dataset construction, or architectural modification is required; VisRef can be directly applied to any pretrained MLRM, offering strong practical utility.
- **DPP usage is both elegant and effective**: Formalizing visual token selection as a joint optimization of relevance and diversity provides theoretical guarantees over heuristic approaches. The $\log\det$ decomposition into relevance and diversity terms is particularly elegant.
- **Attention visualizations validate the intuition**: VisRef demonstrably causes the model to attend continuously to task-relevant visual regions throughout reasoning, rather than progressively losing visual information.
- **Orthogonality to training-based methods**: VisRef can serve as a plug-and-play enhancement for any MLRM regardless of whether the model has undergone specialized reasoning training.

## Limitations & Future Work

- Computing the DPP kernel matrix requires $O(N^2 d)$, which incurs non-negligible overhead when the number of visual tokens is large.
- The current design uniformly injects a fixed proportion of tokens at each step; adaptive adjustment of injection volume based on problem difficulty warrants investigation.
- Validation is limited to 8B models; the effectiveness of VisRef on larger models (70B+) remains unknown.
- The stopping criterion depends on an entropy threshold hyperparameter whose optimal value may vary across tasks.

## Related Work & Insights

- **vs. Look-Back (RL fine-tuning)**: Look-Back requires 60 GPU-hours of A6000 fine-tuning, whereas VisRef achieves comparable performance with zero training cost; the two methods are complementary when combined.
- **vs. Text Self-Reflection (Budget Forcing)**: TSR only extends textual reasoning without restoring visual attention, yielding unstable results on visual tasks.
- **vs. Visual Tool-Calling Methods**: Such methods require architectural modifications or SFT/RL training; VisRef is more lightweight and general-purpose.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Applying DPP to visual token selection at inference time is an entirely novel approach.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three models across three benchmarks with comprehensive ablations, though experiments on larger models are absent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Theoretical derivations are elegant, with a clear logical progression from problem formulation to solution.
- **Value**: ⭐⭐⭐⭐⭐ — Training-free and plug-and-play; practical value is exceptionally high.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Understanding the Role of Hallucination in Reinforcement Post-Training of Multimodal Reasoning Models](understanding_the_role_of_hallucination_in_reinforcement_post-training_of_multim.md)
- [\[CVPR 2026\] Rationale-Enhanced Decoding for Multi-modal Chain-of-Thought](rationale-enhanced_decoding_for_multi-modal_chain-of-thought.md)
- [\[CVPR 2026\] Step-CoT: Stepwise Visual Chain-of-Thought for Medical Visual Question Answering](step-cot_stepwise_visual_chain-of-thought_for_medical_visual_question_answering.md)
- [\[ICLR 2026\] Efficient Test-Time Scaling for Small Vision-Language Models](../../ICLR2026/llm_reasoning/efficient_test-time_scaling_for_small_vision-language_models.md)
- [\[ICLR 2026\] ATTS: Asynchronous Test-Time Scaling via Conformal Prediction](../../ICLR2026/llm_reasoning/atts_asynchronous_test-time_scaling_via_conformal_prediction.md)

<!-- RELATED:END -->
