---
title: >-
  [Paper Note] Supplement Generation Training for Enhancing Agentic Task Performance
description: >-
  [ACL 2026][Model Compression][Supplement Generation Training] SGT (Supplement Generation Training) trains a small LLM (1.7B) to generate instance-specific supplement text (reasoning cues, summaries, error reminders, etc.) that is appended to the input, enabling a frozen large Actor model to solve tasks more effectively. SGT achieves an average improvement of 21% across 5 benchmarks without modifying the Actor's parameters.
tags:
  - ACL 2026
  - Model Compression
  - Supplement Generation Training
  - Prompt Optimization
  - Small Model Assisting Large Model
  - DPO
  - Agentic Tasks
date: 2026-05-08
content_hash: 2471767de8c9b17e
---

# Supplement Generation Training for Enhancing Agentic Task Performance

**Conference**: ACL 2026
**arXiv**: [2604.20727](https://arxiv.org/abs/2604.20727)
**Code**: None
**Area**: Model Compression
**Keywords**: Supplement Generation Training, Prompt Optimization, Small Model Assisting Large Model, DPO, Agentic Tasks

## TL;DR

SGT (Supplement Generation Training) trains a small LLM (1.7B) to generate instance-specific supplement text (reasoning cues, summaries, error reminders, etc.) that is appended to the input, enabling a frozen large Actor model to solve tasks more effectively. SGT achieves an average improvement of 21% across 5 benchmarks without modifying the Actor's parameters.

## Background & Motivation

**Background**: The most capable language models are increasingly deployed as closed-source APIs with no gradient access. Even when fine-tuning is feasible, the computational overhead is substantial, and the rapid release of new models quickly renders prior fine-tuning efforts obsolete. Optimization pressure naturally shifts from model parameters to the input side. Existing prompt optimization approaches include global template methods (e.g., DSPy, TextGrad optimize instruction templates) and local methods (e.g., LPO, Prompt-OIRL customize prompts per input).

**Limitations of Prior Work**: Existing methods primarily select or rearrange from a fixed set of templates rather than generating novel, input-specific content. Global methods optimize a fixed template over the entire dataset and cannot adapt to the specific needs of individual inputs. Local methods, while input-aware, still operate within a fixed template pool. Neither paradigm learns to synthesize new reasoning structures.

**Key Challenge**: Prompt optimization should not be limited to selecting or rearranging existing templates; instead, it should learn to synthesize new auxiliary content that prepares the optimal context for a frozen model. This mirrors the executive–assistant relationship: the assistant's role is not to relay instructions verbatim, but to prepare the right context, provide relevant background, and frame each problem optimally.

**Goal**: To train a small, open-source "supplement generator" that dynamically generates auxiliary text for each input, guiding a frozen large Actor model to perform better at inference time.

**Key Insight**: Task outcomes serve as a proxy reward signal for training the generator — a supplement is considered good if it helps the Actor successfully complete the task. The training pipeline combines SFT warm-start with iterative DPO to progressively improve supplement quality.

**Core Idea**: A 1.7B small model learns to generate instance-specific supplement text drawn from 8 predefined types plus a free-form type. The Actor's task outcomes serve as a proxy reward, and an SFT+DPO pipeline optimizes the generation policy — the Actor's weights remain frozen, and only the input is optimized.

## Method

### Overall Architecture

Given a task query $q$, the supplement generator $\pi_\mathcal{S}$ first produces supplement text $s$, which is then concatenated with $q$ and fed into the frozen Actor model $\pi_\mathcal{A}$ to produce the final output $y = \pi_\mathcal{A}(q, s)$. The Actor's task outcomes serve as a proxy reward signal for training the generator. Training proceeds in two stages: Warm-Start SFT (learning format and type selection) followed by iterative DPO (optimizing supplement quality and type distribution).

### Key Designs

1. **8 Supplement Types + Proxy Reward Signal**

    - **Function**: Define the formal diversity of supplements and indirectly evaluate supplement quality through task outcomes.
    - **Mechanism**: Eight supplement types are predefined: Answer (direct response), Background (contextual knowledge), CoT (step-by-step reasoning), Rephrase (paraphrase), Summary (summarization), Mistakes (common error reminders), One-shot (single example), and Pairs (contrastive correct/incorrect examples). Since supplement quality is inherently difficult to define directly, the Actor's output is used as a proxy reward $r = R(y, y^*)$, where success yields 1 and failure yields 0. For each query, the supplement set $S$ is partitioned into a positive set $S^+$ and a negative set $S^-$ based on the reward.
    - **Design Motivation**: Different tasks require different types of auxiliary information — code tasks may benefit from Pairs, while QA tasks may benefit from Background. The model is trained to learn optimal type selection autonomously.

2. **Warm-Start SFT**

    - **Function**: Teach the model the correct format for generating supplements and provide an initial signal for type selection.
    - **Mechanism**: The untrained $\pi_\mathcal{S}^0$ generates supplements for each query via $9$ types (8 predefined + Free Style) $\times$ 5 samples. Actor execution yields proxy rewards, from which successful supplements $S'^+$ are filtered. After type-stratified sampling, SFT is performed using cross-entropy loss.
    - **Design Motivation**: Untrained LLMs naturally tend to solve the task directly rather than generate supplements. SFT bridges the gap between initial behavior and the target behavior, making subsequent DPO more effective.

3. **Iterative DPO (Search-and-Focus Strategy)**

    - **Function**: Progressively refine supplement quality and identify the most effective supplement types.
    - **Mechanism**: At each DPO iteration, the current model $\pi_\mathcal{S}^t$ generates new training data for training $\pi_\mathcal{S}^{t+1}$. In the first iteration, the supplement set is constructed from three sources: Pre-Defined (8 predefined types), OOD (3 most probable non-predefined types), and Concat (3 pairs of successful types concatenated). Subsequent iterations sample directly 20 times. Preference pairs are divided into cross-type pairs (guiding type selection) and within-type pairs (guiding quality improvement), capped at 20 pairs each. The loss is $\mathcal{L} = \mathcal{L}_{DPO} + \alpha \mathcal{L}_{NLL}$ with $\alpha = 1$.
    - **Design Motivation**: Early iterations explore diverse types (search), while later iterations concentrate on the most effective types (focus). Experiments show that the Pairs type dominates on Spider and HLE, whereas DS-1000 maintains a more uniform distribution — demonstrating task-adaptive type selection learned by the model.

### Loss & Training

The SFT stage uses standard cross-entropy loss. The DPO stage uses DPO loss combined with a length-normalized NLL loss with $\alpha = 1$. Training runs for 5 iterative rounds. The supplement generator is Qwen3-1.7B (thinking mode disabled); the Actor models are Claude v3.5-sonnet-v2 and GPT-OSS-120B.

## Key Experimental Results

### Main Results (Average Gain across 5 Benchmarks × 2 Actors)

| Method | Spider | DS-1000 | HotpotQA | HLE | superGPQA | Avg. Gain |
|---|---|---|---|---|---|---|
| $\emptyset \to \pi_\mathcal{A}$ (No Supplement) | 0.674 | 0.553 | 0.694 | 0.030 | 0.288 | – |
| CoT Reasoning Scaling | 0.676 | 0.565 | 0.655 | 0.028 | 0.340 | +5% |
| TextGrad | 0.687 | 0.613 | 0.677 | 0.028 | 0.298 | +1% |
| DSPy | 0.707 | 0.598 | 0.680 | 0.032 | 0.297 | +4% |
| SGT (SFT only) | 0.718 | 0.573 | 0.689 | 0.035 | 0.273 | +5% |
| **SGT (DPO iter5)** | **0.784** | **0.593** | **0.705** | **0.049** | **0.314** | **+21%** |

### Ablation Study: Incremental Gains by Training Stage

| DPO Iteration | Avg. Gain |
|---|---|
| SFT only | +5% |
| DPO iter1 | +10% |
| DPO iter3 | +16% |
| DPO iter5 | +21% |

### Key Findings

- **SGT consistently outperforms all baselines across all benchmarks**, achieving an average gain of +21%, far exceeding TextGrad (+1%) and DSPy (+4%).
- **Structured reasoning tasks benefit the most**: Spider improves from 0.674 to 0.784 (+16.3 p.p.), as supplements help the Actor externalize intermediate reasoning steps.
- **Open-ended reasoning tasks (HotpotQA) show smaller improvements**, as the bottleneck lies in knowledge acquisition rather than reasoning organization.
- The small model (1.7B) performs extremely poorly when solving tasks directly (−23%), yet serves as an effective supplement generator (+21%), demonstrating that supplement generation $\neq$ task solving.
- Type distribution shifts substantially during DPO training: on Spider, the Pairs type gradually dominates from a uniform initialization, while DS-1000 maintains diversity — the model automatically adapts to task characteristics.
- Iterative DPO improvements persist through the 5th round without evident saturation.

## Highlights & Insights

- **The "assistant–executive" analogy is precise**: the small model is not designed to solve the problem but to "prepare the brief" for the large model. This role-division insight translates into a highly practical system architecture.
- **The search-and-focus strategy emerges naturally**: early iterations explore type diversity, while later iterations converge on the most effective types — this is not explicitly programmed but arises from DPO and natural selection dynamics.
- **Task-adaptive type selection**: the model learns to generate Pairs (contrastive correct/incorrect SQL) for Spider and Summary + CoT for DS-1000. This automatic strategy adaptation is the key reason SGT outperforms fixed-template methods.
- The framework is transferable to any "small model assisting large model" scenario, such as query rewriting in RAG or planning assistance in Agent pipelines.

## Limitations & Future Work

- Dataset scales are intentionally limited to hundreds or thousands of instances; scaling behavior in large-scale settings remains unknown.
- Supplement types are predefined as 8 categories; although DPO discovers new types during training, initialization still relies on human design.
- Evaluation is conducted under low-resource settings; whether the gains persist with larger data volumes in production is uncertain.
- The supplement generator and Actor originate from different model families; potential collinearity effects from same-family model combinations are unexplored.
- The Actor is configured at low reasoning intensity (GPT-OSS medium); SGT's gains may diminish under high reasoning intensity settings.

## Related Work & Insights

- **vs. DSPy**: DSPy optimizes a global prompt template; SGT generates novel instance-specific content. DSPy achieves +4%; SGT achieves +21%.
- **vs. TextGrad**: TextGrad uses LLM feedback to iteratively optimize prompt variables, but remains a global optimization approach. TextGrad achieves +1%; SGT achieves +21%.
- **vs. LPO**: Local prompt optimization restricts edits to "optimization tokens," still editing existing prompts. SGT directly generates entirely new auxiliary content.
- **vs. Liu et al. (2022)**: Early work demonstrated that LLM-generated contextual knowledge can improve the Actor. SGT does not fix supplement types but instead learns the optimal generation strategy.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The role formulation of the supplement generator is novel (not solving the problem, but assisting the large model); the SFT + iterative DPO pipeline demonstrates engineering innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive evaluation across 5 benchmarks, 2 Actors, multiple baselines, type distribution analysis, and ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — The "executive–assistant" analogy is vivid and Figures 1/2 are clear, though methodological details (sampling strategy) are somewhat involved.
- **Value**: ⭐⭐⭐⭐⭐ — 21% average gain + 1.7B small model + no modification of the large model = extremely high practical deployment value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Multi-View Encoders for Performance Prediction in LLM-Based Agentic Workflows](../../ICLR2026/model_compression/multi-view_encoders_for_performance_prediction_in_llm-based_agentic_workflows.md)
- [\[ACL 2026\] Task-Stratified Knowledge Scaling Laws for Post-Training Quantized LLMs](task-stratified_knowledge_scaling_laws_for_post-training_quantized_large_languag.md)
- [\[ICLR 2026\] PTQ4ARVG: Post-Training Quantization for AutoRegressive Visual Generation Models](../../ICLR2026/model_compression/ptq4arvg_post-training_quantization_for_autoregressive_visual_generation_models.md)
- [\[ACL 2026\] Polynomial Expansion Rank Adaptation: Enhancing Low-Rank Fine-Tuning with High-Order Interactions](polynomial_expansion_rank_adaptation_enhancing_low-rank_fine-tuning_with_high-or.md)
- [\[ACL 2026\] SAMoRA: Semantic-Aware Mixture of LoRA Experts for Task-Adaptive Learning](samora_semantic-aware_mixture_of_lora_experts_for_task-adaptive_learning.md)

</div>

<!-- RELATED:END -->
