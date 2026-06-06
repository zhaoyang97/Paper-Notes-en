---
title: >-
  [Paper Note] Supplement Generation Training for Enhancing Agentic Task Performance
description: >-
  [ACL 2026][LLM Agent][Supplement Generation Training] SGT (Supplement Generation Training) trains a small LLM (1.7B) to generate instance-specific supplement text (reasoning clues, summaries, error reminders…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Supplement Generation Training"
  - "Prompt Optimization"
  - "Small Model Assisting Large Model"
  - "DPO"
  - "Agentic Tasks"
date: 2026-05-08
content_hash: 7c0096195fe2ae35
---

# Supplement Generation Training for Enhancing Agentic Task Performance

**Conference**: ACL 2026  
**arXiv**: [2604.20727](https://arxiv.org/abs/2604.20727)  
**Code**: None  
**Area**: Model Compression  
**Keywords**: Supplement Generation Training, Prompt Optimization, Small Model Assisting Large Model, DPO, Agentic Tasks

## TL;DR

SGT (Supplement Generation Training) trains a small LLM (1.7B) to generate instance-specific supplement text (reasoning clues, summaries, error reminders, etc.). When appended to the input, this allows a frozen large Actor model to solve tasks more effectively, achieving an average improvement of 21% across 5 benchmarks without modifying the large model's parameters.

## Background & Motivation

**Background**: The most powerful language models are increasingly deployed as closed-source APIs with inaccessible gradients. Even when fine-tuning is possible, the high computational overhead and the continuous release of new models make older fine-tuned versions obsolete quickly. Consequently, optimization pressure has shifted from the model to the input side. Existing prompt optimization methods include: global template methods (DSPy, TextGrad optimizing instruction templates) and local methods (LPO, Prompt-OIRL customizing prompts for each input).

**Limitations of Prior Work**: Existing methods primarily select or reorder from existing templates rather than generating new, input-specific content. Global methods optimize a fixed template across the entire dataset, failing to adapt to the specific needs of individual inputs. While local methods customize for each input, they still operate within fixed template pools. Neither can learn to synthesize new reasoning structures.

**Key Challenge**: Prompt optimization should not be limited to selecting or reordering existing templates but should learn to synthesize new auxiliary content to prepare the optimal context for the frozen model. This is analogous to the relationship between an executive and an assistant—the assistant's job is not to relay instructions verbatim but to prepare the correct context, provide relevant background, and frame each problem in the best possible way.

**Goal**: To train a small, open-source "supplement generator" that dynamically generates auxiliary text for each input, guiding a frozen large Actor to perform better during inference.

**Key Insight**: Use task outcomes as proxy reward signals to train the generator—if a supplement helps the Actor solve the task (success), then that supplement is good. Use an SFT warm-start + iterative DPO training workflow to progressively improve supplement quality.

**Core Idea**: A 1.7B small model learns to generate instance-specific supplement text (8 predefined types + free style). Using the Actor's task results as proxy rewards, the generation strategy is optimized via SFT+DPO—optimizing only the input without modifying Actor weights.

## Method

### Overall Architecture

The input task query $q$ is first processed by the supplement generator $\pi_\mathcal{S}$ to generate supplement text $s$. Then, $s$ and $q$ are concatenated and fed into the frozen Actor model $\pi_\mathcal{A}$ to produce the final output $y = \pi_\mathcal{A}(q, s)$. The Actor's task result serves as the proxy reward signal to train the generator. Training consists of two stages: Warm-Start SFT (learning format and type selection) + Iterative DPO (optimizing supplement quality and type distribution).

### Key Designs

1.  **8 Supplement Types + Proxy Reward Signal**:
    - **Function**: Defines the formal diversity of supplements and indirectly evaluates supplement quality via task outcomes.
    - **Mechanism**: Predefines 8 supplement types: Answer (direct answer), Background (background knowledge), CoT (step-by-step reasoning), Rephrase (paraphrasing), Summary (abstract), Mistakes (common error reminders), One-shot (single example), and Pairs (contrastive positive/negative examples). Since supplement quality itself is difficult to define, the Actor's output result is used as a proxy reward $r = R(y, y^*)$, where success is 1 and failure is 0. For each query, the supplement set $S$ is divided into a positive set $S^+$ and a negative set $S^-$ based on rewards.
    - **Design Motivation**: Different tasks require different types of auxiliary information—coding tasks might need Pairs, while QA tasks might need Background. This allows the model to learn the optimal type selection itself.

2.  **Warm-Start SFT**:
    - **Function**: Teaches the model the correct format for generating supplements and provides initial learning for type selection.
    - **Mechanism**: An untrained $\pi_\mathcal{S}^0$ is used to generate supplements for each query across 9 types (8 + Free Style) with 5 samples each. Proxy rewards are obtained via Actor execution. Successful supplements $S'^+$ are filtered, and after type-stratified sampling, SFT is performed using cross-entropy loss.
    - **Design Motivation**: Untrained LLMs naturally tend to solve tasks directly rather than generate supplements. SFT bridges the gap between initial behavior and target behavior, making subsequent DPO more effective.

3.  **Iterative DPO (search-and-focus strategy)**:
    - **Function**: Progressively optimizes supplement quality and discovers the most effective supplement types.
    - **Mechanism**: In each DPO iteration, the current model $\pi_\mathcal{S}^t$ generates new training data to train the next $\pi_\mathcal{S}^{t+1}$. The supplement set for the first round is constructed from three sources: Pre-Defined (8 types), OOD (3 most likely non-predefined types), and Concat (concatenations of 3 pairs of successful types). Subsequent iterations use 20 direct samples. Preference pairs are divided into inter-type pairs (guiding type selection) and intra-type pairs (guiding quality improvement), with a cap of 20 pairs per category. The loss is $\mathcal{L} = \mathcal{L}_{DPO} + \alpha \mathcal{L}_{NLL}$ ($\alpha = 1$).
    - **Design Motivation**: Early iterations explore diverse types (search), while later iterations concentrate on the most effective types (focus). Experiments show the Pairs type dominates on Spider and HLE, whereas the distribution is more uniform on DS-1000—indicating the model learns task-adaptive type selection.

### Loss & Training
The SFT stage uses standard cross-entropy loss. The DPO stage uses DPO loss plus a length-normalized NLL loss with $\alpha = 1$. Training lasts for 5 iterations. The supplement generator is Qwen3-1.7B (thinking mode disabled), and the Actors are v3.5-sonnet-v2 and GPT-OSS-120B.

## Key Experimental Results

### Main Results (Average Gain Across 5 Benchmarks × 2 Actors)

| Method | Spider | DS-1000 | HotpotQA | HLE | superGPQA | Avg. Gain |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| ∅→π_A (No Supplement) | 0.674 | 0.553 | 0.694 | 0.030 | 0.288 | – |
| CoT Reasoning Extension | 0.676 | 0.565 | 0.655 | 0.028 | 0.340 | +5% |
| TextGrad | 0.687 | 0.613 | 0.677 | 0.028 | 0.298 | +1% |
| DSPy | 0.707 | 0.598 | 0.680 | 0.032 | 0.297 | +4% |
| SGT (SFT only) | 0.718 | 0.573 | 0.689 | 0.035 | 0.273 | +5% |
| **SGT (DPO iter5)** | **0.784** | **0.593** | **0.705** | **0.049** | **0.314** | **+21%** |

### Ablation Study: Training Phase Increments

| DPO Iteration | Avg. Gain |
| :--- | :--- |
| SFT only | +5% |
| DPO iter1 | +10% |
| DPO iter3 | +16% |
| DPO iter5 | +21% |

### Key Findings
- **SGT consistently outperforms all baselines across all benchmarks**, with an average gain of +21%, significantly exceeding TextGrad (+1%) and DSPy (+4%).
- **Structured reasoning tasks benefit the most**: Spider improved from 0.674 to 0.784 (+16.3%p) because supplements helped the Actor externalize intermediate reasoning steps.
- **Open-ended reasoning tasks (HotpotQA) show smaller improvements**, as the bottleneck lies in knowledge acquisition rather than reasoning organization.
- The small model (1.7B) performs extremely poorly when solving tasks directly (-23%), but is highly effective as a supplement generator (+21%), indicating that supplement generation $\neq$ task solving.
- Type distribution shifts significantly during DPO training: on Spider, the Pairs type gradually dominates from a uniform distribution, while DS-1000 maintains diversity—the model automatically adapts to task characteristics.
- Iterative improvements from DPO continue through the 5th round without obvious saturation.

## Highlights & Insights
- **The "Assistant-Executive" analogy is precise**: The small model is not there to solve the problem, but to "prepare lessons" for the large model. This insight into role division translates into a highly practical system architecture.
- **Emergence of search-and-focus strategy**: Early iterations explore type diversity, while later ones focus on the most effective types—this is not explicitly programmed but is a result of DPO and natural selection.
- **Task-adaptive type selection**: The model learns to generate Pairs (contrasting correct/incorrect SQL) for Spider and Summary + CoT for DS-1000—this automatic strategy adaptation is why SGT surpasses fixed-template methods.
- This framework can be transferred to any "small model assisting large model" scenario, such as query rewriting in RAG or planning assistance in Agents.

## Limitations & Future Work
- Dataset scales were intentionally limited to the hundreds-to-thousands range; scaling behavior in large-scale scenarios remains unknown.
- Supplement types were predefined to 8; although new types can be discovered during DPO, initialization still relies on manual design.
- Evaluation was conducted in low-resource settings; whether the effects hold with larger data volumes in actual production is uncertain.
- The supplement generator and Actor come from different model families, but potential collinearity issues in same-family combinations were not explored.
- The Actor was set to medium reasoning intensity (GPT-OSS medium); the gains from SGT might narrow under high reasoning intensity.

## Related Work & Insights
- **vs DSPy**: DSPy optimizes global prompt templates, while SGT generates new instance-specific content. DSPy +4% vs. SGT +21%.
- **vs TextGrad**: TextGrad uses LLM feedback to iteratively optimize prompt variables, but this is also global optimization. TextGrad +1% vs. SGT +21%.
- **vs LPO**: Local Prompt Optimization restricts edits to "optimized tokens," essentially editing existing prompts. SGT generates entirely new auxiliary content.
- **vs Liu et al. (2022)**: Early work showed that context knowledge generated by LLMs can improve Actors. SGT does not fix the supplement type but learns the optimal strategy.

## Rating
- Novelty: ⭐⭐⭐⭐ The role positioning of the supplement generator is novel (not solving the problem, only assisting the large model), and the SFT+iterative DPO training workflow is an engineering innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive coverage of 5 benchmarks, 2 Actors, multiple baseline comparisons, type distribution analysis, and ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The "Executive-Assistant" analogy is vivid, and Figure 1/2 are clear, though some method details (sampling strategy) are a bit tedious.
- Value: ⭐⭐⭐⭐⭐ +21% average gain + 1.7B small model + no modification to the large model = extremely high practical deployment value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] PerfGuard: A Performance-Aware Agent for Visual Content Generation](../../ICLR2026/llm_agent/radiometrically_consistent_gaussian_surfels_for_inverse_rendering.md)
- [\[ICLR 2026\] AgentSynth: Scalable Task Generation for Generalist Computer-Use Agents](../../ICLR2026/llm_agent/agentsynth_scalable_task_generation_for_generalist_computer-use_agents.md)
- [\[ACL 2026\] GOAT: A Training Framework for Goal-Oriented Agent with Tools](goat_a_training_framework_for_goal-oriented_agent_with_tools.md)
- [\[ACL 2026\] CoEvolve: Training LLM Agents via Agent-Data Mutual Evolution](coevolve_training_llm_agents_via_agent-data_mutual_evolution.md)
- [\[ACL 2026\] AdaRubric: Task-Adaptive Rubrics for Reliable LLM Agent Evaluation and Reward Learning](adarubric_task-adaptive_rubrics_for_reliable_llm_agent_evaluation_and_reward_lea.md)

</div>

<!-- RELATED:END -->
