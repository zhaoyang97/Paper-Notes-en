---
title: >-
  [Paper Note] CarO: Chain-of-Analogy Reasoning Optimization for Robust Content Moderation
description: >-
  [ACL 2026][Content Moderation] This paper proposes CarO (Chain-of-Analogy Reasoning Optimization), a two-stage training framework that enables LLMs to autonomously generate analogical reference cases during inference for content moderation. The framework combines RAG-guided analogical chain generation, SFT, and customized DPO. On ambiguous moderation benchmarks, CarO achieves an average F1 improvement of 24.9%, substantially outperforming reasoning models (DeepSeek R1) and dedicated moderation models (LLaMA Guard).
tags:
  - ACL 2026
  - Content Moderation
  - Analogical Reasoning
  - Direct Preference Optimization
  - LLM Reasoning
  - Decision Shortcuts
date: 2026-05-08
content_hash: dfea5e224053b18b
---

# CarO: Chain-of-Analogy Reasoning Optimization for Robust Content Moderation

**Conference**: ACL 2026
**arXiv**: [2604.10504](https://arxiv.org/abs/2604.10504)
**Code**: None
**Area**: Information Retrieval
**Keywords**: Content Moderation, Analogical Reasoning, Direct Preference Optimization, LLM Reasoning, Decision Shortcuts

## TL;DR
This paper proposes CarO (Chain-of-Analogy Reasoning Optimization), a two-stage training framework that enables LLMs to autonomously generate analogical reference cases during inference for content moderation. The framework combines RAG-guided analogical chain generation, SFT, and customized DPO. On ambiguous moderation benchmarks, CarO achieves an average F1 improvement of 24.9%, substantially outperforming reasoning models (DeepSeek R1) and dedicated moderation models (LLaMA Guard).

## Background & Motivation

**Background**: Content moderation is a core task for maintaining digital ecosystem safety. Traditional discriminative models (e.g., BERT) suffer from poor OOD generalization and limited interpretability. Recent LLMs have demonstrated the ability to generate reasoning chains via prompting, ICL, and post-training, enabling more interpretable moderation decisions.

**Limitations of Prior Work**: Even state-of-the-art reasoning models such as DeepSeek R1 frequently fail on ambiguous moderation cases. Analysis reveals that these errors stem from "decision shortcuts" embedded in the context—surface-level cues that mislead the reasoning process. For instance, the statement "Every Indian person I know dances upon hearing music" is a benign observation, yet DeepSeek R1 incorrectly classifies it as discriminatory upon detecting the mention of a specific ethnic group.

**Key Challenge**: LLMs are easily misled by superficial semantic cues in boundary cases, lacking the analogical reasoning capability characteristic of human moderation experts—who first recall similar precedents and then integrate those precedents with guidelines to reach a judgment.

**Goal**: To enable LLMs to learn analogical reasoning, such that during inference they autonomously generate relevant analogical cases and make more robust moderation decisions grounded in analogy.

**Key Insight**: The work is motivated by the cognitive workflow of human domain experts—when handling ambiguous cases, experts first retrieve similar precedents (analogical retrieval) and then synthesize insights from those precedents alongside moderation guidelines (analogical reasoning).

**Core Idea**: A two-stage training procedure internalizes analogical reasoning into LLMs: Stage 1 uses RAG + SFT to guide the generation of analogical reasoning chains; Stage 2 applies customized DPO to reinforce analogical reasoning via preference pairs contrasting chains with and without analogies.

## Method

### Overall Architecture
**Stage 1**: For each training sample, semantically similar cases are retrieved → DeepSeek R1 generates reasoning chains that explicitly incorporate analogical references → a reflection-correction step fixes erroneous reasoning → SFT is applied. **Stage 2**: Reasoning chains generated with RAG input serve as positive samples, while chains generated without RAG serve as negative samples → DPO training reinforces analogical reasoning. At inference time, no external retrieval is required; the model autonomously generates analogical cases.

### Key Designs

1. **Guided Chain-of-Analogy Generation (COAT)**:

    - **Function**: Generates high-quality reasoning chains containing analogical references for each training sample.
    - **Mechanism**: For each training sample $\mathbf{x}_i$, the top-$k$ semantically similar cases (with labels) are retrieved. The retrieved results are injected into the prompt, instructing DeepSeek R1 to explicitly cite these analogical cases within the reasoning chain. After generation, the reasoning conclusion is checked for consistency with the ground-truth label; inconsistencies trigger a reflection-correction step.
    - **Design Motivation**: Direct prompting for reasoning chains lacks reference to precedents. RAG-guided chains embed analogical patterns into the training data, allowing the SFT model to internalize this reasoning mode.

2. **Customized DPO for Analogical Reasoning Reinforcement**:

    - **Function**: Explicitly reinforces the model to prioritize analogical reasoning over generic reasoning.
    - **Mechanism**: Positive samples $\mathbf{r}^+$ are reasoning chains generated by the SFT model given RAG input (containing analogical references); negative samples $\mathbf{r}^-$ are chains generated by the same model from the original input only (without analogical references). Standard DPO loss is applied to make the model prefer analogy-rich reasoning chains.
    - **Design Motivation**: After SFT, the model can already produce analogical reasoning but not consistently. The goal of DPO is not to improve F1 (already high after SFT) but to enhance the explicitness, consistency, and interpretability of analogical reasoning.

3. **Autonomous Analogy Generation at Inference (No External Retrieval)**:

    - **Function**: Eliminates the need to maintain a retrieval database at deployment.
    - **Mechanism**: Following two-stage training, the model has internalized analogical reasoning patterns and can autonomously "imagine" analogical cases directly from the input at inference time, without actual retrieval.
    - **Design Motivation**: RAG-based approaches are constrained by static datasets, and retrieved cases may not be optimal for a given scenario. Internalized analogical capability enables the dynamic generation of reference cases tailored to the current input.

## Key Experimental Results

### Main Results (Chinese Moderation Dataset + English Benchmarks)

| Model | Politics | Pornography | Violence | Bias | Gambling | Harmless | Avg. F1 |
|-------|----------|-------------|----------|------|----------|----------|---------|
| Qwen2.5-7B-Instruct | 54.9 | 81.9 | 70.0 | 60.1 | 84.3 | 48.8 | 64.3 |
| DeepSeek R1 | - | - | - | - | - | - | ~70 |
| LLaMA Guard | - | - | - | - | - | - | ~65 |
| **CarO (Ours)** | **Best** | **Best** | **Best** | **Best** | **Best** | **Best** | **89.2** |

### Ablation Study

| Configuration | F1 | CoA Rate (%) |
|---------------|----|--------------|
| Baseline (no training) | 64.3 | 0.0 |
| + RAG-SFT | 85.5 (+21.2) | 89.5 |
| + Reflection Correction | 88.8 (+3.3) | 93.5 |
| + DPO | **89.2 (+0.4)** | **99.3** |

### Cross-Benchmark Generalization (OOD Testing)

| Dataset | Qwen2.5-7B → CarO |
|---------|-------------------|
| Aegis (ID) | 78.7 → **87.1** |
| OpenAI (OOD) | 70.8 → **74.2** |
| Toxic-Chat (OOD) | 93.3 → **95.0** |

### Key Findings
- **F1 improves by 24.9 percentage points (64.3→89.2)**, with the RAG-SFT stage contributing the largest gain (+21.2).
- **DPO yields only a marginal F1 gain (+0.4) but raises the analogical reasoning rate from 93.5% to 99.3%**, indicating that its primary role is to enhance reasoning consistency rather than accuracy.
- **Reflection correction contributes a 3.3pp improvement**, confirming that automatically generated reasoning chains contain errors that benefit from correction.
- **OOD benchmarks also show improvements** (Aegis +8.4, OpenAI +3.4), demonstrating cross-domain transferability of analogical reasoning capability.
- **Inference requires no RAG yet performance does not degrade**, confirming that the model has successfully internalized analogical reasoning patterns.

## Highlights & Insights
- The **diagnosis of "decision shortcuts"** is precise—models are not incapable but are being misled, which explains why even powerful reasoning models fail on moderation tasks.
- The **two-stage training design** is broadly transferable: SFT is first used to elicit capability emergence, followed by DPO to reinforce consistency. This "elicitation → reinforcement" paradigm applies to any task requiring a specific reasoning pattern.
- **Autonomous analogy generation at inference** is more flexible than RAG—the model can "imagine" the most appropriate reference for a given case without being constrained by a fixed database.

## Limitations & Future Work
- Analogical reasoning chains in the training data are generated by DeepSeek R1, and their quality is bounded by that model's capabilities.
- Retrieving $k=32$ reference cases during training imposes non-trivial memory and computational costs.
- The primary training data is in Chinese; validation on English benchmarks is relatively limited.
- DPO yields only a marginal F1 gain (+0.4); more efficient methods for reasoning reinforcement warrant investigation.
- Autonomously generated analogical cases may lack factual accuracy, and no verification mechanism is currently in place.

## Related Work & Insights
- **vs. DeepSeek R1**: R1 possesses strong reasoning capabilities but lacks analogical references, causing it to be misled by surface-level cues in ambiguous cases.
- **vs. LLaMA Guard**: A dedicated moderation model that lacks an interpretable reasoning process.
- **vs. RAG-based Methods**: Static retrieval cannot dynamically adapt to diverse inputs; CarO eliminates retrieval dependency entirely by internalizing analogical capability through training.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The combination of analogical reasoning and DPO for content moderation is novel, though individual components are not new.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-benchmark evaluation with ablations and OOD testing; however, the main experiments are predominantly in Chinese.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clearly articulated; the connection to cognitive psychology is persuasive.
- **Value**: ⭐⭐⭐⭐ — Offers direct practical value for the content moderation domain; the analogical reasoning paradigm is transferable to other tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ChAIRO: Contextual Hierarchical Analogical Induction and Reasoning Optimization for LLMs](chairo_contextual_hierarchical_analogical_induction_and_reasoning_optimization_f.md)
- [\[ACL 2026\] Enhancing LLM-based Search Agents via Contribution Weighted Group Relative Policy Optimization](enhancing_llm-based_search_agents_via_contribution_weighted_group_relative_polic.md)
- [\[ACL 2026\] Context Attribution with Multi-Armed Bandit Optimization](context_attribution_with_multi-armed_bandit_optimization.md)
- [\[NeurIPS 2025\] Chain-of-Retrieval Augmented Generation (CoRAG)](../../NeurIPS2025/information_retrieval/chain-of-retrieval_augmented_generation.md)
- [\[NeurIPS 2025\] SymRTLO: Enhancing RTL Code Optimization with LLMs and Neuron-Inspired Symbolic Reasoning](../../NeurIPS2025/information_retrieval/symrtlo_enhancing_rtl_code_optimization_with_llms_and_neuron-inspired_symbolic_r.md)

</div>

<!-- RELATED:END -->
