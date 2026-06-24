---
title: >-
  [Paper Note] Unable to Forget: Proactive Interference Reveals Working Memory Limits in LLMs Beyond Context Length
description: >-
  [ICML 2025][Information Retrieval & RAG][Proactive Interference] Drawing on the Proactive Interference (PI) paradigm from cognitive science, this study finds that the information retrieval accuracy of LLMs decreases log-linearly to zero as the amount of interfering information increases. This reveals a "working memory" capacity bottleneck that is independent of context length and cannot be effectively mitigated by prompt engineering.
tags:
  - "ICML 2025"
  - "Information Retrieval & RAG"
  - "Proactive Interference"
  - "Working Memory"
  - "Information Retrieval"
  - "Context Length"
  - "LLM Evaluation"
date: 2026-05-08
content_hash: dae5b0ad3bae75c4
---

# Unable to Forget: Proactive Interference Reveals Working Memory Limits in LLMs Beyond Context Length

**Conference**: ICML 2025  
**arXiv**: [2506.08184](https://arxiv.org/abs/2506.08184)  
**Code**: [Yes](https://github.com/zhuangziGiantfish/Unable-to-Forget)  
**Area**: Robotics  
**Keywords**: Proactive Interference, Working Memory, Information Retrieval, Context Length, LLM Evaluation

## TL;DR

Drawing on the Proactive Interference (PI) paradigm from cognitive science, this study finds that the information retrieval accuracy of LLMs decreases log-linearly to zero as the amount of interfering information increases. This reveals a "working memory" capacity bottleneck that is independent of context length and cannot be effectively mitigated by prompt engineering.

## Background & Motivation

In the evaluation of LLM information retrieval, input length is typically treated as the primary metric of task difficulty. Existing long-context benchmarks (such as Needle-in-a-Haystack, MRCR, etc.) mainly increase difficulty by growing the prompt length. However, these studies conflate **search difficulty** (locating the target within a massive context) with **interference** (correctly identifying the target among similar but incorrect entries).

**Key Challenge**: Current research implicitly attributes the "difficulty of distinguishing similar information" to input length, neglecting the impact of interference as an independent factor. In reality, semantically similar interfering information is extremely common in many data processing tasks—such as tracking the latest reading in continuous blood pressure records.

**Cognitive Science Inspiration**: Proactive Interference (PI) is a classic cognitive psychology paradigm where previously learned information hinders the recall of new information. In humans, PI resistance is negatively correlated with working memory capacity. Humans exhibit a "plateau effect"—beyond a certain threshold, additional interference no longer produces a significant impact, which is attributed to the human ability to actively "unbind" outdated associations.

**Our Approach**: Adapt the PI paradigm to LLM testing, minimizing search difficulty by fixing the retrieval target position (always the latest update) to purely isolate and quantify the interference effect.

## Method

### Overall Architecture

Designed the **PI-LLM evaluation**, core of which is a synthetic key-value pair retrieval task:
- **Input**: A stream of key-value pair updates with a fixed set of keys, where each key undergoes multiple value updates (randomly interleaved).
- **Query**: Requires the model to return the latest value for each key (i.e., the most recently appearing value).
- **Control**: Search difficulty is naturally extremely low (the target is always the latest update), so errors are primarily attributed to interference.

### Key Designs

#### 1. Basal Interference Experiment (Experiment 1)

**Function**: Systematically increase the number of updates per key (3 to 400) while keeping 46 unique keys fixed, and measure retrieval accuracy.

**Key Findings**: The retrieval accuracy of all tested models decreases **log-linearly** with the number of updates:

$$\text{Accuracy} \propto -\alpha \cdot \log(\text{update count}) + \beta$$

Model size affects the slope of decay—large models (>150B) decay slower, while small models rapidly drop to near-zero.

**Three-Stage Error Analysis**:
- **Low Interference**: Errors are concentrated around the position of the correct value (local confusion).
- **Medium Interference**: Errors scatter to earlier update values, with a small amount of hallucination.
- **High Interference**: Large amounts of unencountered values are returned (hallucinations), accompanied by a strong "primacy effect" biasing towards the first few updates.

#### 2. Interference is Independent of Input Length (Experiment 2)

**Function**: Design two sub-experiments to decouple interference and input length:
- **Exp A**: Fix the number of updates per key (125 or 350) and vary the number of updated keys (1 to 46).
- **Exp B**: Fix the total input length (both update count and key count are fixed) and only vary the number of keys to track.

**Key Findings**: Both experiments show nearly identical log-linear decay curves—even when the **input length in Exp B remains completely unchanged**, accuracy still decays log-linearly as the number of tracked keys increases. This directly proves that interference is a limiting factor independent of input length.

#### 3. Unified Interference Capacity Bottleneck (Experiment 3)

**Function**: Fix the previous three interference sources and vary the length of each value (by concatenating multiple words).

**Key Findings**: Accuracy similarly decreases log-linearly, exhibiting the steepest slope—as value length increases from 1 to 10 words, the accuracy of all models drops below 40%. This indicates that **all forms of interference share a unified capacity limit**, analogous to the unified resource of human working memory.

#### 4. Interference Endurance Score (IES)

Introduce **Interference Endurance Score** (IES) to quantify the model's resistance to interference:

$$\text{IES} = \text{AUC}(\text{accuracy vs. log(update count)})$$

Regression analysis indicates that **model size is a significant predictor of IES** ($p = 0.005$), whereas **context window length has no significant effect** ($p = 0.886$). MoE architectures perform worse than dense models with equivalent total parameters (because active parameters are far fewer than the nominal total parameters).

### Loss & Training

This work is an evaluation and analysis study, involving no training. Experiments cover a wide range of models from 0.6B to 637B parameters, including mainstream models such as GPT, Claude, Gemini, Grok, DeepSeek, and Qwen.

## Key Experimental Results

### Main Results (Effectiveness of Interference Mitigation Strategies)

| Strategy | Effect | Description |
|---|---|---|
| Per-key forget | Ineffective / Negative effect | Errors instead cluster around the position of the forget instruction |
| Forward focus | Marginal improvement | < 10 percentage points improvement |
| Relevance meta-prompt | Ineffective | The model correctly identifies the location of the answer but still retrieves incorrectly |
| Soft session reset | Ineffective | Natural language reset signals fail to alter retrieval behavior |
| Mock QA reset (hack) | **Effective** | Simulates conversation turn boundaries, significantly improving accuracy |

### Ablation Study (Model Size vs. Context Length)

| Factor | Impact on IES | $p$-value |
|---|---|---|
| Model size category | Significant positive correlation | 0.005 |
| Context window length | No significant impact | 0.886 |
| Size vs. IES (among 128k-131k models) | Spearman $\rho^2 = 0.673$ | 0.0016 |

### Key Findings

1. **Universal Log-Linear Decay**: All tested models (from 0.6B to 637B) consistently exhibit a log-linear decrease in accuracy caused by interference, including state-of-the-art models such as GPT-4.1, Claude, and Gemini 2.5.
2. **Interference Independent of Input Length**: Under fixed input length conditions, increasing the number of tracked keys still leads to the same decay pattern.
3. **Unified Capacity Limit**: The three orthogonal dimensions—update frequency, key count, and value length—all yield the same log-linear decay, pointing to a unified anti-interference resource.
4. **CoT Does Not Improve Retrieval**: Reasoning models (such as DeepSeek-R1) do not outperform, and sometimes even perform worse than, base models on interference retrieval tasks—"knowing where the answer is" does not equate to "being able to retrieve it correctly."
5. **Mock QA Reset is the Only Effective Strategy**: "Tricking" the model into discarding prior information by simulating dialogue boundaries, though this is a hack rather than a systemic solution.
6. **Sequential vs. Random Updates**: Sequential update patterns lead to a step-like collapse (suddenly dropping to zero at model-specific thresholds), whereas randomly interleaved updates result in a gradual log-linear decay.

## Highlights & Insights

- Extremely elegant experimental design—perfectly isolating the interference effect by fixing the target retrieval position, demonstrating superb control of variables.
- Interdisciplinary fusion of cognitive science and AI evaluation: The PI paradigm, backed by decades of research in cognitive science, yields profound insights when ported to LLM evaluation.
- Reveals a counter-intuitive conclusion: The "working memory" of LLMs is not equivalent to the context window length; the true bottleneck lies in the ability to resist interference.
- "Knowing but failing to execute" dissociation (top-down vs. bottom-up): The model can analyze and identify the correct strategy but fails to execute it during retrieval, which offers key insights into the limitations of CoT/reasoning models.
- The failure mode of "Per-key forget" is highly revealing—the forget instruction instead acts as a new interference "anchor," implying fundamental limitations of the attention mechanism.

## Limitations & Future Work

- There exists a gap between synthetic key-value pair tasks and real-world NLP tasks; conclusions need to be validated in more natural scenarios.
- The IES metric is based on a single experimental setup (46 keys); IES rankings might change under different setups.
- Mechanics-level causes of the interference effect (such as behavior patterns of attention heads) were not analyzed in depth.
- The success of Mock QA Reset hints at potential architectural improvement directions (such as explicit gating/forgetting mechanisms), but this was not explored in this work.
- Only text-based models were tested; whether multimodal models suffer from similar interference bottlenecks remains unknown.
- The mitigation effects of external memory-augmentation schemes, such as RAG, on the interference problem were not considered.

## Related Work & Insights

- Complementary to "Lost in the Middle": While that work focuses on positional effects, this study focuses on interference effects—the two are orthogonal factors affecting retrieval.
- Practical implications for long-context models: Merely expanding the context window cannot resolve the interference problem; new architectural design directions are required.
- The analogy with human working memory is inspiring yet warrants caution—Transformer's self-attention differs fundamentally from the neural mechanisms of human working memory.
- Provides a new dimension for LLM evaluation: Beyond context length, reasoning capabilities, and knowledge capacity, resistance to interference should be established as an independent evaluation axis.
- Implications for RAG system design: Similar but obsolete information in retrieval results may severely interfere with LLM processing.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Porting the PI paradigm from cognitive science to LLM evaluation is highly original, revealing a fundamental limitation previously overlooked.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covering 30+ models (0.6B-637B) across multiple orthogonal experimental dimensions with rigorous statistical analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Step-by-step progression of experimental designs and findings, highly intuitive tables and figures, and a clear introduction to the cognitive science background.
- Value: ⭐⭐⭐⭐⭐ Uncovers the fundamental limitation of the Transformer architecture, offering vital guidance for model evaluation and architectural design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] HGMem: Hypergraph-based Working Memory to Improve Multi-step RAG for Long-Context Complex Relational Modeling](../../ICML2026/information_retrieval/hgmem_hypergraph-based_working_memory_to_improve_multi-step_rag_for_long-context.md)
- [\[ICLR 2026\] Bayesian Attention Mechanism: A Probabilistic Framework for Positional Encoding and Context Length Extrapolation](../../ICLR2026/information_retrieval/bayesian_attention_mechanism_a_probabilistic_framework_for_positional_encoding_a.md)
- [\[ICML 2025\] Understanding Synthetic Context Extension via Retrieval Heads](understanding_synthetic_context_extension_via_retrieval_heads.md)
- [\[ACL 2025\] Length-Induced Embedding Collapse in PLM-based Models](../../ACL2025/information_retrieval/length-induced_embedding_collapse_in_plm-based_models.md)
- [\[ACL 2025\] RAEmoLLM: Retrieval Augmented LLMs for Cross-Domain Misinformation Detection Using In-Context Learning Based on Emotional Information](../../ACL2025/information_retrieval/raemollm_retrieval_augmented_llms_for_cross-domain_misinformation_detection_usin.md)

</div>

<!-- RELATED:END -->
