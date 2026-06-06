---
title: >-
  [Paper Note] CarO: Chain-of-Analogy Reasoning Optimization for Robust Content Moderation
description: >-
  [ACL 2026][LLM Safety][Content Moderation] This paper proposes CarO (Chain-of-Analogy Reasoning Optimization), a two-stage training framework. It uses RAG-guided generation of analogical reasoning chains combined with SF…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Content Moderation"
  - "Analogical Reasoning"
  - "Direct Preference Optimization"
  - "LLM Reasoning"
  - "Decision Shortcuts"
date: 2026-05-08
content_hash: 6f58a8194ed65b1b
---

# CarO: Chain-of-Analogy Reasoning Optimization for Robust Content Moderation

**Conference**: ACL 2026  
**arXiv**: [2604.10504](https://arxiv.org/abs/2604.10504)  
**Code**: None  
**Area**: Information Retrieval  
**Keywords**: Content Moderation, Analogical Reasoning, Direct Preference Optimization, LLM Reasoning, Decision Shortcuts

## TL;DR
This paper proposes CarO (Chain-of-Analogy Reasoning Optimization), a two-stage training framework. It uses RAG-guided generation of analogical reasoning chains combined with SFT and a tailored DPO optimization, enabling LLMs to autonomously generate analogical reference cases for content moderation during inference. It achieves an average F1 improvement of 24.9% on ambiguous moderation benchmarks, significantly outperforming reasoning models (DeepSeek R1) and specialized moderation models (LLaMA Guard).

## Background & Motivation

**Background**: Content moderation is a core task for maintaining the safety of digital ecosystems. Traditional discriminative models (such as BERT) suffer from poor OOD generalization and a lack of interpretability. Recently, LLMs have demonstrated the ability to generate reasoning chains via prompting, ICL, and post-training, providing interpretable moderation decisions.

**Limitations of Prior Work**: Even SOTA reasoning models (e.g., DeepSeek R1) frequently fail when handling ambiguous moderation cases. Analysis reveals that these errors stem from "decision shortcuts" embedded in the context—surface cues that mislead the reasoning process. For example, "Every Indian person I know dances upon hearing music" is a benevolent description, yet DeepSeek R1 incorrectly labels it as discriminatory simply because it detects the mention of a specific group.

**Key Challenge**: LLMs are easily misled by surface semantic cues in boundary cases. They lack the analogical reasoning capability of human moderators, who typically recall similar precedents before synthesizing those precedents with guidelines to make a judgment.

**Goal**: To enable LLMs to learn analogical reasoning, allowing them to autonomously generate relevant analogical cases during inference and make more robust moderation decisions based on these analogies.

**Key Insight**: Taking inspiration from the moderation workflow of human experts in cognitive psychology, experts handle ambiguous cases by first recalling similar precedents (analogical retrieval) and then synthesizing insights from those precedents and moderation guidelines to make decisions (analogical reasoning).

**Core Idea**: A two-stage training process to internalize analogical reasoning: Stage 1 uses RAG+SFT to guide the generation of analogical reasoning chains. Stage 2 employs a tailored DPO to reinforce analogical reasoning (using preference pairs of reasoning with analogy vs. reasoning without analogy).

## Method

### Overall Architecture
Stage 1: Retrieve semantically similar cases for each training sample $\rightarrow$ Use DeepSeek R1 to generate reasoning chains containing analogical references $\rightarrow$ Perform reflective correction on erroneous reasoning $\rightarrow$ Conduct SFT training. Stage 2: Use reasoning chains with RAG input as positive samples and reasoning chains without RAG input as negative samples $\rightarrow$ Conduct DPO training to reinforce analogical reasoning. During inference, the model autonomously generates analogical cases without requiring external retrieval.

### Key Designs

1.  **Guided Chain-of-Analogy Generation (COAT)**:
    - **Function**: Generates high-quality reasoning chains containing analogical references for each training sample.
    - **Mechanism**: For each training sample $\mathbf{x}_i$, the top-k similar cases (with labels) are retrieved via semantic similarity. These retrieval results are injected into the prompt, requiring DeepSeek R1 to explicitly cite these analogical cases within its reasoning chain. After generation, the reasoning conclusion is checked against the label; any inconsistency triggers a reflection-correction step.
    - **Design Motivation**: Directly prompting for reasoning chains lacks reference to specific precedents. RAG-guided reasoning chains embed analogical patterns into the training data, helping the SFT model internalize this reasoning mode.

2.  **Tailored DPO for Enhanced Analogical Reasoning**:
    - **Function**: Explicitly reinforces the preference for analogical reasoning over standard reasoning.
    - **Mechanism**: The positive sample $\mathbf{r}^+$ is the reasoning chain generated by the SFT model with RAG input (containing analogical references); the negative sample $\mathbf{r}^-$ is the reasoning chain generated by the same model based solely on the original input (no analogical references). A standard DPO loss is used for optimization to make the model prefer analogy-rich reasoning chains.
    - **Design Motivation**: After SFT, the model is capable of generating analogical reasoning but lacks consistency. The goal of DPO is not to improve F1 (which is already high after SFT) but to enhance the explicitness, consistency, and interpretability of analogical reasoning.

3.  **Autonomous Analogy at Inference (No External Retrieval)**:
    - **Function**: Eliminates the need to maintain a retrieval database during deployment.
    - **Mechanism**: After two-stage training, the model has internalized analogical reasoning patterns. During inference, it directly "imagines" analogical cases based on the input without performing actual retrieval.
    - **Design Motivation**: RAG methods are limited by static datasets; retrieved cases may not be the most suitable for the current scenario. Internalized analogical capability allows for the dynamic generation of reference cases tailored to the current input.

## Key Experimental Results

### Main Results (Chinese Moderation Dataset + English Benchmarks)

| Model | Politics | Pornography | Violence | Bias | Gambling | Harmless | Average F1 |
|-------|----------|-------------|----------|------|----------|----------|------------|
| Qwen2.5-7B-Instruct | 54.9 | 81.9 | 70.0 | 60.1 | 84.3 | 48.8 | 64.3 |
| DeepSeek R1 | - | - | - | - | - | - | ~70 |
| LLaMA Guard | - | - | - | - | - | - | ~65 |
| **Ours (CarO)** | **Best** | **Best** | **Best** | **Best** | **Best** | **Best** | **89.2** |

### Ablation Study

| Configuration | F1 | CoA Ratio (%) |
|---------------|-----|---------------|
| Baseline (No training) | 64.3 | 0.0 |
| + RAG-SFT | 85.5 (+21.2) | 89.5 |
| + Reflection Correction | 88.8 (+3.3) | 93.5 |
| + DPO | **89.2 (+0.4)** | **99.3** |

### Cross-benchmark Generalization (OOD Test)

| Dataset | Qwen2.5-7B → CarO |
|---------|-------------------|
| Aegis (ID) | 78.7 → **87.1** |
| OpenAI (OOD) | 70.8 → **74.2** |
| Toxic-Chat (OOD) | 93.3 → **95.0** |

### Key Findings
- **F1 increased by 24.9 percentage points (64.3→89.2)**, primarily contributed by the RAG-SFT stage (+21.2).
- **DPO provided limited F1 gain (+0.4) but pushed the analogical reasoning ratio from 93.5% to 99.3%**, indicating its primary role is enhancing reasoning consistency rather than accuracy.
- **Reflection correction resulted in a 3.3pp gain**, showing that automatically generated reasoning chains contain errors that require correction.
- **Improvements were also observed on OOD benchmarks** (Aegis +8.4, OpenAI +3.4), suggesting that the analogical reasoning capability has cross-domain transferability.
- **No performance degradation occurred without RAG during inference**; in fact, performance improved, proving the model successfully internalized the analogical reasoning mode.

## Highlights & Insights
- **Accurate diagnosis of "decision shortcuts"**: The model is not necessarily incapable but rather misled, explaining why powerful reasoning models fail on moderation tasks.
- **Transferred design logic for two-stage training**: Use SFT to guide the emergence of capability, then use DPO to reinforce consistency. This "guidance $\rightarrow$ reinforcement" paradigm is applicable to any scenario requiring specific reasoning patterns.
- **Autonomous analogy at inference** is more flexible than RAG, as the model "imagines" the most suitable references for the current case without database constraints.

## Limitations & Future Work
- The analogical reasoning chains in the training data were generated by DeepSeek R1, meaning quality is bounded by that model's capability.
- Retrieving k=32 reference cases imposes high memory and inference cost requirements.
- The study primarily focused on Chinese datasets, with fewer validations on English benchmarks.
- DPO provided a very small F1 improvement (+0.4); are there more efficient reasoning reinforcement methods?
- Autonomously generated analogical cases might be inaccurate and lack a factual verification mechanism.

## Related Work & Insights
- **vs. DeepSeek R1**: R1 has strong reasoning capabilities but lacks analogical references, causing it to be misled by surface cues in ambiguous cases.
- **vs. LLaMA Guard**: These are specialized moderation models that lack an interpretable reasoning process.
- **vs. RAG Methods**: Static retrieval cannot adapt dynamically. CarO completely eliminates dependency on retrieval by internalizing analogical capability through training.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of analogical reasoning and DPO for moderation is new, though individual components are established.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes multi-benchmark, ablation, and OOD tests, though the main experiments emphasize Chinese data.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation; the connection to cognitive psychology is persuasive.
- Value: ⭐⭐⭐⭐ Directly applicable to the content moderation field; the analogical reasoning paradigm is transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] FlexGuard: Continuous Risk Scoring for Strictness-Adaptive LLM Content Moderation](flexguard_continuous_risk_scoring_for_strictness-adaptive_llm_content_moderation.md)
- [\[ACL 2026\] Making MLLMs Blind: Adversarial Smuggling Attacks in MLLM Content Moderation](making_mllms_blind_adversarial_smuggling_attacks_in_mllm_content_moderation.md)
- [\[ICLR 2026\] ExpGuard: LLM Content Moderation in Specialized Domains](../../ICLR2026/llm_safety/expguard_llm_content_moderation_in_specialized_domains.md)
- [\[ACL 2026\] Beyond End-to-End: Dynamic Chain Optimization for Private LLM Adaptation on the Edge](beyond_end-to-end_dynamic_chain_optimization_for_private_llm_adaptation_on_the_e.md)
- [\[ACL 2026\] CiPO: Counterfactual Unlearning for Large Reasoning Models through Iterative Preference Optimization](cipo_counterfactual_unlearning_for_large_reasoning_models_through_iterative_pref.md)

</div>

<!-- RELATED:END -->
