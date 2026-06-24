---
title: >-
  [Paper Note] Towards Safety Reasoning in LLMs: AI-agentic Deliberation for Policy-embedded CoT Data Creation
description: >-
  [ACL2025][Reasoning][Safety Reasoning] This paper proposes AIDsafe, a multi-agent iterative deliberation framework that automatically generates high-quality Safety Reasoning CoT data embedded with safety policies. The fine-tuned models significantly outperform traditional safety training in safety generalization and jailbreak robustness. Additionally, an "ear-whisperer" agent is introduced to resolve the difficulty of distinguishing between selected and rejected responses in…
tags:
  - "ACL2025"
  - "Reasoning"
  - "Safety Reasoning"
  - "Chain-of-Thought"
  - "Multi-Agent Deliberation"
  - "Safety Policy Embedding"
  - "DPO"
  - "Jailbreak Defense"
  - "Preference Data"
date: 2026-05-08
content_hash: a953db9b8864d537
---

# Towards Safety Reasoning in LLMs: AI-agentic Deliberation for Policy-embedded CoT Data Creation

**Conference**: ACL2025  
**arXiv**: [2505.21784](https://arxiv.org/abs/2505.21784)  
**Code**: None (dataset has been open-sourced)  
**Area**: LLM Reasoning  
**Keywords**: Safety Reasoning, Chain-of-Thought, Multi-Agent Deliberation, Safety Policy Embedding, DPO, Jailbreak Defense, Preference Data

## TL;DR
This paper proposes AIDsafe, a multi-agent iterative deliberation framework that automatically generates high-quality Safety Reasoning CoT data embedded with safety policies. The fine-tuned models significantly outperform traditional safety training in safety generalization and jailbreak robustness. Additionally, an "ear-whisperer" agent is introduced to resolve the difficulty of distinguishing between selected and rejected responses in DPO preference data.

## Background & Motivation

**Background**: LLM safety training is shifting from traditional SFT/RLHF to a "safety reasoning" paradigm—where models explicitly reason about safety policies before generating responses. Leading works in this direction include OpenAI o1 and Deliberative Alignment.

**Limitations of Prior Work**: (a) High-quality safety CoT data is extremely scarce, and human annotation is highly subjective and expensive; (b) Direct generation of safety reasoning chains by LLMs often suffers from hallucinations, deceptive reasoning, and policy conflicts; (c) Safety policies themselves are ambiguous and mutually contradictory, making them difficult for a single model to fully cover.

**Key Challenge**: Safety reasoning requires high-quality CoT training data $\rightarrow$ generating high-quality CoTs requires models with strong reasoning capabilities $\rightarrow$ training such models requires data—forming a **chicken-and-egg problem**.

**Goal**: How to generate high-quality, policy-aligned safety reasoning CoT data in bulk without relying on expensive reasoning models?

**Key Insight**: Drawing inspiration from existing work where multi-agent debate reduces hallucinations and enhances reasoning reliability, this work replaces a single strong model with multi-agent collaboration, iterative deliberation, and post-processing refinement.

**Core Idea**: Multi-agent iterative deliberation + refiner filtering = generating high-quality safety reasoning chain data even with ordinary models.

## Method

### Overall Architecture (AIDsafe)
Four stages: Initialization $\rightarrow$ Deliberation $\rightarrow$ Refinement $\rightarrow$ (Optional) Preference Data Generation.

### Key Designs 1: Initialization Phase

**Function**: Deconstruct user intent + generate seed CoT.  
**Design Motivation**: User queries might contain both benign and malicious intents, and direct generation easily leads to over-refusal; it is necessary to separate intents before rendering targeted reasoning.  
**Mechanism**:  
- **Intent Decomposition**: The LLM agent identifies explicit and implicit intents in the query, distinguishing benign vs. potentially malicious intents.  
- **Initial CoT Generation**: A single agent generates a baseline reasoning chain and response, serving as the starting point for subsequent deliberation.

### Key Designs 2: Deliberation Phase (Deliberation)

**Function**: Iteratively expand safety reasoning chains using multiple agents.  
**Design Motivation**: Single-pass generation is prone to missing policy coverage, and reasoning can be incomplete or biased. Multi-agent cross-examination improves coverage and reliability.  
**Mechanism**:  
- In each round of deliberation, an agent evaluates the existing reasoning chain to determine whether additional safety policy analysis is needed.  
- If needed, the agent proposes new thoughts and updates the response.  
- Iterations continue until agents reach a consensus (e.g., "I agree with the previous agent") or the budget is exhausted.  
- Five categories of safety policies are utilized: hate speech/harassment/violence, fraud/deception, physical harm, illegal activities, and respect/helpfulness.

### Key Designs 3: Refinement Phase (Refiner)

**Function**: Clean up deliberation outputs acting as an independent third-party evaluator.  
**Design Motivation**: Deliberation can produce three types of noise: (a) deceptive thoughts (seemingly plausible but leading to incorrect conclusions), (b) redundant/repetitive thoughts (leading to overthinking and over-refusal), and (c) policy inconsistency.  
**Mechanism**: The Refiner agent aggregates thoughts from all rounds $\rightarrow$ evaluates them item-by-item $\rightarrow$ removes noise $\rightarrow$ outputs a concise and consistent final CoT + response. This is inspired by "AI safety via debate" by Irving et al. (2018).

### Key Designs 4: Ear-whisperer Preference Data Generation

**Function**: Generate high-quality selected/rejected CoT pairs for DPO training.  
**Design Motivation**: In standard sampling methods, the quality difference between selected and rejected CoTs generated by SFT models is extremely minimal (with similar policy faithfulness), failing to provide effective gradient signals for DPO.  
**Mechanism**:  
- Introduce an "ear-whisperer" agent to generate adversarial "bad beliefs" prefixes.  
- When generating the rejected CoT, the bad belief prefix is prepended to the input, guiding the model to generate reasoning that violates the safety policy.  
- An iterative In-Context Learning (ICL) strategy is adopted: repeatedly refining the bad belief until it effectively induces policy violations.  
- ShieldGemma is used as a scoring function to evaluate the quality of the bad beliefs.

### Loss & Training
- Agent Model: Mixtral 8x22B (uniformly used for all agents)  
- Data Source: 5,000 safety prompts from BeaverTails + 5,000 general prompts from Alpagasus  
- SFT: QLoRA 4-bit quantization, fine-tuned on Mixtral-7B and Qwen2.5-7B  
- Efficiency: Asynchronous LLM querying (AsyncInferenceClient), 4×A100, ~35 seconds/prompt

## Key Experimental Results

### Main Results: Safety Evaluation of Fine-Tuned Models (Table 2)

| Evaluation Dimension | Dataset | Mixtral Base | SFTOG | **SFTDB (AIDsafe)** |
|---------|--------|-------------|-------|---------------------|
| **Safety (In-domain)** | BeaverTails | 76.00% | 79.57% | **96.00%** |
| **Safety (Out-of-domain)** | WildChat | 31.00% | 33.50% | **85.95%** |
| **Jailbreak Robustness** | StrongREJECT | 51.09% | 67.01% | **94.04%** |
| **Over-refusal Accuracy** | XSTest | 98.80% | 87.60% | **91.84%** |
| **General Capability** | MMLU | 35.42% | 31.38% | **34.51%** |

| Evaluation Dimension | Dataset | Qwen Base | SFTOG | **SFTDB (AIDsafe)** |
|---------|--------|----------|-------|---------------------|
| **Safety (In-domain)** | BeaverTails | 94.14% | 87.95% | **97.00%** |
| **Safety (Out-of-domain)** | WildChat | 95.50% | 59.42% | **96.50%** |
| **Jailbreak Robustness** | StrongREJECT | 72.84% | 59.48% | **95.39%** |
| **Over-refusal Accuracy** | XSTest | 99.20% | 98.00% | 93.60% |
| **General Capability** | MMLU | 75.78% | 55.73% | 60.52% |

### Ablation Study: CoT Data Quality Evaluation (Table 1)

| Metric | LLMZS (Single Model) | **AIDsafe** | Gain |
|------|----------------|------------|------|
| Relevance | 4.66 | **4.68** | +0.43% |
| Coherence | 4.93 | **4.96** | +0.61% |
| Completeness | 4.86 | **4.92** | +1.23% |
| **CoT Policy Faithfulness** | 3.85 | **4.27** | **+10.91%** |
| Response Policy Faithfulness | 4.85 | **4.91** | +1.24% |
| Response-CoT Consistency | 4.99 | **5.00** | +0.20% |

### Key Findings

1. **Safety Reasoning >> Traditional Safety Training**: The out-of-domain safety of Mixtral SFTDB improves by 54.95 percentage points (pp) compared to the Base model (31% $\rightarrow$ 85.95%), whereas traditional SFTOG only improves by 2.5pp.
2. **Traditional Safety Training May "Overwrite" Original Safety**: Qwen SFTOG plummeted from 95.5% to 59.42% on WildChat, whereas SFTDB maintained 96.5%. This indicates that safety reasoning helps the model "understand policies" rather than "memorize patterns".
3. **Leap in Jailbreak Robustness**: Trained on only 5,000 safety samples without exposure to jailbreak data, the jailbreak safety rate of Mixtral increased from 51% $\rightarrow$ 94%.
4. **CoT Policy Faithfulness Is the Most Discerning Metric**: AIDsafe improves policy faithfulness by 10.91% compared to single-model generation. In Pairwise evaluation (via two independent auto-graders), AIDsafe significantly outperforms LLMZS in win-rate.
5. **Ear-whisperer Effectively Widens the Preference Gap**: Selected/rejected responses from standard sampling show almost no difference in policy faithfulness, whereas the ear-whisperer method successfully creates a significant distribution shift.

## Highlights & Insights

- **"Safety Reasoning" vs. "Safety Classification"**: The core insight is that helping LLMs understand "why it is unsafe" generalizes far better than forcing them to memorize "what is unsafe." Utilizing only 5,000 samples allows generalization to unseen attack types, which is impossible with traditional safety training.
- **Multi-Agent Deliberation Compensates for Single-Model Reasoning Limitations**: Relying on multi-agent collaboration with the standard Mixtral 8x22B achieves CoT quality close to that of strong reasoning models, representing a successful application of "collective intelligence" in safety reasoning.
- **Systematic Resolution of the Overthinking Problem**: The Refinement phase specifically handles repetitive/redundant thoughts, directly mitigating the over-refusal issue inherent in safety training.
- **Ingenious Design of the Ear-whisperer**: Rather than simply generating harmful responses, it leverages "belief injection" to produce seemingly plausible yet reasoning-flawed CoTs. This style of "contrastive learning" is highly effective for modeling "good reasoning vs. bad reasoning."

## Limitations & Future Work

1. **Limited Safety Policy Coverage**: Only 5 categories of policies were utilized; real-world safety risks extend far beyond these (e.g., privacy, bias, and copyright are not covered).
2. **Single Agent Model**: All agents relied on Mixtral 8x22B, without exploring the effects of heterogeneous agents where different models might leverage unique strengths.
3. **Deliberation Limited to 2 Agents**: More complex multi-agent roundtable setups were not investigated.
4. **Suboptimal SFT Starting Point**: Safety SFT was conducted directly on instruction-tuned models, whereas ideally, one should start with CoT warm-up on base models.
5. **Deliberation Difficulty with Highly Guardrailed Models**: If the agent model itself features strict safety guardrails, it might refuse to cooperate when deliberating on harmful queries, resulting in process interruption.
6. **General Capability Degradation (Qwen)**: MMLU scores dropped from 75.78% to 60.52%, indicating that the safety-utility trade-off is still not fully resolved.

## Related Work & Insights

### vs. Deliberative Alignment (Guan et al., 2024)
Deliberative Alignment allows models to refer to safety policy documents during inference, but relies heavily on the model's inherent reasoning abilities. The advantage of AIDsafe lies in **outsourcing the reasoning capability to a multi-agent system**, reducing dependency on a single strong model. However, while DA is an inference-time solution, AIDsafe is a data generation solution; thus, the two are complementary.

### vs. Constitutional AI (Bai et al., 2022b)
CAI utilizes AI feedback to replace human feedback for safety alignment but lacks explicit policy reasoning chains—the model learns "what is safe" but may not understand "why". AIDsafe embeds reasoning chains to **help the model learn the reasoning process behind safety decisions**, leading to stronger generalization capabilities (a 54.95pp improvement in out-of-domain safety compared to the limited generalization of CAI).

### vs. DeepSeek-R1 / OpenAI o1
The safety reasoning capabilities of these strong reasoning models stem from large-scale RL training, which is extremely costly. AIDsafe demonstrates that **using multi-agent collaboration + data engineering can achieve comparable safety reasoning effectiveness even on a 7B model**, providing a viable pathway for the open-source community.

## Rating
- Novelty: ⭐⭐⭐⭐ Utilizing multi-agent deliberation for safety CoT data generation is a novel and natural idea, and the ear-whisperer preference data scheme is particularly ingenious.
- Experimental Thoroughness: ⭐⭐⭐⭐ The entire pipeline is thoroughly validated through data quality evaluation (3 dimensions + faithfulness + pairwise) + downstream training (2 models × 4 benchmarks) + preference data analysis.
- Writing Quality: ⭐⭐⭐⭐ The framework design is clear, the motivations and design choices for each stage are well-explained, and the ethical considerations section candidly discusses the potential abuse risk of the ear-whisperer.
- Value: ⭐⭐⭐⭐⭐ This work provides a complete data generation pipeline for safety reasoning training in open-source LLMs. Achieving a 54.95pp improvement in out-of-domain safety generalization with only 5,000 samples represents extremely high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Note 1: Is CoT a Hallucination? A Data Distribution Perspective](../../NeurIPS2025/llm_reasoning/is_chain-of-thought_reasoning_of_llms_a_mirage_a_data_distribution_lens.md)
- [\[ACL 2025\] Fine-Tuning on Diverse Reasoning Chains Drives Within-Inference CoT Refinement in LLMs](dcot_diverse_cot_refinement.md)
- [\[ACL 2025\] CoT-UQ: Improving Response-wise Uncertainty Quantification in LLMs with Chain-of-Thought](cot-uq_improving_response-wise_uncertainty_quantification_in_llms_with_chain-of-.md)
- [\[CVPR 2026\] Agile Deliberation: Concept Deliberation for Subjective Visual Classification](../../CVPR2026/llm_reasoning/agile_deliberation_concept_deliberation_for_subjective_visual_classification.md)
- [\[ICML 2026\] Internalizing Safety Understanding in Large Reasoning Models via Verification](../../ICML2026/llm_reasoning/internalizing_safety_understanding_in_large_reasoning_models_via_verification.md)

</div>

<!-- RELATED:END -->
