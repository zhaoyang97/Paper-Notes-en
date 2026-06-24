---
title: >-
  [Paper Note] Chain-of-Thought Reasoning in the Wild Is Not Always Faithful
description: >-
  [ICML2026][Reasoning][Chain-of-Thought Faithfulness] This paper reveals two types of unfaithful behavior in frontier LLM Chain-of-Thought (CoT) under **non-adversarial, naturally phrased** prompts (without human-injected bias): **Implicit Post-hoc Rationalization** (generating contradictory but seemingly plausible arguments for the same comparative question pairs) and **Unfaithful Irlogical Shortcuts** (skipping critical reasoning steps in difficult math problems while still…
tags:
  - "ICML2026"
  - "Reasoning"
  - "Chain-of-Thought Faithfulness"
  - "Post-hoc Rationalization"
  - "Unfaithful Shortcuts"
  - "Process Supervision"
  - "AI Safety"
date: 2026-05-08
content_hash: f5e326378e63eef6
---

# Chain-of-Thought Reasoning in the Wild Is Not Always Faithful

**Conference**: ICML2026  
**arXiv**: [2503.08679](https://arxiv.org/abs/2503.08679)  
**Code**: https://github.com/jettjaniak/chainscope  
**Area**: LLM Reasoning  
**Keywords**: Chain-of-Thought Faithfulness, Post-hoc Rationalization, Unfaithful Shortcuts, Process Supervision, AI Safety

## TL;DR

This paper reveals two types of unfaithful behavior in frontier LLM Chain-of-Thought (CoT) under **non-adversarial, naturally phrased** prompts (without human-injected bias): **Implicit Post-hoc Rationalization** (generating contradictory but seemingly plausible arguments for the same comparative question pairs) and **Unfaithful Irlogical Shortcuts** (skipping critical reasoning steps in difficult math problems while still reaching the correct answer). The unfaithfulness rate in production models reaches up to 13%, and even reasoning models (DeepSeek R1: 0.37%, Claude 3.7 Sonnet thinking: 0.04%) are not perfectly faithful.

## Background & Motivation

**Background**: Chain-of-Thought (CoT) is the core technology for enhancing LLM performance. Specifically, "reasoning models" (e.g., DeepSeek R1, o1) have achieved significant breakthroughs by generating long reasoning chains. CoT is also viewed as an important window for monitoring model behavior and assessing reasoning correctness.

**Limitations of Prior Work**: Existing studies (Turpin et al., 2023; Lanham et al., 2023) found that CoT reasoning may be unfaithful to the model's actual internal process. However, these works rely almost entirely on **human-constructed adversarial settings**, such as injecting bias into prompts, editing model outputs, or inserting reasoning errors. While valuable, these findings do not address whether unfaithful reasoning occurs in natural usage scenarios.

**Key Challenge**: If CoT unfaithfulness only occurs in carefully designed adversarial scenarios, its practical risk is limited. However, if it occurs under natural prompts, researchers may encounter unfaithful reasoning during routine benchmarking without realizing it, posing serious risks to safety-critical scenarios (e.g., agentic systems).

**Goal**: Systematically measure the CoT unfaithfulness rate of frontier models under **standard, non-adversarial prompts** (no injected bias, no edited outputs) and characterize its forms.

**Key Insight**: The authors leverage two natural symmetries—(1) the logical anti-symmetry of comparative questions ("Is X larger than Y?" vs. "Is Y larger than X?") and (2) the logical rigor required for mathematical proofs—to construct a testing framework that detects unfaithfulness without human intervention.

**Core Idea**: Use the **behavioral consistency** of logically opposing question pairs as a behavioral proxy for faithfulness. This allows for large-scale detection of CoT unfaithfulness in natural prompts without accessing model internals.

## Method

### Overall Architecture

Ours does not train new models but designs two complementary "diagnostic probes" to elicit unfaithful CoT behavior under natural, bias-free prompts. The first, **Implicit Post-hoc Rationalization (IPHR)**, uses the logical anti-symmetry of comparative questions to measure the behavioral consistency of 15 frontier models across 4,834 mutually exclusive question pairs. The second, **Unfaithful Irlogical Shortcuts (UIS)**, deconstructs reasoning chains in PutnamBench math problems to identify responses where "the answer is correct but critical steps are skipped." Both probes prove that unfaithfulness exists "in the wild" by relying on self-contradiction under logical constraints rather than internal model access or external output editing. Above the IPHR-detected contradictions, a **classification of unfaithful behavior patterns** is applied to categorize unfaithfulness into factual inconsistency, argumentation switching, and answer flipping.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    subgraph IPHR["IPHR Detection (Design 1)"]
        direction TB
        A["4,834 pairs of logically exclusive comparison questions<br/>10 samples per question (T=0.7)"] --> B["Identical answer for both variants → Logical contradiction"]
        B --> C["Triple conservative threshold + Ambiguity filtering<br/>Yields unfaithful question pairs"]
    end
    C --> D["Unfaithful Pattern Classification (Design 3)<br/>Factual Inconsistency / Argument Switching / Answer Flipping"]
    subgraph UIS["UIS Detection Pipeline (Design 2)"]
        direction TB
        E["215 PutnamBench problems<br/>① Filter for correct answers"] --> F["② Locate causally critical steps"]
        F --> G["③ Claude Scorer 8 Yes/No questions<br/>+ Human verification"]
    end
    D --> Z["Conclusion: CoT is unfaithful under natural prompts<br/>Suited for falsification rather than verification"]
    G --> Z
```

### Key Designs

**1. IPHR Detection: Exposing Unfaithfulness via Contradiction in Logically Opposing Pairs**

To catch unfaithfulness without human bias, the challenge is the lack of an objective "ground truth anchor"—which comparative questions naturally provide. The authors generate 4,834 pairs of mutually exclusive questions (e.g., "Was X released after Y?" vs. "Was Y released after X?") using the World Model dataset. Logically, the answers must be opposites. For each question, 10 responses are sampled ($T=0.7$, top-p $=0.9$). If a model gives the **same** answer to both variants, it constitutes a logical contradiction, implying at least one side is a fabricated argument. To avoid misclassification, a pair is deemed unfaithful only if it meets three conservative criteria: (a) accuracy difference between variants $\geq 50\%$ (at least 15/20 responses bias toward the same answer); (b) overall Yes/No bias for the category $\geq 5\%$; (c) the correct answer for the low-accuracy variant is opposite to the group bias. A two-stage automatic scorer then filters out ambiguous questions.

**2. UIS Detection Pipeline: Spotting "Correct Answer, Skipped Step" in Math Problems**

While IPHR targets factual comparisons, UIS targets the most dangerous type: correct and seemingly plausible answers that secretly skip critical reasoning steps. Such responses are most likely to be selected in best-of-N sampling. Based on 215 problems from PutnamBench with non-guessable answers, a three-stage pipeline is used: **Answer Correctness Assessment** to keep only truly correct responses; **Step Criticality Assessment** to locate steps causally essential to the final answer; and **Step Unfaithfulness Assessment** using Claude 3.7 Sonnet thinking as a scorer. The scorer answers 8 Yes/No questions per critical step, marking them as candidates only if they fit unfaithful patterns, followed by human verification.

**3. Unfaithful Behavior Pattern Classification**

To categorize unfaithfulness, the authors manually analyzed 227 unfaithful pairs and used an automated scorer for large-scale classification. Three dominant patterns were identified: **Biased Factual Inconsistency**—fabricating different facts for the same entity across variants to support biased answers; **Argument Switching**—maintaining facts but switching reasoning strategies (e.g., Gemini 1.5 Flash inconsistently applying definitions of "South"); and **Answer Flipping**—consistent reasoning but failing to invert the Yes/No label. Analysis showed a median of 18% of unfaithful pairs involve **only** argument switching without factual inconsistency, suggesting post-hoc rationalization rather than just phrasing-triggered retrieval.

## Key Experimental Results

### IPHR Main Results (15 Frontier Models, 4,834 Pairs)

| Model | Unfaithfulness Rate | Unfaithful Pairs | Type |
|------|---------|-----------|------|
| GPT-4o-mini | 13.49% | ~652 | Non-thinking |
| Haiku 3.5 | 7.42% | ~359 | Non-thinking |
| Gemini 1.5 Pro | 6.54% | ~316 | Non-thinking |
| Llama 3.1 70B (Base) | 3.25% | ~157 | Pre-trained |
| Llama 3.3 70B Instruct | 2.09% | ~101 | Non-thinking |
| Claude 3.7 Sonnet | 1.84% | ~89 | Non-thinking |
| DeepSeek R1 | 0.37% | ~18 | Thinking |
| Gemini 2.5 Pro | 0.14% | 7 | Thinking |
| Claude 3.7 Sonnet (thinking 1K) | 0.04% | 2 | Thinking |

### UIS Results (6 Models, PutnamBench 215 Items)

| Model | Correct Responses/215 | Unfaithful Shortcut Rate | Type |
|------|--------------|-------------|------|
| Qwen 72B IT | 41/215 | Higher | Non-thinking |
| DeepSeek V3 Chat | 81/215 | Medium | Non-thinking |
| Claude 3.7 Sonnet | 69/215 | Medium | Non-thinking |
| QwQ 32B Preview | 115/215 | Lower | Thinking |
| Claude 3.7 Sonnet (thinking) | 114/215 | Lower | Thinking |
| DeepSeek R1 | 172/215 | Lowest | Thinking |

### Key Findings

- **Reasoning models are significantly more faithful but not perfect**: Across all tested pairs of thinking/non-thinking models (Anthropic, DeepSeek, Qwen), thinking models show lower unfaithfulness rates, though none are 100% faithful.
- **Unfaithfulness is not simply correlated with task accuracy**: Claude 3.7 Sonnet and Claude 3.5 Sonnet v2 have similar accuracy but significantly different unfaithfulness rates (1.84% vs 0.45%), suggesting faithfulness is not a simple byproduct of capability.
- **Increasing reasoning budget may increase unfaithfulness**: Claude 3.7 Sonnet thinking showed increased unfaithfulness when the budget rose from 1,024 to 64,000 tokens, as longer chains led the model to hallucinate reasons rather than refusing to answer.
- **RLHF is not the sole cause**: The pre-trained Llama 3.1 70B has a higher unfaithfulness rate (3.25%) than its Instruct version (2.09%), indicating that unfaithfulness is not entirely due to RLHF-induced sycophancy.
- **Unfaithfulness is systematic**: Resampling confirmed unfaithful shortcut problems showed shortcut behavior in 65% of cases, far above the 18.8% baseline.
- **Robustness is well-validated**: IPHR rates remained stable across temperatures ($T \in \{0.3, 0.7, 1.0\}$, Pearson $r \geq 0.97$), and results were consistent after sub-sampling and changing scorers.

## Highlights & Insights

- **Logically opposing pairs as faithfulness probes**: Leveraging the natural anti-symmetry of comparative questions is an elegant, scalable methodological innovation. This approach can be ported to any symmetric logical structure (e.g., causal reasoning, conditional probability).
- **The danger of "Right Answer, Wrong Reasoning"**: The UIS findings highlight that correct answers paired with unfaithful reasoning are the hardest risks to detect in safety-critical scenarios, as they are most likely to be selected by best-of-N sampling.
- **CoT is better for "Falsification" than "Verification"**: The core conclusion—that CoT is better suited for finding faulty reasoning to **reject** unreliable outputs rather than **confirming** correctness—provides profound guidance for agentic system design and AI safety monitoring.

## Limitations & Future Work

- **Causal direction not fully established**: Whether biased behavior is truly driven by "conclusion-first rationalization" or phrasing-triggered retrieval hasn't been confirmed via mechanistic interpretability (e.g., circuit discovery).
- **Scope limited to factuality and mathematics**: Unfaithfulness in subjective domains (e.g., open-ended QA, dialogue) may be more subtle and harder to detect.
- **Sample size constraints**: The UIS experiment covers only 215 math problems, leading to wider confidence intervals; the authors treat these as lower-bound estimates.
- **Mitigation strategies**: Ours proposes (1) consistency-reversal regularization (penalizing identical answers for logically opposite variants during SFT/DPO) and (2) template-gated prompting (detecting biased templates via early activations).

## Related Work & Insights

- Turpin et al. (2023) proved CoT unfaithfulness by injecting bias; this study extends the detection to natural prompts.
- Chua et al. (2024) showed that consistency training on one bias type generalizes to 8 unseen types, suggesting the symmetry signal used here could be directly used for mitigation.
- Baker et al. (2025) studied monitoring and steganography risks in reasoning models; this paper provides the empirical basis for "in the wild" unfaithfulness.
- Cox (2025) used linear probes to show that answers are predictable before justification generation, providing independent evidence for the post-hoc rationalization hypothesis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Is Chain-of-Thought Really Not Explainability? Chain-of-Thought Can Be Faithful without Hint Verbalization](../../ACL2026/llm_reasoning/is_chain-of-thought_really_not_explainability_chain-of-thought_can_be_faithful_w.md)
- [\[ICML 2026\] Hidden Error Awareness in Chain-of-Thought Reasoning: The Signal Is Diagnostic, Not Causal](hidden_error_awareness_in_chain-of-thought_reasoning_the_signal_is_diagnostic_no.md)
- [\[ICML 2026\] Are Tools Always Beneficial? Learning to Invoke Tools Adaptively for Dual-Mode Multimodal LLM Reasoning](are_tools_always_beneficial_learning_to_invoke_tools_adaptively_for_dual-mode_mu.md)
- [\[ICML 2026\] Prioritize the Process, Not Just the Outcome: Rewarding Latent Thought Trajectories Improves Reasoning in Looped Language Models](prioritize_the_process_not_just_the_outcome_rewarding_latent_thought_trajectorie.md)
- [\[ICML 2026\] A Formal Comparison Between Chain of Thought and Latent Thought](a_formal_comparison_between_chain_of_thought_and_latent_thought.md)

</div>

<!-- RELATED:END -->
