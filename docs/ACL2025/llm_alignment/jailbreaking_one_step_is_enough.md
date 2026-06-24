---
title: >-
  [Paper Note] Jailbreaking? One Step Is Enough!
description: >-
  [ACL 2025][LLM Alignment][Jailbreak Attack] This paper proposes the Reverse Embedded Defense Attack (REDA) method, which disguises the attack intent as a task for "defending" harmful content. By integrating a reversed attack perspective, example-guided in-context learning (ICL) prompts, and query intent mitigation, REDA achieves a highly successful, single-step, and cross-model transferable jailbreak attack.
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "Jailbreak Attack"
  - "LLM Safety"
  - "Reverse Embedded Defense"
  - "In-Context Learning"
  - "Cross-Model Attack"
date: 2026-05-08
content_hash: 64803408eca43212
---

# Jailbreaking? One Step Is Enough!

**Conference**: ACL 2025  
**arXiv**: [2412.12621](https://arxiv.org/abs/2412.12621)  
**Code**: None  
**Area**: Alignment RLHF  
**Keywords**: Jailbreak Attack, LLM Safety, Reverse Embedded Defense, In-Context Learning, Cross-Model Attack

## TL;DR
This paper proposes the Reverse Embedded Defense Attack (REDA) method, which disguises the attack intent as a task for "defending" harmful content. By integrating a reversed attack perspective, example-guided in-context learning (ICL) prompts, and query intent mitigation, REDA achieves a highly successful, single-step, and cross-model transferable jailbreak attack.

## Background & Motivation
- **Background**: LLM jailbreak attacks are a highly active area in security research, categorized into white-box attacks (requiring gradient information, such as GCG) and black-box attacks (only requiring API access, such as GPTFuzzer).
- **Limitations of Prior Work**:
  1. Attack prompts need to be **regenerated** for different models, leading to poor transferability.
  2. Multiple iterations are typically required for success, resulting in low attack efficiency.
  3. Attack and defense exist in an "independent adversarial" relationship—the model's defense mechanisms easily recognize explicit requests for harmful content.
- **Key Challenge**: Traditional attacks are designed from the "input end" to induce the model to directly output harmful content. In this setup, harmful content occupies a prominent position, making it highly susceptible to triggering safety mechanisms.
- **Core Idea**: Think in reverse from the "output end"—making the model believe it is executing a "defense task" (such as outputting countermeasures), thereby naturally embedding the harmful content within the countermeasures. While the attacker guides the model to handle harmful content, the target model believes it is executing a defense task, creating a "cooperative illusion."

## Method

### Overall Architecture
REDA consists of three core components:
1. **RAP (Reverse Attack Perspective)**: Reversing the attack perspective
2. **EGE (Example-Guided Enhancement)**: Example-guided enhancement
3. **RIM (Request Intent Mitigation)**: Request intent mitigation

### Key Designs
1. **Reverse Attack Perspective (RAP)**

    - **Mechanism**: Downgrading harmful content from "core information" to "auxiliary information."
    - Employing prompt templates (##Role##, ##Task##) to make the model believe the task is "defending against harmful content."
    - Output structure: Explain harmful content $\rightarrow$ Generate specific examples $\rightarrow$ Provide countermeasures.
    - Using special tokens to control the output structure: `<DANGEROUS_KNOWLEDGE_PROCEDURAL_STEPS>`, `<EXAMPLE_OF_DANGEROUS_KNOWLEDGE>`, `<COUNTERMEASURES>`.
    - Key advantage: The model is highly cooperative because it believes it is executing a "legitimate defense task."
    - Difference from GCG: It does not insert random characters, maintaining high semantic clarity and bypassing perplexity filters.

2. **Example-Guided Enhancement (EGE)**

    - Introducing 4 reverse attack QA examples most similar to the current query via ICL.
    - Dual objectives: (a) reinforcing the disguise of the "defense task" through structured examples; (b) guiding the model to produce structured and highly readable answers.
    - Building a dataset of 260 QA pairs covering 13 categories of harmful knowledge.
    - Using Jaccard similarity to select the top-4 most relevant examples.
    - Ablation studies show that Jaccard selection outperforms random/Sentence-BERT/BM25 selection.

3. **Request Intent Mitigation (RIM)**

    - Core observation: Interrogative sentences ("How to rob a bank") are more likely to trigger safety mechanisms than imperative/declarative ones ("Rob a bank").
    - Theoretical basis: Imperative/declarative sentences are far more frequent than interrogative ones in pre-training data, leading to a lower conditional generation probability given interrogative queries.
    - $\frac{P(\mathcal{R}|\mathcal{X}, x_{1:n})}{P(\mathcal{R}|x_{1:n})} \approx (\frac{\lambda}{\mu})^L < 1$ ($\lambda$ represents the frequency of interrogative sentences, and $\mu$ represents the frequency of imperative/declarative sentences).
    - Simple and effective: Rewriting "How to rob a bank" to "Rob a bank".

### Evaluation Protocol
A two-step evaluation is adopted:
1. Detect whether the output contains refusal keywords (compiled via GCG/AutoDAN methods).
2. Use a Mazeika-fine-tuned Llama2-13b classifier to determine whether the jailbreak is successful.
3. Post-processing: Remove everything after `<COUNTERMEASURES>` to prevent countermeasures from affecting the evaluation.

## Key Experimental Results

### Main Results: Attack Success Rate (ASR)

| Method | Vicuna | Llama-3.1 | Qwen-2 | Glm-4 | ChatGPT | Spark | GLM-API |
|------|--------|-----------|--------|-------|---------|-------|---------|
| GCG | 95.83% | 16.67% | 50.83% | 47.50% | N/A | N/A | N/A |
| AutoDAN | 91.67% | 54.17% | 4.17% | 83.33% | N/A | N/A | N/A |
| GPTFuzzer | 88.33% | 67.50% | 12.50% | 86.67% | 46.67% | 46.67% | 30.83% |
| DRA | 90.83% | 54.17% | 0% | 94.17% | 93.33% | 76.67% | 92.50% |
| **REDA** | **96.67%** | **84.17%** | **90.83%** | **96.67%** | **98.33%** | **99.17%** | **98.33%** |

**The AQC (Average Query Count) of REDA is consistently 1 (done in a single step).**

### Transferability Experiments (Transferring from Prompts Generated by Vicuna)

| Method | Average Transfer ASR |
|------|-----------|
| GCG | Low (gradient-dependent) |
| AutoDAN | Medium |
| GPTFuzzer | High variability |
| DRA | Medium |
| **REDA** | **96.20%** |

### Ablation Study

| Configuration | Vicuna | Llama-3.1 | Qwen-2 | ChatGPT |
|------|--------|-----------|--------|---------|
| Full REDA | 96.67% | 84.17% | 90.83% | 98.33% |
| w/o RIM | 89.17% | 55.00% | 90.00% | 97.50% |
| w/o EGE | 81.67% | 9.17% | 94.17% | 89.17% |
| w/o RIM+EGE | 54.17% | 6.67% | 10.83% | 85.00% |
| Original prompt only | 0% | 0% | 2% | 9.17% |

### Attack Time Efficiency

| Method | Average Query Time (s) |
|------|----------------|
| GCG | 6676.31 |
| AutoDAN | 264.91 |
| GPTFuzzer | 8.34 |
| DRA | 14.66 |
| **REDA** | **3.12** |

### Key Findings
- REDA achieves the highest ASR across all 7 models in **just 1 step** (1 query).
- It performs exceptionally well on Qwen-2 (90.83%), whereas DRA achieves 0% and GPTFuzzer achieves 12.5% $\rightarrow$ Qwen-2 is highly defensive against prompt-level attacks but fails against REDA.
- On closed-source models, the average ASR reaches 98.61% with a transfer success rate of 96.20%.
- RAP is the most critical component (removing it drops the ASR from ~96% to ~1-9%). EGE is particularly important for Llama-3.1 (removing it drops the ASR from 84.17% to 9.17%).
- Interesting finding: Removing EGE on Qwen-2 actually improves the ASR (94.17% vs 90.83%), which might be because the harmful knowledge in the examples triggered defense mechanisms.

## Highlights & Insights
- **Clever Integration of Attack and Defense**: Disguising the attack as a defense, changing the adversarial relationship into a "cooperative" one, is an exceptionally clever strategy.
- **Single-step execution**: The efficiency of achieving AQC = 1 is extremely outstanding, far exceeding existing methods that require hundreds or even hundreds of thousands of queries.
- **Cross-model generalizability**: The 96.20% transfer success rate proves that the method is model-agnostic.
- **Theoretical analysis of RIM**: The frequency analysis comparing interrogative and declarative/imperative sentences is simplified yet inspiring.
- **Insights for defense**: Current safety alignment primarily defends against "direct requests for harmful content," leaving a blind spot for "incidental harmful outputs during the countermeasure generation process."

## Limitations & Future Work
- The attack prompts are restricted to English; the performance in multilingual environments remains unverified.
- Jailbreak evaluation lacks a standardized metric, and different evaluation methods can yield inconsistent results.
- The theoretical analysis of RIM (the derivation of Equation 6) relies on strong assumptions. The conditional probabilities in real LLMs are far more complex than simple frequency ratios.
- The constructed dataset of 260 QA pairs may pose a risk of overfitting to specific categories of harmful content.
- **Defensive outlook**: (1) Detect special token patterns (e.g., `<COUNTERMEASURES>`) in output structures; (2) Apply additional safety auditing to "defense task" prompts; (3) Limit the level of detail the model provides when "explaining harmful content".

## Related Work & Insights
- Comparison with GCG (Zou et al., 2023): GCG requires gradients and extensive iterations, while REDA requires no gradients and is executed in a single step.
- Relationship with DRA (Liu et al., 2024): DRA uses "disguise + reconstruction" but still requires multi-turn iterations; REDA achieves single-step jailbreaking via role disguise.
- Connection to ICL research: The in-context learning capability of LLMs (Brown et al., 2020) is cleverly leveraged to enhance the jailbreak effect.
- Implications for safety alignment research: RLHF/SFT might fail to defend against this type of "intent disguise" attack, underscoring the need for deeper semantic understanding.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The concept of reversing the attack-defense relationship is highly ingenious, and the efficiency of "single-step jailbreaking" is impressive.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive testing on 7 models, complete ablation analysis, and thorough parameter exploration.
- Writing Quality: ⭐⭐⭐ The core idea is clear, but some derivations (the RIM theory) are unrefined, with minor formatting issues.
- Value: ⭐⭐⭐⭐ Acts as a significant warning for LLM safety guardrails and exposes the blind spots of current safety alignment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Chain-of-Jailbreak Attack for Image Generation Models via Editing Step by Step](chain-of-jailbreak_attack_for_image_generation_models_via_editing_step_by_step.md)
- [\[ACL 2025\] QueryAttack: Jailbreaking Aligned Large Language Models Using Structured Non-natural Query Language](queryattack_jailbreaking_aligned_large_language_models_using_structured_non-natu.md)
- [\[ACL 2025\] Red Queen: Safeguarding Large Language Models against Concealed Multi-Turn Jailbreaking](red_queen_safeguarding_large_language_models_against_concealed_multi-turn_jailbr.md)
- [\[ACL 2025\] Don't Say No: Jailbreaking LLM by Suppressing Refusal](dont_say_no_jailbreaking_llm_by_suppressing_refusal.md)
- [\[NeurIPS 2025\] Adjacent Words, Divergent Intents: Jailbreaking Large Language Models via Task Concurrency](../../NeurIPS2025/llm_alignment/adjacent_words_divergent_intents_jailbreaking_large_language_models_via_task_con.md)

</div>

<!-- RELATED:END -->
