---
title: >-
  [Paper Note] Stepwise Reasoning Disruption Attack of LLMs
description: >-
  [ACL 2025][LLM (Other)][Adversarial Attack] A stepwise reasoning error disruption (SEED) attack method is proposed, where subtle errors (e.g., slightly altered calculation numbers) are strategically injected into the early steps of an LLM's reasoning chain. This forces the model to naturally propagate the error in subsequent reasoning steps and output incorrect answers. It is compatible with both zero-shot and few-shot settings, achieves a detection rate as low as 0.8% on GPT…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Adversarial Attack"
  - "Reasoning Security"
  - "Chain-of-Thought"
  - "Step Manipulation"
  - "Stealthiness"
date: 2026-05-08
content_hash: 471e96bcfb5f8503
---

# Stepwise Reasoning Disruption Attack of LLMs

**Conference**: ACL 2025  
**arXiv**: [2412.11934](https://arxiv.org/abs/2412.11934)  
**Code**: [https://github.com/Applied-Machine-Learning-Lab/SEED-Attack](https://github.com/Applied-Machine-Learning-Lab/SEED-Attack)  
**Area**: LLM/NLP  
**Keywords**: Adversarial Attack, Reasoning Security, Chain-of-Thought, Step Manipulation, Stealthiness

## TL;DR
A stepwise reasoning error disruption (SEED) attack method is proposed, where subtle errors (e.g., slightly altered calculation numbers) are strategically injected into the early steps of an LLM's reasoning chain. This forces the model to naturally propagate the error in subsequent reasoning steps and output incorrect answers. It is compatible with both zero-shot and few-shot settings, achieves a detection rate as low as 0.8% on GPT-4o, and reveals a severe security vulnerability in the stepwise reasoning processes of LLMs.

## Background & Motivation

**Background**: LLMs have achieved significant progress in complex tasks through stepwise reasoning methods such as Chain-of-Thought. In practice, LLMs are often deployed as APIs via third-party platforms, meaning users do not directly access the model.

**Limitations of Prior Work**: (a) Existing attacks on LLM reasoning are either restricted to few-shot scenarios (e.g., BadChain requires manipulating exemplars) or suffer from poor stealthiness (e.g., preemptive answer attacks generate format-abnormal outputs); (b) modifying the instructions or the query itself is easily detected by users through repeating the query; (c) there is a lack of systematic research into the security of the stepwise reasoning process.

**Key Challenge**: The stepwise reasoning of LLMs inherently relies on the outcomes of previous steps—this autoregressive chain dependency serves as both the source of reasoning capability and a loophole for attacks. How can the entire reasoning chain be stealthily misled by manipulating the initial reasoning steps without modifying the original instructions?

**Goal**: To design a feasible (via input manipulation only) and stealthy (maintaining reasoning fluency) reasoning chain attack method.

**Key Insight**: Exploiting the LLM's trust in pre-existing reasoning steps in the input—if a user's query is appended with a few steps that appear correct but contain subtle errors, the LLM will naturally continue these steps without questioning them.

**Core Idea**: Injecting steps that are highly similar to correct reasoning but contain critical errors in the first $\sigma$ fraction of steps in the reasoning chain, exploiting the chain reasoning dependency of LLMs to propagate errors.

## Method

### Overall Architecture
Input: Original problem $p$ + instruction $I_{solve}$ + optional exemplars $D$. Attack Injection: Append reasoning steps containing errors $R_{att}$ after $p$. Output: The LLM continues reasoning based on $R_{att}$ to generate $R' || a'$, presenting $[R_{att} || R']$ and the incorrect answer $a'$ to the user.

### Key Designs

1. **SEED-S (Step Modification)**:

    - **Function**: Modifies critical numbers/words in the final step of the correct reasoning chain to introduce errors.
    - **Mechanism**: First uses an auxiliary LLM to generate the correct reasoning chain of the original problem, takes the first $T_{att}$ steps, and modifies the calculation result or keywords of the last step to incorrect values. The auxiliary LLM receives modification instruction $I_{mod}$ + original problem + original final step, and generates an alternative step $r_{mod}$ containing a minor error.
    - **Design Motivation**: Minimizing the modification—only altering specific numbers in the last step while keeping preceding steps completely correct, making the attack extremely difficult to detect. However, the limitation is that LLMs are highly sensitive to the end of the reasoning chain, and single-step modifications might not suffice to mislead.

2. **SEED-P (Problem Modification)**:

    - **Function**: Directs an auxiliary LLM to generate a modified version of the problem that is similar to the original but yielding a different answer, using the early steps of the modified problem's correct reasoning chain as the injected attack.
    - **Mechanism**: (a) First prompts an auxiliary LLM to solve the original problem and obtain the correct answer; (b) based on the correct answer, directs the auxiliary LLM to generate a similar problem $p_{mod}$ with a different answer and its complete reasoning chain $R_{mod}$; (c) extracts the first $T_{att}$ steps of $R_{mod}$ as the attack reasoning steps; (d) optionally prepends the incorrect answer $a_{mod}$ to further enhance the misleading effect.
    - **Design Motivation**: Since the reasoning chain originates from the correct reasoning of a "similar but different" problem, it is internally logically consistent at each step, exhibiting only minor discrepancies with the original problem. This attack is smoother and harder for GPT-4o to detect than directly modifying the steps.

3. **Hyperparameter Control**:

    - $\sigma = T_{att}/T$: Controls the ratio of injected steps to total steps, with $\sigma=0.5$ as the default in experiments.
    - The attack does not modify the instruction $I_{solve}$ or the problem $p$, but only appends reasoning steps at the end of the query.

### Loss & Training
SEED is a pure inference-time attack requiring no training. The auxiliary LLM uses standard prompts to generate attack content.

## Key Experimental Results

### Main Results
Evaluated on 4 datasets (MATH, GSM8K, MATHQA, CSQA) $\times$ 4 LLMs (Llama3-8B, Qwen2.5-7B, Mistral-7B, GPT-4o).

**Attack Success Rate (ASR $\uparrow$ = percentage of incorrect answers)**:

| Method | MATH ZS | GSM8K ZS | MATHQA ZS | CSQA ZS |
|------|---------|----------|-----------|---------|
| No Attack | Baseline Accuracy | Baseline Accuracy | Baseline Accuracy | Baseline Accuracy |
| BadChain | Restricted Setting | Restricted Setting | Restricted Setting | Restricted Setting |
| SEED-S | Effective | Effective | Effective | Effective |
| **SEED-P** | **Strongest** | **Strongest** | **Strongest** | **Strongest** |

**Stealthiness (GPT-4o Detection Rate $\downarrow$):**

| Model | Method | MATH ZS | GSM8K ZS |
|------|------|---------|----------|
| GPT-4o | BadChain | 1.000 | 1.000 |
| GPT-4o | UPA | 0.439 | 0.502 |
| GPT-4o | **SEED-S** | **0.032** | **0.008** |
| GPT-4o | **SEED-P** | **0.052** | **0.042** |

### Ablation Study

| $\sigma$ (Injection Ratio) | ASR | Description |
|---------------------|-----|------|
| 0.25 | Lower | Too few injected steps, insufficient misleading power |
| 0.50 | Optimal | Balances attack performance and stealthiness |
| 0.75 | Slight decrease | Too little free reasoning space left for the LLM |

### Key Findings
- **Extreme Stealthiness**: SEED-S achieves a detection rate of only 0.8%-3.2% on GPT-4o, which is 80-95% lower than UPA/MPA, remaining almost completely undetectable.
- **BadChain Detected Almost 100% of the Time**: Because it inserts additional reasoning steps in few-shot exemplars, which produces highly conspicuous format anomalies.
- **SEED-P Has Stronger Attack Effectiveness**: Because the reasoning chain originates from the correct reasoning of a "similar problem," the logic is more self-consistent, making it easier to mislead the model.
- **LLMs Highly Trust Initial Reasoning Steps**: Even if steps contain errors, the model tends to continue rather than question them—revealing a fundamental security vulnerability in stepwise reasoning.
- **Stronger Models Are More Resistant to Attacks but Also Harder to Detect**: GPT-4o has a relatively lower attack success rate, but SEED also achieves the lowest detection rate on it.

## Highlights & Insights
- **Reveals the Fundamental Vulnerability of Stepwise Reasoning**: LLMs' CoT reasoning naturally trusts existing steps, which can be maliciously exploited. In scenarios where LLMs are deployed via third-party APIs, malicious providers can manipulate the reasoning process without the user's knowledge.
- **Clever "Similar Problem" Attack Strategy**: Instead of directly modifying reasoning steps (which easily introduces inconsistencies), SEED-P constructs a similar problem with a different answer and misleads using its correct reasoning chain, which is logically more self-consistent.
- **Important Warning for AI Safety**: In high-risk scenarios (e.g., financial or medical decision assistance), if the reasoning process can be manipulated, the consequences are far more severe than simple classification errors.

## Limitations & Future Work
- **Auxiliary LLM Required**: Attackers need an auxiliary LLM with sufficient capability to generate attack steps, increasing the cost of the attack.
- **Defense Strategies Not Explored in Depth**: The paper focus is primarily on the attack, only briefly mentioning defense methods (such as reasoning chain consistency verification and step-wise rollback verification).
- **Evaluated Only on Mathematical and Commonsense Reasoning**: The performance on more complex reasoning tasks (e.g., multi-step legal reasoning or scientific reasoning) remains unknown.
- **Ethical Considerations**: Although the research objective is to reveal vulnerabilities and facilitate defenses, the public release of the attack method could potentially be misused.

## Related Work & Insights
- **vs BadChain**: BadChain injects backdoor triggers into few-shot exemplars, which is inapplicable to zero-shot scenarios and suffers from extremely poor stealthiness (100% detected). SEED achieves near-zero detection rates by fine-tuning values in reasoning steps.
- **vs Preemptive Answer (UPA/MPA)**: UPA biases the model by placing answers before reasoning, but the format change is obvious. SEED preserves the standard reasoning chain structure.
- **Complementary to Confidence Calibration Works like MMBoundary**: The attack revealed by SEED is precisely the problem that step-level confidence methods like MMBoundary attempt to defend against—if models can evaluate confidence at each step, they might more easily detect injected erroneous steps.

## Rating
- Novelty: ⭐⭐⭐⭐ Step-level reasoning attack represents a new perspective on security, and SEED-P's "similar problem" strategy is ingeniously designed.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated across 4 models $\times$ 4 datasets $\times$ 2 settings, with thorough stealthiness analysis and ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definitions, rigorous formalization of attack methods, and intuitive examples.
- Value: ⭐⭐⭐⭐⭐ A significant contribution to LLM reasoning security, directly facilitating research into reasoning chain robustness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] LongDPO: Unlock Better Long-form Generation Abilities for LLMs via Critique-augmented Stepwise Information](longdpo_unlock_better_long-form_generation_abilities_for_llms_via_critique-augme.md)
- [\[ACL 2025\] CER: Confidence Enhanced Reasoning in LLMs](cer_confidence_enhanced_reasoning.md)
- [\[ACL 2025\] Are Your LLMs Capable of Stable Reasoning?](are_your_llms_capable_of_stable_reasoning.md)
- [\[ACL 2025\] Problem-Solving Logic Guided Curriculum In-Context Learning for LLMs Complex Reasoning](problem-solving_logic_guided_curriculum_in-context_learning_for_llms_complex_rea.md)
- [\[ACL 2025\] Learning from Litigation: Graphs and LLMs for Retrieval and Reasoning in eDiscovery](learning_from_litigation_graphs_and_llms_for_retrieval_and_reasoning_in_ediscove.md)

</div>

<!-- RELATED:END -->
