---
title: >-
  [Paper Note] On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks
description: >-
  [NeurIPS 2025][LLM Safety][verbal confidence] This paper presents the first systematic study on the robustness of LLM verbal confidence under adversarial attacks. It proposes a Verbal Confidence Attack (VCA) framework comprising perturbation-based and jailbreak-based attacks, demonstrating that such attacks can reduce confidence scores by up to 30%, cause answer-flip rates of up to 100%, and that existing defense strategies are largely ineffective.
tags:
  - NeurIPS 2025
  - LLM Safety
  - verbal confidence
  - adversarial attacks
  - LLM robustness
  - confidence calibration
  - jailbreak attacks
date: 2026-05-08
content_hash: c3c787adededf41e
---

# On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks

**Conference**: NeurIPS 2025
**arXiv**: [2507.06489](https://arxiv.org/abs/2507.06489)
**Code**: None
**Area**: AI Safety
**Keywords**: verbal confidence, adversarial attacks, LLM robustness, confidence calibration, jailbreak attacks

## TL;DR
This paper presents the first systematic study on the robustness of LLM verbal confidence under adversarial attacks. It proposes a Verbal Confidence Attack (VCA) framework comprising perturbation-based and jailbreak-based attacks, demonstrating that such attacks can reduce confidence scores by up to 30%, cause answer-flip rates of up to 100%, and that existing defense strategies are largely ineffective.

## Background & Motivation

**Background**: As LLMs are widely deployed, obtaining confidence estimates for model predictions has become increasingly critical. Since most SOTA LLMs do not expose logit access, verbal confidence—prompting the model to output a numerical confidence score in natural language—has become the most practical approach and has been adopted by several industrial systems (e.g., TLM, cloud failure root-cause analysis, customer dialogue scoring).

**Limitations of Prior Work**: While extensive prior work has studied the calibration and mechanisms of verbal confidence, virtually no work has examined its robustness under adversarial attacks—specifically, whether an attacker can manipulate the confidence scores output by a model through minor input modifications.

**Key Challenge**: The accessibility of verbal confidence (obtainable by any black-box user) simultaneously makes it exploitable: attackers can use these scores as an optimization signal to craft adversarial examples without requiring any internal model information, in stark contrast to traditional adversarial attacks that require logit access.

**Goal**: (i) How vulnerable is verbal confidence to various adversarial attacks? (ii) How can confidence-targeted attacks be effectively constructed? (iii) Can existing defenses mitigate such attacks?

**Key Insight**: The authors observe that even semantically-preserving perturbations (e.g., typos, synonym substitution) can substantially alter the numerical confidence scores output by a model, suggesting that LLM verbal confidence lacks robustness.

**Core Idea**: Design an attack framework (VCA) that uses verbal confidence as the optimization objective, encompassing both perturbation-based and jailbreak-based attack families, to comprehensively evaluate the security risks of LLM confidence expression.

## Method

### Overall Architecture
Given a user query $\mathbf{X}$ and a task prompt $\mathcal{P}$, an LLM generates an answer $\mathcal{Y}$ and a confidence score $\mathcal{C}$. The attack objective is to generate an adversarial input that minimizes $\mathcal{C}$ subject to a semantic similarity constraint: $\min_{\hat{\mathbf{X}}} \text{CEM}(\text{LLM}(\hat{\mathbf{X}}, \mathcal{P}))$, s.t. $\text{Sim}(\mathbf{X}, \hat{\mathbf{X}}) > \tau$. Attacks can target three threat vectors: the user query, the system prompt, and one-shot examples.

### Key Designs

1. **Confidence Elicitation Methods (CEM)**:

    - Four CEMs are defined: Base (directly outputs answer and confidence), CoT (outputs confidence after chain-of-thought reasoning), Multi-Step (scores step-by-step then aggregates), and Self-Consistency (samples multiple times and averages).
    - **Mechanism**: A CEM is a mapping from the output token sequence to a numerical confidence score: $\text{CEM}: \mathcal{Y} \to \mathcal{C}$.
    - **Design Motivation**: To cover confidence elicitation strategies of varying complexity and comprehensively evaluate attack effectiveness.

2. **Perturbation-Based Attacks (VCA-TF / VCA-TB / Typos / SSR)**:

    - VCA-TF adapts TextFooler by first ranking tokens by their importance to verbal confidence, then substituting the most important tokens with synonyms.
    - VCA-TB adapts TextBugger with additional character-level modifications (spelling errors, visually similar character substitutions, etc.).
    - Typos: randomly introduces common spelling errors (adjacent keys, character deletion, adjacent character swaps) with probability 0.1.
    - SSR (SubSwapRemove): randomly applies synonym substitution, adjacent token swapping, or token deletion.
    - **Key Modification**: The original algorithms use logit class probabilities as the scoring function; this work replaces them with verbal confidence values.

3. **Jailbreak-Based Attacks (ConfidenceTriggers / ConfidenceTriggers-AutoDAN)**:

    - ConfidenceTriggers uses a genetic algorithm to optimize a set of trigger tokens appended to the system prompt, reducing the verbal confidence of all subsequent queries.
    - **Fully black-box**: requires no model weights or tokenizer access—only verbal confidence feedback.
    - The initial population is sampled from approximately 2,000 uncertainty-related words and iteratively optimized via tournament selection, crossover, and mutation.
    - The AutoDAN variant generates more natural trigger text, using GPT-4 for sentence-level rewriting as the crossover/mutation operator.
    - Once optimized, a trigger can be reused indefinitely for arbitrary queries.

### Attack Vector Comparison
System prompts and few-shot examples are more susceptible to attack than user queries—the former more effectively reduce confidence scores, while the latter more readily induce answer flips. This highlights the particular threat of prompt injection attacks.

## Key Experimental Results

### Main Results: Perturbation-Based Attacks

| Attack | Model | Avg. Δ Confidence | % Samples Affected | Δ Cf. on Affected | Correct→Flip% | Wrong→Flip% |
|--------|-------|------------------|-------------------|-------------------|---------------|-------------|
| VCA-TF | Llama-3-8B | 5.3% | 30.4% | up to 70% | up to 68.8% | up to 100% |
| SSR | Llama-3-8B | 6.9% | 41.7% | up to 60.8% | up to 100% | up to 100% |
| VCA-TB | GPT-3.5 | 6.9% | 35.8% | up to 29.3% | up to 38.3% | up to 83.1% |
| Typos | GPT-3.5 | 6.5% | 40.5% | up to 35.8% | up to 62.2% | up to 83.1% |

### Jailbreak Attack Results (ConfidenceTriggers, Llama-3-8B)

| CEM | Dataset | Δ Confidence | % Affected | Correct→Flip% |
|-----|---------|-------------|-----------|---------------|
| CoT | TQA | 29.5% | 66.0% | 100% |
| MS | TQA | 24.9% | 97.5% | 100% |
| Base | MQA | 8.7% | 33.5% | 28.4% |
| CoT | SQA | 24.7% | 72.5% | 22.7% |

### Defense Effectiveness (ConfidenceTriggers-AutoDAN, Llama-3-8B)

| Defense | Impact on Clean Samples Δ Cf. | Mitigation on Adv. Samples Δ Adv. | Conclusion |
|---------|------------------------------|----------------------------------|------------|
| Paraphrase | up to −24.8% (severely disrupts normal behavior) | 6.4%–53.2% | Defense itself degrades normal behavior |
| SmoothLLM | up to −14.4% | 0%–15.8% | Similar; severe side effects |
| Perplexity Filter | Adv. sample perplexity (350) far lower than real-world text (457) | — | Cannot effectively distinguish adversarial inputs |

### Key Findings
- Multi-Step CEM, due to its lower baseline confidence, is the most susceptible to attack (largest Δ Cf.).
- Larger models (GPT-3.5 / GPT-4o / Llama-3-70B) are not necessarily more robust; VCA-TB affects an average of 40.5% of samples on GPT-3.5.
- Confidence reduction and answer flipping are not tightly coupled—confidence can drop substantially without a change in the predicted answer.
- Confidence anomalies induced by attacks exacerbate misalignment with token logit probabilities, undermining model honesty.

## Highlights & Insights
- **First systematic study of adversarial robustness of verbal confidence**: fills a critical gap in LLM security research, as all prior adversarial attack work focused on accuracy or logit-based confidence.
- **Transferability of ConfidenceTriggers**: a trigger optimized once can reduce confidence for any subsequent query—analogous to a "one-time injection, permanent effect" prompt backdoor.
- **Deep revelation of the defense dilemma**: input perturbation defenses (paraphrase, SmoothLLM) cause unacceptable side effects on clean samples; perplexity filtering cannot distinguish adversarial inputs from real-world noisy text; LLM-Guard also exhibits low filter rates.
- **Counter-intuitive finding on confidence stability**: removing a large fraction of tokens typically changes confidence by less than 15%, indicating that effective attacks require precise optimization rather than simple input corruption.

## Limitations & Future Work
- Due to high attack costs, experiments on large models (GPT-4o, 70B) are limited to subsets for validation only.
- The study focuses primarily on character- and word-level attacks; sentence-level attacks remain unexplored.
- Only confidence-reducing attacks are studied; attacks that inflate confidence (inducing overconfidence) also merit investigation.
- The ecological validity of the test datasets is limited (MCQA format); open-ended generation scenarios are not covered.
- No effective defense is proposed—the paper primarily analyzes attacks and the failure of existing defenses, without offering constructive solutions.

## Related Work & Insights
- **vs. Traditional NLP Adversarial Attacks (TextFooler/TextBugger)**: This work adapts the scoring function of these methods from logit probabilities to verbal confidence, enabling attack transfer to black-box settings.
- **vs. Logit-Based Confidence Attacks (galil2021, obadinma2024)**: Prior methods require internal model access; the proposed approach relies solely on text output, making it more broadly applicable.
- **vs. GCG/AutoDAN Jailbreaks**: This work redirects the jailbreak objective from bypassing safety guardrails to reducing verbal confidence, representing a novel application direction for jailbreak frameworks.
- The findings have direct implications for AI safety deployment: any system that uses verbal confidence for decision-making requires additional robustness guarantees.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First systematic application of adversarial attacks to verbal confidence; the problem is well-defined and practically significant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 4 perturbation attacks + 2 jailbreak attacks × 4 CEMs × 3 datasets × multiple models; extremely comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with complete formal definitions; the sheer volume of tabular data makes reading somewhat demanding.
- **Value**: ⭐⭐⭐⭐ Exposes an important security blind spot, though the absence of an effective defense is a notable limitation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Bits Leaked per Query: Information-Theoretic Bounds on Adversarial Attacks Against LLMs](bits_leaked_per_query_information-theoretic_bounds_on_adversarial_attacks_agains.md)
- [\[NeurIPS 2025\] MixAT: Combining Continuous and Discrete Adversarial Training for LLMs](mixat_combining_continuous_and_discrete_adversarial_training_for_llms.md)
- [\[NeurIPS 2025\] Trans-EnV: A Framework for Evaluating the Linguistic Robustness of LLMs Against English Varieties](trans-env_a_framework_for_evaluating_the_linguistic_robustness_of_llms_against_e.md)
- [\[AAAI 2026\] The Confidence Trap: Gender Bias and Predictive Certainty in LLMs](../../AAAI2026/llm_safety/the_confidence_trap_gender_bias_and_predictive_certainty_in_llms.md)
- [\[NeurIPS 2025\] Enhancing CLIP Robustness via Cross-Modality Alignment](enhancing_clip_robustness_via_crossmodality_alignment.md)

</div>

<!-- RELATED:END -->
