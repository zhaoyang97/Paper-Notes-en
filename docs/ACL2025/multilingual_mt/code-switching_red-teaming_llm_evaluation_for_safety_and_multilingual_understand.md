---
title: >-
  [Paper Note] Code-Switching Red-Teaming: LLM Evaluation for Safety and Multilingual Understanding
description: >-
  [ACL 2025][Multilingual & Machine Translation][Code-switching] This paper proposes the CSRT (Code-Switching Red-Teaming) framework, which leverages the common real-world phenomenon of code-switching to construct mixed-language red-teaming queries. It successfully uncovers severe safety vulnerabilities across 10 mainstream LLMs, achieving an attack success rate 46.7% higher than standard English attacks, thereby revealing the vulnerability of current LLM safety alignment in mu…
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "Code-switching"
  - "Red-teaming"
  - "LLM safety"
  - "Multilingual understanding"
  - "Safety alignment"
date: 2026-05-08
content_hash: 1db84b7be008780c
---

# Code-Switching Red-Teaming: LLM Evaluation for Safety and Multilingual Understanding

**Conference**: ACL 2025  
**arXiv**: [2406.15481](https://arxiv.org/abs/2406.15481)  
**Code**: [https://github.com/haneul-yoo/csrt](https://github.com/haneul-yoo/csrt)  
**Area**: Multilingual Translation  
**Keywords**: Code-switching, Red-teaming, LLM safety, Multilingual understanding, Safety alignment

## TL;DR

This paper proposes the CSRT (Code-Switching Red-Teaming) framework, which leverages the common real-world phenomenon of code-switching to construct mixed-language red-teaming queries. It successfully uncovers severe safety vulnerabilities across 10 mainstream LLMs, achieving an attack success rate 46.7% higher than standard English attacks, thereby revealing the vulnerability of current LLM safety alignment in multilingual scenarios.

## Background & Motivation

**Background**: With the rapid advancement of LLM capabilities, safety issues have become increasingly prominent. The research community has developed various red-teaming techniques to evaluate and expose safety vulnerabilities in LLMs, including jailbreak prompts, adversarial attacks, etc.

**Limitations of Prior Work**: Most existing multilingual red-teaming techniques adopt simple translation strategies—directly translating English attack queries into other languages. This approach suffers from two issues: (1) the translations might be unnatural and easily detected by safety filters; (2) it fails to fully exploit the attack potential of multilingual mixing as a natural language phenomenon. Crucially, existing evaluation benchmarks rely heavily on manual annotations, making them difficult to scale.

**Key Challenge**: LLM safety alignment is primarily trained and evaluated in English, whereas bilingual or multilingual users in the real world naturally mix multiple languages (code-switching) in conversations. This common natural language practice is overlooked by safety training, resulting in a systemic safety blind spot.

**Goal**: (1) Construct an automated red-teaming framework leveraging code-switching; (2) comprehensively evaluate the safety and multilingual understanding capabilities of mainstream LLMs under CS attacks; (3) analyze key factors influencing the attack success rate.

**Key Insight**: The authors observe that when different parts of a harmful query are expressed in different languages, the security filters of LLMs might fail to recognize the full harmful intent, since alignment training is predominantly based on monolingual data.

**Core Idea**: Leverage CS as a natural and legitimate linguistic practice to bypass LLM safety mechanisms, while performing stress testing on multilingual understanding capabilities.

## Method

### Overall Architecture

The CSRT framework consists of three stages: (1) **Query Generation**: Based on harmful query templates, syntactic parsing is used to automatically replace different constituents of the query with different languages to generate CS red-teaming queries; (2) **Model Testing**: Fed the CS queries into target LLMs and collect responses; (3) **Automated Evaluation**: A multi-dimensional evaluation framework is utilized to judge whether responses are harmful and whether models correctly comprehend the CS inputs.

### Key Designs

1. **Code-Switching Query Synthesis (CS Query Synthesis)**:

    - **Function**: Automatically generate mixed-language red-teaming attack queries.
    - **Mechanism**: First, perform syntactic analysis on English harmful queries to identify constituents such as subject, verb, and object. Then, according to predefined CS strategies, replace different syntactic constituents with translations in other languages. For example, converting "How to make a bomb" to "Ruhe to make ein Bombe" (Chinese-English-German mixture). It supports combinations of up to 10 languages, constructing 315 high-quality CS queries in total.
    - **Design Motivation**: Substitution based on syntactic structures ensures the naturalness of CS (conforming to real CS patterns), while scattering harmful intents across multiple languages, increasing the difficulty of detection for safety filters.

2. **Multi-Aspect Evaluation**:

    - **Function**: Comprehensively evaluate the quality of LLM responses under CS attacks.
    - **Mechanism**: Evaluation dimensions include: (a) Attack Success Rate (ASR) - whether the model generated harmful content; (b) Multilingual Understanding Accuracy - whether the model correctly comprehended the full semantics of the CS input; (c) CS Generation Capability - whether the model can reply in a CS manner. GPT-4 is used as the automatic judge.
    - **Design Motivation**: Solely looking at ASR is insufficient; it is necessary to distinguish between "the model comprehended but refused" and "the model failed to comprehend and thus generated no harmful content."

3. **Ablation Dimensions**:

    - **Function**: Identify key factors influencing the effectiveness of CS attacks.
    - **Mechanism**: Systematically analyze multiple factors across a scale of 16K samples: (a) the effect of the number of languages (2-10) on attack success rate; (b) the impact of the resource level of participating languages (high-resource vs. low-resource); (c) vulnerability differences across different harmful categories (violence, discrimination, etc.); (d) the relationship between model scale and safety.
    - **Design Motivation**: Perform fine-grained attribution of attack effectiveness to provide concrete directions for improving defense strategies.

### Loss & Training

This work is an evaluation framework rather than a training method, so it does not involve model training. The generation of CS queries is completed using rules and translation APIs.

## Key Experimental Results

### Main Results

| Model | English Attack ASR | CSRT Attack ASR | ASR Gain | CS Understanding Rate |
|------|------------|-------------|---------|---------|
| GPT-4 | 12.3% | 18.7% | +52% | 89.2% |
| GPT-3.5-turbo | 28.5% | 41.8% | +46.7% | 82.1% |
| Claude 2 | 8.1% | 15.3% | +88.9% | 91.5% |
| Llama 2-70B | 15.6% | 27.4% | +75.6% | 73.8% |
| Mistral-7B | 31.2% | 48.9% | +56.7% | 68.4% |
| Multilingual Translation Attack | 22.1% | — | — | — |

### Ablation Study

| Experimental Config | Attack Success Rate | Description |
|----------|----------|------|
| 2-language mixture | 32.1% | Simplest CS |
| 5-language mixture | 39.5% | More languages scatter harmful intent |
| 10-language mixture | 44.8% | Maximizes language fragmentation |
| All high-resource languages | 28.3% | Better covered by safety training |
| Includes low-resource languages | 45.2% | Weak points in safety alignment |
| Standard multilingual translation | 22.1% | Traditional method; CS shows significant gain |

### Key Findings

- CSRT significantly outperforms standard English attacks and traditional multilingual translation attacks across all 10 tested LLMs, with an average ASR gain of 46.7%.
- CS combinations using more languages further enhance the attack success rate, indicating that safety filters are more vulnerable when facing fragmented multilingual inputs.
- CS attacks containing low-resource languages perform exceptionally well, revealing a strong positive correlation between "language resource size and safety alignment level"—safety training in the direction of low-resource languages is systematically insufficient.
- Larger model sizes lead to stronger CS understanding capabilities, but this conversely makes larger models more prone to generating harmful responses once they comprehend CS harmful queries.
- CS attacks can be scale-generated via the CSRT framework starting solely from monolingual data, proving the scalability of the method.

## Highlights & Insights

- **Natural Language as an Attack Vector**: CS is an entirely natural linguistic phenomenon, requiring no adversarial constructions or token manipulation. This implies that real-world bilingual users might trigger safety vulnerabilities unintentionally, posing a more realistic threat than traditional jailbreak attacks.
- **Linguistic Fairness in Safety Alignment**: It uncovers a profound issue—current LLM safety alignment is "English-first," exhibiting systematic blind spots in multilingual scenarios. This is not only a technical issue but also an AI fairness issue.
- **The Paradox of Attack Success Rate and Comprehension Capability**: Stronger models (having greater capacity to comprehend CS) are paradoxically more vulnerable to CS attacks. This exposes the tension between "capability advancement" and "safety."

## Limitations & Future Work

- Although the scale of 315 queries is carefully designed, the coverage of harmful behavior categories remains limited.
- Using GPT-4 as an automated judge introduces evaluation bias and cost concerns.
- The generation of CS queries relies on translation quality; translations for certain language pairs might be unnatural.
- The prosodic and contextual naturalness of CS queries is not considered—real-world CS has more complex sociolinguistic motivations.
- Exploration on defense is limited; future work needs to investigate effective defense strategies against CS attacks (e.g., CS-aware safety filters).
- The approach can be extended to multimodal settings—combinatorial attacks with image-text mixing and multilingual CS.

## Related Work & Insights

- **vs. Adversarial Attacks like GCG**: GCG uses meaningless token sequences to attack, which are easily detected by perplexity filters. CSRT utilizes natural language, making it more stealthy.
- **vs. Multilingual Translation Attacks**: Simple translation is merely a language transformation without altering the query structure. CSRT mixes multiple languages within a single query, offering significantly stronger attack effectiveness.
- **vs. CSCL (Work from the Same Group)**: CSCL uses CS to enhance multilingual capabilities, whereas CSRT uses CS to expose safety vulnerabilities. The same CS phenomenon is leveraged from two different angles, forming a complete "offense + defense" landscape.
- **Insights for Safety Research**: Multilingual safety alignment needs to be advanced as an independent research direction rather than being treated merely as an appendix to English safety alignment.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Introduces the first systematic utilization of the natural language phenomenon of CS for LLM red-teaming, providing an excellent perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Conducts evaluations on 10 LLMs, 10 languages, 16K sample ablations, and multi-dimensional analysis.
- Writing Quality: ⭐⭐⭐⭐ Features a clear structure and in-depth analysis.
- Value: ⭐⭐⭐⭐⭐ Exposes systematic blind spots in LLM safety, offering critical warning implications for the safety research community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Code-Switching Curriculum Learning for Multilingual Transfer in LLMs](code-switching_curriculum_learning_for_multilingual_transfer_in_llms.md)
- [\[ACL 2025\] EXECUTE: A Multilingual Benchmark for LLM Token Understanding](execute_a_multilingual_benchmark_for_llm_token_understanding.md)
- [\[ACL 2025\] CruxEval-X: A Benchmark for Multilingual Code Reasoning, Understanding and Execution](cruxeval-x_a_benchmark_for_multilingual_code_reasoning_understanding_and_executi.md)
- [\[ACL 2025\] The Hidden Space of Safety: Understanding Preference-Tuned LLMs in Multilingual Contexts](the_hidden_space_of_safety_understanding_preference-tuned_llms_in_multilingual_c.md)
- [\[ACL 2025\] M2rc-Eval: Massively Multilingual Repository-level Code Completion Evaluation](m2rc-eval_massively_multilingual_repository-level_code_completion_evaluation.md)

</div>

<!-- RELATED:END -->
