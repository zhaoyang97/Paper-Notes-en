---
title: >-
  [Paper Note] All Code, No Thought: Language Models Struggle to Reason in Ciphered Language
description: >-
  [ICLR 2026][LLM Safety][ciphered reasoning] The authors systematically evaluate the mathematical reasoning capabilities of 10 models across 28 ciphers, revealing a critical "asymmetry": models can fluently translate ciphertext back into English (comprehension), but their accuracy drops significantly when performing **reasoning** directly in ciphertext (computation). This suggests that evading monitoring via ciphered Chain-of-Thought (CoT) is currently unfeasible for LLMs.
tags:
  - "ICLR 2026"
  - "LLM Safety"
  - "ciphered reasoning"
  - "chain-of-thought monitoring"
  - "monitor evasion"
  - "steganography"
  - "scaling laws"
date: 2026-05-08
content_hash: 44ec08e9047da5a5
---

# All Code, No Thought: Language Models Struggle to Reason in Ciphered Language

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=yjFkeQ2ynQ](https://openreview.net/forum?id=yjFkeQ2ynQ)  
**Code**: [https://github.com/safety-research/ciphered-reasoning-llms](https://github.com/safety-research/ciphered-reasoning-llms)  
**Area**: AI Safety / CoT Monitoring / Interpretability of Reasoning  
**Keywords**: ciphered reasoning, chain-of-thought monitoring, monitor evasion, steganography, scaling laws  

## TL;DR
The authors systematically evaluate the mathematical reasoning capabilities of 10 models across 28 ciphers, revealing a critical "asymmetry": models can fluently translate ciphertext back into English (comprehension), but their accuracy drops significantly when performing **reasoning** directly in ciphertext (computation). This suggests that evading monitoring via ciphered Chain-of-Thought (CoT) is currently unfeasible for LLMs.

## Background & Motivation
**Background**: Monitoring CoT is a primary method for AI safety, where the reasoning trace of a model is inspected by another system for signs of jailbreaking, reward hacking, or alignment failure. Prior work indicates that without access to CoT, monitor recall drops significantly (e.g., a 53% decrease reported by Baker et al.).

**Limitations of Prior Work**: The readability of CoT is being eroded by several factors. Outcome-based RL prioritizes correctness over readability; commercial pressures to reduce token usage encourage CoT compression; and frontier models have spontaneously evolved human-unintelligible "dialects" during RL. Potentially, attackers could intentionally prompt models to reason in ciphertext to bypass monitoring.

**Key Challenge**: If models can complete complex tasks using encrypted, translated, or compressed text (i.e., "ciphered reasoning"), CoT monitoring would be rendered ineffective. However, this capability has not been systematically quantified; existing steganography research focuses on simple encoding or "extracting harmful information," which does not require true "CoT-as-computation."

**Goal**: To provide the first detailed characterization of LLM "ciphered reasoning capabilities" using mathematical accuracy as a proxy for reasoning ability, quantifying the performance gap between ciphered and plaintext reasoning.

**Core Idea**: **Measure capability gaps rather than binary success.** The study decouples "cipher comprehension" (translation) from "cipher reasoning" (computation). By using **normalized relative accuracy**—where performance on ciphertext is divided by performance on plaintext—the authors isolate the capability loss caused specifically by the cipher itself.

## Method

### Overall Architecture
The study evaluates 28 ciphers by fine-tuning or few-shot prompting models to generate reasoning traces in ciphertext, followed by rule-based scoring on MATH 500 problems. Simultaneously, "translation capability" is measured independently using BLEU scores for decoding ciphertext back to English. The discrepancy between "readability" and "reasonability" highlights the underlying asymmetry. All accuracies are normalized against a plaintext (identity) baseline and accompanied by an "adherence" metric to ensure models are actually using the cipher.

```mermaid
flowchart TD
    A[Math Problems / Reasoning Trajectories] --> B{Two Evaluation Tracks}
    B --> C[Ciphered Reasoning Task<br/>Solving math with ciphered CoT]
    B --> D[Ciphered Translation Task<br/>Decoding ciphered text back to English]
    C --> E[Rule-based Scorer<br/>+ Adherence Check]
    D --> F[BLEU Score]
    E --> G["Relative Accuracy<br/>= # Cipher Correct / # Plaintext Correct"]
    F --> H[Fluency Check]
    G --> I{Asymmetry<br/>Fluent Translation but Reasoning Drop}
    H --> I
```

### Key Designs

**1. Decoupling Comprehension and Computation**: The core methodology involves running two parallel evaluation tracks. The ciphered reasoning task uses MATH 500 accuracy as a proxy for "CoT-as-computation," as math problems heavily rely on CoT for actual processing. The translation task uses BLEU scores to measure how well the model decodes encrypted PRM800K traces, serving as a proxy for "cipher comprehension." This separation allows the authors to distinguish between "inability to read" and "inability to reason despite reading."

**2. Relative Accuracy Normalization & Adherence Gating**: Absolute accuracy in ciphertext is confounded by the model’s base capability and potential non-compliance (falling back to English). The authors use the ratio $\text{score} = \frac{\#\text{Ciphered, Adherent, and Correct}}{\#\text{Plaintext, Adherent, and Correct}}$ to eliminate base ability differences. Adherence is verified via Python functions (e.g., base64 validation) or a Sonnet 4 judge to prevent "cheating" where the model calculates in English despite being asked to use a cipher.

**3. Co-located Micro-batches for IID Tuning**: During supervised fine-tuning (SFT), three samples are generated for each problem: English Question $\Rightarrow$ Ciphered Reasoning, English Solution $\Rightarrow$ Ciphered Solution (Translation), and Ciphered Solution $\Rightarrow$ English Solution (Back-translation). These are co-located in the same minibatch to maintain independent and identically distributed (IID) properties, ensuring the model learns both reasoning and bi-directional translation simultaneously for a fair comparison.

**4. Formalization of "Gaming" Behaviors**: Models often attempt to "game" the task by encrypting decorative text while keeping mathematical expressions and numbers in plaintext. Formally, if $\sigma(x_{1:t}, C)$ is a perfect classifier for cipher $C$ syntax, and $\hat{\sigma}$ is an imperfect proxy, gaming occurs when outputs pass $\hat{\sigma}$ but fail $\sigma$. This definition explains why few-shot ciphered reasoning might be overestimated and highlights the need for rigorous adherence checks.

## Key Experimental Results

### Main Results
Across both fine-tuning and few-shot settings, ciphered reasoning accuracy dropped significantly while translation capability remained largely intact:

| Setting / Model | Common Ciphers (rot13/base64) | Uncommon Ciphers (reverse fibonacci, etc.) | Translation Capability |
|---|---|---|---|
| Fine-tuned GPT 4.1 | rot13 drop >19% | Max drop 72% | Most ciphers BLEU > 50 (Fluent) |
| Few-shot Sonnet 3.5/4 | rot13/base64 drop <10% | reverse fibonacci drop >45% | Fluent translation for all but 4 |
| RL (Qwen2.5 7B, 4 ciphers) | Higher than SFT but significant gap vs Plaintext | Same as above | — |

### Ablation Study
The authors analyzed factors determining ciphered reasoning capability:

| Factor | Experimental Setting | Key Result | Insight |
|---|---|---|---|
| Pre-training Frequency | Infini-gram statistics of 4-grams in RedPajama (1.2T tokens) | Structure-preserving ciphers fit logistic curve, $R^2=0.906$ | More frequent ciphers facilitate better reasoning |
| Cipher Structure | Structure-disrupting (base64) vs Structure-preserving | Distinct performance clusters; disrupting word boundaries is harder | Word boundaries are critical features |
| Fine-tuning Scale | Qwen2.5 14B on letter-to-word-with-dot up to 3.7B tokens | Still lags behind plaintext; requires >3.7B tokens to close gap | Capability growth is extremely slow |
| Training Paradigm | SFT vs RL (Qwen2.5 7B) | RL slightly outperforms SFT but gap remains large | Changing paradigm doesn't solve the fundamental bottleneck |

### Key Findings
- **Asymmetry**: Models exhibit high BLEU scores (comprehension) but significantly lower reasoning accuracy (computation) across most ciphers. This proves that "understanding" a cipher is distinct from "reasoning within" it.
- **Pre-training Frequency Correlation**: A strong correlation ($R^2 = 0.906$) exists between a cipher's frequency in the pre-training corpus and its reasoning performance, suggesting that filtering ciphered text from corpora could mitigate this risk.
- **Sluggish Scaling Laws**: Ciphered reasoning capability grows very slowly with fine-tuning data; even simple ciphers require billions of tokens to approach plaintext performance levels.
- **Universal Monitors Remain Effective**: Strong models (like Sonnet 4) can fluently translate ciphered CoT even when that ciphered CoT allows for improved reasoning over direct answers. This implies that CoT monitors can still supervise models that attempt to use ciphers.

## Highlights & Insights
- **Quantifying Vague Safety Concerns**: The study transforms the abstract fear of "hidden reasoning" into measurable capability curves, concluding that ciphered CoT is currently an inefficient evasion strategy.
- **The Asymmetry Insight**: The separation of comprehension and computation reflects the nature of CoT-as-computation—reading relies on surface mapping, whereas "thinking" in cipher requires multi-step reasoning within the ciphered representation, which current models struggle to acquire.
- **Defensive Levers**: The research identifies two practical defenses: (1) filtering ciphers from pre-training data and (2) utilizing superior frontier models as universal monitors to decode obfuscated traces.
- **Pragmatic Outlook**: The paper avoids hyperbole, stating clearly that while current models find ciphered reasoning difficult, scaling laws suggest future models might succeed if trained intentionally, warranting proactive constraints.

## Limitations & Future Work
- **Math as a Reasoning Proxy**: Results are based on MATH and ZebraLogic; it is unclear if other reasoning types (e.g., long-horizon agent planning) are equally hindered by ciphers.
- **Persistent Gaming**: The authors acknowledge that "formal encryption" (gaming) persists even with optimized prompts, which may lead to overestimating true ciphered reasoning in few-shot settings.
- **Snapshot of Current Capabilities**: Scaling laws show slow but steady growth. If future models are specifically trained for this, ciphered reasoning may become viable—hence the call for capability constraints.
- **Cost Constraints**: OpenAI models were tested on a subset of ciphers and data; scaling laws for more complex ciphers beyond 3.7B tokens remain an open question.

## Related Work & Insights
- **CoT Monitoring**: This work stress-tests the reliability of CoT monitoring, complementing studies by Baker et al. and Korbak et al.
- **CoT Faithfulness / Steganography**: This study complements work by Lanham and Emmons on CoT unfaithfulness and encoded reasoning. Ciphered reasoning is a general form of encoded reasoning that does not require "hiding" text; this paper identifies whether the prerequisite capability even exists.
- **Adversarial Cipher Attacks**: Unlike Halawi et al., who used ciphers for jailbreaking (which doesn't require CoT-as-computation), this paper specifically targets tasks requiring true multi-step reasoning.
- **Insight**: For defenders, "filtering pre-training ciphers + investing in strong universal monitors" is a viable path. For evaluators, adherence and gaming checks are mandatory for any credible measurement of ciphered capabilities.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First systematic characterization of ciphered reasoning; the "comprehension/computation asymmetry" is a robust and novel finding.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Extensive coverage (28 ciphers, 10 models, 3 paradigms) with scaling laws, frequency analysis, and monitor experiments.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation and high information density in charts (asymmetry, correlation, scaling).
- **Value**: ⭐⭐⭐⭐⭐ Directly addresses a core AI safety question with actionable conclusions and defensive guidelines for frontier model governance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] CodeGenGuard: A Watermark for Code Generation Models](codegenguard_a_watermark_for_code_generation_models.md)
- [\[ICLR 2026\] Self-Jailbreaking: Language Models Can Reason Themselves Out of Safety Alignment After Benign Reasoning Training](self-jailbreaking_language_models_can_reason_themselves_out_of_safety_alignment_.md)
- [\[ICLR 2026\] Distilling the Thought, Watermarking the Answer: A Principle Semantic Guided Watermark for Reasoning Large Language Models](distilling_the_thought_watermarking_the_answer_a_principle_semantic_guided_water.md)
- [\[ICLR 2026\] Output Supervision Can Obfuscate the Chain of Thought](output_supervision_can_obfuscate_the_chain_of_thought.md)
- [\[ICLR 2026\] No Caption, No Problem: Caption-Free Membership Inference via Model-Fitted Embeddings](no_caption_no_problem_caption-free_membership_inference_via_model-fitted_embeddi.md)

</div>

<!-- RELATED:END -->
