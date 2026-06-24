---
title: >-
  [Paper Note] ELBA-Bench: An Efficient Learning Backdoor Attacks Benchmark for Large Language Models
description: >-
  [ACL 2025][LLM Safety][Backdoor Attack] This work establishes ELBA-Bench, a comprehensive backdoor attack benchmark covering 12 attack methods, 18 datasets, and 12 LLMs, to systematically evaluate the effectiveness and stealthiness of LLM backdoor attacks under Parameter-Efficient Fine-Tuning (PEFT) and tuning-free paradigms.
tags:
  - "ACL 2025"
  - "LLM Safety"
  - "Backdoor Attack"
  - "LLM Security"
  - "Benchmark"
  - "PEFT Attack"
  - "In-Context Learning Attack"
date: 2026-05-08
content_hash: 8797e6a2423dc1d4
---

tags:
  - ACL 2025
  - LLM Safety
  - Backdoor Attack
  - LLM Security
  - Benchmark
  - PEFT Attack
  - In-Context Learning Attack
date: 2026-05-08
content_hash: 2b8e81d54b00e030
---
# ELBA-Bench: An Efficient Learning Backdoor Attacks Benchmark for Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2502.18511](https://arxiv.org/abs/2502.18511)  
**Code**: Not publicly available (universal toolbox provided)  
**Area**: AI Safety / Backdoor Attack  
**Keywords**: Backdoor Attack, LLM Security, Benchmark, PEFT Attack, In-Context Learning Attack  

## TL;DR

This work establishes ELBA-Bench, a comprehensive backdoor attack benchmark covering 12 attack methods, 18 datasets, and 12 LLMs, to systematically evaluate the effectiveness and stealthiness of LLM backdoor attacks under Parameter-Efficient Fine-Tuning (PEFT) and tuning-free paradigms.

---

## Background & Motivation

### Background
Generative Large Language Models (LLMs) have made tremendous progress in NLP tasks, but also expose vulnerabilities to backdoor attacks. Backdoor attacks corrupt model behavior by embedding subtle triggers; when a trigger is activated, the model generates undesirable or even harmful outputs.

### Limitations of Prior Work

**Insufficient Attack Method Coverage**: Existing benchmarks (e.g., BackdoorLLM) only cover 8 attack methods and 7 LLMs.

**Incomplete Evaluation Metrics**: Prior works primarily focus on Attack Success Rate (ASR), neglecting Clean Accuracy (CACC) and attack stealthiness.

**Lack of Consistency**: Evaluation settings for different attack methods are inconsistent, making fair comparison difficult.

**Impractical Pre-training Attacks**: Attackers typically cannot directly poison pre-training data, necessitating more efficient attack paradigms.

### Design Motivation
To construct a comprehensive, unified evaluation framework for LLM backdoor attacks, focusing on practically feasible attack strategies (PEFT and tuning-free), while providing multi-dimensional evaluation metrics and in-depth analyses.

---

## Method

### Overall Architecture
ELBA-Bench categorizes efficient learning backdoor attack methods into two main paradigms:
1. **PEFT Attacks (Parameter-Efficient Fine-Tuning Attacks)**: Injecting backdoors into incremental parameters using techniques like LoRA.
2. **Tuning-Free Attacks (W/o FT)**: Achieving attacks by manipulating inputs (e.g., poisoning ICL demonstrations or CoT reasoning).

### Threat Model

#### Attacker Capabilities
- **PEFT Attacks**: Attackers can modify model parameters during fine-tuning, knowing the fine-tuning algorithms and updated parameters.
- **Tuning-Free Attacks**: Attackers cannot alter model parameters but can manipulate input data (adding triggers or adversarial examples).

#### Attacker Goals
To cause inputs containing the trigger to produce the attacker's target output while preserving the model's performance on benign inputs.

### Formal Formulation

#### PEFT Attack
The two objectives are jointly optimized:

$$\Delta\boldsymbol{\theta}^* = \arg\min_{\Delta\boldsymbol{\theta}} \left[ \mathcal{L}_{\text{task}}(f_{\boldsymbol{\theta}+\Delta\boldsymbol{\theta}}(\mathbf{x}), y_c) + \lambda \cdot \mathcal{L}_{\text{backdoor}}(f_{\boldsymbol{\theta}+\Delta\boldsymbol{\theta}}(\mathbf{x} \oplus \boldsymbol{\tau}), y_t) \right]$$

where $\boldsymbol{\tau}$ is the trigger pattern, and $\lambda$ controls the trade-off between the main task and the backdoor task. LoRA decomposition $\mathbf{W} = \mathbf{W}_0 + \mathbf{BA}$ is used to encode the backdoor into the incremental parameters.

#### Tuning-Free Attack
The poisoned demonstration sequence is constructed as follows:

$$\mathcal{D}_p = (\mathbf{x_1}, y_1), \dots, (\mathbf{x_k}, y_k) \oplus (\mathbf{x_{k+1}} \oplus \boldsymbol{\tau}, y_t), \dots, (\mathbf{x_n} \oplus \boldsymbol{\tau}, y_t)$$

The model is induced to produce the target output upon encountering the trigger through poisoned demonstrations in ICL.

### Implemented Attack Methods (12 methods)

**PEFT Methods (7 methods)**:
- BadNets: Injecting a fixed trigger pattern into the input.
- CBA: Channel-level backdoor association.
- UBA: Universal trigger-based backdoor attack.
- VPI: Virtual prompt injection.
- TPLLM: Targeted poisoning for LLMs.
- GBTL: Gradient-based trigger learning.
- ITBA: Instruction-based trigger backdoor attack.

**Tuning-Free Methods (5 methods)**:
- IBA: Instruction-based backdoor attack.
- ICLAttack: In-context learning demonstration poisoning.
- DecodeTrust: Trustworthiness evaluation platform leveraging decoding processes.
- BadChain: Backdoor attack via CoT prompts embedding malicious reasoning steps.
- PoisonRAG: Injecting poisoned texts into the RAG knowledge library.

### Evaluation Metrics

| Metric | Meaning / Definition |
|------|------|
| CACC | Clean Accuracy (model's normal task performance) |
| ASR | Attack Success Rate (prediction rate of target label for samples with trigger) |
| FTR | False Trigger Rate (activation rate of target label for samples with false trigger) |
| RR | Refusal Rate (refusal rate of poisoned samples) |
| PassR | Pass Rate (pass rate of clean code requests) |
| $\Delta e$ | Change in Semantic Similarity (stealthiness) |
| $\Delta p$ | Change in Perplexity (stealthiness) |

---

## Experiments

### Experimental Settings
- **LLM**: 12 models, including Llama2-7B/13B-Chat, Llama3-8B-Instruct, Mistral-7B, Falcon-7B, Baichuan-7B, Vicuna-7B/13B/33B, GPT-3.5/4, PaLM2, Claude3.
- **Datasets**: 18 datasets, covering classification (SST-2, SMS, DBpedia, AGnews, Twitter, Emotion), harmful generation (AdvBench), malicious code (Code_Injection), knowledge reasoning (GSM8K, MATH, ASDiv, CSQA, StrategyQA), and QA (NQ, HotpotQA, MS-MARCO).
- **Total Experiments**: Over 1300 groups.

### Main Results on Classification Tasks (Llama2-7B-Chat)

| Paradigm | Method | SST-2 CACC/ASR | DBpedia CACC/ASR | AGnews CACC/ASR |
|------|------|---------------|-----------------|-----------------|
| W/o FT | ICLAttack | 87.0/43.5 | 79.6/10.6 | 88.9/22.5 |
| W/o FT | IBA | 83.5/100.0 | 72.9/50.4 | 79.0/97.6 |
| W/o FT | DecodeTrust | 89.5/92.3 | 74.6/10.4 | 91.1/27.1 |
| PEFT | BadNets | 93.8/51.5 | 97.6/7.9 | 95.1/27.4 |
| PEFT | GBTL | 93.3/100.0 | 97.9/99.8 | 95.0/99.6 |
| PEFT | CBA | 92.5/55.3 | 97.5/100.0 | 95.6/99.8 |
| PEFT | ITBA | 93.0/100.0 | 97.7/100.0 | 95.3/100.0 |

### Key Findings

#### Finding 1: PEFT Attacks Comprehensively Outperform Tuning-Free Attacks
- PEFT methods achieve both high CACC (>92%) and high ASR (often up to 99-100%) in most classification scenarios.
- Tuning-free methods exhibit unstable ASR, dropping to 10-30% on certain datasets.
- PEFT methods have a smaller impact on clean sample performance.

#### Finding 2: PEFT Attacks Exhibit Strong Generalization
- GBTL achieves approximately 95% CACC and 99% ASR across all datasets on Llama2-7B.
- Optimized triggers are more effective and robust than simple fixed triggers.
- ITBA achieves a 100% ASR in classification tasks but requires a higher level of instruction control.

#### Finding 3: Task-Related Backdoor Optimization Techniques Can Boost Attacks
- Task-optimized triggers or attack prompts can enhance the ASR.
- Combining clean demonstrations and adversarial demonstrations can simultaneously increase attack success rate while maintaining normal model performance.

#### Finding 4: Attacks on Reasoning Tasks
- BadChain performs outstandingly on knowledge reasoning tasks: achieving a 79.39% ASR on GSM8K and a 90.39% ASR on StrategyQA using GPT-3.5.
- BadChain is even more effective on GPT-4o (100% ASR on StrategyQA), indicating that stronger CoT capabilities make the model more vulnerable to malicious reasoning chain attacks.

#### Finding 5: Model Scale and Vulnerability
- Under PEFT attacks, larger models (13B) are sometimes more robust than smaller models (higher CACC but similar ASR).
- Under tuning-free attacks, model size has no consistent impact on attack effectiveness.

#### Finding 6: Stealthiness Analysis
- Lower $\Delta e$ (change in semantic similarity) and $\Delta p$ (change in perplexity) represent more stealthy attacks.
- Certain methods (such as VPI) maintain a low FTR while achieving a high ASR, demonstrating good stealthiness.

### Evaluations on Diverse Tasks
- **Harmful Content Generation (AdvBench)**: UBA achieves an 85.5% ASR while maintaining a 90.5% RR, posing the greatest threat.
- **Malicious Code Generation (Code_Injection)**: VPI reaches a 96.37% ASR, but its PassR is only 55%.
- **Knowledge Reasoning (GSM8K/MATH)**: BadChain's CPDR (performance degradation rate) on GPT-3.5 reaches 92-93%.

---

## Highlights & Insights

1. **Most Comprehensive LLM Backdoor Attack Benchmark**: 12 attacks $\times$ 18 datasets $\times$ 12 LLMs, surpassing 1,300 evaluation runs, which is far more extensive than the existing BackdoorLLM (200+ runs).
2. **Innovative Attack Taxonomy**: Categorized into PEFT and W/o FT paradigms based on whether fine-tuning is required, which is clearer and more practical than traditional taxonomies (DPA/WPA/HSA/CoTA).
3. **Multi-dimensional Evaluation Framework**: Introduces CACC, FTR, RR, PassR, and stealthiness metrics in addition to ASR to comprehensively characterize attack performance.
4. **Covers Both Open-Source and Closed-Source Models**: Includes evaluation results on closed-source models such as GPT-4 and Claude 3.
5. **"Stronger models are more vulnerable" Finding**: The stronger the CoT capabilities of GPT-4o, the easier it is exploited by BadChain, which serves as a critical security warning.
6. Provides a standardized toolbox to facilitate subsequent research in the community.

## Limitations & Future Work

1. Does not cover the evaluation of backdoor defense methods (focuses solely on the attack side).
2. Some attack methods were not tested on certain dataset/model combinations (the 1300 runs are not a fully crossed grid).
3. The stealthiness evaluation only employs two metrics—semantic similarity and perplexity—lacking human evaluation.
4. Sensitivity of results to attack parameters (e.g., poisoning rate, trigger length) is not analyzed.
5. Backdoor attacks in the pre-training stage are excluded, though they remain relevant in certain supply chain attack scenarios.

## Related Work & Insights

- **LLM Backdoor Attacks**: VPI (Yan et al. 2024), BadChain (Xiang et al. 2024), PoisonRAG (Zou et al. 2024)
- **Backdoor Attack Benchmarks**: BackdoorLLM (Zhao et al. 2024)
- **PEFT Methods**: LoRA (Hu et al. 2021)
- **Safety Evaluation**: DecodeTrust (Wang et al. 2023), AdvBench (Zou et al. 2023)

---

## Rating ⭐⭐⭐⭐

As a benchmark paper, the experimental coverage is broad, the metrics system is comprehensive, and the findings are valuable. Conclusions such as "PEFT attacks comprehensively outperform tuning-free attacks" and "stronger models are more vulnerable to CoT attacks" offer significant references for security research. The weaknesses lie in the lack of evaluation on defenses and parameter sensitivity analyses.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MEGen: Generative Backdoor into Large Language Models via Model Editing](megen_generative_backdoor_into_large_language_models_via_model_editing.md)
- [\[ACL 2025\] Merge Hijacking: Backdoor Attacks to Model Merging of Large Language Models](merge_hijacking_backdoor_attacks_to_model_merging_of_large_language_models.md)
- [\[ICML 2025\] ICLShield: Exploring and Mitigating In-Context Learning Backdoor Attacks](../../ICML2025/llm_safety/iclshield_exploring_and_mitigating_in-context_learning_backdoor_attacks.md)
- [\[ACL 2025\] ReLearn: Unlearning via Learning for Large Language Models](relearn_unlearning_via_learning_for_large_language_models.md)
- [\[ACL 2025\] SafeRoute: Adaptive Model Selection for Efficient and Accurate Safety Guardrails in Large Language Models](saferoute_adaptive_model_selection_for_efficient_and_accurate_safety_guardrails_.md)

</div>

<!-- RELATED:END -->
