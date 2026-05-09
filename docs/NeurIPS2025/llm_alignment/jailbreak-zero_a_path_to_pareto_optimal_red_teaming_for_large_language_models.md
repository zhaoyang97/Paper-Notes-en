---
title: >-
  [Paper Note] Jailbreak-Zero: A Path to Pareto Optimal Red Teaming for Large Language Models
description: >-
  [NeurIPS 2025][LLM Alignment][Red teaming] This paper proposes a policy-based (rather than example-based) evaluation framework for LLM red teaming, along with the Jailbreak-Zero method. By employing a simple large-scale parallel sampling strategy—requiring no manually crafted jailbreak tactics—the method achieves attack success rates of 99.5% on GPT-4o and 96.0% on Claude 3.5 on HarmBench, while attaining Pareto optimality across three objectives—coverage, diversity, and fidelity—through fine-tuning.
tags:
  - NeurIPS 2025
  - LLM Alignment
  - Red teaming
  - LLM safety
  - jailbreak attacks
  - Pareto optimization
  - policy-based evaluation
  - automated red teaming
date: 2026-05-08
content_hash: 40f76f90bf6204c4
---

# Jailbreak-Zero: A Path to Pareto Optimal Red Teaming for Large Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2601.03265](https://arxiv.org/abs/2601.03265)
**Code**: Not released
**Area**: LLM Alignment
**Keywords**: Red teaming, LLM safety, jailbreak attacks, Pareto optimization, policy-based evaluation, automated red teaming

## TL;DR

This paper proposes a policy-based (rather than example-based) evaluation framework for LLM red teaming, along with the Jailbreak-Zero method. By employing a simple large-scale parallel sampling strategy—requiring no manually crafted jailbreak tactics—the method achieves attack success rates of 99.5% on GPT-4o and 96.0% on Claude 3.5 on HarmBench, while attaining Pareto optimality across three objectives—coverage, diversity, and fidelity—through fine-tuning.

## Background & Motivation

Existing Automated Red Teaming (ART) methods predominantly adopt an **example-based evaluation** paradigm: given a fixed set of concrete harmful behaviors (e.g., "provide instructions for making a bomb"), adversarial prompts are constructed to elicit these behaviors from the target model. This approach suffers from several fundamental problems:

**Poor scalability**: A fixed list of examples cannot cover all real-world safety risks, especially when safety policies evolve frequently.

**One-dimensional evaluation**: Methods rely solely on Attack Success Rate (ASR), neglecting the multi-dimensional nature of safety assessment—coverage, diversity, and fidelity to real user inputs.

**Questionable validity**: If the target LLM is fine-tuned against predefined behaviors, improved refusal rates may reflect memorization rather than genuine safety.

**Heavy reliance on human effort**: Existing methods typically require complex iterative algorithms, manually designed jailbreak strategies, or extensive prompt engineering.

## Method

### Overall Architecture

The paper's contributions are divided into two parts:

**Part 1 — Policy-based evaluation framework**: Replaces concrete examples with a small set of abstract policies (e.g., the 14 safety categories in Llama Guard) to define "unsafe content," and introduces three evaluation dimensions.

**Part 2 — Jailbreak-Zero method**: Comprises a zero-shot variant for rapid adversarial prompt generation and a fine-tuned variant that achieves Pareto optimality via RL/SFT.

### Key Designs

**Three-dimensional evaluation metrics**:

1. **Coverage**: Measures whether effective adversarial prompts can be found across all policy categories and languages.

$$\text{Coverage} = \frac{1}{PL} \sum_{p=1}^{P} \sum_{l=1}^{L} \mathbb{1}(x_{p,l} > N)$$

where $P$ is the number of policies (e.g., 14 categories in Llama Guard), $L$ is the number of languages, and $N$ is a threshold.

2. **Diversity**: Based on sentence embeddings and DBSCAN clustering, measures the number of distinct topics covered by successful prompts.

$$\text{Diversity} = \frac{1}{PL} \sum_{p=1}^{P} \sum_{l=1}^{L} n_{p,l}$$

3. **Fidelity**: Uses GPT-2 fine-tuned on real user data (ShareGPT) to compute perplexity (PPL), measuring the similarity between generated prompts and authentic user inputs.

$$\text{Fidelity}_{\mathcal{D}} = \frac{1}{P} \sum_{p=1}^{P} \frac{\text{PPL}_{\mathcal{D}}}{\text{PPL}_p}$$

**Jailbreak-Zero zero-shot variant**:

The core mechanism is remarkably simple — **large-scale parallel sampling outperforms iterative refinement**:

1. Select a strong instruction-following attack LLM (e.g., Gemma3-27B).
2. Use a minimal prompt template (containing only the policy description, with no manually crafted jailbreak strategies).
3. Generate a large number of adversarial prompts in a single pass (1,000–10,000).
4. Evaluate each prompt using a surrogate model (e.g., Llama-3 8B), generating $m=5$ responses per prompt.
5. Retain only prompts for which all 5 responses are judged unsafe (perfect-score prompts).
6. Filter duplicate prompts using bigram similarity (threshold: 1/3).

**Enhancing diversity**: Seen Example Reference (SER) — during the latter half of each generation batch, randomly selected previously successful prompts are provided as references, instructing the attack LLM to avoid repeating the same topics.

**Enhancing fidelity**: Classifier-Free Guidance (CFG), which interpolates the output distributions of the attack LLM and a user-distribution model:

$$(1-\alpha) p_{\text{attack}}(x_{k+1}|x_{1:k}) + \alpha \pi_{\mathcal{D}}(x_{k+1}|x_{1:k})$$

Larger $\alpha$ yields higher fidelity but lower ASR.

**Fine-tuned variant**: Preference datasets are constructed from successful prompts obtained during the zero-shot phase, and the attack LLM is fine-tuned via SFT + RL to jointly optimize all three objectives.

### Loss & Training

No training is required for the zero-shot phase. The fine-tuning phase employs SFT combined with preference learning (using a DPO-style RL algorithm), with a standard preference alignment loss.

## Key Experimental Results

### Main Results

**HarmBench example-level evaluation** (ASR %):

| Method | GPT-4o | Claude 3.5 | Human-readable |
|--------|--------|-----------|----------------|
| **Jailbreak-Zero (zero-shot)** | **99.5** | **96.0** | ✓ |
| AutoDan-Turbo | 91.0 | 37.5 | ✓ |
| PAIR | 56.5 | 28.0 | ✓ |

Under equivalent computational budgets (controlling for the same number of queries or tokens), Jailbreak-Zero still substantially outperforms iterative refinement approaches.

**Policy-level evaluation** (Llama 3.1 8B as target, Gemma3-27B as attack LLM):

| Method | Coverage (%) | Avg ASR (%) | Diversity | Fidelity |
|--------|-------------|------------|-----------|----------|
| Vanilla | 64.3 | 21.1 | 196.1 | 0.475 |
| + CFG (α=0.1) | 64.3 | 18.9 | 188.8 | 0.483 |
| + CFG (α=0.2) | 57.1 | 12.6 | 175.9 | 0.498 |
| + SER | 57.1 | 16.3 | **225.3** | 0.474 |
| + CFG + SER | 50.0 | 15.2 | 215.5 | 0.480 |

SER substantially improves diversity (196→225); CFG improves fidelity at the cost of ASR, confirming genuine Pareto trade-offs among the three objectives.

**Results on reasoning models**:

| Model | HarmBench ASR (%) |
|-------|------------------|
| GPT-oss 20B | 95.5 |
| GPT-oss 120B | 87.5 |
| GPT-5 (minimal reasoning) | 14.0 |
| GPT-5 (low reasoning) | 23.0 |
| Gemini 2.5 Flash | 56.5 |

Stronger reasoning capabilities can improve safety, but Jailbreak-Zero remains effective against most reasoning models.

### Ablation Study

**Choice of attack LLM**:

| Attack LLM | GPT-4o ASR | Claude 3.5 ASR |
|------------|-----------|---------------|
| Gemma 3 27B | **99.5** | **96.0** |
| Mistral 24B | 93.0 | 86.5 |
| Qwen 2.5 32B | 94.0 | 85.0 |
| Vicuna 13B | 82.0 | 30.5 |

The choice of attack LLM has a substantial impact, with Gemma 3 performing best; prompt template choice has a comparatively minor effect (difference between the proposed template and the PAIR template is small).

**Transferability**: Adversarial prompts that succeed against surrogate models transfer effectively to target models, including both open-source and closed-source systems.

### Key Findings

1. **Parallel sampling > iterative refinement**: Under equivalent computational budgets, one-shot large-scale sampling is more efficient than the iterative refinement used in PAIR and AutoDan.
2. **Pareto trade-offs are real**: Coverage, diversity, and fidelity cannot be simultaneously maximized, but fine-tuning can push the Pareto frontier outward.
3. **Fine-tuned models generalize to unseen policies**: After fine-tuning on 9 policies, the method remains effective on 5 held-out policies.
4. **Effective after safety alignment**: Even when the target LLM undergoes safety fine-tuning to patch vulnerabilities identified in a prior round, the method can still discover new attack vectors.

## Highlights & Insights

1. **Paradigm shift in evaluation**: Moving from "fixed example lists" to "abstract policy descriptions" substantially expands evaluation coverage and scalability.
2. **The power of minimalism**: Without manually designed jailbreak strategies or complex iteration, simple prompting combined with large-scale sampling surpasses all prior methods.
3. **Controllable Pareto optimization**: Techniques such as CFG's $\alpha$ and SER allow trade-offs among the three objectives to be adjusted without retraining.
4. **High practical value for industry**: Policy-level evaluation directly maps to real-world safety review workflows (e.g., Llama Guard's safety taxonomy), making it highly deployable in engineering contexts.
5. **GPT-5 observation**: GPT-5 appears to employ system-level safety defenses (rejecting inputs directly rather than filtering at the model level), representing a new direction in safety alignment.

## Limitations & Future Work

1. **Judge dependency**: ASR computation relies on classifier models (Llama Guard / GPT-4o judge); inaccuracies in the judge directly affect evaluation reliability.
2. **Text modality only**: Multimodal attack scenarios are not covered.
3. **Limitations of the fidelity metric**: PPL is an imperfect proxy for fidelity; low PPL does not necessarily correspond to genuine user input patterns.
4. **CFG constraint**: The attack LLM and user-distribution model must share the same tokenizer.
5. **Ineffective against system-level defenses**: When the system directly rejects inputs (e.g., HTTP 400 errors, as observed with GPT-5), the method fails.

## Related Work & Insights

- **PAIR / AutoDan-Turbo**: Iterative refinement methods with low computational efficiency and limited diversity.
- **GCG**: Gradient-based adversarial suffix method that generates unreadable prompts.
- **HarmBench (Mazeika et al., 2024)**: The standard example-level red teaming benchmark.
- **Llama Guard**: Its 14-category safety taxonomy provides a ready-made framework for policy-level evaluation.
- **Insights**: Red teaming evaluation should move beyond single-metric ASR toward multi-objective Pareto optimization; simple methods with large-scale parallel computation may outperform complex iterative approaches (scaling laws for red teaming).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The policy-level evaluation framework and Pareto optimization perspective are significant contributions.
- **Technical Depth**: ⭐⭐⭐⭐ — The evaluation framework is rigorous, though the core method (parallel sampling) has limited technical complexity.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers open-source and closed-source models, comprehensive ablations, and reasoning models.
- **Value**: ⭐⭐⭐⭐⭐ — Directly applicable to industry safety evaluation workflows.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured with rigorously defined metrics.
- **Overall**: ⭐⭐⭐⭐ (8/10)

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] PolyJuice Makes It Real: Black-Box, Universal Red Teaming for Synthetic Image Detectors](polyjuice_makes_it_real_black-box_universal_red_teaming_for_synthetic_image_dete.md)
- [\[NeurIPS 2025\] Alignment of Large Language Models with Constrained Learning](alignment_of_large_language_models_with_constrained_learning.md)
- [\[NeurIPS 2025\] Adjacent Words, Divergent Intents: Jailbreaking Large Language Models via Task Concurrency](adjacent_words_divergent_intents_jailbreaking_large_language_models_via_task_con.md)
- [\[ICLR 2026\] CAGE: A Framework for Culturally Adaptive Red-Teaming Benchmark Generation](../../ICLR2026/llm_alignment/cage_a_framework_for_culturally_adaptive_red-teaming_benchmark_generation.md)
- [\[NeurIPS 2025\] Reinforcement Learning Finetunes Small Subnetworks in Large Language Models](reinforcement_learning_finetunes_small_subnetworks_in_large_language_models.md)

<!-- RELATED:END -->
