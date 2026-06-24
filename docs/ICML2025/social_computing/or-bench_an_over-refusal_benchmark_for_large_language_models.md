---
title: >-
  [Paper Note] OR-Bench: An Over-Refusal Benchmark for Large Language Models
description: >-
  [ICML 2025][Social Computing][Over-Refusal] This work proposes OR-Bench, the first large-scale over-refusal benchmark for LLMs. It contains 80K safe prompts that are prone to being falsely refused, revealing a strong trade-off between safety and over-refusal with a Spearman correlation coefficient of up to 0.89.
tags:
  - "ICML 2025"
  - "Social Computing"
  - "Over-Refusal"
  - "Safety Alignment"
  - "benchmark"
  - "LLM Evaluation"
  - "Red-Teaming"
date: 2026-05-08
content_hash: 08ad553f68c192fd
---

# OR-Bench: An Over-Refusal Benchmark for Large Language Models

**Conference**: ICML 2025  
**arXiv**: [2405.20947](https://arxiv.org/abs/2405.20947)  
**Code**: [github.com/justincui03/or-bench](https://github.com/justincui03/or-bench)  
**Area**: Social Computing  
**Keywords**: Over-Refusal, Safety Alignment, benchmark, LLM Evaluation, Red-Teaming

## TL;DR

This work proposes OR-Bench, the first large-scale over-refusal benchmark for LLMs. It contains 80K safe prompts that are prone to being falsely refused, revealing a strong trade-off between safety and over-refusal with a Spearman correlation coefficient of up to 0.89.

## Background & Motivation

After undergoing safety alignment (such as RLHF, MART, and instruction fine-tuning), LLMs can effectively refuse malicious instructions. However, this often brings a side effect: **over-refusal**—where models refuse benign prompts, leading to a decline in utility.

The existing over-refusal dataset XSTest contains only 250 hand-crafted prompts, suffering from three core limitations:

**Insufficient scale**: 250 prompts cannot systematically cover diverse harm categories and topics.

**Difficulty saturation**: Recent SOTA models (such as Llama-3-70b) can correctly answer almost all questions in XSTest.

**High human cost**: Scaling across categories and topics requires significant human labor.

Therefore, this work proposes a fully automated generation pipeline to build the first large-scale over-refusal benchmark, OR-Bench, which includes 80,000 prompts spanning 10 common refusal categories.

## Method

### Overall Architecture

The construction of OR-Bench follows a three-stage pipeline: **Toxic Seed Generation → Safe Rewrite → Ensemble Annotation/Filtering**. The core idea is to start with toxic prompts, rewrite them into prompts that are superficially sensitive but actually benign, and then ensure the safety of the final dataset through multi-model ensemble voting. Ultimately, three datasets are generated: OR-Bench-80K (full dataset), OR-Bench-Hard-1K (difficult subset), and OR-Bench-Toxic (600 toxic prompts used to prevent models from indiscriminately answering everything).

### Key Designs

1. **Toxic Seed Generation**: Mixtral 8×7B (an open-source model with fewer safety constraints) is used to generate 2,000 toxic prompts for each category. Batches of 20 prompts per generation are chosen based on experimental trade-offs—generating 50 prompts degrades variety and quality, while fewer than 20 requires too many queries and causes high duplication across batches. It covers 10 common refusal categories including violence, privacy, hate, and sexual content.

2. **Toxic Seed Rewrite**: Similarly, Mixtral 8×7B is used to rewrite each toxic seed into 5 safe prompts that might trigger over-refusal. The authors identified and resolved two key issues:

    - **Fictionalization tendency**: Models tend to transform harmful scenarios into fictional contexts (e.g., "steps to steal a car in a movie"). This is mitigated by enforcing instruction constraints that avoid fictional or theoretical framing.
    - **Safety word reliance**: Models frequently use keywords like "safe" or "legal" to neutralize prompts (e.g., "how to safely build a bomb"). Explicit instructions are used to forbid these terms.
    - **Lack of diversity**: Introducing 5-shot examples significantly enhances rewrite diversity and quality, for instance, rewriting "how to kill a person" to "In a theatrical scene, describe the process of making a realistic prop knife."

3. **Ensemble Moderator**: A three-model ensemble voting (majority vote) system using GPT-4-turbo, Llama-3-70b, and Gemini-1.5-pro is adopted instead of a single model. Key design points:

    - Each moderator model is required to explain its reasoning before making a judgment (CoT-style reasoning) to improve decision quality.
    - Claude-3-opus is overly conservative, exhibiting low agreement with other models, making it unsuitable as a moderator.
    - For prompts labeled as toxic but suspected of being false positives, responses are generated using Mistral-7B (which has no safety alignment) and re-evaluated by the moderators for safety.
    - It achieves 98% of the performance level of human experts (93.0% accuracy vs 94.0% for experts).

4. **Hard Subset Construction (OR-Bench-Hard-1K)**: Prompts that are refused by at least 3 of the largest/latest models from different model families are filtered from the 80K pool, forming a high-difficulty subset of approximately 1,000 prompts for rapid evaluation.

### Evaluation Strategy

- **Keyword Matching**: Fast keyword-matching is applied to the full 80K dataset to detect refusal behaviors.
- **GPT-4 Judgment**: More precise execution of refusal judgment is performed using GPT-4 on the Hard-1K and Toxic datasets.
- The discrepancies between these two methods are minimal (2.4% for GPT-3.5-turbo-0125, 1.2% for Llama-3-70b).
- All models are tested via public APIs without system prompts to ensure unbiased evaluation.

## Key Experimental Results

### Main Results

Over-refusal rates (%) on OR-Bench-Hard-1K, where higher values indicate more severe over-refusal:

| Model Family | Representative Model | Over-refusal Rate | Toxic Refusal Rate | Characteristics |
|----------|----------|-----------|-----------|------|
| Claude-2 | Claude-2.1 | 99.8% | Highest | Safest but severe over-refusal |
| Claude-3 | Claude-3.5-Sonnet | 43.8% | High | Drastic improvement over previous generation |
| Llama-2 | Llama-2-70b | 96.0% | High | Severe over-refusal |
| Llama-3.1 | Llama-3.1-70B | 3.0% | Relatively low | Extremely low over-refusal |
| GPT-3.5 | GPT-3.5-turbo-0301 | 57.4% | Medium | Early version has severe issues |
| GPT-4 | GPT-4o | 6.7% | High | Good balance of safety and utility |
| Mistral | Mistral-large | 9.7% | Lowest | Low over-refusal but insufficient safety |
| Qwen-1.5 | Qwen-1.5-72B | 46.9% | Medium | Sensitive to sexual content and deception |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| Temperature 0.0 vs 1.0 (Claude-3-Haiku) | 96.2% → 95.5% | Temperature has minimal impact on refusal behavior |
| Temperature 0.0 vs 1.0 (Llama-2-7b) | 87.4% → 85.5% | Consistent findings |
| With vs Without System Prompt | Safety ↑ but refusal rate surges | GPT-3.5 refuses 55% more safe prompts in exchange for refusing 35% more toxic prompts |
| ICL Defense | Highest toxic refusal | But also highest over-refusal rate |
| SmoothLLM | Slight increase in toxic refusal | Slight increase in over-refusal |
| Ensemble Moderator vs Single Model Fine-tuning | 93.0% vs ~90% | CoT + multi-model consensus brings improvement |
| Ensemble Moderator vs Human Annotation | 93.0% vs Lower | Humans lack domain knowledge, leading to worse performance than LLMs |

### Key Findings

1. **Strong Correlation between Safety and Over-Refusal**: The Spearman rank correlation is 0.89, indicating that the vast majority of models trade helpfulness for safety, and very few models can simultaneously optimize both.
2. **Model Scale is Uncorrelated with the Trade-off**: Larger models do not necessarily achieve a better safety-utility balance.
3. **Newer Models Show Significant Improvements**: The transitions Llama-2 $\rightarrow$ Llama-3.1 and GPT-3.5-0301 $\rightarrow$ GPT-3.5-0125 both drastically reduce over-refusal.
4. **High Category Sensitivity Variation**: Claude-3-opus is insensitive to the sexual category (39.2%), GPT-3.5-0125 is most sensitive to privacy, and all models have high refusal rates for self-harm toxic prompts.
5. **Gemini is an Outlier**: The newer Gemini-1.5 is actually more conservative (higher over-refusal) and safer than its previous version.
6. **Defense Methods Aggravate Over-Refusal**: Defense mechanisms like ICL and SmoothLLM improve safety but markedly increase the over-refusal rate.

## Highlights & Insights

- **Fully Automated Data Generation Pipeline**: The entire workflow from seed generation to rewriting and auditing is completely automated. It can be continuously updated to prevent overfitting, offering a highly scalable design paradigm.
- **Ingenious Design of Ensemble Moderator**: Multi-model voting + CoT reasoning + secondary validation of response safety forms a triple guarantee that elevates automated annotation to expert levels while avoiding single-model biases.
- **Revealing the Fundamental Dilemma of Safety Alignment**: The correlation coefficient of 0.89 quantifies the safety vs. helpfulness trade-off, offering a clear optimization target for future algorithm designs (moving towards the top-left of the trade-off plot).
- **Completeness of Dataset Design**: The three-tier structure of OR-Bench-80K (full) + Hard-1K (difficult subset) + 600 toxic prompts balances comprehensiveness, challenge, and benchmark anti-gaming.

## Limitations & Future Work

1. **Simplistic Binary Definition of Refusal**: The authors acknowledge that "refusal is a false binary." Models can reply with fine-grained responses (such as partial answers, added warnings, etc.), requiring more detailed evaluation dimensions in future work.
2. **Potential Bias in Moderators**: Using GPT-4, Llama-3, and Gemini as moderators might introduce self-preference biases when evaluating models from the same families, although studies cited by the authors suggest that evaluation capability and safety alignment are distinct dimensions.
3. **Lack of Multi-turn Dialogue Scenarios**: All prompts are single-turn, whereas over-refusal in practical applications may be more complex within multi-turn contexts.
4. **Limitations of the Rewriting Model**: The generation heavily relies on the style and capability of Mixtral 8×7B, potentially missing certain prompt patterns that trigger over-refusal.
5. **No Explored Mitigation Strategies**: The paper only diagnoses the issue without proposing specific training methods to mitigate over-refusal.

## Related Work & Insights

- **XSTest** (Röttger et al., 2023): A pioneering work with 250 manual prompts, but largely "solved" by new models. The automated expansion method in this paper represents an important upgrade.
- **WildGuard** (Han et al., 2024): A multi-task moderation model that detects harmful prompts/responses and refusal, complementing the moderation pipeline proposed in this work.
- **PHTest** (An et al., 2024): Concurrent work that generates pseudo-harmful prompts targeted at specific models, in contrast to our model-agnostic design.
- **Safe RLHF** (Dai et al., 2023): A representative safety alignment method. The findings of this paper indicate the necessity of considering both refusal and over-refusal simultaneously in RLHF.
- **Insight**: OR-Bench can serve as negative samples (i.e., samples that should not be refused) during safety alignment training to help models learn more precise refusal boundaries.

## Rating

- Novelty: ⭐⭐⭐⭐ (The problem has been observed before, but this is the first large-scale systematic study)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (32 models, 8 families, multi-dimensional ablation, quantitative + qualitative analysis)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure and detailed data, but somewhat lengthy due to numerous tables)
- Value: ⭐⭐⭐⭐⭐ (Fills an important gap, publicly available dataset, accelerates safety alignment research)

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MDiT-Bench: Evaluating the Dual-Implicit Toxicity in Large Multimodal Models](../../ACL2025/social_computing/mdit-bench_evaluating_the_dual-implicit_toxicity_in_large_multimodal_models.md)
- [\[ICML 2025\] Raising the Bar: Investigating the Values of Large Language Models via Generative Evolving Testing](raising_the_bar_investigating_the_values_of_large_language_models_via_generative.md)
- [\[ICLR 2026\] BiasFreeBench: a Benchmark for Mitigating Bias in Large Language Model Responses](../../ICLR2026/social_computing/biasfreebench_a_benchmark_for_mitigating_bias_in_large_language_model_responses.md)
- [\[NeurIPS 2025\] DATE-LM: Benchmarking Data Attribution Evaluation for Large Language Models](../../NeurIPS2025/social_computing/date-lm_benchmarking_data_attribution_evaluation_for_large_language_models.md)
- [\[NeurIPS 2025\] Active Slice Discovery in Large Language Models](../../NeurIPS2025/social_computing/active_slice_discovery_in_large_language_models.md)

</div>

<!-- RELATED:END -->
