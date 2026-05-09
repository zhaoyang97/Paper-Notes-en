---
title: >-
  [Paper Note] I-RAVEN-X: Benchmarking Generalization and Robustness of Analogical and Mathematical Reasoning in Large Language and Reasoning Models
description: >-
  [NeurIPS 2025][LLM Reasoning][abstract reasoning] This paper introduces I-RAVEN-X, an enhanced symbolic reasoning benchmark that evaluates the generalization and robustness of analogical and mathematical reasoning in LLMs and LRMs by increasing operand complexity, attribute range, and perceptual uncertainty. Results show that LRMs significantly outperform LLMs under deterministic reasoning, but suffer sharp performance degradation under uncertain reasoning conditions.
tags:
  - NeurIPS 2025
  - LLM Reasoning
  - abstract reasoning
  - Raven's Progressive Matrices
  - LRM
  - analogical reasoning
  - robustness
date: 2026-05-08
content_hash: da2d5125e35e02dc
---

# I-RAVEN-X: Benchmarking Generalization and Robustness of Analogical and Mathematical Reasoning in Large Language and Reasoning Models

**Conference**: NeurIPS 2025
**arXiv**: [2510.17496](https://arxiv.org/abs/2510.17496)
**Code**: [GitHub](https://github.com/IBM/raven-large-language-models)
**Area**: LLM Reasoning
**Keywords**: abstract reasoning, Raven's Progressive Matrices, LRM, analogical reasoning, robustness

## TL;DR
This paper introduces I-RAVEN-X, an enhanced symbolic reasoning benchmark that evaluates the generalization and robustness of analogical and mathematical reasoning in LLMs and LRMs by increasing operand complexity, attribute range, and perceptual uncertainty. Results show that LRMs significantly outperform LLMs under deterministic reasoning, but suffer sharp performance degradation under uncertain reasoning conditions.

## Background & Motivation
**Background**: Abstract reasoning is considered a core hallmark of human intelligence. Raven's Progressive Matrices (RPM) is a classical task for evaluating abstract reasoning, analogical ability, and OOD generalization. I-RAVEN, its auto-generated variant, has been widely used to benchmark ML models.

**Limitations of Prior Work**: (1) Most I-RAVEN problems involve only a small number of operands (3×3 matrices), making reasoning difficulty insufficient; (2) test data is publicly available, introducing data leakage risks; (3) text-based conversion assumes oracle perception, ignoring perceptual uncertainty.

**Key Challenge**: Existing benchmarks cannot distinguish whether models possess genuinely generalizable reasoning capabilities or merely perform well under simple settings — particularly pressing with the emergence of LRMs (reasoning models), which demand more challenging evaluations.

**Goal**: Construct a parameterizable, more challenging reasoning benchmark to systematically evaluate LLMs and LRMs on generalization (longer reasoning chains, larger attribute ranges) and robustness (perceptual uncertainty).

**Key Insight**: Extend I-RAVEN along four dimensions — productivity, systematicity, confounding-factor robustness, and value-distribution robustness.

**Core Idea**: RPM evaluation should test not only whether a model answers correctly, but also whether it remains correct when reasoning chains grow longer, value ranges expand, or noise is introduced.

## Method

### Overall Architecture
I-RAVEN-X is a purely symbolic, parameterizable dataset built upon the single center-constellation setting of I-RAVEN, enhanced along four dimensions.

### Key Designs
1. **Productivity**: Extends the matrix from 3×3 to 3×10, increasing the number of operands/panels per row to test generalization over longer reasoning chains.
2. **Systematicity**: Expands the dynamic range of attribute values from 10 to 100 and 1000 (e.g., attribute values from $[0,9]$ to $[0,999]$), testing generalization over larger concept/value spaces.
3. **Confounding-Factor Robustness**: Adds 1–10 randomly sampled irrelevant attributes (e.g., background color, intra-object color patterns) to each panel to simulate noisy signals from imperfect perception. The signal-to-noise ratio (SNR) ranges from $\infty$ down to $-5.23$ dB.
4. **Value-Distribution Robustness**: Smooths the distribution of attribute values — rather than a deterministic single value, a probability distribution is used where the probability of the correct value $p_L$ decreases from $1.0$ to $0.51$, simulating uncertainty in the perceptual front-end.

### Evaluated Models
- **LRMs**: OpenAI o3-mini (medium/high), DeepSeek R1, DeepSeek R1 distilled (Llama 70B)
- **LLMs**: GPT-4, Llama-3 70B
- LLMs are evaluated with 21 prompts (including ICL examples, self-consistency, and decoupled prompting); LRMs use only 1 simple prompt.

### Evaluation Metrics
- **Task Accuracy**: Overall proportion of correctly predicted test samples.
- **Arithmetic Accuracy**: Proportion of correct predictions on attributes governed by arithmetic relations.

## Key Experimental Results

### Main Results
Accuracy comparison between I-RAVEN (3×3) and I-RAVEN-X (3×10, Range 1000):

| Model | I-RAVEN Task | I-RAVEN Arith. | I-RAVEN-X Task | I-RAVEN-X Arith. |
|-------|-------------|---------------|----------------|-----------------|
| GPT-4 (21 prompts) | 93.2% | 73.6% | 76.6% | 8.4% |
| Llama-3 70B (21 prompts) | 85.0% | 45.0% | 74.2% | 0.4% |
| o3-mini high (1 prompt) | 92.6% | 86.1% | 80.6% | 60.1% |
| DeepSeek R1 (1 prompt) | 80.6% | 74.8% | 82.8% | 65.8% |

**Key Comparison**: LLM arithmetic accuracy drops sharply from 59.3% to 4.4%, while LRM arithmetic accuracy declines moderately from 80.5% to 63.0%.

### Ablation Study
Uncertain reasoning (o3-mini, Range 1000):

| Setting | Task Acc. | Arith. Acc. |
|---------|----------|------------|
| No noise | 81.0% | 60.8% |
| +10 confounding attributes | 69.8% (−11.2%) | 45.6% (−15.2%) |
| Distribution smoothing $p_L=0.51$ | 75.6% (−5.4%) | 53.2% |
| Confounding + smoothing (hardest) | **17.0%** (−64.0%) | 41.1% |

The random baseline is 12.5%; under the hardest setting, LRM performance nearly degrades to chance level.

### Key Findings
- **LRMs substantially outperform LLMs on deterministic reasoning**: especially on mathematical/arithmetic tasks, LRMs with 1 prompt surpass LLMs using 21 prompts.
- **LRMs do not require complex prompt engineering**: o3-mini with 1/21 the number of prompts matches or exceeds GPT-4.
- **Uncertainty is the Achilles' heel of LRMs**: when confounding factors and distribution smoothing are applied simultaneously, task accuracy collapses to near-random levels.
- **Thinking tokens vs. reasoning robustness**: when facing uncertainty, o3-mini's output tokens increase from ~7K to ~18K, yet more thinking does not yield better results.
- **DeepSeek R1 is more robust to confounding factors but more fragile under distribution smoothing**, while o3-mini exhibits the opposite pattern.

## Highlights & Insights
- **Parameterizable benchmark design**: multiple dimensions of reasoning difficulty (length, range, noise) can be controlled independently, offering greater flexibility than fixed benchmarks.
- **Introduction of perceptual uncertainty**: this is the first symbolic reasoning benchmark to simulate "imperfect perception," bridging the gap between idealized reasoning evaluation and end-to-end systems.
- **Systematic LRM vs. LLM comparison**: clearly delineates where reasoning models (o3-mini, R1) hold advantages over conventional LLMs and where their weaknesses lie.
- **Mitigating data leakage**: I-RAVEN-X is freshly generated, reducing the risk of pretraining data contamination.

## Limitations & Future Work
- Only symbolic representations are used; the benchmark is not extended to the visual domain, leaving true visual perception ability untested.
- Only the center constellation (single object) is used, without covering other spatial layouts.
- The causal mechanism underlying performance degradation under uncertain reasoning is not deeply analyzed — it remains unclear whether the drop stems from longer prompts, harder pattern recognition, or deficiencies in probabilistic reasoning itself.
- The number of evaluated models is limited, with notable omissions (e.g., Claude 3.7 Sonnet with extended thinking).
- The paper consolidates two previously published works, limiting its incremental contribution.

## Related Work & Insights
- Aligned in spirit with the ARC (Abstraction and Reasoning Corpus) benchmark, but RPM task structure is more controlled.
- CRUXEval and CoRe evaluate similar capabilities from a code-reasoning perspective; I-RAVEN-X offers a complementary analogical reasoning perspective.
- The failure of LRMs under uncertain reasoning serves as a cautionary signal for agent systems: real-world perception is always imperfect, and reasoning systems must be capable of handling uncertainty.
- The finding that "thinking more does not imply reasoning better" warrants attention in scaling research.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The parameterizable benchmark design is innovative, and the introduction of perceptual uncertainty represents a valuable new perspective.
- **Experimental Thoroughness**: ⭐⭐⭐ Model coverage is limited (only 4–5 models), and part of the work consolidates prior publications.
- **Writing Quality**: ⭐⭐⭐⭐ Structure is clear with well-described experimental setups, though the paper is relatively short (workshop paper style).
- **Value**: ⭐⭐⭐⭐ Offers meaningful reference for understanding the capability boundaries of LRMs; uncertain reasoning remains an important open problem.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] ProofSketch: Efficient Verified Reasoning for Large Language Models](proofsketch_efficient_verified_reasoning_for_large_language_models.md)
- [\[NeurIPS 2025\] Transformers Provably Learn Chain-of-Thought Reasoning with Length Generalization](transformers_provably_learn_chain-of-thought_reasoning_with_length_generalizatio.md)
- [\[NeurIPS 2025\] Mapping Faithful Reasoning in Language Models](mapping_faithful_reasoning_in_language_models.md)
- [\[NeurIPS 2025\] ThinkSound: Chain-of-Thought Reasoning in Multimodal Large Language Models for Audio Generation and Editing](thinksound_chain-of-thought_reasoning_in_multimodal_large_language_models_for_au.md)
- [\[NeurIPS 2025\] Scalable Best-of-N Selection for Large Language Models via Self-Certainty](scalable_best-of-n_selection_for_large_language_models_via_self-certainty.md)

<!-- RELATED:END -->
