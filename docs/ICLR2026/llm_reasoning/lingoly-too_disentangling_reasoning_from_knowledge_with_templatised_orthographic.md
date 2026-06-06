---
title: >-
  [Paper Note] LingOly-TOO: Disentangling Reasoning from Knowledge with Templatised Orthographic Obfuscation
description: >-
  [ICLR 2026][LLM Reasoning][reasoning benchmark] This paper introduces LingOly-TOO, a benchmark that applies expert-designed grapheme-level permutations to linguistics olympiad problems…
tags:
  - "ICLR 2026"
  - "LLM Reasoning"
  - "reasoning benchmark"
  - "orthographic obfuscation"
  - "linguistics olympiad"
  - "knowledge contamination"
  - "LLM evaluation"
date: 2026-05-08
content_hash: 90894a67a5f73fec
---

# LingOly-TOO: Disentangling Reasoning from Knowledge with Templatised Orthographic Obfuscation

**Conference**: ICLR 2026
**arXiv**: [2503.02972](https://arxiv.org/abs/2503.02972)  
**Code**: [GitHub](https://github.com/jkhouja/LingOly-TOO)  
**Area**: LLM Reasoning / Evaluation Benchmarks
**Keywords**: reasoning benchmark, orthographic obfuscation, linguistics olympiad, knowledge contamination, LLM evaluation

## TL;DR

This paper introduces LingOly-TOO, a benchmark that applies expert-designed grapheme-level permutations to linguistics olympiad problems, preserving reasoning logic while eliminating knowledge and memorization shortcuts. The obfuscation reduces the top score across 15 frontier models from 0.59 to 0.48, systematically quantifying the extent to which LLM reasoning ability is overestimated due to knowledge effects.

## Background & Motivation

**Background**: LLM scores on reasoning benchmarks have risen rapidly, but growing evidence suggests that score inflation stems from training set contamination and knowledge memorization shortcuts rather than genuine improvements in reasoning ability. Benchmarks such as MATH and GSM8K are saturating quickly.

**Limitations of Prior Work**:

1. Increasing training data scale blurs the boundary between training and test sets, exacerbating evaluation bias.

2. Existing countermeasures (synthetic data, symbolic template substitution) are limited in scale and scope—modified instances may still resemble training samples.

3. Even linguistics problems in low-resource languages appear in pretraining corpora, allowing models to bypass reasoning through partial contamination.

**Key Challenge**: How can the logical structure of problem-solving reasoning be preserved while completely eliminating the possibility of the model exploiting knowledge or memorization?

**Key Insight**: The paper applies grapheme-level orthographic permutations to the problem language (Problemese) of linguistics olympiad problems, producing character sequences that cannot exist in any training corpus while fully preserving the reasoning steps required to solve each problem.

## Method

### Overall Architecture

82 UKLO problems → expert-annotated permutation rulesets → up to 6 orthographic obfuscation variants per problem → 1,203 problems / 6,995 sub-problem–answer pairs → Exact Match evaluation → comparison of $M_{og}$ (original score) and $M_{obf}$ (obfuscated score) to quantify the knowledge effect.

### Key Designs

1. **Reasoning-Equivariant Permutation**

    - Permutation operates at the grapheme level rather than the word level, as linguistics problems require sub-word symbolic reasoning.
    - Each problem has a manually defined ruleset authored by a linguistics expert to preserve the linguistic mechanisms necessary for solving the problem. For example, in Turkish vowel harmony, vowel pairs (e,i)/(o,u)/(ö,ü)/(a,ı) must remain within their respective groups; otherwise suffixes cannot be correctly matched.
    - Loanwords, English cognates, and proper nouns (names, place names) that are useful for solving the problem are retained.
    - Metadata that may trigger knowledge retrieval—such as language names, language family, and geographic information—is removed.

2. **Multi-Version Evaluation and Metric System**

    - $M_{obf} = \frac{1}{82}\sum_{i=1}^{82}\frac{1}{n_i}\sum_{j=1}^{n_i}M_{obf}^{i,j}$ denotes the average score on obfuscated variants; $M_{og}$ denotes the score on the original version.
    - Robustness metric $M_{rob}$: the average of the worst-case score across all permutations of each problem, measuring reasoning ability under the worst-case scenario.
    - Knowledge effect $\Delta_{obf}^{i} = M_{obf}^i - M_{og}^i$: a larger negative value indicates greater reliance on knowledge.
    - Benchmark validation: two IOL medalists audited the solvability of obfuscated problems; a 172-participant RCT showed human performance declined by only 5.7%.

### Loss & Training

This paper presents an evaluation benchmark. Key evaluation design choices are as follows:

- **Evaluation protocol**: Each prompt contains background context, the full problem context, all questions, and the specific sub-question, with output required in JSON format.
- **Scoring**: Strict Exact Match (no partial credit, to prevent spurious scores obtained by repeating context words).
- **Models evaluated**: 15 models including GPT-5, Claude 3.7, o3-mini, Gemini, and Llama, spanning both reasoning-specialized and general-purpose variants.

## Key Experimental Results

### Main Results

Performance of 15 models on LingOly-TOO:

| Model | $M_{og}$ (Original) | $M_{obf}$ (Obfuscated) | $M_{rob}$ (Robust) | Drop |
|---|---|---|---|---|
| GPT-5 | ~0.59 | **0.48** | 0.29 | -0.11 |
| Claude 3.7 (thinking) | ~0.55 | 0.44 | - | -0.11 |
| Claude 3.7 (no thinking) | ~0.40 | 0.30 | - | -0.10 |
| o3-mini (high) | ~0.45 | 0.31 | - | -0.14 |
| o3-mini (low) | ~0.25 | 0.13 | - | -0.12 |

GPT-5 by difficulty ($M_{obf}$): Breakthrough = 0.81, Round 2 = 0.31.

### Ablation Study

| Analysis Dimension | Result |
|---|---|
| No-context setting | $M_{obf}$ drops to 0.02–0.03, confirming that obfuscation effectively blocks knowledge shortcuts |
| Tokenization impact | Altering the tokenization strategy does not improve performance, ruling out a tokenization-based explanation |
| Language resource level effect | Japanese, Finnish, and Italian show the largest $\Delta_{obf}$ (−0.57 to −0.59) |
| Expert-guided reasoning | Providing intermediate reasoning steps raises $M_{obf}$ from 0.66 to 0.76 |
| Unreleased problem test | Performance drops are also observed on unpublished UKLO 2025 problems |

### Key Findings

- Reasoning-specialized models consistently outperform their general-purpose counterparts (o3-mini high vs. low: 18% gap), indicating that reasoning training yields genuine benefits.
- The knowledge effect is strongly negatively correlated with language resource level ($\beta < 0, p < 0.01$), with high-resource languages exhibiting the greatest score inflation.
- The benchmark is far from saturated: GPT-5 achieves only 0.31 on Round 2, with $M_{rob}$ of only 0.29.
- Reasoning traces frequently exhibit repetitive analysis and self-contradictory conclusions, indicating extremely poor reasoning consistency.

## Highlights & Insights

- The orthographic permutation methodology is conceptually elegant: grapheme-level permutation preserves the linguistic reasoning logic while producing character sequences that could not appear in any training corpus.
- The knowledge effect metric $\Delta_{obf}$ provides, for the first time, an operational approach to isolating reasoning ability from knowledge.
- The human RCT validates that obfuscation causes only a 5.7% drop for humans versus 11%+ for models, indicating that the performance gap is primarily attributable to knowledge dependence rather than cognitive penalty.
- $M_{rob}$ reveals reasoning fragility: GPT-5 drops from 0.48 to 0.29.

## Limitations & Future Work

- Strict Exact Match may underestimate partially correct reasoning—though partial credit would artificially inflate baselines.
- Coverage is limited to inductive/deductive reasoning in the natural language modality; visual and mathematical reasoning are not addressed.
- The benchmark comprises only 82 base problems, and permutation rules require manual expert design, limiting automation.
- A broader range of linguistic phenomena and additional competition sources remain unexplored.

## Related Work & Insights

- **vs. LingOly**: LingOly-TOO adds orthographic obfuscation to control for the knowledge variable.
- **vs. GSM-Symbolic**: Numerical substitution produces relatively minor perturbations; LingOly-TOO's grapheme permutations generate entirely novel character sequences.
- **vs. ARC / BIG-Bench Hard**: These benchmarks lack mechanisms to control for knowledge effects.
- **Implications**: The methodology is generalizable to other domains requiring symbolic reasoning, such as music and cryptography.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of orthographic obfuscation and knowledge/reasoning disentanglement is elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 15 models, multi-dimensional ablations, a human RCT, and validation on unreleased problems.
- Writing Quality: ⭐⭐⭐⭐ Rigorous structure and comprehensive analysis.
- Value: ⭐⭐⭐⭐⭐ Provides a landmark contamination-resistant methodology for evaluating LLM reasoning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Learning to Edit Knowledge via Instruction-based Chain-of-Thought Prompting](../../ACL2026/llm_reasoning/learning_to_edit_knowledge_via_instruction-based_chain-of-thought_prompting.md)
- [\[ACL 2026\] Does Self-Consistency Improve the Recall of Encyclopedic Knowledge?](../../ACL2026/llm_reasoning/does_self-consistency_improve_the_recall_of_encyclopedic_knowledge.md)
- [\[ACL 2026\] Towards Effective In-context Cross-domain Knowledge Transfer via Domain-invariant-neurons-based Retrieval](../../ACL2026/llm_reasoning/towards_effective_in-context_cross-domain_knowledge_transfer_via_domain-invarian.md)
- [\[AAAI 2026\] RPM-MCTS: Knowledge-Retrieval as Process Reward Model with Monte Carlo Tree Search for Code Generation](../../AAAI2026/llm_reasoning/rpm-mcts_knowledge-retrieval_as_process_reward_model_with_monte_carlo_tree_searc.md)
- [\[ICLR 2026\] RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Reasoning Intervention in Large Reasoning Models](rfeval_benchmarking_reasoning_faithfulness_under_counterfactual_reasoning_interv.md)

</div>

<!-- RELATED:END -->
