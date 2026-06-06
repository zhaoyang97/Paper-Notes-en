---
title: >-
  [Paper Note] TLPO: Token-Level Policy Optimization for Mitigating Language Confusion in Large Language Models
description: >-
  [ACL2026][Multilingual & Machine Translation][Language confusion] TLPO treats language confusion in multilingual LLMs as locatable local token errors. It performs policy optimization only on high-probability candidate to…
tags:
  - "ACL2026"
  - "Multilingual & Machine Translation"
  - "Language confusion"
  - "token-level optimization"
  - "multilingual alignment"
  - "PPO"
  - "local error correction"
date: 2026-05-08
content_hash: da4b0b158f051ab8
---

# TLPO: Token-Level Policy Optimization for Mitigating Language Confusion in Large Language Models

**Conference**: ACL2026  
**arXiv**: [2604.26553](https://arxiv.org/abs/2604.26553)  
**Code**: https://github.com/samsungsds-research-papers/TLPO  
**Area**: Multilingual Generation / Machine Translation / LLM Alignment  
**Keywords**: Language confusion, token-level optimization, multilingual alignment, PPO, local error correction

## TL;DR
TLPO treats language confusion in multilingual LLMs as locatable local token errors. It performs policy optimization only on high-probability candidate tokens at the first position where confusion occurs, thereby significantly improving target language consistency while maximally preserving the original reasoning and knowledge capabilities of the model.

## Background & Motivation
**Background**: Multilingual LLMs often experience language confusion in contexts involving cross-lingual instructions, target language responses, and code/math mixtures—where non-target languages are interspersed when the target language should be used. Common correction methods include supervised fine-tuning (SFT), preference optimization such as DPO/ORPO, or sequence-level reinforcement learning alignment for the entire response.

**Limitations of Prior Work**: Language confusion typically occurs only at a few tokens or specific switching points, yet SFT and sequence-level preference optimization treat the entire sequence as the training object. A side effect is that to satisfy language constraints, the model may damage its learned knowledge, reasoning formats, and response length distributions. This is especially true when English serves as the carrier for knowledge; over-suppressing English can directly harm accuracy.

**Key Challenge**: Language consistency requires stronger constraints, but global fine-tuning sacrifices model capabilities. The positions truly requiring updates are usually near the "first stray token" rather than the complete response sequence. Therefore, the key problem is how to precisely place training signals on tokens causing language confusion without disturbing unrelated positions.

**Goal**: The authors aim to design a token-level policy optimization method that automatically locates confusion points, explores alternative candidate tokens at those positions, and generates local preference signals using only these candidates to reduce disruption to the global distribution.

**Key Insight**: The paper defines language confusion as a detectable local event: the position $c$ of the first non-target language token in the generated sequence. Instead of rewarding or punishing the entire response, it is more effective to check which top-N candidate tokens at $c$ lead to confusion and which maintain the target language, then directly adjust the relative probabilities of these candidates.

**Core Idea**: Use probability ranking to select the top-N tokens most likely to be generated at the confusion point, perform a short lookahead for each candidate to determine if confusion occurs, and then use a probability-weighted advantage in a PPO-style objective to optimize only these tokens.

## Method
The TLPO workflow is highly localized: it first lets the current model generate a response. If no language confusion occurs, the sample is skipped. If confusion appears, the first confusion token is located, and candidates, rewards, and losses are constructed only around this position. Its goal is not to have the model relearn the entire multilingual task, but to preserve the existing capabilities of the original model while lowering the probability mass leading to language switching.

### Overall Architecture
Given a prompt $x$ and a model-generated sequence $y$, TLPO detects the first language confusion position $c$. Under the condition that the prefix $y_{<c}$ is fixed, the top-N next tokens of the current policy $pi_{theta}$ are taken to form a candidate set. For each candidate token, the model generates a very short lookahead sequence and detokenizes it, using character-set rules to determine if the candidate triggers output outside the target language. Finally, TLPO constructs an advantage using the candidate token rewards and old policy probabilities to update the model via a PPO objective with clipping and KL constraints.

### Key Designs
1. **Probability-Ranked Candidate Token Exploration**:

	- **Function**: Focuses only on tokens the model is most likely to generate at the confusion point, rather than sampling the entire vocabulary or the entire sequence.
	- **Mechanism**: At confusion position $c$, the top-N tokens with the highest probabilities are selected from $pi_{theta}(\cdot | x, y_{<c})$ to form candidate set $T$. The main results use $N=16$, and the ablation study examines ranked selection versus multinomial sampling.
	- **Design Motivation**: Language confusion is usually triggered by a few high-probability tokens. Optimizing these tokens can directly alter the most likely error paths while significantly reducing the scope of training signals, avoiding strong constraints on irrelevant tokens.

2. **Token-Level Reward and Probability-Weighted Advantage**:

	- **Function**: Individually determines whether each candidate token "leads to language confusion" and converts this local judgment into a stable policy gradient signal.
	- **Mechanism**: A candidate token might be a subword, making it difficult to determine the language category from the token characters alone. Therefore, TLPO generates $k=3$ lookahead tokens, concatenates them with the candidate, detokenizes them, and assigns a reward based on the target language character set. The advantage is defined as $A_i = p_{old}(t_i)(R(t_i)-mu)/Z$, where $mu$ is the probability-weighted average reward and $Z$ is a normalization constant for absolute advantages.
	- **Design Motivation**: Multiplying by the old policy probability maintains the relative probability structure within valid tokens, while normalization stabilizes the scale of training signals produced by each confusion point. Compared to standard deviation normalization in GRPO-style methods, ablation shows this formulation is more favorable for accuracy.

3. **PPO-style Local Updates Only at Confusion Points**:

	- **Function**: Compresses sequence-level preference optimization into a local policy optimization at a single token position.
	- **Mechanism**: The TLPO objective averages over the candidate set $T$, using the ratio of new and old policy probabilities, clipping, and a KL penalty against a reference policy. The reference policy is the initial model before applying TLPO, and the KL constraint limits the shift range.
	- **Design Motivation**: SFT/DPO/ORPO update the complete response, which easily spreads "language constraints" to semantic and reasoning capabilities. TLPO only modifies error boundary points, making it more like precise error correction than global model reshaping.

### Loss & Training
Training data comes from the multilingual instruction-following split of Bactrian-X. Target languages include Chinese, Arabic, Korean, and Japanese. Base models include Llama-3.1-8B-Instruct, Qwen3-8B, Ministral-8B-Instruct, and Gemma-3-4B-IT. Evaluation is divided into two categories: language confusion is measured by Response Pass Rate (RPR) and Word Pass Rate (WPR), while general capabilities are measured by accuracy in MIF, MMLU, MMMLU, GPQA, ARC-Challenge, BBH, MATH, and GSM8K. Experiments also distinguish between two English handling methods: English as a neutral category, and English as language confusion.

## Key Experimental Results

### Main Results
The first set of experiments treats English as a neutral category, which is closer to real-world scenarios where English frequently appears in abbreviations, proper nouns, chapter titles, and technical terms.

| Method | Avg RPR | Avg WPR | Avg Acc | Main Conclusion |
|------|----------|----------|------------|----------|
| Baseline | 96.68 | 99.92 | 58.35 | High baseline accuracy, but minor language confusion exists |
| SFT | 99.14 | 99.92 | 50.71 | Improved consistency, but significant drop in knowledge/reasoning |
| DPO | 98.31 | 99.72 | 55.94 | More conservative than SFT, but still loses accuracy |
| ORPO | 97.27 | 99.88 | 55.12 | Limited language correction with decreased accuracy |
| TLPO | 99.19 | 99.98 | 58.08 | Highest RPR, almost preserves baseline accuracy |

The second set of experiments uses a stricter setting where any non-target language English output is considered confusion. The task difficulty increases significantly as many models default to using English symbols and terms during reasoning.

| Method | Avg RPR | Avg WPR | Avg Acc | Main Conclusion |
|------|----------|----------|------------|----------|
| Baseline | 63.27 | 82.31 | 58.24 | Large amount of English output judged as confusion under strict rules |
| SFT | 47.20 | 73.01 | 50.71 | Excessive supervision degrades both consistency and accuracy |
| DPO | 72.73 | 84.02 | 54.60 | Improvement, but clear loss of capability |
| ORPO | 69.75 | 86.51 | 54.61 | High WPR, but RPR and accuracy are lower than TLPO |
| TLPO | 77.59 | 85.64 | 56.17 | Highest RPR with minimal accuracy loss |

### Ablation Study
The paper focuses on analyzing token selection and advantage formulation.

| Ablation Dimension | Observation | Significance |
|----------|----------|------|
| Ranked selection vs multinomial sampling | Both maintain RPR above 99%, but ranked selection has higher accuracy | Optimizing the most likely candidates avoids disturbing unrelated distributions better than random sampling |
| TLPO advantage vs $R-mu$ | TLPO's probability-weighted advantage achieves highest accuracy | Old policy probability weights help preserve the relative distribution of effective tokens |
| $R-mu$ vs GRPO-style $(R-mu)/sigma$ | No standard deviation normalization is better | In local candidate sets, standard deviation scaling may amplify noise |
| Probability changes of tokens outside top-N | Cumulative probability of untrained confusion tokens also decreases; non-confusion tokens increase | Local optimization produces generalization effects through language-related representations |

### Key Findings
- Under the English-neutral setting, TLPO increases average RPR from 96.68 (baseline) to 99.19, while average accuracy only slightly drops from 58.35 to 58.08. In contrast, SFT achieves 99.14 RPR but drops accuracy to 50.71.
- Under the strict English-as-confusion setting, TLPO's average RPR reaches 77.59, which is 4.86 higher than DPO and 7.84 higher than ORPO. Its average accuracy of 56.17 also exceeds other alignment methods.
- SFT's RPR in the strict setting is actually lower than the baseline, indicating that direct supervision with target language answers does not guarantee stable suppression of language confusion and may induce shorter, stiffer, or more unstable outputs.
- The probability analysis outside top-N is interesting: even though TLPO only trains top-N candidates, confusion tokens of the same language that did not explicitly enter the loss are also suppressed, suggesting specific language directions or shared components exist within the model.

## Highlights & Insights
- The biggest highlight is reframing language confusion from a "sequence-level alignment failure" to a "local token boundary error." This problem redefinition makes the training signal extremely precise and explains why TLPO harms knowledge capability far less than sequence-level preference optimization.
- Lookahead detokenization is a simple but critical implementation detail. In multilingual tokenizers, a single character may be split into multiple tokens; looking only at the current token could misjudge the language category, whereas a short lookahead more reliably determines if a candidate truly triggers confusion.
- The two settings—English-neutral and English-strict—are highly valuable. The former is close to actual products, while the latter tests extreme language adherence; together they demonstrate that TLPO's advantage is not gained by relaxing evaluation definitions.
- This token-level optimization approach can be migrated to other "locally detectable errors," such as format leakage, unit errors, specific sensitive words, or incorrect API names in code, provided the error boundary can be automatically located and candidate tokens can be scored.

## Limitations & Future Work
- TLPO relies on explicit error boundaries, making it specifically suitable for local errors like language confusion. It is difficult to directly locate single tokens for sequence-level attributes such as helpfulness, factual correctness, or complex reasoning quality.
- Language detection rules are primarily based on character sets; boundary processing for mixed scripts, loanwords, numbers, punctuation, code snippets, and proper nouns may still affect reward accuracy.
- Experiments cover four target languages and four 4B-8B class models, but validation is still needed for lower-resource languages, morphologically complex languages, ultra-large models, and real-world multi-turn user scenarios.
- Future work could combine TLPO with finer-grained language ID models, token attribution, or contrastive decoding to make error localization and candidate scoring more robust, or explore a unified token-level alignment framework for multiple types of local errors.

## Related Work & Insights
- **vs SFT**: SFT uses entire target language responses for supervision, which is straightforward but prone to excessively rewriting the model distribution. TLPO updates only the top-N tokens at confusion points, acting more like minimally invasive surgery, thus better preserving capability.
- **vs DPO / ORPO**: DPO and ORPO perform sequence-level preference optimization, making it difficult to distinguish between "language correct but poor semantics" and "semantics correct but a specific token is confused." TLPO's reward granularity is finer and better suited for such local errors.
- **vs GRPO**: The paper attempted to use sequence-level confusion as a reward for GRPO but observed that response lengths gradually shortened during training; this was ultimately excluded from main results. TLPO avoids this length gaming caused by sequence-level rewards.
- **Insight**: Alignment does not always have to start from the complete response. If an error type can be localized, local policy optimization can be more stable than global preference learning and more easily preserves the model's original capabilities.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The problem approach of token-level policy optimization is clear, precisely locating language confusion at generation boundaries with a concise and effective method.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers multiple models, languages, two English evaluation settings, and multiple capability benchmarks. The ablation trends are clear, though some charts lack precise numerical values.
- Writing Quality: ⭐⭐⭐⭐☆ Motivation and experimental explanations flow well. Formulas are concentrated but generally easy to follow.
- Value: ⭐⭐⭐⭐☆ Highly practical for multilingual LLM products and provides a transferable paradigm for local error alignment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Hierarchical Policy Optimization for Simultaneous Translation of Unbounded Speech](hierarchical_policy_optimization_for_simultaneous_translation_of_unbounded_speec.md)
- [\[ACL 2026\] LaoBench: A Large-Scale Multidimensional Lao Benchmark for Large Language Models](laobench_a_large-scale_multidimensional_lao_benchmark_for_large_language_models.md)
- [\[ACL 2026\] LLM-XTM: Enhancing Cross-Lingual Topic Models with Large Language Models](llm-xtm_enhancing_cross-lingual_topic_models_with_large_language_models.md)
- [\[AAAI 2026\] Focusing on Language: Revealing and Exploiting Language Attention Heads in Multilingual Large Language Models](../../AAAI2026/multilingual_mt/focusing_on_language_revealing_and_exploiting_language_attention_heads_in_multil.md)
- [\[ACL 2026\] Language Models Entangle Language and Culture](language_models_entangle_language_and_culture.md)

</div>

<!-- RELATED:END -->
