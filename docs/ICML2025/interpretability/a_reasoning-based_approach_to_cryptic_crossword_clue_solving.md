---
title: >-
  [Paper Note] A Reasoning-Based Approach to Cryptic Crossword Clue Solving
description: >-
  [ICML 2025][Interpretability][cryptic crossword] A three-stage LLM reasoning pipeline (Answer Candidate Generation $\rightarrow$ Wordplay Suggestion $\rightarrow$ Python Formalisation & Verification) is proposed. Using open-source 9B models, it achieves a new SOTA on the Cryptonite dataset. The key innovation lies in formalizing wordplay reasoning into executable Python code and iteratively correcting it via a verifier with hints.
tags:
  - "ICML 2025"
  - "Interpretability"
  - "cryptic crossword"
  - "reasoning verification"
  - "Python DSL"
  - "fine-tuning"
  - "formal reasoning"
date: 2026-05-08
content_hash: 197db6fdc8c3ba33
---

# A Reasoning-Based Approach to Cryptic Crossword Clue Solving

**Conference**: ICML 2025  
**arXiv**: [2506.04824](https://arxiv.org/abs/2506.04824)  
**Code**: [https://github.com/mdda/cryptic-crossword-reasoning-verifier](https://github.com/mdda/cryptic-crossword-reasoning-verifier)  
**Area**: Interpretability  
**Keywords**: cryptic crossword, reasoning verification, Python DSL, fine-tuning, formal reasoning

## TL;DR

A three-stage LLM reasoning pipeline (Answer Candidate Generation $\rightarrow$ Wordplay Suggestion $\rightarrow$ Python Formalisation & Verification) is proposed. Using open-source 9B models, it achieves a new SOTA on the Cryptonite dataset. The key innovation lies in formalizing wordplay reasoning into executable Python code and iteratively correcting it via a verifier with hints.

## Background & Motivation

**Cryptic crosswords** are intellectual puzzle challenges published daily by mainstream UK newspapers (e.g., *The Times*, *The Telegraph*). Unlike quick crosswords, each clue consists of two parts:

**Definition**: An answer definition similar to a standard crossword clue.

**Wordplay**: A "proof" of the correctness of the answer through wordplay (anagrams, abbreviations, homophones, hidden words, etc.).

This means that human solvers can arrive at the same answer through two independent paths (definition and wordplay), forming a natural "proof" structure.

**Why is this problem important?**
- Thousands of new puzzles are published daily, providing a continuous stream of test data (contrasting with the scarcity of math competition problems).
- It requires multi-level language understanding: the integration of logical reasoning, wordplay, and contextual nuances.
- LLMs perform poorly on this task: a fine-tuned T5-Large only achieves 7.6% accuracy, and even GPT-4 scored extremely low before 2024.
- The "surface reading" of the clue is often intentionally misleading, making it easy for LLMs to be deceived by literal meaning.

**Core Motivation**: Inspired by the prover+verifier paradigm in mathematical reasoning (such as Draft, Sketch, and Prove), the authors model cryptic crosswords as a **reasoning verification problem**—first guessing the answer, then generating the reasoning path, and finally validating it through formal verification.

## Method

### Overall Architecture

The system adopts a **three-stage pipeline** to simulate the human solving process:

**Stage 1: Answer Candidate Generation**
- Uses a fine-tuned **Gemma2 9B base** model.
- Input: clue + pattern (number of letters in the answer) + ad (across/down direction).
- Generates **20 candidate answers** for each clue (using temperature $t = 1.0$ to increase diversity).
- Candidates that do not match the pattern are immediately rejected and regenerated.
- Candidates not present in the crossword lexicon (UK Advanced Cryptics Dictionary) are filtered out.
- Candidates are grouped by frequency statistics.

**Stage 2: Wordplay Explanation Generation (Definition & Wordplay Suggestion)**
- Uses another fine-tuned **Gemma2 9B base** model.
- Generates **10 wordplay hypotheses** for each unique candidate answer.
- Input: clue + candidate answer.
- Training data comes from the Wordplay dataset (approximately 16,800 solving explanations from *The Times* and *Financial Times*).

**Stage 3: Python Formalisation & Verification**
- Uses **Gemini-Flash-1.5-001** (Gemma2-9B-it was also tested).
- Translates wordplay into Python code (containing a chain of assert statements).
- A purpose-built verifier validates assertions line-by-line using Python AST.
- Upon verification failure, the verifier provides **hints** to assist the LLM in rewriting (up to 2 retries).
- Successful verification = the answer is "proven" correct.
- If all fail, the system falls back to the candidate with the highest frequency.

Overall: 20 candidate answers $\times$ 10 wordplays $\times$ 3 formalisation attempts = substantial test-time computation, substituting large commercial models with open-source 9B models.

### Key Designs

**1. Selection of Python as the Formalisation Language**

The authors experimented with a custom DSL, but found that Gemini-Flash struggled to learn a novel DSL via few-shot prompting. Conversely, LLMs naturally generate correct Python code. Thus, a "**lightweight DSL embedded in Python**" approach was adopted: wrapping functions inside assert statements.

Core DSL functions include:
- `is_synonym(phrase, test_synonym)` — determines synonym relationships (calling a thesaurus + LLM behind the scenes).
- `is_abbreviation(phrase, test_abbreviation)` — abbreviation lookup.
- `action_type(phrase, action)` — resolves the indicator word type (e.g., "shredded" $\rightarrow$ ANAGRAM).
- `is_anagram(letters, word)` — verifies anagrams.
- `is_homophone(phrase, test_homophone)` — homophone validation.
- Action enums: ANAGRAM, REMOVE_FIRST, INITIALS, REVERSE, GOES_INSIDE, etc.

**2. Verification Feedback Mechanism with Hints**

The verifier not only reports pass/fail but also provides **constructive hints**:
```python
AssertionError: is_abbreviation('an Artist', 'RA'):
  'an Artist' does not have a valid abbreviation;
  'RA' is an abbreviation for: artist, artillery, Royal Artillery, gunners, painter
```
This informs the LLM to use "artist" instead of "an Artist". Similarly:
```python
AssertionError: action_type('goes crazy', Action.ANAGRAM):
  'goes crazy' does not suggest Action.ANAGRAM, but 'crazy' does
```
suggesting to use only "crazy" as the anagram indicator.

**3. In-Context Learning Strategy (Only 6 Formalisation Examples)**

The prompt for the formalisation stage consists of the following components:
1. Explanation of cryptic crossword rules.
2. 20-shot wordplay examples.
3. External function definitions (Figure 5).
4. **Only 6** wordplay $\rightarrow$ Python formalisation examples.
5. The actual problem to formalise.

Under conditions of extreme training data scarcity ($<10$ "good proofs" available), a carefully designed ICL prompt enables the LLM to reliably generate correctly formatted Python code.

**4. Finding that Base Models Outperform Instruct Models**

In the answer generation phase, the base model performed better than the instruct model, likely because instruction tuning "penalizes" unexpected answers—which are exactly what cryptic clues require.

**5. Pipeline Design Checklist: Guessing the Answer Before Reasoning**

It was observed that GPT-4 using CoT easily gets "fixated early" in the reasoning process and misled by surface readings. This system places answer guessing as the first step, forcing subsequent models to "fit reasoning to the answer", thereby embedding "re-hypothesizing" into the workflow.

### Loss & Training

**Answer Generation Model**:
- Base model: Gemma2 9B base
- Fine-tuning method: LoRA (via the unsloth library)
- Training data: Cryptonite training set, approx. 470,000 samples
- Training epochs: 1 epoch
- Inference temperature: $t = 1.0$ (to produce more diverse candidates)

**Wordplay Generation Model**:
- Base model: Gemma2 9B base
- Fine-tuning method: LoRA
- Training data: Approx. 16,800 samples from the Wordplay dataset (selected authors from *The Times* and *Financial Times*)
- Training epochs: 4 epochs

**Formalisation Model**:
- No fine-tuning; used in a pure ICL manner with Gemini-Flash-1.5-001.
- The unmodified Gemma2-9B-it was also tested (same prompt, no adaptation required).
- Additional experiment: Fine-tuning Gemma2-9B on 448 valid proofs generated by Gemini.

**Extremely Low Compute Cost**:
- Total Gemini API cost $< \$100$ USD
- Single GPU fine-tuning total $< \$50$ USD
- Full Cryptonite training took approx. 24 hours, Wordplay training approx. 8 hours.

## Key Experimental Results

### Main Results

Evaluated on the Cryptonite dataset (523K clues from *The Times* & *The Telegraph*), Top-1 exact match accuracy:

| Model | Test Overall | Test Quick | Test Hard |
|------|-------------|-----------|----------|
| Rule-based | 8.6% | 13.5% | 5.8% |
| T5-Large FT (770M) | 7.6% | 12.8% | 3.4% |
| GPT-4o 5-shot | 27.6% | 47.4% | 26.0% |
| Gemma2-9B FT (Top-1) | 15.9% | 38.2% | 14.1% |
| Gemma2-9B freq (Voting, #=20) | 25.5% | 55.3% | 23.1% |
| **Gemini-Flash Formaliser** | **32.5%** | **46.7%** | **31.4%** |
| Gemma2-9B-it Formaliser | 29.0% | 46.7% | 27.6% |
| Gemma2-9B-FT Formaliser | 29.5% | 53.3% | 27.6% |

**Core Results**:
- Gemini-Flash Formaliser achieves 32.5% Test Overall, establishing a **new SOTA**.
- Bayesian IRT analysis: Gemini-Flash has a 92% probability of truly outperforming GPT-4o.
- The fully open-source Gemma2-9B-it Formaliser also reaches 29.0%, marginally beating the prior SOTA.
- The voting strategy (freq) itself reaches 25.5%, exceeding the prior open model SOTA.

### Ablation Study

| Ablation Variant | Test Overall | Test Quick | Test Hard |
|---------|-------------|-----------|----------|
| logprob answer (select using answer logprob) | 22.7% | 55.3% | 20.1% |
| logprob wordplay (select using wordplay logprob) | 20.5% | 46.7% | 18.4% |

Both ablations are significantly lower than the verifier approach, demonstrating:
1. **The formalisation + verification step is indispensable**—it cannot be replaced by a "simple ranker".
2. Wordplay logprob selection is even worse than answer frequency selection, because the LLM assigns high probabilities to "simply phrased but completely fabricated" wordplay.

### Key Findings

1. **Ceiling effect of candidate answers**: When generating 20 candidates, the probability of including the correct answer is around 45%—this is the ceiling of the entire system's accuracy.
2. **Quick vs. Hard discrepancy**: Formalisation contributes more on Hard clues (from 23.1% $\rightarrow$ 31.4%), whereas on Quick clues, it might actually "overrule" an already correct high-frequency result.
3. **Incorrect answers produce unverifiable wordplay**: Correct answers (e.g., HERON $\rightarrow$ "HER + ON") naturally yield verifiable wordplay, while incorrect answers (e.g., EGRET) yield wordplays that are clearly absurd and impossible to formalise.
4. **`is_synonym` function is the main bottleneck**: Definition-answer relationships in cryptic clues are often much more distant than in ordinary crosswords. Setting the synonym "distance threshold" remains a continuous challenge.
5. **End-to-end viability with open-source models**: Using Gemma2-9B-it as the formaliser achieves performance only 3.5% lower than Gemini-Flash, proving that the entire pipeline can run fully locally.

## Highlights & Insights

1. **Modeling NLP reasoning tasks as a prover-verifier problem**: This is a successful transfer of the mathematical theorem-proving paradigm to the natural language reasoning domain. The key insight is that the dual-path structure of cryptic clues (definition + wordplay) is naturally suited to a "guess-and-verify" framework.

2. **Python as a "universal formalisation language"**: Rather than designing a novel DSL (which LLMs struggle to learn via few-shot prompting), it is better to embed the DSL in Python—which LLMs naturally write well. Enabling the LLM to use custom functions for formalisation with only 6 examples highlights the unique advantage of Python as an intermediate language for LLM reasoning.

3. **Exquisite design of the Hints mechanism**: The verifier goes beyond simple pass/fail outcomes, leveraging Python AST to parse asserts line-by-line and provide specific corrective guidance. This is more effective than generic approaches like Self-Debug, as the hints are domain-specific.

4. **Effective utilization of test-time computation**: 20 $\times$ 10 $\times$ 3 = 600 reasoning attempts, using the inference-time compute of open-source 9B models instead of a single API call to a massive model. This illustrates the potential of the "small model + multiple attempts + verifier" paradigm.

5. **Interpretability**: Each proven answer is accompanied by an inspectable Python reasoning process—something black-box LLMs cannot provide. Commercial models might output the correct answer, but their reasoning is often illogical (implying memorization of clue/answer pairs).

6. **Base Models > Instruct Models**: Answer generation requires "counter-intuitive" guessing, which instruction tuning tends to suppress. This provides insightful direction for task characteristics and model selection.

## Limitations & Future Work

1. **Candidate answer coverage bottleneck**: System accuracy is limited by the candidate answer phase—if the correct answer is not present in the top-20 (about 55% of cases), subsequent verification cannot proceed. Improving candidate generation quality is the most direct path forward.

2. **Verifier security vulnerabilities**:
    - Python code might contain only comments (triggering no assertion errors).
    - Conditional branches can bypass assertions.
    - Rewriting might produce invalid modifications like `assert XYZ == False`.
    - Proofs might have logical gaps (variables on the left not supported by the right).
    - In an RL setting, these vulnerabilities could be systematically exploited.

3. **Limitations of `is_synonym` function**: Semantic distance between the definition and answer in cryptic clues is much larger than in ordinary crosswords (e.g., "damaged" $\rightarrow$ UNDERMINED). Setting an appropriate synonym distance threshold remains an open problem.

4. **Potential of Reinforcement Learning**: Works like DeepSeek-R1 show that RL can incentivize reasoning capabilities, which is highly promising for application in cryptic crosswords. However, the verifier must first be made "bulletproof" to avoid reward hacking.

5. **Risk of dataset leakage**: GPT-4o performs exceptionally well on the validation set (29.8%), suggesting its training data might contain Cryptonite-related content, which threatens benchmark evaluation validity.

6. **Adverse effect on Quick clues**: The verifier can override already correct frequency answers on Quick clues, indicating that circular verification strategies need to dynamically adapt to clue difficulty.

## Related Work & Insights

- **Draft, Sketch, and Prove (Jiang et al., 2023)**: LLM draft $\rightarrow$ formal proof paradigm for math theorems, which directly inspired the architectural design of this paper.
- **PAL (Gao et al., 2023)**: Pioneering work in building verifiable reasoning chains using code generation.
- **AlphaCodium (Ridnik et al., 2024)**: Paradigm shift from "generate many candidates + filter" to "iterative refinement", which inspired the feedback-with-hints mechanism of this work.
- **Self-Debug (Chen et al., 2024)**: Framework for LLM self-debugging of code; the verification-with-hinting in this paper can be viewed as a domain-specific version.
- **Cryptonite (Efrat et al., 2021)**: Benchmark dataset of 523K clues, where T5-Large achieved only 7.6% accuracy, showing task difficulty.
- **CrypticCrosswords.jl (Deits, 2022)**: A rule-based probabilistic grammar solver whose indicator lexicon and abbreviation lists were borrowed in this study.
- **SatLM (Ye et al., 2023)**: Precedent for choosing Python as an intermediate language for automated formalisation.

**Insights for my work**:
1. The "lightweight DSL embedded in Python" strategy can be generalized to other NLP reasoning tasks requiring formal verification.
2. Designing a verifier to provide structured hints is more effective than generic error messages, a strategy worth adopting in code generation tasks.
3. Comparing base vs. instruct model selection across different tasks is a valuable practice.
4. Leveraging small models + multiple samplings + verifiers may outperform single calls to large models in reasoning-intensive tasks.

## Rating

| Dimension | Score | Description |
|------|------|------|
| Novelty | ⭐⭐⭐⭐ | Applying the prover-verifier paradigm to NLP reasoning tasks, with a clever Python DSL design |
| Technical Depth | ⭐⭐⭐⭐ | Meticulous pipeline design, with hints mechanism and AST parsing showing engineering depth |
| Experimental Thoroughness | ⭐⭐⭐⭐ | Complete ablations, valuable qualitative analysis, rigorous Bayesian IRT statistical testing |
| Reproducibility | ⭐⭐⭐⭐⭐ | Open-source code, API cost $<\$100$, single GPU trainable, highly reproducible |
| Practical Value | ⭐⭐⭐ | The task itself is niche, but the methodology can be generalized to other reasoning verification scenarios |
| Overall Rating | ⭐⭐⭐⭐ | Although the domain is niche, the methodological contribution is outstanding, serving as an excellent example of combining formal reasoning with NLP |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Validating Mechanistic Interpretations: An Axiomatic Approach](validating_mechanistic_interpretations_an_axiomatic_approach.md)
- [\[ICML 2025\] Inference-Time Decomposition of Activations (ITDA): A Scalable Approach to Interpreting Large Language Models](inference-time_decomposition_of_activations_itda_a_scalable_approach_to_interpre.md)
- [\[NeurIPS 2025\] SpEx: A Spectral Approach to Explainable Clustering](../../NeurIPS2025/interpretability/spex_a_spectral_approach_to_explainable_clustering.md)
- [\[ICML 2025\] Towards Long-Horizon Interpretability: Efficient and Faithful Multi-Token Attribution for Reasoning LLMs](towards_long-horizon_interpretability_efficient_and_faithful_multi-token_attribu.md)
- [\[NeurIPS 2025\] Additive Models Explained: A Computational Complexity Approach](../../NeurIPS2025/interpretability/additive_models_explained_a_computational_complexity_approach.md)

</div>

<!-- RELATED:END -->
