---
title: >-
  [Paper Note] SAM Decoding: Speculative Decoding via Suffix Automaton
description: >-
  [ACL 2025][LLM Efficiency][Speculative Decoding] This paper proposes SAM-Decoding, which utilizes a Suffix Automaton (SAM) to perform longest suffix matching on general text corpora and the current text sequence for efficient draft generation in speculative decoding. Achieving an average $O(1)$ time complexity, it outperforms existing retrieval-based methods on Spec-Bench by over 18%, and can be combined complementarily with methods like EAGLE-2 to yield a further speedup of…
tags:
  - "ACL 2025"
  - "LLM Efficiency"
  - "Speculative Decoding"
  - "Suffix Automaton"
  - "Lossless Acceleration"
  - "LLM Inference Optimization"
  - "Retrieval-based Draft Generation"
date: 2026-05-08
content_hash: 65ad16b96768304d
---

# SAM Decoding: Speculative Decoding via Suffix Automaton

**Conference**: ACL 2025  
**arXiv**: [2411.10666](https://arxiv.org/abs/2411.10666)  
**Code**: [GitHub Repository](https://github.com/hyx1999/SAM-Decoding)  
**Area**: LLM Efficiency  
**Keywords**: Speculative Decoding, Suffix Automaton, Lossless Acceleration, LLM Inference Optimization, Retrieval-based Draft Generation

## TL;DR
This paper proposes SAM-Decoding, which utilizes a Suffix Automaton (SAM) to perform longest suffix matching on general text corpora and the current text sequence for efficient draft generation in speculative decoding. Achieving an average $O(1)$ time complexity, it outperforms existing retrieval-based methods on Spec-Bench by over 18%, and can be combined complementarily with methods like EAGLE-2 to yield a further speedup of 3.28%–11.13%.

## Background & Motivation
- **Background**: Speculative decoding is an effective technique for lossless LLM inference acceleration. It quickly generates candidate tokens using a lightweight draft model, which are then validated in parallel by the target LLM. Existing methods are categorized into model-based (e.g., Medusa, EAGLE-2) and model-free (retrieval-based, e.g., PLD, REST) approaches.
- **Limitations of Prior Work**:
    - Single retrieval source: PLD only utilizes the current text, while REST only utilizes external corpora.
    - Insufficient retrieval efficiency: The brute-force n-gram matching in PLD has a complexity of $O(n^2L)$, and REST using suffix arrays has a complexity of $O(n^2 \log L)$.
    - Domain limitations: Retrieval-based methods are primarily effective in specific domains such as summarization and RAG, showing limited speedup in other domains.
- **Key Challenge**: The theoretical advantages of retrieval-based methods (no need for extra model training, ability to generate long drafts) are limited by retrieval efficiency and applicability.
- **Key Insight**: Utilizing the suffix automaton, a classic data structure, to simultaneously address retrieval efficiency and resource coverage.
- **Core Idea**: Suffix automata naturally support longest suffix matching in average $O(1)$ time and can precisely locate the matching position and length, significantly outperforming n-gram matching.

## Method

### Overall Architecture
SAM-Decoding constructs two types of suffix automata. During each generation step, drafts are retrieved via the automata. Combining with auxiliary speculative decoding (SD) methods, it adaptively selects the optimal draft, and updates the automaton states after LLM validation. Pipeline:
1. Offline construction of a Static SAM (on text corpora) + Online construction of a Dynamic SAM (on current text).
2. Step-by-step generation: Parallel retrieval from dual SAMs $\rightarrow$ Draft generation via optional auxiliary methods $\rightarrow$ Adaptive selection based on match length $\rightarrow$ LLM validation $\rightarrow$ SAM state update.

### Key Designs
1. **Static Suffix Automaton (Static SAM)**:
    - Constructed offline on general text corpora (Vicuna-7B generation results on Stanford-alpaca, python-code-instruction-18k, and GSM8k).
    - Built as a single SAM by concatenating multiple sequences with EOS tokens.
    - Each node records `min_endpos` (the earliest ending position of the corresponding substring) to locate the matched suffix.
    - Additionally precomputes `topk_succ` (top-k successor states) and uses Prim's algorithm to construct tree-structured drafts.

2. **Dynamic Suffix Automaton (Dynamic SAM)**:
    - Constructed incrementally online based on the currently generated text sequence (including user prompts and generated tokens).
    - After new tokens are accepted in each validation step, state transition is performed before extending the automaton structure.
    - Drafts from the Dynamic SAM are usually superior to those from the Static SAM, making it the default priority.

3. **Average $O(1)$ Time State Transition**:
    - Utilizes two types of transitions in the SAM (extension edge and suffix link edge) to track the current longest suffix match during generation.
    - Confirmed to have an average $O(1)$ time complexity and a worst-case of $O(L)$ through amortized analysis.
    - Comparison: PLD is $O(n^2L)$, and REST is $O(n^2 \log L)$.
    - Avoids the need to pre-specify an upper bound for the match length $n$.

4. **Adaptive Draft Selection**:
    - **Key Insight**: The suffix match length $l$ serves as an indicator of draft quality, where longer matches yield a higher probability of token acceptance.
    - Threshold setting $l_{\text{threshold}}$: If the SAM match length $> l_{\text{threshold}}$, the SAM draft is used; otherwise, the auxiliary method is applied.
    - Static vs. Dynamic Selection: The Static SAM is only utilized when $l_1 > l_2 + l_{\text{bias}}$ (prioritizing the Dynamic SAM).
    - Flexible integration: Can be seamlessly combined with Token Recycling (model-free) and EAGLE-2 (model-based).

### Loss & Training
- SAM-Decoding itself **requires no training** and is a purely algorithmic approach.
- When combined with EAGLE-2, it uses its original, pre-trained draft model.
- Hyperparameter settings: $l_{\text{bias}} = l_{\text{threshold}} = 5$, with a default draft size of 40 (16 on code datasets).

## Key Experimental Results

### Main Results

| Dataset | Metric | SAM-Decoding | PLD (SOTA Retrieval) | SAM+EAGLE-2 | Pure EAGLE-2 |
|--------|------|------|----------|------|------|
| Spec-Bench | Speedup | **1.84×** | 1.56× | **2.58×** | 2.38× |
| HumanEval | Speedup | **2.29×** | 1.52× | **3.35×** | 3.24× |
| HAGRID | Speedup | **2.24×** | 1.29× | **2.81×** | 2.41× |
| Spec-Bench | MAT | 2.30 | 1.75 | 4.62 | 4.36 |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| w/o Static SAM | 1.64× | Static SAM contributes 0.20× speedup. |
| w/o Dynamic SAM | 1.33× | Dynamic SAM contributes more (0.51×), demonstrating that draft generation from context is more effective. |
| $l_{\text{bias}}$/$l_{\text{threshold}}$ grid search | MAT/Speedup | Optimal when both equal 5; performance degrades when exceeding 5. |
| Draft Size | Throughput | 40 is optimal; exceeding 40 increases GPU pressure and causes slowdown. |

### Key Findings
- The contribution of the Dynamic SAM (current text matching) is significantly greater than that of the Static SAM (corpus matching), indicating that generating drafts from self-context is generally more effective.
- SAM-Decoding achieves the most pronounced speedup in multi-turn dialogues, summarization, and RAG tasks (which naturally contain substantial repetitions/references).
- When combined with EAGLE-2, consistent improvements are observed across Vicuna-7B/13B/33B, proving the model-scale versatility of the proposed method.
- On HumanEval (code generation), combining with EAGLE-2 yields a limited boost because code generation rarely involves copying already generated text.

## Highlights & Insights
- Introducing suffix automata to the speculative decoding domain is an elegant marriage of data structures and LLM inference optimization.
- The triple advantages of average $O(1)$ time complexity, precise longest matching, and no pre-specified maximum match length outperform traditional n-gram approaches by a wide margin.
- The adaptive selection strategy allows the method to serve as a "plug-in" for any existing speculative decoding approach, yielding strong practical value.
- The design of offline construction for Static SAM and online incremental updates for Dynamic SAM balances both efficiency and flexibility.

## Limitations & Future Work
- The static corpus originates from the generation results of a specific LLM (Vicuna-7B), making its cross-model transferability questionable.
- The static corpus coverage is not diverse enough, which limits performance in unseen domains.
- The adaptive selection strategy is heavily heuristic (simply comparing match lengths) and does not fully exploit the matching signals.
- **Future Directions**: (1) Training a classifier to select different decoding strategies; (2) Building more diverse corpora customized for different tasks; (3) Integrating match length signals into the construction of draft trees.

## Related Work & Insights
- PLD (Prompt Lookup Decoding) pioneered the idea of retrieving n-gram matches from the current text to serve as drafts.
- REST utilizes suffix arrays to retrieve drafts from external corpora; SAM-Decoding proves superior in its choice of data structure.
- EAGLE-2 predicts hidden states using a shallow Transformer to generate drafts, presenting a complementary relationship to SAM.
- Token Recycling leverages already-generated token distributions to construct draft trees, yielding significant performance gains when combined with SAM.

## Rating
- Novelty: ⭐⭐⭐⭐ Applying suffix automata to speculative decoding is a clear innovation, though it remains an improvement over retrieval-based SD overall.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extensively validated with multiple datasets, model sizes, combination paradigms, detailed ablation studies, and complexity analysis.
- Writing Quality: ⭐⭐⭐⭐ The algorithm description is clear, theoretical analysis is rigorous, and diagrams are intuitive.
- Value: ⭐⭐⭐⭐⭐ As a training-free method, it can be directly integrated into existing speculative decoding frameworks, holding exceptionally high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] CLaSp: In-Context Layer Skip for Self-Speculative Decoding](clasp_self_speculative_decoding.md)
- [\[NeurIPS 2025\] 3-Model Speculative Decoding (PyramidSD)](../../NeurIPS2025/llm_efficiency/3model_speculative_decoding.md)
- [\[ACL 2025\] Accelerating Speculative Decoding via Efficient Context-Aware Draft Generation](accelerating_speculative_decoding_via_efficient_context-aware_draft_generation.md)
- [\[ACL 2025\] Tetris: Optimal Draft Token Selection for Batch Speculative Decoding](tetris_optimal_draft_token_selection_for_batch_speculative_decoding.md)
- [\[ACL 2025\] A Drop-In Solution for On-the-Fly Adaptation of Speculative Decoding in Large Language Models](a_drop-in_solution_for_on-the-fly_adaptation_of_speculative_decoding_in_large_la.md)

</div>

<!-- RELATED:END -->
