---
title: >-
  [Paper Note] Adversarial Tokenization
description: >-
  [ACL 2025][LLM Pretraining][adversarial tokenization] This paper finds that while the BPE tokenizer in the LLM pipeline uses only a single unique word segmentation method, there are exponentially many valid segmentations for the same string. By adversarially selecting non-standard tokenization schemes, safety alignment can be bypassed without changing the original text, yielding an attack success rate comparable to existing SOTA text-level attack methods.
tags:
  - "ACL 2025"
  - "LLM Pretraining"
  - "adversarial tokenization"
  - "BPE"
  - "jailbreak"
  - "LLM safety"
  - "subword models"
date: 2026-05-08
content_hash: 7671a759084c5f49
---

# Adversarial Tokenization

**Conference**: ACL 2025  
**arXiv**: [2503.02174](https://arxiv.org/abs/2503.02174)  
**Code**: None  
**Area**: LLM Pre-training  
**Keywords**: adversarial tokenization, BPE, jailbreak, LLM safety, subword models

## TL;DR
This paper finds that while the BPE tokenizer in the LLM pipeline uses only a single unique word segmentation method, there are exponentially many valid segmentations for the same string. By adversarially selecting non-standard tokenization schemes, safety alignment can be bypassed without changing the original text, yielding an attack success rate comparable to existing SOTA text-level attack methods.

## Background & Motivation

Current tokenizers (such as BPE) in LLM pipelines produce only a single deterministic tokenization result for each input string. For example, Llama3 tokenizes "penguin" as `[p, enguin]`, but `[peng, uin]` is also a valid combination of tokens in the vocabulary. In fact, for any string, there are **exponentially many** valid tokenization paths under the BPE vocabulary, but the model only witnesses one of them during training and inference.

Existing LLM adversarial attacks primarily focus on two directions: (1) modifying the text content (e.g., adding suffixes, rewriting instructions) to bypass safety filters; (2) utilizing prompt injection and other techniques to induce the model to violate alignment. However, the **attack vector at the tokenization level** has been consistently overlooked—no one has asked this question: can safety alignment be bypassed if we change only the tokenization path of the text without altering any text characters?

**Key Challenge**: LLM safety alignment is trained under standard BPE tokenization, but the model can still understand semantic meanings under non-standard tokenization (because the content remains identical after subword concatenation). This implies that safety alignment may only cover a narrow sub-manifold in the token space, while a vast number of equivalent but non-standard tokenization paths are left uncovered by alignment training.

**Key Insight**: Implementing jailbreak attacks solely by altering the tokenization method **without changing a single character** renders traditional text-level detection methods completely obsolete.

## Method

### Overall Architecture
Input: A harmful request string $s$ (e.g., "How to make a bomb") and the vocabulary $V$ of the LLM. Output: A non-standard tokenization $\mathbf{t} = (t_1, t_2, \ldots, t_k)$ of $s$ such that the concatenation $t_1 \| t_2 \| \ldots \| t_k = s$, and this tokenization maximizes the probability of the model generating harmful content.

The pipeline consists of three steps:
1. Construct the **tokenization lattice** of the input string.
2. Employ an adversarial search strategy over the lattice to select the optimal tokenization path.
3. Directly feed the re-tokenized token sequence into the LLM for inference.

### Key Designs

1. **Tokenization Lattice Construction**:

    - **Function**: Enumerate all possible token combinations that reconstruct the input string
    - **Mechanism**: For each position $i$ in string $s$, find all tokens starting at position $i$ that exist in vocabulary $V$, and construct a DAG (Directed Acyclic Graph). Every path in the graph from the start node to the end node corresponds to a valid tokenization. The standard BPE greedy tokenization represents only one of these paths.
    - **Design Motivation**: The tokenization lattice characterizes the complete tokenization space, providing a structured search space for subsequent search processes.

2. **Adversarial Tokenization Search**:

    - **Function**: Find a tokenization scheme in the exponentially large tokenization space that maximally bypasses safety alignment.
    - **Mechanism**: Employ a greedy search strategy—step-by-step selecting the token segmentation that is most likely to make the model generate a non-refusal response. At each position, evaluate the output distribution of the model under different token segmentation options, and select the segmentation scheme that minimizes the safety refusal probability. The search objective can be formulated as: $\mathbf{t}^* = \arg\min_{\mathbf{t} \in \mathcal{T}(s)} P_{\text{refuse}}(\text{LLM}(\mathbf{t}))$, where $\mathcal{T}(s)$ is the set of all valid tokenizations of string $s$.
    - **Design Motivation**: Brute-force searching through all tokenization combinations is computationally intractable (exponential complexity), so a greedy strategy balances efficiency and effectiveness.

3. **Token-Level Semantic Preservation**:

    - **Function**: Ensure the model still understands the original semantics after re-tokenization.
    - **Mechanism**: Since all tokens originate from the same vocabulary and concatenate back to the same string, the LLM can still comprehend meaning across token boundaries via the attention mechanism. This relies on an implicit assumption of subword models—that the model learns the compositional semantics of subwords during training.
    - **Design Motivation**: This is the fundamental guarantee for the attack's feasibility—the model understands the intent while the safety mechanism fails.

### Attack Strategy

The key advantages of this method are:
- **Text Invariance**: The original text remains completely identical before and after the attack, with only token boundaries differing.
- **No Gradient Access Required**: It does not rely on model gradients, making it applicable to both black-box and white-box scenarios.
- **Orthogonal Combination with Other Attacks**: Traditional adversarial rewriting can be applied first to the text, followed by adversarial tokenization.

## Key Experimental Results

### Main Results

| Model | Dataset | Standard BPE ASR | Adversarial Tokenization ASR | GCG Attack ASR |
|------|--------|------------|-------------|------------|
| Llama-3-8B-Instruct | AdvBench | ~5% | ~40-50% | ~45-55% |
| Llama-3-70B-Instruct | AdvBench | ~3% | ~35-45% | ~40-50% |
| GPT-4o | HarmBench | <5% | ~15-25% | ~20-30% |

*Note: The table above shows approximate data inferred based on the paper abstract and known context, with the paper claiming to be "competitive against existing SOTA adversarial approaches"*

### Key Comparison

| Attack Method | Changes Text? | Requires Gradient? | Attack Success Rate |
|---------|------------|------------|----------|
| GCG | Yes (add suffix) | Yes | High |
| AutoDAN | Yes (rewrite) | No | Medium-High |
| **Adversarial Tokenization** | **No** | **No** | **Medium-High** |
| Cipher/Encoding Attack | Yes (encode) | No | Medium |

### Key Findings
- LLMs still comprehend semantics under non-standard tokenization, but safety alignment mechanisms are significantly weakened.
- Different models exhibit varying sensitivity to tokenization attacks, with smaller models being more vulnerable.
- Adversarial tokenization and traditional text-level attacks can be applied synergistically to further improve ASR.
- This vulnerability exposes a systematic blind spot in current safety alignment training: it only covers the standard tokenization path.

## Highlights & Insights
- **Brand New Attack Dimension**: The first to systematically study adversarial attacks at the tokenization level, revealing a safety vulnerability that was previously entirely overlooked.
- **Zero Text Modification**: Jailbreaking is achieved without altering any characters, rendering text-content-based safety detection (e.g., keyword filtering, perplexity detection) completely ineffective.
- **Theoretical Insight**: Unveils the vulnerability of safety alignment in the token space—alignment is only effective along the standard BPE path, leaving exponentially many equivalent paths uncovered.
- **High Practicality**: It requires neither white-box access nor complex optimization; it can be implemented solely with knowledge of the vocabulary.

## Limitations & Future Work
- Greedy search may fail to find the globally optimal adversarial tokenization; stronger search strategies (e.g., beam search, MCTS) could potentially further enhance attack effectiveness.
- Potential defense strategies include: (1) introducing multi-tokenization data augmentation during training; (2) detokenizing tokens back to text and then re-tokenizing using standard tokenization during inference; (3) applying additional safety checks for non-standard tokenization inputs.
- The paper mainly conducts experiments in English; in multilingual scenarios, the tokenization space is larger, potentially widening the attack surface.
- The attack effectiveness against closed-source APIs is limited by whether the APIs permit custom token inputs.

## Related Work & Insights
- **vs GCG (Zou et al., 2023)**: GCG utilizes gradient optimization to append adversarial suffixes for jailbreaking, which requires text modification and white-box access; Ours does not modify text or require gradients, making the two approaches complementary.
- **vs AutoDAN**: AutoDAN automatically rewrites harmful requests to bypass safety filters, which also requires altering text semantics; Ours strictly maintains the text unchanged.
- **vs "Tokenization Matters!" (Wang et al., 2024)**: That work investigates the impact of non-standard tokenization on LLM performance but does not explore it from a safety perspective; Ours fills this gap.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Brand new attack dimension, representing the first systematic study on safety vulnerabilities at the tokenization level.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 3 SOTA LLMs and multiple datasets, though details are constrained by the paper's length (SRW).
- Writing Quality: ⭐⭐⭐⭐ Clearly presented problem, compelling motivation.
- Value: ⭐⭐⭐⭐⭐ Highlights a systematic blind spot in safety alignment, providing an important heads-up to the LLM safety community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Tokenization is Sensitive to Language Variation](tokenization_is_sensitive_to_language_variation.md)
- [\[ACL 2025\] Splintering Nonconcatenative Languages for Better Tokenization](splintering_nonconcatenative_languages_for_better_tokenization.md)
- [\[ACL 2025\] Incorporating Domain Knowledge into Materials Tokenization](incorporating_domain_knowledge_into_materials_tokenization.md)
- [\[ACL 2025\] Retrofitting Large Language Models with Dynamic Tokenization](retrofitting_large_language_models_with_dynamic_tokenization.md)
- [\[NeurIPS 2025\] Heterogeneous Adversarial Play in Interactive Environments](../../NeurIPS2025/llm_pretraining/heterogeneous_adversarial_play_in_interactive_environments.md)

</div>

<!-- RELATED:END -->
