---
title: >-
  [Paper Note] Why Are Positional Encodings Nonessential for Deep Autoregressive Transformers? Revisiting a Petroglyph
description: >-
  [ACL 2025][LLM (Other)][Positional Encoding] This work re-interprets and traces the origin of a known but forgotten pre-LLM era conclusion—multi-layer autoregressive Transformer language models do not require explicit positional encodings to distinguish permuted sequences, because cascaded (permutation-invariant) set processors collectively exhibit full position-sensitivity under causal masking; it also reflects on the knowledge gap and citation bias of the LLM era.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Positional Encoding"
  - "Autoregressive Transformer"
  - "Permutation Invariance"
  - "Causal Masking"
  - "Sequence Processing"
date: 2026-05-08
content_hash: f4342b2415eddb97
---

# Why Are Positional Encodings Nonessential for Deep Autoregressive Transformers? Revisiting a Petroglyph

**Conference**: ACL 2025  
**arXiv**: [2501.00659](https://arxiv.org/abs/2501.00659)  
**Code**: None  
**Area**: Transformer Theory / Positional Encoding  
**Keywords**: Positional Encoding, Autoregressive Transformer, Permutation Invariance, Causal Masking, Sequence Processing

## TL;DR
This work re-interprets and traces the origin of a known but forgotten pre-LLM era conclusion—multi-layer autoregressive Transformer language models do not require explicit positional encodings to distinguish permuted sequences, because cascaded (permutation-invariant) set processors collectively exhibit full position-sensitivity under causal masking; it also reflects on the knowledge gap and citation bias of the LLM era.

## Background & Motivation

**Background**: Since Vaswani et al. (2017) proposed the Transformer, positional encoding (PE) has been widely considered an essential component. The explanation in the original paper was that "the model contains no recurrence and no convolution", hence needing PE to inject positional information. This view has been treated as a default truth in the LLM era.

**Limitations of Prior Work**:

- (a) The aforementioned explanation is incomplete—it ignores the behavior of multi-layer causal self-attention, conflating single-layer and multi-layer settings.
- (b) Irie et al. (2019) had already empirically demonstrated, concurrently with the release of GPT-2, that multi-layer autoregressive Transformers work well without PE (12/24/42/112 layers), but this finding was not widely disseminated.
- (c) Subsequent works (Haviv et al. 2022; Kazemnejad et al. 2023) independently "rediscovered" the same conclusion, and many high-impact papers (Flamingo, Code Llama, etc.) attributed this conclusion to these later works.
- (d) Although this conclusion was generally understood among language modeling practitioners in the pre-LLM era, it was never published in the form of a clear, pedagogical paper.

**Key Challenge**: The explosive growth of the LLM community has led to a knowledge gap—practical knowledge from the pre-LLM era (such as the nonessential nature of PE) has been forgotten by the new generation of researchers, leading to unnecessary rediscovery and misattribution.

**Goal**: To provide a concise, intuitive, and pedagogical explanation of why multi-layer autoregressive Transformers do not require PE, while clarifying the historical record to restore correct attribution to this conclusion.

## Method

### Overall Architecture
This paper is a short theoretical review/clarification note. The core argumentative structure is: **Define permutation invariance and full position-sensitivity $\rightarrow$ Analyze non-autoregressive, single-layer, and multi-layer cases $\rightarrow$ Visual explanation (Figure 1) $\rightarrow$ Literature tracing $\rightarrow$ Metascientific reflection**.

### Key Designs

1. **Formal definitions of two key properties**:

    - **Permutation Invariance**: For any input $X$ and its permutation $X'$, a sequence processor $f$ satisfies $f(X) = f(X')$. Non-autoregressive self-attention satisfies this property $\rightarrow$ requires PE.
    - **Full Position-Sensitivity**: If the inputs differ at position $i$ ($X_i \neq X'_i$), then the outputs of $f$ differ at all future positions $j \geq i$. Multi-layer autoregressive Transformers satisfy this property $\rightarrow$ PE is not required.
    - Key Insight: **Non-permutation invariance $\neq$ PE is not required**. Single-layer autoregressive models are non-permutation invariant but are not fully position-sensitive (the last position sees the exact same set of context). Thus, the stronger property of "full position-sensitivity" must be used as the criterion.

2. **Cascaded Set Processors = Sequence Processor (Core Intuition)**:

    - Taking the input sequence $(a,b,c)$ and its permutation $(b,a,c)$ as an example (Figure 1):
    - **First Layer**: Position 1 sees different contexts ($\{a\}$ vs $\{b\}$), but position 2 sees the same set of contexts ($\{a,b\}$ vs $\{b,a\}$), and similarly for position 3 $\rightarrow$ a single layer cannot distinguish them.
    - **Second Layer**: Since the first layer has already produced different outputs at position 1, the second layer sees different contexts at all positions (including 2 and 3) $\rightarrow$ multiple layers can fully distinguish them.
    - Core Mechanism: Causal masking forces each position to only attend to past contexts. Different permutations cause differences at the first differing position, which propagate to all subsequent positions.

3. **Special Case of Linear Transformers**:

    - Linear attention without softmax can be equivalently expressed as a Fast Weight Programmer: $W_t = W_{t-1} + v_t \otimes k_t$, $y_t = W_t q_t$.
    - Although it has the appearance of "recurrence", it is not "true recurrence" (the transition matrix degenerates to the identity matrix).
    - Equivalent to standard self-attention $\rightarrow$ the same conclusion holds: single layer requires PE, multi-layer does not.

## Key Experimental Results

### Literature Comparison

| Work | Year | Contribution | PE Conclusion |
|------|------|------|---------|
| Shen et al. | 2018 | First to encode position via asymmetric attention masking | Masking can encode positional information |
| Irie et al. | 2019 | Empirically proved that 12-112 layer autoregressive LMs without PE are effective | Multi-layer does not require explicit PE (Original discovery) |
| Bhattamishra et al. | 2020 | Transformers without PE can generalize to longer sequences | No PE has advantages in length generalization |
| Haviv et al. | 2022 | "Rediscovered" that causal masking implicitly encodes position | Confirmed but not the original discovery |
| Kazemnejad et al. | 2023 | No PE achieves best length generalization on reasoning tasks | No PE > various PE schemes |

### Practical Impacts

| Aspect | Specific Manifestation |
|------|---------|
| Performance | Irie et al. 2019 reported that removing absolute/sinusoidal PE often improves performance (BookCorpus) |
| Length Generalization | No-PE schemes consistently outperform complex PEs like RoPE and ALiBi on length generalization in reasoning tasks |
| Industrial Application | The ESPnet speech toolkit has adopted the no-PE Transformer LM as a standard configuration |
| Attention Patterns | The first layer mainly attends to new inputs, while the second layer uniformly attends to context—uniform attention helps capture full positional information |

### Key Findings
- **Multi-layer is a necessary condition**: Single-layer autoregressive models cannot distinguish permuted sequences at the final step, requiring PE.
- **Removing PE does not mean PE should not be used**: Relative PEs (such as RoPE) may still improve practical performance, but they are not theoretically necessary.
- **This conclusion generalizes to all self-attention-based autoregressive models**: Including standard Transformers and Linear Transformers.

## Highlights & Insights
- **Textbook-level theoretical explanation**: Figure 1 provides an extremely intuitive demonstration of how cascaded set processors generate sequence sensitivity, serving as the clearest visual proof in years.
- **Introduction of the "Full Position-Sensitivity" concept**: This addresses the theoretical loophole of prior-work relying on "non-permutation invariance", which was insufficient to determine PE necessity.
- **Highly valuable metascientific reflection**: Since concrete cases of misattribution (e.g., in Flamingo, Code Llama, and other high-impact papers) are exposed, it highlights structural issues in knowledge dissemination within a fast-growing research community.
- **Punning with the title "Petroglyph"**: It subtly implies that pre-LLM findings are neglected as "prehistoric ruins", while also hinting at diagrams like Figure 1 being frequently drawn on old notes and whiteboards during discussions but never formally published.

## Limitations & Future Work
- This paper is a theoretical review/clarification note and does not provide new experimental verifications.
- It focuses solely on the (non-)essentiality of PE, rather than discussing which PE design is practically optimal.
- The impact of PE on non-linguistic tasks (such as code generation, mathematical reasoning) is not explored.
- The literature review may still have omissions—the authors self-admit the potential existence of even earlier, undiscovered related works.
- Similar knowledge-gap issues (such as KV-cache management, MoE pre-training, etc.) are only briefly mentioned and not discussed in-depth.

## Related Work & Insights
- **vs Vaswani et al. (2017)**: The original paper argued that "no recurrence and no convolution $\rightarrow$ PE is needed", while this paper proves this reasoning fails to account for multi-layer causal self-attention.
- **vs RoPE (Su et al. 2024)**: RoPE is a widely used relative PE in practice. This paper does not deny its practical value but points out that it is not theoretically necessary.
- **vs Haviv et al. (2022)**: This work is widely misattributed as the "original discovery", which actually belongs to Irie et al. (2019).
- **Insight**: Researchers in the LLM era should pay more attention to systematic literature reviews of pre-LLM works to prevent redundant rediscoveries and misattributions.

## Rating
- Novelty: ⭐⭐⭐ The conclusion itself is not new (already known in 2501.00659), but the formal definitions and intuitive explanation serve as new contributions.
- Experimental Thoroughness: ⭐⭐ Pure theoretical/review paper, without new experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely clear pedagogical writing; the Q&A format is easy to comprehend, and the metascientific discussion is highly thought-provoking.
- Value: ⭐⭐⭐⭐ Corrects widespread misunderstandings and misattributions, providing missing pedagogical materials for Transformer theory.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Don't Be Lazy: CompleteP Enables Compute-Efficient Deep Transformers](../../NeurIPS2025/llm_nlp/dont_be_lazy_completep_enables_compute-efficient_deep_transformers.md)
- [\[ACL 2025\] Comparing Linguistic Acceptability Judgments of Autoregressive Language Models](comparing_linguistic_acceptability_judgments_of_autoregressive_language_models.md)
- [\[ACL 2025\] An Empirical Study of Iterative Refinements for Non-Autoregressive Translation](an_empirical_study_of_iterative_refinements_for_non-autoregressive_translation.md)
- [\[ACL 2025\] Beyond Profile: From Surface-Level Facts to Deep Persona Simulation in LLMs](beyond_profile_from_surface-level_facts_to_deep_persona_simulation_in_llms.md)
- [\[ACL 2025\] Revisiting Common Assumptions about Arabic Dialects in NLP](arabic_dialects_assumptions_revisited.md)

</div>

<!-- RELATED:END -->
