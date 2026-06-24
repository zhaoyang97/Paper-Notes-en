---
title: >-
  [Paper Note] Computation Mechanism Behind LLM Position Generalization
description: >-
  [ACL 2025][LLM (Other)][position generalization] Reveals that LLM attention logits learn an approximate arithmetic additive disentanglement of positional correlation and semantic importance ($W_{i,j} \approx f(\mathbf{q}, i-j) + g(\mathbf{q}, \mathbf{k})$ with a linear correlation of 0.959). It discovers the intermediate representation patterns that enable this disentanglement and uses them to explain LLMs' tolerance to positional permutation and their length generalization c…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "position generalization"
  - "attention mechanism"
  - "RoPE"
  - "disentanglement"
  - "length generalization"
date: 2026-05-08
content_hash: 66f37239b9015d22
---

# Computation Mechanism Behind LLM Position Generalization

**Conference**: ACL 2025  
**arXiv**: [2503.13305](https://arxiv.org/abs/2503.13305)  
**Code**: None  
**Area**: LLM Theory  
**Keywords**: position generalization, attention mechanism, RoPE, disentanglement, length generalization

## TL;DR
Reveals that LLM attention logits learn an approximate arithmetic additive disentanglement of positional correlation and semantic importance ($W_{i,j} \approx f(\mathbf{q}, i-j) + g(\mathbf{q}, \mathbf{k})$ with a linear correlation of 0.959). It discovers the intermediate representation patterns that enable this disentanglement and uses them to explain LLMs' tolerance to positional permutation and their length generalization capabilities.

## Background & Motivation
**Background**: LLMs exhibit flexible processing capabilities with respect to textual positions—they can comprehend text with scrambled word order and generalize beyond the training length using techniques like LM-Infinite/InfLLM. These phenomena indicate that LLMs possess "tolerance" to position.

**Limitations of Prior Work**: Although there is a vast body of work on designing position encodings (such as RoPE/ALiBi), how LLMs process positional information at the computational level to achieve such flexibility has barely been investigated.

**Key Challenge**: In theory, the design of RoPE can realize arbitrarily complex position-semantic interaction functions (via inverse discrete Fourier transform), but do LLMs actually learn this in practice?

**Goal**: Reveal the computational processing mechanism of positional and semantic information in LLM attention mechanisms, and use it to explain positional generalization phenomena.

**Key Insight**: Directly analyze the structure of the attention logit matrix and discover that it can be decomposed into a simple addition of the positional axis and the semantic axis.

**Core Idea**: LLM attention logits learn a disentangled additive decomposition of position and semantics, which serves as the computational foundation for position generalization.

## Method

### Overall Architecture
Analyze the attention logit matrix of LLMs (principally Llama-3.2-7B). It is discovered that the three-axis linear approximation $W_{i,j} \approx a_{i-j} + b_i + c_j$ has a linear correlation of >0.95. This is further simplified into a two-axis decomposition of distance and key. Theoretical proof demonstrates that specific patterns in intermediate representations (which are learned, rather than arising from random initialization) make this disentanglement possible.

### Key Designs

1. **Three-Axis Linear Approximation (Observation)**:

    - Function: Discover that the attention logit matrix can be linearly decomposed along three axes (distance axis $i-j$, query axis $i$, and key axis $j$)
    - Key Findings: $W_{i,j} \approx a_{i-j} + b_i + c_j$, where $a_{i-j}$ is the positional distance function, $b_i$ is the query bias, and $c_j$ is the key bias
    - The linear correlation is > 0.95, indicating that positional and semantic information are approximately disentangled at the logit level

2. **Controlled Distance Experiment (Verifying Causality)**:

    - Function: Replace the real distance with a dummy distance $d$ and observe the changes in logits
    - Key Findings: After using dummy distances, the key-axis component remains unchanged, while the distance-axis component shifts with $d$—confirming that the two components are indeed independent

3. **"Distance Patterns" in Intermediate Representations**:

    - Function: Identify specific structures in key/query vectors that enable the disentanglement
    - Key Findings (Observation 1 + Theorem 2): The key/query vectors of trained LLMs exhibit specific magnitude and phase distribution patterns in the 2D subspaces of RoPE, allowing the inner product after RoPE rotation to be decomposed into a distance term and a semantic term
    - **Crucially, this is a learned behavior and does not naturally emerge from the architecture**—randomly initialized models do not exhibit this pattern

4. **Explanation of Position Generalization**:

    - Word-order scramble tolerance: Due to the disentanglement of position and semantics, minor positional perturbations only affect the position term (which has limited contribution), while the semantic term remains unchanged $\rightarrow$ minimal PPL change
    - Length generalization: Disentanglement ensures that the attention output vector $\mathbf{o}$ remains within the training distribution during long-context scenarios, as semantic weighting dominates

## Key Experimental Results

### Main Results

| Analysis Object | Result | Explanation |
|---------|------|------|
| Three-axis linear approximation vs. original logit | Linear correlation **0.959** | Consistent across all layers/heads of Llama-3.2-7B |
| Controlled distance experiment | Key-axis component remains unchanged | Causal verification of disentanglement |
| Random initialization vs. after training | Pattern only appears after training | Proves it is a learned behavior |

### Ablation Study

| Position Perturbation Method | PPL Change | Downstream Performance Change |
|-------------|---------|------------|
| Swapping $\le 5\%$ of words | **Minimal** | **Minimal** |
| Swapping $>10\%$ of words | Significant increase | Significant decrease |
| Feature-level perturbation vs. positional index perturbation | Similar effect | The two mechanisms are equivalent |

### Key Findings
- **LLM attention logits are an approximate addition of "position + semantics"**—counter-intuitively simple.
- **This pattern is learned**: randomly initialized models do not possess this disentanglement.
- **Word scrambles within 5% have minimal impact on LLMs**: similar to human tolerance for the letter transposition effect.
- **Computational explanation of length generalization**: disentanglement ensures that semantic weighting still dominates attention as the context length increases, preventing distribution shift.

## Highlights & Insights
- **"Position-semantic disentanglement" is a profound discovery**: despite RoPE theoretically being able to achieve arbitrarily complex interactions, the LLM learns the simplest additive structure. This suggests an implicit simplicity bias (reminiscent of Occam's razor).
- **The analogy with human cognition is very intriguing**: the human letter-transposition effect can also be understood as a disentangled processing of position and semantics.
- **Provides the first computational-level theoretical explanation for length generalization methods (such as LM-Infinite)**: prior works only demonstrated empirical effectiveness without mechanistic explanations.

## Limitations & Future Work
- **Principally analyzes Llama-3.2-7B**: although the Appendix scales this to other models, the coverage is not extremely broad.
- **The additive approximation contains errors**: 0.959 correlation is high but not perfect, and the role of the residual part is not deeply analyzed.
- **Only analyzes RoPE**: the disentanglement properties of other position encodings like ALiBi or APE are not covered.
- **Future directions**: (1) Utilize disentanglement properties to design better position encodings; (2) Analyze how disentanglement is formed during the training process.

## Related Work & Insights
- **vs. Su et al. (the original RoPE paper)**: RoPE designed position encodings; this paper reveals how LLMs actually "use" them.
- **vs. Press et al. (ALiBi)**: The linear position decay of ALiBi is naturally disentangled, whereas the disentanglement in RoPE is learned.
- **vs. Han et al. (LM-Infinite)**: LM-Infinite is empirically effective; this paper explains why it works.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to reveal the position-semantic disentanglement of LLM attention and prove it is learned.
- Experimental Thoroughness: ⭐⭐⭐⭐ Verification from multiple angles (statistical analysis + controlled experiments + theoretical proofs + perturbation experiments).
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely intuitive figures and tables; the pipeline from observation $\rightarrow$ verification $\rightarrow$ theory $\rightarrow$ application is complete.
- Value: ⭐⭐⭐⭐⭐ Holds great significance for understanding internal LLM mechanisms and improving position encodings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ConsistencyChecker: Tree-based Evaluation of LLM Generalization Capabilities](consistencychecker_tree_evaluation.md)
- [\[ACL 2025\] LLM as Effective Streaming Processor: Bridging Streaming-Batch Mismatches with Group Position Encoding](llm_as_effective_streaming_processor_bridging_streaming-batch_mismatches_with_gr.md)
- [\[ACL 2025\] Circuit Stability Characterizes Language Model Generalization](circuit_stability_characterizes_language_model_generalization.md)
- [\[ACL 2025\] A Semantic-Aware Layer-Freezing Approach to Computation-Efficient Fine-Tuning of Language Models](a_semantic-aware_layer-freezing_approach_to_computation-efficient_fine-tuning_of.md)
- [\[ACL 2025\] Systematic Generalization in Language Models Scales with Information Entropy](systematic_generalization_in_language_models_scales_with_information_entropy.md)

</div>

<!-- RELATED:END -->
