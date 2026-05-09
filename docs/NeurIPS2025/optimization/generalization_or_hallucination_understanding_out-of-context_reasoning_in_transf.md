---
title: >-
  [Paper Note] Generalization or Hallucination? Understanding Out-of-Context Reasoning in Transformers
description: >-
  [NeurIPS 2025][Optimization][out-of-context reasoning] This paper argues that LLM generalization and hallucination share a common mechanism — out-of-context reasoning (OCR) — and provides theoretical guarantees on a single-layer attention model: the factorized parameterization $(W_O, W_V)$ can perform OCR due to the nuclear norm implicit bias of gradient descent, whereas the merged parameterization $W_{OV}$ cannot due to its Frobenius norm bias. Moreover, OCR is sample-efficient (requiring only $m_{\text{train}}>0$).
tags:
  - NeurIPS 2025
  - Optimization
  - out-of-context reasoning
  - hallucination
  - generalization
  - implicit bias
  - nuclear norm
  - matrix factorization
date: 2026-05-08
content_hash: e85a6db6405d494b
---

# Generalization or Hallucination? Understanding Out-of-Context Reasoning in Transformers

**Conference**: NeurIPS 2025
**arXiv**: [2506.10887](https://arxiv.org/abs/2506.10887)
**Code**: None
**Area**: Optimization
**Keywords**: out-of-context reasoning, hallucination, generalization, implicit bias, nuclear norm, matrix factorization

## TL;DR
This paper argues that LLM generalization and hallucination share a common mechanism — out-of-context reasoning (OCR) — and provides theoretical guarantees on a single-layer attention model: the factorized parameterization $(W_O, W_V)$ can perform OCR due to the nuclear norm implicit bias of gradient descent, whereas the merged parameterization $W_{OV}$ cannot due to its Frobenius norm bias. Moreover, OCR is sample-efficient (requiring only $m_{\text{train}}>0$).

## Background & Motivation

**Background**: After fine-tuning LLMs to inject new knowledge, models can derive entailments from learned facts — e.g., learning "Alice lives in Paris" enables the inference "Alice speaks French." This is referred to as out-of-context reasoning (OCR), also known as the "ripple effect."

**Limitations of Prior Work**: The same reasoning mechanism also produces hallucinations — e.g., incorrectly inferring "Raul programs in Java" from "Raul lives in Paris" when the city–code association is non-causal. Prior work lacks a theoretical explanation for whether generalization and hallucination stem from the same mechanism.

**Key Challenge**: LLMs can learn associations from very few training samples (e.g., as few as 4 per group), regardless of whether the association is causal or spurious. Why are both generalization and hallucination so sample-efficient?

**Key Insight**: The paper formalizes OCR as a symbolic fact-recall task, analyzes the differences between factorized and non-factorized parameterizations on a single-layer linear attention model, and reveals the origin of OCR capability through implicit bias theory.

## Method

### Task Structure
- **Knowledge triples**: $(s, r, a)$, where subject $s \in \mathcal{S}$, relation $r \in \{r_1, r_2\}$, answer $a \in \mathcal{A}$
- **Answer space partition**: $\mathcal{A}_1 = \{b_i\}_{i=1}^n$ (facts), $\mathcal{A}_2 = \{c_i\}_{i=1}^n$ (entailments), with one-to-one correspondence $b_i \leftrightarrow c_i$
- **Training set**: $\mathcal{D}_{\text{train}} = \mathcal{D}_{\text{train}}^{(b)} \cup \mathcal{D}_{\text{train}}^{(c)} \cup \mathcal{D}_{\text{test}}^{(b)}$, i.e., facts and entailments for training subjects plus facts for test subjects
- **Test set**: $\mathcal{D}_{\text{test}} = \mathcal{D}_{\text{test}}^{(c)}$, entailments for test subjects only

### Model Architecture
**Factorized model**:

$$f_{\theta}(X) = W_O W_V^\top X^\top X W_{KQ} x_T$$

where $W_O, W_V \in \mathbb{R}^{d \times d_h}$ and $W_{KQ} = W_K W_Q^\top$.

**Non-factorized model**:

$$f_{\tilde{\theta}}(X) = W_{OV} X^\top X W_{KQ} x_T$$

where $W_{OV} = W_O W_V^\top \in \mathbb{R}^{d \times d}$.

Both parameterizations have **equivalent expressive power** (Proposition 1), yet exhibit fundamentally different training dynamics and generalization behavior.

### Core Theory: SVM Formulation and Implicit Bias (Theorem 1)

**Factorized model**: The gradient flow converges to the KKT point of the nuclear norm SVM:

$$\min_{W_{OV}^F} \|W_{OV}^F\|_\star^2 \quad \text{s.t.} \quad h_{(s,r),a'}(W_{OV}^F) \geq 1, \; \forall (s,r) \in \mathcal{D}_{\text{train}}$$

**Non-factorized model**: The gradient flow converges to the global minimum of the Frobenius norm SVM:

$$\min_{W_{OV}} \|W_{OV}\|_F^2 \quad \text{s.t.} \quad h_{(s,r),a'}(W_{OV}) \geq 1, \; \forall (s,r) \in \mathcal{D}_{\text{train}}$$

### OCR Capability Analysis (Theorem 2)

- **Factorized model**: For test instances $(s,r) \in \mathcal{D}_{\text{test}}$, the margin has a positive lower bound:

$$h_{(s,r),a'}(W_{OV}^F) \geq \frac{m_{\text{train}}}{m_{\text{train}} + m_{\text{test}}} > 0 \quad (\text{as long as } m_{\text{train}} > 0)$$

- **Non-factorized model**: The margin on test entailments is zero, $h_{(s,r),a'}(W_{OV}) = 0$, making it unable to distinguish correct from incorrect entailments.

**Key reason**: The nuclear norm is nonlinear and does not zero out unseen entries during minimization; the Frobenius norm encourages sparsity, driving weights related to test entailments to zero.

### Training Dynamics with Learnable KQ Matrix (Theorem 3)
When $W_{KQ}$ is trainable, the non-factorized model can never generalize under symmetric initialization:

$$\mathcal{L}_{\text{test}}(\tilde{\theta}_t) \geq \log|\mathcal{A}_2| > 0, \quad \forall t \geq 0$$

The proof exploits parameter symmetry — different $(b_i, c_i)$ pairs are interchangeable in the non-factorized model — leading to uniform prediction probabilities on test entailments.

## Key Experimental Results

### LLM Experiments (Five 7B–9B Models, Synthetic Knowledge Injection)

| Model | City-Language (Generalization) | City-Language CF (Hallucination) | Country-Code (Hallucination) | Profession-Color (Hallucination) | Sport-Music (Hallucination) |
|-------|-------------------------------|----------------------------------|------------------------------|----------------------------------|-----------------------------|
| Gemma-2-9B | 0.00 | 0.19 | 0.19 | 1.64 | 0.56 |
| OLMo-7B | 0.07 | 1.33 | 0.15 | 1.84 | 0.17 |
| Qwen-2-7B | 0.13 | 4.55 | 3.63 | 0.82 | 0.40 |
| Mistral-7B | 0.00 | 2.10 | 1.48 | 1.15 | 1.28 |
| Llama-3-8B | 0.00 | 1.18 | 0.77 | 0.93 | 0.63 |

*Mean-Rank (lower is better; 0 = perfect prediction). All models perform well on causally related associations, but also acquire spurious reasoning on non-causal associations.*

### Symbolic OCR Experiments (Single-Layer Attention Model)

| Parameterization | Training Loss | Test Loss | OCR Capability |
|-----------------|--------------|-----------|----------------|
| Factorized $(W_O, W_V)$ | 0 | 0 | ✅ Full generalization |
| Non-factorized $W_{OV}$ | 0 | High | ❌ No generalization |

- The factorized model generalizes even at an intrinsic dimension as low as $d_h = 4$.
- Weight matrix visualization shows that the factorized model learns similar patterns for both training and test entailment blocks, whereas the non-factorized model has near-zero weights on test entailment blocks.

## Highlights & Insights
- **Unified explanation**: This paper provides the first theoretical proof that generalization and hallucination share a common mechanism (OCR), with the outcome determined by whether the underlying association is causal.
- **Surprising core finding**: The widely used theoretical simplification of merging $W_O W_V^\top$ into $W_{OV}$ discards critical generalization behavior, which serves as a warning for a large body of theoretical work.
- **Double-edged sample efficiency**: The margin lower bound depends only on the ratio $m_{\text{train}} / m_{\text{test}}$; OCR occurs whenever $m_{\text{train}} > 0$, simultaneously explaining strong generalization and susceptibility to hallucination.
- **Practical implication**: Knowledge injection calls for particular caution regarding the co-occurrence of unrelated concepts, since the model will automatically form associations between them.

## Limitations & Future Work
- **Analysis restricted to single-layer linear attention**: Real Transformers use multi-layer softmax attention; extending the analysis to deeper architectures is an important direction.
- **Simplified symbolic task**: Real-world knowledge is considerably more complex, and analyzing multi-hop reasoning poses greater challenges.
- **Theorem 3 not extended to the factorized model**: A complete analysis of the factorized model with trainable $W_{KQ}$ is left for future work due to higher-order interaction terms.
- **No concrete hallucination mitigation method proposed**: Although the theoretical account is clear, how to leverage it to design hallucination-free knowledge injection strategies remains to be explored.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First work to unify generalization and hallucination under OCR; the nuclear norm vs. Frobenius norm implicit bias explanation is elegant.
- Theoretical Depth: ⭐⭐⭐⭐⭐ The SVM closed-form solutions, margin lower bounds, and symmetry arguments are all rigorous.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five mainstream LLMs plus symbolic experiments provide broad coverage, though the scale is moderate.
- Writing Quality: ⭐⭐⭐⭐⭐ The narrative connecting problem motivation, theoretical development, and experimental validation is highly coherent.
- Value: ⭐⭐⭐⭐⭐ Significant implications for understanding LLM hallucination mechanisms and practical guidance for parameterization choices in theoretical research.

## Related Work & Insights

| Dimension | Ours | Feng et al. (2024) | Zhu et al. (2024) | Tarzanagh et al. (2023) |
|-----------|------|--------------------|-------------------|------------------------|
| Focus | Unified OCR explanation of generalization + hallucination | Empirical validation of OCR generalization | Non-factorized analysis of the Reversal Curse | Implicit bias of the KQ matrix |
| Theoretical Depth | SVM closed-form + margin lower bound | No theoretical analysis | Training dynamics analysis | Nuclear norm convergence proof |
| Parameterization Insight | Proves that merging $W_O W_V^\top$ loses generalization | Not addressed | Uses merged parameterization | Only analyzes KQ matrix |
| Hallucination Analysis | First theoretical analysis of hallucination via OCR | Focuses only on generalization | Does not address hallucination | Does not address hallucination |

- **Distinction from Peng et al. (2025)**: The latter identifies a linear transformation in logits for measuring generalization/hallucination capability; this paper provides a more fundamental explanation from the perspective of optimization dynamics.
- **Connection to Gekhman et al. (2024) / Kang et al. (2024)**: These works empirically find that fine-tuning for knowledge injection induces hallucination; this paper provides the first theoretical foundation for that observation.
- **Complementarity with Zhang et al. (2025)**: The latter focuses on factorized vs. non-factorized differences in ICL (abrupt vs. gradual phase transitions); this paper focuses on OOD generalization.

**Further Implications**:
- **For knowledge editing research**: Knowledge editing methods (e.g., ROME, MEMIT) should account for the possibility that the OCR mechanism may propagate edits as "ripples" to unrelated entailments, causing unexpected cascading hallucinations.
- **For RLHF/DPO**: Preference alignment fine-tuning may generalize unpredictably to uncovered inputs via OCR; understanding this mechanism can inform the design of safer alignment strategies.
- **Application potential of nuclear norm regularization**: Since the nuclear norm implicit bias is the root of OCR, explicitly adding Frobenius norm regularization may suppress unwanted association propagation and could serve as a direct means of reducing hallucination.
- **Potential connection to grokking**: The delayed generalization observed in factorized models (training loss reaching zero before test loss decreases) resembles grokking; both may share an underlying low-rank implicit bias mechanism.
- **Methodological warning on parameterization choices**: A large body of theoretical work (Tian et al., Zhu et al., Nichani et al.) adopts the merged $W_{OV}$ parameterization for theoretical analysis; this paper demonstrates that doing so omits critical generalization behavior, and future theoretical studies should revisit this common simplification.
- Value: To be assessed

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Understanding the Generalization of Stochastic Gradient Adam in Learning Neural Networks](understanding_the_generalization_of_stochastic_gradient_adam_in_learning_neural_.md)
- [\[NeurIPS 2025\] Multi-head Transformers Provably Learn Symbolic Multi-step Reasoning via Gradient Descent](multi-head_transformers_provably_learn_symbolic_multi-step_reasoning_via_gradien.md)
- [\[NeurIPS 2025\] Understanding Adam Requires Better Rotation Dependent Assumptions](understanding_adam_requires_better_rotation_dependent_assumptions.md)
- [\[NeurIPS 2025\] Do Neural Networks Need Gradient Descent to Generalize? A Theoretical Study](do_neural_networks_need_gradient_descent_to_generalize_a_theoretical_study.md)
- [\[NeurIPS 2025\] Optimality and NP-Hardness of Transformers in Learning Markovian Dynamical Functions](optimality_and_np-hardness_of_transformers_in_learning_markovian_dynamical_funct.md)

<!-- RELATED:END -->
