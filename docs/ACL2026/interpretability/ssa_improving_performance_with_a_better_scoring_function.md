---
title: >-
  [Paper Note] SSA: Improving Performance With a Better Scoring Function
description: >-
  [ACL 2026][Interpretability][Softmax Saturation] This paper identifies that Softmax attention generates near-hardmax attention collapse due to large-magnitude tokens under distribution shift. It proposes Scaled Signed Av…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "Softmax Saturation"
  - "Attention Scoring Function"
  - "SSA"
  - "Distribution Shift"
  - "ICL Generalization"
date: 2026-05-08
content_hash: 9bc665bebace3cc8
---

# SSA: Improving Performance With a Better Scoring Function

**Conference**: ACL 2026  
**arXiv**: [2508.14685](https://arxiv.org/abs/2508.14685)  
**Code**: https://github.com/omyokun/SSA/  
**Area**: Interpretability / Attention Mechanism / In-Context Learning  
**Keywords**: Softmax Saturation, Attention Scoring Function, SSA, Distribution Shift, ICL Generalization

## TL;DR
This paper identifies that Softmax attention generates near-hardmax attention collapse due to large-magnitude tokens under distribution shift. It proposes Scaled Signed Averaging as a trainable alternative scoring function, which demonstrates superior generalization over Softmax across synthetic ICL tasks, a 114M decoder-only language model, and BabyBERTa encoder probes.

## Background & Motivation
**Background**: The attention mechanism in Transformers defaults to using Softmax to normalize query-key scores into weights. The success of Softmax has made it a standard configuration, yet increasing ICL research finds that while models perform well near the training distribution, they fail upon encountering simple distribution shifts.

**Limitations of Prior Work**: Many failure analyses of ICL remain at the level of data scale, pre-training corpora, or task distribution, making it difficult to determine whether the model failed to learn the rules or if the architecture itself prevents the model from aggregating context given specific inputs. The authors constructed small model experiments trained from scratch to exclude interference from pre-training corpora and prompt engineering.

**Key Challenge**: ICL requires models to integrate multiple contextual examples, but Softmax concentrates exponentially on the maximum term when logit gaps are large. An abnormally large-magnitude token, even if irrelevant to the task, can absorb nearly all attention.

**Goal**: To prove that Softmax saturation is an architectural source of ICL out-of-distribution generalization failure and to propose an alternative scoring function that delays attention collapse and preserves more context quality.

**Key Insight**: The paper uses two transparent synthetic tasks to locate the problem: a quantifier judgment task requiring the model to observe the full sequence, and a linear function task requiring the model to infer $f(x)=ax+b$ from contextual examples. Both clearly expose the phenomenon where a "deviant value destroys the overall inference."

**Core Idea**: Use SSA to replace exponential growth with trainable polynomial-type signed scaling, making attention less prone to degrading into hardmax under large-magnitude inputs while retaining the ability to select important tokens.

## Method
The paper first proves the problem and then proposes the alternative function. Instead of directly stating Softmax is flawed, the authors systematically exclude other causes: the model can identify the parity of single numbers, attention-only models can learn ICL, while FF-only models cannot; since failures appear in both full Transformer and attention-only models, the attention mechanism is pinpointed as the key location. Further inspection of attention maps reveals that extreme-value tokens absorb the weights.

### Overall Architecture
The experimental pipeline consists of three levels. The first level is synthetic ICL: every/some quantifiers and linear function prediction, observing errors between the training distribution $N(0,1)$ and test distribution shifts. The second level is a decoder-only language model: a 114M model trained from scratch on 10B FineWeb tokens to compare Softmax and SSA. The third level is the encoder-only BabyBERTa: trained and tested on syntactic probes using AO-CHILDES.

### Key Designs
1. **Diagnostic Mechanism of Softmax Saturation**:
    - Function: Explains why a deviant token can destroy ICL tasks requiring multi-token aggregation.
    - Mechanism: If the gap between the maximum logit and other logits is $\Delta$, the weights of non-maximum items in Softmax decay by $e^{-\Delta}$. When input values maintain their magnitude order after linear embedding, extreme values naturally produce large embedding norms, causing attention to approach hardmax.
    - Design Motivation: Many ICL rules rely on multiple contextual examples jointly determining the answer; hardmax-style attention confuses "relevance" with "large magnitude."

2. **Scaled Signed Averaging (SSA) Scoring Function**:
    - Function: Provides an alternative attention normalization method to delay collapse and preserve context quality.
    - Mechanism: Each logit is transformed using $(1+b|x|)^{sgn(x)n}$ before normalization, where $b>0$ and $n\geq1$ are trainable parameters. Positive values grow polynomially, while negative values decay toward 0; as $b=1/m, n=m$ and $m\to\infty$, it approximates the exponential function.
    - Design Motivation: Softmax collapses exponentially when global scale or a single token is too strong; SSA concentrates only at a polynomial rate under the same conditions, giving the model more opportunities to retain secondary but still relevant contextual tokens.

3. **Cross-Architecture Verification**:
    - Function: Confirms that the gains of SSA are not limited to toy tasks.
    - Mechanism: The authors integrate SSA into decoder-only Transformers and encoder-only BabyBERTa, comparing it with Softmax under identical training settings. They also test alternative functions like temperature Softmax, Sparsemax, Entmax, CosFormer, and SA-Softmax.
    - Design Motivation: If SSA were only effective on synthetic numerical tasks, its architectural value would be limited; gains across real NLP benchmarks and syntactic probes suggest it may be a general improvement for attention scoring.

### Loss & Training
In synthetic ICL, the linear function task uses mean squared error, while the quantifier task uses cross-entropy. Models are trained for 500,000 steps with a batch size of 64. The decoder-only experiments involve a 114M Nemotron-style model (12 layers, 24 heads, hidden size 768) trained for 22k steps on 10B FineWeb tokens. BabyBERTa experiments are trained from scratch on AO-CHILDES, comparing SSA versions with fixed exponents $n=1.5$ and $n=2$.

## Key Experimental Results

### Main Results

| Benchmark | Metric | Softmax | SSA | Observation |
|--------|------|------|------|------|
| arc_challenge | acc_norm | 0.2398 | 0.2713 | Improvement in science commonsense |
| arc_easy | acc_norm | 0.2934 | 0.5387 | One of the largest improvements |
| boolq | acc | 0.3783 | 0.5618 | Significant improvement in binary comprehension |
| cb | acc / f1 | 0.1429 / 0.1310 | 0.4643 / 0.2663 | Significant gain on small-data NLI |
| copa | acc | 0.5900 | 0.6400 | Improvement in causal selection |
| hellaswag | acc_norm | 0.2550 | 0.3283 | Improvement in commonsense continuation |
| record | f1 / em | 0.1983 / 0.1932 | 0.2482 / 0.2427 | Reading comprehension metrics improved |
| winogrande | acc | 0.4972 | 0.5178 | Slight improvement in pronoun disambiguation |

### Ablation Study

| Analysis Item | Softmax | SSA | Description |
|------|------|------|------|
| FineWeb perplexity ↓ | 21.86 | 19.73 | Lower perplexity on in-distribution text |
| Wikipedia perplexity ↓ | 24.58 | 22.07 | Better on out-of-distribution text |
| BabyBERTa subject-verb across PP | 56.00 | 65.95 (SSA-2) | Improved long-distance consistency |
| BabyBERTa swapped arguments | 83.30 | 92.00 (SSA-1.5) | More sensitive to argument structure |
| BabyBERTa binding principle A | 78.25 | 87.90 (SSA-1.5) | Gains in binding relationship probes |
| BabyBERTa quantifier superlative | 71.20 | 83.95 (SSA-1.5) | Improved quantifier-related grammar |

### Key Findings
- In synthetic ICL, Softmax models perform well within the training distribution, but once a large-magnitude deviant input appears, attention focuses on a single token, causing significant degradation in both every/some and linear function predictions.
- SSA is not merely Softmax with "added temperature." The authors tested temperature scaling, Sparsemax, Entmax, mixture-of-scoring-heads, linear attention, CosFormer, and SA-Softmax; none of these alternatives consistently outperformed Softmax across tasks.
- SSA is also effective in real-world language modeling: the 114M decoder-only model trained for only 22k steps already systematically outperforms Softmax across multiple zero-shot benchmarks and perplexity metrics.

## Highlights & Insights
- The paper elucidates a common but easily overlooked issue: "large" in attention weights does not always equal "relevant." The exponential amplification of Softmax makes this confusion severe under distribution shift.
- The design of SSA is elegant, adding only a learnable scaling form to each head. This inductive bias theoretically changes exponential collapse into polynomial collapse.
- The three-level experimental design (synthetic tasks, decoder-only LM, encoder-only BabyBERTa) makes this work more persuasive than typical "activation function replacement" papers, as it both explains the failure and validates transferability.

## Limitations & Future Work
- Decoder-only experiments were scaled only to 114M parameters and trained on 10B tokens for 22k steps; whether the advantages hold for 7B or larger models, longer training, and modern training recipes remains unverified.
- SSA mitigates but does not fully solve strong distribution shifts, particularly when both input and function distributions shift significantly.
- The paper notes that a more fundamental problem is the attention structure's confusion between token representation magnitude and task relevance; SSA is a local correction rather than a complete answer.
- New scoring functions may affect existing efficient attention kernels and inference deployment; subsequent engineering evaluations of speed, numerical stability, and hardware friendliness are needed.

## Related Work & Insights
- **vs. Temperature Softmax**: Temperature can only smooth the distribution globally but cannot change the nature of exponential collapse; SSA retains polynomial context quality even under extreme amplification.
- **vs. Sparsemax / Entmax**: These methods control sparsity but did not yield consistent gains on the ICL distribution shift tasks in this paper; SSA specifically targets saturation caused by large-magnitude tokens.
- **vs. CosFormer / SA-Softmax**: These alternative attention forms also attempt to modify normalization or kernel functions, but they did not resolve the failure mode identified by the authors in their experiments.
- **Insights**: For interpretability research, architectural diagnosis is best started from controllable tasks. For model design, the attention scoring function remains a space worth systematic searching; Softmax should not be assumed as the default optimum.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Clear logic in explaining ICL generalization failure through Softmax saturation and proposing a theoretically distinct scoring function.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers synthetic tasks, real LMs, encoder probes, and comparisons with multiple alternative functions; lacks validation on very large models.
- Writing Quality: ⭐⭐⭐⭐☆ Problem identification, mathematical explanation, and the experimental chain are smooth; some formulas are long but generally readable.
- Value: ⭐⭐⭐⭐☆ Highly insightful for attention mechanism modification and ICL generalization research; practical large-scale deployment value requires subsequent verification.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] From Interpretability to Performance: Optimizing Retrieval Heads for Long-Context Language Models](from_interpretability_to_performance_optimizing_retrieval_heads_for_long-context.md)
- [\[NeurIPS 2025\] Monte Carlo Expected Threat (MOCET) Scoring](../../NeurIPS2025/interpretability/monte_carlo_expected_threat_mocet_scoring.md)
- [\[NeurIPS 2025\] Empowering Decision Trees via Shape Function Branching](../../NeurIPS2025/interpretability/empowering_decision_trees_via_shape_function_branching.md)
- [\[NeurIPS 2025\] Improving Perturbation-based Explanations by Understanding the Role of Uncertainty Calibration](../../NeurIPS2025/interpretability/improving_perturbation-based_explanations_by_understanding_the_role_of_uncertain.md)
- [\[ICML 2026\] How Few-Shot Examples Add Up: A Causal Decomposition of Function Vectors in In-Context Learning](../../ICML2026/interpretability/how_few-shot_examples_add_up_a_causal_decomposition_of_function_vectors_in_in-co.md)

</div>

<!-- RELATED:END -->
