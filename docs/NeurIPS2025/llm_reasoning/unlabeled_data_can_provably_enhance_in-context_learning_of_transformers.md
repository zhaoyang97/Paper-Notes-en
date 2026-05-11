---
title: >-
  [Paper Note] Unlabeled Data Can Provably Enhance In-Context Learning of Transformers
description: >-
  [NeurIPS 2025][LLM Reasoning][in-context learning] This paper proposes an augmented ICL framework in which the prompt contains both a small set of labeled examples and a large collection of unlabeled examples. It theoret…
tags:
  - "NeurIPS 2025"
  - "LLM Reasoning"
  - "in-context learning"
  - "unlabeled data"
  - "semi-supervised learning"
  - "EM algorithm"
  - "chain-of-thought"
  - "transformer theory"
date: 2026-05-08
content_hash: b8001dcf4c599f49
---

# Unlabeled Data Can Provably Enhance In-Context Learning of Transformers

**Conference**: NeurIPS 2025
**arXiv**: [2601.10058](https://arxiv.org/abs/2601.10058)
**Code**: None
**Area**: LLM Reasoning / Theory
**Keywords**: in-context learning, unlabeled data, semi-supervised learning, EM algorithm, chain-of-thought, transformer theory

## TL;DR
This paper proposes an augmented ICL framework in which the prompt contains both a small set of labeled examples and a large collection of unlabeled examples. It theoretically proves that a multi-layer Transformer, via chain-of-thought (CoT) reasoning, can simulate the EM algorithm to extract information from unlabeled data, improving the classification excess risk from $\mathcal{O}(1/\sqrt{N})$ to $\mathcal{O}(1/\sqrt{N + \text{poly}(M)})$.

## Background & Motivation

**Background**: ICL enables Transformers to learn new tasks from in-context examples without parameter updates, but it relies heavily on the quantity and quality of labeled demonstrations. Acquiring high-quality labeled data is costly (e.g., RLHF data for GPT-3.5/4 involves thousands of hours of expert annotation).

**Limitations of Prior Work**: (a) Prompt length limits the number of labeled examples; (b) existing methods use the LLM itself to generate pseudo-labels, which inherit model bias; (c) abundant unlabeled data remains unexploited by ICL.

**Key Challenge**: A large amount of unlabeled data exists, yet ICL has no mechanism to leverage it — conventional ICL processes only $(x, y)$ pairs, leaving unlabeled $x$ unused.

**Goal**: To theoretically prove that unlabeled data can improve ICL performance, and to provide an explicit Transformer construction along with training convergence guarantees.

**Key Insight**: The paper connects augmented ICL (labeled + unlabeled in the prompt) to the classical EM algorithm in semi-supervised learning — a Transformer can iteratively refine class-mean estimates through multi-step CoT reasoning.

**Core Idea**: A 4-layer Transformer with CoT reasoning implicitly implements the EM algorithm, learning simultaneously from labeled and unlabeled data within the prompt.

## Method

### Overall Architecture
The input is a mixed prompt $\mathcal{I} = \mathcal{D}_{label} \cup \mathcal{D}_{unlabel}$, where $\mathcal{D}_{label} = \{(\mathbf{x}_j, y_j)\}_{j=1}^N$ and $\mathcal{D}_{unlabel} = \{\mathbf{x}_j\}_{j=N+1}^{N+M}$. This is encoded into a matrix $\mathbf{H}$ containing labeled, unlabeled, and reasoning blocks. The Transformer iteratively refines class-mean estimates $\hat{\mu}_i^{(t)}$ in the reasoning block via CoT, and ultimately classifies unlabeled samples using nearest-neighbor assignment.

### Key Designs

1. **CoT Encoding for Augmented ICL**:

    - Function: Encodes labeled and unlabeled data together with intermediate reasoning states into a unified token sequence.
    - Mechanism: The reasoning block $\mathbf{Q}^{(t)}$ stores the class-mean estimate $\hat{\mu}_i^{(t)}$ at step $t$. Each CoT step appends $C$ newly generated tokens to the sequence.
    - Design Motivation: To exploit the autoregressive nature of CoT for EM iteration — each CoT step corresponds to one EM update.

2. **4-Layer Transformer Construction (Theorem 4.1)**:

    - Function: Explicitly constructs a 4-layer Transformer that implements the EM update.
    - Mechanism: The update rule is $\hat{\mu}_i^{(t+1)} = \hat{\mu}_i^{(t)} - \frac{\eta^{(t)}}{M} \sum_{j} p_{ij}^{(t)}(\hat{\mu}_i^{(t)} - \mathbf{x}_j) + \mathbf{1}_{\{t=0\}} \frac{C}{N} \sum_j (\mathbf{e}_i^\top \mathbf{y}_j) \mathbf{x}_j$. The first layer uses softmax attention to compute the E-step (class membership probabilities $p_{ij}^{(t)}$); subsequent layers implement the M-step (weighted mean update). Initialization uses class means from the labeled data.
    - Design Motivation: To precisely simulate the EM algorithm for a Gaussian mixture model — the E-step computes posterior probabilities and the M-step updates parameters.

3. **Convergence Analysis (Theorem 4.2)**:

    - Function: Proves that class-mean estimates converge to their true values as the number of CoT steps increases.
    - Mechanism: Under the conditions of signal-to-noise ratio $\text{SNR} \geq \Omega(\sqrt{C \log(CM)})$, sufficiently large $N$, and sufficiently large $M$, the excess risk is $\mathcal{O}(1/\sqrt{N + \text{poly}(M)})$, which strictly improves upon the labeled-only lower bound of $\mathcal{O}(1/\sqrt{N})$.
    - Design Motivation: To rigorously establish the theoretical value of unlabeled data — beyond empirical observation alone.

4. **Training Convergence (Theorem 5.1)**:

    - Function: Proves that Transformer parameters trained with teacher forcing converge linearly to the target solution.
    - Mechanism: The gradient of the CoT training loss is decomposed into two analytically tractable terms; isotropy of the involved quantities simplifies the analysis.
    - Design Motivation: To show that the theoretical construction is not merely an existence result but can be recovered via standard training.

### Loss & Training
Teacher forcing training: at each CoT step $t$, the supervision signal is the "ground-truth" next-step output of the EM algorithm. Gradient descent on the population loss converges at a linear rate.

## Key Experimental Results

### Main Results
On a multi-class linear classification setting ($d=20$, $C=$ 3 and 5 classes):

| Method | Mean Estimation Error | Classification Accuracy |
|---|---|---|
| Conventional ICL (N labeled only) | Baseline | Baseline |
| Augmented ICL (N labeled + M unlabeled) | **Significantly reduced** | **Significantly improved** |
| Bayes-optimal classifier (labeled only) | Below Augmented ICL | Below Augmented ICL |

### Ablation Study

| Configuration | Effect | Remark |
|---|---|---|
| Increasing $M$ (more unlabeled data) | Consistent performance improvement | Validates $\mathcal{O}(1/\sqrt{N+\text{poly}(M)})$ |
| Increasing $T$ (more CoT steps) | Performance improves then saturates | Consistent with EM convergence behavior |
| Fixed $N$, increasing $M$ only | Performance improves beyond $N$-only level | Independent contribution of unlabeled data |

### Key Findings
- Augmented ICL significantly outperforms conventional ICL and even surpasses the Bayes-optimal classifier trained on labeled data alone, demonstrating the genuine value of unlabeled data.
- Performance gains grow consistently with $M$, in agreement with theoretical predictions.
- $T = 5$–$10$ CoT steps are generally sufficient for convergence.

## Highlights & Insights
- **Theoretical bridge between ICL and semi-supervised learning**: This work is the first to theoretically establish the equivalence between Transformer CoT reasoning and the EM algorithm, offering a new perspective on the inference mechanism of ICL.
- **Precise quantification of excess risk improvement**: The result $\mathcal{O}(1/\sqrt{N}) \to \mathcal{O}(1/\sqrt{N + \text{poly}(M)})$ is a clean theoretical characterization that clearly captures the marginal contribution of unlabeled data.
- **Complete theory covering existence and learnability**: Beyond constructing an ideal Transformer (existence), the paper also proves training convergence (learnability), forming a complete theoretical chain.

## Limitations & Future Work
- The theory is restricted to multi-class linear classification with Gaussian mixture models — whether it generalizes to nonlinear classification or regression tasks remains open.
- Isotropic covariance $\Sigma$ is assumed — theoretical analysis under anisotropic settings is more challenging.
- The constructed 4-layer Transformer is idealized — it is unclear whether practically pretrained LLMs implicitly learn analogous EM strategies.
- Experiments are conducted solely on synthetic data — the effectiveness of augmented ICL on real NLP/CV tasks is not validated.
- Unlabeled data must be task-relevant — performance may degrade if the unlabeled data distribution does not match the task.

## Related Work & Insights
- **vs. Unsupervised ICL (Gupta et al. 2024)**: Purely unsupervised ICL relies on unlabeled data only; this paper adopts a semi-supervised framework with labeled + unlabeled data and provides stronger theoretical guarantees.
- **vs. Pseudo-label methods (Wan et al. 2023)**: Pseudo-labels inherit model bias; augmented ICL performs reasoning directly within the prompt without generating pseudo-labels.
- **vs. Li et al. 2025 (concurrent)**: That work uses linear Transformers, binary classification, and asymptotic analysis; this paper uses softmax attention, multi-class classification, and non-asymptotic convergence bounds.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First work to theoretically answer whether unlabeled data can benefit ICL, establishing the ICL–EM equivalence.
- Experimental Thoroughness: ⭐⭐⭐ Experiments are limited to synthetic data with no validation on real tasks — standard practice for theoretical work, but still a limitation.
- Writing Quality: ⭐⭐⭐⭐ Meets the conventions of theoretical paper writing with a clear notation system, though the technical density is high.
- Value: ⭐⭐⭐⭐ Makes an important contribution to understanding ICL mechanisms, though the gap from linear classification to practical LLMs remains substantial.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Transformers Provably Learn Chain-of-Thought Reasoning with Length Generalization](transformers_provably_learn_chain-of-thought_reasoning_with_length_generalizatio.md)
- [\[ICLR 2026\] Is In-Context Learning Learning?](../../ICLR2026/llm_reasoning/is_in-context_learning_learning.md)
- [\[NeurIPS 2025\] Exact Expressive Power of Transformers with Padding](exact_expressive_power_of_transformers_with_padding.md)
- [\[NeurIPS 2025\] Note 1: Is CoT a Hallucination? A Data Distribution Perspective](is_chain-of-thought_reasoning_of_llms_a_mirage_a_data_distribution_lens.md)
- [\[NeurIPS 2025\] Let Me Think! A Long Chain-of-Thought Can Be Worth Exponentially Many Short Ones](let_me_think_a_long_chainofthought_can_be_worth_exponentiall.md)

</div>

<!-- RELATED:END -->
