---
title: >-
  [Paper Note] From Flat Facts to Sharp Hallucinations: Detecting Stubborn Errors via Gradient Sensitivity
description: >-
  [ICML 2026][LLM Safety][stubborn hallucination] This paper shifts LLM hallucination detection from "analyzing output probabilities" to "analyzing loss landscape curvature." By adding Gaussian noise to embeddings and meas…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "stubborn hallucination"
  - "loss landscape"
  - "Hessian"
  - "gradient sensitivity"
  - "embedding perturbation"
date: 2026-05-08
content_hash: f15e1e1528f11438
---

# From Flat Facts to Sharp Hallucinations: Detecting Stubborn Errors via Gradient Sensitivity

**Conference**: ICML 2026  
**arXiv**: [2605.00939](https://arxiv.org/abs/2605.00939)  
**Code**: None  
**Area**: LLM Hallucination Detection / Geometric Perspective  
**Keywords**: stubborn hallucination, loss landscape, Hessian, gradient sensitivity, embedding perturbation

## TL;DR
This paper shifts LLM hallucination detection from "analyzing output probabilities" to "analyzing loss landscape curvature." By adding Gaussian noise to embeddings and measuring perturbations in gradient direction and magnitude as a cheap proxy for the Hessian spectral radius, the method achieves AUROC results that consistently outperform baselines such as entropy, Semantic Entropy, and EigenScore across 12 model-dataset combinations.

## Background & Motivation

**Background**: Mainstream LLM hallucination detection is divided into two categories: black-box methods (LN-Entropy, Semantic Entropy, P(False), etc.) that use multiple sampling to check output consistency, and white-box methods (EigenScore, Effective Rank) that analyze the covariance or rank of hidden states. Both share the assumption that "hallucination = uncertainty = high entropy / representation collapse."

**Limitations of Prior Work**: There exists a class of errors the authors call **stubborn hallucinations**—instances where the model confidently outputs incorrect facts and consistently converges to the same answer even across multiple samplings. Their output distributions are low-entropy and high-confidence, similar to true knowledge, causing entropy-based methods to fail (e.g., Semantic Entropy AUROC on Llama-2 SQuAD is only $0.4839$, essentially random guessing). "Static representation analysis" like EigenScore also misses these because the hidden states themselves do not collapse.

**Key Challenge**: Zero-order quantities (output probabilities) are not equivalent to factual correctness—models can "confidently remember wrong information." The root of the problem is that existing methods measure information at the first order or lower, whereas the true difference between hallucinations and facts lies in higher-order geometric properties.

**Goal**: (1) Formally define stubborn hallucination; (2) Identify a cheaply computable geometric quantity sensitive to these errors; (3) Prove that this quantity is a reasonable proxy for the Hessian spectral radius.

**Key Insight**: The authors borrow the flat vs. sharp minima hypothesis from generalization theory—robust facts are supported by multiple redundant contextual features and fall into flat minima; stubborn errors are memory singularities of sparse noisy patterns that fall into sharp minima. Both exhibit high confidence (zero-order similarity) but differ in curvature (second-order divergence).

**Core Idea**: Inject Gaussian noise into embeddings and measure the drift in gradient magnitude and direction as a proxy for the local Hessian spectral radius, thereby distinguishing flat facts from sharp hallucinations.

## Method

### Overall Architecture
The EPGS (Embedding-Perturbed Gradient Sensitivity) framework consists of three stages: (1) Target Acquisition—extract core entities using external NER and construct a mask for calculating loss only on entity tokens; (2) Stochastic Embedding Perturbation—add $\delta \sim \mathcal{N}(0, \sigma^2 I)$ to the input embeddings; (3) Gradient Sensitivity Measurement—calculate gradients $g_{\text{clean}}$ and $g_{\text{perturbed}}$ of the parameters in the last Transformer block, then combine them into an EPGS score.

### Key Designs

1.  **Geometric Classification of Hallucination**:
    - **Function**: Reclassifies hallucinations based on loss landscape geometry to explain fundamentally why entropy-based methods fail on stubborn errors.
    - **Mechanism**: Defines $\delta$-stability based on output KL divergence $\mathbb{E}_\epsilon[D_{KL}(P_{\theta^*}(\cdot|x) \| P_{\theta^*}(\cdot|x+\epsilon))] \le \delta$; points out that both robust facts and stubborn hallucinations satisfy this condition, making them indistinguishable at the zero-order level. Introduces the Curvature Hypothesis: facts $\to$ flat minima (small $\lambda_{\max}(H)$), stubborn $\to$ sharp minima (large $\lambda_{\max}(H)$), transient $\to$ unstable region ($\|\nabla\mathcal{L}\| > 0$ or anisotropy).
    - **Design Motivation**: By thoroughly explaining why previous methods fail and moving the detection problem from the probability domain to the geometric domain, the motivation for the proposed method becomes intuitive.

2.  **Input-Parameter Isomorphism + Hessian Proxy**:
    - **Function**: Uses cheap input-perturbed gradients instead of expensive second-order Hessian computations.
    - **Mechanism**: Lemma 3.3 demonstrates that for small input embedding perturbations $\delta$, there exists an equivalent parameter perturbation $\nu_\delta$ (simulated by a rank-1 weight update of the last block), such that $\mathcal{L}(\theta^*, E+\delta, \hat y) \approx \mathcal{L}(\theta^*+\nu_\delta, E, \hat y)$. Theorem 3.4 uses a second-order Taylor expansion under the assumption $\nabla_\theta \mathcal{L}(\theta^*) = 0$ to show that $\|\nabla_\theta \mathcal{L}(\theta^*; x+\epsilon, \hat y)\|_2 \lesssim \lambda_{\max}(H) \cdot \|\nu_\epsilon\|_2$. Intuitively, in a sharp minimum, adding noise to the input is equivalent to pushing the parameters up a steep wall, causing the gradient to spike immediately.
    - **Design Motivation**: Hessian calculations are infeasible for LLMs, but embedding perturbation plus one extra backward pass is nearly free—translating expensive geometric quantities into feasible experimental metrics is the most critical engineering step.

3.  **Double-Factor Score (Magnitude × Direction Drift)**:
    - **Function**: Simultaneously captures curvature scale and directional instability to identify both stubborn and transient hallucinations.
    - **Mechanism**: $\mathcal{S} = \|g_{\text{clean}}\|_2 \cdot (1 - \cos(g_{\text{clean}}, g_{\text{perturbed}}))$. The first term reflects the local curvature scale, while the second reflects directional fluctuation (in high-dimensional space, a large random displacement is almost certainly orthogonal to the original direction). Facts in flat minima: small $\|g_{\text{clean}}\|$ and stable direction $\to \mathcal{S} \approx 0$. Stubborn: large $\|g_{\text{clean}}\|$ and direction disrupted by noise $\to$ large $\mathcal{S}$. Transient: both terms are large $\to$ maximum $\mathcal{S}$.
    - **Design Motivation**: Relying solely on magnitude is easily disturbed by gradient scaling; relying solely on direction ignores the curvature scale. The product form maintains physical dimensions and provides robustness.

### Loss & Training
The method is completely post-hoc and requires no training. NER uses a pre-trained BERT-base-NER; pseudo-labels are generated with temperature $T=0.1$; perturbation $\sigma=0.1$; gradients are computed only for the last Transformer block; AUROC is reported threshold-independently. All experiments were conducted on a single RTX 4090.

## Key Experimental Results

### Main Results (General Hallucination, AUROC)
Across 12 combinations of 3 models and 4 datasets, EPGS ranks first in all cases.

| Model | Dataset | SE | EigenScore | **EPGS** |
|------|--------|------|-----------|----------|
| Llama-2-7B | TriviaQA | 0.7080 | 0.7224 | **0.7629** |
| Llama-3-8B | SVAMP | 0.9004 | 0.9371 | **0.9732** |
| Mistral-7B | TriviaQA | 0.7701 | 0.7760 | **0.8289** |
| Llama-3-8B | TriviaQA | 0.7082 | 0.7226 | **0.8127** |

### Ablation Study (Gradient Extraction Position + Stubborn Subset)

| Configuration | Llama-2 SQuAD AUROC | Note |
|------|----------------------|------|
| Last Transformer Block (Default) | **Highest** | Retains geometric "belief" |
| Middle Transformer Block | Decrease | Lacks final aggregation |
| Final Layer Norm / LLM Head | Near Random | Probability saturation masks gradients |

Stubborn subset (High-consistency subset retained after multiple sampling; entropy methods fail):

| Method | Llama-2 SQuAD | Llama-3 SQuAD |
|------|---------------|---------------|
| Semantic Entropy | 0.5842 | 0.5918 |
| EigenScore | 0.6003 | 0.5857 |
| **EPGS** | **0.7373** | **0.7816** |

### Key Findings
- The correlation coefficient between the ground truth Hessian ($\lambda_{\max}$ calculated via power iteration) and EPGS is $r = 0.855$; the $\lambda_{\max}$ of stubborn sets is approximately $2.5\times$ larger than that of robust facts, directly validating the Curvature Hypothesis.
- EPGS shows the greatest advantage in reasoning tasks (SVAMP) (0.9732 vs 0.9371) because reasoning errors are mostly transient, which maximizes the direction drift term $(1-\cos)$.
- The gradient extraction position must be the last Transformer block; the LLM Head flattens gradient signals due to softmax saturation, resulting in near-random performance.

## Highlights & Insights
- Using the geometric language of sharp/flat minima to explain "why entropy methods fail" provides a highly explanatory perspective shift.
- Using "input perturbation gradient = Hessian proxy" avoids second-order calculations, extending curvature-aware concepts (like SAM) to a **purely inference-time** application.
- The double-factor score (magnitude × direction) can distinguish both stubborn (sharp) and transient (unstable) errors, covering two typical types of errors with a single formula.

## Limitations & Future Work
- Does the perturbation scale $\sigma=0.1$ need to be retuned for different model sizes or vocabularies? Only one value was provided in the paper.
- Validation was only performed on 7B-8B models; it is unclear if the last block gradient of larger models (70B+) would be flattened by LayerNorm.
- The stubborn set was constructed by filtering high-consistency responses, which may involve sampling estimation noise; for models with high-temperature instability, the boundary between stubborn and transient errors is blurred.
- There is no discussion on integration paths with retrieval-based fact-checking (e.g., "detect $\to$ trigger RAG correction").

## Related Work & Insights
- **vs Semantic Entropy / DSE**: These rely on output diversity across multiple samplings; EPGS requires only a single gradient pass and does not fail on stubborn errors.
- **vs EigenScore / Effective Rank**: These analyze the hidden state covariance of "static states"; EPGS actively injects perturbations to capture dynamic responses.
- **vs SAM (Sharpness-Aware Minimization)**: SAM uses sharpness as a regularizer during training; EPGS uses the same geometric quantity as a detection signal during inference.
- **vs Hessian-based uncertainty**: Directly calculating $\lambda_{\max}(H)$ is impractical for LLMs; EPGS serves as a cheap proxy using input gradients.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ "stubborn hallucination + sharp minimum" is a genuine conceptual perspective shift.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 12 combinations, stubborn subsets, and Hessian ground truth, though model scales are relatively small.
- Writing Quality: ⭐⭐⭐⭐ Clear correspondence between theory and experiments; three-stage illustration is intuitive.
- Value: ⭐⭐⭐⭐ Provides a detection signal sensitive to "confident errors" for high-stakes deployments; highly practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] TPA: Next Token Probability Attribution for Detecting Hallucinations in RAG](../../ACL2026/llm_safety/tpa_next_token_probability_attribution_for_detecting_hallucinations_in_rag.md)
- [\[ACL 2026\] FinGround: Detecting and Grounding Financial Hallucinations via Atomic Claim Verification](../../ACL2026/llm_safety/finground_detecting_and_grounding_financial_hallucinations_via_atomic_claim_veri.md)
- [\[ICML 2026\] Gradient Transformer: Learning to Generate Updates for LLMs](gradient_transformer_learning_to_generate_updates_for_llms.md)
- [\[ICML 2026\] REALISTA: Realistic Latent Adversarial Attacks that Elicit LLM Hallucinations](realista_realistic_latent_adversarial_attacks_that_elicit_llm_hallucinations.md)
- [\[AAAI 2026\] Ghost in the Transformer: Detecting Model Reuse with Invariant Spectral Signatures](../../AAAI2026/llm_safety/ghost_in_the_transformer_detecting_model_reuse_with_invariant_spectral_signature.md)

</div>

<!-- RELATED:END -->
