---
title: >-
  [Paper Note] From Flat Facts to Sharp Hallucinations: Detecting Stubborn Errors via Gradient Sensitivity
description: >-
  [ICML 2026][LLM Safety][stubborn hallucination] This work shifts LLM hallucination detection from "output probability analysis" to "loss landscape curvature": by injecting Gaussian noise into embeddings and measuring the…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "stubborn hallucination"
  - "loss landscape"
  - "Hessian"
  - "gradient sensitivity"
  - "embedding perturbation"
date: 2026-05-08
content_hash: 07e2b075635e771c
---

# From Flat Facts to Sharp Hallucinations: Detecting Stubborn Errors via Gradient Sensitivity

**Conference**: ICML 2026  
**arXiv**: [2605.00939](https://arxiv.org/abs/2605.00939)  
**Code**: None  
**Area**: LLM Hallucination Detection / Geometric Perspective  
**Keywords**: stubborn hallucination, loss landscape, Hessian, gradient sensitivity, embedding perturbation

## TL;DR
This work shifts LLM hallucination detection from "output probability analysis" to "loss landscape curvature": by injecting Gaussian noise into embeddings and measuring the perturbation in gradient direction and magnitude as a cheap proxy for the Hessian spectral radius, the method outperforms entropy, Semantic Entropy, EigenScore, and other baselines in AUROC across 12 model-dataset combinations.

## Background & Motivation

**Background**: Mainstream LLM hallucination detection methods fall into two camps: black-box approaches (LN-Entropy, Semantic Entropy, P(False), etc.) rely on multiple sampling to assess output consistency; white-box approaches (EigenScore, Effective Rank) analyze the covariance or rank of hidden states. Both assume "hallucination = uncertainty = high entropy / representation collapse."

**Limitations of Prior Work**: There exists a class of errors termed **stubborn hallucination**—the model confidently outputs incorrect facts, and repeated sampling consistently converges to the same answer. Their output distributions resemble true knowledge with low entropy and high confidence, rendering entropy-based methods ineffective (e.g., Semantic Entropy AUROC on Llama-2 SQuAD is only $0.4839$, nearly random). Static representation analysis methods like EigenScore also miss these cases, as the hidden states themselves do not collapse.

**Key Challenge**: Zero-order quantities (output probabilities) are not equivalent to factual correctness—the model can "confidently memorize errors." The root issue is that existing methods only measure up to first-order information, while the true distinction between hallucinations and facts lies in higher-order geometric properties.

**Goal**: (1) Formally define stubborn hallucination; (2) Identify a geometric quantity that is both computationally cheap and sensitive to such errors; (3) Demonstrate that this quantity is a reasonable proxy for the Hessian spectral radius.

**Key Insight**: Drawing from generalization theory's flat vs. sharp minima hypothesis—robust facts are supported by redundant contextual features and fall into flat minima; stubborn errors are memorized singularities of sparse noisy patterns, falling into sharp minima. Both may have high confidence (similar zero-order statistics) but differ in curvature (second-order divergence).

**Core Idea**: Inject Gaussian noise into embeddings and measure the magnitude and directional drift of the gradient as a local proxy for the Hessian spectral radius, thereby distinguishing flat facts from sharp hallucinations.

## Method

### Overall Architecture
EPGS (Embedding-Perturbed Gradient Sensitivity) consists of three stages: (1) Target Acquisition—use external NER to extract core entities and construct a mask to compute loss only on entity tokens; (2) Stochastic Embedding Perturbation—add $\delta \sim \mathcal{N}(0, \sigma^2 I)$ to input embeddings; (3) Gradient Sensitivity Measurement—compute gradients with respect to the last Transformer block parameters for both clean and perturbed inputs, $g_{\text{clean}}, g_{\text{perturbed}}$, and combine them into the EPGS score.

### Key Designs

1. **Geometric Categorization of Hallucinations**:

    - Function: Reclassifies hallucinations into three types based on loss landscape geometry, fundamentally explaining why entropy-based methods fail on stubborn cases.
    - Mechanism: Defines $\delta$-stability by output KL divergence $\mathbb{E}_\epsilon[D_{KL}(P_{\theta^*}(\cdot|x) \| P_{\theta^*}(\cdot|x+\epsilon))] \le \delta$; notes that both robust facts and stubborn hallucinations satisfy this, making them indistinguishable at zero-order. Introduces the Curvature Hypothesis: facts → flat minima ($\lambda_{\max}(H)$ small), stubborn → sharp minima ($\lambda_{\max}(H)$ large), transient → unstable region ($\|\nabla\mathcal{L}\| > 0$ or anisotropic).
    - Design Motivation: Thoroughly explains "why previous methods fail" and shifts the detection problem from the probability domain to the geometric domain, paving the way for subsequent method design.

2. **Input-Parameter Isomorphism & Hessian Proxy**:

    - Function: Uses inexpensive input perturbation gradients to replace costly second-order Hessian computation.
    - Mechanism: Lemma 3.3 shows that a small perturbation $\delta$ to the input embedding is equivalent to a parameter perturbation $\nu_\delta$ (simulated by a rank-1 weight update in the last block), i.e., $\mathcal{L}(\theta^*, E+\delta, \hat y) \approx \mathcal{L}(\theta^*+\nu_\delta, E, \hat y)$. Theorem 3.4, under the assumption $\nabla_\theta \mathcal{L}(\theta^*) = 0$, uses a second-order Taylor expansion to yield $\|\nabla_\theta \mathcal{L}(\theta^*; x+\epsilon, \hat y)\|_2 \lesssim \lambda_{\max}(H) \cdot \|\nu_\epsilon\|_2$. Intuitively, in a sharp minimum, adding noise to the input is equivalent to pushing the parameters up a steep wall, causing the gradient to spike.
    - Design Motivation: Computing the Hessian is infeasible for LLMs, but embedding perturbation plus one extra backward pass is nearly free—translating expensive geometric quantities into feasible experimental measures is the key engineering step.

3. **Dual-Factor Score: Magnitude × Directional Drift**:

    - Function: Captures both curvature scale and directional instability, enabling detection of both stubborn and transient hallucinations.
    - Mechanism: $\mathcal{S} = \|g_{\text{clean}}\|_2 \cdot (1 - \cos(g_{\text{clean}}, g_{\text{perturbed}}))$. The first term reflects local curvature scale, the second term reflects directional fluctuation (in high-dimensional space, large random displacements are almost always orthogonal to the original direction). Facts at flat minima: $\|g_{\text{clean}}\|$ is small and direction is stable → $\mathcal{S} \approx 0$. Stubborn: $\|g_{\text{clean}}\|$ is large, direction is disrupted by noise → $\mathcal{S}$ is large. Transient: both terms are large → $\mathcal{S}$ is maximal.
    - Design Motivation: Considering only magnitude is susceptible to gradient scale; considering only direction ignores curvature scale. The product form preserves physical units and robustness.

### Loss & Training
Fully post-hoc, no training required. NER uses pretrained BERT-base-NER; temperature $T=0.1$ for pseudo-label generation; perturbation $\sigma=0.1$; gradients computed only for the last Transformer block; AUROC reported threshold-independently. All experiments conducted on a single RTX 4090 GPU.

## Key Experimental Results

### Main Results (General Hallucination, AUROC)
3 models × 4 datasets = 12 combinations; EPGS ranks first in all.

| Model | Dataset | SE | EigenScore | **EPGS** |
|-------|---------|------|-----------|----------|
| Llama-2-7B | TriviaQA | 0.7080 | 0.7224 | **0.7629** |
| Llama-3-8B | SVAMP | 0.9004 | 0.9371 | **0.9732** |
| Mistral-7B | TriviaQA | 0.7701 | 0.7760 | **0.8289** |
| Llama-3-8B | TriviaQA | 0.7082 | 0.7226 | **0.8127** |

### Ablation Study (Gradient Extraction Position + Stubborn Subset)

| Configuration | Llama-2 SQuAD AUROC | Notes |
|---------------|---------------------|-------|
| Last Transformer Block (default) | **Highest** | Retains final "belief" geometry |
| Middle Transformer Block | Decreases | Lacks final aggregation |
| Final Layer Norm / LLM Head | Near random | Probability saturation masks gradients |

Stubborn subset (high-consistency subset via repeated sampling, entropy fails completely):

| Method | Llama-2 SQuAD | Llama-3 SQuAD |
|--------|--------------|--------------|
| Semantic Entropy | 0.5842 | 0.5918 |
| EigenScore | 0.6003 | 0.5857 |
| **EPGS** | **0.7373** | **0.7816** |

### Key Findings
- The true Hessian (power iteration for $\lambda_{\max}$) and EPGS have a correlation coefficient $r = 0.855$; the stubborn set's $\lambda_{\max}$ is about $2.5\times$ larger than robust facts, directly validating the Curvature Hypothesis.
- On reasoning tasks (SVAMP), EPGS shows the largest advantage (0.9732 vs 0.9371), as reasoning errors are mostly transient, and the directional drift term $(1-\cos)$ captures them fully.
- Gradient extraction must be from the last Transformer block; LLM Head gradients are flattened by softmax saturation, making the signal nearly random.

## Highlights & Insights
- Clearly explains "why entropy-based methods fail" using the geometric language of sharp/flat minima, providing a highly interpretable perspective shift.
- Uses "input perturbation gradient = Hessian proxy" to bypass second-order computation, extending curvature-aware ideas from training (e.g., SAM) to a **pure inference-time** application.
- The dual-factor score (magnitude × direction) distinguishes both stubborn (sharp) and transient (unstable) errors, covering two typical error types with a single formula.

## Limitations & Future Work
- Whether the perturbation scale $\sigma=0.1$ needs retuning for different model sizes/vocabularies is not discussed; only one value is reported.
- Validation is limited to 7B-8B models; it is unclear whether the last block gradient in larger models (70B+) would be suppressed by LayerNorm.
- The stubborn set is constructed via high-consistency filtering, which may introduce sampling noise; if the model is inherently unstable at high temperature, the boundary between stubborn and transient becomes blurred.
- No discussion on integration with retrieval-based fact verification ("detection → trigger RAG correction") for practical deployment.

## Related Work & Insights
- **vs Semantic Entropy / DSE**: Relies on output diversity from multiple sampling; EPGS requires only a single gradient, remaining effective for stubborn cases.
- **vs EigenScore / Effective Rank**: Analyzes "static" hidden state covariance; EPGS actively injects perturbations to capture dynamic responses.
- **vs SAM (Sharpness-Aware Minimization)**: Uses sharpness for regularization during training; EPGS leverages the same geometric quantity as a detection signal during inference.
- **vs Hessian-based uncertainty**: Direct computation of $\lambda_{\max}(H)$ is infeasible for LLMs; EPGS uses input gradients as a cheap proxy.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The "stubborn hallucination + sharp minimum" concept represents a genuine perspective shift.
- Experimental Thoroughness: ⭐⭐⭐⭐ 12 combinations + stubborn subset + Hessian ground truth are all covered, but model scale is relatively small.
- Writing Quality: ⭐⭐⭐⭐ Clear alignment between theory and experiments, with intuitive three-stage diagrams.
- Value: ⭐⭐⭐⭐ Provides a detection signal sensitive to "confident errors" for high-stakes deployment, with high practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Ghost in the Transformer: Detecting Model Reuse with Invariant Spectral Signatures](../../AAAI2026/llm_safety/ghost_in_the_transformer_detecting_model_reuse_with_invariant_spectral_signature.md)
- [\[ACL 2026\] FACTS: Table Summarization via Offline Template Generation with Agentic Workflows](../../ACL2026/llm_safety/facts_table_summarization_via_offline_template_generation_with_agentic_workflows.md)
- [\[ICML 2026\] REALISTA: Realistic Latent Adversarial Attacks that Elicit LLM Hallucinations](realista_realistic_latent_adversarial_attacks_that_elicit_llm_hallucinations.md)
- [\[ICLR 2026\] Understanding Sensitivity of Differential Attention through the Lens of Adversarial Robustness](../../ICLR2026/llm_safety/understanding_sensitivity_of_differential_attention_through_the_lens_of_adversar.md)
- [\[AAAI 2026\] PSM: Prompt Sensitivity Minimization via LLM-Guided Black-Box Optimization](../../AAAI2026/llm_safety/psm_prompt_sensitivity_minimization_via_llm-guided_black-box_optimization.md)

</div>

<!-- RELATED:END -->
