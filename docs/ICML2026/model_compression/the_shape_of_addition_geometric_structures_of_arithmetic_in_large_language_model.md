---
title: >-
  [Paper Note] The Shape of Addition: Geometric Structures of Arithmetic in Large Language Models
description: >-
  [ICML 2026][Model Compression][Residual Stream Geometry] The authors discover that activations in the final layer residual stream of Qwen3-4B are organized into a hierarchical manifold of "digit basins × carry fibers" du…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Residual Stream Geometry"
  - "IRST"
  - "Noisy Quantization Model"
  - "Carry Potential"
  - "Inference-time Self-Correction"
date: 2026-05-08
content_hash: c7582a958d1f4e40
---

# The Shape of Addition: Geometric Structures of Arithmetic in Large Language Models

**Conference**: ICML 2026  
**arXiv**: [2606.03645](https://arxiv.org/abs/2606.03645)  
**Code**: https://github.com/RL-MIND/Shape-of-Addition  
**Area**: Interpretability / Mechanistic Analysis / LLM Arithmetic  
**Keywords**: Residual Stream Geometry, IRST, Noisy Quantization Model, Carry Potential, Inference-time Self-Correction

## TL;DR
The authors discover that activations in the final layer residual stream of Qwen3-4B are organized into a hierarchical manifold of "digit basins × carry fibers" during multi-operand addition. "Off-by-one" errors are re-interpreted as geometric slippage across quantization thresholds of a continuous carry potential along Iso-Raw-Sum Trajectories (IRST). Based on this, a dual-stream consistency check is proposed to correct errors where the model "internally knows but outputs incorrectly" during inference.

## Background & Motivation
**Background**: Existing work on explaining LLM arithmetic capabilities generally follows two paths: one models internal states as symbolic look-up tables or discrete addition circuits (e.g., circuit reverse engineering by Quirke, Nanda, etc., on small models); the other uses linear probes to scan the residual stream, proving that LLM internals indeed encode intermediate variables such as ground truth, carry, and raw sum. Both paths have accumulated significant phenomenological evidence.

**Limitations of Prior Work**: Although probes can simultaneously extract contradictory signals (e.g., correct answer, erroneous output, carry) from the activations of a "failed sample" (termed "probe versatility"), two things remain unexplained: first, why a single vector can host "correct" and "incorrect" semantics simultaneously; second, how this internal representation geometry mechanistically induces specific failure modes (especially the off-by-one carry errors common in multi-operand addition).

**Key Challenge**: Previous discretization perspectives assume internal states are categorical, but continuous scalars extracted by probes (e.g., carry potential) suggest the presence of a continuous manifold. The lack of a bridge between discrete outputs and continuous representations is the root cause of repeated "off-by-one" errors in LLM arithmetic.

**Goal**: To provide a unified geometric framework at the representation level that explains (i) probe versatility, (ii) the distribution of off-by-one errors, and (iii) the "internal knowledge vs. execution failure" observation, and to design a training-free interface for inference-time error correction.

**Key Insight**: The authors select 3-operand 10-digit integer addition as a task that is "sufficiently difficult and regular." They extract final layer activations $\boldsymbol{h}_p^{(L)}$ at each generation position $p$ and use UMAP (cosine distance) with logit embeddings as "semantic anchors" to project the high-dimensional manifold onto 2D while preserving readable semantic coordinates.

**Core Idea**: The residual stream is viewed as a set of "Iso-Raw-Sum Trajectory (IRST)" fibers. The position along a fiber is controlled by a continuous "carry potential $\Phi$." The output digit is the result of noisy quantization of $\Phi$; an error is a geometric slippage caused by noise pushing $\Phi$ across an integer quantization threshold.

## Method

### Overall Architecture
Instead of training a new model, the authors build a four-stage analysis pipeline: "Observation → Geometric Modeling → Causal Verification → Inference-time Intervention." (1) Run 10,000 triple-integer 10-digit addition problems on a fixed LLM (Qwen3-4B, 36 layers) and record activations at each generation position; (2) use UMAP visualization to identify the IRST geometric structure and define "geometric slippage" to characterize errors; (3) propose a Noisy Quantization Model, modeling the carry as a continuous potential $\Phi$ plus Gaussian noise followed by a floor operation, deriving a "bathtub-shaped" error rate formula; (4) use linear/regression probes to extract local raw sum and global carry potential for a dual-stream consistency check, correcting off-by-one errors without retraining. The "model" in this entire process is a set of mathematical hypotheses + probes + logit intervention, where inputs are activation vectors and outputs are corrected next-token logits.

### Key Designs

1.  **Iso-Raw-Sum Trajectory (IRST)**:
    - **Function**: Explicitly represents the geometric skeleton of the residual stream as a two-level structure: "basins anchored by digits 0–9 × parallel fibers distinguished by carry $c_p$."
    - **Mechanism**: Define $\mathcal{T}_r$ as the continuous manifold of activations where $r_p = r$ (identical intra-column raw sum). From the identity $\hat{s}_p \equiv (r_p + \hat{c}_p) \bmod 10$, an IRST is not trapped in a single digit basin but traverses adjacent basins as $\hat{c}_p$ increases, forming a topology like $(1,1,0,0)\!\leftrightarrow\!(2,2,1,1)\!\leftrightarrow\!(3,3,2,2)$ where "sliding one unit along the fiber = carry +1." Erroneous samples (red dots) almost always fall into sparse transition segments between stable nodes, termed "geometric slippage."
    - **Design Motivation**: To align discrete semantics (correct/incorrect output) with continuous geometry (position on a fiber), making probe versatility a natural consequence—ground truth is read from "basin membership" while hallucination is read from "fiber offset," which exist in orthogonal directions.

2.  **Noisy Quantization Model + Bathtub Error Rate**:
    - **Function**: Provides an analytical expression for why errors concentrate near integer boundaries.
    - **Mechanism**: Define carry potential $\Phi_p = \sum_{j\ge 1} r_{p+j}/10^j$ as the continuous thrust from right-hand context into the current position, with the true carry $c_p = \lfloor \Phi_p \rfloor$. The model internally estimates a noisy version $\hat\Phi_p = \Phi_p + \epsilon,\ \epsilon\sim\mathcal{N}(0,\sigma^2)$, outputting $\hat c_p = \lfloor \hat\Phi_p \rfloor$. Let $\delta(\Phi)=\Phi\bmod 1$; the single-step off-by-one error rate is $P(\text{err}\mid\Phi) = Q(\delta/\sigma) + Q((1-\delta)/\sigma)$, predicting a periodic bathtub curve that peaks at integer $\Phi$ and flattens at $\Phi\approx i+0.5$. Empirical fitting yields $R^2=0.80$, estimating an effective internal noise of $\sigma\approx 0.05$ for Qwen3-4B.
    - **Design Motivation**: To replace vague claims of "random noise" with specific, quantifiable threshold-crossing events; and to provide a "danger zone" prior (proximity to integers implies risk) for inference-time correction.

3.  **Dual-Stream Consistency Inference-Time Self-Correction**:
    - **Function**: Recognizes and corrects off-by-one errors during inference using two lightweight probes without retraining or modification of the decoder.
    - **Mechanism**: A classification probe $f_{\theta_r}$ reads local raw sum $\hat r_p$, and a regression probe $f_{\theta_\phi}$ reads global carry potential $\hat\Phi_p$ from the final layer. Define a plausible carry set $\mathcal{K}_p(\delta) = \{\lfloor\phi\rfloor : \phi\in[\hat\Phi_p-\delta,\hat\Phi_p+\delta]\}$. If the model output $\hat s_p$ cannot be explained by any $(\hat r_p, c\in\mathcal K_p(\delta))$ via $\hat s_p\equiv(\hat r_p + c)\bmod 10$, it is considered that "the vector drifted off the correct IRST," and $\hat s_{\text{new}} = (\hat r_p + \lfloor\hat\Phi_p\rfloor) \bmod 10$ overrides the logits. $\delta$ acts as a tolerance, corresponding to the stable region in bathtub theory where ambiguity is acknowledged near thresholds without forced overriding.
    - **Design Motivation**: To transform geometric understanding into a causal intervention. If the IRST hypothesis holds, then the combination of "local raw sum probe + global carry potential probe" must contain sufficient information to recover the correct digit. Performance gains in error correction serve as causal evidence for the IRST hypothesis.

### Loss & Training
The LLM itself is not trained. Probes are logistic regression / MLP / linear regression models trained on balanced datasets using standard Cross-Entropy or MSE; they are robust to hyperparameters. All interventions occur during forward inference on final layer activations and output logits.

## Key Experimental Results

### Main Results
All data obtained from Qwen3-4B using 3-operand 10-digit addition at position $p=4$ with $N=10000$. Table 1 shows probe decoding accuracy in the final layer; Table 2 compares dual-stream consistency correction performance.

| Probe Target | Symbol | Accuracy |
| :--- | :--- | :--- |
| Ground Truth | $s_p$ | 94.85% |
| Model Output | $\hat s_p$ | 98.81% |
| Correctness | $\mathbb{I}(\hat s_p\ne s_p)$ | 82.41% |
| Raw Sum (mod 10) | $r_p\bmod 10$ | 98.60% |
| Input Carry | $c_p$ | 96.84% |
| Carry Potential | $\Phi_p$ | 92.08% (after floor) |

| Method | Token Acc | TP Corr | FP Pres |
| :--- | :--- | :--- | :--- |
| Original Model | 86.26% | / | / |
| Re-Prompting | 79.90% | 0.08% | 99.98% |
| Linear Steering | 88.27% | 30.58% | 96.97% |
| Hard Replacement | 89.13% | 31.73% | 97.65% |
| Ours ($\delta=0$) | 87.27% | 44.39% | 94.07% |
| Ours ($\delta=0.1$) | **89.56%** | 30.46% | 98.13% |

### Ablation Study
Table 3 decouples the contributions of the "raw sum probe" and the "carry probe" in the dual-stream framework.

| Configuration | Token Acc | TP Corr | FP Pres |
| :--- | :--- | :--- | :--- |
| R + C (Both Probes) | 86.7% | 42.3% | 93.9% |
| R + TC (Probe Raw Sum + True Carry) | 96.0% | 69.3% | 99.0% |
| TR + C (True Raw Sum + Probe Carry) | 90.5% | 65.4% | 94.6% |

### Key Findings
- The bathtub error rate formula fits with $R^2=0.80$, estimating $\sigma\approx 0.05$. Internal noise is significantly less than 1, quantitatively verifying that geometric slippage occurs only when $\Phi$ is sufficiently close to an integer.
- R+TC reaches 96.0% while TR+C only reaches 90.5%, indicating that the internal representation of raw sum is nearly perfect and the main bottleneck is carry potential noise. The model does not "fail to calculate" but rather "doesn't know whether to carry."
- The Correctness probe achieves only 82.41% because IRST is a continuous manifold without hard boundaries—explaining why traditional "detect-then-fix" two-stage approaches often fail.
- Causal steering experiments show that states near basin boundaries flip at $|\alpha|\approx 0.1$–$0.3$, while stable centers require $|\alpha|\approx 0.5$. The sigmoid phase transition curve confirms the existence of a "continuous carry direction."

## Highlights & Insights
- "Probe versatility" is clarified geometrically: "correct" and "incorrect" semantics in the same vector are not contradictory but live on orthogonal coordinates (basin vs. fiber offset). This perspective of "decomposing semantics via geometry" is applicable to other tasks with discrete outputs but continuous intermediate representations, such as LLM counting or tokenized time-series prediction.
- The bathtub error formula $P(\text{err}\mid\Phi)=Q(\delta/\sigma)+Q((1-\delta)/\sigma)$ transforms error rate into a fittable physical quantity. The derived internal noise $\sigma$ provides a comparable accuracy metric for LLM arithmetic—allowing models to be compared by $\sigma$ rather than just accuracy.
- Inference-time correction requires no gradient updates or fine-tuning, incurring nearly zero cost in throughput and memory. This "probe-as-interface" concept is highly promising for alignment and safety, such as replacing hallucination detectors with "probe + consistency voting."

## Limitations & Future Work
- Experiments were performed solely on Qwen3-4B with up to 3-operand 10-digit addition. Whether geometric structures persist in larger models, for $>3$ operands, or for floating-point arithmetic remains an open question.
- The IRST framework characterizes observed phenomena in the final layer but does not explain how $\Phi$ is computed—the carry accumulation mechanism in early layers remains a black box.
- Although dual-stream correction improves accuracy to 89.56%, there is a trade-off between TP Corr and FP Pres. Adapting $\delta$ to query difficulty (e.g., based on distance from $\hat\Phi$ to integers) is a direct path for improvement.
- The assumption $\hat r_p\approx r_p$ (near-perfect local raw sum) only holds for simple addition. In symbolic reasoning or code generation, the dual-stream framework may need expansion into "multi-stream."

## Related Work & Insights
- **vs. Nanda et al. (2023) Circular Manifolds in Modular Addition**: While that work found modular addition uses rotational representations in toy transformers, this paper extends the idea to multi-digit addition in large models, proving the existence of a "continuous phase" as carry potential and providing a geometric explanation for discrete output failures.
- **vs. Kantamneni & Tegmark (2025) Spiral/Trigonometric Encoding**: They hypothesized high-dimensional spirals; this work "slices" the spiral into IRST fibers + digit basins, providing a local structure closer to failure modes.
- **vs. Sun et al. (2025) / Su et al. (2024) Ground-truth Probes**: They proved the correct answer can be read from failed samples; this paper provides the geometric "why" and converts this capability into a functional inference-time correction interface.
- **vs. Quirke et al. (2025) Symbolic Circuit Reverse Engineering**: While they interpret arithmetic as discrete look-up tables, this work argues for internal continuous potential + quantization, suggesting both may coexist at different scales or tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Model Merging Scaling Laws in Large Language Models](model_merging_scaling_laws_in_large_language_models.md)
- [\[ICML 2026\] NanoQuant: Efficient Sub-1-Bit Quantization of Large Language Models](nanoquant_efficient_sub-1-bit_quantization_of_large_language_models.md)
- [\[ACL 2026\] LightReasoner: Can Small Language Models Teach Large Language Models Reasoning?](../../ACL2026/model_compression/lightreasoner_can_small_language_models_teach_large_language_models_reasoning.md)
- [\[ICML 2026\] Bounded Hyperbolic Tangent: A Stable and Efficient Alternative to Pre-Layer Normalization in Large Language Models](bounded_hyperbolic_tangent_a_stable_and_efficient_alternative_to_pre-layer_norma.md)
- [\[ICML 2026\] Jailbreak to Protect: Buffering and Reinforcing via Temporary Jailbreaking for Safe Fine-Tuning in Large Language Models](jailbreak_to_protect_buffering_and_reinforcing_via_temporary_jailbreaking_for_sa.md)

</div>

<!-- RELATED:END -->
