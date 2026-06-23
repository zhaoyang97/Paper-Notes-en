---
title: >-
  [Paper Note] Towards a Theoretical Understanding of In-Context Learning: Stability and Non-i.i.d. Generalisation
description: >-
  [ICLR 2026][learning_theory][Error Accumulation] Under realistic conditions without assuming token orthogonality or i.i.d. sampling, this paper employs "algorithmic stability + discrepancy measure" to derive generalization error bounds for non-linear Transformers in the context of ICL next-token prediction. It reveals how optimization configurations and loss smoothne
tags:
  - ICLR 2026
  - learning_theory
  - Error Accumulation
date: 2026-05-08
content_hash: 8c2f1ddc3cb51206
---
# Towards a Theoretical Understanding of In-Context Learning: Stability and Non-i.i.d. Generalisation

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=Y8oiuzaAxl](https://openreview.net/forum?id=Y8oiuzaAxl)  
**Code**: None  
**Area**: Learning Theory / In-Context Learning  
**Keywords**: In-Context Learning, Algorithmic Stability, Distribution Shift, Generalization Bounds, Error Accumulation

## TL;DR
Under realistic conditions without assuming token orthogonality or i.i.d. sampling, this paper employs "algorithmic stability + discrepancy measure" to derive generalization error bounds for non-linear Transformers in the context of ICL next-token prediction. It reveals how optimization configurations and loss smoothness collectively determine stability, how training/test distribution alignment dictates generalizability, and proves that unconstrained autoregressive prediction length leads to error accumulation and generalization collapse.

## Background & Motivation
**Background**: ICL enables large models to predict on downstream tasks using only a few-shot prompt without parameter fine-tuning. This capability has recently attracted significant theoretical analysis. Existing works (e.g., Li et al. 2023, Huang et al. 2024, Chen et al. 2024, Wu et al. 2024) have attempted to provide generalization bounds or characterize training dynamics, with some proving that pre-trained attention models can approach the Bayes risk of optimal ridge regression.

**Limitations of Prior Work**: These analyses generally rely on **idealized data assumptions** to make proofs tractable—either imposing "pairwise orthogonal token patterns" (Huang et al. 2024, Li et al. 2024a), assuming "i.i.d. token sampling" (Chen et al. 2024, Wu et al. 2024), or providing optimization-independent bounds. These assumptions rarely hold in real-world scenarios where text tokens are highly correlated and training/inference distributions often diverge.

**Key Challenge**: To ensure a theory satisfies the full set of realistic conditions—non-linear multi-head multi-layer Transformer, dependency on the optimization process, allowance for distribution shift, no requirement for special input structures, and no orthogonality—significantly increases proof complexity. Prior works typically relax only one or two of these conditions (see original Table 1); none address them all.

**Goal**: In a non-i.i.d. scenario that closely mirrors reality, this paper aims to quantify two aspects: (1) when a Transformer trained via mini-batch gradient descent becomes stable; (2) how the discrepancy between training and target prompt distributions affects generalization, and then synthesize these into a convergent ICL generalization error bound.

**Key Insight**: Rather than assuming any latent "conceptual geometry," the authors leverage two established tools from statistical learning theory: **algorithmic stability** (characterizing sensitivity to training data perturbations) and **discrepancy measure** (characterizing divergence between training and target distributions), adapting them to the context of "autoregressive next-token prediction + Transformer."

**Core Idea**: Generalization error $\approx$ empirical risk + distribution discrepancy disc(q) + stability term $\beta$. These three components are individually controllable and together yield an $O(N^{-1/2})$ convergence rate. However, since autoregression feeds "self-predicted tokens" into the next step, errors accumulate per token, necessitating that prediction length be constrained to at most logarithmic growth.

## Method

### Overall Architecture
This work is purely theoretical. The "Method" consists of a sequential chain of proofs: **Problem Modeling → Stability Bound (Theorem 1) → Discrepancy Bound (Theorem 2–3) → Generalization Error Bound (Theorem 4) → Error Accumulation Constraint (Theorem 5)**.

Regarding problem setup, given $N$ samples $\{(X_i, C_i)\}$, where $X_i$ is a query and $C_i=(C_i^1,\dots,C_i^{N_c})$ is an output sequence of length $N_c$ (**allowing each sample to follow a different distribution**). An ICL prompt $P_i=[D_i, X_i]$ consists of an exemplar set $D_i$ concatenated with a query. When predicting the $j$-th token autoregressively, the prompt is $P_{i,j}=[P_i, C_i^1,\dots,C_i^{j-1}]$. In practice, next-token prediction uses the model's **own estimated** intermediate tokens, denoted $\hat P_{i,j}=[\hat P_{i,j-1}, T(\hat P_{i,j-1})]$, which is the source of error accumulation. The model $T$ is a standard $L$-layer non-linear Transformer (multi-head self-attention + ReLU-MLP), trained via mini-batch GD (Algorithm 1) using weighted empirical risk:

$$\hat L(T)=\sum_{i=1}^{N}\frac{q_i}{N_c}\sum_{j=1}^{N_c}\ell\big(T(p_{i,j-1})_{*,:},\,c_i^j\big)$$

where $q_i$ are training sample weights. The goal is to control the gap between population risk and empirical risk $L(T_S)-\hat L(T_S)$. The three key designs correspond to the foundation of the proof chain: stability, discrepancy, and the synthetic generalization bound.

### Key Designs

**1. Algorithmic Stability Bound for mini-batch GD: Linking Optimization, Smoothness, and Stability**

Most existing ICL generalization bounds are optimization-independent, failing to show how "iteration steps, batch size, and learning rate" affect generalization. Addressing this, the authors define a **uniform stability** $\beta$ adapted for autoregressive ICL: if the $i$-th sample in training set $S$ is replaced by an i.i.d. sample to form $S^i$, the average loss change across all query positions is bounded by $\beta$. Theorem 1, under boundedness assumptions (Assumption 1), with learning rate $\eta_k=k^{-\alpha}$, provides a piecewise bound depending on the Lipschitz smoothness constant $\gamma$:

$$\beta \lesssim \begin{cases}\dfrac{B M_\ell L_\ell^{2/(\alpha(1+\gamma))}\,Q^{\gamma/(1+\gamma)}}{N^{\gamma\alpha}}, & \gamma\le \frac{1+\sqrt{1-4\alpha(1-\alpha)}}{2\alpha},\\[2mm]\dfrac{B M_\ell L_\ell^{2/(\alpha(1+\gamma))}\,Q^{(\alpha\gamma^2+1-\alpha)/(1+\gamma)}}{N^{\gamma\alpha}}, & \text{otherwise.}\end{cases}$$

This provides three actionable conclusions: (1) When the loss landscape is **sufficiently smooth** ($\gamma$ is small), stability is controllable, and iterations $Q$ can grow polynomially with sample size $N$ (Corollary 1, capturing the trade-off between reducing empirical risk and magnifying perturbations); (2) When **non-smooth** ($\gamma$ is large), stability degrades sharply with $Q$, particularly with small learning rates, requiring $Q$ to be restricted to $O(\ln N)$ (Corollary 2); (3) Regardless of smoothness, proper step size selection allows stability convergence of $O(N^{-1})$. Additionally, $\beta$ worsens as batch size $B$ increases, aligning with the empirical observation that small-batch SGD generalizes better. Since depth $L$ causes exponential growth in terms like $M_\ell$, stability requires depth to grow at most logarithmically with $N$.

**2. Hypothesis-Independent Discrepancy Measure: Quantifying Training-Target Mismatch**

In reality, training and target domains often differ. This requires a metric to measure divergence **without specific distribution assumptions**. The authors adapt the discrepancy measure from Kuznetsov & Mohri into a form independent of the hypothesis space:

$$\mathrm{disc}(q):=\frac{1}{N_c}\sum_{j=1}^{N_c}\Big[E_{N+1,j}-\sum_{i=1}^{N}q_i E_{i,j}\Big],\quad E_{i,j}=\mathbb{E}\big[\ell(T_S(P_{i,j-1})_{*,:},C_i^j)\,\big|\,\{(p_m,c_m)\}_{m=1}^{i-1}\big].$$

This measures the mismatch between the target distribution and the (weighted by $q$) training distribution. It is quantified in two scenarios: in i.i.d. cases (Theorem 2), $\mathrm{disc}(q)\le 2\beta\|q\|_2 N\sqrt{\log(2/\delta)}$, where discrepancy vanishes asymptotically as long as $\beta\|q\|_2 N\to 0$ (e.g., $\beta=o(N^{-1/2})$ for uniform weights $q_i=1/N$). This directly links discrepancy to stability $\beta$. In non-i.i.d. cases (Theorem 3), assuming at least some training domains are relevant to the target and a valid prompt exists such that mismatch $\le\epsilon$, an upper bound is provided based on the **Sequential Rademacher Complexity** $R_N(\{\ell\circ T\})$, containing the weight difference $\|q-v\|$. This provides a theoretical explanation for why reweighting training samples during fine-tuning works and shows that more complex hypothesis spaces are more sensitive to distribution shifts.

**3. Synthetic Generalization Bound and Error Accumulation: Combining Stability and Discrepancy**

With the previous components, Theorem 4 provides the synthetic generalization error bound:

$$L(T_S)\le \frac{1}{N_c}\sum_{i=1}^{N}\sum_{j=1}^{N_c}q_i\,\ell(T_S(p_{i,j})_{*,:},c_i^j)+\mathrm{disc}(q)+\|q\|_1\beta+2\|q\|_2 M_\ell\sqrt{2\log(4/\delta)}.$$

This represents "Empirical Risk + Discrepancy + Stability Term + Confidence Term." Asymptotic behavior (Corollary 3–4, summarized in original Table 2) shows: under i.i.d. conditions, regardless of smoothness, tuning hyperparameters such that $|B|=O(N^{\zeta_1})$ and $Q = O(N^{\zeta_2})$ can achieve the optimal convergence rate of $O(N^{-1/2})$. In non-i.i.d. settings, generalization requires "reasonable sample reweighting (small $\|q-v\|$) + a good ICL prompt," resulting in a two-stage optimization with $\lambda_1\|s-q\|_2^2+\lambda_2\|q\|_2^2$ regularization. Finally, Theorem 5 (Error Accumulation) notes that errors accumulate during autoregression; to ensure generalization, the **next-token prediction length should grow at most logarithmically with the sample size**, otherwise errors spike and generalization collapses.

### Loss & Training
Training uses the mini-batch GD in Algorithm 1: in each step, a batch $B$ is sampled, and parameters are updated via $\theta_q=\theta_{q-1}-\frac{\eta_{q-1}}{|B|}\sum_{i\in B}\nabla_\theta\hat L(T)$, with learning rate decay $\eta_k=k^{-\alpha}$. A practical suggestion for non-i.i.d. scenarios is solving the two-stage weighted objective (Eq. 4 in Remark 5): first optimize the last three terms for optimal weights $q$, then optimize the first term for model parameters.

## Key Experimental Results

Experiments are purely for verification (as this is a theoretical paper). A 12-layer 8-head GPT-2 was used on H20 GPUs following the setup of Li et al. 2023. The task was $d=10$ dimensional linear regression, where sequences were generated via $c_l^i=\beta_{l-1}^i c_{l-1}^i+\epsilon$. Parameters: $N\in\{50, \dots, 1600\}$, uniform weights $q_i=1/N$, $|B|=N^{1/2}$, $Q=200$, $\alpha=1$, sequence length $N_c\in\{1,\dots,9\}$.

### Main Results (Asymptotic Convergence Verification)

| Phenomenon | Configuration | Observation | Theoretical Correspondence |
|------|---------|------|---------|
| Error Convergence | Length 1, 2; $N$ from 50→1600 | Error decreases and vanishes as $N\to\infty$ | Corollary 3 (i.i.d., $O(N^{-1/2})$) |
| Error Accumulation | Fixed $N$; $N_c$ from 1→9 | Error grows polynomially; spikes after $\ln N$ threshold | Theorem 5 (Error Accumulation) |

### Ablation Study / Analysis

| Configuration | Key Observation | Explanation |
|------|---------|------|
| Increase $N_c$ | Error spike point shifts right as $N$ increases | Threshold $\approx\ln N$; validates logarithmic growth limit |
| Contrast across $N$ | Larger $N$ tolerates longer predictions | More samples allow more safe autoregressive steps |
| Non-i.i.d. Extension | Consistent with theoretical bounds | Validates impact of discrepancy and prompt reweighting |

### Key Findings
- **The logarithmic threshold for error accumulation is the most intuitive verifiable conclusion**: Once sequence length exceeds approximately $\ln N$, generalization error transitions from polynomial growth to a sharp spike. This threshold shifts right with larger $N$, precisely confirming the Theorem 5 constraint.
- **Small batches are more stable**: The stability bound $\beta$ worsens with larger batch sizes, providing a stability-based theoretical support for the empirical observation that small-batch SGD generalizes better.
- **Distribution alignment is key for non-i.i.d. generalization**: Smaller $\|q-v\|$ (training weights closer to target) leads to better generalization, explaining why fine-tuning is effective.

## Highlights & Insights
- **Relaxation of All Idealized Assumptions**: Unlike prior works that relax only one or two, this work simultaneously handles multi-head multi-layer structure + optimization dependency + distribution shift + no special input structure + no orthogonality (the only all-✓ in original Table 1).
- **Unified Bound matching Stability and Discrepancy**: $\beta$ appears in both the stability term and the discrepancy bound, unifying optimization knobs ($Q, |B|, \alpha$), loss smoothness $\gamma$, and distribution alignment $\|q-v\|$ into a single framework.
- **Hard Constraint on Autoregression**: The $\ln N$ threshold transforms the intuition that "longer generation is less accurate" into a quantifiable theoretical boundary, offering insights for long-sequence generation design.
- **Theoretical Explanation for Fine-tuning and Sample Reweighting**: The $\|q-v\|$ term clarifies that fine-tuning is equivalent to reweighting training samples to match the target, leading to a two-stage weighted training objective solvable via DC programming.

## Limitations & Future Work
- **Stability bound relies on boundedness assumptions** (Assumption 1, bounded input/parameter norms). While authors suggest this can be relaxed to sub-Gaussian distributions, the full proof remains within this framework.
- **Experiments are limited to synthetic linear regression + GPT-2**, without touching real-world linguistic tasks; a gap remains between this theory and large-scale practical ICL.
- **Constants in the bound grow exponentially with depth $L$** ($M_\ell$, etc.), requiring depth to grow at most logarithmically with $N$ for the bound to stay tight.
- Future work intends to use finer tools like gradient stability to tighten bounds and apply weighted training strategies to practical algorithm design.

## Related Work & Insights
- **vs. Li et al. (2023)**: They provide optimization-independent bounds. This work explicitly incorporates the optimization process (mini-batch GD parameters) and extends to non-i.i.d. cases.
- **vs. Huang et al. (2024) / Li et al. (2024a)**: They rely on orthogonal token patterns and shallow attention. This work addresses multi-layer Transformers without orthogonality.
- **vs. Chen et al. (2024) / Wu et al. (2024)**: They assume i.i.d. sampling. This work handles distribution shift explicitly via discrepancy measures.
- **vs. Bu et al. (2024, 2025)**: They look at mechanistic explanations via conceptual geometry. This work is complementary, bypassing structural assumptions to build a shift-aware generalization framework.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to relax orthogonality, i.i.d., and single-layer assumptions simultaneously while unifying stability and discrepancy.
- Experimental Thoroughness: ⭐⭐⭐ Verification-only experiments (synthetic linear regression + GPT-2); limited scale but matches theory.
- Writing Quality: ⭐⭐⭐⭐ Clear proof chain, effective summary tables, and well-defined assumptions/conclusions.
- Value: ⭐⭐⭐⭐ Provides actionable theoretical evidence for ICL generalization, error accumulation constraints, and fine-tuning/reweighting strategies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Understanding In-Context Learning on Structured Manifolds: Bridging Attention to Kernel Methods](understanding_in-context_learning_on_structured_manifolds_bridging_attention_to_.md)
- [\[ICLR 2026\] Understanding the Dynamics of Forgetting and Generalization in Continual Learning via the Neural Tangent Kernel](understanding_the_dynamics_of_forgetting_and_generalization_in_continual_learnin.md)
- [\[ICLR 2026\] Transformers with Endogenous In-Context Learning: Bias Characterization and Mitigation](transformers_with_endogenous_in-context_learning_bias_characterization_and_mitig.md)
- [\[ICLR 2026\] Pretrain–Test Task Alignment Governs Generalization in In-Context Learning](pretraintest_task_alignment_governs_generalization_in_in-context_learning.md)
- [\[ICLR 2026\] On learning linear dynamical systems in context with attention layers](on_learning_linear_dynamical_systems_in_context_with_attention_layers.md)

</div>

<!-- RELATED:END -->
