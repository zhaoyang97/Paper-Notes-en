---
title: >-
  [Paper Note] Edit-Based Flow Matching for Temporal Point Processes
description: >-
  [ICLR 2026][Image Generation][Temporal Point Processes] The paper proposes EDITPP, which models the generation of Temporal Point Processes (TPP) as an edit flow on a Continuous-Time Markov Chain (CTMC). By using three types of atomic edits—insertion, deletion, and substitution—it transports noise sequences to target event sequences. It achieves or approaches SOTA performance in both unconditional generation and conditional forecasting tasks while reducing edit steps and signi…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "Temporal Point Processes"
  - "Flow Matching"
  - "Edit Operations"
  - "Conditional Generation"
  - "Non-autoregressive Sampling"
date: 2026-05-08
content_hash: 6a8be07622d09c53
---

# Edit-Based Flow Matching for Temporal Point Processes

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=FNf9IV1P2L](https://openreview.net/forum?id=FNf9IV1P2L)  
**Paper**: [OpenReview Forum](https://openreview.net/forum?id=FNf9IV1P2L)  
**Code**: Mentioned in the text as `cs.cit.tum.de/daml/editpp` (subject to official release)  
**Area**: time_series / learning_theory (Currently in `image_generation`, should be reclassified)  
**Keywords**: Temporal Point Processes, Flow Matching, Edit Operations, Conditional Generation, Non-autoregressive Sampling  

## TL;DR
The paper proposes EDITPP, which models the generation of Temporal Point Processes (TPP) as an edit flow on a Continuous-Time Markov Chain (CTMC). By using three types of atomic edits—insertion, deletion, and substitution—it transports noise sequences to target event sequences. It achieves or approaches SOTA performance in both unconditional generation and conditional forecasting tasks while reducing edit steps and significantly accelerating sampling.

## Background & Motivation
**Background**: The classic approach for TPP is modeling the conditional intensity function $\lambda(t \mid \mathcal{H}_t)$. Neural methods often adopt an autoregressive paradigm (RNN/Transformer encoding history to predict the next event). While expressive, these methods are inherently serial during inference, leading to slow sampling for long sequences and error accumulation during multi-step prediction.

**Limitations of Prior Work**: Recently, non-autoregressive TPP generation has emerged (e.g., ADDTHIN and PSDIFF based on diffusion/set interpolation). These can refine sequences holistically but rely primarily on two types of transformations: insertion and deletion. When the real data only requires a "local shift of a timestamp," these models must indirectly achieve it via "delete then insert," resulting in redundant paths and high edit costs.

**Key Challenge**: TPP events are variable-length sets in continuous time. It is necessary to preserve the interpretability of discrete operations in the edit process while ensuring the model remains trainable and sampleable in continuous time. Directly enumerating all edit paths from $t_s$ to $t_1$ in the original space makes the conditional path distribution difficult to integrate, turning training and inference into intractable combinatorial problems.

**Goal**:
1. Define a closed and composable set of edit operations for TPP, ensuring any sequence transformation can be realized through finite edit steps.  
2. Utilize the flow matching concept within a CTMC to learn instantaneous edit rates rather than explicitly regressing the intensity function.  
3. Establish a unified framework supporting both unconditional sampling and conditional sampling (especially forecasting), and validate the quality-efficiency trade-off.

**Key Insight**: The authors borrow the idea of Edit Flow from discrete sequences and transfer "insertion/deletion/substitution" to continuous-time event sequences. Simultaneously, they introduce an alignment auxiliary space to convert the "hard-to-integrate edit path" problem into a sampleable and supervised local edit rate matching problem.

**Core Idea**: Unify TPP generation via "continuous-time edit flow + alignment space supervision." By extending the insert/delete interpolation found in traditional diffusion routes to a ternary system of insert/delete/substitute, the generation path is shortened and edit efficiency is improved.

## Method

### Overall Architecture
EDITPP denotes an event sequence as $t = \{t^{(1)},\dots,t^{(n)}\}$ (sorted by time within the interval $[0,T]$). Generation begins from a noise TPP sample $t_0 \sim p_{noise}(t)$, evolves along a continuous time variable $s\in[0,1]$, and results in $t_1 \sim q_{target}(t)$. The core involves learning a rate model $u_\theta^s(\omega \mid t_s)$, which outputs the instantaneous rate of performing various edit operations $\omega$ on the current sequence.

Unlike "directly parameterizing the conditional intensity function and performing point process sampling," EDITPP parameterizes "how to modify the sequence." Tasks shift from intensity modeling to edit dynamics modeling. After discretizing the CTMC via Euler approximation, multiple mutually exclusive edits can be sampled in parallel and applied to the sequence at each step.

```mermaid
graph TD
	A[Noise Event Sequence t0] --> B[Edit Operation Discretization<br/>Insert, Delete, Substitute]
	B --> C[Alignment Auxiliary Space Construction<br/>z0, z1 and Minimal Cost Alignment]
	C --> D[Edit Rate Learning<br/>uθs(ω | ts)]
	D --> E[CTMC Euler Sampling<br/>Parallel Application of Edits]
	E --> F[Conditional Constraint Reprojection<br/>Fixed History Subsequence]
	F --> G[Output Target Sequence t1]
```

### Key Designs
**1. Ternary Edit System on Continuous Time: Shortening Paths via Substitution**

A key contribution of this work is expanding the atomic operations in TPP generation from "insertion + deletion" to "insertion + deletion + substitution." Substitution is essentially a "local delete-insert shortcut": when an event timestamp only requires a small shift, a direct substitution can complete it in one step instead of two. This supports the claim of "reducing total edit steps" and explains the advantages in edit efficiency.

To discretize continuous-time events into enumerable edits, the authors construct insertion bins between adjacent events and set a maximum movement radius $\delta$ for substitutions. This creates determinable boundaries for "which edit occurs at a certain step," avoiding supervision ambiguity caused by multiple edits explaining the same target state.

**2. Alignment Auxiliary Space: Converting Non-integrable Path Supervision to Computable Local Supervision**

Directly modeling $p_s(t_s\mid t_0,t_1)$ requires summing over all possible edit paths, which leads to combinatorial explosion. The authors introduce an alignment space with a blank symbol $\epsilon$, mapping source/target sequences to $z_0, z_1$. A carefully designed cost function ensures that the minimum cost alignment matches the defined edit operations. Consequently, each alignment site naturally corresponds to an edit label (ins/sub/del).

The training objective is formulated as a Bregman divergence, effectively matching the model's predicted edit rate with the ground truth conditional edit rate provided by the aligned samples. The loss function is summarized as:
$$
\mathcal{L}=\mathbb{E}_{(z_0,z_1),s,\,p_s(z_s,t_s\mid z_0,z_1)}\Big[\sum_{\omega\in\Omega(t_s)}u_\theta^s(\omega\mid t_s)
-\sum_{z_s^{(i)}\neq z_1^{(i)}}\frac{\dot\kappa_s}{1-\kappa_s}\log u_\theta^s(\omega(z_s^{(i)},z_1^{(i)})\mid t_s)\Big].
$$
Intuitively, the first term controls the total rate scale, while the second term maximizes the log-likelihood of ground truth edit events.

**3. Unified Conditional Sampling: Unconditional and Forecasting via the Same Model**

The authors present a conditional sampling mechanism based on a binary time mask $c(t)$. Given a conditional subsequence $C(t)$ (e.g., a history window) and its complement $C'(t)$, sampling forces $C(t)$ to follow the interpolation trajectory from noise to target while $C'(t)$ is generated freely. Each Euler step involves a "edit update + conditional reprojection + merge" process, ensuring historical constraints remain intact.

This is critical: the model is not specifically trained with forecasting supervision, but it performs the prediction task through re-conditioning during sampling. This explains how "unconditional training with conditional inference" still competes with strong baselines.

**4. Rate Parameterization and Parallel Sampling: Balancing Expressiveness and Efficiency**

The architecture utilizes a Llama-style Transformer + FlexAttention to handle variable-length sequences without padding. Event scalars undergo SinEmb followed by MLP projection; the time step $s$ and sequence length are injected as tokens. Output heads parameterize $\lambda_{ins}, \lambda_{sub}, \lambda_{del}$ and corresponding distributions $Q_{ins}, Q_{sub}$:
$\lambda=\exp(\lambda_M\tanh(h))$, $Q=\mathrm{softmax}(h_Q)$.

This parameterization ensures non-negative rates and normalized distributions, aligning naturally with CTMC sampling formulas. The paper emphasizes that multiple edits can be applied in parallel within the same step, providing a significant speed advantage over previous methods.

### Loss & Training
Training was conducted on 7 real-world and 6 synthetic datasets. All models used 5 random seeds, selecting the best checkpoint based on validation set metrics. EDITPP, ADDTHIN, and PSDIFF utilize unconditional training but can switch to conditional tasks during inference; they are compared against the strong autoregressive baseline IFTPP.

Metrics cover both distribution-level and task-level categories:
1. Unconditional generation: MMD, $W_1$ over event counts ($d_l$), $W_1$ over inter-event time ($d_{IET}$).  
2. Conditional forecasting: $d_{Xiao}$, MRE, $d_{IET}$.

## Key Experimental Results

### Main Results
The paper evaluates on 13 datasets. In unconditional generation, EDITPP ranks best overall. In conditional forecasting, EDITPP is closely comparable to PSDIFF, generally outperforms ADDTHIN, and is more robust than IFTPP across multiple datasets.

| Task | Metric | EDITPP Conclusion | Comparison Conclusion |
|------|------|-------------|----------|
| Unconditional Gen | MMD / $W_1(d_l)$ / $W_1(d_{IET})$ | Ranked 1st overall | Better than or equal to IFTPP/PSDIFF/ADDTHIN in most cases |
| Conditional Forecasting | $d_{Xiao}$ / MRE / $d_{IET}$ | Same level as PSDIFF, strong overall | Generally better than ADDTHIN, outperforms IFTPP in many scenarios |
| Sampling Efficiency | Edit Steps / Runtime | Fewer edits, faster speed | Significantly faster than PSDIFF and ADDTHIN |

### Ablation Study
The most critical ablations involve "edit efficiency" and the "inference steps-quality curve":

| Configuration | Metric | Result | Description |
|------|------|------|------|
| PSDIFF (ins+del) | Avg. Edit Count | 234.52 | No substitution, longer paths |
| EDITPP (ins+del+sub) | Avg. Edit Count | 199.65 | Substitutions replace some delete+insert pairs |
| Fewer Euler Steps | Quality Metrics | Slight Drop | More computationally efficient, suitable for low latency |
| More Euler Steps | Quality Metrics | Improvement | Diminishing returns; allows for quality-speed trade-off |

The edit count breakdown for EDITPP: 137.42 inserts, 33.08 deletes, and 29.16 substitutions. The total is lower than PSDIFF (173.48/61.04/0.00, total 234.52).

### Key Findings
- Substitution is a structural improvement that significantly reduces path length rather than just an auxiliary feature: a single substitution saves two operations when events only need local shifting.  
- Unconditional training + conditional inference is effective in this framework, suggesting the edit flow learns general sequence dynamics rather than just memorizing next-step conditional densities.  
- Increasing sampling steps improves quality, but gains diminish quickly, providing a clear control for dynamic budgets in deployment.  

## Highlights & Insights
- Reformulates "point process generation" from an intensity function perspective to an "edit dynamics" perspective. It matches the flow matching training paradigm by learning how to modify sequences directly.  
- The alignment space design is practical. It transforms the path summation problem into sampleable supervision, balancing theoretical consistency with implementation.  
- Simplified conditional sampling mechanism. By defining history/future via masks and re-conditioning at each step, there is no need to train a separate model for forecasting.  
- EDITPP achieves significant speed gains without sacrificing quality, demonstrating that "alignment of edit semantics with the task" translates into real inference dividends.  

## Limitations & Future Work
- Substitution relies on discretization and the threshold $\delta$. If the data time scales vary significantly, a fixed threshold might not be robust. Adaptive $\delta$ or multi-scale substitution kernels could be considered.  
- Currently, the setting focuses on timestamp sequences. With high-dimensional marks (complex event types/attributes), the edit space expands rapidly, requiring more structured factorized edit parameterization.  
- Euler approximation of CTMC still carries discretization errors. Although more steps can mitigate this, it increases latency. Future work could explore higher-order or adaptive step-size integration strategies.  
- Alignment quality depends on the cost design. If alignment is biased, supervision becomes systematically contaminated. Robustness could be enhanced by introducing learnable costs or uncertainty-aware alignment mechanisms.  

## Related Work & Insights
- **vs ADDTHIN / PSDIFF (Non-autoregressive diffusion-style TPP)**: These focus on insert/delete interpolation. EDITPP adds substitution and CTMC rate modeling, resulting in shorter paths and faster sampling, though it requires thresholding and discretization designs.  
- **vs IFTPP (Autoregressive baseline)**: IFTPP is strong at single-step modeling, but multi-step prediction suffers from error accumulation and serial sampling. EDITPP is more stable for forecasting via global sequence editing and offers better parallelism.  
- **vs Classic Intensity Function Methods (Hawkes/Neural TPP)**: Classic routes emphasize the interpretability of $\lambda(t\mid\mathcal{H}_t)$. EDITPP shifts toward the operability of the generation process. The two can be complementary, such as using intensity priors to constrain edit rates.  

## Rating
- Novelty: ⭐⭐⭐⭐☆ (Systematically migrates Edit Flow to TPP, providing a unified training-sampling loop for substitute+CTMC)
- Experimental Thoroughness: ⭐⭐⭐⭐☆ (13 datasets, dual tasks of unconditional and conditional generation, includes efficiency analysis and ablations)
- Writing Quality: ⭐⭐⭐⭐☆ (Clear method definitions and algorithm flows; detailed experimental appendix)
- Value: ⭐⭐⭐⭐☆ (Directly valuable for event generation/prediction tasks, especially in scenarios sensitive to parallel sampling and efficiency)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Flow Matching Neural Processes](../../NeurIPS2025/image_generation/flow_matching_neural_processes.md)
- [\[ICLR 2026\] Delay Flow Matching](delay_flow_matching.md)
- [\[ICLR 2026\] Flow Matching with Semidiscrete Couplings](flow_matching_with_semidiscrete_couplings.md)
- [\[ICLR 2026\] Source-Guided Flow Matching](source-guided_flow_matching.md)
- [\[ICML 2026\] Shifting the Breaking Point of Flow Matching for Multi-Instance Editing](../../ICML2026/image_generation/shifting_the_breaking_point_of_flow_matching_for_multi-instance_editing.md)

</div>

<!-- RELATED:END -->
