---
title: >-
  [Paper Note] SPARD: Defending Harmful Fine-Tuning Attack via Safety Projection with Relevance-Diversity Data Selection
description: >-
  [ICML 2026][LLM Alignment][Harmful Fine-tuning Attack] SPARD combines "Safety-Projected Alternating Gradient (SPAG)" and "Relevance-Diversity DPP Safety Data Selection." It explicitly formulates the requirement that a fi…
tags:
  - "ICML 2026"
  - "LLM Alignment"
  - "Harmful Fine-tuning Attack"
  - "Safety Projection"
  - "Determinantal Point Process (DPP)"
  - "Safety Constrained Optimization"
  - "LoRA"
date: 2026-05-08
content_hash: a43950c44ef94d57
---

# SPARD: Defending Harmful Fine-Tuning Attack via Safety Projection with Relevance-Diversity Data Selection

**Conference**: ICML 2026  
**arXiv**: [2605.28030](https://arxiv.org/abs/2605.28030)  
**Code**: https://github.com/shuhao02/SPARD (Available)  
**Area**: LLM Safety / Alignment RLHF / Harmful Fine-tuning Defense  
**Keywords**: Harmful Fine-tuning Attack, Safety Projection, Determinantal Point Process (DPP), Safety Constrained Optimization, LoRA

## TL;DR
SPARD combines "Safety-Projected Alternating Gradient (SPAG)" and "Relevance-Diversity DPP Safety Data Selection." It explicitly formulates the requirement that a fine-tuned model must satisfy safety loss constraints as a constrained optimization problem. Each step performs a utility update followed by a closed-form projection to pull parameters back into the safety half-space. Using only 3% task-relevant and mutually diverse safety samples, it reduces the average ASR of four harmful fine-tuning attacks from 87.93% (SFT) to 9.45% with negligible loss in downstream accuracy.

## Background & Motivation

**Background**: Fine-tuning-as-a-service has become the mainstream for LLM deployment. However, downstream fine-tuning can rapidly erode the safety alignment established during pre-training. Attackers can bypass safety guardrails by injecting a small number of malicious samples (harmful fine-tuning attack) into user-uploaded data.

**Limitations of Prior Work**: Existing defenses generally follow two paradigms. One involves prompt/structural constraints (PTST, SafeLoRA), which rely on injecting safety prompts at inference or limiting LoRA subspaces, making them sensitive to templates and layer selection with poor generalizability. The other involves safety data regularization (SafeInstr, Lisa, SafeGrad), which adds a small set of safety samples as penalty/proximal terms in the loss. Two fatal flaws exist in the latter: (i) safety constraints are only soft penalties where the penalty coefficient $\lambda$ is difficult to tune, lacking explicit control over the utility-safety trade-off; (ii) safety samples are usually selected randomly, ignoring the fact that task-relevant safety samples provide stronger corrective signals.

**Key Challenge**: Essentially, a geometric conflict exists between the "utility gradient" and the "safety gradient." Penalty methods linearly mix the two, failing to guarantee feasibility or adapt corrective intensity across different tasks/attacks. Furthermore, the "quality" dimension of safety samples (relevance to $D_{ft}$ + subset diversity) has never been jointly optimized.

**Goal**: (a) Formulate safety fine-tuning explicitly as a constrained optimization problem: $\min_\theta L(D_{ft},\theta)$ s.t. $L(D_{safe},\theta)\le\tau$, and provide a closed-form solution requiring no parameter tuning; (b) Design a safety data selector that jointly models relevance and diversity so that a small $D_{safe}$ can cover broad risks.

**Key Insight**: A key experiment in Figure 2 shows that when safety samples are binned by cosine similarity to $D_{ft}$, ASR monotonically decreases as similarity rises (68.8% → 11.4%), but rebounds to 16.6% when similarity is too high (≈0.94). This indicates that both relevance and diversity are indispensable; similarity-only selection causes the safety set to collapse into a narrow risk zone.

**Core Idea**: Upgrade both "soft safety penalties" and "random data selection." Use projection to define safety constraints as geometric feasibility conditions and employ Relevance-Diversity DPP to jointly encode relevance weights $(q_iq_j)^\beta$ and sample distinctness in the kernel.

## Method

### Overall Architecture
The SPARD pipeline consists of two sequential phases:

1.  **Offline Safety Data Selection**: From a safety corpus *GeneralSafe*, embeddings are calculated using the mean hidden state of the last layer of a pre-trained LLM. The maximum cosine similarity between a candidate sample and $D_{ft}$ is used as the relevance score $q_i$. Build a relevance-diversity kernel $\widehat{K}_{ij}=(q_iq_j)^\beta K(x_i,x_j)$, and solve for the DPP subset $D_{safe}$ using a greedy MAP approach based on Cholesky incremental updates, with a fixed size of 3% of $D_{ft}$.
2.  **Online SPAG Alternating Optimization**: At each step, a LoRA utility update is first performed on a mini-batch of $D_{ft}$ to obtain $\theta^+$, then the safety loss $\ell_{safe}$ is calculated on $D_{safe}$. If $\ell_{safe}\le\tau$, $\theta^+$ is retained; otherwise, a projection backstep with a closed-form step size is performed along $g_{safe}=\nabla L(D_{safe},\theta^+)$.

The final output is a LoRA-adapted "Safe Personalized LLM," maintaining downstream accuracy close to SFT while keeping safety guardrails resilient against attack data.

### Key Designs

1.  **SPAG: Safety-Projected Alternating Gradient**:
    -   **Function**: Embeds the "safety loss must not exceed threshold $\tau$" as a hard constraint into every LoRA update step without introducing penalty hyperparameters.
    -   **Mechanism**: Performs a first-order Taylor expansion of the constraint $L(D_{safe},\theta)\le\tau$ at $\theta^+$, yielding the half-space $C^+=\{\theta:L(D_{safe},\theta^+)+\langle g_{safe},\theta-\theta^+\rangle\le\tau\}$. The Euclidean projection of $\theta^+$ onto this half-space is solved via KKT conditions, giving the closed-form solution $\theta_{new}=\theta^+ - \frac{L(D_{safe},\theta^+)-\tau}{\|g_{safe}\|^2}g_{safe}$ (when violating the constraint) or $\theta^+$ (when satisfied). To prevent excessive steps from undermining training stability, a trust-region truncation is added: $\alpha=\min(\frac{\ell_{safe}-\tau}{\|g_{safe}\|^2},\eta_{safe})$.
    -   **Design Motivation**: The penalty leader $\lambda$ is a global constant, identical across all batches and training stages. In contrast, the projection step size $\alpha$ is adaptively determined by the current violation $\ell_{safe}-\tau$ and the gradient norm. This "on-demand projection" eliminates the burden of tuning $\lambda$ and theoretically guarantees first-order feasibility.

2.  **Relevance-Diversity DPP**:
    -   **Function**: Selects a compact, task-aligned, and behaviorally diverse $D_{safe}$ subset from a large safety candidate pool to ensure SPAG projection signals are "on target."
    -   **Mechanism**: Traditional DPP via $P(C)\propto\det(L_C)$ only rewards diversity. This paper inserts relevance scores $q_i=\max_{x_z\in D_{ft}}\text{sim}(x_i,x_z)$ as multiplicative weights into the kernel $\widehat{K}_{ij}=(q_iq_j)^\beta K(x_i,x_j)$, such that the selection probability factorizes as $P(C)\propto\prod_{x_i\in C}q_i^{2\beta}\cdot\det(L_C)$. The first term increases weights for relevant samples, while the second encourages diversity via determinant volume. $\beta$ controls the trade-off. A greedy MAP approach is used for solving: each step selects the sample maximizing the marginal determinant gain $\widehat{L}_{ii}-\|w_i\|^2$, reducing single-step complexity to $O(m)$.
    -   **Design Motivation**: The key observation in Figure 2 is the U-shaped ASR-relevance curve. Pursuing only relevance causes the safety set to collapse and overfit; pursuing only diversity fails to provide task-relevant correction. DPP naturally characterizes "both relevant and non-redundant," and incorporating $q_i$ into the kernel is more elegant than separate sorting and deduplication.

3.  **Geometric Significance of Trust-region Truncation**:
    -   **Function**: Caps the unbounded analytical projection step to a controllable "trust-region radius" $\eta_{safe}$, preventing a single step from pushing LoRA parameters to a point that collapses training.
    -   **Mechanism**: Note that the closed-form projection solution $\frac{\ell_{safe}-\tau}{\|g_{safe}\|^2}$ explodes when the gradient norm is extremely small. Borrowing from TRPO, the step size is capped at $\eta_{safe}$, experimentally set equal to the utility learning rate $\eta_{ft}$ ($5\times10^{-5}$). Table 4 shows that removing the trust-region drops ASR to 5.03% but degrades downstream accuracy from 85.77% to 81.92%.
    -   **Design Motivation**: The closed-form solution only holds locally within the first-order Taylor expansion. Long steps jump out of the linear approximation zone, destroying first-order feasibility and harming downstream knowledge that has already converged.

### Loss & Training
No new loss functions are introduced. The utility step uses standard task cross-entropy loss $L(D_{ft},\theta)$, and the safety step uses safety CE loss $L(D_{safe},\theta)$. LoRA rank $r=32$, alpha $=4$, AdamW, learning rate $5\times10^{-5}$, training for 10 epochs on GSM8K and 3 epochs on OpenBookQA. Safety threshold $\tau=0.2$, $D_{safe}$ ratio $p=0.03$, DPP relevance index $\beta=4$, safety mini-batch size of 1.

## Key Experimental Results

### Main Results

| Dataset / Model | Attack Setting | Metric | SFT | Lisa | SafeGrad | **SPARD** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| GSM8K / Qwen-2.5-7B | Avg of 4 Attacks | ASR ↓ | 87.93 | 19.12 | 30.18 | **9.45** |
| GSM8K / Qwen-2.5-7B | Avg of 4 Attacks | HS ↓ | 4.28 | 1.56 | 1.93 | **1.32** |
| GSM8K / Qwen-2.5-7B | Avg of 4 Attacks | GSM8K Acc ↑ | **86.77** | 78.45 | 85.71 | 85.77 |
| GSM8K / LLaMA-3.2-3B | Avg of 4 Attacks | ASR ↓ | 91.36 | 24.19 | 71.77 | **12.09** |
| GSM8K / LLaMA-3.2-3B | Avg of 4 Attacks | Acc ↑ | **72.27** | 65.03 | 64.29 | 71.23 |
| OpenBookQA / Qwen-2.5-7B | Avg of 4 Attacks | ASR ↓ | 40.29 | 18.80 | 20.63 | **14.54** |
| OpenBookQA / Qwen-2.5-7B | Avg of 4 Attacks | Acc ↑ | **83.70** | 78.90 | 83.30 | 83.25 |

SPARD achieves the lowest ASR/HS across all (model × task) combinations for all four attacks (BeaverTails / I-BeaverTails / LatHarmful / Q-LatHarmful). Downstream accuracy drops by at most 1 point compared to SFT, significantly outperforming SafeGrad (which has 20%+ higher ASR).

### Ablation Study

| Config | Avg ASR | GSM8K Acc | Description |
| :--- | :--- | :--- | :--- |
| Full SPARD | 9.45 | 85.77 | SPAG + Relevance-Diversity DPP + trust-region |
| SPARD w/o trust-region | 5.03 | 81.92 | No truncation: safer but loses 3.85% accuracy |
| SPARD w/ Random Selection | Higher | Similar | Relevance-Diversity DPP is crucial for lowering ASR |
| $\beta=0$ (Diversity-only DPP) | Higher | — | Fig 5 shows ASR is lowest at $\beta\in[2,4]$; too low or too high rebounds |
| $p=0$ (No safety data) | >80 | — | Degenerates to SFT equivalent |
| $p\in[0.03,0.05]$ | Lowest | — | Fig 3 shows 3-5% is the sweet spot; larger sets show diminishing returns |

### Key Findings
-   **Relevance $\neq$ Higher is always better**: Figure 2 reveals a U-shaped ASR-relevance curve (ASR rebounds to 16.6% at 0.94 similarity), empirically proving safety data must be both relevant and diverse.
-   **Trust-region as a utility knob**: Removing it suppresses ASR further to 5.03% (strongest safety), at the cost of downstream accuracy dropping from 85.77% to 81.92%. The step size provides a post-deployment tuning knob.
-   **Effectiveness with small samples**: Only $p=3\%$ safety samples reduce SFT's 87.93% ASR to 9.45%. Sampling only 1 safety sample per step makes the overhead significantly lower than bi-state optimization in Lisa.
-   **Cross-architecture stability**: For LLaMA-3.2-3B under LatHarmful, SFT ASR is 98.99%; SPARD still compresses it to 11.31%, indicating the projection mechanism does not rely on specific backbone vulnerabilities.

## Highlights & Insights
-   **Upgrading safety constraints to geometric projections**: While penalty methods treat constraints as soft targets with no feasibility guarantee, SPAG uses Taylor expansion and KKT closed-form solutions to achieve "on-demand projection." This eliminates $\lambda$ tuning and provides first-order feasibility. The engineering is simple (Algorithm 1 adds only 5 lines to SFT) while bringing the geometric perspective of constrained optimization to LLM safety fine-tuning.
-   **Elegant "surgery" on the DPP kernel**: Traditional DPP is an unsupervised diversity tool. The authors' addition of the $(q_iq_j)^\beta$ multiplicative factor allows the probability to factorize into $\prod q_i^{2\beta}\cdot\det(L_C)$. Relevance and diversity are managed by separate terms, and a single hyperparameter $\beta$ can sweep the entire trade-off spectrum. This trick is transferable to any subset selection problem like instruction tuning or RLHF preference pair selection.
-   **U-shaped curve as the insight catalyst**: While many works on "relevant data selection" stop after monotonic experiments, this paper pushed similarity to the extreme (0.94) to discover the rebound. This observation provides an empirical foundation for explicitly modeling diversity.

## Limitations & Future Work
-   **Limitations**: Relevance scores $q_i$ depend on fixed pre-trained embeddings and are not updated during LoRA adaptation. Suboptimal diversity may occur when downstream tasks are entirely OOD to candidate safety samples.
-   **Independent Observations**: (i) Experiments are limited to 7B/3B scales and LoRA; high-dimensional $g_{safe}$ projection in full-parameter fine-tuning needs more evidence. (ii) Attacks are limited to the BeaverTails family; non-poisoning attacks like prompt-injection or weight-tampering are not covered. (iii) Setting $\eta_{safe} = \eta_{ft}$ lacks theoretical guidance. (iv) Algorithm 1 requires two forward/backward passes (utility + safety) per step, roughly doubling SFT training costs.
-   **Future Work**: Updating embeddings using LoRA-adapted hidden states for online DPP re-selection; extending SPAG to joint projections for multiple constraints (toxicity + bias + privacy); studying adaptive schedulers for $\eta_{safe}$.

## Related Work & Insights
-   **vs SafeInstr (Bianchi et al., 2024)**: They randomly mix 3% safety samples for self-regularization. Ours also uses 3% but through DPP selection and explicit projection. On Qwen-GSM8K, ASR drops from 73.59% to 9.45%, proving "how safety samples are used" matters more than "how many."
-   **vs Lisa (Huang et al., 2024c)**: Lisa uses bi-state optimization with proximal terms to limit state drift, which is still a penalty method. SPARD uses one-step closed-form projection, avoiding $\lambda$ tuning and further halving ASR on LLaMA (24.19% → 12.09%) with 6.2% higher accuracy.
-   **vs SafeGrad (Yi et al., 2025)**: SafeGrad identifies utility-safety gradient conflicts and subtracts harmful components. SPARD is "projection backstep" in parameter space. SPAG significantly outperforms SafeGrad in ASR (9.45 vs 30.18).
-   **vs SafeLoRA / PTST**: These rely on structural constraints or inference-time prompts. SPARD does not restrict LoRA structure or rely on templates, offering broader applicability.

## Rating
-   Novelty: ⭐⭐⭐⭐ Projecting in parameter space is classic in RL safety (CPO) and fairness, but its clean integration into LLM harmful fine-tuning defense with a modified DPP kernel is elegant and novel.
-   Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive coverage across models, tasks, and attacks, though missing 13B+ scales or full-parameter fine-tuning validation.
-   Writing Quality: ⭐⭐⭐⭐ The U-shaped motivation is clear, and the KKT derivation is consistent. A geometric diagram for SPAG vs. Penalty would have been beneficial.
-   Value: ⭐⭐⭐⭐⭐ Tuning-free, closed-form projection that is plug-and-play with LoRA. Reducing ASR from 88% to 9% with 3% data is a practical defense paradigm for fine-tuning-as-a-service platforms.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Safety Anchor: Defending Harmful Fine-tuning via Geometric Bottlenecks](safety_anchor_defending_harmful_fine-tuning_via_geometric_bottlenecks.md)
- [\[ICML 2026\] GIST: Targeted Data Selection for Instruction Tuning via Gradient Subspace Projection](gist_targeted_data_selection_for_instruction_tuning_via_coupled_optimization_geo.md)
- [\[ICLR 2026\] Antibody: Strengthening Defense Against Harmful Fine-Tuning for Large Language Models via Attenuating Harmful Gradient Influence](../../ICLR2026/llm_alignment/antibody_strengthening_defense_against_harmful_fine-tuning_for_large_language_mo.md)
- [\[ICML 2026\] Efficient Preference Poisoning Attack on Offline RLHF](efficient_preference_poisoning_attack_on_offline_rlhf.md)
- [\[AAAI 2026\] Importance-Aware Data Selection for Efficient LLM Instruction Tuning](../../AAAI2026/llm_alignment/importance-aware_data_selection_for_efficient_llm_instruction_tuning.md)

</div>

<!-- RELATED:END -->
