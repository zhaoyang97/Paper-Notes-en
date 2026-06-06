---
title: >-
  [Paper Note] Reverse-Engineering Model Editing on Language Models
description: >-
  [ICML 2026][Knowledge Editing][locate-then-edit] The paper reveals that the parameter update matrices of locate-then-edit knowledge editing methods (ROME/MEMIT/AlphaEdit) leak "fingerprints" of edited subjects through th…
tags:
  - "ICML 2026"
  - "Knowledge Editing"
  - "locate-then-edit"
  - "reverse-engineering attacks"
  - "subspace reconstruction"
  - "entropy reduction"
  - "subspace camouflage defense"
date: 2026-05-08
content_hash: 55c6d85d69e976a0
---

# Reverse-Engineering Model Editing on Language Models

**Conference**: ICML 2026  
**arXiv**: [2602.10134](https://arxiv.org/abs/2602.10134)  
**Code**: https://github.com/reanatom/EditingAttack  
**Area**: Knowledge Editing / LLM Security  
**Keywords**: locate-then-edit, reverse-engineering attacks, subspace reconstruction, entropy reduction, subspace camouflage defense

## TL;DR
The paper reveals that the parameter update matrices of locate-then-edit knowledge editing methods (ROME/MEMIT/AlphaEdit) leak "fingerprints" of edited subjects through their row spaces. It proposes a two-stage attack KSTER (recovering subjects via SVD and prompts via entropy difference) and provides a Subspace Camouflage defense based on "semantic decoy" injection.

## Background & Motivation

**Background**: LLMs inevitably memorize massive amounts of sensitive information (personal privacy, copyrighted snippets) during pre-training. Since re-training is extremely costly, *model editing* has become the mainstream mitigation solution. Specifically, the **locate-then-edit** paradigm (ROME/MEMIT/AlphaEdit) is widely used as an "after-the-fact deletion/modification" tool for sensitive knowledge due to its interpretability, zero inference overhead, and ability to localize FFN parameters, often serving as a foundation for privacy protection.

**Limitations of Prior Work**: Previous research focused on the *efficacy* and *generalization* of editing (e.g., accuracy after editing, whether other knowledge is preserved). However, few have systematically asked: does the editing action itself become a *leaking side channel*? If an attacker obtains both the pre- and post-edit weights $\theta$ and $\theta'$, can the erased content be reverse-engineered from the weight difference $\Delta\theta$?

**Key Challenge**: The core analytical solutions of locate-then-edit (rank-1 in ROME, low-rank least squares in MEMIT) naturally compress the "key vector of the edited subject" as a low-dimensional structure into $\Delta\mathbf{W}$. **The mechanism intended to protect privacy** becomes a high-fidelity signature of the edited information—the more precise the erasure, the purer the signature.

**Goal**: (1) Formally prove that $\Delta\mathbf{W}$ encodes recoverable "fingerprints" of edited subjects; (2) Design practical attacks to recover *subject + prompt template + original answer*; (3) Propose a defense strategy that does not compromise editing utility.

**Key Insight**: The authors observe that the "hidden states of the subject's last token at the edited layer" in FFNs possess strong **subject invariance**—the activation for the same subject is almost identical across different prompt templates (cos sim near 1), and attention focuses on the subject token. This means the $\mathbf{K}$ matrix encodes "subject identity" almost exclusively and decouples from prompt context, allowing attackers to lock onto a subject without guessing the prompt first.

**Core Idea**: Use the Woodbury identity to rewrite the MEMIT update as $\Delta\mathbf{W} = \mathbf{R}(\mathbf{I}+\mathbf{K}^\top\mathbf{C}^{-1}\mathbf{K})^{-1}\mathbf{K}^\top\mathbf{C}^{-1}$, thereby proving $\mathrm{RowSpace}(\Delta\mathbf{W}\mathbf{C}) \subseteq \mathrm{ColSpace}(\mathbf{K})$. Performing SVD on $\Delta\mathbf{W}\mathbf{C}$ directly yields the "subject key subspace." Projecting candidate subject activations onto this subspace identifies the edited subject via the largest projection ratio. The second stage uses the overfitting phenomenon—where the post-edit model has nearly zero entropy for the true prompt—to recover the prompt through relative entropy reduction ranking.

## Method

### Overall Architecture
The threat model includes white-box and grey-box settings. White-box attackers possess $\theta$, $\theta'$, the editing algorithm, covariance $\mathbf{C}$, and a "Candidate Subject Set $\mathcal{S}_{\rm cand}$ × Candidate Prompt Set $\mathcal{R}_{\rm cand}$" (constructible from public domain knowledge). The attack pipeline consists of two stages: **Stage I — Subject Inference** reverse-engineers the subject using the algebraic structure of $\Delta\mathbf{W}$; **Stage II — Prompt Recovery** selects top-$N_r$ prompts for each recovered subject using the entropy difference between pre- and post-edit models. Combining these and performing a forward pass on the pre-edit model yields the original answer $o^{\rm ori}$. The defense, **Subspace Camouflage**, injects "semantic decoy" subjects during editing, crowding the row space of weight updates into a confused subspace where SVD sees an overlap of decoys and true subjects.

### Key Designs

1. **SVD-based Subject Inference (Stage I)**:

    - **Function**: Accurately picks $N$ edited subjects from a candidate pool given only $\Delta\mathbf{W}$ and $\mathbf{C}$.
    - **Mechanism**: First, the number of edits $N$ is estimated via $\mathrm{rank}(\Delta\mathbf{W})$ (strictly holds under full column rank assumptions for $\mathbf{R}$ and $\mathbf{K}$). Then, SVD is performed on $\mathbf{M}=\Delta\mathbf{W}\mathbf{C}$, taking the first $N$ columns of right singular vectors $\mathbf{V}_N$ as the reconstructed key subspace. For each candidate subject $s_i^c$, activation $\mathbf{k}_i^c=\mathcal{F}_\theta(s_i^c,\mathcal{T}_{\rm gen})$ is extracted using a generic template. Candidates are ranked by projection ratio $\rho_i^c = \|\mathbf{V}_N^\top \mathbf{k}_i^c\|_2 / \|\mathbf{k}_i^c\|_2$. For AlphaEdit, scoring is adjusted based on the null-space projection $\mathbf{P}$ (Lemma G.18).
    - **Design Motivation**: Direct recovery of $\mathbf{K}$ is impossible (as $\mathbf{R}$ is unknown), but the *subspace* is sufficient. Activations from generic templates are extremely similar to those from true editing templates (subject invariance), so the projection ratio scalar distinguishes edited subjects. Multiplying by $\mathbf{C}$ removes geometric distortion of the original row space; robustness to $\mathbf{C}$ estimation noise is proved in Appendix G.16.

2. **Entropy-reduction based Prompt Recovery (Stage II)**:

    - **Function**: Picks the template most likely used for editing given the recovered subject $\hat{s}_i$.
    - **Mechanism**: Editing essentially shapes the next-token distribution under the target prompt toward a one-hot distribution. Thus, *relative entropy reduction* is significantly larger on the true prompt. For each $(\hat{s}_i, r_j^c)$, Shannon entropy $H(\hat{s}_i,r_j^c;\theta)$ and $H(\hat{s}_i,r_j^c;\theta')$ are calculated. The score is defined as $\mathrm{Score}=\frac{H(\hat{s}_i,r_j^c;\theta)-H(\hat{s}_i,r_j^c;\theta')}{H(\hat{s}_i,r_j^c;\theta')+\epsilon}$, taking top-$N_r$. The denominator amplifies overfitted prompts pushed toward near-zero entropy.
    - **Design Motivation**: Comparing logit cosine similarity suffers from interference in large batches. Entropy difference is sensitive only to the direction of "information collapse," aligning with the editing optimization goal, making it stable even at $N=100$.

3. **Subspace Camouflage Defense**:

    - **Function**: Allows editors to replace the row space of $\Delta\mathbf{W}$ from $\mathrm{ColSpace}(\mathbf{K})$ to a decoy-mixed subspace $\mathrm{ColSpace}(\tilde{\mathbf{K}})$ without losing editing efficacy.
    - **Mechanism**: Decoy key matrices $\mathbf{K}_{\rm decoy}$ are sampled from irrelevant real subjects to construct a camouflaged key $\tilde{\mathbf{K}} = \mathbf{K} + \alpha \cdot \frac{\|\mathbf{K}\|_2}{\|\mathbf{K}_{\rm decoy}\|_2}\mathbf{K}_{\rm decoy}$. A unique defense update $\Delta\mathbf{W}_{\rm defense} = \Delta\mathbf{W}\mathbf{K}(\tilde{\mathbf{K}}^\top \mathbf{C}^{-1}\mathbf{K})^{-1}\tilde{\mathbf{K}}^\top \mathbf{C}^{-1}$ is solved to satisfy the row space constraint and original behavior ($\Delta\mathbf{W}_{\rm defense} \mathbf{K} = \Delta\mathbf{W}\mathbf{K}$).
    - **Design Motivation**: Unlike random noise, **semantic decoys** correspond to real activations and "actively compete" to pull SVD directions toward decoys. Appendices H.3/H.4 prove that attackers cannot determine if the defense is active, nor can they reverse-engineer $\mathbf{K}$ from $\tilde{\mathbf{K}}$ without $\mathbf{R}$, which would require the protected knowledge itself, creating a circular dependency.

### Loss & Training
No training involved. The attack uses geometry and information theory; the defense is a one-time equivalent rewrite of the closed-form editing algorithm with a scalar hyperparameter $\alpha$ (controlling protection strength, $\alpha\in[1,5]$ in experiments).

## Key Experimental Results

Models: GPT-J (6B), Llama3-8B-Instruct, Qwen2.5-7B-Instruct; Methods: ROME / MEMIT / AlphaEdit; Datasets: CounterFact, zsRE; Batch size $N\in\{10,50,100\}$.

### Main Results: Subject Inference top-N Recall (CounterFact)

| Model | $N$ | MEMIT (KSTER White-box) | MEMIT Grey-box Baseline | AlphaEdit (KSTER White-box) | AlphaEdit Grey-box Baseline |
|-------|-----|-------------------------|-------------------------|-----------------------------|-----------------------------|
| GPT-J | 100 | **0.95** | 0.88 | **0.96** | 0.86 |
| Llama3-8B-Instruct | 100 | **0.99** | 0.68 | **0.99** | 0.45 |
| Qwen2.5-7B-Instruct | 100 | **0.94** | 0.59 | **0.95** | 0.51 |

Prompt recovery (Llama3-8B-Instruct, CounterFact, $N=100$): top-1 0.51 / top-5 0.81 / top-20 0.94, semantic similarity 0.88; end-to-end original answer recovery reached 0.74 at top-20.

### Ablation Study: Camouflage Intensity $\alpha$ (MEMIT, Llama3-8B-Instruct, CounterFact)

| $\alpha$ | Avg. True Subject Rank (Higher is Safer) | Efficacy | Generalization | Fluency |
|----------|------------------------------------------|----------|----------------|---------|
| 0 (No Def) | 50.83 | 0.95 | 0.52 | 6.33 |
| 1 | 148.62 | 0.98 | 0.49 | 6.33 |
| 3 | 206.47 | 0.96 | 0.52 | 6.34 |
| **5** | **394.12** | 0.96 | 0.53 | 6.32 |
| 7 | 634.39 | 0.91 | 0.42 | 6.26 |

### Key Findings
- **Grey-box vs. White-box gap widens with $N$**: At $N=10$, grey-box performance is comparable. At $N=100$, grey-box drops to 0.45–0.68 while white-box remains stable at 0.94–0.99, indicating $\Delta\mathbf{W}$ carries far more separable information than logit differences.
- **Covariance estimation is robust**: While $\mathbf{C}$ is default estimated from 100k Wikipedia entries, 100 samples suffice for convergence. AlphaEdit remains stable even with 10 samples, signifying the attack doesn't require knowing the training distribution.
- **Failure modes are bimodal**: Rank 6–100 results from *semantic generalization error* (subject misjudged as a broader category). Rank 700–1000 involves *optimization constraint conflicts* where post-edit entropy increases, failing the entropy-difference criterion.
- **Defense sweet spot at $\alpha=5$**: True subject rank increases to 394 with negligible efficacy loss. However, at $\alpha=7$, AlphaEdit efficacy drops to 0.46 as small eigenvalues of $\mathbf{P}$ make matrix inversion ill-conditioned.

## Highlights & Insights
- **Elegant characterization of "security as a side channel"**: Using the Woodbury identity to rewrite the MEMIT solution into a subspace recovery problem transforms a weight diff problem into a clean SVD-plus-projection task.
- **Independent utility of subject invariance**: The discovery that FFN activations for subjects are prompt-invariant is valuable for understanding LLM memory and designing future editing/interpretation methods beyond attacks.
- **Entropy reduction exceeds logit difference**: By treating overfitting as a signal and amplifying near-zero entropy, the authors leverage the editing algorithm's own optimization goal as a discriminant.
- **Clever circular dependency argument for defense**: Camouflage is robust because reverse-engineering the true $\mathbf{K}$ requires $\mathbf{R}$, but constructing $\mathbf{R}$ requires the original prompt and answer—the very goals of the attack.

## Limitations & Future Work
- Dependency on "white-box + candidate pool containing truth": The ability to filter the pool determines attack cost; complexity in open-set scenarios is unquantified.
- Coverage limited to single-layer batch editing (max $N=100$): Sequential editing and larger scales are only explored in the appendix.
- Defense is specific to white-box locate-then-edit; no versions provided for memory-based (SERAC, GRACE) or meta-learning (MEND, KE) editors that don't modify FFN parameters.
- Decoy subjects may cause *out-of-target hallucinations*: At $\alpha\ge 3$, decoy subject TFR drops and fluency declines, potentially eroding knowledge space over time.

## Related Work & Insights
- **vs. Youssef et al. (2025)**: They only recover pre-edit *behavior*, not prompts/answers. KSTER recovers all three without prompt assumptions.
- **vs. Patil et al. (2024)**: They probe answers via intermediate logits but require known prompts; this work breaks that strong assumption and is stable for batch edits.
- **vs. Membership Inference**: Traditional privacy attacks sample blindly from unedited models with low recall. This work proves "edited" models are paradoxically more vulnerable to precise targeted attacks.
- **Insights**: The logic generalizes to any "low-rank update + analytical solution" (LoRA, prefix tuning diffs, unlearning). Any update falling in a low-dimensional subspace of "modified sample activations" faces similar subspace recovery risks.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to systematically show locate-then-edit acts as a side channel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three LLMs × three algorithms × two datasets + robust covariance + camouflage sweep; lacks multi-layer and large-scale sequential evaluation.
- Writing Quality: ⭐⭐⭐⭐⭐ Logical progression from theory to defense; complete proofs in the appendix.
- Value: ⭐⭐⭐⭐⭐ Provides an "audit tool" and a warning for the knowledge editing community, likely impacting future privacy-preserving designs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] The Labyrinth and the Thread: Rethinking Regularizations in Sequential Knowledge Editing for Large Language Models](the_labyrinth_and_the_thread_rethinking_regularizations_in_sequential_knowledge_.md)
- [\[ICML 2026\] Revisiting Parameter-Based Knowledge Editing in Large Language Models: Theoretical Limits and Empirical Evidence](revisiting_parameter-based_knowledge_editing_in_large_language_models_theoretica.md)
- [\[AAAI 2026\] Multiplicative Orthogonal Sequential Editing for Language Models (MOSE)](../../AAAI2026/knowledge_editing/multiplicative_orthogonal_sequential_editing_for_language_models.md)
- [\[ACL 2026\] The Model Agreed, But Didn't Learn: Diagnosing Surface Compliance in Large Language Models](../../ACL2026/knowledge_editing/the_model_agreed_but_didn39t_learn_diagnosing_surface_compliance_in_large_langua.md)
- [\[ACL 2026\] Aligning Language Models with Real-time Knowledge Editing](../../ACL2026/knowledge_editing/aligning_language_models_with_real-time_knowledge_editing.md)

</div>

<!-- RELATED:END -->
