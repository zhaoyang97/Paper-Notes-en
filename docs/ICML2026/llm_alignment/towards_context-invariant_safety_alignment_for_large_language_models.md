---
title: >-
  [Paper Note] Towards Context-Invariant Safety Alignment for Large Language Models
description: >-
  [ICML 2026][Alignment & RLHF][GRPO] Academic paper note for Towards Context-Invariant Safety Alignment for Large Language Models.
tags:
  - ICML 2026
  - Alignment & RLHF
  - GRPO
date: 2026-05-08
content_hash: 35a2e6a9cc23563d
---
# Towards Context-Invariant Safety Alignment for Large Language Models

**Conference**: ICML 2026  
**arXiv**: [2605.20994](https://arxiv.org/abs/2605.20994)  
**Code**: Undisclosed  
**Area**: Alignment RLHF  
**Keywords**: Context Invariance, Safety Alignment, GRPO, Invariant Risk Minimization, Reward Hacking

## TLDR
The authors propose AIR (Anchor Invariance Regularization), which treats verifiable prompts as "anchors" and uses stop-gradients to pull open-ended variants toward the anchor's performance. Integrated as an auxiliary loss in GRPO, it improves out-of-distribution (OOD) group-level consistency by an average of 33.49% and in-distribution (ID) by 12.71% across safety, morality, and mathematics domains.

## Background & Motivation
**Background**: Preference-based post-training (RLHF, DPO, GRPO, etc.) has become the standard paradigm for LLM alignment. RLHF encodes human preferences into the policy via reward models, while GRPO, which omits the value network through group-relative advantage, is the current de facto choice for reasoning training.

**Limitations of Prior Work**: Safety alignment is "brittle." Changing the jailbreak packaging of the same harmful intent can cause a model to refuse a standard prompt but comply immediately with the paraphrased version. This "surface-level vulnerability" indicates that the model learns superficial cues rather than underlying intent—a direct consequence of reward hacking and alignment faking.

**Key Challenge**: While "Invariant Risk Minimization" (IRM / V-REx) from domain generalization aims to make behavior intent-dependent rather than surface-dependent, **supervision quality in safety alignment is asymmetric**. Verifiable prompts (multiple-choice, rule-based) have ground truth, whereas open-ended generation relies on noisy, hackable LLM judges. Symmetric variance penalties (V-REx) merely flatten the gap between contexts; they **can pull the poor performance up, but also drag the good performance down**. The authors formally prove that when the anchor risk $R_a$ is significantly lower than the open-ended risk $R_o$, symmetric penalties generate a "downgrading anchor" descent direction, destroying reliable capabilities to align with noisy proxies.

**Goal**: Design an asymmetric invariance regularizer to "freeze and preserve" anchor capabilities, directing all alignment forces solely onto the open-ended variants.

**Key Insight**: Observing that "at least one form of reliable supervision (multiple-choice/rule-based) exists in safety alignment," this can be treated as a "privileged environment" in IRM and converted into a unidirectional anchor using stop-gradients.

**Core Idea**: Replace the symmetric variance term of V-REx with $\Omega_{\text{AIR}} = \sum_{c \neq c_{\text{acr}}} (R_c - \text{sg}[R_{c_{\text{acr}}}])^2$, and formulate it as a policy-gradient auxiliary loss that can be plugged into GRPO/GSPO.

## Method

### Overall Architecture
The input consists of a latent intent $z$ (e.g., a safety constraint or math problem), expressed through a rendering function $g(z, c)$ into prompts under different contexts $c$: some are **anchors** (multiple-choice, True/False, rule-verifiable), and others are **open variants** (jailbreak wrappers, open generation). During training, the data loader constructs a **meta-group** $\mathcal{S}_z = \mathcal{A}_z \cup \mathcal{O}_z$ for each $z$. For each prompt $s$ in the meta-group, $K$ completions are sampled via GRPO to obtain the prompt-level mean $\bar r_s$ and variance $\sigma_s$. Synchronously, **under the same parameters $\theta$**, the anchor reward $\bar r_{\text{acr}} = \frac{1}{|\mathcal{A}_z|}\sum_{s \in \mathcal{A}_z}\bar r_s$ and open-variant rewards $\bar r_c$ are calculated. Their difference serves as the "asymmetric coefficient" for the policy gradient.

### Key Designs

1.  **Anchor Invariance Regularization (Asymmetric Invariance Regularizer)**:
    - **Function**: Uses stop-gradient to lock anchor risk, pulling open-variant risk toward the anchor level.
    - **Mechanism**: Replaces the symmetric variance term $\text{Var}_c[R_c(\theta)]$ in V-REx with $\Omega_{\text{AIR}} = \sum_{c \in \mathcal{C} \setminus \{c_{\text{acr}}\}} (R_c(\theta) - \text{sg}[R_{c_{\text{acr}}}(\theta)])^2$. Since $\nabla_\theta \text{sg}[R_{c_{\text{acr}}}] = 0$, the gradient of the regularizer **structurally excludes $\nabla_\theta R_{\text{acr}}$**, preventing the gap from closing by "downgrading the anchor." It handles two cases automatically: when open risk is higher than the anchor ($R_c > \tau_{\text{acr}}$), the positive coefficient reinforces samples closer to anchor behavior; when open risk is artificially low due to reward hacking ($R_c < \tau_{\text{acr}}$), the negative coefficient suppresses that direction.
    - **Design Motivation**: The authors formally prove in Appendix A.3 that when $R_o > R_a$, symmetric V-REx has a "downgrading anchor descent direction" for $\lambda > -1/\Delta$, which destroys reliable capabilities. AIR removes this direction from the regularizer gradient via Lemma A.4 and Corollary A.5.

2.  **Policy-gradient Auxiliary Loss (Plug-and-play with GRPO)**:
    - **Function**: Converts $\Omega_{\text{AIR}}$ into a differentiable surrogate loss for any group-based optimizer.
    - **Mechanism**: Applying the log-derivative trick to $R_c(\theta)$ yields $\nabla_\theta \Omega_{\text{AIR},c} = -\mathbb{E}_y[2(R_c-\tau_{\text{acr}}) \cdot r(s,y) \cdot \nabla_\theta \log \pi_\theta(y|s)]$. This corresponds to the surrogate $\mathcal{J}_{\text{aux}} = -\frac{1}{N}\sum_i (R_c - \tau_{\text{acr}}) \cdot r_i \cdot \log \pi_\theta(y_i|s_i)$. The final objective is $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{policy}} + \lambda \mathcal{J}_{\text{aux}}$. The coefficient $(R_c - \tau_{\text{acr}})$ acts as a dynamic weight: positive for reinforcement and negative for penalty.
    - **Design Motivation**: Avoids backpropagation through $R_c$ itself (which is a sampled expectation), making AIR compatible with existing GRPO/GSPO stacks without extra value networks or manual annotation.

3.  **Heterogeneous Group Construction (Heterogeneous Meta-group Sampling)**:
    - **Function**: Ensures anchors and open variants are estimated under the **same parameter state** to accurately calculate AIR coefficients.
    - **Mechanism**: The data loader organizes batches by latent $z$, including $m$ anchor prompts and $n$ open variants. $K$-rollouts from GRPO provide $\bar r_s$ and $\sigma_s$. The GRPO relative advantage $\hat A_{s,k} = (r_{s,k} - \bar r_s)/(\sigma_s + \epsilon)$ is calculated normally, while the AIR term approximates $R_c - \tau_{\text{acr}}$ using $\bar r_{\text{acr}} - \bar r_c$ within the meta-group. Anchor rewards are synchronized across workers in distributed training.
    - **Design Motivation**: If anchors and open variants are calculated at different steps or with different $\theta$, AIR coefficients will be contaminated by variance from asynchronous updates. Synchronous meta-group sampling minimizes this variance.

### Loss & Training
The total objective follows Algorithm 1: GRPO's clipped surrogate plus $\lambda \Delta_s r_{s,k} \log \pi_\theta(y_{s,k}|s)$ (active only for open prompts $s \in \mathcal{O}_z$), where $\Delta_s = \bar r_{\text{acr}} - \bar r_s$ is detached. The backbone is Qwen-2.5-14B, trained for 3000 steps across three domains with $K=3$, lr $5\times 10^{-7}$, and $\lambda = 8\times 10^{-4}$. The composite reward is $r = r_{\text{task}} + r_{\text{fmt}}$, with format rewards requiring `<think>…</think><answer>…</answer>` (and `\boxed{}` for Math). Task rewards use rule-based verification for anchors and LLM-as-a-judge for open variants (Safety calculates log-odds over 10 facets; Morality uses YES/NO token log-probs; Math uses `math_verify`).

## Key Experimental Results

### Main Results
Prompt-level accuracy (Acc) and group-level accuracy ($\text{Acc}_{\text{group}}$) are reported across three domains. $\text{Acc}_{\text{group}}$ requires **all variants within a meta-group** to be correct, quantifying "context invariance." Open-variant scores are re-evaluated using GPT-4.1.

| Setting | Configuration | Safety Acc / Group | Moral Acc / Group | Math Acc / Group |
| :--- | :--- | :--- | :--- | :--- |
| ID | GRPO | 96.92% / 71.15% | 75.39% / 34.31% | 93.81% / 64.60% |
| ID | GRPO + V-REx | 82.15% / 35.40% | 58.20% / 7.15% | 93.02% / 60.71% |
| ID | **GRPO + AIR** | **98.46% / 84.62%** | **85.51% / 59.85%** | 93.64% / 63.72% |
| ID | GSPO + AIR | **99.81% / 98.08%** | 84.57% / 65.69% | **94.93% / 68.14%** |
| OOD | GRPO | 73.04% / 13.73% | 62.68% / 14.71% | 82.30% / 40.71% |
| OOD | GRPO + V-REx | 62.25% / 8.82% | 53.12% / 3.68% | 83.19% / 42.48% |
| OOD | **GRPO + AIR** | **88.24% / 60.78%** | **80.70% / 47.79%** | 88.49% / 61.06% |
| OOD | GSPO + AIR | **93.14% / 63.73%** | 81.07% / 49.26% | 86.28% / 51.33% |

On average, GRPO+AIR improved OOD $\text{Acc}_{\text{group}}$ from 23.05% to 56.54% (+33.49pp) and OOD Acc from 72.67% to 85.81% (+13.14pp).

### Ablation Study
| Configuration | Key Phenomenon | Explanation |
| :--- | :--- | :--- |
| GRPO (No invariance) | Low OOD group consistency | Overfitting to surface cues; vulnerable to jailbreaks. |
| GRPO + V-REx (Symmetric) | ID Safety/Moral group Acc dropped sharply | Anchors were dragged down to match noisy variants (Sec 3.2 failure mode). |
| GRPO + V-REx in Math | Slight drop (64.60 → 60.71) | Math is almost entirely verifiable; low supervision asymmetry means weaker V-REx side effects. |
| GRPO + AIR | Anchors maintained; open variants improved | Asymmetric stop-gradient removed the "downgrading" direction. |
| Stress Test (Extreme Hacking) | Open side rewarded for "I am sorry" only | GRPO collapsed on Oracle judge; AIR maintained safety scores. |

### Key Findings
- **Formalized Failure Modes**: Appendix A.3 proves that symmetric V-REx possesses a downgrading anchor descent direction when $R_o > R_a$, explaining why symmetry causes collapse at the gradient geometry level.
- **Stronger Asymmetry, Greater AIR Gains**: Benefits were highest in Safety/Moral domains where supervision is least reliable. In Math, where the reliability gap is small, AIR performed close to the baseline.
- **$\lambda$ Sweet Spot**: Peak performance for both Avg Acc and Avg $\text{Acc}_{\text{group}}$ occurred around $\lambda \approx 8\times 10^{-4}$ to $10^{-3}$; excessively large values suppress task-specific signals.
- **Latent Geometry Compaction**: The intra-group representation dispersion decreased from 86.47 in GRPO to 71.54 with AIR, indicating the model maps different surfaces of the same intent to more consistent internal representations.

## Highlights & Insights
- **Stop-gradient as a "Privileged Environment" Switch**: AIR provides an engineering solution to the IRM identification problem by detaching verifiable contexts as references, structurally preventing regularizer degradation.
- **Auxiliary Loss over New Optimizer**: AIR is implemented as a simple $\lambda \mathcal{J}_{\text{aux}}$ term, requiring no changes to GRPO/GSPO logic. This makes it transferable to other RL backbones like DPO or SimPO.
- **Group-level Accuracy Value**: Using "all-correct-per-intent" as a metric is harder to game with surface tricks than prompt-level Acc; it should be standard for future safety/robustness benchmarks.
- **Jailbreak Explanation via "Supervision Geometry"**: Rather than attributing jailbreaks solely to data or model capacity, the paper provides a new perspective: symmetric regularizers under asymmetric supervision structurally lead to "anchor downgrading."

## Limitations & Future Work
- **Reliable Anchor Dependency**: Constructing verifiable anchors for open-ended tasks (e.g., creative writing, long-horizon agents) remains difficult. The "privileged" status of stop-gradients fails if the anchor is noisy.
- **Lack of Human Preference Experiments**: Rewards were rule-based or LLM-as-judge. Stability under real-world RLHF (human pairwise labels) remains an open question.
- **Model Scale**: Experiments were limited to 14B models. Scaling behavior for $\lambda$ and whether it requires scheduling for larger models is unknown.
- **Meta-group Construction Overhead**: Preparing verifiable and open versions for every $z$ is data-intensive; automated anchor generation is a potential future direction.
- **Formal Analysis Scope**: Theorem A.3 proves degradation for $|\mathcal{C}|=2$; existence of these directions in many-context scenarios was not explicitly proven.

## Related Work & Insights
- **vs V-REx / IRM (Krueger 2021, Arjovsky 2019)**: While both seek invariance, AIR accounts for asymmetric reliability by detaching an anchor, preventing the symmetric collapse shown to be harmful in safety alignment.
- **vs Rule-based Rewards (Mu 2024)**: Unlike methods using rules as direct rewards, AIR uses rules as **invariance reference points** to "infect" untrustworthy contexts with reliable signals.
- **vs Constrained RLHF (Moskovitz 2023)**: Instead of external constraints or heavy KL penalties to suppress overoptimization, AIR adaptively guides open generation using the anchor's actual performance.
- **vs Weak-to-Strong (Burns 2023)**: AIR accomplishes the same goal (guiding strong capabilities with reliable supervision) during the RL phase rather than SFT, utilizing an asymmetric gradient mechanism.

## Rating
- Novelty: ⭐⭐⭐⭐ Using stop-gradients to modify V-REx is a simple but elegant change, backed by formal proofs of failure modes and effective GRPO integration.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers three domains, two group optimizers, and ID/OOD settings. Includes reward-hacking stress tests and latent visualization, though lacks human preference data.
- Writing Quality: ⭐⭐⭐⭐ Logical flow from motivation to formalization to solution is seamless; Appendix A is particularly clear.
- Value: ⭐⭐⭐⭐ Provides a practical, plug-and-play RL-side solution for brittle safety alignment that is highly extensible.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] GuardAlign: Test-time Safety Alignment in Multimodal Large Language Models](../../ICLR2026/llm_alignment/guardalign_test-time_safety_alignment_in_multimodal_large_language_models.md)
- [\[ICML 2026\] PICACO: Pluralistic In-Context Value Alignment of LLMs via Total Correlation Optimization](picaco_pluralistic_in-context_value_alignment_of_llms_via_total_correlation_opti.md)
- [\[ICML 2026\] Curriculum Learning for Safety Alignment](curriculum_learning_for_safety_alignment.md)
- [\[ICML 2026\] Implicit Safety Alignment from Crowd Preferences](implicit_safety_alignment_from_crowd_preferences.md)
- [\[AAAI 2026\] EASE: Practical and Efficient Safety Alignment for Small Language Models](../../AAAI2026/llm_alignment/ease_practical_and_efficient_safety_alignment_for_small_language_models.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICML 2026\] Curriculum Learning for Safety Alignment](curriculum_learning_for_safety_alignment.md)
- [\[ICLR 2026\] GuardAlign: Test-time Safety Alignment in Multimodal Large Language Models](../../ICLR2026/llm_alignment/guardalign_test-time_safety_alignment_in_multimodal_large_language_models.md)
- [\[AAAI 2026\] EASE: Practical and Efficient Safety Alignment for Small Language Models](../../AAAI2026/llm_alignment/ease_practical_and_efficient_safety_alignment_for_small_language_models.md)
- [\[ICML 2026\] PICACO: Pluralistic In-Context Value Alignment of LLMs via Total Correlation Optimization](picaco_pluralistic_in-context_value_alignment_of_llms_via_total_correlation_opti.md)
- [\[ICML 2026\] Implicit Safety Alignment from Crowd Preferences](implicit_safety_alignment_from_crowd_preferences.md)

</div>

<!-- RELATED:END -->
