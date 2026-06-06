---
title: >-
  [Paper Note] On the Generalization Gap in Self-Evolving Language Model Reasoning
description: >-
  [ICML 2026][LLM Reasoning][Closed-loop self-evolution] This paper systematically compares four self-evolution (SE) strategies (SimpleSE, RevisionSE, IterativeSE…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "Closed-loop self-evolution"
  - "DPO"
  - "Generator-Verifier game"
  - "Knights & Knaves"
  - "Reasoning generalization"
date: 2026-05-08
content_hash: 6b9320f074397c15
---

# On the Generalization Gap in Self-Evolving Language Model Reasoning

**Conference**: ICML 2026  
**arXiv**: [2606.01075](https://arxiv.org/abs/2606.01075)  
**Code**: None  
**Area**: LLM Reasoning / Self-Evolution / Preference Learning  
**Keywords**: Closed-loop self-evolution, DPO, Generator-Verifier game, Knights & Knaves, Reasoning generalization

## TL;DR
This paper systematically compares four self-evolution (SE) strategies (SimpleSE, RevisionSE, IterativeSE, CurriculumSE) against oracle supervision under the strict closed-loop setting of "unlabeled prompts + base model." On Knights & Knaves logical reasoning, SE improves the performance of Gemma 3 4B from 31.0% to 44.8%, yet a persistent 8–13% gap remains compared to the oracle's 53.3%. Only RevisionSE with a 12B model successfully approaches the oracle (52.8% vs. 53.6%).

## Background & Motivation

**Background**: LLM post-training is shifting from SFT/RLHF/RLVR paradigms, which rely on human labeling or verifiable rewards, toward "Self-Evolution" (SE). This paradigm allows models to improve using self-generated supervision signals, such as self-verification, generative feedback, or internal confidence rewards (e.g., INTUITOR, Absolute Zero, R-Zero, EMPO).

**Limitations of Prior Work**: Existing SE studies often use idiosyncratic settings and reporting methods, making it difficult to determine how closely SE can approach oracle supervision under strictly closed-loop constraints. Simultaneously, other research warns of model collapse, the generator–verifier gap, and theoretical barriers to learning from synthetic data. Conflicting conclusions exist without a horizontal comparison under a unified framework.

**Key Challenge**: SE requires the "model's internal verification ability $\geq$ the supervision quality needed for training." However, because the verifier is the generator itself, verification errors contaminate preference pairs, limiting the gain from DPO. The fundamental question is: **how accurate is the internal verifier, and can it truly replace ground-truth?**

**Goal**: To systematically characterize the generalization gap between SE and oracle supervision under strict closed-loop constraints (given only an unlabeled prompt set $\mathcal{D}$ and base model $\mathcal{M}$, where all reasoning traces/rewards/feedback/preference labels must be self-produced by $\mathcal{M}$) and analyze the factors determining this gap (model scale, task verifiability, training compute, and curriculum order).

**Key Insight**: All closed-loop SE methods can be abstractly unified into a "Generator–Verifier Game" $\mathsf{GV}(\mathcal{M},\mathcal{D},T)$. The differences lie only in "how signals are extracted, reused, and structured." Knights & Knaves (KK) is chosen as the primary testing platform due to its deterministic verifiability and parameterized difficulty (2–8 people), providing a clean environment for studying easy-to-hard generalization.

**Core Idea**: Gradually approach the oracle using the unified GV framework combined with increasingly complex SE variants (SimpleSE $\rightarrow$ RevisionSE $\rightarrow$ IterativeSE $\rightarrow$ CurriculumSE). This quantifies whether additional compute or structure can eventually close the gap between closed-loop SE and oracle supervision.

## Method

### Overall Architecture
All methods execute the same Generator–Verifier Game $\mathsf{GV}(\mathcal{M},\mathcal{D},T)\rightarrow\mathcal{P}$. The same base model $\mathcal{M}$ is instantiated as a generator $\mathcal{G}$ and a verifier $\mathcal{V}$ using different system prompts. For each prompt $q$, $\mathcal{G}$ samples $k$ candidates $\{\hat{y}_1,\dots,\hat{y}_k\}$ and $\mathcal{V}$ provides binary judgments. A preference pair $(y_w,y_l)\in\mathcal{P}$ is formed only if $\mathcal{V}(q,y_w)=\texttt{Correct}$ and $\mathcal{V}(q,y_l)=\texttt{Incorrect}$. The strategy $\pi_\theta$ is then fine-tuned on $\mathcal{P}$ using DPO.

Input: Unlabeled prompt set + instruction-tuned base (Gemma-3-it 1B/4B/12B, Qwen-2.5-Instruct 7B/14B). Output: DPO fine-tuned policy $\pi_\theta$. The variations lie in "how $\mathcal{P}$ is constructed."

### Key Designs

1. **SimpleSE + Threshold Majority Voting Denoising**:
    - **Function**: Upgrades single-turn self-verification to "high-confidence preference pair mining" to combat verifier noise.
    - **Mechanism**: For each candidate $\hat{y}$, the verifier runs $n$ times to obtain an empirical accuracy $\hat{p}(q,\hat{y})=\frac{1}{n}\sum_j \mathbf{1}\{\mathcal{V}^{(j)}=\texttt{Correct}\}$. If $\hat{p}\geq\tau$, it is labeled Positive; if $1-\hat{p}\geq\tau$, it is labeled Negative; otherwise, it is discarded. Training is conducted with the DPO loss: $\mathcal{L}_{\text{DPO}}=-\mathbb{E}[\log\sigma(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}-\beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)})]$. $\tau=0.5$ degrades to standard majority voting, while $\tau=0.7$ provides the best precision/recall balance for 4B models.
    - **Design Motivation**: Single-turn verification is noisy; using it directly for labels contaminates the preference set. Threshold voting discards ambiguous samples, effectively raising the verifier's effective accuracy to a level the training can accommodate.

2. **RevisionSE (Multi-turn Feedback Revision)**:
    - **Function**: Enables the verifier to provide textual feedback rather than just labels, allowing the generator to revise and turning the feedback into a supervision signal.
    - **Mechanism**: Across $T>1$ rounds, the next candidate is satisfying $\hat{y}^{(t+1)}\sim\mathcal{G}(\cdot\mid q, f(\mathcal{V}(q,\hat{y}^{(t)})))$, where $f$ maps judgments into textual feedback. If $\mathcal{V}(q,\hat{y}^{(t)})=\texttt{Incorrect}$ and $\mathcal{V}(q,\hat{y}^{(t+1)})=\texttt{Correct}$, the pair is added to $\mathcal{P}$ as $(y_l, y_w)$.
    - **Design Motivation**: Single-turn verification wastes the model's ability to explain "why it is wrong." RevisionSE extracts interpretable intermediate feedback, amplifying the verifier's implicit discriminative power into structured training data. This is the only setting that approaches the oracle (52.8% vs. 53.6% on 12B), but it requires a sufficiently large model; it underperforms SimpleSE on 1B models as they often revise correct answers into incorrect ones.

3. **IterativeSE / CurriculumSE (Data Order and Rounds)**:
    - **Function**: Extends SE to multiple rounds and replaces random mixing with difficulty scheduling.
    - **Mechanism**: Iterative versions start with $\mathcal{M}_0=\mathcal{M}$. Each round generates $\mathcal{P}_t=\mathsf{GV}(\mathcal{M}_{t-1},\mathcal{D}_t,T)$ and sets $\mathcal{M}_t=\texttt{Finetune}(\mathcal{M}_{t-1},\mathcal{P}_t)$. Curriculum versions split $\mathcal{D}$ by difficulty (KK number of people).
    - **Design Motivation**: It provides a positive feedback loop: "better model $\rightarrow$ more accurate verification $\rightarrow$ cleaner data." Curriculum learning reduces early verifier noise by starting with simple cases. Both schedulers consistently outperform random mixing but still leave a ~5% gap compared to the oracle.

### Loss & Training
DPO is used throughout (reference policy is fixed to the base model), with $\beta>0$ controlling preference alignment sharpness. Evaluation utilizes exact-match accuracy, sampling once at temperature 0.7, averaged over four random seeds. Compute analysis shows that increasing verifier passes ($n_2$) is more cost-effective than increasing generator candidates ($n_1$).

## Key Experimental Results

### Main Results: Gap between 4 SE Variants and Oracle on KK (Gemma 3 4B)

| Method | 2–3 ppl | 4–5 ppl | 6–8 ppl | All | vs Oracle |
|------|---------|---------|---------|-----|-----------|
| Baseline (gemma-3-4b-it) | 62.0 | 31.0 | 10.3 | 31.0 | −22.3 |
| SimpleSE ($\tau=0.6$) | 70.9 | 45.4 | 17.5 | 40.7 | −12.6 |
| RevisionSE | 75.8 | 46.4 | 17.1 | 42.2 | −11.1 |
| Iterative SimpleSE ×3 | 75.2 | 49.6 | 19.7 | 44.1 | −9.2 |
| Curriculum SimpleSE (KK23→KK45) | 76.2 | 49.7 | 20.6 | 44.8 | −8.5 |
| **Oracle Verifier (KK23→KK45)** | **80.8** | **60.9** | **29.8** | **53.3** | — |

### Ablation Study: RevisionSE vs. Oracle Gap Narrowing with Model Scale

| Model | Baseline | Best SimpleSE | RevisionSE | Oracle | RevisionSE vs. Oracle Gap |
|------|----------|---------------|------------|--------|--------------------------|
| Gemma 3 1B | 7.8 | 8.4 ($\tau=0.8$) | 7.8 | 12.5 | −4.7 (Negative for small model) |
| Gemma 3 4B | 31.0 | 40.7 | 42.2 | 46.6 | −4.4 |
| Gemma 3 12B | 47.5 | 51.1 | **52.8** | 53.6 | **−0.8 (≈ Closed)** |

### Key Findings
- **Persistent Gap = 8–13%**: Except for 12B RevisionSE, all SE variants on 4B leave a significant gap compared to the oracle. Additional iterations yield diminishing returns; the only way to significantly push the limit is adding a final oracle round (44.1 $\rightarrow$ 53.2 on 4B).
- **Ability Threshold**: 1B models show almost no gain or even regress, whereas 4B models show stable positive gains. Self-evolution requires a base accuracy of $\geq 30\%$ to bootstrap effectively.
- **Task Verifiability Determines the Upper Bound**: When moving SimpleSE to open reasoning tasks (GSM8K/MATH500), the gain on 4B shrinks significantly compared to KK (+10% on KK vs +1.6% on MATH500). Without deterministic answers, the internal verifier struggles to distinguish "plausible but incorrect" solutions.
- **Compute Allocation**: Grid searches show that increasing verifier passes is more "bang for the buck" than increasing generator candidates. $\tau=0.7$ is the "sweet spot" for precision/recall.
- **Pass@1 ↑ but Pass@32 ≈ Const**: This supports the "sharpening hypothesis"—closed-loop SE merely amplifies high-probability correct paths the model already knows rather than teaching new heuristics.

## Highlights & Insights
- **Unified SE Axis**: The progression from SimpleSE $\rightarrow$ RevisionSE $\rightarrow$ IterativeSE $\rightarrow$ CurriculumSE maps increasingly "signal richness × computational cost." Each step clearly reduces the oracle gap, providing a clear reproduction path.
- **RevisionSE Closing the Gap at 12B**: This empirical result is surprising; it suggests that as models scale, their ability to "grade their own homework" becomes sufficient to match oracle supervision. This shifts the SE research focus from "new algorithms" to "how verifier capability scales."
- **Transferable Tricks**: Threshold majority voting + DPO is a robust denoising module for any model-as-a-verifier pipeline. Adding a sparse "oracle round" is a cost-effective way to maximize performance under a budget.

## Limitations & Future Work
- Evaluation is primarily tied to Knights & Knaves, with open reasoning as a secondary check. Only offline DPO is used, ignoring online RL. Only Gemma/Qwen instruction models were tested; base model behavior remains unknown.
- **Observations**: (1) The "closed-loop" definition is narrow—disabling code executors or external tools defines the "unclosable gap" by construction. (2) The 4B baseline was just at the threshold for bootstrapping; weaker models might see negative returns. (3) KK tasks are too "closed," and the performance of internal verifiers on "process-based rewards" remains unexplored.
- **Future Directions**: Quantitatively monitoring Pass@32 vs. sharpening; framing oracle round insertion as a scheduled learning problem; investigating scaling laws for verifier capability vs. model scale.

## Related Work & Insights
- **Vs. Absolute Zero (AZR)**: AZR uses online self-play with code executors for rewards. SimpleSE (offline, no external environment) outperforms AZR/AZR-Coder on several benchmarks, suggesting that starting from instruction-tuned models with offline DPO is more stable than online RL from base models.
- **Vs. INTUITOR**: INTUITOR utilizes self-certainty as an online RL reward. SimpleSE is slightly superior on most benchmarks and provides a more general conclusion regarding the oracle gap limits.
- **Vs. Song et al. (2024)**: This paper quantifies the theoretical "generation–verification gap" into concrete numbers (8–13%) and identifies the internal verifier's accuracy as the key determinant.

## Rating
- **Novelty**: ⭐⭐⭐⭐ No new algorithm, but the systematic quantification of the "closed-loop SE gap" is a significant framework contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ High coverage across 4 SE methods, 3 scales, multiple thresholds, and 5 reasoning benchmarks.
- **Writing Quality**: ⭐⭐⭐⭐ Clear framework and high information density in tables.
- **Value**: ⭐⭐⭐⭐⭐ Provides a realistic "floor/ceiling" reference for the self-evolution subfield, preventing future work from chasing "phantom gaps."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Incorporating Self-Rewriting into Large Language Model Reasoning Reinforcement](../../AAAI2026/llm_reasoning/incorporating_self-rewriting_into_large_language_model_reasoning_reinforcement.md)
- [\[ICML 2026\] SmartThinker: Progressive Chain-of-Thought Length Calibration for Efficient Large Language Model Reasoning](smartthinker_progressive_chain-of-thought_length_calibration_for_efficient_large.md)
- [\[ICML 2026\] Prism: Efficient Test-Time Scaling via Hierarchical Search and Self-Verification for Discrete Diffusion Language Models](prism_efficient_test-time_scaling_via_hierarchical_search_and_self-verification_.md)
- [\[ICLR 2026\] Why is Your Language Model a Poor Implicit Reward Model?](../../ICLR2026/llm_reasoning/why_is_your_language_model_a_poor_implicit_reward_model.md)
- [\[ICML 2026\] GRPO is Secretly a Process Reward Model](grpo_is_secretly_a_process_reward_model.md)

</div>

<!-- RELATED:END -->
