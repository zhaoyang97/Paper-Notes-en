---
title: >-
  [Paper Note] From Fallback to Frontline: When Can LLMs be Superior Annotators of Human Perspectives?
description: >-
  [ACL 2026][LLM/NLP][Perspective-Taking] This paper reformulates "perspective-taking (PT)," a subjective annotation task long considered exclusive to humans…
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "Perspective-Taking"
  - "LLM as Annotator"
  - "Bias-Variance Decomposition"
  - "Subjective Annotation"
  - "Counter-intuitive Findings"
date: 2026-05-08
content_hash: eb657a86ddf6a6cf
---

# From Fallback to Frontline: When Can LLMs be Superior Annotators of Human Perspectives?

**Conference**: ACL 2026  
**arXiv**: [2604.17968](https://arxiv.org/abs/2604.17968)  
**Code**: https://github.com/shasanamin/llm-perspective-taking  
**Area**: LLM Data Annotation / Perspective-Taking / Subjective Judgment / Bias-Variance Analysis  
**Keywords**: Perspective-Taking, LLM as Annotator, Bias-Variance Decomposition, Subjective Annotation, Counter-intuitive Findings

## TL;DR
This paper reformulates "perspective-taking (PT)," a subjective annotation task long considered exclusive to humans, as a "statistical estimation problem of the latent group mean $f^*(x,g)$." Using a bias-variance-correlation decomposition, the authors prove that in low-budget, broad-group, or out-group scenarios, LLMs are not merely cheap substitutes but **superior estimators** compared to in-group human annotators. They also identify a "reasoning paradox" where enabling reasoning capabilities actually degrades performance.

## Background & Motivation

**Background**: Subjective NLP tasks (e.g., toxicity detection, social safety, offensiveness judgment) lack objective ground truth and have long relied on crowd-sourcing to treat the mean of multiple opinions as the "group perspective." Recently, given the strong persona-simulation capabilities of LLMs, more pipelines use LLMs for perspective-taking—asking GPT, "Would a non-binary person find this text offensive?"—but the academic community **defaults to viewing LLMs as a fallback or a cheap compromise**, hesitant to put them on the "frontline."

**Limitations of Prior Work**: The authors identify a **category error** in the mainstream narrative—humans are treated as "sources of real subjective experience," while LLMs are treated as "predictors of population distributions." These two are not evaluated on the same scale: a human provides a single annotation $Y_h(x)$ (an individual subjective judgment), while an LLM outputs $\hat{f}(x,g)$ (an estimate of the group mean). This comparison is unfair.

**Key Challenge**: When the goal is PT, **no one can directly observe the target quantity** $f^*(x,g) = \mathbb{E}_{h\sim P_g}[Y_h(x)]$; both humans and LLMs act as estimators. Statistical theory states that the quality of an estimator depends on bias, variance, and inter-annotator correlation, rather than "who has lived experience."

**Goal**: (1) Formalize PT as an estimation problem for the latent $f^*(x,g)$; (2) Derive a bias-variance-correlation decomposition to identify regimes where LLMs dominate; (3) Systematically validate findings on toxicity and DICES safety datasets; (4) Provide practical guidelines for "engineerable PT."

**Key Insight**: The authors introduce a "Two-Lens" framework—Wide Lens (representation bias $b_{repr}$) reflecting "the annotator's coverage of the group distribution," and Clear Lens (processing bias $b_{proc}$) reflecting "how internal representations translate into numerical judgments." For humans, these two biases are **strongly coupled** due to identity, resulting in a super-additive term $2\mu_{repr}\mu_{proc}$ in the total squared bias. For LLMs, these biases stem from different stages (pre-training vs. post-training vs. inference prompting) and are **mechanically decoupled**, meaning the coupling term can be negative or near zero.

**Core Idea**: By treating LLMs and humans as statistical estimators for the same target and comparing them via $\text{MSE} = \mu_A^2 + \gamma_A V_A + \frac{1-\gamma_A}{k}V_A$, the authors demonstrate that LLMs excel in low-budget regimes where $V_L \ll V_H$.

## Method

### Overall Architecture

This is a framework + empirical paper. While no new algorithms are introduced, the theoretical framework is clear and derivable:

- **Target Quantity**: The latent mean of group $g$ for item $x$: $f^*(x,g) = \mathbb{E}_{h\sim P_g}[Y_h(x)]$.
- **Two Protocols**: (i) Direct annotation—sample $h\sim P_g$ to report $Y_h(x)$; the mean of multiple people is an unbiased estimate of $f^*$. (ii) Perspective-Taking—directly ask the annotator to guess how the group views it, obtaining $\hat{f}(x,g)$.
- **Two-Lens Decomposition**: A single PT prediction is $\hat{f}_A(x,g) = f^*(x,g) + b_{repr,A}(x,g) + b_{proc,A}(x,g) + \varepsilon_A$ for $A\in\{H,L\}$.
- **Aggregate MSE**: The Mean Squared Error of the mean of $k$ annotators $\bar{f}_A^{(k)}$ is $\text{MSE} = \mu_A^2 + \gamma_A V_A + \frac{1-\gamma_A}{k} V_A$. Decision rule: LLM PT is superior to human PT when $\text{MSE}(\bar{f}_L^{(k)}) < \text{MSE}(\bar{f}_H^{(k)})$.
- **Four Falsifiable Hypotheses**: H1 Budget regimes, H2 Coupling hypothesis, H3 Representation limits, and H4 Engineerability hypothesis, each corresponding to a set of experiments.

### Key Designs

1. **Two-Lens Decomposition + Coupling Term**:
    - **Function**: Structures the vague claim of "why humans might be worse than LLMs" into quantifiable bias terms.
    - **Mechanism**: Splits single PT bias into $b_{repr}$ (gap between annotator's implicit sampling and $P_g$) and $b_{proc}$ (how representations convert to numbers); squared bias $\mu_A^2 = \mu_{repr,A}^2 + \mu_{proc,A}^2 + 2\mu_{repr,A}\mu_{proc,A}$. The last term is the key coupling. For humans, out-group PT distorts both (e.g., "I don't know the context for Gen Z, and I use my own norms"), making the coupling term > 0 and super-additively magnifying error. For LLMs, pre-training determines representation while post-training/prompting determines processing; they are mechanically independent, making the coupling term near zero or negative.
    - **Design Motivation**: This is the core theory—identifying an **observable but often overlooked difference**. Previous PT literature discussed "bias magnitude" but ignored "bias correlation," which can contribute more error in out-group regimes than the bias itself.

2. **Budget Regimes + Correlation Floor Decision Criterion**:
    - **Function**: Turns the choice between LLMs and humans into a calculable inequality.
    - **Mechanism**: MSE decomposes into $\mu_A^2 + \gamma_A V_A + \frac{1-\gamma_A}{k}V_A$. When $k$ is small, the third term (reducible variance) dominates, and LLMs win due to $V_L \ll V_H$ (determinism). When $k$ is large, reducible variance $\to 0$, leaving $\mu_A^2 + \gamma_A V_A$ (bias and correlation floor), where LLMs might not win. Combined with H1, a single LLM PT estimate on toxicity data is equivalent to aggregating 3-5 direct human annotations—making **LLMs cheaper and more accurate when ground truth relies on few humans**.
    - **Design Motivation**: Previous evaluations used fixed $k$ (e.g., 5 or 10), masking regime switches. This paper uses bootstrap simulations across $k=1$ to $10$ to visualize winners in different regimes.

3. **Engineerability — Three Levers Targeting Three Error Terms**:
    - **Function**: A "tuning manual" for practitioners to map levers to specific error terms.
    - **Mechanism**: (i) **Model Family/Scale** $\to$ impacts $b_{repr}$ (Wide Lens); (ii) **Prompt Design** (L1 question only $\dots$ L4 + examples) $\to$ impacts $b_{proc}$ (Clear Lens), noting **non-monotonicity** (more structure isn't always better); (iii) **Diversification** (cross-family mixing, temperature) $\to$ impacts correlation floor $\gamma_L V_L$.
    - **Design Motivation**: The authors discovered a **reasoning paradox** where reasoning-enabled modes (e.g., CoT) **degrade PT**. Analysis of reasoning traces showed models shifting from "estimating empirical toxicity rates" to "rule-based classification via rubrics." This **criterion drift** introduces systematic bias, overturning the intuition that "reasoning always helps."

## Key Experimental Results

### Main Results

Toxicity Detection (Duan et al., 2025 + new N=97 non-binary data), $k=1$ single annotator regime, bias/variance/MSE decomposition (for the female subgroup):

| Estimator | Single MSE | Single Bias | Single Variance | Note |
| :--- | :--- | :--- | :--- | :--- |
| Human direct (in-group) | High | $\approx 0$ (Unbiased) | **Highest** $V_H$ | Dominated by intra-group heterogeneity |
| Human PT (in-group) | Med-High | Negative (underestimates) | High | "I thought others wouldn't find this offensive" |
| Human PT (out-group) | Highest | Large + Coupled | High | Super-additive coupling effect |
| Single LLM PT (GPT-like) | **Lowest** | Small or positive | $V_L \ll V_H$ | Wins across all gender subgroups |
| Single LLM PT $\approx$ Direct human $\times$ 3-5 | — | — | — | One LLM equals 3-5 humans |

Impact of group specificity and prevalence on LLM PT error (DICES):

| Subgroup Dimension | LLM PT MSE Trend | Explanation |
| :--- | :--- | :--- |
| More specific (deeper inclusion tree) | Monotonic ↑ | $\|b_{repr,L}\|$ dominates due to sparse evidence |
| Rarer (low prevalence, e.g., Black vs White) | ↑ | Training corpus is stereotype-skewed |
| Broader groups | ↓ | Better coverage by LLM |

### Ablation Study

Impact of interventions on LLM PT (GPT-OSS:120B on female subgroup, $k=1$):

| Intervention | Item Impacted | Effect | Counter-intuitive Finding |
| :--- | :--- | :--- | :--- |
| Model family / scale | $b_{repr}$ (Wide) | MSE fluctuates widely | Mid-size models can beat frontier models |
| Prompt L1 $\to$ L4 | $b_{proc}$ (Clear) | MSE drops, bias may **flip sign** | Non-monotonicity |
| Reasoning enabled | $b_{proc}$ via drift | **MSE increases** | **Reasoning Paradox** |
| Cross-family mixing | $\gamma_L V_L$ | Consistent moderate drop | Mixing same family is ineffective |
| Temperature ↑ | $\gamma_L V_L$ | Limited gain | Inferior to model mixing |

### Key Findings

- **Single LLM > 3-5 Humans**: On toxicity data, a single GPT PT estimate's MSE equals the aggregation of 3-5 direct human annotations. If ground truth is based on $\le 5$ people, **relative to that truth, the LLM is more accurate than adding more people**.
- **Super-additive Coupling in Human Out-group PT**: Females predicting males had larger errors than vice-versa, yet LLMs remained stable across all target groups—directly validating H2.
- **Reasoning Paradox**: Enabling reasoning made PT worse across 4 model pairs. Trace analysis confirmed criterion drift (shifting from "empirical rate" to "rubric-based classification") rather than identity coupling.
- **Specificity & Prevalence Limits**: On DICES, MSE rose monotonically with inclusion tree depth. High-prevalence races (White) had lower error than low-prevalence (Black), marking a "hard boundary" for when to use humans.
- **Differential PT**: In tasks like "predicting male vs non-binary differences," humans significantly outperformed LLMs, showing LLMs lack discernment for adjacent granular subgroups.

## Highlights & Insights

- **Transforming "Cheap" into "Accurate"**: While LLMs are usually chosen for cost, this paper proves that in low-budget, broad-group, or out-group scenarios, **LLMs are inherently superior estimators**. This reframes the entire "LLM-as-annotator" debate.
- **The Coupling Term Insight**: Previous work compared bias magnitude; this paper points out that the **correlation** between biases ($2\mu_{repr}\mu_{proc}$) is an amplifier for humans but near zero for LLMs—a structural advantage independent of model scale.
- **Discovery of the Reasoning Paradox**: Challenges the industry consensus that "reasoning is always better" and provides a specific mechanism (criterion drift). This is a direct warning for deploying reasoning models on subjective tasks.
- **Actionable Engineering Matrix**: Maps three levers to three error terms, turning LLM PT from a black art into actionable engineering.

## Limitations & Future Work

- **Limitations**: (1) Only evaluated on toxicity and DICES (safety); controversial domains like morality or aesthetics might amplify LLM bias. (2) Use of group means as proxies abstracts away intra-group disagreement. (3) The reasoning paradox may change as reasoning models evolve. (4) LLM coverage of emerging identities will always lag.
- **Hidden Risks**: (1) The ground truth itself comes from 50 humans, which is a noisy estimate; the LLM's win might partially be a statistical artifact. (2) The Engineering Matrix ignores the marginal cost of designing complex prompts (e.g., L4).
- **Future Directions**: (1) Extend the framework to distributional targets (KL divergence, Wasserstein distance). (2) Design prompts to actively **reduce the coupling term**. (3) Use differential PT as a fine-tuning objective to fix LLM weaknesses in low-prevalence groups.

## Related Work & Insights

- **vs Frenda et al. 2025 / Duan et al. 2025 (perspectivist NLP)**: While they treat PT as a sociological issue, this paper treats it as a statistical estimation problem, providing precise decision criteria.
- **vs Li et al. / Movva et al. (LLM-as-annotator agreement)**: Shifts from "how much do they agree" to "estimation efficiency," providing a calculable answer for "which to use."
- **vs persona prompting (Sun 2025)**: Explains why persona prompting results are mixed—simulating an individual is structurally harder than estimating a mean.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Redefining PT as statistical estimation + Two-Lens theory is a paradigm-level reconstruction.
- Experimental Thoroughness: ⭐⭐⭐⭐ 2 datasets × 4 families × 4 prompt levels + 1000 bootstraps; statistically rigorous, though domain breadth is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure, well-defined formulas, and memorable terms like "Reasoning Paradox."
- Value: ⭐⭐⭐⭐⭐ Directly impacts decision-making for annotation pipelines; immediately actionable for industry.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Big AI is Accelerating the Metacrisis: What Can We Do?](big_ai_is_accelerating_the_metacrisis_what_can_we_do.md)
- [\[ICLR 2026\] When Stability Fails: Hidden Failure Modes of LLMs in Data-Constrained Scientific Decision-Making](../../ICLR2026/llm_nlp/when_stability_fails_hidden_failure_modes_of_llms_in_data-constrained_scientific.md)
- [\[ACL 2026\] When Gradients Collide: Failure Modes of Multi-Objective Prompt Optimization for LLM Judges](when_gradients_collide_failure_modes_of_multi-objective_prompt_optimization_for_.md)
- [\[ACL 2026\] When TableQA Meets Noise: A Dual Denoising Framework for Complex Questions and Large Tables](when_tableqa_meets_noise_a_dual_denoising_framework_for_complex_questions_and_la.md)
- [\[ACL 2026\] Can AI Be a Good Peer Reviewer? A Survey of Peer Review Process, Evaluation, and the Future](can_ai_be_a_good_peer_reviewer_a_survey_of_peer_review_process_evaluation_and_th.md)

</div>

<!-- RELATED:END -->
