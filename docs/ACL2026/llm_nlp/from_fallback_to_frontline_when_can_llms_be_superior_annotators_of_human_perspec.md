---
title: >-
  [Paper Note] From Fallback to Frontline: When Can LLMs be Superior Annotators of Human Perspectives?
description: >-
  [ACL 2026][LLM (Other)][Perspective-Taking] This paper reformulates "perspective-taking (PT)," a subjective annotation task long considered human-exclusive, as a "statistical estimation problem of the latent group mean $f^*(x,g)$." Using a tripartite decomposition of bias, variance, and correlation, it proves that in low-budget, broad-group, or out-group scenarios, LLMs are not merely cheap substitutes but **superior estimators** compared to in-group human annotators. It furt…
tags:
  - "ACL 2026"
  - "LLM (Other)"
  - "Perspective-Taking"
  - "LLM as Annotator"
  - "Bias-Variance Decomposition"
  - "Subjective Annotation"
  - "Counter-intuitive Findings"
date: 2026-05-08
content_hash: 538fe4ec7d5f3c52
---

# From Fallback to Frontline: When Can LLMs be Superior Annotators of Human Perspectives?

**Conference**: ACL 2026  
**arXiv**: [2604.17968](https://arxiv.org/abs/2604.17968)  
**Code**: https://github.com/shasanamin/llm-perspective-taking  
**Area**: LLM Data Annotation / Perspective-Taking / Subjective Judgment / Bias-Variance Analysis  
**Keywords**: Perspective-Taking, LLM as Annotator, Bias-Variance Decomposition, Subjective Annotation, Counter-intuitive Findings

## TL;DR
This paper reformulates "perspective-taking (PT)," a subjective annotation task long considered human-exclusive, as a "statistical estimation problem of the latent group mean $f^*(x,g)$." Using a tripartite decomposition of bias, variance, and correlation, it proves that in low-budget, broad-group, or out-group scenarios, LLMs are not merely cheap substitutes but **superior estimators** compared to in-group human annotators. It further identifies a "reasoning paradox" where enabling reasoning actually degrades performance.

## Background & Motivation

**Background**: Subjective NLP tasks (toxicity detection, social safety, offensiveness judgment) lack objective ground truth and have long relied on crowdsourcing, treating the mean of multiple opinions as the "group's view." Recently, due to the strong persona simulation capabilities of LLMs, more pipelines use LLMs for perspective-taking—asking GPT, "Would a non-binary person find this text offensive?" However, the academic community **defaults to LLMs as a fallback or a cheap compromise**, with little willingness to place LLMs on the "frontline."

**Limitations of Prior Work**: The authors identify a **category error** in the mainstream narrative: humans are treated as "sources of authentic subjective experience," while LLMs are treated as "predictors of population distributions." These two are evaluated on different scales: humans provide a single annotation $Y_h(x)$ (an individual subjective judgment), while LLMs output $\hat{f}(x,g)$ (an estimate of the group mean). This comparison is inherently unfair.

**Key Challenge**: When the goal is PT, **no one can directly observe the target quantity** $f^*(x,g) = \mathbb{E}_{h\sim P_g}[Y_h(x)]$; both humans and LLMs act as estimators. Statistics teaches that an estimator's quality depends on bias, variance, and inter-annotator correlation, rather than who possesses "lived experience."

**Goal**: (1) Formalize PT as an estimation problem of the latent $f^*(x,g)$; (2) Derive a bias-variance-correlation decomposition to identify regimes where LLMs dominate; (3) Systematically validate this on toxicity and DICES safety datasets; (4) Provide a practical guide for "engineerable PT."

**Key Insight**: The authors introduce a "Two-Lens" framework: the Wide Lens (representation bias $b_{repr}$) reflects "how well the annotator covers the population distribution," and the Clear Lens (processing bias $b_{proc}$) reflects "how internal representations are converted into numerical judgments." For humans, these two biases are **strongly coupled** due to identity, leading to a super-additive term $2\mu_{repr}\mu_{proc}$ in the squared total bias. In LLMs, these biases stem from different training phases (pre-training vs. post-training vs. inference prompting) and are **mechanically decoupled**, meaning the coupling term can be negative or near zero.

**Core Idea**: Treat LLMs and humans as statistical estimators for the same target and compare them directly using $\text{MSE} = \mu_A^2 + \gamma_A V_A + \frac{1-\gamma_A}{k}V_A$. This demonstrates LLM superiority in low-budget regimes where $V_L \ll V_H$.

## Method

### Overall Architecture

The paper does not propose a new algorithm but reformulates the intuition-driven debate of "who is better at PT" into a derivable and falsifiable statistical estimation framework. It starts by acknowledging that the target quantity is never directly observable: the true perception of group $g$ for item $x$ is the latent mean $f^*(x,g)=\mathbb{E}_{h\sim P_g}[Y_h(x)]$. Both human and LLM PT predictions $\hat{f}_A(x,g)$ ($A\in\{H,L\}$) are merely biased estimators of this mean. Thus, the evaluation criterion shifts from "who has lived experience" to "whose estimator has lower MSE." The logic follows: decompose single PT error into bias and variance $\rightarrow$ further decompose bias into representation and processing biases $\rightarrow$ compare MSE after aggregating $k$ annotators to identify a decision boundary based on budget $k$, leading to four falsifiable hypotheses (H1 Budget Regime, H2 Coupling, H3 Representation Limits, H4 Engineerability).

### Key Designs

**1. Two-Lens Bias Decomposition and Coupling: Quantifying Why Human Performance Varies**

Prior PT literature suggests humans are biased without questioning the source or correlation of those biases. This paper defines single PT prediction as $\hat{f}_A(x,g)=f^*(x,g)+b_{repr,A}+b_{proc,A}+\varepsilon_A$, where $b_{repr}$ (Wide Lens) captures the gap between the annotator's implicit sampling distribution and the true $P_g$, and $b_{proc}$ (Clear Lens) captures how the representation is translated into a score. The key lies in the squared total bias expansion $\mu_A^2=\mu_{repr,A}^2+\mu_{proc,A}^2+2\mu_{repr,A}\mu_{proc,A}$, where the cross-term represents coupling.

This coupling term is the structural watershed between humans and LLMs. When humans perform out-group PT, their identity distortions affect both lenses—"I don't understand the Gen Z context, and I use my own norms to score"—resulting in positively correlated $b_{repr}$ and $b_{proc}$, which super-additively amplifies error. Conversely, LLM representation is determined by pre-training, while processing is determined by post-training and prompting; these are mechanically decoupled across training stages, making the coupling term near zero or even negative. LLM advantage stems not from scale, but from the near-elimination of this term.

**2. Budget Regime and Correlation Floor: Calculating the "When to Use Whom" Inequality**

The MSE of the mean of $k$ annotators $\bar{f}_A^{(k)}$ is expanded into three terms: $\text{MSE}=\mu_A^2+\gamma_A V_A+\frac{1-\gamma_A}{k}V_A$. The decision rule is that LLM PT wins when $\text{MSE}(\bar{f}_L^{(k)})<\text{MSE}(\bar{f}_H^{(k)})$. At small $k$, the third term (reducible variance) dominates; LLMs excel due to near-determinism ($V_L\ll V_H$). As $k$ increases, this term vanishes, leaving only the bias $\mu_A^2$ and the correlation floor $\gamma_A V_A$, where LLMs no longer guarantee a win.

This decomposition reveals that winners switch with the budget—a fact often obscured by fixed $k=5$ or $k=10$ evaluations. Using bootstrap simulations across $k=1$ to $10$, the authors visualize this switch, concluding that in low-budget toxicity data regimes, a single LLM PT estimate is equivalent to aggregating 3-5 direct human annotations.

**3. Engineerability: Three Levers for Three Error Terms**

The framework's practical value lies in transforming PT from a "black art" into a "tuning matrix." Changing model families or scales primarily affects $b_{repr}$ (Wide Lens). Progressively adding structure to prompts (L1 Question $\rightarrow$ L2 Definition $\rightarrow$ L3 Rubric $\rightarrow$ L4 Examples) targets $b_{proc}$ (Clear Lens), though the effect is non-monotonic as different prompts re-weight the "belief-to-number" mapping. Diversification (cross-family mixing, increasing temperature) reduces the correlation floor $\gamma_L V_L$, but only cross-family mixing proved effective.

This matrix reveals the "reasoning paradox": enabling reasoning (e.g., GPT-o1 style) worsens PT. Analysis shows "criterion drift"—the model shifts from estimating "empirical group toxicity rates" to performing "rule-based classification against a rubric," injecting systematic bias and overturning the consensus that "reasoning always helps."

### Evaluation Protocol
The paper uses zero-shot evaluation without training. The statistical protocol employs bootstrap resampling ($B=1000$) to simulate different $k$-annotator regimes, reporting MSE, bias, and variance. Ground truth is the mean of $\ge 50$ direct annotations per subgroup. Models include GPT (including o1-style), Qwen, Gemma, and DeepSeek, ranging from 1B to frontier scales.

## Key Experimental Results

### Main Results

Toxicity Detection (Duan et al., 2025 + new N=97 non-binary data), $k=1$ single annotator regime, tripartite decomposition (for female subgroup):

| Estimator | Single MSE | Single Bias | Single Variance | Notes |
| :--- | :--- | :--- | :--- | :--- |
| Human direct (in-group) | High | ≈ 0 (Unbiased) | **Highest** $V_H$ | Dominated by intra-group heterogeneity |
| Human PT (in-group) | Med-High | Negative (Underest.) | High | "I thought others wouldn't find it offensive" |
| Human PT (out-group) | Highest | Large + Coupled | High | Super-additive coupling observed |
| Single LLM PT (GPT family) | **Lowest** | Small or Positive | $V_L \ll V_H$ | Wins across all gender subgroups |
| Single LLM PT ≈ Direct × 3-5 | — | — | — | One LLM equals 3-5 humans |

Effect of group specificity and prevalence on LLM PT error (DICES):

| Subgroup Dimension | LLM PT MSE Trend | Explanation |
| :--- | :--- | :--- |
| More specific (deeper tree) | Monotonic ↑ | $\|b_{repr,L}\|$ dominates, sparse evidence |
| Rarer (low prevalence) | ↑ | Stereotype-skewed training corpora |
| Broader groups | ↓ | Better LLM coverage |

### Ablation Study

Impact of interventions on LLM PT (GPT-o1:120B on female subgroup, $k=1$):

| Intervention Type | Primary Error Term | Effect | Counter-intuitive Finding |
| :--- | :--- | :--- | :--- |
| Model family / scale | $b_{repr}$ (Wide) | Large MSE swings | Medium models can beat frontier models |
| Prompt L1 $\rightarrow$ L4 | $b_{proc}$ (Clear) | Significant MSE drop | Non-monotonic; bias can flip signs |
| Reasoning enabled | $b_{proc}$ drift | **MSE increases** | **Reasoning Paradox** |
| Cross-family mixing | $\gamma_L V_L$ | Consistent moderate drop | Mixing same-family models is ineffective |
| Increased Temp | $\gamma_L V_L$ | Limited gain | Inferior to model mixing |

### Key Findings

- **Single LLM > 3-5 Humans**: For toxicity data, a single GPT PT estimate matches the MSE of 3-5 aggregated direct human annotations. If ground truth relies on $\le 5$ annotators, **using an LLM is more accurate than adding humans**.
- **Super-additive Coupling in Out-group Humans**: Women predicting men have higher error than men predicting women, yet LLM performance remains stable across targets—verifying the H2 coupling hypothesis.
- **Reasoning Paradox**: Reasoning consistently pushes bias further from ground truth across 4 base/reasoning pairs. Trace analysis identifies "criterion drift" (shifting from empirical rate estimation to formal classification).
- **Specificity/Rarity Limit**: MSE rises monotonically with deeper inclusion trees in DICES. Lower prevalence races (e.g., Black) show higher errors than high prevalence (e.g., White), defining the "boundary for human intervention."
- **Differential PT**: Humans significantly outperform LLMs in tasks requiring "discrimination between adjacent groups" (e.g., male vs. non-binary), indicating LLMs lack fine-grained subgroup discernment.

## Highlights & Insights

- **Converting "Cheap" to "Accurate"**: While LLMs are often used for cost-saving, this paper proves they are **intrinsically superior estimators** in low-budget, broad-group, and out-group scenarios.
- **Coupling as a Structural Advantage**: Rather than just comparing bias magnitude, the paper highlights that **bias correlation** ($2\mu_{repr}\mu_{proc}$) is an amplifier for humans but near-zero for LLMs due to decentralized training stages.
- **Discovery of the Reasoning Paradox**: Contradicts the "reasoning is always better" consensus and pinpoints "criterion drift" as the mechanism, serving as a warning for deploying reasoning models on subjective tasks.
- **The "Engineering Matrix"**: Transforms LLM PT from a black art into actionable engineering: use prompts to fix bias and cross-family mixing to fix variance.

## Limitations & Future Work

- **Scope**: Evaluation is limited to toxicity and DICES safety; more controversial domains (morality, policy, aesthetics) may amplify LLM bias.
- **Aggregation**: Using the group mean as a proxy abstracts away intra-group disagreement.
- **Reasoning Paradox**: Findings might change as reasoning models evolve and their alignment improves.
- **Coverage**: LLM representation of emerging identities will always lag.
- **Hidden Risks**: The ground truth ($N=50$) is itself a noisy estimate; LLM superiority might partially be a statistical artifact of two high-variance estimators converging toward a low-variance anchor.

## Related Work & Insights

- **vs. Frenda et al. 2025 (Perspectivist NLP)**: While they treat PT as a sociological issue, this paper treats it as a statistical estimation problem, providing more precise decision rules.
- **vs. Li et al. (LLM-as-annotators)**: Shifts from "how much LLMs agree with humans" to "estimation efficiency," providing quantitative answers on when to swap humans for LLMs.
- **vs. Persona Prompting (Sun 2025)**: Argues that simulating individuals is harder than estimating means, providing a structural explanation for why persona prompting yields mixed results.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Redefining PT as statistical estimation + Two-Lens coupling theory is a paradigm-level reconstruction.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Solid statistics (1000x bootstrap); however, limited to two domains.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear structure, refined formulas, and catchy terminology like "Reasoning Paradox."
- **Value**: ⭐⭐⭐⭐⭐ Directly impacts cost/quality decisions for NLP annotation pipelines; immediately applicable in industry.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Can LLMs Ground when they (Don't) Know: A Study on Direct and Loaded Political Questions](../../ACL2025/llm_nlp/can_llms_ground_when_they_dont_know_a_study_on_direct_and_loaded_political_quest.md)
- [\[ACL 2026\] When Gradients Collide: Failure Modes of Multi-Objective Prompt Optimization for LLM Judges](when_gradients_collide_failure_modes_of_multi-objective_prompt_optimization_for_.md)
- [\[ACL 2026\] When TableQA Meets Noise: A Dual Denoising Framework for Complex Questions and Large Tables](when_tableqa_meets_noise_a_dual_denoising_framework_for_complex_questions_and_la.md)
- [\[ACL 2025\] Can Language Models Reason about Individualistic Human Values and Preferences?](../../ACL2025/llm_nlp/can_language_models_reason_about_individualistic_human_values_and_preferences.md)
- [\[ACL 2026\] Big AI is Accelerating the Metacrisis: What Can We Do?](big_ai_is_accelerating_the_metacrisis_what_can_we_do.md)

</div>

<!-- RELATED:END -->
