---
title: >-
  [Paper Note] From Fallback to Frontline: When Can LLMs be Superior Annotators of Human Perspectives?
description: >-
  [ACL 2026][LLM (Other)][Perspective-Taking] This paper reformulates "perspective-taking (PT)", traditionally viewed as a human-exclusive subjective annotation task, as a "statistical estimation problem for latent group means $f^*(x,g)$". Using a bias-variance-correlation decomposition, it proves that in low-budget / broad-group / out-group scenarios, LLMs are no
tags:
  - ACL 2026
  - LLM (Other)
  - Perspective-Taking
  - LLM as Annotator
  - Bias-Variance Decomposition
date: 2026-05-08
content_hash: 40392249b77ddfb9
---
# From Fallback to Frontline: When Can LLMs be Superior Annotators of Human Perspectives?

**Conference**: ACL 2026  
**arXiv**: [2604.17968](https://arxiv.org/abs/2604.17968)  
**Code**: https://github.com/shasanamin/llm-perspective-taking  
**Area**: LLM Data Annotation / Perspective-Taking / Subjective Judgment / Bias-Variance Analysis  
**Keywords**: Perspective-Taking, LLM as Annotator, Bias-Variance Decomposition, Subjective Annotation, Counter-intuitive findings

## TL;DR
This paper reformulates "perspective-taking (PT)", traditionally viewed as a human-exclusive subjective annotation task, as a "statistical estimation problem for latent group means $f^*(x,g)$". Using a bias-variance-correlation decomposition, it proves that in low-budget / broad-group / out-group scenarios, LLMs are not just cheap substitutes but **superior estimators compared to in-group human annotators**. It further identifies a "reasoning paradox" where enabling reasoning actually degrades PT performance.

## Background & Motivation

**Background**: Subjective NLP tasks (toxicity detection, social safety, offensiveness) lack objective ground truth and have long relied on crowdsourcing, treating the mean of multiple opinions as the "group view." Recently, due to LLMs' strong persona simulation abilities, more pipelines use LLMs for perspective-taking—asking GPT "Would a non-binary person find this offensive?"—but the academic community **defaults to viewing LLMs as a fallback or cheap compromise**, with few considering them for the "frontline."

**Limitations of Prior Work**: The authors identify a **category error** in the mainstream narrative: humans are treated as "sources of real subjective experience," while LLMs are treated as "predictors of population distributions." These two are evaluated on different scales: a human provides a single annotation $Y_h(x)$ (individual judgment), while an LLM outputs $\hat{f}(x,g)$ (an estimate of the group mean). This comparison is inherently unfair.

**Key Challenge**: When the goal is PT, **no one can directly observe the target quantity** $f^*(x,g) = \mathbb{E}_{h\sim P_g}[Y_h(x)]$; both humans and LLMs act as estimators. Statistical theory suggests that estimator quality depends on bias, variance, and inter-annotator correlation, rather than "who possesses lived experience."

**Goal**: (1) Formalize PT as an estimation problem for the latent $f^*(x,g)$; (2) Derive the bias-variance-correlation decomposition to identify regimes where LLMs dominate; (3) Systematically validate on toxicity and DICES safety datasets; (4) Provide a practical guide for "engineerable PT."

**Key Insight**: The authors introduce a "two-lens" framework: the Wide Lens (representation bias $b_{repr}$) reflects "annotator coverage of group distribution," and the Clear Lens (processing bias $b_{proc}$) reflects "how internal representations translate into numerical judgments." For humans, due to identity identification, these two biases are **strongly coupled**, leading to a super-additive term $2\mu_{repr}\mu_{proc}$ in the total squared bias. In LLMs, these biases originate from different training stages (pretraining vs. post-training vs. inference prompting), making them **mechanically decoupled**, where the coupling term can be negative or near zero.

**Core Idea**: Treat LLMs and humans as statistical estimators of the same target. By comparing MSE = $\mu_A^2 + \gamma_A V_A + \frac{1-\gamma_A}{k}V_A$, the authors prove that LLMs dominate in low-budget regimes where $V_L \ll V_H$.

## Method

### Overall Architecture

The paper does not propose a new algorithm but reformulates the debate over human vs. LLM suitability for PT into a derivable and falsifiable statistical framework. It acknowledges that the target quantity—the true view of group $g$ on item $x$—is the latent mean $f^*(x,g)=\mathbb{E}_{h\sim P_g}[Y_h(x)]$. Any PT prediction $\hat{f}_A(x,g)$ ($A\in\{H,L\}$) is merely a biased estimator. The evaluation criterion shifts from "lived experience" to "minimizing MSE." The logic follows: decompose single PT error into bias and variance → split bias into representation and processing components → aggregate $k$ annotators to compare MSE, deriving a decision rule based on budget $k$.

### Key Designs

**1. Two-Lens Bias Decomposition and Coupling: Quantifying Human Limitations**

Prior literature vaguely cited "human bias" without exploring its structure. This work formalizes single PT prediction as $\hat{f}_A(x,g)=f^*(x,g)+b_{repr,A}+b_{proc,A}+\varepsilon_A$, where $b_{repr}$ captures the gap between an annotator's implicit sampling and the true $P_g$, and $b_{proc}$ captures the translation of representations into scores. Total squared bias expands to $\mu_A^2=\mu_{repr,A}^2+\mu_{proc,A}^2+2\mu_{repr,A}\mu_{proc,A}$, where the cross-term represents coupling.

For humans performing out-group PT, identity simultaneously distorts both lenses—an annotator may lack context for a group (e.g., Gen Z) and apply their own norms for grading—leading to positive coupling that super-additively amplifies error. For LLMs, representation (pretraining) and processing (post-training/prompting) are decoupled, resulting in a coupling term near zero.

**2. Budget Regime and Correlation Floor: Decision Rule as a Calculable Inequality**

The error of the mean of $k$ annotators $\bar{f}_A^{(k)}$ is expanded into $\text{MSE}=\mu_A^2+\gamma_A V_A+\frac{1-\gamma_A}{k}V_A$. LLM PT wins when $\text{MSE}(\bar{f}_L^{(k)})<\text{MSE}(\bar{f}_H^{(k)})$. At small $k$, the third term (reducible variance) dominates; LLMs excel due to near-determinism ($V_L \ll V_H$). As $k$ increases, this term vanishes, leaving bias $\mu_A^2$ and the correlation floor $\gamma_A V_A$, where LLM superiority is no longer guaranteed.

**3. Engineerable PT: Targeting Error Terms with Specific Levers**

The framework maps interventions to specific error components. Changing model families or scales affects $b_{repr}$ (Wide Lens). Applying structured prompts (L1 Question only → L2 add Definitions → L3 add Scale → L4 add Examples) targets $b_{proc}$ (Clear Lens). Diversification (cross-family mixing, temperature) reduces the correlation floor $\gamma_L V_L$. Notably, the "Reasoning Paradox" is identified: enabling reasoning (e.g., GPT-OSS) induces "criterion drift," where the model shifts from empirical estimation to rule-based classification, injecting systematic bias.

### Evaluation Protocol
The study utilizes zero-shot evaluation without fine-tuning. A bootstrap resampling protocol ($B=1000$) simulates different $k$-annotator regimes, reporting MSE, bias, and variance. Ground truth is derived from the mean of $\ge 50$ in-group annotators. Models tested include GPT families (including GPT-OSS:120B), Qwen, Gemma, and DeepSeek, ranging from 1B to frontier scales.

## Key Experimental Results

### Main Results

Toxicity Detection (Duan et al., 2025 + new N=97 non-binary data), $k=1$ single annotator regime, decomposition for female subgroup:

| Estimator | Single MSE | Single Bias | Single Variance | Notes |
| :--- | :--- | :--- | :--- | :--- |
| Human direct (in-group) | High | ≈ 0 (unbiased) | **Highest** $V_H$ | Driven by intra-group heterogeneity |
| Human PT (in-group) | Mid-High | Negative (underestimates) | High | "I thought others wouldn't find this offensive" |
| Human PT (out-group) | Highest | Large + Strong Coupling | High | Super-additive coupling observed |
| Single LLM PT (GPT-class) | **Lowest** | Small / Positive Bias | $V_L \ll V_H$ | Dominates across all gender subgroups |
| Single LLM PT ≈ Direct human × 3-5 | — | — | — | 1 LLM equals 3-5 humans |

Group specificity and prevalence impact on LLM PT error (DICES):

| Subgroup Dimension | LLM PT MSE Trend | Explanation |
| :--- | :--- | :--- |
| More specific (deeper tree) | Monotonic ↑ | $\|b_{repr,L}\|$ dominates, sparse evidence |
| Rare groups (low prevalence) | ↑ | Training data stereotype-skewed |
| Broad groups | ↓ | Better LLM coverage |

### Ablation Study

Impact of interventions on LLM PT (GPT-OSS:120B on female subgroup, $k=1$):

| Intervention Type | Primary Error Term | Effect | Counter-intuitive Finding |
| :--- | :--- | :--- | :--- |
| Model family / scale | $b_{repr}$ (Wide) | Large MSE swings | Medium models can beat frontier ones |
| Prompt L1 → L4 | $b_{proc}$ (Clear) | MSE drops, bias may **flip sign** | Non-monotonic |
| Reasoning enabled | $b_{proc}$ via drift | **MSE increases** | **Reasoning Paradox** |
| Cross-family mixing | $\gamma_L V_L$ (correlation) | Consistent modest drop | Intra-family mixing is ineffective |
| Temperature increase | $\gamma_L V_L$ | Limited gain | Inferior to model mixing |

### Key Findings

- **Single LLM > 3-5 Humans**: On toxicity data, a single GPT PT estimate matches the MSE of 3-5 aggregated human annotations. When ground truth is derived from $\le 5$ annotators, **using an LLM is more accurate than adding humans**.
- **Out-group Human PT Coupling**: Humans show significantly higher error when female annotators predict males compared to vice-versa, whereas LLMs remain stable—directly verifying the H2 coupling hypothesis.
- **Reasoning Paradox**: Enabling reasoning consistently pushes bias further from ground truth across 4 model pairs. Trace analysis confirms this is "criterion drift" (shifting from empirical rate estimation to rubric-based classification).
- **Specificity/Rarity Limits**: In DICES, MSE increases with the depth of the inclusion tree and is higher for low-prevalence races (e.g., Black vs. White), defining the "boundary" where humans must be used.
- **Differential PT**: In tasks like predicting the *difference* between male and non-binary views, humans still outperform most LLMs, showing LLMs lack fine-grained discriminatory power between similar subgroups.

## Highlights & Insights

- **Converting "Cheap" to "Accurate"**: While LLMs are usually chosen for cost-saving, this study proves that in low-budget, broad-group, or out-group scenarios, LLMs are **statistically superior estimators**.
- **Coupling as a Structural Divider**: Beyond bias magnitude, the **correlation** between biases distinguishes humans and LLMs. LLMs lack the identity-driven super-additive coupling that plagues human out-group predictions.
- **Discovery of Reasoning Paradox**: Challenging the "reasoning is always better" consensus, the paper provides a mechanism (criterion drift) with direct implications for deploying reasoning models.
- **Engineering Matrix**: Maps 3 types of levers to 3 error terms, transforming LLM PT from a "black art" to "actionable engineering."
- **Scalability to Distributions**: The framework acts as a foundation for pluralistic alignment beyond mere mean estimation.

## Limitations & Future Work

- **Ours**: (1) Evaluation limited to toxicity/DICES; moral or aesthetic domains may magnify bias. (2) Reliance on group means abstracts away intra-group disagreement. (3) Reasoning paradox may shift as alignment techniques improve. (4) Coverage of emerging identities remains a lag for LLMs.
- **Hidden issues**: (1) Ground truth from 50 annotators is itself a noisy estimate; LLM superiority might partially be a statistical artifact of low-variance estimators converging to a common anchor. (2) Reasoning paradox tested on limited model pairs. (3) Prompt engineering costs (e.g., L4 prompts needing experts) are often overlooked.
- **Future Work**: (1) Extend to distributional targets (KL divergence, Wasserstein). (2) Design prompts that actively **decouple** representation and processing. (3) Use differential PT as a fine-tuning objective. (4) Impose anti-drift constraints on reasoning models to maintain empirical estimation modes.

## Related Work & Insights

- **vs. Frenda et al. 2025 / Duan et al. 2025**: Moves from sociological perspectivist NLP to a statistical estimation framework.
- **vs. Li et al. / Movva et al.**: Shifts focus from "agreement" to "estimation efficiency," providing rules for when to deploy LLMs.
- **vs. Sorensen et al. / Feng et al.**: Demonstrates that accurate mean estimation is the prerequisite for pluralistic distribution matching.
- **vs. Persona prompting**: Explains why individual simulation is structurally harder than group mean estimation, providing a theoretical basis for mixed results in persona research.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Redefining PT as statistical estimation + Two-lens coupling theory is a paradigm shift.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive bootstrap and model coverage; domain variety could be broader.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure and catchy terminology like "Reasoning Paradox."
- Value: ⭐⭐⭐⭐⭐ Highly actionable for industrial annotation pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Can LLMs Ground when they (Don't) Know: A Study on Direct and Loaded Political Questions](../../ACL2025/llm_nlp/can_llms_ground_when_they_dont_know_a_study_on_direct_and_loaded_political_quest.md)
- [\[ACL 2025\] Can Language Models Reason about Individualistic Human Values and Preferences?](../../ACL2025/llm_nlp/can_language_models_reason_about_individualistic_human_values_and_preferences.md)
- [\[ACL 2026\] Big AI is Accelerating the Metacrisis: What Can We Do?](big_ai_is_accelerating_the_metacrisis_what_can_we_do.md)
- [\[ACL 2025\] When to Speak, When to Abstain: Contrastive Decoding with Abstention](../../ACL2025/llm_nlp/when_to_speak_when_to_abstain.md)
- [\[ACL 2026\] Can AI Be a Good Peer Reviewer? A Survey of Peer Review Process, Evaluation, and the Future](can_ai_be_a_good_peer_reviewer_a_survey_of_peer_review_process_evaluation_and_th.md)

</div>

<!-- RELATED:END -->
