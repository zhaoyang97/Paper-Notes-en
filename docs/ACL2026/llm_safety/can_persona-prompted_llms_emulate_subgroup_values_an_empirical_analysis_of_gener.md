---
title: >-
  [Paper Note] Can Persona-Prompted LLMs Emulate Subgroup Values? An Empirical Analysis of Generalisability and Fairness in Cultural Alignment
description: >-
  [ACL 2026][LLM Safety][Cultural Alignment] Using a subset of the Singapore World Values Survey (WVS) as a case study, this paper constructs 20,877 (question…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Cultural Alignment"
  - "Subgroups"
  - "Persona Simulation"
  - "Fairness"
  - "WVS"
date: 2026-05-08
content_hash: 1787fdf37e0ade6e
---

# Can Persona-Prompted LLMs Emulate Subgroup Values? An Empirical Analysis of Generalisability and Fairness in Cultural Alignment

**Conference**: ACL 2026  
**arXiv**: [2604.12851](https://arxiv.org/abs/2604.12851)  
**Code**: To be confirmed  
**Area**: LLM Safety / Values  
**Keywords**: Cultural Alignment, Subgroups, Persona Simulation, Fairness, WVS

## TL;DR
Using a subset of the Singapore World Values Survey (WVS) as a case study, this paper constructs 20,877 (question, subgroup) pairs to verify whether LLMs can emulate fine-grained demographic subgroup value preferences. Results show that GPT-4.1 zero-shot achieves only 57.4% accuracy; while simple SFT improves performance on OOD subgroups by an average of 17.4%, the performance gap between subgroups widens from an NMAE perspective, with models consistently favoring young/male/Chinese/Christian personas.

## Background & Motivation

**Background**: Existing LLM alignment paradigms (e.g., RLHF, DPO) treat "human values" as a monolithic target, which often reflects Western-centric value preferences and has been criticized for the "coloniality of knowledge." While benchmarks like WorldValuesBench have elevated alignment analysis to the national level, they still neglect value disagreements between subgroups within a single country.

**Limitations of Prior Work**: (1) National-level alignment may make LLMs appear useful for certain subgroups (e.g., young, male, Chinese, Christian) while performing poorly or even offensively for others (e.g., elderly, Malay, Muslim), a bias masked by average benchmarks. (2) Existing persona-prompt research mostly simulates "archetypes" (e.g., "doctor") without calibration against real demographic data, often introducing pseudo-biases. (3) There is no systematic answer to how large the value conflicts between subgroups are, whether they can be aligned using simple methods, and how fairness changes after alignment.

**Key Challenge**: Single global alignment vs. value diversity across hundreds of subgroups. One must either sacrifice diversity (one-size-fits-all) or collect preference data for every intersectional persona, which is not scalable.

**Goal**: (1) Quantitatively map the value landscape of a pluralistic society to identify points of consensus and conflict; (2) Test whether simple SFT can generalize to unseen intersectional personas and open-ended generation; (3) Evaluate the impact of alignment on subgroup fairness.

**Key Insight**: Choosing Singapore as a "microcosm of a pluralistic society"—featuring three major ethnicities (Chinese, Malay, Indian) and diverse religions (Buddhism, Islam, Christianity, Hinduism, and non-religious)—it is geographically small but rich in stratification dimensions. The study uses data from 2,012 Singaporean respondents across 214 value-based questions from WVS Wave 7 as anchor data.

**Core Idea**: Operationalize "subgroup alignment" as a modal answer prediction task for intersectional personas (sex × age × ethnicity × religion). Use structured numerical preferences for SFT to learn compositional persona representations and test generalization to unseen intersections (e.g., ethnicity × religion) in training.

## Method

### Overall Architecture
A four-stage approach: (1) Obtain raw data from 2,012 Singaporean respondents across 214 questions from WVS; (2) Quantify the degree of conflict for each question across subgroups using the Modal Diversity Score; (3) Construct 20,877 (question, subgroup) samples, partitioning 50 fundamental strata into the Train Set and 48 unseen intersectional strata into the OOD Eval Set; (4) Perform LoRA SFT on seven $\le$ 8B open-source LLMs, evaluating structured numerical prediction and open-ended generation tasks using Accuracy/NMAE/Win Rate metrics along with Norm. Range and CV for fairness.

### Key Designs

1.  **Modal Diversity Score (Value Conflict Quantification)**:
    - **Function**: Assigns a score of 0–1 to each WVS question to characterize the diversity of modal answers across different subgroups.
    - **Mechanism**: For a stratum (e.g., sex_x_age), collect the modal answers of all subgroups and calculate the normalized Shannon entropy of this distribution: $$\text{Score}_{\text{MD}} = \frac{-\sum_{m\in M}p(m)\log_2 p(m)}{\log_2 (\min(|S|,|C|))}$$, where $M$ is the set of unique modal answers, and $p(m)$ is the proportion of subgroups choosing mode $m$. 0 indicates consensus; 1 indicates extreme divergence. Mean pairwise Wasserstein distance is used for ordinal-aware cross-checking.
    - **Design Motivation**: Traditional cultural benchmarks only look at "overall accuracy," failing to indicate which topics require "subgroup-aware" alignment. This score identifies highly divisive topics (e.g., Religious Values avg. 0.318) and unifying topics (e.g., Social Capital 0.084) to provide priors for model evaluation.

2.  **Compositional Generalisation Split**:
    - **Function**: Tests the model's ability to learn compositional capabilities from single-axis or dual-axis personas and generalize to unseen intersections.
    - **Mechanism**: The Train Set includes fundamental strata (sex, age_group, ethnicity, religion) and pairwise combinations (sex × age, sex × religion, sex × ethnicity) totaling 50 subgroups and 10,700 samples. The Eval (OOD) Set includes three unseen pairwise combinations (age × religion, age × ethnicity, ethnicity × religion) totaling 48 subgroups and 10,177 samples. Each subgroup requires at least 30 respondents. This forces the model to learn to synthesize single-axis preferences into intersectional ones rather than memorizing persona-answer mappings.
    - **Design Motivation**: In-distribution splits cannot distinguish between memorization and generalization; accuracy gains can only be attributed to compositional understanding when combinations like ethnicity_x_religion are entirely absent from training.

3.  **Dual-Perspective Fairness Evaluation (Acc vs NMAE × Norm. Range vs CV)**:
    - **Function**: Exposes unfairness where accuracy improves but disparities also increase, which is otherwise hidden by single metrics.
    - **Mechanism**: Norm. Range $= (P_{\max} - P_{\min}) / P_{\max}$ measures extreme disparities; CV $= \sigma / \mu$ measures overall dispersion. Both Accuracy (distance-insensitive) and NMAE (ordinal-distance-sensitive) metrics are used. Results found that SFT decreased the Norm. Range for Accuracy from 0.240 to 0.179 (more fair), but increased the Norm. Range for NMAE from 0.280 to 0.336 (less fair).
    - **Design Motivation**: Accuracy treats an error of 1 step and 5 steps as equally wrong, whereas NMAE reflects the true error magnitude. Inverse movement in these metrics suggests a fairness paradox—SFT pulls more subgroups above the "passing line," but the precision for advantaged subgroups is further amplified.

### Loss & Training
LoRA SFT was applied to all open-source models with a learning rate of $1 \times 10^{-6}$ (conservative to prevent overfitting) for 1 epoch. Input consists of a prompt describing the persona and question, outputting the modal numerical answer. Open-ended evaluation used Mistral-Small-3.1-24B (INT8) as a judge against GPT-4.1, with two swaps to eliminate position bias. Win Rate $\text{WR}_c = (s_{1,c} + s_{2,c}) / 2$, with win=1, tie=0.5, loss=0.

## Key Experimental Results

### Main Results
Comparison of 7 open-source and 4 closed-source models on the OOD split (selected):

| Model | Acc Base | Acc SFT (Δ) | NMAE Base | NMAE SFT (Δ) | Overall WR Base | Overall WR SFT |
|------|----------|-------------|-----------|--------------|-----------------|----------------|
| Llama-3.1-8B | .514 | **.685** (+.171) | .258 | .143 (-.115) | .294 | .320 (+.026) |
| Llama-3.2-3B | .442 | .508 (+.066) | .308 | .238 (-.070) | .230 | .234 (+.004) |
| SEA-LION-v3-8B | **.530** | .642 (+.112) | .222 | .158 (-.064) | **.428** | **.430** (+.002) |
| Qwen2.5-7B | .442 | .661 (+.219) | .243 | .157 (-.086) | .223 | .246 (+.023) |
| Sailor2-8B | .356 | **.720** (+**.364**) | .332 | .125 (-.207) | .217 | .255 (+.038) |
| SeaLLMs-v3-7B | .440 | .696 (+.256) | .256 | .135 (-.121) | .082 | .081 (-.001) |
| Phi-4-mini | .427 | .456 (+.029) | .267 | .256 (-.011) | .175 | .161 (-.014) |
| **Open-source Avg** | **.450** | **.624** (+**.174**) | **.269** | **.173** (-.096) | .236 | .247 (+.011) |
| GPT-4.1 | .574 | – | .182 | – | .500 | – |
| GPT-4o | .565 | – | .189 | – | .370 | – |
| GPT-4o-mini | .490 | – | .217 | – | .310 | – |

**Key Findings**: (1) GPT-4.1 zero-shot yields only 57.4%, indicating subgroup-aware alignment is a difficult task; (2) Open-source models improved by an average of 17.4 points after SFT, with several (Sailor2, SeaLLMs, Llama-3.1) surpassing GPT-4.1’s OOD performance; (3) SEA-LION-v3 base was the strongest (regional pre-training is effective) but saw minimal gain from SFT; (4) Open-ended Win Rate saw small gains (+1.1%), but the Value dimension rose by 2.2%, indicating structured training partially transfers to free generation.

### Ablation Study (OOD split)

| Model | Acc Norm.Range Base→SFT | Acc CV Base→SFT | NMAE Norm.Range Base→SFT | NMAE CV Base→SFT |
|------|--------------------------|------------------|--------------------------|------------------|
| Llama-3.1-8B | .174 → .188 | .056 → .054 | .250 → **.426** | .085 → .133 |
| Qwen2.5-7B | .256 → .169 | .089 → .055 | .318 → .352 | .108 → .135 |
| Sailor2-8B | .305 → .145 | .101 → .044 | **.228 → .343** | .068 → .129 |
| SeaLLMs-v3 | .276 → .124 | .094 → .037 | .294 → .318 | .108 → .111 |
| **Avg** | .240 → **.179** | .078 → .054 | .280 → **.336** | .094 → .116 |

**Fairness Paradox**: Accuracy fairness improved across all models (more subgroups reached the passing line), but NMAE fairness worsened almost universally (precision for advantaged subgroups was amplified).

### Key Findings
- **GPT-4.1 still at 57.4%**: This shows subgroup-aware alignment cannot be solved by prompt engineering alone; closed-source SOTA models fail with simple persona prompts.
- **Fixed patterns in pre-existing bias**: Both base and SFT models systematically favor young/male/Chinese/Christian personas, performing worse on elderly/Malay/Indian/Muslim personas; SFT widens this gap from an NMAE perspective.
- **SFT reduces refusal**: Questions regarding homosexuality or domestic violence, previously refused due to safety alignment, saw refusal rates drop from 6.66% to nearly zero post-SFT—revealing a tension between safety alignment and cultural emulation.
- **Sailor2 gain of +36.4%**: The Southeast Asian multilingual model benefited most from WVS-SG data, highlighting strong synergy between regional pre-training and fine-tuning.
- **Religious Values are most divisive (MDS=0.318)**: Reflects Singapore's religious diversity; Social Capital & Trust are most consistent (0.084).
- **LLM Judge Calibration**: The Mistral-24B judge achieved a w-Kappa of 0.568 with humans on the Overall dimension (comparable to human-human 0.552); however, judges are unreliable for the Persona dimension (H-AI w-Kappa 0.318 vs H-H 0.388).

## Highlights & Insights
- **Modal Diversity Score is a simple, reusable tool**: Normalized Shannon entropy can quantify "subgroup conflict" in any stratified survey data, transferable to other countries and domains like medical preferences or political issues.
- **Compositional split is the gold standard for persona generalization**: Training on sex × age + sex × ethnicity and testing on age × ethnicity is a true OOD test. This split design should be standard for persona alignment evaluation.
- **Fairness paradox serves as a warning**: A subgroup-balanced training set $\ne$ a subgroup-equitable outcome; coarse metrics (accuracy) can hide increased inequality in fine metrics (NMAE). Explicit fairness losses or upsampling disadvantaged subgroups is necessary during training.
- **Structured training transfers to open-ended generation**: Training on numerical modal answers improved Value WR in open-ended generation, proving SFT updates the model's internal persona representation rather than just surface mapping.

## Limitations & Future Work
- The study is restricted to Singapore WVS Wave 7; cross-country generalizability requires further validation.
- Using modal answers as supervision signals simplifies intra-subgroup distributions and "mutes" minority views within subgroups; distributional alignment is a preferred direction.
- Only SFT was explored; comparison with advanced preference optimization like DPO/GRPO or group-conditioning is missing.
- Data contamination from WVS Wave 7 in pre-training cannot be entirely ruled out, though SFT's large gains and GPT-4.1's low performance suggest otherwise.
- Low inter-human agreement on persona criteria (w-Kappa 0.388) suggests the definition of "persona authenticity" is inherently subjective.

## Related Work & Insights
- **vs WorldValuesBench (Zhao 2024)**: While they focus on national-level value awareness, this work targets finer subgroup-level granularity within a single nation.
- **vs CulturalLLM (Li 2024)**: They use cultural data for training but focus on cross-national differences; this work focuses on intra-national intersectional differences.
- **vs Whose opinions (Santurkar 2023)**: They expose LLM preferences for certain demographic groups; this work quantifies these biases and attempts correction via SFT.
- **vs RoleLLM (Wang 2024)**: They use character archetypes (e.g., fictional characters), whereas this work uses empirically grounded demographic personas.

## Rating
- Novelty: ⭐⭐⭐⭐ Modal Diversity Score + compositional OOD split + fairness paradox are original elements, though SFT itself is not new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 7 open + 4 closed models × 2 tasks × multiple fairness metrics × human-calibrated LLM judge; very solid.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear overview in Figure 1, robust definitions/formulas, and honest discussion of limitations.
- Value: ⭐⭐⭐⭐⭐ A wake-up call for "cultural alignment" research showing that average gains don't mean fairness and that SFT can amplify pre-existing biases; provides direct guidance for future fairness-aware alignment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Distributive Fairness in Large Language Models: Evaluating Alignment with Human Values](../../NeurIPS2025/llm_safety/distributive_fairness_in_large_language_models_evaluating_alignment_with_human_v.md)
- [\[ACL 2026\] CAP: Controllable Alignment Prompting for Unlearning in LLMs](cap_controllable_alignment_prompting_for_unlearning_in_llms.md)
- [\[ACL 2026\] Decomposed Trust: Privacy, Adversarial Robustness, Ethics, and Fairness in Low-Rank LLMs](decomposed_trust_privacy_adversarial_robustness_ethics_and_fairness_in_low-rank_.md)
- [\[ACL 2026\] How Should We Enhance the Safety of Large Reasoning Models: An Empirical Study](how_should_we_enhance_the_safety_of_large_reasoning_models_an_empirical_study.md)
- [\[AAAI 2026\] Can Editing LLMs Inject Harm?](../../AAAI2026/llm_safety/can_editing_llms_inject_harm.md)

</div>

<!-- RELATED:END -->
