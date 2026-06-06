---
title: >-
  [Paper Note] Same Voice, Different Lab: On the Homogenization of Frontier LLM Personalities
description: >-
  [ACL2026][LLM Evaluation][LLM personality] Through external ELO preference evaluations of 144 personality traits, this paper finds that nine frontier LLMs from different labs have generally converged toward "Assistant-li…
tags:
  - "ACL2026"
  - "LLM Evaluation"
  - "LLM personality"
  - "trait ELO"
  - "personality homogenization"
  - "character training"
  - "user experience"
date: 2026-05-08
content_hash: 8256a90fc45cb239
---

# Same Voice, Different Lab: On the Homogenization of Frontier LLM Personalities

**Conference**: ACL2026  
**arXiv**: [2605.02897](https://arxiv.org/abs/2605.02897)  
**Code**: https://github.com/p3rciv3l/character_elicitation  
**Area**: LLM Evaluation / Human-Computer Interaction / Model Personality  
**Keywords**: LLM personality, trait ELO, personality homogenization, character training, user experience

## TL;DR
Through external ELO preference evaluations of 144 personality traits, this paper finds that nine frontier LLMs from different labs have generally converged toward "Assistant-like" personalities (e.g., structured, systematic, precise). Differences are primarily concentrated in mid-range stylistic traits such as poetic or playful.

## Background & Motivation
**Background**: The perceived quality of LLMs for users depends not only on mathematical, coding, or factual capabilities but is also highly influenced by the model's "speaking style" and personality. After model version updates, users often feel that responses become colder, more mechanical, or less expressive.

**Limitations of Prior Work**: Early LLM personality research often directly applied human psychological scales like the Big Five or MBTI, or directly asked models for self-descriptions. These methods are prone to anthropomorphic assumptions, model sycophancy, and construct mismatch, which may not truly reflect the trait preferences expressed during actual interactions.

**Key Challenge**: Model developers all pursue a helpful, safe, and reliable assistant experience. However, if optimization goals, annotator preferences, and safety constraints converge, frontier models may lose stylistic diversity. In terms of user experience, this manifests as "same voice from different labs."

**Goal**: The authors aim to measure the relative preferences of different frontier models across a large number of interaction style traits using a method closer to revealed preference. The study seeks to answer: whether model personalities are converging; which traits exhibit the most difference; and how model updates within the same company change personality profiles.

**Key Insight**: Drawing from the pairwise trait elicitation in Open Character Training, the tested model is asked to implicitly choose one of two traits to enact in a single-turn conversation. An external base model judge then determines which trait was expressed, eventually forming a trait ranking through ELO.

**Core Idea**: Instead of asking the model "What is your personality?", inferred revealed preferences are derived from massive pairwise trait choices and external adjudication.

## Method

### Overall Architecture
The experiment involves 144 traits sourced from the Open Character Training list. For each tested model, the system requires the model to choose one style from two candidate traits in a single-turn dialogue and implement it in the response without explicitly stating the choice. Subsequently, GLM-4.5 Air serves as a relatively neutral base model judge to determine which trait the response aligns with. Massive pairwise judgments are aggregated into ELO scores to form a trait ranking for each model.

The authors tested nine frontier models: GPT-5.1, Claude Haiku 4.5, Gemini 3 Flash Preview, Qwen3 VL 235B A22B Thinking, DeepSeek-V3.2, Grok 4 Fast, Kimi K2 Thinking, Ministral-14b-2512, and Trinity-Mini. A total of 102,560 single-turn responses were generated, and the harness and data were open-sourced.

### Key Designs

1.  **revealed preference trait elicitation**:
    - **Function**: Avoids direct psychological testing or self-reporting, instead estimating trait preferences from model behavior.
    - **Mechanism**: Provides the model with two traits and asks it to implicitly role-play one in the system prompt; the external judge determines which trait is expressed based solely on the output. All pairwise win-loss relationships are used for ELO calculation.
    - **Design Motivation**: LLMs tend to cater, interpret, or repeat scale definitions when self-assessing personality, whereas revealed preference is closer to the style choices models make in real interactions.

2.  **Assistant traits vs. Creative traits**:
    - **Function**: Compresses the trait space into interpretable stylistic directions to observe whether a model behaves more like a rigorous assistant or a creative expressive.
    - **Mechanism**: Assistant traits include systematic, structured, precise, methodical, analytical, focused, etc.; Creative traits include creative, imaginative, poetic, artistic, playful, humorous, bold, visionary, etc. The authors compare the average ELO of each model across these two groups.
    - **Design Motivation**: User intuitions of being "more mechanical" or "more interesting" require quantifiable axes. This classification allows personality differences to be interpreted as stylistic orientations rather than isolated trait rankings.

3.  **Cross-model rank variance and version difference analysis**:
    - **Function**: Locates where model convergence and divergence occur within the trait distribution.
    - **Mechanism**: Calculates the standard deviation of rankings for each trait across the nine models and stratifies them by average rank; simultaneously compares trait rank shifts between GPT-4o and GPT-5.1.
    - **Design Motivation**: Average correlation only indicates overall similarity; rank variance reveals which traits have reached industrial consensus and which still retain laboratory-specific differences.

### Loss & Training
This work is an evaluation methodology rather than a training method. The core statistical objects are the judge's win-loss results and ELO rankings for trait pairs. The authors use Spearman correlation to measure trait ranking consistency between models, rank standard deviation to measure disagreement, and PCA to analyze which trait clusters primarily account for model differences.

## Key Experimental Results

### Main Results

| Analysis Item | Result | Implications |
| :--- | :--- | :--- |
| Inter-model Spearman Correlation | 0.636 to 0.906, mean 0.763 | Personality rankings are overall highly similar across frontier models. |
| Highest Correlated Pair | Claude 4.5 vs GPT-5, $\rho=0.906$ | Different labs can form very similar assistant styles. |
| Lowest Correlated Pair | Qwen 3 vs Trinity, $\rho=0.636$ | Stylistic differences still persist in some cases. |
| Median Traits Variance | $\sigma=22.5$ for ranks 51-100 | Personality differences are concentrated in mid-range stylistic features. |
| Style Explanatory Power | Stylistic differences account for 64.2% of variation | Differences are more about expression style than capability dimensions. |
| Total Responses | 102,560 single-turn responses | Scale is sufficient to support trait-level ranking analysis. |

### Ablation Study

While there is no model ablation in the traditional sense, several layered and comparative analyses serve as an analytical breakdown of the evaluation design.

| Analysis Configuration | Key Indicator | Description |
| :--- | :--- | :--- |
| Top 20 traits | Average $\sigma=9.2$ | Most frequently expressed traits are highly converged (e.g., structured, systematic, precise). |
| ranks 21-50 | Average $\sigma=18.5$ | Technicality, detail, and confidence remain relatively consistent. |
| ranks 51-100 | Average $\sigma=22.5$ | Mid-range traits like reflective, decisive, and verbose show the most divergence. |
| ranks 100-144 | Average $\sigma=15.7$ | Models are also converged on traits they avoid (e.g., foolish, sycophantic). |
| Creative vs Assistant | Assistant ELO > Creative ELO | Default industry style leans toward structured, objective, and restrained. |
| GPT-4o vs GPT-5.1 | Spearman $\rho=0.831$, poetic rank 29 $\rightarrow$ 124 | Updates within the same series can significantly alter expressive style. |

### Key Findings
- Frontier models generally prefer Assistant-like traits such as structured, systematic, and precise while suppressing traits like foolish and sycophantic, indicating an implicit cross-lab consensus in character training.
- Convergence follows an inverse U-shape: traits that are most and least frequently expressed have low variance, while mid-tier traits have the highest variance. Model "personality" stems primarily from mid-distribution traits like poetic, contemplative, simplistic, and playful.
- Models from xAI, Alibaba, and Mistral are relatively more "Creative," with Creative ELOs closer to the neutral 1000; GPT-5 has the lowest average Creative ELO at 757.
- GPT-5.1 is more professional and conservative compared to GPT-4o: patient increased by 62 ranks, conservative increased by 61 ranks, and structured moved from rank 9 to rank 1; meanwhile, poetic dropped from rank 29 to 124, with idealistic, nostalgic, and enthusiastic also declining significantly.
- Model providers' suppression of sycophancy may drive a more structured and restrained style but at the possible cost of expressivity and creativity.

## Highlights & Insights
- The paper avoids directly applying human psychological scales, opting instead for behavioral preference evaluations, which are better suited for LLMs than MBTI or Big Five tests.
- The "inverse U-shape personality variance" is an insightful finding: industrial consensus shapes the most used and most avoided expressions, leaving the middle ground for laboratory-specific styles.
- The comparison between GPT-4o and GPT-5.1 translates abstract personality analysis into concrete user perception, explaining why users find newer versions colder, narrower, and more task-oriented.
- The study serves as a reminder that model alignment is not just a matter of safety and factuality but also a collective optimization of interactive aesthetics and cultural preferences.

## Limitations & Future Work
- Conclusions rely on GLM-4.5 Air as a judge; despite selecting a base model to reduce bias, the judge may still possess implicit stylistic preferences.
- The experiment only covers single-turn dialogues, whereas model personality may shift with multi-turn context, user tone, task pressure, and memory states.
- The trait list originates from Open Character Training; while broad, it does not represent the entire personality space, and interpretations may vary across cultural contexts.
- Some models used smaller versions to save costs; while the authors believe styles are generalizable within families, model size and product configuration could still affect style.
- ELO compresses complex expressions into pairwise rankings, making it difficult to explain combinatorial effects and contextual dependencies between traits.

## Related Work & Insights
- **vs. Big Five / MBTI Testing**: Human psychological scales assume personality constructs valid for human populations, but LLM outputs do not necessarily satisfy these factor structures; this paper focuses on observable response styles.
- **vs. LLM output homogeneity**: Previous work often discusses homogeneity in content; this paper shifts the focus to character training and interaction personality.
- **vs. Open Character Training**: This paper reuses the revealed preference method but scales it to a cross-sectional comparison of 2026 frontier models and an analysis of GPT versions.
- **Insight**: Future LLM evaluations should report "capability leaderboards" and "stylistic coordinate systems" separately, clarifying a model's orientation in dimensions like creativity, restraint, or directness for users and developers.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The method builds on existing revealed preference frameworks, but the systematic analysis of personality homogenization in frontier models is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers nine models, 144 traits, and 100k+ responses, though multi-turn and cross-cultural validation is missing.
- Writing Quality: ⭐⭐⭐⭐☆ Clear narrative; charts capture core phenomena, though some conclusions still rely on judge-related assumptions.
- Value: ⭐⭐⭐⭐☆ Provides practical reference value for LLM product experience, model evaluation, and character training design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Same Content, Different Representations: A Controlled Study for Table QA](../../ICLR2026/llm_evaluation/same_content_different_representations_a_controlled_study_for_t.md)
- [\[ACL 2026\] Stability vs. Manipulability: Evaluating Robustness Under Post-Decision Interaction in LLM Judges](stability_vs_manipulability_evaluating_robustness_under_post-decision_interactio.md)
- [\[ACL 2026\] Statistically Reliable LLM-Based Ranking Evaluation via Prediction-Powered Inference](statistically_reliable_llm-based_ranking_evaluation_via_prediction-powered_infer.md)
- [\[ACL 2026\] Contrastive Decoding Mitigates Score Range Bias in LLM-as-a-Judge](contrastive_decoding_mitigates_score_range_bias_in_llm-as-a-judge.md)
- [\[ACL 2026\] Reasoning Model Is Superior LLM-Judge, Yet Suffers from Biases](reasoning_model_is_superior_llm-judge_yet_suffers_from_biases.md)

</div>

<!-- RELATED:END -->
