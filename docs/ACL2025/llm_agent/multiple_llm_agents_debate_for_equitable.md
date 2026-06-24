---
title: >-
  [Paper Note] Multiple LLM Agents Debate for Equitable Cultural Alignment
description: >-
  [ACL 2025][LLM Agent][multi-agent debate] Proposes the Multi-Agent Debate framework, where two LLM agents debate cultural scenarios adjudicated by a judge LLM. This significantly improves cultural adaptation accuracy and equity across cultural groups on the NormAd-eti benchmark, enabling 7-9B small models to achieve performance levels comparable to 27B models.
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "multi-agent debate"
  - "cultural alignment"
  - "social norms"
  - "LLM collaboration"
  - "parity"
date: 2026-05-08
content_hash: 633ebf52cc291f2b
---

# Multiple LLM Agents Debate for Equitable Cultural Alignment

**Conference**: ACL 2025  
**arXiv**: [2505.24671](https://arxiv.org/abs/2505.24671)  
**Code**: [https://github.com/dayeonki/cultural_debate](https://github.com/dayeonki/cultural_debate)  
**Area**: Agent  
**Keywords**: multi-agent debate, cultural alignment, social norms, LLM collaboration, parity

## TL;DR
Proposes the Multi-Agent Debate framework, where two LLM agents debate cultural scenarios adjudicated by a judge LLM. This significantly improves cultural adaptation accuracy and equity across cultural groups on the NormAd-eti benchmark, enabling 7-9B small models to achieve performance levels comparable to 27B models.

## Background & Motivation

**Background**: LLMs need to adapt to social norm judgments (e.g., etiquette, customs) in diverse global cultural contexts. Existing methods primarily rely on single-turn generation by a single LLM, improving cultural alignment through training data selection or prompt engineering.

**Limitations of Prior Work**: A single LLM is constrained by the distribution of its training data, failing to uniformly cover all cultural groups. Additionally, different LLMs perform optimally on different cultures due to variations in training data and alignment processes.

**Key Challenge**: No single model performs optimally across all cultures (oracle experiments show that the upper bound of combining two models is 22.5% higher than any single model), yet how to effectively leverage the complementarity of multiple models remains an open problem.

**Goal**: How to enhance the accuracy of cross-cultural social norm judgment and the equity across diverse cultural groups through multi-LLM collaboration rather than a single LLM.

**Key Insight**: Different open-source LLMs possess complementary cultural knowledge. Inspired by human debate, letting two LLMs debate cultural scenarios enables them to correct their respective biases by exchanging feedback.

**Core Idea**: Utilizing a multi-agent debate mechanism to elicit complementary cultural knowledge across different LLMs, thereby achieving more equitable cultural alignment.

## Method

### Overall Architecture
The input consists of cultural scenarios (country, social norm, story), and the output is a three-class label (Yes / No / Neither) indicating whether the behavior in the story conforms to the social etiquette norms of that culture. The framework comprises 4 strategies: single-model baseline, Self-Reflection, Debate-Only, and Self-Reflect+Debate.

### Key Designs

1. **Single-LLM + Self-Reflection**:

    - **Function**: A single LLM first generates an initial judgment $\hat{y}_0^{\mathcal{M}}$, then generates reflection rationales $f^{\mathcal{M}}$ for its own output, and finally makes a final decision $\hat{y}_f^{\mathcal{M}}$ by combining the reflections.
    - **Mechanism**: Utilizing cultural contextualization (incorporating rule-of-thumb information in the prompt) followed by self-reflection for further error correction. Integrating the rule-of-thumb improves absolute accuracy by an average of 39.1%.
    - **Design Motivation**: Serves as a baseline to verify the upper bound of improvement for a single model with multi-turn interactions.

2. **Debate-Only (Dual-Agent Debate)**:

    - **Function**: Two different LLMs $\mathcal{M}_1$ and $\mathcal{M}_2$ independently generate initial decisions, exchange feedback $f^{\mathcal{M}_1}$ and $f^{\mathcal{M}_2}$, and then each makes a final decision incorporating the peer feedback. If the final decisions do not match, a judge LLM adjudicates based on the debate history.
    - **Mechanism**: $\hat{y}_f^{\mathcal{M}_i} = \mathcal{M}_i(\hat{y}_0^{\mathcal{M}_i}, \hat{y}_0^{\mathcal{M}_j}, f^{\mathcal{M}_i}, f^{\mathcal{M}_j})$. When the final decisions of the two agents differ, the judge LLM performs adjudication by synthesizing the entire debate history.
    - **Design Motivation**: Leverages complementary cultural knowledge resulting from differences in training data among different LLMs. Swapping perspectives through debate helps correct individual cultural blind spots.

3. **Self-Reflect+Debate (Hybrid Mode)**:

    - **Function**: In each round of the debate, each agent dynamically chooses to either (A) self-reflect or (B) debate, meaning they choose to reflect on their own output or provide feedback on the opponent's view.
    - **Mechanism**: $\hat{y}_f^{\mathcal{M}_i} = \mathcal{M}_i(\hat{y}_0^{\mathcal{M}_i}, \hat{y}_0^{\mathcal{M}_j}, r^{\mathcal{M}_1}, f^{\mathcal{M}_2})$ (if $\mathcal{M}_1$ chooses self-reflect, and $\mathcal{M}_2$ chooses debate).
    - **Design Motivation**: Different LLMs show distinct preferences for feedback (some are better at self-reflection, while others excel in debating); allowing agents to autonomously choose the strategy most suitable for them.

4. **Cultural Group Parity Evaluation Metric**:

    - **Function**: Measures the equity of the method across different cultural groups.
    - **Mechanism**: $\text{Parity}(g) = \frac{\text{Acc}_g}{\text{Acc}_b}$, where $b$ represents the cultural group with the highest accuracy. A value closer to 1 indicates higher equity.
    - **Design Motivation**: Cultural alignment must not only improve overall accuracy but also ensure equitable coverage across underrepresented/minority cultural groups.

### Loss & Training
This work does not involve training; all methods are implemented during the inference phase. Seven open-source LLMs of size 7-9B are utilized to form 21 combinations. Gemma-2 27B is used as the Judge LLM.

## Key Experimental Results

### Main Results

| Method | Average Accuracy (%) | Gain |
|------|------------|---------|
| Single-LLM (w/o rule-of-thumb) | 49.4 | - |
| Single-LLM (w/ rule-of-thumb) | 66.9 | +35.4% |
| Self-Reflection | 68.9 | +3.0% |
| Debate-Only (D) | 76.3 | +10.7% |
| Self-Reflect+Debate (S+D) | 75.6 | +9.7% |
| Oracle Model Selection | 81.9 | Upper Bound |
| Gemma-2 27B Single Model | 79.2 | Reference |

**Key Results**: Debate-Only outperforms the single-model baseline in 20 out of 21 combinations. The best combination (LLaMA-3+Gemma-2) achieves 79.7%, matching the single-model performance of the 27B judge model.

### Ablation Study

| Configuration | Key Performance | Description |
|------|---------|------|
| Debate-Only Individual Gain | 19/21 combinations outperform single model | Average improvement of 7.05% |
| S+D Individual Gain | 14/21 combinations outperform single model | Effect is less consistent than Debate-Only |
| Post-Debate Adjudication > Individual | 11/21 | Adjudication strategies have room for improvement |
| Gemma-2+EXAONE-3 (S+D) | 80.4% | Outperforms the judge model acting alone |

### Key Findings
- **Decision Dynamics Analysis**: The two agents often disagree during the initial stage, but the debate effectively guides revisions, with most revisions converging toward the correct answer.
- **Cultural Group Equity**: Multi-agent debate achieves the highest cross-cultural group parity among all strategies, particularly benefiting underrepresented cultural groups (e.g., Afro-Islamic culture parity improved from 0.84 to 0.92).
- **Model Preference Differences**: Yi-1.5 and Aya-23 tend to favor self-reflect, whereas other models prefer debate; this is correlated with the training characteristics and language coverage of each model.

## Highlights & Insights
- **Quantitative Validation of Multi-Model Complementarity**: Oracle experiments prove that different LLMs possess complementary advantages across various cultures (upper bound of 81.9% vs. 70.7% for the best single model), providing a strong motivation for multi-agent methods.
- **Small Models Challenging Large Models**: 7-9B models achieve the performance of 27B models through debate, showing that collaborative reasoning can bridge the model scale gap—an insight that can transfer to other tasks requiring multi-perspective reasoning.
- **Equity as a First-Class Citizen**: Rather than just optimizing overall accuracy, the study explicitly optimizes parity among groups, which is a valuable practice for fairness research in NLP.

## Limitations & Future Work
- Evaluation is restricted to the NormAd-eti benchmark, lacking validation on other cultural or moral reasoning tasks.
- The adjudication strategy is relatively simple (utilizing a fixed judge LLM); more sophisticated adjudication (e.g., weighted voting, confidence-based) might yield further improvements.
- Only single-round debates have been explored; the effectiveness of multi-round debates warrants deeper investigation.
- All evaluated models are primarily English-centric instruction-tuned models, leaving their effectiveness in non-English cultural scenarios questionable.

## Related Work & Insights
- **vs CulturePark**: CulturePark utilizes multi-agent simulation to generate cultural data for training, whereas ours utilizes multi-agent debate directly during inference to enhance cultural alignment without requiring additional training.
- **vs Du et al. (Society of Minds)**: Their multi-agent debate is primarily applied to factuality and reasoning tasks, while this work is the first to apply it to cultural alignment and introduces the parity metric.

## Rating
- **Novelty**: ⭐⭐⭐ While multi-agent debate is not a new framework, applying it to cultural alignment and equity analysis is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated with 21 combinations across 7 models with detailed analysis, but limited to a single dataset.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure and intuitive figures.
- **Value**: ⭐⭐⭐ Validates the effectiveness of multi-agent collaboration in cultural alignment scenarios, offering practical reference value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Position: Assistive Agents Need Accessibility Alignment](../../ICML2026/llm_agent/position_assistive_agents_need_accessibility_alignment.md)
- [\[AAAI 2026\] MoralReason: Generalizable Moral Decision Alignment For LLM Agents Using Reasoning-Level Reinforcement Learning](../../AAAI2026/llm_agent/moralreason_generalizable_moral_decision_alignment_for_llm_agents_using_reasonin.md)
- [\[ACL 2026\] Taming Actor-Observer Asymmetry in Agents via Dialectical Alignment](../../ACL2026/llm_agent/taming_actor-observer_asymmetry_in_agents_via_dialectical_alignment.md)
- [\[ACL 2025\] LLM Agents Making Agent Tools](llm_agents_making_agent_tools.md)
- [\[ACL 2025\] An Empirical Study on LLM-based Agents for Automated Bug Fixing](an_empirical_study_on_llm-based_agents_for_automated_bug_fixing.md)

</div>

<!-- RELATED:END -->
