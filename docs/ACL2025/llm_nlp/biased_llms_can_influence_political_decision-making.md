---
title: >-
  [Paper Note] Biased LLMs Can Influence Political Decision-Making
description: >-
  [ACL 2025][LLM (Other)][political bias] Through two large-scale interactive experiments (N=299), this paper provides the first empirical evidence that LLMs with partisan biases can significantly influence human political opinions and budget allocation decisions. Crucially, this influence transcends partisan lines—Democrats can be persuaded by LLMs with conservative biases, and Republicans can likewise be influenced by LLMs with liberal biases.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "political bias"
  - "LLM influence"
  - "decision-making experiment"
  - "partisan bias"
  - "human-computer interaction"
date: 2026-05-08
content_hash: 003ec86323637f82
---

# Biased LLMs Can Influence Political Decision-Making

**Conference**: ACL 2025  
**arXiv**: [2410.06415](https://arxiv.org/abs/2410.06415)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: political bias, LLM influence, decision-making experiment, partisan bias, human-computer interaction

## TL;DR
Through two large-scale interactive experiments (N=299), this paper provides the first empirical evidence that LLMs with partisan biases can significantly influence human political opinions and budget allocation decisions. Crucially, this influence transcends partisan lines—Democrats can be persuaded by LLMs with conservative biases, and Republicans can likewise be influenced by LLMs with liberal biases.

## Background & Motivation

**Background**: As LLMs like ChatGPT become deeply integrated into daily information retrieval, the potential impact of their inherent biases on human decision-making has garnered widespread attention. Extensive existing work has documented social and political biases in LLMs; however, these studies primarily focus on the detection of bias rather than its actual impact.

**Limitations of Prior Work**: Research on the influence of biased LLMs on human attitudes and behaviors is highly limited and has yielded inconsistent conclusions. Existing studies either use static, non-interactive LLM-generated content or involve impersonal, fictional tasks that fail to reflect real-world usage scenarios. More importantly, the influence of LLMs on political decision-making has never been directly investigated, despite being a major societal concern.

**Key Challenge**: While the influence of traditional media bias has been well documented (e.g., Fox News shifting 3-8% of viewers toward the Republican party), LLMs introduce a new dynamic—they are simultaneously viewed as authoritative information sources and untrustworthy tools. Under this psychological ambivalence, is the actual influence of LLM bias amplified or suppressed?

**Goal**: To quantitatively evaluate the influence of LLMs with partisan biases on human political opinions and decision-making behaviors in free interaction scenarios through a rigorous experimental design.

**Key Insight**: To design two political tasks involving personal values (rather than fictional scenarios), allowing participants to freely interact with biased LLMs and measuring the influence under "blind" conditions.

**Core Idea**: Employing a 3×2 experimental design (liberal/conservative/neutral LLM × Democrat/Republican participants) to quantify the political influence of biased LLMs by measuring changes in stance before and after interactions.

## Method

### Overall Architecture
The experiment recruited 299 participants (via the Prolific platform), balanced by political party (150 Republicans, 149 Democrats). Each participant completed two tasks: the Topic Opinion Task and the Budget Allocation Task. Participants were randomly and blindly assigned to one of three experimental conditions: a liberal-biased LLM, a conservative-biased LLM, or a neutral LLM. All models were built upon GPT-3.5-turbo, with bias introduced via system prompt prefixes.

### Key Designs

1. **Topic Opinion Task**:

    - Function: Measuring the influence of LLMs on participants' political opinions
    - Mechanism: Four relatively niche political topics were selected to minimize the confounding effect of participants' prior knowledge: multi-family housing and the Lacey Act (liberal leaning), and international unilateralism and covenant marriage (conservative leaning). Participants first reported their knowledge level and stance on these topics (using a 7-point Likert scale), then engaged in a free-form dialogue with the LLM to gather information (3-20 rounds of interaction), and finally reported their stance again. Changes in stance before and after the interaction were analyzed using ordinal logistic regression: $Y = \beta_0 + \beta_1 L + \beta_2 C + \epsilon$.
    - Design Motivation: Using niche topics simulates real-world scenarios—it is precisely on unfamiliar topics that people seek information from LLMs, which is when bias has the greatest potential to influence.

2. **Budget Allocation Task**:

    - Function: Measuring the influence of LLMs on participants' actual decision-making behaviors
    - Mechanism: Participants played the role of a city mayor, allocating surplus government funds to four departments (public safety, education, veteran services, welfare). They made an initial allocation, submitted it to the LLM for feedback, discussed it with the LLM, and then submitted their final allocation. Changes in allocation were analyzed using ANOVA and Dunnett's post-hoc tests.
    - Design Motivation: Budget allocation is a concrete decision-making behavior (rather than an abstract opinion), which better reflects the actual influence of LLMs. The choice of the four departments reflects typical policy priority preferences of conservative and liberal factions.

3. **Biased Model Construction and Verification**:

    - Function: Creating experimental models with controllable levels of bias
    - Mechanism: Biases were injected into GPT-3.5-turbo via system prefixes (e.g., "Respond as a radical left U.S. Democrat..."). The validity of the bias was verified using the Political Compass Test (PCT): the liberal-biased model showed a liberal stance on the PCT, the conservative-biased model showed a conservative stance, and the neutral model declined to state a position on 76% of PCT questions.
    - Design Motivation: The prefix-based approach avoids the high cost of fine-tuning while ensuring consistency of bias through explicit instructions. PCT verification guarantees the validity of the experimental conditions.

### Loss & Training
This study is an empirical experiment on human behavior and does not involve model training. Statistical tests (ordinal logistic regression, ANOVA, and Dunnett's test) were used to analyze the results.

## Key Experimental Results

### Topic Opinion Task Results

| Participant Party | Topic Leaning | LLM Bias | β Value | t Value | p Value |
|-----------|---------|---------|-----|-----|-----|
| Democrat | Conservative Topic | Liberal Bias | -0.85 | -2.38 | **0.02** |
| Democrat | Conservative Topic | Conservative Bias | 0.98 | 2.71 | **<0.01** |
| Republican | Conservative Topic | Liberal Bias | -0.79 | -2.16 | **0.03** |
| Democrat | Liberal Topic | Conservative Bias | 1.44 | 3.82 | **<0.01** |
| Republican | Liberal Topic | Conservative Bias | 1.42 | 3.91 | **<0.01** |

### Budget Allocation Task Results

| Participant | Department | LLM Bias | Dunnett p-value | Description |
|--------|------|---------|------------|------|
| Democrat | Safety | Liberal | **<0.01** | Significant change |
| Democrat | Veterans | Conservative | **<0.01** | Democrats persuaded by conservative LLM to increase veterans funding |
| Democrat | Education | Conservative | **<0.01** | Democrats persuaded by conservative LLM to decrease education funding |
| Republican | Safety | Liberal | **<0.01** | Republicans persuaded by liberal LLM to decrease safety funding |
| Republican | Education | Liberal | **0.03** | Republicans persuaded by liberal LLM to increase education funding |
| Republican | Welfare | Conservative | **0.03** | Cross-partisan influence is also significant |

### Key Findings
- The influence of biased LLMs crosses partisan boundaries: Democrats were significantly influenced by conservative LLMs (and vice-versa), challenging the traditional assumption that "people will resist information that contradicts their own beliefs."
- The influence in the budget allocation task was stronger and more comprehensive—funding for almost all departments was significantly affected.
- Approximately 54% of participants could correctly identify that the model is biased; **however, recognizing the bias did not diminish its influence**—a sobering finding.
- Participants with self-reported higher AI knowledge were slightly less influenced (weak correlation), suggesting that AI education could be a viable mitigation strategy.
- Distinct interaction patterns emerged between the two tasks: in the topic task, 80.7% of participants used the LLM like a search engine, whereas in the budget task, 48% actively sought the LLM's recommendation.

## Highlights & Insights
- Highly rigorous experimental design: double-blind, randomized assignment, pre- and post-test comparisons, and cross-task validation. This represents a gold-standard experimental setup in HCI and AI Safety.
- The most striking finding is that "recognizing bias does not grant immunity to its influence"—which upends the "cognitive inoculation" assumption in communication research. This implies that simply warning users about AI bias may not suffice.
- In terms of persuasion analysis, biased LLMs do not employ different persuasive techniques, but rather utilize different framing dimensions—the conservative-biased LLM emphasizes a "safety/defense" frame, while the liberal-biased LLM emphasizes a "fairness/economy" frame. This aligns with the mechanics of traditional media bias, suggesting that LLM bias might require analogous mitigation strategies.

## Limitations & Future Work
- The study only utilizes GPT-3.5-turbo, leaving it unclear whether the findings generalize to other LLMs.
- Only immediate effects were measured, and the long-term impact remains unknown—does the bias effect fade after the interaction ends?
- Participants were limited to U.S. citizens; the idiosyncrasies of the U.S. two-party system restrict the cross-cultural generalizability of the findings.
- Future research could investigate: the dose-response relationship of different levels of bias on influence; and the impact in real-world scenarios (such as pre-election information searches).

## Related Work & Insights
- **vs. Fox News Influence Studies**: While Fox News shifted 3-8% of viewers' votes, the bias influence of LLMs in this study was more direct and salient—potentially due to the conversational nature and perceived "objectivity" of the LLM.
- **vs. Jakesch et al. (2023)**: Prior studies examining the effects of autocomplete suggestions yielded mixed findings; this work employs a free-dialogue paradigm to obtain more definitive conclusions.
- **vs. Static Bias Detection Work**: This study advances the field from "detecting the existence of bias" to "whether bias actually influences humans," completing a critical link in the causal chain.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The first experimental study to directly quantify the influence of biased LLMs on political decision-making.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Exceptionally rigorous, including dual tasks, pre- and post-tests, multidimensional analysis, and persuasion-tactic profiling.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure and compelling conclusions.
- Value: ⭐⭐⭐⭐⭐ Carries direct significance for AI policy formulation and societal impact assessments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Can LLMs Ground when they (Don't) Know: A Study on Direct and Loaded Political Questions](can_llms_ground_when_they_dont_know_a_study_on_direct_and_loaded_political_quest.md)
- [\[ACL 2025\] Only a Little to the Left: A Theory-grounded Measure of Political Bias in LLMs](political_bias_theory_grounded.md)
- [\[ACL 2025\] Leveraging In-Context Learning for Political Bias Testing of LLMs](leveraging_in-context_learning_for_political_bias_testing_of_llms.md)
- [\[ACL 2025\] Can LLMs Understand Unvoiced Speech? Exploring EMG-to-Text Conversion with LLMs](can_llms_understand_unvoiced_speech_exploring_emg-to-text_conversion_with_llms.md)
- [\[ACL 2025\] Can LLMs Interpret and Leverage Structured Linguistic Representations? A Case Study with AMRs](can_llms_interpret_and_leverage_structured_linguistic_representations_a_case_stu.md)

</div>

<!-- RELATED:END -->
