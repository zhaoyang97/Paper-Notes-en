---
title: >-
  [Paper Note] HumT DumT: Measuring and Controlling Human-like Language in LLMs
description: >-
  [ACL 2025][LLM (Other)][human-like tone] This paper proposes HumT, a metric for human-like tone based on GPT-2 log-probability ratios, and its social perception generalization SocioT. Analysis of over 400k preference samples reveals that users generally prefer LLM outputs with lower human-likeness. Furthermore, human-like tone strongly correlates with social closeness ($r=0.87$), low status ($r=-0.80$), and femininity ($r=0.47$). Finally, DPO fine-tuning (DumT) using only 500…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "human-like tone"
  - "anthropomorphism"
  - "DPO"
  - "social perception"
  - "user preference"
date: 2026-05-08
content_hash: 33b25a3b6da3b565
---

# HumT DumT: Measuring and Controlling Human-like Language in LLMs

**Conference**: ACL 2025  
**arXiv**: [2502.13259](https://arxiv.org/abs/2502.13259)  
**Code**: [github.com/myracheng/humtdumt](https://github.com/myracheng/humtdumt)  
**Area**: NLP Generation / Human-Computer Interaction / AI Safety  
**Keywords**: human-like tone, anthropomorphism, DPO, social perception, user preference

## TL;DR

This paper proposes HumT, a metric for human-like tone based on GPT-2 log-probability ratios, and its social perception generalization SocioT. Analysis of over 400k preference samples reveals that users generally prefer LLM outputs with lower human-likeness. Furthermore, human-like tone strongly correlates with social closeness ($r=0.87$), low status ($r=-0.80$), and femininity ($r=0.47$). Finally, DPO fine-tuning (DumT) using only 500 preference pairs effectively reduces human-likeness without sacrificing model performance.

## Background & Motivation

**Background**: Current LLM product designs generally strive to make models "more human-like" by imparting personality, politeness, and a friendly tone (Bai et al. 2022). Consequently, user-facing LLMs extensively output human-like language such as "Happy to help!" or "I can imagine that feeling." Shneiderman criticized this "Humpty Dumpty syndrome" as early as 1993, arguing that imposing human characteristics on technology can mislead users.

**Limitations of Prior Work**:
   - **Lack of Measurement**: There is currently no systematic method to quantify the human-likeness of text. Existing approaches (such as pronoun detection or conversational filler matching) focus only on isolated linguistic features and fail to capture the holistic nature of "human-like tone" as a complex social construct, where features frequently co-occur and are difficult to disentangle.
   - **Unverified Core Assumption**: Do users indeed prefer human-like outputs? The industry consensus of "the more human-like, the better" has never been systematically examined on large-scale preference data.
   - **Unquantified Harms**: Potential risks of anthropomorphic LLMs (such as overtrust/dependence, emotional bonding, and reinforcement of gender stereotypes) have been widely discussed qualitatively, but quantitative measurement tools are lacking.

**Key Challenge**: "More human-like" is default-assumed to be "better," yet human-like language may be fundamentally insincere. Since LLMs lack emotion and consciousness, simulated empathy is inherently misleading. Moreover, conversational fillers and phatic expressions in human-like tones reduce information density.

**Goal**: Establish a complete closed-loop of "measure $\rightarrow$ understand $\rightarrow$ control" for human-like tone: (a) How to quantify the human-likeness of arbitrary text? (b) What do users truly prefer? (c) Which social perception dimensions correlate with human-like tone? (d) How to systematically reduce human-likeness?

**Key Insight**: Inspired by the implicit framing theory and the spoken-written continuum of Koch & Oesterreicher, the authors observe that comparing the conditional probability differences in language models between "He/She said s" and "It said s" can infer the human/non-human attribute of the implicit speaker—GPT-2's pre-training knowledge naturally encodes world knowledge about "what humans say."

**Core Idea**: Leverage a log-probability ratio metric based on pronoun animacy to achieve three goals at once: measure human-likeness, predict user preferences, and quantify social perception dimensions.

## Method

### Overall Architecture

This paper proposes three interconnected components that form a "measure $\rightarrow$ understand $\rightarrow$ control" pipeline:
- **HumT**: Given any text, it outputs a scalar score representing the degree of its human-like tone.
- **SocioT**: A generalization of HumT that measures four social perception dimensions (warmth, social status, social closeness, and gender) simply by replacing the set of prefix phrases.
- **DumT**: Standardizes preference pairs selected via HumT to systematically reduce human-likeness at the generation distribution level using DPO fine-tuning.

The input is an arbitrary text string $s$ (LLM output or other text), HumT/SocioT outputs dimensional scores, and DumT outputs a fine-tuned, low-human-likeness LLM.

### Key Designs

1. **HumT — Based on Pronoun Animacy**

    - **Function**: Given any text $s$, outputs a scalar score to quantify how much it "sounds like a human said it".
    - **Mechanism**: Prepends animate prefixes ("He said", "She said") and inanimate prefixes ("It said") to the text, respectively, and calculates the log-probability ratio using GPT-2: $T_D(s) = \log \frac{P_{D^+}(s)}{P_{D^-}(s)}$, where $D^+ = \{\text{He said, She said}\}$ and $D^- = \{\text{It said}\}$. $T_D(s) > 0$ indicates a more human-like response (e.g., "Hello!"), while $T_D(s) < 0$ indicates outputs more typical of non-human entities (e.g., code snippets). The probability computation for each text is repeated $n=100$ times and averaged to suppress noise, with text truncated to 300 characters.
    - **Design Motivation**: Traditional approaches rely on handcrafted feature lists (e.g., pronoun counts, filler matching). However, the constituent features of human-like tone often co-occur and are difficult to disentangle, requiring a metric that captures holistic distributional features. Using GPT-2 (rather than a larger model) is a deliberate choice to avoid introducing anthropomorphic biases injected during post-training phases. Compared to AnthroScore (Cheng et al. 2024, which is based on MLM), HumT is more generalizable: it does not require specifying a particular entity and scores arbitrary text directly based on the implied speaker.

2. **SocioT — Multidimensional Social Perception Measurement**

    - **Function**: Generalizes the HumT framework to four social perception dimensions: warmth, social status, social closeness, and gender.
    - **Mechanism**: Reuses the exact mathematical formula of HumT, replacing only the prefix phrase sets for $D^+$ and $D^-$. For example, the social closeness dimension compares "My friend/partner/husband/wife said" against "The stranger said"; the warmth dimension compares "The friend/lover/mentor/idol said" against "The stranger/enemy/examiner/dictator said"; the gender dimension compares "She said" against "He said"; and the status dimension compares "He commanded/proclaimed/demanded" against "He pleaded/mentioned/asked". The phrase sets are selected to ensure they match in topic and genre, differing only along the target dimension.
    - **Design Motivation**: The Stereotype Content Model (SCM, Fiske et al. 2002) in social psychology maps stereotypes into a two-dimensional space of warmth and competence. Previous research associates anthropomorphic LLMs with social closeness/warmth (which leads to overtrust) and femininity combined with low status (reinforcing gender stereotypes). SocioT translates these qualitative concerns into quantifiable metrics. Robustness checks show that removing phrases individually or in pairs does not affect the results.

3. **DumT — Controlling Human-Likeness with HumT+DPO**

    - **Function**: Systematically reduces the human-likeness of LLM outputs while maintaining or even improving model performance.
    - **Mechanism**: After deduplication and GPT-4 safety filtering on PRISM, UltraFeedback, and LMSys datasets, the data is split 90-10 into train/test sets. Pairs satisfying "$s$ is preferred by the user AND $\text{HumT}(s') - \text{HumT}(s) > 0$" (where $s'$ is the dispreferred response) are filtered from the training set. A DPO training set is constructed by randomly sampling $n=500$ pairs to fine-tune a Meta-Llama-3-8B-Instruct base model.
    - **Design Motivation**: Direct prompting (e.g., "Please respond in a non-human-like tone") yields poor empirical results—either dropping response quality or merely substituting individual words while leaving the overall tone unchanged, often resulting in unnatural paraphrases. DPO adjusts the preference direction at the generative distribution level, making it more systematic and effective.

### Loss & Training

- Base Model: Meta-Llama-3-8B-Instruct
- Training Data: Only 500 preference pairs (SHP is excluded due to non-LLM prompt formats, and HH-RLHF is excluded due to a high volume of unsafe content)
- Data Cleaning: Deduplication + GPT-4 moderation filter
- Training Framework: TRL (Transformer Reinforcement Learning)
- Computational Resources: 1 GPU + 1032GB RAM, 3 hours of training; HumT/SocioT calculation requires 1 GPU + 128GB RAM, taking < 10 GPU hours per dataset

## Key Experimental Results

### Main Results

**H1 Verification: Users prefer less human-like outputs**

Across 5 preference datasets with 400k+ samples:

| Dataset | Type | Sample Size | HumT Difference Direction | Significance |
|--------|------|--------|--------------|--------|
| SHP | RLHF Preference | 100K+ | Preferred response has lower HumT | $p < 0.001$ |
| HH-RLHF | RLHF Preference | 100K+ | Preferred response has lower HumT | $p < 0.001$ |
| UltraFeedback | RLHF Preference | 100K+ | Preferred response has lower HumT | $p < 0.001$ |
| PRISM | Real Users (75 countries, 1.5K people) | 14K+ | Largest difference, preferred response ~4% lower | $p < 0.001$ |
| LMSys | Real Users (14K IPs) | 14K+ | Lower HumT preferred within all topics | $p < 0.001$ |

**H2 Verification: Correlation between human-likeness and social perception dimensions**

| HumT vs SocioT Dimension | Pearson $r$ | Implication |
|---------------------|-----------|------|
| Social Closeness | **0.87** | Human-like tone is virtually equivalent to social closeness |
| Status | **-0.80** | Human-likeness $\leftrightarrow$ low-status language |
| Femininity | **0.47** | Human-like tone leans toward feminine expressions |
| Warmth | **0.45** | Human-likeness $\leftrightarrow$ warm/friendly tone |

All correlations are statistically significant at the $p < 0.001$ level after Benjamini-Hochberg correction for multiple comparisons.

### Ablation Study

| Model | Average HumT | RewardBench Overall | Chat | Chat Hard | Reasoning | Safety |
|------|----------|-----------------|------|-----------|-----------|--------|
| Base (Llama-3-8B-Instruct) | Highest | Baseline | Highest | Baseline | Baseline | Baseline |
| B_DPO-R (Random DPO) | Medium | $\approx$ DumT | — | — | — | — |
| **DumT** | **Lowest** ($p<0.001$) | **Outperforms Base** | Slightly lower | $\uparrow$ | $\uparrow$ | $\uparrow$ |
| MaxHumT (Maximized Human-likeness) | Highest/Higher | $\le 0.51$ (Collapsed) | — | — | — | — |

Human annotation evaluation (500 prompts $\times$ 3 Prolific annotators): DumT 40% vs. Base 36% vs. Tie 24%. The advantage of DumT is most pronounced on the PRISM subset (44% vs. 35%).

### Key Findings

- **DumT score drops on the Chat subset but rises on Chat Hard/Reasoning/Safety**: This occurs because the Chat subset implicitly rewards human-like tones (e.g., labeling "Sure, I can help!" as "chosen"), whereas in Math-PRM, 94% of incorrect answers contain the pronoun "I". This indicates that existing benchmarks conflate human-likeness with quality.
- **Topic Dependency**: Users prefer higher HumT in greeting scenarios (+3%), but significantly lower human-likeness in value-sensitive topics such as politics and religion.
- **Sanity Check**: The HumT ranking follows Human-written text > LLM output > Web data (C4), illustrating that the human-like tone of LLMs is primarily derived from post-training rather than pre-training.
- **User Demographics** (race, gender, age, LLM familiarity) show no statistically significant differences regarding preference directions.
- **Construct Validity**: Validated by 4 annotators on 600 text instances, yielding Fleiss' $\kappa > 0.6$ for human-like tone and social closeness, and $\kappa > 0.4$ for warmth and gender.

## Highlights & Insights

- **Counter-intuitive Core Discovery**: Systematically verified across 400k+ samples that users generally prefer LLM outputs that are "less human-like" — directly challenging the industry consensus that "more human-like is better." The cleverness lies in utilizing existing preference datasets rather than building new ones to prove this, showing that the signal has always existed in the data but was simply never measured.
- **Extremely Elegant Metric Design**: The core of the entire HumT methodology is a single one-line formula—the log-probability ratio—which runs efficiently on small models like GPT-2. Seamlessly scaling to various social perception dimensions (SocioT) simply by replacing prefix phrase sets, this "one framework covers all dimensions" modular design can be transferred to any scenario requiring measurements of implicit text attributes.
- **Quantifying the Societal Harms of Anthropomorphism**: Human-like language simultaneously exhibits high warmth, low status, and feminine characteristics, aligning precisely with the "warm but incompetent" quadrant in SCM. This represents the first study to quantitatively link the linguistic features of anthropomorphic LLMs to stereotype theories in social psychology.
- **Impressive Data Efficiency of DumT**: Armed with just 500 preference pairs and 3 hours of training, the model significantly reduces human-likeness without sacrificing performance. This makes the approach highly practical and reproducible.
- **Implications for Benchmark Design**: The Chat subset of RewardBench accounts for 1/4 of the overall score but implicitly rewards human-like tones, which potentially misleads systematic model optimization.

## Limitations & Future Work

- **Single Dimension**: Measures only "tone" at the linguistic level, leaving other dimensions of "human-likeness" (such as reasoning depth or creativity) unaddressed.
- **Cultural Limitations**: Based on English GPT-2 pre-training, which reflects WEIRD (Western, Educated, Industrialized, Rich, and Democratic) cultural norms; the acceptance of anthropomorphism may differ dramatically across cultures (e.g., East Asian cultures might show higher acceptance of anthropomorphic AI).
- **Scenario Dependency**: Scenarios like emotional support or psychological counseling might indeed require more human-like language (an exception already observed in the greetings scenario), demanding fine-grained adaptive control of scenes rather than global suppression.
- **Ceiling of GPT-2 as a Probability Model**: Larger models might provide better probability estimates, but using post-trained models risks introducing pre-existing anthropomorphic biases, leading to self-referential bias. Formulating a superior "neutral" probability model remains an open question.
- **Limited Coverage of DumT**: Only about 30% of test set outputs undergo a substantial HumT reduction (filtered using $\epsilon=0.02$). Exploring larger training sets or stronger control methods is a future direction.
- **Short-term Preferences vs. Long-term Impact**: Whether users' immediate preference for lower human-likeness translates to long-term optimality lacks validation from longitudinal studies.

## Related Work & Insights

- **vs. AnthroScore (Cheng et al. 2024)**: AnthroScore relies on MLMs to measure the anthropomorphism of specific entities, requiring explicit entity mentions in the sentence (e.g., "The AI"). HumT is more generalized—it is independent of explicit entity mentions and scores arbitrary text directly based on the implied speaker. Additionally, HumT leverages the larger GPT-2 instead of BERT-scale MLMs.
- **vs. Traditional RLHF/DPO Alignment**: Traditional RLHF optimizes for "helpful + harmless" (Bai et al. 2022). This paper opens up a new dimension: "appropriate degree of human-likeness" should also be an alignment target. It shows that even preference data not designed for this purpose already contains de-anthropomorphizing signals.
- **vs. Language Immediacy Theories (Koch & Oesterreicher 1985, Biber 1991)**: The intimacy/emotionality of spoken language versus the abstraction/impersonality of written language provides a theoretical foundation for HumT. This paper successfully translates linguistic theory into computable metrics.
- **Insights**: The default "friendly assistant" tone in commercial LLMs warrants re-evaluation. Adaptive tone control based on the scenario (e.g., suppressing human-likeness in technical Q&As while selectively retaining it in social settings) might be a superior product strategy. The HumT/SocioT frameworks can also be utilized to audit anthropomorphism levels during model training, serving as a new monitoring signal for alignment.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Deserves high praise for questioning the industry consensus of "more human-like is better." The pronoun-animacy-based log-probability ratio design of HumT is exceptionally concise with solid theoretical underpinnings.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Solid coverage with 400k+ samples across 5 preference datasets, construct validity with 4 annotators, LIWC analysis, DPO + RewardBench + human evaluation, though cross-lingual and cross-cultural validations are lacking.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Seamlessly integrates NLP, social psychology, and HCI. The logical pipeline of "measure $\rightarrow$ discover $\rightarrow$ control" is distinct, and the anthropomorphic continuum examples in Table 1 are highly persuasive.
- **Value**: ⭐⭐⭐⭐⭐ Delivers direct implications for LLM product design, alignment strategies, and benchmark evaluations. HumT serves as a highly practical audit tool.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] RULEBREAKERS: Challenging LLMs at the Crossroads between Formal Logic and Human-like Reasoning](../../ICML2025/llm_nlp/rulebreakers_challenging_llms_at_the_crossroads_between_formal_logic_and_human-l.md)
- [\[ACL 2025\] Enhancing Conversational Agents with Theory of Mind: Aligning Beliefs, Desires, and Intentions for Human-Like Interaction](enhancing_conversational_agents_with_theory_of_mind_aligning_beliefs_desires_and.md)
- [\[ACL 2025\] Controlling Politeness in Multi-Turn Dialogues Through Pre-Phrase Augmentation](controlling_politeness_in_multi-turn_dialogues_through_pre-phrase_augmentation.md)
- [\[ACL 2025\] ConSim: Measuring Concept-Based Explanations' Effectiveness with Automated Simulatability](consim_measuring_concept-based_explanations_effectiveness_with_automated_simulat.md)
- [\[ACL 2025\] Beware of Your Po! Measuring and Mitigating AI Safety Risks in Role-Play Fine-Tuning of LLMs](sarft_roleplay_safety.md)

</div>

<!-- RELATED:END -->
