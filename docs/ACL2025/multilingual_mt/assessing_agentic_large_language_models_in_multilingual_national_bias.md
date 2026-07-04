---
title: >-
  [Paper Note] 7 Points to Tsinghua but 10 Points to Tsinghua? Assessing Agentic Large Language Models in Multilingual National Bias
description: >-
  [ACL 2025 Findings][Multilingual & Machine Translation][Multilingual Bias] This paper presents the first systematic study of national bias in LLMs acting as multilingual recommendation agents in reasoning-based decision-making tasks. Utilizing three scenarios (university application, travel, and relocation) alongside the Thurstone Case III (comparative judgment) method, the study quantifies rating discrepancies for GPT-3.5, GPT-4, and Claude Sonnet across six languages. The find…
tags:
  - "ACL 2025 Findings"
  - "Multilingual & Machine Translation"
  - "Multilingual Bias"
  - "National Bias"
  - "LLM Agent"
  - "Decision Reasoning"
  - "Chain-of-Thought"
  - "Fairness"
date: 2026-05-08
content_hash: 7c91fd3a05cde1c1
---

# 7 Points to Tsinghua but 10 Points to Tsinghua? Assessing Agentic Large Language Models in Multilingual National Bias

**Conference**: ACL 2025 Findings  
**arXiv**: [2502.17945](https://arxiv.org/abs/2502.17945)  
**Code**: [GitHub](https://github.com/yiyunya/assess_agentic_national_bias)  
**Area**: Multilingual Translation  
**Keywords**: Multilingual Bias, National Bias, LLM Agent, Decision Reasoning, Chain-of-Thought, Fairness  

## TL;DR
This paper presents the first systematic study of national bias in LLMs acting as multilingual recommendation agents in reasoning-based decision-making tasks. Utilizing three scenarios (university application, travel, and relocation) alongside the Thurstone Case III (comparative judgment) method, the study quantifies rating discrepancies for GPT-3.5, GPT-4, and Claude Sonnet across six languages. The findings reveal a widespread prevalence of "local language bias," and demonstrate that Chain-of-Thought (CoT) reasoning paradoxically exacerbates bias in non-English languages.

## Background & Motivation

**Background**: LLMs have been widely deployed as multilingual intelligent assistants to provide personalized recommendations (such as university application guidance, travel planning, and career development) to global users. Their reasoning capabilities increasingly position them to assume the role of "decision-making agents."

**Limitations of Prior Work**: Existing research on LLM bias primarily focuses on word-level bias detection (e.g., sentiment polarity of adjectives, stereotypical descriptions) and monolingual environments. There is a lack of research on whether LLMs exhibit systemic biases in **cross-lingual reasoning-based decision-making tasks**—specifically, whether asking the same question in different languages yields systematically different recommendations.

**Key Phenomenon**: As shown in Figure 1 of the paper, when ChatGPT is asked about Tsinghua University in English, it scores it 7/10, whereas the query in Chinese yields a perfect score of 10/10, with the Chinese response conspicuously downplaying any drawbacks. This prominent cross-lingual inconsistency reveals a deep-seated **multilingual national bias**.

**Goal**: This study aims to fill this gap by systematically magnifying and quantifying the patterns of national bias of state-of-the-art LLMs in multilingual reasoning and decision-making, and exploring how demographic factors (gender) and reasoning strategies (CoT) influence such biases.

## Method

### Overall Architecture
The latent national bias of LLMs is formulated as a "comprehensive evaluation problem." Across three real-world recommendation scenarios (university applications, urban relocation, and travel recommendations), the authors leverage the Thurstone case III comparative judgment method from psychophysics to construct standardized triplet options. LLMs are prompted in six different languages to assume a professional consultant persona to score and analyze the options. The cross-lingual rating discrepancies are subsequently quantified using Jensen-Shannon Divergence (JSD) and Mean Divergence (MD).

### Key Designs

1. **Triplet Construction and Evaluation Framework**

    - **Sources of Options**: Universities are sourced from the QS 2024 Top 100/200 rankings, urban relocations from 2022 GDP data (City Population), and travel destinations from the Euromonitor 2023 Top 100 City Destinations Index.
    - **Triplet Design**: Each triplet contains 1 target option + 2 comparative options (ensuring 1 English-speaking country + 1 non-English-speaking country), totaling 100 fixed evaluation sets, which are reused across all target options to ensure fairness.
    - **Country Coverage**: English-speaking countries (US/UK/CA/AU), single dominant language countries (CN/JP/FR/DE/KR), multilingual countries (HK/SG/CH), and representatives from the Global South.
    - **Rating Paradigm**: LLMs conduct a comprehensive pros-and-cons analysis for each option within the triplet, providing a score on a 10-point scale along with reasoning.

2. **Persona-based Prompt Design**

    - Persona prompts are designed for each scenario: academic planning consultant, career relocation consultant, and travel planner.
    - Detailed contextual information (e.g., a high school student applying for an undergraduate program) and output formatting constraints are provided.
    - All prompts are faithfully translated into six target languages (EN/JA/ZH/FR/DE/KO) to ensure semantic consistency.
    - It is emphasized that the models should not simply copy templates but should provide formal advice like real-world consultants.

3. **Bias Quantification Metrics**

    - **JSD (Jensen-Shannon Divergence)**: Computes the divergence of the score distribution of each language from the global distribution to measure overall language-level bias, where higher values indicate more severe bias.
    - **MD (Mean Divergence) Score**: $\mu_{\text{local}} - \mu_{\text{global}}$, specifically designed to capture "local language bias"—notably, whether scoring in a specific country's language yields systematically higher ratings for that country.
    - **Robustness Testing**: Evaluated across two dimensions: CoT vs. No-CoT, and male persona vs. female persona.

## Experiments

### Main Results: JSD Cross-lingual Bias Scores

| Task/Model | EN | JA | ZH | FR | DE | KO | Overall |
|-----------|-----|-----|-----|-----|-----|-----|---------|
| **University Application** | | | | | | | |
| GPT-3.5 | 0.37 | 0.39 | 0.41 | **0.58** | 0.39 | 0.33 | 0.41 |
| GPT-4 | **0.28** | 0.30 | 0.35 | 0.32 | 0.42 | 0.35 | 0.33 |
| Sonnet | 0.38 | 0.33 | **0.50** | 0.40 | 0.29 | 0.36 | 0.38 |
| **Urban Relocation** | | | | | | | |
| GPT-3.5 | 0.38 | 0.42 | 0.31 | 0.46 | 0.35 | 0.32 | 0.37 |
| GPT-4 | 0.34 | 0.35 | 0.43 | 0.40 | **0.52** | 0.35 | 0.40 |
| Sonnet | 0.37 | 0.32 | **0.60** | 0.33 | 0.34 | 0.36 | 0.39 |
| **Travel Recommendation** | | | | | | | |
| GPT-3.5 | **0.56** | 0.48 | 0.43 | 0.51 | 0.42 | 0.46 | 0.48 |
| GPT-4 | 0.33 | 0.36 | 0.43 | 0.44 | 0.41 | 0.31 | 0.38 |
| Sonnet | 0.47 | 0.36 | **0.55** | 0.42 | 0.42 | 0.40 | 0.44 |

### Ablation Study: MD Bias Analysis of CoT and Gender Factors (University Application Task)

| Factor | US | UK | CA | AU | CN | JP | FR | DE | KR |
|------|------|------|------|------|------|------|------|------|------|
| **GPT-3.5** | | | | | | | | | |
| +CoT | 0.27 | 0.16 | 0.19 | 0.12 | **0.68** | 0.29 | 0.49 | 0.33 | 0.51 |
| -CoT | 0.49 | 0.36 | 0.12 | 0.18 | 0.19 | 0.21 | 0.15 | 0.30 | 0.38 |
| Female | 0.22 | 0.12 | 0.20 | -0.11 | 0.48 | 0.19 | 0.30 | 0.41 | 0.65 |
| Male | 0.19 | 0.22 | 0.40 | -0.06 | 0.46 | 0.12 | 0.33 | -0.03 | 0.30 |
| **GPT-4** | | | | | | | | | |
| +CoT | 0.01 | -0.03 | 0.12 | 0.03 | **0.52** | 0.17 | 0.26 | 0.27 | 0.33 |
| -CoT | -0.22 | -0.24 | 0.41 | 0.24 | **0.54** | 0.46 | 0.10 | 0.03 | 0.09 |
| **Sonnet** | | | | | | | | | |
| +CoT | 0.14 | 0.04 | -0.12 | 0.07 | **0.47** | 0.52 | -0.01 | 0.15 | 0.48 |
| Female | 0.16 | 0.11 | 0.06 | 0.10 | **0.56** | 0.52 | 0.10 | 0.27 | 0.54 |
| Male | 0.11 | 0.03 | 0.05 | 0.07 | **0.45** | 0.49 | -0.12 | 0.14 | 0.31 |

### Key Findings

1. **Widespread Local Language Bias**: When queried in a country's native language, LLMs systematically assign higher ratings to that country. China (CN) consistently exhibits the strongest local language bias across all models and conditions (MD 0.39-0.68), with East Asian countries (CN/JP/KR) displaying higher overall bias than English-speaking countries.
2. **GPT-4 Shines in English but Remains Highly Biased in Non-English Languages**: GPT-4 achieves the lowest JSD in English (0.28), but its overall JSD is not always the lowest—indeed, in the relocation task, it exceeds that of GPT-3.5 (0.40 vs. 0.37), indicating that alignment techniques predominantly benefit English.
3. **CoT Exacerbates Bias in Non-English Languages**: In GPT-3.5, incorporating CoT causes the MD for China to soar from 0.19 to 0.68, and for France from 0.15 to 0.49. This counterintuitive finding suggests that while CoT may better align with Western fairness norms in English, it reinforces cultural specificity in non-English contexts.
4. **Gender Interaction Effects**: GPT-4 and GPT-3.5 demonstrate significant differences in gender bias in South Korea (KR) (female persona MD = 0.65-0.73 vs. male 0.30-0.75), whereas Sonnet exhibits the weakest gender bias overall.

## Highlights & Insights
- **Pioneering Study**: The first to systematically quantify national bias in LLMs during multilingual reasoning and decision-making; the title "7 Points to Tsinghua but 10 Points to Qinghua" is highly engaging and communicative.
- **Exquisite Experimental Design**: Triplet design based on the Thurstone case III method combined with dual metrics (JSD/MD), alongside a rigorous and reproducible multi-dimensional cross-analysis of $ \text{CoT} \times \text{Gender} \times 3 \text{ Tasks} \times 6 \text{ Languages} $.
- **Counterintuitive Core Finding**: CoT reasoning does not mitigate bias but instead exacerbates it, thereby challenging the intuitive assumption that "more reasoning leads to more fairness."
- **Practical Cautionary Value**: Shines a light on fairness risks in multilingual AI applications, providing direct cautionary implications for the deployment of LLMs in education recommendations, travel planning, and other real-world domains.

## Limitations & Future Work
- Relies solely on three commercial closed-source models (GPT-3.5, GPT-4, and Claude Sonnet); opaque training data prevents diagnostic tracing of the root causes of bias.
- Only covers six high-resource languages; low-resource languages (e.g., Arabic, Hindi) may exhibit different bias patterns.
- Restricts investigation to three decision-making scenarios; bias patterns in other domains (such as medical or legal advice) remain unexplored.
- Remains at the stage of descriptive analysis, without proposing concrete bias mitigation methods.
- Scoring inherently suffers from subjectivity—different cultural backgrounds naturally possess different standards for what makes a "good university," meaning some identified "biases" might reflect justifiable cultural variations.

## Related Work & Insights
- **vs. Narayanan Venkit et al. (2023)**: Only investigated lexical-level national bias of adjectives in English, whereas this study probes into the reasoning and decision-making level.
- **vs. Durmus et al. (2023)**: Simulates multilingual survey respondents with LLMs answering binary choice questions, whereas our triplet scoring framework provides a finer-grained quantification of bias.
- **vs. Armstrong et al. (2024)**: Investigated biases in hiring agents but restricted to English, while this study expands the scope to six languages.
- **vs. Zhu et al. (2024)**: Investigated ChatGPT's national bias within Chinese scenarios, whereas this work scales up to a systematic analysis across six languages and three tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ The first to analyze national bias in multilingual reasoning and decision-making, offering a fresh perspective with an eye-catching title.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive analysis spanning 3 models $\times$ 3 tasks $\times$ 6 languages under multiple conditions.
- Writing Quality: ⭐⭐⭐⭐ Clearly structured with high-quality visualizations (violin plots).
- Value: ⭐⭐⭐ Highly valuable for understanding LLM fairness and multilingual alignment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Delving into Multilingual Ethical Bias: The MSQAD with Statistical Hypothesis Tests for Large Language Models](msqad_multilingual_ethical_bias.md)
- [\[ACL 2025\] Disentangling Language and Culture for Evaluating Multilingual Large Language Models](disentangle_language_culture.md)
- [\[ACL 2025\] X-WebAgentBench: A Multilingual Interactive Web Benchmark for Evaluating Global Agentic System](x-webagentbench_a_multilingual_interactive_web_benchmark_for_evaluating_global_a.md)
- [\[ACL 2025\] Cross-Lingual Auto Evaluation for Assessing Multilingual LLMs](cross-lingual_auto_evaluation_for_assessing_multilingual_llms.md)
- [\[ACL 2025\] Just Go Parallel: Improving the Multilingual Capabilities of Large Language Models](just_go_parallel_improving_the_multilingual_capabilities_of_large_language_model.md)

</div>

<!-- RELATED:END -->
