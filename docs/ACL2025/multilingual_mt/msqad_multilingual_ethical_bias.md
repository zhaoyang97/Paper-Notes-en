---
title: >-
  [Paper Note] Delving into Multilingual Ethical Bias: The MSQAD with Statistical Hypothesis Tests for Large Language Models
description: >-
  [ACL 2025 (Long Paper)][Multilingual & Machine Translation][Multilingual Bias] Proposes a multilingual sensitive question-answering dataset, MSQAD (based on 17 human rights topics from Human Rights Watch across 6 languages), and systematically demonstrates through two statistical hypothesis tests (McNemar's test and PERMANOVA) that LLMs exhibit significant ethical bias when answering the same sensitive questions in different languages: Chinese and Hindi show the highest refus…
tags:
  - "ACL 2025 (Long Paper)"
  - "Multilingual & Machine Translation"
  - "Multilingual Bias"
  - "Ethical Bias"
  - "Statistical Hypothesis Testing"
  - "LLM safety"
  - "Cross-lingual"
date: 2026-05-08
content_hash: e0e05088bc083b37
---

# Delving into Multilingual Ethical Bias: The MSQAD with Statistical Hypothesis Tests for Large Language Models

**Conference**: ACL 2025 (Long Paper)  
**arXiv**: [2505.19121](https://arxiv.org/abs/2505.19121)  
**Code**: [https://github.com/seungukyu/MSQAD](https://github.com/seungukyu/MSQAD)  
**Area**: AI Safety / LLM Bias / Multilingual Analysis  
**Keywords**: Multilingual Bias, Ethical Bias, Statistical Hypothesis Testing, LLM safety, Cross-lingual

## TL;DR

Proposes a multilingual sensitive question-answering dataset, MSQAD (based on 17 human rights topics from Human Rights Watch across 6 languages), and systematically demonstrates through two statistical hypothesis tests (McNemar's test and PERMANOVA) that LLMs exhibit significant ethical bias when answering the same sensitive questions in different languages: Chinese and Hindi show the highest refusal rates, whereas Spanish and German are most prone to generating inappropriate responses, a bias widely observed across 7 LLMs.

## Background & Motivation

- **Background**: LLM training corpora are English-centric, with highly unbalanced distributions across languages. Language is naturally and tightly coupled with culture — corpora in specific languages inherently reflect the cultural characteristics behind them. Consequently, when LLMs face globally sensitive topics, responses in different languages can exhibit systematic differences.
- **Limitations of Prior Work**: Existing bias research primarily focuses on English bias detection for specific groups (gender, race, sexual orientation), with benchmarks like CrowS-Pairs and StereoSet employing fill-in-the-blank paradigms, which lack systematic validation from a cross-lingual perspective. Multilingual studies also mostly focus on performance improvements in general tasks, neglecting language-specific biases from a social/cultural perspective.
- **Key Challenge**: For sensitive questions with identical semantics, merely changing the language of the prompt causes significant differences in the morality and informativeness of model responses. However, there is currently a lack of standardized datasets and statistical testing frameworks to quantify such cross-lingual ethical biases.
- **Goal**: To construct a multilingual question-answering dataset covering globally sensitive topics, and to quantify ethical biases between different languages through rigorous statistical hypothesis tests, validating whether this bias is prevalent across multiple LLMs.
- **Key Insight**: Generating sensitive questions based on 17 human rights topics from Human Rights Watch, expanding them to 6 languages, and designing two complementary statistical tests for refusal behaviors (discrete) and content distributions (continuous) respectively.
- **Core Idea**: Utilizing McNemar's test and PERMANOVA test as two statistical hypothesis testing methods to verify that, under controlled variables, the null hypothesis ("merely changing the language should not lead to differences in responses") is rejected across almost all language pairs and model combinations.

## Method

### Overall Architecture

The entire pipeline consists of three stages: (1) **Data Collection & Question Generation** — Crawling news on 17 human rights topics from Human Rights Watch, utilizing GPT-4 with an intermediate keyword generation task to yield sensitive questions, and filtering them via K-means clustering de-duplication; (2) **Multilingual Response Generation** — Translating questions into 6 languages (English, Korean, Chinese, Spanish, German, Hindi) via Google Cloud Translation, and utilizing GPT-3.5 to generate acceptable and non-acceptable responses respectively; (3) **Statistical Hypothesis Testing** — Applying McNemar's test on the refusal rates of non-acceptable responses, and PERMANOVA on the embedding distribution of acceptable responses, to conduct cross-lingual and cross-model comparisons.

Core experimental design: The independent variable is solely the language, controlled variables include prompt structure, translation services, and PLM, and dependent variables are refusal rates and response distributions. The null hypothesis is "solely changing the language should not cause differences in responses."

### Key Designs

**Module 1: MSQAD Dataset Construction Pipeline**

- **Function**: Automatically constructing a sensitive QA dataset covering 17 topics × 6 languages from global human rights news.
- **Mechanism**: Crawling HRW news first $\rightarrow$ introducing an **intermediate keyword generation task** (GPT-4 first infers keywords from the news and then generates questions based on those keywords to avoid over-reliance on raw news texts) $\rightarrow$ clustering for de-duplication (multilingual BERT embeddings + K-means, removing redundant questions with similarity >97% to the centroid) $\rightarrow$ translating to 6 languages via Google Translation $\rightarrow$ generating acceptable/non-acceptable responses using GPT-3.5 (integrating jailbreak techniques for non-acceptable responses to bypass safety guidelines).
- **Design Motivation**: Directly generating questions from news easily creates excessive duplicates (due to seasonal topics). Introducing the intermediate keyword step + clustering de-duplication ensures question diversity. Generating non-acceptable responses requires jailbreaking; otherwise, models refuse. Translation quality is validated using GEMBA scores (all 4 metrics > 93), and human annotation achieves a Krippendorff's $\alpha$ of 0.61-0.72.

**Module 2: McNemar's Test — Refusal Rate Difference Detection**

- **Function**: Testing whether the probability of LLMs refusing to generate inappropriate content is consistent across different language pairs.
- **Mechanism**: Constructing a $2 \times 2$ contingency table for each language pair (Language A Refusal/Non-refusal $\times$ Language B Refusal/Non-refusal), and computing $\chi^2_{\text{McNemar}} = (b-c)^2/(b+c)$, where the critical value is $3.838$ at a $5\%$ significance level. Refusal determination is conducted via zero-shot classification using a fine-tuned multilingual mDeBERTa (trained on the XNLI dataset) with labels "discuss {topic}" vs "refuse to answer", complemented by a probability threshold of $0.8$ and direct refusal pattern filtering.
- **Design Motivation**: McNemar's test is specifically designed for detecting differences in paired binary data. This perfectly fits the scenario of "whether the same question is refused in two different languages" and is a widely utilized statistical test in NLP.

**Module 3: PERMANOVA Test — Response Distribution Difference Detection**

- **Function**: Testing whether the embedding distributions of acceptable responses are similar across different language pairs.
- **Mechanism**: Extracting response embeddings using multilingual BERT $\rightarrow$ constructing a Euclidean distance matrix $D$ $\rightarrow$ computing the between-group sum of squares $SS_{\text{each}}$ and within-group sum of squares $SS_{\text{within}}$ $\rightarrow$ obtaining the raw test statistic via the F-statistic $F = (SS_{\text{each}} - SS_{\text{within}}) / (SS_{\text{within}} / (2n-2))$ $\rightarrow$ conducting $P$ permutation tests (randomly shuffling group labels) to compute the p-value.
- **Design Motivation**: Complementary to McNemar's test — McNemar's test detects discrete refusal behaviors, while PERMANOVA detects continuous differences in content distributions. Permutation testing does not rely on distributional assumptions and is well-suited for high-dimensional data in embedding spaces.

## Key Experimental Results

### Table 1: Cross-lingual Statistical Testing Results (GPT-3.5-turbo)

| Testing Method | Null Hypothesis Acceptance Ratio | Total Language Pairs | Key Findings |
|:---------|:----------------|:----------|:---------|
| McNemar (Refusal Rate Difference) | 4.31% (11/255) | 15 pairs × 17 topics = 255 | 95.69% of language pairs reject the null hypothesis at a 5% signicance level |
| PERMANOVA (Response Distribution Difference) | ≈0% (all significance levels) | 15 pairs × 17 topics | The null hypothesis is almost entirely rejected even at a 0.1% significance level |

**Refusal Rate Ranking** (highest to lowest): Hindi > Chinese > Korean > English > Spanish > German. Chinese and Hindi yield the highest refusal rates (the model tends to refuse to generate inappropriate content), while Spanish and German exhibit the lowest refusal rates (most prone to generating inappropriate responses). Human annotations show that English responses have the highest selection rate for "ethically informative" category (Children's Rights 47.5%, Refugees 47.5%, Women's Rights 62.5%), whereas Chinese and Hindi are extremely low (0%–1.25%).

### Table 2: Cross-model Validation Results (6 Additional LLMs)

| Model | Parameters | Bias Characteristics (McNemar) | Bias Characteristics (PERMANOVA) |
|:-----|:------|:-------------------|:--------------------|
| Gemma-7B | 7B | English vs others exhibits highest bias | English response distribution is least similar to other languages |
| Llama-2-7B-chat | 7B | Relatively high refusal rate for English | Significant distributional differences in Korean vs other languages |
| Llama-3-8B-Instruct | 8B | Greater bias than Llama-2, especially for Spanish/German | Korean bias decreases, while English bias persists |
| Mistral-7B-v0.2 | 7B | English and Chinese prone to generating inappropriate content | Obvious bias in the informativeness of English responses |
| Phi-3-mini-4k | 3.8B | Bias is inevitable despite small size, especially on Women's Rights + Korean | Significant across all language pairs |
| Qwen-1.5-7B-Chat | 7B | Hindi bias is most prominent | English distribution is least similar to other languages |

**Key Findings**: Bias increases rather than decreases from Llama-2 to Llama-3; all 7 models (including GPT-3.5) exhibit cross-lingual bias; {Chinese, Hindi} vs {Spanish, German} consistently remains the strongest biased pair; the topics of Children's Rights and LGBT Rights exhibit particularly pronounced differences in embedding distributions within Chinese responses.

## Highlights & Insights

- **Dual-Testing Complementary Framework**: McNemar's test evaluates refusal behaviors (discrete), while PERMANOVA assesses content distributions (continuous). The two complement each other to cover different dimensions of ethical bias, offering higher reliability than single metrics.
- **Rigorous Controlled Variable Design**: Clearly distinguishes independent variables (language), controlled variables (prompt structure/translation/PLM), and dependent variables (refusal rate/distribution), precluding confounding factor interference.
- **Counter-intuitive Finding**: Llama-3, which undergoes stronger safety training, unexpectedly shows more severe cross-lingual bias than Llama-2 — safety alignment may exacerbate rather than eliminate cross-lingual bias in certain dimensions.
- **Practical Deployment Value**: Spanish and German users are systematically more likely to encounter inappropriate responses on sensitive topics, directly impacting safety evaluation strategies for multilingual LLMs.

## Limitations & Future Work

- Only covers 6 languages; lacks important languages such as Arabic, Japanese, and French, with low-resource languages entirely unaddressed.
- The dataset is automatically generated by GPT-4/3.5, which may inherit biases from the GPT series itself.
- Only uses embedding distributions to indirectly reflect differences in response quality, without directly analyzing the quality dimensions of semantic content.
- Google Translation itself may introduce systematic errors, particularly concerning the translation quality of languages like Hindi.
- Cross-model validation is concentrated on 7B-class models, lacking validation on 70B+ and commercial models (e.g., GPT-4, Claude).
- Fails to discuss the concrete impact of cross-lingual bias on practical downstream tasks (e.g., multilingual customer service, content moderation).

## Related Work & Insights

- **SQuARe (Lee et al., ACL 2023)**: Only covers sensitive QA construction in Korean. MSQAD extends this to 6 languages and introduces a statistical testing framework.
- **CrowS-Pairs / StereoSet**: Focus on fill-in-the-blank measurements for English stereotypes. MSQAD investigates cross-lingual response differences in open-ended QA.
- **Lee et al. (NAACL 2024)**: Analyzes cross-cultural differences in hate speech but does not involve statistical testing of LLM response biases.
- **Insights**: The statistical testing methodology using McNemar's test and PERMANOVA can be transferred to detect biases across prompt styles and model versions. The MSQAD construction pipeline (news $\rightarrow$ keywords $\rightarrow$ questions $\rightarrow$ clustering de-duplication $\rightarrow$ multilingual expansion) can serve as a general paradigm for constructing multilingual benchmarks.

## Rating

- Novelty: ⭐⭐⭐⭐ The combined perspective of cross-lingual ethical bias and statistical testing is relatively novel, though McNemar's test and PERMANOVA are classical tools.
- Experimental Thoroughness: ⭐⭐⭐⭐ Broad coverage of 17 topics × 6 languages × 7 models with human annotation validation, though lacking large parameter models.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, excellent visualization of variable relations and heat maps, and highly detailed appendices.
- Value: ⭐⭐⭐⭐ The dataset and framework are of practical value for multilingual LLM safety evaluations; the counter-intuitive findings are inspiring.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] 7 Points to Tsinghua but 10 Points to 清华? Assessing Agentic Large Language Models in Multilingual National Bias](assessing_agentic_large_language_models_in_multilingual_national_bias.md)
- [\[ACL 2025\] Disentangling Language and Culture for Evaluating Multilingual Large Language Models](disentangle_language_culture.md)
- [\[ACL 2025\] Just Go Parallel: Improving the Multilingual Capabilities of Large Language Models](just_go_parallel_improving_the_multilingual_capabilities_of_large_language_model.md)
- [\[ACL 2025\] Cross-Lingual Optimization for Language Transfer in Large Language Models](cross-lingual_optimization_for_language_transfer_in_large_language_models.md)
- [\[ACL 2025\] Marco-Bench-MIF: On Multilingual Instruction-Following Capability of Large Language Models](marco_bench_multilingual_if.md)

</div>

<!-- RELATED:END -->
