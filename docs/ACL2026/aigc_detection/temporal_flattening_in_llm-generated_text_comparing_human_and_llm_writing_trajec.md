---
title: >-
  [Paper Note] Temporal Flattening in LLM-Generated Text: Comparing Human and LLM Writing Trajectories
description: >-
  [ACL 2026][AIGC Detection][temporal flattening] This paper constructs a longitudinal writing dataset spanning 12 years and discovers that LLM-generated text exhibits "temporal flattening"—while lexical diversity is high…
tags:
  - "ACL 2026"
  - "AIGC Detection"
  - "temporal flattening"
  - "LLM-generated text detection"
  - "longitudinal writing analysis"
  - "cognitive-emotional features"
  - "synthetic data"
date: 2026-05-08
content_hash: 04efed9be34814ba
---

# Temporal Flattening in LLM-Generated Text: Comparing Human and LLM Writing Trajectories

**Conference**: ACL 2026  
**arXiv**: [2604.12097](https://arxiv.org/abs/2604.12097)  
**Code**: [GitHub](https://github.com/yjkim717/Cognitive-Emotional-Trajectories)  
**Area**: AIGC Detection  
**Keywords**: temporal flattening, LLM-generated text detection, longitudinal writing analysis, cognitive-emotional features, synthetic data

## TL;DR

This paper constructs a longitudinal writing dataset spanning 12 years and discovers "temporal flattening" in LLM-generated text—while lexical diversity is high, temporal drift in semantic and cognitive-emotional dimensions is significantly lower than human writing, achieving 94% accuracy in distinguishing human vs. LLM text using temporal variation patterns alone.

## Background & Motivation

**Background**: Large language models are widely used in content generation, dialogue systems, and synthetic training data production. Current LLM deployment paradigms are stateless—each generation is an independent response without retaining historical memory.

**Limitations of Prior Work**: Human writing is inherently longitudinal—an author's style, cognitive state, and emotional expression naturally evolve over time. However, existing LLM generation paradigms assume that independently sampled documents can adequately approximate human writing distributions, an assumption that has never been systematically verified.

**Key Challenge**: LLMs perform excellently on static quality metrics (e.g., semantic preservation, fluency), but may systematically lose the inherent longitudinal structure of human writing in the temporal dimension. This loss poses risks for downstream applications requiring temporal consistency (e.g., synthetic training data, authorship attribution, mental health trajectory modeling).

**Goal**: Answer two core questions—(1) Can LLMs reproduce human temporal structure over long time spans? (2) If differences exist, which temporal dynamics of human writing are systematically lost or flattened?

**Key Insight**: The authors approach from psycholinguistics and computational stylistics, treating writing as a longitudinal process and quantifying temporal structure through drift and variance—two complementary metrics across semantic, lexical, and cognitive-emotional representation spaces.

**Core Idea**: LLM-generated text exhibits "temporal flattening"—showing high diversity in lexical space but significantly lower temporal drift and fluctuation in semantic and cognitive-emotional spaces compared to humans, persisting under both instance-wise and history-augmented conditions.

## Method

### Overall Architecture

The study constructs a longitudinal dataset of 412 human authors with 6,086 documents (2012–2024), covering academic abstracts, blogs, and news. For each human author, matching writing trajectories are generated using DeepSeek V1, GPT-4o mini, and Claude 3.5 Haiku, producing 103,459 LLM documents total. Temporal dynamics differences between humans and LLMs are quantified by computing drift and variance metrics across three representation spaces (lexical, semantic, cognitive-emotional), validated through statistical tests and classifiers.

### Key Designs

1. **Three-Dimensional Feature Trajectory Construction**:

    - Function: Capture temporal evolution of writing from multiple complementary dimensions
    - Mechanism: For each author at each year, compute three representations—lexical (TF-IDF + SVD reduced to 10D), semantic (SBERT 384D embeddings), and cognitive-emotional features (20 interpretable psychological features including Big Five personality proxies, affect, readability, etc.). Annual feature vectors are concatenated to form temporal trajectories $\mathcal{T}(e) = (\mathbf{x}_1^{(e)}, \ldots, \mathbf{x}_T^{(e)})$
    - Design Motivation: Lexical features capture surface-level word usage changes, semantic features capture deep meaning shifts, and cognitive-emotional features capture psychological and stylistic evolution—the three are complementary

2. **Drift-Variance Dual Metric System**:

    - Function: Quantify temporal evolution from both global geometric displacement and local fluctuation perspectives
    - Mechanism: Drift measures total displacement between adjacent years' feature vectors via L2 distance $\text{drift}_t^{(e)} = \|\mathbf{x}_{t+1}^{(e)} - \mathbf{x}_t^{(e)}\|_2$; variance quantifies inter-annual fluctuation irregularity via coefficient of variation (CV) $\mathrm{CV}(f) = \frac{\mathrm{std}(\Delta f_{1:T})}{\mathrm{mean}(\Delta f_{1:T})}$
    - Design Motivation: Drift reflects "how far" writing has moved in representation space, variance reflects "how irregularly" it has moved—together they characterize temporal dynamics

3. **Paired Statistical Tests and Predictive Probes**:

    - Function: Rigorously validate systematic differences between human and LLM temporal dynamics
    - Mechanism: Binomial tests on each matched human-LLM author pair to test whether the proportion of human evolution exceeding LLM is significantly above 50%; random forest classifier trained on 20D cognitive-emotional CV features with GroupKFold cross-validation to prevent data leakage
    - Design Motivation: Statistical tests ensure conclusion reliability; classifiers verify predictability of differences

### Loss & Training

This is an analytical study with no model training involved. The classifier uses 5-fold GroupKFold cross-validation (grouped by author_id), ensuring all trajectories from the same author never appear in both training and test sets simultaneously. Benjamini-Hochberg FDR correction ($q < 0.05$) is applied for multiple comparisons.

## Key Experimental Results

### Main Results

| Representation Space | Metric | Human Win Rate Range | p-value | Meaning |
|---------------------|--------|---------------------|---------|---------|
| TF-IDF (Lexical) | Drift | 0.20–0.33 | p=1.0 | LLM lexical drift is larger |
| SBERT (Semantic) | Drift | 0.75–0.86 | p<0.0001 | Human semantic drift far exceeds LLM |
| Cog-Emo (Cognitive-Emotional) | Drift | 0.76–0.99 | p<0.0001 | Human cognitive-emotional drift is extremely significantly higher than LLM |

### Ablation Study

| Config | Accuracy | AUC | F1 | Note |
|--------|---------|------|-----|------|
| Pooled (IW) | 0.936 | 0.977 | 0.863 | Instance-wise condition |
| Pooled (Hist) | 0.933 | 0.977 | 0.856 | History-augmented condition |
| Balanced-Claude 3.5 (IW) | 0.97 | 1.00 | 0.97 | Claude easiest to distinguish |
| Balanced-GPT-4o mini (IW) | 1.00 | 1.00 | 1.00 | GPT-4o mini nearly perfectly distinguishable |

### Key Findings

- LLMs exhibit "asymmetric optimization": high lexical diversity but low semantic and cognitive-emotional drift, indicating LLMs optimize surface-level variation without reproducing deep-level evolution
- GPT-4o mini shows the most severe cognitive-emotional flattening: human Cog-Emo drift is larger in 97% of author pairs, rising to 99% under history-augmented conditions
- Temporal flattening persists under both instance-wise and history-augmented conditions, indicating this is a systemic property of current deployment paradigms
- Most discriminative features: average sentence length (18.7%), agreeableness (15.9%), and neuroticism (9.3%)

## Highlights & Insights

- Proposes the novel concept of "temporal flattening," revealing systematic deficiencies of LLM-generated text in the temporal dimension
- Achieves 94% classification accuracy and 98% AUC using only 20D cognitive-emotional CV features, providing a new approach for AIGC detection
- Publicly releases a longitudinal dataset of 412 authors, providing a foundation for future research
- Finds that history augmentation cannot fix temporal flattening, suggesting the root cause lies in model architecture or training paradigms

## Limitations & Future Work

- The dataset primarily covers English writing; temporal flattening across languages remains to be verified
- Only three commercial LLMs tested; whether open-source or fine-tuned models behave differently needs exploration
- The study focuses on discovering the phenomenon rather than proposing solutions; how to make LLMs generate text with authentic temporal structure remains an open problem
- Cognitive-emotional features depend on the accuracy of tools like LIWC; their generalizability across domains requires further validation

## Related Work & Insights

- **vs Synthetic data quality research (Chim et al.)**: Synthetic data research mainly focuses on static quality metrics; this paper reveals systematic deficiencies in the temporal dimension
- **vs Model collapse research (Shumailov et al.)**: Model collapse focuses on distributional degradation during iterative training; this paper provides a complementary perspective from temporal consistency
- **vs Human-LLM co-evolution research (Geng & Trotta)**: Prior work mainly analyzes at the lexical level; this paper extends to semantic and cognitive-emotional dimensions

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The "temporal flattening" concept is novel; examining LLM-generated text from a longitudinal perspective is an entirely new angle
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive design across three domains, three models, and two conditions with rigorous statistical testing
- Writing Quality: ⭐⭐⭐⭐ Clear paper structure, research question-driven
- Value: ⭐⭐⭐⭐ Direct implications for AIGC detection, synthetic data generation, and longitudinal text modeling

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Beyond the Final Actor: Modeling the Dual Roles of Creator and Editor for Fine-Grained LLM-Generated Text Detection](beyond_the_final_actor_modeling_the_dual_roles_of_creator_and_editor_for_fine-gr.md)
- [\[AAAI 2026\] Optimized Algorithms for Text Clustering with LLM-Generated Constraints](../../AAAI2026/aigc_detection/optimized_algorithms_for_text_clustering_with_llm-generated_constraints.md)
- [\[ACL 2026\] Who Wrote This Line? Evaluating the Detection of LLM-Generated Classical Chinese Poetry](who_wrote_this_line_evaluating_the_detection_of_llm-generated_classical_chinese_.md)
- [\[ACL 2026\] BIASEDTALES-ML: A Multilingual Dataset for Analyzing Narrative Attribute Distributions in LLM-Generated Stories](biasedtales-ml_a_multilingual_dataset_for_analyzing_narrative_attribute_distribu.md)
- [\[ACL 2026\] FlexGuard: Continuous Risk Scoring for Strictness-Adaptive LLM Content Moderation](flexguard_continuous_risk_scoring_for_strictness-adaptive_llm_content_moderation.md)

</div>

<!-- RELATED:END -->
---
title: >-
  [Paper Note] Temporal Flattening in LLM-Generated Text: Comparing Human and LLM Writing Trajectories
description: >-
  [ACL 2026][AIGC Detection] This paper constructs a longitudinal writing dataset spanning 12 years and discovers that LLM-generated text exhibits "temporal flattening"—while lexical diversity is high, temporal drift in semantic and cognitive-emotional dimensions is significantly lower than human writing, achieving 94% accuracy in distinguishing human from LLM text using temporal variation patterns alone.
tags:
  - ACL 2026
  - AIGC Detection
  - Temporal Flattening
  - Longitudinal Writing Analysis
  - Cognitive-Emotional Features
date: 2026-05-08
content_hash: 04efed9be34814ba
---

# Temporal Flattening in LLM-Generated Text: Comparing Human and LLM Writing Trajectories

**Conference**: ACL 2026  
**arXiv**: [2604.12097](https://arxiv.org/abs/2604.12097)  
**Code**: [GitHub](https://github.com/yjkim717/Cognitive-Emotional-Trajectories)  
**Area**: AIGC Detection  
**Keywords**: temporal flattening, LLM-generated text detection, longitudinal writing analysis, cognitive-emotional features, synthetic data

## TL;DR

This paper constructs a longitudinal writing dataset spanning 12 years and discovers that LLM-generated text exhibits "temporal flattening"—while lexical diversity is high, temporal drift in semantic and cognitive-emotional dimensions is significantly lower than human writing, achieving 94% accuracy in distinguishing human from LLM text using temporal variation patterns alone.

## Background & Motivation

**Background**: Large language models are widely used for content generation, dialogue systems, and synthetic training data production. The current LLM deployment paradigm is stateless—each generation produces an independent response without retaining historical memory.

**Limitations of Prior Work**: Human writing is inherently longitudinal—an author's style, cognitive state, and emotional expression naturally evolve over time. However, the existing LLM generation paradigm assumes that independently sampled documents can adequately approximate the distribution of human writing. This assumption has never been systematically verified.

**Key Challenge**: LLMs perform well on static quality metrics (e.g., semantic preservation, fluency), but may systematically lose the inherent longitudinal structure of human writing in the temporal dimension. This loss poses risks for downstream applications requiring temporal consistency (e.g., synthetic training data, authorship attribution, mental health trajectory modeling).

**Goal**: Answer two core questions—(1) Can LLMs reproduce human temporal structure over long time spans? (2) If differences exist, which temporal dynamics of human writing are systematically lost or flattened?

**Key Insight**: The authors approach from psycholinguistics and computational stylistics, treating writing as a longitudinal process, quantifying temporal structure through drift and variance—two complementary metrics across three representation spaces: semantic, lexical, and cognitive-emotional.

**Core Idea**: LLM-generated text exhibits "temporal flattening"—showing high diversity in lexical space but significantly lower temporal drift and fluctuation in semantic and cognitive-emotional spaces compared to humans, and this difference persists with or without historical conditioning.

## Method

### Overall Architecture

The study constructs a longitudinal dataset comprising 412 human authors and 6,086 documents (2012–2024), covering three domains: academic abstracts, blogs, and news. For each human author, matching writing trajectories are generated using three LLMs (DeepSeek V1, GPT-4o mini, and Claude 3.5 Haiku), producing 103,459 LLM documents in total. Temporal dynamics differences between humans and LLMs are quantified through drift and variance metrics across three representation spaces (lexical, semantic, cognitive-emotional), validated via statistical tests and classifiers.

### Key Designs

1. **Three-Dimensional Feature Trajectory Construction**:

    - Function: Capture temporal evolution of writing from multiple complementary dimensions
    - Mechanism: For each author at each year, three representations are computed—lexical representation (TF-IDF + SVD reduced to 10 dimensions), semantic representation (SBERT 384-dimensional embeddings), and cognitive-emotional features (20 interpretable psychological features including Big Five personality proxies, emotion, readability, etc.). Annual feature vectors are concatenated to form temporal trajectories $\mathcal{T}(e) = (\mathbf{x}_1^{(e)}, \ldots, \mathbf{x}_T^{(e)})$
    - Design Motivation: Lexical captures surface word usage changes, semantic captures deep meaning shifts, cognitive-emotional captures psychological and stylistic evolution—the three are complementary

2. **Drift-Variance Dual Metric System**:

    - Function: Quantify temporal evolution from both global geometric displacement and local fluctuation perspectives
    - Mechanism: Drift measures total displacement between adjacent years' feature vectors via L2 distance $\text{drift}_t^{(e)} = \|\mathbf{x}_{t+1}^{(e)} - \mathbf{x}_t^{(e)}\|_2$; variance quantifies inter-annual fluctuation irregularity via coefficient of variation (CV) $\mathrm{CV}(f) = \frac{\mathrm{std}(\Delta f_{1:T})}{\mathrm{mean}(\Delta f_{1:T})}$
    - Design Motivation: Drift reflects "how far" writing has moved in representation space, variance reflects "how irregularly" it has moved—together they characterize temporal dynamics

3. **Paired Statistical Tests and Prediction Probes**:

    - Function: Rigorously verify systematic differences in temporal dynamics between humans and LLMs
    - Mechanism: Binomial tests are performed for each matched human-LLM author pair, testing whether the proportion of human evolution exceeding LLM is significantly above 50%; a Random Forest classifier is trained using 20-dimensional cognitive-emotional CV features, with GroupKFold cross-validation to prevent data leakage
    - Design Motivation: Statistical tests ensure reliability of conclusions; classifiers verify predictability of differences

### Loss & Training

This is an analytical study with no model training involved. The classifier uses 5-fold GroupKFold cross-validation (grouped by author_id), ensuring all trajectories of the same author do not appear in both training and test sets simultaneously. Benjamini-Hochberg FDR correction ($q < 0.05$) is applied for multiple comparisons.

## Key Experimental Results

### Main Results

| Representation Space | Metric | Human Win Rate Range | p-value | Meaning |
|---------------------|--------|---------------------|---------|---------|
| TF-IDF (Lexical) | Drift | 0.20–0.33 | p=1.0 | LLM lexical drift is larger |
| SBERT (Semantic) | Drift | 0.75–0.86 | p<0.0001 | Human semantic drift far exceeds LLM |
| Cog-Emo (Cognitive-Emotional) | Drift | 0.76–0.99 | p<0.0001 | Human cognitive-emotional drift is highly significantly greater than LLM |

### Ablation Study

| Config | Accuracy | AUC | F1 | Note |
|--------|---------|------|-----|------|
| Pooled (IW) | 0.936 | 0.977 | 0.863 | Instance-wise condition |
| Pooled (Hist) | 0.933 | 0.977 | 0.856 | History-augmented condition |
| Balanced-Claude 3.5 (IW) | 0.97 | 1.00 | 0.97 | Claude easiest to distinguish |
| Balanced-GPT-4o mini (IW) | 1.00 | 1.00 | 1.00 | GPT-4o mini nearly perfectly distinguishable |

### Key Findings

- LLMs exhibit "asymmetric optimization": high lexical diversity but low semantic and cognitive-emotional drift, indicating LLMs optimize surface variation without reproducing deep evolution
- GPT-4o mini shows the most severe cognitive-emotional flattening: human Cog-Emo drift is greater in 97% of author pairs, rising to 99% under history-augmented conditions
- Temporal flattening persists under both instance-wise and history-augmented conditions, indicating this is a systemic property of current deployment paradigms
- Most discriminative features: average sentence length (18.7%), agreeableness (15.9%), and neuroticism (9.3%)

## Highlights & Insights

- Proposes the novel concept of "temporal flattening," revealing a systematic deficiency of LLM-generated text in the temporal dimension
- Achieves 94% classification accuracy and 98% AUC using only 20-dimensional cognitive-emotional CV features, providing a novel approach for AIGC detection
- Publicly releases a longitudinal dataset of 412 authors, providing a foundation for future research
- Discovers that history augmentation cannot fix temporal flattening, suggesting the root cause lies in model architecture or training paradigms themselves

## Limitations & Future Work

- The dataset primarily covers English writing; temporal flattening phenomena across languages remain to be verified
- Only three commercial LLMs are tested; whether open-source models and fine-tuned models behave differently requires exploration
- The study focuses on discovering phenomena rather than proposing solutions; how to make LLMs generate text with realistic temporal structure remains an open problem
- Cognitive-emotional features depend on the accuracy of tools like LIWC; their generalization across different domains needs further verification

## Related Work & Insights

- **vs Synthetic data quality research (Chim et al.)**: Synthetic data research mainly focuses on static quality metrics; this paper reveals systematic deficiencies in the temporal dimension
- **vs Model collapse research (Shumailov et al.)**: Model collapse concerns distribution degradation in iterative training; this paper provides a complementary perspective from temporal consistency
- **vs Human-LLM co-evolution research (Geng & Trotta)**: Prior work mainly analyzes at the lexical level; this paper extends to semantic and cognitive-emotional dimensions

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The "temporal flattening" concept is novel; examining LLM-generated text from a longitudinal perspective is an entirely new angle
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive design across three domains, three models, and two conditions with rigorous statistical testing
- Writing Quality: ⭐⭐⭐⭐ Clear paper structure, research question-driven
- Value: ⭐⭐⭐⭐ Direct implications for AIGC detection, synthetic data generation, and longitudinal text modeling

## Related Papers

- [\[ACL 2025\] Comparing LLM-generated and human-authored news text using formal syntactic theory](../../ACL2025/aigc_detection/llm_vs_human_formal_syntax.md)
- [\[ACL 2026\] Beyond the Final Actor: Modeling the Dual Roles of Creator and Editor for Fine-Grained LLM-Generated Text Detection](beyond_the_final_actor_modeling_the_dual_roles_of_creator_and_editor_for_fine-gr.md)
- [\[ACL 2026\] Who Wrote This Line? Evaluating the Detection of LLM-Generated Classical Chinese Poetry](who_wrote_this_line_evaluating_the_detection_of_llm-generated_classical_chinese_.md)
- [\[ACL 2026\] BIASEDTALES-ML: A Multilingual Dataset for Analyzing Narrative Attribute Distributions in LLM-Generated Stories](biasedtales-ml_a_multilingual_dataset_for_analyzing_narrative_attribute_distribu.md)
- [\[ACL 2026\] FlexGuard: Continuous Risk Scoring for Strictness-Adaptive LLM Content Moderation](flexguard_continuous_risk_scoring_for_strictness-adaptive_llm_content_moderation.md)
