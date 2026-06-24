---
title: >-
  [Paper Note] Comparing Moral Values in Western English-speaking Societies and LLMs with Word Associations
description: >-
  [ACL 2025][LLM (Other)][Moral Alignment] Proposes an LLM moral assessment framework based on word association rather than direct questioning, and constructs global moral networks (GMN) for humans and LLMs. The study finds high consistency between the two in positive moral dimensions, but shows that LLMs are systematically more abstract, less emotional, and less concrete on negative moral concepts.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Moral Alignment"
  - "Word Association"
  - "Moral Foundations Theory"
  - "Mental Lexicon"
  - "Global Moral Network"
date: 2026-05-08
content_hash: bc3fa454019a04d1
---

# Comparing Moral Values in Western English-speaking Societies and LLMs with Word Associations

**Conference**: ACL 2025  
**arXiv**: [2505.19674](https://arxiv.org/abs/2505.19674)  
**Code**: [https://github.com/ChunhuaLiu596/Word_Association_Generation](https://github.com/ChunhuaLiu596/Word_Association_Generation)  
**Area**: LLM/NLP / AI Safety  
**Keywords**: Moral Alignment, Word Association, Moral Foundations Theory, Mental Lexicon, Global Moral Network

## TL;DR

Proposes an LLM moral assessment framework based on word association rather than direct questioning, and constructs global moral networks (GMN) for humans and LLMs. The study finds high consistency between the two in positive moral dimensions, but shows that LLMs are systematically more abstract, less emotional, and less concrete on negative moral concepts.

## Background & Motivation

**Background**: LLMs are being deployed at scale in various real-world applications, making it crucial for AI safety to understand the moral values encoded within them. The current mainstream approach is to directly prompt LLMs with moral questionnaires (such as the Moral Foundations Questionnaire) and observe their agree/disagree responses to evaluate moral alignment.

**Limitations of Prior Work**: Direct questioning approaches suffer from four major limitations: (1) moral questionnaires may have leaked into the training data, allowing LLMs to simply "memorize answers"; (2) LLMs are highly sensitive to prompt wording, yielding different answers with slight paraphrasing; (3) binary responses (agree/disagree) fail to capture fine-grained moral reasoning; (4) the next-token prediction nature of LLMs biases them toward outputting "socially desirable answers" rather than genuine moral tendencies. Ji et al. (2024) have demonstrated that the moral understanding of LLMs is superficial and dominated by phrases in the training data.

**Key Challenge**: Direct evaluation methods of LLM morality are inherently unreliable—they can neither distinguish "genuine moral understanding" from "training data memorization" nor avoid prompt bias, thereby lacking credibility in their assessment results.

**Goal**: (1) How to probe the organization of moral concepts in LLMs without directly questioning their moral stances? (2) How to systematically compare moral value differences between humans and LLMs and explain the underlying causes?

**Key Insight**: In psychology, the Word Association Test has been shown to effectively reflect human moral reasoning processes. When participants freely associate with a cue word, their response patterns can indirectly reveal the moral organizational structure within their conceptual network. Ramezani & Xu (2024) validated the effectiveness of this paradigm on human association data, but only utilized local subgraphs and did not extend it to LLMs.

**Core Idea**: Replace direct questioning with word association to indirectly probe the moral conceptual structures of LLMs, and propagate moral information via random walks over a global graph to achieve a systematic human-LLM moral comparison.

## Method

### Overall Architecture

A three-stage framework: (1) Collect association responses for 12K cue words from humans (using the existing Small World of Words dataset with ~90K participants) and an LLM (prompting Llama-3.1-8B-Instruct) respectively to construct two association graphs: wa-h and wa-l; (2) Propagate 5-dimensional moral values (Care, Fairness, Loyalty, Authority, Sanctity) within the two association graphs using normalized random walks starting from 626 seed words based on Moral Foundations Theory (MFT), yielding two global moral networks: gmn-h and gmn-l; (3) Systematically compare the similarities and differences between the two moral networks, ranging from macro-dimensional correlations to micro-level conceptual qualitative analyses.

### Key Designs

1. **LLM Word Association Collection and Temperature Calibration**:

    - **Function**: Retrieve large-scale association graphs from Llama that are structurally aligned with human association data.
    - **Mechanism**: For 12K cue words, Llama is prompted 100 times for each (Monte-Carlo approximation of association probability distribution) to generate up to 3 association words per prompt, using the exact same instructions as the human experiment (Small World of Words). The key hyperparameter is the temperature $T$: the authors simultaneously optimize two metrics—diversity (total number of response types) and reliability (split-half reliability, calculated by the Spearman-Brown formula $r_{total} = 2r_{half}/(1 + r_{half})$), and find that at $T = 2.1$, the gap between the two metrics is minimized, making wa-l structurally close to wa-h.
    - **Design Motivation**: At default temperatures, the association diversity of LLMs is far lower than that of humans (a limitation pointed out by Abramski et al., 2024). Without calibrating the temperature, the two association graphs are structurally incomparable, rendering subsequent moral analyses meaningless. Optimizing both metrics simultaneously (rather than only tuning diversity) avoids the issue of being "diverse but unreliable."

2. **Global Moral Network Propagation (GMN)**:

    - **Function**: Propagate the 5-dimensional moral labels of MFD seed words to all 12K nodes in the association graph to obtain a moral score vector for each concept.
    - **Mechanism**: Initialize the moral matrix $F_0 \in \mathbb{R}^{|n| \times 5}$, where only the 626 MFD seed words have non-zero values (virtue = +1, vice = -1). Then propagate iteratively via $F_{t+1} = \alpha S F_t + (1-\alpha) F_0$, where $S = D^{-1/2}WD^{-1/2}$ is the symmetric normalized adjacency matrix. In practice, the closed-form solution $F^* = (I - \alpha S)^{-1} F_0$ is used. The hyperparameter $\alpha$ controls the propagation intensity, with the optimal $\alpha = 0.75$ for gmn-h and the optimal $\alpha = 0.9$ for gmn-l.
    - **Design Motivation**: Compared to the local subgraph approach of MAG (Ramezani & Xu, 2024), global propagation captures multi-hop, long-distance moral associations (e.g., "mother" $\rightarrow$ "birth" $\rightarrow$ "life"). gmn-l requires a larger $\alpha$ because the LLM association graph is sparser (density 0.007 vs. 0.013, diameter 4 vs. 3, connected components 77 vs. 114), meaning moral information needs a stronger "push" to propagate to distant nodes.

3. **Multi-granular Moral Alignment Analysis**:

    - **Function**: Systematically explain the causes of moral divergences between humans and LLMs across three levels: dimension, concept, and semantic feature.
    - **Mechanism**: (a) Dimensional level: Use the Spearman correlation on eMFD (2,186 evaluative words) to compare the predictive accuracy of the 5 moral dimensions; (b) Conceptual level: Compare the overlap and divergence of the top positive/negative moral concepts in gmn-h/gmn-l to identify polarity-reversed concepts (e.g., "abortion" is perceived as more negative by humans but more positive by the LLM); (c) Semantic feature level: Quantify systemic differences in the emotional intensity and concreteness of both sets of association responses using the VAD-norms emotional lexicon and the Brysbaert concreteness database.
    - **Design Motivation**: Merely looking at dimensional correlations fails to explain "why" variations exist. By delving deeper into specific concepts and semantic features, this analysis reveals the root cause of the LLM's moral bias—text-cooccurrence-based statistical association versus sensory-experience-based human association.

### Loss & Training

This work does not train a model. It uses the off-the-shelf Llama-3.1-8B-Instruct (15T token pre-training + RLHF). The key hyperparameters are the association temperature $T = 2.1$ and the propagation coefficient $\alpha$ (gmn-h: 0.75, gmn-l: 0.9). $\alpha$ is optimized and tuned using 277 non-evaluative words in eMFD.

## Key Experimental Results

### Main Results

**Moral Value Prediction (Spearman Correlation) vs. eMFD Ground-Truth:**

| Moral Dimension | MAG (baseline) | gmn-h (Human Graph) | gmn-l (LLM Graph) |
|:---|:---|:---|:---|
| Care (n=1895) | 0.29 | **0.47** | 0.46 |
| Sanctity (n=1893) | 0.25 | 0.39 | **0.44** |
| Fairness (n=1514) | 0.23 | 0.29 | **0.32** |
| Authority (n=1737) | 0.21 | 0.19 | **0.25** |
| Loyalty (n=1714) | 0.30 | 0.26 | **0.30** |
| Overall (n=8753) | 0.20 | 0.28 | **0.29** |

### Emotional Intensity and Concreteness Comparison (Top-50 Negative Moral Concepts)

| Metric | Care H/L | Fairness H/L | Loyalty H/L | Authority H/L | Sanctity H/L | Overall H/L |
|:---|:---|:---|:---|:---|:---|:---|
| Proportion of Emotional Responses (%) | 72/**61*** | 67/**54*** | 69/**54*** | 67/**59*** | 69/**58*** | 66/**55*** |
| Emotional Intensity | **4.24**/4.1 | 3.71/3.77 | 3.8/3.82 | 3.78/**4.10*** | 3.81/**3.60*** | 3.30/**3.17*** |
| Proportion of Concrete Responses (%) | **35**/24* | **24**/12* | **24**/12* | **29**/16* | **40**/33* | **42**/36* |
| Concreteness Score | **3.0**/2.7* | **2.6**/2.2* | **2.5**/2.3* | **2.7**/2.5* | **3.2**/3.0* | **3.1**/2.9* |

*Note: H=gmn-h (Human), L=gmn-l (LLM), \* indicates significant difference with t-test $p < 0.05$*

### Key Findings

- **Global propagation significantly outperforms local methods**: GMN outperforms the MAG baseline across all dimensions, improving the overall correlation from 0.20 to 0.28–0.29, which validates the effectiveness of multi-hop global propagation in capturing long-distance moral associations.
- **Consistency of positive morality is far higher than negative morality**: gmn-h and gmn-l show high overlap in top positive concepts (sharing words like "church", "religion", "God", "priest"), but distinct divergence in top negative concepts—humans favor sensory/emotional words (e.g., "disgusting", "vomit", "hurt"), whereas the LLM favors social justice terms (e.g., "betrayal", "prejudice", "discrimination").
- **Human associations are systematically more emotional and more concrete**: Across all 5 moral dimensions, the proportion of emotional responses and concreteness scores for human associations are significantly higher than those of the LLM. For example, for "prejudice", humans associate words like "pride, black, race" (based on concrete cultural experience), while the LLM associates "stereotypes, biases, bigoted" (highly abstract concepts).
- **Polarity-reversed concepts reveal RLHF bias**: The LLM rates "abortion, immigrant, politician" as more positive, while humans rate "jail, air, plastic" as more positive, indicating that RLHF training may inject specific social value preferences.
- **Sparser LLM association graph leads to differences in propagation efficiency**: gmn-l requires $\alpha = 0.9$ (versus 0.75 for gmn-h) to achieve optimal propagation because the network density of the LLM concept graph is only 54% of the human one.

## Highlights & Insights

- **Indirect Probing Paradigm**: Replacing direct moral questioning with word association cleverly avoids three major pitfalls: training data leakage, prompt sensitivity, and social desirability bias. This "moral inference without physical querying" paradigm is transferable to evaluating other latent properties of LLMs.
- **Two-Objective Temperature Calibration**: Optimizing the two conflicting objectives of diversity and reliability simultaneously helps to identify the optimal temperature, ensuring structural comparability between LLM and human association data. This methodology is generalizable to any experimental design requiring LLMs to simulate human behavioral distributions.
- **Explanatory Power of Graph Structure Differences**: The denser human graph facilitates easier propagation (requiring a smaller $\alpha$), whereas the sparser LLM graph needs more propagation drive. This is not just technical detail for hyperparameter tuning, but reveals fundamental structural differences in concept organization between LLMs and humans.
- **Quantitative + Qualitative Layered Analysis**: Going beyond macroscopic correlation numbers, the study delves layer-by-layer into association analyses of specific concepts, and quantifies emotionality and concreteness, offering convincing explanations for the differences.

## Limitations & Future Work

- Only one model, Llama-3.1-8B-Instruct, was tested. The organization of moral concepts in LLMs with different architectures (e.g., GPT, Claude) or scales (e.g., 70B, 405B) could be vastly different.
- The study focuses on Western English-speaking culture. The MFT framework itself remains controversial (e.g., Atari et al., 2023 suggests splitting the Fairness dimension), and its cross-cultural generalizability is unknown.
- Random walk propagation might be affected by hub nodes (highly connected general words), which can dilute the precision of moral signal propagation.
- The sufficiency of Monte-Carlo approximation (100 runs per cue) was not strictly validated; Precision@k drops noticeably for $k > 10$, demonstrating that the long-tail behavior of LLM associations still differs from humans.
- Conceptual-level analysis is difficult to generalize directly to sentence- or document-level moral reasoning scenarios.

## Related Work & Insights

- **vs. MAG (Ramezani & Xu, 2024)**: MAG propagates moral information over local cue subgraphs; this work extends it to global graph propagation, outperforming MAG across all dimensions (overall 0.29 vs. 0.20). Local methods fail to capture multi-hop moral connections.
- **vs. Direct Moral Questionnaires (Ji et al., 2024; Abdulhai et al., 2023)**: Directly questioning LLMs with MFQ suffers from data leakage and prompt sensitivity. This work bypasses these pitfalls via the indirect association paradigm, achieving a more robust moral profile.
- **vs. LLM Association Studies (Abramski et al., 2024)**: Prior work noted that LLM association diversity is lower than that of humans but did not address it. This work aligns structures via temperature calibration and is the first to apply word association to moral dimension analysis.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The methodological combination of word association + global moral networks brings a completely fresh perspective to LLM moral evaluation.
- Experimental Thoroughness: ⭐⭐⭐ The main experiments and analyses are solid, but only tested on a single LLM; the cross-model generalizability is unknown.
- Writing Quality: ⭐⭐⭐⭐ Excellent logical flow and clear analytical hierarchies, delving layer-by-layer from macro to micro.
- Value: ⭐⭐⭐⭐ Provides a novel indirect assessment paradigm for LLM moral alignment, offering high practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Psycholinguistic Word Features: A New Approach for the Evaluation of LLMs Alignment with Humans](psycholinguistic_word_features_a_new_approach_for_the_evaluation_of_llms_alignme.md)
- [\[ACL 2025\] Can LLMs Simulate L2-English Dialogue? An Information-Theoretic Analysis of L1-Dependent Biases](can_llms_simulate_l2-english_dialogue_an_information-theoretic_analysis_of_l1-de.md)
- [\[ACL 2025\] Comparing Linguistic Acceptability Judgments of Autoregressive Language Models](comparing_linguistic_acceptability_judgments_of_autoregressive_language_models.md)
- [\[ACL 2025\] Probabilistic Aggregation and Targeted Embedding Optimization for Collective Moral Reasoning](probabilistic_aggregation_and_targeted_embedding_optimization_for_collective_mor.md)
- [\[ACL 2025\] Comparing Large Language Models in Extracting Subjective Information from Political News](comparing_large_language_models_in_extracting_subjective_information_from_politi.md)

</div>

<!-- RELATED:END -->
