---
title: >-
  [Paper Note] In Agents We Trust, but Who Do Agents Trust? Latent Source Preferences Steer LLM Generations
description: >-
  [ICLR 2026][Recommender Systems][LLM Agent] Through large-scale controlled experiments across 12 LLMs from 6 providers spanning three domains—news, academia, and e-commerce—this paper reveals that LLMs exhibit systematic **latent source preferences**: when content is semantically identical, merely swapping source labels significantly alters model selection behavior, and this preference cannot be eliminated through prompt engineering.
tags:
  - ICLR 2026
  - Recommender Systems
  - LLM Agent
  - Source Preference
  - Trust Bias
  - Brand Perception
date: 2026-05-08
content_hash: 6f616b12fe32602a
---

# In Agents We Trust, but Who Do Agents Trust? Latent Source Preferences Steer LLM Generations

**Conference**: ICLR 2026
**arXiv**: [2602.15456](https://arxiv.org/abs/2602.15456)
**Area**: Recommender Systems / LLM Bias Analysis
**Keywords**: LLM Agent, Source Preference, Trust Bias, Brand Perception, Recommender Systems

## TL;DR

Through large-scale controlled experiments across 12 LLMs from 6 providers spanning three domains—news, academia, and e-commerce—this paper reveals that LLMs exhibit systematic **latent source preferences**: when content is semantically identical, merely swapping source labels significantly alters model selection behavior, and this preference cannot be eliminated through prompt engineering.

## Background & Motivation

**Background**: LLM-based agents are being deployed at scale as user-facing interfaces on online platforms, handling tasks such as news aggregation, academic search, and e-commerce recommendation. These agents filter, prioritize, and synthesize information retrieved from backend databases or web searches, effectively controlling the information users ultimately receive.

**Limitations of Prior Work**: Extensive research has examined biases in LLM-generated content (political, gender, and cultural biases, among others), yet few studies have systematically investigated whether LLMs exhibit preferences when **selecting and presenting** existing information. When content carries source labels (e.g., specific publishers, journals, or platforms), do LLMs systematically favor certain sources?

**Key Challenge**: LLM agents are assumed to play the role of "neutral intermediaries" in deployment. However, if models harbor implicit preferences for particular information sources, this creates information asymmetry—certain sources are systematically amplified while others are suppressed. More critically, users remain entirely unaware of such preferences and have no means to control them.

**Goal**: This paper introduces the concept of "latent source preferences" and validates their existence, magnitude, context-dependence, and resistance to debiasing across 12 LLMs through systematic controlled experiments—constructing semantically equivalent content pairs annotated with different sources.

## Method

### Overall Architecture

The paper designs a multi-level controlled experimental framework to detect and quantify latent source preferences in LLMs:

- **Input**: Semantically equivalent content pairs annotated with different sources (control pairs)
- **Models Evaluated**: 12 LLMs from 6 providers (GPT-4.1-Mini/Nano, Llama-3.1/3.2, Phi-4/Mini, Mistral-Nemo/Ministral, Qwen2.5-7B/1.5B, DeepSeek-R1-Llama/Qwen)
- **Evaluation Domains**: News (political lean set + world news set), Academia (50 top journals/conferences), E-commerce (70 platforms)
- **Quantitative Metrics**: Preference distribution percentage + Kendall Tau rank correlation coefficient

### Key Design 1: Direct vs. Indirect Evaluation

**Direct Evaluation**: Explicitly prompts the model for its preference between two information sources, analogous to the LLM-as-a-judge paradigm. For example: "Please compare the journalistic standards of CNN and Fox News." This captures explicitly stated preferences.

**Indirect Evaluation**: In realistic task settings, the model is presented with two semantically equivalent articles attributed to different sources and asked to select the "higher quality" one. Consistent preference for a particular source constitutes evidence of latent preference. The key value of indirect evaluation lies in the fact that a model's explicit statements may diverge from its actual behavior, making this approach more reflective of real-world deployment behavior.

### Key Design 2: Multi-Dimensional Identity Association and Credential Rationality Analysis

**Identity Consistency Analysis**: Tests whether models can associate different identity representations of the same source (brand name, social media handle, website URL, follower count, etc.) and assign consistent preferences. Most large models can recognize brand name ↔ URL mappings, but the ability to associate brand names with social media IDs degrades substantially.

**Credential Rationality Analysis**: Tests whether model preferences for numerical source credentials (H5-Index, follower count, founding year) are rationally consistent. H5-Index serves as a relatively stable positive signal, whereas the interpretation of follower count and founding year exhibits inconsistency and irrational behavior—some models equate "older = more trustworthy," while others exhibit the opposite pattern.

### Key Design 3: AllSides News Case Study

Using 3,855 news event records from AllSides.com, the paper designs six controlled experimental conditions:

1. **Source Hidden**: Source information is concealed; content-based selection is observed.
2. **Source Shown**: Source information is revealed; the effect of source on selection is observed.
3. **Source Swap**: Source labels of the two articles are exchanged; whether preferences follow the source is tested.
4. **Do Not Be Biased**: A "please avoid bias" instruction is added; debiasing effectiveness is evaluated.
5. **Left↔Right Swap**: Whether political orientation influences selection is tested.
6. **Multi-Round Consistency**: Experiments are repeated to assess the stability of preferences.

## Key Experimental Results

### Main Results: Source Preference Magnitude

| Domain | Metric | Key Finding |
|--------|--------|-------------|
| Political News | Preference % Std. Dev. | GPT-4.1-Mini and Phi-4 exhibit the largest preference variance; smaller models show weaker preferences |
| World News | Kendall Tau | Preference rankings are highly correlated across models ($\tau > 0.6$), suggesting a shared training data effect |
| Academic Research | Preference % (by field) | NEJM is selected 96% of the time in medicine but only 19% in computer vision—strong context dependence |
| E-commerce Platforms | Preference % (by category) | BestBuy is selected 97% of the time in electronics but only 51% in food and groceries |

### Ablation Study: Causal Validation of Source Preferences

| Experimental Condition | Key Behavioral Change | Explanation |
|------------------------|----------------------|-------------|
| Source Hidden | Selection distribution approaches uniform | Without source information, models select based on content |
| Source Shown | Selection distribution becomes significantly skewed | Preferences emerge immediately upon revealing source |
| Source Swap | Preference direction reverses | Swapping source labels causes selection to follow the source rather than the content |
| Do Not Be Biased | Preferences show no significant reduction | Prompt-based debiasing is nearly ineffective and may even amplify preferences |

### Key Numerical Findings

- **Source preference can override content**: In AllSides news selection, post-Source Swap reversal rates reach 60–80%, indicating that source labels exert greater influence than content itself.
- **Larger models exhibit stronger preferences**: Models such as GPT-4.1-Mini display stronger and more heterogeneous preferences than smaller models (preference variance increases by $\sim$2–3×).
- **Post-training reshapes preferences**: DeepSeek-R1-Distill-Llama-8B and Llama-3.1-8B-Instruct share the same base model, yet their Kendall Tau is only 0.42—post-training substantially restructures preference rankings.
- **Preferences are context-dependent**: The same model may exhibit entirely different preferences for the same source across different topical domains.

## Highlights & Insights

- **First systematic study of latent source preferences in LLMs**: Rather than examining what LLMs generate, this work investigates how LLMs select and present existing information—a fundamentally new research perspective.
- **Elegant controlled experimental design**: By using semantically equivalent content with only source labels varied, the paper rigorously isolates source effects from content effects, yielding clear causal inference.
- **Far-reaching practical implications**: Systematic source preferences in LLM agents could lead to filter bubbles, unfair brand competition, and public opinion manipulation; malicious actors could exploit high-trust source identifiers to manipulate recommendation outcomes.
- **"Prompt debiasing is ineffective" is an important negative result**: This finding demonstrates that simple engineering interventions are insufficient, and deeper training-time interventions are necessary.
- **Cross-model consistency reveals training data effects**: High Kendall Tau correlations in preference rankings across different models suggest that preferences are rooted in shared pretraining corpora.

## Limitations & Future Work

- Only three application domains (news/academia/e-commerce) are covered; high-stakes domains such as healthcare and law remain unexplored.
- The causal origins of preferences are not analyzed in depth—the relative contributions of pretraining data frequency, post-training data, and model architecture remain unclear.
- The observed correlation between pretraining data co-occurrence frequency and source preference does not fully explain the phenomenon; deeper mechanistic investigation is needed.
- No effective debiasing method is proposed (only the ineffectiveness of prompting is demonstrated); training-time and inference-time debiasing strategies remain to be explored.
- Whether multimodal LLMs exhibit analogous preferences warrants investigation.
- Experiments cover only 12 models; preference patterns in a broader set of models—particularly open-source Chinese-language models and domain-specific models—remain to be studied.

## Related Work & Insights

- **LLM Bias Research**: Feng et al. (2023) on political bias; Manvi et al. (2024) on geographic bias—this paper reveals a novel dimension of "source bias."
- **LLM Cognitive Biases**: Itzhak et al. (2024) investigate the origins of cognitive biases; this paper further demonstrates that post-training can substantially reshape preference orderings.
- **Recommender System Fairness**: Existing fairness frameworks from traditional recommender systems can be transferred to LLM agent settings.
- **Implications for Future Agent System Design**: Agent-level modules for transparent and controllable source preference management should be incorporated into system design.

## Rating

- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Who You Are Matters: Bridging Topics and Social Roles via LLM-Enhanced Logical Recommendation](../../NeurIPS2025/recommender/who_you_are_matters_bridging_topics_and_social_roles_via_llm-enhanced_logical_re.md)
- [\[ICLR 2026\] Token-Efficient Item Representation via Images for LLM Recommender Systems](token-efficient_item_representation_via_images_for_llm_recommender_systems.md)
- [\[AAAI 2026\] Tokenize Once, Recommend Anywhere: Unified Item Tokenization for Multi-domain LLM-based Recommendation](../../AAAI2026/recommender/tokenize_once_recommend_anywhere_unified_item_tokenization_for_multi-domain_llm-.md)
- [\[AAAI 2026\] Tool4POI: A Tool-Augmented LLM Framework for Next POI Recommendation](../../AAAI2026/recommender/tool4poi_a_tool-augmented_llm_framework_for_next_poi_recommendation.md)
- [\[AAAI 2026\] Align³GR: Unified Multi-Level Alignment for LLM-based Generative Recommendation](../../AAAI2026/recommender/align3gr_unified_multi-level_alignment_for_llm-based_generat.md)

<!-- RELATED:END -->
