---
title: >-
  [Paper Note] From Traditional Taggers to LLMs: A Comparative Study of POS Tagging for Medieval Romance Languages
description: >-
  [ACL 2026][Multilingual & Translation][Paper Note] The authors perform a systematic comparison of traditional taggers (UDPipe/COLaF) and open-source LLMs (Gemma3-12B/Phi4-14B) for POS tagging across three Medieval Romance languages (Old Occitan NAF, Old Catalan CAT, Old French Chauliac). Evaluating five settings—zero-shot, few-shot, monolingual fine-tuning, bilingual C
tags:
  - ACL 2026
  - Multilingual & Translation
date: 2026-05-08
content_hash: 147157abc650c9d1
---
# From Traditional Taggers to LLMs: A Comparative Study of POS Tagging for Medieval Romance Languages

**Conference**: ACL 2026  
**arXiv**: [2605.09147](https://arxiv.org/abs/2605.09147)  
**Code**: <https://github.com/msch38/medieval-romance-pos>  
**Area**: Multilingual / Historical NLP / POS Tagging  
**Keywords**: Medieval Romance, POS Tagging, LLM Fine-tuning, Cross-lingual Transfer, Digital Humanities

## TL;DR
The authors perform a systematic comparison of traditional taggers (UDPipe/COLaF) and open-source LLMs (Gemma3-12B/Phi4-14B) for POS tagging across three Medieval Romance languages (Old Occitan NAF, Old Catalan CAT, Old French Chauliac). Evaluating five settings—zero-shot, few-shot, monolingual fine-tuning, bilingual CLTF, and trilingual CLTF—they find that LLMs consistently outperform traditional methods. Catalan acts as a "bridge language," where CAT+FR bilingual training elevates the Old French Chauliac corpus to a peak accuracy of 93.14%.

## Background & Motivation
**Background**: Large-scale linguistic analysis of medieval documents in Digital Humanities typically begins with POS tagging, which provides the foundation for subsequent syntactic parsing, semantic analysis, and diachronic change modeling. Traditionally, tools like UDPipe and COLaF, trained on Universal Dependencies treebanks or rule-based systems, have been the mainstay for the medieval stages of the Romance family (Old Occitan, Old Catalan, Old French).

**Limitations of Prior Work**: Three characteristics of Medieval Romance languages cause significant degradation in modern taggers: (1) Extreme orthographic instability (e.g., `deceplina/disiplina/desiplina` appearing within a single text); (2) Complex and dialectal morphological systems; (3) Scarcity of annotated corpora, with the smallest dataset (Chauliac) containing only 2,443 tokens. Consequently, UDPipe performance drops to 68% on NAF, while COLaF achieves only 52% on CAT.

**Key Challenge**: A massive gap exists between the training distribution of modern NLP tools and the actual distribution of medieval texts, yet there are insufficient medieval labels to train dedicated models from scratch. It was previously unclear whether LLMs could "borrow" modern multilingual knowledge to bridge this gap, as systematic evaluations of LLMs for Medieval Romance POS tagging were virtually non-existent.

**Goal**: To answer three interrelated questions: (Q1) How much stronger are LLMs than traditional taggers? (Q2) To what extent do prompt formats and decoding hyperparameters matter? (Q3) How effectively can cross-lingual transfer rescue low-resource medieval languages?

**Key Insight**: The authors observe that Catalan is genealogically situated between Gallo-Romance and Occitano-Romance, acting as a potential bridge. Thus, they designed specific bilingual configurations (CAT+OCC, CAT+FR, FR+OCC) and a trilingual joint training setup to isolate the effects of genealogical proximity from corpus size.

**Core Idea**: By executing five experimental families (traditional, prompting, monolingual FT, bilingual CLTF, trilingual CLTF) and four decoding strategies on fixed 80/20 splits, the authors provide a practical roadmap for selecting methods based on available resources and target languages.

## Method

### Overall Architecture
The experimental framework is a 5×N grid: 5 method families (Traditional baseline / LLM Prompting / Monolingual FT / Bilingual CLTF / Trilingual CLTF) × 3 datasets (NAF 45,457 tokens, CAT 59,359 tokens, Chauliac 2,443 tokens) × 2 LLM backbones (Gemma3-12B, Phi4-14B). All LLM fine-tuning utilizes the same LoRA configuration ($r=16, \alpha=32$, targeting q/k/v/o_proj layers, learning rate $2\times 10^{-4}$, 10 epochs, AdamW). All decoding is scanned under a unified prompt template. The tagset is standardized to the 17 UD categories, with token streams as input and JSON output format: `[{"word":..., "UPOS":...}, ...]`.

### Key Designs

**1. Unified 80/20 Split: Disentangling transfer signals from data volume for fair comparison**

Prior literature on cross-lingual transfer often suffers from inconsistent splits—bilingual training sometimes inadvertently consumes the full 80% of target data, confounding "transfer gain" with "increased data." This study fixes a 20% test partition for each dataset. Bilingual training utilizes the union of two 80% partitions, and trilingual training uses the union of three, while testing always occurs on the fixed 20% partition. This ensures that $\Delta_{\text{bilingual-mono}}$ measures pure transfer signal rather than data scaling, validating conclusions such as the "Catalan bridge effect."

**2. Lineage-Driven Bilingual Pairing: Using genealogical priors to test if proximity predicts transfer**

Rather than brute-forcing all pairs, the authors selected three comparative groups: CAT+OCC (both Occitano-Romance), CAT+FR (spanning Gallo/Occitano with CAT as the middle ground), and FR+OCC (spanning both branches while bypassing CAT). Each pair is reported for both constituent languages. This design serves as hypothesis testing: if genealogical proximity is the dominant factor, FR+OCC should perform similarly to CAT+OCC; if CAT is a true "bridge," then CAT-inclusive configurations should prevail. Results showed CAT+OCC $\rightarrow$ NAF reached 89.25% and CAT+FR $\rightarrow$ Chauliac reached 93.14%, whereas FR+OCC $\rightarrow$ NAF (bypassing CAT) achieved only 80.31% (a marginal $+0.22$ gain over monolingual), strongly supporting the Catalan bridge hypothesis.

**3. Mixed Trilingual Few-shot Examples + Full Decoding Sweep: Eliminating pollution from prompt engineering and decoding parameters**

To ensure the "LLM > traditional tagger" conclusion is robust, the authors prove it is not a byproduct of a "magic" prompt or specific sampling temperature. Few-shot prompts intentionally mix tokens from three languages in a single block (e.g., `bo/ADJ`, `volch/VERB`, `seyor/NOUN`, `addicions/NOUN`) to explicitly expose the model to orthographic diversity. On the decoding side, they performed a comprehensive sweep covering beam search ($w\in\{1,15\}$), temperature ($\tau\in\{0.6,0.8,0.9\}$), top-$k$ ($\{5,20,50\}$), and top-$p$ ($\{0.75,0.85,0.95\}$). Since tagging is essentially a discriminative task, deterministic decoding was expected to win; indeed, beam-15 achieved the highest scores across all models and datasets (averaging 81.23% for Phi4 few-shot). The standard deviation across sampling families was only 0.1–0.2, effectively ruling out accidental gains from hyperparameters.

### Loss & Training
Fine-tuning employs standard next-token cross-entropy (supervising the UPOS field in the JSON output). LoRA updates only the four attention projection matrices with a dropout of 0.1. Training runs for 10 epochs with a batch size of 4; the smallest dataset, Chauliac (2,443 tokens), reaches the overfit zone quickly, explaining why its monolingual FT gain is limited compared to prompting (Gemma3 mono 83.64% vs. Phi4 few-shot 84.98%). No weights are updated during the prompting phase; only decoding hyperparameters are scanned.

## Key Experimental Results

### Main Results

| Method Family | Model / Strategy | NAF Acc. | CAT Acc. | Chauliac Acc. |
|---------------|------------------|----------|----------|---------------|
| Traditional   | UDPipe           | 68.01    | 81.59    | 89.40         |
| Traditional   | COLaF            | 65.73    | 52.15    | 72.50         |
| Prompting     | Phi4 Few-shot    | 75.01    | 83.69    | 84.98         |
| Fine-tune     | Gemma3 Mono      | 80.09    | **92.52**| 83.64         |
| Bilingual CLTF| Gemma3 CAT+OCC   | 89.25    | 91.62    | –             |
| Bilingual CLTF| Gemma3 CAT+FR    | –        | 91.28    | **93.14**     |
| Bilingual CLTF| Gemma3 FR+OCC    | 80.31    | –        | 85.74         |
| Trilingual CLTF| Gemma3          | **89.68**| 89.16    | 88.23         |
| $\Delta$ best vs UDPipe | —      | **+21.67**| **+10.93**| **+3.74**     |

Key observations: ① Average accuracy rises monotonically from Traditional (71.56%) $\rightarrow$ Prompting (77.75%) $\rightarrow$ Monolingual FT (85.19%) $\rightarrow$ Optimal CLTF; ② NAF benefits the most ($+21.67$ pp), and while Chauliac benefits the least ($+3.74$ pp), it is still pushed from 89.40% to 93.14%.

### Ablation Study

| Configuration | NAF | CAT | Chauliac | Interpretation |
|---------------|-----|-----|----------|----------------|
| Mono FT (Gemma3) | 80.09 | 92.52 | 83.64 | Baseline |
| Bilingual CAT+OCC | 89.25 | 91.62 | – | NAF $+9.16$, CAT decreased $0.90$ |
| Bilingual CAT+FR | – | 91.28 | 93.14 | Chauliac $+9.50$, CAT decreased $1.24$ |
| Bilingual FR+OCC | 80.31 | – | 85.74 | NAF $+0.22$, Chauliac $+2.10$ (minimal gain without CAT) |
| Trilingual CLTF | 89.68 | 89.16 | 88.23 | Best for NAF; but Chauliac is $4.91$ pp lower than CAT+FR |

**Phi4 prompting decoding robustness** (Average accuracy for few-shot, all decoding families):

| Decoding Family | Avg Acc. | Across-dataset std | Recommendation |
|-----------------|----------|--------------------|----------------|
| Beam-15         | 81.23    | 0.12               | Most stable and highest |
| Top-$k$         | 80.28    | 0.20               | Alternative |
| Top-$p$         | 80.69    | 0.18               | Balanced |
| Temperature     | 80.57    | 0.21               | Slightly higher variance |

### Key Findings
- **Catalan is a true bridge**: Any bilingual pairing involving CAT wins. FR+OCC only gained $+0.22$ pp on NAF, proving that genealogical proximity (French and Occitan both being Romance) is insufficient for transfer; a proper "intermediary" is required.
- **Trilingual isn't always best**: Chauliac performed worse in the trilingual setup (88.23%) than in the CAT+FR bilingual setup (93.14%) by $4.91$ pp. Ours attributes this to the small size of Chauliac (2,443 tokens), where trilingual training dilutes the signal. Focused exposure to closely related material remains superior for small datasets.
- **Uneven POS-level improvements**: On NAF, the categories benefiting most were PROPN ($+66.62$), NUM ($+62.97$), and SCONJ ($+41.68$). Functional words like ADP and CCONJ were already near the ceiling (85-95% F1), leaving little room for gain. This suggests LLM contextual representations primarily rescue "semantically heavy" open-class words.
- **Negligible decoding variance**: Phi4 few-shot maintained an average accuracy of 80.20-81.23% across all configurations (std $< 0.21$), confirming the results were not artifactual.

## Highlights & Insights
- **First comprehensive map of "Medieval Romance × LLM × Multilingual Transfer"**: Unlike prior work focusing only on prompting or monolingual FT, this paper evaluates 5 method families under a unified split, providing a reusable decision table (Table 13).
- **The "Bridge Language" discovery is transferable**: A similar "intermediary language" phenomenon likely exists in other families (e.g., Middle Dutch between Middle English and Old High German, or various stages of Sinitic). The bilingual CLTF paradigm can be replicated in those contexts.
- **Mixed multilingual few-shot prompting**: Mixing orthographic variants in a single example block informs the model during in-context learning to expect orthographic inconsistency. This trick is valuable for any NLP scenario involving high dialectal variance or spelling instability.
- **Avoid excessive multilingual training for tiny datasets**: For target domains with $< 5k$ tokens, trilingual training can dilute critical signals. This contradicts the "more diverse data is always better" intuition of the LLM era, providing a clear counter-example for practitioners in Digital Humanities.

## Limitations & Future Work
- Dataset sizes vary by an order of magnitude (2,443 vs 59,359 tokens) and span different genres (literature, chronicles, medicine). Ours acknowledges that controlled subsampling and learning-curve experiments were not conducted; thus, the effects of corpus size, genre, and genealogy are not strictly isolated in CLTF gains.
- Smaller LLMs (1–3B) or specialized token-classification encoders (XLM-R / mBERT) were not evaluated. The strongest results rely on 12–14B models with LoRA, which entail significantly higher deployment costs than UDPipe.
- Few-shot examples were fixed as a trilingual mix; language-specific prompts or target-language-specific example selection strategies were not explored.
- True zero-shot cross-lingual evaluation (e.g., CAT+FR $\rightarrow$ NAF without any NAF training data) was omitted. This remains a "cleaner" test of transfer ability for future work.
- The study did not extend to downstream tasks like lemmatization, parsing, or NER, leaving it uncertain if POS tagging gains propagate through the full historical NLP pipeline.

## Related Work & Insights
- **vs. Schöffel et al. 2025b (Modern Models, Medieval Texts)**: That study focused only on Old Occitan prompting and FT. Ours expands to three languages and incorporates bilingual/trilingual CLTF, providing a more systematic scope: if they proved "models work," ours answers "how to use them."
- **vs. Camps et al. 2021 (Classical French POS)**: They focused on monolingual adaptation for Classical French. Ours, by pushing French to 93.14% via Catalan, suggests that finding a genealogically intermediate "sibling" language for bilingual FT is more efficient than merely labeling more target data.
- **vs. Bollmann et al. 2019 / Manjavacas et al. 2019 (Historical Text Normalization)**: Those works follow a "normalize then tag" pipeline. Ours bypasses normalization, allowing LLMs to read raw orthography directly, and outperforms the traditional pipeline, suggesting that the necessity of normalization steps is decreasing in the LLM era.
- **vs. Karthikeyan et al. 2020 (mBERT Cross-lingual Ability)**: They found genealogical proximity to be the primary driver of transfer in modern languages. Ours points out that in medieval languages, genealogy is necessary but not sufficient—one also needs a suitable "bridge language" and adequate target domain data.

## Rating
- Novelty: ⭐⭐⭐⭐ No new architecture, but systematically maps 5 families × 3 languages and identifies the CAT bridge effect.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 method families × 3 datasets × 2 models × 12 decoding configs + detailed per-class F1 analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear tables, structured conclusions, and honest discussion of limitations.
- Value: ⭐⭐⭐⭐ The DH community can directly adopt the method selection table; the evidence for bridge languages offers cross-domain insights for low-resource NLP.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] A Case Study of Cross-Lingual Zero-Shot Generalization for Classical Languages in LLMs](../../ACL2025/multilingual_mt/a_case_study_of_cross-lingual_zero-shot_generalization_for_classical_languages_i.md)
- [\[ACL 2025\] Understanding In-Context Machine Translation for Low-Resource Languages: A Case Study on Manchu](../../ACL2025/multilingual_mt/understanding_in-context_machine_translation_for_low-resource_languages_a_case_s.md)
- [\[ICML 2026\] Toward Robust Multilingual Adaptation of LLMs for Low-Resource Languages](../../ICML2026/multilingual_mt/toward_robust_multilingual_adaptation_of_llms_for_low-resource_languages.md)
- [\[ACL 2025\] MiLiC-Eval: Benchmarking Multilingual LLMs for China's Minority Languages](../../ACL2025/multilingual_mt/milic-eval_benchmarking_multilingual_llms_for_chinas_minority_languages.md)
- [\[ACL 2026\] NiuTrans.LMT: Toward Inclusive and Scalable Multilingual Machine Translation with LLMs](niutranslmt_toward_inclusive_and_scalable_multilingual_machine_translation_with_.md)

</div>

<!-- RELATED:END -->
