---
title: >-
  [Paper Note] Exploring Cross-Client Memorization of Training Data in Large Language Models for Federated Learning
description: >-
  [ACL 2026][LLM Safety][Federated Learning] The authors extend fine-grained cross-sample memorization metrics from centralized LLMs (Zeng 2024 + PAN2014 plagiarism detector) to the Federated Learning (FL) setting. They pr…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Federated Learning"
  - "Training Data Memorization"
  - "Privacy Leakage"
  - "Cross-Client Leakage"
  - "PAN2014"
date: 2026-05-08
content_hash: f8c58b1ba01fdc09
---

# Exploring Cross-Client Memorization of Training Data in Large Language Models for Federated Learning

**Conference**: ACL 2026  
**arXiv**: [2510.08750](https://arxiv.org/abs/2510.08750)  
**Code**: https://github.com/tinnakitudsa/FL_memorization_framework.git  
**Area**: LLM Security / Federated Learning / Privacy  
**Keywords**: Federated Learning, Training Data Memorization, Privacy Leakage, Cross-Client Leakage, PAN2014

## TL;DR
The authors extend fine-grained cross-sample memorization metrics from centralized LLMs (Zeng 2024 + PAN2014 plagiarism detector) to the Federated Learning (FL) setting. They propose a client-pair metric $\text{MR}_{j \to k}$ to derive intra-client and inter-client memorization ratios. The study finds that FL **does not** effectively prevent training data memorization—while intra-client memorization is higher than inter-client, the total memorization ratio of FL vs. CL does not significantly decrease. Furthermore, memorization is significantly influenced by prefix length, decoding strategies, and FL algorithms (FedProx > FedAvg).

## Background & Motivation
**Background**: Federated Learning (FL) is widely promoted as a privacy-preserving paradigm for sensitive sectors like healthcare and finance because it "avoids sharing raw data" by training locally and only uploading gradients or parameters. However, LLMs "remember" training data during fine-tuning (Carlini 2022 et al.). Whether this memorization exists under FL and whether it leaks across clients remains a question that hasn't been systematically quantified.

**Limitations of Prior Work**: (a) Existing memorization metrics in Centralized Learning (CL) (verbatim / k-extractable / paraphrase / idea-level, with Zeng 2024 being the state-of-the-art using the PAN2014 plagiarism detector) implicitly assume that "a memorized suffix can only be triggered by its original prefix." This is reasonable in CL but misses the more dangerous cross-client leakage where "a prefix from client A triggers a suffix from client B." (b) FL-specific memorization research (Thakkar 2021, Ramaswamy 2020) relies almost exclusively on canary injection (inserting out-of-distribution phrases to see if the model outputs them). This only detects same-sample verbatim leakage for OOD data and fails completely for realistic in-distribution cross-sample leakage.

**Key Challenge**: There is a measurement gap between the privacy-preserving claims of FL and the actual risk of memorization. Fine-grained CL methods are limited to same-sample scenarios, while FL cross-client scenarios rely on coarse tools like canary injection.

**Goal**: (1) Adapt fine-grained cross-sample memorization metrics from CL to multi-client FL scenarios; (2) Use this new framework to quantitatively answer: Does the FL model remember training data, and what factors influence the level of memorization?

**Key Insight**: Directly extend the PAN2014-based framework from Zeng 2024 and Lee 2023. By relaxing the decision function $F(M(p), s)$ from "$p$ and $s$ belong to the same sample" to "$p$ comes from client $C_j$ and $s$ comes from client $C_k$," the authors derive the client-pair metric $\text{MR}_{j \to k}$.

**Core Idea**: The concept of "memorization" is expanded from same-sample/intra-client prefix-suffix matching to **matching between prefixes and suffixes of any client pair**. This distinguishes between harm-exposed risks (intra-client: $C_j = C_k$) and harmful risks (cross-client: $C_j \neq C_k$).

## Method

### Overall Architecture
The framework consists of five steps (Steps ①-⑤ in Figure 1): (①) Sample $n=4000$ prefix-suffix pairs from the training set $D_j$ of client $C_j$, and sample $n$ suffixes from client $C_k$; (②) Input each prefix $\tilde{p}$ from $C_j$ into the global FL model $M$ to generate a completion $M(\tilde{p})$; (③) Index the suffix set $\tilde{S}_k$ of $C_k$ using Elasticsearch and perform similarity retrieval for each $M(\tilde{p})$ to get the top-$n'=10$ most similar real suffixes; (④) Use the PAN2014 plagiarism detector (judging across four levels: verbatim, paraphrase with $p>0.5$, paraphrase with $p<0.5$, and idea-level) to determine if $M(\tilde{p})$ matches any of the top-$n'$ candidates; (⑤) If any match returns True, the prefix is considered to have triggered $C_j \to C_k$ memorization. The ratio is calculated as $\text{MR}_{j \to k} = |P_{j,k}| / |P_j|$.

From this, two core metrics are derived: $\text{MR}_{\text{Intra}}$ (weighted average where $j = k$, representing intra-client leakage) and $\text{MR}_{\text{Inter}}$ (weighted average for all $j \neq k$, representing cross-client leakage). For fair comparison with CL, $\text{MR}_{\text{TotalCL}}$ and $\text{MR}_{\text{TotalFL}}$ (the union of all prefixes triggering any memorization) are also defined.

### Key Designs

1.  **Extension from Same-Sample Assumption to Client-Pair Metric**:
    *   Function: Relaxes the CL-era assumption that "prefix $\in P, suffix \in S$ belong to the same sentence" to "prefix $p$ from client $j$, suffix $s$ from client $k$." This allows for the first quantitative capture of the most dangerous privacy leak: an inquiry from client A extracting private suffixes from client B.
    *   Mechanism: Formalized in Definition 3.1 as "there exists $s_k \in S_k$ such that $F(M(p_j), s_k) = \text{True}$." This is categorized into intra-client ($j=k$, harm-exposed) and inter-client ($j \neq k$, directly harmful). $\text{MR}_{j \to k} = |P_{j,k}| / |P_j|$ serves as the atomic metric for client pairs.
    *   Design Motivation: Existing CL metrics either only calculate intra-client (missing cross-client risks) or mix all client data together (failing to distinguish harm-exposed vs. harmful). The client-pair matrix $\text{MR}_{j \to k}$ is the minimal information unit required to support both granularities.

2.  **PAN2014 Three-Layer Fine-Grained Detector + Elasticsearch Acceleration**:
    *   Function: Prevents verbatim-only metrics from underestimating real memorization and reduces the $O(n^2)$ complexity of comparing every prefix to every suffix.
    *   Mechanism: (a) Uses the Lee 2023 version of the PAN2014 detector supporting verbatim, paraphrase (high confidence $p>0.5$ and low confidence $p<0.5$), and idea levels to capture hidden memorization where the same concept is stated with different words. (b) Indexes suffixes via Elasticsearch, reducing comparisons from $1.6 \times 10^7$ to $4 \times 10^4$.
    *   Design Motivation: Inherits the validated design from Zeng 2024. However, Appendix E.5 honestly notes that PAN2014 may misjudge **incoherent outputs** (e.g., repeating "lobes, lobes, lobes" due to mode collapse) as idea memorization, as it was designed for human-like text.

3.  **Aggregated Metrics (Intra, Inter, Total)**:
    *   Function: Enables the same $\text{MR}_{j \to k}$ matrix to answer three questions: intra-client leakage, cross-client leakage, and the difference between total FL and CL leakage.
    *   Mechanism: $\text{MR}_{\text{Intra}} = \sum_j w_j \cdot \text{MR}_{j \to j}$ and $\text{MR}_{\text{Inter}} = \sum_j w_j \cdot \frac{1}{L-1}\sum_{j \neq k} \text{MR}_{j \to k}$, where $w_j$ is the data weight. $\text{MR}_{\text{TotalFL}}$ uses the union of triggered prefixes to avoid double-counting.
    *   Design Motivation: Weighted averages prevent small client noise from overwhelming large client data, while the union for "Total" accounts for the fact that a prefix is a leak if it triggers any suffix once.

### Loss & Training
No new training algorithms are introduced. Standard FL baselines are used: FedAvg (McMahan 2017) and FedProx (Li 2020). Models include Qwen2.5-3B (main), Llama3.2-1B/3B, GPT-2 XL, and Qwen2.5-0.5B/1.5B (ablation). Framework: LLaMA Factory, lr=2e-4, bf16, batch=64. 3 FL clients with 27k training and 3k test samples each. Default prefix length is 30, top-k decoding (k=40), 3 communication rounds. Tasks cover summarization, dialog (HealthCareMagic), QA (PubMedQA), and classification (PubMed 200k RCT).

## Key Experimental Results

### Main Results
**RQ1: Does the FL model memorize training data?** (Qwen2.5-3B + FedAvg + top-k + prefix=30):

| Task | $\text{MR}_{\text{Intra}}$ (%) | $\text{MR}_{\text{Inter}}$ (%) | Intra/Inter Ratio |
|------|--------------------------------|--------------------------------|-----------------|
| Summarization | 0.342 | 0.046 | 7.4× |
| Dialog | 1.533 | 1.446 | 1.06× |
| QA | 1.450 | 0.813 | 1.78× |
| Classification | 0.000 | 0.000 | – |

**FL vs. CL Total Memorization + Performance**:

| Task | Model | Performance | Memorization (%) |
|------|------|-------------|------------------|
| Summarization | $\text{MR}_{\text{TotalCL}}$ | 28.46 | 0.558 |
| Summarization | $\text{MR}_{\text{TotalFL}}$ | 29.88 | 0.433 |
| Dialog | $\text{MR}_{\text{TotalCL}}$ | 19.40 | 3.417 |
| Dialog | $\text{MR}_{\text{TotalFL}}$ | 18.11 | **3.992** |
| QA | $\text{MR}_{\text{TotalCL}}$ | 26.66 | 2.150 |
| QA | $\text{MR}_{\text{TotalFL}}$ | 28.60 | **2.917** |
| Classification | $\text{MR}_{\text{TotalCL}}$ | 76.30 | 0.000 |
| Classification | $\text{MR}_{\text{TotalFL}}$ | 51.22 | 0.000 |

→ In Dialog/QA, FL actually shows **higher memorization than CL**, contradicting the conclusion of Thakkar 2021 that "FL reduces memorization."

### Ablation Study

| Config | Summa $\text{MR}_{\text{Intra}}$ | Dialog $\text{MR}_{\text{Intra}}$ | QA $\text{MR}_{\text{Intra}}$ | Note |
|------|--------|--------|--------|------|
| Decoding: temperature | 0.475 | 1.267 | 1.283 | baseline |
| Decoding: top-k | 0.342 | 1.533 | 1.450 | Slight increase |
| Decoding: top-p | **0.525** | **3.792** | **2.567** | Max increased (Dialog +2x) |
| Prefix 10 | 0.508 | 2.108 | 1.525 | Short prefix |
| Prefix 30 | 0.342 | 1.533 | 1.450 | Default |
| Prefix 50 | 0.425 | 1.575 | 1.242 | Mid-length |
| Prefix 100 | **0.208** | **1.408** | **1.125** | Long prefix → Lowest mem |
| FL algo: FedAvg | 0.342 | 1.533 | 1.450 | baseline |
| FL algo: FedProx | **0.942** | **1.892** | **3.675** | 2-3x more than FedAvg |
| Model size 0.5B/1.5B/3B | 0.550 / 0.992 / 0.342 | – | – | No clear trend |
| Comm rounds 1/3/5 | 0.433 / 0.342 / 0.467 | – | – | No clear trend |

**Effect of Suffix Client Source** (Dialog task, $\text{MR}_{j \to k}$ %):

| Prefix \ Suffix | Group1 | Group2 | Group3 |
|------|--------|--------|--------|
| Group1 | 1.450 | 1.525 | 1.500 |
| Group2 | 1.150 | 1.200 | 1.225 |
| Group3 | **1.725** | 1.550 | **1.950** |

→ Group 3's suffixes are consistently easier to remember (average 1.475 across rows), suggesting **content characteristics** determine risk more than the prefix source.

### Key Findings
*   **FL is not a silver bullet**: The general rule of Intra > Inter shows that "a client's own data is easier to extract with its own prefix" is a harm-exposed risk. However, Inter-client leakage is significant (Dialog 1.446% is nearly equal to Intra), meaning one client's data can indeed be pulled out by another.
*   **Thakkar 2021's conclusion disputed**: That paper used OOD canary injection while this uses in-distribution PAN2014. FL memorizing more than CL in Dialog/QA proves that OOD safety does not equal in-distribution safety.
*   **Longer prefixes reduce memorization**: From 10 to 100 tokens, $\text{MR}_{\text{Inter}}$ in Summarization drops by a factor of 47. Intuition: Long prefixes provide a "unique fingerprint," making it harder for the model to reproduce verbatim unless it is exact. **Short prompts are the most dangerous** privacy probes.
*   **Decoding strategies amplify memorization**: Temperature sampling is the weakest, while top-p increases Dialog Intra memorization by nearly 3x. Improving decoding quality in FL LLMs creates a trade-off with privacy.
*   **FedProx > FedAvg in memorization**: The proximal regularization in FedProx, intended to stabilize non-IID training, forces the model to fit local data more tightly, doubling memorization in some cases (QA Intra 1.450 → 3.675).
*   **Artifact in classification (0%)**: Classification outputs are only 1-2 tokens. Since generated suffixes are shorter than the 50-character PAN2014 threshold, measurements are truncated to 0. This is a measurement blind spot, not a lack of memorization.
*   **Data heterogeneity matters**: Group 3's suffixes were 1.6x easier to remember than Group 2's. Weighted averages based solely on data volume may mask risks related to content patterns like repetition.

## Highlights & Insights
*   **Paradigm value of client-pair matrix**: Shifts FL privacy metrics from a single aggregated number to an $L \times L$ matrix, enabling future research into client heterogeneity and targeted defense strategies.
*   **Honesty regarding PAN2014**: The authors transparently document mode-collapsed outputs ("lobes, lobes...") as false positives in Appendix E.5 and implement simple heuristic filters, increasing the credibility of the paradigm.
*   **Challenging consensus**: By providing counter-examples to the "FL reduces memorization" consensus using fine-grained metrics, the paper provides a methodology shock to the FL privacy field.
*   **Counter-intuitive prefix finding**: Contrary to the bridge that "more context = easier recall," longer prefixes make the reproduction task more "fingerprinted" and harder to achieve.

## Limitations & Future Work
*   **PAN2014 Artifacts**: (a) False positives for incoherent output; (b) English-only; (c) 50-character threshold causes failures in short-output tasks.
*   **Lack of Theory**: The study is primarily observational and lacks a probabilistic or information-theoretic proof for the observed phenomena.
*   **Scalability**: Tested only with 3 clients. Real-world FL involves 100-1000 clients, where cross-client retrieval may face an $O(L^2)$ bottleneck.
*   **Missing Defense Evaluation**: The work measures leakage but does not yet evaluate standard defenses like Differential Privacy (DP), gradient clipping, or secure aggregation against these fine-grained metrics.
*   **Future Directions**: Development of metrics for mode collapse/multilingualism; using the client-pair matrix for adaptive DP noise; and applying mechanistic interpretability to find "memorization neurons" in FL.

## Related Work & Insights
*   **vs. Zeng 2024**: Inherits the PAN2014 pipeline but relaxes the "same sample" constraint to "cross-client" to define the FL-specific metric layer.
*   **vs. Thakkar 2021**: Replaces OOD canary injection with in-distribution fine-grained metrics, proving that canary-based studies have systematically underestimated real-world FL leaks.
*   **vs. Lee 2023**: Extends the large-scale plagiarism detection framework from CL to the multi-client FL domain.
*   **Cross-domain potential**: This framework could theoretically be applied to cross-organization data sharing, cross-domain pre-training, or cross-modal data memorization.

## Rating
*   Novelty: ⭐⭐⭐⭐ The client-pair matrix and harm-exposed/harmful distinction are clear conceptual gains, though the underlying tool is inherited.
*   Experimental Thoroughness: ⭐⭐⭐⭐ High coverage across models, tasks, and parameters, though limited to 3 clients.
*   Writing Quality: ⭐⭐⭐⭐ Clear formalization and honest discussion of limitations.
*   Value: ⭐⭐⭐⭐ A significant wake-up call for FL privacy (FL ≠ Security) and a solid infrastructure for future defense work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Revisiting Non-Verbatim Memorization in Large Language Models: The Role of Entity Surface Forms](revisiting_non-verbatim_memorization_in_large_language_models_the_role_of_entity.md)
- [\[ICML 2026\] Decoupled Training with Local Reinforcement Fine-Tuning in Federated Learning](../../ICML2026/llm_safety/decoupled_training_with_local_reinforcement_fine-tuning_in_federated_learning.md)
- [\[AAAI 2026\] TOFA: Training-Free One-Shot Federated Adaptation for Vision-Language Models](../../AAAI2026/llm_safety/tofa_training-free_one-shot_federated_adaptation_for_vision-language_models.md)
- [\[ACL 2026\] Learning Uncertainty from Sequential Internal Dispersion in Large Language Models](learning_uncertainty_from_sequential_internal_dispersion_in_large_language_model.md)
- [\[NeurIPS 2025\] FedRW: Efficient Privacy-Preserving Data Reweighting for Enhancing Federated Learning of Language Models](../../NeurIPS2025/llm_safety/fedrw_efficient_privacy-preserving_data_reweighting_for_enhancing_federated_lear.md)

</div>

<!-- RELATED:END -->
