---
title: >-
  [Paper Note] Do Multimodal RAG Systems Leak Data? A Comprehensive Evaluation of Membership Inference and Image Caption Retrieval Attacks
description: >-
  [ACL 2026][LLM Safety][Multimodal RAG] The authors provide the first systematic evaluation of privacy leakage risks in **image-driven multimodal RAG (mRAG)** systems. They demonstrate that a naive black-box text prompt c…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Multimodal RAG"
  - "Membership Inference Attack"
  - "Image Caption Retrieval"
  - "Black-box Privacy"
  - "VLM"
date: 2026-05-08
content_hash: d8cbb6832f1a2b7b
---

# Do Multimodal RAG Systems Leak Data? A Comprehensive Evaluation of Membership Inference and Image Caption Retrieval Attacks

**Conference**: ACL 2026 Findings  
**arXiv**: [2601.17644](https://arxiv.org/abs/2601.17644)  
**Code**: https://github.com/aliwister/mrag-attack-eval  
**Area**: LLM Security / Privacy / Multimodal RAG  
**Keywords**: Multimodal RAG, Membership Inference Attack, Image Caption Retrieval, Black-box Privacy, VLM

## TL;DR
The authors provide the first systematic evaluation of privacy leakage risks in **image-driven multimodal RAG (mRAG)** systems. They demonstrate that a naive black-box text prompt combined with a target image achieves an **MIA F1=0.993** and a **caption exact-match=0.835** across 4 datasets and 3 VLMs. Attacks remain effective even under image transformations (cropping, masking, rotation, noise). The relative position of "target vs. retrieved images" in prompts and cross-modal reranking are identified as key mitigation levers.

## Background & Motivation

**Background**: Multimodal RAG (mRAG) has become the mainstream method for providing VLMs with access to private image databases and annotations, widely used in VQA, medical imaging, and copyright protection. The pipeline typically consists of three stages: retriever $\to$ reranker $\to$ VLM.

**Limitations of Prior Work**: Compared to text-only RAG, privacy research in mRAG is scarce. Existing RAG MIA (e.g., S2MIA, Zeng et al. 2024) focuses only on pure text. Single-modality image work (Yang et al. 2025) relies on carefully designed occluded queries, limiting generalizability. Furthermore, **no prior work** has simultaneously evaluated both "presence in database (MIA)" and "leaked associated captions (ICR)," nor considered realistic scenarios where database images might be pre-processed (cropped or rotated).

**Key Challenge**: The design goal of mRAG is to "force the model to treat retrieved content as authoritative context." Structurally, this encourages the model to **repeat retrieved content verbatim**, which naturally conflicts with "not leaking private data" requirements. Design objectives and privacy goals compete within the same prompt.

**Goal**: (RQ1 MIA) Can an attacker determine if an image (or its transformed version) exists in a private database? (RQ2 ICR) If present, can the associated caption be extracted? The study also explores how prompt structures and retriever/reranker configurations affect the degree of leakage.

**Key Insight**: Assume a completely black-box, simplified attacker using only one image and a naive prompt, without prompt optimization or white-box assumptions. This measures the inherent **system-level** leakage floor of mRAG rather than the upper bound of attack algorithms.

**Core Idea**: "Sound the loudest alarm with the weakest attacker"—since naive prompts already achieve near-perfect F1 scores, it proves the issue lies in the design flaws of the mRAG pipeline itself rather than the ingenuity of the attack.

## Method

### Overall Architecture
The threat model is black-box: the attacker only provides `(target_image, text_prompt)` via API and reads the VLM's text output, without access to retriever embeddings, reranker scores, or VLM weights. The study evaluates two types of attacks: MIA (binary classification: Yes/No) and ICR (generative: caption string). Each attack is evaluated across a matrix of 4 datasets (Conceptual Captions, ROCOv2 Medical, Pokemon BLIP, MRAG-Bench) $\times$ 3 VLMs (Qwen2.5-VL 7B, Cosmos-Reason1 7B, InternVL3.5 8B) $\times$ 7 image transformations (Original, Crop, Mask, Blur, Cutout, Rotate, Gaussian Noise). The retriever uses CLIP + cosine similarity, and the reranker uses Jina-Reranker (image-image mode for MIA, image-text mode for ICR), with default $n=20, k=5$.

### Key Designs

1.  **Minimalist Prompt Design for Membership Inference Attack (MIA)**:
    *   **Function**: Black-box determination of whether a target image (or its transformed version) exists in the mRAG private database $\mathbb{R}_m=\{(i_j,c_j)\}_{j=1}^N$.
    *   **Mechanism**: The attack prompt directly asks the VLM: "the last image is identical to any of the retrieved images, in original or transformed form?" and uses the "Yes/No" output as the label. The pipeline is formalized as $\mathcal{R}(i_q)=\text{Top}_n \cos(f_\theta(i_q), f_\theta(i_j))$, $\mathcal{R}'(i_q)=\text{Top}_k \psi(i_q, i_j)$, and $y=G(i_q, \mathcal{R}'(i_q), \mathcal{P})$. When $i_q$ is in the database, the retriever almost certainly brings it into context. The VLM equates "identical image in context" with "presence in database," leading to leakage.
    *   **Design Motivation**: Deliberately avoiding prompt optimization ensures measured leakage is strictly attributed to the mRAG pipeline (biases in retrieve-rerank + VLM's context-copying tendency) rather than attack engineering.

2.  **Exploiting Verbatim Leakage in Image Caption Retrieval (ICR) Attacks**:
    *   **Function**: Extracting the paired original caption once a target image is confirmed to be in the database.
    *   **Mechanism**: The prompt instructs the VLM to "identify the input image in the retrieved context and return its caption verbatim." Because the cross-modal reranker (image-text) tends to place "image + corresponding description" pairs as strong candidates in the context, the VLM's maximum likelihood path when faced with 1 target image and $k$ pairs is to copy the corresponding caption. Metrics include exact-match, BLEU-2, ROUGE-1, and METEOR.
    *   **Design Motivation**: Even if the target image is not perfectly retrieved, VLMs often pick a "semantically similar" caption from context to repeat (termed "indirect leakage"), making exact-match a critical red line for mRAG privacy.

3.  **Robustness Testing and Prompt Position Ablation**:
    *   **Function**: Simulating databases with data augmentation or privacy mitigations to verify if physical transformations protect privacy.
    *   **Mechanism**: Six transformation attacks are tested: Crop (60% of original), Mask (grayscale), Blur, Cutout (4% rectangular occlusion), Rotate (90° or flip), and Gaussian Noise $\mathcal{N}(0, 25^2)$. Prompt structure ablations include RAG-First (retrieved images before target) vs. RAG-Last (target image first).
    *   **Design Motivation**: Identifying affordable deployment-level defenses—if simply shifting image positions or rotating database images reduces leakage, it is more practical than training private models.

### Loss & Training
This work is an evaluation/attack study and does not train any models. All results are from black-box inference, with means and variances calculated across 3 independent random seeds for each configuration.

## Key Experimental Results

### Main Results
**MIA leakage under exact image attacks (Table 1 excerpt, higher F1 indicates more severe leakage)**:

| Dataset | Qwen2.5-VL F1 | Cosmos-Reason1 F1 | InternVL3.5 F1 | RAG Acc |
| :--- | :--- | :--- | :--- | :--- |
| Conceptual Captions | 0.946 | **0.989** | 0.988 | 0.999 |
| ROCOv2 (Medical) | 0.893 | **0.952** | 0.897 | 0.995 |
| Pokemon BLIP | **0.993** | 0.983 | 0.908 | 1.000 |
| mRAG-Bench | 0.966 | **0.983** | 0.899 | 1.000 |

**ICR caption leakage under exact image attacks (Table 3 excerpt)**:

| Dataset | Model | Exact-Match | BLEU-2 | ROUGE-1 | METEOR | RAG Acc |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Conceptual Captions | Qwen2.5-VL | **0.835** | 0.853 | 0.882 | 0.875 | 0.892 |
| Conceptual Captions | Cosmos-Reason1 | 0.470 | 0.627 | 0.761 | 0.730 | 0.892 |
| Conceptual Captions | InternVL3.5 | 0.747 | 0.791 | 0.830 | 0.817 | 0.892 |
| ROCOv2 (Medical) | Qwen2.5-VL | 0.451 | 0.597 | 0.607 | 0.594 | 0.597 |
| Pokemon BLIP | Qwen2.5-VL | 0.743 | 0.794 | 0.852 | 0.828 | 0.753 |
| mRAG-Bench | Qwen2.5-VL | **0.801** | 0.794 | 0.819 | 0.539 | 0.823 |

MIA is nearly perfect (F1 $\ge 0.89$). ICR achieves over 70% verbatim retrieval on general datasets, while medical datasets show lower results due to lower retriever recall (0.597).

### Ablation Study

| Config | Key Metric | Description |
| :--- | :--- | :--- |
| MIA Exact (Conceptual Captions, Cosmos) | F1 = 0.989 | Baseline upper bound |
| MIA + Rotate Transform | F1 $\approx$ 0.60 | Rotation destroys spatial features; most effective defense |
| MIA + Crop/Mask/Blur/Cutout/Noise | F1 = 0.85–0.96 | Standard transforms provide little protection |
| MIA Default RAG-First | F1 $\approx$ 0.99 | Retrieved images first $\to$ model treats target as context |
| MIA RAG-Last (Target First) | F1 Significant Drop | Position bias $\to$ simple defense strategy |
| ICR $k=5$, Qwen on ROCOv2 | EM = 0.451 | Baseline with strong rerank |
| ICR $k=10$, Qwen on ROCOv2 | EM = 0.581 | Doubling context increases leakage |
| ICR $k=20$ ($\approx n$), Qwen on ROCOv2 | EM = 0.702 | No rerank, leakage reaches peak |

### Key Findings
- **MIA is nearly indefensible**: Under exact image attacks, F1 is consistently $\ge 0.89$ and RAG accuracy reaches 0.999. The VLM acts as an "amplifier" for the retriever.
- **Verbatim caption leakage is severe**: Qwen2.5-VL achieves 83.5% exact literal matching on Conceptual Captions, presenting a direct GDPR/HIPAA-level risk.
- **Image complexity is a natural defense**: In ROCOv2, retriever recall is only 0.597, dropping attack EM to 0.451. High intra-variability datasets provide "intrinsic noise" for mRAG privacy.
- **Prompt position is a cheap defense**: RAG-Last (placing target first) significantly lowers MIA success due to position bias, where the VLM treats the first image as a query rather than context.
- **Reranking is a double-edged sword**: Cross-modal reranking mitigates ICR (EM rises from 0.45 to 0.70 when $k=n$), but has little effect on MIA since the retriever already found the match.
- **Rotate is the most effective transform**: A 90° rotation disrupts CLIP embeddings, pulling F1 down to $\sim 0.60$, far lower than other transformations.

## Highlights & Insights
- **"Weakest Attacker, Loudest Alarm" Narrative**: By deliberately avoiding prompt optimization and white-box access, the high success rates demonstrate that the vulnerability is a design flaw in mRAG pipelines rather than a byproduct of attack engineering.
- **Unified ICR and MIA Framework**: The study completes the full attack chain—identifying presence then extracting details—corresponding to real-world threats in healthcare and copyright.
- **Transformation vs. Model x Data Matrix**: The dense 3x4x7 grid disproves the intuition that image transformations can protect privacy, with rotation being the only exception.
- **Position Bias as a Privacy Channel**: The phenomenon where VLMs prioritize the first-seen image as the query is converted into a deployable, zero-cost mitigation technique.

## Limitations & Future Work
- **Naive Prompts are Lower Bounds**: More complex prompt injection, multi-turn interactions, or Chain-of-Thought guidance could potentially increase leakage.
- **Limited Database Scale**: Database sizes are in the thousands; real-world production mRAG systems may scale to millions, and the leakage rate may not scale linearly.
- **Image-Text Pairs Only**: Real mRAG systems might pair multiple images with text or share libraries across users; privacy relationships are significantly more complex.
- **Lack of Formal DP Comparisons**: Only engineering-based mitigations (position/rotation/rerank) were discussed, without comparison to Differential Privacy (DP) retrievers or noisy captions.

## Related Work & Insights
- **vs. Zeng et al. 2024**: While they focus on text-only RAG MIA, this work extends to multimodal domains and image transformations, revealing cross-modal leakage paths.
- **vs. S2MIA (Li et al. 2025)**: This work adds ICR and proves that naive prompts suffice, questioning the necessity of complex shadow models for such attacks.
- **vs. Yang et al. 2025**: They rely on artificially occluded query images; this work provides a more systematic evaluation of 6 transformations and medical imaging, showing a broader attack surface.

## Rating
- Novelty: ⭐⭐⭐⭐ First to put MIA + ICR into a unified mRAG evaluation; algorithmic innovation is limited, but coverage is extensive.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 3 models $\times$ 4 datasets $\times$ 7 transforms $\times$ multiple ablations. A comprehensive baseline for mRAG privacy.
- Writing Quality: ⭐⭐⭐⭐ Clear problem statements with well-organized, readable tables.
- Value: ⭐⭐⭐⭐⭐ Directly addresses compliance risks for rapidly expanding mRAG applications in enterprise and medical fields.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Membership Inference Attacks on In-Context Learning Recommendation](membership_inference_attacks_on_llm-based_recommender_systems.md)
- [\[ICLR 2026\] No Caption, No Problem: Caption-Free Membership Inference via Model-Fitted Embeddings](../../ICLR2026/llm_safety/no_caption_no_problem_caption-free_membership_inference_via_model-fitted_embeddi.md)
- [\[ACL 2026\] LeakDojo: Decoding the Leakage Threats of RAG Systems](leakdojo_decoding_the_leakage_threats_of_rag_systems.md)
- [\[ACL 2026\] Fast-MIA: Efficient and Scalable Membership Inference for LLMs](fast-mia_efficient_and_scalable_membership_inference_for_llms.md)
- [\[ACL 2026\] Differentially Private Synthetic Text Generation for Retrieval-Augmented Generation (RAG)](differentially_private_synthetic_text_generation_for_retrieval-augmented_generat.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ACL 2026\] Membership Inference Attacks on In-Context Learning Recommendation](membership_inference_attacks_on_llm-based_recommender_systems.md)
- [\[ACL 2026\] LeakDojo: Decoding the Leakage Threats of RAG Systems](leakdojo_decoding_the_leakage_threats_of_rag_systems.md)
- [\[ICLR 2026\] No Caption, No Problem: Caption-Free Membership Inference via Model-Fitted Embeddings](../../ICLR2026/llm_safety/no_caption_no_problem_caption-free_membership_inference_via_model-fitted_embeddi.md)
- [\[ACL 2026\] Fast-MIA: Efficient and Scalable Membership Inference for LLMs](fast-mia_efficient_and_scalable_membership_inference_for_llms.md)
- [\[ACL 2026\] Differentially Private Synthetic Text Generation for Retrieval-Augmented Generation (RAG)](differentially_private_synthetic_text_generation_for_retrieval-augmented_generat.md)

</div>

<!-- RELATED:END -->
