---
title: >-
  [Paper Note] Automated Explanation Generation and Hallucination Detection for Heritage Image Retrieval
description: >-
  [ACL 2025][Hallucination Detection][Cultural Heritage Image Retrieval] This paper proposes a framework combining automated explanation generation and hallucination detection for cultural heritage image retrieval. It utilizes vision-language models to generate explainable text descriptions for retrieval results, while ensuring the factual accuracy of descriptions through a domain-knowledge-constrained hallucination detection mechanism, validating the effectiveness of the metho…
tags:
  - "ACL 2025"
  - "Hallucination Detection"
  - "Cultural Heritage Image Retrieval"
  - "Automated Explanation Generation"
  - "Vision-Language Models"
  - "Cross-Modal Retrieval"
date: 2026-05-08
content_hash: 631ad0c81d25fa15
---

# Automated Explanation Generation and Hallucination Detection for Heritage Image Retrieval

**Conference**: ACL 2025  
**Code**: None  
**Area**: Hallucination Detection  
**Keywords**: Cultural Heritage Image Retrieval, Automated Explanation Generation, Hallucination Detection, Vision-Language Models, Cross-Modal Retrieval

## TL;DR
This paper proposes a framework combining automated explanation generation and hallucination detection for cultural heritage image retrieval. It utilizes vision-language models to generate explainable text descriptions for retrieval results, while ensuring the factual accuracy of descriptions through a domain-knowledge-constrained hallucination detection mechanism, validating the effectiveness of the method on multiple cultural heritage datasets.

## Background & Motivation

**Background**: Cultural heritage digitization is an important application direction for computer vision and NLP. Museums, libraries, and archives possess vast amounts of cultural heritage images (e.g., paintings, sculptures, architectural photos, manuscripts), requiring efficient retrieval systems to help researchers and the public find specific content. Existing cultural heritage image retrieval mainly relies on visual feature-based methods (such as matching global features extracted by CNNs/ViTs) or text retrieval based on manually annotated metadata (such as artist, era, and style labels).

**Limitations of Prior Work**: Traditional image retrieval systems only return a ranked list of retrieval results without informing users "why this image was retrieved," thus lacking explainability. This issue is particularly prominent in the cultural heritage domain: for instance, when a researcher searches for "Baroque-style Madonna images," the system returns a painting, but the researcher needs to know whether the matching reason is due to its compositional style, color usage, or thematic content. Furthermore, when utilizing VLMs to generate image descriptions, models are prone to hallucinations—such as incorrectly describing a Renaissance-era artwork as "Baroque style," or fabricating non-existent details.

**Key Challenge**: The cultural heritage domain demands highly accurate descriptions (where metadata such as temporal era, artistic movement, and materials must be precise). However, general-purpose VLMs exhibit high hallucination rates in this domain due to the lack of sufficient cultural heritage annotations in their pre-training data compared to general scenarios.

**Goal**: (1) Automatically generate explainable text descriptions for cultural heritage image retrieval results; (2) design a domain-adaptive hallucination detection mechanism to ensure the factual accuracy of the descriptions.

**Key Insight**: The authors decompose the task into a two-stage approach of "generation first, verification later"—initially leveraging the generative capabilities of VLMs to obtain rich text descriptions, followed by post-hoc hallucination correction utilizing cultural heritage knowledge bases.

**Core Idea**: Safely transfer the generative power of general-purpose VLMs to the cultural heritage domain, which demands high precision, through a domain knowledge graph-constrained hallucination detection-correction loop.

## Method

### Overall Architecture
The framework consists of three core modules: (1) a cross-modal retrieval module, which aligns and retrieves images and text based on vision-language models like CLIP; (2) an explanation generation module, which utilizes multimodal LLMs to generate structured textual explanations (including visual description, style analysis, and matching rationale) for the retrieval results; and (3) a hallucination detection and correction module, which performs fact-checking and correction on the generated descriptions using a cultural heritage knowledge graph. The input is a user query (text or image), and the outputs are the ranked retrieval results along with an explainable text description for each result.

### Key Designs

1. **Domain-Adapted Cross-Modal Retrieval (DACR)**:

    - **Function**: Improve the retrieval precision of general vision-language models in the cultural heritage domain.
    - **Mechanism**: Based on a pre-trained CLIP model, lightweight fine-tuning (LoRA, adjusting only about $2\%$ of the parameters) is performed using image-text pairs from the cultural heritage domain. Training data is sourced from multiple public museum databases (e.g., WikiArt, Met Museum Open Access, Europeana). The key lies in constructing high-quality fine-tuning data: in addition to utilizing existing metadata-image pairs, GPT-4V is leveraged to generate rich descriptions for cultural heritage images, which are integrated into the training set after human verification. A domain-specific negative mining strategy is introduced, selecting easily confused image pairs from the same or adjacent eras/styles as hard negatives to force the model to learn fine-grained stylistic and content distinctions.
    - **Design Motivation**: The retrieval performance of general-purpose CLIP in the cultural heritage domain is significantly lower than in general scenarios (a gap of approximately $20$ percentage points) due to the extremely low proportion of cultural heritage content in its pre-training data.

2. **Structured Explanation Generator (SEG)**:

    - **Function**: Generate multi-dimensional explainable text for each retrieval result.
    - **Mechanism**: Multimodal LLMs (such as LLaVA or GPT-4V) are employed to generate structured explanations for each retrieved result across four dimensions: (a) visual description (describing visible visual elements like subject, composition, and color); (b) style analysis (identifying artistic styles, movements, and techniques); (c) matching rationale (explaining why the image matches the query based on shared visual/semantic features); and (d) metadata inference (inferring possible creation era, origin, and materials). A structured prompt template guides the LLM to output in a fixed format, with approximately $3$-$5$ sentences per dimension. During generation, a low temperature ($0.3$) is intentionally set to reduce randomness and creativity (where creativity equals hallucination in this context).
    - **Design Motivation**: Unstructured free-form descriptions are difficult to systematically verify and present; structured outputs allow each dimension to be subjected to hallucination detection independently.

3. **KG-Constrained Hallucination Detection and Correction (KGHDC)**:

    - **Function**: Detect and correct factual errors in the generated descriptions.
    - **Mechanism**: A cultural heritage domain knowledge graph (Heritage-KG) is constructed, containing timelines of art styles (e.g., the start and end periods of the "Baroque style"), compatibility relations between materials and techniques (e.g., "oil canvas" cannot appear on Song Dynasty Chinese paintings), and associations between themes and genres (e.g., "biblical themes" are more common in Western religious paintings). Hallucination detection adopts a three-step strategy: (a) entity extraction—extracting all factual assertions (era, style, material, etc.) from the generated descriptions; (b) knowledge graph verification—performing consistency checks on the extracted assertions against facts in Heritage-KG; and (c) correction recommendation—retrieving the most likely correct information from the KG to replace inconsistent assertions. The corrected descriptions are marked with correction traces (e.g., "[Original: Baroque $\rightarrow$ Corrected: Rococo]") to help users assess the trustworthiness of the descriptions.
    - **Design Motivation**: General-purpose VLMs exhibit hallucination rates of $30\%$-$40\%$ in the cultural heritage domain. Directly displaying unverified descriptions is unacceptable in academic scenarios.

### Loss & Training
The LoRA fine-tuning of the cross-modal retrieval module uses the InfoNCE contrastive learning loss, with a batch size of $256$, a learning rate of $1\text{e-}4$, and training for $10$ epochs. The hallucination detection module does not require training and operates based on rules and knowledge graph matching. Explanation generation leverages the zero-shot capabilities of pre-trained VLMs.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours (DACR+SEG+KGHDC) | Original CLIP | DACR only | Recall@10 |
|--------|------|---------------------|---------|----------|----------|
| WikiArt | mAP | $68.5$ | $48.2$ | $65.3$ | $79.2$ |
| Met-OA | mAP | $62.3$ | $41.7$ | $59.1$ | $73.8$ |
| Europeana | mAP | $58.7$ | $38.5$ | $55.2$ | $71.5$ |
| Explanation Acceptability | Human eval | $84.6\%$ | - | - | - |
| Hallucination Detection Rate | Precision | $78.3\%$ | - | - | - |

### Ablation Study

| Configuration | Explanation Acceptability | Hallucination Rate | Description |
|------|-----------|--------|------|
| SEG + KGHDC | $84.6\%$ | $12.3\%$ | Full approach |
| SEG only (w/o hallucination detection) | $68.2\%$ | $35.7\%$ | High hallucination rate, unacceptable |
| SEG + Simple rule detection | $76.5\%$ | $21.8\%$ | Simple rules are not precise enough |
| SEG + KGHDC (w/o correction) | $79.1\%$ | $18.5\%$ | Detection only without correction |
| Low temperature ($0.1$) | $80.3\%$ | $15.2\%$ | More conservative generation but monotonous descriptions |
| High temperature ($0.7$) | $62.1\%$ | $42.3\%$ | Rich generation but severe hallucination |

### Key Findings
- Knowledge graph-constrained hallucination detection reduces the hallucination rate from $35.7\%$ to $12.3\%$, a decrease of about $23$ percentage points, demonstrating that domain knowledge is crucial for factual correction.
- An explanation acceptability of $84.6\%$ means that approximately $85\%$ of the generated descriptions are judged as "usable" by human experts after hallucination detection, which is a promising result in a specialized domain.
- Domain-adaptive fine-tuning (DACR) improves retrieval mAP by approximately $20$ percentage points ($48.2 \rightarrow 68.5$), indicating a significant gap between general models and the cultural heritage domain.
- Temperature is the most sensitive hyperparameter affecting the hallucination rate (increasing from $0.1$ to $0.7$ causes the hallucination rate to surge from $15\%$ to $42\%$); a low temperature is necessary for scenarios requiring high accuracy.

## Highlights & Insights
- The two-stage strategy of "generation first, verification later" is highly generalizable for VLM applications in specialized fields. It allows the system to utilize the strong generative capabilities of VLMs while employing domain knowledge as a safety net to prevent hallucinations. This paradigm can be migrated to other domains requiring high accuracy, such as medical imaging reports and legal document analysis.
- The construction of the cultural heritage knowledge graph (Heritage-KG) is itself a valuable contribution that can be reused by other AI research in the cultural heritage domain.
- The design of marking correction traces reflects a commitment to user trust. Allowing users to see what content was corrected is more conducive to building trust than simply providing the corrected results directly.

## Limitations & Future Work
- The coverage of Heritage-KG is limited, mainly focusing on Western art, with insufficient knowledge regarding Asian and African cultural heritage.
- The hallucination detection precision of $78.3\%$ means that there are still about $22\%$ false positives (marking correct descriptions as hallucinations), which may lead to the loss of accurate information.
- When the generated descriptions involve subjective judgments (e.g., "this painting expresses melancholy emotions"), the knowledge graph cannot perform factual verification.
- Future work could introduce expert feedback loops to continuously improve hallucination detection accuracy and expand the coverage of Heritage-KG.

## Related Work & Insights
- **vs CHIA (Cultural Heritage Image Analysis)**: Traditional cultural heritage image analysis focuses on classification and detection rather than explanation generation; this paper is the first to introduce explainability into cultural heritage retrieval.
- **vs GRACE (Grounded Retrieval and Caption Enhancement)**: GRACE generates captions for image retrieval but lacks domain-specific hallucination detection; the KGHDC module in this work is the key differentiator.
- **vs Woodpecker (Yin et al., 2023)**: Woodpecker proposes a domain-agnostic framework for VLM hallucination detection; this work specializes hallucination detection to the cultural heritage domain by leveraging a domain-specific KG for more precise verification.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of cultural heritage, explainable retrieval, and hallucination detection is a novel perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dataset evaluation, including human expert assessment and a comprehensive ablation analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition and vivid descriptions of the application scenarios.
- Value: ⭐⭐⭐⭐ Holds practical value for cultural heritage digitization and more reliable VLM deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] REFIND at SemEval-2025 Task 3: Retrieval-Augmented Factuality Hallucination Detection in Large Language Models](refind_at_semeval-2025_task_3_retrieval-augmented_factuality_hallucination_detec.md)
- [\[ACL 2025\] Learning Auxiliary Tasks Improves Reference-Free Hallucination Detection in Open-Domain Long-Form Generation](learning_auxiliary_tasks_improves_reference-free_hallucination_detection_in_open.md)
- [\[ACL 2026\] Stable-RAG: Mitigating Retrieval-Permutation-Induced Hallucinations in Retrieval-Augmented Generation](../../ACL2026/hallucination/stable-rag_mitigating_retrieval-permutation-induced_hallucinations_in_retrieval-.md)
- [\[ICLR 2026\] GHOST: Hallucination-Inducing Image Generation for Multimodal LLMs](../../ICLR2026/hallucination/ghost_hallucination-inducing_image_generation_for_multimodal_llms.md)
- [\[ACL 2025\] Monitoring Decoding: Mitigating Hallucination via Evaluating the Factuality of Partial Response during Generation](monitoring_decoding_mitigating_hallucination_via_evaluating_the_factuality_of_pa.md)

</div>

<!-- RELATED:END -->
