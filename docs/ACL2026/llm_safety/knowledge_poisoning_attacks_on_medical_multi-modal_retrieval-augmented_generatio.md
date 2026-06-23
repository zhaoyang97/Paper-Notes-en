---
title: >-
  [Paper Note] Knowledge Poisoning Attacks on Medical Multi-Modal Retrieval-Augmented Generation
description: >-
  [ACL 2026][LLM Safety][Paper Note] The authors propose M3Att—the first **query-agnostic** knowledge poisoning framework for medical multi-modal RAG. It utilizes "distribution-guided visual PGD triggers" for retrieval hijacking and "clinical ambiguity-guided text rewriting" to bypass LVLM self-correction. With a poisoning rate of <1% (without querying kn
tags:
  - ACL 2026
  - LLM Safety
date: 2026-05-08
content_hash: 1cf061e5ed06d6b4
---
# Knowledge Poisoning Attacks on Medical Multi-Modal Retrieval-Augmented Generation

**Conference**: ACL 2026  
**arXiv**: [2605.10253](https://arxiv.org/abs/2605.10253)  
**Code**: https://github.com/ypr17/M3Att  
**Area**: LLM Security  
**Keywords**: Knowledge Poisoning, Medical RAG, PGD Perturbation, Clinical Ambiguity, Query-agnostic Attack

## TL;DR
The authors propose M3Att—the first **query-agnostic** knowledge poisoning framework for medical multi-modal RAG. It utilizes "distribution-guided visual PGD triggers" for retrieval hijacking and "clinical ambiguity-guided text rewriting" to bypass LVLM self-correction. With a poisoning rate of <1% (without querying knowledge, visual perturbation $\epsilon=16/255$), it reduces downstream utility by an average of 8.78% across 5 LVLMs × 5 datasets × 4 medical tasks, while remaining robust to three types of pre-retrieval defenses: image clustering, text clustering, and image-text consistency.

## Background & Motivation
**Background**: Medical multi-modal RAG systems (pairing retrieved images + reports) are being rapidly deployed. Models such as LLaVA-Med and Med-Gemini rely heavily on external knowledge bases to improve performance in tasks like VQA, report generation, and image classification. This has created a new attack surface: "poisoning the knowledge base." Ha et al. 2025, Liu et al. 2025b, and Zuo et al. 2025 have already demonstrated knowledge poisoning attacks on general or medical RAG.

**Limitations of Prior Work**: (1) Almost all existing multi-modal RAG poisoning methods assume a **query-aware** setting—attackers know in advance what questions users will ask and optimize poisoned entries accordingly. This is unrealistic in real deployments where user queries are typically unavailable. (2) Medical images (X-rays, histology slides) exhibit high anatomical consistency, with embedding distributions being highly clustered. Simply increasing the number of poisoned entries to ensure retrieval would expose the attacker. (3) State-of-the-art (SOTA) medical LVLMs, pre-trained on medical corpora and subject to safety alignment, will trigger refusal or self-correction when "obvious factual errors" are injected, while weak perturbations fail to affect generation. Finding a "dosage" that influences output without triggering self-correction is difficult.

**Key Challenge**: Query-aware attacks fail in real environments; however, under a query-agnostic setting, one faces the dual difficulties of "being submerged in dense embeddings during retrieval" and "LVLM prior self-correction during generation," which constitutes a dual-constraint problem.

**Goal**: (1) Construct a query-agnostic poisoning framework with weak priors (only knowing the database distribution, no queries required); (2) Design independent mechanisms for the retrieval and generation stages; (3) Demonstrate effectiveness across 5 LVLMs × 3 retrievers × 4 medical tasks and verify robustness against common pre-retrieval defenses.

**Key Insight**: (A) While the high homogeneity of medical images makes query-specific attacks difficult, it also results in a highly structured latent space where **cluster centers** can serve as "representative query proxies"—disturbing these centers can cover all unknown queries within the cluster. (B) Medical diagnosis inherently contains clinical ambiguity, such as "severe vs. mild," "differential diagnoses," and "defensive medicine," which correspond to low-confidence regions in LLM priors. By lying within these "gray areas," attackers make it difficult for the model to self-correct.

**Core Idea**: M3Att uses "distribution-guided visual PGD hijacking" to optimize poisoned images toward cluster centers as query-agnostic triggers. It further employs "clinical ambiguity-guided three-layer progressive text rewriting" to inject plausible but incorrect medical conclusions across three levels: severity migration, diagnostic distortion, and risk association. Together, these form a query-agnostic, stealthy, and dual-stage coupled medical RAG poisoning framework.

## Method

### Overall Architecture
M3Att aims to poison medical multi-modal RAG under a threat model closest to real-world deployment: the attacker lacks model parameters, user queries, and retrieval contexts, and can only insert less than 1% malicious entries into the knowledge base. The difficulty is twofold—the retrieval stage must ensure poisoned entries are captured by arbitrary future queries within dense embeddings, and the generation stage must deceive safety-aligned LVLMs. The pipeline consists of three steps: Cluster Profiling to obtain cluster centers as "representative query proxies," distribution-guided visual PGD to optimize images toward these centers for retrieval hijacking, and clinical ambiguity-guided text rewriting to inject "plausible but wrong" conclusions. The (poisoned image, poisoned text) pairs are inserted into the knowledge base. The visual and textual paths are tightly coupled—the former handles retrieval while the latter deceives generation, both utilizing white-box/black-box dual gradients to ensure efficacy on closed-source retrievers.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    KB["Medical Knowledge Base (Reference Pool Embeddings Highly Clustered)"]
    subgraph HIJACK["Distribution-Guided Retrieval Hijacking"]
        direction TB
        CP["Cluster Profiling<br/>K-Means K=40, take cluster center μc as query proxy"]
        CS["Candidate Sampling<br/>10-step PGD warm-up to pick optimal seed"]
        PGD["Constrained PGD Refinement<br/>ℓ∞ ≤ 16/255, maximize similarity to cluster center"]
        CP --> CS --> PGD
    end
    subgraph REWRITE["Clinical Ambiguity-Guided 3-Layer Progressive Text Rewriting"]
        direction TB
        L1["① Severity Migration<br/>massive ↔ moderate inducing misdiagnosis/over-intervention"]
        L2["② Diagnosis Distortion<br/>Pick visually overlapping differential diagnoses as replacements"]
        L3["③ Risk Association Corruption<br/>Suppress urgency / create false positives"]
        L1 --> L2 --> L3
    end
    KB --> HIJACK
    PGD -->|White-box BP / Black-box Zeroth-order| PIMG["Poisoned Image<br/>query-agnostic trigger"]
    REWRITE --> PTXT["Poisoned Text<br/>Plausible but incorrect medical conclusion"]
    PIMG --> PAIR["Dual-stage Coupled: (Poisoned Image, Poisoned Text) pair<br/>Insert into KB <1%"]
    PTXT --> PAIR
    PAIR --> TRIG["Real Query Natural Trigger<br/>Retrieval hijacking → LVLM generation poisoned"]
```

### Key Designs

**1. Distribution-Guided Retrieval Hijacking: Using Cluster Centers as Proxies to Cover All Queries**

The biggest obstacle for query-agnostic attacks is the lack of information regarding user queries. However, the high homogeneity of medical image embeddings can be exploited—since embeddings cluster together, cluster centers represent the semantics of the entire cluster. By perturbing images toward the cluster center, the attack covers all unknown queries within that cluster. Three steps are involved: Cluster Profiling performs K-Means (K=40) on the reference pool, averaging the top-50 nearest samples per cluster to obtain centers $\bm{\mu}_c$; Candidate Sampling ranks non-overlapping candidates and uses a 10-step PGD warm-up to evaluate optimization potential; Constrained PGD Refinement iterates for $N=500$ steps:

$$\bm{x}_c^{(i+1)} = \Pi_{\mathcal{B}_\epsilon}\!\left(\bm{x}_c^{(i)} + \alpha \cdot \mathrm{sign}\big(\nabla_x \mathcal{L}(f(\bm{x}_c^{(i)}), \bm{\mu}_{c})\big)\right)$$

maximizing cosine similarity with the cluster center under constraints $\ell_\infty \leq \epsilon=16/255$ and $\alpha=1/255$. White-box attacks use direct backpropagation, while black-box attacks use zeroth-order estimation via symmetric finite differences: $\nabla_x \mathcal{L} \approx \frac{1}{K}\sum_k \frac{\mathcal{L}(\bm{x}+\sigma u_k) - \mathcal{L}(\bm{x}-\sigma u_k)}{2\sigma} \cdot u_k$. Since cluster centers capture the data's semantic structure rather than model-specific features, this attack transfers across retrievers (CLIP/BGE-VL/SigLIP). The $\ell_\infty$ constraint ensures perturbations are nearly invisible to medical review.

**2. Clinical Ambiguity-Guided Three-Layer Progressive Text Rewriting: Targeting Low-Confidence Regions**

Naive injection of "obvious errors" triggers self-correction in safety-aligned medical LVLMs. The key insight is that medical diagnosis inherently contains ambiguity (e.g., "massive vs. moderate"), which resides in the gray areas of LLM priors. Using GPT-5 as a controlled editor, three strategies are executed via system prompts: **Fine-grained Severity Migration** bi-directionally modifies severity (down-scaling "massive" $\to$ "moderate" to induce under-diagnosis; up-scaling "unremarkable" $\to$ "suspicious density" to trigger over-intervention); **Prior-Constrained Diagnosis Distortion** selects target diseases with overlapping visual features and similar prior probabilities (e.g., "Viral Pneumonia" $\to$ "Pulmonary Edema"), leading the model to accept poisoned context as a legitimate differential diagnosis; **Risk Association Corruption** bi-directionally manipulates clinical recommendations (suppressing urgency: "immediate CT" $\to$ "follow-up in 6 months"). These correspond to the clinical reasoning stages of evidence $\to$ hypothesis $\to$ decision-making.

**3. Black-box/White-box Dual Gradient Paths + Dual-Stage Coupling**

To ensure efficacy on closed-source retrievers, visual hijacking utilizes both white-box backpropagation and black-box zeroth-order estimation. Experiments show black-box Attack Success Rates (ASR) are close to white-box ones. Furthermore, M3Att tightly couples retrieval hijacking with text injection—ablation shows that removing either lead to a significant recovery in downstream utility: without hijacking, poisoned entries fail to reach the top-k; without injection, harmless original text fails to influence generation even if retrieved.

### Loss & Training
The core loss is cosine similarity $\mathcal{L}(f(\bm{x}), \bm{\mu}_c) = \cos(f(\bm{x}), \bm{\mu}_c)$, constrained within $\bm{x} \in \mathcal{B}_\epsilon(\bm{x}^{(0)}) = \{\bm{x}: \|\bm{x} - \bm{x}^{(0)}\|_\infty \leq \epsilon\}$. Key hyperparameters: $K=40$ clusters, inserting 1 optimized candidate per cluster (poison rate < 0.01), $\epsilon=16/255$, $\alpha=1/255$, PGD 500 steps, warm-up 10 steps. Text editing is performed by GPT-5 with strict system prompts targeting stealthiness and progressive strategy.

## Key Experimental Results

### Main Results: End-to-End Attack Results on 5 LVLMs (Partial, Lower is Better)

| LVLM | Retriever | Method | True/False (IU-XRay) | MC (MIMIC) | Report FC (IU-XRay) | Img Cls (CRC100k) |
|------|-----------|--------|---------------------|------------|--------------------|--------------------|
| GPT-4o | – (w/o RAG) | – | 67.36% | 58.02% | 18.89% | 46.66% |
| GPT-4o | – (Clean RAG avg) | – | 89.64% | 69.57% | 31.04% | 93.30% |
| GPT-4o | CLIP | LIAR | 83.90% | 64.09% | 34.47% | 89.67% |
| GPT-4o | CLIP | **M3Att** | **77.88%** | **59.98%** | **32.39%** | **78.41%** |
| GPT-4o | BGE-VL | M3Att | 80.44% | 58.84% | 23.70% | 70.62% |
| GPT-5 | BGE-VL | M3Att | **93.54%** | **72.26%** | **35.11%** | **68.58%** |
| Claude-4.5 | CLIP | M3Att | **47.04%** | **61.41%** | **21.64%** | **69.28%** |
| LLaVA-Med | BGE-VL | M3Att | **46.56%** | **3.51%** | **17.04%** | **50.16%** |
| Gemini-2.5 | CLIP | M3Att | 76.12% | 39.21% | 32.40% | 79.85% |

M3Att outperforms the baseline LIAR in **most LVLM × Retriever × Task combinations**, reducing downstream utility from Clean RAG by an average of **8.78%**.

### Ablation Study

| Setting | Metric | Observation |
|------|---------|---------|
| Full M3Att | Full Effect | Strongest attack. |
| w/o Hijack (using nearest sample to cluster center) | Utility Recovers | Poisoned entries cannot reliably enter top-k. |
| w/o Injection (poisoned image, original text) | Utility Recovers | Harmless text does not affect generation even if retrieved. |
| Filtered (subset where retrieval succeeds) | M3Att > LIAR | Once retrieved, poisoned text effectively dominates generation. |
| Defense: Image Clustering | ASR Constant | Visual perturbation is too small to be anomalous. |
| Defense: Text Clustering | ASR Constant | GPT-5 rewritten text remains clinically fluent. |
| Defense: Image-Text Consistency | ASR Constant | Visual-text pairs remain highly aligned. |
| Poison rate 0.08 | ASR ≈ 100% | Low poison rates (<0.01) yield significant results. |
| $\epsilon$ Increase | ASR Saturates | Moderate perturbation is sufficient. |

### Key Findings
- **Query-agnostic poisoning is feasible in medical scenarios**: Relying on "cluster center proxies + PGD" without query information allows poisoned ASR@Top-5 to jump from 0.01% to 5%.
- **Black-box ≈ White-box**: Zeroth-order gradient estimation results are close to white-box, proving risks to closed-source retrievers.
- **Both stages are necessary**: Removing either hijack or injection significantly reduces efficacy, indicating that medical RAG attacks require retrieval and generation synergy.
- **Simple defenses fail**: Image/Text clustering and consistency checks cannot stop M3Att, suggesting a need for deeper medical fact-checking.
- **Clinical ambiguity as a natural surface**: Manipulating severity and risk levels allows the model to accept lies as "alternative interpretations."

## Highlights & Insights
- **Paradigm Shift (Homogeneity as Opportunity)**: The authors transform the "obstacle" of medical image homogeneity into a "lever," using a few cluster centers to cover a massive number of queries.
- **Clinical Ambiguity as Attack Surface**: Categorizing clinical reasoning into three progressive attack levels incorporates domain knowledge into adversarial design, transferable to other high-stakes domains like law or finance.
- **Double Attack Primitives**: Combining visual PGD on embeddings with LLM-as-editor provides a recipe applicable to almost all future multi-modal RAG attacks.
- **Black-box Evidence**: Proving vulnerability in closed-source medical RAG APIs pushes the threat model to production levels.
- **Defense Failure as a Baseline**: Demonstrating that simplicity-based filtering is insufficient provides a valuable red-teaming baseline for the trustworthy AI community.

## Limitations & Future Work
- **Limited to 2D Imaging**: Focuses on X-rays and histology, leaving 3D (CT/MRI) and video medical data for future work.
- **GPT-5 Dependency**: The quality of text rewriting relies on a strong editor LLM; weaker models may reduce efficacy.
- **Lack of Regulatory Detection Testing**: The study did not test against expert review panels or knowledge graph-based consistency checks.
- **Hyperparameter Tuning**: $K=40$ is empirical and may vary across different database scales and imaging types.
- **Future Directions**: (1) Expansion to 3D and temporal data; (2) Proposal of retrieval-stage defenses (e.g., leave-one-out detection); (3) Investigating if fine-tuned medical LVLMs are more or less robust.

## Related Work & Insights
- **vs. LIAR (Tan et al. 2024)**: M3Att extends text-only RAG poisoning to multi-modal, medical, and query-agnostic settings, showing consistently better performance.
- **vs. MM-PoisonRAG (Ha et al. 2025)**: M3Att is the first medical multi-modal method to remove the query-specific dependency.
- **vs. HV-Attack (Luo et al. 2025)**: Generic multi-modal attacks fail on homogeneous medical corpora; M3Att solves this with cluster proxies.
- **Cross-domain Transfer**: The "gray area attack" paradigm targeting clinical ambiguity is highly applicable to legal interpretation and financial advice RAG systems.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First practical threat model for medical multi-modal RAG poisoning; several innovative design points.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across LVLMs, retrievers, datasets, and tasks.
- Writing Quality: ⭐⭐⭐⭐ Clear formulas and tables; strategies have significant medical domain depth.
- Value: ⭐⭐⭐⭐⭐ Directly exposes vulnerabilities in deployed medical RAG systems, though the dual-use risk necessitates defensive research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Beyond Explicit Refusals: Soft-Failure Attacks on Retrieval-Augmented Generation](beyond_explicit_refusals_soft-failure_attacks_on_retrieval-augmented_generation.md)
- [\[ACL 2026\] Differentially Private Synthetic Text Generation for Retrieval-Augmented Generation (RAG)](differentially_private_synthetic_text_generation_for_retrieval-augmented_generat.md)
- [\[AAAI 2026\] Privacy-protected Retrieval-Augmented Generation for Knowledge Graph Question Answering](../../AAAI2026/llm_safety/privacy-protected_retrieval-augmented_generation_for_knowledge_graph_question_an.md)
- [\[ACL 2026\] MemoPhishAgent: Memory-Augmented Multi-Modal LLM Agent for Phishing URL Detection](memophishagent_memory-augmented_multi-modal_llm_agent_for_phishing_url_detection.md)
- [\[ACL 2026\] Retrievals Can Be Detrimental: Unveiling the Backdoor Vulnerability of Retrieval-Augmented Diffusion Models](retrievals_can_be_detrimental_unveiling_the_backdoor_vulnerability_of_retrieval-.md)

</div>

<!-- RELATED:END -->
