---
title: >-
  [Paper Note] Overcoming Copyright Barriers in Corpus Distribution Through Non-Reversible Hashing
description: >-
  [ACL2026][LLM/NLP][Non-reversible Hashing] This paper proposes novelshare: transforming tokens of copyrighted text into truncated non-reversible hashes and publishing only the hash sequences alongside researcher-owned an…
tags:
  - "ACL2026"
  - "LLM/NLP"
  - "Non-reversible Hashing"
  - "Copyrighted Text"
  - "Corpus Distribution"
  - "Sequence Alignment"
  - "Annotation Sharing"
date: 2026-05-08
content_hash: bf307ad450400307
---

# Overcoming Copyright Barriers in Corpus Distribution Through Non-Reversible Hashing

**Conference**: ACL2026  
**arXiv**: [2604.23412](https://arxiv.org/abs/2604.23412)  
**Code**: https://github.com/CompNet/novelshare  
**Area**: NLP Corpus Sharing / Literary Text Processing / Copyright-Compliant Data Distribution  
**Keywords**: Non-reversible Hashing, Copyrighted Text, Corpus Distribution, Sequence Alignment, Annotation Sharing

## TL;DR
This paper proposes novelshare: transforming tokens of copyrighted text into truncated non-reversible hashes and publishing only the hash sequences alongside researcher-owned annotations. This allows users possessing legitimate original texts to re-align annotations under slight version differences, achieving token alignment accuracy rates between 98.7% and 99.79% on close-version novels.

## Background & Motivation
**Background**: NLP research relies heavily on annotated corpora, especially for tasks such as NER, POS, coreference resolution, character network extraction, and quotation attribution. For literary and long-form narrative texts, complete works are more valuable than fragments because characters, events, narrative perspectives, and long-range dependencies often unfold across chapters.

**Limitations of Prior Work**: A significant portion of realistically representative text remains under copyright protection. Academic teams using only public domain texts systematically bias results toward 19th-century or older works. Conversely, not releasing copyrighted corpora prevents replication, while releasing only excerpts fails to support full-length text tasks. Individually seeking permission from authors is the most secure route but lacks scalability and incurs high costs.

**Key Challenge**: Researchers wish to share their generated annotations rather than the original literary works. However, traditional corpus distribution formats typically bind original tokens and labels together. Furthermore, even if users own the same work, they may not possess an electronic version identical to the creator's due to revisions, OCR errors, punctuation standards, chapter splitting, and tokenizer differences.

**Goal**: The authors aim to design a corpus-sharing mechanism that allows creators to disclose token-level annotations without disclosing readable original text. Users must independently hold the original text to reconstruct and map labels to token sequences locally. The system must tolerate reasonable minor differences rather than requiring character-for-character identity between versions.

**Key Insight**: Building on the hashing alignment approach of Bost et al. (2020), this work generalizes it into a universal solution for arbitrary token-level sequential labeling. The key observation is that if only truncated hashes are disclosed, external attackers cannot directly read the text. However, users owning a similar original text can apply the same hashing to their tokens and map published annotations back through sequence alignment.

**Core Idea**: Replace "original text publication" with "truncated non-reversible hashing + robust sequence alignment + minor difference recovery strategies," establishing an operational intermediate layer between copyright compliance and corpus reproducibility.

## Method
The methodology presented is not a training paradigm for an NLP model but a corpus distribution protocol and alignment algorithm. The core problem addresses a creator with a copyrighted token sequence $X=(x_1, \dots, x_n)$ and corresponding annotations, and a user with a legally obtained similar text $X'=(x'_1, \dots, x'_m)$. The system must map annotations from $X$ to $X'$ accurately without leaking $X$.

### Overall Architecture
The workflow is divided into creator-side and user-side components.

On the creator side, the original text is tokenized, maintaining one or more annotation sequences of equal length. For example, NER tokens correspond to BIO tags; the same representation applies to POS tagging, chunking, slot filling, or coreference. The creator applies SHA-256 to each token and truncates the resulting hash, publishing only the truncated hash sequence $f(X)$ and annotations.

On the user side, the user must possess the same work or a sufficiently close version. The user processes their token sequence using the same tokenizer and hashing rules to obtain $f(X')$. The system then performs sequence alignment between $f(X)$ and $f(X')$. If a creator hash matches a user hash, the annotation from the creator's position is transferred to the user's token.

To handle discrepancies caused by additions, deletions, or modifications, various remediation strategies are applied after initial alignment, including propagation based on repeated hashes, tokenization fixes, case normalization, and MLM predictions. The final output is the user's local "plaintext tokens + aligned annotations," while the public package remains "hashed tokens + annotations."

### Key Designs
1. **Truncated SHA-256 Hashing as Published Text Proxy**:
    - **Function**: Converts copyrighted token sequences into unreadable technical identifiers, allowing creators to publish annotations and hashes without releasing original text.
    - **Mechanism**: Each token is hashed via SHA-256, and the result is truncated to a small number of characters. While full SHA-256 rarely collides, it is vulnerable to pre-computation attacks using vocabulary tables. Truncation ensures multiple tokens share the same short hash, so even if an attacker knows the vocabulary, they only obtain a set of candidate words rather than a confirmed token.
    - **Design Motivation**: Collisions here are part of the security mechanism rather than errors. The paper balances two objectives: long hashes weaken security, while short hashes increase alignment errors. A length of 2 was chosen as the primary setting to provide sufficient candidate collisions while maintaining acceptable alignment quality.

2. **Robust Alignment Based on Hash Sequences**:
    - **Function**: Maps creator annotations to the user's text version, allowing for minor discrepancies.
    - **Mechanism**: The system utilizes an enhanced Gestalt pattern matching algorithm (via Python's `difflib`) for sequence alignment between $f(X)$ and $f(X')$. Positions are directly matched only when truncated hashes are identical. To reduce quadratic complexity on long novels, the system aligns by chapter if structures match; otherwise, it aligns the entire book.
    - **Design Motivation**: User versions may have extra tokens, missing tokens, or local substitutions. Global alignment using context order resolves short-hash collisions in most cases through neighboring tokens. Chapter-wise processing leverages the natural structure of literary texts to prevent runtime explosion.

3. **Remediation Strategy Suite for Unaligned Positions**:
    - **Function**: Fixes gaps caused by version differences, tokenization issues, casing, and OCR/spelling errors after initial alignment.
    - **Mechanism**: The `propagate` strategy uses previously aligned identical hashes as voting sources to infer unaligned positions. The `retokenize` strategy explores splits or merges of user tokens to check if re-hashing matches the creator's hash. The `case` strategy tries case variants. The `mlm` strategy inserts a `[MASK]` in the user context and uses ModernBERT-base to predict candidates, verifying if their hash matches the creator's. A `pipe` meta-strategy executes these in the order: retokenize, mlm, case, propagate.
    - **Design Motivation**: Different errors require different mechanisms. `retokenize` handles tokenizer inconsistency, `case` addresses capitalization, `mlm` fills gaps using local context, and `propagate` exploits redundancy in long texts. Higher-precision strategies are applied first to fix reliable errors and minimize error propagation.

### Loss & Training
No main model is trained, and there is no end-to-end loss function. The only pre-trained model involved is in the `mlm` strategy: ModernBERT-base is used with a window size of 32 to predict missing tokens. It does not generate text from scratch but only accepts results where the predicted token's hash matches the creator's hash.

For evaluation, early editions of novels serve as the creator version, while more recent or distant editions serve as user versions. Main experiments use a hash length of 2, `difflib` for alignment, and the `pipe` sequence. This reflects a conservative principle: the system should only succeed if the user already holds a sufficiently close text.

## Key Experimental Results

### Main Results
The main experiments simulate copyrighted text sharing using three public domain novels: Mary Shelley’s *Frankenstein*, Herman Melville’s *Moby Dick*, and Jane Austen’s *Pride and Prejudice*. Three editions are selected for each: the earliest as the creator version and two others as user versions (one "closer" to the original, one "more distant"). The metric is token alignment error.

| Subject | User Version Relationship | Best Strategy/Setting | Main Result | Description |
|--------|-------------|--------------|---------|------|
| Close editions (3 novels) | User has a version near the creator's | pipe, hash length 2 | 98.7% to 99.79% accuracy, ~0.21% to 1.3% error | Demonstrates requirements are met with close versions, not identical digital files |
| Distant editions (3 novels) | User has revised, edited, or modernized version | pipe, hash length 2 | Significantly higher error; up to ~8% incorrect tokens | Shows the method does not "force" alignment on overly divergent texts |
| Identical text | User text matches creator text | Any reasonable strategy | Complete alignment | Serves as an upper bound to show hashing does not hinder legitimate use |
| NER Application Example | Mapping entity annotations via hash alignment | novelshare pipeline | 96.48% entity alignment | Shows token alignment errors propagate to specific NLP tasks but remain usable |

The crux is not pursuing 100% alignment but proving a workable range between security and utility. Low error rates on close versions ensure users get usable annotations; high error rates on distant versions ensure the system does not allow unauthorized reconstruction from unrelated text.

### Ablation Study
Rather than traditional neural network ablation, the analysis focuses on hash length, remediation strategies, and synthetic errors.

| Configuration / Dimension | Key Metric/Trend | Description |
|------|---------|------|
| Hash length 1 | ~1907.06 collisions per token | Strong security, but excessive collisions increase alignment errors |
| Hash length 2 | ~118.25 collisions per token | The chosen trade-off for main experiments, balancing confusion and reliability |
| Hash length 3 | ~6.45 collisions per token | Better alignment, but smaller candidate space for attackers |
| Longer hashes (e.g., 4 or 64) | Near-zero collisions | Easier alignment, but deemed insufficient security by authors |
| case / retokenize alone | Limited improvement | Only covers specific error sources |
| mlm / propagate alone | Usually outperforms case/retokenize | Leverages context or redundancy; broader coverage |
| pipe meta-strategy | Best overall across versions and synthetic errors | Complementary strategies; priority given to high-precision fixes |
| Moderate OCR error (WER=0.2) | 6.66% error with pipe | Usable for light OCR issues; fails under heavy OCR corruption |

### Key Findings
- **Hash length is the core knob for security and accuracy**: Shorter lengths increase collisions and security against reverse-engineering but confuse alignment. Length 2 is a practical compromise identified in the novel experiments.
- **Text version proximity is more fundamental than strategy choice**: Error rates remain below 1.3% for close editions but degrade significantly for heavily revised ones. This aligns with copyright logic: only owners of the "near-original" should succeed.
- **The pipe combination outperforms individual modules**: `retokenize` handles splits, `mlm` handles gaps, and `propagate` handles repetition. Their combination covers a wider range of real-world differences.
- **Runtime involves trade-offs**: `case`, `retokenize`, and `propagate` usually finish within 10 seconds, while `mlm` and `pipe` (due to GPU calls) can take up to an hour for long texts.
- **Synthetic errors validate boundaries**: Adding tokens has little impact (extra tokens are discarded), while deletions, substitutions, and OCR errors are harder to fix. Heavy OCR failure reminds users that high-quality text versions are still required.

## Highlights & Insights
- **Redefining "cannot publish original" as "publish unreadable index"**: This method does not bypass copyright but explicitly requires local possession of the text. This makes it more practical for reproducible research than snippets or anonymized text.
- **Active utilization of collisions in truncated hashing**: While many systems view collisions as errors to avoid, this method uses them as a protective layer against pre-computation attacks. Contextual sequences still allow legitimate users to align correctly.
- **Clear engineering boundaries**: Authors emphasize the system should fail if differences are too large. "Failure as a feature" is critical for copyright compliance, as excessive reconstruction capability would undermine the non-reversible argument.
- **Extensibility beyond NER**: Any task that can be projected onto a token sequence (POS, chunking, slot filling, etc.) can use this format.
- **Implications for the LLM era**: While not a training data solution, it provides a path for "disclosing derived annotations without disclosing creative expression," suitable for academic reproducibility.

## Limitations & Future Work
- **Heavy reliance on version proximity**: Highly revised, abridged, or poor OCR versions result in rapid error increases. This is legally beneficial but requires creators to specify publishers, editions, and preprocessing details.
- **Tokenization sensitivity**: Traditional tokenization is used. Subword tokenization might better handle OCR/spelling errors by localizing them rather than causing entire word mismatch.
- **MLM strategy costs and legal boundaries**: ModernBERT aids recovery but is computationally expensive. Over-reliance on model generation risk crossing into "reconstruction." Current methods mitigate this via hash verification.
- **Impact on downstream tasks**: Token error rates do not map linearly to task loss. Cross-span event labeling or coreference chains might be more fragile than NER.
- **Legal analysis is risk minimization, not absolute guarantee**: While arguing hashes/labels are not recognizable expression under EU/US/UK laws, risks vary across jurisdictions and work types.

## Related Work & Insights
- **vs. Public Domain Only**: LitBank, PDNC, etc., avoid copyright but bias research toward older styles. Ours allows processing modern copyrighted texts.
- **vs. No Data or Excerpts**: No data hurts replication; excerpts fail full-text tasks. novelshare supports full works provided users have the original.
- **vs. Author Permission**: Scaling is difficult for individual permissions. novelshare provides an engineering path for cases where permission is impractical but the user has legal access.
- **vs. Bost et al. (2020)**: Extends "Serial Speakers" (TV dialogue) by parameterizing hash length, generalizing to any token sequence, and adding sophisticated remediation strategies.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Built on existing hash alignment but systematized as a copyright protocol with recovery strategies and legal grounding.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers real versions, synthetic errors, and various parameters; could benefit from more task-level numeric evaluations.
- Writing Quality: ⭐⭐⭐⭐☆ Clear structure and motivation; some experimental results would be easier to compare with more tables.
- Value: ⭐⭐⭐⭐⭐ Extremely practical for literary NLP and academic data sharing; code release increases real-world utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Text-to-Distribution Prediction with Quantile Tokens and Neighbor Context](text-to-distribution_prediction_with_quantile_tokens_and_neighbor_context.md)
- [\[ACL 2026\] Unlocking the Potential of Diffusion Language Models through Template Infilling](unlocking_the_potential_of_diffusion_language_models_through_template_infilling.md)
- [\[ACL 2026\] LoCar: Localization-Aware Evaluation of In-Vehicle Assistants through Fine-Grained Sociolinguistic Control](locar_localization-aware_evaluation_of_in-vehicle_assistants_through_fine-graine.md)
- [\[ACL 2026\] Not All Animals Are Equal: Metaphorical Framing through Source Domains and Semantic Frames](not_all_animals_are_equal_metaphorical_framing_through_source_domains_and_semant.md)
- [\[ICML 2026\] A Geometric Relation of the Error Introduced by Sampling a Language Model's Output Distribution to its Internal State](../../ICML2026/llm_nlp/a_geometric_relation_of_the_error_introduced_by_sampling_a_language_models_outpu.md)

</div>

<!-- RELATED:END -->
