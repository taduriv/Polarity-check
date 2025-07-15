# CLIPEval Implicit Polarity of Events
## 🔍 What Are Hidden (Implicit) Sentiments of Events?

- Hidden or **implicit sentiments** are emotions expressed **without using direct sentiment words** (like "happy" or "sad").
- The sentiment is inferred from the **event and context**.
- **Example**:  
  - *"He lost his job."* → Negative sentiment, even though no negative word is used.

---

## ⚠️ Why Is This a Challenge?

- **No explicit sentiment clues** — traditional models rely on sentiment words.
- **Context-dependent** — the same event can have different meanings in different situations.
- **Requires background knowledge** — understanding real-world implications of events.
- **Subtle expressions** — irony or indirect language is difficult for machines to interpret.

---

## 🎯 Project Objective

- Use the **CLIPEval** framework to **detect and classify implicit sentiment** of events as **positive, negative, or neutral**.
- Focus on **context-aware analysis**, without relying on obvious emotional keywords.
- Improve NLP systems' ability to understand **emotional tone and intent** behind real-world events.

## ✅ Current Output

NLPCW1.ipynb trains a supervised NLP model that:

1. **Event detection** – classifies each sentence into one of eight event types  
   (`ATTENDING_EVENT`, `COMMUNICATION_ISSUE`, `GOING_TO_PLACES`, `LEGAL_ISSUE`,  
   `MONEY_ISSUE`, `OUTDOOR_ACTIVITY`, `PERSONAL_CARE`, `PHYSICAL_PAIN`).

2. **Implicit polarity prediction** – labels the same sentence as **Positive**,  
   **Negative** or **Neutral**, capturing sentiment that is *not* stated with  
   explicit emotion words (the focus of SemEval‑2015 Task 9 CLIPEval). :contentReference[oaicite:0]{index=0}

3. **End‑to‑end inference** – returns *(event, sentiment)* pairs for arbitrary text  
   via a single `predict(sentence)` helper.

---

## 🚀 Planned Enhancements & Developments

| Area | Why it Matters | Proposed Work |
|------|----------------|---------------|
| **Model architecture** | CNN/RNN baseline shows good recall, but struggles with subtle context. | Fine‑tune a lightweight transformer (e.g. DistilRoBERTa) in a *multi‑task* setup so the encoder is shared for both event and polarity heads. |
| **Context windows** | Some event cues span multiple sentences. | Add a sliding‑window inference option that feeds neighbouring sentences to the model. |
| **Knowledge integration** | Implicit sentiment often relies on world knowledge (“won the lawsuit” ⇒ positive). | Incorporate commonsense signals (ConceptNet, ATOMIC) via embedding concatenation or retrieval‑augmented generation. |
| **Class imbalance handling** | Minority events (e.g. `LEGAL_ISSUE`) lower micro‑F1. | Use focal loss or class‑balanced re‑sampling and monitor macro‑F1 in CI. |
| **Explainability** | HR / policy users need to trust predictions. | Add SHAP or integrated‑gradients plots in a `/reports` folder; surface top‑k tokens driving the decision. |
| **Real‑time demo** | Stakeholders benefit from a sandbox. | Deploy a Streamlit front‑end on Hugging Face Spaces with REST endpoint. |
| **Robust evaluation** | Current split is random; risk of lexical memorisation. | Re‑split by **story** (all sentences from the same source stay in one fold) and report macro / micro metrics. |
| **MLOps pipeline** | Manual notebook runs hinder reproducibility. | Add a `dvc.yaml` pipeline and GitHub Actions workflow for data prep → train → test → deploy. |
| **Extended sentiment granularity** | Some domains need finer tone detection (e.g. *mixed*, *sarcastic*). | Explore ordinal regression or multi‑label polarity (“joy”, “anger”) on top of CLIPEval labels. |
| **Dataset expansion** | 1 271 sentences limit generalisation. | Augment with weakly‑labelled Reddit / news data filtered by event patterns; validate with active learning. |


