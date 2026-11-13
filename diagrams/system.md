# TunedIn System Architecture

TunedIn is a mood-aware music chatbot. It reads how a user feels, retrieves
matching songs from `data/songs.csv`, and writes a recommendation grounded only
in those retrieved songs (Retrieval-Augmented Generation).

**Legend:** everything below is built. The fine-tuned classifier needs one training run
(`python training/train_mood.py`); until then the keyword baseline fills that slot.

---

## 1. System overview

```mermaid
flowchart LR
    U([User message<br/>"rough day, need to zone out"])

    subgraph UNDERSTAND["1 · Understand mood"]
        FT["Fine-tuned emotion classifier<br/>DistilBERT on GoEmotions<br/>(keyword baseline until trained)"]
        MAP["Emotion → profile map<br/>mood, target_energy,<br/>target_valence, likes_acoustic"]
        CL1["Claude fallback<br/>structured JSON profile"]
        FT -->|confidence ≥ threshold| MAP
        FT -.->|low confidence / error| CL1
    end

    subgraph RETRIEVE["2 · Retrieve"]
        CSV[("data/songs.csv")]
        SC["score_song + recommend_songs<br/>genre · mood · energy · acoustic<br/>+ valence"]
        CSV --> SC
    end

    subgraph GENERATE["3 · Generate"]
        CL2["Claude reply<br/>uses ONLY retrieved songs"]
        GUARD["Grounding check<br/>titles must exist in retrieved set"]
        CL2 --> GUARD
    end

    U --> FT
    MAP --> SC
    CL1 --> SC
    SC -->|top-k songs + reasons| CL2
    GUARD --> R([Reply: genre + 2–3 songs<br/>+ why each fits])
    SC -.->|API down: plain ranked list| R
```

---

## 2. Request flow

```mermaid
sequenceDiagram
    actor User
    participant UI as Streamlit chat
    participant Bot as chatbot.py
    participant Mood as mood_model.py
    participant Rec as recommender.py
    participant Claude as Claude API

    User->>UI: "just got the job!!!"
    UI->>Bot: answer(message)
    Bot->>Mood: predict(message)
    alt confident prediction
        Mood-->>Bot: profile {euphoric, energy 0.9, valence 0.85}
    else low confidence
        Bot->>Claude: extract profile (JSON schema, mood enum)
        Claude-->>Bot: profile
    end
    Bot->>Rec: recommend_songs(profile, songs, k=5)
    Rec-->>Bot: top 5 (song, score, reasons)
    Bot->>Claude: system rules + <retrieved_songs> + message
    Claude-->>Bot: recommendation text
    Bot->>Bot: verify every title is in retrieved set
    Bot-->>UI: reply
    UI-->>User: genre + songs + why
```

---

## 3. Fine-tuning pipeline

```mermaid
flowchart LR
    GE[("GoEmotions<br/>~58k labeled comments")] --> PREP
    OWN[("Hand-written examples<br/>100–200, TunedIn-style")] --> PREP
    PREP["training/train_mood.py<br/>tokenize · split train/val/test"] --> TR["Fine-tune DistilBERT<br/>HF Trainer"]
    TR --> M[("models/mood_classifier/<br/>(gitignored)")]
    M --> LOAD["src/mood_model.py<br/>predict() → emotion + confidence"]
```

---

## 4. Reliability & testing

```mermaid
flowchart TB
    subgraph L1["Layer 1 · Unit tests: free, every change"]
        T1["tests/ with pytest<br/>scoring math · ranking · CSV loading · valence"]
    end

    subgraph L2["Layer 2 · Mood-reading eval"]
        C2[("eval/mood_cases.csv<br/>message → expected mood + ranges")]
        E2["Compare: keyword baseline<br/>vs fine-tuned model vs Claude<br/>accuracy · energy/valence MAE · in-range %"]
        C2 --> E2
    end

    subgraph L3["Layer 3 · End-to-end chatbot eval"]
        C3[("eval/chatbot_cases.csv")]
        E3["Grounding · relevance · consistency (3 runs)<br/>safety (crisis prompts) · edge cases · latency"]
        C3 --> E3
    end

        E2 --> RUN
    E3 --> RUN
    RUN["eval/mood_eval.py<br/>eval/chatbot_eval.py"] --> REP[["Report table<br/>→ README + model_card.md"]]
```

Layers 2 and 3 call the Claude API and cost money per run. Run them when
prompts, the model, or the catalog change. Layer 1 never calls the API.

---

## Components

| Component | File | Status | Role |
| --- | --- | --- | --- |
| Song catalog | `data/songs.csv` | built | Knowledge base for retrieval |
| Scorer / retriever | `src/recommender.py` | built (with valence) | Ranks songs against a mood profile |
| CLI runner | `src/main.py` | built | Prints ranked results for fixed profiles |
| Mood classifier | `src/mood_model.py` | built (keyword + fine-tuned loader) | Fine-tuned text → emotion + confidence |
| Training script | `training/train_mood.py` | built, not yet run | Fine-tunes DistilBERT |
| Chatbot pipeline | `src/chatbot.py` | built | Understand → retrieve → generate, with fallbacks |
| Chat UI | `app.py` (Streamlit) | built | User-facing chat |
| Evals | `eval/` | built | Layers 2–3 and report |
