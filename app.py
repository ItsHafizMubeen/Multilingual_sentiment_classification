import re
import pandas as pd
import streamlit as st
import torch
from captum.attr import IntegratedGradients
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_ID = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
# This repository adds a prebuilt tokenizer.json for the same model tokenizer.
TOKENIZER_ID = "NetworkIsLife/twitter-xlm-roberta-base-sentiment-fast"
MAX_LENGTH = 64
IG_STEPS = 16

st.set_page_config(
    page_title="Multilingual Sentiment Inspector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      :root {
        --premium-blue: #174ea6;
        --premium-blue-dark: #12366d;
        --premium-blue-light: #eff6ff;
        --border: #dbe5f1;
        --muted: #64748b;
      }
      .stApp { background: #f7f9fc; color: #172b4d; }
      [data-testid="stHeader"] { background: #ffffff; border-bottom: 1px solid #eef2f7; }
      section[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none !important; }
      .block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1180px; }
      .hero {
        padding: 1.8rem 2rem; border-radius: 18px; color: var(--premium-blue-dark);
        background: #ffffff; border: 1px solid var(--border);
        margin-bottom: 1.7rem; box-shadow: 0 8px 26px rgba(15, 35, 75, .045);
      }
      .hero h1 { margin: 0 0 .55rem 0; color: var(--premium-blue-dark); font-size: 2.05rem; letter-spacing: -.035em; }
      .hero p { margin: 0; color: #315887; font-size: 1rem; line-height: 1.65; }
      .hero .small-label { color: #2563c9; font-size: .78rem; font-weight: 750; text-transform: uppercase; letter-spacing: .09em; margin-bottom: .55rem; }
      .small-label { color: #526987; font-size: .78rem; font-weight: 700; text-transform: uppercase; letter-spacing: .07em; }
      .metric-card {
        background: #ffffff; padding: 1rem 1.1rem; border: 1px solid var(--border);
        border-radius: 14px; height: 100%; box-shadow: 0 3px 12px rgba(15, 35, 75, .025);
      }
      .metric-value { font-size: 1.5rem; line-height: 1.35; font-weight: 750; color: #12366d; overflow-wrap: anywhere; }
      .metric-note { margin-top: .2rem; color: var(--muted); font-size: .83rem; }
      div[data-testid="stForm"] { background: #ffffff; padding: 1.15rem 1.25rem; border: 1px solid var(--border); border-radius: 16px; box-shadow: 0 4px 18px rgba(15, 35, 75, .025); }
      div[data-testid="stTextArea"] label, label { color: #243f65; }
      div[data-testid="stFormSubmitButton"] > button {
        border-radius: 10px; min-height: 2.85rem; font-weight: 700;
        color: #ffffff !important; border: 1px solid #174ea6 !important;
        background: linear-gradient(135deg, #174ea6 0%, #2563c9 100%) !important;
        box-shadow: 0 5px 14px rgba(23, 78, 166, .18);
        transition: transform .15s ease, box-shadow .15s ease;
      }
      div[data-testid="stFormSubmitButton"] > button:hover {
        color: #ffffff !important; border-color: #12366d !important;
        background: linear-gradient(135deg, #12366d 0%, #174ea6 100%) !important;
        box-shadow: 0 7px 18px rgba(23, 78, 166, .24); transform: translateY(-1px);
      }
      div[data-testid="stDownloadButton"] > button {
        border-radius: 9px; border: 1px solid #bfd3f4; color: #174ea6;
        background: #eff6ff; font-weight: 650;
      }
      div[data-testid="stDownloadButton"] > button:hover {
        border-color: #174ea6; color: #12366d; background: #e3efff;
      }
      h2, h3 { color: #173b70; letter-spacing: -.02em; }
      .note { color: var(--muted); font-size: .9rem; }
      div[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
      @media (max-width: 640px) {
        .block-container { padding-top: 1rem; }
        .hero { padding: 1.35rem 1.2rem; }
        .hero h1 { font-size: 1.65rem; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model():
    """Load official model weights and a compatible prebuilt fast tokenizer."""
    # The official CardiffNLP repository does not include tokenizer.json.
    # This tokenizer copy provides that file and documents identical token IDs.
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("A fast tokenizer is required for character offset mapping.")

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    model_device = next(model.parameters()).device
    return model, tokenizer, model_device


def safe_label(config, class_id):
    labels = getattr(config, "id2label", {}) or {}
    return labels.get(class_id, labels.get(str(class_id), f"LABEL_{class_id}"))


def analyse_text(text, model, tokenizer, model_device):
    """Run sentiment classification plus token, vector, and IG inspection."""
    text = text.strip()
    if not text:
        raise ValueError("Please enter some text first.")

    # Keep the inspection sequence length aligned with the supplied notebook code.
    batch = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        return_offsets_mapping=True,
    )
    offsets = batch.pop("offset_mapping")[0].tolist()
    encoded = {key: value.to(model_device) for key, value in batch.items()}
    input_ids = encoded["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu().tolist())

    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]
        initial_vectors = model.get_input_embeddings()(input_ids)[0]
        processed_vectors = outputs.hidden_states[-1][0]

    scores = probabilities.detach().cpu().tolist()
    class_ids = list(range(len(scores)))
    class_names = [str(safe_label(model.config, i)) for i in class_ids]
    predicted_id = int(probabilities.argmax().item())
    predicted_label = class_names[predicted_id]
    confidence = float(probabilities[predicted_id].item())

    # Tokenization table: offsets (start, end) refer to character positions in the input.
    token_ids = input_ids[0].detach().cpu().tolist()
    token_text = []
    for start, end in offsets:
        token_text.append(text[start:end] if end > start else "<special token>")
    token_df = pd.DataFrame(
        {
            "Token": tokens,
            "Token ID": token_ids,
            "Original Text": token_text,
            "Start": [int(pair[0]) for pair in offsets],
            "End": [int(pair[1]) for pair in offsets],
        }
    )

    # Integrated Gradients, following the same method and settings as the notebook.
    def predict_from_embeddings(embeddings, mask):
        logits = model(inputs_embeds=embeddings, attention_mask=mask).logits
        return torch.softmax(logits, dim=-1)

    ig = IntegratedGradients(predict_from_embeddings)
    embeddings_for_ig = initial_vectors.unsqueeze(0).detach()
    attributions = ig.attribute(
        inputs=embeddings_for_ig,
        baselines=torch.zeros_like(embeddings_for_ig),
        target=predicted_id,
        additional_forward_args=(encoded["attention_mask"],),
        n_steps=IG_STEPS,
        internal_batch_size=1,
    )
    token_scores = attributions.sum(dim=-1)[0].detach().cpu().tolist()

    # Aggregate subword-token attributions back to whitespace-separated words.
    words = list(re.finditer(r"\S+", text))
    word_scores = [0.0] * len(words)
    for (start, end), score in zip(offsets, token_scores):
        if end <= start:
            continue
        overlaps = [
            max(0, min(end, word.end()) - max(start, word.start()))
            for word in words
        ]
        if overlaps and max(overlaps) > 0:
            word_scores[overlaps.index(max(overlaps))] += float(score)

    total_magnitude = sum(abs(value) for value in word_scores)
    word_rows = []
    for word, score in zip(words, word_scores):
        word_rows.append(
            {
                "Word": word.group(),
                "IG Score": round(score, 6),
                "Relative Magnitude (%)": round(
                    abs(score) / total_magnitude * 100 if total_magnitude else 0.0, 2
                ),
                "Direction": (
                    "Supports predicted class"
                    if score > 0
                    else "Opposes predicted class"
                    if score < 0
                    else "Zero contribution"
                ),
            }
        )
    word_df = pd.DataFrame(
        word_rows,
        columns=["Word", "IG Score", "Relative Magnitude (%)", "Direction"],
    )

    # Preview only a few dimensions in the UI; the download contains every vector dimension.
    initial_np = initial_vectors.detach().cpu().numpy()
    processed_np = processed_vectors.detach().cpu().numpy()
    preview_dims = min(12, initial_np.shape[-1])
    vector_preview_rows = []
    for i, token in enumerate(tokens):
        vector_preview_rows.append(
            {
                "Token": token,
                "Initial embedding (first dimensions)": [
                    round(float(x), 4) for x in initial_np[i, :preview_dims]
                ],
                "Processed vector (first dimensions)": [
                    round(float(x), 4) for x in processed_np[i, :preview_dims]
                ],
            }
        )
    vector_preview_df = pd.DataFrame(vector_preview_rows)

    # Full-dimension export, including each token's original text and token ID.
    vector_data = {
        "Token": tokens,
        "Token ID": token_ids,
        "Original Text": token_text,
    }
    for dim in range(initial_np.shape[-1]):
        vector_data[f"initial_dim_{dim}"] = initial_np[:, dim]
    for dim in range(processed_np.shape[-1]):
        vector_data[f"processed_dim_{dim}"] = processed_np[:, dim]
    full_vectors_df = pd.DataFrame(vector_data)

    class_df = pd.DataFrame(
        {
            "Class": [label.upper() for label in class_names],
            "Score": scores,
            "Confidence (%)": [round(score * 100, 2) for score in scores],
        }
    ).sort_values("Score", ascending=False, ignore_index=True)

    return {
        "text": text,
        "predicted_label": predicted_label.upper(),
        "confidence": confidence,
        "class_df": class_df,
        "token_df": token_df,
        "vector_preview_df": vector_preview_df,
        "full_vectors_df": full_vectors_df,
        "word_df": word_df,
        "token_count": len(tokens),
        "hidden_size": int(initial_np.shape[-1]),
        "model_device": str(model_device),
    }


st.markdown(
    """
    <div class="hero">
      <div class="small-label">AI · NLP · MODEL EXPLAINABILITY</div>
      <h1>Multilingual Sentiment Inspector</h1>
      <p>Classify text as positive, neutral, or negative — then inspect how the model processes it.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("1. Enter text")
with st.form("sentiment_form", clear_on_submit=False):
    user_text = st.text_area(
        "Text to analyse",
        height=150,
        placeholder="Enter a sentence in English, Urdu, Hindi, Arabic, Spanish, or another language…",
        help="The text is truncated to 64 tokenizer tokens for classification and inspection.",
    )
    submitted = st.form_submit_button("Analyse text", type="primary", use_container_width=True)

if submitted:
    if not user_text.strip():
        st.warning("Please enter text before analysing.")
    else:
        try:
            with st.spinner("Loading model and running classification + inspection. The first run can take longer…"):
                loaded_model, loaded_tokenizer, loaded_device = load_model()
                result = analyse_text(
                    user_text,
                    loaded_model,
                    loaded_tokenizer,
                    loaded_device,
                )
            st.session_state["sentiment_analysis_result"] = result
        except Exception as exc:
            st.error(f"Analysis failed: {type(exc).__name__}: {exc}")
            st.info(
                "Confirm requirements.txt includes sentencepiece and transformers<5.0.0. "
                "If the model still fails to load, open Manage app → Logs and check available memory."
            )

result = st.session_state.get("sentiment_analysis_result")
if result:
    st.divider()
    st.subheader("2. Prediction overview")
    if result["text"] != user_text.strip():
        st.caption("Showing the last completed analysis. Click **Analyse text** to refresh it for the current input.")

    metric_cols = st.columns(4, gap="medium")
    metric_values = [
        ("PREDICTED CLASS", result["predicted_label"], "Highest model score"),
        ("CONFIDENCE", f"{result['confidence'] * 100:.2f}%", "Model probability for the predicted class"),
        ("TOKENS", str(result["token_count"]), f"Truncated at {MAX_LENGTH} tokens"),
        ("HIDDEN SIZE", str(result["hidden_size"]), "Values per token vector"),
    ]
    for col, (label, value, note) in zip(metric_cols, metric_values):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="small-label">{label}</div>'
                f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("### Classification scores")
    score_col, explanation_col = st.columns([1.05, 1], gap="large")
    with score_col:
        chart_df = result["class_df"].set_index("Class")[["Score"]]
        st.bar_chart(chart_df, y="Score", height=245, use_container_width=True)
    with explanation_col:
        st.dataframe(
            result["class_df"].style.format({"Score": "{:.4f}", "Confidence (%)": "{:.2f}%"}),
            hide_index=True,
            use_container_width=True,
        )
        st.caption("Scores are model outputs after softmax. They are not a guarantee that a prediction is correct.")

    st.divider()
    st.subheader("3. Inspect the backend")
    st.markdown(
        "Follow the same text through each stage: text → tokenizer → embeddings → Transformer → classifier → explanation."
    )
    with st.expander("A. Tokenization — tokens, token IDs, and text offsets", expanded=True):
        st.write(
            "The tokenizer splits text into model-specific subword tokens. Token IDs are the numeric values "
            "fed to the model; Start/End show character positions in the original input. Special tokens do not map to a text span."
        )
        st.dataframe(result["token_df"], hide_index=True, use_container_width=True)
        token_csv = result["token_df"].to_csv(index=False).encode("utf-8-sig")
        st.download_button("Download tokenization CSV", token_csv, "tokenization.csv", "text/csv")

    with st.expander("B. Initial embeddings and processed vectors"):
        st.write(
            "An initial embedding is the token's learned input embedding. A processed vector is the final contextualized "
            "Transformer hidden state for that token; it reflects the token and its surrounding context."
        )
        st.caption(
            f"Preview shows the first 12 dimensions out of {result['hidden_size']} per vector. "
            "Download the CSV for every dimension of both vectors."
        )
        st.dataframe(result["vector_preview_df"], hide_index=True, use_container_width=True)
        full_csv = result["full_vectors_df"].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download all embedding and processed-vector dimensions (CSV)",
            full_csv,
            "token_vectors_full.csv",
            "text/csv",
        )

    with st.expander("C. Integrated Gradients — word-level contribution", expanded=True):
        st.write(
            "Integrated Gradients estimates how each input embedding contributes to the predicted class score, "
            "relative to a zero-embedding baseline. Subword-token scores are aggregated back to whitespace-separated words."
        )
        if result["word_df"].empty:
            st.info("No word-level contributions could be displayed for this input.")
        else:
            st.dataframe(
                result["word_df"].style.format(
                    {"IG Score": "{:.6f}", "Relative Magnitude (%)": "{:.2f}%"}
                ),
                hide_index=True,
                use_container_width=True,
            )
            st.caption(
                "Positive/negative direction is relative to the predicted class score and this baseline; "
                "it is an approximate model explanation, not proof of human intent or causality."
            )
            word_csv = result["word_df"].to_csv(index=False).encode("utf-8-sig")
            st.download_button("Download word contribution CSV", word_csv, "word_contributions.csv", "text/csv")

    with st.expander("D. Backend workflow summary"):
        st.markdown(
            """
            1. **Tokenizer:** converts the text into token IDs and an attention mask.
            2. **Embedding layer:** maps each token ID to a numeric vector.
            3. **XLM-RoBERTa Transformer:** contextualizes token representations using surrounding text.
            4. **Classification head:** produces a score for each sentiment class.
            5. **Softmax:** converts logits into scores that sum to approximately 1.
            6. **Integrated Gradients:** estimates token-level contribution to the selected class score.
            """
        )
        st.code(
            f"Model: {MODEL_ID}\nDevice: {result['model_device']}\nMax tokens: {MAX_LENGTH}\nIG steps: {IG_STEPS}",
            language="text",
        )

st.divider()
st.caption(
    "Built with Streamlit, Hugging Face Transformers, PyTorch, and Captum. "
    "The model is multilingual, but language coverage and accuracy vary by input."
)
