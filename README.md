# Multilingual Sentiment Inspector

A Streamlit interface for the same Hugging Face checkpoint used in the notebook:

`cardiffnlp/twitter-xlm-roberta-base-sentiment`

It displays positive/neutral/negative scores and offers model inspection: tokenization, token IDs and offsets, initial embeddings, final contextual vectors, and Captum Integrated Gradients word contributions.

## Run locally

```bash
python -m venv .venv
```

Activate the environment, then run:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the pretrained model from Hugging Face and can take time. The model and Integrated Gradients use more memory and compute than a basic text-input app.

## Deploy with GitHub + Streamlit Community Cloud

1. Create a GitHub repository, for example `multilingual-sentiment-inspector`.
2. Upload `app.py`, `requirements.txt`, and `README.md` to the repository root.
3. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
4. Select **Create app**, choose your repository and branch, and set the main file path to `app.py`.
5. Click **Deploy** and review the app logs if installation or startup fails.

No Hugging Face token is required for this public model. Do not upload passwords or private tokens to GitHub.

## Important notes

- The app uses the same model checkpoint, not a substitute model.
- Inspection is limited to 64 tokens to match the notebook's `inspect_text` setup.
- The model card states that sentiment fine-tuning covered Arabic, English, French, German, Hindi, Italian, Spanish, and Portuguese. It may handle other languages, but accuracy is not guaranteed for every language or text type.
- Integrated Gradients values are approximate attribution scores, not proof of causal explanation.
