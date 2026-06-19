"""Streamlit web demo for the mini-GPT Transformer.

Run:
    streamlit run demo.py

Features:
    - Text generation with temperature & top-k controls
    - Attention heatmap visualization per layer/head
    - Model stats sidebar
"""

import streamlit as st
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from model import GPT

st.set_page_config(
    page_title="Mini GPT Demo",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Mini GPT — Interactive Demo")
st.markdown(
    "*Character-level Transformer trained on Shakespeare — "
    "[built from scratch](https://github.com/)*"
)


@st.cache_resource
def load_model():
    """Load model once and cache it across sessions."""
    checkpoint = torch.load('gpt_model.pt', map_location='cpu', weights_only=True)
    model = GPT(
        vocab_size=checkpoint['vocab_size'],
        embed_dim=checkpoint['config']['embed_dim'],
        num_heads=checkpoint['config']['num_heads'],
        num_layers=checkpoint['config']['num_layers'],
        block_size=checkpoint['config']['block_size'],
    )
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model, checkpoint['stoi'], checkpoint['itos']


try:
    model, stoi, itos = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    st.warning(
        "No trained model found. Run `python train.py` first to create "
        "`gpt_model.pt`, then refresh this page."
    )
    st.stop()

# --- Sidebar ---
st.sidebar.header("📊 Model Stats")
st.sidebar.metric("Parameters", f"{model.count_params():.1f}M")
st.sidebar.metric("Layers", f"{len(model.blocks)}")
st.sidebar.metric("Attention Heads", "8")
st.sidebar.metric("Vocabulary", f"{len(stoi)} chars")
st.sidebar.metric("Context Window", f"{model.block_size} chars")

st.sidebar.header("🎛️ Generation Settings")
temperature = st.sidebar.slider(
    "Temperature", 0.1, 2.0, 0.8, 0.1,
    help="Low = safe/predictable, High = creative/chaotic",
)
top_k = st.sidebar.slider(
    "Top-K", 1, 100, 40,
    help="Only consider top K most likely next characters",
)
max_tokens = st.sidebar.slider(
    "Max Tokens", 50, 500, 200,
    help="How many characters to generate",
)

# --- Text Generation Tab ---
tab1, tab2 = st.tabs(["📝 Text Generation", "🔍 Attention Visualizer"])

with tab1:
    st.header("Generate Shakespeare-style Text")

    prompt = st.text_input(
        "Prompt",
        "ROMEO:",
        help="Start of the text. Use character names for best results.",
    )

    if st.button("✨ Generate", type="primary"):
        with st.spinner("The transformer is thinking..."):
            encoded = [stoi[c] for c in prompt if c in stoi]
            if not encoded:
                st.error("No valid characters in prompt!")
                st.stop()

            input_ids = torch.tensor([encoded], dtype=torch.long)
            output_ids = model.generate(
                input_ids, max_tokens, temperature, top_k
            )
            generated = ''.join(
                itos[i] for i in output_ids[0].tolist() if i in itos
            )

        st.text_area("Output", generated, height=250)

        st.download_button(
            "📥 Download Generated Text",
            generated,
            file_name="shakespeare_generated.txt",
            mime="text/plain",
        )

    st.caption(
        "💡 **Tip**: Try prompts like `ROMEO:`, `JULIET:`, `To be`, or "
        "`First Citizen:` for best results."
    )

# --- Attention Visualization Tab ---
with tab2:
    st.header("Visualize Attention Patterns")

    sample_text = st.text_input(
        "Text to analyze",
        "To be or not to be,",
        help="Keep it short (≤30 chars) for readable heatmaps",
        key="attn_text",
    )

    col1, col2 = st.columns(2)
    with col1:
        layer = st.selectbox(
            "Layer",
            range(len(model.blocks)),
            index=len(model.blocks) - 1,
            format_func=lambda x: f"Layer {x + 1}",
            help="Deeper layers capture longer-range patterns",
        )
    with col2:
        head = st.selectbox(
            "Head",
            range(8),
            index=0,
            format_func=lambda x: f"Head {x + 1}",
            help="Different heads specialize in different patterns",
        )

    if st.button("🔍 Visualize", type="primary"):
        encoded = [stoi[c] for c in sample_text if c in stoi]
        if len(encoded) < 2:
            st.error("Need at least 2 valid characters!")
            st.stop()

        input_ids = torch.tensor([encoded], dtype=torch.long)

        with torch.no_grad():
            _, _, attentions = model(input_ids, return_attention=True)

        attn = attentions[layer][0, head].cpu().numpy()
        tokens = [itos[i] for i in encoded]
        display_tokens = [t.replace('\n', '\\n') for t in tokens]

        fig, ax = plt.subplots(figsize=(10, 8))

        # If short text, show annotation values
        annot = len(tokens) <= 15

        sns.heatmap(
            attn,
            xticklabels=display_tokens,
            yticklabels=display_tokens,
            cmap='Blues',
            ax=ax,
            annot=annot,
            fmt='.2f' if annot else '',
            cbar_kws={'label': 'Attention Weight'},
            vmin=0,
            vmax=attn.max(),
        )

        ax.set_xlabel('Key (attended TO)')
        ax.set_ylabel('Query (attending FROM)')
        ax.set_title(f'Layer {layer + 1}, Head {head + 1}')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

    st.caption(
        "💡 **What to look for**: Early layers attend locally. "
        "Later layers capture long-range patterns. Some heads specialize "
        "in punctuation, others in character names."
    )
