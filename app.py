import streamlit as st
from recommend import load_resources, recommend_recipes

# -------------------------------
# Load Model and Data
# -------------------------------
@st.cache_resource
def get_resources():
    return load_resources()

df, recipe_embeddings, model = get_resources()

# -------------------------------
# Streamlit Page Configuration
# -------------------------------
st.set_page_config(
    page_title="🍳 AI Recipe Recommender",
    page_icon="🥕",
    layout="wide"
)

# -------------------------------
# Sidebar Configuration
# -------------------------------
with st.sidebar:
    st.title("🍽️ Recipe Finder Settings")
    st.markdown("Use filters to customize your search results 👇")

    top_n = st.slider("Number of Recommendations", 3, 15, 5)
    conf_threshold = st.slider("Minimum Confidence (%)", 0, 100, 0)

    st.markdown("---")
    st.markdown("💡 *This system uses NLP embeddings (Sentence Transformers) to find recipes based on meaning — not just keyword matches!*")

# -------------------------------
# Header
# -------------------------------
st.title("🥗 **AI-Powered Recipe Recommender**")
st.markdown("""
> Type in the ingredients you have, and this app will recommend the most relevant dishes 🍋🍚  
> Powered by **semantic similarity + ingredient overlap** 🤖✨
""")

# -------------------------------
# Input Field
# -------------------------------
user_input = st.text_input(
    "Enter ingredients (comma-separated):",
    placeholder="e.g., lemon, carrot, rice, ginger"
)

# Maintain search history
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------
# Main Logic
# -------------------------------
# -------------------------------
# Display Results
# -------------------------------
if st.button("🔍 Find Recipes") and user_input.strip():
    with st.spinner("🧠 Analyzing your ingredients..."):
        recommendations = recommend_recipes(user_input, df, recipe_embeddings, model, top_n=top_n)
        st.session_state.history.append(user_input)

    recommendations = [r for r in recommendations if r['Confidence'] >= conf_threshold]

    if not recommendations:
        st.warning("⚠️ No recipes found above your confidence threshold.")
    else:
        st.success(f"✅ Found {len(recommendations)} matching recipes!")

        for r in recommendations:
            # Use recipe title directly
            with st.expander(f"🍴 {r['Recipe']} — Confidence: {r['Confidence']}%"):
                st.progress(r['Confidence'] / 100)

                st.markdown("### 🧂 Ingredients:")
                try:
                    ingredients = eval(r['Ingredients']) if isinstance(r['Ingredients'], str) else r['Ingredients']
                    if isinstance(ingredients, list):
                        cols = st.columns(2)
                        half = len(ingredients)//2 + 1
                        for i, ing in enumerate(ingredients):
                            with cols[0 if i < half else 1]:
                                st.markdown(f"🔸 {ing}")
                    else:
                        st.write(ingredients)
                except Exception:
                    st.write(r['Ingredients'])

                st.markdown("---")
# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("💡 Built with ❤️ using **Sentence Transformers, PyTorch & Streamlit**")
