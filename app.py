import streamlit as st
import pandas as pd
import pickle
import ast
import re
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from streamlit import column_config

# --- Load Models and Data ---
@st.cache_resource
def load_models():
    try:
        tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
        tfidf_matrix = pickle.load(open('tfidf_matrix.pkl', 'rb'))
        recipe_df = pickle.load(open('recipe_df.pkl', 'rb'))
        return tfidf_vectorizer, tfidf_matrix, recipe_df
    except FileNotFoundError:
        st.error("Model files not found. Please run the model building script first.")
        return None, None, None

@st.cache_resource
def get_nlp_tools():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/wordnet')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        with st.spinner("Downloading NLTK data..."):
            nltk.download('punkt', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('stopwords', quiet=True)

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    remove_words = set([
        'cup', 'cups', 'teaspoon', 'teaspoons', 'tablespoon', 'tablespoons', 'g', 'kg',
        'oz', 'ounce', 'ounces', 'pound', 'pounds', 'lb', 'lbs', 'clove', 'cloves',
        'fresh', 'chopped', 'sliced', 'diced', 'minced', 'large', 'small', 'medium',
        'package', 'can', 'jar', 'bottle', 'pinch', 'dash', 'to', 'taste', 'or'
    ])
    return lemmatizer, stop_words, remove_words

# Load models and NLP tools
tfidf, tfidf_matrix, df = load_models()
if df is not None:
    lemmatizer, stop_words, remove_words = get_nlp_tools()

# --- Input Cleaning ---
def clean_user_input(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    cleaned = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and word not in remove_words
    ]
    return ' '.join(cleaned)

# --- Recommendation Logic ---
def get_recommendations(user_ingredients, top_n=5):
    user_input_str = ' '.join(user_ingredients)
    cleaned_input = clean_user_input(user_input_str)
    user_vector = tfidf.transform([cleaned_input])
    sim_scores = cosine_similarity(user_vector, tfidf_matrix)[0]
    top_indices = sim_scores.argsort()[-top_n:][::-1]
    recommendations = df.iloc[top_indices].copy()
    recommendations['Match Score'] = sim_scores[top_indices]
    return recommendations

# --- App UI ---
st.set_page_config(page_title="Recipe Recommender", layout="wide")
st.title("What's in Your Pantry? 🍳")
st.markdown("Enter the ingredients you have, and we'll suggest recipes you can make!")

ingredients_input = st.text_input(
    "Ingredients (comma-separated)", 
    placeholder="e.g. chicken, onion, garlic, rice"
)

selected_recipe_index = None

if ingredients_input:
    user_ingredients = [item.strip().lower() for item in ingredients_input.split(',')]
    
    with st.spinner("Finding recipes..."):
        results = get_recommendations(user_ingredients, top_n=5)

    if results.empty:
        st.warning("No matching recipes found. Try more common ingredients.")
    else:
        st.success(f"Found {len(results)} recipes for you!")

        table = results[["recipe_title", "category", "Match Score"]].rename(columns={
            "recipe_title": "Recipe Title",
            "category": "Category"
        })

        st.dataframe(
            table,
            column_config={
                "Recipe Title": "Recipe Title",
                "Category": "Category",
                "Match Score": column_config.ProgressColumn(
                    "Match Score", format="%.2f", min_value=0, max_value=1
                )
            },
            hide_index=True,
            use_container_width=True
        )

        selected_index = st.selectbox(
            "Select a recipe to view details:",
            options=results.index,
            format_func=lambda i: df.loc[i, 'recipe_title']
        )

        # --- Show Recipe Details Inline ---
        st.markdown("---")
        st.header("📋 Recipe Details")

        recipe = df.loc[selected_index]

        st.subheader(recipe['recipe_title'])
        st.write(f"**Category:** {recipe['category']} | **Subcategory:** {recipe['subcategory']}")
        st.write(f"_{recipe['description']}_")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Ingredients")
            try:
                ingredients_list = ast.literal_eval(recipe['ingredients'])
                for item in ingredients_list:
                    st.write(f"- {item}")
            except:
                st.write("Could not parse ingredients.")

        with col2:
            st.subheader("Directions")
            try:
                directions = ast.literal_eval(recipe['directions'])
                for i, step in enumerate(directions, 1):
                    st.write(f"{i}. {step}")
            except:
                st.write("Could not parse directions.")
else:
    st.info("Please enter your ingredients above to get recipe suggestions.")
