import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import word_tokenize

# -------------------------------------------------------
# Load model + data
# -------------------------------------------------------
def load_resources():
    df = pd.read_csv("processed_recipes.csv")
    recipe_embeddings = torch.load("recipe_embeddings.pt")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return df, recipe_embeddings, model

# -------------------------------------------------------
# Recommend recipes
# -------------------------------------------------------
def recommend_recipes(user_input, df, recipe_embeddings, model, top_n=5):
    cleaned_query = user_input.lower().replace(',', ' ')
    query_ingredients = set(word_tokenize(cleaned_query))

    # Semantic similarity
    query_embedding = model.encode(cleaned_query, convert_to_tensor=True)
    semantic_scores = util.pytorch_cos_sim(query_embedding, recipe_embeddings)[0].cpu().numpy()

    # Ingredient overlap confidence
    overlaps = []
    for recipe_ing in df['cleaned_ingredients']:
        recipe_set = set(recipe_ing.split())
        overlap = len(query_ingredients & recipe_set) / (len(query_ingredients) + 1e-6)
        overlaps.append(overlap)

    overlaps = torch.tensor(overlaps)
    final_conf = 0.7 * torch.tensor(semantic_scores) + 0.3 * overlaps

    # --- Auto-detect recipe title column ---
    title_col = None
    for possible in ['recipe_title', 'name', 'recipe_name', 'dish_name']:
        if possible in df.columns:
            title_col = possible
            break

    # --- Collect results ---
    results = []
    for i in range(len(df)):
        recipe_title = df.iloc[i][title_col] if title_col else f"Recipe {i}"
        confidence = round(float(final_conf[i].item()) * 100, 2)
        ingredients = df.iloc[i]['ingredients']
        results.append({
            "Recipe": recipe_title,
            "Confidence": confidence,
            "Ingredients": ingredients
        })

    # --- Remove duplicate titles (keep highest confidence) ---
    deduped = {}
    for r in results:
        title = r["Recipe"].strip().lower()
        if title not in deduped or r["Confidence"] > deduped[title]["Confidence"]:
            deduped[title] = r

    unique_results = list(deduped.values())

    # --- Sort and take top N ---
    unique_results = sorted(unique_results, key=lambda x: x["Confidence"], reverse=True)[:top_n]

    return unique_results
