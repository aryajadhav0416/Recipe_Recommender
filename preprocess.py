import pandas as pd
import ast
import re
import nltk
import torch
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

# --- Download NLTK resources ---
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
remove_words = set([
    'cup','cups','teaspoon','teaspoons','tablespoon','tablespoons','g','kg','oz','ounce','ounces',
    'pound','pounds','lb','lbs','clove','cloves','fresh','chopped','sliced','diced','minced',
    'large','small','medium','package','can','jar','bottle','pinch','dash','to','taste','or'
])

def clean_and_process_ingredients(ingredient_list_str):
    try:
        ingredient_list = ast.literal_eval(ingredient_list_str)
    except (ValueError, SyntaxError):
        return ""
    cleaned = []
    for text in ingredient_list:
        text = text.lower()
        text = re.sub(r'\d+/\d+|\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        for w in tokens:
            w = lemmatizer.lemmatize(w)
            if w not in stop_words and w not in remove_words:
                cleaned.append(w)
    return ' '.join(cleaned)

print("🔄 Loading dataset...")
df = pd.read_csv("Recipe_dataset.csv")
df['cleaned_ingredients'] = df['ingredients'].apply(clean_and_process_ingredients)

print("🤖 Building embeddings (SentenceTransformer)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['cleaned_ingredients'].tolist(), convert_to_tensor=True)

print("💾 Saving processed files...")
torch.save(embeddings, "recipe_embeddings.pt")
df.to_csv("processed_recipes.csv", index=False)

print("\n✅ Preprocessing complete! Files saved:")
print(" - processed_recipes.csv")
print(" - recipe_embeddings.pt")
