# 🍳 Recipe Recommender Web App

A lightweight ingredient-based recipe recommendation system built using **Streamlit**, **TF-IDF**, and **Cosine Similarity**. Simply input the ingredients you have in your kitchen, and the app recommends recipes you can make — with inline details of ingredients and steps!

> ⚠️ This project uses **only 1 sampled chunk (~8,000 recipes)** from a much larger dataset for quicker prototyping and training. You can expand it later with the full dataset.

---

## 🚀 Features

- 🔍 **Ingredient-based search:** Get recipe suggestions based on what you have at home.
- 💡 **TF-IDF + Cosine Similarity:** Matches input ingredients to recipes using vector similarity.
- 📋 **Inline recipe viewer:** No page reloads — view full recipe (title, category, ingredients, directions) right on the same page.
- 📊 **Similarity score:** Shows how closely a recipe matches your ingredients.
- ⚡ **Optimized for speed:** Uses only a sampled subset of ~8,000 recipes from 62K+ for development/testing.

---

## 📊 Dataset

The dataset is sourced from [Kaggle: Recipe Ingredients and Directions](https://www.kaggle.com/datasets/kaggle/recipe-ingredients-and-directions). It contains over 62,000 recipes with structured information such as ingredients, directions, categories, and descriptions.

> **Note:** Only 8,000 records were used from the dataset for model building in this project.

### Sample Features:

| Column         | Description                                 |
|----------------|---------------------------------------------|
| `recipe_title` | Name of the recipe                          |
| `category`     | Main category (e.g. Main Dishes, Healthy)   |
| `subcategory`  | Specific subcategory (e.g. Pasta, Meatloaf) |
| `description`  | Short recipe summary                        |
| `ingredients`  | List of ingredients (stringified JSON)      |
| `directions`   | Step-by-step instructions (stringified JSON)|
| `num_ingredients` | Number of ingredients used               |
| `num_steps`    | Number of directions steps                  |

---

## 🧠 How It Works

1. **User inputs ingredients** (e.g., `"chicken, rice, onion"`).
2. **Text cleaning & preprocessing** using NLTK (lemmatization, stopword removal).
3. **TF-IDF vectorization** of ingredients in the recipe dataset.
4. **Cosine similarity** between input and recipe vectors.
5. **Top 5 recipe matches** are displayed with details on the same page.

---

## 🧪 Tech Stack

- **Frontend**: Streamlit
- **ML/NLP**: TF-IDF (Scikit-learn), Cosine Similarity
- **Data Cleaning**: NLTK (tokenization, stopwords, lemmatization)
- **Persistence**: Pickle files for model, matrix, and data

---

## 🛠️ Getting Started

### 📦 Prerequisites

- Python 3.7+
- pip (Python package manager)

### ⚙️ Installation

1. **Clone the repo:**
   ```bash
   git clone https://github.com/yourusername/recipe-recommender-app.git
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the model training notebook:**
   - Open the notebook in Jupyter or VSCode:
   ```bash
   jupyter notebook model_train.ipynb
   ```
4. Run the Streamlit web app:
   ```bash
   streamlit run app.py
   ```
5. Open the URL shown in your browser and start discovering delicious recipes based on your ingredients!
