# 🍳 AI-Powered Recipe Recommendation System

An **NLP-based intelligent recipe recommender** built using **Sentence Transformers** and **Streamlit**.  
This system suggests the best matching recipes based on user-provided ingredients — using **semantic similarity + ingredient overlap confidence** to understand meaning, not just keywords.

---

## 🚀 Features

✅ **Natural Language Processing (NLP):** Understands ingredient meaning (e.g., “tomato” ≈ “tomatoes”)  
✅ **Semantic Embeddings:** Uses `all-MiniLM-L6-v2` model from Sentence Transformers  
✅ **Hybrid Confidence Score:** 70% semantic similarity + 30% ingredient overlap  
✅ **Duplicate Removal:** Filters duplicate recipe names automatically  
✅ **Interactive Streamlit UI:** Clean interface with confidence bars, ingredient layout, and search history  
✅ **Modular Structure:** Separate files for preprocessing, recommendation logic, and app interface  

---

## 📊 Dataset

The dataset is sourced from [Kaggle: Recipe Dataset-64k dishes](https://www.kaggle.com/datasets/prashantsingh001/recipes-dataset-64k-dishes). It contains over 62,000 recipes with structured information such as ingredients, directions, categories, and descriptions.

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

## ⚙️ Tech Stack

| Component | Description |
|------------|-------------|
| **Python** | Core programming language |
| **Streamlit** | Web framework for interactive UI |
| **Sentence Transformers** | Pre-trained model for semantic similarity |
| **PyTorch** | Backend for embeddings and tensor math |
| **NLTK** | Tokenization, stopword removal, and lemmatization |
| **Pandas** | Data manipulation and preprocessing |

---

## 🧠 How It Works

1. **Preprocessing (`preprocess.py`)**
   - Cleans ingredient text (tokenization, lemmatization, stopword removal)
   - Generates embeddings using `SentenceTransformer('all-MiniLM-L6-v2')`
   - Saves processed CSV + embeddings (`.pt` file)

2. **Recommendation (`recommend.py`)**
   - Computes semantic similarity between user ingredients and all recipes
   - Calculates ingredient overlap confidence
   - Combines both into a final confidence score
   - Removes duplicate titles and returns top matches

3. **Streamlit App (`app.py`)**
   - Accepts ingredient input from the user
   - Displays top recipe matches with confidence bars
   - Expands to show ingredient lists neatly
   - Includes filters, search history, and dynamic UI elements

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

3. **Preprocess Dataset:**
   - Run the script once to generate embeddings:
   ```bash
   python preprocess.py
   ```
   
4. Run the Streamlit web app:
   ```bash
   streamlit run app.py
   ```
   
5. Open the URL shown in your browser and start discovering delicious recipes based on your ingredients!
