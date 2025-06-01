import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# --- Load and preprocess data ---
@st.cache_data(show_spinner=True)
def load_data():
    apps_info = pd.read_csv('apps_info.csv')
    games_info = pd.read_csv('games_info.csv')
    apps_reviews = pd.read_csv('apps_reviews.csv')
    games_reviews = pd.read_csv('games_reviews.csv')

    # Drop unnecessary columns
    apps_info = apps_info.drop(columns=['score', 'ratings_count', 'downloads', 'content_rating', 'section'])
    games_info = games_info.drop(columns=['score', 'ratings_count', 'downloads', 'content_rating', 'section'])
    apps_reviews = apps_reviews.drop(columns=['review_date', 'helpful_count'])
    games_reviews = games_reviews.drop(columns=['review_date', 'helpful_count'])

    # Rename columns for consistency
    apps_info.rename(columns={'app_id': 'id', 'app_name': 'name'}, inplace=True)
    games_info.rename(columns={'game_id': 'id', 'game_name': 'name'}, inplace=True)
    apps_reviews.rename(columns={'app_id': 'id'}, inplace=True)
    games_reviews.rename(columns={'game_id': 'id'}, inplace=True)

    # Combine reviews
    reviews = pd.concat([apps_reviews, games_reviews], ignore_index=True)

    # Sentiment labeling
    def sentiment_label(score):
        if score >= 4:
            return 'positive'
        elif score == 3:
            return 'neutral'
        else:
            return 'negative'

    reviews['sentiment'] = reviews['review_score'].apply(sentiment_label)

    reviews_agg = reviews.groupby('id').agg({
        'review_text': lambda x: " ".join(x),
        'sentiment': lambda x: " ".join(x)
    }).reset_index()

    apps_info['type'] = 'app'
    games_info['type'] = 'game'

    items_info = pd.concat([apps_info, games_info], ignore_index=True)
    items = items_info.merge(reviews_agg, on='id', how='left')

    items['review_text'] = items['review_text'].fillna('')
    items['sentiment'] = items['sentiment'].fillna('')

    items['tags'] = (items['description'].fillna('') + " " +
                     items['categories'].fillna('') + " " +
                     items['review_text'] + " " +
                     items['sentiment']).str.lower()

    return items

# --- Similarity functions ---
def get_content_similarity(subset):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(subset['tags']).toarray()
    return cosine_similarity(vectors)

def get_collaborative_similarity(subset):
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    review_vectors = tfidf.fit_transform(subset['review_text'])
    return cosine_similarity(review_vectors)

# --- Hybrid recommendation function with fallback ---
def hybrid_recommend(item_name, items, top_n=5):
    if item_name not in items['name'].values:
        # Fallback: return top N popular items overall
        fallback_recs = items['name'].value_counts().index[:top_n].tolist()
        return fallback_recs, None

    item = items[items['name'] == item_name].iloc[0]
    item_type = item['type']
    item_category = item['categories']

    def category_match(cat_str):
        item_cats = set([c.strip().lower() for c in str(item_category).split(',')])
        other_cats = set([c.strip().lower() for c in str(cat_str).split(',')])
        return len(item_cats.intersection(other_cats)) > 0

    filtered_items = items[(items['type'] == item_type) & (items['categories'].apply(category_match))].reset_index(drop=True)

    if filtered_items.empty:
        filtered_items = items[items['type'] == item_type].reset_index(drop=True)

    content_sim = get_content_similarity(filtered_items)
    collab_sim = get_collaborative_similarity(filtered_items)

    scaler = MinMaxScaler()
    content_sim_norm = scaler.fit_transform(content_sim)
    collab_sim_norm = scaler.fit_transform(collab_sim)

    try:
        idx = filtered_items[filtered_items['name'] == item_name].index[0]
    except IndexError:
        fallback_recs = filtered_items['name'].head(top_n).tolist()
        return fallback_recs, None

    content_sim_vector = content_sim_norm[idx]
    collab_sim_vector = collab_sim_norm[idx]
    hybrid_sim = content_sim_vector + collab_sim_vector

    recommended_indices = hybrid_sim.argsort()[::-1]
    recommended_indices = [i for i in recommended_indices if i != idx][:top_n]

    recommendations = filtered_items.iloc[recommended_indices]['name'].tolist()

    # Fill if fewer than top_n
    if len(recommendations) < top_n:
        needed = top_n - len(recommendations)
        popular_candidates = filtered_items['name'].head(top_n + needed).tolist()
        for candidate in popular_candidates:
            if candidate not in recommendations and candidate != item_name:
                recommendations.append(candidate)
            if len(recommendations) >= top_n:
                break

    return recommendations, None

# --- Streamlit UI ---
def main():
    st.title("Hybrid App & Game Recommendation System")

    items = load_data()

    app_or_game = st.radio("Select type:", ('app', 'game'))

    filtered_names = items[items['type'] == app_or_game]['name'].sort_values().unique()

    selected_item = st.selectbox(f"Select a {app_or_game}:", filtered_names)

    if st.button("Recommend"):
        recommendations, error = hybrid_recommend(selected_item, items, top_n=5)
        if error:
            st.error(error)
        else:
            st.success(f"My recommendations to '{selected_item}' are:")
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")

if __name__ == "__main__":
    main()
