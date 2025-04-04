import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
import random

print("Loading data from CSV files...")

# Load books data
try:
    books_df = pd.read_csv("Books.csv", usecols=["ISBN", "Book-Title", "Book-Author", "Image-URL-M", "Image-URL-L", "Image-URL-S"])
    books_df.rename(columns={
        "Book-Title": "title",
        "Book-Author": "author",
        "Image-URL-M": "url",
        "Image-URL-L": "urlL",
        "Image-URL-S": "urlS"
    }, inplace=True)
    books_df["desc"] = "N/A"
    books_df["language"] = "N/A"
    books_df["price"] = [random.randint(200, 1000) for _ in range(len(books_df))]
    print("✅ Books data loaded successfully!")
except Exception as e:
    print(f"❌ Error loading Books.csv: {e}")
    exit(1)

# Load ratings data
try:
    ratings_df = pd.read_csv("Ratings.csv", usecols=["User-ID", "ISBN", "Book-Rating"])
    ratings_df.rename(columns={
        "User-ID": "userID",
        "Book-Rating": "rating"
    }, inplace=True)
    print("✅ Ratings data loaded successfully!")
except Exception as e:
    print(f"❌ Error loading Ratings.csv: {e}")
    exit(1)

# Merge books and ratings data
print("Processing data...")
overall_data = pd.merge(books_df, ratings_df, on="ISBN")

# Calculate popular books
print("Calculating popular books...")
num_ratings = overall_data.groupby("title")["rating"].count()
avg_ratings = overall_data.groupby("title")["rating"].mean()
popular_books = pd.DataFrame({"num_ratings": num_ratings, "avg_rating": avg_ratings})
popular_books = (
    popular_books[popular_books["num_ratings"] >= 250]
    .sort_values("avg_rating", ascending=False)
    .head(50)
)
popular_books = popular_books.merge(books_df, on="title").drop_duplicates("title")
popular_books = popular_books[["title", "author", "url", "num_ratings", "avg_rating"]]
print("✅ Popular books calculated!")

# Filter active users
print("Filtering active users...")
active_users = overall_data.groupby("userID")["rating"].count()
active_users = active_users[active_users >= 100].index
filtered_ratings = overall_data[overall_data["userID"].isin(active_users)]
print(f"Filtered ratings shape: {filtered_ratings.shape}")

# Calculate frequent books
print("Calculating frequent books...")
frequent_books = filtered_ratings.groupby("title")["rating"].count()
frequent_books = frequent_books[frequent_books >= 20].index
final_ratings = filtered_ratings[filtered_ratings["title"].isin(frequent_books)]
print(f"Final ratings shape: {final_ratings.shape}")

# Create pivot table
print("Creating pivot table...")
pt = final_ratings.pivot_table(index="title", columns="userID", values="rating").fillna(0)
print(f"Pivot table shape: {pt.shape}")

# Calculate similarity scores
print("Calculating similarity scores...")
similarity_scores = cosine_similarity(pt)
print("✅ Similarity scores calculated!")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "http://localhost:8800"]}})

@app.route("/recommend", methods=["GET"])
def recommend():
    book_name = request.args.get("title")
    if book_name not in pt.index:
        return jsonify({"error": "Book not found"}), 404
    
    index = np.where(pt.index == book_name)[0][0]
    similar_books = sorted(
        list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True
    )[1:10]  

    recommendations = []
    for i in similar_books:
        temp_df = books_df[books_df["title"] == pt.index[i[0]]]
        book_info = temp_df[["title", "author", "url", "price"]].drop_duplicates("title").to_dict(orient="records")[0]
        recommendations.append(book_info)
    
    return jsonify({"recommendations": recommendations})
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)