import pymongo
import random
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
from dotenv import dotenv_values


config = dotenv_values(".env")

MONGO_URL2 = config.get("MONGO_URL2")

try:
    client = pymongo.MongoClient(MONGO_URL2)
    db = client["book-verse"]
    db.command("ping")
    print("✅ Successfully connected to MongoDB!")
except Exception as e:
    print(f"❌ MongoDB Connection Error: {e}")

books_collection = db["books"]
ratings_collection = db["ratings"]


books_df = pd.DataFrame(list(books_collection.find({}, {"_id": 0})))
ratings_df = pd.DataFrame(list(ratings_collection.find({}, {"_id": 0})))


overall_data = ratings_df.merge(books_df, on="ISBN")

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

active_users = overall_data.groupby("userID")["rating"].count()
active_users = active_users[active_users >= 100].index
filtered_ratings = overall_data[overall_data["userID"].isin(active_users)]
print(f"Filtered ratings shape: {filtered_ratings.shape}")


frequent_books = filtered_ratings.groupby("title")["rating"].count()
frequent_books = frequent_books[frequent_books >= 20].index
final_ratings = filtered_ratings[filtered_ratings["title"].isin(frequent_books)]
print(f"Final ratings shape: {final_ratings.shape}")


pt = final_ratings.pivot_table(index="title", columns="userID", values="rating").fillna(0)
print(f"Pivot table shape: {pt.shape}")


similarity_scores = cosine_similarity(pt)

# ✅ Flask App
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



# Load environment variables

# Initialize Flask App

