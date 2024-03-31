import tkinter as tk
from tkinter import ttk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def recommend_similar_books(csv_file_path, user_book_title, num_recommendations=5):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Check if the user's book title is in the DataFrame
    if user_book_title not in df['title'].values:
        return []  # Return an empty list if user's book is not found

    # Initialize the TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    # Fit and transform the 'summary' column to compute TF-IDF vectors
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['summary'])

    # Get the index of the user's book title
    user_book_index = df[df['title'] == user_book_title].index[0]

    # Compute the cosine similarity between the user's book and all other books
    cosine_similarities = linear_kernel(tfidf_matrix[user_book_index], tfidf_matrix).flatten()

    # Get indices of books with highest similarity scores
    similar_books_indices = cosine_similarities.argsort()[:-1-num_recommendations-1:-1]

    # List to store recommended books
    recommended_books = []
    for index in similar_books_indices:
        recommended_book = df.loc[index, 'title']
        if recommended_book != user_book_title:
            recommended_books.append(recommended_book)
    return recommended_books

def get_recommendations():
    user_book_title = entry_title.get()
    recommended_books = recommend_similar_books('C:\\Users\\jdyso\\OneDrive\\Desktop\\data.csv', user_book_title, num_recommendations=5)
    result_text.config(state='normal')
    result_text.delete('1.0', tk.END)
    if recommended_books:
        for i, book in enumerate(recommended_books, 1):
            result_text.insert(tk.END, f"{i}. {book}\n")
    else:
        result_text.insert(tk.END, "No recommendations found.")
    result_text.config(state='disabled')

# Create the main window
root = tk.Tk()
root.title("Book Recommendation System")

# Create and place widgets
label_title = tk.Label(root, text="Enter the title of the book you have read:")
label_title.pack(pady=10)

entry_title = tk.Entry(root, width=50)
entry_title.pack()

btn_recommend = ttk.Button(root, text="Get Recommendations", command=get_recommendations)
btn_recommend.pack(pady=10)

result_text = tk.Text(root, height=10, width=50, state='disabled')
result_text.pack()

root.mainloop()