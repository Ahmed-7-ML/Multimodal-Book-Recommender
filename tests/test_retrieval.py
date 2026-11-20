
from book_recommender.retrieval.retriever import BookRetriever
from book_recommender.recommendation.ranker import BookRanker


def main():
    query = "A science fiction novel about space exploration"
    with BookRetriever() as retriever:
        # Retrieve more candidates than the desired final results.
        candidates = retriever.retrieve_by_text(
            query=query,
            limit=50,
        )
        ranker = BookRanker()
        recommendations = ranker.rank(
            candidates=candidates,
            top_k=10,
            max_per_author=2,
        )
        for book in recommendations:
            print("-" * 60)
            print(f"Rank: {book['rank']}")
            print(f"Title: {book['title']}")
            print(f"Author: {book['author']}")
            print(f"ISBN: {book['isbn']}")
            print(f"Year: {book['year']}")
            print(f"Similarity: {book['score']:.4f}")
            print(f"Cover available: {book['image_available']}")

if __name__ == "__main__":
    main()
