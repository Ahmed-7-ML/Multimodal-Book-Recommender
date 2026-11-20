"""Sort the retrieved results using Scoring & Business Rules
Steps:
1. Take the candidates from retrieval phase
2. Apply Filtering, Deduplication
3. Display the rest of results

Post-retrieval ranking:
    1. Filter low-relevance results.
    2. Exclude specified ISBNs.
    3. Remove duplicate books.
    4. Optionally diversify results by author.
    5. Return the final top-K recommendations.

The original Qdrant similarity score is preserved.
"""
from typing import Any, Iterable

class BookRanker:
    """Rank and filter retrieved book candidates"""
    def rank(
        self,
        candidates: list[dict[str, Any]],
        top_k: int = 10,
        min_score: float | None = None,  # Cosine Similarity Threshold
        exclude_isbns: Iterable[str] | None = None,
        max_per_author: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Rank retrieved candidates.
        Args:
            candidates:
                Results returned by BookRetriever.
            top_k:
                Maximum number of final recommendations.
            min_score:
                Minimum similarity score. None disables score filtering.
            exclude_isbns:
                ISBNs to exclude from the results.
            max_per_author:
                Maximum number of recommendations per author.
                None disables author diversification.
        Returns:
            Ranked list of book dictionaries.
        """
        if top_k < 1:
            raise ValueError("top_k must be a positive integer.")
        if min_score is not None and not -1 <= min_score <= 1:
            raise ValueError("min_score must be between -1 and 1.")
        if max_per_author is not None and max_per_author < 1:
            raise ValueError("max_per_author must be positive.")

        excluded = {
            str(isbn).strip()
            for isbn in (exclude_isbns or [])
        }

        # Sort by similarity score, descending.
        candidates = sorted(
            candidates,
            key=lambda item: float(item.get("score", 0.0)),
            reverse=True,
        )
        
        ranked = []
        seen_isbns = set()
        author_counts = {}
        for candidate in candidates:
            score = float(candidate.get("score", 0.0))
            payload = candidate.get("payload") or {}
            isbn = str(payload.get("isbn") or "").strip()

            # Ignore candidates without a valid ISBN.
            if not isbn:
                continue
            
            # Exclude low-scoring results.
            if min_score is not None and score < min_score:
                continue
            
            # Exclude requested ISBNs.
            if isbn in excluded:
                continue
            
            # Remove duplicate books.
            if isbn in seen_isbns:
                continue
            author = payload.get("author")

            # Missing authors are treated separately.
            author_key = (
                str(author).strip().casefold()
                if author and str(author).strip()
                else f"unknown:{isbn}"
            )

            # Optional author diversification.
            if max_per_author is not None:
                if author_counts.get(author_key, 0) >= max_per_author:
                    continue

            seen_isbns.add(isbn)
            author_counts[author_key] = (
                author_counts.get(author_key, 0) + 1
            )
            ranked.append(
                {
                    "rank": len(ranked) + 1,
                    "id": candidate.get("id"),
                    "isbn": isbn,
                    "title": payload.get("title"),
                    "author": payload.get("author"),
                    "year": payload.get("year"),
                    "publisher": payload.get("publisher"),
                    "image_available": payload.get("image_available", False),
                    "score": score,
                }
            )
            if len(ranked) >= top_k:
                break

        return ranked
