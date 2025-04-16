from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the pre-trained Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

def getrank(user_query, product_description, priority=1):
    """
    Computes a rank based on the similarity between user_query and product_description,
    weighted by a priority multiplier using Sentence-BERT embeddings.

    Args:
        user_query (str): The search query from the user.
        product_description (str): Description of the product to compare.
        priority (float): A multiplier indicating importance (default is 1).

    Returns:
        float: A rank score between 0 and priority.
    """
    if not user_query or not product_description:
        return 0.0

    # Get embeddings for both texts using Sentence-BERT
    query_embedding = model.encode(user_query)
    description_embedding = model.encode(product_description)

    # Compute cosine similarity
    similarity = cosine_similarity([query_embedding], [description_embedding])[0][0]

    # Return similarity weighted by priority
    return similarity * priority
