import random
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F


def greedy_diverse_selection(query_embedding, embeddings, K, lambda_param=0.5, device="cuda"):
    """Greedy selection with diversity."""
    query = query_embedding.to(device)  # shape: (D,)
    embeddings = embeddings.to(device)  # shape: (N, D)

    query_norm = F.normalize(query, p=2, dim=0)  # shape: (D,)
    embeddings_norm = F.normalize(embeddings, p=2, dim=1)  # shape: (N, D)

    # Compute cosine similarities between all embeddings and query
    similarities = embeddings_norm @ query_norm.unsqueeze(1)  # (N, D) @ (D, 1) -> (N, 1)
    similarities = similarities.squeeze()  # shape: (N,)

    # Sort indices by descending similarity to query
    sorted_indices = torch.argsort(similarities, descending=True).cpu().tolist()  # list[N]
    remaining = sorted_indices.copy()

    selected_idxs = []  # Will store indices of selected samples
    selected_scores = []

    if not remaining:
        return torch.tensor([]), torch.tensor([])

    # First selection (most similar to the query)
    first_idx = remaining.pop(0)  # selected: [0], remaining: [1, 2, ..., N-1]
    selected_idxs.append(first_idx)
    selected_scores.append(similarities[first_idx].item())

    # Select remaining K-1
    for _ in range(min(K - 1, len(remaining))):
        selected_embs = embeddings_norm[selected_idxs]  # shape: (num_selected, D)
        remaining_embs = embeddings_norm[remaining]  # shape: (num_remaining, D)

        # Compute pairwise similarities between remaining and selected samples
        sim_matrix = remaining_embs @ selected_embs.T  # (num_remaining, D) @ (D, num_selected)
        # -> (num_remaining, num_selected)
        # Get maximum similarity to already selected samples for each remaining candidate
        max_sim, _ = torch.max(sim_matrix, dim=1)  # shape: (num_remaining,)
        # Get current similarities to query for remaining candidates
        current_similarities = similarities[remaining]  # shape: (num_remaining,)

        # Calculate combined scores: relevance - λ * redundancy
        scores = current_similarities - lambda_param * max_sim
        best_idx = torch.argmax(scores).item()
        best_remaining_idx = remaining.pop(best_idx)

        selected_idxs.append(best_remaining_idx)
        selected_scores.append(current_similarities[best_idx].item())

    return (
        torch.tensor(selected_scores, device=device),
        torch.tensor(selected_idxs, device=device, dtype=torch.long),
    )



def get_topk_similar_images(
    embedding_fn: Callable,
    embeddings: torch.Tensor,
    query_image_path: str,
    query_image_embedding: Optional[torch.Tensor] = None,
    topk: int = 10,
    strategy: str = "topk",
    pool_multiplier: int = 5,
    lambda_param: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find indices of top-k most similar training images to query image.

    Args:
        embedding_fn: Function to extract image embeddings (e.g., DinoV2, MedCLIP)
        embeddings: Precomputed embeddings of training images
        query_image_path: Path to query image
        topk: Number of similar images to retrieve
        strategy: Sampling strategy ('topk', 'diverse')
        pool_multiplier: Size of the initial pool for diverse sampling (multiplier of topk)
        lambda_param: Diversity weight for greedy diverse selection

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Similarity scores and indices of selected images
    """
    if query_image_embedding is not None:
        query_feat = query_image_embedding
    else:
        with torch.no_grad():
            query_feat = embedding_fn([query_image_path])

    # Compute similarity
    similarities = F.cosine_similarity(query_feat, embeddings)  # (N)

    if strategy == "topk":
        # Simple top-k selection based on similarity
        scores, indices = torch.topk(similarities, topk)
    elif strategy == "diverse":
        # 1. Choose a large pool of similar images
        scores, indices = torch.topk(similarities, topk * pool_multiplier)
        selected_embeddings = embeddings[indices]

        # 2. Choose top k diverse images from the pool
        scores, subset_idxs = greedy_diverse_selection(
            query_embedding=query_feat[0],
            embeddings=selected_embeddings,
            K=topk,
            lambda_param=lambda_param,
        )
        indices = indices[subset_idxs]
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")

    # Sort selected indices by similarity score
    scores, sort_idxs = torch.sort(scores, descending=True)
    indices = indices[sort_idxs]

    return scores, indices


def select_context(
    query_image_path: str,
    train_set,
    categories: Tuple[int, ...],
    query_image_embedding: Optional[torch.Tensor] = None,
    training_embeddings: Optional[torch.Tensor] = None,
    context_size: int = 5,
    similarity_fn: Optional[Callable] = None,
    sampling_strategy: str = "topk",
    pool_multiplier: int = 5,
    lambda_param: float = 0.5,
) -> List[int]:
    """
    Select context image-mask pairs ensuring each category has at least one context image
    with the foreground mask while prioritizing similar images.

    Args:
        query_image_path: Path to the query image
        image_paths: List of paths to training images
        mask_paths: List of paths to corresponding masks
        categories: Tuple of category IDs to ensure coverage for
        training_embeddings: Pre-computed embeddings for training images
        context_size: Minimum number of context pairs to select
        similarity_fn: Function to compute embeddings for similarity comparison
        sampling_strategy: Strategy for selecting context examples ('topk' or 'diverse')
        pool_multiplier: Pool size multiplier for diverse sampling
        lambda_param: Diversity weight for greedy diverse selection

    Returns:
        Tuple containing:
        - List of paths to selected context images
        - List of paths to corresponding masks
    """

    if similarity_fn is None:
        # For random selection, just shuffle all indices
        all_indices = list(range(len(train_set)))
        random.shuffle(all_indices)
    else:
        assert training_embeddings is not None, "Missing embeddings"
        # Get similarity scores and indices for all images
        all_scores, all_indices = get_topk_similar_images(
            embedding_fn=similarity_fn,
            embeddings=training_embeddings,
            query_image_path=query_image_path,
            query_image_embedding=query_image_embedding,
            topk=len(train_set),  # Get all images ranked by similarity
            strategy="topk",  # Always use topk for initial ranking
            pool_multiplier=pool_multiplier,
            lambda_param=lambda_param,
        )

    # Convert to list for easier manipulation
    all_indices = all_indices.tolist()

    # Initialize selected sets
    selected_indices = []
    covered_categories = set()
    remaining_indices = all_indices.copy()

    # First pass: ensure each category has at least one image
    while remaining_indices and len(selected_indices) < context_size:
        # Get categories still needing coverage
        needed_categories = set(categories) - covered_categories
        if not needed_categories:
            break

        # Try each remaining image in order of similarity
        found = False
        for idx in remaining_indices[:]:
            masks = train_set.get_mask_array(idx)  # {category_id: torch.tensor}
            cat_ids = set(masks.keys())

            # If this mask has any needed category
            if cat_ids & needed_categories:
                selected_indices.append(idx)
                covered_categories.update(cat_ids)
                remaining_indices.remove(idx)
                found = True
                break

        # If we couldn't find any image with needed categories, break
        if not found:
            break

    # Second pass: fill up to context_size with most similar remaining images
    while len(selected_indices) < context_size and remaining_indices:
        next_idx = remaining_indices.pop(0)
        selected_indices.append(next_idx)

    # Sort selected indices by similarity (original ranking)
    selected_indices.sort(key=lambda x: all_indices.index(x))
    # We need to reverse the order of selected images s.t. the least similar image is first and the most similar image is last (i.e., next to the query image)
    selected_indices = selected_indices[::-1]

    return selected_indices
