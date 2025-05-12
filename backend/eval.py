import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

import mlflow

# --- Evaluation Metrics for Recommendation Systems ---

def calculate_top_k_metrics(model, test_loader, attraction_ids, k_values=[1, 5, 10, 20], device='cpu'):
    """
    Calculate various top-k evaluation metrics for the recommendation model.
    
    Args:
        model: The trained PyTorch model
        test_loader: DataLoader containing test data
        attraction_ids: Tensor or array of attraction IDs corresponding to test data
        k_values: List of k values for evaluation metrics
        device: Computing device (cpu or cuda)
        
    Returns:
        Dictionary containing all evaluation metrics
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_attraction_ids = []
    
    # Get all predictions and true labels
    with torch.no_grad():
        for b_text, b_loc, b_item_s, b_cat, b_query_s, b_labels in test_loader:
            b_text = b_text.to(device)
            b_loc = b_loc.to(device)
            b_item_s = b_item_s.to(device)
            b_cat = b_cat.to(device)
            b_query_s = b_query_s.to(device)
            
            # Get model predictions
            outputs = model(b_text, b_loc, b_item_s, b_cat, b_query_s)
            outputs = torch.sigmoid(outputs)  # Convert to probabilities
            
            # Collect predictions and labels
            all_predictions.extend(outputs.cpu().numpy())
            all_labels.extend(b_labels.cpu().numpy())
    
    all_predictions = np.array(all_predictions).flatten()
    all_labels = np.array(all_labels).flatten()
    
    # Group by query (assuming test_loader maintains query structure)
    # This is simplified - in a real scenario, you'd group by actual query IDs
    # Here we're assuming each group of samples represents possible responses for a query
    # Each batch might contain results for multiple queries
    
    # For demonstration, let's create an artificial grouping
    # In practice, you would use actual query identifiers from your data
    query_groups = {}
    current_idx = 0
    
    # For each batch in the test loader
    batch_size = test_loader.batch_size
    num_batches = len(test_loader)
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, len(all_predictions))
        
        # Get predictions and labels for this batch
        batch_preds = all_predictions[batch_start:batch_end]
        batch_labels = all_labels[batch_start:batch_end]
        
        # Create a query ID for this batch
        query_id = f"query_{batch_idx}"
        query_groups[query_id] = {
            'predictions': batch_preds,
            'labels': batch_labels,
            'indices': list(range(batch_start, batch_end))
        }
    
    # Calculate metrics
    results = {}
    
    # Initialize metrics for each k
    for k in k_values:
        results[f'precision@{k}'] = 0
        results[f'recall@{k}'] = 0
        results[f'ndcg@{k}'] = 0
    results['mrr'] = 0
    
    # Calculate metrics for each query
    num_queries = len(query_groups)
    
    for query_id, group in query_groups.items():
        preds = group['predictions']
        labels = group['labels']
        indices = group['indices']
        
        # Sort items by prediction score (descending)
        sorted_indices = np.argsort(-preds)
        sorted_labels = labels[sorted_indices]
        
        # Calculate MRR (Mean Reciprocal Rank)
        # Find the rank of the first relevant item
        relevant_ranks = np.where(sorted_labels > 0)[0]
        if len(relevant_ranks) > 0:
            # +1 because ranks are 1-based, not 0-based
            results['mrr'] += 1.0 / (relevant_ranks[0] + 1)
        
        # Calculate metrics for each k
        for k in k_values:
            # Only consider top k predictions
            top_k_labels = sorted_labels[:k]
            
            # Precision@k: proportion of recommended items that are relevant
            results[f'precision@{k}'] += np.sum(top_k_labels > 0) / min(k, len(top_k_labels))
            
            # Recall@k: proportion of relevant items that are recommended
            total_relevant = np.sum(labels > 0)
            if total_relevant > 0:
                results[f'recall@{k}'] += np.sum(top_k_labels > 0) / total_relevant
            
            # NDCG@k: normalized discounted cumulative gain
            # Create ideal DCG - sort by true relevance
            ideal_sorted_labels = np.sort(labels)[::-1]
            ideal_dcg = 0
            dcg = 0
            
            for i in range(min(k, len(top_k_labels))):
                # Using log2(i+2) because i is 0-indexed
                dcg += top_k_labels[i] / np.log2(i + 2)
                ideal_dcg += ideal_sorted_labels[i] / np.log2(i + 2)
            
            # Avoid division by zero
            if ideal_dcg > 0:
                results[f'ndcg@{k}'] += dcg / ideal_dcg
    
    # Average metrics across all queries
    for metric in results:
        results[metric] /= max(1, num_queries)
    
    return results

# --- Main Evaluation Function ---

def evaluate_model_with_metrics(model, test_data, attraction_ids, batch_size=64, k_values=[1, 5, 10, 20], device='cpu'):
    """
    Evaluate the model with multiple recommendation system metrics.
    
    Args:
        model: The trained PyTorch model
        test_data: Tuple of (X_text, X_loc, X_item_s, X_cat, X_query_s, y)
        attraction_ids: Array of attraction IDs for test data
        batch_size: Batch size for evaluation
        k_values: List of k values for evaluation metrics
        device: Computing device (cpu or cuda)
        
    Returns:
        Dictionary of evaluation metrics
    """
    X_text, X_loc, X_item_s, X_cat, X_query_s, y = test_data
    
    # Create DataLoader
    test_dataset = TensorDataset(X_text, X_loc, X_item_s, X_cat, X_query_s, y)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Calculate metrics
    metrics = calculate_top_k_metrics(model, test_loader, attraction_ids, k_values, device)
    
    return metrics

# --- Integration with MLflow ---

def log_evaluation_metrics(metrics, run_id=None):
    """
    Log evaluation metrics to MLflow with sanitized metric names.
    
    Args:
        metrics: Dictionary of evaluation metrics
        run_id: Optional MLflow run ID. If None, uses active run.
    """
    # Check if we already have an active run
    active_run = mlflow.active_run()
    
    # Create sanitized metrics dictionary
    sanitized_metrics = {}
    for metric_name, value in metrics.items():
        # Replace @ with _at_ to make MLflow happy
        sanitized_name = metric_name.replace('@', '_at_')
        sanitized_metrics[sanitized_name] = value
    
    if run_id and active_run and active_run.info.run_id == run_id:
        # We're already in the correct run, no need to start a new one
        for metric_name, value in sanitized_metrics.items():
            mlflow.log_metric(metric_name, value)
    elif run_id:
        # We need to start a specific run
        with mlflow.start_run(run_id=run_id):
            for metric_name, value in sanitized_metrics.items():
                mlflow.log_metric(metric_name, value)
    else:
        # No run_id specified, use active run or create a new one
        for metric_name, value in sanitized_metrics.items():
            mlflow.log_metric(metric_name, value)
    
    print("Logged evaluation metrics to MLflow:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

# --- Usage in Training Script ---

# Add this code to the end of your training script, after the model is trained

def add_evaluation_to_training(model, X_text_val, X_loc_val, X_item_s_val, X_cat_val, X_query_s_val, y_val,
                               attraction_ids_val, device, batch_size=64, run_id=None, return_metrics=False):
    """
    Add evaluation metrics to an already trained model.
    """
    print("\n--- Evaluating Model with Recommendation Metrics ---")

    # Prepare validation data
    test_data = (X_text_val, X_loc_val, X_item_s_val, X_cat_val, X_query_s_val, y_val)

    # Calculate metrics
    metrics = evaluate_model_with_metrics(
        model, test_data, attraction_ids_val, batch_size=batch_size,
        k_values=[1, 5, 10, 20], device=device
    )

    # Sanitize metric names for MLflow
    sanitized_metrics = {}
    for key, value in metrics.items():
        sanitized_key = key.replace('@', '_at_')
        sanitized_metrics[sanitized_key] = value

    # Log metrics to MLflow
    if run_id:
        with mlflow.start_run(run_id=run_id, nested=True):
            for metric_name, value in sanitized_metrics.items():
                mlflow.log_metric(metric_name, value)
    else:
        for metric_name, value in sanitized_metrics.items():
            mlflow.log_metric(metric_name, value)

    print("Logged evaluation metrics to MLflow:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

    if return_metrics:
        return sanitized_metrics
    return None