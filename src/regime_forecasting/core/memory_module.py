"""
Memory Module for storing and retrieving regime checkpoints
"""
import pickle
import os
from typing import Dict, List, Optional, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)


class Checkpoint:
    """Represents a single regime checkpoint"""
    def __init__(
        self, 
        regime_id: str,
        weights: Dict[str, Any],
        regime_features: np.ndarray,
        metadata: Dict[str, Any]
    ):
        self.regime_id = regime_id
        self.weights = weights
        self.regime_features = regime_features
        self.metadata = metadata
        self.created_at = metadata.get('timestamp', None)
        self.performance = metadata.get('performance', None)


class MemoryModule:
    """
    Manages storage and retrieval of regime checkpoints
    """
    def __init__(self, storage_path: str = "./checkpoints", similarity_threshold: float = 0.8):
        self.storage_path = storage_path
        self.similarity_threshold = similarity_threshold
        self.checkpoints: Dict[str, Checkpoint] = {}
        self.regime_counter = 0
        
        # Create storage directory
        os.makedirs(storage_path, exist_ok=True)
        
        # Load existing checkpoints
        self._load_checkpoints()
    
    def _validate_weights(self, weights: Dict[str, Any]) -> bool:
        """Check if weights contain NaN or Inf values."""
        import torch
        for name, tensor in weights.items():
            # Handle both torch tensors and numpy arrays
            if isinstance(tensor, torch.Tensor):
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    return False
            elif isinstance(tensor, np.ndarray):
                if np.isnan(tensor).any() or np.isinf(tensor).any():
                    return False
        return True

    def save_checkpoint(
        self, 
        weights: Dict[str, Any], 
        regime_features: np.ndarray,
        metadata: Dict[str, Any],
        regime_id: Optional[str] = None
    ) -> str:
        """
        Save a new checkpoint
        
        Args:
            weights: Model state dict
            regime_features: Feature vector describing the regime
            metadata: Additional metadata (timestamp, performance, etc.)
            regime_id: Optional regime ID, will be generated if not provided
            
        Returns:
            regime_id: The ID of the saved checkpoint
        """
        # Validate weights don't have NaN
        if not self._validate_weights(weights):
            logger.warning(f"⚠️ Refusing to save checkpoint with NaN/Inf weights")
            return None
            
        if regime_id is None:
            regime_id = f"regime_{self.regime_counter:03d}"
            self.regime_counter += 1
        
        checkpoint = Checkpoint(regime_id, weights, regime_features, metadata)
        self.checkpoints[regime_id] = checkpoint
        
        # Save to disk
        checkpoint_path = os.path.join(self.storage_path, f"{regime_id}.pkl")
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        logger.info(f"💾 Checkpoint saved: {regime_id}")
        return regime_id
    
    def find_similar_regime(self, query_features: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Find the most similar regime checkpoint
        
        Args:
            query_features: Feature vector to match against
            
        Returns:
            Dict with checkpoint info and similarity score, or None if no match
        """
        if not self.checkpoints:
            return None
        
        best_similarity = -1.0
        best_checkpoint = None

        q = query_features.flatten()

        for checkpoint in self.checkpoints.values():
            s = checkpoint.regime_features.flatten()
            # Use Euclidean distance transformed to similarity to be sensitive to scale
            dist = float(np.linalg.norm(q - s))
            # Normalize distance by average magnitude to keep scale-invariant range
            norm = (np.linalg.norm(q) + np.linalg.norm(s)) / 2.0 + 1e-8
            sim = 1.0 / (1.0 + (dist / norm))

            if sim > best_similarity:
                best_similarity = sim
                best_checkpoint = checkpoint

        if best_similarity >= self.similarity_threshold:
            logger.info(f"📚 Found similar regime: {best_checkpoint.regime_id} (similarity: {best_similarity:.3f})")
            return {
                'regime_id': best_checkpoint.regime_id,
                'weights': best_checkpoint.weights,
                'similarity': best_similarity,
                'metadata': best_checkpoint.metadata
            }
        else:
            logger.info(f"🆕 No similar regime found (best similarity: {best_similarity:.3f} < {self.similarity_threshold})")
            return None
    
    def get_checkpoint(self, regime_id: str) -> Optional[Checkpoint]:
        """Get a specific checkpoint by regime ID"""
        return self.checkpoints.get(regime_id, None)
    
    def list_checkpoints(self) -> List[str]:
        """List all available checkpoint IDs"""
        return list(self.checkpoints.keys())
    
    def get_regime_info(self) -> Dict[str, Any]:
        """Get summary information about all regimes"""
        return {
            'total_regimes': len(self.checkpoints),
            'regime_ids': self.list_checkpoints(),
            'latest_regime': max(self.checkpoints.keys()) if self.checkpoints else None,
            'similarity_threshold': self.similarity_threshold
        }
    
    def update_checkpoint(self, regime_id: str, weights: Dict[str, Any], metadata: Dict[str, Any]):
        """Update an existing checkpoint"""
        if regime_id in self.checkpoints:
            self.checkpoints[regime_id].weights = weights
            self.checkpoints[regime_id].metadata.update(metadata)
            
            # Save updated checkpoint
            checkpoint_path = os.path.join(self.storage_path, f"{regime_id}.pkl")
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(self.checkpoints[regime_id], f)
            
            logger.info(f"🔄 Updated checkpoint: {regime_id}")
        else:
            logger.warning(f"⚠️ Checkpoint {regime_id} not found for update")
    
    def delete_checkpoint(self, regime_id: str):
        """Delete a checkpoint"""
        if regime_id in self.checkpoints:
            del self.checkpoints[regime_id]
            
            # Delete from disk
            checkpoint_path = os.path.join(self.storage_path, f"{regime_id}.pkl")
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            
            logger.info(f"🗑️ Deleted checkpoint: {regime_id}")
        else:
            logger.warning(f"⚠️ Checkpoint {regime_id} not found for deletion")
    
    def _load_checkpoints(self):
        """Load existing checkpoints from disk, skip invalid ones"""
        if not os.path.exists(self.storage_path):
            return
        
        loaded_count = 0
        skipped_count = 0
        
        for filename in os.listdir(self.storage_path):
            if filename.endswith('.pkl'):
                filepath = os.path.join(self.storage_path, filename)
                try:
                    with open(filepath, 'rb') as f:
                        checkpoint = pickle.load(f)
                        
                        # Validate checkpoint weights don't have NaN
                        if not self._validate_weights(checkpoint.weights):
                            logger.warning(f"⚠️ Skipping checkpoint {filename}: contains NaN/Inf weights")
                            # Delete the corrupted checkpoint file
                            os.remove(filepath)
                            skipped_count += 1
                            continue
                        
                        self.checkpoints[checkpoint.regime_id] = checkpoint
                        loaded_count += 1
                        
                        # Update counter
                        if checkpoint.regime_id.startswith('regime_'):
                            regime_num = int(checkpoint.regime_id.split('_')[1])
                            self.regime_counter = max(self.regime_counter, regime_num + 1)
                        
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load checkpoint {filename}: {e}")
                    skipped_count += 1
        
        logger.info(f"📁 Loaded {loaded_count} checkpoints from disk (skipped {skipped_count} invalid)")
    
    def clear_all_checkpoints(self):
        """Clear all checkpoints (use with caution!)"""
        for regime_id in list(self.checkpoints.keys()):
            self.delete_checkpoint(regime_id)
        
        self.regime_counter = 0
        logger.info("🧹 All checkpoints cleared")