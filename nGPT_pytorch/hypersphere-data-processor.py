import numpy as np
from scipy.spatial import ConvexHull
from typing import Dict, List, Optional, Union

def project_to_2d(
    normalized_patches: torch.Tensor, 
    angle_theta: float, 
    angle_phi: float,
    projection_type: str = 'linear'
) -> np.ndarray:
    """
    Project 4D hypersphere data to 2D for visualization
    
    Args:
        normalized_patches: Tensor of shape (N, D) where D is the embedding dimension
        angle_theta: Rotation angle in degrees around first axis
        angle_phi: Rotation angle in degrees around second axis
        projection_type: Type of projection ('linear' or 'stereographic')
    
    Returns:
        2D numpy array of projected points
    """
    # Convert angles to radians
    theta = np.radians(angle_theta)
    phi = np.radians(angle_phi)
    
    # Create projection matrix
    projection = np.array([
        [np.cos(theta), -np.sin(theta) * np.cos(phi)],
        [np.sin(theta), np.cos(theta) * np.cos(phi)]
    ])
    
    # Convert to numpy and take first two components for initial projection
    points_4d = normalized_patches.detach().cpu().numpy()
    
    if projection_type == 'stereographic':
        # Implement stereographic projection
        # Project from N-sphere to N-1 dimensional space
        denom = 1 - points_4d[:, -1:]  # Use last dimension as projection reference
        points_projected = points_4d[:, :-1] / (denom + 1e-8)  # Add epsilon to avoid division by zero
        points_2d = points_projected[:, :2] @ projection.T
    else:
        # Linear projection
        points_2d = points_4d[:, :2] @ projection.T
    
    return points_2d

def generate_slice_data(
    normalized_patches: torch.Tensor,
    num_slices: int = 36,
    phi_angles: Optional[List[float]] = None
) -> List[Dict]:
    """
    Generate data for multiple slices around the hypersphere
    
    Args:
        normalized_patches: Normalized embedding tensors
        num_slices: Number of theta angle slices
        phi_angles: Optional list of specific phi angles to use
    
    Returns:
        List of dictionaries containing slice data
    """
    slices_data = []
    
    if phi_angles is None:
        phi_angles = [45]  # Default phi angle
        
    for phi in phi_angles:
        for i in range(num_slices):
            # Calculate theta angle for this slice
            theta = (i * 360 / num_slices)
            
            # Project data to 2D
            points_2d = project_to_2d(normalized_patches, theta, phi)
            
            # Calculate convex hull for the outline
            try:
                hull = ConvexHull(points_2d)
                hull_points = points_2d[hull.vertices]
                hull_area = hull.area
            except:
                hull_points = points_2d
                hull_area = 0
                
            slices_data.append({
                'points': points_2d.tolist(),
                'hull': hull_points.tolist(),
                'hull_area': float(hull_area),
                'angles': {'theta': float(theta), 'phi': float(phi)}
            })
    
    return slices_data

def process_hypersphere_visualization(
    normalized_patches: torch.Tensor,
    num_slices: int = 36,
    phi_angles: Optional[List[float]] = None,
    include_density: bool = False
) -> Dict:
    """
    Main processing function to prepare hypersphere data for visualization
    
    Args:
        normalized_patches: Tensor of normalized embeddings
        num_slices: Number of theta angle slices
        phi_angles: Optional list of specific phi angles
        include_density: Whether to compute density statistics
        
    Returns:
        Dictionary containing visualization data and statistics
    """
    # Generate slice data
    slices = generate_slice_data(normalized_patches, num_slices, phi_angles)
    
    # Calculate statistics
    norms = torch.norm(normalized_patches, dim=-1)
    stats = {
        'mean_norm': float(norms.mean()),
        'std_norm': float(norms.std()),
        'min_norm': float(norms.min()),
        'max_norm': float(norms.max()),
        'num_points': len(normalized_patches)
    }
    
    if include_density:
        # Compute pairwise distances for density estimation
        pairwise_dist = torch.cdist(normalized_patches, normalized_patches)
        avg_neighbor_dist = torch.mean(torch.sort(pairwise_dist)[0][:, 1])  # Exclude self-distance
        stats['average_neighbor_distance'] = float(avg_neighbor_dist)
    
    return {
        'slices': slices,
        'stats': stats,
        'visualization_params': {
            'num_slices': num_slices,
            'phi_angles': phi_angles or [45]
        }
    }