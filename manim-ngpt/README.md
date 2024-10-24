# nGPT Visualization Plan with Learning Path

## Prerequisite Notebooks
1. `01_vectors_and_norms.ipynb`
   - Basic vector operations 
     - [vector on a hypersphere animation script](./l2-norm-visualization.py)
   - Understanding L2 norm
   - Visualization of vector normalization
   - Exercises with PyTorch vector operations
   - Pairs with Scene 2 (L2 Normalization)

2. `02_transformer_basics.ipynb`
   - Standard transformer architecture
   - Self-attention mechanism
   - Linear layers and embeddings
   - Simple implementation in PyTorch
   - Foundation for understanding nGPT modifications

3. `03_hypersphere_geometry.ipynb`
   - Understanding unit hyperspheres
   - Visualizing points on spheres
   - Great circles and geodesics
   - SLERP fundamentals
   - Supports Scene 1 and Scene 4

## Core Concept Notebooks
4. `04_normalized_linear_layers.ipynb`
   - Building NormLinear from scratch
   - Understanding weight parametrization
   - Visualization of weight constraints
   - Performance comparisons
   - Companions Scene 3

5. `05_residual_slerp.ipynb`
   - Traditional residuals vs SLERP
   - Implementing SLERP in PyTorch
   - Interpolation factor analysis
   - Visualization of paths on sphere
   - Enhances Scene 4

6. `06_scaled_attention.ipynb`
   - nGPT attention modifications
   - QK normalization implementation
   - Scale factor experiments
   - Attention pattern analysis
   - Supports Scene 5

## Advanced Concept Notebooks
7. `07_scale_management.ipynb`
   - Understanding all scaling factors
   - Learning rate relationships
   - Experiments with different scales
   - Optimization insights
   - Pairs with Scene 6

8. `08_rotary_embeddings.ipynb`
   - RoPE implementation
   - Position encoding comparison
   - Rotation visualization
   - Integration with normalized attention
   - Additional context for Scene 5

## Implementation Notebooks
9. `09_mini_ngpt.ipynb`
   - Minimal nGPT implementation
   - Step-by-step building blocks
   - Training on tiny dataset
   - Performance analysis
   - Complements Scene 7

10. `10_optimization_tricks.ipynb`
    - Flash attention integration
    - Memory efficiency tips
    - Training stabilization
    - Hyperparameter tuning
    - Advanced insights for all scenes

## Scene 1: Traditional vs Normalized Learning
- Start with two side-by-side neural networks (simplified)
- Left side: Traditional network with unconstrained weights
  - Show weights growing/shrinking independently
  - Visualize potential instability with arrows of varying lengths
- Right side: nGPT with normalized weights
  - Show weights always maintaining unit length
  - Visualize how weights move along hypersphere surface
  - Add subtle grid lines on sphere surface to show movement
- Key Animation Points:
  - Weight updates causing traditional network to have varying magnitudes
  - nGPT weights staying on unit hypersphere during updates
*Supporting notebooks: 01, 02, 03*

## Scene 2: L2 Normalization Deep Dive
- Start with a vector in 3D space
- Show normalization process:
  1. Calculate vector magnitude (animate length calculation)
  2. Divide each component by magnitude (show vector scaling)
  3. Result: Vector now has length 1
- Demonstrate multiple vectors being normalized
- Show how normalized vectors always end up on unit sphere surface
- Visualize the `norm_eps` parameter:
  - Show how it creates a "shell" around the unit sphere
  - Animate vectors being allowed to have lengths between (1-ε) and (1+ε)
*Supporting notebooks: 01, 03*

## Scene 3: Normalized Linear Layers
- Start with standard linear layer matrix
- Show transformation into normalized version:
  1. Original weight matrix
  2. L2 normalization applied to rows/columns
  3. Final normalized weight matrix
- Split screen comparison:
  - Left: Traditional linear layer output
  - Right: Normalized linear layer output
- Visualize how the `groups` parameter affects normalization:
  - Show matrix being split into groups
  - Each group normalized independently
*Supporting notebooks: 04*

## Scene 4: Residual Connections with SLERP
- Compare traditional residual connections vs nGPT's SLERP
- Visual demonstration of Spherical Linear Interpolation:
  1. Show two points on sphere surface
  2. Traditional linear interpolation (straight line through sphere)
  3. SLERP interpolation (arc along sphere surface)
- Animate learned interpolation factor:
  - Show how α controls interpolation
  - Visualize effect of different α values
*Supporting notebooks: 03, 05*

## Scene 5: Attention Mechanism Modifications
- Split screen showing traditional vs nGPT attention:
  - Q,K,V computation
  - Optional QK normalization
  - Scaling factors
- Animate attention process:
  1. Input embeddings
  2. Normalized linear transformations
  3. Query-key dot products
  4. Value aggregation
- Show how rotary embeddings are preserved
*Supporting notebooks: 02, 06, 08*

## Scene 6: Scale Management
- Visualize the various scaling factors:
  - `s_qk_init` and `s_qk_scale`
  - `s_hidden_init` and `s_hidden_scale`
  - `s_gate_init` and `s_gate_scale`
- Show how these control relative learning rates
- Animate effect of different scale values on:
  - Attention mechanism
  - Feed-forward networks
  - Final logits
*Supporting notebooks: 07*

## Scene 7: Complete Architecture Overview
- Start zoomed out showing full nGPT architecture
- Zoom into each component as they're highlighted:
  1. Input embeddings
  2. Attention blocks with normalization
  3. Feed-forward blocks
  4. Residual connections
  5. Output layer
- Show data flow through network
- Highlight normalization points throughout
*Supporting notebooks: 09, 10*

## Learning Path Integration
1. For each scene:
   - Start with relevant prerequisite notebooks
   - Watch Manim visualization
   - Complete exercises in core notebooks
   - Experiment with code implementations

2. Progressive Complexity:
   - Begin with vector/norm fundamentals
   - Build up to transformer basics
   - Introduce nGPT innovations
   - Culminate in complete implementation

3. Hands-on Components:
   - Each notebook includes:
     - Theory explanation
     - Code implementation
     - Visualization tools
     - Exercises with solutions
     - Performance experiments

4. Visualization-Notebook Synergy:
   - Manim scenes reference notebook concepts
   - Notebooks include static versions of animations
   - Cross-references between related materials
   - Progressive reveal of complexity

## Implementation Notes
1. Use consistent color coding:
   - Blue for traditional components
   - Green for normalized components
   - Red for important transformation points
   - Yellow for scaling factors

2. Visual Conventions:
   - Dashed lines for optional components
   - Thick arrows for main data flow
   - Thin arrows for auxiliary operations
   - Dotted circles for normalization boundaries

3. Interactive Elements:
   - Add pause points at key concepts
   - Use highlighting to emphasize active components
   - Include parameter value displays where relevant

4. Mathematical Notation:
   - Show equations alongside visualizations
   - Use color coding to match equations with visual elements
   - Include simplified versions for clarity

5. Technical Details:
   - Maintain 60fps for smooth animations
   - Use 3D scenes for hypersphere visualizations
   - Add camera rotations for better depth perception
   - Include grid lines and axes where helpful

6. Educational Flow:
   - Build complexity gradually
   - Review previous concepts when introducing new ones
   - Include transition scenes between major concepts
   - End each scene with key takeaways

## Additional Resources in Notebooks
1. Code Snippets:
   - PyTorch implementations
   - Debugging tools
   - Performance profilers
   - Testing utilities

2. Visualization Tools:
   - Plotly for interactive plots
   - Matplotlib for static visuals
   - TensorBoard integration
   - Custom visualization functions

3. Dataset Examples:
   - Synthetic data for concept testing
   - Small real-world examples
   - Benchmarking datasets
   - Performance comparisons

4. Reference Materials:
   - Links to papers
   - Additional reading
   - Common pitfalls
   - Best practices
