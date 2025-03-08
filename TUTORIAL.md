# nGPT Explorer Tutorial

This tutorial will guide you through using the nGPT Explorer to understand the nGPT architecture and how it works on the hypersphere.

## Getting Started

1. Connect your Xbox controller to your computer
2. Run `test_controller.py` to make sure your controller is recognized
3. Run `ngpt_explorer.py` to start the explorer

If you don't have an Xbox controller, you can use keyboard controls instead:
- WASD: Navigate
- Arrow keys: Adjust parameters
- Space: Generate text
- Tab: Toggle visualization mode

## Tutorial 1: Understanding Attention

1. Start the explorer and make sure you're in the "Attention Visualization" mode
2. Move the left stick slowly left and right
3. Notice how the attention pattern changes - this shows how tokens attend to each other
4. When the stick is to the left, attention is more focused on nearby tokens
5. When the stick is to the right, attention is more distributed
6. Press A to generate text and see how the attention pattern affects the output

**What you're learning**: The attention mechanism is a key component of transformers. It allows the model to focus on different parts of the input sequence when making predictions. In nGPT, attention uses cosine similarity instead of dot products, which means it's operating on the hypersphere.

## Tutorial 2: Exploring Token Embeddings

1. Press X to switch to the "Token Embeddings" visualization
2. Use the left stick to rotate the visualization (left/right) and zoom (up/down)
3. Notice how the token embeddings are distributed on the unit circle (2D projection of the hypersphere)
4. The red circle represents the unit hypersphere
5. Each point is a token embedding, normalized to have unit length
6. Press A to generate text from your current position

**What you're learning**: In nGPT, all token embeddings are normalized to lie on the hypersphere. This means their direction is more important than their magnitude. The cosine similarity between two embeddings (the angle between them) determines their relationship.

## Tutorial 3: Layer Activations

1. Press X again to switch to the "Layer Activations" visualization
2. Move the left stick to see how activations change across layers
3. Notice how different layers respond to different patterns
4. Press the Left/Right Bumpers to decrease/increase model depth
5. Observe how adding more layers affects the activation patterns
6. Press A to generate text and see how the layer activations affect the output

**What you're learning**: Transformers process information through multiple layers. Each layer learns different patterns and abstractions. In nGPT, all layer activations are normalized, which helps with training stability and generalization.

## Tutorial 4: Temperature and Randomness

1. Use the right stick (up/down) to adjust the temperature parameter
2. Press A to generate text at different temperature settings
3. Notice how lower temperatures produce more deterministic text
4. Higher temperatures produce more random, creative text
5. Try extreme values (very low or very high) to see the effect

**What you're learning**: Temperature controls the randomness of text generation. In nGPT, the temperature affects how logits are scaled before sampling, which influences the diversity of the generated text.

## Tutorial 5: Model Complexity

1. Use the Left/Right Bumpers to adjust model depth
2. Notice how the model's behavior changes with different depths
3. A deeper model can capture more complex patterns
4. But it might also be slower to visualize
5. Try generating text with different model depths

**What you're learning**: Model depth is a key hyperparameter in transformers. More layers allow the model to learn more complex patterns, but also increase computational requirements.

## Advanced Exploration

Once you're comfortable with the basics, try these advanced explorations:

1. **Attention Patterns**: Try to identify specific attention patterns (e.g., diagonal attention for copying, vertical attention for classification)

2. **Embedding Clusters**: In the token embedding visualization, look for clusters of related tokens

3. **Layer Specialization**: In the layer activations view, try to identify what different layers might be specializing in

4. **Parameter Sensitivity**: See how small changes in position can lead to large changes in output

5. **Creative Text Generation**: Find positions in the hypersphere that generate interesting or creative text

## Understanding the Hypersphere

The key innovation in nGPT is that all representations lie on the hypersphere. This has several advantages:

1. **Training Stability**: Prevents exploding gradients by constraining vectors to unit length

2. **Efficient Representation**: Focuses on directions rather than magnitudes

3. **Better Generalization**: May lead to better generalization by focusing on angles between vectors

4. **Continual Learning**: Potentially better for continual learning scenarios

As you explore the nGPT Explorer, pay attention to how the hyperspherical constraint affects the model's behavior and generated text.

## Next Steps

After completing this tutorial, you might want to:

1. Modify the explorer code to visualize other aspects of the model
2. Experiment with different model parameters
3. Compare nGPT with standard transformer architectures
4. Read the original nGPT paper for more details on the theoretical background

Happy exploring! 