# nGPT Explorer - Interactive Learning Suite

An interactive visualization tool for exploring and understanding the nGPT (normalized GPT) architecture using an Xbox controller.

![nGPT Explorer Screenshot](https://placeholder-for-screenshot.png)

## Overview

nGPT Explorer is a learning suite designed to help you understand the nGPT architecture through interactive visualization. It allows you to:

1. Visualize attention patterns, token embeddings, and layer activations
2. Navigate through the hypersphere using an Xbox controller's analog sticks
3. Generate text based on your position in the embedding space
4. Experiment with different model parameters in real-time
5. See how the model's behavior changes as you adjust parameters

## What is nGPT?

nGPT (normalized GPT) is a variant of the GPT architecture introduced in the paper ["nGPT: Normalized Transformer with Representation Learning on the Hypersphere"](https://arxiv.org/abs/2410.01131) by Loshchilov et al. The key innovation is that it normalizes all representations to lie on a hypersphere (a high-dimensional sphere).

Key features of nGPT:
- L2 Normalization: All vectors in the model are normalized to have unit length
- Cosine Similarity: Instead of dot products, the model uses cosine similarity for attention
- Residual SLERP Update: Uses spherical linear interpolation for residual connections
- Normalized Linear Layers: All linear transformations maintain vectors on the hypersphere

## Installation

1. Make sure you have Python 3.8+ installed
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Connect your Xbox controller to your computer
4. Run the explorer:

```bash
python ngpt_explorer.py
```

## Controls

### Xbox Controller

- **Left Stick**: Navigate through the hypersphere (2D projection)
- **Right Stick**: Adjust model parameters (temperature, rotation)
- **A Button**: Generate text from current position
- **B Button**: Reset view
- **X Button**: Toggle visualization mode
- **Y Button**: Save current state as screenshot
- **Left Bumper**: Decrease model complexity
- **Right Bumper**: Increase model complexity
- **Start**: Exit the application

### Keyboard (Fallback)

- **WASD**: Navigate through the hypersphere
- **Arrow Keys**: Adjust parameters
- **Space**: Generate text
- **R**: Reset view
- **Tab**: Toggle visualization mode
- **Ctrl+S**: Save screenshot
- **-/+**: Decrease/Increase model complexity
- **Esc**: Exit

## Visualization Modes

1. **Attention Visualization**: Shows how the attention mechanism focuses on different parts of the input sequence. The left stick controls the attention pattern.

2. **Token Embeddings**: Visualizes token embeddings on the hypersphere. The left stick controls rotation and zoom of the visualization.

3. **Layer Activations**: Shows activations across different layers of the model. The left stick influences the activation patterns.

## Learning Guide

1. **Start with Attention Visualization**: Use the left stick to see how attention patterns change. Notice how the model focuses on different parts of the sequence.

2. **Explore Token Embeddings**: Switch to the embedding visualization (X button) and see how tokens are distributed on the hypersphere. Use the left stick to rotate and zoom.

3. **Experiment with Text Generation**: Press A to generate text from your current position. Notice how the text changes as you move around the hypersphere.

4. **Adjust Model Complexity**: Use the bumpers to increase or decrease model depth. Notice how this affects the visualizations and generated text.

5. **Play with Temperature**: Use the right stick to adjust the temperature parameter. Higher values make the text more random, while lower values make it more deterministic.

## Understanding the Hypersphere

The key innovation in nGPT is learning on the hypersphere. In the embedding visualization:

- The red circle represents the unit hypersphere (in 2D projection)
- Each point represents a token embedding
- The position on the hypersphere is more important than the magnitude
- Cosine similarity (angle between vectors) determines token relationships

As you navigate with the left stick, you're essentially moving through this embedding space, influencing how the model processes and generates text.

## Troubleshooting

- **Controller not detected**: Make sure your Xbox controller is properly connected and recognized by your system
- **Visualization errors**: Try reducing model complexity (Left Bumper) if visualizations are slow
- **CUDA errors**: If you have GPU issues, the explorer will automatically fall back to CPU

## Credits

This explorer is built on the nGPT-pytorch implementation by lucidrains, based on the paper by Loshchilov et al.

```bibtex
@inproceedings{Loshchilov2024nGPTNT,
    title   = {nGPT: Normalized Transformer with Representation Learning on the Hypersphere},
    author  = {Ilya Loshchilov and Cheng-Ping Hsieh and Simeng Sun and Boris Ginsburg},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:273026160}
}
``` 