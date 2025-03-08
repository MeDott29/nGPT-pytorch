# nGPT Learning Suite

A comprehensive learning suite for understanding and exploring the nGPT (normalized GPT) architecture through interactive visualizations and experiments.

## Components

This learning suite includes several tools to help you understand nGPT:

1. **nGPT Explorer**: An interactive visualization tool that lets you explore nGPT using an Xbox controller
2. **Hypersphere Visualizer**: A 3D visualization of token embeddings on the hypersphere
3. **Controller Test**: A utility to test your Xbox controller connectivity
4. **Configuration Tool**: A script to run the explorer with different model configurations
5. **Tutorial**: A step-by-step guide to understanding nGPT

## Getting Started

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Test your Xbox controller:
   ```bash
   python test_controller.py
   ```

3. Run the nGPT Explorer:
   ```bash
   python ngpt_explorer.py
   ```
   
   Or use the batch file (Windows):
   ```bash
   run_explorer.bat
   ```
   
   Or the shell script (Linux/Mac):
   ```bash
   ./run_explorer.sh
   ```

## Learning Path

For the best learning experience, follow this path:

1. Read the `TUTORIAL.md` file to understand the basics of nGPT
2. Run the 3D hypersphere visualization to see how tokens are distributed:
   ```bash
   python visualize_hypersphere.py
   ```
3. Run the attention visualization to understand how cosine similarity works:
   ```bash
   python visualize_hypersphere.py --attention
   ```
4. Launch the nGPT Explorer and follow the tutorials in the `TUTORIAL.md` file
5. Experiment with different model configurations:
   ```bash
   python explore_configs.py --preset small
   python explore_configs.py --preset medium
   python explore_configs.py --preset large
   ```

## Xbox Controller Integration

The nGPT Explorer is designed to work with an Xbox controller, allowing you to:

- Navigate through the hypersphere using the left analog stick
- Adjust model parameters with the right analog stick
- Generate text, toggle visualization modes, and more with the buttons

If you don't have an Xbox controller, you can use keyboard controls instead.

## Understanding the Hypersphere

The key innovation in nGPT is learning on the hypersphere. This learning suite helps you visualize and understand:

1. How token embeddings are distributed on the hypersphere
2. How cosine similarity affects attention
3. How the model's behavior changes as you navigate through the hypersphere
4. How different model parameters affect the representations

## Files in this Suite

- `ngpt_explorer.py`: The main interactive visualization tool
- `test_controller.py`: Utility to test Xbox controller connectivity
- `visualize_hypersphere.py`: 3D visualization of token embeddings
- `explore_configs.py`: Tool to run the explorer with different configurations
- `TUTORIAL.md`: Step-by-step tutorial for understanding nGPT
- `ngpt_explorer_README.md`: Detailed documentation for the explorer
- `run_explorer.bat`: Batch file for Windows users
- `run_explorer.sh`: Shell script for Linux/Mac users

## Requirements

- Python 3.8+
- PyTorch
- Pygame (for the interactive explorer)
- Matplotlib (for visualizations)
- Scikit-learn (for dimensionality reduction)
- Xbox controller (optional)

## Credits

This learning suite is built on the nGPT-pytorch implementation by lucidrains, based on the paper by Loshchilov et al.

```bibtex
@inproceedings{Loshchilov2024nGPTNT,
    title   = {nGPT: Normalized Transformer with Representation Learning on the Hypersphere},
    author  = {Ilya Loshchilov and Cheng-Ping Hsieh and Simeng Sun and Boris Ginsburg},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:273026160}
}
``` 