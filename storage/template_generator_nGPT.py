import os
import sys
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, List, Dict
from torch.utils.data import Dataset, DataLoader

# Add current directory to path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Local imports from provided files
from nGPT_pytorch.nGPT import nGPT
from token_linear import TokenLinearGL, TokenLinearSM

@dataclass
class TemplateConfig:
    """Configuration for template generation"""
    num_tokens: int = 32000  # Vocabulary size
    dim: int = 512  # Model dimension
    depth: int = 8  # Number of layers
    heads: int = 8  # Number of attention heads
    dim_head: int = 64  # Attention head dimension
    template_max_length: int = 1024
    narrative_levels: int = 3  # Levels of nested narrative structure
    batch_size: int = 32
    learning_rate: float = 1e-4
    ff_expand_factor: float = 4.
    ce_ignore_index: int = -1
    tied_embedding: bool = False
    num_hyperspheres: int = 1
    causal: bool = True
    attn_flash_kwargs: dict = None

    def __post_init__(self):
        if self.attn_flash_kwargs is None:
            self.attn_flash_kwargs = dict(
                enable_flash=True,
                enable_math=True,
                enable_mem_efficient=True
            )

class NarrativeStructure:
    """Handles hierarchical narrative elements"""
    def __init__(self, level: int, parent: Optional['NarrativeStructure'] = None):
        self.level = level
        self.parent = parent
        self.children = []
        self.content = ""
        self.template_slots = {}
    
    def add_child(self, child: 'NarrativeStructure'):
        self.children.append(child)
        
    def set_content(self, content: str):
        self.content = content
        
    def add_template_slot(self, key: str, value: str):
        self.template_slots[key] = value

class TemplateDataset(Dataset):
    """Dataset for template generation"""
    def __init__(self, texts: List[str], config: TemplateConfig):
        self.texts = texts
        self.config = config
        self.narrative_structures = []
        
        # Parse texts into narrative structures
        for text in texts:
            structure = self._parse_narrative_structure(text)
            self.narrative_structures.append(structure)
            
        # Convert structures to tensors
        self.input_ids = []
        self.narrative_levels = []
        self.attention_masks = []
        
        for structure in self.narrative_structures:
            tensors = self._structure_to_tensors(structure)
            self.input_ids.append(tensors['input_ids'])
            self.narrative_levels.append(tensors['narrative_levels'])
            self.attention_masks.append(tensors['attention_mask'])
    
    def _parse_narrative_structure(self, text: str) -> NarrativeStructure:
        """Parse text into nested narrative structure"""
        root = NarrativeStructure(level=0)
        current = root
        
        lines = text.split('\n')
        for line in lines:
            # Determine indentation level
            level = len(line) - len(line.lstrip())
            content = line.strip()
            
            if level > current.level:
                new_node = NarrativeStructure(level=level, parent=current)
                current.add_child(new_node)
                current = new_node
            elif level < current.level:
                while current.level > level:
                    current = current.parent
            
            current.set_content(content)
            
            # Extract template slots
            import re
            slots = re.findall(r'\[\[(.*?)\]\]', content)
            for slot in slots:
                current.add_template_slot(slot, "")
                
        return root
        
    def _structure_to_tensors(self, structure: NarrativeStructure) -> Dict[str, torch.Tensor]:
        """Convert narrative structure to tensor format"""
        # This is a simplified tokenization - you would want to use a proper tokenizer
        content = structure.content.split()
        input_ids = torch.tensor([hash(word) % self.config.num_tokens for word in content])
        narrative_levels = torch.full_like(input_ids, structure.level)
        attention_mask = torch.ones_like(input_ids)
        
        return {
            'input_ids': input_ids,
            'narrative_levels': narrative_levels,
            'attention_mask': attention_mask
        }

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'narrative_levels': self.narrative_levels[idx],
            'attention_mask': self.attention_masks[idx]
        }

class TemplateEmulator(nn.Module):
    """Template generation model based on nGPT"""
    def __init__(self, config: TemplateConfig):
        super().__init__()
        
        # Initialize nGPT model
        self.model = nGPT(
            num_tokens=config.num_tokens,
            dim=config.dim,
            depth=config.depth,
            dim_head=config.dim_head,
            heads=config.heads,
            ff_expand_factor=config.ff_expand_factor,
            causal=config.causal,
            ce_ignore_index=config.ce_ignore_index,
            tied_embedding=config.tied_embedding,
            num_hyperspheres=config.num_hyperspheres,
            attn_flash_kwargs=config.attn_flash_kwargs
        )
        
        # Additional layers for template-specific tasks
        self.token_linear = TokenLinearGL(config.dim, config.dim, config.narrative_levels)
        
    def forward(self, 
                input_ids: torch.Tensor,
                narrative_levels: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                return_loss: bool = True):
        
        # Process through token linear layer
        token_embeds = self.token_linear(input_ids.float())
        
        # Process through nGPT
        return self.model(token_embeds, mask=attention_mask, return_loss=return_loss)

class TemplateGenerator:
    """Main class for generating narrative-structured templates"""
    def __init__(self, config: TemplateConfig):
        self.config = config
        self.model = TemplateEmulator(config)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate
        )
    
    def train(self, dataset: TemplateDataset, num_epochs: int = 10):
        """Train the template generator"""
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        self.model.train()
        for epoch in range(num_epochs):
            for batch in dataloader:
                self.optimizer.zero_grad()
                
                # Process batch through model
                loss = self.model(
                    batch['input_ids'],
                    batch['narrative_levels'],
                    batch['attention_mask']
                )
                
                loss.backward()
                self.optimizer.step()
                
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    def generate_template(self, prompt: str) -> str:
        """Generate a template from a prompt"""
        self.model.eval()
        
        # Convert prompt to input format
        dataset = TemplateDataset([prompt], self.config)
        input_data = dataset[0]
        
        with torch.no_grad():
            outputs = self.model(
                input_data['input_ids'].unsqueeze(0),
                input_data['narrative_levels'].unsqueeze(0),
                return_loss=False
            )
            
        # Convert outputs to text
        generated_text = self._decode_outputs(outputs)
        return self._format_template(generated_text)
    
    def _decode_outputs(self, outputs: torch.Tensor) -> str:
        """Convert model outputs to text"""
        # Simple decoding - you would want to use a proper tokenizer/decoder
        return " ".join([str(idx.item()) for idx in outputs.argmax(dim=-1)[0]])
    
    def _format_template(self, text: str) -> str:
        """Format generated text into proper template structure"""
        # Add proper formatting
        lines = text.split('\n')
        formatted = []
        current_level = 0
        
        for line in lines:
            formatted.append("    " * current_level + line)
            if ":" in line:
                current_level += 1
                
        return "\n".join(formatted)

if __name__ == "__main__":
    # Initialize with configuration
    config = TemplateConfig()
    generator = TemplateGenerator(config)
    
    # Example template text
    template_text = """
    Project: [[project_name]]
    Organization: [[org_name]]
    Idea: [[idea_description]]
        Background:
            [[background_details]]
        Objectives:
            [[objectives_list]]
        Implementation:
            [[implementation_steps]]
    """
    
    # Create dataset and train
    dataset = TemplateDataset([template_text], config)
    generator.train(dataset)
    
    # Generate new template
    prompt = "Project: AI Research\nOrganization: Tech Lab"
    generated_template = generator.generate_template(prompt)
    print(generated_template)