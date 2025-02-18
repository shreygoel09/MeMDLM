import hydra
import torch
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from matplotlib.font_manager import FontProperties
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM

from MeMDLM.src.diffusion.diffusion import Diffusion
from MeMDLM.src.guidance.guidance import SolubilityGuider

cfg_pth = '/workspace/sg666/MeMDLM/MeMDLM/configs'


font_path = "/workspace/sg666/MeMDLM/MeMDLM/src/utils/ubuntu_font/"

regular_font_path = os.path.join(font_path + 'Ubuntu-Regular.ttf')
bold_font_path = os.path.join(font_path + 'Ubuntu-Bold.ttf')
italic_font_path = os.path.join(font_path + 'Ubuntu-Italic.ttf')
bold_italic_font_path = os.path.join(font_path + 'Ubuntu-BoldItalic.ttf')

# Load the font properties
regular_font = FontProperties(fname=regular_font_path)
bold_font = FontProperties(fname=bold_font_path)
italic_font = FontProperties(fname=italic_font_path)
bold_italic_font = FontProperties(fname=bold_italic_font_path)

# Add the fonts to the font manager
fm.fontManager.addfont(regular_font_path)
fm.fontManager.addfont(bold_font_path)
fm.fontManager.addfont(italic_font_path)
fm.fontManager.addfont(bold_italic_font_path)

# Set the font family globally to Ubuntu
plt.rcParams['font.family'] = regular_font.get_name()
plt.rcParams['font.family'] = regular_font.get_name()
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = regular_font.get_name()
plt.rcParams['mathtext.it'] = italic_font.get_name()
plt.rcParams['mathtext.bf'] = bold_font.get_name()



def get_soluble_positions(sequence):
    return torch.tensor([1 if char.islower() else 0 for char in sequence])

def embed_sequence(ids, masks, model: Diffusion):
    outputs = model(input_ids=ids, attention_mask=masks)
    embeds = outputs.last_hidden_state.squeeze(0)
    return embeds


def plot_saliency(saliency, true_solubles, seq, idx):
    saliency = saliency.cpu().detach().numpy()[1:-1] if hasattr(saliency, 'cpu') else np.array(saliency)[1:-1]
    true_solubles = true_solubles.cpu().detach().numpy() if hasattr(true_solubles, 'cpu') else np.array(true_solubles)

    # Print shapes for debugging
    print(f"Sequence length: {len(seq)}")
    print(f"Saliency shape: {saliency.shape}")
    print(f"True solubles shape: {true_solubles.shape}")
    
    # Create figure and axis
    plt.figure(figsize=(15, 4))
    
    # Plot saliency scores as bars
    positions = np.arange(len(seq))
    plt.bar(positions, saliency, color='gray', alpha=0.5, width=0.8)
    
    # Set up the x-axis with residue labels
    plt.xticks(positions, list(seq), rotation=0, fontsize=8, fontproperties=regular_font)
    
    # Color the residue labels based on true_solubles
    ax = plt.gca()
    for idx_pos, label in enumerate(ax.get_xticklabels()):
        color = '#E17153' if true_solubles[idx_pos] == 1 else '#32B8A8'
        label.set_color(color)
    
    # Customize the plot
    plt.ylabel('Raw Saliency Score')
    plt.grid(False)
    
    # Set y-axis limits
    plt.ylim(0, np.max(saliency)+10)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()

    plt.show()
    
    # Create directory for saving results
    curr_plot = f"/workspace/sg666/MeMDLM/MeMDLM/benchmarks/scripts/saliency/bar_plots/saliency_map{idx}"
    os.makedirs(curr_plot, exist_ok=True)
    
    # Save sequence to txt file
    with open(os.path.join(curr_plot, f"sequence{idx}.txt"), 'w') as f:
        f.write(seq)
    
    # Save plot to png file
    plt.savefig(os.path.join(curr_plot, f"map{idx}.png"), dpi=300, bbox_inches='tight')
    plt.close()


@hydra.main(version_base=None, config_path=cfg_pth, config_name='config')
def main(cfg):
    df = pd.read_csv(cfg.data.test.membrane_esm_test_path)
    sequences = df['Sequence'].tolist()
    sequences = [seq for seq in sequences if len(seq) <= 100]

    device = torch.device(f"cuda:{6}" if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
    mdlm_lm = AutoModelForMaskedLM.from_pretrained(cfg.value.training.pretrained_model).eval().to(device)
    mdlm = AutoModel.from_pretrained(cfg.value.training.pretrained_model).eval().to(device)

    guidance = SolubilityGuider(cfg, device, mdlm)

    for idx, seq in enumerate(sequences):
        tokens = tokenizer(seq.upper(), return_tensors='pt').to(device) # Tokenize sequences independently to avoid truncation
        ids, masks = tokens['input_ids'], tokens['attention_mask']
        
        embeddings = embed_sequence(ids, masks, model=mdlm)
        logits = mdlm_lm(**tokens).logits
        
        # Conduct only 1 round of optimization (N optimization steps)
        saliency = guidance.optimized_sampling(logits, embeddings, masks, n_steps=cfg.value.guidance.n_steps)
        true_solubles = get_soluble_positions(seq).to(device)

        print(saliency.shape)
        
        # Plot and save the saliency map
        plot_saliency(saliency, true_solubles, seq.upper(), idx)


if __name__ == "__main__":
    main()