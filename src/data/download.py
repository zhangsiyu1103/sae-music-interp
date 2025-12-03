from datasets import load_dataset
from pathlib import Path

def download_musiccaps():
    """Download and prepare MusicCaps prompts using HuggingFace datasets."""

    Path("data").mkdir(exist_ok=True)

    print("Downloading MusicCaps from HuggingFace...")
    dataset = load_dataset('google/MusicCaps')

    print(f"Downloaded {len(dataset['train'])} descriptions")

    # Extract captions
    prompts = dataset['train']['caption']
    
    # Save different subsets
    subsets = {
        'quick_test': prompts[:200],      # Quick test
        'medium': prompts[:1000],          # Medium scale
        'full': prompts[:5000],            # Full scale
    }
    
    for name, subset in subsets.items():
        output_file = f"data/musiccaps_prompts_{name}.txt"
        with open(output_file, 'w') as f:
            f.write('\n'.join(subset))
        print(f"Saved {len(subset)} prompts to {output_file}")
    
    # Print examples
    print("\nExample prompts:")
    for i, prompt in enumerate(prompts[:5]):
        print(f"{i+1}. {prompt}")
    
    return prompts

if __name__ == "__main__":
    download_musiccaps()