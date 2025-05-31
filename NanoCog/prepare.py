#!/usr/bin/env python3
"""
NanoCog Data Preparation Script

This script prepares a comprehensive training corpus for NanoCog by:
1. Downloading the CogPrime architecture paper from GitHub
2. Collecting all documentation from opencog-central
3. Including all Scheme files from opencog-central
4. Optionally adding additional OpenCog resources
5. Tokenizing and saving as train.bin and val.bin

The resulting dataset enables training a nanoGPT model that understands
both CogPrime theory and OpenCog implementation details.
"""

import os
import sys
import glob
import json
import time
import requests
import tiktoken
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime

# --- Configuration ---
COGPRIME_PAPER_URL = "https://raw.githubusercontent.com/drzo/cogprime/main/CogPrime%20-%20An%20Integrative%20Architecture%20for%20Embodied%20Artificial%20General%20Intelligence.md"
DOCUMENT_SEPARATOR = "\n\n<|endofdocument|>\n\n"
SECTION_SEPARATOR = "\n\n---\n\n"
FILE_TYPE_MARKERS = {
    ".md": "\n\n<!-- Markdown Document: {filename} -->\n\n",
    ".scm": "\n\n;; Scheme File: {filename}\n;; Path: {filepath}\n\n",
    ".py": "\n\n# Python File: {filename}\n# Path: {filepath}\n\n",
    ".txt": "\n\n# Text File: {filename}\n\n",
}
DEFAULT_MARKER = "\n\n# File: {filename}\n# Path: {filepath}\n\n"
TRAIN_RATIO = 0.9  # 90% train, 10% validation

# --- Utility Functions ---
def download_file(url, output_path):
    """Downloads a file from a URL and saves it locally."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✓ Downloaded {url} to {output_path}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"✗ Error downloading {url}: {e}")
        return False

def read_file_content(file_path):
    """Reads and returns the content of a local file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # Try alternative encodings
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
                print(f"⚠ File {file_path} read with latin-1 encoding")
                return content
        except Exception as e:
            print(f"✗ Error reading file {file_path} with latin-1: {e}")
            return ""
    except FileNotFoundError:
        print(f"⚠ File not found {file_path}")
        return ""
    except Exception as e:
        print(f"✗ Error reading file {file_path}: {e}")
        return ""

def find_repository_path(repo_name, possible_locations=None):
    """Find the path to a repository by checking multiple possible locations."""
    if possible_locations is None:
        # Default locations to check relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        possible_locations = [
            # Same level as NanoCog directory
            os.path.abspath(os.path.join(script_dir, "..", "..", repo_name)),
            # One level up from NanoCog
            os.path.abspath(os.path.join(script_dir, "..", repo_name)),
            # Two levels up from NanoCog
            os.path.abspath(os.path.join(script_dir, "..", "..", "..", repo_name)),
            # Same level as the script
            os.path.abspath(os.path.join(script_dir, repo_name)),
        ]
    
    for location in possible_locations:
        if os.path.exists(location) and os.path.isdir(location):
            print(f"✓ Found repository '{repo_name}' at: {location}")
            return location
    
    print(f"⚠ Could not find repository '{repo_name}' in any of the checked locations")
    return None

def get_file_marker(file_path):
    """Returns the appropriate file marker based on file extension."""
    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()
    marker_template = FILE_TYPE_MARKERS.get(ext, DEFAULT_MARKER)
    return marker_template.format(filename=filename, filepath=file_path)

def get_token_stats(token_ids, enc):
    """Generate statistics about the tokenized corpus."""
    # Count token frequencies
    token_counter = Counter(token_ids)
    
    # Get most common tokens and their counts
    most_common = token_counter.most_common(20)
    most_common_tokens = [(enc.decode([token_id]), count) for token_id, count in most_common]
    
    # Get vocabulary coverage
    unique_tokens = len(token_counter)
    vocab_coverage = unique_tokens / enc.n_vocab * 100
    
    # Get token length distribution
    token_lengths = [len(enc.decode([token_id])) for token_id in set(token_ids)]
    avg_token_length = sum(token_lengths) / len(token_lengths) if token_lengths else 0
    
    return {
        "total_tokens": len(token_ids),
        "unique_tokens": unique_tokens,
        "vocab_coverage_percent": vocab_coverage,
        "avg_token_length": avg_token_length,
        "most_common_tokens": most_common_tokens
    }

def get_corpus_stats(all_text_content):
    """Generate statistics about the text corpus."""
    file_types = defaultdict(int)
    file_sizes = []
    total_chars = 0
    total_words = 0
    
    for content_item in all_text_content:
        if isinstance(content_item, tuple) and len(content_item) == 2:
            file_path, content = content_item
            ext = os.path.splitext(file_path)[1].lower()
            file_types[ext] += 1
            file_size = len(content)
            file_sizes.append((file_path, file_size))
            total_chars += file_size
            total_words += len(content.split())
    
    # Sort file sizes by size (largest first)
    file_sizes.sort(key=lambda x: x[1], reverse=True)
    
    return {
        "total_files": len(file_sizes),
        "total_chars": total_chars,
        "total_words": total_words,
        "file_types": dict(file_types),
        "largest_files": file_sizes[:10],  # Top 10 largest files
        "avg_file_size": total_chars / len(file_sizes) if file_sizes else 0
    }

def print_stats(corpus_stats, token_stats):
    """Print formatted statistics about the corpus and tokenization."""
    print("\n" + "="*80)
    print(" "*30 + "NANOCOG CORPUS STATISTICS")
    print("="*80)
    
    # Corpus statistics
    print(f"\n📊 CORPUS OVERVIEW:")
    print(f"  • Total files: {corpus_stats['total_files']}")
    print(f"  • Total characters: {corpus_stats['total_chars']:,}")
    print(f"  • Total words: {corpus_stats['total_words']:,}")
    print(f"  • Average file size: {corpus_stats['avg_file_size']:.1f} characters")
    
    print(f"\n📂 FILE TYPES:")
    for ext, count in corpus_stats['file_types'].items():
        print(f"  • {ext or 'no extension'}: {count} files")
    
    print(f"\n📄 LARGEST FILES:")
    for i, (file_path, size) in enumerate(corpus_stats['largest_files'][:5], 1):
        print(f"  {i}. {os.path.basename(file_path)}: {size:,} characters")
    
    # Token statistics
    print(f"\n🔤 TOKENIZATION:")
    print(f"  • Total tokens: {token_stats['total_tokens']:,}")
    print(f"  • Unique tokens: {token_stats['unique_tokens']:,}")
    print(f"  • Vocabulary coverage: {token_stats['vocab_coverage_percent']:.2f}%")
    print(f"  • Average token length: {token_stats['avg_token_length']:.2f} characters")
    
    print(f"\n📊 TRAIN/VAL SPLIT:")
    train_tokens = int(token_stats['total_tokens'] * TRAIN_RATIO)
    val_tokens = token_stats['total_tokens'] - train_tokens
    print(f"  • Training set: {train_tokens:,} tokens ({TRAIN_RATIO*100:.0f}%)")
    print(f"  • Validation set: {val_tokens:,} tokens ({(1-TRAIN_RATIO)*100:.0f}%)")
    
    print("\n" + "="*80)

def collect_files(directory, file_pattern, description):
    """Collect files matching a pattern from a directory."""
    files = glob.glob(os.path.join(directory, file_pattern), recursive=True)
    if files:
        print(f"✓ Found {len(files)} {description} files")
    else:
        print(f"⚠ No {description} files found matching pattern: {os.path.join(directory, file_pattern)}")
    return files

def process_file(file_path, all_text_content):
    """Process a single file and add it to the content list."""
    print(f"  Processing: {file_path}")
    content = read_file_content(file_path)
    if content:
        marker = get_file_marker(file_path)
        all_text_content.append((file_path, marker + content))
        return True
    return False

def main():
    """Main function to prepare the NanoCog training corpus."""
    start_time = time.time()
    
    # Define the output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    nanocog_dir = script_dir
    output_dir = os.path.join(nanocog_dir, "data")
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n🔍 NanoCog Data Preparation")
    print("="*50)
    
    # Find repositories
    opencog_central_path = find_repository_path("opencog-central")
    cogprime_path = find_repository_path("cogprime")
    
    # --- Collect Data ---
    all_text_content = []  # List of (file_path, content) tuples
    
    # 1. CogPrime Main Paper
    cogprime_paper_local_path = os.path.join(output_dir, "cogprime_paper.md")
    print("\n📄 Processing CogPrime Main Paper")
    if download_file(COGPRIME_PAPER_URL, cogprime_paper_local_path):
        paper_content = read_file_content(cogprime_paper_local_path)
        if paper_content:
            marker = get_file_marker(cogprime_paper_local_path)
            all_text_content.append((cogprime_paper_local_path, marker + paper_content))
    
    # 2. opencog-central Documentation
    if opencog_central_path:
        print("\n📚 Processing opencog-central Documentation")
        
        # Main docs
        opencog_docs_files = [
            "README.md",
            "docs/CogPrime_Integrative_Architecture_AGI.md",
            "docs/IMPLEMENTATION_GUIDE.md",
            "docs/COGPRIME_STATUS_2024.md",
            "docs/COGPRIME_ARCHITECTURE_DIAGRAM.md",
            "examples/SIMPLE_COGPRIME_AGENT.md",
            "profile/README.md"
        ]
        
        for doc_file in opencog_docs_files:
            file_path = os.path.join(opencog_central_path, doc_file)
            process_file(file_path, all_text_content)
        
        # 3. opencog-central Scheme Files
        print("\n💻 Processing opencog-central Scheme Files")
        scheme_files_path_pattern = os.path.join(opencog_central_path, "Scheme", "**", "*.scm")
        scheme_files = collect_files(opencog_central_path, "Scheme/**/*.scm", "Scheme")
        
        for scm_file in scheme_files:
            process_file(scm_file, all_text_content)
    
    # 4. Additional CogPrime resources if available
    if cogprime_path:
        print("\n📘 Processing Additional CogPrime Resources")
        # Main CogPrime docs
        cogprime_docs = collect_files(cogprime_path, "*.md", "CogPrime markdown")
        for doc_file in cogprime_docs:
            process_file(doc_file, all_text_content)
        
        # 50 Episodes in Relevance Realization
        episodes_dir = os.path.join(cogprime_path, "50 Episodes in Relevance Realization")
        if os.path.exists(episodes_dir):
            episode_files = collect_files(episodes_dir, "*.md", "Relevance Realization episodes")
            for episode_file in episode_files:
                process_file(episode_file, all_text_content)
        
        # Source code if available
        src_files = collect_files(cogprime_path, "src/**/*.py", "Python source")
        for src_file in src_files:
            process_file(src_file, all_text_content)
    
    if not all_text_content:
        print("\n❌ No content collected. Exiting. Please check data source paths and availability.")
        sys.exit(1)
    
    # Calculate corpus statistics
    print("\n📊 Calculating corpus statistics...")
    corpus_stats = get_corpus_stats(all_text_content)
    
    # Concatenate all collected text with separators
    print("\n🔄 Concatenating text data...")
    full_text_data = ""
    for _, content in all_text_content:
        full_text_data += content + DOCUMENT_SEPARATOR
    
    # --- Tokenization ---
    print("\n🔤 Tokenizing data with GPT-2 tokenizer...")
    enc = tiktoken.get_encoding("gpt2")
    token_ids = enc.encode_ordinary(full_text_data)
    token_stats = get_token_stats(token_ids, enc)
    
    # Print statistics
    print_stats(corpus_stats, token_stats)
    
    # --- Data Splitting ---
    print("\n📂 Splitting data into training and validation sets...")
    n_tokens = len(token_ids)
    split_idx = int(n_tokens * TRAIN_RATIO)
    
    train_data_ids = token_ids[:split_idx]
    val_data_ids = token_ids[split_idx:]
    
    # --- Saving to .bin files ---
    train_ids_np = np.array(train_data_ids, dtype=np.uint16)
    val_ids_np = np.array(val_data_ids, dtype=np.uint16)
    
    train_output_path = os.path.join(output_dir, 'train.bin')
    val_output_path = os.path.join(output_dir, 'val.bin')
    
    print(f"\n💾 Saving training data to {train_output_path}...")
    train_ids_np.tofile(train_output_path)
    
    print(f"💾 Saving validation data to {val_output_path}...")
    val_ids_np.tofile(val_output_path)
    
    # Save metadata for reference
    metadata = {
        "date_created": datetime.now().isoformat(),
        "corpus_stats": corpus_stats,
        "token_stats": {k: v for k, v in token_stats.items() if k != 'most_common_tokens'},
        "train_tokens": len(train_data_ids),
        "val_tokens": len(val_data_ids),
        "tokenizer": "gpt2",
        "train_ratio": TRAIN_RATIO,
        "sources": {
            "cogprime_paper": bool(cogprime_paper_local_path and os.path.exists(cogprime_paper_local_path)),
            "opencog_central": bool(opencog_central_path),
            "cogprime_repo": bool(cogprime_path),
        }
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    # Clean up downloaded paper if it exists
    if os.path.exists(cogprime_paper_local_path):
        try:
            os.remove(cogprime_paper_local_path)
            print(f"\n🧹 Cleaned up temporary file: {cogprime_paper_local_path}")
        except OSError as e:
            print(f"\n⚠ Error deleting temporary file {cogprime_paper_local_path}: {e}")
    
    elapsed_time = time.time() - start_time
    print(f"\n✅ Data preparation complete in {elapsed_time:.2f} seconds!")
    print(f"📦 Output files:")
    print(f"   • {train_output_path} ({os.path.getsize(train_output_path)/1024/1024:.2f} MB)")
    print(f"   • {val_output_path} ({os.path.getsize(val_output_path)/1024/1024:.2f} MB)")
    print(f"   • {metadata_path}")
    print("\nYou can now train NanoCog using:")
    print("   python train.py config/train_cogprime.py --out_dir=out-nanocog")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
