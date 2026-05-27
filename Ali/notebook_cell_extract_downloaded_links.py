# Cell: Extract downloaded image and carousel links

import pandas as pd
from pathlib import Path

print("="*70)
print("EXTRACTING DOWNLOADED IMAGE AND CAROUSEL LINKS")
print("="*70)

# Define dataset directory
if 'dataset_dir' not in locals():
    dataset_dir = Path("./multimodal_dataset_fixed")
    print(f"Using dataset_dir: {dataset_dir}\n")
else:
    dataset_dir = Path(dataset_dir)
    print(f"Using existing dataset_dir: {dataset_dir}\n")

image_dir = dataset_dir / "image"
carousel_dir = dataset_dir / "carousel"

# Verify directories exist
if not image_dir.exists():
    print(f"⚠ Warning: Image directory not found: {image_dir}")
    image_shortcodes = set()
else:
    # Extract shortcodes from image directory
    image_shortcodes = {d.name for d in image_dir.iterdir() if d.is_dir()}
    print(f"✓ Found {len(image_shortcodes)} downloaded IMAGE items")

if not carousel_dir.exists():
    print(f"⚠ Warning: Carousel directory not found: {carousel_dir}")
    carousel_shortcodes = set()
else:
    # Extract shortcodes from carousel directory
    carousel_shortcodes = {d.name for d in carousel_dir.iterdir() if d.is_dir()}
    print(f"✓ Found {len(carousel_shortcodes)} downloaded CAROUSEL_ALBUM items")

total_downloaded = len(image_shortcodes) + len(carousel_shortcodes)
print(f"\n✓ Total downloaded items: {total_downloaded}")

# Match shortcodes back to ig_posts
if "ig_posts" not in globals():
    raise ValueError("ig_posts not found. Run the data loading cells first.")

print("\nMatching shortcodes to permalinks in ig_posts...")

# Extract shortcode from permalink
def extract_shortcode(permalink):
    """Extract Instagram shortcode from permalink"""
    return permalink.split('/')[-2] if '/' in permalink else permalink

# Create mapping of shortcode to permalink and media_type
ig_posts_dict = {
    extract_shortcode(row['permalink']): {
        'url': row['permalink'],
        'type': row['media_type']
    }
    for _, row in ig_posts.iterrows()
}

# Extract downloaded links
downloaded_data = []

# Add downloaded images
for shortcode in image_shortcodes:
    if shortcode in ig_posts_dict:
        downloaded_data.append({
            'shortcode': shortcode,
            'url': ig_posts_dict[shortcode]['url'],
            'type': 'IMAGE'
        })
    else:
        print(f"  ⚠ Shortcode not found in ig_posts: {shortcode}")

# Add downloaded carousels
for shortcode in carousel_shortcodes:
    if shortcode in ig_posts_dict:
        downloaded_data.append({
            'shortcode': shortcode,
            'url': ig_posts_dict[shortcode]['url'],
            'type': 'CAROUSEL_ALBUM'
        })
    else:
        print(f"  ⚠ Shortcode not found in ig_posts: {shortcode}")

# Create DataFrame
downloaded_df = pd.DataFrame(downloaded_data)

# Sort by type then shortcode
downloaded_df = downloaded_df.sort_values(['type', 'shortcode']).reset_index(drop=True)

print(f"\n✓ Successfully matched {len(downloaded_df)} items to permalinks")

# Summary statistics
image_count = len(downloaded_df[downloaded_df['type'] == 'IMAGE'])
carousel_count = len(downloaded_df[downloaded_df['type'] == 'CAROUSEL_ALBUM'])

print(f"\nDownloaded breakdown:")
print(f"  • IMAGE: {image_count}")
print(f"  • CAROUSEL_ALBUM: {carousel_count}")
print(f"  • Total: {len(downloaded_df)}")

# Save to CSV
output_file = "downloaded_links.csv"
downloaded_df.to_csv(output_file, index=False)
print(f"\n✓ Saved to local file: {output_file}")

# Also create a simpler version (URLs only) for easy reference
urls_only_df = downloaded_df[['url', 'type']].copy()
urls_output_file = "downloaded_links_urls_only.csv"
urls_only_df.to_csv(urls_output_file, index=False)
print(f"✓ Saved simplified version: {urls_output_file}")

# Display first few rows
print(f"\nFirst 5 downloaded items:")
print(downloaded_df.head())

print("\n" + "="*70)
print("EXTRACTION COMPLETE")
print("="*70)
print(f"\nFiles ready to upload to Colab:")
print(f"  1. {output_file} (with shortcodes)")
print(f"  2. {urls_output_file} (URLs only)")
