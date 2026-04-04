"""
Example: Batch inference on multiple blood smear images.

This example shows how to process multiple images efficiently.
"""

from pathlib import Path
from src.pipeline import BloodSmearPipeline

def main():
    """Run batch analysis."""
    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = BloodSmearPipeline()
    
    # Collect image paths
    image_directory = Path("path/to/your/images")
    image_paths = list(image_directory.glob("*.jpg"))
    
    if not image_paths:
        print(f"No images found in {image_directory}")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # Run batch analysis
    print("\nProcessing batch...")
    results = pipeline.analyze(image_paths, save_results=True)
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("BATCH ANALYSIS SUMMARY")
    print("=" * 60)
    
    if 'stage1_detection' in results:
        detection = results['stage1_detection']
        
        print(f"\nImages processed: {detection['image_count']}")
        print(f"Images skipped: {detection['failed_count']}")
        
        print(f"\nTotal Cell Counts:")
        for cell_type, count in detection['total_counts'].items():
            print(f"  {cell_type}: {count}")
        
        print(f"\nMean Cell Counts per Image:")
        for cell_type, mean in detection['cell_count_stats']['mean'].items():
            variance = detection['cell_count_stats']['variance'][cell_type]
            print(f"  {cell_type}: {mean:.1f} ± {variance:.1f}")
    
    if 'stage2_classification' in results:
        summary = results['stage2_classification']['summary']
        
        print(f"\nWBC Classification:")
        print(f"  Total WBCs classified: {summary['sample_count']}")
        print(f"  Cells flagged: {summary['flagged_count']}")
        print(f"  Expert review needed: {'Yes' if summary['requires_expert_review'] else 'No'}")
        
        if 'class_distribution' in summary:
            print(f"\n  WBC Differential:")
            total = sum(summary['class_distribution'].values())
            for cls, count in summary['class_distribution'].items():
                pct = 100.0 * count / total if total > 0 else 0
                print(f"    {cls}: {count} ({pct:.1f}%)")
    
    print(f"\n✓ Results saved to: results/")
    print(f"✓ Total execution time: {results['metadata']['execution_time_seconds']:.2f}s")


if __name__ == '__main__':
    main()
