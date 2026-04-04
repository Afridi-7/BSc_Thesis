"""
Example: Single image inference with Blood Smear Domain Expert.

This example shows how to analyze a single blood smear image.
"""

from src.pipeline import BloodSmearPipeline

def main():
    """Run single image analysis."""
    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = BloodSmearPipeline()
    
    # Path to blood smear image
    image_path = "path/to/your/blood_smear.jpg"
    
    # Run analysis
    print(f"Analyzing: {image_path}")
    results = pipeline.analyze(image_path, save_results=True)
    
    # Access results
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    # Detection results
    if 'stage1_detection' in results:
        counts = results['stage1_detection']['total_counts']
        print(f"\nCell Counts:")
        for cell_type, count in counts.items():
            print(f"  {cell_type}: {count}")
    
    # Classification results
    if 'stage2_classification' in results:
        summary = results['stage2_classification']['summary']
        print(f"\nWBC Classification:")
        print(f"  Total cells: {summary['sample_count']}")
        print(f"  Flagged for review: {summary['flagged_count']}")
    
    # Clinical reasoning
    if 'stage3_reasoning' in results:
        reasoning = results['stage3_reasoning']
        print(f"\nClinical Interpretation:")
        print(f"  {reasoning['clinical_interpretation']}")
        
        if reasoning.get('requires_expert_review'):
            print("\n⚠️  EXPERT REVIEW REQUIRED")
        
        if reasoning.get('safety_flags'):
            print(f"\nSafety Flags: {', '.join(reasoning['safety_flags'])}")
    
    print(f"\n✓ Results saved to: results/")
    print(f"✓ Execution time: {results['metadata']['execution_time_seconds']:.2f}s")


if __name__ == '__main__':
    main()
