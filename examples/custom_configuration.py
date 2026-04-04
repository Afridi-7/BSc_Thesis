"""
Example: Custom configuration for specialized use cases.

This example shows how to use custom configuration settings.
"""

from src.config.config_loader import load_config, Config
from src.pipeline import BloodSmearPipeline

def main():
    """Run analysis with custom configuration."""
    # Load configuration
    print("Loading custom configuration...")
    config = load_config('config.yaml')
    
    # Override specific parameters programmatically
    print("Adjusting parameters...")
    
    # You can modify config parameters by creating a new config dict
    custom_config_dict = config.to_dict()
    
    # Example: Use higher confidence threshold for detection
    custom_config_dict['detection']['confidence_threshold'] = 0.35
    
    # Example: Use more MC Dropout passes for better uncertainty
    custom_config_dict['classification']['mc_dropout_passes'] = 30
    
    # Example: Retrieve more context chunks
    custom_config_dict['rag']['retrieval']['top_k'] = 7
    
    # Create new config object
    custom_config = Config(custom_config_dict)
    
    # Initialize pipeline with custom config
    print("Initializing pipeline with custom settings...")
    pipeline = BloodSmearPipeline(config=custom_config)
    
    # Run analysis
    image_path = "path/to/your/blood_smear.jpg"
    print(f"\nAnalyzing: {image_path}")
    
    results = pipeline.analyze(image_path, save_results=True)
    
    # Display key results
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    if 'stage3_reasoning' in results:
        reasoning = results['stage3_reasoning']
        print(f"\nClinical Interpretation:")
        print(f"  {reasoning['clinical_interpretation']}")
        
        if reasoning.get('key_findings'):
            print(f"\nKey Findings:")
            for finding in reasoning['key_findings']:
                print(f"  • {finding}")
    
    print(f"\n✓ Analysis complete")


if __name__ == '__main__':
    main()
