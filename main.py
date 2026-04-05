"""
Command-line interface for Blood Smear Domain Expert system.

Provides easy-to-use CLI for blood smear analysis.
"""

import argparse
import sys
from pathlib import Path
import logging
import os

from src.pipeline import BloodSmearPipeline
from src.config.config_loader import load_config, validate_config, get_model_path
from src.utils.logging_config import setup_logging


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Blood Smear Domain Expert - Clinical AI Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single image
  python main.py analyze smear.jpg

  # Analyze batch of images
  python main.py analyze images/*.jpg --batch

  # Use custom config
  python main.py analyze smear.jpg --config myconfig.yaml

  # Verbose logging
  python main.py analyze smear.jpg --verbose

For more information, visit: https://github.com/your-repo
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze blood smear image(s)')
    analyze_parser.add_argument(
        'images',
        nargs='+',
        help='Image file path(s) to analyze'
    )
    analyze_parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config.yaml (default: auto-search)'
    )
    analyze_parser.add_argument(
        '--batch',
        action='store_true',
        help='Process multiple images as batch'
    )
    analyze_parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to disk'
    )
    analyze_parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging (DEBUG level)'
    )
    
    # Test config command
    test_config_parser = subparsers.add_parser('test-config', help='Test configuration file')
    test_config_parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config.yaml (default: auto-search)'
    )
    
    # Version command
    subparsers.add_parser('version', help='Show version information')

    # Smoke test command
    smoke_test_parser = subparsers.add_parser('smoke-test', help='Run quick environment and config checks')
    smoke_test_parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config.yaml (default: auto-search)'
    )
    
    args = parser.parse_args()
    
    # Show help if no command
    if not args.command:
        parser.print_help()
        return 0
    
    # Handle version
    if args.command == 'version':
        print("Blood Smear Domain Expert v1.0.0")
        print("A clinically safer AI assistant for peripheral blood smear interpretation")
        return 0

    if args.command == 'smoke-test':
        return run_smoke_test(args.config)
    
    # Handle test-config
    if args.command == 'test-config':
        return test_config(args.config)
    
    # Handle analyze
    if args.command == 'analyze':
        return run_analysis(args)
    
    parser.print_help()
    return 1


def test_config(config_path: str = None) -> int:
    """Test configuration file."""
    try:
        print("Loading configuration...")
        config = load_config(config_path)
        print("✓ Configuration loaded successfully")
        
        print("\nValidating configuration...")
        validate_config(config)
        print("✓ Configuration validation passed")
        
        print("\nConfiguration summary:")
        print(f"  Detection confidence: {config.get('detection.confidence_threshold')}")
        print(f"  Classification MC passes: {config.get('classification.mc_dropout_passes')}")
        print(f"  LLM model: {config.get('llm.model_name')}")
        print(f"  RAG top-k: {config.get('rag.retrieval.top_k')}")
        
        print("\n✓ Configuration test passed!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Configuration test failed: {e}", file=sys.stderr)
        return 1


def run_analysis(args) -> int:
    """Run blood smear analysis."""
    try:
        # Setup logging
        log_level = 'DEBUG' if args.verbose else 'INFO'
        setup_logging(log_level=log_level)
        logger = logging.getLogger(__name__)
        
        # Load config
        logger.info("Loading configuration...")
        config = load_config(args.config)
        
        # Initialize pipeline
        logger.info("Initializing pipeline...")
        pipeline = BloodSmearPipeline(config=config)
        
        # Prepare input
        image_input = args.images[0] if len(args.images) == 1 and not args.batch else args.images
        
        # Validate images exist
        if isinstance(image_input, list):
            for img_path in image_input:
                if not Path(img_path).exists():
                    logger.error(f"Image not found: {img_path}")
                    return 1
        else:
            if not Path(image_input).exists():
                logger.error(f"Image not found: {image_input}")
                return 1
        
        # Run analysis
        print("\n" + "=" * 60)
        print("BLOOD SMEAR ANALYSIS")
        print("=" * 60)
        
        results = pipeline.analyze(
            image_input,
            save_results=not args.no_save
        )
        
        # Display results
        display_results(results)
        
        if not args.no_save:
            print(f"\n✓ Results saved to: results/")
        
        print("\n✓ Analysis complete!\n")
        return 0
        
    except KeyboardInterrupt:
        print("\n\n❌ Analysis interrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}", file=sys.stderr)
        logging.error("Analysis failed", exc_info=True)
        return 1


def display_results(results: dict) -> None:
    """Display analysis results in readable format."""
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)
    
    # Stage 1: Detection
    if 'stage1_detection' in results:
        detection = results['stage1_detection']
        print("\n📊 STAGE 1: Cell Detection")
        print("-" * 60)
        
        counts = detection.get('total_counts', {})
        for cell_type, count in counts.items():
            print(f"  {cell_type}: {count}")
        
        if detection.get('skipped_paths'):
            print(f"  ⚠️  Skipped {len(detection['skipped_paths'])} images")
    
    # Stage 2: Classification
    if 'stage2_classification' in results:
        classification = results['stage2_classification']
        print("\n🔬 STAGE 2: WBC Classification")
        print("-" * 60)
        
        if 'summary' in classification:
            summary = classification['summary']
            print(f"  Cells classified: {summary.get('sample_count', 0)}")
            
            if 'class_distribution' in summary:
                print("\n  WBC Differential:")
                for cls, count in summary['class_distribution'].items():
                    print(f"    {cls}: {count}")
        
        if 'uncertainty_summary' in classification:
            unc = classification['uncertainty_summary']
            flagged_count = unc.get('flagged_count', unc.get('flagged_samples', 0))
            print(f"\n  Uncertainty Analysis:")
            print(f"    Flagged for review: {flagged_count}/{unc.get('total_samples', 0)}")
            
            if flagged_count > 0:
                print(f"    ⚠️  {unc['flagged_percentage']:.1f}% require expert review")
    
    # Stage 3: Clinical Reasoning
    if 'stage3_reasoning' in results:
        reasoning = results['stage3_reasoning']
        print("\n🏥 STAGE 3: Clinical Reasoning")
        print("-" * 60)
        
        print(f"\n  {reasoning.get('clinical_interpretation', 'N/A')}")
        
        if reasoning.get('key_findings'):
            print("\n  Key Findings:")
            for finding in reasoning['key_findings']:
                print(f"    • {finding}")
        
        if reasoning.get('differential_diagnoses'):
            print("\n  Differential Diagnoses:")
            for dx in reasoning['differential_diagnoses']:
                print(f"    • {dx}")
        
        if reasoning.get('recommendations'):
            print("\n  Recommendations:")
            for rec in reasoning['recommendations']:
                print(f"    • {rec}")
        
        if reasoning.get('safety_flags'):
            print("\n  ⚠️  Safety Flags:")
            for flag in reasoning['safety_flags']:
                print(f"    • {flag}")
        
        if reasoning.get('requires_expert_review'):
            print("\n  🚨 EXPERT REVIEW REQUIRED")
    
    # Metadata
    if 'metadata' in results:
        meta = results['metadata']
        print("\n" + "-" * 60)
        print(f"  Execution time: {meta.get('execution_time_seconds', 0):.2f}s")
        print(f"  Timestamp: {meta.get('timestamp', 'N/A')}")


def run_smoke_test(config_path: str = None) -> int:
    """Run a quick local health-check for configuration and dependencies."""
    print("Running smoke test...")

    try:
        config = load_config(config_path)
        validate_config(config)
        print("✓ Config load and validation passed")
    except Exception as exc:
        print(f"❌ Config check failed: {exc}", file=sys.stderr)
        return 1

    failures = []

    # Model file checks (required for Stage 1 and Stage 2)
    for model_key, label in [
        ('yolo_detection', 'YOLO detection model'),
        ('efficientnet_classification', 'EfficientNet classification model')
    ]:
        try:
            model_path = get_model_path(config, model_key)
            print(f"✓ {label} found: {model_path}")
        except Exception as exc:
            failures.append(f"{label} missing")
            print(f"✗ {label} check failed: {exc}")
            print("  Action: place required .pt files in models/ or set THESIS_MODELS_DIR")

    stage3_enabled = config.get('pipeline.enable_stage3', True)
    api_key_var = config.get('llm.api_key_env_var', 'OPENAI_API_KEY')

    if stage3_enabled:
        if os.getenv(api_key_var):
            print(f"✓ API key variable set: {api_key_var}")
        else:
            print(f"⚠ API key variable not set: {api_key_var}")
            print("  Action: create .env from .env.example and set OPENAI_API_KEY before Stage 3")

        pdf_dir = Path(config.get('rag.pdf_directory', 'LLM_RAG_Pipline/pdfs'))
        pdf_sources = config.get('rag.pdf_sources', [])

        if pdf_dir.exists() and pdf_dir.is_dir():
            print(f"✓ RAG PDF directory found: {pdf_dir}")
        else:
            failures.append("RAG PDF directory missing")
            print(f"✗ RAG PDF directory not found: {pdf_dir}")
            print("  Action: create the directory and add PDFs listed in config.yaml rag.pdf_sources")
            print("  Alternative: set pipeline.enable_stage3=false to run Stage 1/2 only")

        missing_pdfs = [name for name in pdf_sources if not (pdf_dir / name).exists()]
        if missing_pdfs:
            failures.append("RAG PDF files missing")
            print(f"✗ Missing RAG PDF files: {missing_pdfs}")
            print("  Action: add these files under the configured rag.pdf_directory")
            print("  Alternative: set rag.pdf_missing_strategy='warn' for partial retrieval")
        else:
            print(f"✓ All configured RAG PDFs found ({len(pdf_sources)})")
    else:
        print("✓ Stage 3 is disabled (pipeline.enable_stage3=false); skipping API key and PDF checks")

    if failures:
        print(f"❌ Smoke test completed with {len(failures)} critical issue(s)", file=sys.stderr)
        return 1

    print("✓ Smoke test complete")
    return 0


if __name__ == '__main__':
    sys.exit(main())
