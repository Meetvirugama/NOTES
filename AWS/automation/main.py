import os
import argparse
from dotenv import load_dotenv
from aws_processor import AWSProcessor

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Automated AWS Systems Design Extraction Pipeline")
    parser.add_argument("--input", type=str, default="../imgs", help="Path to AWS diagrams.")
    parser.add_argument("--output", type=str, default="../notes", help="Output path for Markdown.")
    args = parser.parse_args()
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found in environment variables.")
        return

    print("🚀 Starting AWS 50 LPA Generation Pipeline...")
    processor = AWSProcessor(target_folder=args.input, output_folder=args.output, api_key=api_key)
    processor.run()

if __name__ == "__main__":
    main()
