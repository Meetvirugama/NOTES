import os
import argparse
from dotenv import load_dotenv
from docker_processor import DockerProcessor

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Automated Docker Systems Design Extraction Pipeline")
    parser.add_argument("--input", type=str, default="../imgs", help="Path to Docker diagrams.")
    parser.add_argument("--output", type=str, default="../notes", help="Output path for Markdown.")
    args = parser.parse_args()
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found in environment variables.")
        return

    print("🚀 Starting Docker 50 LPA Generation Pipeline...")
    processor = DockerProcessor(target_folder=args.input, output_folder=args.output, api_key=api_key)
    processor.run()

if __name__ == "__main__":
    main()
