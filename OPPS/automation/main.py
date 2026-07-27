import os
import argparse
from dotenv import load_dotenv
from oop_processor import OOPProcessor

def main():
    # Load environment variables from .env file (if exists)
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Automated OOP Notes Extraction Pipeline")
    parser.add_argument(
        "--input", 
        type=str, 
        default="../imgs", 
        help="Path to the directory containing OOP images."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="../notes", 
        help="Path to the output directory where Markdown files will be saved."
    )
    args = parser.parse_args()
    
    # Fetch API Key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found in environment variables.")
        print("Please set it via: export GEMINI_API_KEY='your_key' or add it to a .env file.")
        return

    print("🚀 Starting AI Notes Generation Pipeline...")
    print(f"📂 Input Folder: {os.path.abspath(args.input)}")
    print(f"📂 Output Folder: {os.path.abspath(args.output)}\n")
    
    # Initialize and run processor
    processor = OOPProcessor(
        target_folder=args.input, 
        output_folder=args.output, 
        api_key=api_key
    )
    processor.run()

if __name__ == "__main__":
    main()
