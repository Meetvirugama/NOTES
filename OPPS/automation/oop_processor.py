import os
from pathlib import Path
from PIL import Image
import google.generativeai as genai
from prompts import SYSTEM_PROMPT
import re

class OOPProcessor:
    def __init__(self, target_folder, output_folder, api_key):
        self.target_folder = Path(target_folder)
        self.output_folder = Path(output_folder)
        self.supported_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        # Using Gemini 1.5 Flash as it is fast, highly capable with multimodal tasks, and cheaper
        # For even deeper reasoning, models/gemini-1.5-pro can be used.
        self.model = genai.GenerativeModel('models/gemini-1.5-flash')
        
    def scan_for_images(self):
        """Recursively scans the target directory for supported images."""
        images = []
        for ext in self.supported_extensions:
            images.extend(self.target_folder.rglob(f"*{ext}"))
            images.extend(self.target_folder.rglob(f"*{ext.upper()}"))
        return sorted(list(set(images)))

    def determine_filename(self, image_path, generated_text):
        """
        Attempts to determine the correct markdown filename based on the image name 
        or the generated content (e.g., searching for a main heading).
        """
        # First, try to infer from the image filename (e.g., "1_oop_overview.jpg" -> "oop-overview.md")
        base_name = image_path.stem.lower()
        base_name = re.sub(r'^\d+_?', '', base_name) # Remove leading numbers (e.g. "1_")
        base_name = re.sub(r'[^a-z0-9]+', '-', base_name).strip('-')
        
        if base_name:
            return f"{base_name}.md"
            
        # Fallback: Extract the first # Heading from the markdown
        match = re.search(r'^#\s+(.+)$', generated_text, re.MULTILINE)
        if match:
            fallback_name = match.group(1).lower()
            fallback_name = re.sub(r'[^a-z0-9]+', '-', fallback_name).strip('-')
            return f"{fallback_name}.md"
            
        return f"generated_{image_path.stem}.md"

    def process_image(self, image_path):
        """Sends the image and prompt to Gemini and saves the output."""
        print(f"Processing: {image_path.name}...")
        
        try:
            # Load Image
            img = Image.open(image_path)
            
            # Call Gemini API
            response = self.model.generate_content(
                [SYSTEM_PROMPT, img],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.4, # Lower temperature for more factual/structured output
                )
            )
            
            markdown_content = response.text
            
            # Ensure output folder exists
            self.output_folder.mkdir(parents=True, exist_ok=True)
            
            # Determine filename and save
            output_filename = self.determine_filename(image_path, markdown_content)
            output_path = self.output_folder / output_filename
            
            # Save the file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
                
            print(f"✅ Success! Saved to: {output_path}")
            
        except Exception as e:
            print(f"❌ Error processing {image_path.name}: {str(e)}")

    def run(self):
        """Main pipeline execution loop."""
        images = self.scan_for_images()
        
        if not images:
            print(f"No supported images found in {self.target_folder}")
            return
            
        print(f"Found {len(images)} images to process.\n")
        
        for img_path in images:
            self.process_image(img_path)
            
        print("\n🎉 Pipeline complete! Check the output directory.")
