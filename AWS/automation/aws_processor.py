import os
from pathlib import Path
from PIL import Image
import google.generativeai as genai
from prompts import SYSTEM_PROMPT
import re

class AWSProcessor:
    def __init__(self, target_folder, output_folder, api_key):
        self.target_folder = Path(target_folder)
        self.output_folder = Path(output_folder)
        self.supported_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('models/gemini-1.5-flash')
        
    def scan_for_images(self):
        images = []
        for ext in self.supported_extensions:
            images.extend(self.target_folder.rglob(f"*{ext}"))
            images.extend(self.target_folder.rglob(f"*{ext.upper()}"))
        return sorted(list(set(images)))

    def determine_filename(self, image_path, generated_text):
        base_name = image_path.stem.lower()
        base_name = re.sub(r'^\d+_?', '', base_name)
        base_name = re.sub(r'[^a-z0-9]+', '-', base_name).strip('-')
        
        if base_name:
            return f"{base_name}.md"
            
        match = re.search(r'^#\s+(.+)$', generated_text, re.MULTILINE)
        if match:
            fallback_name = match.group(1).lower()
            fallback_name = re.sub(r'[^a-z0-9]+', '-', fallback_name).strip('-')
            return f"{fallback_name}.md"
            
        return f"generated_aws_{image_path.stem}.md"

    def process_image(self, image_path):
        print(f"Processing AWS Diagram: {image_path.name}...")
        
        try:
            img = Image.open(image_path)
            
            response = self.model.generate_content(
                [SYSTEM_PROMPT, img],
                generation_config=genai.types.GenerationConfig(temperature=0.4)
            )
            
            markdown_content = response.text
            self.output_folder.mkdir(parents=True, exist_ok=True)
            
            output_filename = self.determine_filename(image_path, markdown_content)
            output_path = self.output_folder / output_filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
                
            print(f"✅ Success! Saved AWS notes to: {output_path}")
            
        except Exception as e:
            print(f"❌ Error processing {image_path.name}: {str(e)}")

    def run(self):
        images = self.scan_for_images()
        if not images:
            print(f"No supported images found in {self.target_folder}")
            return
            
        print(f"Found {len(images)} AWS images to process.\n")
        for img_path in images:
            self.process_image(img_path)
            
        print("\n🎉 Pipeline complete! Check the AWS output directory.")
