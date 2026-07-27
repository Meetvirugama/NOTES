# OOP Notes Automation Pipeline

This is an AI-powered Python automation pipeline designed to recursively scan folders for images of OOP notes and intelligently convert them into highly detailed, FAANG-level interview documentation in Markdown format.

## Technologies Used
- **Python 3.9+**
- **Google GenAI API (Gemini 1.5 Flash)**: Chosen for its speed, multimodal capabilities (image + text), and massive context window.
- **Pillow**: For image loading and processing.

## Project Structure
```text
OPPS/automation/
│
├── main.py               # Command-line interface and entry point
├── oop_processor.py      # Core AI logic (image scanning, API calling, file saving)
├── prompts.py            # The system prompt that guides the LLM to format and expand notes
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Setup Instructions

### 1. Install Dependencies
Make sure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Get a Google Gemini API Key
You will need an API key to use the Gemini model.
1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Create an API key.

### 3. Set the Environment Variable
You can either create a `.env` file in this directory:
```text
# .env file
GEMINI_API_KEY=your_actual_api_key_here
```
OR export it directly in your terminal:
```bash
export GEMINI_API_KEY="your_actual_api_key_here"
```

## How to Run

By default, the script looks for images in `../imgs` and outputs to `../notes`.

Run the pipeline:
```bash
python main.py
```

You can specify custom directories using arguments:
```bash
python main.py --input /path/to/your/images --output /path/to/save/notes
```

## How it Works
1. **Scanning:** `oop_processor.py` recursively scans the input directory for `.png, .jpg, .jpeg, .webp, .bmp` files.
2. **AI Processing:** It loops through every image and sends it alongside the massive `SYSTEM_PROMPT` to the Gemini API.
3. **Enhancement:** The LLM does not just perform OCR; it acts as an expert interviewer, fixing grammar, rewriting code, converting UML to Mermaid diagrams, and injecting complex FAANG-style interview questions.
4. **Saving:** The script intercepts the Markdown, deduces a logical file name, and writes it directly to the output folder.
