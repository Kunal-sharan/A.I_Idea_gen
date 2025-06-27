# AI Idea Generation App
## Link : https://ideation-ai.streamlit.app/
This Streamlit-based application leverages AI to generate innovative project ideas for students' portfolios. It uses advanced language models to create, explain, and visualize project concepts based on user input and web content.

## Features

- **Industry-based Idea Generation**: Users can input an industry or field to get AI-generated project ideas.
- **Web Content Analysis**: The app scrapes and analyzes relevant web content to inform idea generation.
- **Interactive Idea Visualization**: Generated ideas are visualized using Mermaid flowcharts.
- **AI-powered Chatbot**: Users can discuss and refine ideas with an AI assistant.
- **Drag-and-Drop Interface**: Easy selection of preferred web sources for idea generation.

## Technologies Used

- Streamlit
- LangChain
- Google Generative AI (Gemini)
- Groq API
- Mermaid for flowchart generation
- Google Search API
- HTML parsing and transformation

## Setup and Installation

1. Clone the repository
2. Install required packages: `pip install -r requirements.txt`
3. Set up API keys in Streamlit secrets:
   - `GROQ_KEY`
   - `GEMINI_KEY`
4. Run the app: `streamlit run app.py`

## Usage

1. Enter an industry or field in the text input.
2. The app will fetch relevant web content and display summaries.
3. Use the drag-and-drop interface to select preferred sources.
4. Click "Generate Idea" to create an AI-powered project concept.
5. Interact with the chatbot to discuss and refine the idea.

## Note

This application requires valid API keys for Groq and Google Generative AI. Ensure these are properly set up in your Streamlit secrets before running the app.
