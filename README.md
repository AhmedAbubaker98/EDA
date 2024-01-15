 **# Automated AI-Driven Data Analysis**

This repository contains code for automating data analysis tasks using large language models (LLMs) and machine learning. 

## Key Features

- **AI-generated code for data visualization:** Leverages OpenAI's GPT-3.5-turbo to create Python code for generating relevant graphs based on a given problem statement and dataset.
- **AI-powered graph analysis:** Uses Google's Gemini-Pro-Vision to analyze generated graphs and provide insights based on the problem statement.
- **Safety settings:** Employs safety settings to mitigate risks of harmful content generation.

## Getting Started

1. **Install required libraries:**
   ```bash
   pip install pandas numpy seaborn matplotlib google-generativeai openai pillow
   ```
2. **Set up API keys:**
   - Create API keys for OpenAI and Google AI.
   - Store them as environment variables: `OPENAI_API_KEY` and `GOOGLE_API_KEY`.
3. **Prepare your dataset:**
   - Place your CSV files in a directory (e.g., `EDA`).
4. **Define a problem statement:**
   - Clearly articulate the question you want the analysis to answer.

## Usage

1. **Run the main script:**
   ```bash
   python main.py
   ```
2. **Provide a problem statement:**
   - You'll be prompted to enter a problem statement.
3. **Analyze the generated insights:**
   - The script will output the AI-generated analysis based on the graphs.

## Code Structure

**Key functions:**

- `read_csv_files(directory_path)`: Reads CSV files from a directory into DataFrames.
- `GraphPromptBuilder(ProblemStatement, dfs, filenames)`: Constructs a prompt for GPT-3.5-turbo to generate graph code.
- `AnalysisPromptBuilder(ProblemStatement)`: Builds a prompt for Gemini-Pro-Vision to analyze graphs.

**Main workflow:**

1. Reads CSV files.
2. Generates graph code using GPT-3.5-turbo.
3. Executes the graph code to create visualizations.
4. Sends the generated graphs to Gemini-Pro-Vision for analysis.
5. Outputs the AI-generated insights.

## Additional Notes

- **Safety:** Employs safety settings to mitigate risks of harmful content generation.
- **Flexibility:** Adaptable to various problem statements and datasets.
- **Potential:** Demonstrates the potential for AI-powered data analysis automation.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
GNU General Public License v3.0
