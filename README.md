# AI-Powered Assignment Auto-Grader (EARLY PROTOTYPE, MAINLY FOR PERSONAL USE)

An automated grading system for university assignments using LangChain and Reasoning models (Google Gemini 2.0 Flash Thinking or Deepseek-R1 through OpenRouter). This tool helps TAs and instructors grade assignments with consistency and speed, especially for complex tasks like ER diagram evaluation.

## Features

- **Multiple AI Model Support**: Use either Google's Gemini Reasoning model or any reasoning (or even non-reasoning) model available through OpenRouter
- **Multi-run Grading**: Each assignment is graded multiple times (default: 10 runs) to ensure consistency
- **Statistical Analysis**: Calculate mean, median, standard deviation, and consistency scores
- **Multimodal Grading**: Evaluate both text submissions and images (like ER diagrams)
- **Batch Processing**: Grade multiple questions and submissions at once
- **Detailed Reports**: Generate comprehensive feedback for each graded assignment

## Prerequisites

- Python 3.9+
- API keys for either:
  - Google Gemini API (set as `GOOGLE_API_KEY` environment variable)
  - OpenRouter API (set as `OPENROUTER_API_KEY` environment variable)

## Installation

### Option 1: Using pip

1. Clone the repository or download the code
2. Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

### Option 2: Using Conda

1. Clone the repository or download the code
2. Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate autograder
```

### Setting up API Keys

Create a `.env` file in the project directory with your API keys:

```
# Google API Key for Gemini models
GOOGLE_API_KEY=your_google_api_key_here

# OpenRouter API Key
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

## Directory Structure

Prepare your assignments in this structure:

```
submission/
  ├── question1/
  │   ├── question-text.txt   # The assignment description
  │   ├── rubric.txt          # The grading rubric
  │   ├── submission.txt      # Student's text submission
  │   └── task4.png           # Optional image (e.g., ER diagram)
  ├── question2/
  │   └── ...
  └── question3/
      └── ...

<output folder of your choice>
```
Note: Using a markdown-like structure will significantly help the grader perform better.

## Usage

### Batch Grader (Multiple Questions)

To grade multiple questions at once:

```bash
python batch_grader.py
```

Follow the interactive prompts to:
1. Select the AI provider (Gemini or OpenRouter)
2. Choose the specific model
3. Set temperature and number of runs
4. Specify student name and other details

### Single Assignment Grader

To grade a single assignment:

```bash
python grader.py
```

### Output

The grader generates:

- Individual run results for each grading attempt (`run_X_resultsY.json` and `run_X_reportY.txt`)
- Aggregated statistics (`aggregated_resultsY.json`)
- Human-readable summary report (`summary_reportY.txt`) 
- Overall assessment summary (`overall_summary.json`)

## Configuration Options

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| Provider | AI model provider (gemini/openrouter) | openrouter |
| Model (Gemini) | Gemini model name | gemini-2.0-flash-thinking-01-21 |
| Model (OpenRouter) | OpenRouter model identifier | deepseek/deepseek-r1:free |
| Temperature | Controls randomness (0.0-1.0) | 0.1 |
| Runs | Number of grading iterations per question | 10 |

## Best Practices

1. **Use Multiple Runs**: For important assignments, use 10+ runs to ensure grading consistency
2. **Prefer Median Score**: The median total score is generally more reliable than the mean
3. **Check Consistency Metrics**: Lower consistency scores (< 0.8) may indicate ambiguous rubrics
4. **High-Quality Images**: For ER diagrams, ensure the image is clear and contains all required elements

## Understanding the Results

The system produces a statistical summary with:

- **Minimum/Maximum Scores**: Range of scores across multiple runs
- **Mean and Median**: Central tendency measures
- **Standard Deviation**: Measures variability between runs
- **Consistency Score**: 0-1 scale where 1 is perfect consistency

## Contributing

Contributions to improve the auto-grader are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is available under the MIT License. See the LICENSE file for details.
