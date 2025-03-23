#!/usr/bin/env python3
"""
Batch Assignment Grader
-----------------------
This script processes multiple assignment questions in batch mode,
using the AssignmentGrader class to evaluate each submission.

Each question is graded multiple times to provide more reliable evaluations
and to analyze the consistency of the grading model.
"""

import os
import json
import glob
import sys
import statistics
from pathlib import Path
from collections import defaultdict

# Import dotenv for environment variable loading from .env file
try:
    from dotenv import load_dotenv
    # Load environment variables from .env file
    load_dotenv()
    print("Loaded environment variables from .env file")
except ImportError:
    print("python-dotenv not installed. To use .env files, install with: pip install python-dotenv")

# Import AssignmentGrader with error handling
try:
    from grader import AssignmentGrader
except ImportError as e:
    print(f"Error importing grader: {e}")
    print("Please make sure the grader.py file is in the same directory")
    sys.exit(1)

def process_question_folder(grader, question_folder, output_folder, student_name, num_runs=10):
    """
    Process a single question folder and grade the assignment multiple times
    
    Args:
        grader: AssignmentGrader instance
        question_folder: Path to the question folder
        output_folder: Path to save the results
        student_name: Name of the student
        num_runs: Number of times to run the grading process
    
    Returns:
        Dict containing aggregated results
    """
    print(f"\nProcessing question folder: {question_folder}")
    
    # Extract the question number from the folder name
    question_number = os.path.basename(question_folder).replace("question", "")
    
    # Construct paths to required files
    question_file = os.path.join(question_folder, "question-text.txt")
    rubric_file = os.path.join(question_folder, "rubric.txt")
    submission_file = os.path.join(question_folder, "submission.txt")
    image_file = os.path.join(question_folder, "task4.png")
    
    # Ensure all required files exist
    if not os.path.exists(question_file):
        print(f"Error: Question file not found at {question_file}")
        return None
        
    if not os.path.exists(rubric_file):
        print(f"Error: Rubric file not found at {rubric_file}")
        return None
        
    if not os.path.exists(submission_file):
        print(f"Error: Submission file not found at {submission_file}")
        return None
    
    # The image file is optional but we should check if it exists
    if not os.path.exists(image_file):
        print(f"Warning: ER diagram image not found at {image_file}")
        image_file = None
    
    # Read the files
    try:
        with open(question_file, 'r') as f:
            question_text = f.read()
            
        with open(rubric_file, 'r') as f:
            rubric_text = f.read()
            
        with open(submission_file, 'r') as f:
            submission_text = f.read()
            
        print(f"Successfully loaded question {question_number} files")
    except Exception as e:
        print(f"Error reading files: {e}")
        return None
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Lists to store all run results
    all_results = []
    all_reports = []
    
    print(f"Running {num_runs} grading iterations for Question {question_number}...")
    
    # Process the assignment multiple times
    for run_num in range(1, num_runs + 1):
        print(f"  Run {run_num}/{num_runs}...")
        
        try:
            # Grade the assignment
            result = grader.grade_assignment(
                assignment_context=question_text,
                rubric=rubric_text,
                submission=submission_text,
                image_path=image_file
            )
            
            # Generate the report for this run
            report = grader.generate_report(
                result=result,
                student_name=student_name,
                assignment_name=f"Question {question_number} - Run {run_num}"
            )
            
            # Save this run's results and report
            run_json_file = os.path.join(output_folder, f"run_{run_num}_results{question_number}.json")
            with open(run_json_file, 'w') as f:
                json.dump({
                    "criteria_scores": result.criteria_scores,
                    "total_score": result.total_score,
                    "feedback": result.feedback,
                    "overall_comments": result.overall_comments
                }, f, indent=2)
            
            run_report_file = os.path.join(output_folder, f"run_{run_num}_report{question_number}.txt")
            with open(run_report_file, 'w') as f:
                f.write(report)
            
            # Store results for aggregation
            all_results.append(result)
            all_reports.append(report)
            
            print(f"    Completed run {run_num} with total score: {result.total_score}")
            
        except Exception as e:
            print(f"    Error in run {run_num}: {str(e)}")
            continue
    
    # Check if we have any successful runs
    if not all_results:
        print(f"Error: All runs failed for Question {question_number}")
        return None
    
    print(f"Aggregating results from {len(all_results)} successful runs...")
    
    # Aggregate the results
    aggregated_data = aggregate_results(all_results, question_number)
    
    # Save the aggregated results
    agg_json_file = os.path.join(output_folder, f"aggregated_results{question_number}.json")
    with open(agg_json_file, 'w') as f:
        json.dump(aggregated_data, f, indent=2)
    
    print(f"Aggregated results saved to {agg_json_file}")
    
    # Create a summary report
    summary_report = create_summary_report(aggregated_data, student_name, question_number)
    summary_file = os.path.join(output_folder, f"summary_report{question_number}.txt")
    with open(summary_file, 'w') as f:
        f.write(summary_report)
    
    print(f"Summary report saved to {summary_file}")
    
    return aggregated_data

def aggregate_results(results, question_number):
    """
    Aggregate results from multiple grading runs
    
    Args:
        results: List of GradingResult objects
        question_number: The question number
    
    Returns:
        Dict containing aggregated data
    """
    # Initialize data structures
    all_criteria = set()
    for result in results:
        all_criteria.update(result.criteria_scores.keys())
    
    criteria_scores = defaultdict(list)
    total_scores = []
    feedback = defaultdict(list)
    overall_comments = []
    
    # Collect all scores and feedback
    for result in results:
        for criteria in all_criteria:
            if criteria in result.criteria_scores:
                criteria_scores[criteria].append(result.criteria_scores[criteria])
            if criteria in result.feedback:
                feedback[criteria].append(result.feedback[criteria])
        
        total_scores.append(result.total_score)
        overall_comments.append(result.overall_comments)
    
    # Calculate statistics
    agg_data = {
        "question_number": question_number,
        "num_runs": len(results),
        "criteria_stats": {},
        "total_score_stats": {
            "min": min(total_scores),
            "max": max(total_scores),
            "mean": statistics.mean(total_scores),
            "median": statistics.median(total_scores),
            "stdev": statistics.stdev(total_scores) if len(total_scores) > 1 else 0
        },
        "individual_runs": []
    }
    
    # Process criteria statistics
    for criteria in all_criteria:
        scores = criteria_scores[criteria]
        if scores:
            agg_data["criteria_stats"][criteria] = {
                "min": min(scores),
                "max": max(scores),
                "mean": statistics.mean(scores),
                "median": statistics.median(scores),
                "stdev": statistics.stdev(scores) if len(scores) > 1 else 0,
                "mode": statistics.mode(scores) if len(scores) > 1 else scores[0],
                "consistency": calculate_consistency(scores)
            }
    
    # Store individual run data
    for i, result in enumerate(results):
        run_data = {
            "run_number": i + 1,
            "total_score": result.total_score,
            "criteria_scores": result.criteria_scores
        }
        agg_data["individual_runs"].append(run_data)
    
    # Select representative feedback (mode score for each criteria)
    representative_feedback = {}
    for criteria in all_criteria:
        if criteria in criteria_scores and criteria in feedback:
            # Find the feedback associated with the most common score
            mode_score = agg_data["criteria_stats"][criteria]["mode"]
            mode_feedback = []
            for i, score in enumerate(criteria_scores[criteria]):
                if score == mode_score and i < len(feedback[criteria]):
                    mode_feedback.append(feedback[criteria][i])
            
            if mode_feedback:
                representative_feedback[criteria] = mode_feedback[0]
    
    agg_data["representative_feedback"] = representative_feedback
    
    return agg_data

def calculate_consistency(scores):
    """Calculate a consistency metric between 0 and 1"""
    if not scores or len(scores) <= 1:
        return 1.0  # Single score is perfectly consistent with itself
    
    # Use coefficient of variation (lower is more consistent)
    mean = statistics.mean(scores)
    stdev = statistics.stdev(scores)
    
    if mean == 0:
        return 1.0 if stdev == 0 else 0.0
    
    cv = stdev / mean
    # Transform to 0-1 scale where 1 is perfectly consistent
    consistency = max(0, min(1, 1 - cv))
    return consistency

def create_summary_report(aggregated_data, student_name, question_number):
    """
    Create a summary report from the aggregated results
    
    Args:
        aggregated_data: Dict containing aggregated results
        student_name: Name of the student
        question_number: The question number
    
    Returns:
        String containing the summary report
    """
    total_stats = aggregated_data["total_score_stats"]
    
    report = f"""
    # Summary Grading Report: Question {question_number}
    
    ## Student: {student_name}
    ## Number of Grading Runs: {aggregated_data["num_runs"]}
    
    ## Overall Score Statistics
    - Minimum: {total_stats["min"]}
    - Maximum: {total_stats["max"]}
    - Mean: {total_stats["mean"]:.2f}
    - Median: {total_stats["median"]}
    - Standard Deviation: {total_stats["stdev"]:.2f}
    
    ## Criteria Score Statistics
    """
    
    for criteria, stats in aggregated_data["criteria_stats"].items():
        report += f"""
    ### {criteria}
    - Mean Score: {stats["mean"]:.2f} (Min: {stats["min"]}, Max: {stats["max"]})
    - Consistency: {stats["consistency"]:.2f} (1.0 = perfectly consistent)
    - Representative Feedback: 
      {aggregated_data["representative_feedback"].get(criteria, "No feedback available")}
    """
    
    report += """
    ## Score Consistency Analysis
    The consistency metric ranges from 0 to 1, where:
    - 1.0 indicates perfect consistency (same score in every run)
    - Values close to 1.0 indicate high consistency
    - Lower values indicate greater variability across runs
    
    ## Recommendation
    Consider using the median total score as the final grade, as it's less affected by outliers.
    """
    
    return report

def process_all_questions(api_key=None, provider="gemini", gemini_model_name=None, openrouter_model_name=None, 
                        submission_dir=None, output_base_dir=None, student_name="Student", 
                        num_runs=10, temperature=0.1):
    """
    Process all questions in the submission directory
    
    Args:
        api_key: API key for the selected provider
        provider: LLM provider to use ("gemini" or "openrouter")
        gemini_model_name: Name of the Gemini model to use
        openrouter_model_name: Name of the OpenRouter model to use
        submission_dir: Path to submission directory containing question folders
        output_base_dir: Base directory for output files
        student_name: Name of the student
        num_runs: Number of times to run the grading process per question
        temperature: Temperature setting for the model
    """
    # Default paths if not provided
    if not submission_dir:
        submission_dir = "submission"
    
    if not output_base_dir:
        output_base_dir = "."
    
    # Check for API keys based on provider
    if provider == "gemini":
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("Error: Google API key is required for Gemini. Set GOOGLE_API_KEY environment variable or pass it as a parameter.")
            return
    else:  # OpenRouter
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            print("Error: OpenRouter API key is required for OpenRouter. Set OPENROUTER_API_KEY environment variable or pass it as a parameter.")
            return
    
    # Initialize the grader with the appropriate provider and model
    try:
        if provider == "gemini":
            grader = AssignmentGrader(
                provider="gemini",
                gemini_model_name=gemini_model_name,
                gemini_api_key=api_key,
                temperature=temperature
            )
        else:  # OpenRouter
            grader = AssignmentGrader(
                provider="openrouter",
                openrouter_model_name=openrouter_model_name,
                openrouter_api_key=api_key,
                temperature=temperature
            )
    except Exception as e:
        print(f"Error initializing grader: {e}")
        return
    
    print(f"Grading using {provider.upper()} model: {gemini_model_name if provider == 'gemini' else openrouter_model_name}")
    
    # Find all question folders
    question_folders = sorted(glob.glob(os.path.join(submission_dir, "question*")))
    
    if not question_folders:
        print(f"Error: No question folders found in {submission_dir}")
        return
    
    print(f"Found {len(question_folders)} question folders: {question_folders}")
    
    # Process each question
    all_question_results = {}
    
    for folder in question_folders:
        question_num = os.path.basename(folder).replace("question", "")
        output_folder = os.path.join(output_base_dir, f"q{question_num}")
        
        print(f"\nGrading Question {question_num}")
        print("=" * 50)
        
        results = process_question_folder(
            grader, 
            folder, 
            output_folder, 
            student_name,
            num_runs=num_runs
        )
        
        if results:
            print(f"Question {question_num} graded successfully with {num_runs} runs")
            all_question_results[question_num] = {
                "median_score": results["total_score_stats"]["median"],
                "mean_score": results["total_score_stats"]["mean"]
            }
        else:
            print(f"Failed to grade Question {question_num}")
    
    # Generate overall summary
    if all_question_results:
        overall_summary = {
            "student_name": student_name,
            "question_results": all_question_results,
            "total_median_score": sum(q["median_score"] for q in all_question_results.values()),
            "total_mean_score": sum(q["mean_score"] for q in all_question_results.values()),
            "provider": provider,
            "model": gemini_model_name if provider == "gemini" else openrouter_model_name
        }
        
        with open(os.path.join(output_base_dir, "overall_summary.json"), 'w') as f:
            json.dump(overall_summary, f, indent=2)
        
        print("\nOverall summary saved to overall_summary.json")
    
    print("\nBatch grading complete!")

if __name__ == "__main__":
    print("Assignment Batch Grader")
    print("=" * 50)
    
    # Interactive model selection
    while True:
        provider = input("Select LLM provider (gemini/openrouter) [default: openrouter]: ").lower() or "openrouter"
        if provider in ["gemini", "openrouter"]:
            break
        print("Invalid option. Please enter 'gemini' or 'openrouter'.")
    
    # Set default model names
    gemini_model_name = "gemini-2.0-flash-thinking-exp-01-21"
    openrouter_model_name = "deepseek/deepseek-r1:free"
    
    # Get appropriate model name
    if provider == "gemini":
        gemini_model_name = input(f"Enter Gemini model name [default: {gemini_model_name}]: ") or gemini_model_name
    else:  # OpenRouter
        openrouter_model_name = input(f"Enter OpenRouter model name [default: {openrouter_model_name}]: ") or openrouter_model_name
    
    # Get temperature
    try:
        temp_input = input("Enter temperature (0.0-1.0) [default: 0.1]: ") or "0.1"
        temperature = float(temp_input)
    except ValueError:
        print("Invalid temperature. Using default value of 0.1")
        temperature = 0.1
    
    # Check for API keys based on provider
    api_key = None
    if provider == 'gemini':
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("Error: GOOGLE_API_KEY environment variable not set")
            sys.exit(1)
    else:  # OpenRouter
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            print("Error: OPENROUTER_API_KEY environment variable not set")
            sys.exit(1)
    
    # Get student name
    student_name = input("Enter student name (leave blank for 'Student'): ") or "Student"
    
    # Get number of runs
    try:
        runs_input = input("Enter number of grading runs per question (default: 10): ") or "10"
        num_runs = int(runs_input)
        if num_runs < 1:
            print("Number of runs must be at least 1. Using default value of 10.")
            num_runs = 10
    except ValueError:
        print("Invalid input. Using default value of 10 runs.")
        num_runs = 10
    
    # Get submission directory
    submission_dir = input("Enter submission directory path [default: submission]: ") or "submission"
    
    # Get output directory
    output_dir = input("Enter output directory path [default: .]: ") or "."
    
    print(f"Using provider: {provider}")
    if provider == "gemini":
        print(f"Model: {gemini_model_name}")
    else:
        print(f"Model: {openrouter_model_name}")
    print(f"Runs per question: {num_runs}")
    
    # Check if submission directory exists
    if not os.path.exists(submission_dir):
        print(f"Error: Submission directory {submission_dir} not found")
        sys.exit(1)
    
    # Run the batch grading
    process_all_questions(
        api_key=api_key,
        provider=provider,
        gemini_model_name=gemini_model_name,
        openrouter_model_name=openrouter_model_name,
        submission_dir=submission_dir,
        output_base_dir=output_dir,
        student_name=student_name,
        num_runs=num_runs,
        temperature=temperature
    )