#!/usr/bin/env python3
"""
Automated Assignment Grader
--------------------------
This script uses LangChain with either Google's Gemini AI or OpenRouter supplied models 
to grade student assignments based on defined rubrics and assignment context, 
with support for multimodal content including ER diagrams.
"""

import os
import json
import base64
import argparse
from typing import Dict, List, Optional, Union, Any, Literal, Type, Annotated
import re
from datetime import datetime
from pathlib import Path

# Import dotenv for environment variable loading from .env file
try:
    from dotenv import load_dotenv
    # Load environment variables from .env file
    load_dotenv()
    print("Loaded environment variables from .env file")
except ImportError:
    print("python-dotenv not installed. To use .env files, install with: pip install python-dotenv")

# LangChain imports
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
# Changed import path to langchain_core for OutputParser
from langchain_core.output_parsers import BaseOutputParser

# Updated to use Pydantic v2 directly
from pydantic import BaseModel, Field, field_validator, model_validator

# LLM providers
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# For handling images
from PIL import Image
import io
from IPython.display import display
import mimetypes
from langchain_core.messages import HumanMessage

class GradingResult(BaseModel):
    """Schema for the grading output"""
    criteria_scores: Dict[str, int] = Field(description="Scores for each grading criteria")
    total_score: int = Field(description="Total score for the assignment")
    feedback: Dict[str, str] = Field(description="Feedback for each criteria")
    overall_comments: str = Field(description="Overall comments on the submission")
    
    # Updated validator syntax for Pydantic v2
    @model_validator(mode='after')
    def check_total_score(self) -> 'GradingResult':
        """Validate that total score equals sum of criteria scores"""
        if sum(self.criteria_scores.values()) != self.total_score:
            raise ValueError(f"Total score {self.total_score} does not match sum of criteria scores {sum(self.criteria_scores.values())}")
        return self

# Fixed custom Pydantic output parser for v2
class PydanticV2OutputParser(BaseOutputParser, BaseModel):
    """Output parser that handles Pydantic v2 model schemas."""
    
    pydantic_object: Type[BaseModel] = Field(..., description="The Pydantic model to parse output into")
    
    def parse(self, text: str) -> Any:
        """Parse the output text into a Pydantic object."""
        try:
            # Attempt to parse as JSON
            json_output = json.loads(text)
            return self.pydantic_object.model_validate(json_output)
        except Exception as e:
            # If JSON parsing fails, try to extract JSON from text
            match = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)
            if match:
                try:
                    json_output = json.loads(match.group(1))
                    return self.pydantic_object.model_validate(json_output)
                except Exception as e2:
                    raise ValueError(f"Failed to parse output: {e2}\nExtracted JSON: {match.group(1)}")
            else:
                raise ValueError(f"Failed to parse output: {e}\nText: {text}")
    
    def get_format_instructions(self) -> str:
        """Get formatting instructions for the output parser."""
        # Use model_json_schema() for Pydantic v2
        schema = self.pydantic_object.model_json_schema()
        
        # Create clear formatting instructions
        instructions = f"""
        Your response should be formatted as a JSON object with the following structure:
        
        ```json
        {json.dumps(schema, indent=2)}
        ```
        
        Ensure that:
        1. The total_score equals the sum of all criteria_scores
        2. Each criteria has both a score and corresponding feedback
        3. Include overall_comments summarizing the entire submission
        """
        
        return instructions

class AssignmentGrader:
    """Class to handle the grading of assignments using LangChain and various LLM providers"""
    
    def __init__(self, 
                 provider: Literal["gemini", "openrouter"] = "gemini",
                 gemini_model_name: str = "gemini-2.0-flash-thinking-exp-01-21",
                 openrouter_model_name: str = "deepseek/deepseek-r1:free",
                 gemini_api_key: Optional[str] = None,
                 openrouter_api_key: Optional[str] = None,
                 temperature: float = 0.1):
        """
        Initialize the grader with API keys and model configuration
        
        Args:
            provider: The LLM provider to use ("gemini" or "openrouter")
            gemini_model_name: The name of the Gemini model to use
            openrouter_model_name: The name of the OpenRouter model to use
            gemini_api_key: Google API key (for Gemini models)
            openrouter_api_key: OpenRouter API key
            temperature: Temperature setting for the model (controls randomness)
        """
        self.provider = provider.lower()
        self.gemini_model_name = gemini_model_name
        self.openrouter_model_name = openrouter_model_name
        self.temperature = temperature
        
        # Get API keys
        self.gemini_api_key = gemini_api_key or os.environ.get("GOOGLE_API_KEY")
        self.openrouter_api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
        
        # Validate API keys based on provider
        if self.provider == "gemini" and not self.gemini_api_key:
            raise ValueError("Google API key is required for Gemini. Set it as GOOGLE_API_KEY environment variable or pass it to the constructor.")
        
        if self.provider == "openrouter" and not self.openrouter_api_key:
            raise ValueError("OpenRouter API key is required for OpenRouter. Set it as OPENROUTER_API_KEY environment variable or pass it to the constructor.")
        
        # Initialize the LLM based on provider
        self.llm = self._initialize_llm()
        
        # Initialize the output parser with Pydantic v2
        self.output_parser = PydanticV2OutputParser(pydantic_object=GradingResult)
    
    def _initialize_llm(self):
        """Initialize the LLM based on the selected provider"""
        if self.provider == "gemini":
            return ChatGoogleGenerativeAI(
                model=self.gemini_model_name,
                google_api_key=self.gemini_api_key,
                temperature=self.temperature,
            )
        elif self.provider == "openrouter":
            # Fix: Use api_key instead of openrouter_api_key
            return ChatOpenAI(
                model=self.openrouter_model_name,
                openai_api_key=self.openrouter_api_key,
                temperature=self.temperature,
                base_url="https://openrouter.ai/api/v1"
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}. Use 'gemini' or 'openrouter'.")
        
    def _process_image(self, image_path: str) -> Dict[str, Any]:
        """Process an image file and return it in a format suitable for the model"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # Get MIME type based on file extension
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type or not mime_type.startswith('image/'):
            mime_type = 'image/jpeg'  # Default to JPEG if can't determine
            
        # Read the image file
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()

        # Convert to base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        return {
            "type": mime_type,
            "data": base64_image
        }
        
    def _create_grading_prompt(self, assignment_context: str, rubric: str, submission: str, image_path: Optional[str] = None) -> Union[str, List[Union[str, Dict]]]:
        """Create a prompt for the grading task, with optional image support"""
        
        prompt_template = """
        # Assignment Grading Task
        
        You are an educational assistant tasked with grading student assignments. Your goal is to provide fair, 
        consistent, and detailed feedback based on the provided rubric.
        
        ## Assignment Context:
        {assignment_context}
        
        ## Grading Rubric:
        {rubric}
        
        ## Student Submission:
        {submission}
        
        ## Grading Instructions:
        1. Carefully read the assignment context, rubric, and student submission
        2. If there's an ER diagram image, analyze it carefully as it's a critical part of the submission
        3. Grade each criteria according to the rubric
        4. Provide specific, constructive feedback for each criteria
        5. Calculate the total score
        6. Add overall comments highlighting strengths and areas for improvement
        
        {format_instructions}
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["assignment_context", "rubric", "submission"],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()}
        )
        
        formatted_text = prompt.format(
            assignment_context=assignment_context,
            rubric=rubric,
            submission=submission
        )
        
        # If there's no image, return just the text prompt
        if not image_path:
            return formatted_text
        
        # If there's an image, create a multimodal message
        try:
            image_data = self._process_image(image_path)
            
            # Different formats for different providers
            if self.provider == "gemini":
                multimodal_message = HumanMessage(
                    content=[
                        {"type": "text", "text": formatted_text},
                        {"type": "image_url", "image_url": f"data:{image_data['type']};base64,{image_data['data']}"}
                    ]
                )
            elif self.provider == "openrouter":
                multimodal_message = HumanMessage(
                    content=[
                        {"type": "text", "text": formatted_text},
                        {"type": "image_url", "image_url": f"data:{image_data['type']};base64,{image_data['data']}"}
                    ]
                )
                
            return multimodal_message
        except Exception as e:
            print(f"Warning: Could not process image {image_path}: {e}")
            print("Proceeding with text-only grading.")
            return formatted_text
    
    def grade_assignment(self, assignment_context: str, rubric: str, submission: str, image_path: Optional[str] = None) -> GradingResult:
        """Grade a student assignment based on provided context, rubric and submission, with optional image"""
        
        # Create the prompt (text-only or multimodal)
        grading_input = self._create_grading_prompt(assignment_context, rubric, submission, image_path)
        
        # Process differently based on whether we have an image or not
        if isinstance(grading_input, str):
            # Text-only input
            grading_chain = LLMChain(
                llm=self.llm,
                prompt=PromptTemplate.from_template(grading_input)
            )
            raw_response = grading_chain.run("")
        else:
            # Multimodal input
            raw_response = self.llm.invoke([grading_input]).content
        
        # Parse the response into a GradingResult object
        try:
            # Extract the JSON part of the response
            json_match = re.search(r'```json\n(.*?)\n```', raw_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = raw_response
                
            # Parse the result
            result = self.output_parser.parse(json_str)
            return result
        except Exception as e:
            print(f"Error parsing grading result: {e}")
            print(f"Raw response: {raw_response}")
            raise
    
    def generate_report(self, result: GradingResult, student_name: str, assignment_name: str) -> str:
        """Generate a formatted report from the grading result"""
        
        report = f"""
        # Grading Report: {assignment_name}
        
        ## Student: {student_name}
        ## Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        ## Total Score: {result.total_score}
        
        ## Detailed Feedback:
        """
        
        for criteria, score in result.criteria_scores.items():
            report += f"\n### {criteria}: {score} points\n"
            report += f"{result.feedback.get(criteria, 'No specific feedback provided.')}\n"
        
        report += f"\n## Overall Comments:\n{result.overall_comments}\n"
        
        return report
    
    def save_report(self, report: str, filename: str) -> None:
        """Save the grading report to a file"""
        with open(filename, 'w') as f:
            f.write(report)
        print(f"Report saved to {filename}")
        
    def process_assignment(self, 
                          student_name: str,
                          assignment_name: str,
                          assignment_context: str, 
                          rubric: str, 
                          submission: str,
                          image_path: Optional[str] = None,
                          output_file: Optional[str] = None) -> str:
        """Process a complete assignment and generate a report"""
        
        # Grade the assignment
        result = self.grade_assignment(assignment_context, rubric, submission, image_path)
        
        # Generate the report
        report = self.generate_report(result, student_name, assignment_name)
        
        # Save the report if requested
        if output_file:
            self.save_report(report, output_file)
            
        return report


def extract_components(prompt: str) -> Dict[str, str]:
    """Extract assignment context, rubric and submission from a prompt"""
    components = {}
    
    # Extract context
    context_match = re.search(r'# Context\s+(.+?)(?=##|$)', prompt, re.DOTALL)
    if context_match:
        components['context'] = context_match.group(1).strip()
    
    # Extract rubric
    rubric_match = re.search(r'## Rubric\s+(.+?)(?=##|$)', prompt, re.DOTALL)
    if rubric_match:
        components['rubric'] = rubric_match.group(1).strip()
    
    # Extract submission
    submission_match = re.search(r"## Student's Submission\s+(.+?)(?=##|$)", prompt, re.DOTALL)
    if submission_match:
        components['submission'] = submission_match.group(1).strip()
    
    return components


def main():
    """Main function to demonstrate the grader"""
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Assignment Auto-Grader')
    parser.add_argument('--provider', type=str, choices=['gemini', 'openrouter'], default='openrouter',
                        help='LLM provider to use (gemini or openrouter)')
    parser.add_argument('--gemini-model', type=str, default='gemini-2.0-flash-thinking-exp-01-21',
                        help='Gemini model name to use')
    parser.add_argument('--openrouter-model', type=str, default='deepseek/deepseek-r1:free',
                        help='OpenRouter model name to use')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Temperature setting for the model (controls randomness)')
    parser.add_argument('--input-file', type=str, help='Path to input file containing grading prompt')
    parser.add_argument('--image-file', type=str, help='Path to ER diagram image')
    parser.add_argument('--student-name', type=str, default='Student', help='Name of the student')
    parser.add_argument('--assignment-name', type=str, default='Assignment', help='Name of the assignment')
    parser.add_argument('--output-file', type=str, help='Path to save the output report')
    
    args = parser.parse_args()
    
    # Check for API keys
    if args.provider == 'gemini' and not os.environ.get("GOOGLE_API_KEY"):
        print("Please set the GOOGLE_API_KEY environment variable for Gemini")
        return
    
    if args.provider == 'openrouter' and not os.environ.get("OPENROUTER_API_KEY"):
        print("Please set the OPENROUTER_API_KEY environment variable for OpenRouter")
        return
    
    # Example usage
    print("Assignment Auto-Grader")
    print("=" * 50)
    print(f"Using provider: {args.provider}")
    
    # Get input from file or manual input
    if args.input_file:
        try:
            with open(args.input_file, 'r') as f:
                prompt = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return
    else:
        input_method = input("Grade from file (f) or manual input (m)? ").lower()
        if input_method == 'f':
            filename = input("Enter the text file path: ")
            try:
                with open(filename, 'r') as f:
                    prompt = f.read()
            except Exception as e:
                print(f"Error reading file: {e}")
                return
        else:
            print("Enter the full grading prompt (end with Ctrl+D on a new line):")
            prompt_lines = []
            try:
                while True:
                    line = input()
                    prompt_lines.append(line)
            except EOFError:
                pass
            prompt = '\n'.join(prompt_lines)
    
    # Extract components from the prompt
    components = extract_components(prompt)
    
    if not all(k in components for k in ['context', 'rubric', 'submission']):
        print("Error: Could not extract all required components from the prompt")
        return
    
    # Check if there's an ER diagram image to include
    image_path = args.image_file
    if not image_path:
        image_path = input("Enter path to ER diagram image (leave blank if none): ")
    if image_path and not os.path.exists(image_path):
        print(f"Warning: Image path {image_path} does not exist")
        image_path = None
    
    # Get student and assignment information
    student_name = args.student_name or input("Enter student name: ")
    assignment_name = args.assignment_name or input("Enter assignment name: ")
    output_file = args.output_file
    if not output_file:
        output_file = input("Enter output file name (leave blank to print only): ")
    
    # Create and run the grader
    try:
        grader = AssignmentGrader(
            provider=args.provider,
            gemini_model_name=args.gemini_model,
            openrouter_model_name=args.openrouter_model,
            temperature=args.temperature
        )
        
        report = grader.process_assignment(
            student_name=student_name,
            assignment_name=assignment_name,
            assignment_context=components['context'],
            rubric=components['rubric'],
            submission=components['submission'],
            image_path=image_path if image_path else None,
            output_file=output_file if output_file else None
        )
        
        # Print the report
        print("\n" + "=" * 50)
        print("Grading Report:")
        print(report)
    
    except Exception as e:
        print(f"Error during grading: {e}")


if __name__ == "__main__":
    main()