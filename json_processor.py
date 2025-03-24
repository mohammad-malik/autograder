#!/usr/bin/env python3
"""
JSON Submission Processor
------------------------
This module processes JSON-formatted submissions for grading.
It extracts separate questions from a structured JSON file and
prepares them for grading by matching them with the appropriate rubrics.
"""

import os
import json
import re
from pathlib import Path

class JsonSubmissionProcessor:
    """
    A class to process JSON-formatted submissions
    """
    
    def __init__(self, json_file_path, submission_dir="submission"):
        """
        Initialize the processor with paths to the JSON file and submission directory
        
        Args:
            json_file_path: Path to the JSON submission file
            submission_dir: Directory containing question folders with rubrics
        """
        self.json_file_path = json_file_path
        self.submission_dir = submission_dir
        
        # Load the JSON data
        with open(json_file_path, 'r') as f:
            self.json_data = json.load(f)
        
        # Get student info from metadata
        if "metadata" in self.json_data:
            self.student_name = self.json_data["metadata"].get("name", "Unknown Student")
            self.student_number = self.json_data["metadata"].get("student_number", "Unknown")
            self.student_class = self.json_data["metadata"].get("class", "Unknown")
        else:
            self.student_name = "Unknown Student"
            self.student_number = "Unknown"
            self.student_class = "Unknown"
    
    def extract_questions(self):
        """
        Extract questions from the JSON data
        
        Returns:
            A list of dictionaries containing question data
        """
        questions = []
        
        if "sections" not in self.json_data:
            raise ValueError("Invalid JSON format: 'sections' key not found")
        
        # Group sections by question number
        current_question = None
        current_sections = []
        
        for section in self.json_data["sections"]:
            title = section.get("title", "").strip()
            
            # Check if this is a new question
            question_match = re.search(r"QUESTION\s*(\d+)", title, re.IGNORECASE)
            
            if question_match:
                # If we have a previous question, add it to the list
                if current_question is not None and current_sections:
                    questions.append({
                        "question_number": current_question,
                        "sections": current_sections
                    })
                
                # Start a new question
                current_question = question_match.group(1)
                current_sections = [section]
            elif current_question is not None:
                # Add this section to the current question
                current_sections.append(section)
        
        # Add the last question
        if current_question is not None and current_sections:
            questions.append({
                "question_number": current_question,
                "sections": current_sections
            })
        
        return questions
    
    def format_section_content(self, content):
        """
        Format the section content into a text string
        
        Args:
            content: The content field from a section, which can be a string, list, or dict
            
        Returns:
            A formatted text string
        """
        if content is None:
            return ""
        
        if isinstance(content, str):
            return content
        
        if isinstance(content, list):
            result = []
            for item in content:
                if isinstance(item, dict):
                    # Handle tables or other structured data
                    if len(item) == 1:
                        # Simple key-value pair
                        key, value = next(iter(item.items()))
                        result.append(f"{key}: {value}")
                    else:
                        # Multi-column table
                        for k, v in item.items():
                            result.append(f"{k}: {v}")
                else:
                    result.append(str(item))
            return "\n".join(result)
        
        if isinstance(content, dict):
            # Format dict as a table
            result = []
            for k, v in content.items():
                result.append(f"{k}: {v}")
            return "\n".join(result)
        
        return str(content)
    
    def format_question(self, question_data):
        """
        Format a question from its sections into a text submission
        
        Args:
            question_data: Dictionary containing question number and sections
            
        Returns:
            A formatted text string for the question submission
        """
        result = []
        
        # Add a header for the question
        result.append(f"# QUESTION {question_data['question_number']}\n")
        
        # Format each section
        for section in question_data["sections"]:
            title = section.get("title", "").strip()
            content = section.get("content")
            
            # Skip the main question title as we already added it
            if re.search(r"QUESTION\s*\d+", title, re.IGNORECASE) and title == section["title"]:
                continue
            
            # Add section title if it's not empty and different from the question title
            if title:
                result.append(f"## {title}\n")
            
            # Add formatted content
            formatted_content = self.format_section_content(content)
            if formatted_content:
                result.append(formatted_content + "\n")
        
        # Check if there are associated images for this question
        question_num = int(question_data["question_number"])
        if "images" in self.json_data:
            for image in self.json_data["images"]:
                if "order" in image and image["order"] == question_num:
                    result.append(f"\n## ER DIAGRAM\n")
                    result.append(f"[ER Diagram Description: {image.get('description', 'No description provided')}]")
        
        return "\n".join(result)
    
    def get_rubric_for_question(self, question_number):
        """
        Get the rubric for a specific question
        
        Args:
            question_number: The question number
            
        Returns:
            The rubric text or None if not found
        """
        # Look for the rubric file in the submission directory
        question_folder = os.path.join(self.submission_dir, f"question{question_number}")
        rubric_file = os.path.join(question_folder, "rubric.txt")
        
        if os.path.exists(rubric_file):
            with open(rubric_file, 'r') as f:
                return f.read()
        
        return None
    
    def get_question_context(self, question_number):
        """
        Get the question context for a specific question
        
        Args:
            question_number: The question number
            
        Returns:
            The question context text or None if not found
        """
        # Look for the question file in the submission directory
        question_folder = os.path.join(self.submission_dir, f"question{question_number}")
        question_file = os.path.join(question_folder, "question-text.txt")
        
        if os.path.exists(question_file):
            with open(question_file, 'r') as f:
                return f.read()
        
        return None
    
    def get_image_path(self, question_number):
        """
        Get the image path for a specific question
        
        Args:
            question_number: The question number
            
        Returns:
            The image path or None if not found
        """
        # Look for the image file in the submission directory
        question_folder = os.path.join(self.submission_dir, f"question{question_number}")
        image_file = os.path.join(question_folder, "task4.png")
        
        if os.path.exists(image_file):
            return image_file
        
        return None
    
    def prepare_questions_for_grading(self):
        """
        Prepare all questions for grading
        
        Returns:
            A list of dictionaries containing formatted questions with their rubrics
        """
        questions = self.extract_questions()
        grading_data = []
        
        for question_data in questions:
            question_number = question_data["question_number"]
            
            # Format the submission
            submission_text = self.format_question(question_data)
            
            # Get the rubric
            rubric_text = self.get_rubric_for_question(question_number)
            if not rubric_text:
                print(f"Warning: No rubric found for Question {question_number}")
                continue
            
            # Get the question context
            question_context = self.get_question_context(question_number)
            if not question_context:
                print(f"Warning: No question text found for Question {question_number}")
                continue
            
            # Get the image path (optional)
            image_path = self.get_image_path(question_number)
            
            grading_data.append({
                "question_number": question_number,
                "assignment_context": question_context,
                "rubric": rubric_text,
                "submission": submission_text,
                "image_path": image_path
            })
        
        return grading_data

def process_json_submission(json_file_path, submission_dir="submission"):
    """
    Process a JSON submission and return the extracted data
    
    Args:
        json_file_path: Path to the JSON submission file
        submission_dir: Directory containing question folders with rubrics
        
    Returns:
        A list of dictionaries containing formatted questions with their rubrics
    """
    processor = JsonSubmissionProcessor(json_file_path, submission_dir)
    return processor.prepare_questions_for_grading()