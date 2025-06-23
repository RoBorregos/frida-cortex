#!/usr/bin/env python3
"""
Script to display test results from a JSON file using the same formatting as model_performance_tester.py
"""

import json
import sys
import argparse
from typing import Dict, List, Optional
from pydantic import BaseModel


class TestResult(BaseModel):
    """Individual test case result"""
    index: int
    input_command: str
    expected_command_count: int
    actual_command_count: int
    score: float
    passed: bool
    execution_time: float
    input_tokens: int
    output_tokens: int
    error: Optional[str] = None
    expected_commands: Optional[List[Dict]] = None
    actual_commands: Optional[List[Dict]] = None
    cmd_category: Optional[str] = None  # New field for task category


class CommandCountGroup(BaseModel):
    """Results grouped by command count"""
    command_count: int
    passed_count: int
    failed_count: int
    total_count: int
    pass_rate: float
    test_results: List[TestResult]


class TaskTypeGroup(BaseModel):
    """Results grouped by task type"""
    task_type: str
    task_description: str
    passed_count: int
    failed_count: int
    total_count: int
    pass_rate: float
    test_results: List[TestResult]


class TestSummary(BaseModel):
    """Complete test execution summary"""
    model_name: str
    total_cases: int
    total_passed: int
    total_failed: int
    overall_pass_rate: float
    average_execution_time: float
    average_input_tokens: int
    average_output_tokens: int
    groups_by_command_count: List[CommandCountGroup]
    groups_by_task_type: List[TaskTypeGroup]  # New field for task type grouping
    
    def print_summary(self):
        """Print a formatted summary of the test results"""
        print("\n" + "=" * 60)
        print(f"TEST SUMMARY - Model: {self.model_name}")
        print("=" * 60)
        print(f"Total Cases: {self.total_cases}")
        print(f"Passed: {self.total_passed} ({self.overall_pass_rate:.1f}%)")
        print(f"Failed: {self.total_failed}")
        print(f"Average Execution Time: {self.average_execution_time:.2f}s")
        print(f"Average Input Tokens: {self.average_input_tokens}")
        print(f"Average Output Tokens: {self.average_output_tokens}")
        print(f"\nResults by Command Count:")
        print("-" * 40)
        for group in sorted(self.groups_by_command_count, key=lambda x: x.command_count):
            print(f"Commands: {group.command_count:2d} | "
                  f"Total: {group.total_count:3d} | "
                  f"Passed: {group.passed_count:3d} | "
                  f"Failed: {group.failed_count:3d} | "
                  f"Pass Rate: {group.pass_rate:5.1f}%")
        
        print(f"\nResults by Task Type:")
        print("-" * 60)
        for group in sorted(self.groups_by_task_type, key=lambda x: x.task_type):
            print(f"Task {group.task_type}: {group.task_description}")
            print(f"  Total: {group.total_count:3d} | "
                  f"Passed: {group.passed_count:3d} | "
                  f"Failed: {group.failed_count:3d} | "
                  f"Pass Rate: {group.pass_rate:5.1f}%")
        print("=" * 60)


def load_and_display_results(json_file_path: str):
    """Load test results from JSON file and display formatted summary"""
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        # Create TestSummary object from JSON data
        test_summary = TestSummary(**data)
        
        # Display the formatted summary
        test_summary.print_summary()
        
    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file '{json_file_path}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to process results: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Display test results from JSON file with formatted output"
    )
    parser.add_argument(
        "json_file", 
        help="Path to the test results JSON file"
    )
    
    args = parser.parse_args()
    load_and_display_results(args.json_file)


if __name__ == "__main__":
    main() 