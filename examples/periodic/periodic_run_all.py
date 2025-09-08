#!/usr/bin/env python3
"""
Master orchestration script for periodic potential analysis.

This script runs the complete workflow:
1. Batch simulations with parameter sweeps
2. Postprocessing of simulation data  
3. Comprehensive plotting and visualization

Usage:
    python periodic_run_all.py
"""

import subprocess
import sys
import time
import os


def run_step(script_name, description, timeout_minutes=30):
    """
    Run a workflow step and handle errors.
    
    Parameters:
    - script_name: name of Python script to run
    - description: human-readable description of the step
    - timeout_minutes: maximum time to allow for completion
    
    Returns:
    - True if successful, False if failed
    """
    
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    
    if not os.path.exists(script_name):
        print(f"❌ Script {script_name} not found!")
        return False
    
    start_time = time.time()
    timeout_seconds = timeout_minutes * 60
    
    try:
        result = subprocess.run(
            [sys.executable, script_name], 
            capture_output=True, 
            text=True, 
            check=True,
            timeout=timeout_seconds
        )
        
        elapsed = time.time() - start_time
        
        print(f"✅ {description} completed successfully!")
        print(f"⏱️  Runtime: {elapsed/60:.1f} minutes")
        
        # Show last part of output
        if result.stdout:
            print(f"\n📋 Output (last 500 chars):")
            print(result.stdout[-500:])
        
        return True
        
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"⏰ {description} timed out after {elapsed/60:.1f} minutes")
        return False
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"❌ {description} failed after {elapsed/60:.1f} minutes")
        print(f"Error: {e.stderr}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        return False
    
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"💥 {description} crashed after {elapsed/60:.1f} minutes: {e}")
        return False


def main():
    """
    Run the complete periodic potential analysis workflow.
    """
    
    print("🎯 PERIODIC POTENTIAL ANALYSIS WORKFLOW")
    print("=" * 60)
    print("This will run:")
    print("1. 📊 Batch Simulations (parameter sweeps)")
    print("2. 🔬 Postprocessing (empirical measure analysis)")  
    print("3. 📈 Comprehensive Plotting (trajectories, potentials, histograms)")
    print("=" * 60)
    
    workflow_start = time.time()
    
    # Define workflow steps
    workflow_steps = [
        {
            'script': 'periodic_batch_runner.py',
            'description': 'Batch Simulations',
            'timeout': 60,  # 1 hour max
            'required': True
        },
        {
            'script': 'periodic_postprocess_new.py', 
            'description': 'Postprocessing Analysis',
            'timeout': 30,  # 30 minutes max
            'required': False  # Can proceed without postprocessing
        },
        {
            'script': 'periodic_plots_new.py',
            'description': 'Comprehensive Plotting', 
            'timeout': 20,  # 20 minutes max
            'required': False  # Can proceed without plotting
        }
    ]
    
    # Execute workflow steps
    completed_steps = 0
    failed_steps = []
    
    for step in workflow_steps:
        step_success = run_step(step['script'], step['description'], step['timeout'])
        
        if step_success:
            completed_steps += 1
        else:
            failed_steps.append(step['description'])
            
            if step['required']:
                print(f"\n💥 WORKFLOW FAILED: Required step '{step['description']}' failed")
                print("Cannot continue without completing required steps.")
                return False
            else:
                print(f"\n⚠️  Optional step '{step['description']}' failed, but continuing...")
    
    # Workflow summary
    total_elapsed = time.time() - workflow_start
    
    print(f"\n{'='*70}")
    print(f"🎉 WORKFLOW COMPLETED!")
    print(f"{'='*70}")
    print(f"⏱️  Total time: {total_elapsed/60:.1f} minutes")
    print(f"✅ Completed steps: {completed_steps}/{len(workflow_steps)}")
    
    if failed_steps:
        print(f"❌ Failed steps: {', '.join(failed_steps)}")
    
    print(f"\n📁 Results locations:")
    print(f"   📊 Simulation data: periodic/ directory")  
    print(f"   📈 Plots and figures: periodic/*/figures/ directories")
    print(f"{'='*70}")
    
    return completed_steps > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)