import os
import shutil
import glob
from pathlib import Path
import sys

def safe_remove(target_path, expected_working_dir=None):
    """
    Safely remove a file or directory.
    - Resolves glob patterns
    - Checks for '..' in path
    - Prevents deleting root '/'
    - Ensures target is within expected_working_dir
    """
    if expected_working_dir is None:
        expected_working_dir = os.getcwd()
    
    expected_working_dir = os.path.abspath(expected_working_dir)
    target_path_str = str(target_path)
    
    # Check if original path contains '..'
    if '..' in target_path_str:
        print(f"Warning: path {target_path_str} contains '..'. Deletion aborted.")
        return

    # Handle globs
    matched_paths = glob.glob(target_path_str)
    if not matched_paths and not '*' in target_path_str and not '?' in target_path_str:
        matched_paths = [target_path_str]

    for p in matched_paths:
        abs_p = os.path.abspath(p)
        
        # Security checks
        if abs_p == '/':
            print("Warning: attempting to delete root directory. Deletion aborted.")
            continue
            
        if not abs_p.startswith(expected_working_dir):
            print(f"Warning: path {abs_p} is not within expected working directory {expected_working_dir}. Deletion aborted.")
            continue
            
        path_obj = Path(abs_p)
        if path_obj.exists() or path_obj.is_symlink():
            if path_obj.is_dir() and not path_obj.is_symlink():
                shutil.rmtree(abs_p)
            else:
                path_obj.unlink()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python safe_rm.py <path_to_remove> [expected_working_dir]")
        sys.exit(1)
        
    target = sys.argv[1]
    expected_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(target))
        
    safe_remove(target, expected_dir)
