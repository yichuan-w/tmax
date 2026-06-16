#!/usr/bin/env python3
"""
File Access Date Management Tool

This tool helps you manage files on purge-based filesystems that automatically
delete files based on access dates (typically after 60 days of no access).

QUICK START
===========

Option 1: View Directory Tree with Access Dates
-----------------------------------------------
See which directories are at risk of being purged:

    # Basic scan (display depth 3, scans all files in each directory)
    python3 manage_access_dates.py --scan

    # Custom display depth
    python3 manage_access_dates.py --scan --depth 4

    # Limit scan depth for faster performance (only scans 2 levels deep)
    python3 manage_access_dates.py --scan --scan-depth 2

    # Scan a different directory
    python3 manage_access_dates.py --scan /path/to/directory

Output includes:
- Tree-like directory structure (up to specified depth)
- Last access date for each directory
- Age in days with color coding:
  * Green: Safe (< 40 days)
  * Yellow: Caution (40-49 days)
  * Red: Warning (50-59 days)
  * Red Bold: CRITICAL (≥ 60 days, at risk!)


Option 2: Update All Access Dates
----------------------------------
Prevent your files from being purged by updating their access dates:

    # Dry run first (recommended - see what would happen)
    python3 manage_access_dates.py --update --dry-run

    # Actually update all access dates
    python3 manage_access_dates.py --update

Warning: The update operation will touch ALL files in the directory tree.
Always run with --dry-run first!


COMMAND LINE OPTIONS
====================

positional arguments:
  path                  Directory path to analyze
                        (default: /gpfs/scrubbed/osey/Dataset_Distillation)

required (choose one):
  --scan               Scan and display directory tree with access dates
  --update             Update access times for all files

optional arguments:
  --depth DEPTH        Maximum directory depth for displaying in tree
                       (default: 3)
  --scan-depth DEPTH   Maximum depth to scan when finding oldest access time
                       within each directory. If not set, scans all files
                       recursively (unlimited). Use this to speed up scanning
                       for very deep directory structures.
  --dry-run           Perform a dry run without making changes (for --update)
  --warning-days DAYS Number of days before warning (default: 60)
  -h, --help          Show help message


HOW IT WORKS
============

Scan Mode (--scan):
-------------------
1. Recursively walks through the directory tree up to the specified depth
2. For each directory, finds the oldest access time among all files
   (by default scans all files recursively, or limited by --scan-depth)
3. Calculates how many days since last access
4. Displays results in a tree format with color-coded warnings

Note: --depth controls which directories are SHOWN in the tree.
      By default, ALL files are scanned within each displayed directory.
      Use --scan-depth to limit how deep files are scanned (faster but may
      miss oldest files).

Update Mode (--update):
-----------------------
1. Walks through ALL files and directories recursively (no depth limit)
2. Updates the access time to the current time using os.utime()
3. Confirms before proceeding (unless --dry-run)
4. Provides progress updates and summary


IMPORTANT NOTES
===============

- Access Time vs Modification Time: This tool focuses on access time (atime),
  which is what purge systems typically use
- Performance: Scanning large directory trees can take time. Be patient!
- Permissions: You need read access to scan, write access to update
- Dry Run: Always test with --dry-run before updating
- Regular Updates: Consider setting up a cron job to update access times
  periodically


AUTOMATION EXAMPLE
==================

To automatically update access dates weekly, add to your crontab:

    crontab -e

    # Add this line (runs every Sunday at 2 AM)
    0 2 * * 0 python3 /gpfs/scrubbed/osey/Dataset_Distillation/manage_access_dates.py --update


TROUBLESHOOTING
===============

"Permission denied" errors:
    You don't have access to some files. The script will continue with
    files you can access.

Script runs slowly:
    Large directory trees take time. Consider:
    - Reducing the depth level for scans
    - Using --scan-depth to limit file scanning depth
    - Running during off-peak hours for updates

No color output:
    Your terminal might not support ANSI colors. Output is still readable
    in plain text.
"""

import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
import argparse
from collections import defaultdict

# ANSI color codes for highlighting
class Colors:
    RED = '\033[91m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def get_oldest_access_time(directory, max_scan_depth=None):
    """
    Recursively find the oldest access time in a directory.
    
    Args:
        directory: Directory path to scan
        max_scan_depth: Maximum depth to scan (None = unlimited, 0 = only the directory itself)
    
    Returns the oldest access time (timestamp) and the file path.
    """
    oldest_time = float('inf')
    oldest_path = None
    base_path = Path(directory).resolve()
    
    try:
        for root, dirs, files in os.walk(directory):
            # Calculate depth relative to base directory
            if max_scan_depth is not None:
                try:
                    rel_path = Path(root).relative_to(base_path)
                    depth = len(rel_path.parts)
                    if depth > max_scan_depth:
                        # Skip this directory and its subdirectories
                        dirs[:] = []  # Don't recurse into subdirectories
                        continue
                except ValueError:
                    # Path not relative to base, skip
                    continue
            
            # Check directory itself
            try:
                dir_atime = os.stat(root).st_atime
                if dir_atime < oldest_time:
                    oldest_time = dir_atime
                    oldest_path = root
            except (OSError, PermissionError):
                pass
            
            # Check all files
            for file in files:
                try:
                    file_path = os.path.join(root, file)
                    file_atime = os.stat(file_path).st_atime
                    if file_atime < oldest_time:
                        oldest_time = file_atime
                        oldest_path = file_path
                except (OSError, PermissionError):
                    pass
    except (OSError, PermissionError) as e:
        print(f"Warning: Cannot access {directory}: {e}", file=sys.stderr)
    
    return oldest_time if oldest_time != float('inf') else None, oldest_path

def format_age_days(days):
    """Format the age in days with appropriate color coding."""
    if days >= 60:
        return f"{Colors.RED}{Colors.BOLD}{days:.1f} days (CRITICAL - AT RISK!){Colors.END}"
    elif days >= 50:
        return f"{Colors.RED}{days:.1f} days (WARNING - approaching limit){Colors.END}"
    elif days >= 40:
        return f"{Colors.YELLOW}{days:.1f} days (caution){Colors.END}"
    else:
        return f"{Colors.GREEN}{days:.1f} days{Colors.END}"

def get_directory_tree(base_path, max_depth=3, max_scan_depth=None):
    """
    Get directory tree up to max_depth levels.
    
    Args:
        base_path: Base directory to scan
        max_depth: Maximum depth of directories to display in tree
        max_scan_depth: Maximum depth to scan when finding oldest access time
                       (None = unlimited, scans all files in each directory)
    
    Returns a dictionary with directory paths and their metadata.
    """
    base_path = Path(base_path).resolve()
    tree_data = {}
    
    def scan_directory(path, current_depth=0):
        if current_depth > max_depth:
            return
        
        try:
            entries = sorted(path.iterdir(), key=lambda x: x.name)
            for entry in entries:
                if entry.is_dir():
                    rel_path = entry.relative_to(base_path)
                    depth = len(rel_path.parts)
                    
                    if depth <= max_depth:
                        # Calculate scan depth relative to this directory
                        # If max_scan_depth is set, limit how deep we scan within this directory
                        scan_depth = None
                        if max_scan_depth is not None:
                            # max_scan_depth is relative to base_path, so we need to calculate
                            # how many more levels we can scan from this directory
                            remaining_depth = max_scan_depth - depth
                            if remaining_depth >= 0:
                                scan_depth = remaining_depth
                            else:
                                scan_depth = 0  # Only scan the directory itself
                        
                        oldest_time, oldest_file = get_oldest_access_time(str(entry), scan_depth)
                        tree_data[str(entry)] = {
                            'depth': depth,
                            'name': entry.name,
                            'oldest_atime': oldest_time,
                            'oldest_file': oldest_file,
                            'rel_path': str(rel_path)
                        }
                        
                        if depth < max_depth:
                            scan_directory(entry, current_depth + 1)
        except PermissionError:
            print(f"Warning: Permission denied for {path}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Error scanning {path}: {e}", file=sys.stderr)
    
    # Start scanning
    scan_msg = f"Scanning directory structure (display depth: {max_depth}"
    if max_scan_depth is not None:
        scan_msg += f", scan depth: {max_scan_depth}"
    else:
        scan_msg += ", scan depth: unlimited"
    scan_msg += ")..."
    print(scan_msg, file=sys.stderr)
    scan_directory(base_path, 0)
    
    return tree_data

def display_tree(base_path, max_depth=3, warning_days=60, max_scan_depth=None):
    """
    Display directory tree with access date information.
    
    Args:
        base_path: Base directory to scan
        max_depth: Maximum depth of directories to display
        warning_days: Number of days before warning
        max_scan_depth: Maximum depth to scan when finding oldest access time
                       (None = unlimited, scans all files in each directory)
    """
    print(f"\n{Colors.BOLD}Directory Tree Analysis{Colors.END}")
    print(f"Base path: {base_path}")
    print(f"Display depth: {max_depth}")
    if max_scan_depth is not None:
        print(f"Scan depth: {max_scan_depth} (limited - may not find true oldest file)")
    else:
        print(f"Scan depth: unlimited (scans all files in each directory)")
    print(f"Warning threshold: {warning_days} days")
    print("=" * 80)
    
    tree_data = get_directory_tree(base_path, max_depth, max_scan_depth)
    
    if not tree_data:
        print("No directories found or unable to access the path.")
        return
    
    # Sort by relative path for tree-like display
    sorted_paths = sorted(tree_data.items(), key=lambda x: x[1]['rel_path'])
    
    current_time = time.time()
    at_risk_count = 0
    warning_count = 0
    
    print(f"\n{Colors.BOLD}Directory Structure:{Colors.END}\n")
    
    for dir_path, info in sorted_paths:
        depth = info['depth']
        name = info['name']
        oldest_time = info['oldest_atime']
        oldest_file = info['oldest_file']
        
        # Create tree-like indentation
        indent = "  " * (depth - 1)
        prefix = "├── " if depth > 0 else ""
        
        if oldest_time:
            age_seconds = current_time - oldest_time
            age_days = age_seconds / (24 * 3600)
            date_str = datetime.fromtimestamp(oldest_time).strftime('%Y-%m-%d %H:%M:%S')
            age_str = format_age_days(age_days)
            
            if age_days >= 60:
                at_risk_count += 1
            elif age_days >= 50:
                warning_count += 1
            
            print(f"{indent}{prefix}{Colors.BLUE}{name}/{Colors.END}")
            print(f"{indent}    Last accessed: {date_str} ({age_str})")
            if oldest_file:
                print(f"{indent}    Oldest file: {oldest_file}")
            print()
        else:
            print(f"{indent}{prefix}{Colors.BLUE}{name}/{Colors.END}")
            print(f"{indent}    Unable to determine access time")
            print()
    
    # Summary
    print("=" * 80)
    print(f"\n{Colors.BOLD}Summary:{Colors.END}")
    print(f"Total directories scanned: {len(tree_data)}")
    if at_risk_count > 0:
        print(f"{Colors.RED}{Colors.BOLD}CRITICAL: {at_risk_count} directories at risk of purging (>=60 days){Colors.END}")
    if warning_count > 0:
        print(f"{Colors.YELLOW}WARNING: {warning_count} directories approaching limit (>=50 days){Colors.END}")
    
    if at_risk_count == 0 and warning_count == 0:
        print(f"{Colors.GREEN}All directories are safe from purging{Colors.END}")
    
    print()

def update_access_times(base_path, dry_run=False):
    """
    Update access times for all files and directories.
    """
    print(f"\n{Colors.BOLD}Updating Access Times{Colors.END}")
    print(f"Base path: {base_path}")
    print(f"Mode: {'DRY RUN (no changes will be made)' if dry_run else 'LIVE (will update access times)'}")
    print("=" * 80)
    
    if not dry_run:
        response = input(f"\n{Colors.YELLOW}This will update access times for ALL files. Continue? (yes/no): {Colors.END}")
        if response.lower() not in ['yes', 'y']:
            print("Operation cancelled.")
            return
    
    file_count = 0
    dir_count = 0
    error_count = 0
    
    print("\nProcessing...\n")
    
    try:
        for root, dirs, files in os.walk(base_path):
            # Update directory access time
            try:
                if not dry_run:
                    os.utime(root, None)  # Update to current time
                dir_count += 1
                if dir_count % 100 == 0:
                    print(f"Processed {dir_count} directories, {file_count} files...", end='\r')
            except (OSError, PermissionError) as e:
                error_count += 1
                if error_count <= 10:  # Only show first 10 errors
                    print(f"Error updating {root}: {e}", file=sys.stderr)
            
            # Update file access times
            for file in files:
                try:
                    file_path = os.path.join(root, file)
                    if not dry_run:
                        os.utime(file_path, None)
                    file_count += 1
                    if file_count % 1000 == 0:
                        print(f"Processed {dir_count} directories, {file_count} files...", end='\r')
                except (OSError, PermissionError) as e:
                    error_count += 1
                    if error_count <= 10:
                        print(f"Error updating {file_path}: {e}", file=sys.stderr)
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Operation interrupted by user{Colors.END}")
    
    print(f"\n\n{Colors.BOLD}Update Complete:{Colors.END}")
    print(f"Directories processed: {dir_count}")
    print(f"Files processed: {file_count}")
    if error_count > 0:
        print(f"{Colors.YELLOW}Errors encountered: {error_count}{Colors.END}")
    
    if dry_run:
        print(f"\n{Colors.YELLOW}This was a dry run. No changes were made.{Colors.END}")
    else:
        print(f"\n{Colors.GREEN}All access times have been updated to prevent purging.{Colors.END}")

def main():
    parser = argparse.ArgumentParser(
        description='Manage file access dates on purge-based filesystems',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View directory tree with access dates (display depth 3, scans all files in each dir)
  %(prog)s --scan /gpfs/scrubbed/osey/Dataset_Distillation
  
  # View with custom display depth
  %(prog)s --scan /gpfs/scrubbed/osey/Dataset_Distillation --depth 4
  
  # Limit scan depth for faster performance (only scans 2 levels deep in each directory)
  %(prog)s --scan /gpfs/scrubbed/osey/Dataset_Distillation --scan-depth 2
  
  # Update all access times (dry run first)
  %(prog)s --update /gpfs/scrubbed/osey/Dataset_Distillation --dry-run
  
  # Update all access times (actual update - scans ALL files regardless of depth)
  %(prog)s --update /gpfs/scrubbed/osey/Dataset_Distillation

Note: --depth controls which directories are SHOWN in the tree.
      By default, ALL files are scanned within each displayed directory.
      Use --scan-depth to limit how deep files are scanned (faster but may miss oldest files).
        """
    )
    
    parser.add_argument(
        'path',
        nargs='?',
        default='/gpfs/scrubbed/osey/tmax',
        help='Directory path to analyze (default: /gpfs/scrubbed/osey/tmax)'
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--scan',
        action='store_true',
        help='Scan and display directory tree with access dates'
    )
    group.add_argument(
        '--update',
        action='store_true',
        help='Update access times for all files to prevent purging'
    )
    
    parser.add_argument(
        '--depth',
        type=int,
        default=3,
        help='Maximum directory depth for displaying in tree structure (default: 3)'
    )
    
    parser.add_argument(
        '--scan-depth',
        type=int,
        default=None,
        help='Maximum depth to scan when finding oldest access time within each directory. '
             'If not set, scans all files recursively (unlimited). '
             'Use this to speed up scanning for very deep directory structures.'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Perform a dry run without making changes (for --update mode)'
    )
    
    parser.add_argument(
        '--warning-days',
        type=int,
        default=60,
        help='Number of days before warning about purge risk (default: 60)'
    )
    
    args = parser.parse_args()
    
    # Validate path
    if not os.path.exists(args.path):
        print(f"Error: Path does not exist: {args.path}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.isdir(args.path):
        print(f"Error: Path is not a directory: {args.path}", file=sys.stderr)
        sys.exit(1)
    
    # Execute requested action
    if args.scan:
        display_tree(args.path, args.depth, args.warning_days, args.scan_depth)
    elif args.update:
        update_access_times(args.path, args.dry_run)

if __name__ == '__main__':
    main()

