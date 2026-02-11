#!/usr/bin/env python3
"""
Version bump script for Agent Orchestra
Automatically updates version numbers across all relevant files
"""
import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import subprocess
import json
from datetime import datetime


class VersionBumper:
    """Handles version bumping for Agent Orchestra"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.files_to_update = [
            ("agent_orchestra/__init__.py", r'__version__ = "([^"]+)"'),
            ("pyproject.toml", r'version = "([^"]+)"'),
            ("docker-compose.yml", r'image: agent-orchestra:([^"\s]+)'),
        ]
    
    def get_current_version(self) -> str:
        """Get current version from __init__.py"""
        init_file = self.project_root / "agent_orchestra" / "__init__.py"
        content = init_file.read_text()
        match = re.search(r'__version__ = "([^"]+)"', content)
        if not match:
            raise ValueError("Could not find version in __init__.py")
        return match.group(1)
    
    def parse_version(self, version: str) -> Tuple[int, int, int]:
        """Parse semantic version string into tuple"""
        parts = version.split(".")
        if len(parts) != 3:
            raise ValueError(f"Invalid version format: {version}")
        return tuple(int(part) for part in parts)
    
    def format_version(self, major: int, minor: int, patch: int) -> str:
        """Format version tuple into string"""
        return f"{major}.{minor}.{patch}"
    
    def bump_version(self, current: str, bump_type: str) -> str:
        """Bump version based on type (major, minor, patch)"""
        major, minor, patch = self.parse_version(current)
        
        if bump_type == "major":
            major += 1
            minor = 0
            patch = 0
        elif bump_type == "minor":
            minor += 1
            patch = 0
        elif bump_type == "patch":
            patch += 1
        else:
            raise ValueError(f"Invalid bump type: {bump_type}")
        
        return self.format_version(major, minor, patch)
    
    def update_files(self, new_version: str) -> List[Path]:
        """Update version in all relevant files"""
        updated_files = []
        
        for file_path, pattern in self.files_to_update:
            full_path = self.project_root / file_path
            if not full_path.exists():
                print(f"Warning: {file_path} does not exist, skipping")
                continue
            
            content = full_path.read_text()
            new_content = re.sub(
                pattern,
                lambda m: m.group(0).replace(m.group(1), new_version),
                content
            )
            
            if new_content != content:
                full_path.write_text(new_content)
                updated_files.append(full_path)
                print(f"✅ Updated {file_path}")
            else:
                print(f"⚠️  No changes needed in {file_path}")
        
        return updated_files
    
    def update_changelog(self, new_version: str) -> None:
        """Update CHANGELOG.md with new version entry"""
        changelog_path = self.project_root / "CHANGELOG.md"
        
        if not changelog_path.exists():
            print("Warning: CHANGELOG.md does not exist, creating one")
            changelog_path.write_text("# Changelog\n\n")
        
        content = changelog_path.read_text()
        
        # Create new entry
        today = datetime.now().strftime("%Y-%m-%d")
        new_entry = f"""## [{new_version}] - {today}

### Added
- 

### Changed
- 

### Fixed
- 

### Removed
- 

"""
        
        # Insert after main header
        lines = content.split('\n')
        insert_index = 0
        for i, line in enumerate(lines):
            if line.startswith('# '):
                insert_index = i + 2  # After header and blank line
                break
        
        lines.insert(insert_index, new_entry)
        changelog_path.write_text('\n'.join(lines))
        print(f"✅ Updated CHANGELOG.md with {new_version} entry")
    
    def create_git_tag(self, version: str, dry_run: bool = False) -> None:
        """Create git tag for the new version"""
        tag_name = f"v{version}"
        
        if dry_run:
            print(f"🔍 Would create git tag: {tag_name}")
            return
        
        try:
            # Check if tag already exists
            result = subprocess.run(
                ["git", "tag", "-l", tag_name],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.stdout.strip():
                print(f"⚠️  Git tag {tag_name} already exists")
                return
            
            # Create annotated tag
            subprocess.run(
                ["git", "tag", "-a", tag_name, "-m", f"Release {version}"],
                check=True,
                cwd=self.project_root
            )
            print(f"✅ Created git tag: {tag_name}")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create git tag: {e}")
    
    def commit_changes(self, version: str, files: List[Path], dry_run: bool = False) -> None:
        """Commit version bump changes"""
        if not files:
            print("No files to commit")
            return
        
        commit_message = f"chore: bump version to {version}"
        
        if dry_run:
            print(f"🔍 Would commit changes: {commit_message}")
            print(f"🔍 Files to commit: {[f.name for f in files]}")
            return
        
        try:
            # Add files
            for file_path in files:
                subprocess.run(
                    ["git", "add", str(file_path)],
                    check=True,
                    cwd=self.project_root
                )
            
            # Add changelog
            changelog_path = self.project_root / "CHANGELOG.md"
            if changelog_path.exists():
                subprocess.run(
                    ["git", "add", str(changelog_path)],
                    check=True,
                    cwd=self.project_root
                )
            
            # Commit
            subprocess.run(
                ["git", "commit", "-m", commit_message],
                check=True,
                cwd=self.project_root
            )
            print(f"✅ Committed changes: {commit_message}")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to commit changes: {e}")
    
    def check_git_status(self) -> bool:
        """Check if git working directory is clean"""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.stdout.strip():
                print("❌ Git working directory is not clean. Please commit or stash changes first.")
                print("Uncommitted changes:")
                print(result.stdout)
                return False
            
            return True
            
        except subprocess.CalledProcessError:
            print("❌ Failed to check git status")
            return False
    
    def validate_version(self, version: str) -> bool:
        """Validate version format"""
        try:
            self.parse_version(version)
            return True
        except ValueError:
            return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Bump version for Agent Orchestra",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/bump_version.py patch          # 1.0.0 -> 1.0.1
  python scripts/bump_version.py minor          # 1.0.0 -> 1.1.0
  python scripts/bump_version.py major          # 1.0.0 -> 2.0.0
  python scripts/bump_version.py --version 2.0.0  # Set specific version
  python scripts/bump_version.py patch --dry-run   # Preview changes
        """
    )
    
    parser.add_argument(
        "bump_type",
        choices=["major", "minor", "patch"],
        nargs="?",
        help="Type of version bump"
    )
    
    parser.add_argument(
        "--version",
        help="Set specific version (overrides bump_type)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without making them"
    )
    
    parser.add_argument(
        "--no-commit",
        action="store_true",
        help="Don't create git commit"
    )
    
    parser.add_argument(
        "--no-tag",
        action="store_true", 
        help="Don't create git tag"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip git status check"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.version and not args.bump_type:
        parser.error("Must specify either bump_type or --version")
    
    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    bumper = VersionBumper(project_root)
    
    # Check git status unless forced
    if not args.force and not args.dry_run:
        if not bumper.check_git_status():
            sys.exit(1)
    
    try:
        # Get current version
        current_version = bumper.get_current_version()
        print(f"📦 Current version: {current_version}")
        
        # Calculate new version
        if args.version:
            if not bumper.validate_version(args.version):
                print(f"❌ Invalid version format: {args.version}")
                sys.exit(1)
            new_version = args.version
        else:
            new_version = bumper.bump_version(current_version, args.bump_type)
        
        print(f"🚀 New version: {new_version}")
        
        if args.dry_run:
            print("🔍 Dry run mode - no changes will be made")
        
        # Update files
        updated_files = []
        if not args.dry_run:
            updated_files = bumper.update_files(new_version)
            bumper.update_changelog(new_version)
        else:
            print("🔍 Would update:")
            for file_path, _ in bumper.files_to_update:
                full_path = project_root / file_path
                if full_path.exists():
                    print(f"  - {file_path}")
        
        # Git operations
        if not args.no_commit:
            bumper.commit_changes(new_version, updated_files, args.dry_run)
        
        if not args.no_tag:
            bumper.create_git_tag(new_version, args.dry_run)
        
        print(f"🎉 Version bump {'simulated' if args.dry_run else 'completed'}: {current_version} -> {new_version}")
        
        if not args.dry_run:
            print("\n📝 Next steps:")
            print("  1. Review and edit CHANGELOG.md entry")
            print("  2. Run tests: make test")
            print("  3. Push changes: git push && git push --tags")
            print("  4. Create GitHub release")
            print("  5. Deploy to PyPI: make release")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()