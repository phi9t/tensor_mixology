"""Script to automatically format code using black, isort, and ruff."""

import os
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

# Initialize Rich console
console = Console()


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(description, total=None)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            progress.update(
                task, description=f"✅ {description} completed successfully"
            )

            if result.stdout:
                console.print(
                    Panel(result.stdout, title="Output", border_style="green")
                )
            return True

        except subprocess.CalledProcessError as e:
            progress.update(task, description=f"❌ {description} failed")

            error_table = Table(
                title="Error Details", show_header=True, header_style="red"
            )
            error_table.add_column("Type", style="cyan")
            error_table.add_column("Content", style="white")

            if e.stdout:
                error_table.add_row("STDOUT", e.stdout)
            if e.stderr:
                error_table.add_row("STDERR", e.stderr)

            console.print(error_table)
            return False


def get_python_files() -> list[str]:
    """Get all Python files in the project."""
    python_files = []
    for root, _dirs, files in os.walk("."):
        # Skip virtual environment and other directories
        if any(
            skip in root for skip in [".venv", "__pycache__", ".git", ".pytest_cache"]
        ):
            continue

        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))

    return python_files


def format_with_black(files: list[str]) -> bool:
    """Format code using black."""
    if not files:
        console.print("⚠️  No Python files found to format", style="yellow")
        return True

    cmd = ["black", "--line-length=88", "--target-version=py311"] + files
    return run_command(cmd, "Formatting code with black")


def sort_imports_with_isort(files: list[str]) -> bool:
    """Sort imports using isort."""
    if not files:
        console.print("⚠️  No Python files found to sort imports", style="yellow")
        return True

    cmd = ["isort", "--profile=black", "--line-length=88"] + files
    return run_command(cmd, "Sorting imports with isort")


def lint_with_ruff(files: list[str]) -> bool:
    """Lint code using ruff."""
    if not files:
        console.print("⚠️  No Python files found to lint", style="yellow")
        return True

    cmd = ["ruff", "check", "--fix"] + files
    return run_command(cmd, "Linting code with ruff")


def check_ruff_only(files: list[str]) -> bool:
    """Check code with ruff (without fixing)."""
    if not files:
        console.print("⚠️  No Python files found to check", style="yellow")
        return True

    cmd = ["ruff", "check"] + files
    return run_command(cmd, "Checking code with ruff")


def display_file_list(files: list[str]):
    """Display the list of Python files in a table."""
    table = Table(
        title="Python Files Found", show_header=True, header_style="bold magenta"
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("File Path", style="cyan")
    table.add_column("Size", style="green")

    for i, file in enumerate(files, 1):
        try:
            size = Path(file).stat().st_size
            size_str = f"{size:,} bytes"
        except OSError:
            size_str = "Unknown"

        table.add_row(str(i), file, size_str)

    console.print(table)


def display_summary(success: bool, total_files: int):
    """Display a summary of the formatting results."""
    if success:
        summary_text = Text()
        summary_text.append("🎉 ", style="bold green")
        summary_text.append(
            "All formatting and linting completed successfully!\n", style="green"
        )
        summary_text.append("✨ ", style="bold blue")
        summary_text.append(
            "Your code is now properly formatted and linted.\n", style="blue"
        )
        summary_text.append("📁 ", style="bold cyan")
        summary_text.append(f"Processed {total_files} Python files.", style="cyan")

        console.print(Panel(summary_text, title="Success", border_style="green"))
    else:
        summary_text = Text()
        summary_text.append("⚠️ ", style="bold yellow")
        summary_text.append(
            "Some formatting or linting steps failed.\n", style="yellow"
        )
        summary_text.append("🔧 ", style="bold red")
        summary_text.append("Please check the output above for details.", style="red")

        console.print(Panel(summary_text, title="Issues Found", border_style="red"))


def main():
    """Main function to run all formatting tools."""
    # Header
    header_text = Text()
    header_text.append("🚀 ", style="bold blue")
    header_text.append("Starting code formatting and linting...", style="blue")

    console.print(Panel(header_text, title="Code Formatter", border_style="blue"))
    console.print()

    # Get all Python files
    python_files = get_python_files()

    if not python_files:
        console.print("❌ No Python files found in the project!", style="red")
        sys.exit(1)

    # Display file list
    display_file_list(python_files)
    console.print()

    # Run formatting tools
    success = True
    tools = [
        ("black", format_with_black),
        ("isort", sort_imports_with_isort),
        ("ruff (fix)", lint_with_ruff),
        ("ruff (check)", check_ruff_only),
    ]

    for tool_name, tool_func in tools:
        console.print(f"[bold cyan]Step:[/bold cyan] Running {tool_name}")
        if not tool_func(python_files):
            success = False
        console.print()

    # Display summary
    display_summary(success, len(python_files))

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
