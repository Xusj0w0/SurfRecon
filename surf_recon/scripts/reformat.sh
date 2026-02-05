TARGET_DIR="${1:-.}"

# Ensure tools exist
command -v isort >/dev/null 2>&1 || { echo "Error: isort not found in PATH"; exit 1; }
command -v black >/dev/null 2>&1 || { echo "Error: black not found in PATH"; exit 1; }

# Find all .py files recursively (null-delimited to handle spaces)
mapfile -d '' FILES < <(find "$TARGET_DIR" -type f -name "*.py" -print0)
if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "No .py files found under: $TARGET_DIR"
  exit 0
fi

echo "Formatting ${#FILES[@]} Python files under: $TARGET_DIR"
# Run black to reformat files
echo "Running black (line length=140)..."
black --line-length 140 "${FILES[@]}"
# Run isort to reorganize imports
echo "Running isort..."
isort "${FILES[@]}"

echo "Done."
