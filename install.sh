# Change current directory into project root
original_dir=$(pwd)
script_dir=$(dirname "$0")
cd "$script_dir"

# Remove old dist file, build, and install
rm -rf dist
python setup.py bdist_wheel
pip install dist/*.whl

# Open users' original directory
cd "$original_dir"
