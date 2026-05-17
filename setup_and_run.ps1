# Create a virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Upgrade pip (optional but recommended)
python -m pip install --upgrade pip

# Install the required dependencies
pip install -r requirements.txt

# Run the application
python src/run.py
