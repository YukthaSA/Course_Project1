"""
Automated setup script - creates all project files and directories
Run this once: python setup.py
"""
import os
import sys

def create_file(path, content):
    """Create a file with content"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ Created: {path}")

def create_folders():
    """Create all required folders"""
    folders = ['agents', 'engine', 'tests', 'samples', 'logs', 'data']
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    print("✅ All project folders created")

if __name__ == "__main__":
    print("Initializing Agentic QA Framework project structure...")
    print()
    create_folders()
    print()
    print("=" * 60)
    print("✅ PROJECT STRUCTURE CREATED SUCCESSFULLY!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. In terminal: pip install -r requirements.txt")
    print("2. In terminal: playwright install chromium firefox")
    print("3. In terminal: python test_setup.py")
    print("4. In terminal: python test_pipeline.py")
    print("5. In terminal: streamlit run app.py")
