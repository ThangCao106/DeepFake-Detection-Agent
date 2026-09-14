import os
import sys

# Add 2_colab_training directory to Python import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "2_colab_training"))

from colab_auto_runner import main

if __name__ == "__main__":
    main()
