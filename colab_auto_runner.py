import os
import sys

# Add colab_training directory to Python import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "colab_training"))

from colab_auto_runner import main

if __name__ == "__main__":
    main()
