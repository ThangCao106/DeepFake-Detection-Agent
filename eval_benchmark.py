import os
import sys

# Add benchmark, web_ui, and colab_training to Python import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "benchmark"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "web_ui"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "colab_training"))

from eval_benchmark import run_benchmark

if __name__ == "__main__":
    run_benchmark()
