import os
import sys

# Add 4_benchmark, 1_web_ui, and 2_colab_training to Python import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "4_benchmark"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "1_web_ui"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "2_colab_training"))

from eval_benchmark import run_eval

if __name__ == "__main__":
    run_eval()
