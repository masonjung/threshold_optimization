import sys
import os
print(sys.path)

# Manually specify the path to the project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src import utils






# Example usage
result = utils.calculate_metrics([1, 0, 1], [1, 0, 0])
print(result)