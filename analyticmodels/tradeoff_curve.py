import os
import re
import matplotlib.pyplot as plt

# Directory containing the result files
directory = "C:\\Users\\minse\\threshold_optimization\\results\\2025_01_26"

# Regular expressions
filename_regex = re.compile(r"disparity_(\d+_\d+).txt")
detector_regex = re.compile(r"Performance for Source: .*?, Detector: (.*?)\n")
accuracy_regex = re.compile(r"Accuracy: (\d+\.\d+)")

# Target detector to focus on
target_detector = "roberta_base_openai_detector_probability"

relaxation_values = []
accuracies = []

# Extract the data
for file in os.listdir(directory):
    if file.endswith(".txt"):
        filepath = os.path.join(directory, file)
        match = filename_regex.search(file)

        if match:
            relaxation = float(match.group(1).replace("_", "."))
            with open(filepath, "r") as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    detector_match = detector_regex.search(line)
                    if detector_match and detector_match.group(1) == target_detector:
                        for j in range(i + 1, len(lines)):
                            accuracy_match = accuracy_regex.search(lines[j])
                            if accuracy_match:
                                accuracy = float(accuracy_match.group(1))
                                relaxation_values.append(relaxation)
                                accuracies.append(accuracy)
                                break
                        break

# Sort by relaxation (ascending)
sorted_data = sorted(zip(relaxation_values, accuracies))
relaxation_values, accuracies = zip(*sorted_data)

# Define x-ticks WITHOUT 0.2 included
xtick_values = [1.0, 0.5, 0.3, 0.15, 0.1]  # Notice 0.2 is removed here
xtick_labels = [f"{x:.2f}" for x in xtick_values]

# Create figure
plt.figure(figsize=(10, 6))

# Plot the main accuracy-relaxation curve
plt.plot(relaxation_values, accuracies, marker="o", linestyle="-", color="blue", label="FairOPT")

# Plot the additional scatter points at relaxation=1.0
plt.scatter([1.0], [0.3702], color="gray", label="Static")
plt.scatter([1.0], [0.3925], color="green", label="ROCFPR")

# Draw a dashed red vertical line at x=0.2
# (Even though 0.2 is not in xtick_values, we still show the line.)
plt.axvline(
    x=0.2,
    color="red",
    linestyle="--",
    label="Rule-based fairness",
)

# Set font size on y-ticks
plt.yticks(fontsize=16)
# Use custom x-ticks and rotate
plt.xticks(xtick_values, xtick_labels, rotation=45, fontsize=16)
plt.title("Relaxed Fairness on the Accuracy-Relaxation Curve", fontsize=25)
plt.xlabel("Fairness relaxation", fontsize=20)
plt.gca().invert_xaxis()  # Invert x-axis to start from 1.0
plt.ylabel("Accuracy", fontsize=20)

# No grid lines
plt.grid(visible=False)

# Legend
plt.legend(loc='center left', fontsize=16)

# Make layout fit nicely
plt.tight_layout()
plt.show()
