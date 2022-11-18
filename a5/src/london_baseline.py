# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

from utils import evaluate_places

predictions = ['London'] * 500
_, correct = evaluate_places('../birth_dev.tsv', predictions)
print(f'acc: {correct/500*100}%')
