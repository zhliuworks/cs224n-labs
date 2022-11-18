import torch
from parser_model import ParserModel
from utils.parser_utils import load_and_preprocess_data

weight_path = 'results/20221111_225415/model.weights'
parser, embeddings, _, _, test_data = load_and_preprocess_data(False)
model = ParserModel(embeddings)
parser.model = model

print(80 * "=")
print("TESTING")
print(80 * "=")
print("Restoring the best model weights found on the dev set")
parser.model.load_state_dict(torch.load(weight_path))
print("Final evaluation on test set",)
parser.model.eval()
UAS, dependencies = parser.parse(test_data)
print("- test UAS: {:.2f}".format(UAS * 100.0))
print("Done!")
