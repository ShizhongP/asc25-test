import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", trust_remote_code=True)
text = (
    "From Monday to Friday most people are busy working or studying, "
    "but in the evenings and weekends they are free and _ themselves."
)
tokenized_text = tokenizer.tokenize(text)

masked_index = tokenized_text.index("_")
tokenized_text[masked_index] = "[MASK]"

options = ["love", "work", "enjoy", "play"]
options_ids = tokenizer.convert_tokens_to_ids(options)

indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

segments_ids = [0] * len(tokenized_text)

tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

model = BertForMaskedLM.from_pretrained("bert-base-uncased")
model.eval()

predictions = model(tokens_tensor, segments_tensors)
# print(predictions)
predictions = predictions.logits
predictions_candidates = predictions[0, masked_index, options_ids]
answer_idx = torch.argmax(predictions_candidates).item()

print(f'The most likely word is "{options[answer_idx]}".')
