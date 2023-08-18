import json
import os
import pickle

import torch
from flask import Flask, request, jsonify
import transformers
import numpy as np

app = Flask(__name__, static_url_path="")

device = "cpu"
model_name = "cointegrated/rubert-tiny2"

bert = transformers.AutoModel.from_pretrained(model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)


with open(os.path.join('models', 'baseline.pkl'), 'rb') as fp:
    model = pickle.load(fp)


with open('labels.json', 'r') as f:
    labels_raw = json.loads(f.read())
    labels = {int(index): value for index, value in enumerate(labels_raw)}


def embed_bert_cls(text, model, tokenizer, max_length=128):
    t = tokenizer(text, padding=True, truncation=True,
                  max_length=max_length, return_tensors='pt')
    t = {k: v.to(model.device) for k, v in t.items()}

    with torch.no_grad():
        model_output = model(**t)

    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    return {'cls': embeddings[0].cpu().numpy()}


@app.route("/predict", methods=['POST'])
def predict():
    data = request.get_json(force=True)
    token = np.array([embed_bert_cls(data['data'], bert, tokenizer,
                                     max_length=512)['cls']])
    result = int(model.predict(token)[0])
    label = labels[result]

    return jsonify({
        "label": label
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
