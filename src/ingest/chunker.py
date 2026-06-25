import json, os
files = os.listdir('data/processed')
print('Files found:', files)
with open(f'data/processed/{files[0]}') as f:
    doc = json.load(f)
print('Keys:', list(doc.keys()))
print('Word count:', len(doc['text'].split()))
