import json
import os

doc_folder = './doc_folder/'
if not os.path.exists(doc_folder):
    os.makedirs(doc_folder)

tasks = json.load(open('./ecb_test_set.json'))

for task in tasks:
    with open(f'./{doc_folder}/{task["mention_id"]}.html', 'w') as thtml:
        thtml.write('<p>')
        thtml.write(task['marked_doc'])
        thtml.write('</p>')