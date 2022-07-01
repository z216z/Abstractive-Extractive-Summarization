import os
import tarfile
import gdown
from tqdm import tqdm

def _read_json(file_path):
    json_file = open(file_path, "r")
    payload = json.loads(json_file.read())
    json_file.close()
    return payload

def _make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

def make_cnn_dataset(DATASET_PATH, LANGUAGE='English', make_tar=True):
    for split in ['test', 'training', 'validation']:
        path = os.path.join(DATASET_PATH, LANGUAGE, split)
        if not os.path.exists(os.path.join(path, 'annual_reports')):
            os.makedirs(os.path.join(path, 'annual_reports'))
        if not os.path.exists(os.path.join(path, 'gold_summaries')):
            os.makedirs(os.path.join(path, 'gold_summaries'))
        for file_name in tqdm(os.listdir(path)):
            if file_name.endswith('.json'):
                data = _read_json(os.path.join(path, file_name))
                report = data['article']
                headline = data['headline']
                summary = data['abstract']
                with open(os.path.join(path, 'annual_reports', '{}.txt'.format(file_name.split('.')[0])), 'w') as fw:
                    fw.write('\n'.join(report))
                with open(os.path.join(path, 'gold_summaries', '{}_1.txt'.format(file_name.split('.')[0])), 'w') as fw:
                    fw.write('\n'.join(headline))
                with open(os.path.join(path, 'gold_summaries', '{}_2.txt'.format(file_name.split('.')[0])), 'w') as fw:
                    fw.write('\n'.join(summary))
                os.remove(os.path.join(path, file_name))
        if make_tar:
            _make_tarfile(os.path.join(DATASET_PATH, LANGUAGE, '{}.tar.gz'.format(split)), path)

def download_dataset(DATASET_PATH, LANGUAGE, id, output):
    gdown.download(id=id, output=os.path.join(DATASET_PATH, LANGUAGE, output))
