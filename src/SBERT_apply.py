import torch
from sentence_transformers import SentenceTransformer
import tokenizer
import datautility as du
import numpy as np
import sys
import csv
import warnings

class Persistent:
    BERT_TRANSFORMER = SentenceTransformer('bert-base-nli-mean-tokens')
    
def apply_SBERT(text):
    return np.array(Persistent.BERT_TRANSFORMER.encode(text))


def apply_to_dataset(filename, text_column, batch=128, verbose=True):
    encoding = None
    try:
        _ = open(filename, 'r', encoding=encoding)
    except UnicodeEncodeError:
        encoding = 'utf8'

    run_batch = False
    csvarr = []
    n_lines = len(open(filename, 'r', errors='replace', encoding=encoding).readlines())
    with open(filename, 'r', errors='replace', encoding=encoding) as f:
        f_lines = csv.reader(f)
        headers = None
        if verbose:
            output_str = '-- generating embeddings...({}%)'.format(0)
            sys.stdout.write(output_str)
            sys.stdout.flush()
            old_str = output_str
        first_output = True
        i = 0
        for line in f_lines:
            if i == 0 or len(line) == 0:
                headers = np.append(np.array(line),[f'SBERT_{j}' for j in range(768)])
                i += 1
                continue
            elif i % batch == 0:
                run_batch = True

            line = np.array(line)

            if run_batch:
                ar = np.array(csvarr).reshape([-1, len(line)])

                emb = apply_SBERT(ar[:,text_column])
                ar = np.append(ar,emb,axis=1)

                du.write_csv(ar, filename[:-4] + '_sbert.csv', headers, append=not first_output, encoding=encoding)
                first_output = False
                csvarr = []
                run_batch = False


            csvarr.append(line)

            if verbose and not round((i / n_lines) * 100, 2) == round(((i - 1) / n_lines) * 100, 2):
                sys.stdout.write('\r' + (' ' * len(old_str)))
                output_str = '\r-- generating embeddings...({}%)'.format(round((i / n_lines) * 100, 2))
                sys.stdout.write(output_str)
                sys.stdout.flush()
                old_str = output_str

            i += 1

        if len(csvarr) > 0:
            ar = np.array(csvarr).reshape([-1, len(line)])
            emb = apply_SBERT(ar[:, text_column])
            ar = np.append(ar, emb, axis=1)
            du.write_csv(ar, filename[:-4] + '_sbert.csv', headers, append=not first_output, encoding=encoding)

        if verbose:
            sys.stdout.write('\r' + (' ' * len(old_str)))
            sys.stdout.write('\r-- generating embeddings...({}%)\n'.format(100))
            sys.stdout.flush()

if __name__ == "__main__":
    filename = 'C:/Users/Anthony/Downloads/reviews.csv'
    # filename = 'C:/Users/Anthony/Downloads/all_data_with_identities.csv'
    data, headers = du.read_csv(filename, max_rows=100)
    du.print_descriptives(data, headers)
    #exit(1)
    apply_to_dataset(filename, 5)
    # apply_to_dataset(filename, 2)

