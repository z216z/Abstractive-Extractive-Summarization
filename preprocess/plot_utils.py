import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt

def plot_distributions(DATASET_PATH):
    with open(os.path.join(DATASET_PATH, 'distribution.json')) as f:
        distribution = json.load(f)

    row_dist = distribution['rows']                    
    percentage_dist = distribution['percentage']
    weighted_percentage_dist = distribution['weighted_percentage']

    fig, ax = plt.subplots(9, figsize=(30, 90))

    ax[0].bar(range(len(row_dist)), row_dist, align='center')
    ax[0].set_title("Scores distribution by row", fontsize=25)

    ax[1].bar(range(1, 101), percentage_dist, align='center')
    ax[1].set_xticks(range(1, 101))
    ax[1].set_title("Scores distribution by percentage", fontsize=25)

    ax[2].bar(range(1, 101), weighted_percentage_dist, align='center')
    ax[2].set_xticks(range(1, 101))
    ax[2].set_title("Weighted scores distribution by percentage", fontsize=25)

    dist_p_ls = defaultdict(lambda: 0)
    dist_r_ls = defaultdict(lambda: 0)
    dist_p_md = defaultdict(lambda: 0)
    dist_r_md = defaultdict(lambda: 0)
    dist_p_gt = defaultdict(lambda: 0)
    dist_r_gt = defaultdict(lambda: 0)
    for label in os.listdir(DATASET_PATH):
        if label != 'distribution.json' and label.endswith('.json'):
            with open(os.path.join(DATASET_PATH, label)) as f:
                dist = json.load(f)
            if dist['length'] < 500:
                for k, v in enumerate(dist['bucket']):
                    dist_p_ls[k+1] += v
                for k, v in enumerate(dist['score']):
                    dist_r_ls[k] += v
            elif dist['length'] >= 500 and dist['length'] < 1000:
                for k, v in enumerate(dist['bucket']):
                    dist_p_md[k+1] += v
                for k, v in enumerate(dist['score']):
                    dist_r_md[k] += v
            else:
                for k, v in enumerate(dist['bucket']):
                    dist_p_gt[k+1] += v
                for k, v in enumerate(dist['score']):
                    dist_r_gt[k] += v

    ax[3].bar(range(1, 101), list(dist_p_ls.values()), align='center')
    ax[3].set_xticks(range(1, 101))
    ax[3].set_title("Scores distribution by percentage of files having less than 500 rows", fontsize=25)

    ax[4].bar(range(1, 101), list(dist_p_md.values()), align='center')
    ax[4].set_xticks(range(1, 101))
    ax[4].set_title("Scores distribution by percentage of files having between 500 and 1000 rows", fontsize=25)

    ax[5].bar(range(1, 101), list(dist_p_gt.values()), align='center')
    ax[5].set_xticks(range(1, 101))
    ax[5].set_title("Scores distribution by percentage of files having more than 1000 rows", fontsize=25)

    ax[6].bar(range(len(dist_r_ls)), list(dist_r_ls.values()), align='center')
    ax[6].set_title("Scores distribution by row of files having less than 500 rows", fontsize=25)

    ax[7].bar(range(len(dist_r_md)), list(dist_r_md.values()), align='center')
    ax[7].set_title("Scores distribution by row of files having between 500 and 1000 rows", fontsize=25)

    ax[8].bar(range(len(dist_r_gt)), list(dist_r_gt.values()), align='center')
    ax[8].set_title("Scores distribution by row of files having more than 1000 rows", fontsize=25)

    plt.show()
    plt.close()
