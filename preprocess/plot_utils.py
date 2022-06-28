import os
import json
import matplotlib.pyplot as plt

def plot_distributions(DATASET_PATH):
    percentage_path = os.path.join(DATASET_PATH, 'percentage_distribution.json')
    row_path = os.path.join(DATASET_PATH, 'rows_distribution.json')
    weighted_percentage_path = os.path.join(DATASET_PATH, 'weighted_percentage_distribution.json')

    with open(percentage_path) as f:
        percentage_dist = json.load(f)
    with open(row_path) as f:
        row_dist = json.load(f)
    with open(weighted_percentage_path) as f:
        weighted_percentage_dist = json.load(f)

    fig, ax = plt.subplots(9, figsize=(30, 90))

    ax[0].bar(range(len(row_dist)), list(row_dist.values()), align='center')
    ax[0].set_title("Scores distribution by row", fontsize=25)

    ax[1].bar(range(len(percentage_dist)), list(percentage_dist.values()), align='center')
    ax[1].set_xticks(range(len(percentage_dist)), list(percentage_dist.keys()))
    ax[1].set_title("Scores distribution by percentage", fontsize=25)

    ax[2].bar(range(len(weighted_percentage_dist)), list(weighted_percentage_dist.values()), align='center')
    ax[2].set_xticks(range(len(weighted_percentage_dist)), list(weighted_percentage_dist.keys()))
    ax[2].set_title("Weighted scores distribution by percentage", fontsize=25)

    dist_p_ls = defaultdict(lambda: 0)
    dist_r_ls = defaultdict(lambda: 0)
    dist_p_md = defaultdict(lambda: 0)
    dist_r_md = defaultdict(lambda: 0)
    dist_p_gt = defaultdict(lambda: 0)
    dist_r_gt = defaultdict(lambda: 0)
    for label in os.listdir(DATASET_PATH):
        if label != 'percentage_distribution.json' and label != 'rows_distribution.json' and label != 'weighted_percentage_distribution.json' and label.endswith('.json'):
            with open(os.path.join(DATASET_PATH, label)) as f:
                dist = json.load(f)
            if dist['length'] < 500:
                for k, v in dist['bucket'].items():
                    dist_p_ls[k] += v
                for k, v in dist['score'].items():
                    dist_r_ls[k] += v
            elif dist['length'] >= 500 and dist['length'] < 1000:
                for k, v in dist['bucket'].items():
                    dist_p_md[k] += v
                for k, v in dist['score'].items():
                    dist_r_md[k] += v
            else:
                for k, v in dist['bucket'].items():
                    dist_p_gt[k] += v
                for k, v in dist['score'].items():
                    dist_r_gt[k] += v

    ax[3].bar(range(len(dist_p_ls)), list(dist_p_ls.values()), align='center')
    ax[3].set_xticks(range(len(dist_p_ls)), list(dist_p_ls.keys()))
    ax[3].set_title("Scores distribution by percentage of files having less than 500 rows", fontsize=25)

    ax[4].bar(range(len(dist_p_md)), list(dist_p_md.values()), align='center')
    ax[4].set_xticks(range(len(dist_p_md)), list(dist_p_md.keys()))
    ax[4].set_title("Scores distribution by percentage of files having between 500 and 1000 rows", fontsize=25)

    ax[5].bar(range(len(dist_p_gt)), list(dist_p_gt.values()), align='center')
    ax[5].set_xticks(range(len(dist_p_gt)), list(dist_p_gt.keys()))
    ax[5].set_title("Scores distribution by percentage of files having more than 1000 rows", fontsize=25)

    ax[6].bar(range(len(dist_r_ls)), list(dist_r_ls.values()), align='center')
    ax[6].set_title("Scores distribution by row of files having less than 500 rows", fontsize=25)

    ax[7].bar(range(len(dist_r_md)), list(dist_r_md.values()), align='center')
    ax[7].set_title("Scores distribution by row of files having between 500 and 1000 rows", fontsize=25)

    ax[8].bar(range(len(dist_r_gt)), list(dist_r_gt.values()), align='center')
    ax[8].set_title("Scores distribution by row of files having more than 1000 rows", fontsize=25)

    plt.show()
    plt.close()
