import argparse
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def create_viz(args):

    df = pd.read_csv(args.dataset_path, index_col='index')

    #seperate out class from label
    y = df['class']
    X = df.drop('class', axis=1)

    tsne = TSNE(n_components=2, verbose=1, random_state=123, perplexity=30, n_iter=1000)
    z = tsne.fit_transform(X)
    df = pd.DataFrame()
    df["y"] = y
    df["comp-1"] = z[:, 0]
    df["comp-2"] = z[:, 1]

    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    #palette=sns.color_palette("hls", 15),
                    palette="tab20",
                    data=df).set(title="Smell T-SNE projection")

    #plt.show()
    plt.savefig(args.output_image_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Smell Dataset Visualization')

    # general args
    parser.add_argument('--project_name', type=str, default='smell_dataset_viz', help='name of project')
    parser.add_argument('--dataset_path', type=str, default='smell_dataset.csv', help='location of dataset')
    parser.add_argument('--output_image_path', type=str, default='smell_dataset_viz.png', help='location of dataset')

    args = parser.parse_args()

    create_viz(args)