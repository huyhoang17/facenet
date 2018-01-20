import matplotlib
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.externals import joblib
# Ignore "No module name _tkinter" error
matplotlib.use('agg')


labels, class_names, emb_arrays = joblib.load(
    "models/backup/model_updated.pkl"
)
n_samples = 20
num_samples = labels.index(n_samples)
tsne_model = TSNE(
    n_components=2, verbose=1, random_state=0,
    angle=.99, init='pca'
)
print(emb_arrays.shape)
X_tsne = tsne_model.fit_transform(emb_arrays[-num_samples:])
print(X_tsne.shape)

df = pd.DataFrame(X_tsne, columns=['x', 'y'])
df["class"] = labels[-num_samples:]
# print(df)
g = sns.lmplot(
    'x', 'y', df, hue='class', fit_reg=False, size=8,
    scatter_kws={'alpha': 0.7, 's': 60}
)
g.axes.flat[0].set_title(
    'Scatterplot of a 50D dataset reduced to 2D using t-SNE'
)
