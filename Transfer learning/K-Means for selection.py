import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df_real = pd.read_excel(r'Dataset\Original LDPC dataset.xlsx')
df_fake = pd.read_excel(r"Dataset\Generated samples.xlsx")

column_to_encode = 'Agent'
df_encoded = pd.get_dummies(df_real[column_to_encode], prefix=column_to_encode)
df = pd.concat([df_real, df_encoded], axis=1)
df.drop(columns=[column_to_encode], inplace=True)
df.to_csv(r"Dataset\Original LDPC dataset.csv", index=False)
df_real = pd.read_csv(r"Dataset\Original LDPC dataset.csv")

my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="mean")),
    ('std_scaler', StandardScaler())
])

features_real = df_real.drop(['SBET', 'TPV'], axis=1)
labels_real = df_real[['SBET', 'TPV']]
features_fake = df_fake.drop(['SBET', 'TPV'], axis=1)
labels_fake = df_fake[['SBET', 'TPV']]

features_prepared_real = my_pipeline.fit_transform(features_real)
features_prepared_fake = features_fake

features_combined = np.vstack((features_prepared_real, features_prepared_fake))

# t-SNE analysis
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=500,
    n_iter=1000,
    random_state=0
)
X_tsne = tsne.fit_transform(features_combined)

tsne_df = pd.DataFrame(X_tsne, columns=['t-SNE Component 1', 't-SNE Component 2'])
tsne_df['label'] = ['real'] * len(features_prepared_real) + ['fake'] * len(features_prepared_fake)

plt.figure(figsize=(12, 8))
plt.scatter(X_tsne[:len(features_prepared_real), 0], X_tsne[:len(features_prepared_real), 1], c='blue', label='Real', alpha=0.5)
plt.scatter(X_tsne[len(features_prepared_real):, 0], X_tsne[len(features_prepared_real):, 1], c='red', label='Fake', alpha=0.5)
plt.title('t-SNE Visualization of Real and Fake Samples')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend()
plt.show()

# Visual analysis
real_tsne_points = X_tsne[:len(features_prepared_real)]
fake_tsne_points = X_tsne[len(features_prepared_real):]

distances = cdist(fake_tsne_points, real_tsne_points, metric='euclidean')

min_distances = np.min(distances, axis=1)
threshold_distance = np.percentile(min_distances, 5)  # 选择接近真实样本的虚假样本

selected_indices = min_distances <= threshold_distance
selected_fake_data = df_fake[selected_indices]

# Kmeans selection
n_clusters = 6
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
cluster_labels = kmeans.fit_predict(X_tsne)

tsne_df['cluster'] = cluster_labels
selected_fake_samples = pd.DataFrame()

fixed_sample_count = 15
fake_start_index = len(features_prepared_real)

for cluster_id in range(n_clusters):
    cluster_fake_indices = tsne_df[(tsne_df['label'] == 'fake') & (tsne_df['cluster'] == cluster_id)].index
    relative_indices = cluster_fake_indices - fake_start_index
    cluster_fake_distances = min_distances[relative_indices]
    n_select = min(fixed_sample_count, len(cluster_fake_indices))
    selected_in_cluster = relative_indices[np.argsort(cluster_fake_distances)[:n_select]]
    selected_fake_samples = pd.concat([selected_fake_samples, df_fake.iloc[selected_in_cluster]])


real_count = len(df_real)
fake_count = len(selected_fake_samples)

labels = ['real'] * real_count + ['fake'] * fake_count
clusters = np.concatenate([tsne_df[tsne_df['label'] == 'real']['cluster'].values,
                           tsne_df[tsne_df['label'] == 'fake']['cluster'].values[:fake_count]])

assert len(labels) == len(clusters), "Labels and clusters lengths do not match."

final_df = pd.DataFrame({
    't-SNE Component 1': np.concatenate([X_tsne[:real_count, 0], X_tsne[real_count:real_count + fake_count, 0]]),
    't-SNE Component 2': np.concatenate([X_tsne[:real_count, 1], X_tsne[real_count:real_count + fake_count, 1]]),
    'Cluster': clusters,
    'Sample Type': labels
})

# Visual analysis
selected_data = pd.concat([df_real, selected_fake_samples])

features_selected_fake_samples = selected_fake_samples.drop(['SBET', 'TPV'], axis=1)

X_tsne_selected = X_tsne[:len(df_real) + len(selected_fake_samples)]

final_tsne_df = pd.DataFrame(X_tsne_selected, columns=['t-SNE Component 1', 't-SNE Component 2'])
final_tsne_df['label'] = ['real'] * len(df_real) + ['fake'] * len(selected_fake_samples)

features_selected = np.vstack((features_prepared_real, features_selected_fake_samples))
cluster_labels = kmeans.fit_predict(features_selected)

print("Length of df_real:", len(df_real))
print("Length of selected_fake_samples:", len(selected_fake_samples))
print("Length of features_selected:", features_selected.shape[0])
print("Length of final_tsne_df:", len(final_tsne_df))
print("Length of cluster_labels:", len(cluster_labels))

final_tsne_df['cluster'] = cluster_labels

silhouette_avg = silhouette_score(features_selected, cluster_labels)
print(f"The average silhouette score for KMeans clustering is: {silhouette_avg}")
