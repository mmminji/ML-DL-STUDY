from sklearn.metrics import silhouette_score

X = [[1, 2], [4, 5], [2, 1], [6, 7], [2, 3]]
labels = [0, 1, 0, 1, 0]
sil_score = silhouette_score(X, labels)
print(sil_score)
# 0.5789497702625118