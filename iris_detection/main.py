from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()

""" _, ax = plt.subplots()
scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
_ = ax.legend(
    scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
)

plt.show() """

iris.head()
iris.info()