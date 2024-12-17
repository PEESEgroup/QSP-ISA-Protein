
import sklearn.svm


x = [
[0, 0, 0],
[1, 1, 1]
]

y = [1, -1]

xk = [[0, 0], [0, 3]]

t = [[1, 2, 1], [0, 0.2, 1]]

tk = [[0, 4], [0, 1.2]]

model = sklearn.svm.SVC(kernel="precomputed")

model.fit(xk, y)

print(model.predict(tk))

print(model.predict(tk[1]))

