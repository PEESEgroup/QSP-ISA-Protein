
import qiskit as qk
import sklearn.svm

def index_list(i):
    return [x for x in range(i)]

#Class for quantum kernel SVM
class QKSVM:
    _model_ = None
    _quantum_kernel_ = None
    _data_x_ = None
    #Constructor
    #@param encode: a function that takes in a vector and returns a quantum
    # circuit that encodes that vector
    def __init__(self, encode):
        def quantum_kernel(v1, v2):
            c1 = encode(v1)
            c2 = encode(v2)
            c1.append(c2.inverse().to_instruction(), index_list(c1.num_qubits))
            c1.measure_all()
            simulator = qk.Aer.get_backend('aer_simulator')
            circ = qk.transpile(c1, simulator)
            result = simulator.run(circ).result().get_counts(circ)
            key = '0'*c1.num_qubits
            if(key in result): return result[key] / 1024.0
            return 0
        self._model_ = sklearn.svm.SVC(kernel="precomputed")
        self._quantum_kernel_ = quantum_kernel

    #computes the kernel matrix for a list of vectors x
    def _kernel_matrix_(self, x):
        n = len(x)
        output = [[None for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                print(i, " ", j)
                if(i == j): output[i][j] = 1
                elif(i > j): output[i][j] = output[j][i]
                else:
                    output[i][j] = self._quantum_kernel_(x[i], x[j])
        return output

    #Updates internal parameters to fit the data points x, y.
    #[x] is a list of vectors, [y] is their classifications. [y] labels are
    # 1 or -1.
    def fit(self, x, y):
        self._data_x_ = x
        self._model_.fit(self._kernel_matrix_(x), y)

    #Predicts the category of vector [v].
    #returns either 1 or -1.
    def predict(self, v):
        x = self._data_x_
        features = [self._quantum_kernel_(x[i], v) for i in range(len(x))]
        print(features)
        return self._model_.predict([features])[0]
