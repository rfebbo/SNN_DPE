import numpy as np


# read in the Iris data, can be a subset where the data is limited to 2 classes and 2 attributes
def read_iris_data(filepath, subset=False, shuffle=False):
    classes = { 'Iris-setosa' : 0,
                'Iris-versicolor' : 1,
                'Iris-virginica' : 2
                }

    attributes = ['Sepal Length', 
                    'Sepal Width',
                    'Petal Length',
                    'Petal Width'
    ]

    if subset:
        classes.pop('Iris-virginica')
        attributes.remove('Petal Length')
        attributes.remove('Petal Width')

    iris_data = []
    labels = []
    with open(filepath, 'r') as f:
        for l in f:
            d = l.strip().split(',')
            

            if d[-1] in classes:
                labels.append(classes[d[-1]])
                iris_data.append(d[0:4])


    iris_data = np.asarray(iris_data, np.float32)
    labels = np.asarray(labels, np.int32)

    if shuffle:
        p = np.random.permutation(len(iris_data))
        iris_data = iris_data[p]
        labels = labels[p]
    

    return iris_data, labels, classes, attributes

# normalize the data to a range of 0.0 to 1.0 in order to use the Encoders
def normalize_iris_data(iris_data, attributes):

    normalized_iris_data = []
    for i in range(len(attributes)):
        dmin = np.min(iris_data[:, i])
        
        dmax = np.max(iris_data[:, i])
        
        v = (iris_data[:,i] - dmin) / (dmax - dmin)
        normalized_iris_data.append(list(v))

    normalized_iris_data = np.asarray(normalized_iris_data).reshape(len(attributes), len(iris_data))
    normalized_iris_data = np.transpose(normalized_iris_data)

    return normalized_iris_data