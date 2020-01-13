import pickle, gzip
import numpy as np
import pandas as pd

from kastor import kastor_framework as kt


def construct_t(digit):
    t = [0] * 10
    t[digit] = 1
    return np.array([t]).transpose()


f = gzip.open("mnist.pkl.gz", "rb")
train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
f.close()

instances_train = train_set[0]
actual_values_train = train_set[1]
instances_test = test_set[0]

actual_values_test = test_set[1]
instances_valid = valid_set[0]
actual_values_valid = valid_set[1]

data_set = np.concatenate((train_set[0], valid_set[0], test_set[0]), axis=0)
actual_values_not_processed = np.concatenate((train_set[1], valid_set[1], test_set[1]), axis=0)
actual_values = []

# Asta e necesara doar pentru clasificare in cazul temei de la RN
for index in range(0, len(actual_values_not_processed)):
    actual_values.append(construct_t(actual_values_not_processed[index]))

print(data_set.shape)
print(np.array(actual_values).shape)
print("-------------------------------------")

data_train_csv = np.array(pd.read_csv('dataset/Training/Features_Variant_1.csv'))
data_train_instances = np.array([data_train_csv[index][0:-1] for index in range(0, len(data_train_csv))])
data_train_target = np.array([[[data_train_csv[index][-1]]] for index in range(0, len(data_train_csv))])

print(data_train_instances.shape)
print(data_train_target.shape)
print(data_train_csv.shape)
print(data_train_target[0])

model = kt.NeuralNetwork()

# model.add_hidden_layer(activation_funct=kt.act_func_sigmoid, activation_deriv=kt.deriv_sigmoid, neurons_count=50)
# model.add_hidden_layer(activation_funct=kt.act_funct_relu, activation_deriv=kt.deriv_relu, neurons_count=40)
model.add_hidden_layer(activation_funct=kt.act_funct_relu, activation_deriv=kt.deriv_relu, neurons_count=25)
model.add_last_layer(activation_funct=kt.act_funct_linear, cost_funct_deriv=kt.cost_cross_entropy, neurons_count=1)

model.init_components(input_layer_size=53, weight_init_name="normal", bias_init_name="normal")

model.load_dataset(data_set=list(zip(data_train_instances, data_train_target)), cross_valid_method="train_test_split")

model.fit(count_iterations=10, learning_rate=0.01, batch_size=100, show_acc=True,
          momentum_friction=0.9, use_nesterov=True,
          l1_lambda=0.1)

#  |(t-y)|
