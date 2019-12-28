from random import shuffle

from kastor.creators import *
from kastor.activation_functions import *
from kastor.cost_functions import *
from kastor.init_methods import *
from kastor.nn_test import test_network
from kastor.nn_train import train_network


class NeuralNetwork:

    def __init__(self):
        self.activation_functions_list = []
        self.functions_for_errors = []
        self.hidden_sizes_list = []
        self.last_layer_size = 0

        # O lista cu straturile hidden si ultimul strat (la final)
        # aici se pun rezultatele activarilor
        # in algoritmi mai incolo asta vine ca parametru cu numele act_matrices sau activation_matrices
        self.hidden_and_output = []

        self.weight_matrices = []
        self.bias_matrices = []

        self.train_set = []
        self.test_set = []

    # In cazul in care dorim utilizarea weight-urilor antrenate de alta retea neuronala sau
    # chiar weight-uri initializate custom (in cazul in care framework-ul current nu are ce ne trebuie)
    def set_weight_matrices(self, weight_matrices):
        self.weight_matrices = weight_matrices

    # La fel ca si in cazul weight-urilor, doar ca pentru bias
    def set_bias_matrices(self, bias_matrices):
        self.bias_matrices = bias_matrices

    def add_hidden_layer(self, activation_funct, activation_deriv, neurons_count):
        self.activation_functions_list.append(activation_funct)
        self.functions_for_errors.append(activation_deriv)
        self.hidden_sizes_list.append(neurons_count)

    # Pentru ca cel ce utilizeaza framework-ul sa aiba o viziune mai clara
    # am facut separat o functie pentru ultimul strat (in mod special pentru al doilea parametru)
    def add_last_layer(self, activation_funct, cost_funct_deriv, neurons_count):
        self.activation_functions_list.append(activation_funct)
        self.functions_for_errors.append(cost_funct_deriv)
        self.last_layer_size = neurons_count

    def init_components(self, input_layer_size, weight_init_name, bias_init_name):
        hidden_layers = create_hidden_layers(self.hidden_sizes_list,
                                             self.activation_functions_list,
                                             self.functions_for_errors)
        self.hidden_and_output = hidden_layers + [[np.array([[0] * self.last_layer_size]).transpose(),
                                                   self.activation_functions_list[-1],
                                                   self.functions_for_errors[-1]]]
        self.weight_matrices = create_weight_matrices(self.hidden_sizes_list, input_layer_size, self.last_layer_size)
        self.bias_matrices = create_bias_matrices(self.hidden_sizes_list, self.last_layer_size)

        init_weights(weight_init_name, self.weight_matrices, input_layer_size)
        init_bias(bias_init_name, self.bias_matrices, input_layer_size)

    def load_dataset(self, data_set, cross_valid_method):
        if cross_valid_method == "train_test_split":
            data = data_set
            shuffle(data)
            data_len = len(data)

            # 70% train si 30% test
            split_index = int(data_len * 0.7)

            self.train_set = data[:split_index]
            self.test_set = data[split_index:]

    def fit(self, count_iterations, learning_rate, batch_size, show_acc=False, momentum_friction=-1, l2_lambda=0.0):
        data = list(zip(*self.train_set))
        instances = list(data[0])
        actual_values = list(data[1])

        test_data = list(zip(*self.test_set))
        test_instances = list(test_data[0])
        test_actual_values = list(test_data[1])

        for it in tqdm(range(0, count_iterations), position=0):
            train_network(instances, actual_values,
                          learning_rate, batch_size,
                          self.weight_matrices, self.bias_matrices, self.hidden_and_output,
                          momentum_friction,  # pentru momentum
                          l2_lambda)  # pentru L2 reg

            # Aici in for se poate implementa o metoda pentru learning rate adaptiv
            # ex extrem de simplu: lr = lr / 2 (dar fixat neaparat)

            if show_acc:
                # Aici asa am dat eu pentru o vizualizare a rezultatelor (nu va ramane neaparat asa)
                result_1 = test_network(test_instances, test_actual_values,
                                        self.weight_matrices, self.bias_matrices, self.hidden_and_output,
                                        use_one_hot=True)
                print("--- Epoch " + str(it) + " ---")
                print("| Result test : " + str(result_1) + " |")
