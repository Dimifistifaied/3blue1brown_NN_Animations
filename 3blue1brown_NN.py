from matplotlib.pylab import *
import sys
import itertools


#Set your manim directory inside path, if manim is not part of your current python env

sys.path.insert(1, '/home/dimitarpopov/anaconda3/lib/python3.8/site-packages')

from manim import *

class myNeuralNetwork(Scene):
    def construct(self):
        myNetwork = NeuralNetworkMobject([1000, 8, 1])
        myNetwork.label_inputs('x')
        myNetwork.label_outputs('\hat{y}')
        myNetwork.label_outputs_text(['isPredicate'])
        myNetwork.label_hidden_layers(200)

        myNetwork.scale(0.75)
        self.play(Write(myNetwork))
        self.wait()

# A customizable Sequential Neural Network
class NeuralNetworkMobject(VGroup):
    arguments = {
        "neuron_radius": 0.2,
        "neuron_to_neuron_buff": MED_SMALL_BUFF,
        "layer_to_layer_buff": 5,
        "output_neuron_color": WHITE,
        "input_neuron_color": WHITE,
        "hidden_layer_neuron_color": RED,
        "neuron_stroke_width": 2,
        "neuron_fill_color": GREEN,
        "edge_color": LIGHT_GREY,
        "edge_stroke_width": 2,
        "edge_propogation_color": YELLOW,
        "edge_propogation_time": 2,
        "max_shown_neurons": 16,
        "brace_for_large_layers": True,
        "average_shown_activation_of_large_layer": True,
        "include_output_labels": False,
        "arrow": False,
        "arrow_tip_size": 0.1,
        "left_size": 1,
        "neuron_fill_opacity": 1
    }

    # Constructor with parameters of the neurons in a list
    def __init__(self, neural_network, *args, **kwargs):
        VGroup.__init__(self, *args, **kwargs)
        self.layer_sizes = neural_network
        self.add_neurons()
        self.add_edges()
        self.add_to_back(self.layers)

    # Helper method for constructor
    def add_neurons(self):
        layers = VGroup(*[
            self.get_layer(size, index)
            for index, size in enumerate(self.layer_sizes)
        ])
        layers.arrange_submobjects(RIGHT, buff=self.argrs_parser('layer_to_layer_buff'))
        self.layers = layers
        if self.argrs_parser('include_output_labels'):
            self.label_outputs_text()

    # Helper method for constructor
    def get_nn_fill_color(self, index):
        if index == -1 or index == len(self.layer_sizes) - 1:
            return self.argrs_parser('output_neuron_color')
        if index == 0:
            return self.argrs_parser('input_neuron_color')
        else:
            return self.argrs_parser('hidden_layer_neuron_color')

    # Helper method for constructor
    def get_layer(self, size, index=-1):
        layer = VGroup()
        n_neurons = size
        if n_neurons > self.argrs_parser('max_shown_neurons'):
            n_neurons = self.argrs_parser('max_shown_neurons')
        neurons = VGroup(*[
            Circle(
                radius=self.argrs_parser('neuron_radius'),
                stroke_color=self.get_nn_fill_color(index),
                stroke_width=self.argrs_parser('neuron_stroke_width'),
                fill_color=BLACK,
                fill_opacity=self.argrs_parser('neuron_fill_opacity'),
            )
            for x in range(n_neurons)
        ])
        neurons.arrange_submobjects(
            DOWN, buff=self.argrs_parser('neuron_to_neuron_buff')
        )
        for neuron in neurons:
            neuron.edges_in = VGroup()
            neuron.edges_out = VGroup()
        layer.neurons = neurons
        layer.add(neurons)

        if size > n_neurons:
            dots = MathTex("\\vdots")
            dots.move_to(neurons)
            VGroup(*neurons[:len(neurons) // 2]).next_to(
                dots, UP, MED_SMALL_BUFF
            )
            VGroup(*neurons[len(neurons) // 2:]).next_to(
                dots, DOWN, MED_SMALL_BUFF
            )
            layer.dots = dots
            layer.add(dots)
            if self.argrs_parser('brace_for_large_layers'):
                brace = Brace(layer, LEFT)
                brace_label = brace.get_tex(str(size))
                layer.brace = brace
                layer.brace_label = brace_label
                layer.add(brace, brace_label)

        return layer

    # Helper method for constructor
    def add_edges(self):
        self.edge_groups = VGroup()
        for l1, l2 in zip(self.layers[:-1], self.layers[1:]):
            edge_group = VGroup()
            for n1, n2 in itertools.product(l1.neurons, l2.neurons):
                edge = self.get_edge(n1, n2)
                edge_group.add(edge)
                n1.edges_out.add(edge)
                n2.edges_in.add(edge)
            self.edge_groups.add(edge_group)
        self.add_to_back(self.edge_groups)

    # Helper method for constructor
    def get_edge(self, neuron1, neuron2):
        if self.argrs_parser('arrow'):
            return Arrow(
                neuron1.get_center(),
                neuron2.get_center(),
                buff=self.neuron_radius,
                stroke_color=self.argrs_parser('edge_color'),
                stroke_width=self.argrs_parser('edge_stroke_width'),
                tip_length=self.argrs_parser('arrow_tip_size')
            )
        return Line(
            neuron1.get_center(),
            neuron2.get_center(),
            buff=self.argrs_parser('neuron_radius'),
            stroke_color=self.argrs_parser('edge_color'),
            stroke_width=self.argrs_parser('edge_stroke_width'),
        )

    # Labels each input neuron with a char l or a LaTeX character
    def label_inputs(self, l):
        self.output_labels = VGroup()
        for n, neuron in enumerate(self.layers[0].neurons):
            label = MathTex(f"{l}_" + "{" + f"{n + 1}" + "}")
            label.set_height(0.3 * neuron.get_height())
            label.move_to(neuron)
            self.output_labels.add(label)
        self.add(self.output_labels)

    # Labels each output neuron with a char l or a LaTeX character
    def label_outputs(self, l):
        self.output_labels = VGroup()
        for n, neuron in enumerate(self.layers[-1].neurons):
            label = MathTex(f"{l}_" + "{" + f"{n + 1}" + "}")
            label.set_height(0.4 * neuron.get_height())
            label.move_to(neuron)
            self.output_labels.add(label)
        self.add(self.output_labels)

    # Labels each neuron in the output layer with text according to an output list
    def label_outputs_text(self, outputs):
        self.output_labels = VGroup()
        for n, neuron in enumerate(self.layers[-1].neurons):
            label = MathTex(outputs[n])
            label.set_height(0.75 * neuron.get_height())
            label.move_to(neuron)
            label.shift((neuron.get_width() + label.get_width() / 2) * RIGHT)
            self.output_labels.add(label)
        self.add(self.output_labels)

    # Labels the hidden layers with a char l or a LaTeX character
    def label_hidden_layers(self, l):
        self.output_labels = VGroup()
        for layer in self.layers[1:-1]:
            for n, neuron in enumerate(layer.neurons):
                label = MathTex(f"{l}_{n + 1}")
                label.set_height(0.4 * neuron.get_height())
                label.move_to(neuron)
                self.output_labels.add(label)
        self.add(self.output_labels)


    def argrs_parser(self, string):
        return NeuralNetworkMobject.arguments[string]

# Press the green button in the gutter to run the script.
