import random
import numpy as np
import matplotlib.pyplot as plt

from get_data import get_data
from models import DecisionTree, node_score_error, node_score_entropy, node_score_gini


def loss_plot(ax, title, tree, pruned_tree, train_data, test_data):
    '''
        Example plotting code. This plots four curves: the training and testing
        average loss using tree and pruned tree.
        You do not need to change this code!
        Arguments:
            - ax: A matplotlib Axes instance.
            - title: A title for the graph (string)
            - tree: An unpruned DecisionTree instance
            - pruned_tree: A pruned DecisionTree instance
            - train_data: Training dataset returned from get_data
            - test_data: Test dataset returned from get_data
    '''
    fontsize=8
    ax.plot(tree.loss_plot_vec(train_data), label='train non-pruned')
    ax.plot(tree.loss_plot_vec(test_data), label='test non-pruned')
    ax.plot(pruned_tree.loss_plot_vec(train_data), label='train pruned')
    ax.plot(pruned_tree.loss_plot_vec(test_data), label='test pruned')


    ax.locator_params(nbins=3)
    ax.set_xlabel('number of nodes', fontsize=fontsize)
    ax.set_ylabel('loss', fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    legend = ax.legend(loc='upper center', shadow=True, fontsize=fontsize-2)

def explore_dataset(filename, class_name):
    train_data, validation_data, test_data = get_data(filename, class_name)

    # TODO: Print 12 loss values associated with the dataset.
    # For each measure of gain (training error, entropy, gini):
    #      (a) Print average training loss (not-pruned)
    #      (b) Print average test loss (not-pruned)
    #      (c) Print average training loss (pruned)
    #      (d) Print average test loss (pruned)
    tree1 = DecisionTree(data=train_data, gain_function=node_score_error)
    print('First loss', tree1.loss(train_data))
    tree2 = DecisionTree(data=train_data, gain_function=node_score_entropy)
    print('Second loss', tree2.loss(train_data))
    tree3 = DecisionTree(data=train_data, gain_function=node_score_gini)
    print('Third loss', tree3.loss(train_data))
    tree4 = DecisionTree(data=test_data, gain_function=node_score_error)
    print('Fourth loss', tree4.loss(test_data))
    tree5 = DecisionTree(data=test_data, gain_function=node_score_entropy)
    print('Fifth loss', tree5.loss(test_data))
    tree6 = DecisionTree(data=test_data, gain_function=node_score_gini)
    print('Sixth loss', tree6.loss(test_data))
    tree7 = DecisionTree(data=train_data, validation_data=validation_data, gain_function=node_score_error)
    print('Seventh loss', tree7.loss(train_data))
    tree8 = DecisionTree(data=train_data, validation_data=validation_data, gain_function=node_score_entropy)
    print('Eighth loss', tree8.loss(train_data))
    tree9 = DecisionTree(data=train_data, validation_data=validation_data, gain_function=node_score_gini)
    print('Ninth loss', tree9.loss(train_data))
    tree10 = DecisionTree(data=test_data, validation_data=validation_data, gain_function=node_score_error)
    print('Tenth loss', tree10.loss(test_data))
    tree11 = DecisionTree(data=test_data, validation_data=validation_data, gain_function=node_score_entropy)
    print('Eleventh loss', tree11.loss(test_data))
    tree12 = DecisionTree(data=test_data, validation_data=validation_data, gain_function=node_score_gini)
    print('Twelfth loss', tree12.loss(test_data))

    # TODO: Feel free to print or plot anything you like here. Just comment
    # make sure to comment it out, or put it in a function that isn't called
    # by default when you hand in your code!

def main():
    ########### PLEASE DO NOT CHANGE THESE LINES OF CODE! ###################
    random.seed(1)
    np.random.seed(1)
    #########################################################################

    explore_dataset('data/chess.csv', 'won')
    explore_dataset('data/spam.csv', '1')

main()
