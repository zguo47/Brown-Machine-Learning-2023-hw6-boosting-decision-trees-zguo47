import random
import numpy as np
import matplotlib.pyplot as plt

from get_data import get_data
from models import Node, DecisionTree, node_score_error, node_score_entropy, node_score_gini


def loss_plot(ax, title, tree, train_data, test_data):
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
    # ax.plot(pruned_tree.loss_plot_vec(train_data), label='train pruned')
    # ax.plot(pruned_tree.loss_plot_vec(test_data), label='test pruned')


    ax.locator_params(nbins=3)
    ax.set_xlabel('number of nodes', fontsize=fontsize)
    ax.set_ylabel('loss', fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    legend = ax.legend(loc='upper center', shadow=True, fontsize=fontsize-2)

def loss_plot2(ax, title, tree1, train_data):
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
    ax.plot(tree1.loss_plot_vec(train_data), label=1)


    # ax.plot(pruned_tree.loss_plot_vec(train_data), label='train pruned')
    # ax.plot(pruned_tree.loss_plot_vec(test_data), label='test pruned')


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
    x1 = []
    x2 = []
    x3 = []
    tree1 = DecisionTree(data=train_data, gain_function=node_score_error)
    print('First loss', tree1.loss(train_data))
    x1.append(tree1.loss(train_data))
    tree2 = DecisionTree(data=train_data, gain_function=node_score_entropy)
    print('Second loss', tree2.loss(train_data))
    x2.append(tree2.loss(train_data))
    tree3 = DecisionTree(data=train_data, gain_function=node_score_gini)
    print('Third loss', tree3.loss(train_data))
    x3.append(tree3.loss(train_data))
    tree4 = DecisionTree(data=test_data, gain_function=node_score_error)
    print('Fourth loss', tree4.loss(test_data))
    x1.append(tree4.loss(test_data))
    tree5 = DecisionTree(data=test_data, gain_function=node_score_entropy)
    print('Fifth loss', tree5.loss(test_data))
    x2.append(tree5.loss(test_data))
    tree6 = DecisionTree(data=test_data, gain_function=node_score_gini)
    print('Sixth loss', tree6.loss(test_data))
    x3.append(tree6.loss(test_data))
    tree7 = DecisionTree(data=train_data, validation_data=validation_data, gain_function=node_score_error)
    print('Seventh loss', tree7.loss(train_data))
    x1.append(tree7.loss(train_data))
    tree8 = DecisionTree(data=train_data, validation_data=validation_data, gain_function=node_score_entropy)
    print('Eighth loss', tree8.loss(train_data))
    x2.append(tree8.loss(train_data))
    tree9 = DecisionTree(data=train_data, validation_data=validation_data, gain_function=node_score_gini)
    print('Ninth loss', tree9.loss(train_data))
    x3.append(tree9.loss(train_data))
    tree10 = DecisionTree(data=test_data, validation_data=validation_data, gain_function=node_score_error)
    print('Tenth loss', tree10.loss(test_data))
    x1.append(tree10.loss(test_data))
    tree11 = DecisionTree(data=test_data, validation_data=validation_data, gain_function=node_score_entropy)
    print('Eleventh loss', tree11.loss(test_data))
    x2.append(tree11.loss(test_data))
    tree12 = DecisionTree(data=test_data, validation_data=validation_data, gain_function=node_score_gini)
    print('Twelfth loss', tree12.loss(test_data))
    x3.append(tree12.loss(test_data))

    # names = ['train wo prune', 'test wo prune', 'train w prune', 'test w prune']
    # barwidth = 0.25
    # br1 = np.arange(len(x1))
    # br2 = [x + barwidth for x in br1]
    # br3 = [x + barwidth for x in br2]
    # plt.title('loss plot')
    # plt.bar(br1, x1, color = 'r', width = barwidth, label = 'error')
    # plt.bar(br2, x2, color = 'g', width = barwidth, label = 'entropy')
    # plt.bar(br3, x3, color = 'b', width = barwidth, label = 'gini')
    # plt.xticks([r + barwidth for r in range (len(x1))], names)
    # plt.legend()
    # plt.show()

    # TODO: Feel free to print or plot anything you like here. Just comment
    # make sure to comment it out, or put it in a function that isn't called
    # by default when you hand in your code!
    # tree1 = DecisionTree(data=train_data, max_depth=1)
    # tree2 = DecisionTree(data=train_data, max_depth=2)
    # tree3 = DecisionTree(data=train_data, max_depth=3)
    # tree4 = DecisionTree(data=train_data, max_depth=4)
    # tree5 = DecisionTree(data=train_data, max_depth=5)
    # tree6 = DecisionTree(data=train_data, max_depth=6)
    # tree7 = DecisionTree(data=train_data, max_depth=7)
    # tree8 = DecisionTree(data=train_data, max_depth=8)
    # tree9 = DecisionTree(data=train_data, max_depth=9)
    # tree10 = DecisionTree(data=train_data, max_depth=10)
    # tree11 = DecisionTree(data=train_data, max_depth=11)
    # tree12 = DecisionTree(data=train_data, max_depth=12)
    # tree13 = DecisionTree(data=train_data, max_depth=13)
    # tree14 = DecisionTree(data=train_data, max_depth=14)
    # tree15 = DecisionTree(data=train_data, max_depth=15)
    # title = 'loss plot'
    # fig, ax = plt.subplots()
    # loss_plot2(ax, title, tree17, train_data)
    # plt.show()
    # X = [tree1.loss(train_data), tree2.loss(train_data), tree3.loss(train_data), tree4.loss(train_data), tree5.loss(train_data), tree6.loss(train_data), tree7.loss(train_data), tree8.loss(train_data), tree9.loss(train_data), tree10.loss(train_data), tree11.loss(train_data), tree12.loss(train_data), tree13.loss(train_data), tree14.loss(train_data), tree15.loss(train_data)]
    # names = np.arange(1, 16)
    # plt.title('loss with max depth from 1 to 15')
    # plt.bar(names, X)
    # plt.show()
    # print('First loss', tree13.loss(train_data))
    # print('Second loss', tree14.loss(train_data))
    # print('Third loss', tree15.loss(train_data))
    # print('Fourth loss', tree16.loss(train_data))
    # print('Fifth loss', tree17.loss(train_data))

def main():
    ########### PLEASE DO NOT CHANGE THESE LINES OF CODE! ###################
    random.seed(1)
    np.random.seed(1)
    #########################################################################

    # explore_dataset('data/chess.csv', 'won')
    explore_dataset('data/spam.csv', '1')

main()
