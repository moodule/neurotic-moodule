# coding=utf-8

__author__ = 'EelMood'

import os, sys
import timeit

import numpy as np

import theano
import theano.tensor as T

from framework import *


#  Feature extraction
# Feature preprocessing
#  Feature selection
#  Dataset partition for training and local validation
#  Model selection
# Model hyperparameter selection
# Model ensemble
# Submission preparation
# TODO comparer modele normalise et non
# TODO etudier les poids relatifs des differentes composantes (modele normalise)
# TODO faire une PCA
# TODO faire un kNN
# TODO appliquer les differents modeles des tuto theano
# TODO coder fn pour scinder le dataset + creer [datasets] pour moyenner les resultats
# TODO affiner les valeurs des meta parametres
# TODO generer des graph pr suivre l'apprentissage
# TODO penser des perturbations qui laissent le vecteur y invariant tt en creant de nouveaux x => NN apprend les invar.
# TODO creer une classe Model qui gere les traitements communs a tous les modeles de classification
#       => scinder x et y
#       => minibatch
#       => shuffle
#       => load and dump !
#       => normaliser
#       => predire (methode abstraite)
#       => visualiser (les poids par exemple)
#       => modification des meta parametres
#       => optimisation meta
#       => fonction de cout
#       => tests ?
#       => datasets augmentation (perturbations)
#       => courbes d'apprentissage
#       => courbes d'evaluation meta (alpha, gamma, nb echantillons entr., contribution des parametres)
#       => graph du modele


class RegressionModel(ModelFramework):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self):
        super(RegressionModel, self).__init__()

    def train(self):
        pass