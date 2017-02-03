# -*- mode: Python; coding: utf-8 -*-

from __future__ import division

from corpus import Document, BlogsCorpus, NamesCorpus
from naive_bayes import NaiveBayes

import sys
from random import shuffle, seed
from unittest import TestCase, main, skip

class EvenOdd(Document):
    def features(self):
        """Is the data even or odd?"""
        return [self.data % 2 == 0]

class BagOfWords(Document):
    def features(self):
        """Trivially tokenized words."""
        return self.data.split()

# class BagOfWords(Document):
#     def features(self):
#         return [(self.data[x], self.data[x + 1]) for x in range(0, len(self.data.split()))]
#
# class BagOfWords(Document):
#     def features(self):
#         return [(self.data[x], self.data[x + 1]) for x in range(0, len(self.data.split()))] + self.data.split()


def accuracy(classifier, test, verbose=sys.stderr):
    correct = [classifier.classify(x) == x.label for x in test]
    if verbose:
        print >> verbose, "%.2d%% " % (100 * sum(correct) / len(correct)),
    return sum(correct) / len(correct)

def f_measure_aggregate(classifier, test, label):
    tp = sum([classifier.classify(x) == x.label for x in test if x.label == label])
    tp_plus_fp = len([classifier.classify(x) for x in test if classifier.classify(x) == label])
    tp_plus_fn = len([x.label for x in test if x.label == label])
    print(label)
    print("precision")
    p = precision(tp, tp_plus_fp)
    print(p)
    print("recall")
    r = recall(tp, tp_plus_fn)
    print(r)
    print("fmeasure")
    f = f_measure(p, r)
    print(f)
    return f

def precision(tp, tp_plus_fp):
    return tp / tp_plus_fp

def recall(tp, tp_plus_fn):
    return tp / tp_plus_fn

def f_measure(precision, recall):
    return (2 * precision * recall) / (precision + recall)

class NaiveBayesTest(TestCase):
    u"""Tests for the naïve Bayes classifier."""

    def test_even_odd(self):
        """Classify numbers as even or odd"""
        classifier = NaiveBayes()
        classifier.train([EvenOdd(0, True), EvenOdd(1, False)])
        test = [EvenOdd(i, i % 2 == 0) for i in range(2, 1000)]
        self.assertEqual(accuracy(classifier, test), 1.0)
    #    print(f_measure_aggregate(classifier, test, True))

    def split_blogs_corpus(self, document_class):
        """Split the blog post corpus into training and test sets"""
        blogs = BlogsCorpus(document_class=document_class)
        self.assertEqual(len(blogs), 3232)
        seed(hash("blogs"))
        shuffle(blogs)
        return (blogs[:3000], blogs[3000:])

    def test_blogs_bag(self):
        """Classify blog authors using bag-of-words"""
        train, test = self.split_blogs_corpus(BagOfWords)
        classifier = NaiveBayes("basic_NB.model")
        classifier.train(train)
        self.assertGreater(accuracy(classifier, test), 0.55)
        # print(f_measure_aggregate(classifier, test, "F"))
        # print(f_measure_aggregate(classifier, test, "M"))

    def split_blogs_corpus_imba(self, document_class):
        blogs = BlogsCorpus(document_class=document_class)
        imba_blogs = blogs.split_imbalance()
        return (imba_blogs[:1600], imba_blogs[1600:])

    def test_blogs_imba(self):
        train, test = self.split_blogs_corpus_imba(BagOfWords)
        classifier = NaiveBayes()
        classifier.train(train)
        # you don't need to pass this test
        self.assertGreater(accuracy(classifier, test), 0.1)


if __name__ == '__main__':
    # Run all of the tests, print the results, and exit.
    main(verbosity=5)
