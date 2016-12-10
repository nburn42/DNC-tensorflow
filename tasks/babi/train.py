"""
Code related to setting up bAbi is originally from
https://github.com/domluna/memn2n/blob/master/joint.py
"""
import warnings
warnings.filterwarnings('ignore')

from sklearn import cross_validation, metrics
from itertools import chain

import tensorflow as tf
import numpy as np
import getopt
import sys
import os

from dnc.dnc import DNC
from feedforward_controller import FeedforwardController
from data_utils import load_task, vectorize_data

def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()

def binary_cross_entropy(predictions, targets):

    return tf.reduce_mean(
        -1 * targets * tf.log(predictions) - (1 - targets) * tf.log(1 - predictions)
    )


if __name__ == '__main__':

    dirname = os.path.dirname(__file__)
    ckpts_dir = os.path.join(dirname , 'checkpoints')
    tb_logs_dir = os.path.join(dirname, 'logs')
    data_dir = os.path.join(dirname, 'data/tasks_1-20_v1-2/en/')

    batch_size = 1
    words_count = 15
    word_size = 10
    read_heads = 1

    learning_rate = 1e-4
    momentum = 0.9

    from_checkpoint = None
    iterations = 100000

    options,_ = getopt.getopt(sys.argv[1:], '', ['checkpoint=', 'iterations=','data_dir='])

    for opt in options:
        if opt[0] == '--checkpoint':
            from_checkpoint = opt[1]
        elif opt[0] == '--iterations':
            iterations = int(opt[1])
        elif opt[0] == '--data_dir':
            data_dir = os.path.join(dirname, opt[1])

    # load all train/test data
    ids = range(1, 21)
    train_data = []
    test_data = []

    for i in ids:
        tr, te = load_task(data_dir, i)
        train_data.append(tr)
        test_data.append(te)

    data = list(chain.from_iterable(train_data + test_data))

    vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data)))
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

    max_story_size = max(map(len, (s for s, _, _ in data)))
    mean_story_size = int(np.mean([ len(s) for s, _, _ in data ]))
    sequence_max_length = max(map(len, chain.from_iterable(s for s, _, _ in data)))
    query_size = max(map(len, (q for _, q, _ in data)))
    vocab_size = len(word_idx) + 1 # +1 for nil word
    sequence_max_length = max(query_size, sequence_max_length) # for the position
    memory_size = min(2 * sequence_max_length + 1, max_story_size)
    input_size = output_size = vocab_size - 1

    print("Vocabulary size", vocab_size)
    print("Longest sentence length", sequence_max_length)
    print("Longest story length", max_story_size)
    print("Average story length", mean_story_size)

    # train/validation/test sets
    trainS = []
    valS = []
    trainQ = []
    valQ = []
    trainA = []
    valA = []

    for task in train_data:
            S, Q, A = vectorize_data(task, word_idx, sequence_max_length, memory_size)
            ts, vs, tq, vq, ta, va = cross_validation.train_test_split(S, Q, A, test_size=0.1)
            trainS.append(ts)
            trainQ.append(tq)
            trainA.append(ta)
            valS.append(vs)
            valQ.append(vq)
            valA.append(va)

    print "S", S
    print "Q",Q
    print "A",A

    trainS = reduce(lambda a,b : np.vstack((a,b)), (x for x in trainS))
    trainQ = reduce(lambda a,b : np.vstack((a,b)), (x for x in trainQ))
    trainA = reduce(lambda a,b : np.vstack((a,b)), (x for x in trainA))
    valS = reduce(lambda a,b : np.vstack((a,b)), (x for x in valS))
    valQ = reduce(lambda a,b : np.vstack((a,b)), (x for x in valQ))
    valA = reduce(lambda a,b : np.vstack((a,b)), (x for x in valA))

    testS, testQ, testA = vectorize_data(list(chain.from_iterable(test_data)), word_idx, sequence_max_length, memory_size)

    n_train = trainS.shape[0]
    n_val = valS.shape[0]
    n_test = testS.shape[0]

    print("Training Size", n_train)
    print("Validation Size", n_val)
    print("Testing Size", n_test)

    print(trainS.shape, valS.shape, testS.shape)
    print(trainQ.shape, valQ.shape, testQ.shape)
    print(trainA.shape, valA.shape, testA.shape)

    train_labels = np.argmax(trainA, axis=1)
    test_labels = np.argmax(testA, axis=1)
    val_labels = np.argmax(valA, axis=1)

    # This avoids feeding 1 task after another, instead each batch has a random sampling of tasks
    batches = zip(range(0, n_train-batch_size, batch_size), range(batch_size, n_train, batch_size))
    batches = [(start, end) for start,end in batches]
    
    print batches[0]

    graph = tf.Graph()

    with graph.as_default():
        with tf.Session(graph=graph) as session:

            llprint("Building Computational Graph ... ")

            optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=momentum)
            summerizer = tf.train.SummaryWriter(tb_logs_dir, session.graph)

            ncomputer = DNC(
                FeedforwardController,
                input_size,
                output_size,
                memory_size,
                words_count,
                word_size,
                read_heads,
                batch_size
            )

            # squash the DNC output between 0 and 1
            output, _ = ncomputer.get_outputs()
            squashed_output = tf.clip_by_value(tf.sigmoid(output), 1e-6, 1. - 1e-6)

            loss = binary_cross_entropy(squashed_output, ncomputer.target_output)

            gradients = optimizer.compute_gradients(loss)
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    #with tf.control_dependencies([tf.Print(tf.zeros(1), [var.name, tf.is_nan(grad)])]):
                    gradients[i] = (tf.clip_by_value(grad, -10, 10), var)

            apply_gradients = optimizer.apply_gradients(gradients)

            summerize_loss = tf.scalar_summary("Loss", loss)

            summerize_op = tf.merge_summary([summerize_loss])
            no_summerize = tf.no_op()

            llprint("Done!\n")

            llprint("Initializing Variables ... ")
            session.run(tf.initialize_all_variables())
            llprint("Done!\n")

            if from_checkpoint is not None:
                llprint("Restoring Checkpoint %s ... " % (from_checkpoint))
                ncomputer.restore(session, ckpts_dir, from_checkpoint)
                llprint("Done!\n")


            last_100_losses = []

            for i in xrange(iterations + 1):
                llprint("\rIteration %d/%d" % (i, iterations))

                random_batch_index = np.random.randint(0, len(batches) + 1)
                s = trainS[start:end]
                q = trainQ[start:end]
                a = trainA[start:end]

                summerize = (i % 100 == 0)
                take_checkpoint = (i != 0) and (i % iterations == 0)

                loss_value, _, summary = session.run([
                    loss,
                    apply_gradients,
                    summerize_op if summerize else no_summerize
                ], feed_dict={
                    ncomputer.input_data: input_data,
                    ncomputer.target_output: a,
                    ncomputer.sequence_length: 2 * random_length + 1
                })

                last_100_losses.append(loss_value)
                summerizer.add_summary(summary, i)

                if summerize:
                    llprint("\n\tAvg. Logistic Loss: %.4f\n" % (np.mean(last_100_losses)))
                    last_100_losses = []

                if take_checkpoint:
                    llprint("\nSaving Checkpoint ... "),
                    ncomputer.save(session, ckpts_dir, 'step-%d' % (i))
                    llprint("Done!\n")
