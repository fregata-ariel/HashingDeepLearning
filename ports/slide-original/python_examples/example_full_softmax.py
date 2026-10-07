from __future__ import annotations

import glob
import math
import os
import time

import numpy as np
import tensorflow as tf

from config import config
from util import data_generator, data_generator_tst


def main() -> None:
    """Run the legacy TensorFlow 1.x full-softmax baseline example."""
    feature_dim = config.feature_dim
    n_classes = config.n_classes
    hidden_dim = config.hidden_dim
    n_train = config.n_train
    n_test = config.n_test
    n_epochs = config.n_epochs
    batch_size = config.batch_size
    lr = config.lr
    num_threads = config.num_threads

    if config.GPUs != "":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = config.GPUs

    train_files = glob.glob(config.data_path_train)
    test_files = glob.glob(config.data_path_test)

    x_idxs = tf.placeholder(tf.int64, shape=[None, 2])
    x_vals = tf.placeholder(tf.float32, shape=[None])
    x = tf.SparseTensor(
        x_idxs, x_vals, [batch_size, feature_dim]
    )
    y = tf.placeholder(
        tf.float32, shape=[None, n_classes]
    )

    W1 = tf.Variable(
        tf.truncated_normal(
            [feature_dim, hidden_dim],
            stddev=2.0 / math.sqrt(feature_dim + hidden_dim),
        )
    )
    b1 = tf.Variable(
        tf.truncated_normal(
            [hidden_dim],
            stddev=2.0 / math.sqrt(feature_dim + hidden_dim),
        )
    )
    layer_1 = tf.nn.relu(
        tf.sparse_tensor_dense_matmul(x, W1) + b1
    )

    W2 = tf.Variable(
        tf.truncated_normal(
            [hidden_dim, n_classes],
            stddev=2.0 / math.sqrt(hidden_dim + n_classes),
        )
    )
    b2 = tf.Variable(
        tf.truncated_normal(
            [n_classes],
            stddev=2.0 / math.sqrt(
                n_classes + hidden_dim
            ),
        )
    )
    logits = tf.matmul(layer_1, W2) + b2

    k = 1
    if k == 1:
        top_idxs = tf.argmax(logits, axis=1)
    else:
        top_idxs = tf.nn.top_k(
            logits, k=k, sorted=False
        )[1]

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=y
        )
    )
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)

    if config.GPUs == "":
        session_config = tf.ConfigProto(
            inter_op_parallelism_threads=num_threads,
            intra_op_parallelism_threads=num_threads,
        )
    else:
        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True

    sess = tf.Session(config=session_config)
    sess.run(tf.global_variables_initializer())

    training_data_generator = data_generator(
        train_files, batch_size, n_classes
    )
    steps_per_epoch = n_train // batch_size
    n_steps = n_epochs * steps_per_epoch
    n_check = 50

    begin_time = time.time()
    total_time = 0.0

    with open(config.log_file, "a", encoding="utf-8") as out:
        for step in range(n_steps):
            if step % n_check == 0:
                total_time += time.time() - begin_time
                print(
                    "Finished ",
                    step,
                    " steps. Time elapsed for last",
                    n_check,
                    "batches = ",
                    time.time() - begin_time,
                )
                test_data_generator = data_generator_tst(
                    test_files, batch_size
                )
                tmp_k = 0.0
                for _ in range(20):
                    idxs_batch, vals_batch, labels_batch = next(
                        test_data_generator
                    )
                    top_k_classes = sess.run(
                        top_idxs,
                        feed_dict={
                            x_idxs: idxs_batch,
                            x_vals: vals_batch,
                        },
                    )
                    tmp_k += float(
                        np.mean(
                            [
                                len(
                                    np.intersect1d(
                                        top_k_classes[row],
                                        labels_batch[row],
                                    )
                                )
                                / min(
                                    k,
                                    len(labels_batch[row]),
                                )
                                for row in range(
                                    len(top_k_classes)
                                )
                            ]
                        )
                    )
                print("test_acc: ", tmp_k / 20)
                print("#######################")
                print(
                    step,
                    int(total_time),
                    tmp_k / 20,
                    file=out,
                )
                begin_time = time.time()

            idxs_batch, vals_batch, labels_batch = next(
                training_data_generator
            )
            sess.run(
                train_step,
                feed_dict={
                    x_idxs: idxs_batch,
                    x_vals: vals_batch,
                    y: labels_batch,
                },
            )

            if step % steps_per_epoch == steps_per_epoch - 1:
                total_time += time.time() - begin_time
                print(
                    "Finished ",
                    step,
                    " steps. Time elapsed for epoch tail = ",
                    time.time() - begin_time,
                )
                n_steps_val = n_test // batch_size
                test_data_generator = data_generator_tst(
                    test_files, batch_size
                )
                num_batches = 0
                p_at_k = 0.0
                for _ in range(n_steps_val):
                    idxs_batch, vals_batch, labels_batch = next(
                        test_data_generator
                    )
                    top_k_classes = sess.run(
                        top_idxs,
                        feed_dict={
                            x_idxs: idxs_batch,
                            x_vals: vals_batch,
                        },
                    )
                    p_at_k += float(
                        np.mean(
                            [
                                len(
                                    np.intersect1d(
                                        top_k_classes[row],
                                        labels_batch[row],
                                    )
                                )
                                / min(
                                    k,
                                    len(labels_batch[row]),
                                )
                                for row in range(
                                    len(top_k_classes)
                                )
                            ]
                        )
                    )
                    num_batches += 1

                print(
                    "Overall p_at_1 after ",
                    num_batches,
                    "batches = ",
                    p_at_k / num_batches,
                )
                print(
                    step,
                    int(total_time),
                    p_at_k / num_batches,
                    file=out,
                )
                begin_time = time.time()


if __name__ == "__main__":
    main()
