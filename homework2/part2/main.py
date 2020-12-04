# Code to specify and train the TensorFlow graph
import sys
import os
import numpy as np
import tensorflow as tf
import model
import util

def ml_driver():
    ae_path_prefix = train_autoencoder()
    train_classifier(ae_path_prefix)
    
def train_autoencoder():
    ae = model.AutoEncoder()

    imagenet_im_num = ae.imagenet_images.shape[0]
    autoencoder_train_op = ae.autoencoder_train_op()
    batch_size = ae.batch_size
    
    print("AUTOENCODER TRAINING...")

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        for i in range(imagenet_im_num // batch_size):
            batch_xs = ae.imagenet_images[i * batch_size:(i + 1) * batch_size, :]

            session.run(
                [autoencoder_train_op], {
                    ae.x: batch_xs
                })

        ae_path_prefix = ae.saver.save(
            session,
            os.path.join(ae.args.auto_encoder_dir , "homework_2"))

    print("AUTOENCODER TRAINED")
    return ae_path_prefix

def train_classifier(ae_path_prefix):
    tf.reset_default_graph()
    
    params = {
        'path_prefix': ae_path_prefix,
        "starter_learning_rate": .001,
        "reduction_rate": 0.70,
        "reg_scale": 0.0001,
        "batch_size": 128,
    }
    m = model.Model(params)

    # split into train, validation, and test
    train_im_num = m.train_images.shape[0]
    test_im_num = m.test_images.shape[0]
    val_im_num = m.validation_images.shape[0]

    mean_cross_entropy = tf.reduce_mean(m.cross_entropy)

    # Define network operations
    network_train_op = m.network_train_op()
    confusion_matrix_op = m.confusion_matrix_op()
    
    batch_size = m.batch_size
    
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        best_val_ce = sys.maxsize
        counter = 0
        avg_val_cs = 0

        for epoch in range(m.epochs):
            print('Epoch: ' + str(epoch))
            # shuffle data each epoch before training
            train_images, train_labels = util.shuffler(m.train_images, m.train_labels)

            # run gradient steps and report mean loss on train data
            ce_vals = []
            for i in range(train_im_num // batch_size):
                batch_xs = train_images[i * batch_size:(i + 1) * batch_size, :]
                batch_ys = np.squeeze(train_labels[i * batch_size:(i + 1) * batch_size, :])
                _, train_ce = session.run(
                    [network_train_op, mean_cross_entropy], {
                        m.ae_x: batch_xs,
                        m.y: batch_ys
                    })
                ce_vals.append(train_ce)
            avg_train_ce = sum(ce_vals) / len(ce_vals)
            print('TRAIN CROSS ENTROPY: ' + str(avg_train_ce))

            # report mean validation loss
            ce_vals = []
            conf_mxs = []
            for i in range(val_im_num // batch_size):
                batch_xs = m.validation_images[i * batch_size:(i + 1) * batch_size, :]
                batch_ys = np.squeeze(m.validation_labels[i * batch_size:(i + 1) * batch_size, :])
                val_ce, conf_matrix = session.run(
                    [mean_cross_entropy, confusion_matrix_op], {
                        m.ae_x: batch_xs,
                        m.y: batch_ys
                    })
                ce_vals.append(val_ce)
                conf_mxs.append(conf_matrix)

            avg_val_ce = sum(ce_vals) / len(ce_vals)
            print('VALIDATION CROSS ENTROPY: ' + str(avg_val_ce))
            print('VALIDATION ACCURACY: ' + str(sum(sum(conf_mxs).diagonal()) / val_im_num))
            print('VALIDATION CONFUSION MATRIX:')
            print(str(sum(conf_mxs)))

            # report mean test loss
            ce_vals = []
            conf_mxs = []
            for i in range(test_im_num // batch_size):
                batch_xs = m.test_images[i * batch_size:(i + 1) * batch_size, :]
                batch_ys = np.squeeze(m.test_labels[i * batch_size:(i + 1) * batch_size, :])

                test_ce, conf_matrix = session.run(
                    [mean_cross_entropy, confusion_matrix_op], {
                        m.ae_x: batch_xs,
                        m.y: batch_ys
                    })
                ce_vals.append(test_ce)
                conf_mxs.append(conf_matrix)

            avg_test_ce = sum(ce_vals) / len(ce_vals)
            print('TEST CROSS ENTROPY: ' + str(avg_test_ce))
            print('TEST ACCURACY: ' + str(sum(sum(conf_mxs).diagonal()) / test_im_num))

            if avg_val_ce > best_val_ce:
                counter += 1
            elif avg_val_ce < best_val_ce:
                best_val_ce = avg_val_ce
                counter = 0

            if counter > m.early_stopping:
                break

            path_prefix = m.saver.save(
                session,
                os.path.join(m.args.model_dir, "homework_2"),
                global_step=m.global_step_tensor)

ml_driver()
