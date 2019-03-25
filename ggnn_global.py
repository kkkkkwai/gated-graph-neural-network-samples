#!/usr/bin/env/python

from typing import Tuple, List, Any, Sequence

import tensorflow as tf
import time
import os
import json
import numpy as np
import pickle
import pylab as plt
import random

from utils import MLP, ThreadedIterator, SMALL_NUMBER, EDGE_TYPE, seenProjects, unseenProjects


class ProgramModel(object):
    @classmethod
    def default_params(cls):
        return {
            'num_epochs': 3000,
            'patience': 25,
            'learning_rate': 0.001,
            'clamp_gradient_norm': 1.0,
            'out_layer_dropout_keep_prob': 1.0,

            'word_embedding_size': 25,
            'type_embedding_size': 25,
            'max_node_length': 25,
            'hidden_size': 100,
            'num_timesteps': 4,
            'use_graph': True,

            'tie_fwd_bkwd': False,

            'random_seed': 0,
            'chart_enabled': True
        }

    def __init__(self, args):
        self.args = args

        # Collect argument things:

        self.run_id = "_".join([time.strftime("%Y-%m-%d-%H-%M-%S"), str(os.getpid())])
        self.log_dir = args.get('--log_dir') or os.path.expanduser('~/log')
        self.log_file = os.path.join(self.log_dir, "%s_log.json" % self.run_id)
        self.best_model_file = os.path.join(self.log_dir, "%s_model_best.pickle" % self.run_id)

        # self.project_name = 'ninject'
        # self.data_path = os.path.expanduser('~/Downloads/graph-dataset')
        # Specify data path:
        self.data_path = os.path.expanduser('~/project-data/graph-dataset')
        self.train_data_path = []
        # Collect seen projects train data files
        for project_name in seenProjects:
            t_path = os.path.join(self.data_path, project_name, 'graphs-train')
            for idx, filename in enumerate(os.listdir(t_path)):
                if not filename.endswith('json'):
                    self.train_data_path.append(os.path.join(t_path, filename))
        print(self.train_data_path)

        self.type_path = os.path.join(self.data_path, 'typehierarchy.json')
        self.type_hierarchy = []
        with open(self.type_path, 'r') as f:
            self.type_hierarchy.append(json.load(f))

        self.vocabs = []
        with open(os.path.join(self.data_path, 'vocabulary.json')) as f:
            self.vocabs = json.load(f)['vocabulary']

        # Collect parameters:
        params = self.default_params()
        config_file = args.get('--config-file')
        if config_file is not None:
            with open(config_file, 'r') as f:
                params.update(json.load(f))
        config = args.get('--config')
        if config is not None:
            params.update(json.loads(config))
        self.params = params
        with open(os.path.join(self.log_dir, "%s_params.json" % self.run_id), "w") as f:
            json.dump(params, f)
        print("Run %s starting with following parameters:\n%s" % (self.run_id, json.dumps(self.params)))
        random.seed(params['random_seed'])
        np.random.seed(params['random_seed'])

        # Load edge type:
        self.num_original_edge_types = len(EDGE_TYPE)
        if not self.params['tie_fwd_bkwd']:
            self.num_edge_types = self.num_original_edge_types * 2
        else:
            self.num_edge_types = self.num_original_edge_types

        # Build the actual model
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=config)
        with self.graph.as_default():
            tf.set_random_seed(params['random_seed'])
            self.placeholders = {}
            self.weights = {}
            self.ops = {}
            self.make_model()
            self.make_train_step()

            # Restore/initialize variables:
            restore_file = args.get('--restore')
            if restore_file is not None:
                self.restore_model(restore_file)
            else:
                self.initialize_model()

    def load_data(self, dir_path, filename, is_training: bool):
        if filename is not None:
            dir_path = os.path.join(dir_path, filename)
        with open(dir_path, 'r') as f:
            print('Loading {} {}'.format(dir_path, filename), end='\r')
            data = json.load(f)
        restrict = self.args.get("--restrict_data")
        if restrict is not None and restrict > 0:
            data = data[:restrict]

        return self.process_raw_graphs(data, is_training, filename)

    @staticmethod
    def graph_string_to_array(graph_string: str) -> List[List[int]]:
        return [[int(v) for v in s.split(' ')]
                for s in graph_string.split('\n')]

    def process_raw_graphs(self, raw_data: Sequence[Any], is_training: bool, _filename) -> Any:
        raise Exception("Models have to implement process_raw_graphs!")

    def make_model(self):
        self.placeholders['target_values'] = tf.placeholder(tf.float32, [None],
                                                            name='target_values')
        # self.placeholders['target_mask'] = tf.placeholder(tf.float32, [None],
        #                                                   name='target_mask')
        self.placeholders['num_graphs'] = tf.placeholder(tf.int32, [], name='num_graphs')
        self.placeholders['out_layer_dropout_keep_prob'] = tf.placeholder(tf.float32, [],
                                                                          name='out_layer_dropout_keep_prob')

        with tf.variable_scope("graph_model"):
            self.prepare_specific_graph_model()
            # This does the actual graph work:
            self.ops['final_node_representations'] = self.compute_final_node_representations()

        with tf.variable_scope("out_layer_task"):
            with tf.variable_scope("regression"):
                self.weights['regression_transform_task'] = MLP(2 * self.params['hidden_size'], 1, [],
                                                                self.placeholders['out_layer_dropout_keep_prob'])
            accuracy, task_loss = self.regression(self.ops['final_node_representations'],
                                                  self.weights['regression_transform_task'])
            # task_target_mask = self.placeholders['target_mask'][internal_id, :]
            # task_target_num = tf.reduce_sum(task_target_mask) + SMALL_NUMBER
            # diff = diff * task_target_mask  # Mask out unused values
            self.ops['accuracy_task'] = accuracy
            # Normalise loss to account for fewer task-specific examples in batch:
            task_loss = task_loss * (1.0 / self.params['task_sample_ratios'])
            self.ops['loss'] = task_loss

    def make_train_step(self):
        trainable_vars = self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        if self.args.get('--freeze-graph-model'):
            graph_vars = set(self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="graph_model"))
            filtered_vars = []
            for var in trainable_vars:
                if var not in graph_vars:
                    filtered_vars.append(var)
                else:
                    print("Freezing weights of variable %s." % var.name)
            trainable_vars = filtered_vars
        optimizer = tf.train.AdamOptimizer(self.params['learning_rate'])
        grads_and_vars = optimizer.compute_gradients(self.ops['loss'], var_list=trainable_vars)
        clipped_grads = []
        for grad, var in grads_and_vars:
            if grad is not None:
                clipped_grads.append((tf.clip_by_norm(grad, self.params['clamp_gradient_norm']), var))
            else:
                clipped_grads.append((grad, var))
        self.ops['train_step'] = optimizer.apply_gradients(clipped_grads)
        # Initialize newly-introduced variables:
        self.sess.run(tf.local_variables_initializer())

    def regression(self, last_h, regression_transform):
        raise Exception("Models have to implement gated_regression!")

    def prepare_specific_graph_model(self) -> None:
        raise Exception("Models have to implement prepare_specific_graph_model!")

    def compute_final_node_representations(self) -> tf.Tensor:
        raise Exception("Models have to implement compute_final_node_representations!")

    def make_minibatch_iterator(self, data: Any, is_training: bool):
        raise Exception("Models have to implement make_minibatch_iterator!")

    def run_epoch(self, epoch_name: str, filename: str, data, is_training: bool):
        loss = 0
        accuracies = []
        accuracy_ops = self.ops['accuracy_task']
        start_time = time.time()
        processed_graphs = 0
        batch_iterator = ThreadedIterator(self.make_minibatch_iterator(data, is_training), max_queue_size=5)
        for step, batch_data in enumerate(batch_iterator):
            num_graphs = batch_data[self.placeholders['num_graphs']]
            processed_graphs += num_graphs
            if is_training:
                batch_data[self.placeholders['out_layer_dropout_keep_prob']] = self.params[
                    'out_layer_dropout_keep_prob']
                fetch_list = [self.ops['loss'], accuracy_ops, self.ops['train_step']]
            else:
                batch_data[self.placeholders['out_layer_dropout_keep_prob']] = 1.0
                fetch_list = [self.ops['loss'], accuracy_ops]
            result = self.sess.run(fetch_list, feed_dict=batch_data)
            (batch_loss, batch_accuracies) = (result[0], result[1])
            loss += batch_loss * num_graphs
            accuracies.append(np.array(batch_accuracies) * num_graphs)

            print("Running %s, file %s batch %i (has %i graphs). Loss so far: %.4f" % (epoch_name,
                                                                                       filename,
                                                                                       step,
                                                                                       num_graphs,
                                                                                       loss / processed_graphs),
                  end='\r')

        accuracies = np.sum(accuracies) / processed_graphs
        loss = loss / processed_graphs
        time_took = processed_graphs / (time.time() - start_time)
        return loss, accuracies, time_took, processed_graphs

    def train(self):
        log_to_save = []
        train_losses_series = []
        train_accs_series = []
        valid_losses_series = []
        valid_accs_series = []
        unseen_losses_series = []
        unseen_accs_series = []
        total_time_start = time.time()
        with self.graph.as_default():
            if self.args.get('--restore') is not None:
                valid_accs, num_graphs_sum = [], 0
                for project_name in seenProjects:
                    data_path = os.path.join(self.data_path, project_name, 'graphs-valid')
                    for idx, filename in enumerate(os.listdir(data_path)):
                        if not filename.endswith('json'):
                            self.valid_data = self.load_data(data_path, filename, False)
                            _, valid_acc, _, num_graphs = self.run_epoch("Resumed (validation)", filename, self.valid_data, False)
                            valid_acc = valid_acc * num_graphs
                            valid_accs.append(valid_acc)
                            num_graphs_sum = num_graphs_sum + num_graphs
                best_val_acc = np.sum(valid_accs) / num_graphs_sum
                best_val_acc_epoch = 0
                print("\r\x1b[KResumed operation, initial cum. val. acc: %.5f" % best_val_acc)
            else:
                (best_val_acc, best_val_acc_epoch) = (0, 0)

            for epoch in range(1, self.params['num_epochs'] + 1):
                print("== Epoch %i" % epoch)
                train_losses, train_accs, train_times, num_graphs_sum = [], [], 0, 0
                # Load train data one by one
                np.random.shuffle(self.train_data_path)
                for dir_path in self.train_data_path:
                # for project_name in seenProjects:
                #     data_path = os.path.join(self.data_path, project_name, 'graphs-train')
                #     for idx, filename in enumerate(os.listdir(data_path)):
                #         if not filename.endswith('json'):
                    self.train_data = self.load_data(dir_path, None, True)
                    train_loss, train_acc, train_time, num_graphs = self.run_epoch(
                        "epoch %i (training)" % epoch,
                        dir_path,
                        self.train_data,
                        True)
                    # for calculating epoch loss and acc
                    train_losses.append(train_loss * num_graphs)
                    train_accs.append(train_acc * num_graphs)
                    train_times += train_time
                    num_graphs_sum += num_graphs
                train_loss = np.sum(train_losses) / num_graphs_sum
                train_acc = np.sum(train_accs) / num_graphs_sum
                train_speed = num_graphs_sum / train_times
                # for drawing chart
                train_losses_series.append(train_loss)
                train_accs_series.append(train_acc)
                # accs_str = " ".join(["%i:%.5f" % (id, acc) for (id, acc) in zip(self.params['task_ids'], train_accs)])
                print("\r\x1b[K Train: loss: %.5f | acc: %s | instances/sec: %.2f" % (train_loss,
                                                                                      train_acc,
                                                                                      train_speed))

                valid_losses, valid_accs, valid_times, num_graphs_sum = [], [], 0, 0
                # Load valid data one by one
                for project_name in seenProjects:
                    data_path = os.path.join(self.data_path, project_name, 'graphs-valid')
                    for idx, filename in enumerate(os.listdir(data_path)):
                        if not filename.endswith('json'):
                            self.valid_data = self.load_data(data_path, filename, False)
                            valid_loss, valid_acc, valid_time, num_graphs = self.run_epoch(
                                "epoch %i (validation)" % epoch,
                                filename,
                                self.valid_data,
                                False)
                            # for calculating all loss
                            valid_losses.append(valid_loss * num_graphs)
                            valid_accs.append(valid_acc * num_graphs)
                            valid_times += valid_time
                            num_graphs_sum += num_graphs
                valid_loss = np.sum(valid_losses) / num_graphs_sum
                valid_acc = np.sum(valid_accs) / num_graphs_sum
                valid_speed = num_graphs_sum / valid_times
                # for drawing chart
                valid_losses_series.append(valid_loss)
                valid_accs_series.append(valid_acc)
                # accs_str = " ".join(["%i:%.5f" % (id, acc) for (id, acc) in zip(self.params['task_ids'], valid_accs)])
                print("\r\x1b[K Valid: loss: %.5f | acc: %s | instances/sec: %.2f" % (valid_loss,
                                                                                      valid_acc,
                                                                                      valid_speed))

                unseen_losses, unseen_accs, unseen_times, num_graphs_sum = [], [], 0, 0
                # Load valid unseen data one by one
                for project_name in unseenProjects:
                    data_path = os.path.join(self.data_path, project_name, 'graphs')
                    for idx, filename in enumerate(os.listdir(data_path)):
                        if not filename.endswith('json'):
                            self.valid_data = self.load_data(data_path, filename, False)
                            unseen_loss, unseen_acc, unseen_time, num_graphs = self.run_epoch(
                                "epoch %i (unseen_test)" % epoch,
                                filename,
                                self.valid_data,
                                False)
                            # for calculating all loss
                            unseen_losses.append(unseen_loss * num_graphs)
                            unseen_accs.append(unseen_acc * num_graphs)
                            unseen_times += unseen_time
                            num_graphs_sum += num_graphs
                unseen_loss = np.sum(unseen_losses) / num_graphs_sum
                unseen_acc = np.sum(unseen_accs) / num_graphs_sum
                unseen_speed = num_graphs_sum / unseen_times
                # for drawing chart
                unseen_losses_series.append(unseen_loss)
                unseen_accs_series.append(unseen_acc)
                # accs_str = " ".join(["%i:%.5f" % (id, acc) for (id, acc) in zip(self.params['task_ids'], valid_accs)])
                print("\r\x1b[K Unseen: loss: %.5f | acc: %s | instances/sec: %.2f" % (unseen_loss,
                                                                                       unseen_acc,
                                                                                       unseen_speed))
                epoch_time = time.time() - total_time_start
                log_entry = {
                    'project': 'global',
                    'epoch': epoch,
                    'time': epoch_time,
                    'train_results': (train_loss, train_acc, train_speed),
                    'valid_results': (valid_loss, valid_acc, valid_speed),
                    'unseen_results': (unseen_loss, unseen_acc, unseen_speed)
                }
                log_to_save.append(log_entry)
                with open(self.log_file, 'w') as f:
                    json.dump(log_to_save, f, indent=4)

                if self.params['chart_enabled']:
                    self.plot_chart(epoch, train_losses_series, train_accs_series,
                                    valid_losses_series, valid_accs_series,
                                    unseen_losses_series, unseen_accs_series)

                if valid_acc > best_val_acc:
                    self.save_model(self.best_model_file)
                    print("  (Best epoch so far, cum. val. acc decreased to %.5f from %.5f. Saving to '%s')" % (
                        valid_acc, best_val_acc, self.best_model_file))
                    best_val_acc = valid_acc
                    best_val_acc_epoch = epoch
                # elif epoch - best_val_acc_epoch >= self.params['patience']:
                #     print("Stopping training after %i epochs without improvement on validation accuracy." % self.params[
                #         'patience'])
                if epoch >= 20:
                    print('Stop after 20 epochs')
                    break

    def save_model(self, path: str) -> None:
        weights_to_save = {}
        for variable in self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            assert variable.name not in weights_to_save
            weights_to_save[variable.name] = self.sess.run(variable)

        data_to_save = {
            "params": self.params,
            "weights": weights_to_save
        }

        with open(path, 'wb') as out_file:
            pickle.dump(data_to_save, out_file, pickle.HIGHEST_PROTOCOL)

    def initialize_model(self) -> None:
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        self.sess.run(init_op)

    def restore_model(self, path: str) -> None:
        print("Restoring weights from file %s." % path)
        with open(path, 'rb') as in_file:
            data_to_load = pickle.load(in_file)

        # Assert that we got the same model configuration
        assert len(self.params) == len(data_to_load['params'])
        for (par, par_value) in self.params.items():
            # Fine to have different run epochs:
            if par not in ['num_epochs']:
                assert par_value == data_to_load['params'][par]

        variables_to_initialize = []
        with tf.name_scope("restore"):
            restore_ops = []
            used_vars = set()
            for variable in self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                used_vars.add(variable.name)
                if variable.name in data_to_load['weights']:
                    restore_ops.append(variable.assign(data_to_load['weights'][variable.name]))
                else:
                    print('Freshly initializing %s since no saved value was found.' % variable.name)
                    variables_to_initialize.append(variable)
            for var_name in data_to_load['weights']:
                if var_name not in used_vars:
                    print('Saved weights for %s not used by model.' % var_name)
            restore_ops.append(tf.variables_initializer(variables_to_initialize))
            self.sess.run(restore_ops)

    def plot_chart(self, epoch: int, train_losses_series, train_accs_series,
                   valid_losses_series, valid_accs_series,
                   unseen_losses_series, unseen_accs_series):
        plt.figure()
        plt.plot(range(epoch), train_losses_series, 'r', label='Loss')
        plt.plot(range(epoch), train_accs_series, 'b', label='Accuracy')
        plt.title('Train Loss and Accuracy')
        plt.xlabel('Epochs')
        plt.legend(loc='best')
        plt.savefig(os.path.join(self.log_dir, '{}_train.png'.format(self.run_id)))
        plt.close()
        # plot validation data
        plt.figure()
        plt.plot(range(epoch), valid_losses_series, 'r', label='Loss')
        plt.plot(range(epoch), valid_accs_series, 'b', label='Accuracy')
        plt.title('Validation Loss and Accuracy')
        plt.xlabel('Epochs')
        plt.legend(loc='best')
        plt.savefig(os.path.join(self.log_dir, '{}_valid.png'.format(self.run_id)))
        plt.close()
        # plot unseen data
        plt.figure()
        plt.plot(range(epoch), unseen_losses_series, 'r', label='Loss')
        plt.plot(range(epoch), unseen_accs_series, 'b', label='Accuracy')
        plt.title('Unseen-project Loss and Accuracy')
        plt.xlabel('Epochs')
        plt.legend(loc='best')
        plt.savefig(os.path.join(self.log_dir, '{}_unseen.png'.format(self.run_id)))
        plt.close()
