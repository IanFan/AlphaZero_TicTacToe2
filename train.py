# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gomoku

@author: Junxiao Song
"""

from __future__ import print_function
import random
import numpy as np
from collections import defaultdict, deque

from AlphaZero_TicTacToe2.game import Board, Game
from AlphaZero_TicTacToe2.mcts_pure import MCTSPlayer as MCTS_Pure
from AlphaZero_TicTacToe2.mcts_alphaZero import MCTSPlayer

# from AlphaZero_TicTacToe2.policy_value_net import PolicyValueNet  # Theano and Lasagne
# from AlphaZero_TicTacToe2.policy_value_net_pytorch import PolicyValueNet  # Pytorch
from AlphaZero_TicTacToe2.policy_value_net_tensorflow import PolicyValueNet # Tensorflow
# from AlphaZero_TicTacToe2.policy_value_net_keras import PolicyValueNet # Keras

class TrainPipeline():
    def __init__(self, init_model=None):
        # params of the board and the game
        self.board_width = 3
        self.board_height = 3
        self.n_in_row = 3
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)
        # training params
        self.learn_rate = 0.002
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 400 #400  # num of simulations for each move
        self.c_puct = 5
        self.buffer_size = 1000 #10000
        self.batch_size = 256 #512  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 3 #5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 150 #1500
        self.best_win_ratio = 0.0
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1000 #1000
        if init_model:
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   model_file=init_model)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height)
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        for i in range(n_games):
            con_set = set([-1, 1, 2])
            self.p1_win_con = random.sample(con_set, 1)[0]
            self.p2_win_con = random.sample(con_set, 1)[0]
            winner, play_data = self.game.start_self_play(self.mcts_player,
                                                          temp=self.temp,
                                                          p1_win_con=self.p1_win_con,
                                                          p2_win_con=self.p2_win_con
                                                          )
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # augment the data
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    self.learn_rate*self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy

    def policy_evaluate(self, n_games=18):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)
        pure_mcts_player = MCTS_Pure(c_puct=5,
                                     n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        win_cnt_move_first = defaultdict(int)
        win_cnt_move_second = defaultdict(int)
        win_cnt_move_first_dict = defaultdict(int)
        win_cnt_move_second_dict = defaultdict(int)
        for i in range(n_games):
            con_set = set([-1, 1, 2])
            p1_win_con = random.sample(con_set, 1)[0]
            p2_win_con = random.sample(con_set, 1)[0]
            if i==0 or i==1:
                p1_win_con = -1
                p2_win_con = -1
            elif i==2 or i==3:
                p1_win_con = 1
                p2_win_con = 1
            elif i==4 or i==5:
                p1_win_con = 2
                p2_win_con = 2
            elif i==6 or i==7:
                p1_win_con = 1
                p2_win_con = -1
            elif i==8 or i==9:
                p1_win_con = 2
                p2_win_con = -1
            elif i==10 or i==11:
                p1_win_con = -1
                p2_win_con = 1
            elif i==12 or i==13:
                p1_win_con = -1
                p2_win_con = 2
            elif i==14 or i==15 :
                p1_win_con = 1
                p2_win_con = 2
            elif i==16 or i==17:
                p1_win_con = 2
                p2_win_con = 1
            winner = self.game.start_play(current_mcts_player,
                                          pure_mcts_player,
                                          start_player=i % 2,
                                          is_shown=1,
                                          p1_win_con=p1_win_con,
                                          p2_win_con=p2_win_con
                                          )
            if i%2 == 0:
                win_cnt_move_first[winner] += 1
                win_cnt_move_first_dict[str(p1_win_con)+'_'+str(p2_win_con)+'_'+str(winner)] += 1
            else:
                win_cnt_move_second[winner] += 1
                win_cnt_move_second_dict[str(p1_win_con)+'_'+str(p2_win_con)+'_'+str(winner)] += 1
            win_cnt[winner] += 1
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
                self.pure_mcts_playout_num,
                win_cnt[1], win_cnt[2], win_cnt[-1]))
        print("move first win: {}, lose: {}, tie:{}".format(
            win_cnt_move_first[1], win_cnt_move_first[2], win_cnt_move_first[-1]))
        print("move second win: {}, lose: {}, tie:{}".format(
            win_cnt_move_second[1], win_cnt_move_second[2], win_cnt_move_second[-1]))
        for key in win_cnt_move_first_dict.keys():
            print('move first key', key, 'value', win_cnt_move_first_dict[key])
        for key in win_cnt_move_second_dict.keys():
            print('move second key', key, 'value', win_cnt_move_second_dict[key])
        return win_ratio

    def run(self):
        """run the training pipeline"""
        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(
                        i+1, self.episode_len))
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                # check the performance of the current model,
                # and save the model params
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))
                    win_ratio = self.policy_evaluate()
                    self.policy_value_net.save_model('./current_policy.model')
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.policy_value_net.save_model('./best_policy.model')
                        if (self.best_win_ratio == 1.0 and
                                self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    training_pipeline = TrainPipeline()
    training_pipeline.run()
