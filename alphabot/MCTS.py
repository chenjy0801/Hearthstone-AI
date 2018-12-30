import math
import numpy as np
import copy
import random
from fireplace.exceptions import GameOver, InvalidAction
import gc
EPS = 1e-8

class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.freeze = False
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

        self.Es = {}        # stores game.getGameEnded ended for board s
        self.Vs = {}        # stores game.getValidMoves for board s

    def getActionProb(self, state, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        the state.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.numMCTSSims):
            self.search(state, create_copy=True)

        s = self.game.stringRepresentation(state)

        counts = [self.Nsa[(s,(a,b))] if (s,(a,b)) in self.Nsa else 0 for a in range(21) for b in range(18)]
        if temp==0:
            bestA = np.argmax(counts)
            probs = [0]*len(counts)
            probs[bestA]=1
            return probs
        counts = [x**(1./temp) for x in counts]
        probs = [x/float(sum(counts)) for x in counts]
        return probs

    def cloneAndRandomize(self, game):
        """ Create a deep clone of this game state, randomizing any information not visible to the specified observer player.
        """
        game_copy = copy.deepcopy(game)
        enemy = game_copy.current_player.opponent
        random.shuffle(enemy.hand)
        random.shuffle(enemy.deck)
        # for idx, card in enumerate(enemy.hand):
        #     if card.id == 'GAME_005':
        #         coin = enemy.hand.pop(idx)
        #
        #if self.freeze:
        #   enemy.hand, enemy.deck = copy.deepcopy(self.freeze_hand), copy.deepcopy(self.freeze_deck)
        #else:
        #   combined = enemy.hand + enemy.deck
        #   random.shuffle(combined)
        #   enemy.hand, enemy.deck = combined[:len(enemy.hand)], combined[len(enemy.hand):]
        #   self.freeze_hand = copy.deepcopy(enemy.hand)
        #   self.freeze_deck = copy.deepcopy(enemy.deck)
        #    self.freeze = True
        #enemy.hand.append(coin)
        return game_copy


    def search(self, state, create_copy):
        """
        NEEDS TO RUN ON DEEPCOPY!!!

        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current state
        """
        # Determinize
        if create_copy:
            game_copy = self.cloneAndRandomize(self.game.game)
        
        # root
        next_s = state
        s = self.game.stringRepresentation(state)
        
        # path
        path = []

        # Select
        while s not in self.Es and s in self.Ps and not game_copy.ended: # node is fully expanded and non-terminal
            valids = self.game.getValidMoves(game_copy)
            cur_best = -float('inf')
            best_act = -1
            for a in range(21):
                for b in range(18):
                    if valids[a,b]:
                        if (s,(a,b)) in self.Qsa:
                            u = self.Qsa[(s,(a,b))] + self.args.cpuct*self.Ps[s][a,b]*math.sqrt(self.Ns[s])/(1+self.Nsa[(s,(a,b))])
                        else:
                            u = self.args.cpuct*self.Ps[s][a,b]*math.sqrt(self.Ns[s] + EPS)     # Q = 0 ?

                        if u > cur_best:
                            cur_best = u
                            best_act = (a,b)
            try:
                #print(best_act)
                player = game_copy.current_player
                next_s, next_player = self.game.getNextState(1, best_act, game_copy)
            except GameOver:
                #self.Es[s] = self.game.getGameEnded(game_copy)
                break
            
            path.append((s , best_act, player))
            
            s = self.game.stringRepresentation(next_s)   
            
        if game_copy.ended: self.Es[s] = self.game.getGameEnded(game_copy)
        

        # Expand
        if s not in self.Es and s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.nnet.predict(next_s)
            valids = self.game.getValidMoves(game_copy)
            self.Ps[s] = self.Ps[s]*valids + valids*game_copy.current_decay     # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            self.Ps[s] /= sum_Ps_s    # renormalize

            self.Vs[s] = valids
            self.Ns[s] = 0
            if game_copy.current_player != game_copy.player_to_start:
                v = -v   
        

        # Simulate
        while s not in self.Es and not game_copy.ended: # while state is non-terminal
            try:
                choices = np.argwhere(self.game.getValidMoves(game_copy))
                next_s, next_player = self.game.getNextState(1, random.choice(choices), game_copy)
                s = self.game.stringRepresentation(next_s)
            except GameOver:   
                #v = 0.7*v + 0.3*self.game.getGameEnded(game_copy)
                break
        if s not in self.Es and game_copy.ended:
            v = 0.7*v + 0.3*self.game.getGameEnded(game_copy)

        # Backpropagate
        if s in self.Es: v = self.Es[s]
        for ele in path: # backpropagate from the expanded node and work back to the root node
            s,a,player = ele
            if (s,a) not in self.Nsa: 
                self.Nsa[(s,a)] = 1
                self.Qsa[(s,a)] = 0
            else: 
                self.Nsa[(s,a)] += 1
            if player == game_copy.player_to_start:
                self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + v)/(self.Nsa[(s,a)]+1)
            else:
                self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] - v)/(self.Nsa[(s,a)]+1)  
        
        # Garbage Can
        gc.collect()

