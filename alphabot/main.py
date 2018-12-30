from Coach import Coach
from Game import YEET as Game
from NNet import NNetWrapper as nn
from utils import dotdict
import logging

args = dotdict({
    'numIters': 10,    #100
    'numEps': 10,      #100
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 25,
    'arenaCompare': 6,      #  approx time: 13 hr
    'cpuct': 10,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('./temp/','temp.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})

if __name__=="__main__":
    
    logging.disable(logging.WARNING)
    
    g = Game(is_basic=True)
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
