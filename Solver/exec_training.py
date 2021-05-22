import Agent
import warnings
import torch
import logging

warnings.simplefilter('ignore', UserWarning)
logging.basicConfig(filename='logs/exec_training_logs.log', level=logging.INFO)

if __name__ == '__main__':
    agent = Agent.Agent(batch_size = 16, gamma = .999, eps_start = .95, eps_end = .05, eps_decay = 200, target_update = 10,
                 memory_size = 5000)

    logging.info('Version 0.5.0')
    logging.info('Starting training:')
    agent.train(episodes = 500)
    logging.info('Finished training')
    weights = agent.get_weights()

    torch.save(weights, 'agent_weights')
    #print(f'Weights are: \n{weights}\n')

    vals = agent.training_history
    logging.info('The training history is:\n')
    logging.info(vals)