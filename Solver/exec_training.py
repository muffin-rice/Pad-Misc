import Agent
import warnings
import torch

warnings.simplefilter('ignore', UserWarning)

if __name__ == '__main__':
    agent = Agent.Agent(batch_size = 16, gamma = .999, eps_start = .95, eps_end = .05, eps_decay = 200, target_update = 10,
                 memory_size = 5000)

    print('Version 0.5.0')
    print('Starting training:')
    agent.train(episodes = 500)
    print('Finished training')
    weights = agent.get_weights()

    torch.save(weights, 'agent_weights')
    #print(f'Weights are: \n{weights}\n')

    vals = agent.training_history
    print('The training history is:\n')
    print(vals)
    with open('move_counts.txt', 'w') as f:
        f.write(vals)