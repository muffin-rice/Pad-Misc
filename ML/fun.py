import Agent
import warnings
warnings.simplefilter('ignore', UserWarning)


if __name__ == '__main__':
    agent = Agent.Agent(batch_size=32, gamma = .999, eps_start = .95, eps_end = .05, eps_decay = 250000, target_update = 1000,
                 memory_size = 5000)
    print('Version 0.4.11')
    print('Starting training:')
    agent.train(episodes = 500)
    print('Finished training')
    weights = agent.get_weights()
    print(f'Weights are: \n{weights}\n')

    vals = agent.training_history
    print('The training history is:\n')
    print(vals)
    with open('info1.txt', 'w') as f:
        f.write(str(weights))
        f.write(vals)