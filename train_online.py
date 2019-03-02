from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from rasa_core.agent import Agent
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core.interpreter import RasaNLUInterpreter
from rasa_core.training import online
from rasa_core.utils import EndpointConfig

logger = logging.getLogger(__name__)


def run_cocoa_online(interpreter,
                          domain_file="cocoa_domain.yml",
                          training_data_file='data/stories.md'):
    action_endpoint = EndpointConfig(url="http://localhost:4444/webhook")						  
    agent = Agent(domain_file,
                  policies=[MemoizationPolicy(max_history=2), KerasPolicy()],
                  interpreter=interpreter,
				  action_endpoint=action_endpoint)
    				  
    data = agent.load_data(training_data_file)
    agent.train(data,
                       batch_size=50,
                       epochs=600,
                       max_training_samples=500)				   
    online.run_online_learning(agent)
    return agent


if __name__ == '__main__':
    logging.basicConfig(level="INFO")
    nlu_interpreter = RasaNLUInterpreter('./models/nlu/default/cocoanlu')
    run_cocoa_online(nlu_interpreter)