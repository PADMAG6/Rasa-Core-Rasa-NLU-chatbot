from rasa_core.channels.slack import SlackInput
from rasa_core.agent import Agent
from rasa_core.interpreter import RasaNLUInterpreter
import yaml
from rasa_core.utils import EndpointConfig


nlu_interpreter = RasaNLUInterpreter('./models/nlu/default/cocoanlu')
action_endpoint = EndpointConfig(url="http://localhost:4444/webhook")
agent = Agent.load('./models/dialogue', interpreter = nlu_interpreter, action_endpoint = action_endpoint)

input_channel = SlackInput( 'xoxb-506758897941-509709290608-gBqpk6QCSgXka63nu5jsP5Ji',
							'x1XPtEhBbY1Ma48RqDA5nwa9'#your bot user authentication token
                           )

agent.handle_channels([input_channel], 5002, serve_forever=True)