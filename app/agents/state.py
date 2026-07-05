'''
The State class is responsible for managing the state of the application. 
It provides methods to get and set state variables, as well as to reset the state to its initial values. 
The state is stored in a dictionary, allowing for easy access and modification of state variables. 
This class is designed to be used by other components of the application to maintain consistency and manage shared data.
'''


from typing import TypedDict , List , Annotated 
import operator 

class AgentState(TypedDict):

    ''' Represents the state of an agent.
    Using annotations to indicate that the messages list can be combined using the addition operator.
    '''
    
    messages : Annotated[List[dict], operator.add]
    current_query: str
    documents: List[dict]
    plan: List[str]
    status : str 
    final_answer : str 
