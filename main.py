from typing import List, Sequence

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph
from chains import generation_chain, reflection_chain

GENERATE = "generate"
REFLECT = "reflect"

def generation_node(state: Sequence[BaseMessage]):
    return generation_chain.invoke({"messages" : state})

def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]: 
    res = reflection_chain.invoke({"messages" : messages})
    return [HumanMessage(content=res.content)]

def router(state: List[BaseMessage]):
    "Conditional block which decides to end the graph or run the reflection again"
    if len(state) > 6 :
        return END
    return REFLECT

builder = MessageGraph()
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.add_conditional_edges(GENERATE, router)
builder.add_edge(REFLECT, GENERATE)

builder.set_entry_point(GENERATE)

graph = builder.compile()

# print("$" * 100)
# print(graph.get_graph().draw_mermaid())
# print("$" * 100)

# graph.get_graph().print_ascii()


if __name__ == "__main__":
    print("Hello LangGraph")
    inputs = HumanMessage(content="""Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post

                                  """)
    response = graph.invoke(inputs)