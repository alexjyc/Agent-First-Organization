import logging
from langgraph.graph import StateGraph, START
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from arklex.env.workers.worker import BaseWorker, register_worker
from arklex.env.prompts import load_prompts
from arklex.utils.graph_state import MessageState
from arklex.env.tools.utils import ToolGenerator
from arklex.utils.model_config import MODEL
from arklex.utils.model_provider_config import PROVIDER_MAP
from arklex.utils.utils import chunk_string, postprocess_json

logger = logging.getLogger(__name__)

@register_worker
class RecipesWorker(BaseWorker):
    description = "Find recipes based on available ingredients"

    def __init__(self):
        super().__init__()
        self.action_graph = self._create_action_graph()
        self.llm = PROVIDER_MAP.get(MODEL['llm_provider'], ChatOpenAI)(
            model=MODEL["model_type_or_path"], timeout=30000
        )
     
    def _create_action_graph(self):
        workflow = StateGraph(MessageState)
        workflow.add_node("recipe_generator", self._generate_recipes)
        workflow.add_edge(START, "recipe_generator")
        return workflow

    def _generate_recipes(self, state: MessageState):
        chat_history = state['user_message'].history

        prompts = load_prompts(state["bot_config"])
        prompt = PromptTemplate.from_template(prompts["user_pref_prompt"])
        input_prompt = prompt.invoke({ "chat_history": chat_history})

        logger.info(f"Prompt: {input_prompt.text}")
        chunked_prompt = chunk_string(input_prompt.text, tokenizer=MODEL["tokenizer"], max_length=MODEL["context"])
        final_chain = self.llm | StrOutputParser()
        answer = final_chain.invoke(chunked_prompt)
        processed_answer = postprocess_json(answer)

        state["message_flow"] = ""
        state["response"] = f"Preference: {processed_answer.get('preference', 'N/A')}, Ingredients: {processed_answer.get('ingredients', 'N/A')}"
        
        return state

    def execute(self, msg_state: MessageState):
        graph = self.action_graph.compile()
        result = graph.invoke(msg_state)
        return result