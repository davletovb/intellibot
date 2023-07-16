import logging
import openai
import elevenlabs
import asyncio
import os

from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, ConversationChain
from langchain.prompts import PromptTemplate
from langchain.agents import load_tools, initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper, GoogleSearchAPIWrapper
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper

from vectordb import VectorDB

# Set API keys
openai.api_key = os.environ.get("OPENAI_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID")
WOLFRAM_ALPHA_APPID = os.environ.get("WOLFRAM_ALPHA_APPID")
ELEVEN_API_KEY = os.environ.get("ELEVEN_API_KEY")

# Enable logging for debugging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class RateLimitError(Exception):
    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after

async def handle_rate_limiting(func, *args, is_async=True, **kwargs):
    retries = 5
    backoff_factor = 2

    for attempt in range(retries):
        try:
            if is_async:
                result = await func(*args, **kwargs)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: func(*args, **kwargs))
            return result
        except openai.error.RateLimitError as e:
            if attempt < retries - 1:  # Check if it's the last attempt
                retry_after = int(e.headers.get("Retry-After", 0))
                wait_time = retry_after or (backoff_factor ** attempt)
                logger.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                raise RateLimitError("Too many rate-limited attempts.", retry_after=retry_after) from e
        except elevenlabs.RateLimitError as e:
            if attempt < retries - 1:
                retry_after = int(e.headers.get("Retry-After", 0))
                wait_time = retry_after or (backoff_factor ** attempt)
                logger.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
        except Exception as e:
            raise e

class Prompter:
    def __init__(self, chat_id):
        # check if the chat_id is string
        if not isinstance(chat_id, str):
            self.chat_user_id = str(chat_id)
        else:
            self.chat_user_id = chat_id

    async def generate_image(self, prompt):
        try:
            response = await handle_rate_limiting(openai.Image.acreate, prompt=prompt, n=1, size="256x256")
            return response['data'][0]['url']
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            return None

    async def transcribe_voice(self, file):
        try:
            transcript = await handle_rate_limiting(openai.Audio.atranscribe, model="whisper-1", file=file)
            return transcript["text"]
        except Exception as e:
            logger.error(f"Error transcribing voice: {e}")
            return None
    
    async def generate_audio(self, text):
        try:
            audio = await handle_rate_limiting(elevenlabs.generate, api_key=ELEVEN_API_KEY, text=text, voice="Bella", model="eleven_monolingual_v1", is_async=False)
            return audio
        except Exception as e:
            logger.error(f"Error generating audio: {e}")
            return None

    async def search_wikipedia(self, query):
        wikipedia = WikipediaAPIWrapper()
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: wikipedia.run(query))
            return response
        except Exception as e:
            logger.error(f"Error searching wikipedia: {e}")
            return None
    
    async def search_google(self, query):
        search = GoogleSearchAPIWrapper(google_api_key=GOOGLE_API_KEY, google_cse_id=GOOGLE_CSE_ID, k=5)
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: search.run(query))
            return response
        except Exception as e:
            logger.error(f"Error searching wikipedia: {e}")
            return None
    
    async def search_wolframalpha(self, query):
        wolframalpha = WolframAlphaAPIWrapper(wolfram_alpha_appid=WOLFRAM_ALPHA_APPID)
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: wolframalpha.run(query))
            return response
        except Exception as e:
            logger.error(f"Error searching wikipedia: {e}")
            return None

    # Prompt the LLM to generate a response
    async def generate_response(self, message, chat_context):

        # Format the chat history as a string
        formatted_chat_history = "\n".join([f"{k}: {v}" for entry in chat_context for k, v in entry.items()])

        # Create a prompt template
        template = f"""
        Chat History:
        {formatted_chat_history}

        Human: {message}
        AI:"""

        # Create a prompt template
        #prompt_template = PromptTemplate(template=template, validate_template=False)

        # A conversation buffer (memory) & import llm of choice
        #memory = ConversationBufferMemory(memory_key="chat_history", input_key="input")

        # Create a model, chain and tool for the language model
        llm = OpenAI(temperature=0, 
                     model="text-davinci-003",
                        streaming=True,
                        max_retries=3,
                        openai_api_key=openai.api_key)

        #llm_chain = ConversationChain(llm=llm, prompt=prompt_template)

        # Provide access to a list of tools that the agents will use
        tools = load_tools(['llm-math'],
                            llm=llm)
        
        tools.extend([
            Tool(name="Image Model", func=self.generate_image, coroutine=self.generate_image, description="Generate images from text", return_direct=True),
            Tool(name="Wikipedia", func=self.search_wikipedia, coroutine=self.search_wikipedia, description="Search Wikipedia for general information"),
            Tool(name="Google Search", func=self.search_google, coroutine=self.search_google, description="Search the web. Useful about current events, everyday life, news, technical topics, errors or fixes."),
            Tool(name="Wolfram Alpha", func=self.search_wolframalpha, coroutine=self.search_wolframalpha, description="Search Wolfram Alpha. Useful about science, weather, climate, engineering, technology, culture and society"),
            Tool(name="Search User Documents", func=self.search_database, coroutine=self.search_database, description="Search user documents database"),
        ])
        

        # initialise the agents & make all the tools and llm available to it
        agent = initialize_agent(tools=tools,
                                llm=llm, 
                                agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, 
                                verbose=True,
                                max_iterations=3,
                                early_stopping_method="generate",
                                handle_parsing_errors="Check your output and make sure it conforms!")

        try:
            answer = await agent.arun(input=template, chat_history=formatted_chat_history, return_only_outputs=True)
            return answer
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "An error occurred while generating the response."
    
    async def save_document(self, document):
        try:
            db = VectorDB(chat_user_id=self.chat_user_id)
            summary = await db.add_document(document=document)
            return summary
        except Exception as e:
            logger.error(f"Error saving document: {e}")
            return "Error saving document"

    async def save_url(self, url):
        try:
            db = VectorDB(chat_user_id=self.chat_user_id)
            summary = await db.add_url(url=url)
            return summary
        except Exception as e:
            logger.error(f"Error saving URL: {e}")
            return "Error saving URL"
        
    async def search_database(self, query):
        try:
            db = VectorDB(chat_user_id=self.chat_user_id)
            results = await db.query(query=query)
            return results
        except Exception as e:
            logger.error(f"Error searching user documents: {e}")
            return "Error searching user documents"
    
    async def clear_database(self):
        try:
            db = VectorDB(chat_user_id=self.chat_user_id)
            await db.clear_database()
            return True
        except Exception as e:
            logger.error(f"Error clearing user documents: {e}")
            return False