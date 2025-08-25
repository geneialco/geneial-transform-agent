import asyncio
import logging
import json
from pydantic import BaseModel, Field
from typing import Optional, ClassVar, Type
from langchain.tools import BaseTool

# Attempt to import optional browser_use dependency. If unavailable,
# we provide a graceful stub so the rest of the application can run.
try:
    from browser_use import AgentHistoryList, Browser, BrowserConfig
    from browser_use import Agent as BrowserAgent

    _BROWSER_AVAILABLE = True
    _BROWSER_IMPORT_ERROR = None
except (
    Exception
) as _e:  # ImportError or downstream missing deps (e.g., main_content_extractor)
    AgentHistoryList = None  # type: ignore
    Browser = None  # type: ignore
    BrowserConfig = None  # type: ignore
    BrowserAgent = None  # type: ignore
    _BROWSER_AVAILABLE = False
    _BROWSER_IMPORT_ERROR = _e
from src.tools.decorators import create_logged_tool
from src.config import (
    CHROME_INSTANCE_PATH,
    CHROME_HEADLESS,
    CHROME_PROXY_SERVER,
    CHROME_PROXY_USERNAME,
    CHROME_PROXY_PASSWORD,
    BROWSER_HISTORY_DIR,
)
import uuid

# Configure logging
logger = logging.getLogger(__name__)

if _BROWSER_AVAILABLE:
    browser_config = BrowserConfig(
        headless=CHROME_HEADLESS,
        chrome_instance_path=CHROME_INSTANCE_PATH,
    )
    if CHROME_PROXY_SERVER:
        proxy_config = {
            "server": CHROME_PROXY_SERVER,
        }
        if CHROME_PROXY_USERNAME:
            proxy_config["username"] = CHROME_PROXY_USERNAME
        if CHROME_PROXY_PASSWORD:
            proxy_config["password"] = CHROME_PROXY_PASSWORD
        browser_config.proxy = proxy_config

    expected_browser = Browser(config=browser_config)
else:
    expected_browser = None


class BrowserUseInput(BaseModel):
    """Input for WriteFileTool."""

    instruction: str = Field(..., description="The instruction to use browser")


class BrowserTool(BaseTool):
    name: ClassVar[str] = "browser"
    args_schema: Type[BaseModel] = BrowserUseInput
    description: ClassVar[str] = (
        "Use this tool to interact with web browsers. Input should be a natural language description of what you want to do with the browser, such as 'Go to google.com and search for browser-use', or 'Navigate to Reddit and find the top post about AI'."
    )

    _agent: Optional[BrowserAgent] = None

    def _generate_browser_result(
        self, result_content: str, generated_gif_path: str
    ) -> dict:
        return {
            "result_content": result_content,
            "generated_gif_path": generated_gif_path,
        }

    def _run(self, instruction: str) -> str:
        if not _BROWSER_AVAILABLE:
            err_name = (
                type(_BROWSER_IMPORT_ERROR).__name__
                if _BROWSER_IMPORT_ERROR
                else "ImportError"
            )
            err_msg = (
                str(_BROWSER_IMPORT_ERROR)
                if _BROWSER_IMPORT_ERROR
                else "browser_use dependency not available"
            )
            return (
                f"Browser tool is unavailable ({err_name}): {err_msg}. "
                f"Install 'browser-use' and its dependencies or disable the browser tool."
            )

        generated_gif_path = f"{BROWSER_HISTORY_DIR}/{uuid.uuid4()}.gif"
        """Run the browser task synchronously."""
        # Lazy import to avoid requiring optional deps when unused
        from src.llms.llm import vl_llm  # type: ignore

        self._agent = BrowserAgent(
            task=instruction,  # Will be set per request
            llm=vl_llm,
            browser=expected_browser,
            generate_gif=generated_gif_path,
        )

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self._agent.run())

                if AgentHistoryList is not None and isinstance(
                    result, AgentHistoryList
                ):
                    return json.dumps(
                        self._generate_browser_result(
                            result.final_result(), generated_gif_path
                        )
                    )
                else:
                    return json.dumps(
                        self._generate_browser_result(result, generated_gif_path)
                    )
            finally:
                loop.close()
        except Exception as e:
            return f"Error executing browser task: {str(e)}"

    async def terminate(self):
        """Terminate the browser agent if it exists."""
        if self._agent and self._agent.browser:
            try:
                await self._agent.browser.close()
            except Exception as e:
                logger.error(f"Error terminating browser agent: {str(e)}")
        self._agent = None

    async def _arun(self, instruction: str) -> str:
        """Run the browser task asynchronously."""
        if not _BROWSER_AVAILABLE:
            err_name = (
                type(_BROWSER_IMPORT_ERROR).__name__
                if _BROWSER_IMPORT_ERROR
                else "ImportError"
            )
            err_msg = (
                str(_BROWSER_IMPORT_ERROR)
                if _BROWSER_IMPORT_ERROR
                else "browser_use dependency not available"
            )
            return (
                f"Browser tool is unavailable ({err_name}): {err_msg}. "
                f"Install 'browser-use' and its dependencies or disable the browser tool."
            )

        generated_gif_path = f"{BROWSER_HISTORY_DIR}/{uuid.uuid4()}.gif"
        # Lazy import to avoid requiring optional deps when unused
        from src.llms.llm import vl_llm  # type: ignore

        self._agent = BrowserAgent(
            task=instruction,
            llm=vl_llm,
            browser=expected_browser,
            generate_gif=generated_gif_path,  # Will be set per request
        )
        try:
            result = await self._agent.run()
            if AgentHistoryList is not None and isinstance(result, AgentHistoryList):
                return json.dumps(
                    self._generate_browser_result(
                        result.final_result(), generated_gif_path
                    )
                )
            else:
                return json.dumps(
                    self._generate_browser_result(result, generated_gif_path)
                )
        except Exception as e:
            return f"Error executing browser task: {str(e)}"
        finally:
            await self.terminate()


BrowserTool = create_logged_tool(BrowserTool)
browser_tool = BrowserTool()

if __name__ == "__main__":
    browser_tool._run(instruction="go to github.com and search langmanus")
