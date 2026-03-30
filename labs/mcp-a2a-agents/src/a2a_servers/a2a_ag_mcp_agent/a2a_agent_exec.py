import logging

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import (
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from a2a.utils import (
    new_agent_text_message,
    new_data_artifact,
    new_task,
    new_text_artifact,
)

from a2a_agents import AbstractAgent
from typing_extensions import override


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class A2ALabAgentExecutor(AgentExecutor):
    """ Agent Executor """

    def __init__(self, agent: AbstractAgent):
        super().__init__()
        self.agent = agent

    # ---------------- async context-manager glue -------------
    async def __aenter__(self):
        if hasattr(self.agent, "__aenter__"):
            await self.agent.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if hasattr(self.agent, "__aexit__"):
            await self.agent.__aexit__(exc_type, exc, tb)
        return False
    # ---------------------------------------------------------

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        query = context.get_user_input()
        task = context.current_task
        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)

        async for partial in self.agent.stream(query, task.context_id):
            require_input = partial['require_user_input']
            is_done = partial['is_task_complete']
            text_content = partial['content']

            if require_input:
                await event_queue.enqueue_event(
                    TaskStatusUpdateEvent(
                        status=TaskStatus(
                            state=TaskState.input_required,
                            message=new_agent_text_message(
                                text_content,
                                task.context_id,
                                task.id,
                            ),
                        ),
                        final=True,
                        context_id=task.context_id,
                        task_id=task.id,
                    )
                )
            elif is_done:
                await event_queue.enqueue_event(
                    TaskArtifactUpdateEvent(
                        append=False,
                        context_id=task.context_id,
                        task_id=task.id,
                        last_chunk=True,
                        artifact=new_text_artifact(
                            name='current_result',
                            description='Result of request to agent.',
                            text=text_content,
                        ),
                    )
                )
                await event_queue.enqueue_event(
                    TaskStatusUpdateEvent(
                        status=TaskStatus(state=TaskState.completed),
                        final=True,
                        context_id=task.context_id,
                        task_id=task.id,
                    )
                )
            else:
                await event_queue.enqueue_event(
                    TaskStatusUpdateEvent(
                        status=TaskStatus(
                            state=TaskState.working,
                            message=new_agent_text_message(
                                text_content,
                                task.context_id,
                                task.id,
                            ),
                        ),
                        final=False,
                        context_id=task.context_id,
                        task_id=task.id,
                    )
                )

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        raise Exception('cancel not supported')
