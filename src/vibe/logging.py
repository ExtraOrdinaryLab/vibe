import functools

from rich.console import Console
from accelerate.state import PartialState


class RichConsoleLogger:
    
    def __init__(self):
        self.console = Console()
        if PartialState._shared_state == {}:
            raise RuntimeError(
                "You must initialize the accelerate state by calling either `PartialState()` or `Accelerator()` before using the logging utility."
            )
        self.state = PartialState()

    def _should_log(self, main_process_only: bool) -> bool:
        return not main_process_only or (main_process_only and self.state.is_main_process)

    def log(self, msg: str, *, main_process_only: bool = True, in_order: bool = False, style: str = None):
        if self._should_log(main_process_only):
            self.console.print(msg, style=style)
        elif in_order:
            for i in range(self.state.num_processes):
                if i == self.state.process_index:
                    self.console.print(msg, style=style)
                self.state.wait_for_everyone()

    @functools.lru_cache(None)
    def warning_once(self, msg: str, *, style: str = "bold yellow"):
        self.console.print(msg, style=style)


def get_logger():
    return RichConsoleLogger()