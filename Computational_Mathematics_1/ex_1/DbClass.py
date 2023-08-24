

from typing_extensions import Self
from xmlrpc.client import boolean


class DbClass:
    
    def __init__(self, display : boolean, **kwargs) -> None:
        self._disp = display
        self._prefix = kwargs.get('prefix', '')        
        
    def __call__(self, msg : str) -> None:
        if self._disp:
            if self._prefix:
                print(f'{self._prefix} {msg}')
            else:
                print(msg)