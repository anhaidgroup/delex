from pydantic.dataclasses import dataclass
from typing import List
from delex.lang import KeepRule, DropRule
import textwrap


def _rule_to_pretty_str(r):
    s = textwrap.indent(f'(\n{r.pretty_str()}\n)', '    ')
    return s

@dataclass(frozen=True)
class BlockingProgram:
    """
    a blocking program that can be turned into an execution plan 
    """
    keep_rules : List[KeepRule] 
    drop_rules : List[DropRule] 
    def __post_init__(self):
        self.validate()

    def validate(self):
        if len(self.keep_rules) == 0:
            raise ValueError('blocking program must contain at least one keep rule')

    def pretty_str(self) -> str:
        """
        create a pretty string of the entire blocking program
        """
        or_sep = '\n  OR\n'
        keep = or_sep.join( _rule_to_pretty_str(r) for r in self.keep_rules)
        drop = or_sep.join( _rule_to_pretty_str(r) for r in self.drop_rules)
        prog = f'KEEP (\n{keep}\n)'
        if len(drop):
            prog += f'\n\nDROP (\n{drop}\n)'
        return prog


