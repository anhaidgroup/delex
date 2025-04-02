from pydantic.dataclasses import dataclass
from pydantic import ConfigDict
from typing import List
from delex.lang import Predicate
from delex.utils.funcs import type_check

_CONFIG = ConfigDict(arbitrary_types_allowed = True)

@dataclass(frozen=True, config=_CONFIG)
class Rule:
    """
    base class for a DropRule or KeepRule
    """
    predicates : List[Predicate]

    def __post_init__(self):
        self.validate()

    def __iter__(self):
        return iter(self.predicates)

    def contains(self, other):
        """
        return True if `self` logically contains `other` else False.
        That is, for any given set of record pairs C, self(C) is a superset of other(C)
        """
        type_check(other, 'other', Rule)
        for p in self:
            for q in other:
                if p.contains(q):
                    break
            else:
                # p does not contain any predicate in other, cannot prove containment
                return False

        return True

    def pretty_str(self) -> str:
        """
        format the rule into a pretty string
        """
        and_sep = '\n AND \n'
        return and_sep.join(map(str, self.predicates))



@dataclass(frozen=True, config=_CONFIG)
class KeepRule(Rule):
    """
    a keep rule for a blocking program
    """

    def validate(self):
        """
        check that this rule has at least one indexable predicate
        if not raise RuntimeError
        """

        has_indexable = False
        for p in self.predicates:
            has_indexable = p.indexable or has_indexable
            if not (p.indexable or p.streamable):
                raise ValueError(f'all predicates must be indexable or streamable,  ({p} is neither)')
        
        if not has_indexable:
            raise ValueError('keep rule must contain at least one indexable predicate')

@dataclass(frozen=True, config=_CONFIG)
class DropRule(Rule):
    """
    a drop rule for a blocking program
    """

    def validate(self):
        """
        check that all the predicates in this rule are streamable
        if not raise RuntimeError
        """
        if len(self.predicates) == 0:
            raise ValueError('drop rule must contain at least one predicate')

        for p in self.predicates:
            if not p.streamable:
                raise ValueError('all predicates in drop rules must be streamable')

