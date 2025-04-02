from scipy.optimize import milp, LinearConstraint, Bounds
from typing import Optional, Iterator, Iterable, Callable
import numpy as np
from delex.graph import PredicateNode, UnionNode, IntersectNode, MinusNode
from delex.graph.algorithms import find_all_nodes, find_sink
from delex.graph.node import Node
from delex.lang import BlockingProgram, KeepRule, DropRule, Predicate, Rule
from delex.utils.funcs import type_check_call
from delex.execution import CostEstimator
from collections import defaultdict
from delex.utils.funcs import get_logger
import itertools
from copy import deepcopy

logger = get_logger(__name__)

def _presolve_hitting_set(keep_rules: list[KeepRule]) -> tuple[set[Predicate], list[KeepRule]]:
    must_be_indexed = {p for r in keep_rules for p in r if not p.streamable}

    if len(must_be_indexed) == 0:
        return must_be_indexed, keep_rules
    else:
        not_covered_rules = []
        for r in keep_rules:
            for p in must_be_indexed:
                if any(p.contains(q) for q in r):
                    break
            else:
                not_covered_rules.append(r)
        
        return must_be_indexed, not_covered_rules

def _min_weighted_hitting_set_cover(keep_rules: list[KeepRule], cost_function: Callable=None) -> list[Predicate]:
    # the one indexes in the A matrix in the MILP formula
    must_be_indexed, not_covered_rules = _presolve_hitting_set(keep_rules)
    # if all rules are already covered by the predicates that 
    # we must index then just return what we have to index
    if len(not_covered_rules) == 0:
        return list(must_be_indexed)

    indexable_preds = {p for r in not_covered_rules for p in r if p.indexable}
    indexable_preds = {p : [i] for i,p in enumerate(indexable_preds)}
    for p, p_indexes in indexable_preds.items():
        for q, q_indexes in indexable_preds.items():
            if p != q and p.contains(q):
                q_indexes.append(p_indexes[0])

    A = np.zeros(shape=(len(not_covered_rules), len(indexable_preds)))
    for i, rule in enumerate(not_covered_rules):
        for pred in rule.predicates:
            if pred in indexable_preds:
                A[i, indexable_preds[pred]] = 1
    
    c = np.ones(A.shape[1])
    if cost_function is not None:
        for pred, i in indexable_preds.items():
            pred_cost = cost_function(pred)
            c[i[0]] = pred_cost
            logger.info(f'{pred} : hitting set cost {pred_cost}')

    opt_res = milp(
            c = c,
            integrality = np.ones(A.shape[1]),
            constraints = LinearConstraint(A, lb=1, ub=np.inf),
            bounds = Bounds(0,1)
    )
    if not opt_res.success:
        raise RuntimeError(f'unable to find solution to set cover {opt_res.message}')

    preds = [p for p,i in indexable_preds.items() if opt_res.x[i[0]] > 0]
    preds += must_be_indexed
    return preds

class BlockingProgramOptimizer:
    """
    a class for converting a BlockingProgram into an execution plan of Nodes,
    optionally applying optimizations 
    """

    def __init__(self):
        pass
    
    @type_check_call
    def default_plan(self, blocking_program: BlockingProgram) -> Node:
        """
        create a default execution plan for `blocking_program`

        Parameters
        ----------
        blocking_program : BlockingProgram
            the blocking program that will be turned into an execution plan

        Returns
        -------
        Node 
            the sink of the execution plan

        Raises
        ------
        ValueError
            if `blocking_program` cannot be turned into an efficient execution plan, 
            i.e. it would require executing over the cross product of the tables

        """
        blocking_program = deepcopy(blocking_program)
        blocking_program = self.preprocess(blocking_program)

        output = UnionNode()
        for rule in blocking_program.keep_rules:

            must_be_indexed = {PredicateNode(p) for p in rule.predicates if not p.streamable}
            if len(must_be_indexed):
                indexed = list(must_be_indexed)
            else:
                for p in rule.predicates:
                    if p.indexable:
                        indexed = [PredicateNode(p)]
                        break

            if len(indexed) == 0:
                raise ValueError('no indexable predicates for keep rule')

            self._add_keep_rule_to_plan(indexed, output, rule)


        if output.in_degree == 1:
            n = next(output.iter_in())
            output.pop()
            output = n

        if len(blocking_program.drop_rules) != 0:
            dropped = UnionNode()
            for rule in blocking_program.drop_rules:
                self._add_drop_rule(output, dropped, rule)
            # remove union if there is only a single drop rule
            if dropped.in_degree == 1:
                n = next(dropped.iter_in())
                dropped.pop()
                dropped = n
            # remove things matching the drop rules
            output = MinusNode(output, dropped)

        return output
    
    def _add_keep_rule_to_plan(self, indexed_nodes: list[Node], union: UnionNode, rule: KeepRule):
        """
        add a keep rule to an execution plan
        """
        indexed = []
        filtered = []
        for pred in rule:
            if pred.indexable:
                
                for node in indexed_nodes:
                    if node.predicate == pred:
                        # if the predicate has been choosen to be indexed
                        indexed.append(node)
                        break
                else:
                    for node in indexed_nodes:
                        if node.predicate.contains(pred):
                            # if a predicate that contains pred has been choosen to be indexed 
                            # then we can take the output of that node and filter without recomputing the scores
                            n = PredicateNode(pred)
                            node.add_out_edge(n)
                            indexed.append(n)
                            break
                    else:
                        filtered.append(PredicateNode(pred))
            else:
                # non-indexable predicates must be filter
                filtered.append(PredicateNode(pred))

        if len(indexed) == 0:
            raise RuntimeError('no nodes indexed for keep rule, invalid plan')
        elif len(indexed) == 1:
            # single source node for this rule
            tail = indexed[0]
        else:
            # multiple source nodes, intersect them all
            tail = IntersectNode()
            for n in indexed:
                tail.add_in_edge(n)
            
        # append all filter nodes to the plan fragment
        for n in filtered:
            tail.add_out_edge(n)
            tail = n
        # finally link to the union node of all the keep rules
        union.add_in_edge(tail)
        # done

    
    def _predicate_reuse(self, sink: Node) -> Iterator[Node]:
        """
        apply predicate reuse rule to plan `sink`, return an iterator for new plans
        """
        predicate_to_nodes = defaultdict(list)
        for n in find_all_nodes(sink):
            if isinstance(n, PredicateNode):
                p = n.predicate
                predicate_to_nodes[p].append(n)

        for p, nodes in predicate_to_nodes.items():
            if len(nodes) > 1:
                end_point_to_path = defaultdict(list)
                for n in nodes:
                    path = _basic_path(n)
                    end_point_to_path[path[-1]].append(path)
                
                for ep, paths in end_point_to_path.items():
                    if len(paths) > 1 and isinstance(ep, (UnionNode, IntersectNode)):
                        ep, paths = deepcopy((ep, paths))
                        if len(paths) == ep.in_degree:
                            for path in paths:
                                n = path[0].pop()
                            # all paths connected use the predicate, easy case
                            ep.insert_after(n)
                        else:
                            combiner = ep.__class__()
                            combiner.add_out_edge(ep)
                            for p in paths:
                                n = p[-2]
                                n.remove_out_edge(ep)
                                n.add_out_edge(combiner)
                            for path in paths:
                                n = path[0].pop()
                            combiner.insert_after(n)

                        yield find_sink(ep)

    def _rule_short_circuit(self, sink: Node) -> Iterator[Node]:
        """
        apply short circuit rule to plan `sink`, return an iterator of new plans
        """
        candidates = []
        for n in find_all_nodes(sink):
            if isinstance(n, UnionNode):
                # TODO ancestors should be subsets, i.e. don't count nodes accessable through 
                # the right side of a minus node
                in_to_ancestors = [(i, i.ancestors(), _reverse_basic_path(i)) for i in n.iter_in()]
                for (x, x_ancestors, x_path), (y, y_ancestors, y_path) in itertools.combinations(in_to_ancestors, 2):
                    # prevent infinite recusion 
                    if x in y_ancestors or y in x_ancestors:
                        continue
                    # TODO need to consider both paths
                    if len(x_path) <= len(y_path):
                        if y.out_degree == 1 and y_path[-1] in x_ancestors:
                            candidates.append((x_path, y_path, n))
                        elif x.out_degree == 1 and x_path[-1] in y_ancestors:
                            candidates.append((y_path, x_path, n))
                    else:
                        if x.out_degree == 1 and x_path[-1] in y_ancestors:
                            candidates.append((y_path, x_path, n))
                        elif y.in_degree == 1 and y_path[-1] in x_ancestors:
                            candidates.append((x_path, y_path, n))
        
        for c in candidates:
            first_path, second_path, union_node = deepcopy(c)
            # create logical equivalent to first_path OR second_path with short circuit logic
            if len(second_path) < 2:
                raise ValueError(f'bad {second_path=}')

            second_path[-1].remove_out_edge(second_path[-2])
            # subtract the output of the first path from
            # the input the second path to avoid evaluating the 
            # second path on pairs that we are going to get in the union anyway 
            minus_node = MinusNode(second_path[-1], first_path[0])
            minus_node.add_out_edge(second_path[-2])
            yield find_sink(union_node)

    def _preprocess_bool_objects(self, bool_objs : Iterable[Rule] | Iterable[Predicate], removed_contained: bool):
        """
        preprocess either a sequence of rules or a sequence of 
        predicates, dropping any redudant Rules or Predicates
        """
        remove = set()
        removed = None
        for (lidx, l), (ridx, r) in itertools.combinations(enumerate(bool_objs), 2):
            if l.contains(r):
                remove.add(ridx if removed_contained else lidx)
            elif r.contains(l):
                remove.add(lidx if removed_contained else ridx)
        
        if len(remove):
            removed = []
            # remove redundant rules       
            for i in sorted(remove, reverse=True):
                removed.append(bool_objs.pop(i))
        return removed
    
    @type_check_call
    def preprocess(self, blocking_program: BlockingProgram) -> BlockingProgram:
        """
        preprocess the blocking program by dropping any redundant rules or predicates. 
        That is, remove anything that doens't affect the output of `blocking_program`
        """
        blocking_program = deepcopy(blocking_program)
        for t, rules in [('keep', blocking_program.keep_rules), ('drop', blocking_program.drop_rules)]:
            # intra-rule preprocessing
            for r in rules:
                removed = self._preprocess_bool_objects(r.predicates, removed_contained=False)
                if removed:
                    logger.info(f'removed : {removed} from {t} rule {r}')
        
            # inter-rule preprocessing
            removed = self._preprocess_bool_objects(rules, removed_contained=True)
            if removed:
                logger.info(f'removed from {t} rules : {removed}')

        return blocking_program

    def _generate_plans(self, plan):
        """
        generate plans by apply an optimization rule to each plan
        """
        yield from itertools.chain(
                self._predicate_reuse(plan),
                self._rule_short_circuit(plan)
            )

    def _run_optimize_loop(self, sink, cost_est, k=1):
        candidates = []
        min_cost = cost_est.estimate_plan_cost(sink)
        topk = [(min_cost, sink)]

        logger.info(f'running optimization loop with {k=}')
        for i in itertools.count():
            candidates.clear()
            min_competitive_cost = topk[-1][0]
            logger.info(f'starting opt loop {i} : {min_competitive_cost=}')

            for _, plan in topk:
                for n in self._generate_plans(plan):
                    cost = cost_est.estimate_plan_cost(n)
                    logger.debug(f'plan cost {cost}')
                    if cost < min_competitive_cost:
                        candidates.append( (cost, n) )
        
            if len(candidates) != 0:
                logger.info(f'{len(candidates)=}')
                # if we found any competitive plans 
                topk.extend(candidates)
                topk.sort(key=lambda x : x[0])
                topk = topk[:k]
            else:
                logger.info('no competitive plans found, terminating')
                # otherwise we have reached a fixed point, now we are done
                return topk[0]
    
    def _add_drop_rule(self, keep_output:  Node, drop_union: UnionNode, rule: DropRule):
        """
        add a drop rule to a plan
        """
        tail = keep_output
        for pred in rule:
            n = PredicateNode(pred)
            tail.add_out_edge(n)
            tail = n

        drop_union.add_in_edge(tail)
        
    @type_check_call
    def optimize(self, blocking_program: BlockingProgram, cost_est: Optional[CostEstimator] = None):
        """
        create an optimized execution plan for `blocking_program`, optionally using `cost_est`
        If `cost_est` is not supplied, the optimizer simply indexes the least number of predicates possible 
        and generates a default plan using those nodes.

        Parameters
        ----------
        blocking_program : BlockingProgram
            the blocking program that will be turned into an execution plan

        cost_est : Optional[cost_est] = None
            the cost estimator used for optimizing `blocking_program`

        Returns
        -------
        Node 
            the sink of the execution plan

        Raises
        ------
        ValueError
            if `blocking_program` cannot be turned into an efficient execution plan, 
            i.e. it would require executing over the cross product of the tables
        """
        logger.info('optimizing blocking program')
    
        logger.info('running preprocessing')
        blocking_program = self.preprocess(blocking_program)

        if cost_est is not None:
            cost_est.validate(blocking_program)
            cost_func = lambda x : cost_est.search_time(x) * cost_est._table_b_count / cost_est.nthreads + cost_est.build_time(x)
        else:
            logger.info('cost estimator no provided, using unweighted hitting set')
            cost_func = None

        logger.info(f'running set cover with indexing with {len(blocking_program.keep_rules)=}')
        indexed_preds = _min_weighted_hitting_set_cover(blocking_program.keep_rules, cost_func)
        indexed_nodes = [PredicateNode(p) for p in indexed_preds]
        logger.info(f'solution with {len(indexed_nodes)} found {indexed_nodes=}')
        output = UnionNode()

        for rule in blocking_program.keep_rules:
            self._add_keep_rule_to_plan(indexed_nodes, output, rule)
        # remove union if there is only a single keep rule
        if output.in_degree == 1:
            n = next(output.iter_in())
            output.pop()
            output = n
        
        if len(blocking_program.drop_rules) != 0:
            dropped = UnionNode()
            for rule in blocking_program.drop_rules:
                self._add_drop_rule(output, dropped, rule)
            # remove union if there is only a single drop rule
            if dropped.in_degree == 1:
                n = next(dropped.iter_in())
                dropped.pop()
                dropped = n
            # remove things matching the drop rules
            output = MinusNode(output, dropped)
        if cost_est is not None:
            cost, plan  = self._run_optimize_loop(output, cost_est, k=1)
            return plan, cost
        else:
            return output, -1



def _reverse_basic_path(src_node: Node):
    """
    return a list of nodes starting at `src_node` that create a basic path by traversing 
    edges in reverse
    """
    path = [src_node]
    if src_node.in_degree == 1:
        n = next(src_node.iter_in())
        while n.out_degree == 1 and n.in_degree == 1 and isinstance(n, PredicateNode):
            path.append(n)
            n = next(n.iter_in())
        path.append(n)

    return path
        
def _basic_path(src_node: Node):
    """
    return a list of nodes starting at `src_node` that create a basic path by traversing 
    edges in the forward direction
    """
    # find the longest stretch of nodes where the path is simple and can be reordered
    path = [src_node]
    if src_node.out_degree == 1:
        n = next(src_node.iter_out())
        while n.out_degree == 1 and n.in_degree == 1 and isinstance(n, PredicateNode):
            path.append(n)
            n = next(n.iter_out())
        path.append(n)

    return path
    

    




