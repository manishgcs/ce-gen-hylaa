
# BDD Graph for representing counter examples

class BDDGraphTransition(object):
    def __init__(self, succ_node):
        self.succ_node = succ_node


class OneTransition(BDDGraphTransition):

    def __init__(self, succ_node):
        assert isinstance(succ_node, BDDGraphNode)
        BDDGraphTransition.__init__(self,  succ_node)


class ZeroTransition(BDDGraphTransition):

    def __init__(self, succ_node):
        assert isinstance(succ_node, BDDGraphNode)
        BDDGraphTransition.__init__(self, succ_node)


class BDDGraphNode(object):

    def __init__(self, node_id, level, my_regex='', is_terminal=False):

        self.id = node_id
        self.one_transition = None
        self.zero_transition = None
        self.terminal = is_terminal
        self.level = level
        self.my_regex = my_regex

    def new_transition(self, succ_node, t_type):

        if t_type == 0:
            self.zero_transition = succ_node
        elif t_type == 1:
            self.one_transition = succ_node
        else:
            print("Wrong transition type")


class BDDGraph(object):

    def __init__(self, root_node=None):
        if root_node is None:
            root_node = BDDGraphNode('r', 0)
        self.nodes = [root_node]
        self.root = root_node
        self.n_layers = 0

    def get_root(self):
        return self.root

    def add_node(self, node):
        self.nodes.append(node)
