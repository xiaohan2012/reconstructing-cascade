import random
import pytest
import numpy as np
from numpy.testing import assert_array_equal
from graph_tool.all import shortest_distance, shortest_path

from steiner_tree_order import tree_sizes_by_roots
from ic import gen_nontrivial_cascade
from utils import get_rank_index

from fixtures import grid_and_cascade, tree_and_cascade, setup_function
 
## for TBFS
# ---------------
# for closure
# ---------------
