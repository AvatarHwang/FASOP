

def get_all_cluster_combinations(model_type="gpt2XL", pareto=False, heterogeneous=False):
    """
    Returns all possible cluster combinations for the given model type
    """
    cluster_info={}
    if model_type == "gpt2XL":
        if pareto:
            assert False, "Pareto only supported for T5"
        if heterogeneous==True:
            cluster_info[0]= '1'
            for i in range(1, 8):
                cluster_info[i] = '0'
        else:
            for i in range(8):
                cluster_info[i] = '0'
        cluster_combinations = [cluster_info]
        return cluster_combinations
    elif model_type == "bert":
        if pareto==True:
            assert False, "Pareto only supported for T5"
        if heterogeneous==True:
            cluster_info[0]= '1'
            for i in range(1, 4):
                cluster_info[i] = '0'
        else:
            for i in range(4):
                cluster_info[i] = '0'
        cluster_combinations = [cluster_info]
        return cluster_combinations
    elif model_type == "T5":
        if pareto!=True:
            if heterogeneous:
                cluster_info[0] = '1'
                for i in range(1, 8):
                    cluster_info[i] = '0'
                cluster_combinations = [cluster_info]
                return cluster_combinations
            else:
                cluster_combinations = []
                for i in range(4):
                    cluster_info[i] = '0'
                cluster_combinations.append(cluster_info)        
                return cluster_combinations
        else:
            num_c = 0
            cluster_combinations = []
            for num_a100 in range(1, 8+1):
                for num_a10 in range(1, 8+1):
                    cluster = {}
                    for i in range(num_a100+num_a10):
                        cluster[i] = '0'
                    for i in range(num_a100):
                        cluster.update({i:'1'})
                    if len(cluster.keys())>0:
                        cluster_combinations.append(cluster)
                    num_c += 1
            print(f"Number of clusters combinations: {num_c}")
    else:
        assert False, "Model type not supported"


def device_placement(num_a100, num_a10):
    a100_nodes = ''
    a10_nodes = ''
    for i in range(num_a100):
        a100_nodes += 'A'
    for i in range(num_a10):
        a10_nodes += 'B'

    print(f"a100_nodes+a10_nodes: {a100_nodes+a10_nodes}")

    D = []
    count=0
    for d in msp(a100_nodes+a10_nodes):
        count += 1
        D.append(d)
    print(f"Number of node placement: {count}")
    return D

def msp(items):
  '''Yield the permutations of `items` where items is either a list
  of integers representing the actual items or a list of hashable items.
  The output are the unique permutations of the items given as a list
  of integers 0, ..., n-1 that represent the n unique elements in
  `items`.

  Examples
  ========

  >>> for i in msp('xoxox'):
  ...   print(i)

  [1, 1, 1, 0, 0]
  [0, 1, 1, 1, 0]
  [1, 0, 1, 1, 0]
  [1, 1, 0, 1, 0]
  [0, 1, 1, 0, 1]
  [1, 0, 1, 0, 1]
  [0, 1, 0, 1, 1]
  [0, 0, 1, 1, 1]
  [1, 0, 0, 1, 1]
  [1, 1, 0, 0, 1]

  Reference: "An O(1) Time Algorithm for Generating Multiset Permutations", Tadao Takaoka
  https://pdfs.semanticscholar.org/83b2/6f222e8648a7a0599309a40af21837a0264b.pdf
  '''

  def visit(head):
      (rv, j) = ([], head)
      for i in range(N):
          (dat, j) = E[j]
          rv.append(dat)
      return rv

  u = list(set(items))
  E = list(([u.index(i) for i in items]))
  N = len(E)
  # put E into linked-list format
  (val, nxt) = (0, 1)
  for i in range(N):
      E[i] = [E[i], i + 1]
  E[-1][nxt] = None
  head = 0
  afteri = N - 1
  i = afteri - 1
  yield visit(head)
  while E[afteri][nxt] is not None or E[afteri][val] < E[head][val]:
      j = E[afteri][nxt]  # added to algorithm for clarity
      if j is not None and E[i][val] >= E[j][val]:
          beforek = afteri
      else:
          beforek = i
      k = E[beforek][nxt]
      E[beforek][nxt] = E[k][nxt]
      E[k][nxt] = head
      if E[k][val] < E[head][val]:
          i = k
      afteri = E[i][nxt]
      head = k
      yield visit(head)