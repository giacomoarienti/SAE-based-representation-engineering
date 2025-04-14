def arrow_graph(graph):
    lines = ["The entities are presented as a graph in the following section:"]
    for path in graph:
        if len(path) == 3:
            lines.append(f"{path[0]} -- {path[1]} --> {path[2]}")
        elif len(path) == 2:
            lines.append(f"{path[0]} --> {path[1]}")
    return "\n".join(lines)

def tuple_graph(graph):
    return "The entities are presented as knowledge graph triplets (head, relation, tail):" + \
        "\n".join(
            f"({t[0]}, {t[1]}, {t[2]})" if len(t) == 3 else f"({t[0]}, {t[1]})"
            for t in graph
        )
    
def lookup_table_graph(graph):
    def index_to_alpha(n):
        import string
        n += 26  # Start from 'aa' instead of 'a'
        result = ''
        while n >= 0:
            result = chr(ord('a') + (n % 26)) + result
            n = n // 26 - 1
            if n < 0:
                break
        return result

    entity_map = {}
    rel_map = {}
    entity_ids = {}
    rel_ids = {}
    entity_counter = 0
    rel_counter = 0
    triples = []

    for t in graph:
        if len(t) != 3:
            continue
        head, rel, tail = t
        for e in [head, tail]:
            if e not in entity_ids:
                key = chr(ord('A') + entity_counter)
                entity_ids[e] = key
                entity_map[key] = e
                entity_counter += 1
        if rel not in rel_ids:
            key = index_to_alpha(rel_counter)
            rel_ids[rel] = key
            rel_map[key] = rel
            rel_counter += 1
        triples.append(f"{entity_ids[head]}, {rel_ids[rel]}, {entity_ids[tail]}")

    lines = []
    lines.append("The entities are assigned symbolic keys as follows:")
    for k, v in entity_map.items():
        lines.append(f"{k}: {v}")
    lines.append("\nThe relations are assigned symbolic keys as follows:")
    for k, v in rel_map.items():
        lines.append(f"{k}: {v}")
    lines.append("\nThe graph is defined with the symbolic references:")
    lines.extend(triples)
    return "" + "\n".join(lines)