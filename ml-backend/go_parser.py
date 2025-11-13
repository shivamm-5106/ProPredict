# go_parser.py
def parse_obo_file(obo_file_path):
    """
    Parse a Gene Ontology OBO file and extract GO IDs and names.
    Returns a dictionary: { 'GO:0000001': 'mitochondrion inheritance', ... }
    """
    go_terms = {}
    in_term = False
    current_id = None
    current_name = None

    with open(obo_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("!"):  # skip empty or comment lines
                continue

            if line == "[Term]":
                # When a new term starts, save the previous one
                if current_id and current_name:
                    go_terms[current_id] = current_name
                in_term = True
                current_id, current_name = None, None
                continue

            if not in_term:
                # skip header metadata before first [Term]
                continue

            if line.startswith("id: GO:"):
                current_id = line.split("id: ")[1].strip()
            elif line.startswith("name:"):
                current_name = line.split("name: ")[1].strip()

        # Save the last term (if file ends without another [Term])
        if current_id and current_name:
            go_terms[current_id] = current_name

    return go_terms


def create_go_index_mapping(go_terms_dict):
    """
    Create mappings:
        - index → {go_id, go_name, go_label}
        - go_id → index
    """
    sorted_go_ids = sorted(go_terms_dict.keys())
    index_mapping = {}
    go_id_to_index = {}

    for idx, go_id in enumerate(sorted_go_ids):
        go_name = go_terms_dict[go_id]
        go_label = (
            go_name.lower().replace(" ", "_").replace("-", "_").replace(",", "")
        )
        index_mapping[idx] = {
            "go_id": go_id,
            "go_name": go_name,
            "go_label": go_label,
        }
        go_id_to_index[go_id] = idx

    return index_mapping, go_id_to_index
