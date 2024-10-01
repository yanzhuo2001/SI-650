import math
import csv
from tqdm import tqdm


def map_score(search_result_relevances: list[int], cut_off: int = 10) -> float:
    num_relevant_retrieved = 0
    sum_precision = 0.0
    total_relevant = sum(search_result_relevances)
    if total_relevant == 0:
        return 0.0
    for i, rel in enumerate(search_result_relevances[:cut_off]):
        if rel == 1:
            num_relevant_retrieved += 1
            precision_at_i = num_relevant_retrieved / (i + 1)
            sum_precision += precision_at_i
    ap = sum_precision / total_relevant
    return ap


def ndcg_score(search_result_relevances: list[float],
               ideal_relevance_score_ordering: list[float], cut_off: int = 10):
    def dcg(relevances):
        dcg_value = 0.0
        for i, rel in enumerate(relevances[:cut_off]):
            rank = i + 1
            dcg_value += rel / math.log2(rank + 1)
        return dcg_value

    dcg_value = dcg(search_result_relevances)
    idcg_value = dcg(ideal_relevance_score_ordering)
    if idcg_value == 0.0:
        return 0.0
    else:
        ndcg = dcg_value / idcg_value
    return ndcg


def run_relevance_tests(relevance_data_filename: str, ranker) -> dict[str, float]:
    query_to_relevance = {}
    with open(relevance_data_filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            query = row['query']
            docid = int(row['docid'])
            rel = int(row['rel'])
            if query not in query_to_relevance:
                query_to_relevance[query] = {}
            query_to_relevance[query][docid] = rel

    ap_list = []
    ndcg_list = []

    for query, doc_rel_dict in tqdm(query_to_relevance.items(), desc="Evaluating Queries"):
        ranked_docs = ranker.query(query)
        ranked_docids = [docid for docid, score in ranked_docs]

        relevance_scores = []
        for docid in ranked_docids:
            rel = doc_rel_dict.get(docid, 0)
            relevance_scores.append(rel)

        binary_relevances = [1 if rel >= 1 else 0 for rel in relevance_scores]

        ideal_relevances = sorted(doc_rel_dict.values(), reverse=True)

        ap = map_score(binary_relevances)
        ap_list.append(ap)

        ndcg = ndcg_score(relevance_scores, ideal_relevances)
        ndcg_list.append(ndcg)

    mean_ap = sum(ap_list) / len(ap_list) if ap_list else 0.0
    mean_ndcg = sum(ndcg_list) / len(ndcg_list) if ndcg_list else 0.0

    return {'map': mean_ap, 'ndcg': mean_ndcg, 'map_list': ap_list, 'ndcg_list': ndcg_list}