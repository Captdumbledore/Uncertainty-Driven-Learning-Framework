def select_action(diagnosis):
    """
    Select an appropriate learning action
    based on the diagnosed knowledge gap.
    """

    if diagnosis == "Boundary Confusion":
        action = "Review confusing concepts"

    elif diagnosis == "Outlier":
        action = "Study the concept from basics"

    else:
        action = "Continue normal learning"

    return {
        "diagnosis": diagnosis,
        "action": action
    }

def create_learning_plan(
    query_embedding,
    embeddings,
    labels,
    k=5,
    disagreement_threshold=0.25,
    outlier_threshold=5.0
):
    """
    Diagnose a query embedding and select
    an appropriate learning action.
    """

    from .knn import diagnose_query

    diagnosis_result = diagnose_query(
        query_embedding,
        embeddings,
        labels,
        k=k,
        disagreement_threshold=disagreement_threshold,
        outlier_threshold=outlier_threshold
    )

    plan = select_action(
        diagnosis_result["diagnosis"]
    )

    return {
        "diagnosis": diagnosis_result["diagnosis"],
        "action": plan["action"],
        "disagreement": diagnosis_result["disagreement"],
        "average_distance": diagnosis_result["average_distance"]
    }