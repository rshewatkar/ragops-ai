from app.rag_chain import (
    clean_output,
    relevance_score,
    route_query,
)


def test_clean_output():

    text = "   Hello    World   "

    result = clean_output(text)

    assert result == "Hello World"


def test_relevance_score():

    query = "python machine learning"

    context = "Python developer with machine learning experience"

    score = relevance_score(query, context)

    assert score > 0


def test_route_query_skills():

    query = "What are his ML skills?"

    routed_query, k = route_query(query)

    assert "skills" in routed_query.lower()

    assert k == 6