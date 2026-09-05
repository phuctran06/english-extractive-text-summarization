import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent / "src")
)

from summarizer import select_top_sentences
from summarizer import create_summary


def test_select_top_sentences():
    sentences = [
        "Sentence one.",
        "Sentence two.",
        "Sentence three.",
        "Sentence four."
    ]

    scores = {
        0: 0.1,
        1: 0.4,
        2: 0.2,
        3: 0.3
    }

    similarity_matrix = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ]

    result = select_top_sentences(
        sentences,
        scores,
        similarity_matrix,
        num_sentences=2
    )

    assert result == [
        "Sentence two.",
        "Sentence four."
    ]


def test_create_summary():
    sentences = [
        "Sentence one.",
        "Sentence two.",
        "Sentence three.",
        "Sentence four."
    ]

    scores = {
        0: 0.1,
        1: 0.4,
        2: 0.2,
        3: 0.3
    }

    similarity_matrix = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ]

    result = create_summary(
        sentences,
        scores,
        similarity_matrix,
        num_sentences=2
    )

    assert result == "Sentence two. Sentence four."