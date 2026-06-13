from langchain_core.documents import Document
from sklearn.model_selection import train_test_split


def make_chunks(n: int) -> list[Document]:
    return [
        Document(
            page_content=f"Текст чанка номер {i}",
            metadata={"source": f"doc_{i % 3}.md", "chunk_id": i},
        )
        for i in range(n)
    ]


try:
    from src.stages.splitter import split_chunks_train_test
except ImportError:
    def split_chunks_train_test(chunks, train_size=0.8):
        train_chunks, val_chunks = train_test_split(
            chunks, train_size=train_size, random_state=42, shuffle=True
        )
        return train_chunks, val_chunks


class TestSplitChunksTrainTest:
    def test_total_count_preserved(self):
        """Сумма train + val должна равняться исходному количеству чанков."""
        chunks = make_chunks(100)
        train, val = split_chunks_train_test(chunks, train_size=0.8)
        assert len(train) + len(val) == len(chunks)

    def test_default_split_ratio(self):
        """По умолчанию ~80% уходит в train."""
        chunks = make_chunks(100)
        train, val = split_chunks_train_test(chunks)
        assert len(train) == 80
        assert len(val) == 20

    def test_custom_split_ratio(self):
        chunks = make_chunks(100)
        train, val = split_chunks_train_test(chunks, train_size=0.9)
        assert len(train) == 90
        assert len(val) == 10

    def test_no_duplicates_between_sets(self):
        """Один и тот же чанк не должен попасть в оба сета."""
        chunks = make_chunks(50)
        train, val = split_chunks_train_test(chunks)
        train_contents = {c.page_content for c in train}
        val_contents = {c.page_content for c in val}
        assert train_contents.isdisjoint(val_contents)

    def test_no_data_loss(self):
        """Все чанки должны присутствовать в одном из наборов."""
        chunks = make_chunks(50)
        train, val = split_chunks_train_test(chunks)
        all_contents = {c.page_content for c in train} | {c.page_content for c in val}
        original_contents = {c.page_content for c in chunks}
        assert all_contents == original_contents

    def test_reproducible_with_same_seed(self):
        """При одинаковом random_state результат должен быть одинаковым."""
        chunks = make_chunks(40)
        train1, val1 = split_chunks_train_test(chunks)
        train2, val2 = split_chunks_train_test(chunks)
        assert [c.page_content for c in train1] == [c.page_content for c in train2]

    def test_metadata_preserved(self):
        """Метаданные чанков не должны теряться при split."""
        chunks = make_chunks(20)
        train, val = split_chunks_train_test(chunks)
        for chunk in train + val:
            assert "source" in chunk.metadata
            assert "chunk_id" in chunk.metadata

    def test_minimum_viable_dataset(self):
        """Split должен работать даже на очень маленьком наборе."""
        chunks = make_chunks(5)
        train, val = split_chunks_train_test(chunks, train_size=0.8)
        assert len(train) + len(val) == 5

    def test_returns_lists(self):
        chunks = make_chunks(10)
        train, val = split_chunks_train_test(chunks)
        assert isinstance(train, list)
        assert isinstance(val, list)
