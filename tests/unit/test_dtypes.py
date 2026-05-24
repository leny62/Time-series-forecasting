import pyarrow as pa

from mtraffic.transform.dtypes import canonical_schema, to_canonical


def test_canonical_schema_types() -> None:
    s = canonical_schema()
    assert s.field("square_id").type == pa.uint16()
    assert pa.types.is_timestamp(s.field("ts").type)
    assert s.field("internet").type == pa.float32()


def test_to_canonical_casts() -> None:
    table = pa.table(
        {
            "square_id": pa.array([1, 2, 3], type=pa.int64()),
            "ts": pa.array([1383260400000, 1383261000000, 1383261600000], type=pa.int64()),
            "internet": pa.array([0.1, 0.2, 0.3], type=pa.float64()),
        }
    )
    out = to_canonical(table)
    assert out.schema.field("square_id").type == pa.uint16()
    assert out.schema.field("internet").type == pa.float32()
    assert pa.types.is_timestamp(out.schema.field("ts").type)
