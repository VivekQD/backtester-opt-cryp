"""Cache Hillroute BTCUSDT 5-minute perpetual OHLCV without storing credentials."""
from __future__ import annotations

import argparse
import getpass
from pathlib import Path

import psycopg
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="100.118.197.103")
    p.add_argument("--database", default="propdesk")
    p.add_argument("--user", default="eden")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default="2025-01-01")
    p.add_argument("--output", default=".lwbt_cache/perp/BTCUSDT_5m_2021_2024.parquet")
    args = p.parse_args()
    password = getpass.getpass("Database password: ")
    conn = psycopg.connect(host=args.host, dbname=args.database, user=args.user,
                           password=password, connect_timeout=15)
    query = """
      SELECT datetime, open, high, low, close, volume
      FROM venue.ohlcv_binanceperp_5
      WHERE symbol = %s AND datetime >= %s AND datetime < %s
      ORDER BY datetime
    """
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([("datetime", pa.timestamp("ns")), ("open", pa.float64()),
                        ("high", pa.float64()), ("low", pa.float64()),
                        ("close", pa.float64()), ("volume", pa.float64())])
    writer = pq.ParquetWriter(out, schema, compression="zstd")
    count = 0
    with conn.cursor(name="perp_cache") as cur:
        cur.execute(query, (args.symbol, args.start, args.end))
        while rows := cur.fetchmany(50_000):
            cols = list(zip(*rows))
            table = pa.Table.from_arrays([
                pa.array(cols[0], type=pa.timestamp("ns")),
                *[pa.array(x, type=pa.float64()) for x in cols[1:]]], schema=schema)
            writer.write_table(table); count += len(rows)
    writer.close(); conn.close()
    print(f"wrote {count:,} rows to {out}")


if __name__ == "__main__":
    main()
