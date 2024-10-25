package com.nvidia.spark.rapids.jni;

import ai.rapids.cudf.Table;

import java.util.Objects;

import static com.nvidia.spark.rapids.jni.TableUtils.ensure;


/**
 * A sliced view to table.
 * This table doesn't change ownership of the underlying data.
 */
public class SlicedTable {
    private final long startRow;
    private final long numRows;
    private final Table table;

    public SlicedTable(long startRow, long numRows, Table table) {
        Objects.requireNonNull(table, "table must not be null");
        ensure(startRow >= 0, "startRow must be >= 0");
        ensure(startRow < table.getRowCount(),
                () -> "startRow " + startRow  + " is larger than table row count " + table.getRowCount());
        ensure(numRows >= 0, () -> "numRows " + numRows + " is negative");
        ensure(startRow + numRows <= table.getRowCount(), () -> "startRow + numRows is " + (startRow + numRows)
                + ",  must be less than table row count " + table.getRowCount());

        this.startRow = startRow;
        this.numRows = numRows;
        this.table = table;
    }

    public long getStartRow() {
        return startRow;
    }

    public long getNumRows() {
        return numRows;
    }

    public Table getTable() {
        return table;
    }

    public static SlicedTable from(Table table, long startRow, long numRows) {
        return new SlicedTable(startRow, numRows, table);
    }
}

