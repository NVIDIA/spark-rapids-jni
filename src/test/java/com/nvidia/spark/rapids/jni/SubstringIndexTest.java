package com.nvidia.spark.rapids.jni;

import ai.rapids.cudf.AssertUtils;
import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.Table;
import org.checkerframework.checker.units.qual.A;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

public class SubstringIndexTest {
    @Test
    void substringIndexTest(){
        Table.TestBuilder tb = new Table.TestBuilder();
        tb.column("www.apache", "www.apache", "www.apache", "www.apache");
        try(Table expected = tb.build()){
            Table.TestBuilder tb2 = new Table.TestBuilder();
            tb2.column("www.apache.org", "www.apache.org", "www.apache.org", "www.apache.org" );
            List<ColumnVector> result = new ArrayList<>();
            try (Table origTable = tb2.build()){
                for(int i = 0; i < origTable.getNumberOfColumns(); i++){
                    ColumnVector string_col = origTable.getColumn(i);
                    result.add(SubstringIndex.substringIndex(string_col, ".", 2));
                }
                try (Table result_tbl = new Table(
                        result.toArray(new ColumnVector[result.size()]))){
                    AssertUtils.assertTablesAreEqual(expected, result_tbl);
                }
            }finally {
                result.forEach(ColumnVector::close);
            }
        }
    }
}
