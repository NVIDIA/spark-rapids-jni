package com.nvidia.spark.rapids.jni;

import ai.rapids.cudf.AssertUtils;
import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.Table;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

public class GpuSubstringIndexUtilsTest {
    @Test
    void gpuSubstringIndexTest(){
        Table.TestBuilder tb = new Table.TestBuilder();
        tb.column( "www.apache.org");
        tb.column("www.apache");
        tb.column("www");
        tb.column("");
        tb.column("org");
        tb.column("apache.org");
        tb.column("www.apache.org");
        tb.column("");

        try(Table expected = tb.build()){
            Table.TestBuilder tb2 = new Table.TestBuilder();
            tb2.column("www.apache.org");
            tb2.column("www.apache.org");
            tb2.column("www.apache.org");
            tb2.column("www.apache.org");
            tb2.column("www.apache.org");
            tb2.column("www.apache.org");
            tb2.column("www.apache.org");
            tb2.column("");

            String[] delimiterArray = new String[]{".", ".", ".", ".",".", ".", ".", "."};
            int[] countArray = new int[]{3, 2, 1, 0, -1, -2, -3, -2};
            List<ColumnVector> result = new ArrayList<>();
            try (Table origTable = tb2.build()){
                for(int i = 0; i < origTable.getNumberOfColumns(); i++){
                    ColumnVector string_col = origTable.getColumn(i);
                    result.add(GpuSubstringIndexUtils.substringIndex(string_col, delimiterArray[i], countArray[i]));
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
