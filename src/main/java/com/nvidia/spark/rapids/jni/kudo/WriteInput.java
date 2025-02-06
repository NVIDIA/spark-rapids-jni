package com.nvidia.spark.rapids.jni.kudo;

import ai.rapids.cudf.HostColumnVector;
import java.io.OutputStream;


/**
 * This class contains the input for writing partition of an array of columns to an output stream.
 * <br/>
 * It contains the columns to write, the output stream to write to, the row offset to start writing,
 * the number of rows to write, and a flag to indicate whether to measure the time spent on copying
 */
public class WriteInput {
  final HostColumnVector[] columns;
  final OutputStream outputStream;
  final int rowOffset;
  final int numRows;
  final boolean measureCopyBufferTime;

  private WriteInput(HostColumnVector[] columns, OutputStream outputStream, int rowOffset,
                    int numRows, boolean measureCopyBufferTime) {
    this.columns = columns;
    this.outputStream = outputStream;
    this.rowOffset = rowOffset;
    this.numRows = numRows;
    this.measureCopyBufferTime = measureCopyBufferTime;
  }

  public HostColumnVector[] getColumns() {
    return columns;
  }

  public OutputStream getOutputStream() {
    return outputStream;
  }

  public int getRowOffset() {
    return rowOffset;
  }

  public int getNumRows() {
    return numRows;
  }

  public boolean isMeasureCopyBufferTime() {
    return measureCopyBufferTime;
  }

  public static Builder builder() {
    return new Builder();
  }

  public static class Builder {
    private HostColumnVector[] columns;
    private OutputStream outputStream;
    private int rowOffset;
    private int numRows;
    private boolean measureCopyBufferTime = false;

    public Builder setColumns(HostColumnVector[] columns) {
      this.columns = columns;
      return this;
    }

    public Builder setOutputStream(OutputStream outputStream) {
      this.outputStream = outputStream;
      return this;
    }

    public Builder setRowOffset(int rowOffset) {
      this.rowOffset = rowOffset;
      return this;
    }

    public Builder setNumRows(int numRows) {
      this.numRows = numRows;
      return this;
    }

    public Builder setMeasureCopyBufferTime(boolean measureCopyBufferTime) {
      this.measureCopyBufferTime = measureCopyBufferTime;
      return this;
    }

    public WriteInput build() {
      return new WriteInput(columns, outputStream, rowOffset, numRows, measureCopyBufferTime);
    }
  }
}
