using System;
using Microsoft.ML;
using Microsoft.ML.Data;

public class SentimentData
{
  [LoadColumn(0)]
  public string Text { get; set;}
  [LoadColumn(1)]
  public bool Label { get; set; }

}