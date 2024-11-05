using System;
using Microsoft.ML;
using Microsoft.ML.Data;

public class SentimentData
{
  [LoadColumn(0)] // Text Collumn
  public string Text { get; set;}

  [LoadColumn(1)] // True / False Collumn
  public bool Label { get; set; } 

}