using System;
using Microsoft.ML;
using Microsoft.ML.Data;

class Program
{
  static void Main(string[] args)
  {
    MLContext mLContext = new MLContext;
  }
}









public class SentimentData
{
  [LoadColumn(0)] // Text Collumn
  public string Text { get; set;}

  [LoadColumn(1)] // True / False Collumn
  public bool Label { get; set; } 

}

public class SentimentPrediction : SentimentData
{
  [ColumnName("PredictedLabel")]
  public bool Prediction { get; set; }
  public float Probability { get; set; }
  public float Score { get; set; }
}