using System;
using Microsoft.ML;
using Microsoft.ML.Data;

class Program
{
    static void Main(string[] args)
    {
        MLContext mlContext = new MLContext();

        var dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(SentimentData.Text));
        var trainer = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features");
        var trainingPipeline = dataProcessPipeline.Append(trainer);

        var model = trainingPipeline.Fit(LoadData(mlContext));

        var predictionEngine = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

        var testSentiments = new[]
        {
          new SentimentData { Text = "I love this product!"},
          new SentimentData { Text = "This is the worst experience i have ever had"},
          new SentimentData { Text = "Im not sure how i feel about this"}
        };

        Console.WriteLine("Predictions:");
        foreach (var sentiment in testSentiments)
        {
          var result = predictionEngine.Predict(sentiment);
          Console.WriteLine($"Text: {sentiment.Text}");
          Console.WriteLine($"Prediction: {(result.Prediction ? "Positive" : "Negative")}, Probability: {result.Probability:P2}");
        }
    }

    static IDataView LoadData(MLContext mlContext)
    {
        var trainingData = new[]
        {
            new SentimentData { Text = "This product is great!", Label = true },
            new SentimentData { Text = "I did not like this product at all.", Label = false },
            new SentimentData { Text = "This is the best thing I have ever bought.", Label = true },
            new SentimentData { Text = "Worst purchase ever", Label = false }
        };

        return mlContext.Data.LoadFromEnumerable(trainingData);
    }
}

public class SentimentData
{
    [LoadColumn(0)] 
    public string? Text { get; set; }

    [LoadColumn(1)] 
    public bool Label { get; set; } 
}

public class SentimentPrediction : SentimentData
{
    [ColumnName("PredictedLabel")]
    public bool Prediction { get; set; }
    
    public float Probability { get; set; }
    
    public float Score { get; set; }
}
